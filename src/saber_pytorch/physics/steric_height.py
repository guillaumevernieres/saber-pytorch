"""Steric height emulator compatible with the SABER TorchBalance contract.

Physics
-------
In a Boussinesq ocean the steric SSH anomaly is:

    η = -(1/ρ₀) · Σ_k [ρ(T_k, S_k, p_k) - ρ_ref] · dz_k

The linearised Jacobian w.r.t. the background state (T_bg, S_bg) is:

    ∂η/∂T_k = -(1/ρ₀) · (∂ρ/∂T)|_(bg,k) · dz_k
    ∂η/∂S_k = -(1/ρ₀) · (∂ρ/∂S)|_(bg,k) · dz_k

Both dz (layer thicknesses) and depth (mid-level depths in metres) are
received as runtime input fields from SABER.

SABER TorchBalance contract (setupVerticalEmulator)
----------------------------------------------------
The emulator is a TorchScript module with:

  Attributes:
    input_names  : List[str]  — 3 variable names: [T, S, dz]
    output_names : List[str]  — exactly one output variable name (SSH)

  Method called by C++:
    jac_physical(inputs: Tensor, mask: Tensor, row_indices: Tensor, col_indices: Tensor) -> Tensor

  inputs is packed by the C++ as all levels of each input_names entry:
    inputs[:, 0*nlevels : 1*nlevels] = T background profiles
    inputs[:, 1*nlevels : 2*nlevels] = S background profiles
    inputs[:, 2*nlevels : 3*nlevels] = dz (layer thicknesses, m)

  Return shape: [nnodes, nRequestedPairs]
    Compact column layout:
      0*nlevels : 1*nlevels  = ∂η/∂T    (non-zero)
      1*nlevels : 2*nlevels  = ∂η/∂S    (non-zero)
      2*nlevels : 3*nlevels  = ∂η/∂dz   (zeros — geometry, not DA state)

Python-friendly interface
-------------------------
  jac_from_state(state: Dict[str, Tensor], mask: Tensor) -> Tensor

  Accepts a dict keyed by variable name. Use this from Python scripts and
  tests; jac_physical keeps the packed-tensor signature required by C++.

Assumptions
-----------
- All 4 input fields share the same nlevels.
- All inputs arrive as float32 from the C++ layer.
- rho0 == rho_ref == 1025.0 kg/m³.
"""

from typing import Dict, List

import torch
import torch.nn as nn

from .roquet_eos import RoquetEOS

RHO0: float = 1025.0
G: float = 9.81  # m/s²


def depth_to_pressure(depth_m: torch.Tensor, rho0: float = RHO0) -> torch.Tensor:
    """Convert depth [m] to pressure [dbar] via the hydrostatic approximation.

        p [dbar] = ρ₀ · g · depth [m] / 1e4

    Args:
        depth_m: depths in metres, positive downward.
        rho0:    reference density [kg/m³].

    Returns:
        pressure [dbar], same shape as depth_m.
    """
    return depth_m * (rho0 * G / 1.0e4)


class StericHeightEmulator(nn.Module):
    """Steric height Jacobian emulator for SABER TorchBalance.

    dz (layer thicknesses) is received as a runtime input field; mid-level
    depths and pressure are derived from it on the fly.
    """

    # TorchScript attribute declarations
    input_names: List[str]
    output_names: List[str]
    rho0: float

    def _to_pressure(self, depth_m: torch.Tensor) -> torch.Tensor:
        # p [dbar] = rho0 * g * depth [m] / 1e4;  g=9.81 inlined (TorchScript
        # cannot close over module-level float variables).
        return depth_m * (self.rho0 * 9.81 / 1.0e4)

    def __init__(
        self,
        input_names: List[str],
        output_names: List[str],
        rho0: float = RHO0,
    ) -> None:
        """
        Args:
            input_names:  3-element list [T_name, S_name, dz_name].
                          Must match the field names in the SABER configuration
                          and the order in which C++ packs the inputs tensor.
            output_names: 1-element list [ssh_name].
            rho0:         Boussinesq reference density [kg/m³].
        """
        super().__init__()
        if len(input_names) != 3:
            raise ValueError(
                "StericHeightEmulator requires exactly 3 input names: "
                "[T_name, S_name, dz_name]"
            )
        if len(output_names) != 1:
            raise ValueError("StericHeightEmulator requires exactly 1 output name (SSH)")

        self.eos = RoquetEOS()
        self.rho0 = float(rho0)
        self.input_names = input_names
        self.output_names = output_names

    # ------------------------------------------------------------------
    # Forward pass  (nonlinear steric SSH)
    # Required for TorchScript tracing; NOT called by SABER C++ at runtime.
    # ------------------------------------------------------------------

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Nonlinear steric SSH from packed T/S/dz background profiles.

        Args:
            inputs: [nnodes, 3*nlevels] — T, S, dz profiles.

        Returns:
            ssh: [nnodes, 1] — steric SSH (m).
        """
        n = inputs.shape[1] // 3
        T  = inputs[:, 0*n:1*n]
        S  = inputs[:, 1*n:2*n]
        dz = inputs[:, 2*n:3*n]
        depth = torch.cumsum(dz, dim=1) - dz * 0.5
        pressure = self._to_pressure(depth)
        sigma = self.eos.density_anomaly(T, S, pressure, self.rho0)
        return -(1.0 / self.rho0) * (sigma * dz).sum(dim=1, keepdim=True)

    # ------------------------------------------------------------------
    # Core Jacobian computation (shared by both public methods)
    # ------------------------------------------------------------------

    def _compute_jacobian(
        self,
        T_bg: torch.Tensor,
        S_bg: torch.Tensor,
        dz: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return full Jacobian [nnodes, 1, 3*nlevels].

        depth_m is derived from dz:  depth_m[k] = cumsum(dz)[k] - dz[k]/2.
        The dz column block is zero: layer thickness is grid geometry, not a
        DA state variable, so its Jacobian block is unused by SABER.
        """
        depth_m = torch.cumsum(dz, dim=1) - dz * 0.5
        pressure = self._to_pressure(depth_m)
        rho_th, rho_s = self.eos.derivatives(T_bg, S_bg, pressure)

        jac_T = -(1.0 / self.rho0) * rho_th * dz                  # [nnodes, nlevels]
        jac_S = -(1.0 / self.rho0) * rho_s * dz                   # [nnodes, nlevels]
        zeros = torch.zeros_like(jac_T)

        # [nnodes, 3*nlevels] → [nnodes, 1, 3*nlevels]
        jac = torch.cat([jac_T, jac_S, zeros], dim=1).unsqueeze(1)
        return jac * mask.unsqueeze(2)                             # mask: [nnodes, 1, 1]

    # ------------------------------------------------------------------
    # Python-friendly interface
    # ------------------------------------------------------------------

    @torch.jit.export
    def jac_from_state(
        self, state: Dict[str, torch.Tensor], mask: torch.Tensor
    ) -> torch.Tensor:
        """Jacobian from a dict of background fields.

        Args:
            state: keys must include all three entries of input_names.
            mask:  [nnodes, 1].

        Returns:
            jac: [nnodes, 1, 3*nlevels].
        """
        T_bg = state[self.input_names[0]]
        S_bg = state[self.input_names[1]]
        dz   = state[self.input_names[2]]
        return self._compute_jacobian(T_bg, S_bg, dz, mask)

    # ------------------------------------------------------------------
    # SABER C++ entry point
    # ------------------------------------------------------------------

    @torch.jit.export
    def jac_physical(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor,
        requested_row_indices: torch.Tensor,
        requested_col_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Jacobian from the packed background tensor assembled by SABER C++.

        inputs layout (all fields share nlevels = inputs.shape[1] // 3):
            [:, 0*n:1*n] = T background
            [:, 1*n:2*n] = S background
            [:, 2*n:3*n] = dz  (layer thicknesses, m)

        Args:
            inputs:                [nnodes, 3*nlevels].
            mask:                  [nnodes, 1].
            requested_row_indices: 1-D int64 tensor of output row indices.
            requested_col_indices: 1-D int64 tensor of column indices into the
                                   full [nnodes, 1, 3*nlevels] Jacobian.

        Returns:
            jac: [nnodes, len(requested_col_indices)].
        """
        n = inputs.shape[1] // 3
        T_bg = inputs[:, 0*n:1*n]
        S_bg = inputs[:, 1*n:2*n]
        dz   = inputs[:, 2*n:3*n]
        jac_full = self._compute_jacobian(T_bg, S_bg, dz, mask)
        jac_rows = jac_full.index_select(1, requested_row_indices)
        gather_cols = requested_col_indices.view(1, -1, 1).expand(jac_full.shape[0], -1, 1)
        return jac_rows.gather(2, gather_cols).squeeze(2)
