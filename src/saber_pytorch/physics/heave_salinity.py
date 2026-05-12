"""Weaver/Ricci-style local temperature-to-salinity physical balance.

The balance maps temperature increments to balanced salinity increments via a
level-local T/S slope relationship.  For each level k:

    δS_k = K_ST,k · δT_k

where:

    K_ST,k = α₀ · w_T,k · S_z,k · T_z,k / (T_z,k² + ε)

and the optional temperature-gradient taper is:

    w_T,k = T_z,k² / (T_z,k² + ε_T)

T_z and S_z are finite-difference vertical derivatives of the background
state (z positive downward).  The Jacobian is diagonal in vertical level:

    ∂δS_k/∂δT_j = 0  for j ≠ k

This avoids the dense vertical coupling of displacement-based approaches and
keeps the Jacobian block-diagonal, compatible with the SABER TorchBalance
memory assumptions.

SABER TorchBalance contract
----------------------------
Same packed-tensor layout as StericHeightEmulator:

    inputs[:, 0*n:1*n] = background potential temperature
    inputs[:, 1*n:2*n] = background salinity
    inputs[:, 2*n:3*n] = layer thickness (m)

Jacobian shape: [nnodes, nlevels, 3*nlevels]
    Non-zero block: diagonal of [:, :, 0:nlevels]  (temperature → salinity)
    Zero blocks:    [:, :, nlevels:]               (S and dz columns unused)
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class WeaverTSBalance(nn.Module):
    """Weaver/Ricci-style local physical T→S balance for SABER TorchBalance.

    inputs layout (all fields share nlevels = inputs.shape[1] // 3):
        [:, 0*n:1*n] = background potential temperature
        [:, 1*n:2*n] = background salinity
        [:, 2*n:3*n] = layer thickness, positive in metres

    output rows are salinity levels.  The Jacobian has shape
    [nnodes, nlevels, 3*nlevels]; only the diagonal of the temperature
    block is non-zero.
    """

    # TorchScript attribute declarations
    input_names: List[str]
    output_names: List[str]
    epsilon: float
    epsilon_taper: float
    amplitude: float
    use_temperature_gradient_taper: bool
    suppress_shallow_weak_stratification: bool
    shallow_taper_depth_m: float
    shallow_epsilon_taper: float
    mask_var_name: str
    mask_min: float
    mask_max: float

    def __init__(
        self,
        input_names: List[str],
        output_names: List[str],
        epsilon: float = 1.0e-12,
        epsilon_taper: float = 1.0e-3,
        amplitude: float = 1.0,
        use_temperature_gradient_taper: bool = True,
        suppress_shallow_weak_stratification: bool = False,
        shallow_taper_depth_m: float = 50.0,
        shallow_epsilon_taper: float = 1.0e-4,
        mask_var_name: str = "",
        mask_min: float = 0.0,
        mask_max: float = 1.0,
    ) -> None:
        """
        Args:
            input_names:  3-element list [T_name, S_name, dz_name].
            output_names: 1-element list [salinity_output_name].
            epsilon:      regularization for T_z² denominator.
            epsilon_taper: 50 % suppression threshold at |T_z| = sqrt(ε_T).
                Use ~1.0e-3 for a threshold near 0.032 °C/m (thermocline).
            amplitude:    scalar multiplier α₀ for K_ST (default 1.0).
            use_temperature_gradient_taper: whether to apply w_T taper.
            suppress_shallow_weak_stratification: apply a stronger
                temperature-gradient taper above shallow_taper_depth_m.
            shallow_taper_depth_m: layer-centre depth threshold for the
                shallow weak-stratification taper.
            shallow_epsilon_taper: ε_T used above shallow_taper_depth_m when
                suppress_shallow_weak_stratification is enabled.
        """
        super().__init__()
        if len(input_names) != 3:
            raise ValueError(
                "WeaverTSBalance requires exactly 3 input names: "
                "[T_name, S_name, dz_name]"
            )
        if len(output_names) != 1:
            raise ValueError(
                "WeaverTSBalance requires exactly 1 output name"
            )

        self.input_names = input_names
        self.output_names = output_names
        self.epsilon = float(epsilon)
        self.epsilon_taper = float(epsilon_taper)
        self.amplitude = float(amplitude)
        self.use_temperature_gradient_taper = bool(
            use_temperature_gradient_taper
        )
        self.suppress_shallow_weak_stratification = bool(
            suppress_shallow_weak_stratification
        )
        self.shallow_taper_depth_m = float(shallow_taper_depth_m)
        self.shallow_epsilon_taper = float(shallow_epsilon_taper)
        self.mask_var_name = mask_var_name
        self.mask_min = float(mask_min)
        self.mask_max = float(mask_max)

    def _split_inputs(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = inputs.shape[1] // 3
        return (
            inputs[:, 0 * n : 1 * n],
            inputs[:, 1 * n : 2 * n],
            inputs[:, 2 * n : 3 * n],
        )

    def _vertical_derivative(
        self,
        profile: torch.Tensor,
        dz: torch.Tensor,
        min_thickness: float = 1.0e-12,
    ) -> torch.Tensor:
        """Finite-difference d(profile)/dz at layer centres, z positive down.

        Centered differences in the interior, one-sided at top and bottom.
        Levels with zero or near-zero thickness are masked to zero.
        """
        n = profile.shape[1]
        safe_dz = torch.clamp(dz, min=min_thickness)
        z = torch.cumsum(safe_dz, dim=1) - safe_dz * 0.5
        deriv = torch.zeros_like(profile)

        if n == 1:
            return deriv

        top_dz = torch.clamp(z[:, 1] - z[:, 0], min=min_thickness)
        deriv[:, 0] = (profile[:, 1] - profile[:, 0]) / top_dz

        bottom_dz = torch.clamp(
            z[:, n - 1] - z[:, n - 2], min=min_thickness
        )
        deriv[:, n - 1] = (
            (profile[:, n - 1] - profile[:, n - 2]) / bottom_dz
        )

        if n > 2:
            interior_dz = torch.clamp(
                z[:, 2:n] - z[:, 0 : n - 2],
                min=min_thickness,
            )
            deriv[:, 1 : n - 1] = (
                (profile[:, 2:n] - profile[:, 0 : n - 2]) / interior_dz
            )

        valid = dz > min_thickness
        return torch.where(valid, deriv, torch.zeros_like(deriv))

    def _layer_center_depth(
        self,
        dz: torch.Tensor,
        min_thickness: float = 1.0e-12,
    ) -> torch.Tensor:
        """Return layer-centre depth, z positive downward."""
        safe_dz = torch.clamp(dz, min=min_thickness)
        return torch.cumsum(safe_dz, dim=1) - safe_dz * 0.5

    def _compute_jacobian(
        self,
        T_bg: torch.Tensor,
        S_bg: torch.Tensor,
        dz: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return full Jacobian [nnodes, nlevels, 3*nlevels]."""
        T_z = self._vertical_derivative(T_bg, dz)
        S_z = self._vertical_derivative(S_bg, dz)

        T_z2 = T_z * T_z
        if self.use_temperature_gradient_taper:
            epsilon_taper = torch.full_like(T_z, self.epsilon_taper)
            if self.suppress_shallow_weak_stratification:
                z = self._layer_center_depth(dz)
                shallow_epsilon = torch.full_like(T_z, self.shallow_epsilon_taper)
                epsilon_taper = torch.where(
                    z <= self.shallow_taper_depth_m,
                    shallow_epsilon,
                    epsilon_taper,
                )
            w_T = T_z2 / (T_z2 + epsilon_taper)
        else:
            w_T = torch.ones_like(T_z)

        K_ST = self.amplitude * w_T * S_z * T_z / (T_z2 + self.epsilon)

        # Diagonal Jacobian: K_ST is the only non-zero entry per level.
        jac_T = torch.diag_embed(K_ST)     # [nnodes, nlevels, nlevels]
        zeros = torch.zeros_like(jac_T)
        jac = torch.cat(
            [jac_T, zeros, zeros], dim=2   # [nnodes, nlevels, 3*nlevels]
        )
        return jac * mask.unsqueeze(2)

    def forward(
        self,
        inputs: torch.Tensor,
        temperature_increment: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the balance to a temperature increment.

        Args:
            inputs:                packed background profiles [T, S, dz].
            temperature_increment: [nnodes, nlevels].

        Returns:
            balanced salinity increment [nnodes, nlevels].
        """
        T_bg, S_bg, dz = self._split_inputs(inputs)
        mask = torch.ones(
            inputs.shape[0], 1, dtype=inputs.dtype, device=inputs.device
        )
        jac = self._compute_jacobian(T_bg, S_bg, dz, mask)
        n = T_bg.shape[1]
        return torch.bmm(
            jac[:, :, :n], temperature_increment.unsqueeze(2)
        ).squeeze(2)

    @torch.jit.export
    def apply_from_state(
        self,
        state: Dict[str, torch.Tensor],
        temperature_increment: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the balance using a dict of background fields."""
        T_bg = state[self.input_names[0]]
        S_bg = state[self.input_names[1]]
        dz = state[self.input_names[2]]
        jac = self._compute_jacobian(T_bg, S_bg, dz, mask)
        n = T_bg.shape[1]
        return torch.bmm(
            jac[:, :, :n], temperature_increment.unsqueeze(2)
        ).squeeze(2)

    @torch.jit.export
    def jac_from_state(
        self, state: Dict[str, torch.Tensor], mask: torch.Tensor
    ) -> torch.Tensor:
        """Jacobian from a dict of background fields."""
        T_bg = state[self.input_names[0]]
        S_bg = state[self.input_names[1]]
        dz = state[self.input_names[2]]
        return self._compute_jacobian(T_bg, S_bg, dz, mask)

    @torch.jit.export
    def compute_mask(self, mask_var: torch.Tensor) -> torch.Tensor:
        """Compute binary domain mask from raw background values.

        Args:
            mask_var: [nnodes, 1] — background values of the mask variable
                      (CF name: self.mask_var_name).

        Returns:
            [nnodes, 1] float32 mask (1 = active, 0 = outside valid range).
        """
        if self.mask_var_name == "":
            return torch.ones(mask_var.shape[0], 1, dtype=mask_var.dtype, device=mask_var.device)
        v = mask_var[:, 0]
        valid = (v >= self.mask_min) & (v <= self.mask_max)
        return valid.unsqueeze(1).to(mask_var.dtype)

    @torch.jit.export
    def jac_physical(
        self,
        inputs: torch.Tensor,
        mask_var: torch.Tensor,
        requested_row_indices: torch.Tensor,
        requested_col_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Jacobian entries requested by TorchBalanceVerticalEmulator.cc."""
        mask = self.compute_mask(mask_var)
        T_bg, S_bg, dz = self._split_inputs(inputs)
        jac_full = self._compute_jacobian(T_bg, S_bg, dz, mask)
        jac_rows = jac_full.index_select(1, requested_row_indices)
        gather_cols = requested_col_indices.view(1, -1, 1).expand(
            jac_full.shape[0], -1, 1
        )
        jac = jac_rows.gather(2, gather_cols).squeeze(2)
        return torch.where(torch.isfinite(jac), jac, torch.zeros_like(jac))
