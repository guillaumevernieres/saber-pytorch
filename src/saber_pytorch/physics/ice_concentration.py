"""Surface ice concentration prior-state Jacobian provider for SABER TorchBalance.

Contract
--------
This module supplies a linearised d(aice)/d[sst, sss, hi, hs] Jacobian
evaluated at a prior (background) state that also includes the prior aice
value.  It is NOT a nonlinear SIC predictor.

Packed input tensor (5 columns):
    inputs[:, 0] = sst_bg   — sea surface temperature background
    inputs[:, 1] = sss_bg   — sea surface salinity background
    inputs[:, 2] = hi_bg    — sea ice thickness background
    inputs[:, 3] = hs_bg    — snow depth over sea ice background
    inputs[:, 4] = aice_prior — prior sea ice area fraction

The column for aice_prior is used only to compute the state-dependent weight
w(aice_prior); it is NOT included in the returned Jacobian.

Output (jac_physical):
    [nnodes, 1, 4] = d(aice) / d[sst, sss, hi, hs]

Jacobian formula
----------------
The prior ice concentration determines a state-dependent sensitivity weight:

    a = clamp(aice_prior, 0, 1)
    w = w_min + (1 - w_min) * 4 * a * (1 - a)

This weight is maximum at a = 0.5 (maximum marginal ice zone sensitivity)
and equals w_min at a = 0 or a = 1 (open ocean or complete ice cover).

The local freezing temperature is approximated as:

    S = max(SSS, 0)
    Tf = tf0 + tf_s_linear * S + tf_s_pow * S * sqrt(S)

The Jacobian is active only when the background state is close enough to the
freezing point:

    active = |SST - Tf| <= freezing_tolerance

Partial derivatives for active nodes:

    d(aice)/dSST = -alpha_t * w
    d(aice)/dSSS =  alpha_t * dTf/dSSS * w
    d(aice)/dHI  =  alpha_hi * exp(-hi / hi_scale) * w
    d(aice)/dHS  =  alpha_hs * exp(-hs / hs_scale) * w

where

    dTf/dSSS = tf_s_linear + 1.5 * tf_s_pow * sqrt(max(SSS, 0))

For inactive nodes, all Jacobian entries are zero.

Sign expectations:
    d(aice)/dSST <= 0
    d(aice)/dSSS <= 0  for normal ocean salinity (Tf decreases with salinity)
    d(aice)/dHI  >= 0
    d(aice)/dHS  >= 0

SABER TorchBalance surface contract
------------------------------------
Attributes:
  input_names  : List[str]  (5 names: [sst, sss, hi, hs, aice])
    input_levels : List[int]  (5 levels, typically all zeros for surface fields)
  output_names : List[str]  (1 name:  [aice])
    output_levels: List[int]  (1 level, typically [0])

Method called by C++:
  jac_physical(inputs: Tensor, mask: Tensor) -> Tensor

  inputs: [nnodes, 5]
  mask:   [nnodes, 1]
  return: [nnodes, 1, 4]

forward():
  Applies the Jacobian as a linear map to a perturbation vector:
    inputs:       [nnodes, 5]
    perturbations:[nnodes, 4]  = [d_sst, d_sss, d_hi, d_hs]
    return:       [nnodes, 1]  = delta_aice
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class SurfaceIceConcentrationEmulator(nn.Module):
    """Prior-state Jacobian provider for sea-ice concentration balance."""

    input_names: List[str]
    input_levels: List[int]
    output_names: List[str]
    output_levels: List[int]
    alpha_t: float
    alpha_hi: float
    alpha_hs: float
    hi_scale: float
    hs_scale: float
    w_min: float
    tf0: float
    tf_s_linear: float
    tf_s_pow: float
    freezing_tolerance: float

    def __init__(
        self,
        input_names: List[str],
        output_names: List[str],
        input_levels: Optional[List[int]] = None,
        output_levels: Optional[List[int]] = None,
        alpha_t: float = 1.0,
        alpha_hi: float = 0.2,
        alpha_hs: float = 0.1,
        hi_scale: float = 0.5,
        hs_scale: float = 0.1,
        w_min: float = 0.05,
        tf0: float = 0.0901,
        tf_s_linear: float = -0.0575,
        tf_s_pow: float = 1.710523e-3,
        freezing_tolerance: float = 1.0,
    ) -> None:
        super().__init__()

        if len(input_names) != 5:
            raise ValueError(
                "SurfaceIceConcentrationEmulator requires exactly 5 input names: "
                "[sst_name, sss_name, hi_name, hs_name, aice_prior_name]"
            )
        if len(output_names) != 1:
            raise ValueError(
                "SurfaceIceConcentrationEmulator requires exactly 1 output name"
            )
        resolved_input_levels = (
            [0] * len(input_names) if input_levels is None else list(input_levels)
        )
        resolved_output_levels = (
            [0] * len(output_names) if output_levels is None else list(output_levels)
        )
        if len(resolved_input_levels) != len(input_names):
            raise ValueError(
                f"input_levels length ({len(resolved_input_levels)}) must match "
                f"input_names length ({len(input_names)})"
            )
        if len(resolved_output_levels) != len(output_names):
            raise ValueError(
                f"output_levels length ({len(resolved_output_levels)}) must match "
                f"output_names length ({len(output_names)})"
            )
        if alpha_t < 0.0:
            raise ValueError("alpha_t must be >= 0.0")
        if alpha_hi < 0.0:
            raise ValueError("alpha_hi must be >= 0.0")
        if alpha_hs < 0.0:
            raise ValueError("alpha_hs must be >= 0.0")
        if hi_scale <= 0.0:
            raise ValueError("hi_scale must be > 0.0")
        if hs_scale <= 0.0:
            raise ValueError("hs_scale must be > 0.0")
        if w_min < 0.0 or w_min > 1.0:
            raise ValueError("w_min must be in [0, 1]")
        if freezing_tolerance < 0.0:
            raise ValueError("freezing_tolerance must be >= 0.0")

        self.input_names = list(input_names)
        self.input_levels = resolved_input_levels
        self.output_names = list(output_names)
        self.output_levels = resolved_output_levels
        self.alpha_t = float(alpha_t)
        self.alpha_hi = float(alpha_hi)
        self.alpha_hs = float(alpha_hs)
        self.hi_scale = float(hi_scale)
        self.hs_scale = float(hs_scale)
        self.w_min = float(w_min)
        self.tf0 = float(tf0)
        self.tf_s_linear = float(tf_s_linear)
        self.tf_s_pow = float(tf_s_pow)
        self.freezing_tolerance = float(freezing_tolerance)

    def _freezing_temperature(self, sss_bg: torch.Tensor) -> torch.Tensor:
        sss_eff = torch.clamp(sss_bg, min=0.0)
        return self.tf0 + self.tf_s_linear * sss_eff + self.tf_s_pow * sss_eff * torch.sqrt(
            sss_eff
        )

    def _compute_jacobian(
        self,
        sst_bg: torch.Tensor,
        sss_bg: torch.Tensor,
        hi_bg: torch.Tensor,
        hs_bg: torch.Tensor,
        aice_bg: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        a = torch.clamp(aice_bg, 0.0, 1.0)
        w = self.w_min + (1.0 - self.w_min) * 4.0 * a * (1.0 - a)

        sss_eff = torch.clamp(sss_bg, min=0.0)
        tf_bg = self._freezing_temperature(sss_bg)
        dTf_dS = self.tf_s_linear + 1.5 * self.tf_s_pow * torch.sqrt(sss_eff)
        active = (torch.abs(sst_bg - tf_bg) <= self.freezing_tolerance).to(mask.dtype)
        effective_mask = mask * active.unsqueeze(1)

        hi_eff = torch.clamp(hi_bg, min=0.0)
        hs_eff = torch.clamp(hs_bg, min=0.0)

        d_aice_d_sst = -self.alpha_t * w
        d_aice_d_sss = self.alpha_t * dTf_dS * w
        d_aice_d_hi = self.alpha_hi * torch.exp(-hi_eff / self.hi_scale) * w
        d_aice_d_hs = self.alpha_hs * torch.exp(-hs_eff / self.hs_scale) * w

        jac = torch.stack(
            [d_aice_d_sst, d_aice_d_sss, d_aice_d_hi, d_aice_d_hs],
            dim=1,
        ).unsqueeze(1)
        return jac * effective_mask.unsqueeze(2)

    def forward(
        self,
        inputs: torch.Tensor,
        perturbations: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the prior-state Jacobian as a linear map to perturbations.

        Args:
            inputs:       [nnodes, 5] packed as [sst_bg, sss_bg, hi_bg, hs_bg, aice_prior].
            perturbations:[nnodes, 4] = [d_sst, d_sss, d_hi, d_hs].

        Returns:
            [nnodes, 1] linearised SIC increment delta_aice = J @ perturbations.
        """
        if inputs.shape[1] != 5:
            raise ValueError("forward expects inputs with shape [nnodes, 5]")
        if perturbations.shape[1] != 4:
            raise ValueError("forward expects perturbations with shape [nnodes, 4]")

        mask = torch.ones(
            inputs.shape[0], 1, dtype=inputs.dtype, device=inputs.device
        )
        jac = self.jac_physical(inputs, mask)  # [nnodes, 1, 4]
        return torch.bmm(jac, perturbations.unsqueeze(2)).squeeze(2)

    @torch.jit.export
    def jac_from_state(
        self,
        state: Dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Jacobian from a dict keyed by input_names entries (Python interface)."""
        sst_bg = state[self.input_names[0]]
        sss_bg = state[self.input_names[1]]
        hi_bg = state[self.input_names[2]]
        hs_bg = state[self.input_names[3]]
        aice_bg = state[self.input_names[4]]
        return self._compute_jacobian(sst_bg, sss_bg, hi_bg, hs_bg, aice_bg, mask)

    @torch.jit.export
    def jac_physical(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Jacobian from the 5-column packed background tensor assembled by SABER C++.

        Args:
            inputs: [nnodes, 5] packed as [sst_bg, sss_bg, hi_bg, hs_bg, aice_prior].
            mask:   [nnodes, 1] binary validity mask (1 = active node).

        Returns:
            jac: [nnodes, 1, 4]  d(aice)/d[sst, sss, hi, hs].
        """
        if inputs.shape[1] != 5:
            raise ValueError("jac_physical expects inputs with shape [nnodes, 5]")

        sst_bg = inputs[:, 0]
        sss_bg = inputs[:, 1]
        hi_bg = inputs[:, 2]
        hs_bg = inputs[:, 3]
        aice_bg = inputs[:, 4]
        return self._compute_jacobian(sst_bg, sss_bg, hi_bg, hs_bg, aice_bg, mask)
