"""Surface ice concentration balance Jacobian for SABER TorchBalance.

The module evaluates a local prior-state Jacobian for sea-ice area fraction:

    daice = J_sst dSST + J_sss dSSS + J_hi dHI + J_hs dHS

Packed background input layout:

    inputs[:, 0] = SST background
    inputs[:, 1] = SSS background
    inputs[:, 2] = sea-ice thickness background
    inputs[:, 3] = sea-ice snow thickness background
    inputs[:, 4] = prior sea-ice area fraction

The returned Jacobian has four columns because the prior ice fraction is only
used to evaluate the local balance strength. It is not a perturbation input.
"""

from typing import List, Optional

import torch
import torch.nn as nn


class SurfaceIceConcentrationJacobian(nn.Module):
    """Prior-state SIC Jacobian provider for SABER TorchBalance."""

    input_names: List[str]
    output_names: List[str]
    input_levels: List[int]
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
    mask_var_name: str
    mask_min: float
    mask_max: float

    def __init__(
        self,
        input_names: List[str],
        output_names: List[str],
        input_levels: List[int],
        output_levels: List[int],
        alpha_t: float = 1.0,
        alpha_hi: float = 0.2,
        alpha_hs: float = 0.1,
        hi_scale: float = 0.5,
        hs_scale: float = 0.1,
        w_min: float = 0.05,
        tf0: float = 0.0901,
        tf_s_linear: float = -0.0575,
        tf_s_pow: float = 1.710523e-3,
        mask_var_name: str = "",
        mask_min: float = 0.0,
        mask_max: float = 1.0,
    ) -> None:
        super().__init__()
        if len(input_names) != 5:
            raise ValueError(
                "SurfaceIceConcentrationJacobian requires 5 input names: "
                "[sst, sss, hi, hs, aice_prior]"
            )
        if len(output_names) != 1:
            raise ValueError(
                "SurfaceIceConcentrationJacobian requires exactly 1 output name"
            )
        if len(input_levels) != len(input_names):
            raise ValueError("input_levels length must match input_names length")
        if len(output_levels) != len(output_names):
            raise ValueError("output_levels length must match output_names length")
        if alpha_t < 0.0:
            raise ValueError("alpha_t must be >= 0")
        if alpha_hi < 0.0:
            raise ValueError("alpha_hi must be >= 0")
        if alpha_hs < 0.0:
            raise ValueError("alpha_hs must be >= 0")
        if hi_scale <= 0.0:
            raise ValueError("hi_scale must be > 0")
        if hs_scale <= 0.0:
            raise ValueError("hs_scale must be > 0")
        if w_min < 0.0 or w_min > 1.0:
            raise ValueError("w_min must be in [0, 1]")

        self.input_names = list(input_names)
        self.output_names = list(output_names)
        self.input_levels = list(input_levels)
        self.output_levels = list(output_levels)
        self.alpha_t = float(alpha_t)
        self.alpha_hi = float(alpha_hi)
        self.alpha_hs = float(alpha_hs)
        self.hi_scale = float(hi_scale)
        self.hs_scale = float(hs_scale)
        self.w_min = float(w_min)
        self.tf0 = float(tf0)
        self.tf_s_linear = float(tf_s_linear)
        self.tf_s_pow = float(tf_s_pow)
        self.mask_var_name = mask_var_name
        self.mask_min = float(mask_min)
        self.mask_max = float(mask_max)

    def _check_inputs(self, inputs: torch.Tensor, label: str) -> None:
        if inputs.dim() != 2 or inputs.shape[1] != 5:
            raise ValueError(label + " expects inputs with shape [nnodes, 5]")

    def _ice_activity_weight(self, aice_bg: torch.Tensor) -> torch.Tensor:
        a = torch.clamp(aice_bg, 0.0, 1.0)
        edge = 4.0 * a * (1.0 - a)
        return self.w_min + (1.0 - self.w_min) * edge

    def _d_freezing_point_d_sss(self, sss_bg: torch.Tensor) -> torch.Tensor:
        sss_eff = torch.clamp(sss_bg, min=0.0)
        return self.tf_s_linear + 1.5 * self.tf_s_pow * torch.sqrt(sss_eff)

    def _compute_jacobian(
        self,
        sst_bg: torch.Tensor,
        sss_bg: torch.Tensor,
        hi_bg: torch.Tensor,
        hs_bg: torch.Tensor,
        aice_bg: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        w = self._ice_activity_weight(aice_bg)
        dTf_dS = self._d_freezing_point_d_sss(sss_bg)
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
        return jac * mask.unsqueeze(2)

    def forward(
        self,
        inputs: torch.Tensor,
        perturbations: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the local SIC Jacobian as a linear map.

        Args:
            inputs: [nnodes, 5] background tensor used to evaluate the Jacobian.
            perturbations: optional [nnodes, 4] tensor [dSST, dSSS, dHI, dHS].
                If omitted, inputs[:, :4] is used as the mapped vector.

        Returns:
            [nnodes, 1] balanced SIC increment.
        """
        self._check_inputs(inputs, "forward")
        if perturbations is None:
            dx = inputs[:, 0:4]
        else:
            if perturbations.dim() != 2 or perturbations.shape[1] != 4:
                raise ValueError(
                    "forward expects perturbations with shape [nnodes, 4]"
                )
            dx = perturbations

        mask_var = torch.ones(
            inputs.shape[0], 1, dtype=inputs.dtype, device=inputs.device
        )
        jac = self.jac_physical(inputs, mask_var)
        return torch.bmm(jac, dx.unsqueeze(2)).squeeze(2)

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
    ) -> torch.Tensor:
        """Return d(aice) / d[sst, sss, hi, hs] for SABER C++."""
        self._check_inputs(inputs, "jac_physical")
        mask = self.compute_mask(mask_var)
        sst_bg  = inputs[:, 0]
        sss_bg  = inputs[:, 1]
        hi_bg   = inputs[:, 2]
        hs_bg   = inputs[:, 3]
        aice_bg = inputs[:, 4]
        jac = self._compute_jacobian(sst_bg, sss_bg, hi_bg, hs_bg, aice_bg, mask)
        return torch.where(torch.isfinite(jac), jac, torch.zeros_like(jac))
