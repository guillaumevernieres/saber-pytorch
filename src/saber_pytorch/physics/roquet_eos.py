"""Roquet EOS for ocean density computation.

Ported from eosall_from_theta.f90 (UNESCO standard). Computes in-situ density
and its partial derivatives w.r.t. potential temperature and salinity.

Pressure is passed explicitly to every method so that the module carries no
state and the caller decides how pressure is sourced (e.g. from a stored
profile, or derived at runtime from depth fields).

Differences from ocean-balance RoquetEOS:
- No base class (no EquationOfState ABC, no LinearEOS/PolynomialEOS)
- No Dict[str, Tensor] interface; returns plain tensors
- No stored pressure buffer; pressure is an explicit argument
- Pressure conditional removed: pressure terms are always applied
- Float32 throughout (matches SABER C++ float32 tensors)
- rho_p derivative removed (not needed for steric height Jacobian)
"""

from typing import Tuple

import torch
import torch.nn as nn


class RoquetEOS(nn.Module):
    """Stateless Roquet EOS — pressure is passed explicitly at call time."""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @torch.jit.export
    def density(
        self, T: torch.Tensor, S: torch.Tensor, pressure: torch.Tensor
    ) -> torch.Tensor:
        """In-situ density [kg/m³].

        Args:
            T, S:     potential temperature and salinity, same shape.
            pressure: pressure [dbar], broadcastable to T shape.
        """
        rho, _, _ = self._rho_and_derivs(T, S, pressure)
        return rho

    @torch.jit.export
    def density_anomaly(
        self,
        T: torch.Tensor,
        S: torch.Tensor,
        pressure: torch.Tensor,
        rho_ref: float = 1025.0,
    ) -> torch.Tensor:
        """Density anomaly rho(T, S, p) - rho_ref."""
        return self.density(T, S, pressure) - rho_ref

    @torch.jit.export
    def derivatives(
        self, T: torch.Tensor, S: torch.Tensor, pressure: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Partial derivatives (∂ρ/∂T, ∂ρ/∂S) evaluated at (T, S, pressure).

        Args:
            T, S:     [batch, nlevels] or [nlevels].
            pressure: broadcastable to T shape.

        Returns:
            (rho_th, rho_s): same shape as T.
        """
        _, rho_th, rho_s = self._rho_and_derivs(T, S, pressure)
        return rho_th, rho_s

    # ------------------------------------------------------------------
    # Internal computation
    # ------------------------------------------------------------------

    def _rho_and_derivs(
        self, T: torch.Tensor, S: torch.Tensor, pressure: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute (rho, drho/dT, drho/dS) with the full Roquet EOS.

        Pressure terms are always included (no zero-pressure shortcut).
        Salinity is clamped ≥ 0 before sqrt to guard increments that drift
        slightly negative.

        pressure is broadcast to match T if it has fewer dimensions.
        """
        p: torch.Tensor = pressure.to(dtype=T.dtype)
        if p.dim() < T.dim():
            # [nlevels] → [1, nlevels] → broadcast to [batch, nlevels]
            p = p.unsqueeze(0).expand_as(T)

        th = T
        s = torch.clamp(S, min=0.0)
        th2 = th * th
        sqrts = torch.sqrt(s)
        pth = p * th

        # ---- numerator and denominator of the rational EOS ----

        anum = (
            9.9984085444849347e02
            + th * (
                7.3471625860981584
                + th * (-5.3211231792841769e-02 + th * 3.6492439109814549e-04)
            )
            + s * (
                2.5880571023991390
                - th * 6.7168282786692355e-03
                + s * 1.9203202055760151e-03
            )
            + p * (
                1.1798263740430364e-02
                + th2 * 9.8920219266399117e-08
                + s * 4.6996642771754730e-06
                - p * (2.5862187075154352e-08 + th2 * 3.2921414007960662e-12)
            )
        )

        aden = (
            1.0
            + th * (
                7.2815210113327091e-03
                + th * (
                    -4.4787265461983921e-05
                    + th * (3.3851002965802430e-07 + th * 1.3651202389758572e-10)
                )
            )
            + s * (
                1.7632126669040377e-03
                - th * (8.8066583251206474e-06 + th2 * 1.8832689434804897e-10)
                + sqrts * (5.7463776745432097e-06 + th2 * 1.4716275472242334e-09)
            )
            + p * (
                6.7103246285651894e-06
                - pth * (th2 * 2.4461698007024582e-17 + p * 9.1534417604289062e-18)
            )
        )

        # ---- numerator/denominator partial derivatives ----

        anum_s = (
            2.5880571023991390
            - th * 6.7168282786692355e-03
            + s * 3.8406404111520300e-03
            + p * 4.6996642771754730e-06
        )

        aden_s = (
            1.7632126669040377e-03
            + th * (-8.8066583251206470e-06 - th2 * 1.8832689434804897e-10)
            + sqrts * (8.6195665118148150e-06 + th2 * 2.2074413208363504e-09)
        )

        anum_th = (
            7.3471625860981580
            + th * (-1.0642246358568354e-01 + th * 1.0947731732944364e-03)
            - s * 6.7168282786692355e-03
            + pth * (1.9784043853279823e-07 - p * 6.5842828015921320e-12)
        )

        aden_th = (
            7.2815210113327090e-03
            + th * (
                -8.9574530923967840e-05
                + th * (1.0155300889740728e-06 + th * 5.4604809559034290e-10)
            )
            + s * (
                -8.8066583251206470e-06
                - th2 * 5.6498068304414700e-10
                + th * sqrts * 2.9432550944484670e-09
            )
            - p * p * (th2 * 7.3385094021073750e-17 + p * 9.1534417604289060e-18)
        )

        # ---- density and derivatives via quotient rule ----

        rec_aden = 1.0 / aden
        rho = anum * rec_aden
        rho_s = (anum_s - aden_s * rho) * rec_aden
        rho_th = (anum_th - aden_th * rho) * rec_aden

        return rho, rho_th, rho_s
