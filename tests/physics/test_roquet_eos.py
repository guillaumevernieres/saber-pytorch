"""Tests for RoquetEOS.

Pressure is now an explicit argument to every method (no stored buffer).

Validates:
1. Density values against known reference points
2. Derivative self-consistency via finite differences
3. TorchScript compilation
"""

import torch

from saber_pytorch.physics.roquet_eos import RoquetEOS


NLEVELS = 4
PRESSURE = torch.tensor([5.0, 15.0, 35.0, 75.0])  # dbar


def make_eos() -> RoquetEOS:
    return RoquetEOS()


def test_density_shape_1d():
    eos = make_eos()
    T = torch.full((NLEVELS,), 10.0)
    S = torch.full((NLEVELS,), 35.0)
    rho = eos.density(T, S, PRESSURE)
    assert rho.shape == (NLEVELS,)


def test_density_shape_2d():
    eos = make_eos()
    T = torch.full((8, NLEVELS), 10.0)
    S = torch.full((8, NLEVELS), 35.0)
    p = PRESSURE.unsqueeze(0).expand(8, -1)
    rho = eos.density(T, S, p)
    assert rho.shape == (8, NLEVELS)


def test_density_at_reference():
    """At T=10°C, S=35 PSU, near-surface, density should be close to 1026 kg/m³."""
    eos = make_eos()
    T = torch.tensor([10.0])
    S = torch.tensor([35.0])
    p = torch.tensor([5.0])
    rho = eos.density(T, S, p)
    assert 1025.0 < rho.item() < 1028.0, f"Unexpected density: {rho.item()}"


def test_pressure_broadcast_1d_to_2d():
    """1D pressure should broadcast correctly across the batch dimension."""
    eos = make_eos()
    T = torch.full((8, NLEVELS), 10.0)
    S = torch.full((8, NLEVELS), 35.0)
    # 1D pressure [nlevels] should broadcast to [8, nlevels]
    rho_broadcast = eos.density(T, S, PRESSURE)
    rho_explicit  = eos.density(T, S, PRESSURE.unsqueeze(0).expand(8, -1))
    assert torch.allclose(rho_broadcast, rho_explicit)


def test_derivatives_finite_difference():
    """Analytical derivatives should match finite differences (float64 for precision)."""
    eos = make_eos()
    T0 = torch.tensor([[10.0, 12.0, 8.0, 6.0]], dtype=torch.float64)
    S0 = torch.tensor([[35.0, 34.5, 35.5, 36.0]], dtype=torch.float64)
    p  = PRESSURE.to(torch.float64)
    eps = 1e-4

    rho_th, rho_s = eos.derivatives(T0, S0, p)

    for k in range(NLEVELS):
        dT = torch.zeros_like(T0)
        dT[0, k] = eps
        fd = (eos.density(T0 + dT, S0, p)[0, k] - eos.density(T0 - dT, S0, p)[0, k]) / (2 * eps)
        assert abs(rho_th[0, k].item() - fd.item()) < 1e-5 * abs(fd.item()) + 1e-12, (
            f"rho_th mismatch at level {k}"
        )

    for k in range(NLEVELS):
        dS = torch.zeros_like(S0)
        dS[0, k] = eps
        fd = (eos.density(T0, S0 + dS, p)[0, k] - eos.density(T0, S0 - dS, p)[0, k]) / (2 * eps)
        assert abs(rho_s[0, k].item() - fd.item()) < 1e-5 * abs(fd.item()) + 1e-12, (
            f"rho_s mismatch at level {k}"
        )


def test_salinity_clamp():
    """Slightly negative salinity should not produce NaN (guarding sqrt)."""
    eos = make_eos()
    T = torch.full((NLEVELS,), 10.0)
    S = torch.tensor([-0.001, 35.0, 35.0, 35.0])
    rho = eos.density(T, S, PRESSURE)
    assert not torch.any(torch.isnan(rho))


def test_torchscript_compilable():
    eos = make_eos()
    scripted = torch.jit.script(eos)
    T = torch.full((3, NLEVELS), 10.0)
    S = torch.full((3, NLEVELS), 35.0)
    p = PRESSURE.unsqueeze(0).expand(3, -1)
    rho_th, rho_s = scripted.derivatives(T, S, p)
    assert rho_th.shape == (3, NLEVELS)
    assert rho_s.shape == (3, NLEVELS)
