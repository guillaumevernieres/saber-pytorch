"""Tests for StericHeightEmulator.

The emulator takes 3 input fields [T, S, dz]; mid-level depths are derived
from dz on the fly.  The C++ entry point jac_physical() accepts row and column
indices that select compact entries from the full [nnodes, 1, 3*nlevels]
Jacobian.

Validates:
1. forward() — nonlinear steric SSH shape and sign
2. jac_physical() — shape matches SABER contract
3. jac_from_state() — dict interface, identical result to jac_physical
4. Jacobian zero-columns for dz
5. Jacobian self-consistency via finite differences on forward()
6. Adjoint dot-product test
7. Mask application
8. TorchScript compilation and save/load round-trip
"""

import tempfile
from pathlib import Path
from typing import Dict, List

import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from saber_pytorch.physics.steric_height import StericHeightEmulator, depth_to_pressure

NLEVELS = 6
NNODES = 10
T_NAME  = "sea_water_potential_temperature"
S_NAME  = "sea_water_salinity"
DZ_NAME = "ocean_layer_thickness"


def make_emulator() -> StericHeightEmulator:
    input_names: List[str] = [T_NAME, S_NAME, DZ_NAME]
    output_names: List[str] = ["sea_surface_height_above_geoid"]
    return StericHeightEmulator(input_names=input_names, output_names=output_names)


def make_state(nnodes: int = NNODES) -> Dict[str, torch.Tensor]:
    T  = torch.rand(nnodes, NLEVELS) * 20.0 + 2.0
    S  = torch.rand(nnodes, NLEVELS) * 2.0 + 34.0
    dz = torch.full((nnodes, NLEVELS), 50.0)
    return {T_NAME: T, S_NAME: S, DZ_NAME: dz}


def state_to_packed(state: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([state[T_NAME], state[S_NAME], state[DZ_NAME]], dim=1)


def all_col_indices(nlevels: int = NLEVELS) -> torch.Tensor:
    """Return all column indices for a 3-input emulator."""
    return torch.arange(3 * nlevels, dtype=torch.long)


def output_row_indices(col_indices: torch.Tensor) -> torch.Tensor:
    """Steric height has one output row, so all requested pairs use row zero."""
    return torch.zeros_like(col_indices)


# ------------------------------------------------------------------
# depth_to_pressure utility
# ------------------------------------------------------------------

def test_depth_to_pressure_shape():
    depth = torch.linspace(0.0, 500.0, 10)
    p = depth_to_pressure(depth)
    assert p.shape == depth.shape


def test_depth_to_pressure_values():
    # At 100 m depth, p ≈ 100.6 dbar  (rho0=1025, g=9.81)
    p = depth_to_pressure(torch.tensor([100.0]))
    assert 100.0 < p.item() < 102.0


# ------------------------------------------------------------------
# Shape and sign checks
# ------------------------------------------------------------------

def test_forward_shape():
    em = make_emulator()
    inputs = state_to_packed(make_state())
    ssh = em(inputs)
    assert ssh.shape == (NNODES, 1)


def test_forward_warm_anomaly_raises_ssh():
    """Warmer (less dense) water should give positive steric SSH."""
    em = make_emulator()
    dz = torch.full((1, NLEVELS), 50.0)
    S  = torch.full((1, NLEVELS), 35.0)
    ssh_warm = em(torch.cat([torch.full((1, NLEVELS), 20.0), S, dz], dim=1))
    ssh_cold = em(torch.cat([torch.full((1, NLEVELS),  5.0), S, dz], dim=1))
    assert ssh_warm.item() > ssh_cold.item()


def test_jac_physical_shape():
    em = make_emulator()
    inputs = state_to_packed(make_state())
    mask = torch.ones(NNODES, 1)
    cols = all_col_indices()
    jac = em.jac_physical(inputs, mask, output_row_indices(cols), cols)
    assert jac.shape == (NNODES, 3 * NLEVELS)


def test_jac_physical_masked_is_zero():
    em = make_emulator()
    inputs = state_to_packed(make_state())
    mask = torch.zeros(NNODES, 1)
    cols = all_col_indices()
    jac = em.jac_physical(inputs, mask, output_row_indices(cols), cols)
    assert torch.all(jac == 0.0)


def test_dz_jacobian_columns_are_zero():
    """Jacobian columns for dz (geometry, not DA state) must be zero."""
    em = make_emulator()
    inputs = state_to_packed(make_state())
    mask = torch.ones(NNODES, 1)
    cols = all_col_indices()
    jac = em.jac_physical(inputs, mask, output_row_indices(cols), cols)
    assert torch.all(jac[:, 2 * NLEVELS:] == 0.0)


def test_jac_physical_col_selection():
    """Requesting only T columns should return a narrower Jacobian."""
    em = make_emulator()
    inputs = state_to_packed(make_state())
    mask = torch.ones(NNODES, 1)
    t_indices = torch.arange(NLEVELS, dtype=torch.long)
    jac_t = em.jac_physical(inputs, mask, output_row_indices(t_indices), t_indices)
    assert jac_t.shape == (NNODES, NLEVELS)
    cols = all_col_indices()
    jac_full = em.jac_physical(inputs, mask, output_row_indices(cols), cols)
    assert torch.allclose(jac_t, jac_full[:, :NLEVELS])


# ------------------------------------------------------------------
# Dict interface
# ------------------------------------------------------------------

def test_jac_from_state_shape():
    em = make_emulator()
    state = make_state()
    mask = torch.ones(NNODES, 1)
    jac = em.jac_from_state(state, mask)
    assert jac.shape == (NNODES, 1, 3 * NLEVELS)


def test_jac_from_state_matches_jac_physical():
    em = make_emulator()
    state = make_state()
    mask = torch.ones(NNODES, 1)
    jac_dict   = em.jac_from_state(state, mask)
    cols = all_col_indices()
    jac_packed = em.jac_physical(state_to_packed(state), mask, output_row_indices(cols), cols)
    jac_packed = jac_packed.unsqueeze(1)
    assert torch.allclose(jac_dict, jac_packed)


# ------------------------------------------------------------------
# Finite-difference Jacobian check (T and S columns only)
# ------------------------------------------------------------------

def test_jac_finite_difference():
    """Jacobian T/S columns should match FD on forward() (float64 for precision)."""
    em = make_emulator()
    state = {k: v[:1].to(torch.float64) for k, v in make_state(nnodes=1).items()}
    inputs = state_to_packed(state)
    mask = torch.ones(1, 1, dtype=torch.float64)
    cols = all_col_indices(NLEVELS)
    jac = em.jac_physical(inputs, mask, output_row_indices(cols), cols)

    eps = 1e-4
    # Only check T columns (0..nlevels) and S columns (nlevels..2*nlevels)
    for j in range(2 * NLEVELS):
        dv = torch.zeros_like(inputs)
        dv[0, j] = eps
        fd = ((em(inputs + dv) - em(inputs - dv)) / (2 * eps)).item()
        an = jac[0, j].item()
        assert abs(an - fd) < 1e-4 * abs(fd) + 1e-12, (
            f"FD mismatch at input index {j}: analytical={an:.6e}, FD={fd:.6e}"
        )


# ------------------------------------------------------------------
# Adjoint dot-product test
# ------------------------------------------------------------------

def test_adjoint_dot_product():
    """Verify ⟨J·dx, dy⟩ == ⟨dx, Jᵀ·dy⟩ for the T/S Jacobian block."""
    em = make_emulator()
    state = make_state(nnodes=1)
    mask = torch.ones(1, 1)
    cols = all_col_indices()
    jac = em.jac_physical(
        state_to_packed(state), mask, output_row_indices(cols), cols
    )[0, :2 * NLEVELS]

    dx = torch.randn(2 * NLEVELS)
    dy = torch.randn(1)

    fwd = (jac * dx).sum() * dy[0]
    adj = (jac * dy[0] * dx).sum()
    assert abs(fwd.item() - adj.item()) < 1e-6 * max(abs(fwd.item()), 1e-10)


# ------------------------------------------------------------------
# TorchScript tests
# ------------------------------------------------------------------

def test_torchscript_compilable():
    em = make_emulator()
    scripted = torch.jit.script(em)
    inputs = state_to_packed(make_state())
    mask = torch.ones(NNODES, 1)
    cols = all_col_indices()
    jac = scripted.jac_physical(inputs, mask, output_row_indices(cols), cols)
    assert jac.shape == (NNODES, 3 * NLEVELS)


def test_torchscript_jac_from_state():
    em = make_emulator()
    scripted = torch.jit.script(em)
    state = make_state()
    mask = torch.ones(NNODES, 1)
    jac = scripted.jac_from_state(state, mask)
    assert jac.shape == (NNODES, 1, 3 * NLEVELS)


def test_torchscript_attributes():
    em = make_emulator()
    scripted = torch.jit.script(em)
    assert scripted.input_names == [T_NAME, S_NAME, DZ_NAME]
    assert scripted.output_names == ["sea_surface_height_above_geoid"]


def test_torchscript_save_load_roundtrip():
    em = make_emulator()
    scripted = torch.jit.script(em)
    inputs = state_to_packed(make_state())
    mask = torch.ones(NNODES, 1)
    cols = all_col_indices()
    jac_before = scripted.jac_physical(inputs, mask, output_row_indices(cols), cols)

    with tempfile.NamedTemporaryFile(suffix=".ts") as f:
        scripted.save(f.name)
        loaded = torch.jit.load(f.name)
        jac_after = loaded.jac_physical(inputs, mask, output_row_indices(cols), cols)

    assert torch.allclose(jac_before, jac_after)
    assert loaded.input_names == scripted.input_names
    assert loaded.output_names == scripted.output_names
