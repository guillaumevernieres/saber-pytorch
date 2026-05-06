"""Tests for WeaverTSBalance."""

import tempfile
from typing import Dict, List

import torch

from saber_pytorch.physics.heave_salinity import WeaverTSBalance

NLEVELS = 6
NNODES = 4
T_NAME = "sea_water_potential_temperature"
S_NAME = "sea_water_salinity"
DZ_NAME = "ocean_layer_thickness"
OUT_NAME = "sea_water_salinity"


def make_emulator(
    amplitude: float = 1.0,
    use_taper: bool = True,
    epsilon_taper: float = 1.0e-3,
    suppress_shallow_weak_stratification: bool = False,
    shallow_taper_depth_m: float = 50.0,
    shallow_epsilon_taper: float = 1.0e-4,
) -> WeaverTSBalance:
    input_names: List[str] = [T_NAME, S_NAME, DZ_NAME]
    output_names: List[str] = [OUT_NAME]
    return WeaverTSBalance(
        input_names=input_names,
        output_names=output_names,
        epsilon=1.0e-12,
        epsilon_taper=epsilon_taper,
        amplitude=amplitude,
        use_temperature_gradient_taper=use_taper,
        suppress_shallow_weak_stratification=suppress_shallow_weak_stratification,
        shallow_taper_depth_m=shallow_taper_depth_m,
        shallow_epsilon_taper=shallow_epsilon_taper,
    )


def make_state(nnodes: int = NNODES) -> Dict[str, torch.Tensor]:
    # Linear T and S profiles with known constant gradients:
    #   T_z = -0.2 / dz  (dz = 10 m  →  T_z = -0.02 °C/m)
    #   S_z = +0.03 / dz (dz = 10 m  →  S_z = +0.003 PSU/m)
    z = torch.arange(NLEVELS, dtype=torch.float64).view(1, -1)
    dz = torch.full((nnodes, NLEVELS), 10.0, dtype=torch.float64)
    T = 20.0 - 0.2 * z
    S = 34.0 + 0.03 * z
    return {
        T_NAME: T.repeat(nnodes, 1),
        S_NAME: S.repeat(nnodes, 1),
        DZ_NAME: dz,
    }


def state_to_packed(state: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([state[T_NAME], state[S_NAME], state[DZ_NAME]], dim=1)


def all_col_indices(nlevels: int = NLEVELS) -> torch.Tensor:
    return torch.arange(3 * nlevels, dtype=torch.long)


def all_row_indices(nlevels: int = NLEVELS) -> torch.Tensor:
    return torch.arange(nlevels, dtype=torch.long).repeat_interleave(
        3 * nlevels
    )


def repeated_col_indices(nlevels: int = NLEVELS) -> torch.Tensor:
    return torch.arange(3 * nlevels, dtype=torch.long).repeat(nlevels)


def test_jac_from_state_shape():
    em = make_emulator()
    state = make_state()
    mask = torch.ones(NNODES, 1, dtype=torch.float64)
    jac = em.jac_from_state(state, mask)
    assert jac.shape == (NNODES, NLEVELS, 3 * NLEVELS)


def test_jac_physical_row_col_selection_matches_full_jacobian():
    em = make_emulator()
    state = make_state()
    mask = torch.ones(NNODES, 1, dtype=torch.float64)
    rows = all_row_indices()
    cols = repeated_col_indices()
    selected = em.jac_physical(state_to_packed(state), mask, rows, cols)
    full = em.jac_from_state(state, mask)
    assert selected.shape == (NNODES, NLEVELS * 3 * NLEVELS)
    assert torch.allclose(selected, full[:, rows, cols])


def test_salinity_and_thickness_jacobian_columns_are_zero():
    em = make_emulator()
    state = make_state()
    mask = torch.ones(NNODES, 1, dtype=torch.float64)
    jac = em.jac_from_state(state, mask)
    assert torch.all(jac[:, :, NLEVELS:] == 0.0)


def test_jacobian_temperature_block_is_diagonal():
    """Off-diagonal entries of the T block must be zero."""
    em = make_emulator()
    state = make_state()
    mask = torch.ones(NNODES, 1, dtype=torch.float64)
    jac = em.jac_from_state(state, mask)
    jac_T = jac[:, :, :NLEVELS]  # [nnodes, nlevels, nlevels]
    # zero out the diagonal; everything remaining must be zero
    diag_mask = torch.eye(NLEVELS, dtype=torch.bool).unsqueeze(0)
    off_diag = jac_T.masked_fill(diag_mask, 0.0)
    assert torch.all(off_diag == 0.0)


def test_synthetic_slope_recovery():
    """With linear T/S and no taper, δS should equal K_ST · δT exactly."""
    em = make_emulator(use_taper=False)
    state = make_state(nnodes=1)
    mask = torch.ones(1, 1, dtype=torch.float64)

    T_z = em._vertical_derivative(state[T_NAME], state[DZ_NAME])
    S_z = em._vertical_derivative(state[S_NAME], state[DZ_NAME])
    T_z2 = T_z * T_z
    K_ST_expected = S_z * T_z / (T_z2 + em.epsilon)

    dT = torch.randn(1, NLEVELS, dtype=torch.float64)
    expected_dS = K_ST_expected * dT
    predicted_dS = em.apply_from_state(state, dT, mask)

    assert torch.allclose(predicted_dS, expected_dS, atol=1.0e-10)


def test_weak_temperature_gradient_is_finite_and_tapered():
    em = make_emulator()
    state = make_state()
    state[T_NAME] = torch.full_like(state[T_NAME], 10.0)
    mask = torch.ones(NNODES, 1, dtype=torch.float64)
    dT = torch.randn(NNODES, NLEVELS, dtype=torch.float64)

    dS = em.apply_from_state(state, dT, mask)

    assert torch.isfinite(dS).all()
    assert torch.allclose(dS, torch.zeros_like(dS))


def test_shallow_weak_stratification_taper_only_changes_shallow_levels():
    base = make_emulator(epsilon_taper=1.0e-6)
    shallow = make_emulator(
        epsilon_taper=1.0e-6,
        suppress_shallow_weak_stratification=True,
        shallow_taper_depth_m=30.0,
        shallow_epsilon_taper=1.0e-4,
    )
    state = make_state(nnodes=1)
    mask = torch.ones(1, 1, dtype=torch.float64)

    base_diag = torch.diagonal(
        base.jac_from_state(state, mask)[:, :, :NLEVELS], dim1=1, dim2=2
    )
    shallow_diag = torch.diagonal(
        shallow.jac_from_state(state, mask)[:, :, :NLEVELS], dim1=1, dim2=2
    )

    assert torch.all(torch.abs(shallow_diag[:, :3]) < torch.abs(base_diag[:, :3]))
    assert torch.allclose(shallow_diag[:, 3:], base_diag[:, 3:])


def test_zero_thickness_layers_are_finite_and_zeroed():
    em = make_emulator()
    state = make_state()
    state[DZ_NAME][:, 2] = 0.0
    mask = torch.ones(NNODES, 1, dtype=torch.float64)
    jac = em.jac_from_state(state, mask)

    assert torch.isfinite(jac).all()
    assert torch.all(jac[:, 2, :] == 0.0)


def test_masked_jacobian_is_zero():
    em = make_emulator()
    state = make_state()
    mask = torch.zeros(NNODES, 1, dtype=torch.float64)
    jac = em.jac_from_state(state, mask)
    assert torch.all(jac == 0.0)


def test_linearity_for_fixed_background():
    em = make_emulator()
    state = make_state()
    mask = torch.ones(NNODES, 1, dtype=torch.float64)
    dT1 = torch.randn(NNODES, NLEVELS, dtype=torch.float64)
    dT2 = torch.randn(NNODES, NLEVELS, dtype=torch.float64)
    a, b = 1.7, -0.4

    lhs = em.apply_from_state(state, a * dT1 + b * dT2, mask)
    rhs = (
        a * em.apply_from_state(state, dT1, mask)
        + b * em.apply_from_state(state, dT2, mask)
    )

    assert torch.allclose(lhs, rhs, atol=1.0e-12)


def test_torchscript_compilable():
    em = make_emulator()
    scripted = torch.jit.script(em)
    state = make_state()
    inputs = state_to_packed(state)
    mask = torch.ones(NNODES, 1, dtype=torch.float64)
    cols = all_col_indices()
    rows = torch.zeros_like(cols)
    jac = scripted.jac_physical(inputs, mask, rows, cols)
    assert jac.shape == (NNODES, 3 * NLEVELS)


def test_torchscript_save_load_roundtrip():
    em = make_emulator()
    scripted = torch.jit.script(em)
    state = make_state()
    inputs = state_to_packed(state)
    mask = torch.ones(NNODES, 1, dtype=torch.float64)
    cols = all_col_indices()
    rows = torch.zeros_like(cols)
    jac_before = scripted.jac_physical(inputs, mask, rows, cols)

    with tempfile.NamedTemporaryFile(suffix=".ts") as f:
        scripted.save(f.name)
        loaded = torch.jit.load(f.name)
        jac_after = loaded.jac_physical(inputs, mask, rows, cols)

    assert torch.allclose(jac_before, jac_after)
    assert loaded.input_names == [T_NAME, S_NAME, DZ_NAME]
    assert loaded.output_names == [OUT_NAME]
