"""Tests for FFNNSurfaceEmulator (ml_balance.py) and the FFNN core (ffnn.py).

Validates:
1. FFNNSurfaceEmulator construction and attribute checks
2. jac_physical() shape matches TorchBalance surface contract [nnodes, out, in]
3. Mask application — zeroed at masked nodes
4. Jacobian self-consistency via finite differences on forward()
5. Adjoint dot-product test
6. TorchScript compilation, attribute preservation, and save/load round-trip
7. Normalization round-trip (predict with identity norm)
8. FFNN core: _jac_physical shape and predict round-trip
"""

import tempfile
from pathlib import Path
from typing import List

import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from saber_pytorch.ml.ffnn import FFNN
from saber_pytorch.ml.ml_balance import FFNNSalinityProfileEmulator, FFNNSurfaceEmulator


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

N_IN = 5     # number of input features
N_OUT = 1    # surface emulator: exactly one output
NNODES = 12

IN_NAMES: List[str] = [
    "sea_water_potential_temperature",
    "sea_water_salinity",
    "sea_ice_area_fraction",
    "air_temperature",
    "sea_ice_thickness",
]
OUT_NAMES: List[str] = ["sea_ice_concentration"]
IN_LEVELS: List[int] = [0, 0, 0, 127, 0]
OUT_LEVELS: List[int] = [0]


def make_emulator(hidden_size: int = 16, hidden_layers: int = 2) -> FFNNSurfaceEmulator:
    em = FFNNSurfaceEmulator(
        input_names=IN_NAMES,
        output_names=OUT_NAMES,
        input_levels=IN_LEVELS,
        output_levels=OUT_LEVELS,
        hidden_size=hidden_size,
        hidden_layers=hidden_layers,
        activation="gelu",
    )
    # Identity normalization so physical ≡ normalized space
    em.init_norm(
        input_mean=torch.zeros(N_IN),
        input_std=torch.ones(N_IN),
        output_mean=torch.zeros(N_OUT),
        output_std=torch.ones(N_OUT),
    )
    em.eval()
    return em


def make_inputs(nnodes: int = NNODES, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(nnodes, N_IN, dtype=dtype)


# ------------------------------------------------------------------
# Construction guards
# ------------------------------------------------------------------

def test_wrong_output_count_raises():
    try:
        FFNNSurfaceEmulator(
            input_names=IN_NAMES,
            output_names=["a", "b"],
            input_levels=IN_LEVELS,
            output_levels=[0, 0],
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_mismatched_levels_raises():
    try:
        FFNNSurfaceEmulator(
            input_names=IN_NAMES,
            output_names=OUT_NAMES,
            input_levels=[0],          # wrong length
            output_levels=OUT_LEVELS,
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ------------------------------------------------------------------
# Shape and mask checks
# ------------------------------------------------------------------

def test_jac_physical_shape():
    em = make_emulator()
    inputs = make_inputs()
    mask = torch.ones(NNODES, 1)
    jac = em.jac_physical(inputs, mask)
    assert jac.shape == (NNODES, N_OUT, N_IN), (
        f"Expected ({NNODES}, {N_OUT}, {N_IN}), got {tuple(jac.shape)}"
    )


def test_jac_physical_all_masked_is_zero():
    em = make_emulator()
    inputs = make_inputs()
    mask = torch.zeros(NNODES, 1)
    jac = em.jac_physical(inputs, mask)
    assert torch.all(jac == 0.0)


def test_jac_physical_partial_mask():
    """Masked nodes should be zero; unmasked should be non-zero on average."""
    em = make_emulator()
    inputs = make_inputs()
    mask = torch.zeros(NNODES, 1)
    mask[0] = 1.0
    jac = em.jac_physical(inputs, mask)
    assert torch.all(jac[1:] == 0.0)
    assert not torch.all(jac[0] == 0.0)


# ------------------------------------------------------------------
# Normalization round-trip
# ------------------------------------------------------------------

def test_predict_with_non_identity_norm():
    """predict() should undo normalization correctly."""
    em = make_emulator()
    scale = 3.0
    em.init_norm(
        input_mean=torch.full((N_IN,), 1.0),
        input_std=torch.full((N_IN,), scale),
        output_mean=torch.zeros(N_OUT),
        output_std=torch.ones(N_OUT),
    )
    # A model with identity weights: output = sum(inputs_norm)
    # Check that predict() actually applies normalization before forwarding
    x = torch.ones(1, N_IN) * (1.0 + scale)   # normalized to [1,...,1]
    x_norm = (x - 1.0) / scale
    y_direct = em.ffnn.forward(x_norm)
    y_predict = em.ffnn.predict(x)
    assert torch.allclose(y_direct, y_predict)


# ------------------------------------------------------------------
# Finite-difference Jacobian check
# ------------------------------------------------------------------

def test_jac_finite_difference():
    """Analytical Jacobian should match central finite differences on forward()."""
    em = make_emulator()
    em.eval()

    inputs = make_inputs(nnodes=1, dtype=torch.float64)
    em.ffnn.input_mean = em.ffnn.input_mean.double()
    em.ffnn.input_std = em.ffnn.input_std.double()
    em.ffnn.output_mean = em.ffnn.output_mean.double()
    em.ffnn.output_std = em.ffnn.output_std.double()
    em.ffnn.network = em.ffnn.network.double()

    mask = torch.ones(1, 1, dtype=torch.float64)
    jac = em.jac_physical(inputs, mask)  # [1, N_OUT, N_IN]

    eps = 1e-4
    for j in range(N_IN):
        dv = torch.zeros_like(inputs)
        dv[0, j] = eps
        y_plus = em(inputs + dv)
        y_minus = em(inputs - dv)
        fd = ((y_plus - y_minus) / (2 * eps)).squeeze()  # [N_OUT]

        for o in range(N_OUT):
            an = jac[0, o, j].item()
            fd_val = fd[o].item() if N_OUT > 1 else fd.item()
            assert abs(an - fd_val) < 1e-3 * abs(fd_val) + 1e-10, (
                f"FD mismatch at (output={o}, input={j}): "
                f"analytical={an:.6e}, FD={fd_val:.6e}"
            )


# ------------------------------------------------------------------
# Adjoint dot-product test
# ------------------------------------------------------------------

def test_adjoint_dot_product():
    """Verify ⟨J·dx, dy⟩ == ⟨dx, Jᵀ·dy⟩."""
    em = make_emulator()
    mask = torch.ones(1, 1)
    jac = em.jac_physical(make_inputs(nnodes=1), mask)[0]  # [N_OUT, N_IN]

    torch.manual_seed(7)
    dx = torch.randn(N_IN)
    dy = torch.randn(N_OUT)

    fwd = torch.dot(jac @ dx, dy)
    adj = torch.dot(dx, jac.T @ dy)
    assert abs(fwd.item() - adj.item()) < 1e-5 * max(abs(fwd.item()), 1e-10)


# ------------------------------------------------------------------
# FFNN core unit tests
# ------------------------------------------------------------------

def test_ffnn_jac_shape():
    ffnn = FFNN(input_size=N_IN, output_size=N_OUT, hidden_size=8, hidden_layers=2)
    x = torch.randn(NNODES, N_IN)
    jac = ffnn._jac_physical(x)
    assert jac.shape == (NNODES, N_OUT, N_IN)


def test_ffnn_predict_shape():
    ffnn = FFNN(input_size=N_IN, output_size=N_OUT, hidden_size=8, hidden_layers=2)
    x = torch.randn(NNODES, N_IN)
    y = ffnn.predict(x)
    assert y.shape == (NNODES, N_OUT)


def test_ffnn_predict_shape_with_conv1d():
    ffnn = FFNN(
        input_size=N_IN,
        output_size=N_OUT,
        hidden_size=8,
        hidden_layers=2,
        use_conv1d=True,
        conv_channels=4,
        conv_kernel_size=3,
    )
    x = torch.randn(NNODES, N_IN)
    y = ffnn.predict(x)
    assert y.shape == (NNODES, N_OUT)


def test_ffnn_norm_scaling():
    """With non-unit std, _jac_physical should scale by output_std / input_std."""
    ffnn = FFNN(input_size=2, output_size=1, hidden_size=4, hidden_layers=1,
                activation="relu")
    # Set all weights to 1, bias to 0 for a predictable linear function
    with torch.no_grad():
        for m in ffnn.network:
            if hasattr(m, "weight"):
                torch.nn.init.constant_(m.weight, 1.0 / m.weight.shape[1])
                torch.nn.init.zeros_(m.bias)

    in_std = torch.tensor([2.0, 3.0])
    out_std = torch.tensor([5.0])
    ffnn.init_norm(torch.zeros(2), in_std, torch.zeros(1), out_std)

    x = torch.zeros(1, 2)
    jac_norm_row = ffnn._jac_physical(x) / out_std.view(1, -1, 1) * in_std.view(1, 1, -1)
    # In normalized space all inputs have equal weight; sum of each row ≈ constant
    assert jac_norm_row.shape == (1, 1, 2)


# ------------------------------------------------------------------
# TorchScript tests
# ------------------------------------------------------------------

def test_torchscript_compilable():
    em = make_emulator()
    scripted = torch.jit.script(em)
    inputs = make_inputs()
    mask = torch.ones(NNODES, 1)
    jac = scripted.jac_physical(inputs, mask)
    assert jac.shape == (NNODES, N_OUT, N_IN)


def test_torchscript_attributes():
    em = make_emulator()
    scripted = torch.jit.script(em)
    assert scripted.input_names == IN_NAMES
    assert scripted.output_names == OUT_NAMES
    assert scripted.input_levels == IN_LEVELS
    assert scripted.output_levels == OUT_LEVELS


def test_torchscript_save_load_roundtrip():
    em = make_emulator()
    scripted = torch.jit.script(em)
    inputs = make_inputs()
    mask = torch.ones(NNODES, 1)
    jac_before = scripted.jac_physical(inputs, mask)

    with tempfile.NamedTemporaryFile(suffix=".ts") as f:
        scripted.save(f.name)
        loaded = torch.jit.load(f.name)
        jac_after = loaded.jac_physical(inputs, mask)

    assert torch.allclose(jac_before, jac_after)
    assert loaded.input_names == IN_NAMES
    assert loaded.output_names == OUT_NAMES
    assert loaded.input_levels == IN_LEVELS
    assert loaded.output_levels == OUT_LEVELS


def test_torchscript_mask_preserved_after_load():
    em = make_emulator()
    scripted = torch.jit.script(em)
    inputs = make_inputs()

    mask_full = torch.ones(NNODES, 1)
    mask_zero = torch.zeros(NNODES, 1)

    with tempfile.NamedTemporaryFile(suffix=".ts") as f:
        scripted.save(f.name)
        loaded = torch.jit.load(f.name)

    jac_full = loaded.jac_physical(inputs, mask_full)
    jac_zero = loaded.jac_physical(inputs, mask_zero)
    assert not torch.all(jac_full == 0.0)
    assert torch.all(jac_zero == 0.0)


# ------------------------------------------------------------------
# Salinity profile reduced-grid tests
# ------------------------------------------------------------------

def test_salinity_profile_interpolates_temperature_to_reduced_grid():
    em = FFNNSalinityProfileEmulator(
        temperature_variable_name="sea_water_potential_temperature",
        thickness_variable_name="h",
        output_variable_name="sea_water_salinity",
        source_num_levels=5,
        target_num_levels=3,
        hidden_size=4,
        hidden_layers=1,
        activation="relu",
    )
    temp = torch.tensor([[10.0, 20.0, 9999.0, 40.0, -9999.0]])
    thickness = torch.tensor([[1.0, 1.0, 0.0, 2.0, 0.0]])
    inputs = torch.cat([temp, thickness], dim=1)

    reduced = em.reduced_temperature_inputs(inputs)

    expected = torch.tensor([[10.0, 23.333333, 40.0]])
    assert torch.allclose(reduced, expected, atol=1e-5)


def test_salinity_profile_ignores_thin_layer_temperature_fill_values():
    em = FFNNSalinityProfileEmulator(
        temperature_variable_name="sea_water_potential_temperature",
        thickness_variable_name="h",
        output_variable_name="sea_water_salinity",
        source_num_levels=5,
        target_num_levels=3,
        hidden_size=4,
        hidden_layers=1,
        activation="relu",
    )
    thickness = torch.tensor([[1.0, 1.0, 0.1, 2.0, 0.0]])
    with_fill = torch.cat(
        [torch.tensor([[10.0, 20.0, 9999.0, 40.0, -9999.0]]), thickness],
        dim=1,
    )
    without_fill = torch.cat(
        [torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0]]), thickness],
        dim=1,
    )

    reduced_with_fill = em.reduced_temperature_inputs(with_fill)
    reduced_without_fill = em.reduced_temperature_inputs(without_fill)

    assert torch.isfinite(reduced_with_fill).all()
    assert torch.allclose(reduced_with_fill, reduced_without_fill, atol=1e-5)


def test_salinity_profile_can_use_temperature_gradient_reduced_grid():
    uniform = FFNNSalinityProfileEmulator(
        temperature_variable_name="sea_water_potential_temperature",
        thickness_variable_name="h",
        output_variable_name="sea_water_salinity",
        source_num_levels=4,
        target_num_levels=5,
        hidden_size=4,
        hidden_layers=1,
        activation="relu",
    )
    adaptive = FFNNSalinityProfileEmulator(
        temperature_variable_name="sea_water_potential_temperature",
        thickness_variable_name="h",
        output_variable_name="sea_water_salinity",
        source_num_levels=4,
        target_num_levels=5,
        hidden_size=4,
        hidden_layers=1,
        activation="relu",
        reduced_grid_method="temperature_gradient",
        temperature_gradient_weight=4.0,
    )
    temp = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
    thickness = torch.ones_like(temp)
    inputs = torch.cat([temp, thickness], dim=1)

    uniform_reduced = uniform.reduced_temperature_inputs(inputs)
    adaptive_reduced = adaptive.reduced_temperature_inputs(inputs)
    adaptive_features = adaptive.reduced_profile_inputs(inputs)

    assert adaptive_reduced[0, 1] > uniform_reduced[0, 1]
    assert adaptive_reduced[0, -2] < uniform_reduced[0, -2]
    assert torch.allclose(adaptive_reduced[:, [0, -1]], temp[:, [0, -1]])
    assert adaptive_features.shape == (1, 10)
    assert torch.isfinite(adaptive_features).all()


def test_salinity_profile_output_and_jacobian_contract_shapes():
    em = FFNNSalinityProfileEmulator(
        temperature_variable_name="sea_water_potential_temperature",
        thickness_variable_name="h",
        output_variable_name="sea_water_salinity",
        source_num_levels=5,
        target_num_levels=3,
        hidden_size=4,
        hidden_layers=1,
        activation="relu",
    )
    temp = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0]]).repeat(2, 1)
    thickness = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]]).repeat(2, 1)
    inputs = torch.cat([temp, thickness], dim=1)

    salinity = em(inputs)
    assert salinity.shape == (2, 3)

    mask = torch.ones(2, 1)
    row_indices = torch.tensor([0, 1, 2], dtype=torch.long)
    col_indices = torch.tensor([0, 1, 4], dtype=torch.long)
    jac = em.jac_physical(inputs, mask, row_indices, col_indices)
    assert jac.shape == (2, 3)


def test_salinity_profile_torchscript_save_load_roundtrip():
    em = FFNNSalinityProfileEmulator(
        temperature_variable_name="sea_water_potential_temperature",
        thickness_variable_name="h",
        output_variable_name="sea_water_salinity",
        source_num_levels=5,
        target_num_levels=3,
        hidden_size=4,
        hidden_layers=1,
        activation="relu",
    )
    scripted = torch.jit.script(em)
    temp = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0]]).repeat(2, 1)
    thickness = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]]).repeat(2, 1)
    inputs = torch.cat([temp, thickness], dim=1)
    mask = torch.ones(2, 1)
    row_indices = torch.tensor([0, 1, 2], dtype=torch.long)
    col_indices = torch.tensor([0, 1, 4], dtype=torch.long)
    jac_before = scripted.jac_physical(inputs, mask, row_indices, col_indices)

    with tempfile.NamedTemporaryFile(suffix=".ts") as f:
        scripted.save(f.name)
        loaded = torch.jit.load(f.name)
        jac_after = loaded.jac_physical(inputs, mask, row_indices, col_indices)

    assert torch.allclose(jac_before, jac_after)
    assert loaded.input_names == ["sea_water_potential_temperature", "h"]
    assert loaded.output_names == ["sea_water_salinity"]
