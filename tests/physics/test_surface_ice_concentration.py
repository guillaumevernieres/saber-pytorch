from typing import List

import torch

from saber_pytorch.physics.surface_ice_concentration import (
    SurfaceIceConcentrationJacobian,
)


INPUT_NAMES: List[str] = [
    "sea_water_potential_temperature",
    "sea_water_salinity",
    "sea_ice_thickness",
    "sea_ice_snow_thickness",
    "sea_ice_area_fraction",
]
OUTPUT_NAMES: List[str] = ["sea_ice_area_fraction"]
INPUT_LEVELS: List[int] = [0, 0, 0, 0, 0]
OUTPUT_LEVELS: List[int] = [0]


def make_emulator() -> SurfaceIceConcentrationJacobian:
    return SurfaceIceConcentrationJacobian(
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        input_levels=INPUT_LEVELS,
        output_levels=OUTPUT_LEVELS,
    ).eval()


def make_inputs() -> torch.Tensor:
    return torch.tensor(
        [
            [-1.8, 34.0, 0.1, 0.02, 0.0],
            [-1.7, 35.0, 0.5, 0.10, 0.5],
            [-1.6, 36.0, 2.0, 0.40, 1.0],
        ],
        dtype=torch.float32,
    )


def test_jac_physical_shape():
    em = make_emulator()
    jac = em.jac_physical(make_inputs(), torch.ones(3, 1))
    assert jac.shape == (3, 1, 4)


def test_jac_physical_signs():
    em = make_emulator()
    jac = em.jac_physical(make_inputs(), torch.ones(3, 1))[:, 0, :]
    assert torch.all(jac[:, 0] <= 0.0)
    assert torch.all(jac[:, 1] <= 0.0)
    assert torch.all(jac[:, 2] >= 0.0)
    assert torch.all(jac[:, 3] >= 0.0)


def test_masked_rows_are_zero():
    em = make_emulator()
    mask = torch.tensor([[1.0], [0.0], [1.0]])
    jac = em.jac_physical(make_inputs(), mask)
    assert torch.all(jac[1] == 0.0)
    assert not torch.all(jac[0] == 0.0)
    assert not torch.all(jac[2] == 0.0)


def test_ice_activity_weight_extrema():
    em = make_emulator()
    inputs = make_inputs()
    jac = em.jac_physical(inputs, torch.ones(3, 1))
    sst_sensitivity = -jac[:, 0, 0]

    assert torch.allclose(sst_sensitivity[0], torch.tensor(em.w_min))
    assert torch.allclose(sst_sensitivity[1], torch.tensor(1.0))
    assert torch.allclose(sst_sensitivity[2], torch.tensor(em.w_min))


def test_hi_and_hs_sensitivities_decay_with_thickness():
    em = make_emulator()
    inputs = make_inputs()
    inputs[:, 4] = 0.5
    jac = em.jac_physical(inputs, torch.ones(3, 1))[:, 0, :]

    assert jac[0, 2] > jac[1, 2] > jac[2, 2]
    assert jac[0, 3] > jac[1, 3] > jac[2, 3]


def test_forward_applies_jacobian_to_supplied_perturbations():
    em = make_emulator()
    inputs = make_inputs()
    dx = torch.tensor(
        [
            [0.2, -0.1, 0.05, 0.01],
            [0.0, 0.3, -0.2, 0.04],
            [-0.1, 0.2, 0.0, -0.03],
        ],
        dtype=torch.float32,
    )
    jac = em.jac_physical(inputs, torch.ones(3, 1))
    expected = torch.bmm(jac, dx.unsqueeze(2)).squeeze(2)

    assert torch.allclose(em(inputs, dx), expected)


def test_forward_defaults_to_first_four_input_columns():
    em = make_emulator()
    inputs = make_inputs()
    jac = em.jac_physical(inputs, torch.ones(3, 1))
    expected = torch.bmm(jac, inputs[:, :4].unsqueeze(2)).squeeze(2)

    assert torch.allclose(em(inputs), expected)


def test_torchscript_jacobian_and_forward():
    em = make_emulator()
    scripted = torch.jit.script(em)
    inputs = make_inputs()
    dx = torch.randn(3, 4)
    jac = scripted.jac_physical(inputs, torch.ones(3, 1))
    y = scripted(inputs, dx)
    expected = torch.bmm(jac, dx.unsqueeze(2)).squeeze(2)

    assert jac.shape == (3, 1, 4)
    assert torch.allclose(y, expected)
