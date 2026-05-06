"""Tests for SurfaceIceConcentrationEmulator (prior-state Jacobian provider)."""

import tempfile
from pathlib import Path
from typing import Dict, List

import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from saber_pytorch.physics.ice_concentration import SurfaceIceConcentrationEmulator  # noqa: E402

NNODES = 12
SST_NAME = "sea_surface_temperature"
SSS_NAME = "sea_surface_salinity"
HI_NAME = "sea_ice_thickness"
HS_NAME = "surface_snow_thickness"
AICE_NAME = "sea_ice_area_fraction"
OUT_NAME = "sea_ice_area_fraction"


def make_emulator() -> SurfaceIceConcentrationEmulator:
    input_names: List[str] = [SST_NAME, SSS_NAME, HI_NAME, HS_NAME, AICE_NAME]
    output_names: List[str] = [OUT_NAME]
    return SurfaceIceConcentrationEmulator(
        input_names=input_names,
        output_names=output_names,
    )


def make_state(
    nnodes: int = NNODES,
    aice_val: float = 0.5,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    sst = torch.rand(nnodes, dtype=dtype) * 6.0 - 2.0
    sss = torch.rand(nnodes, dtype=dtype) * 3.0 + 32.0
    hi = torch.rand(nnodes, dtype=dtype) * 3.0
    hs = torch.rand(nnodes, dtype=dtype) * 0.6
    aice = torch.full((nnodes,), aice_val, dtype=dtype)
    return {SST_NAME: sst, SSS_NAME: sss, HI_NAME: hi, HS_NAME: hs, AICE_NAME: aice}


def state_to_packed(state: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack(
        [state[SST_NAME], state[SSS_NAME], state[HI_NAME], state[HS_NAME], state[AICE_NAME]],
        dim=1,
    )


# ------------------------------------------------------------------
# Shape and output contract
# ------------------------------------------------------------------

def test_jac_physical_shape():
    em = make_emulator()
    inputs = state_to_packed(make_state())
    mask = torch.ones(NNODES, 1)
    jac = em.jac_physical(inputs, mask)
    assert jac.shape == (NNODES, 1, 4)


def test_jac_from_state_matches_jac_physical():
    em = make_emulator()
    state = make_state()
    mask = torch.ones(NNODES, 1)
    jac_dict = em.jac_from_state(state, mask)
    jac_packed = em.jac_physical(state_to_packed(state), mask)
    assert torch.allclose(jac_dict, jac_packed)


def test_masked_jacobian_is_zero():
    em = make_emulator()
    inputs = state_to_packed(make_state())
    mask = torch.zeros(NNODES, 1)
    jac = em.jac_physical(inputs, mask)
    assert torch.all(jac == 0.0)


# ------------------------------------------------------------------
# aice_prior weight behaviour
# ------------------------------------------------------------------

def test_aice_prior_half_gives_max_weight():
    """w is maximised at aice_prior = 0.5."""
    em = make_emulator()
    mask = torch.ones(1, 1)
    base = torch.tensor([[-1.0, 34.0, 0.5, 0.1]])

    jac_half = em.jac_physical(
        torch.cat([base, torch.tensor([[0.5]])], dim=1), mask
    )
    jac_zero = em.jac_physical(
        torch.cat([base, torch.tensor([[0.0]])], dim=1), mask
    )
    jac_one = em.jac_physical(
        torch.cat([base, torch.tensor([[1.0]])], dim=1), mask
    )
    # Magnitude at a=0.5 must be >= a=0 and a=1 for d/dSST (which is < 0)
    assert jac_half.abs().sum() >= jac_zero.abs().sum() - 1e-6
    assert jac_half.abs().sum() >= jac_one.abs().sum() - 1e-6


def test_aice_prior_extremes_give_w_min():
    """At aice_prior = 0 and 1, all sensitivities should scale by w_min."""
    em = make_emulator()
    mask = torch.ones(1, 1)
    base = torch.tensor([[-1.0, 34.0, 0.5, 0.1]])

    jac_a0 = em.jac_physical(torch.cat([base, torch.tensor([[0.0]])], dim=1), mask)
    jac_a1 = em.jac_physical(torch.cat([base, torch.tensor([[1.0]])], dim=1), mask)
    # At a=0 and a=1 the weight is exactly w_min; magnitudes should match
    assert torch.allclose(jac_a0.abs(), jac_a1.abs(), atol=1e-6)


# ------------------------------------------------------------------
# Sign constraints
# ------------------------------------------------------------------

def test_jacobian_sign_sst_negative():
    em = make_emulator()
    state = make_state(dtype=torch.float64)
    mask = torch.ones(NNODES, 1, dtype=torch.float64)
    jac = em.jac_from_state(state, mask)
    assert torch.all(jac[:, 0, 0] <= 0.0)   # d/dSST <= 0


def test_jacobian_sign_hi_positive():
    em = make_emulator()
    state = make_state(dtype=torch.float64)
    mask = torch.ones(NNODES, 1, dtype=torch.float64)
    jac = em.jac_from_state(state, mask)
    assert torch.all(jac[:, 0, 2] >= -1e-14)   # d/dHI >= 0


def test_jacobian_sign_hs_positive():
    em = make_emulator()
    state = make_state(dtype=torch.float64)
    mask = torch.ones(NNODES, 1, dtype=torch.float64)
    jac = em.jac_from_state(state, mask)
    assert torch.all(jac[:, 0, 3] >= -1e-14)   # d/dHS >= 0


# ------------------------------------------------------------------
# Thickness / snow saturation (exponential decay)
# ------------------------------------------------------------------

def test_d_aice_d_hi_decreases_with_hi():
    """d(aice)/dHI must decrease as HI increases (exponential saturation)."""
    em = make_emulator()
    mask = torch.ones(1, 1)
    aice = 0.5
    sst, sss, hs = -1.0, 34.0, 0.1
    hi_low = 0.1
    hi_high = 2.0

    jac_lo = em.jac_physical(
        torch.tensor([[sst, sss, hi_low, hs, aice]]), mask
    )
    jac_hi = em.jac_physical(
        torch.tensor([[sst, sss, hi_high, hs, aice]]), mask
    )
    assert jac_lo[0, 0, 2].item() > jac_hi[0, 0, 2].item()


def test_d_aice_d_hs_decreases_with_hs():
    """d(aice)/dHS must decrease as HS increases (exponential saturation)."""
    em = make_emulator()
    mask = torch.ones(1, 1)
    aice = 0.5
    sst, sss, hi = -1.0, 34.0, 0.5
    hs_low = 0.01
    hs_high = 0.5

    jac_lo = em.jac_physical(
        torch.tensor([[sst, sss, hi, hs_low, aice]]), mask
    )
    jac_hi = em.jac_physical(
        torch.tensor([[sst, sss, hi, hs_high, aice]]), mask
    )
    assert jac_lo[0, 0, 3].item() > jac_hi[0, 0, 3].item()


# ------------------------------------------------------------------
# Freezing-point cutoff
# ------------------------------------------------------------------


def test_jacobian_zero_when_far_from_freezing_point():
    em = make_emulator()
    mask = torch.ones(1, 1)
    sss = 34.0
    far_warm_sst = 2.0
    jac = em.jac_physical(
        torch.tensor([[far_warm_sst, sss, 0.5, 0.1, 0.5]]), mask
    )
    assert torch.all(jac == 0.0)



def test_jacobian_nonzero_when_near_freezing_point():
    em = make_emulator()
    mask = torch.ones(1, 1)
    sss = torch.tensor(34.0)
    tf = em.tf0 + em.tf_s_linear * sss + em.tf_s_pow * sss * torch.sqrt(sss)
    near_sst = tf.item() + 0.1
    jac = em.jac_physical(
        torch.tensor([[near_sst, sss.item(), 0.5, 0.1, 0.5]]), mask
    )
    assert torch.any(jac != 0.0)


# ------------------------------------------------------------------
# forward() linear-map contract
# ------------------------------------------------------------------

def test_forward_equals_bmm_of_jac_and_perturbation():
    em = make_emulator()
    state = make_state(nnodes=4)
    inputs = state_to_packed(state)
    perturb = torch.randn(4, 4)
    mask = torch.ones(4, 1)

    delta_from_forward = em(inputs, perturb)
    jac = em.jac_physical(inputs, mask)
    delta_from_bmm = torch.bmm(jac, perturb.unsqueeze(2)).squeeze(2)

    assert torch.allclose(delta_from_forward, delta_from_bmm)


def test_forward_output_shape():
    em = make_emulator()
    inputs = state_to_packed(make_state())
    perturb = torch.randn(NNODES, 4)
    out = em(inputs, perturb)
    assert out.shape == (NNODES, 1)


# ------------------------------------------------------------------
# TorchScript
# ------------------------------------------------------------------

def test_torchscript_compilable():
    em = make_emulator()
    scripted = torch.jit.script(em)
    inputs = state_to_packed(make_state())
    mask = torch.ones(NNODES, 1)
    jac = scripted.jac_physical(inputs, mask)
    assert jac.shape == (NNODES, 1, 4)


def test_torchscript_save_load_roundtrip():
    em = make_emulator()
    scripted = torch.jit.script(em)
    inputs = state_to_packed(make_state())
    mask = torch.ones(NNODES, 1)
    jac_before = scripted.jac_physical(inputs, mask)

    with tempfile.NamedTemporaryFile(suffix=".ts") as f:
        scripted.save(f.name)
        loaded = torch.jit.load(f.name)
        jac_after = loaded.jac_physical(inputs, mask)

    assert torch.allclose(jac_before, jac_after)
    assert loaded.input_names == [SST_NAME, SSS_NAME, HI_NAME, HS_NAME, AICE_NAME]
    assert loaded.input_levels == [0, 0, 0, 0, 0]
    assert loaded.output_names == [OUT_NAME]
    assert loaded.output_levels == [0]
