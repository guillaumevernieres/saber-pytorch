#!/usr/bin/env python3
"""Build TorchScript analytic surface ice-concentration emulator.

Usage
-----
python scripts/build_surface_ice_concentration_emulator.py \
    --output surface_ice_concentration.ts
"""

import argparse
import sys
from pathlib import Path
from typing import List

import torch

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from saber_pytorch.physics.ice_concentration import SurfaceIceConcentrationEmulator  # noqa: E402


# input tensor has 5 columns (background state incl. aice_prior);
# jac_physical returns [nnodes, 1, 4] (excludes aice_prior column)
_INPUT_SIZE = 5
_OUTPUT_JAC_COLS = 4


def _parse_names(raw: str, expected: int, label: str) -> List[str]:
    names = [x.strip() for x in raw.split(",") if x.strip()]
    if len(names) != expected:
        raise ValueError(f"--{label}: expected {expected} names, got {len(names)}")
    return names


def build_and_save(
    output_path: str,
    input_names: List[str],
    output_names: List[str],
    alpha_t: float,
    alpha_hi: float,
    alpha_hs: float,
    hi_scale: float,
    hs_scale: float,
    w_min: float,
    tf0: float,
    tf_s_linear: float,
    tf_s_pow: float,
) -> None:
    emulator = SurfaceIceConcentrationEmulator(
        input_names=input_names,
        output_names=output_names,
        alpha_t=alpha_t,
        alpha_hi=alpha_hi,
        alpha_hs=alpha_hs,
        hi_scale=hi_scale,
        hs_scale=hs_scale,
        w_min=w_min,
        tf0=tf0,
        tf_s_linear=tf_s_linear,
        tf_s_pow=tf_s_pow,
    ).eval()

    scripted = torch.jit.script(emulator)
    scripted.save(output_path)

    loaded = torch.jit.load(output_path)
    test_x = torch.randn(4, _INPUT_SIZE)
    test_mask = torch.ones(4, 1)
    jac = loaded.jac_physical(test_x, test_mask)
    expected_shape = (4, 1, _OUTPUT_JAC_COLS)
    if tuple(jac.shape) != expected_shape:
        raise RuntimeError(
            f"Verification failed: expected jac shape {expected_shape}, got {tuple(jac.shape)}"
        )

    print(f"Saved: {output_path}")
    print(f"  input_names  ({_INPUT_SIZE}): {input_names}")
    print(f"  output_names      (1): {output_names}")
    print(
        "  coefficients: "
        f"alpha_t={alpha_t}, alpha_hi={alpha_hi}, alpha_hs={alpha_hs}, "
        f"hi_scale={hi_scale}, hs_scale={hs_scale}, w_min={w_min}, "
        f"tf0={tf0}, tf_s_linear={tf_s_linear}, tf_s_pow={tf_s_pow}"
    )
    print(f"  jac_physical runtime shape: [nnodes, 1, {_OUTPUT_JAC_COLS}]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build TorchScript analytic surface ice concentration emulator"
    )
    parser.add_argument("--output", required=True, help="Output .ts path")
    parser.add_argument(
        "--input-names",
        default="sea_surface_temperature,sea_surface_salinity,sea_ice_thickness,surface_snow_thickness,sea_ice_area_fraction",
        help=(
            "Comma-separated input variable names in packed order [sst,sss,hi,hs,aice_prior]"
        ),
    )
    parser.add_argument(
        "--output-names",
        default="sea_ice_area_fraction",
        help="Comma-separated output variable names (exactly one)",
    )
    parser.add_argument("--alpha-t", type=float, default=1.0)
    parser.add_argument("--alpha-hi", type=float, default=0.2)
    parser.add_argument("--alpha-hs", type=float, default=0.1)
    parser.add_argument("--hi-scale", type=float, default=0.5)
    parser.add_argument("--hs-scale", type=float, default=0.1)
    parser.add_argument("--w-min", type=float, default=0.05)
    parser.add_argument("--tf0", type=float, default=0.0901)
    parser.add_argument("--tf-s-linear", type=float, default=-0.0575)
    parser.add_argument("--tf-s-pow", type=float, default=1.710523e-3)
    args = parser.parse_args()

    build_and_save(
        output_path=args.output,
        input_names=_parse_names(args.input_names, _INPUT_SIZE, "input-names"),
        output_names=_parse_names(args.output_names, 1, "output-names"),
        alpha_t=args.alpha_t,
        alpha_hi=args.alpha_hi,
        alpha_hs=args.alpha_hs,
        hi_scale=args.hi_scale,
        hs_scale=args.hs_scale,
        w_min=args.w_min,
        tf0=args.tf0,
        tf_s_linear=args.tf_s_linear,
        tf_s_pow=args.tf_s_pow,
    )


if __name__ == "__main__":
    main()
