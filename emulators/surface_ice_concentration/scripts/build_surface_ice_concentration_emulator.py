#!/usr/bin/env python3
"""Build TorchScript analytic surface ice-concentration emulator.

Usage
-----
python emulators/surface_ice_concentration/scripts/build_surface_ice_concentration_emulator.py \
    --output surface_ice_concentration.ts
"""

import argparse
import sys
from pathlib import Path
from typing import List

import torch

_SRC = str(Path(__file__).resolve().parents[3] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from saber_pytorch.physics.ice_concentration import SurfaceIceConcentrationEmulator  # noqa: E402
from saber_pytorch.ml.cf_mappings import CF_ATM, CF_OCN  # noqa: E402

_CF = {**CF_ATM, **CF_OCN}

# Fixed packed order: [sst, sss, hi, hs, aice_prior]
_DEFAULT_INPUT_NAMES  = ",".join([_CF["sst"], _CF["sss"], _CF["hi"], _CF["hs"], _CF["aice"]])
_DEFAULT_OUTPUT_NAMES = _CF["aice"]

# input tensor has 5 columns (background state incl. aice_prior);
# jac_physical returns [nnodes, 1, 4] (excludes aice_prior column)
_INPUT_SIZE = 5
_OUTPUT_JAC_COLS = 4


def _parse_names(raw: str, expected: int, label: str) -> List[str]:
    names = [x.strip() for x in raw.split(",") if x.strip()]
    if len(names) != expected:
        raise ValueError(f"--{label}: expected {expected} names, got {len(names)}")
    return names


def _parse_levels(raw: str, expected: int, label: str) -> List[int]:
    levels = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if len(levels) != expected:
        raise ValueError(f"--{label}: expected {expected} levels, got {len(levels)}")
    return levels


def build_and_save(
    output_path: str,
    input_names: List[str],
    output_names: List[str],
    input_levels: List[int],
    output_levels: List[int],
    alpha_t: float,
    alpha_hi: float,
    alpha_hs: float,
    hi_scale: float,
    hs_scale: float,
    w_min: float,
    tf0: float,
    tf_s_linear: float,
    tf_s_pow: float,
    freezing_tolerance: float,
    mask_var_name: str = "",
    mask_min: float = 0.0,
    mask_max: float = 1.0,
) -> None:
    emulator = SurfaceIceConcentrationEmulator(
        input_names=input_names,
        output_names=output_names,
        input_levels=input_levels,
        output_levels=output_levels,
        alpha_t=alpha_t,
        alpha_hi=alpha_hi,
        alpha_hs=alpha_hs,
        hi_scale=hi_scale,
        hs_scale=hs_scale,
        w_min=w_min,
        tf0=tf0,
        tf_s_linear=tf_s_linear,
        tf_s_pow=tf_s_pow,
        freezing_tolerance=freezing_tolerance,
        mask_var_name=mask_var_name,
        mask_min=mask_min,
        mask_max=mask_max,
    ).eval()

    scripted = torch.jit.script(emulator)
    scripted.save(output_path)

    loaded = torch.jit.load(output_path)
    test_x = torch.randn(4, _INPUT_SIZE)
    if mask_var_name:
        test_mask_var = torch.full((4, 1), (mask_min + mask_max) / 2.0)
    else:
        test_mask_var = torch.ones(4, 1)
    jac = loaded.jac_physical(test_x, test_mask_var)
    expected_shape = (4, 1, _OUTPUT_JAC_COLS)
    if tuple(jac.shape) != expected_shape:
        raise RuntimeError(
            f"Verification failed: expected jac shape {expected_shape}, got {tuple(jac.shape)}"
        )

    print(f"Saved: {output_path}")
    print(f"  input_names  ({_INPUT_SIZE}): {input_names}")
    print(f"  input_levels ({_INPUT_SIZE}): {input_levels}")
    print(f"  output_names      (1): {output_names}")
    print(f"  output_levels     (1): {output_levels}")
    print(f"  mask_var_name        : {mask_var_name or '(none)'}")
    if mask_var_name:
        print(f"  mask range           : [{mask_min}, {mask_max}]")
    print(
        "  coefficients: "
        f"alpha_t={alpha_t}, alpha_hi={alpha_hi}, alpha_hs={alpha_hs}, "
        f"hi_scale={hi_scale}, hs_scale={hs_scale}, w_min={w_min}, "
        f"tf0={tf0}, tf_s_linear={tf_s_linear}, tf_s_pow={tf_s_pow}, "
        f"freezing_tolerance={freezing_tolerance}"
    )
    print(f"  jac_physical runtime shape: [nnodes, 1, {_OUTPUT_JAC_COLS}]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build TorchScript analytic surface ice concentration emulator"
    )
    parser.add_argument("--output", required=True, help="Output .ts path")
    parser.add_argument(
        "--input-names",
        default=_DEFAULT_INPUT_NAMES,
        help=(
            "Comma-separated CF input variable names in packed order "
            "[sst, sss, hi, hs, aice_prior]. Defaults derived from cf_mappings.py."
        ),
    )
    parser.add_argument(
        "--output-names",
        default=_DEFAULT_OUTPUT_NAMES,
        help="Comma-separated CF output variable names (exactly one).",
    )
    parser.add_argument(
        "--input-levels",
        default="0,0,0,0,0",
        help=(
            "Comma-separated input level indices in packed order [sst,sss,hi,hs,aice_prior]"
        ),
    )
    parser.add_argument(
        "--output-levels",
        default="0",
        help="Comma-separated output level indices (exactly one)",
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
    parser.add_argument(
        "--freezing-tolerance",
        type=float,
        default=1.0,
        help="Maximum |sst_bg - Tf(sss_bg)| allowed before the Jacobian is zeroed",
    )
    parser.add_argument(
        "--mask-var", default="",
        help="CF name of the background variable used to gate the Jacobian "
             "(e.g. 'sea_ice_area_fraction'). Omit for no masking.",
    )
    parser.add_argument("--mask-min", type=float, default=0.0,
                        help="Lower bound of the active range (inclusive).")
    parser.add_argument("--mask-max", type=float, default=1.0,
                        help="Upper bound of the active range (inclusive).")
    args = parser.parse_args()

    build_and_save(
        output_path=args.output,
        input_names=_parse_names(args.input_names, _INPUT_SIZE, "input-names"),
        output_names=_parse_names(args.output_names, 1, "output-names"),
        input_levels=_parse_levels(args.input_levels, _INPUT_SIZE, "input-levels"),
        output_levels=_parse_levels(args.output_levels, 1, "output-levels"),
        alpha_t=args.alpha_t,
        alpha_hi=args.alpha_hi,
        alpha_hs=args.alpha_hs,
        hi_scale=args.hi_scale,
        hs_scale=args.hs_scale,
        w_min=args.w_min,
        tf0=args.tf0,
        tf_s_linear=args.tf_s_linear,
        tf_s_pow=args.tf_s_pow,
        freezing_tolerance=args.freezing_tolerance,
        mask_var_name=args.mask_var,
        mask_min=args.mask_min,
        mask_max=args.mask_max,
    )


if __name__ == "__main__":
    main()
