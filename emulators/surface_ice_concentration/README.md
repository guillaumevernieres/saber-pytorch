# Surface Ice Concentration Emulator

Analytic (non-ML) emulator that provides the Jacobian of sea ice area fraction (aice)
with respect to prior-state inputs: SST, SSS, and sea ice thickness.

The module is TorchScript-compatible and satisfies the SABER TorchBalance surface
contract for C++ integration.

## Build

```bash
python emulators/surface_ice_concentration/scripts/build_surface_ice_concentration_emulator.py \
    --output surface_ice_concentration.ts
```

Run with `--help` to see all variable-name and level flags.

## Source

`src/saber_pytorch/physics/ice_concentration.py` — `SurfaceIceConcentrationEmulator`
