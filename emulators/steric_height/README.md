# Steric Height Emulator

Physics-based emulator that computes sea surface height (SSH) from temperature (T),
salinity (S), and layer thickness (dz) via the Roquet equation of state.

The module is TorchScript-compatible and satisfies the SABER TorchBalance vertical
contract for C++ integration.

## Build

```bash
python emulators/steric_height/scripts/build_steric_height_vertical_emulator.py \
    --output steric_height_emulator.ts
```

Run with `--help` to see all variable-name flags (`--T-name`, `--S-name`, `--dz-name`,
`--ssh-name`, `--rho0`).

## Source

`src/saber_pytorch/physics/steric_height.py` — `StericHeightEmulator`
