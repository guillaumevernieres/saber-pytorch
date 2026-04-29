# Localized Heave Salinity Jacobian Demo

This directory demonstrates the physical heave balance Jacobian using the 1-degree `geom100` ocean background:

`../i-jedi/test-soca/geom100/MOM.res.nc`

Run:

```bash
python emulators/heave_salinity/scripts/plot_geom100_jacobian.py \
  --config emulators/heave_salinity/config.yaml
```

The demo intentionally does not save the full dense Jacobian for every grid point. For 75 levels and three packed input fields, that would be very large. Instead it writes compact diagnostics:

- `geom100_heave_column_matrix.png`: full output-level by input-temperature-level Jacobian for one representative column.
- `geom100_heave_diag_maps.png`: maps of diagonal sensitivity $\partial \delta S_k / \partial \delta T_k$ at selected levels.
- `geom100_heave_vertical_profiles.png`: mean and high-percentile vertical summaries of the diagonal and row-sum absolute sensitivity.
- `geom100_heave_summary.csv`: per-level summary statistics.

The plotted Jacobian is the temperature block of the vertical TorchBalance Jacobian. Salinity and thickness input blocks are zero for this physical balance.

## Mixed-Layer Noise Control

The physical balance uses `model.epsilon_taper` to damp the T/S slope where the
vertical temperature gradient is weak.  To keep a permissive full-column value
while damping mixed-layer noise more strongly, enable:

```yaml
model:
  suppress_shallow_weak_stratification: true
  shallow_taper_depth_m: 50.0
  shallow_epsilon_taper: 1.0e-4
```

This applies the stronger `shallow_epsilon_taper` only to layer centres shallower
than `shallow_taper_depth_m`; deeper levels continue to use `epsilon_taper`.

## Single-Profile Reconstruction

To test whether the heave Jacobian can reconstruct a nearby truth salinity profile from a background profile:

```bash
python emulators/heave_salinity/scripts/reconstruct_salinity_profile.py \
  --config emulators/heave_salinity/config.yaml \
  --lon -119.5 \
  --lat 4.36
```

The scripts use `mom6_structured_grid.nc` to find the nearest valid grid point
to the configured or requested lon/lat. You should not need to enter horizontal
grid indices by hand.

The reconstruction script then chooses a nearby valid background column, applies:

$$
S_{\mathrm{heave}} = S_{\mathrm{bg}} + K_{ST}(T_{\mathrm{bg}}, S_{\mathrm{bg}}, h_{\mathrm{bg}})
(T_{\mathrm{truth}} - T_{\mathrm{bg}})
$$

and writes a profile plot, CSV, and NPZ under `outputs/`.

If `ml_salinity.enabled` is true, the same plot also includes a green
ML-Jacobian salinity estimate. This is computed on the trained ML emulator's
reduced vertical grid:

$$
S_{\mathrm{ML}} =
S_{\mathrm{bg,reduced}} +
J_{\mathrm{ML}}(T_{\mathrm{bg,reduced}}, h_{\mathrm{bg,reduced}})
(T_{\mathrm{truth,reduced}} - T_{\mathrm{bg,reduced}})
$$

The script writes a separate `*_ml_reduced.csv` file with the reduced-grid
depths and ML salinity values.

By default the script first tries the configured offset, then searches outward
for a valid background column whose surface temperature differs from the truth
by at least `background_min_surface_temperature_difference` in `config.yaml`
(currently `1.0` degC). You can change that with `--background-offset-iy`,
`--background-offset-ix`, `--background-search-radius`,
`--background-min-grid-distance`, and `--background-min-surface-temp-diff`.

## Argo Profile Reconstruction

To repeat the same investigation with real matched Argo T/S profile pairs, first
build the reusable profile dataset:

```bash
python scripts/build_real_argo_ts_profiles.py
```

By default this reads all paired 00Z Argo temperature/salinity diagnostics
matching:

```text
/home/gvernier/Documents/gomo/gomo-mom6/gdas.202604*/00/analysis/ocean/diags/insitu_{temp,salt}_profile_argo.nc
```

Then run:

```bash
python emulators/heave_salinity/scripts/reconstruct_argo_salinity_profile.py \
  --config emulators/heave_salinity/config.yaml \
  --lon -160 \
  --lat 30
```

The script uses the nearest retained Argo profile as truth and chooses a nearby
retained Argo profile as background, requiring a minimum common valid depth
range and a configurable shallow temperature contrast. It writes an Argo profile
plot, CSV, summary CSV, NPZ, and, when enabled, the ML reduced-grid CSV under
`outputs/`.
