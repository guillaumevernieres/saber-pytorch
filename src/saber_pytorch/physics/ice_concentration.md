# SurfaceIceConcentrationEmulator: Equations and Parameters

This document describes the model implemented in `ice_concentration.py`.

## Purpose

`SurfaceIceConcentrationEmulator` provides an analytic Jacobian for sea-ice concentration (`aice`) with respect to background surface variables.

It is a **linearized balance operator**, not a nonlinear predictor of absolute sea-ice concentration.

## Inputs and Outputs

### Packed input tensor

`inputs` has shape `[nnodes, 5]` and is ordered as:

1. `inputs[:, 0] = sst_bg` (sea surface temperature)
2. `inputs[:, 1] = sss_bg` (sea surface salinity)
3. `inputs[:, 2] = hi_bg` (sea ice thickness)
4. `inputs[:, 3] = hs_bg` (snow thickness on ice)
5. `inputs[:, 4] = aice_prior` (prior sea-ice area fraction)

### Jacobian output

`jac_physical(inputs, mask)` returns shape `[nnodes, 1, 4]`:

- Output row dimension is 1 (`aice`)
- Input column dimension is 4 (`sst, sss, hi, hs`)
- The `aice_prior` column is used internally to build the state-dependent weight, but it is not part of the returned Jacobian columns.

## Core Equations

Let:

- `a = clamp(aice_prior, 0, 1)`
- `w = w_min + (1 - w_min) * 4 * a * (1 - a)`
- `sss_eff = max(sss, 0)`
- `Tf(sss) = tf0 + tf_s_linear * sss_eff + tf_s_pow * sss_eff * sqrt(sss_eff)`
- `dTf_dS = tf_s_linear + 1.5 * tf_s_pow * sqrt(sss_eff)`

The emulator is active only near the local freezing point. Define the freezing mismatch:

- `delta_freeze = abs(sst - Tf(sss))`

and the node-activity flag:

- `active = 1` if `delta_freeze <= freezing_tolerance`, else `0`

Then the partial derivatives are:

- `d(aice)/d(sst) = -alpha_t * w`
- `d(aice)/d(sss) = alpha_t * dTf_dS * w`
- `d(aice)/d(hi)  = alpha_hi * exp(-max(hi,0) / hi_scale) * w`
- `d(aice)/d(hs)  = alpha_hs * exp(-max(hs,0) / hs_scale) * w`

The full Jacobian per node is assembled as:

- `J = [d(aice)/d(sst), d(aice)/d(sss), d(aice)/d(hi), d(aice)/d(hs)]`

with final masking applied by multiplication with `mask * active`.

## Interpretation of the Weight Function

`w(aice_prior)` is a parabola on `[0,1]`:

- maximum at `aice_prior = 0.5`
- minimum value `w_min` at `aice_prior = 0` and `aice_prior = 1`

This means sensitivity is strongest in marginal ice-zone conditions and weakest in fully open water or full ice cover.

## Parameters

Constructor signature:

```python
SurfaceIceConcentrationEmulator(
    input_names,
    output_names,
    input_levels=None,
    output_levels=None,
    alpha_t=1.0,
    alpha_hi=0.2,
    alpha_hs=0.1,
    hi_scale=0.5,
    hs_scale=0.1,
    w_min=0.05,
    tf0=0.0901,
    tf_s_linear=-0.0575,
    tf_s_pow=1.710523e-3,
    freezing_tolerance=1.0,
)
```

### Metadata parameters

- `input_names`: length 5, names for `[sst, sss, hi, hs, aice_prior]`
- `output_names`: length 1, output variable name (`aice`)
- `input_levels`: length 5 level indices (defaults to `[0,0,0,0,0]`)
- `output_levels`: length 1 level index (defaults to `[0]`)

### Physical/empirical parameters

- `alpha_t` (>= 0): temperature sensitivity scale
- `alpha_hi` (>= 0): ice-thickness sensitivity amplitude
- `alpha_hs` (>= 0): snow-thickness sensitivity amplitude
- `hi_scale` (> 0): decay scale for `hi` exponential term
- `hs_scale` (> 0): decay scale for `hs` exponential term
- `w_min` in `[0,1]`: minimum sensitivity weight at `aice_prior` extremes
- `tf0`: constant term in the salinity-dependent freezing-temperature approximation
- `tf_s_linear`: linear salinity term in the freezing-temperature approximation
- `tf_s_pow`: `S^(3/2)` coefficient in the freezing-temperature approximation
- `freezing_tolerance` (>= 0): maximum allowed `|sst - Tf(sss)|` before the Jacobian is zeroed

## Sign Expectations

Given parameter constraints and typical ocean salinities:

- `d(aice)/d(sst) <= 0`
- `d(aice)/d(sss) <= 0` (because `dTf_dS` is typically negative)
- `d(aice)/d(hi)  >= 0`
- `d(aice)/d(hs)  >= 0`

## Forward Linear Application

`forward(inputs, perturbations)` applies the Jacobian as a linear map:

- `perturbations` shape `[nnodes, 4]` ordered as `[d_sst, d_sss, d_hi, d_hs]`
- output shape `[nnodes, 1]`
- mathematically: `delta_aice = J * perturbations`

## TorchScript/C++ Contract Notes

Exported fields expected by SABER/JEDI surface balance:

- `input_names`
- `input_levels`
- `output_names`
- `output_levels`

Exported method:

- `jac_physical(inputs, mask)`
