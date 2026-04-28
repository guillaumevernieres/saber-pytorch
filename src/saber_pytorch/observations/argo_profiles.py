"""Utilities for building matched Argo temperature/salinity profile pairs.

IODA profile diagnostics are stored on a flat ``Location`` dimension.  These
helpers group locations into vertical profiles, match temperature and salinity
profiles with metadata checks, convert in-situ temperature to potential
temperature, and interpolate the matched profiles to a model depth grid.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from netCDF4 import Dataset


_FLOAT_FILL_ABS_LIMIT = 1.0e30


@dataclass(frozen=True)
class ArgoProfile:
    """One grouped vertical profile from an IODA-style file."""

    key: tuple[Any, ...]
    station_id: Any
    date_time: int | None
    original_date_time: int | None
    latitude: float
    longitude: float
    ocean_basin: int | None
    depth: np.ndarray
    value: np.ndarray
    n_obs: int


def _as_array(variable: Any) -> np.ndarray:
    values = variable[:]
    if np.ma.isMaskedArray(values):
        fill_value = getattr(variable, "_FillValue", np.nan)
        values = values.filled(fill_value)
    return np.asarray(values)


def _fill_mask(values: np.ndarray, fill_value: Any | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind in "f":
        mask = ~np.isfinite(array) | (np.abs(array) > _FLOAT_FILL_ABS_LIMIT)
        if fill_value is not None:
            mask |= np.isclose(array, fill_value, equal_nan=True)
        return mask
    if array.dtype.kind in "iu":
        mask = np.zeros(array.shape, dtype=bool)
        if fill_value is not None:
            mask |= array == fill_value
        return mask
    if array.dtype.kind in "SUO":
        return np.asarray(array == "", dtype=bool)
    return np.zeros(array.shape, dtype=bool)


def _variable_fill_value(group: Any, name: str) -> Any | None:
    variable = group.variables[name]
    return getattr(variable, "_FillValue", None)


def _valid_float(values: np.ndarray, fill_value: Any | None = None) -> np.ndarray:
    return ~_fill_mask(values, fill_value)


def _valid_scalar(value: Any, fill_value: Any | None = None) -> bool:
    return not bool(_fill_mask(np.asarray([value]), fill_value)[0])


def _scalar_or_none(value: Any, fill_value: Any | None = None) -> Any | None:
    if not _valid_scalar(value, fill_value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return value


def _longitude_delta(lon: float, target_lon: float) -> float:
    return float(((lon - target_lon + 180.0) % 360.0) - 180.0)


def _mean_longitude(longitudes: np.ndarray) -> float:
    radians = np.deg2rad(longitudes)
    mean_angle = np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())
    return float((np.rad2deg(mean_angle) + 360.0) % 360.0)


def _clean_depth_value(depth: np.ndarray, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(depth) & np.isfinite(value) & (depth >= 0.0)
    depth = np.asarray(depth[mask], dtype=np.float64)
    value = np.asarray(value[mask], dtype=np.float64)
    if depth.size == 0:
        return depth, value

    order = np.argsort(depth)
    depth = depth[order]
    value = value[order]

    unique_depths = []
    unique_values = []
    for unique_depth in np.unique(depth):
        at_depth = depth == unique_depth
        unique_depths.append(float(unique_depth))
        unique_values.append(float(np.nanmean(value[at_depth])))
    return np.asarray(unique_depths), np.asarray(unique_values)


def _auto_obs_variable(obs_group: Any, obs_variable: str | None) -> str:
    variables = list(obs_group.variables)
    if obs_variable is not None:
        if obs_variable not in obs_group.variables:
            raise KeyError(f"ObsValue/{obs_variable} not found; available: {variables}")
        return obs_variable
    if len(variables) == 1:
        return variables[0]
    salinity_candidates = [
        name
        for name in variables
        if "sal" in name.lower() or "salt" in name.lower()
    ]
    if len(salinity_candidates) == 1:
        return salinity_candidates[0]
    temperature_candidates = [
        name
        for name in variables
        if "temp" in name.lower() or "temperature" in name.lower()
    ]
    if len(temperature_candidates) == 1:
        return temperature_candidates[0]
    raise ValueError(
        "Could not infer ObsValue variable; pass obs_variable explicitly. "
        f"Available variables: {variables}"
    )


def read_ioda_profile_file(path: str | Path, obs_variable: str | None = None) -> dict[str, Any]:
    """Read one IODA-style profile diagnostic file.

    Parameters
    ----------
    path
        NetCDF file with ``MetaData`` and ``ObsValue`` groups.
    obs_variable
        Optional variable name inside ``ObsValue``.  When omitted, the function
        infers the variable if the group contains one observation variable or a
        single salinity/temperature-like name.
    """

    path = Path(path).expanduser()
    with Dataset(path) as dataset:
        if "MetaData" not in dataset.groups or "ObsValue" not in dataset.groups:
            raise KeyError(f"{path} is missing required MetaData/ObsValue groups")

        metadata_group = dataset.groups["MetaData"]
        obs_group = dataset.groups["ObsValue"]
        obs_variable = _auto_obs_variable(obs_group, obs_variable)

        metadata = {
            name: _as_array(metadata_group.variables[name])
            for name in metadata_group.variables
        }
        metadata_fill_values = {
            name: _variable_fill_value(metadata_group, name)
            for name in metadata_group.variables
        }

        obs_values = _as_array(obs_group.variables[obs_variable])
        obs_fill_value = _variable_fill_value(obs_group, obs_variable)

    return {
        "path": path,
        "obs_variable": obs_variable,
        "value": obs_values,
        "value_fill_value": obs_fill_value,
        "metadata": metadata,
        "metadata_fill_values": metadata_fill_values,
        "n_locations": int(obs_values.shape[0]),
    }


def _profile_key_for_location(
    metadata: dict[str, np.ndarray],
    fill_values: dict[str, Any],
    index: int,
) -> tuple[Any, ...]:
    station_id = _scalar_or_none(
        metadata.get("stationID", np.asarray([None]))[index],
        fill_values.get("stationID"),
    )
    date_time = _scalar_or_none(
        metadata.get("dateTime", np.asarray([None]))[index],
        fill_values.get("dateTime"),
    )
    original_date_time = _scalar_or_none(
        metadata.get("originalDateTime", np.asarray([None]))[index],
        fill_values.get("originalDateTime"),
    )
    latitude = float(metadata["latitude"][index])
    longitude = float(metadata["longitude"][index])

    if station_id is not None and date_time is not None:
        return ("station_date", station_id, int(date_time))
    if station_id is not None and original_date_time is not None:
        return ("station_original_date", station_id, int(original_date_time))
    if date_time is not None:
        return ("geo_date", round(latitude, 3), round(longitude % 360.0, 3), int(date_time))
    if original_date_time is not None:
        return (
            "geo_original_date",
            round(latitude, 3),
            round(longitude % 360.0, 3),
            int(original_date_time),
        )
    if station_id is not None:
        return ("station_geo", station_id, round(latitude, 3), round(longitude % 360.0, 3))
    return ("geo", round(latitude, 3), round(longitude % 360.0, 3))


def group_locations_into_profiles(
    data: dict[str, Any],
    min_valid_levels: int = 1,
    value_min: float | None = None,
    value_max: float | None = None,
) -> list[ArgoProfile]:
    """Group flat ``Location`` observations into sorted vertical profiles.

    The primary grouping key is ``stationID + dateTime``.  If either field is
    unavailable or filled, the fallback key uses ``originalDateTime`` or rounded
    latitude/longitude plus time.  Invalid/fill observations are removed before
    grouping.
    """

    metadata = data["metadata"]
    fill_values = data["metadata_fill_values"]
    value = np.asarray(data["value"])
    value_fill_value = data.get("value_fill_value")

    required_metadata = ["latitude", "longitude", "depth"]
    missing = [name for name in required_metadata if name not in metadata]
    if missing:
        raise KeyError(f"Missing required MetaData variables: {missing}")

    valid = _valid_float(value, value_fill_value)
    if value_min is not None:
        valid &= np.asarray(value, dtype=np.float64) >= float(value_min)
    if value_max is not None:
        valid &= np.asarray(value, dtype=np.float64) <= float(value_max)
    for name in required_metadata:
        valid &= _valid_float(metadata[name], fill_values.get(name))
    valid &= np.asarray(metadata["depth"], dtype=np.float64) >= 0.0

    grouped_indices: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index in np.flatnonzero(valid):
        grouped_indices[_profile_key_for_location(metadata, fill_values, int(index))].append(
            int(index)
        )

    profiles: list[ArgoProfile] = []
    for key, indices in grouped_indices.items():
        index_array = np.asarray(indices, dtype=np.int64)
        depth, profile_value = _clean_depth_value(
            np.asarray(metadata["depth"][index_array], dtype=np.float64),
            np.asarray(value[index_array], dtype=np.float64),
        )
        if depth.size < min_valid_levels:
            continue

        latitudes = np.asarray(metadata["latitude"][index_array], dtype=np.float64)
        longitudes = np.asarray(metadata["longitude"][index_array], dtype=np.float64)
        first = int(index_array[0])
        station_id = _scalar_or_none(
            metadata.get("stationID", np.asarray([None]))[first],
            fill_values.get("stationID"),
        )
        date_time = _scalar_or_none(
            metadata.get("dateTime", np.asarray([None]))[first],
            fill_values.get("dateTime"),
        )
        original_date_time = _scalar_or_none(
            metadata.get("originalDateTime", np.asarray([None]))[first],
            fill_values.get("originalDateTime"),
        )
        ocean_basin = _scalar_or_none(
            metadata.get("oceanBasin", np.asarray([None]))[first],
            fill_values.get("oceanBasin"),
        )

        profiles.append(
            ArgoProfile(
                key=key,
                station_id=station_id,
                date_time=int(date_time) if date_time is not None else None,
                original_date_time=(
                    int(original_date_time) if original_date_time is not None else None
                ),
                latitude=float(np.nanmean(latitudes)),
                longitude=_mean_longitude(longitudes),
                ocean_basin=int(ocean_basin) if ocean_basin is not None else None,
                depth=depth,
                value=profile_value,
                n_obs=int(index_array.size),
            )
        )

    profiles.sort(key=lambda profile: profile.key)
    return profiles


def _profile_time(profile: ArgoProfile) -> int | None:
    return profile.date_time if profile.date_time is not None else profile.original_date_time


def _profiles_are_plausible_match(
    temperature: ArgoProfile,
    salinity: ArgoProfile,
    max_position_difference_deg: float,
    max_time_difference_seconds: int | None,
    min_depth_overlap_m: float,
) -> bool:
    lat_delta = abs(temperature.latitude - salinity.latitude)
    lon_delta = abs(_longitude_delta(temperature.longitude, salinity.longitude))
    if lat_delta > max_position_difference_deg or lon_delta > max_position_difference_deg:
        return False

    if max_time_difference_seconds is not None:
        temperature_time = _profile_time(temperature)
        salinity_time = _profile_time(salinity)
        if temperature_time is not None and salinity_time is not None:
            if abs(temperature_time - salinity_time) > max_time_difference_seconds:
                return False

    overlap_top = max(float(temperature.depth.min()), float(salinity.depth.min()))
    overlap_bottom = min(float(temperature.depth.max()), float(salinity.depth.max()))
    return (overlap_bottom - overlap_top) >= min_depth_overlap_m


def match_temperature_salinity_profiles(
    temperature_profiles: Iterable[ArgoProfile],
    salinity_profiles: Iterable[ArgoProfile],
    *,
    max_position_difference_deg: float = 0.02,
    max_time_difference_seconds: int | None = 3600,
    min_depth_overlap_m: float = 1.0,
) -> list[dict[str, Any]]:
    """Match grouped temperature profiles to grouped salinity profiles."""

    salinity_by_key = {profile.key: profile for profile in salinity_profiles}
    matched: list[dict[str, Any]] = []
    for temperature in temperature_profiles:
        salinity = salinity_by_key.get(temperature.key)
        if salinity is None:
            continue
        if not _profiles_are_plausible_match(
            temperature,
            salinity,
            max_position_difference_deg=max_position_difference_deg,
            max_time_difference_seconds=max_time_difference_seconds,
            min_depth_overlap_m=min_depth_overlap_m,
        ):
            continue
        matched.append({"key": temperature.key, "temperature": temperature, "salinity": salinity})
    return matched


def _interpolate_1d(
    source_depth: np.ndarray,
    source_value: np.ndarray,
    target_depth: np.ndarray,
    *,
    allow_extrapolation: bool = False,
) -> np.ndarray:
    source_depth, source_value = _clean_depth_value(source_depth, source_value)
    target_depth = np.asarray(target_depth, dtype=np.float64)
    output = np.full(target_depth.shape, np.nan, dtype=np.float64)
    if source_depth.size < 2:
        return output

    if allow_extrapolation:
        return np.interp(target_depth, source_depth, source_value)

    inside = (target_depth >= source_depth[0]) & (target_depth <= source_depth[-1])
    output[inside] = np.interp(target_depth[inside], source_depth, source_value)
    return output


def convert_insitu_to_potential_temperature(matched_profile: dict[str, Any]) -> dict[str, Any]:
    """Convert an in-situ temperature profile to potential temperature.

    Salinity is interpolated to the temperature observation depths for the
    thermodynamic conversion only.  The original salinity observations are kept
    for later interpolation to the model grid.
    """

    try:
        import gsw
    except ImportError as exc:
        raise ImportError(
            "gsw is required to convert Argo in-situ temperature to potential "
            "temperature. Install the Gibbs SeaWater package before building "
            "real Argo T/S profile datasets."
        ) from exc

    temperature = matched_profile["temperature"]
    salinity = matched_profile["salinity"]
    salinity_at_temperature_depth = _interpolate_1d(
        salinity.depth,
        salinity.value,
        temperature.depth,
        allow_extrapolation=False,
    )
    valid = np.isfinite(salinity_at_temperature_depth)
    if valid.sum() < 2:
        raise ValueError("Temperature and salinity profiles do not overlap enough")

    depth = temperature.depth[valid]
    in_situ_temperature = temperature.value[valid]
    salinity_for_conversion = salinity_at_temperature_depth[valid]
    latitude = float(temperature.latitude)
    longitude = float(temperature.longitude)

    pressure = gsw.p_from_z(-depth, latitude)
    absolute_salinity = gsw.SA_from_SP(
        salinity_for_conversion,
        pressure,
        np.full(depth.shape, longitude),
        np.full(depth.shape, latitude),
    )
    potential_temperature = gsw.pt0_from_t(
        absolute_salinity,
        in_situ_temperature,
        pressure,
    )

    return {
        "key": matched_profile["key"],
        "stationID": temperature.station_id,
        "dateTime": temperature.date_time,
        "originalDateTime": temperature.original_date_time,
        "latitude": latitude,
        "longitude": longitude,
        "oceanBasin": temperature.ocean_basin,
        "temperature_depth": depth,
        "in_situ_temperature": in_situ_temperature,
        "potential_temperature": np.asarray(potential_temperature, dtype=np.float64),
        "salinity_depth": salinity.depth,
        "salinity": salinity.value,
        "n_temperature_obs": temperature.n_obs,
        "n_salinity_obs": salinity.n_obs,
        "n_overlapping_obs": int(valid.sum()),
    }


def interpolate_profile_to_model_depth(
    profile: dict[str, Any],
    model_depth: np.ndarray,
    *,
    allow_extrapolation: bool = False,
    min_valid_levels: int = 5,
) -> dict[str, Any] | None:
    """Interpolate one converted matched profile to the model depth grid."""

    model_depth = np.asarray(model_depth, dtype=np.float64)
    potential_temperature = _interpolate_1d(
        np.asarray(profile["temperature_depth"], dtype=np.float64),
        np.asarray(profile["potential_temperature"], dtype=np.float64),
        model_depth,
        allow_extrapolation=allow_extrapolation,
    )
    salinity = _interpolate_1d(
        np.asarray(profile["salinity_depth"], dtype=np.float64),
        np.asarray(profile["salinity"], dtype=np.float64),
        model_depth,
        allow_extrapolation=allow_extrapolation,
    )
    valid_mask = np.isfinite(potential_temperature) & np.isfinite(salinity)
    if int(valid_mask.sum()) < min_valid_levels:
        return None

    output = dict(profile)
    output.update(
        {
            "model_depth": model_depth,
            "model_potential_temperature": potential_temperature,
            "model_salinity": salinity,
            "model_valid_mask": valid_mask,
            "n_valid_model_levels": int(valid_mask.sum()),
            "interpolation_allow_extrapolation": bool(allow_extrapolation),
        }
    )
    return output


def read_model_depth_grid(path: str | Path, variable: str | None = None) -> np.ndarray:
    """Read a one-dimensional model depth coordinate from a NetCDF file."""

    path = Path(path).expanduser()
    candidates = [variable] if variable is not None else [
        "Layer",
        "st_ocean",
        "zt",
        "z_l",
        "depth",
        "Depth",
    ]
    with Dataset(path) as dataset:
        for name in candidates:
            if name is None:
                continue
            if name in dataset.variables:
                values = np.asarray(dataset.variables[name][:], dtype=np.float64).squeeze()
                if values.ndim != 1:
                    raise ValueError(f"{path}:{name} is not one-dimensional")
                return values
    raise KeyError(f"No model depth variable found in {path}; tried {candidates}")


def build_matched_argo_ts_dataset(
    temperature_path: str | Path,
    salinity_path: str | Path,
    model_depth: np.ndarray,
    *,
    temperature_variable: str = "waterTemperature",
    salinity_variable: str | None = None,
    min_profile_levels: int = 2,
    min_valid_model_levels: int = 5,
    allow_extrapolation: bool = False,
    max_position_difference_deg: float = 0.02,
    max_time_difference_seconds: int | None = 3600,
    min_depth_overlap_m: float = 1.0,
    max_profiles: int | None = None,
    salinity_min: float | None = 20.0,
    salinity_max: float | None = 45.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build interpolated real Argo T/S profile pairs."""

    temperature_data = read_ioda_profile_file(temperature_path, temperature_variable)
    salinity_data = read_ioda_profile_file(salinity_path, salinity_variable)
    temperature_profiles = group_locations_into_profiles(
        temperature_data,
        min_valid_levels=min_profile_levels,
    )
    salinity_profiles = group_locations_into_profiles(
        salinity_data,
        min_valid_levels=min_profile_levels,
        value_min=salinity_min,
        value_max=salinity_max,
    )
    matched_profiles = match_temperature_salinity_profiles(
        temperature_profiles,
        salinity_profiles,
        max_position_difference_deg=max_position_difference_deg,
        max_time_difference_seconds=max_time_difference_seconds,
        min_depth_overlap_m=min_depth_overlap_m,
    )
    if max_profiles is not None:
        matched_profiles = matched_profiles[:max_profiles]

    rejection_reasons: Counter[str] = Counter()
    output: list[dict[str, Any]] = []
    for matched_profile in matched_profiles:
        try:
            converted = convert_insitu_to_potential_temperature(matched_profile)
        except Exception:
            rejection_reasons["potential_temperature_conversion"] += 1
            continue
        interpolated = interpolate_profile_to_model_depth(
            converted,
            model_depth,
            allow_extrapolation=allow_extrapolation,
            min_valid_levels=min_valid_model_levels,
        )
        if interpolated is None:
            rejection_reasons["insufficient_valid_model_levels"] += 1
            continue
        output.append(interpolated)

    summary = {
        "temperature_file": str(Path(temperature_path).expanduser()),
        "salinity_file": str(Path(salinity_path).expanduser()),
        "temperature_obs_variable": temperature_data["obs_variable"],
        "salinity_obs_variable": salinity_data["obs_variable"],
        "temperature_locations": temperature_data["n_locations"],
        "salinity_locations": salinity_data["n_locations"],
        "temperature_profiles": len(temperature_profiles),
        "salinity_profiles": len(salinity_profiles),
        "matched_profiles": len(matched_profiles),
        "retained_profiles": len(output),
        "rejection_reasons": dict(rejection_reasons),
        "model_depth_levels": int(np.asarray(model_depth).size),
        "allow_extrapolation": bool(allow_extrapolation),
        "min_valid_model_levels": int(min_valid_model_levels),
        "salinity_min": salinity_min,
        "salinity_max": salinity_max,
    }
    return output, summary
