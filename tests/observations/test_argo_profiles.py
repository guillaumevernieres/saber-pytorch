import numpy as np
from netCDF4 import Dataset

from saber_pytorch.observations.argo_profiles import (
    build_matched_argo_ts_dataset,
    group_locations_into_profiles,
    match_temperature_salinity_profiles,
    read_ioda_profile_file,
)


def _write_ioda_file(path, obs_variable, values, depths, station_ids, times):
    values = np.asarray(values, dtype=np.float32)
    depths = np.asarray(depths, dtype=np.float32)
    station_ids = np.asarray(station_ids, dtype=np.int32)
    times = np.asarray(times, dtype=np.int64)
    nloc = values.size
    with Dataset(path, "w") as ds:
        ds.createDimension("Location", nloc)
        metadata = ds.createGroup("MetaData")
        obs = ds.createGroup("ObsValue")

        lat = metadata.createVariable("latitude", "f4", ("Location",), fill_value=-3.368795e38)
        lon = metadata.createVariable("longitude", "f4", ("Location",), fill_value=-3.368795e38)
        depth = metadata.createVariable("depth", "f4", ("Location",), fill_value=-3.368795e38)
        station = metadata.createVariable("stationID", "i4", ("Location",), fill_value=-2147483643)
        date_time = metadata.createVariable("dateTime", "i8", ("Location",), fill_value=-9223372036854775801)
        original_date_time = metadata.createVariable(
            "originalDateTime",
            "i8",
            ("Location",),
            fill_value=-9223372036854775801,
        )
        basin = metadata.createVariable("oceanBasin", "i4", ("Location",), fill_value=-2147483643)
        value = obs.createVariable(obs_variable, "f4", ("Location",), fill_value=-3.368795e38)

        lat[:] = np.where(station_ids == 1001, 10.0, 20.0)
        lon[:] = np.where(station_ids == 1001, 240.0, 250.0)
        depth[:] = depths
        station[:] = station_ids
        date_time[:] = times
        original_date_time[:] = times
        basin[:] = 1
        value[:] = values


def test_group_and_match_ioda_profile_pairs(tmp_path):
    temp_path = tmp_path / "temp.nc"
    salt_path = tmp_path / "salt.nc"
    depths = [0.0, 50.0, 100.0, 0.0, 50.0]
    station_ids = [1001, 1001, 1001, 1002, 1002]
    times = [10, 10, 10, 20, 20]
    _write_ioda_file(
        temp_path,
        "waterTemperature",
        [20.0, 16.0, 12.0, 9.0, 8.0],
        depths,
        station_ids,
        times,
    )
    _write_ioda_file(
        salt_path,
        "salinity",
        [35.0, 35.2, 35.4, 34.0, 34.1],
        depths,
        station_ids,
        times,
    )

    temp_data = read_ioda_profile_file(temp_path, "waterTemperature")
    salt_data = read_ioda_profile_file(salt_path)
    assert temp_data["obs_variable"] == "waterTemperature"
    assert salt_data["obs_variable"] == "salinity"

    temp_profiles = group_locations_into_profiles(temp_data, min_valid_levels=3)
    salt_profiles = group_locations_into_profiles(salt_data, min_valid_levels=3)
    assert len(temp_profiles) == 1
    assert len(salt_profiles) == 1

    matched = match_temperature_salinity_profiles(temp_profiles, salt_profiles)
    assert len(matched) == 1
    assert matched[0]["temperature"].station_id == 1001


def test_group_locations_can_filter_implausible_salinity(tmp_path):
    salt_path = tmp_path / "salt.nc"
    _write_ioda_file(
        salt_path,
        "salinity",
        [35.0, 0.0, 35.4],
        [0.0, 50.0, 100.0],
        [1001, 1001, 1001],
        [10, 10, 10],
    )

    salt_data = read_ioda_profile_file(salt_path)
    profiles = group_locations_into_profiles(
        salt_data,
        min_valid_levels=2,
        value_min=20.0,
        value_max=45.0,
    )

    assert len(profiles) == 1
    np.testing.assert_allclose(profiles[0].depth, [0.0, 100.0])
    np.testing.assert_allclose(profiles[0].value, [35.0, 35.4])


def test_build_matched_argo_ts_dataset_interpolates_to_model_grid(tmp_path):
    temp_path = tmp_path / "temp.nc"
    salt_path = tmp_path / "salt.nc"
    depths = [0.0, 25.0, 50.0, 75.0, 100.0]
    station_ids = [1001] * 5
    times = [10] * 5
    _write_ioda_file(
        temp_path,
        "waterTemperature",
        [20.0, 18.0, 16.0, 14.0, 12.0],
        depths,
        station_ids,
        times,
    )
    _write_ioda_file(
        salt_path,
        "salinity",
        [35.0, 35.1, 35.2, 35.3, 35.4],
        depths,
        station_ids,
        times,
    )

    model_depth = np.asarray([0.0, 20.0, 40.0, 60.0, 80.0, 120.0])
    profiles, summary = build_matched_argo_ts_dataset(
        temp_path,
        salt_path,
        model_depth,
        min_valid_model_levels=5,
    )

    assert summary["retained_profiles"] == 1
    assert summary["salinity_obs_variable"] == "salinity"
    profile = profiles[0]
    np.testing.assert_array_equal(
        profile["model_valid_mask"],
        np.asarray([True, True, True, True, True, False]),
    )
    assert np.all(np.isfinite(profile["model_potential_temperature"][:5]))
    np.testing.assert_allclose(profile["model_salinity"][:5], [35.0, 35.08, 35.16, 35.24, 35.32])
