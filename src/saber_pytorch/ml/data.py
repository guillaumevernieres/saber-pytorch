"""
Data preparation utilities for UFS emulator training.
Supports CF-1 NetCDF inputs from separate atmosphere and ocean/ice files.
"""

import numpy as np
import netCDF4 as nc
import torch
from typing import Tuple, Dict, Optional
from pathlib import Path
from .cf_mappings import CF_ATM, CF_OCN, DEFAULT_ATM_LEVEL


class UFSEmulatorDataBuilder:
    """
    Prepares training data from CF-1 NetCDF files (atmosphere + ocean/ice).
    """

    def __init__(self, config: Dict):
        self.config = config
        dcfg = config.get('domain', {})
        vcfg = config.get('variables', {})
        mcfg = config.get('model', {})
        self.emulator_type = mcfg.get('emulator_type', 'vertical')
        self.target_num_levels = vcfg.get('target_num_levels')
        self.min_layer_thickness = 0.1
        rgcfg = vcfg.get(
            'reduced_grid',
            config.get('reduced_grid', config.get('data', {}).get('reduced_grid', {}))
        )
        self.reduced_grid_method = str(
            rgcfg.get('method', 'uniform_depth')
        ).lower()
        if self.reduced_grid_method == 'uniform':
            self.reduced_grid_method = 'uniform_depth'
        if self.reduced_grid_method in ('temperature_gradient', 'temp_gradient'):
            self.reduced_grid_method = 'temperature_gradient'
        if self.reduced_grid_method not in ('uniform_depth', 'temperature_gradient'):
            raise ValueError(
                "reduced_grid.method must be 'uniform_depth' or "
                f"'temperature_gradient', got {self.reduced_grid_method!r}"
            )
        default_gradient_weight = (
            2.0 if self.reduced_grid_method == 'temperature_gradient' else 0.0
        )
        self.reduced_grid_gradient_weight = float(
            rgcfg.get('gradient_weight', default_gradient_weight)
        )
        self.min_ice = dcfg.get('min_ice_concentration', 0.0)
        self.use_synthetic = dcfg.get('use_synthetic_data', False)
        self.mask_mode = dcfg.get('mask_mode', 'sea_ice')
        # Atmospheric vertical level info (for interpolated files with single level)
        self.atm_level_index = dcfg.get('atm_level_index', 127)  # Default: level 127 (typical nlevs-1 for 128-level model)
        # Inputs/outputs
        self.input_variables = vcfg.get('input_variables')
        self.output_variables = vcfg.get('output_variables', ['aice'])
        mxf = vcfg.get('max_input_features')
        if mxf and mxf > 0:
            self.input_variables = self.input_variables[:mxf]
        self.input_size = len(self.input_variables)
        self.output_size = len(self.output_variables)
        # CF-1 variable map (imported from cf_mappings module)
        self.cf_atm = CF_ATM
        self.cf_ocn = CF_OCN

    def _uses_salinity_profile_emulator(self) -> bool:
        return self.emulator_type == 'salinity_profile'

    def _read_var(self, ds, name):
        """Read variable from NetCDF, converting masked arrays to regular arrays with NaN for fill values."""
        var_data = ds.variables[name][:]
        # NetCDF4 returns masked arrays - convert to regular array with NaN for masked values
        if np.ma.is_masked(var_data):
            var_data = np.ma.filled(var_data, np.nan)
        return var_data

    def read_netcdf_data_pair(self, atm_file: Optional[str], ocn_file: str) -> Dict[str, np.ndarray]:
        # Read ocean dataset (always required)
        do = nc.Dataset(ocn_file, 'r')
        data: Dict[str, np.ndarray] = {}

        # Determine if we need atmospheric data
        atm_vars_needed = set()
        for var in self.input_variables + self.output_variables:
            if var in self.cf_atm:
                atm_vars_needed.add(var)

        # Read atmospheric dataset only if needed and file provided
        da = None
        if atm_file and atm_vars_needed:
            da = nc.Dataset(atm_file, 'r')

            # Coordinates from atmosphere file
            lat = self._read_var(da, self.cf_atm['lat']).astype(np.float32)
            lon = self._read_var(da, self.cf_atm['lon']).astype(np.float32)
        else:
            # Coordinates from ocean file
            # Try different possible coordinate names
            lat_names = ['latitude', 'geolat', 'yh', 'ny', 'lat', 'y', 'yaxis_1', 'yaxis_2']
            lon_names = ['longitude', 'geolon', 'xh', 'nx', 'lon', 'x', 'xaxis_1', 'xaxis_2']

            lat = None
            for name in lat_names:
                if name in do.variables:
                    lat = self._read_var(do, name).astype(np.float32)
                    print(f"  Found latitude coordinate: '{name}'")
                    break
            if lat is None:
                if 'ny' in do.dimensions:
                    lat = np.arange(len(do.dimensions['ny']), dtype=np.float32)
                    print("  Latitude coordinate not found; using ny index coordinate")
                else:
                    raise ValueError(f"Latitude coordinate not found. Tried: {lat_names}. Available: {list(do.variables.keys())[:20]}")

            lon = None
            for name in lon_names:
                if name in do.variables:
                    lon = self._read_var(do, name).astype(np.float32)
                    print(f"  Found longitude coordinate: '{name}'")
                    break
            if lon is None:
                if 'nx' in do.dimensions:
                    lon = np.arange(len(do.dimensions['nx']), dtype=np.float32)
                    print("  Longitude coordinate not found; using nx index coordinate")
                else:
                    raise ValueError(f"Longitude coordinate not found. Tried: {lon_names}. Available: {list(do.variables.keys())[:20]}")

        # Broadcast to 2D
        lat2 = np.repeat(lat[:, None], lon.size, axis=1)
        lon2 = np.repeat(lon[None, :], lat.size, axis=0)
        data['lat'] = lat2.flatten()
        data['lon'] = lon2.flatten()

        # Read atmospheric variables if atmosphere file is provided
        if da is not None:
            # Determine atmospheric level and dimensionality
            # Use tair as reference to check if file has vertical dimension
            tair_var = da.variables[self.cf_atm['tair']]
            has_vertical_dim = len(tair_var.shape) == 3

            if has_vertical_dim:
                nlevs = tair_var.shape[2]
                lev_idx = nlevs - 1
                atm_level = nlevs - 1
            else:
                lev_idx = None
                atm_level = self.atm_level_index  # Use configured level index

            for var in atm_vars_needed:
                cf_name = self.cf_atm[var]
                if cf_name in da.variables:
                    var_data = da.variables[cf_name]
                    if has_vertical_dim and len(var_data.shape) == 3:
                        data[var] = var_data[:, :, lev_idx].astype(np.float32).flatten()
                    else:
                        data[var] = var_data[:].astype(np.float32).flatten()
                else:
                    raise ValueError(f"Required atmospheric variable '{var}' (CF name: '{cf_name}') not found in {atm_file}")

            # Store atmospheric level index for metadata
            data['atm_level_index'] = atm_level
        else:
            data['atm_level_index'] = -1  # No atmospheric data

        # Get vertical level limit from config (default 50)
        vcfg = self.config.get('variables', {})
        VERTICAL_LEVEL_LIMIT = int(vcfg.get('num_levels', 50))

        # Dynamically read all ocean/ice variables needed
        ocn_vars_needed = set()
        for var in self.input_variables + self.output_variables:
            if var in self.cf_ocn:
                ocn_vars_needed.add(var)

        # Track if we have 3D data to get spatial dimensions
        nlat, nlon, nlevs = None, None, None

        h_data_3d = None  # Store h data if we need to compute depth
        temperature_vars = {
            'Temp', 'temp', 'sst', 'thetao', 'sea_water_potential_temperature'
        }
        salinity_vars = {
            'Salt', 'salt', 'sss', 'so', 'sea_water_salinity'
        }
        thickness_vars = {
            'h', 'ho', 'thick', 'thkcello', 'sea_water_cell_thickness'
        }

        for var in ocn_vars_needed:
            cf_name = self.cf_ocn[var]
            var_data = None

            # Special handling for common MOM6 variables - try multiple possible names
            if var in thickness_vars:
                possible_names = ['h', 'ho', 'sea_water_cell_thickness', 'thkcello', 'dz']

                # Debug: Print all available variables
                all_vars = list(do.variables.keys())
                print(f"  DEBUG: Total variables in file: {len(all_vars)}")
                print(f"  DEBUG: First 30 variables: {all_vars[:30]}")

                for name in possible_names:
                    if name in do.variables:
                        cf_name = name
                        var_data = do.variables[name]
                        print(f"  Found thickness variable: '{name}' with dimensions {var_data.dimensions}")
                        break

                if var_data is None:
                    # Thickness not found - try to compute from depth coordinate
                    print("  Thickness variable not found, attempting to compute from depth coordinate...")
                    depth_coord_names = ['zaxis_1', 'zaxis_2', 'z_l', 'z_i', 'depth', 'lev', 'zl']
                    depth_coord = None
                    depth_coord_name = None
                    for name in depth_coord_names:
                        if name in do.variables:
                            depth_coord = do.variables[name][:]
                            depth_coord_name = name
                            print(f"  Found depth coordinate: '{name}', will compute thickness")
                            break

                    if depth_coord is not None:
                        # Compute thickness from depth coordinate
                        # Assuming depth_coord is at layer centers
                        nlevs = len(depth_coord)
                        thickness = np.zeros(nlevs, dtype=np.float32)

                        # First layer: thickness = 2 * depth
                        thickness[0] = 2.0 * depth_coord[0]

                        # Middle layers: thickness = depth[i+1] - depth[i-1]
                        for i in range(1, nlevs - 1):
                            thickness[i] = depth_coord[i+1] - depth_coord[i-1]

                        # Last layer: use same as second-to-last
                        thickness[-1] = thickness[-2]

                        print(f"  Computed thickness from depth coordinate (nlevs={nlevs})")
                        print(f"    Thickness range: {thickness.min():.2f} - {thickness.max():.2f} m")

                        # Create a synthetic variable - will be broadcast to all spatial points later
                        var_data = None  # Signal to handle as 1D coordinate
                        # Store computed thickness for later use
                        data['_computed_thickness'] = thickness
                        data[f'{var}_is_coord'] = True
                        nlat = None  # Will be set when reading other 3D variables
                        nlon = None
                        nlevs = len(thickness)
                    else:
                        available_vars = all_vars
                        raise ValueError(f"Thickness variable not found and cannot compute from depth. Tried: {possible_names}. Available depth coords tried: {depth_coord_names}. All variables: {available_vars}")
            elif var in temperature_vars:
                possible_names = ['Temp', 'temp', 'sea_water_potential_temperature', 'thetao', 'temperature']
                for name in possible_names:
                    if name in do.variables:
                        cf_name = name
                        var_data = do.variables[name]
                        print(f"  Found temperature variable: '{name}'")
                        break
                if var_data is None:
                    available_vars = list(do.variables.keys())
                    raise ValueError(f"Temperature variable not found. Tried: {possible_names}. Available variables: {available_vars[:20]}...")
            elif var in salinity_vars:
                possible_names = ['Salt', 'so', 'sea_water_salinity', 'salt', 'salinity']
                for name in possible_names:
                    if name in do.variables:
                        cf_name = name
                        var_data = do.variables[name]
                        print(f"  Found salinity variable: '{name}'")
                        break
                if var_data is None:
                    available_vars = list(do.variables.keys())
                    raise ValueError(f"Salinity variable not found. Tried: {possible_names}. Available variables: {available_vars[:20]}...")
            elif cf_name in do.variables:
                var_data = do.variables[cf_name]
            else:
                raise ValueError(f"Required ocean variable '{var}' (CF name: '{cf_name}') not found in {ocn_file}")

            # Special handling for computed thickness (when var_data is None but we computed it)
            if var_data is None and var in thickness_vars and '_computed_thickness' in data:
                # Thickness was computed from depth coordinate - treat as 1D coordinate
                # Will be broadcast to all spatial points during filtering
                continue  # Skip to next variable, already stored in data
            elif var_data is not None:
                # Handle time dimension if present (squeeze out time dimension)
                raw_data = var_data[:]
                # NetCDF4 returns masked arrays - convert to regular array with NaN for masked values
                if np.ma.is_masked(raw_data):
                    raw_data = np.ma.filled(raw_data, np.nan)

                if len(raw_data.shape) == 4 and raw_data.shape[0] == 1:
                    # Shape: (time=1, z_l, yh, xh) -> squeeze to (z_l, yh, xh)
                    raw_data = raw_data.squeeze(axis=0)
                    print(f"  Squeezed time dimension from '{cf_name}': {var_data.shape} -> {raw_data.shape}")

                # Check dimensionality
                if len(raw_data.shape) == 3:
                    # 3D ocean variable (e.g., temp, so, ho)
                    # Shape: (nlevs, nlat, nlon) OR (nlat, nlon, nlevs)
                    # Determine which dimension is vertical
                    shape = raw_data.shape

                    # Check if first dimension is vertical (typical for z_l first)
                    if shape[0] < 100:  # Typical vertical levels < 100
                        # Shape is (nlevs, nlat, nlon) - need to transpose
                        raw_data = np.transpose(raw_data, (1, 2, 0))  # -> (nlat, nlon, nlevs)
                        print(f"  Transposed '{cf_name}' from (nlevs, nlat, nlon) to (nlat, nlon, nlevs)")

                    # Limit to first 50 levels
                    full_data = raw_data.astype(np.float32)[..., :VERTICAL_LEVEL_LIMIT]
                    if nlat is None:
                        nlat, nlon, nlevs = full_data.shape
                    nlevs = min(nlevs, VERTICAL_LEVEL_LIMIT)

                    # Special handling for 'h' (thickness)
                    if var in thickness_vars:
                        # Check for invalid values (fill values now converted to NaN)
                        n_nan = np.sum(np.isnan(full_data))
                        n_total = full_data.size
                        if n_nan > 0:
                            print(f"  INFO: Thickness has {n_nan} NaN values ({100*n_nan/n_total:.1f}%) - land points")

                        # Store thickness as-is (don't convert to depth for now - simpler)
                        data[var] = full_data.reshape(nlat * nlon, nlevs)
                        data[f'{var}_nlevs'] = nlevs

                        # Compute depth as cumulative sum of thickness for each profile
                        # depth[i, k] = sum_{j=0}^k h[i, j]
                        thickness_profiles = data[var]  # shape (n_profiles, nlevs)
                        depth_profiles = np.cumsum(thickness_profiles, axis=1)
                        data['depth'] = depth_profiles  # shape (n_profiles, nlevs)
                        data['depth_nlevs'] = nlevs

                        # Also use for surface masking
                        h_data_3d = full_data
                    else:
                        # Regular 3D variable - store as-is
                        n_nan = np.sum(np.isnan(full_data))
                        if n_nan > 0:
                            pct = 100 * n_nan / full_data.size
                            print(f"  INFO: '{cf_name}' has {n_nan} NaN values ({pct:.1f}%) - land points")

                        data[var] = full_data.reshape(nlat * nlon, nlevs)
                        data[f'{var}_nlevs'] = nlevs
                elif len(raw_data.shape) == 1:
                    # 1D coordinate variable (e.g., z_l depth levels)
                    # This needs to be broadcast to all spatial points
                    # Store the 1D array for now, will broadcast during filtering
                    coord_data = raw_data.astype(np.float32)[:VERTICAL_LEVEL_LIMIT]
                    data[var] = coord_data
                    data[f'{var}_is_coord'] = True
                    if nlevs is None:
                        nlevs = len(coord_data)
                elif len(raw_data.shape) == 2:
                    # 2D surface variable - shape (nlat, nlon) -> (nlat*nlon,)
                    data[var] = raw_data.astype(np.float32).flatten()
                else:
                    raise ValueError(f"Unexpected dimensionality for variable '{var}': {raw_data.shape}")

        # Store spatial dimensions for coordinate broadcasting
        if nlat is not None:
            data['_nlat'] = nlat
            data['_nlon'] = nlon
            data['_nlevs'] = nlevs

        # Always read 'thick' for masking even if not in input/output (backward compatibility)
        # If we already read 'h' and converted to depth, use h_data_3d for masking
        if 'thick' not in data:
            if h_data_3d is not None:
                # Use surface layer thickness from h for masking
                data['thick'] = h_data_3d[:, :, 0].astype(np.float32).flatten()
            elif self.cf_ocn['thick'] in do.variables:
                thick_raw = self._read_var(do, self.cf_ocn['thick'])
                if len(thick_raw.shape) == 3:
                    data['thick'] = thick_raw[:, :, 0].astype(np.float32).flatten()
                else:
                    data['thick'] = thick_raw[:].astype(np.float32).flatten()

        # Mask: thickness validity (0 < thick <= 500) AND tair bounds (183.15 to 333.15 K)
        # Only apply tair bounds if tair was actually read
        # NaN values (from masked arrays) are excluded by np.isfinite()
        if 'tair' in data and 'thick' in data:
            thick_valid = np.isfinite(data['thick']) & (data['thick'] > 0.0) & (data['thick'] <= 500.0)
            tair_valid = (data['tair'] >= 183.15) & (data['tair'] <= 333.15)
            data['mask'] = (thick_valid & tair_valid).astype(np.int32)
        elif 'thick' in data:
            # Treat NaN as invalid (masked out)
            thick_valid = np.isfinite(data['thick']) & (data['thick'] > 0.0) & (data['thick'] <= 500.0)
            data['mask'] = thick_valid.astype(np.int32)
        else:
            # No masking possible
            data['mask'] = np.ones(len(data['lat']), dtype=np.int32)

        if da is not None:
            da.close()
        do.close()
        return data

    def _check_for_invalid_values(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Mark any profile with NaN or large fill values (>9000) as invalid.

        Returns boolean array: True = valid profile, False = has invalid data
        """
        FILL_VALUE_THRESHOLD = 9000.0  # CICE/MOM6 fill values are typically ~1e4

        n_spatial = len(data['lat'])
        valid_profiles = np.ones(n_spatial, dtype=bool)

        print(f"  QC Debug: Starting with {n_spatial} spatial points")

        # Check all 3D variables for NaN or fill values
        for var in self.input_variables + self.output_variables:
            if var in data:
                if data[var].ndim == 2:
                    # 3D variable: shape (n_spatial, n_levels)
                    var_min = np.nanmin(data[var])
                    var_max = np.nanmax(data[var])
                    print(f"  QC Debug: Checking '{var}' - shape: {data[var].shape}")
                    print(f"    Data range (excluding NaN): [{var_min:.2e}, {var_max:.2e}]")

                    has_nan = np.any(np.isnan(data[var]), axis=1)
                    has_fill = np.any(np.abs(data[var]) >= FILL_VALUE_THRESHOLD, axis=1)
                    n_nan_profiles = np.sum(has_nan)
                    n_fill_profiles = np.sum(has_fill & ~has_nan)

                    print(f"    Found {n_nan_profiles} profiles with NaN (land/invalid points)")
                    if n_fill_profiles > 0:
                        print(f"    Found {n_fill_profiles} profiles with fill values (|x|>={FILL_VALUE_THRESHOLD:.0e})")

                    valid_profiles &= ~has_nan & ~has_fill

                    if n_nan_profiles > 0:
                        print(f"  QC: Rejecting {n_nan_profiles} profiles with NaN in '{var}'")

        n_valid = np.sum(valid_profiles)
        n_invalid = n_spatial - n_valid
        print(f"  QC: Keeping {n_valid}/{n_spatial} profiles ({100*n_valid/n_spatial:.1f}%)")

        return valid_profiles

    def _check_salinity_profile_valid_values(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Validate salinity-profile data only on sufficiently thick source levels."""
        fill_threshold = 9000.0
        n_spatial = len(data['lat'])
        valid_profiles = np.ones(n_spatial, dtype=bool)
        temperature_var = self.input_variables[0]
        thickness_var = self.input_variables[1]
        salinity_var = self.output_variables[0]

        thickness = data[thickness_var]
        valid_thickness = (
            np.isfinite(thickness)
            & (thickness > self.min_layer_thickness)
        )
        valid_profiles &= np.any(valid_thickness, axis=1)

        for var in (temperature_var, salinity_var):
            values = data[var]
            valid_values = np.isfinite(values) & (np.abs(values) < fill_threshold)
            valid_profiles &= ~np.any(valid_thickness & ~valid_values, axis=1)
            valid_profiles &= np.any(valid_thickness & valid_values, axis=1)

        n_valid = np.sum(valid_profiles)
        print(
            f"  QC: Keeping {n_valid}/{n_spatial} salinity profiles "
            f"({100*n_valid/n_spatial:.1f}%)"
        )
        return valid_profiles

    def _target_depths_for_profile(
        self,
        source_depths: np.ndarray,
        temperature_values: np.ndarray,
        target_levels: int,
    ) -> np.ndarray:
        if len(source_depths) == 0:
            return np.full(target_levels, np.nan, dtype=np.float32)
        if len(source_depths) == 1:
            return np.full(target_levels, source_depths[0], dtype=np.float32)

        if (
            self.reduced_grid_method != 'temperature_gradient'
            or self.reduced_grid_gradient_weight <= 0.0
        ):
            return np.linspace(
                source_depths[0],
                source_depths[-1],
                target_levels,
                dtype=np.float32,
            )

        dz = np.diff(source_depths)
        valid_dz = dz > 1.0e-12
        if not np.any(valid_dz):
            return np.linspace(
                source_depths[0],
                source_depths[-1],
                target_levels,
                dtype=np.float32,
            )

        slopes = np.zeros_like(dz, dtype=np.float32)
        slopes[valid_dz] = (
            np.abs(np.diff(temperature_values)[valid_dz])
            / dz[valid_dz]
        )
        max_slope = float(np.max(slopes))
        if not np.isfinite(max_slope) or max_slope <= 1.0e-12:
            return np.linspace(
                source_depths[0],
                source_depths[-1],
                target_levels,
                dtype=np.float32,
            )

        metric_steps = dz * (
            1.0 + self.reduced_grid_gradient_weight * slopes / max_slope
        )
        metric_steps = np.where(valid_dz, metric_steps, 0.0).astype(np.float32)
        cumulative_metric = np.concatenate(
            (
                np.array([0.0], dtype=np.float32),
                np.cumsum(metric_steps, dtype=np.float32),
            )
        )
        total_metric = float(cumulative_metric[-1])
        if not np.isfinite(total_metric) or total_metric <= 1.0e-12:
            return np.linspace(
                source_depths[0],
                source_depths[-1],
                target_levels,
                dtype=np.float32,
            )

        target_metric = np.linspace(
            0.0,
            total_metric,
            target_levels,
            dtype=np.float32,
        )
        return np.interp(
            target_metric,
            cumulative_metric,
            source_depths,
        ).astype(np.float32)

    def _target_depths_from_temperature(
        self,
        temperature: np.ndarray,
        thickness: np.ndarray,
    ) -> np.ndarray:
        if self.target_num_levels is None:
            raise ValueError("target_num_levels is required for salinity_profile")

        fill_threshold = 9000.0
        target_levels = int(self.target_num_levels)
        target_depths = np.zeros((temperature.shape[0], target_levels), dtype=np.float32)

        valid_thickness = (
            np.isfinite(thickness)
            & (thickness > self.min_layer_thickness)
        )
        safe_thickness = np.where(valid_thickness, thickness, 0.0).astype(np.float32)
        depths = np.cumsum(safe_thickness, axis=1) - 0.5 * safe_thickness
        valid_temperature = (
            np.isfinite(temperature)
            & (np.abs(temperature) < fill_threshold)
        )
        valid = valid_thickness & valid_temperature

        for i in range(temperature.shape[0]):
            valid_indices = np.where(valid[i])[0]
            source_depths = depths[i, valid_indices]
            source_temperature = temperature[i, valid_indices]
            target_depths[i, :] = self._target_depths_for_profile(
                source_depths,
                source_temperature,
                target_levels,
            )

        return target_depths

    def _thickness_from_target_depths(self, target_depths: np.ndarray) -> np.ndarray:
        reduced_thickness = np.zeros_like(target_depths, dtype=np.float32)
        n_levels = target_depths.shape[1]
        if n_levels == 1:
            reduced_thickness[:, 0] = np.maximum(2.0 * target_depths[:, 0], 0.0)
            return reduced_thickness

        interfaces = np.zeros((target_depths.shape[0], n_levels + 1), dtype=np.float32)
        top_spacing = target_depths[:, 1] - target_depths[:, 0]
        interfaces[:, 0] = np.maximum(target_depths[:, 0] - 0.5 * top_spacing, 0.0)
        interfaces[:, 1:n_levels] = 0.5 * (
            target_depths[:, :-1] + target_depths[:, 1:]
        )
        bottom_spacing = target_depths[:, -1] - target_depths[:, -2]
        interfaces[:, n_levels] = target_depths[:, -1] + 0.5 * bottom_spacing

        reduced_thickness[:, :] = np.diff(interfaces, axis=1)
        reduced_thickness = np.where(
            reduced_thickness > self.min_layer_thickness,
            reduced_thickness,
            self.min_layer_thickness,
        )
        return reduced_thickness.astype(np.float32)

    def _interp_profiles_at_target_depths(
        self,
        values: np.ndarray,
        thickness: np.ndarray,
        target_depths: np.ndarray,
    ) -> np.ndarray:
        """Interpolate profiles to precomputed reduced-grid target depths."""
        if self.target_num_levels is None:
            raise ValueError("target_num_levels is required for salinity_profile")

        fill_threshold = 9000.0
        target_levels = int(self.target_num_levels)
        reduced = np.zeros((values.shape[0], target_levels), dtype=np.float32)

        valid_thickness = (
            np.isfinite(thickness)
            & (thickness > self.min_layer_thickness)
        )
        safe_thickness = np.where(valid_thickness, thickness, 0.0).astype(np.float32)
        depths = np.cumsum(safe_thickness, axis=1) - 0.5 * safe_thickness
        valid_values = np.isfinite(values) & (np.abs(values) < fill_threshold)
        valid = valid_thickness & valid_values

        for i in range(values.shape[0]):
            valid_indices = np.where(valid[i])[0]
            if len(valid_indices) == 0:
                reduced[i, :] = np.nan
            elif len(valid_indices) == 1:
                reduced[i, :] = values[i, valid_indices[0]]
            else:
                source_depths = depths[i, valid_indices]
                source_values = values[i, valid_indices]
                reduced[i, :] = np.interp(
                    target_depths[i, :], source_depths, source_values
                ).astype(np.float32)

        return reduced

    def _interp_profiles_to_target_grid(
        self,
        values: np.ndarray,
        thickness: np.ndarray,
    ) -> np.ndarray:
        """Interpolate profiles to the salinity emulator's reduced vertical grid."""
        target_depths = self._target_depths_from_temperature(values, thickness)
        return self._interp_profiles_at_target_depths(values, thickness, target_depths)

    def filter_data(self, data: Dict[str, np.ndarray], max_patterns: int = 400000) -> Tuple[np.ndarray, ...]:
        # Simple QC: exclude any profile with NaN or fill values
        if self._uses_salinity_profile_emulator():
            valid_profiles_qc = self._check_salinity_profile_valid_values(data)
        else:
            valid_profiles_qc = self._check_for_invalid_values(data)

        # Apply thickness mask (basic validity check on surface thickness)
        mask = data['mask'] == 1

        # Combine masks
        mask = mask & valid_profiles_qc

        # Handle aice for masking (if it exists)
        if 'aice' in data:
            aice_data = data['aice']
            # If aice is 3D, take surface layer
            if aice_data.ndim == 2:
                aice = aice_data[:, 0]
            else:
                aice = aice_data
            has_aice = True
        else:
            aice = np.zeros(len(data['lat']), dtype=np.float32)
            has_aice = False

        if self.mask_mode == "sea_ice":
            if has_aice:
                valid = mask & (aice > self.min_ice)
            else:
                print("  Warning: mask_mode='sea_ice' but no aice data found. Using all masked points.")
                valid = mask
        elif self.mask_mode == "ocean":
            if has_aice:
                valid = mask & (aice < self.min_ice)
            else:
                # For ocean-only data without aice, just use thickness mask
                print("  Ocean-only mode: using all valid ocean points (no ice filtering)")
                valid = mask
        else:  # "both"
            valid = mask

        indices = np.where(valid)[0][:max_patterns]
        n = len(indices)
        print(f"Selected {n} points from {len(data['lat'])} total")

        if self._uses_salinity_profile_emulator():
            temperature_var = self.input_variables[0]
            thickness_var = self.input_variables[1]
            salinity_var = self.output_variables[0]
            thickness = data[thickness_var][indices, :]
            temperature = data[temperature_var][indices, :]
            target_depths = self._target_depths_from_temperature(
                temperature, thickness
            )
            reduced_temperature = self._interp_profiles_at_target_depths(
                temperature, thickness, target_depths
            )
            reduced_thickness = self._thickness_from_target_depths(target_depths)
            patterns = np.concatenate(
                (reduced_temperature, reduced_thickness),
                axis=1,
            ).astype(np.float32)
            targets = self._interp_profiles_at_target_depths(
                data[salinity_var][indices, :], thickness, target_depths
            )
            self.input_size = patterns.shape[1]
            self.output_size = targets.shape[1]
            lons = data['lon'][indices]
            lats = data['lat'][indices]
            if np.any(np.isnan(patterns)) or np.any(np.isnan(targets)):
                raise ValueError("Reduced-grid salinity training data contains NaN values")
            print(f"  Training data validation: OK (no NaN values)")
            return patterns, targets, lons, lats

        # Calculate total input/output size dynamically based on 3D variables
        total_input_size = 0
        total_output_size = 0

        for name in self.input_variables:
            var_data = data.get(name)
            if var_data is not None and var_data.ndim == 2:
                # 3D variable: (nspatial, nlevs)
                total_input_size += var_data.shape[1]
            elif var_data is not None and data.get(f'{name}_is_coord', False):
                # 1D coordinate variable - broadcast to all levels
                total_input_size += len(var_data)
            else:
                # 2D variable: single feature
                total_input_size += 1

        for name in self.output_variables:
            var_data = data.get(name)
            if var_data is not None and var_data.ndim == 2:
                # 3D variable: (nspatial, nlevs)
                total_output_size += var_data.shape[1]
            elif var_data is not None and data.get(f'{name}_is_coord', False):
                # 1D coordinate variable - broadcast to all levels
                total_output_size += len(var_data)
            else:
                # 2D variable: single feature
                total_output_size += 1

        patterns = np.zeros((n, total_input_size), dtype=np.float32)
        targets = np.zeros((n, total_output_size), dtype=np.float32)

        # Fill patterns with input variables
        col_idx = 0
        for name in self.input_variables:
            # If the input variable is 'h', use 'depth' instead
            if name == 'h' and 'depth' in data:
                var_data = data['depth']
            else:
                var_data = data.get(name)
            if var_data is not None and var_data.ndim == 2:
                # 3D variable: all levels become features
                nlevs = var_data.shape[1]
                patterns[:, col_idx:col_idx+nlevs] = var_data[indices, :]
                col_idx += nlevs
            elif var_data is not None and data.get(f'{name}_is_coord', False):
                # 1D coordinate variable (e.g., depth z_l) - broadcast to all selected points
                # Each spatial point gets the same depth values
                nlevs = len(var_data)
                patterns[:, col_idx:col_idx+nlevs] = np.tile(var_data, (n, 1))
                col_idx += nlevs
            elif var_data is not None:
                # 2D variable: single feature
                patterns[:, col_idx] = var_data[indices]
                col_idx += 1
            else:
                # Missing variable - fill with zeros
                patterns[:, col_idx] = 0.0
                col_idx += 1

        # Fill targets with output variables
        col_idx = 0
        for name in self.output_variables:
            var_data = data.get(name)
            if var_data is not None and var_data.ndim == 2:
                # 3D variable: all levels become features
                nlevs = var_data.shape[1]
                targets[:, col_idx:col_idx+nlevs] = var_data[indices, :]
                col_idx += nlevs
            elif var_data is not None and data.get(f'{name}_is_coord', False):
                # 1D coordinate variable - broadcast to all selected points
                nlevs = len(var_data)
                targets[:, col_idx:col_idx+nlevs] = np.tile(var_data, (n, 1))
                col_idx += nlevs
            elif var_data is not None:
                # 2D variable: single feature
                targets[:, col_idx] = var_data[indices]
                col_idx += 1
            else:
                # Missing variable - fill with zeros
                targets[:, col_idx] = 0.0
                col_idx += 1

        # Update self.input_size and self.output_size based on actual dimensions
        self.input_size = total_input_size
        self.output_size = total_output_size

        lons = data['lon'][indices]
        lats = data['lat'][indices]

        # Check for NaN in patterns and targets before returning
        # (Fill values were converted to NaN during reading)
        n_nan_patterns = np.sum(np.isnan(patterns))
        n_nan_targets = np.sum(np.isnan(targets))

        if n_nan_patterns > 0:
            print(f"  ERROR: Invalid values found in training patterns!")
            print(f"    - {n_nan_patterns}/{patterns.size} NaN values ({100*n_nan_patterns/patterns.size:.2f}%)")
            raise ValueError("Training patterns contain NaN - QC failed to filter invalid profiles")

        if n_nan_targets > 0:
            print(f"  ERROR: Invalid values found in training targets!")
            print(f"    - {n_nan_targets}/{targets.size} NaN values ({100*n_nan_targets/targets.size:.2f}%)")
            raise ValueError("Training targets contain NaN - QC failed to filter invalid profiles")

        print(f"  Training data validation: OK (no NaN values)")

        return patterns, targets, lons, lats

    def compute_normalization_stats(self, patterns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.mean(patterns, axis=0).astype(np.float32)
        std = np.std(patterns, axis=0).astype(np.float32)
        std = np.where(std > 1e-6, std, 1.0)

        # Check for NaNs in normalization statistics
        n_nan_mean = np.sum(np.isnan(mean))
        n_nan_std = np.sum(np.isnan(std))

        if n_nan_mean > 0 or n_nan_std > 0:
            print(f"  ERROR: NaN values in normalization statistics!")
            if n_nan_mean > 0:
                print(f"    - {n_nan_mean}/{len(mean)} NaN values in mean")
            if n_nan_std > 0:
                print(f"    - {n_nan_std}/{len(std)} NaN values in std")
            raise ValueError("Normalization statistics contain NaN - input data is corrupted")

        return mean, std

    def thin_patterns(self, patterns, targets, lons, lats, fraction, target_depths=None):
        if fraction < 1.0:
            n = len(targets)
            m = int(n * fraction)
            idx = np.random.choice(n, m, replace=False)
            if target_depths is not None:
                return patterns[idx], targets[idx], lons[idx], lats[idx], target_depths[idx]
            return patterns[idx], targets[idx], lons[idx], lats[idx], None
        return patterns, targets, lons, lats, target_depths

    def prepare_training_data(self, atm_file: Optional[str] = None,
                              ocn_file: Optional[str] = None,
                              max_patterns: int = 400000,
                              output_file: Optional[str] = None,
                              thin_fraction: float = 1.0) -> Dict:
        data = self.read_netcdf_data_pair(atm_file, ocn_file)
        filtered = self.filter_data(data, max_patterns)
        if len(filtered) == 5:
            patterns, targets, lons, lats, target_depths = filtered
        else:
            patterns, targets, lons, lats = filtered
            target_depths = None
        patterns, targets, lons, lats, target_depths = self.thin_patterns(
            patterns, targets, lons, lats, thin_fraction, target_depths
        )

        if len(patterns) == 0:
            raise ValueError(
                f"No valid training points found after filtering. "
                f"Check mask_mode='{self.mask_mode}' in the config domain section, "
                f"and ensure the data file contains valid points."
            )

        # Compute normalization statistics for both inputs and outputs
        print(f"\nComputing normalization statistics...")
        print(f"  Input patterns shape: {patterns.shape}")
        print(f"  Input range: [{np.min(patterns):.6f}, {np.max(patterns):.6f}]")

        input_mean, input_std = self.compute_normalization_stats(patterns)

        print(f"  Input normalization computed:")
        print(f"    - Mean range: [{np.min(input_mean):.6f}, {np.max(input_mean):.6f}]")
        print(f"    - Std range: [{np.min(input_std):.6f}, {np.max(input_std):.6f}]")

        print(f"  Output patterns shape: {targets.shape}")
        print(f"  Output range: [{np.min(targets):.6f}, {np.max(targets):.6f}]")

        output_mean, output_std = self.compute_normalization_stats(targets)

        print(f"  Output normalization computed:")
        print(f"    - Mean range: [{np.min(output_mean):.6f}, {np.max(output_mean):.6f}]")
        print(f"    - Std range: [{np.min(output_std):.6f}, {np.max(output_std):.6f}]")

        # Create CF-1 standard name mappings for DA system
        # Atmospheric variables use nlevs-1 (stored from read), ocean variables use 0
        atm_level = data.get('atm_level_index', self.atm_level_index)

        input_cf_mapping = {}
        output_cf_mapping = {}

        for var in self.input_variables:
            if var in self.cf_atm:
                input_cf_mapping[var] = {
                    'cf_name': self.cf_atm[var],
                    'source': 'atmosphere',
                    'level_index': atm_level
                }
            elif var in self.cf_ocn:
                input_cf_mapping[var] = {
                    'cf_name': self.cf_ocn[var],
                    'source': 'ocean',
                    'level_index': 0
                }

        for var in self.output_variables:
            if var in self.cf_atm:
                output_cf_mapping[var] = {
                    'cf_name': self.cf_atm[var],
                    'source': 'atmosphere',
                    'level_index': atm_level
                }
            elif var in self.cf_ocn:
                output_cf_mapping[var] = {
                    'cf_name': self.cf_ocn[var],
                    'source': 'ocean',
                    'level_index': 0
                }

        result = {
            'inputs': patterns, 'targets': targets, 'lons': lons, 'lats': lats,
            'input_mean': input_mean, 'input_std': input_std,
            'output_mean': output_mean, 'output_std': output_std,
            'metadata': {
                'n_patterns': len(patterns),
                'input_features': self.input_variables,
                'output_features': self.output_variables,
                'input_size': self.input_size,
                'output_size': self.output_size,
                'emulator_type': self.emulator_type,
                'target_num_levels': self.target_num_levels,
                'reduced_grid': {
                    'method': self.reduced_grid_method,
                    'gradient_weight': self.reduced_grid_gradient_weight,
                },
                'input_cf_mapping': input_cf_mapping,
                'output_cf_mapping': output_cf_mapping
            }
        }
        if output_file:
            self.save_processed_data(result, output_file)
        return result

    def save_processed_data(self, data: Dict, filename: str) -> None:
        filepath = Path(filename); filepath.parent.mkdir(parents=True, exist_ok=True)
        if filename.endswith('.npz'):
            np.savez_compressed(filename, **data)
        elif filename.endswith('.pt'):
            torch_data = {
                'inputs': torch.from_numpy(data['inputs']),
                'targets': torch.from_numpy(data['targets']),
                'lons': torch.from_numpy(data['lons']),
                'lats': torch.from_numpy(data['lats']),
                'input_mean': torch.from_numpy(data['input_mean']),
                'input_std': torch.from_numpy(data['input_std']),
                'metadata': data['metadata']
            }
            torch.save(torch_data, filename)


def create_training_data_from_netcdf(netcdf_file: str,
                                     config: Dict,
                                     output_file: str,
                                     max_patterns: int = 400000) -> str:
    # Use config-specified pair if available
    data_config = config.get('data', {})
    atm = data_config.get('atm_file')
    ocn = data_config.get('ocean_file')
    max_patterns = data_config.get('max_patterns', max_patterns)
    thin_fraction = data_config.get('thin_fraction', 1.0)
    preparer = UFSEmulatorDataBuilder(config)
    preparer.prepare_training_data(atm, ocn, max_patterns, output_file, thin_fraction)
    return output_file


if __name__ == "__main__":
    # Minimal CLI example (expects --atm and --ocn)
    import argparse
    p = argparse.ArgumentParser(description='Prepare training data from CF-1 NetCDF')
    p.add_argument('--atm', required=True)
    p.add_argument('--ocn', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max', type=int, default=400000)
    a = p.parse_args()
    cfg = {'domain': {}, 'data': {}}
    UFSEmulatorDataBuilder(cfg).prepare_training_data(a.atm, a.ocn, a.max, a.out)
