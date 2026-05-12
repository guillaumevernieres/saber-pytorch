"""CF-1 standard name mappings for atmospheric and ocean/ice variables.

Short names are the keys used in YAML configs and training data files.
Values are the CF-1 standard names expected in NetCDF input files and by
SABER/JEDI at runtime.
"""

# Atmospheric variables
CF_ATM = {
    "lat": "latitude",
    "lon": "longitude",
    "tair": "air_temperature",
    "uatm": "eastward_wind",
    "vatm": "northward_wind",
    "tsfc": "snow_ice_surface_temperature",
    "qref": "water_vapor_mixing_ratio_wrt_moist_air",
    "flwdn": "surface_downwelling_longwave_flux_in_air",
    "fswdn": "surface_downwelling_shortwave_flux_in_air",
    "pressfc": "air_pressure_at_surface",
}

# Ocean / sea-ice variables
CF_OCN = {
    "lat": "latitude",
    "lon": "longitude",
    "sst": "sea_water_potential_temperature",
    "sea_water_potential_temperature": "sea_water_potential_temperature",
    "sss": "sea_water_salinity",
    "sea_water_salinity": "sea_water_salinity",
    "aice": "sea_ice_area_fraction",
    "hi": "sea_ice_volume",
    "hs": "sea_ice_snow_volume",
    "thick": "sea_water_cell_thickness",
    "sea_water_cell_thickness": "sea_water_cell_thickness",
    "h": "h",           # MOM6 layer thickness (alternative name)
    "h_surface": "sea_water_cell_thickness",  # Surface layer thickness only
    "sice": "sea_ice_salinity",
    "uocn": "eastward_sea_water_velocity",
    "vocn": "northward_sea_water_velocity",
    "Temp": "Temp",     # MOM6 temperature variable name
    "Salt": "Salt",     # MOM6 salinity variable name
    "uo": "uo",         # UFS ocean x-velocity
    "vo": "vo",         # UFS ocean y-velocity
    "so": "so",         # UFS salinity
    "temp": "temp",     # UFS temperature
    "ho": "ho",         # UFS layer thickness
}

# Default atmospheric level index used when the atmospheric file has no vertical
# dimension (single-level / already-interpolated files).
# Override per-run via domain.atm_level_index in the config YAML.
# Typical values: 63 (64-level), 126 (127-level), 127 (128-level).
DEFAULT_ATM_LEVEL = 126
