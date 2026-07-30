import logging
import os
from pathlib import Path
from typing import Callable

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

#### Default conventions to try on dataset ####

type DataVarsConfig = dict[str, str | float | None]

CONFIG_EP: DataVarsConfig = {
    "acouvar": "Sv",
    "cvar": "channel",
    "fvar": "frequency_nominal",
    "tvar": "ping_time",
    "zvar": "depth",
}

CONFIG_IMOS: DataVarsConfig = {
    "acouvar": "Sv",
    "cvar": "channel",
    "fvar": None,
    "fvar_unit_scale": 1000,
    "tvar": "time",
    "zvar": "depth",
}

DEFAULT_CONFIGS: dict[str, DataVarsConfig] = {
    "Echopype": CONFIG_EP,
    "IMOS SOOP-BA": CONFIG_IMOS,
}


#### Normalization functions
#
def _normalize_vars_from_config(
    ds: xr.Dataset,
    config: DataVarsConfig,
) -> xr.Dataset:
    """Normalize a dataset using a given data variables config"""

    config = config.copy()

    # Fetch unit scale if it exists (if not assume 1)
    try:
        fvar_unit_scale = config.pop("fvar_unit_scale")
    except KeyError:
        fvar_unit_scale = 1
    if fvar_unit_scale is None:
        fvar_unit_scale = 1

    # Check that all provided var names exist in dataset
    non_none_values = {v for v in config.values() if v is not None}
    if not (non_none_values <= set(ds.variables)):
        raise ValueError(
            "Variables config does not match actual variable names."
            f"\n* Config variable names not in Dataset: {non_none_values - set(ds.variables)}"
        )

    # Check for "fvar" config: if none, use "cvar" (must be numerical)
    fvar = config.get("fvar")
    if fvar is None:
        # Rename to avoid using None as var name
        fvar = "fvar"
        config["fvar"] = "fvar"

        # Assign cvar to fvar after checks
        cvar = config.get("cvar")
        if cvar is None:
            raise ValueError("Channel variable name is None is config.")
        if not np.issubdtype(ds[cvar].dtype, np.number):
            raise ValueError("In the absence of fvar, channel variable should have a numeric dtype.")
        ds["fvar"] = ds[cvar].astype(np.float64)

    # fvar engineering
    ds = ds.set_coords(fvar)  # Make a coord
    ds[fvar] = fvar_unit_scale * ds[fvar]  # Scale to Hz

    # Normalize variable names
    # Create renaming dictionaries
    rename_dict = {v: k for k, v in config.items()}
    rename_dict[fvar] = "fvar"
    rename_dims_dict = {k: v for k, v in rename_dict.items() if v in ["cvar", "tvar", "zvar"]}

    ds = ds.rename_dims(rename_dims_dict)  # rename dimensions
    ds = ds.rename_vars(rename_dict)  # rename variables

    # Drop dimensions other than normalized
    ds = ds.drop_dims(set(ds.dims) - set(["cvar", "tvar", "zvar"]))

    # Swap cvar dim for fvar, enabling sel(fvar=...) and isel(fvar=...)
    ds = ds.swap_dims({"cvar": "fvar"})

    return ds


def normalize_vars(
    ds: xr.Dataset,
    default_configs: dict[str, DataVarsConfig] = DEFAULT_CONFIGS,
    custom_config: DataVarsConfig | None = None,
) -> xr.Dataset:

    errors = {}

    configs = default_configs.copy()
    if custom_config is not None:
        configs["CUSTOM"] = custom_config

    for convention_name, config in configs.items():
        try:
            ds = _normalize_vars_from_config(ds, config)
        except ValueError as e:
            errors[convention_name] = e
            continue
        else:
            logger.debug(f"Normalized dataset using {convention_name} convention.")
            return ds
    raise ValueError(f"Acoustic dataset could not be normalized. Per-convention errors:\n{errors}")


def open_dataset(
    path: Path,
    preprocess_fn: Callable[[xr.Dataset], xr.Dataset],
) -> xr.Dataset:
    """
    Lazy-load an acoustic dataset.

    Support formats:
    - Single .zarr directory
    - Single .nc file
    - Directory containing multiple .nc files (concatenated along ping_time)

    Supports acoustic data conventions:
    - IMOS SOOP-BA ('time', 'depth', 'channel' - containing freq in kHz, 'Sv')
    - Echopype ('ping_time', 'depth', 'channel' - str, 'frequency_nominal' - in Hz, 'Sv')
    - Custom (TODO)
    """

    if path.is_dir():
        # Check if it's a .zarr directory
        if path.suffix == ".zarr":
            ds = xr.open_dataset(path, engine="zarr", chunks=None)
            ds = preprocess_fn(ds)
            return ds

        # Otherwise, assume directory of .nc files
        nc_files = sorted([str(path / f) for f in os.listdir(str(path)) if f.endswith(".nc") and not f.startswith(".")])

        if not nc_files:
            raise ValueError(f"No .nc files found in directory: {path}")

        return xr.open_mfdataset(
            nc_files,
            engine="netcdf4",
            combine="nested",
            concat_dim="tvar",
            data_vars="minimal",
            chunks="auto",
            preprocess=preprocess_fn,
            compat="no_conflicts",
            join="outer",
        )

    # Single file
    if path.suffix == ".nc":
        ds = xr.open_dataset(path, engine="netcdf4", chunks="auto")
    else:
        raise ValueError(
            f"Invalid file format: '{path.suffix}'. Expected .nc, .zarr, or directory containing .nc files"
        )

    # Apply preprocessing
    ds = preprocess_fn(ds)

    return ds
