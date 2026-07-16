import os
from pathlib import Path

import xarray as xr

# Required variable and dimensions in the dataset. Currently based on the MVBS format produced by Echopype v0.10.0
REQUIRED_VARS = {"acoustic": "Sv", "channel": "frequency_nominal"}
REQUIRED_DIMS = {"time": "ping_time", "depth": "depth", "channel": "channel"}


def open_dataset(path: Path) -> xr.Dataset:
    """
    Lazy-load an acoustic dataset.

    Supports:
    - Single .zarr directory
    - Single .nc file
    - Directory containing multiple .nc files (concatenated along ping_time)
    """

    if path.is_dir():
        # Check if it's a .zarr directory
        if path.suffix == ".zarr":
            ds_MVBS = xr.open_dataset(path, engine="zarr", chunks=None)
            _validate_dataset(ds_MVBS)
            return ds_MVBS

        # Otherwise, assume directory of .nc files
        nc_files = sorted([str(path / f) for f in os.listdir(str(path)) if f.endswith(".nc") and not f.startswith(".")])

        if not nc_files:
            raise ValueError(f"No .nc files found in directory: {path}")

        return xr.open_mfdataset(
            nc_files,
            engine="netcdf4",
            combine="nested",
            concat_dim=REQUIRED_DIMS["ping_time"],
            data_vars="minimal",
            chunks="auto",
            preprocess=_validate_dataset,
        )

    # Single file
    if path.suffix == ".nc":
        ds_MVBS = xr.open_dataset(path, engine="netcdf4", chunks="auto")
    else:
        raise ValueError(f"Invalid file format: {path.suffix}. Expected .nc, .zarr, or directory containing .nc files")

    _validate_dataset(ds_MVBS)
    return ds_MVBS


def _validate_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Preprocess function to validate dataset structure."""
    for type, name in REQUIRED_VARS.items():
        if name not in ds.variables:
            raise ValueError(f"Acoustic dataset must contain a {type} variable with name '{name}'.")
    for type, name in REQUIRED_DIMS.items():
        if name not in ds.dims:
            raise ValueError(f"Acoustic dataset must contain a {type} dimension with name '{name}'.")
    return ds
