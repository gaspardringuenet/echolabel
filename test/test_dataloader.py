from pathlib import Path

import pytest

from echolabel.core.dataloader import _normalize_vars, open_dataset

DATA_DIR: Path = Path(__file__).parent


@pytest.mark.parametrize(
    "format, convention, file",
    [
        ("Zarr store", "Echopype", DATA_DIR / "data/echopype_ds_MVBS.zarr"),
        ("NetCDF file", "IMOS SOOP-BA", DATA_DIR / "data/imos_nc_dir/imos_0.nc"),
        ("NetCDF folder", "IMOS SOOP-BA", DATA_DIR / "data/imos_nc_dir"),
    ],
)
def test_open_dataset(format: str, convention: str, file: Path):
    """Open acoustic file(s) with a given format and convention"""

    # Open dataset with custom function and assert there is no error
    ds_Sv = open_dataset(file, preprocess_fn=_normalize_vars)

    # Check class returned
    assert ds_Sv.__class__.__name__ == "Dataset", f"Wrong data class returned: {ds_Sv.__class__.__name__}"

    required_coords = {"fvar", "cvar", "tvar", "zvar"}
    required_vars = required_coords | {"acouvar"}
    valid_dims = {"fvar", "tvar", "zvar"}

    # Check dimensions are correct
    assert set(ds_Sv.dims) == valid_dims, (
        f"Normalized dataset should have exactly the required dimensions: {valid_dims}. Got {set(ds_Sv.dims)}."
    )

    # Check coords normalization
    assert required_coords <= set(ds_Sv.coords), (
        f"Normalized dataset does not contain required coords: {required_coords}. Coords: {set(ds_Sv.coords)}."
    )

    # Check vars normalization
    assert required_vars <= set(ds_Sv.variables), (
        f"Normalized dataset does not contain required variables: {required_vars}."
    )
