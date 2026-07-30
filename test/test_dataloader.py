import pytest
from xarray import Dataset

from echolabel.core.dataloader import normalize_vars, open_dataset


@pytest.mark.parametrize(
    "format, convention, file",
    [
        ("Zarr store", "Echopype", "sample_echopype_zarr_store"),
        ("NetCDF file", "IMOS SOOP-BA", "sample_IMOS_netCDF_file"),
        ("NetCDF folder", "IMOS SOOP-BA", "sample_IMOS_netCDF_dir"),
    ],
)
def test_open_dataset(format, convention, file, request):
    """Open acoustic file(s) with a given format and convention"""
    # Get actual fixture (path to file)
    file = request.getfixturevalue(file)

    # Open dataset with custom function and assert there is no error
    ds_Sv = open_dataset(file, preprocess_fn=normalize_vars)

    # Check class returned
    assert isinstance(ds_Sv, Dataset), f"Wrong class returned: {ds_Sv.__class__.__name__}"

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
