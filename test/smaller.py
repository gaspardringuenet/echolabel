import xarray as xr

if __name__ == "__main__":
    # For Zarr
    ds = xr.open_zarr("test/data/echopype_ds_MVBS.zarr")
    ds.isel(ping_time=slice(0, 100)).to_zarr("test/data_new/echopype_ds_MVBS_small.zarr", mode="w")

    # For NetCDF
    ds = xr.open_dataset("test/data/imos_nc_dir/imos_1.nc")
    ds.isel(time=slice(0, 200)).to_netcdf("test/data_new/imos_nc_dir/imos_1.0_small.nc", mode="w")
    ds.isel(time=slice(100, 200)).to_netcdf("test/data_new/imos_nc_dir/imos_1.1_small.nc", mode="w")
