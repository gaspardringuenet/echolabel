"""Shared test fixtures for echolabel tests"""

from datetime import UTC, datetime
from pathlib import Path

import echoregions as er
import numpy as np
import pandas as pd
import pytest

from echolabel.config.cache import CachePathsConfig
from echolabel.config.config import Config
from echolabel.core.manifest import ImageMetadata, ImagesDatasetManifest


@pytest.fixture
def tmp_cache(tmp_path):
    """Create a temporary cache directory with proper structure"""
    cache = CachePathsConfig(root=tmp_path / "cache")
    cache.mkdir()
    return cache


@pytest.fixture
def default_config():
    """Load default configuration for testing"""
    return Config.from_defaults()


@pytest.fixture
def custom_config(default_config):
    """Create a custom config with modified parameters for testing"""
    config = default_config.copy()
    config.vmin = -80.0
    config.vmax = -30.0
    config.frame_width = 100
    return config


@pytest.fixture
def custom_config_rgb(default_config):
    """Create a custom config with modified parameters for testing"""
    config = default_config.copy()
    config.channels_frequency_nominal = [38e3, 70e3, 120e3]  # works with all test datasets
    config.echogram_cmap = "RGB"
    config.frame_width = 100
    return config


@pytest.fixture
def sample_manifest(tmp_cache):
    """Create a sample manifest for testing"""
    images = [
        ImageMetadata(
            filename="image_0000.png",
            ping_sample_start=0,
            ping_sample_end=60,
            range_sample_start=1,
            range_sample_end=101,
        ),
        ImageMetadata(
            filename="image_0001.png",
            ping_sample_start=60,
            ping_sample_end=120,
            range_sample_start=1,
            range_sample_end=101,
        ),
        ImageMetadata(
            filename="image_0002.png",
            ping_sample_start=120,
            ping_sample_end=180,
            range_sample_start=1,
            range_sample_end=101,
        ),
    ]

    manifest = ImagesDatasetManifest(
        source="test_source.nc",
        created_at=datetime.now(UTC).isoformat(),
        shape={"tvar": 60, "zvar": 100},
        freqs_hz=[38e3, 70e3, 120e3],
        viz_params={"vmin": -80.0, "vmax": -30.0, "cmap": "RGB"},
        images=images,
    )

    return manifest


@pytest.fixture
def sample_manifest_with_coords(tmp_cache, sample_manifest):
    """
    Create a sample manifest and save coordinate arrays to disk
    Three consecutive 1 min long frames:
    - time axis: 60 ping samples with 1s interval
    - depth axis: 100 depths samples in [1, 100m]
    """

    ping_coords = np.arange(
        np.datetime64("2000-01-01 00:00:00"),
        np.datetime64("2000-01-01 00:01:00"),
        np.timedelta64(1, "s"),
    ).astype("datetime64[ns]")
    range_coords = np.arange(1, 101)

    for img_meta in sample_manifest.images:
        img_meta.save_coordinates(
            tmp_cache.img_dataset,
            ping_coords,
            range_coords,
        )

        # Shift ping coords to the right
        ping_coords = ping_coords + np.timedelta64(1, "m")

    # Save manifest
    sample_manifest.save(tmp_cache.img_dataset)
    return sample_manifest


@pytest.fixture
def data_dir():
    """Get path to test data directory"""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_echopype_zarr_store(data_dir):
    """Get path to sampe Echopype Zarr store (if it exists)"""
    zarr_path = data_dir / "echopype_ds_MVBS.zarr"
    if zarr_path.exists():
        return zarr_path
    pytest.skip("Sample Echopype Zarr store not found")


@pytest.fixture
def sample_IMOS_netCDF_dir(data_dir):
    """Get path to sample IMOS NetCDF directory (if it exists)"""
    nc_dirpath = data_dir / "imos_nc_dir"
    if nc_dirpath.is_dir():
        return nc_dirpath
    pytest.skip("Sample IMOS NetCDF directory not found")


@pytest.fixture
def sample_IMOS_netCDF_file(sample_IMOS_netCDF_dir):
    """Get path to sample IMOS NetCDF file (if it exists)"""
    nc_path = sample_IMOS_netCDF_dir / "imos_0.nc"
    if nc_path.is_file():
        return nc_path
    pytest.skip("Sample IMOS NetCDF file not found")


@pytest.fixture
def sample_labelme_annotation_0():
    """
    Create a sample LabelMe annotation structure for the
    image_0000.png in sample_manifest_with_coords.
    """
    return {
        "version": "6.0.0",
        "flags": {},
        "shapes": [
            {
                "label": "Fish_0",
                "points": [[27, 17], [18, 20], [23, 28], [48, 23], [45, 15]],
                "group_id": None,
                "description": "Some fish region",
                "shape_type": "polygon",
                "flags": {},
                "mask": None,
            },
        ],
        "imagePath": "../images/image_0000.png",
        "imageData": None,
        "imageHeight": 100,
        "imageWidth": 60,
    }


@pytest.fixture
def sample_labelme_annotation_1():
    """
    Create a sample LabelMe annotation structure for the
    image_0001.png in sample_manifest_with_coords.
    """
    return {
        "version": "6.0.0",
        "flags": {},
        "shapes": [
            {
                "label": "Other_1",
                "points": [[19, 15], [48, 26]],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
                "mask": None,
            },
        ],
        "imagePath": "../images/image_0001.png",
        "imageData": None,
        "imageHeight": 100,
        "imageWidth": 60,
    }


@pytest.fixture
def sample_labelme_annotation_2():
    """
    Create a sample LabelMe annotation structure for the
    image_0002.png in sample_manifest_with_coords.
    """
    return {
        "version": "6.0.0",
        "flags": {},
        "shapes": [
            {
                "label": "Fish_2",
                "points": [[27, 17], [18, 20], [23, 28], [48, 23], [45, 15]],
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
                "mask": None,
            },
            {
                "label": "Other_2",
                "points": [[19, 15], [48, 26]],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
                "mask": None,
            },
        ],
        "imagePath": "../images/image_0002.png",
        "imageData": None,
        "imageHeight": 100,
        "imageWidth": 60,
    }


@pytest.fixture
def sample_csv_file(tmp_path):
    """
    Create a sample CSV file for testing annotation loading.
    Corresponds to images in sample_manifest_with_coords.
    """
    csv_path = tmp_path / "regions.csv"

    # Create minimal CSV with required structure for echoregions
    # Generated from LabelMe annotations with corrected depth values
    csv_content = """
region_id,region_name,region_class,time,depth,region_bbox_left,region_bbox_right,region_bbox_top,region_bbox_bottom,echoview_version,region_structure_version,region_creation_type,region_type,region_notes,flags,group_id,description
1,Fish_0,Fish_0,"['2000-01-01T00:00:27' '2000-01-01T00:00:18' '2000-01-01T00:00:23' '2000-01-01T00:00:48' '2000-01-01T00:00:45']",[18 21 29 24 16],2000-01-01 00:00:18,2000-01-01 00:00:48,16,29,13.0.378.44817,13,2,1,[],[],,"Some fish region"
2,Other_1,Other_1,"['2000-01-01T00:01:19' '2000-01-01T00:01:19' '2000-01-01T00:01:48' '2000-01-01T00:01:48']",[16 27 27 16],2000-01-01 00:01:19,2000-01-01 00:01:48,16,27,13.0.378.44817,13,3,1,[],[],,""
3,Fish_2,Fish_2,"['2000-01-01T00:02:27' '2000-01-01T00:02:18' '2000-01-01T00:02:23' '2000-01-01T00:02:48' '2000-01-01T00:02:45']",[18 21 29 24 16],2000-01-01 00:02:18,2000-01-01 00:02:48,16,29,13.0.378.44817,13,2,1,[],[],,""
4,Other_2,Other_2,"['2000-01-01T00:02:19' '2000-01-01T00:02:19' '2000-01-01T00:02:48' '2000-01-01T00:02:48']",[16 27 27 16],2000-01-01 00:02:19,2000-01-01 00:02:48,16,27,13.0.378.44817,13,3,1,[],[],,""
    """

    csv_path.write_text(csv_content.strip())
    return csv_path


@pytest.fixture
def sample_regions2d(sample_csv_file):
    """Get Regions2D object"""
    regions = er.read_regions_csv(sample_csv_file)
    # Convert columns to match parse_labelme output
    for col in ["region_bbox_left", "region_bbox_right"]:
        regions.data[col] = pd.to_datetime(regions.data[col])
    # Convert string '[]' to actual empty lists
    for col in ["region_notes", "flags"]:
        regions.data[col] = regions.data[col].apply(lambda x: [] if x == "[]" else x)  # type: ignore
    return regions


@pytest.fixture
def mock_labelme_subprocess(monkeypatch):
    """Mock subprocess.run to prevent actual LabelMe execution"""
    called = []

    def mock_run(*args, **kwargs):
        called.append({"args": args, "kwargs": kwargs})

    import subprocess

    monkeypatch.setattr(subprocess, "run", mock_run)
    return called
