import json
import math

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from labelme._label_file import LabelFile
from PIL import Image

from echolabel.core import builder
from echolabel.core.manifest import ImagesDatasetManifest
from echolabel.regions2d_extension import labelme_reader, labelme_writer
from echolabel.utils.echograms import RGB, OKLab
from echolabel.utils.images import array2image


def test_oklab_defaults_custom_settings_and_background():
    data = np.array([[[-70, -55, -80], [np.nan, np.nan, np.nan]]], dtype=np.float32)
    expected = OKLab(vmin=-90, vmax=-50, sat_max=math.sqrt(800), l_range=(0.3, 0.75), c_max=0.25, theta=0.85 * math.pi)(
        data
    )
    encoder = OKLab(vmin=-90, vmax=-50, sat_max=math.sqrt(800), l_range=(0.3, 0.75), c_max=0.25, theta=0.85 * math.pi)
    image = np.asarray(array2image(data, encoder))
    np.testing.assert_array_equal(image[0, 0], np.round(expected[0, 0] * 255).astype(np.uint8))
    np.testing.assert_array_equal(image[0, 1], [128, 128, 128])

    custom_encoder = OKLab(vmin=-90, vmax=-50, sat_max=math.sqrt(800), l_range=(0.3, 0.75), c_max=0.25, theta=0.0)
    custom = np.asarray(array2image(data, custom_encoder))
    expected_custom = OKLab(vmin=-90, vmax=-50, sat_max=math.sqrt(800), l_range=(0.3, 0.75), c_max=0.25, theta=0.0)(
        data
    )
    np.testing.assert_array_equal(custom[0, 0], np.round(expected_custom[0, 0] * 255).astype(np.uint8))
    assert not np.array_equal(image[0, 0], custom[0, 0])


def test_other_colormaps_unchanged():
    rgb = np.array([[[-90, -70, -50], [np.nan, np.nan, np.nan]]], dtype=np.float32)
    np.testing.assert_array_equal(
        np.asarray(array2image(rgb, RGB(vmin=-90, vmax=-50))),
        [[[0, 128, 255], [128, 128, 128]]],
    )
    single = np.array([[-90, -70, -50, np.nan]], dtype=np.float32)
    image = np.asarray(array2image(single, vmin=-90, vmax=-50, echogram_cmap="viridis"))
    np.testing.assert_array_equal(image[0, -1], [128, 128, 128, 255])


@pytest.mark.parametrize(
    "setting,value",
    [("sat_max", 10.0), ("l_range", (0.1, 0.6)), ("c_max", 0.05), ("theta", 0.0)],
)
def test_each_oklab_setting_changes_output(setting, value):
    data = np.array([[[-70, -55, -80]]], dtype=np.float32)
    settings = {
        "vmin": -90,
        "vmax": -50,
        "sat_max": math.sqrt(800),
        "l_range": (0.3, 0.75),
        "c_max": 0.25,
        "theta": 0.85 * math.pi,
    }
    original = np.asarray(array2image(data, OKLab(**settings)))
    changed = np.asarray(array2image(data, OKLab(**(settings | {setting: value}))))
    assert not np.array_equal(original, changed)


def test_custom_encoder_receives_raw_data():
    data = np.array([[[-90, -50, np.nan], [np.nan, np.nan, np.nan]]], dtype=np.float32)
    calls = []

    def encode(values):
        calls.append(values.copy())
        return np.full_like(values, 0.4)

    image = np.asarray(array2image(data, encode, bg_color="red"))
    np.testing.assert_array_equal(calls[0], data)
    np.testing.assert_array_equal(image, [[[102, 102, 102], [255, 0, 0]]])


def test_image_shape_must_match_encoder():
    with pytest.raises(ValueError, match="encoder"):
        array2image(np.zeros((2, 2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="encoder"):
        array2image(np.zeros((2, 2), dtype=np.float32), RGB())


def test_builder_records_and_uses_effective_oklab_settings(tmp_path, monkeypatch):
    data = np.array([[[-70, -55, -80], [-75, -60, -85]], [[-65, -50, -75], [-80, -65, -90]]], dtype=np.float32)
    dataset = xr.Dataset(
        {"acouvar": (("zvar", "tvar", "fvar"), data)},
        coords={
            "zvar": [1, 2],
            "tvar": np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"),
            "fvar": [38000, 70000, 120000],
        },
    )
    monkeypatch.setattr(builder, "open_dataset", lambda *_args: dataset)
    source = tmp_path / "source.zarr"
    source.mkdir()
    manifest = builder.build_images_dataset(
        source=source,
        output_dir=tmp_path,
        datavars_config={},
        freqs_hz=[38000, 70000, 120000],
        vmin=-90,
        vmax=-50,
        echogram_cmap="OKLab",
        bg_color="grey",
        oklab={"theta": 0.0},
    )
    settings = ImagesDatasetManifest.load(tmp_path).viz_params["oklab"]
    assert settings == manifest.viz_params["oklab"]
    assert settings["theta"] == 0.0
    assert settings["sat_max"] == 28.3
    with Image.open(tmp_path / "image_0000.png") as image:
        encoder = OKLab(
            vmin=-90,
            vmax=-50,
            sat_max=settings["sat_max"],
            l_range=tuple(settings["l_range"]),
            c_max=settings["c_max"],
            theta=settings["theta"],
        )
        np.testing.assert_array_equal(np.asarray(image), np.asarray(array2image(data, encoder)))


def test_builder_selects_rgb_encoder(tmp_path):
    data = np.array([[[-90, -70, -50], [np.nan, -80, -40]]], dtype=np.float32)
    array = xr.DataArray(data, dims=("zvar", "tvar", "fvar"))
    output = tmp_path / "rgb.png"
    builder.save_echogram_image(array, output, -90, -50, "RGB", "grey")
    with Image.open(output) as image:
        np.testing.assert_array_equal(np.asarray(image), [[[0, 128, 255], [0, 64, 255]]])


def test_builder_preserves_single_ping_and_depth(sample_echopype_zarr_store, default_config, tmp_path):
    manifest = builder.build_images_dataset(
        sample_echopype_zarr_store,
        tmp_path,
        default_config.datavars,
        freqs_hz=[38000.0],
        frame_width=1,
        range_samples_slice=slice(0, 1),
        vmin=-90,
        vmax=-50,
        echogram_cmap="viridis",
        bg_color="grey",
    )
    assert len(manifest.images) == 100
    with Image.open(tmp_path / manifest.images[-1].filename) as image:
        assert image.size == (1, 1)


def test_builder_merges_single_ping_remainder(sample_echopype_zarr_store, default_config, tmp_path):
    manifest = builder.build_images_dataset(
        sample_echopype_zarr_store,
        tmp_path,
        default_config.datavars,
        freqs_hz=[38000.0],
        frame_width=99,
        vmin=-90,
        vmax=-50,
        echogram_cmap="viridis",
        bg_color="grey",
    )
    assert len(manifest.images) == 1
    with Image.open(tmp_path / manifest.images[0].filename) as image:
        assert image.width == 100


def test_real_data_labelme_roundtrip_with_stepped_depth(sample_echopype_zarr_store, default_config, tmp_cache):
    manifest = builder.build_images_dataset(
        sample_echopype_zarr_store,
        tmp_cache.img_dataset,
        default_config.datavars,
        freqs_hz=[38000.0],
        frame_width=100,
        range_samples_slice=slice(0, 10, 2),
        vmin=-90,
        vmax=-50,
        echogram_cmap="viridis",
        bg_color="grey",
    )
    times, depths = manifest.images[0].load_coordinates(tmp_cache.img_dataset)
    original = pd.DataFrame(
        [
            {
                "region_id": 1,
                "region_name": "fish",
                "time": np.array([times[1], times[3], times[3], times[1]]),
                "depth": np.array([depths[1], depths[1], depths[3], depths[3]]),
            }
        ]
    )
    annotations = labelme_writer.regions2d_to_labelme(original, manifest, tmp_cache.img_dataset)
    assert len(annotations) == 1
    filename, annotation = next(iter(annotations.items()))
    assert (annotation["imageWidth"], annotation["imageHeight"]) == (100, 5)
    path = tmp_cache.labelme / filename
    path.write_text(json.dumps(annotation))
    assert len(LabelFile(str(path)).shapes) == 1
    result = labelme_reader.parse_echolabel(tmp_cache, manifest, original_df=original)
    np.testing.assert_array_equal(result.loc[0, "time"], original.loc[0, "time"])
    np.testing.assert_array_equal(result.loc[0, "depth"], original.loc[0, "depth"])


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"frame_width": 0}, "frame_width"),
        ({"range_samples_slice": slice(0, None, -1)}, "step"),
        ({"range_samples_slice": slice(0, 0)}, "must not be empty"),
        ({"freqs_hz": [18000.0, 38000.0]}, "one or three"),
        ({"vmin": -50, "vmax": -90}, "vmin and vmax"),
    ],
)
def test_builder_rejects_invalid_input(sample_echopype_zarr_store, default_config, tmp_path, overrides, message):
    params = {
        "source": sample_echopype_zarr_store,
        "output_dir": tmp_path,
        "datavars_config": default_config.datavars,
        "vmin": -90,
        "vmax": -50,
        "echogram_cmap": "viridis",
        "bg_color": "grey",
    }
    with pytest.raises(ValueError, match=message):
        builder.build_images_dataset(**(params | overrides))


@pytest.mark.parametrize("axis", ["time", "depth"])
def test_builder_rejects_descending_coordinates(
    sample_echopype_zarr_store, default_config, tmp_path, monkeypatch, axis
):
    dataset = builder.open_dataset(sample_echopype_zarr_store, builder.normalize_vars)
    if axis == "time":
        dataset = dataset.isel(tvar=slice(None, None, -1))
    else:
        dataset = dataset.isel(zvar=slice(None, None, -1))
    monkeypatch.setattr(builder, "open_dataset", lambda *_args: dataset)
    with pytest.raises(ValueError, match=f"{axis} axis.*strictly increasing"):
        builder.build_images_dataset(
            sample_echopype_zarr_store,
            tmp_path,
            default_config.datavars,
            vmin=-90,
            vmax=-50,
            echogram_cmap="viridis",
            bg_color="grey",
        )
