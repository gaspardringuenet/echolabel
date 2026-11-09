import json
import subprocess

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from echolabel import app
from echolabel.core.manifest import ImagesDatasetManifest


def test_repeated_reuse_regenerates_only_current_annotations(
    tmp_cache, sample_manifest_with_coords, sample_csv_file, default_config, monkeypatch, tmp_path
):
    image = tmp_cache.img_dataset / "image_0000.png"
    for metadata in sample_manifest_with_coords.images:
        Image.new("RGB", (60, 100)).save(tmp_cache.img_dataset / metadata.filename)
    original_image = image.read_bytes()
    coords = tmp_cache.img_dataset / "image_0000_coords.npz"
    original_coords = coords.read_bytes()
    extra_metadata = tmp_cache.img_dataset / "metadata.json"
    extra_metadata.write_text('{"keep": true}')
    extra_file = tmp_cache.labelme / "keep.txt"
    extra_file.write_text("keep")
    stale = tmp_cache.labelme / "stale.json"
    stale.write_text('{"stale": true}')
    default_config.reuse_images = True
    source = tmp_path / "source.nc"
    source.touch()
    sample_manifest_with_coords.cache_key = app.image_cache_key(
        source=source,
        datavars_config=default_config.datavars,
        freqs_hz=default_config.channels_frequency_nominal,
        frame_width=default_config.frame_width,
        range_samples_slice=default_config.range_samples_slice,
        vmin=default_config.vmin,
        vmax=default_config.vmax,
        echogram_cmap=default_config.echogram_cmap,
        bg_color=default_config.bg_color,
        oklab=default_config.oklab,
    )
    sample_manifest_with_coords.save(tmp_cache.img_dataset)
    application = app.EcholabelApp(default_config, cache_dir=tmp_cache.root)

    def no_rebuild(**_kwargs):
        raise AssertionError("cached images must not be rebuilt")

    seen = []

    def inspect_annotations(cache):
        seen.append({p.name: json.loads(p.read_text()) for p in cache.labelme.glob("*.json")})

    monkeypatch.setattr(app, "build_images_dataset", no_rebuild)
    monkeypatch.setattr(app, "_run_labelme", inspect_annotations)
    monkeypatch.setattr(app, "_parse_to_csv", lambda *_args: None)

    application.run(source, sample_csv_file)
    assert "stale.json" not in seen[0]
    assert any(shape["region_id"] == 1 for entry in seen[0].values() for shape in entry["shapes"])

    stale.write_text('{"stale": true}')
    application.run(source, tmp_path / "new.csv")
    assert seen[1] == {}

    stale.write_text('{"stale": true}')
    application.run(source, sample_csv_file)
    assert seen[2] == seen[0]
    assert image.read_bytes() == original_image
    assert coords.read_bytes() == original_coords
    assert extra_metadata.read_text() == '{"keep": true}'
    assert extra_file.read_text() == "keep"
    assert ImagesDatasetManifest.load(tmp_cache.img_dataset) == sample_manifest_with_coords


def test_missing_manifest_rebuilds(tmp_cache, sample_manifest, monkeypatch):
    builds = []
    unrelated = tmp_cache.root / "important.txt"
    unrelated.write_text("keep")

    def build(**kwargs):
        builds.append(kwargs)
        sample_manifest.save(tmp_cache.img_dataset)
        return sample_manifest

    monkeypatch.setattr(app, "build_images_dataset", build)
    assert app._prepare_labelme_dataset(tmp_cache, True, source="source.zarr") == sample_manifest
    assert builds == [{"source": "source.zarr"}]
    assert ImagesDatasetManifest.load(tmp_cache.img_dataset) == sample_manifest
    assert unrelated.read_text() == "keep"


def test_app_passes_effective_oklab_settings(
    tmp_cache, sample_manifest_with_coords, default_config, monkeypatch, tmp_path
):
    default_config.echogram_cmap = "OKLab"
    default_config.update({"oklab": {"sat_max": 15, "l_range": [0.2, 0.8], "c_max": 0.1, "theta": 1.0}})
    captured = {}

    def prepare(_cache, _reuse, **kwargs):
        captured.update(kwargs)
        return sample_manifest_with_coords

    monkeypatch.setattr(app, "_prepare_labelme_dataset", prepare)
    monkeypatch.setattr(app, "_run_labelme", lambda *_args: None)
    monkeypatch.setattr(app, "_parse_to_csv", lambda *_args: None)
    app.EcholabelApp(default_config, tmp_cache.root).run("source.zarr", tmp_path / "new.csv")
    assert captured["oklab"] == {"sat_max": 15, "l_range": [0.2, 0.8], "c_max": 0.1, "theta": 1.0}


def test_labelme_failure_does_not_export(tmp_cache, sample_manifest_with_coords, default_config, monkeypatch, tmp_path):
    output = tmp_path / "regions.csv"
    output.write_text("original")
    monkeypatch.setattr(app, "_prepare_labelme_dataset", lambda *_args, **_kwargs: sample_manifest_with_coords)
    monkeypatch.setattr(app, "_read_regions_csv", lambda *_args: None)
    monkeypatch.setattr(app, "_parse_to_csv", lambda *_args: pytest.fail("CSV export must not run"))

    def fail(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, "labelme")

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="LabelMe exited.*labelme.log"):
        app.EcholabelApp(default_config, tmp_cache.root).run("source.zarr", output)
    assert output.read_text() == "original"


def test_annotation_rejects_single_sample_axis(
    tmp_cache, sample_manifest_with_coords, default_config, monkeypatch, tmp_path
):
    sample_manifest_with_coords.images[0].save_coordinates(
        tmp_cache.img_dataset,
        np.array(["2020-01-01"], dtype="datetime64[ns]"),
        np.array([1.0, 2.0]),
    )
    monkeypatch.setattr(app, "_prepare_labelme_dataset", lambda *_args, **_kwargs: sample_manifest_with_coords)
    monkeypatch.setattr(app, "_run_labelme", lambda *_args: pytest.fail("LabelMe must not run"))
    with pytest.raises(ValueError, match="at least two samples on each axis"):
        app.EcholabelApp(default_config, tmp_cache.root).run("source.zarr", tmp_path / "regions.csv")


def test_csv_export_preserves_original_and_prior_backups(tmp_cache, sample_manifest, monkeypatch, tmp_path):
    output = tmp_path / "regions.csv"
    output.write_text("original")
    backup = tmp_path / "regions_backup.csv"
    backup.write_text("older backup")
    monkeypatch.setattr(app.labelme_reader, "parse_echolabel", lambda *_args, **_kwargs: pd.DataFrame({"x": [1]}))
    app._parse_to_csv(tmp_cache, sample_manifest, output)
    assert output.read_text() == "x\n1\n"
    assert backup.read_text() == "older backup"
    assert (tmp_path / "regions_backup_1.csv").read_text() == "original"


def test_csv_write_failure_preserves_original(tmp_cache, sample_manifest, monkeypatch, tmp_path):
    output = tmp_path / "regions.csv"
    output.write_text("original")
    monkeypatch.setattr(app.labelme_reader, "parse_echolabel", lambda *_args, **_kwargs: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        app._parse_to_csv(tmp_cache, sample_manifest, output)
    assert output.read_text() == "original"
    assert set(tmp_path.iterdir()) == {output, tmp_cache.root}


@pytest.mark.parametrize(
    "change", ["settings", "source", "missing_image", "missing_coordinates", "corrupt_image", "corrupt_coordinates"]
)
def test_cache_mismatch_or_missing_files_rebuilds(
    tmp_cache, sample_manifest, default_config, monkeypatch, tmp_path, change
):
    source = tmp_path / "source.nc"
    source.write_text("original")
    params = {
        "source": source,
        "output_dir": tmp_cache.img_dataset,
        "datavars_config": default_config.datavars,
        "freqs_hz": default_config.channels_frequency_nominal,
        "frame_width": default_config.frame_width,
        "range_samples_slice": default_config.range_samples_slice,
        "vmin": default_config.vmin,
        "vmax": default_config.vmax,
        "echogram_cmap": default_config.echogram_cmap,
        "bg_color": default_config.bg_color,
        "oklab": default_config.oklab,
    }
    key_params = {k: v for k, v in params.items() if k != "output_dir"}
    sample_manifest.cache_key = app.image_cache_key(**key_params)
    sample_manifest.save(tmp_cache.img_dataset)
    for metadata in sample_manifest.images:
        Image.new("RGB", (2, 2)).save(tmp_cache.img_dataset / metadata.filename)
        metadata.save_coordinates(
            tmp_cache.img_dataset,
            np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"),
            np.array([1.0, 2.0]),
        )
    if change == "settings":
        params["vmin"] = -80
    elif change == "source":
        source.write_text("changed source size")
    elif change == "missing_image":
        (tmp_cache.img_dataset / sample_manifest.images[0].filename).unlink()
    elif change == "missing_coordinates":
        (tmp_cache.img_dataset / "image_0000_coords.npz").unlink()
    elif change == "corrupt_image":
        (tmp_cache.img_dataset / "image_0000.png").write_bytes(b"invalid")
    else:
        (tmp_cache.img_dataset / "image_0000_coords.npz").write_bytes(b"invalid")
    builds = []
    monkeypatch.setattr(app, "build_images_dataset", lambda **kwargs: builds.append(kwargs) or sample_manifest)
    app._prepare_labelme_dataset(tmp_cache, True, **params)
    assert builds == [params]
