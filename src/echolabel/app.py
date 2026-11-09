import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import BadZipFile

import echoregions as er
import numpy as np
import pandas as pd
import platformdirs
from echoregions.regions2d import Regions2D
from PIL import Image, UnidentifiedImageError

from echolabel.config.cache import CachePathsConfig
from echolabel.config.config import Config, resolve_oklab_settings
from echolabel.core.builder import build_images_dataset, image_cache_key
from echolabel.core.manifest import ImagesDatasetManifest
from echolabel.regions2d_extension import labelme_reader, labelme_writer, shapes


class EcholabelApp:
    """Container class for the Echolabel app."""

    def __init__(self, config: Config, cache_dir: Path | None = None):
        self.name = "echolabel"
        self.cfg = config
        cache_dir = cache_dir or Path(platformdirs.user_cache_dir(self.name))
        self.cache = CachePathsConfig(root=cache_dir)
        self.cache.mkdir()

    def run(self, source: str | Path, output: str | Path):
        """Run the Labelme wrapper.

        Parameters
        ----------
        output : str | Path
            Path to the regions .csv file.
        reuse_images : bool
            Whether to reuse the existing images dataset in cache.
        """

        # Defensive typing of output
        output = Path(output)

        # Preparing params for builder
        builder_params = {
            "source": source,
            "output_dir": self.cache.img_dataset,
            "datavars_config": self.cfg.datavars,
            "freqs_hz": self.cfg.channels_frequency_nominal,
            "frame_width": self.cfg.frame_width,
            "range_samples_slice": self.cfg.range_samples_slice,
            "vmin": self.cfg.vmin,
            "vmax": self.cfg.vmax,
            "echogram_cmap": self.cfg.echogram_cmap,
            "bg_color": self.cfg.bg_color,
            "oklab": resolve_oklab_settings(self.cfg.oklab),
        }

        # Build images dataset and fetch metadata manifest
        manifest = _prepare_labelme_dataset(self.cache, self.cfg.reuse_images, **builder_params)

        for image in manifest.images:
            times, depths = image.load_coordinates(self.cache.img_dataset)
            if len(times) < 2 or len(depths) < 2:
                raise ValueError(
                    f"{image.filename} has fewer than two time or depth samples; "
                    "annotation requires at least two samples on each axis"
                )

        # Read the original regions once: used for export and to keep region identity
        original = _read_regions_csv(output)

        # LabelMe annotations belong to the current CSV, not the cached images.
        for annotation in self.cache.labelme.glob("*.json"):
            annotation.unlink()

        # Parse shape annotation file into per-image annotation JSON in LabelMe format
        if original is not None:
            _load_annotations(original, self.cache, manifest)

        # Open LabelMe on images dataset to edit / create new shapes
        _run_labelme(self.cache)

        # Parse LabelMe internal shapes JSON to a single CSV file with (time, depth) attributes
        _parse_to_csv(self.cache, manifest, output, original)


REQUIRED_CSV_COLUMNS = ["region_id", "time", "depth"]


def _read_regions_csv(file: Path) -> pd.DataFrame | None:
    """Read an Echoregions-compatible regions CSV. Returns None if the file does not exist."""
    if not file.is_file():
        return None
    if file.suffix != ".csv":
        raise ValueError(f"Invalid file format for library. Expected '.csv', got '{file.suffix}'")

    regions: Regions2D = er.read_regions_csv(file)
    df = pd.DataFrame(regions.data.copy())

    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Regions CSV {file} is missing required columns: {missing}")
    return df


def _prepare_labelme_dataset(
    cache_cfg: CachePathsConfig,
    reuse_images: bool,
    **builder_params,
) -> ImagesDatasetManifest:
    """
    Build the images dataset in cache and return manifest (clear cache
    beforehand).

    If reuse_images is True, try to load the existing manifest from cache.
    """

    if reuse_images:
        try:
            # Load existing manifest
            manifest = ImagesDatasetManifest.load(cache_cfg.img_dataset)
        except FileNotFoundError as e:
            print(f"Failed to load manifest. Rebuilding images dataset.\n{e}")
        else:
            key_params = {key: value for key, value in builder_params.items() if key != "output_dir"}
            current_key = image_cache_key(**key_params)
            if manifest.cache_key == current_key and manifest.images and _valid_cached_images(cache_cfg, manifest):
                return manifest
            print("Cached images do not match the source or settings, or are invalid. Rebuilding images.")

    # Remove only files belonging to the image dataset, not the user-supplied cache root.
    for pattern in ("image_*.png", "image_*_coords.npz", "manifest.json"):
        for path in cache_cfg.img_dataset.glob(pattern):
            path.unlink()
    cache_cfg.mkdir()
    # Build new dataset
    manifest = build_images_dataset(**builder_params)

    return manifest


def _valid_cached_images(cache_cfg: CachePathsConfig, manifest: ImagesDatasetManifest) -> bool:
    for metadata in manifest.images:
        try:
            times, depths = metadata.load_coordinates(cache_cfg.img_dataset)
            if times.ndim != 1 or depths.ndim != 1 or not len(times) or not len(depths):
                raise ValueError("invalid coordinate arrays")
            if not np.issubdtype(times.dtype, np.datetime64) or np.any(np.isnat(times)):
                raise ValueError("invalid time coordinates")
            if np.any(np.diff(times) <= np.timedelta64(0, "ns")):
                raise ValueError("time coordinates are not strictly increasing")
            if not np.issubdtype(depths.dtype, np.number):
                raise ValueError("invalid depth coordinates")
            if not np.all(np.isfinite(depths)) or np.any(np.diff(depths) <= 0):
                raise ValueError("depth coordinates are not finite and strictly increasing")
            with Image.open(cache_cfg.img_dataset / metadata.filename) as image:
                if image.size != (len(times), len(depths)):
                    raise ValueError("image and coordinate dimensions differ")
                image.verify()
        except (OSError, EOFError, KeyError, TypeError, ValueError, BadZipFile, UnidentifiedImageError) as exc:
            print(f"Invalid cached image {metadata.filename}: {exc}")
            return False
    return True


def _load_annotations(
    original: pd.DataFrame,
    cache_cfg: CachePathsConfig,
    manifest: ImagesDatasetManifest,
):
    """Write the annotations of an existing regions CSV to Labelme format"""
    labelme_data = labelme_writer.regions2d_to_labelme(original, manifest, cache_cfg.img_dataset)

    for filename, data in labelme_data.items():
        with open(cache_cfg.labelme / filename, "w") as f:
            json.dump(data, f, indent=4)


def _run_labelme(cache_cfg: CachePathsConfig) -> None:
    """Run Labelme as subprocess"""
    with open(cache_cfg.labelme_logs, "w") as log:
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "labelme",
                    str(cache_cfg.img_dataset),
                    "--output",
                    str(cache_cfg.labelme),
                ],
                stdout=log,
                stderr=log,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"LabelMe exited with status {exc.returncode}; see {cache_cfg.labelme_logs}") from exc


def _parse_to_csv(
    cache_cfg: CachePathsConfig,
    manifest: ImagesDatasetManifest,
    outfile: Path,
    original: pd.DataFrame | None = None,
) -> None:
    """Convert LabelMe files to Regions2D compatible DataFrame and save CSV"""

    # Parse output to Dataframe
    data = labelme_reader.parse_echolabel(cache_cfg, manifest, original_df=original)

    # Save to library file
    library_dir = outfile.parent
    library_dir.mkdir(parents=True, exist_ok=True)

    data = shapes.csv_safe(data)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=library_dir, delete=False) as temp:
            temp_path = Path(temp.name)
            data.to_csv(temp, index=False)

        if outfile.exists():
            index = 0
            while True:
                suffix = f"_{index}" if index else ""
                backup = library_dir / f"{outfile.stem}_backup{suffix}{outfile.suffix}"
                try:
                    target = backup.open("xb")
                except FileExistsError:
                    index += 1
                    continue
                try:
                    with target, outfile.open("rb") as source:
                        shutil.copyfileobj(source, target)
                except BaseException:
                    backup.unlink()
                    raise
                break
        os.replace(temp_path, outfile)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
