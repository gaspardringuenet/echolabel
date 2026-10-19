import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import echoregions as er
import pandas as pd
import platformdirs
from echoregions.regions2d import Regions2D

from echolabel.config.cache import CachePathsConfig
from echolabel.config.config import Config
from echolabel.core.builder import build_images_dataset
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
        }

        # Build images dataset and fetch metadata manifest
        manifest = _prepare_labelme_dataset(self.cache, self.cfg.reuse_images, **builder_params)

        # Read the original regions once: used for export and to keep region identity
        original = _read_regions_csv(output)

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

    If the reuse_images is True, tries to load existing manifest from cache
    (only clear JSON files).
    """

    if reuse_images:
        try:
            # Load existing manifest
            manifest = ImagesDatasetManifest.load(cache_cfg.img_dataset)
        except FileNotFoundError as e:
            print(f"Failed to load manifest. Rebuilding images dataset.\n{e}")
        else:
            # Clear JSON files
            json_files = glob.glob(str(cache_cfg.img_dataset / "*.json"))
            for file in json_files:
                os.remove(file)
            # Return manifest
            return manifest

    # Clear cache directory
    shutil.rmtree(cache_cfg.root)
    cache_cfg.mkdir()
    # Build new dataset
    manifest = build_images_dataset(**builder_params)

    return manifest


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
            check=False,
        )


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

    if outfile.is_file():
        update_safe_name = outfile.stem + "_backup" + outfile.suffix
        os.rename(outfile, library_dir / update_safe_name)  # Safety guard: rename the existing file before overwriting

    data = shapes.csv_safe(data)
    data.to_csv(outfile, index=False)
