import json
import os
import shutil
import subprocess
from pathlib import Path

import echoregions as er
import platformdirs
from echoregions.regions2d import Regions2D

from echolabel.config.cache import CachePathsConfig
from echolabel.core.builder import build_images_dataset
from echolabel.core.manifest import ImagesDatasetManifest
from echolabel.echoregions_extension import echolabel_writer, regions2d_parser


class EcholabelApp:
    """Container class for the Echolabel processing"""

    def __init__(self, cache_dir: Path | None = None):

        self.name = "echolabel_v2"
        self.cache = CachePathsConfig(root=Path(platformdirs.user_cache_dir(self.name)))
        self.cache.mkdir()

    def run(self, output: str | Path, reuse_images: bool, **builder_params):
        """Run the Labelme wrapper.

        Parameters
        ----------
        output : str | Path
            Path to the regions .csv file.
        reuse_images : bool
            Whether to reuse the existing images dataset in cache.
        """

        output = Path(output)
        manifest = _prepare_labelme_dataset(self.cache, reuse_images, **builder_params)
        _load_annotations(output, self.cache, manifest)
        _run_labelme(self.cache)
        _parse_to_csv(self.cache, manifest, output)


def _prepare_labelme_dataset(
    cache_cfg: CachePathsConfig,
    reuse_images: bool,
    **builder_params,
) -> ImagesDatasetManifest:

    if reuse_images:
        try:
            manifest = ImagesDatasetManifest.load(cache_cfg.img_dataset)
        except Exception as e:
            print(f"Failed to load manifest. Rebuilding images dataset.\n{e}")
            # Clear cache_cfg directory
            shutil.rmtree(cache_cfg.root)
            cache_cfg.mkdir()
            # Build new dataset
            manifest = build_images_dataset(**builder_params)
    else:
        # Clear cache_cfg directory
        shutil.rmtree(cache_cfg.root)
        cache_cfg.mkdir()
        # Build new dataset
        manifest = build_images_dataset(**builder_params)

    return manifest


def _load_annotations(file: Path, cache_cfg: CachePathsConfig, manifest: ImagesDatasetManifest):
    # If the library already contains annotation: write to Labelme format
    if file.is_file():
        if file.suffix == ".evr":
            regions: Regions2D = er.read_evr(file)
        elif file.suffix == ".csv":
            regions: Regions2D = er.read_regions_csv(file)
        else:
            raise ValueError(f"Invalid file format for library. Expected one of ['.evr', '.csv'], got '{file.suffix}'")

        labelme_data = echolabel_writer.regions2d_to_labelme(regions, manifest, cache_cfg.img_dataset)

        for filename, data in labelme_data.items():
            with open(cache_cfg.labelme / filename, "w") as f:
                json.dump(data, f, indent=4)


def _run_labelme(cache_cfg: CachePathsConfig):
    # Run Labelme as subprocess
    with open(cache_cfg.labelme_logs, "w") as log:
        subprocess.run(
            [
                "labelme",
                str(cache_cfg.img_dataset),
                "--output",
                str(cache_cfg.labelme),
            ],
            stdout=log,
            stderr=log,
        )


def _parse_to_csv(cache_cfg: CachePathsConfig, manifest: ImagesDatasetManifest, outfile: Path):

    # Parse output to Dataframe
    data = regions2d_parser.parse_echolabel(cache_cfg, manifest)

    # Save to library file
    library_dir = outfile.parent
    library_dir.mkdir(parents=True, exist_ok=True)

    if outfile.is_file():
        update_safe_name = outfile.stem + "_prev" + outfile.suffix
        os.rename(outfile, library_dir / update_safe_name)  # Safety guard: rename the existing file before overwriting

    data.to_csv(outfile)
