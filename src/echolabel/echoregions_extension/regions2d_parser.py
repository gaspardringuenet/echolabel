import glob
import json
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from echoregions.utils.io import check_file

from ..config.cache import CachePathsConfig
from ..core.manifest import ImagesDatasetManifest

CUSTOM_COLUMNS = [
    "region_id",
    "region_name",
    "region_class",
    "time",
    "depth",
    "region_bbox_left",
    "region_bbox_right",
    "region_bbox_top",
    "region_bbox_bottom",
    "echoview_version",
    "region_structure_version",
    "region_creation_type",
    "region_type",
    "region_notes",
    "flags",
    "group_id",
    "description",
]


def parse_echolabel(
    cache_cfg: CachePathsConfig,
    manifest: ImagesDatasetManifest | None = None,
) -> pd.DataFrame:
    """Parse all the Labelme JSON file produced during an session"""

    if manifest is None:
        manifest = ImagesDatasetManifest.load(cache_cfg.img_dataset)

    data_list = []

    # Parse all JSON files in the labelme output subfolder
    for labelme_file in glob.glob(str(cache_cfg.labelme / "*.json")):
        data_list.append(
            parse_labelme(
                input_file=labelme_file,
                cache_dir=cache_cfg.img_dataset,
                manifest=manifest,
            )
        )

    # Return default if empty
    if len(data_list) == 0:
        data = pd.DataFrame(columns=CUSTOM_COLUMNS)
        return data

    # Concatenate (silence FutureWarning)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        data = pd.concat(data_list, ignore_index=True)

    # region_id must be unique (now it start with 1 for each json file): order by time of start and count
    data = data.sort_values(by="region_bbox_left")

    # Reassign region_id to be globally unique
    data = data.reset_index(drop=True)
    data["region_id"] = range(1, len(data) + 1)

    return data


def parse_labelme(
    input_file: str,
    cache_dir: str,
    manifest: ImagesDatasetManifest,
) -> pd.DataFrame:
    """Parse a Labelme JSON file. Points are related to a
    subset of the acoustic data printed as an echogram image. The data manifest file
    provides information on the images' metadata (especially time and depth values)._
    """

    creation_types_conversion_dict: dict = {"polygon": "2", "rectangle": "3"}

    # Check for validity of input file
    check_file(input_file, "JSON")

    # Read files
    with open(input_file) as f:
        labelme_annotation: dict = json.load(f)

    # Find image file in annotation
    img_filename = Path(labelme_annotation["imagePath"]).name

    rows = []

    for idx, shape in enumerate(labelme_annotation["shapes"]):
        # Assume the shape is a polygon (would need reshaping if not)
        if shape["shape_type"] not in ["polygon", "rectangle"]:
            continue

        # Convert to time x depth coordinates
        time, depth = manifest.labelme_polygon_to_real_coords(
            Path(cache_dir), img_filename, shape["points"]
        )

        # Calculate bounding box
        left = np.min(time)
        right = np.max(time)
        top = np.min(depth)
        bottom = np.max(depth)

        if shape["shape_type"] == "rectangle":
            time, depth = _format_rectangle(left, right, top, bottom)

        # Create row
        row = {
            # Minimal requirements for Regions2D methods (except .to_evr)
            "region_id": shape.get("id", idx),  # Must be an int
            "region_name": shape.get("label", ""),
            "region_class": shape.get("label", ""),  # Using label as class
            "time": time,
            "depth": depth,
            "region_bbox_left": left,
            "region_bbox_right": right,
            "region_bbox_top": top,
            "region_bbox_bottom": bottom,
            # EVR compatibility
            "echoview_version": "13.0.378.44817",  # from EchoRegions' doc - https://echoregions.readthedocs.io/en/latest/Regions2D_functionality.html
            "region_structure_version": "13",
            "region_creation_type": creation_types_conversion_dict.get(
                shape.get("shape_type"), "-1"
            ),  # noqa "Polygon tool" (3 for rectangle)
            "region_type": "1",  # "analysis"
            "region_notes": [],
            # Additional attributes
            "flags": [k for (k, v) in shape.get("flags", {}).items() if v],
            "group_id": shape.get("group_id", None),
            "description": shape.get("description", ""),
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    return df


def _format_rectangle(
    left: np.datetime64, right: np.datetime64, top: float, bottom: float
) -> Tuple[np.ndarray[np.datetime64], np.ndarray[float]]:
    """
    Format rectangle to be EVR compatible: a list of points in order:
        1    4
        2 -> 3
    """
    time = np.array([left, left, right, right])
    depth = np.array([top, bottom, bottom, top])

    return time, depth
