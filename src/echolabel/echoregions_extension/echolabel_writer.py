from pathlib import Path

import numpy as np
import pandas as pd
from echoregions.regions2d import Regions2D

from ..core.manifest import ImagesDatasetManifest


def regions2d_to_labelme(
    regions: Regions2D,
    manifest: ImagesDatasetManifest,
    img_dataset_dir: Path,
    labelme_version="6.0.0",
) -> dict:
    """
    Writes shapes in a Regions2D object to the Labelme JSON format

    For each image in the dataset:
    -   Find all regions contained in that image (using bbox)
    -   Convert real-world (time, depth) coordinates to pixel coordinates
    -   Write shapes into JSON file

    Warning: Handle out of bounds points (depth > image edge)
    Special case: regions that extend beyond a single image
    --> For now not handled: simply print error and skip
    """

    labelme_data = {}
    parsed_ids = []
    failed_ids = []

    df = regions.data.copy()

    # Convert bbox time columns from string to datetime64 (should be done by Echoregions ideally)
    for bbox_col in ["region_bbox_left", "region_bbox_right"]:
        if bbox_col in df.columns:
            df[bbox_col] = pd.to_datetime(df[bbox_col], errors="coerce")

    for img_meta in manifest.images:
        time_coord, depth_coord = img_meta.load_coordinates(img_dataset_dir)

        # Grab all regions contained within the image's bbox
        contained_mask = (
            (time_coord[0] <= df["region_bbox_left"])
            & (df["region_bbox_right"] <= time_coord[-1])
            & (depth_coord[0] <= df["region_bbox_top"])
            & (df["region_bbox_bottom"] <= depth_coord[-1])
        )
        contained = df[contained_mask]

        # Continue if no regions is contained in image
        if len(contained) == 0:
            continue

        # Create image annotation dict
        img_annotations = {
            "version": labelme_version,
            "flags": {},
            "shapes": [],
            "imagePath": str(img_dataset_dir / img_meta.filename),
            "imageData": None,
            "imageHeight": img_meta.range_sample_end - img_meta.range_sample_start + 1,
            "imageWidth": img_meta.ping_sample_end - img_meta.ping_sample_start + 1,
        }
        json_filename = Path(img_meta.filename).stem + ".json"
        labelme_data[json_filename] = img_annotations

        for _, row in contained.iterrows():
            try:
                # Extract coordinate arrays from row
                times = row["time"]
                depths = row["depth"]

                if not isinstance(times, (list, np.ndarray)) or not isinstance(depths, (list, np.ndarray)):
                    raise ValueError(f"Expected time/depth arrays, got {type(times)}/{type(depths)}")

                # Convert real-world coordinates to pixel indices
                x_pixels = np.searchsorted(time_coord, times)
                y_pixels = np.searchsorted(depth_coord, depths)

                # Clamp to image bounds to catch any floating point issues
                x_pixels = np.clip(x_pixels, 0, len(time_coord) - 1)
                y_pixels = np.clip(y_pixels, 0, len(depth_coord) - 1)

                # Shape type
                region_creation_type = row.get("region_creation_type", None)

                if int(region_creation_type) == 3:  # rectangle
                    shape_type = "rectangle"
                    points = [
                        [float(np.min(x_pixels)), float(np.min(y_pixels))],
                        [float(np.max(x_pixels)), float(np.max(y_pixels))],
                    ]
                else:
                    shape_type = "polygon"
                    points = [[float(x), float(y)] for x, y in zip(x_pixels, y_pixels)]

                # Build shape metadata
                shape = {
                    "label": row["region_name"],
                    "points": points,
                    "group_id": None if pd.isna(row.get("group_id")) else int(row.get("group_id")),
                    "description": "" if pd.isna(row.get("description")) else row.get("description", ""),
                    "shape_type": shape_type,
                    "mask": None,
                    "flags": {},
                }

                img_annotations["shapes"].append(shape)
                parsed_ids.append(row["region_id"])

            except Exception as e:
                print(f"Failed to parse region {row['region_id']}: {e}")
                failed_ids.append(row["region_id"])
                continue

    total = len(df)
    print(f"Parsed {len(parsed_ids)}/{total} regions into Labelme shapes.")
    if failed_ids:
        print(f"Failed: {failed_ids}")

    return labelme_data
