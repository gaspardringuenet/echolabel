from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from echoregions.regions2d import Regions2D

from ..core.manifest import ImagesDatasetManifest


def _rectangle_to_labelme(
    times: npt.NDArray[np.datetime64],
    depths: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.datetime64], npt.NDArray[np.floating]]:
    """
    Select 2 diagonaly opposed points in a 4 points rectangle polygon.

    The result corresponds to the LabelMe representation of a rectangle.
    (Still requires converting to a list of [i, j] pixel coordinates,
    done using manifest method.)
    """
    return (
        np.array([np.min(times), np.max(times)]),
        np.array([np.min(depths), np.max(depths)]),
    )


shape_tranforms_registry: dict[str, Callable] = {
    "polygon": (lambda *x: x),
    "rectangle": _rectangle_to_labelme,
}


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

    df: pd.DataFrame = pd.DataFrame(regions.data.copy())

    # Convert bbox time columns from string to datetime64 (should be done by Echoregions ideally)
    for bbox_col in ["region_bbox_left", "region_bbox_right"]:
        if bbox_col in df.columns:
            df[bbox_col] = pd.to_datetime(df[bbox_col], errors="coerce")

    for img_meta in manifest.images:
        time_coord, depth_coord = img_meta.load_coordinates(img_dataset_dir)

        # Grab all regions contained within the image's bbox
        contained_mask: pd.Series = (
            (time_coord[0] <= df["region_bbox_left"])
            & (df["region_bbox_right"] <= time_coord[-1])
            & (depth_coord[0] <= df["region_bbox_top"])
            & (df["region_bbox_bottom"] <= depth_coord[-1])
        )
        contained = df.loc[contained_mask]

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

                if not isinstance(times, np.ndarray) or not isinstance(depths, np.ndarray):
                    raise TypeError(f"Expected time/depth arrays, got {type(times)}/{type(depths)}")

                region_creation_type = row.get("region_creation_type", None)

                if region_creation_type is not None and int(region_creation_type) == 3:
                    shape_type = "rectangle"
                else:
                    shape_type = "polygon"

                trans_fn = shape_tranforms_registry[shape_type]
                times, depths = trans_fn(times, depths)

                points = manifest.real_coords_to_labelme_polygon(
                    cache_dir=img_dataset_dir,
                    filename=img_meta.filename,
                    points_ping_axis_values=times,
                    points_range_axis_values=depths,
                )

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

            except (ValueError, TypeError) as e:
                print(f"Failed to parse region {row['region_id']}: {e}")
                failed_ids.append(row["region_id"])
                continue

    total = len(df)
    print(f"Parsed {len(parsed_ids)}/{total} regions into Labelme shapes.")
    if failed_ids:
        print(f"Failed: {failed_ids}")

    return labelme_data
