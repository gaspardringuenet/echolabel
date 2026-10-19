from pathlib import Path

import pandas as pd
from echoregions.regions2d import Regions2D

from ..core.manifest import ImagesDatasetManifest
from .shapes import (
    MAPPED_COLUMNS,
    flags_to_dict,
    is_missing,
    is_rectangle_row,
    json_safe,
    label_of,
    load_image_grids,
    region_to_shapes,
    row_geometry,
)


def regions2d_to_labelme(
    regions: Regions2D | pd.DataFrame,
    manifest: ImagesDatasetManifest,
    img_dataset_dir: Path,
    labelme_version="6.0.0",
) -> dict:
    """
    Writes the regions of a Regions2D object into LabelMe JSON annotations.

    Each region is clipped to every image it overlaps in depth and time. The
    part of a region outside an image is not written as a shape: it stays in
    the CSV only. A region spanning several images gives one shape per image,
    all sharing the same `region_id`.

    Non-mapped CSV columns are passed on as extra keys of each shape.
    Regions with no overlap with any image, or with invalid geometry, are
    skipped with a printed reason.

    Returns a dict mapping JSON filename to the LabelMe annotation dict.
    """
    labelme_data: dict = {}
    grids = load_image_grids(manifest, img_dataset_dir)

    for grid in grids:
        json_filename = Path(grid.filename).stem + ".json"
        labelme_data[json_filename] = {
            "version": labelme_version,
            "flags": {},
            "shapes": [],
            "imagePath": str(img_dataset_dir / grid.filename),
            "imageData": None,
            "imageHeight": grid.meta.range_sample_end - grid.meta.range_sample_start + 1,
            "imageWidth": grid.meta.ping_sample_end - grid.meta.ping_sample_start + 1,
        }

    df: pd.DataFrame = regions if isinstance(regions, pd.DataFrame) else pd.DataFrame(regions.data.copy())
    passthrough_cols = [c for c in df.columns if c not in MAPPED_COLUMNS]
    parsed_ids = []
    failed_ids = []

    for _, row in df.iterrows():
        region_id = row["region_id"]
        try:
            geom = row_geometry(row["time"], row["depth"])
        except (ValueError, TypeError) as e:
            print(f"Warning: region {region_id} skipped, invalid geometry ({e})")
            failed_ids.append(region_id)
            continue

        shape_type = "polygon"
        if is_rectangle_row(row.get("shape_type"), row["time"], row["depth"]):
            shape_type = "rectangle"
        elif str(row.get("shape_type")) == "rectangle":
            print(
                f"Warning: region {region_id} has shape_type 'rectangle' but is not an axis-aligned box, imported as polygon"
            )
        specs = region_to_shapes(geom, grids, shape_type)
        if not specs:
            print(f"Warning: region {region_id} skipped, no overlap with any image (kept in CSV only)")
            failed_ids.append(region_id)
            continue

        for spec in specs:
            shape = {
                "label": label_of(row),
                "points": spec.points,
                "group_id": None if is_missing(row.get("group_id")) else int(row["group_id"]),
                "description": "" if is_missing(row.get("description")) else str(row["description"]),
                "shape_type": spec.shape_type,
                "mask": None,
                "flags": flags_to_dict(row.get("flags")),
                "region_id": json_safe(region_id),
            }
            for col in passthrough_cols:
                shape[col] = json_safe(row[col])
            labelme_data[Path(spec.filename).stem + ".json"]["shapes"].append(shape)

        parsed_ids.append(region_id)

    # Only keep image annotations that contain shapes
    labelme_data = {k: v for k, v in labelme_data.items() if v["shapes"]}

    print(f"Parsed {len(parsed_ids)}/{len(df)} regions into Labelme shapes.")
    if failed_ids:
        print(f"Failed: {failed_ids}")

    return labelme_data
