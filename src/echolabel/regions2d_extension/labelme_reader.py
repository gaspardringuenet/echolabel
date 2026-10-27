import glob
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
from echoregions.utils.io import check_file
from shapely import unary_union

from ..config.cache import CachePathsConfig
from ..core.manifest import ImagesDatasetManifest
from .shapes import (
    SHAPE_RESERVED_KEYS,
    ImageGrid,
    flags_to_dict,
    is_box,
    is_missing,
    is_rectangle_row,
    label_of,
    load_image_grids,
    pixels_to_geometry,
    polygon_parts,
    region_to_shapes,
    row_geometry,
    seconds_to_time,
)

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
    "flags",
    "group_id",
    "description",
]


def parse_echolabel(
    cache_cfg: CachePathsConfig,
    manifest: ImagesDatasetManifest | None = None,
    original_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Parse all LabelMe JSON files of a session back into a regions DataFrame.

    If `original_df` (the regions CSV read at the start of the session) is
    given, region identity is preserved:
    -   A region whose shapes are unchanged is written back as its original row.
    -   An edited region gets its rows rebuilt from the current shapes. The part
        outside all images (never shown in LabelMe) is kept.
    -   Shapes without a known `region_id` become new regions.
    """
    if manifest is None:
        manifest = ImagesDatasetManifest.load(cache_cfg.img_dataset)

    grids = load_image_grids(manifest, cache_cfg.img_dataset)

    shapes_by_id: dict = {}
    for labelme_file in sorted(glob.glob(str(cache_cfg.labelme / "*.json"))):
        json_data = _load_json(labelme_file)
        for entry in _shape_entries(json_data, grids):
            shapes_by_id.setdefault(entry["region_id"], []).append(entry)

    columns = list(original_df.columns if original_df is not None else CUSTOM_COLUMNS)
    if "shape_type" not in columns:
        columns.insert(columns.index("depth") + 1, "shape_type")
    # Always exported, filled from the LabelMe JSON even if the source CSV lacks them
    for col in ("flags", "group_id", "description", "echolabel_version"):
        if col not in columns:
            columns.append(col)
    next_id = _next_free_id(original_df, shapes_by_id)
    rows: list[dict] = []

    if original_df is not None:
        for _, orig in original_df.iterrows():
            rid = orig["region_id"]
            current = shapes_by_id.pop(rid, [])
            region_rows, next_id = _reconcile_region(orig.to_dict(), current, grids, next_id)
            rows.extend(region_rows)

    # Shapes not linked to an original region (new shapes, or no region_id)
    for entries in shapes_by_id.values():
        group_rows, next_id = _group_rows(entries, None, entries[0]["region_id"], next_id)
        rows.extend(group_rows)

    data = pd.DataFrame(rows)
    for col in columns:
        if col not in data.columns:
            data[col] = np.nan
    extra = [c for c in data.columns if c not in columns]
    data = data.loc[:, columns + extra]

    # Rows from original CSV without a shape_type value are polygons
    data["shape_type"] = data["shape_type"].where(data["shape_type"].notna(), "polygon")

    try:
        data["echolabel_version"] = version("echolabel")
    except PackageNotFoundError:
        data["echolabel_version"] = "unknown"

    # Ensure correct dtypes for columns that may contain only NaN values
    if "description" in data.columns:
        data["description"] = data["description"].astype("object")

    return data


def _next_free_id(original_df: pd.DataFrame | None, shapes_by_id: dict) -> int:
    ids = []
    if original_df is not None and "region_id" in original_df.columns:
        ids += [int(i) for i in original_df["region_id"].dropna()]
    ids += [int(i) for i in shapes_by_id if i is not None and not is_missing(i)]
    return max(ids, default=0) + 1


def _shape_entries(data: dict, grids: list[ImageGrid]) -> list[dict]:
    """Shapes of one LabelMe JSON file, converted to real-world geometries"""
    img_filename = Path(data["imagePath"]).name
    grid = next(g for g in grids if g.filename == img_filename)

    entries = []
    for shape in data["shapes"]:
        if shape["shape_type"] not in ("polygon", "rectangle"):
            continue
        rid = shape.get("region_id")
        entries.append(
            {
                "region_id": None if is_missing(rid) else rid,
                "label": shape.get("label", ""),
                "points": shape["points"],
                "shape_type": shape["shape_type"],
                "geometry": pixels_to_geometry(shape["points"], shape["shape_type"], grid),
                "filename": img_filename,
                "group_id": shape.get("group_id"),
                "description": shape.get("description") or "",
                "flags": [k for k, v in (shape.get("flags") or {}).items() if v],
                "extras": {k: v for k, v in shape.items() if k not in SHAPE_RESERVED_KEYS},
            }
        )
    return entries


def _expected_key(filename: str, points: list, label: str, group_id, description: str, flags: list) -> tuple:
    return (
        filename,
        tuple(tuple(p) for p in points),
        label,
        None if group_id is None else int(group_id),
        description,
        tuple(sorted(flags)),
    )


def _reconcile_region(orig: dict, current: list[dict], grids: list[ImageGrid], next_id: int) -> tuple[list[dict], int]:
    """Rows for one original region, given its current LabelMe shapes."""
    try:
        geom = row_geometry(orig["time"], orig["depth"])
    except (ValueError, TypeError):
        # Region cannot be represented: keep the original row untouched
        return [orig], next_id

    label = label_of(orig)
    group_id = None if is_missing(orig.get("group_id")) else int(orig["group_id"])
    description = "" if is_missing(orig.get("description")) else str(orig["description"])
    flags = list(flags_to_dict(orig.get("flags")).keys())
    shape_type = "rectangle" if is_rectangle_row(orig.get("shape_type"), orig["time"], orig["depth"]) else "polygon"

    expected = {
        _expected_key(s.filename, s.points, label, group_id, description, flags)
        for s in region_to_shapes(geom, grids, shape_type)
    }
    actual = {
        _expected_key(e["filename"], e["points"], e["label"], e["group_id"], e["description"], e["flags"])
        for e in current
    }
    if expected == actual:
        return [orig], next_id

    # All visible shapes deleted: the region is deleted, outside part included
    outside = geom.difference(unary_union([g.box for g in grids]))
    if not current:
        _warn_outside_dropped(orig["region_id"], polygon_parts(outside))
        return [], next_id

    # Edited region: keep the part outside any image, add the shapes drawn by the user.
    # If a visible shape was deleted, the outside part is dropped too (deletion stays simple to follow).
    entries = list(current)
    if expected - actual:
        _warn_outside_dropped(orig["region_id"], polygon_parts(outside))
    elif polygon_parts(outside):
        entries.append(
            {
                "label": label,
                "geometry": outside,
                "clipped": True,
                "group_id": group_id,
                "description": description,
                "flags": flags,
                "extras": {},
            }
        )
    return _group_rows(entries, orig, orig["region_id"], next_id)


def _warn_outside_dropped(region_id, outside_parts: list) -> None:
    if outside_parts:
        print(
            f"Warning: region {region_id} was edited by deleting visible shapes; "
            "its part outside the images (CSV-only) is deleted too."
        )


def _group_rows(entries: list[dict], orig: dict | None, rid, next_id: int) -> tuple[list[dict], int]:
    """
    Build rows from shape entries of one region: shapes are unioned per label,
    each polygon part becomes a row. A part is a rectangle only when all its
    shapes are rectangles and the union is still an axis-aligned box.
    The first row reuses `rid` (if not None).
    """
    groups: dict[str, dict] = {}
    for e in entries:
        group = groups.setdefault(e["label"], {"geoms": [], "meta": e, "rectangles": True})
        group["geoms"].append(e["geometry"])
        if not e.get("clipped"):
            group["rectangles"] &= e.get("shape_type") == "rectangle"

    rows = []
    for label, group in groups.items():
        meta = dict(group["meta"], shape_type="rectangle" if group["rectangles"] else "polygon")
        for part in polygon_parts(unary_union(group["geoms"])):
            if rid is not None and not rows:
                part_id = rid
            else:
                part_id = next_id
                next_id += 1
            rows.append(_build_row(part, label, part_id, meta, orig))
    return rows, next_id


def _build_row(geom, label: str, region_id, meta: dict, orig: dict | None) -> dict:
    """
    One output row for a region part. Unmapped columns (including region_class)
    come unchanged from the original row; region_bbox_* are recomputed from the geometry.
    Rectangles are exported as their four corners.
    """
    row = dict(orig) if orig is not None else dict(meta.get("extras", {}))

    minx, miny, maxx, maxy = geom.bounds
    is_rect = meta.get("shape_type") == "rectangle" and is_box(geom)
    if is_rect:
        coords = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
    else:
        coords = list(geom.exterior.coords)[:-1]
    time = seconds_to_time([c[0] for c in coords])
    depth = np.array([c[1] for c in coords], dtype=float)

    group_id = meta.get("group_id")
    description = meta.get("description") or ""
    row.update(
        {
            "region_id": region_id,
            "region_name": label,
            "time": time,
            "depth": depth,
            "shape_type": "rectangle" if is_rect else "polygon",
            "region_bbox_left": seconds_to_time(minx),
            "region_bbox_right": seconds_to_time(maxx),
            "region_bbox_top": miny,
            "region_bbox_bottom": maxy,
            "flags": list(meta.get("flags", [])),
            "group_id": np.nan if group_id is None else int(group_id),
            "description": description if description else np.nan,
        }
    )
    return row


def parse_labelme(
    data: dict,
    manifest: ImagesDatasetManifest,
    cache_dir: str | Path,
) -> pd.DataFrame:
    """
    Parse one LabelMe JSON file into one row per shape (no reconciliation with
    an original CSV). Shapes keep their `region_id` when present.
    """
    grids = load_image_grids(manifest, Path(cache_dir))
    rows = [_build_row(e["geometry"], e["label"], e["region_id"], e, None) for e in _shape_entries(data, grids)]
    return pd.DataFrame(rows)


def _load_json(input_file: str) -> dict:
    """Load JSON file. Check file validity using Echoregions' IO utility."""

    # Check for validity of input file
    check_file(input_file, "JSON")

    # Read files
    with open(input_file) as f:
        data: dict = json.load(f)

    return data
