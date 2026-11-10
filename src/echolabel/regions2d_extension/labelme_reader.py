import glob
import json
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
from echoregions.utils.io import check_file
from shapely import unary_union
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from ..config.cache import CachePathsConfig
from ..core.manifest import ImagesDatasetManifest
from .shapes import (
    SHAPE_RESERVED_KEYS,
    ImageGrid,
    ShapeType,
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

type RegionRow = dict[Any, Any]
type ShapeComparisonKey = tuple[str, tuple[tuple[float, ...], ...], str, int | None, str, tuple[str, ...]]


class LabelMeShapeEntry(TypedDict):
    region_id: Any
    label: str
    points: list[list[float]]
    shape_type: ShapeType
    geometry: BaseGeometry
    filename: str
    group_id: Any
    description: str
    flags: list[str]
    extras: dict[str, Any]
    clipped: NotRequired[bool]


class ClippedShapeEntry(TypedDict):
    label: str
    geometry: BaseGeometry
    group_id: Any
    description: str
    flags: list[str]
    extras: dict[str, Any]
    shape_type: NotRequired[ShapeType]
    clipped: NotRequired[Literal[True]]


type ShapeEntry = LabelMeShapeEntry | ClippedShapeEntry


class LabelGroup(TypedDict):
    geoms: list[BaseGeometry]
    meta: ShapeEntry
    rectangles: bool


CUSTOM_COLUMNS: list[str] = [
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
    -   An edited region is rebuilt from its current shapes. Its part outside
        all images is kept only if every original visible shape remains
        unchanged; deleting or modifying an original visible shape drops it.
        Deleting all visible shapes removes the region, including that part.
    -   Shapes without a known `region_id` become new regions.
    """
    if manifest is None:
        manifest = ImagesDatasetManifest.load(cache_cfg.img_dataset)

    grids = load_image_grids(manifest, cache_cfg.img_dataset)

    # Build dict with list of shape entries for each region id (incl. None)
    shapes_by_id: dict[Any, list[LabelMeShapeEntry]] = {}
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
    rows: list[RegionRow] = []

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


def _next_free_id(
    original_df: pd.DataFrame | None,
    shapes_by_id: Mapping[Any, Sequence[LabelMeShapeEntry]],
) -> int:
    ids: list[int] = []
    if original_df is not None and "region_id" in original_df.columns:
        ids += [int(i) for i in original_df["region_id"].dropna()]
    ids += [int(i) for i in shapes_by_id if i is not None and not is_missing(i)]
    return max(ids, default=0) + 1


def _shape_entries(data: dict[str, Any], grids: list[ImageGrid]) -> list[LabelMeShapeEntry]:
    """Extract polygon and rectangle shapes, converting their points to real-world geometries."""
    img_filename = Path(data["imagePath"]).name
    grid = next(g for g in grids if g.filename == img_filename)

    entries: list[LabelMeShapeEntry] = []
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


def _expected_key(
    filename: str,
    points: list[list[float]],
    label: str,
    group_id: Any,
    description: str,
    flags: Sequence[str],
) -> ShapeComparisonKey:
    """Canonical comparison key used to detect unchanged LabelMe shapes."""
    return (
        filename,
        tuple(tuple(p) for p in points),
        label,
        None if group_id is None else int(group_id),
        description,
        tuple(sorted(flags)),
    )


def _reconcile_region(
    orig: RegionRow,
    current: Sequence[LabelMeShapeEntry],
    grids: list[ImageGrid],
    next_id: int,
) -> tuple[list[RegionRow], int]:
    """Reconcile one original row with its current LabelMe shapes.

    An unrepresentable region or one with unchanged shapes is returned as its
    original row. Otherwise, rows are rebuilt from the current shapes. The
    outside-image part is preserved only when every expected visible shape is
    still present unchanged; if any is missing or changed, that part is
    dropped. If no visible shapes remain, the region is removed entirely.
    """
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

    expected: set[ShapeComparisonKey] = {
        _expected_key(s.filename, s.points, label, group_id, description, flags)
        for s in region_to_shapes(geom, grids, shape_type)
    }
    actual: set[ShapeComparisonKey] = {
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
    entries: list[ShapeEntry] = list(current)
    if expected - actual:
        _warn_outside_dropped(orig["region_id"], polygon_parts(outside))
    elif polygon_parts(outside):
        entries.append(
            ClippedShapeEntry(
                label=label,
                geometry=outside,
                clipped=True,
                group_id=group_id,
                description=description,
                flags=flags,
                extras={},
            )
        )
    return _group_rows(entries, orig, orig["region_id"], next_id)


def _warn_outside_dropped(region_id: Any, outside_parts: Sequence[Polygon]) -> None:
    if outside_parts:
        print(
            f"Warning: region {region_id} was edited by deleting visible shapes; "
            "its part outside the images (CSV-only) is deleted too."
        )


def _group_rows(
    entries: Sequence[ShapeEntry],
    orig: RegionRow | None,
    rid: Any,
    next_id: int,
) -> tuple[list[RegionRow], int]:
    """
    Build rows from shape entries of one region: shapes are unioned per label,
    each polygon part becomes a row. A part is a rectangle only when all its
    shapes are rectangles and the union is still an axis-aligned box.
    The first row reuses `rid` (if not None).
    """
    groups: dict[str, LabelGroup] = {}
    for e in entries:
        group = groups.setdefault(e["label"], {"geoms": [], "meta": e, "rectangles": True})
        group["geoms"].append(e["geometry"])
        if not e.get("clipped"):
            group["rectangles"] &= e.get("shape_type") == "rectangle"

    rows: list[RegionRow] = []
    for label, group in groups.items():
        meta: dict[str, Any] = dict(group["meta"])
        meta["shape_type"] = "rectangle" if group["rectangles"] else "polygon"
        for part in polygon_parts(unary_union(group["geoms"])):
            if rid is not None and not rows:
                part_id = rid
            else:
                part_id = next_id
                next_id += 1
            rows.append(_build_row(part, label, part_id, meta, orig))
    return rows, next_id


def _build_row(
    geom: BaseGeometry,
    label: str,
    region_id: Any,
    meta: Mapping[str, Any],
    orig: RegionRow | None,
) -> RegionRow:
    """
    One output row for a region part. Unmapped columns (including region_class)
    come unchanged from the original row; region_bbox_* are recomputed from the geometry.
    Rectangles are exported as their four corners.
    """
    if not isinstance(geom, Polygon):
        raise TypeError(f"Expected a polygon geometry, got {geom.geom_type}")

    row = dict(orig) if orig is not None else dict(meta.get("extras", {}))

    minx, miny, maxx, maxy = geom.bounds
    is_rect = meta.get("shape_type") == "rectangle" and is_box(geom)
    if is_rect:
        coords = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
    else:
        coords = list(geom.exterior.coords)[:-1]
    time = seconds_to_time([c[0] for c in coords])
    depth: npt.NDArray[np.float64] = np.array([c[1] for c in coords], dtype=float)

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
    data: dict[str, Any],
    manifest: ImagesDatasetManifest,
    cache_dir: str | Path,
) -> pd.DataFrame:
    """
    Parse one LabelMe JSON file into one row per shape (no reconciliation with
    an original CSV). Shapes keep their `region_id` when present.
    """
    grids = load_image_grids(manifest, Path(cache_dir))
    rows: list[RegionRow] = [
        _build_row(e["geometry"], e["label"], e["region_id"], e, None) for e in _shape_entries(data, grids)
    ]
    return pd.DataFrame(rows)


def _load_json(input_file: str) -> dict[str, Any]:
    """Load JSON file. Check file validity using Echoregions' IO utility."""

    # Check for validity of input file
    check_file(input_file, "JSON")

    # Read files
    with open(input_file) as f:
        data: dict[str, Any] = json.load(f)

    return data
