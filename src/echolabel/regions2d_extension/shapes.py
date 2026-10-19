"""
Shared geometry helpers between the CSV -> LabelMe writer and the LabelMe -> CSV reader.

Real-world geometries are shapely objects in (x, y) = (time in seconds since
epoch, depth in metres). Both directions use the same image grids, so a region
exported without edits maps back to exactly the same pixel points.
"""

import datetime
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from shapely import box, get_parts, make_valid
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from ..core.manifest import ImageMetadata, ImagesDatasetManifest

# Columns of the regions CSV that are not passed on to LabelMe shapes.
# Mapped: region_name -> label, time/depth -> points, shape_type -> LabelMe shape type,
# flags/group_id/description -> shape fields.
# Derived on export: region_bbox_* (recomputed from the clipped geometry), echolabel_version.
MAPPED_COLUMNS = {
    "region_id",
    "region_name",
    "time",
    "depth",
    "shape_type",
    "region_bbox_left",
    "region_bbox_right",
    "region_bbox_top",
    "region_bbox_bottom",
    "echolabel_version",
    "group_id",
    "description",
    "flags",
}

SHAPE_RESERVED_KEYS = {"label", "points", "shape_type", "mask", "flags", "group_id", "description", "region_id"}

NONAME_LABEL = ".NONAME"


@dataclass(frozen=True)
class ImageGrid:
    """Real-world extent and pixel grid of one image in the images dataset."""

    meta: ImageMetadata
    times: npt.NDArray[np.floating]  # seconds since epoch, one per ping column
    depths: npt.NDArray[np.floating]  # metres, one per range row

    @property
    def filename(self) -> str:
        return self.meta.filename

    @property
    def box(self) -> Polygon:
        """Real-world extent of the image"""
        return box(self.times[0], self.depths[0], self.times[-1], self.depths[-1])


@dataclass(frozen=True)
class ShapeSpec:
    """A shape drawn in one image, in pixels, with its real-world geometry."""

    filename: str
    points: list[list[float]]
    geometry: BaseGeometry
    shape_type: str = "polygon"


def load_image_grids(manifest: ImagesDatasetManifest, cache_dir: Path) -> list[ImageGrid]:
    """Load the real-world grid of every image in the manifest."""
    grids = []
    for img_meta in manifest.images:
        time_coord, depth_coord = img_meta.load_coordinates(cache_dir)
        grids.append(
            ImageGrid(
                meta=img_meta,
                times=time_to_seconds(time_coord),
                depths=np.asarray(depth_coord, dtype=float),
            )
        )
    return grids


def find_grid(grids: list[ImageGrid], filename: str) -> ImageGrid:
    for grid in grids:
        if grid.filename == filename:
            return grid
    raise ValueError(f"Image {filename} not found in manifest")


def time_to_seconds(values) -> npt.NDArray[np.floating]:
    """datetime-like values (datetime64, strings or objects) -> float seconds since epoch"""
    arr = np.asarray(values, dtype="datetime64[ns]")
    return arr.astype(np.int64) / 1e9


def seconds_to_time(values) -> npt.NDArray[np.datetime64]:
    """Float seconds since epoch -> datetime64[ns]"""
    return np.round(np.asarray(values, dtype=float) * 1e9).astype(np.int64).astype("datetime64[ns]")


def is_missing(value) -> bool:
    """True for None / NaN / NaT scalars. Arrays and lists are never missing."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def label_of(row: pd.Series | dict) -> str:
    """LabelMe label from the region_name column, with a fallback when absent."""
    value = row.get("region_name")
    if is_missing(value) or str(value) == "":
        return NONAME_LABEL
    return str(value)


def row_geometry(time_values, depth_values) -> BaseGeometry:
    """Real-world geometry of a region row (valid polygon, in seconds x metres)."""
    times = time_to_seconds(time_values)
    depths = np.asarray(depth_values, dtype=float)
    if len(times) < 3 or len(depths) != len(times):
        raise ValueError(f"Expected at least 3 matching time/depth points, got {len(times)}/{len(depths)}")
    return make_valid(Polygon(zip(times, depths)))


def polygon_parts(geom: BaseGeometry) -> list[Polygon]:
    """Non-empty polygonal parts of a geometry (drops points and lines from clipping)."""
    return [p for p in get_parts(geom) if isinstance(p, Polygon) and p.area > 0]


def is_rectangle_row(shape_type, time_values, depth_values) -> bool:
    """
    True when a CSV row is an axis-aligned rectangle: shape_type == "rectangle"
    and exactly the four corners of a box (4 points, 2 distinct times, 2 distinct depths).
    """
    if is_missing(shape_type) or str(shape_type) != "rectangle":
        return False
    times = time_to_seconds(time_values)
    depths = np.asarray(depth_values, dtype=float)
    if len(times) != 4 or len(depths) != 4:
        return False
    corners = set(zip(times.tolist(), depths.tolist()))
    return len(set(times.tolist())) == 2 and len(set(depths.tolist())) == 2 and len(corners) == 4


def is_box(geom: BaseGeometry) -> bool:
    """True when a polygon is an axis-aligned rectangle (up to float tolerance)."""
    if not isinstance(geom, Polygon) or geom.area <= 0:
        return False
    return abs(geom.area - box(*geom.bounds).area) <= 1e-9 * geom.area


def region_to_shapes(geom: BaseGeometry, grids: list[ImageGrid], shape_type: str = "polygon") -> list[ShapeSpec]:
    """
    Split a real-world region across images and clip it to each image extent.

    A region overlapping several images gives one shape per image, of type
    `shape_type` ("polygon" or "rectangle"). The part beyond an image's depth
    range is clipped away (kept only in the CSV). Clipping a rectangle by an
    image box gives a rectangle, so rectangles stay rectangles.
    """
    specs = []
    for grid in grids:
        clipped = geom.intersection(grid.box)
        for part in polygon_parts(clipped):
            specs.append(
                ShapeSpec(
                    filename=grid.filename,
                    points=_geometry_to_pixels(part, grid, shape_type),
                    geometry=part,
                    shape_type=shape_type,
                )
            )
    return specs


def _geometry_to_pixels(part: Polygon, grid: ImageGrid, shape_type: str = "polygon") -> list[list[float]]:
    corners = list(part.exterior.coords)[:-1]
    # Fractional pixel positions (LabelMe accepts floats). Linear interpolation is a
    # monotonic map on each axis, so the polygon keeps its topology (no self-intersections).
    xs = np.interp([c[0] for c in corners], grid.times, np.arange(len(grid.times)))
    ys = np.interp([c[1] for c in corners], grid.depths, np.arange(len(grid.depths)))
    if shape_type == "rectangle":
        # LabelMe rectangles are stored as two opposite corners
        return [[float(xs.min()), float(ys.min())], [float(xs.max()), float(ys.max())]]
    return [[float(x), float(y)] for x, y in zip(xs, ys)]


def pixels_to_geometry(points: list[list[float]], shape_type: str, grid: ImageGrid) -> BaseGeometry:
    """LabelMe shape points (fractional pixels) -> real-world geometry, inverse of _geometry_to_pixels."""
    arr = np.asarray(points, dtype=float).reshape(-1, 2)
    times = np.interp(arr[:, 0], np.arange(len(grid.times)), grid.times)
    depths = np.interp(arr[:, 1], np.arange(len(grid.depths)), grid.depths)
    if shape_type == "rectangle":
        return box(times.min(), depths.min(), times.max(), depths.max())
    return make_valid(Polygon(zip(times, depths)))


def csv_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Serialize array cells without numpy's summarization ('...') so Echoregions can re-read them"""
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(lambda v: np.array2string(v, threshold=sys.maxsize) if isinstance(v, np.ndarray) else v)
    return out


def json_safe(value):
    """Convert a pandas / numpy value into something json.dump accepts."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp, np.datetime64, datetime.datetime)):
        return str(pd.Timestamp(value).isoformat())
    if isinstance(value, np.generic):
        value = value.item()
    if is_missing(value):
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def flags_to_dict(value) -> dict[str, bool]:
    """Echoregions flags (list, '[]' string, or NaN) -> LabelMe flags dict."""
    if is_missing(value):
        return {}
    if isinstance(value, str):
        value = value.strip().strip("[]")
        names = [v.strip().strip("'\"") for v in value.split(",") if v.strip()]
    else:
        names = [str(v) for v in value]
    return {name: True for name in names}


def flags_to_list(flags: dict) -> list[str]:
    return [k for k, v in (flags or {}).items() if v]
