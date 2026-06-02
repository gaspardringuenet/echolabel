import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import xarray as xr


@dataclass
class ImageMetadata:
    """Metadata for a single echogram image"""

    filename: str
    ping_sample_start: int
    ping_sample_end: int
    range_sample_start: int
    range_sample_end: int

    def save_coordinates(
        self,
        output_dir: Path,
        ping_axis_values: np.ndarray,
        range_axis_values: np.ndarray,
    ):
        """Save coordinate arrays as compressed numpy fimes"""
        stem = Path(self.filename).stem
        np.savez_compressed(
            output_dir / f"{stem}_coords.npz",
            ping_axis=ping_axis_values,
            range_axis=range_axis_values,
        )

    def load_coordinates(self, output_dir: Path):
        """Load coordinate arrays"""
        stem = Path(self.filename).stem
        data = np.load(output_dir / f"{stem}_coords.npz")
        return data["ping_axis"], data["range_axis"]

    def pixel_to_real_coords(
        self,
        output_dir: Path,
        x_pixel: int | List[int] | np.ndarray[int],
        y_pixel: int | List[int] | np.ndarray[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert pixel coordinates to real-world (ping_axis, range_axis) coordinates"""

        # Load real coordinates arrays
        ping_axis_values, range_axis_values = self.load_coordinates(output_dir)

        # Enforce numpy array type
        x_pixel = np.array(x_pixel)
        y_pixel = np.array(y_pixel)

        # Round to int & change dtype
        x_pixel = np.rint(x_pixel).astype(np.uint32)
        y_pixel = np.rint(y_pixel).astype(np.uint32)

        # Validate pixel coordinates
        if not np.all((0 <= x_pixel) & (x_pixel < len(ping_axis_values))):
            raise ValueError(f"x_pixel {x_pixel} out of bounds [0, {len(ping_axis_values)}")
        if not np.all((0 <= y_pixel) & (y_pixel < len(range_axis_values))):
            raise ValueError(f"y_pixel {y_pixel} ouf of bounds [0, {len(range_axis_values)}")

        return ping_axis_values[x_pixel], range_axis_values[y_pixel]

    # def get_real_bbox(self, output_dir: Path):
    #     ping_axis_values, range_axis_values = self.load_coordinates(output_dir)
    #     return ping_axis_values[0], ping_axis_values[-1], range_axis_values[0], range_axis_values[-1]


@dataclass
class ImagesDatasetManifest:
    """Manifest for an images dataset"""

    source: str | List[str]
    created_at: str
    sv_shape: List[int]
    channels: float | List[float]
    viz_params: dict
    images: List[ImageMetadata]

    def save(self, cache_dir: Path):
        """Save manifest to JSON"""
        manifest_path = cache_dir / "manifest.json"
        data = asdict(self)
        with open(manifest_path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, cache_dir: Path):
        """Load manifest from JSON"""
        manifest_path = cache_dir / "manifest.json"
        with open(manifest_path, "r") as f:
            data = json.load(f)
        data["images"] = [ImageMetadata(**img) for img in data["images"]]
        return cls(**data)

    def get_image_metadata(self, filename: str) -> Optional[ImageMetadata]:
        """Get metadata for a specific image by filename"""
        for img_meta in self.images:
            if img_meta.filename == filename:
                return img_meta
        return None

    def pixel_to_real_coords(
        self,
        cache_dir: Path,
        filename: str,
        x_pixel: int | List[int] | np.ndarray[int],
        y_pixel: int | List[int] | np.ndarray[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert pixel coordinates to real-worlds coordinates for a given image"""
        img_meta = self.get_image_metadata(filename)
        if img_meta is None:
            raise ValueError(f"Image {filename} not found in manifest")

        return img_meta.pixel_to_real_coords(cache_dir, x_pixel, y_pixel)

    def labelme_polygon_to_real_coords(
        self,
        cache_dir: Path,
        filename: str,
        points: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a Labelme polygon annotation to rCeal-world coordinates
        """
        points = np.array(points)

        return self.pixel_to_real_coords(cache_dir, filename, x_pixel=points[:, 0], y_pixel=points[:, 1])

    def real_coords_to_labelme_polygon(
        self,
        cache_dir: Path,
        filename: str,
        points_ping_axis_values: List[float | np.datetime64],
        points_range_axis_values: List[float],
    ) -> List[Tuple[int, int]]:
        """
        Convert real-world coordinates to a Labelme polygon
        """

        img_meta = self.get_image_metadata(filename)
        ping_axis_values, range_axis_values = img_meta.load_coordinates(cache_dir)

        xs = np.where(points_ping_axis_values == ping_axis_values)
        ys = np.where(points_range_axis_values == range_axis_values)

        points = [[x, y] for (x, y) in zip(xs, ys)]

        return points
