import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
import numpy.typing as npt


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

    def load_coordinates(self, output_dir: Path) -> Tuple[npt.NDArray, npt.NDArray]:
        """Load coordinate arrays"""
        stem = Path(self.filename).stem
        data = np.load(output_dir / f"{stem}_coords.npz")
        return data["ping_axis"], data["range_axis"]

    def pixel_to_real_coords(
        self,
        output_dir: Path,
        x_pixel: int | List[int] | npt.NDArray[np.integer],
        y_pixel: int | List[int] | npt.NDArray[np.integer],
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Convert pixel coordinates to real-world (ping_axis, range_axis) coordinates"""

        # Load real coordinates arrays
        ping_axis_values, range_axis_values = self.load_coordinates(output_dir)
        imax_ping, imax_range = len(ping_axis_values) - 1, len(range_axis_values) - 1

        # Enforce numpy array type
        x_pixel = np.array(x_pixel)
        y_pixel = np.array(y_pixel)

        # Round to int & change dtype
        x_pixel = np.floor(x_pixel).astype(np.uint32)
        y_pixel = np.floor(y_pixel).astype(np.uint32)

        # Enforce image boundaries (LabelMe sometimes allow x = img.shape[1] for instance)
        x_pixel = np.clip(x_pixel, 0, imax_ping)
        y_pixel = np.clip(y_pixel, 0, imax_range)

        return ping_axis_values[x_pixel], range_axis_values[y_pixel]


@dataclass
class ImagesDatasetManifest:
    """Manifest for an images dataset"""

    source: str | List[str]
    created_at: str
    shape: dict[str, int]
    channels: List[float] | Literal["first"]
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

    def get_image_metadata(self, filename: str) -> ImageMetadata:
        """Get metadata for a specific image by filename"""
        for img_meta in self.images:
            if img_meta.filename == filename:
                return img_meta
        raise FileNotFoundError(f"No image metada corresponds to filename: {filename}")

    def pixel_to_real_coords(
        self,
        cache_dir: Path,
        filename: str,
        x_pixel: int | List[int] | npt.NDArray[np.integer],
        y_pixel: int | List[int] | npt.NDArray[np.integer],
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Convert pixel coordinates to real-worlds coordinates for a given image
        """
        img_meta = self.get_image_metadata(filename)
        if img_meta is None:
            raise ValueError(f"Image {filename} not found in manifest")

        return img_meta.pixel_to_real_coords(cache_dir, x_pixel, y_pixel)

    def labelme_polygon_to_real_coords(
        self,
        cache_dir: Path,
        filename: str,
        points: List[int],
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Convert a Labelme polygon annotation to real-world coordinates
        """
        points_array: npt.NDArray[np.integer] = np.array(points)

        return self.pixel_to_real_coords(
            cache_dir,
            filename,
            x_pixel=points_array[:, 0],
            y_pixel=points_array[:, 1],
        )

    def real_coords_to_labelme_polygon(
        self,
        cache_dir: Path,
        filename: str,
        points_ping_axis_values: List[float | np.datetime64],
        points_range_axis_values: List[float],
    ) -> List[List[int]]:
        """
        Convert real-world coordinates to a Labelme polygon
        """

        img_meta = self.get_image_metadata(filename)
        ping_axis_values, range_axis_values = img_meta.load_coordinates(cache_dir)

        xs = np.where(points_ping_axis_values == ping_axis_values)
        ys = np.where(points_range_axis_values == range_axis_values)

        points = [[int(x), int(y)] for (x, y) in zip(xs, ys)]

        return points
