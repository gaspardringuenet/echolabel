import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

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

    def load_coordinates(self, output_dir: Path) -> tuple[npt.NDArray, npt.NDArray]:
        """Load coordinate arrays"""
        stem = Path(self.filename).stem
        data = np.load(output_dir / f"{stem}_coords.npz")
        return data["ping_axis"], data["range_axis"]


@dataclass
class ImagesDatasetManifest:
    """Manifest for an images dataset"""

    source: str | list[str]
    created_at: str
    shape: dict[str, int]
    freqs_hz: float | list[float] | Literal["first"]
    viz_params: dict
    images: list[ImageMetadata]

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
