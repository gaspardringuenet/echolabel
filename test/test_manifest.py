import random

import numpy as np
import pytest

from echolabel.core.manifest import ImagesDatasetManifest

random.seed(a=None, version=42)


class TestImageMetadata:
    def test_save_load_coordinates(self, tmp_cache, sample_manifest):
        """Save and load coordinate arrays"""
        # Create coords
        ping_coords = np.arange(
            np.datetime64("2000-01-01 00:00:00"),
            np.datetime64("2000-01-01 00:08:20"),
            np.timedelta64(5, "s"),
        ).astype("datetime64[ns]")
        range_coords = np.linspace(0, 100, 50)

        # Save
        img_meta = sample_manifest.images[0]
        img_meta.save_coordinates(tmp_cache.img_dataset, ping_coords, range_coords)

        # Load & verify
        loaded_ping, loaded_range = img_meta.load_coordinates(tmp_cache.img_dataset)
        assert np.allclose(loaded_ping.astype("int64"), ping_coords.astype("int64"))
        assert np.allclose(loaded_range, range_coords)

    def test_pixel_to_real_coords_scalar(self, tmp_cache, sample_manifest_with_coords):
        """Convert scalar pixel coordinates to real coords"""
        img_meta = sample_manifest_with_coords.images[0]

        # Get real coords at pixel (10, 20)
        ping_real, range_real = img_meta.pixel_to_real_coords(tmp_cache.img_dataset, x_pixel=10, y_pixel=20)

        assert ping_real.shape == ()  # Scalar
        assert range_real.shape == ()

    def test_pixel_to_real_coords_array(self, tmp_cache, sample_manifest_with_coords):
        """Convert multiple pixel coordinates"""
        img_meta = sample_manifest_with_coords.images[0]

        pixels_x = [0, 50, 99]
        pixels_y = [0, 25, 49]

        ping_real, range_real = img_meta.pixel_to_real_coords(tmp_cache.img_dataset, x_pixel=pixels_x, y_pixel=pixels_y)

        assert len(ping_real) == 3
        assert len(range_real) == 3

    def test_pixel_to_real_coords_boundary_clipping(self, tmp_cache, sample_manifest_with_coords):
        """Test boundary clipping for out-of-range pixels"""
        img_meta = sample_manifest_with_coords.images[0]

        # Query beyond image bounds
        ping_real, range_real = img_meta.pixel_to_real_coords(tmp_cache.img_dataset, x_pixel=9999, y_pixel=9999)

        # Should clip to max valid coords
        assert ping_real is not None
        assert range_real is not None


class TestImagesDatasetManifest:
    def test_save_load_manifest(self, tmp_cache, sample_manifest):
        """Save and load manifest JSON"""
        sample_manifest.save(tmp_cache.img_dataset)

        loaded = ImagesDatasetManifest.load(tmp_cache.img_dataset)
        assert loaded.source == sample_manifest.source
        assert len(loaded.images) == len(sample_manifest.images)
        assert loaded.images[0].filename == "image_0000.png"

    def test_get_image_metadata(self, sample_manifest):
        """Retrieve image metadata by filename"""
        meta = sample_manifest.get_image_metadata("image_0000.png")
        assert meta.filename == "image_0000.png"
        assert meta.ping_sample_start == 0

    def test_get_image_metadata_not_found(self, sample_manifest):
        """Raise error for missing image"""
        with pytest.raises(FileNotFoundError):
            sample_manifest.get_image_metadata("nonexistent.png")

    def test_labelme_polygon_to_real_coords(self, tmp_cache, sample_manifest_with_coords):
        """Convert LabelMe polygon points to real coords"""
        sample_manifest_with_coords.save(tmp_cache.img_dataset)

        # LabelMe points in pixel space
        labelme_points = [[10, 20], [50, 30], [40, 10]]

        ping_real, range_real = sample_manifest_with_coords.labelme_polygon_to_real_coords(
            tmp_cache.img_dataset, filename="image_0000.png", points=labelme_points
        )

        assert len(ping_real) == 3
        assert len(range_real) == 3

    def test_real_coords_to_labelme_polygon(self, tmp_cache, sample_manifest_with_coords):
        """Convert real coords back to LabelMe pixel points"""
        sample_manifest_with_coords.save(tmp_cache.img_dataset)

        img_meta = sample_manifest_with_coords.images[0]

        # Create mock polygon with real coords
        # Get full image axes coords
        ping_coords, range_coords = img_meta.load_coordinates(tmp_cache.img_dataset)
        # Subsample 10 points
        points_i = random.sample(range(len(ping_coords)), 10)
        points_j = random.sample(range(len(range_coords)), 10)
        ping_points, range_points = ping_coords[points_i], range_coords[points_j]

        assert len(ping_points) == 10
        assert len(range_points) == 10

        # Convert to LabelMe points list
        labelme_points = sample_manifest_with_coords.real_coords_to_labelme_polygon(
            tmp_cache.img_dataset,
            filename="image_0000.png",
            points_ping_axis_values=ping_points,
            points_range_axis_values=range_points,
        )

        # Should get back valid pixel coordinates
        assert len(labelme_points) == 10
        assert all(len(p) == 2 for p in labelme_points)  # [x, y] pairs
        assert all(isinstance(k, int) for point in labelme_points for k in point)  # ints
        assert 0 <= max([i for [i, _] in labelme_points]) < len(ping_coords)  # xaxis bounds
        assert 0 <= max([j for [_, j] in labelme_points]) < len(range_coords)  # yxaxis bounds

    def test_pixel_to_real_coords_via_manifest(self, tmp_cache, sample_manifest_with_coords):
        """Test pixel_to_real_coords wrapper method"""
        sample_manifest_with_coords.save(tmp_cache.img_dataset)

        ping_real, range_real = sample_manifest_with_coords.pixel_to_real_coords(
            tmp_cache.img_dataset, filename="image_0000.png", x_pixel=10, y_pixel=20
        )

        assert ping_real is not None
        assert range_real is not None
