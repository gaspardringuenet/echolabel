import numpy as np
import pytest

from echolabel.core.manifest import ImagesDatasetManifest


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
