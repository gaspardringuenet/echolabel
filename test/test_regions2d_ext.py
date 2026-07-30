"""Tests for regions2d_extension module (labelme_reader and labelme_writer)"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from echolabel.regions2d_extension import labelme_reader, labelme_writer


@pytest.mark.parametrize(
    "shape_types, image, sample_labelme_json",
    [
        ("polygon", "image_0000.png", "sample_labelme_annotation_0"),
        ("rectangle", "image_0001.png", "sample_labelme_annotation_1"),
        ("both", "image_0002.png", "sample_labelme_annotation_2"),
    ],
)
def test_regions2d_to_labelme(
    tmp_cache,
    shape_types,
    image,
    sample_manifest_with_coords,
    sample_regions2d,
    sample_labelme_json,
    request,
):
    """
    Convert a Regions2D to LabelMe format. Check conversion against
    ground-truth hand-made LabelMe annotation. A region should be assigned
    to the correct image based on metadata (store in sample_manifest_with_coords)

    Cases checked (see parametrization):
    - Single polygon shape (first image)
    - Single rectangle shape (2nd image; rectangles require special conversion)
    - Both a polygon and a rectangle (3rd image)
    """

    # Get actual value for labelme annotation
    # This is the ground truth for one image
    true_annotation = request.getfixturevalue(sample_labelme_json)
    json_fname = Path(true_annotation["imagePath"]).stem + ".json"

    # Save manifest coordinates
    sample_manifest_with_coords.save(tmp_cache.img_dataset)

    # Convert regions2d object to LabelMe
    labelme_data = labelme_writer.regions2d_to_labelme(
        sample_regions2d,
        sample_manifest_with_coords,
        tmp_cache.img_dataset,
    )

    # Check conversion against ground truth
    assert len(labelme_data) > 0
    assert json_fname in list(labelme_data.keys())

    annotation = labelme_data[json_fname]
    assert annotation["version"] == "6.0.0"

    shapes = annotation["shapes"]
    true_shapes = true_annotation["shapes"]
    assert len(shapes) == len(true_shapes)

    for s, ts in zip(shapes, true_shapes):
        assert s["label"] == ts["label"]
        assert s["shape_type"] == ts["shape_type"]
        assert len(s["points"]) == len(s["points"])


@pytest.mark.parametrize(
    "shape_types, image, sample_labelme_json",
    [
        ("polygon", "image_0000.png", "sample_labelme_annotation_0"),
        # ("rectangle", "image_0001.png", "sample_labelme_annotation_1"),
        # ("both", "image_0002.png", "sample_labelme_annotation_2"),
    ],
)
def test_parse_labelme(
    tmp_cache,
    shape_types,
    image,
    sample_manifest_with_coords,
    sample_labelme_json,
    request,
):
    """
    Convert LabelMe image annotation dict (loaded JSON) into an
    echoregions-compatible dataframe using the manifest as key.

    Check that the dataframe structure and compare values to JSON
    data.
    """
    # Get actual value for labelme annotation
    annotation = request.getfixturevalue(sample_labelme_json)

    # Save manifest coordinates
    sample_manifest_with_coords.save(tmp_cache.img_dataset)

    # Convert image LabelMe annotation to Regions2D
    image_regions_df = labelme_reader.parse_labelme(
        annotation,
        sample_manifest_with_coords,
        tmp_cache.img_dataset,
    )

    # Check dataframe structure
    assert isinstance(image_regions_df, pd.DataFrame)
    assert len(image_regions_df) == len(annotation["shapes"])

    # Check required columns exist
    evr_required_cols = [
        "region_id",
        "region_name",
        "region_class",
        "time",
        "depth",
        "region_bbox_left",
        "region_bbox_right",
        "region_bbox_top",
        "region_bbox_bottom",
        "echoview_version",
        "region_structure_version",
        "region_creation_type",
    ]
    for col in evr_required_cols:
        assert col in image_regions_df.columns, f"Missing column: {col}"

    # Check data types and values (= check match with dict)
    for i, (_, row) in enumerate(image_regions_df.iterrows()):
        # Get corresponding shape (df is built in the order of json "shapes")
        shape = annotation["shapes"][i]

        # Label should match (double usage for now)
        assert row["region_name"] == shape["label"]
        assert row["region_class"] == shape["label"]

        # Coordinates should be arrays
        assert isinstance(row["time"], np.ndarray)
        assert isinstance(row["depth"], np.ndarray)


def test_parse_round_trip(
    tmp_cache,
    sample_manifest_with_coords,
    sample_regions2d,
):
    """
    Round-trip conversion Regions2D.data (DataFrame) → LabelMe cache
    → Regions2D.data.
    Check full equality between DataFrame.
    """

    # Save manifest coordinates
    sample_manifest_with_coords.save(tmp_cache.img_dataset)

    # Parse Regions2D -> JSON and save in tmp cache
    labelme_data = labelme_writer.regions2d_to_labelme(
        sample_regions2d,
        sample_manifest_with_coords,
        tmp_cache.img_dataset,
    )
    for filename, data in labelme_data.items():
        with open(tmp_cache.labelme / filename, "w") as f:
            json.dump(data, f, indent=4)

    # Parse all JSON files into a Regions2D dataframe
    regions_df = labelme_reader.parse_echolabel(
        tmp_cache,
        sample_manifest_with_coords,
    )

    # Check dataframe structure
    assert isinstance(regions_df, pd.DataFrame)

    # Check match with ground-truth DataFrame
    # Fetch ground-truth data
    true_df: pd.DataFrame = sample_regions2d.data

    assert regions_df.equals(true_df)
