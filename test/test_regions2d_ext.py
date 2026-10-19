"""Tests for regions2d_extension module (labelme_reader and labelme_writer)"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely.ops import unary_union

from echolabel.regions2d_extension import labelme_reader, labelme_writer
from echolabel.regions2d_extension.shapes import region_to_shapes, row_geometry, seconds_to_time


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
    required_cols = [
        "region_id",
        "region_name",
        "time",
        "depth",
        "region_bbox_left",
        "region_bbox_right",
        "region_bbox_top",
        "region_bbox_bottom",
        "shape_type",
    ]
    for col in required_cols:
        assert col in image_regions_df.columns, f"Missing column: {col}"

    # Check data types and values (= check match with dict)
    for i, (_, row) in enumerate(image_regions_df.iterrows()):
        # Get corresponding shape (df is built in the order of json "shapes")
        shape = annotation["shapes"][i]

        # Label should match
        assert row["region_name"] == shape["label"]

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
        original_df=sample_regions2d.data,
    )

    # Check dataframe structure
    assert isinstance(regions_df, pd.DataFrame)

    # Check match with ground-truth DataFrame (original columns; the reader adds echolabel_version)
    true_df: pd.DataFrame = sample_regions2d.data
    assert "echolabel_version" in regions_df.columns
    assert regions_df[true_df.columns].equals(true_df)


def test_rectangle_round_trip_keeps_shape_type(tmp_cache, sample_manifest_with_coords, sample_regions2d):
    """Rectangles export as rectangle shapes and come back as 4-corner rows with shape_type rectangle"""
    sample_manifest_with_coords.save(tmp_cache.img_dataset)
    labelme_data = labelme_writer.regions2d_to_labelme(
        sample_regions2d, sample_manifest_with_coords, tmp_cache.img_dataset
    )

    rect_shapes = [s for d in labelme_data.values() for s in d["shapes"] if s["shape_type"] == "rectangle"]
    assert len(rect_shapes) == 2
    assert all(len(s["points"]) == 2 for s in rect_shapes)

    for filename, data in labelme_data.items():
        with open(tmp_cache.labelme / filename, "w") as f:
            json.dump(data, f, indent=4)

    regions_df = labelme_reader.parse_echolabel(
        tmp_cache, sample_manifest_with_coords, original_df=sample_regions2d.data
    )
    rect_rows = regions_df[regions_df["shape_type"] == "rectangle"]
    assert sorted(rect_rows["region_id"]) == [2, 4]
    assert all(len(t) == 4 for t in rect_rows["time"])


def test_edited_rectangle_stays_rectangle(tmp_cache, sample_manifest_with_coords, sample_regions2d):
    """A rectangle resized in LabelMe is still exported as a 4-corner rectangle"""
    sample_manifest_with_coords.save(tmp_cache.img_dataset)
    labelme_data = labelme_writer.regions2d_to_labelme(
        sample_regions2d, sample_manifest_with_coords, tmp_cache.img_dataset
    )
    for filename, data in labelme_data.items():
        for shape in data["shapes"]:
            if shape["shape_type"] == "rectangle":
                # Shrink the rectangle by one pixel on each side
                (x0, y0), (x1, y1) = shape["points"]
                shape["points"] = [[x0 + 1, y0 + 1], [x1 - 1, y1 - 1]]
        with open(tmp_cache.labelme / filename, "w") as f:
            json.dump(data, f, indent=4)

    regions_df = labelme_reader.parse_echolabel(
        tmp_cache, sample_manifest_with_coords, original_df=sample_regions2d.data
    )
    rect_rows = regions_df[regions_df["region_id"] == 2]
    assert len(rect_rows) == 1
    assert rect_rows.iloc[0]["shape_type"] == "rectangle"
    assert len(rect_rows.iloc[0]["time"]) == 4
    assert len(rect_rows.iloc[0]["depth"]) == 4


def test_skipped_region_prints_warning(tmp_cache, sample_manifest_with_coords, sample_regions2d, capsys):
    """A region outside every image is skipped with a printed warning"""
    sample_manifest_with_coords.save(tmp_cache.img_dataset)
    df = sample_regions2d.data.iloc[[0]].copy()
    df["time"] = [np.array(["1990-01-01T00:00:00"] * 3 + ["1990-01-01T00:00:01"], dtype="datetime64[ns]")]
    df["depth"] = [np.array([10.0, 20.0, 30.0, 20.0])]

    labelme_data = labelme_writer.regions2d_to_labelme(df, sample_manifest_with_coords, tmp_cache.img_dataset)

    assert labelme_data == {}
    assert "Warning: region 1 skipped" in capsys.readouterr().out


def test_csv_safe_long_arrays_are_rereadable(tmp_path):
    """Arrays longer than numpy's print threshold must not be summarized with '...'"""
    import echoregions as er

    from echolabel.regions2d_extension.shapes import csv_safe

    n = 1500
    times = np.array(
        ["2017-04-09T21:33:17.000000000"] * n,
        dtype="datetime64[ns]",
    )
    depths = np.linspace(7.5, 749.5, n)
    df = pd.DataFrame(
        {
            "region_id": [1],
            "time": [times],
            "depth": [depths],
        }
    )
    csv_path = tmp_path / "long.csv"
    csv_safe(df).to_csv(csv_path, index=False)

    reread = er.read_regions_csv(csv_path).data
    assert len(reread["depth"][0]) == n
    assert np.allclose(reread["depth"][0], depths)


def test_polygon_between_columns_stays_valid_in_pixels():
    """Vertices between two image columns must not collapse into a self-intersecting shape"""
    from shapely.geometry import Polygon

    from echolabel.core.manifest import ImageMetadata
    from echolabel.regions2d_extension.shapes import ImageGrid, pixels_to_geometry, region_to_shapes

    times = np.arange(10, dtype=float) * 1.0
    depths = np.arange(10, dtype=float) * 1.0
    grid = ImageGrid(meta=ImageMetadata("img.png", 0, 9, 0, 9), times=times, depths=depths)

    # Vertices at x = 3.2, 3.4 and 3.6 all fall between columns 3 and 4
    geom = Polygon([(1.0, 2.0), (3.2, 2.5), (3.4, 6.0), (3.6, 2.6), (6.0, 2.0), (6.0, 8.0), (1.0, 8.0)])
    specs = region_to_shapes(geom, [grid])

    assert len(specs) == 1
    assert Polygon([tuple(p) for p in specs[0].points]).is_valid
    rebuilt = pixels_to_geometry(specs[0].points, "polygon", grid)
    assert rebuilt.symmetric_difference(geom).area < 1e-9


def test_misaligned_rectangle_warns_and_becomes_polygon(
    tmp_cache, sample_manifest_with_coords, sample_regions2d, capsys
):
    """A shape_type 'rectangle' row whose corners are not axis-aligned is imported as a polygon with a warning"""
    sample_manifest_with_coords.save(tmp_cache.img_dataset)
    df = sample_regions2d.data.copy()
    rect_idx = df.index[df["shape_type"] == "rectangle"][0]
    times = np.asarray(df.at[rect_idx, "time"]).copy()
    depths = np.asarray(df.at[rect_idx, "depth"], dtype=float).copy()
    # Shift one corner so the shape is a non-aligned quadrilateral
    depths[0] += 3.0
    df.at[rect_idx, "depth"] = depths
    df.at[rect_idx, "time"] = times

    labelme_data = labelme_writer.regions2d_to_labelme(df, sample_manifest_with_coords, tmp_cache.img_dataset)

    assert "has shape_type 'rectangle' but is not an axis-aligned box" in capsys.readouterr().out
    shapes = [s for d in labelme_data.values() for s in d["shapes"] if s["region_id"] == df.at[rect_idx, "region_id"]]
    assert shapes
    assert all(s["shape_type"] == "polygon" for s in shapes)


def test_fully_deleted_region_is_removed(tmp_cache, sample_manifest_with_coords, sample_regions2d):
    """When all visible shapes of a region are deleted in LabelMe, the region is deleted"""
    sample_manifest_with_coords.save(tmp_cache.img_dataset)
    labelme_data = labelme_writer.regions2d_to_labelme(
        sample_regions2d, sample_manifest_with_coords, tmp_cache.img_dataset
    )
    for filename, data in labelme_data.items():
        data["shapes"] = [s for s in data["shapes"] if s.get("region_id") != 2]
        with open(tmp_cache.labelme / filename, "w") as f:
            json.dump(data, f, indent=4)

    regions_df = labelme_reader.parse_echolabel(
        tmp_cache, sample_manifest_with_coords, original_df=sample_regions2d.data
    )
    assert 2 not in set(regions_df["region_id"])


def test_export_adds_metadata_columns_from_json(tmp_cache, sample_manifest_with_coords, sample_regions2d):
    """flags, group_id, description and echolabel_version are exported even if the source CSV lacks them"""
    sample_manifest_with_coords.save(tmp_cache.img_dataset)
    source_df = sample_regions2d.data.drop(
        columns=["flags", "group_id", "description", "echolabel_version"], errors="ignore"
    )
    labelme_data = labelme_writer.regions2d_to_labelme(
        sample_regions2d, sample_manifest_with_coords, tmp_cache.img_dataset
    )
    for filename, data in labelme_data.items():
        for shape in data["shapes"]:
            if shape.get("region_id") == 2:
                shape["group_id"] = 7
                shape["description"] = "checked"
                shape["flags"] = {"verified": True}
        with open(tmp_cache.labelme / filename, "w") as f:
            json.dump(data, f, indent=4)

    regions_df = labelme_reader.parse_echolabel(tmp_cache, sample_manifest_with_coords, original_df=source_df)

    for col in ("flags", "group_id", "description", "echolabel_version"):
        assert col in regions_df.columns
    row = regions_df[regions_df["region_id"] == 2].iloc[0]
    assert row["group_id"] == 7
    assert row["description"] == "checked"
    assert list(row["flags"]) == ["verified"]
    assert regions_df.loc[regions_df["region_id"] != 2, "group_id"].isna().all()


def test_partial_deletion_drops_outside_part_with_warning(
    tmp_cache, sample_manifest_with_coords, sample_regions2d, capsys
):
    """Deleting one visible shape of a region also deletes its outside-image part, with a warning"""
    sample_manifest_with_coords.save(tmp_cache.img_dataset)
    grids = labelme_reader.load_image_grids(sample_manifest_with_coords, tmp_cache.img_dataset)
    union_boxes = unary_union([g.box for g in grids])
    # Rewrite the first row as a polygon spanning two images, extending below the 100 m bottom
    df = sample_regions2d.data.copy()
    target = int(df.iloc[0]["region_id"])
    mid = [int(g.times.size // 2) for g in grids[:2]]
    t_a, t_b = grids[0].times[mid[0]], grids[1].times[mid[1]]
    df.at[df.index[0], "time"] = seconds_to_time([t_a, t_b, t_b, t_a])
    df.at[df.index[0], "depth"] = np.array([20.0, 20.0, 300.0, 300.0])
    df.at[df.index[0], "shape_type"] = "polygon"
    assert len(region_to_shapes(row_geometry(df.at[df.index[0], "time"], df.at[df.index[0], "depth"]), grids)) >= 2

    labelme_data = labelme_writer.regions2d_to_labelme(df, sample_manifest_with_coords, tmp_cache.img_dataset)
    removed = False
    for filename, data in labelme_data.items():
        kept = []
        for s in data["shapes"]:
            if s.get("region_id") == target and not removed:
                removed = True
                continue
            kept.append(s)
        data["shapes"] = kept
        with open(tmp_cache.labelme / filename, "w") as f:
            json.dump(data, f, indent=4)

    regions_df = labelme_reader.parse_echolabel(tmp_cache, sample_manifest_with_coords, original_df=df)

    assert f"region {target} was edited by deleting visible shapes" in capsys.readouterr().out
    rebuilt = unary_union(
        [row_geometry(r["time"], r["depth"]) for _, r in regions_df[regions_df["region_id"] == target].iterrows()]
    )
    assert rebuilt.difference(union_boxes).area < 1e-9
