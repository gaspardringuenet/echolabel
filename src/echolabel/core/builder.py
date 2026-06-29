from datetime import datetime
from pathlib import Path
from typing import List

import xarray as xr
from tqdm import tqdm

from ..utils.images import sv_array2image  # , sv_norm2image, normalize_sv_array,
from .dataloader import open_dataset
from .manifest import ImageMetadata, ImagesDatasetManifest

# ---- New function working with manifest


def build_images_dataset(
    source: str | Path | List[str | Path],
    output_dir: Path,
    freqs_hz: float | List[float] | None = None,
    frame_width: int = 1000,
    range_samples_slice: slice = slice(0, None),
    **viz_params,
) -> ImagesDatasetManifest:
    """Builds an echogram images dataset with metadata"""

    output_dir.mkdir(parents=True, exist_ok=True)

    source = Path(source)
    ds_MVBS: xr.Dataset = open_dataset(source)  # Open source dataset

    images_metadata = []
    ping_axis_coord = ds_MVBS.ping_time.values
    range_axis_coord = ds_MVBS.depth.values

    for i, ping_start in tqdm(enumerate(range(0, len(ping_axis_coord), frame_width)), desc="Building images"):
        ping_end = min(ping_start + frame_width, len(ping_axis_coord))

        # Extract subset
        if freqs_hz is None:  # not provided: use the first channel
            da_sub = ds_MVBS.isel(ping_time=slice(ping_start, ping_end), depth=range_samples_slice, channel=0).Sv

        else:
            if isinstance(freqs_hz, float):
                freqs_hz = [freqs_hz]
            da_sub = (
                ds_MVBS.isel(ping_time=slice(ping_start, ping_end), depth=range_samples_slice)
                .sel(channel=ds_MVBS.frequency_nominal.isin(freqs_hz))
                .squeeze()  # drop channel if single frequency was selected
                .Sv
            )

        # Save image
        filename = f"image_{i:04d}.png"
        save_echogram_image(da_sub, output_dir / filename, **viz_params)

        # Create metadata
        img_meta = ImageMetadata(
            filename=filename,
            ping_sample_start=ping_start,
            ping_sample_end=ping_end,
            range_sample_start=range_samples_slice.start or 0,
            range_sample_end=range_samples_slice.stop or da_sub.sizes["depth"] - 1,
        )

        # Save coordinates
        img_meta.save_coordinates(
            output_dir,
            ping_axis_values=ping_axis_coord[ping_start:ping_end],
            range_axis_values=range_axis_coord[range_samples_slice],
        )

        images_metadata.append(img_meta)

    # Save manifest file
    if isinstance(source, list):
        source = list(map(str, source))
    else:
        source = str(source)

    manifest = ImagesDatasetManifest(
        source=source,
        created_at=datetime.now().isoformat(),
        sv_shape=dict(zip(da_sub.dims, da_sub.shape)),
        channels=list(da_sub.channel.values) if "channel" in da_sub.dims else "first",
        viz_params=viz_params,
        images=images_metadata,
    )
    manifest.save(output_dir)

    return manifest


def save_echogram_image(
    sv_da: xr.DataArray,
    outfile: Path,
    vmin: float,
    vmax: float,
    echogram_cmap: str,
    bg_color: str,
):
    """Computes an echogram image from acoustic data and saves it as png"""

    # Transpose and convert to np array
    try:
        sv_array = sv_da.transpose("depth", "ping_time", "channel").values
    except ValueError:
        sv_array = sv_da.transpose("depth", "ping_time").values

    # Convert array to PIL image by applying cmap and bg_color
    img = sv_array2image(sv_array, vmin, vmax, echogram_cmap, bg_color)

    # Save
    img.save(outfile)
