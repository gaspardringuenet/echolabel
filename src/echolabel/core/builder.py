"""
Build an image dataset from an acoustic dataset and save metadata using ImagesDatasetManifest.

Note: currently some confusion between naming conventions:

* Normalized dataset variables: 'tvar', 'zvar', 'cvar', 'fvar', 'acouvar' (time, depth, channel, frequency, acoustic variable)
* Real-world naming : 'ping_time_*', 'range_*'
"""

from datetime import datetime
from functools import partial
from pathlib import Path
from typing import List

import xarray as xr
from tqdm import tqdm

from ..utils.images import array2image
from .dataloader import normalize_vars, open_dataset  # open_mask
from .manifest import ImageMetadata, ImagesDatasetManifest


def build_images_dataset(
    source: str | Path,
    output_dir: Path,
    datavars_config: dict[str, str | float | None],
    freqs_hz: float | List[float] | None = None,
    frame_width: int = 1000,
    range_samples_slice: slice = slice(0, None),
    bin_mask: str | Path | None = None,  # binary mask: overlay negative region with alpha to highlight positive
    **viz_params,
) -> ImagesDatasetManifest:
    """Builds an echogram images dataset with metadata"""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Open source dataset
    source = Path(source)
    preprocess_fn = partial(normalize_vars, custom_config=datavars_config)
    ds_acou: xr.Dataset = open_dataset(source, preprocess_fn)

    # TODO Open binary mask dataset (if it exists)
    # if bin_mask is not None:
    #     da_mask: xr.Dataset = open_mask(bin_mask)

    images_metadata = []
    t_axis_coord = ds_acou.tvar.values
    z_axis_coord = ds_acou.zvar.values

    for i, t_start in tqdm(enumerate(range(0, len(t_axis_coord), frame_width)), desc="Building images"):
        t_end = min(t_start + frame_width, len(t_axis_coord))

        # Extract subset
        if freqs_hz is None:  # not provided: use the first channel
            da_sub: xr.DataArray = ds_acou.isel(
                tvar=slice(t_start, t_end),
                zvar=range_samples_slice,
                cvar=0,
            ).acouvar

        else:
            if isinstance(freqs_hz, float):
                freqs_hz = [freqs_hz]
            da_sub: xr.DataArray = (
                ds_acou.isel(tvar=slice(t_start, t_end), zvar=range_samples_slice)
                .sel(fvar=freqs_hz)
                .squeeze()  # drop channel if single frequency was selected
                .acouvar
            )

        # TODO Extract binary mask subset (if it exists)
        # if bin_mask:
        #     da_mask_sub = da_mask.isel(ping_time=slice(ping_start, ping_end), depth=range_samples_slice)

        # Save image
        filename = f"image_{i:04d}.png"
        save_echogram_image(da_sub, output_dir / filename, **viz_params)

        # Create metadata
        img_meta = ImageMetadata(
            filename=filename,
            ping_sample_start=t_start,
            ping_sample_end=t_end,
            range_sample_start=range_samples_slice.start or 0,
            range_sample_end=range_samples_slice.stop or da_sub.sizes["zvar"] - 1,
        )

        # Save coordinates
        img_meta.save_coordinates(
            output_dir,
            ping_axis_values=t_axis_coord[t_start:t_end],
            range_axis_values=z_axis_coord[range_samples_slice],
        )

        images_metadata.append(img_meta)

    # Save manifest file
    manifest = ImagesDatasetManifest(
        source=str(source),
        created_at=datetime.now().isoformat(),
        shape=dict(zip(da_sub.dims, da_sub.shape)),  # type: ignore
        channels=list(da_sub.channel.values) if "channel" in da_sub.dims else "first",  # type: ignore
        viz_params=viz_params,
        images=images_metadata,
    )
    manifest.save(output_dir)

    return manifest


def save_echogram_image(
    da: xr.DataArray,
    outfile: Path,
    vmin: float,
    vmax: float,
    echogram_cmap: str,
    bg_color: str,
    da_mask: xr.DataArray | None = None,
):
    """Computes an echogram image from acoustic data and saves it as png"""

    # Transpose and convert to np array
    try:
        array = da.transpose("zvar", "tvar", "fvar").values
    except ValueError:
        array = da.transpose("zvar", "tvar").values

    # Convert array to PIL image by applying cmap and bg_color
    # TODO overlay mask
    img = array2image(array, vmin, vmax, echogram_cmap, bg_color)

    # Save
    img.save(outfile)
