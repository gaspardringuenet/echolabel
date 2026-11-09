"""
Build an image dataset from an acoustic dataset and save metadata using ImagesDatasetManifest.

Note: currently some confusion between naming conventions:

* Normalized dataset variables: 'tvar', 'zvar', 'cvar', 'fvar', 'acouvar' (time, depth, channel, frequency, acoustic variable)
* Real-world naming : 'ping_time_*', 'range_*'
"""

from collections.abc import Mapping
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

from ..config.config import resolve_oklab_settings
from ..utils.echograms import RGB, Encoding, OKLab
from ..utils.images import array2image
from .dataloader import normalize_vars, open_dataset  # open_mask
from .manifest import ImageMetadata, ImagesDatasetManifest


def image_cache_key(
    source: str | Path,
    datavars_config: dict,
    freqs_hz: float | list[float] | None = None,
    frame_width: int = 1000,
    range_samples_slice: slice = slice(0, None),
    bin_mask: str | Path | None = None,
    **viz_params,
) -> dict:
    """Describe the source and settings used to build the cached images."""
    path = Path(source).resolve()
    stat = path.stat()
    if isinstance(freqs_hz, (int, float)):
        freqs_hz = [float(freqs_hz)]
    if freqs_hz == []:
        freqs_hz = None
    return {
        "source": str(path),
        "source_mtime_ns": stat.st_mtime_ns,
        "source_size": stat.st_size,
        "datavars": datavars_config,
        "freqs_hz": freqs_hz,
        "frame_width": frame_width,
        "range_samples_slice": [range_samples_slice.start, range_samples_slice.stop, range_samples_slice.step],
        "bin_mask": str(bin_mask) if bin_mask is not None else None,
        "viz_params": viz_params,
    }


def build_images_dataset(
    source: str | Path,
    output_dir: Path,
    datavars_config: dict[str, str | float | None],
    freqs_hz: float | list[float] | None = None,
    frame_width: int = 1000,
    range_samples_slice: slice = slice(0, None),
    bin_mask: str | Path | None = None,  # binary mask: overlay negative region with alpha to highlight positive
    **viz_params,
) -> ImagesDatasetManifest:
    """Builds an echogram images dataset with metadata"""

    if viz_params.get("echogram_cmap") == "OKLab":
        viz_params["oklab"] = resolve_oklab_settings(viz_params.get("oklab"))

    if frame_width <= 0:
        raise ValueError("frame_width must be positive")
    if range_samples_slice.step is not None and range_samples_slice.step <= 0:
        raise ValueError("range_samples_slice step must be positive")
    if isinstance(freqs_hz, (int, float)):
        freqs_hz = [float(freqs_hz)]
    if freqs_hz == []:
        freqs_hz = None
    if freqs_hz is not None and len(freqs_hz) not in (1, 3):
        raise ValueError("Select one or three frequencies")
    vmin, vmax = viz_params.get("vmin"), viz_params.get("vmax")
    if vmin is not None and vmax is not None and (not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax):
        raise ValueError("vmin and vmax must be finite and vmin must be less than vmax")

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
    selected_depths = z_axis_coord[range_samples_slice]
    if len(t_axis_coord) == 0 or len(selected_depths) == 0:
        raise ValueError("The selected time and depth axes must not be empty")
    if not np.issubdtype(t_axis_coord.dtype, np.datetime64) or np.any(np.isnat(t_axis_coord)):
        raise ValueError("The time axis must contain valid datetime values")
    if np.any(np.diff(t_axis_coord) <= np.timedelta64(0, "ns")):
        raise ValueError("The time axis must be strictly increasing")
    if not np.issubdtype(selected_depths.dtype, np.number):
        raise ValueError("The selected depth axis must be numeric")
    if not np.all(np.isfinite(selected_depths)) or np.any(np.diff(selected_depths) <= 0):
        raise ValueError("The selected depth axis must be finite and strictly increasing")

    previous_end = 0
    for i, t_start in tqdm(enumerate(range(0, len(t_axis_coord), frame_width)), desc="Building images"):
        if t_start < previous_end:
            continue
        t_end = min(t_start + frame_width, len(t_axis_coord))
        if frame_width > 1 and len(t_axis_coord) - t_end == 1:
            t_end += 1

        # Extract subset
        if freqs_hz is None:  # not provided: use the first channel
            da_sub: xr.DataArray = ds_acou.isel(
                tvar=slice(t_start, t_end),
                zvar=range_samples_slice,
                fvar=0,
            ).acouvar

        else:
            da_sub: xr.DataArray = (
                ds_acou.isel(tvar=slice(t_start, t_end), zvar=range_samples_slice).sel(fvar=freqs_hz).acouvar
            )
            if da_sub.sizes["fvar"] == 1:
                da_sub = da_sub.squeeze("fvar", drop=True)

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
        previous_end = t_end

    # Save manifest file
    manifest = ImagesDatasetManifest(
        source=str(source),
        created_at=datetime.now(UTC).isoformat(),
        shape=dict(da_sub.sizes),  # type: ignore
        freqs_hz=freqs_hz if (freqs_hz is not None) else "first",
        viz_params=viz_params,
        images=images_metadata,
        cache_key=image_cache_key(
            source, datavars_config, freqs_hz, frame_width, range_samples_slice, bin_mask, **viz_params
        ),
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
    oklab: Mapping[str, object] | None = None,
):
    """Computes an echogram image from acoustic data and saves it as png"""

    # Transpose and convert to np array
    try:
        array = da.transpose("zvar", "tvar", "fvar").values
    except ValueError:
        array = da.transpose("zvar", "tvar").values

    enc: Encoding | None = None
    if echogram_cmap == "RGB":
        enc = RGB(vmin=vmin, vmax=vmax)
    elif echogram_cmap == "OKLab":
        settings = resolve_oklab_settings(oklab)
        enc = OKLab(
            vmin=vmin,
            vmax=vmax,
            sat_max=settings["sat_max"],
            l_range=(settings["l_range"][0], settings["l_range"][1]),
            c_max=settings["c_max"],
            theta=settings["theta"],
        )

    # Convert array to PIL image by applying the encoder or scalar colormap.
    # TODO overlay mask
    img = array2image(array, enc=enc, vmin=vmin, vmax=vmax, echogram_cmap=echogram_cmap, bg_color=bg_color)

    # Save
    img.save(outfile)
