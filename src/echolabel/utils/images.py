from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import xarray
from PIL import Image


def sv2array(
    sv: xarray.DataArray,
    time_idx_slice: slice = slice(0, 100),
    depth_idx_slice: slice = slice(0, 100),
    channels: float | Sequence[float] = (38.0, 70.0, 120.0),
) -> np.ndarray:
    """Converts a Sv xarray.DataArray into a numpy array for plotting.

    Parameters
    ----------
    sv : xarray.DataArray
        Acoustic volume backscattering DataArray with coordinates time, depth, channel (in any order)
    time_idx_slice : slice, optional
        Time dimension index slice, by default slice(0, 100)
    depth_idx_slice : slice, optional
        Depth dimension index slice, by default slice(0, 100)
    channels : float | Sequence[float], optional
        Frequency channels to select, by default (38., 70., 120.)

    Returns
    -------
    np.ndarray
        Numpy array with shape (depth, time) or (depth, time, channels)
    """

    if isinstance(channels, (tuple, list)) and len(channels) == 1:
        channels = channels[0]

    sv_array = (
        sv.transpose("depth", "time", "channel")
        .isel(time=time_idx_slice, depth=depth_idx_slice)
        .sel(channel=channels)
        .values
    )

    return sv_array


def normalize_sv_array(a: np.ndarray, vmin: float = None, vmax: float = None) -> np.ndarray:
    """Normalizes array values to [0, 1]. If vmin (resp. vmax) is provided, values below vmin
    (resp. above vmax) are clipped to 0 (resp. 1). Else min and max values in a are used as
    defaults.

    Parameters
    ----------
    a : np.ndarray
        Array
    vmin : float, optional
        Minimal value, corresponding to 0, by default None
    vmax : float, optional
        Maximal value, corresponding to 1, by default None

    Returns
    -------
    np.ndarray
        Normalized array
    """

    if not vmin:
        vmin = a.min()
    if not vmax:
        vmax = a.max()

    a = (a - vmin) / (vmax - vmin)
    a = np.clip(a, 0, 1)
    a = np.nan_to_num(a, nan=0)

    return a


def sv_norm2image(a: np.ndarray, echogram_cmap: str = "RGB") -> Image.Image:
    """Converts numpy array in shape (H, W) or (H, W, 3) into a PIL Image.

    Parameters
    ----------
    a : np.ndarray
        Array
    echogram_cmap : str, optional
        Colormap for the image. 'RGB' is the only colormap accepted for (H, W, 3) array.
        For (H, W) arrays, colormap argument must be a matplotlib colormap name, by default 'RGB'

    Returns
    -------
    Image
        Output image

    Raises
    ------
    ValueError
        Errors when the number of channels is neither 0 nor 3, or when it doesn't correspond
        to the provided colormap argument.
    """

    if (len(a.shape) == 3) and (a.shape[2] == 3) and (echogram_cmap == "RGB"):
        img = Image.fromarray(np.uint8(a * 255))
    elif (len(a.shape) == 2) and (echogram_cmap != "RGB"):
        cmap = plt.get_cmap(echogram_cmap)
        img = Image.fromarray(np.uint8(cmap(a) * 255))
    else:
        raise ValueError(f"sv_array is of shape {a.shape}, which doesn't match the cmap '{echogram_cmap}'.")

    return img
