import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import colors
from matplotlib.typing import ColorType
from PIL import Image

from .echograms import Encoding


def array2image(
    a: npt.NDArray[np.float32],
    enc: Encoding | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    echogram_cmap: str = "viridis",
    bg_color: ColorType = "grey",
) -> Image.Image:
    """
    Convert numpy array in shape (H, W) or (H, W, 3) into a PIL Image

    Parameters
    ----------
    a : np.ndarray
        Array
    enc : Encoding, optional
        Encoder for three-channel data; required for arrays of shape (H, W, 3).
    vmin, vmax : float, optional
        Bounds for single-channel color mapping. Three-channel bounds belong to enc.
    echogram_cmap : str, optional
        Matplotlib colormap for single-channel arrays, by default 'viridis'.
    bg_color : ColorType, optional
        Background color to fill where data is missing, by default "grey"

    Returns
    -------
    Image.Image
        Echogram image

    Raises
    ------
    ValueError
        When the array shape does not match the provided encoder.
    """
    bg_rgb: npt.NDArray = colors.to_rgba_array(bg_color)[0]

    if a.ndim == 3 and a.shape[2] == 3 and enc is not None:
        rgb = enc(a)
        all_channels_na = np.all(np.isnan(a), axis=-1, keepdims=True)
        rgb = np.where(all_channels_na, bg_rgb[:3], rgb)
        return Image.fromarray((rgb * 255).round().astype(np.uint8))

    if a.ndim == 2 and enc is None:
        min_value: float = np.nanmin(a) if vmin is None else vmin  # type: ignore
        max_value: float = np.nanmax(a) if vmax is None else vmax  # type: ignore
        a = np.clip((a - min_value) / (max_value - min_value), 0, 1)

        # Get cmap object
        cmap = plt.get_cmap(echogram_cmap)

        # Create mask for NaN values
        nan_mask = np.isnan(a)

        # Replace NaN with 0 for colormap conversion
        a_filled = np.nan_to_num(a, nan=0)
        rgb_array = np.uint8(cmap(a_filled) * 255)
        bg_color_uint8 = np.uint8(bg_rgb * 255)

        # Apply background color where NaN (expand mask to match RGBA channels)
        rgb_array = np.where(nan_mask[..., np.newaxis], bg_color_uint8, rgb_array)
        img = Image.fromarray(rgb_array)

    else:
        raise ValueError(f"Array shape {a.shape} does not match the provided encoder.")

    return img
