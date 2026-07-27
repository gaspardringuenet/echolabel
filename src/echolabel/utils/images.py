import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import colors
from matplotlib.typing import ColorType
from PIL import Image


def sv_array2image(
    a: npt.NDArray[np.float32],
    vmin: float | None = None,
    vmax: float | None = None,
    echogram_cmap: str = "RGB",
    bg_color: ColorType = "grey",
) -> Image.Image:
    """
    Convert numpy array in shape (H, W) or (H, W, 3) into a PIL Image

    Parameters
    ----------
    a : np.ndarray
        Array
    vmin : float, optional
        Minimal value for color mapping, by default None
    vmax : float, optional
        Maximal value for color mapping, by default None
    echogram_cmap : str, optional
        Colormap for the image. 'RGB' is the only colormap accepted for (H, W, 3) array.
        For (H, W) arrays, colormap argument must be a matplotlib colormap name, by default 'RGB'
    bg_color : ColorType, optional
        Background color to fill where data is missing, by default "grey"

    Returns
    -------
    Image.Image
        Echogram image

    Raises
    ------
    ValueError
        When the number of channels is neither 0 nor 3, or when it doesn't correspond
        to the provided colormap argument.
    """

    # Handle missing vmin, vmax params
    min_value: float = vmin or a.min()
    max_value: float = vmax or a.min()

    # Convert array values to (0, 1) range (use vmin, vmax and clip)
    a = (a - min_value) / (max_value - min_value)
    a = np.clip(a, 0, 1)

    # Get rgba vector for background color
    bg_rgb: npt.NDArray = colors.to_rgba_array(bg_color)[0]

    # If array as 3 channels, convert to RGB echogram image
    if (len(a.shape) == 3) and (a.shape[2] == 3) and (echogram_cmap == "RGB"):
        # Where all channels are NA --> apply bg_color
        all_channels_na = np.all(np.isnan(a), axis=-1, keepdims=True)

        # Disable channel when it is NA (data still displayed as long as one channel is active)
        a = np.nan_to_num(a, nan=0)

        # But show as bg_color where all channels are NA
        a = np.where(all_channels_na, np.ones_like(a) * bg_rgb[:3], a)

        # Convert to image
        img = Image.fromarray(np.uint8(a * 255))

    # If array has a single channel - (depth, ping_time) shape - apply Matplotlib cmap
    elif (len(a.shape) == 2) and (echogram_cmap != "RGB"):
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
        raise ValueError(f"sv_array is of shape {a.shape}, which doesn't match the cmap '{echogram_cmap}'.")

    return img
