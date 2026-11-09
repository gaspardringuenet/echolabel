from dataclasses import dataclass
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.dates import ConciseDateFormatter
from matplotlib.typing import ColorType

# OKLab to sRGB conversion matrices (https://bottosson.github.io/posts/oklab/)
_LAB_TO_LMS = np.array(
    [
        [1.0, 0.3963377774, 0.2158037573],
        [1.0, -0.1055613458, -0.0638541728],
        [1.0, -0.0894841775, -1.2914855480],
    ]
)
_LMS_TO_RGB = np.array(
    [
        [4.0767416621, -3.3077115913, 0.2309699292],
        [-1.2684380046, 2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147, 1.7076147010],
    ]
)


class Encoding(Protocol):
    def __call__(self, sv: np.ndarray) -> np.ndarray:  # type: ignore
        """Sv [dB] of (reference, f1, f2) along the last axis, shape (..., 3), NaN if missing
        -> RGB (..., 3) in [0, 1]."""


def _max_and_deltas(sv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """max over channels (NaN ignored), and ΔSv(f1), ΔSv(f2) relative to the reference."""
    max_Sv = np.fmax.reduce(sv, axis=-1)
    ref = sv[..., 0]
    return max_Sv, sv[..., 1] - ref, sv[..., 2] - ref


@dataclass(frozen=True)
class RGB:
    """Encoding of 3-channel Sv data using a simple RGB mapping: Sv(f0) -> R, Sv(f1) -> G,
    Sv(f3) -> B.

    Parameters
    ----------
    vmin : float, default -90.0
        Minimum Sv [dB] bound of the colormap.
    vmax : float, default -50.0
        Maximum Sv [dB] bound of the colormap.
    """

    vmin: float = -90.0
    vmax: float = -50.0

    def __call__(self, sv: np.ndarray) -> np.ndarray:

        if not sv.shape[-1] == 3:
            raise ValueError(f"Invalid Sv array shape {sv.shape}. Expected HWC with C==3.")

        rgb = ((sv - self.vmin) / (self.vmax - self.vmin)).clip(0, 1)
        # A missing channel is rendered as 0 (dark), not as equal to the reference
        return np.nan_to_num(rgb, nan=0.0)


@dataclass(frozen=True)
class OKLab:
    """Perceptual encoding in OKLab: max Sv sets lightness, ΔSv's set hue and chroma.

    Parameters
    ----------
    vmin, vmax : float
        Sv [dB] mapped linearly to lightness, clipped outside this range.
    sat_max : float
        ΔSv [dB] at which chroma reaches c_max.
    l_range : tuple[float, float]
        OKLab lightness at vmin and vmax.
    c_max : float
        Maximum OKLab chroma, reached at a ΔSv = sat_max.
    theta : float
        Rotation [rad] applied to the (ΔSv(f1), ΔSv(f2)) direction to get the (a, b) hue.
    """

    vmin: float = -90
    vmax: float = -50
    sat_max: float = 25
    l_range: tuple[float, float] = (0.25, 0.75)
    c_max: float = 0.15
    theta: float = 0.85 * np.pi

    def __call__(self, sv: np.ndarray) -> np.ndarray:

        if not sv.shape[-1] == 3:
            raise ValueError(f"Invalid Sv array shape {sv.shape}. Expected HWC with C==3.")

        max_Sv, df1, df2 = _max_and_deltas(sv)
        # NaN (empty pixels, masked at plot time) must not reach the gamut check: NaN fails every
        # comparison, so those pixels would all go through the costly chroma bisection
        value: np.ndarray = np.nan_to_num(
            ((max_Sv - self.vmin) / (self.vmax - self.vmin)).clip(0, 1),
            nan=0.0,
        )
        # Missing channel deltas are neutral (grey)
        return _oklab_rgb(
            value,
            np.nan_to_num(df1),
            np.nan_to_num(df2),
            sat_max=self.sat_max,
            l_range=self.l_range,
            c_max=self.c_max,
            theta=self.theta,
        )


def echogram(
    da_Sv: xr.DataArray,
    enc: Encoding,
    *,
    channel_idx: tuple[int, int, int] = (0, 1, 2),
    max_span: float | None = None,
    hwc: tuple[str, str, str] = ("depth", "ping_time", "channel"),
    ax: Axes | None = None,
    **plot_kwargs,
) -> QuadMesh:
    """
    Plot a trivariate echogram given a DataArray of volume backscattering strength.

    Parameters
    ----------
    da_Sv : xr.DataArray
        Sv data to plot.
    enc : Encoding
        Colour encoding, e.g. RGB() or OKLab().
    channel_idx : tuple[int, int, int], optional
        Indices of the reference, second and third channels in da_Sv.
    max_span : float, optional
        Maximum time span [s] between consecutive pings. Larger gaps are filled with NaN,
        and each quad is clipped to max_span / 2 on either side. By default no clipping.
    hwc : tuple[int, int, int], optional
        The name of the dimensions for height, width and channel in da_Sv.
    ax : Axes, optional
        Axes on which to draw the echogram. Uses the current Axes by default.
    **plot_kwargs
        Keyword arguments passed to plt.pcolormesh. `rasterized` defaults to True.

    Returns
    -------
    QuadMesh
        Echogram.
    """

    zdim, tdim, cdim = hwc

    # Keep the three channels, in (reference, f1, f2) order
    da_Sv = da_Sv.isel({cdim: list(channel_idx)}).transpose(*hwc)
    da_Sv.load()
    # Get data as numpy arrays, shape (depth, time, 3)
    sv: np.ndarray = da_Sv.values
    depth: np.ndarray = da_Sv[zdim].values
    time: np.ndarray = da_Sv[tdim].values

    # Compute RGB values
    rgb: np.ndarray = enc(sv)

    # Mask out invalid data (mostly bottom + some low insentity pixels)
    valid_data_mask: np.ndarray = ~np.all(np.isnan(sv), axis=-1)

    # Create echogram grid
    X, Y, gaps = _echogram_grid(time, depth, max_span)

    # Plot Quadmesh
    # Rasterize by default: one vector path per cell makes SVG export extremely slow
    plot_kwargs = {"rasterized": True} | plot_kwargs
    return _plot_mesh(ax, X, Y, rgb, valid_data_mask, gaps, plot_kwargs)


def colormap(
    enc: Encoding,
    max_Sv: float,
    lim: float,
    n=501,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    df1, df2 = np.meshgrid(np.linspace(-lim, lim, n), np.linspace(-lim, lim, n))
    # Reference Sv such that max(Sv_ref, Sv_f1, Sv_f2) = max_Sv
    ref = np.full_like(df1, max_Sv) - np.fmax(0, np.fmax(df1, df2))
    sv = np.stack([ref, ref + df1, ref + df2], axis=-1)
    rgb = enc(sv)
    return df1, df2, rgb


def plot_colormap(
    enc: Encoding,
    max_Sv: float,
    lim: float = 20,
    ax: Axes | None = None,
    samples: np.ndarray | None = None,
    paths_color: ColorType = "0.5",
    ylabel: str | None = None,
    xlabel: str | None = None,
):

    ax = ax or plt.gca()

    xlabel_ = r"$\Delta \text{MVBS}_{f_1 - f_{ref}}$ [dB]" if xlabel is None else xlabel
    ylabel_ = r"$\Delta \text{MVBS}_{f_2 - f_{ref}}$ [dB]" if ylabel is None else ylabel

    linekwargs = {
        "linestyle": "dashed",
        "linewidth": 0.5,
        "alpha": 0.4,
        "color": paths_color,
    }

    X, Y, C = colormap(enc, max_Sv, lim)
    ax.set_aspect(1)
    ax.pcolormesh(X, Y, C, rasterized=True)
    ax.axvline(0, **linekwargs)
    ax.axhline(0, **linekwargs)
    ax.axline((0, 0), slope=1, **linekwargs)
    ax.set_xlabel(xlabel_)
    ax.set_ylabel(ylabel_)
    ax.set_title(rf"$\max Sv = {max_Sv}$ dB")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # Scale markers drawn above the colormap mesh and the reference lines
    linekwargs = {
        "linestyle": "dashed",
        "linewidth": 0.5,
        "color": "red",
        "label": "2D scale",
        "zorder": 10,
        "clip_on": False,
    }
    dotkwargs = {
        "facecolor": "none",
        "edgecolor": "red",
        "label": "1D scale",
        "zorder": 10,
        "clip_on": False,
    }
    if isinstance(enc, RGB):
        ax.axhline(-lim, **linekwargs)
        ax.scatter(-lim, -lim, **dotkwargs)
    elif isinstance(enc, OKLab):
        ax.axhline(0, **linekwargs)
        ax.scatter(0, 0, **dotkwargs)
    else:
        print(f"Warning: Unknown encodingn {type(enc)}, unable to plot 1D and 2D scales info.")

    if samples is not None:
        ax.scatter(
            samples[0],
            samples[1],
            s=3,
            linewidths=0.2,
            facecolors="none",
            edgecolors=paths_color,
            zorder=2,
        )


# ---- Helper functions ---- #


def _oklab_rgb(
    value: np.ndarray,
    df1: np.ndarray,
    df2: np.ndarray,
    sat_max: float = 25.0,
    l_range: tuple[float, float] = (0.25, 0.75),
    c_max: float = 0.15,
    theta: float = 0.85 * np.pi,
) -> np.ndarray:
    """Convert OKLab coordinates to sRGB, with gamut clipping by chroma reduction.

    Lightness is linear in value. (a, b) is linear in (df1, df2): the chroma c_max is reached at
    a spectral difference of sat_max, then clipped radially. Colours outside the sRGB gamut have
    their chroma reduced at constant lightness and hue, as oklab.clamp does.

    Parameters
    ----------
    max_Sv : np.ndarray
        max(Sv) accross channel, normalised in [0, 1], mapped linearly to lightness over l_range.
    df1, df2 : np.ndarray
        ΔSv(f1), ΔSv(f2) [dB], same shape as value. Must not contain NaN.
    sat_max, l_range, c_max, theta
        See OKLab.

    Returns
    -------
    np.ndarray
        sRGB values in [0, 1], shape (*value.shape, 3).
    """

    lightness = l_range[0] + (l_range[1] - l_range[0]) * value
    r = np.hypot(df1, df2)
    chroma = c_max * np.minimum(r / sat_max, 1.0)

    # Rotate the unit direction (df1, df2) / r by theta without trigonometric calls per pixel
    unit = np.divide(1.0, r, out=np.zeros_like(r), where=r > 0)
    cos_a = unit * (df1 * np.cos(theta) - df2 * np.sin(theta))
    sin_a = unit * (df1 * np.sin(theta) + df2 * np.cos(theta))

    ab = (chroma * np.stack([cos_a, sin_a])).reshape(2, -1)
    light = lightness.ravel()

    def to_srgb_linear(idx, scale):
        # scale multiplies chroma only, so bisection can shrink it while keeping L and hue
        lab = np.stack([light[idx], scale * ab[0, idx], scale * ab[1, idx]], axis=-1)
        return ((lab @ _LAB_TO_LMS.T) ** 3) @ _LMS_TO_RGB.T

    def in_gamut(lin):
        # Small tolerance so that floating-point noise at the gamut boundary is not reduced
        return np.all((lin >= -1e-6) & (lin <= 1 + 1e-6), axis=-1)

    # Full chroma first; only out-of-gamut pixels go through the bisection below
    lin = to_srgb_linear(slice(None), 1.0)

    # Chroma reduction by bisection, only for out-of-gamut pixels
    bad = np.flatnonzero(~in_gamut(lin))
    if bad.size:
        lo, hi = np.zeros(bad.size), np.ones(bad.size)
        for _ in range(16):
            mid = (lo + hi) / 2
            good = in_gamut(to_srgb_linear(bad, mid))
            lo = np.where(good, mid, lo)
            hi = np.where(good, hi, mid)
        lin[bad] = to_srgb_linear(bad, lo)

    lin = lin.clip(0, 1).reshape(*lightness.shape, 3)
    # sRGB transfer function (linear -> gamma-encoded)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


def _echogram_grid(
    time: np.ndarray,
    depth: np.ndarray,
    max_span: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell edges of the echogram mesh and the indices of the time gaps to fill.

    Returns
    -------
    X, Y : np.ndarray
        Meshgrid of time and depth cell edges, with one extra column per gap.
    gaps : np.ndarray
        Indices i such that time[i + 1] - time[i] > max_span.
    """
    # Time cell edges: midpoints between pings, with each half-width capped at max_span / 2.
    # Consecutive pings further apart than max_span get a masked column filling the gap.
    dt = np.diff(time)
    half_left = np.concatenate([dt[:1], dt]) / 2
    half_right = np.concatenate([dt, dt[-1:]]) / 2
    gaps = np.array([], dtype=int)
    if max_span is not None:
        span = np.timedelta64(round(max_span * 1000), "ms")
        half_left = np.minimum(half_left, span / 2)
        half_right = np.minimum(half_right, span / 2)
        gaps = np.flatnonzero(dt > span)

    time_edges = np.insert(
        np.concatenate([[time[0] - half_left[0]], time + half_right]),
        gaps + 2,
        time[gaps + 1] - half_left[gaps + 1],
    )

    # Depth cell edges: midpoints between samples, extrapolated at both ends
    depth_mid = (depth[1:] + depth[:-1]) / 2
    depth_edges = np.concatenate([[2 * depth[0] - depth_mid[0]], depth_mid, [2 * depth[-1] - depth_mid[-1]]])

    X, Y = np.meshgrid(time_edges, depth_edges)
    return X, Y, gaps


def _plot_mesh(
    ax: Axes | None,
    X: np.ndarray,
    Y: np.ndarray,
    C: np.ndarray,
    valid_data_mask: np.ndarray,
    gaps: np.ndarray,
    plot_kwargs: dict,
) -> QuadMesh:
    """Fill gaps with masked columns, draw the mesh and format the time and depth axes.

    C is either a (depth, time) scalar array or a (depth, time, 3) RGB array.
    """
    C = np.insert(C, gaps + 1, 0.0, axis=1)
    valid_data_mask = np.insert(valid_data_mask, gaps + 1, False, axis=1)
    C = np.ma.array(C, mask=np.dstack([~valid_data_mask] * 3))

    ax = ax or plt.gca()
    im = ax.pcolormesh(X, Y, C, shading="flat", **plot_kwargs)

    ax.xaxis.set_major_formatter(ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.yaxis.set_inverted(True)

    return im
