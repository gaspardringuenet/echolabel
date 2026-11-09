"""
Generate colormaps for the two multifrequency composite echogram modes: RGB and OKLab. Uses parameters from echolabel's default configuration.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from echolabel import Config
from echolabel.config.config import OKLabSettings, resolve_oklab_settings
from echolabel.utils.echograms import RGB, Encoding, OKLab, plot_colormap

MAX_SVS: list[int] = [-90, -80, -70, -60, -50]  # max Sv [dB] of the colormap panels
SAVE_DIR: Path = Path(".")


def main():

    mpl.rcParams["axes.titlesize"] = 10

    cfg: Config = Config.load()

    settings: OKLabSettings = resolve_oklab_settings(cfg.oklab)

    enc_oklab = OKLab(
        vmin=cfg.vmin,
        vmax=cfg.vmax,
        sat_max=settings["sat_max"],
        l_range=(settings["l_range"][0], settings["l_range"][1]),
        c_max=settings["c_max"],
        theta=settings["theta"],
    )
    enc_rgb = RGB(
        vmin=cfg.vmin,
        vmax=cfg.vmax,
    )

    names: list[str] = ["RGB", "OKLab"]

    for enc, name in zip([enc_rgb, enc_oklab], names):
        make_cmap(enc, name)


def make_cmap(enc: Encoding, name: str):
    fig, axes = plt.subplots(
        figsize=(10, 2.5),
        ncols=5,
        sharey=True,
        layout="constrained",
    )

    for i, (ax, max_Sv) in enumerate(zip(axes, MAX_SVS)):
        plot_colormap(
            enc,
            max_Sv,
            ax=ax,
            paths_color=f"{1 - i / len(MAX_SVS)}",
        )
        if i != 0:
            ax.set_ylabel("")
    fig.suptitle(f"--cmap {name}")
    fig.savefig(SAVE_DIR / f"cmap_{name}.svg", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
