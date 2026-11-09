from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from echolabel.utils.echograms import RGB, Encoding, OKLab, plot_colormap

MAX_SVS: list[int] = [-90, -80, -70, -60, -50]  # max Sv [dB] of the colormap panels
SAVE_DIR: Path = Path(".")


def main():

    mpl.rcParams["axes.titlesize"] = 10

    # --- RGB --- #
    enc: Encoding = RGB()
    fig, axes = plt.subplots(
        figsize=(10, 2.5),
        ncols=5,
        sharey=True,
        layout="constrained",
    )

    for i, (ax, max_Sv) in enumerate(zip(axes, MAX_SVS)):
        plot_colormap(enc, max_Sv, ax=ax, paths_color=f"{1 - i / len(MAX_SVS)}")
        if i != 0:
            ax.set_ylabel("")
    fig.suptitle("--cmap RGB")
    fig.savefig(SAVE_DIR / "cmap_RGB.svg", dpi=300)

    # --- OKLab --- #
    enc: Encoding = OKLab(
        sat_max=28.3,
        l_range=(0.3, 0.75),
        c_max=0.25,
        theta=2.67,
    )
    fig, axes = plt.subplots(
        figsize=(10, 2.5),
        ncols=5,
        sharey=True,
        layout="constrained",
    )

    for i, (ax, max_Sv) in enumerate(zip(axes, MAX_SVS)):
        plot_colormap(enc, max_Sv, ax=ax, paths_color=f"{1 - i / len(MAX_SVS)}")
        if i != 0:
            ax.set_ylabel("")
    fig.suptitle("--cmap OKLab")
    fig.savefig(SAVE_DIR / "cmap_OKLab.svg", dpi=300)


if __name__ == "__main__":
    main()
