from pathlib import Path

from echolabel.builder import build_images_dataset


def main():

    build_images_dataset(
        source=Path(
            "/Users/gaspardringuenet/Projects/sv-extraction/private/data/input/Amazomix/IRD_SOOP-BA_A_20210828T223155Z_ANTEA_FV02_AMAZOMIX2021-38-70-120-200_END-20211008T082151Z_C-20251014T104337Z.nc"
        ),
        cache_dir=Path("/Users/gaspardringuenet/Library/Caches/echolabel/test-builder"),
        channels=[38, 70, 120],
        frame_width=10_000,
        range_samples_slice=slice(0, None),
        vmin=-90,
        vmax=-50,
        echogram_cmap="RGB",
    )


if __name__ == "__main__":
    main()
