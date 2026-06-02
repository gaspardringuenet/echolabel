from pathlib import Path

from echolabel.app import EcholabelApp

LIBRARY = Path("/Users/gaspardringuenet/Library/Caches/echolabel/test-builder/output_library/regions.csv")


def main():

    app = EcholabelApp()

    builder_params = dict(
        source="/Users/gaspardringuenet/Library/Caches/echolabel_v2/sample_data.nc",
        output_dir=app.cache.img_dataset,
        channels=[38.0, 70.0, 120.0],
        frame_width=1000,
        range_samples_slices=slice(0, None),
        vmin=-90.0,
        vmax=-50.0,
        echogram_cmap="RGB",
    )

    app.run(
        output="/Users/gaspardringuenet/Library/Caches/echolabel/test-builder/output_library/regions.csv",
        reuse_images=False,
        builder_params=builder_params,
    )


if __name__ == "__main__":
    main()
