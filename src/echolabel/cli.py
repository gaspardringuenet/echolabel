import os
from pathlib import Path

import click

from echolabel.app import EcholabelApp
from echolabel.config.config import Config, init_user_config
from echolabel.utils.demo import download_demo_data


@click.group()
def cli():
    """Echolabel - Minimalist echogram annotation software"""


@cli.command()
def config():
    """Initialize user configuration file"""
    config_path = init_user_config()
    click.echo(f"User configuration initialized at: {config_path}")
    click.echo("Edit this file to customize your defaults")


@cli.command()
def cache():
    """Show default cache directory"""
    cfg = Config.load()
    app = EcholabelApp(cfg)
    click.echo(app.cache.root)


@cli.command()
@click.option("--output-dir", "-o", type=click.Path(path_type=Path))
def demo(output_dir):
    """Run echolabel in demo mode"""

    if not output_dir:
        output_dir = Path(os.getcwd())

    cfg = Config.load()
    app = EcholabelApp(cfg)

    source = download_demo_data(output_dir)
    output = output_dir / "regions.csv"

    click.echo(f"Processing {source} → {output}")
    app.run(source, output)
    click.echo("Complete!")


@cli.command()
@click.option(
    "--source",
    "-s",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="""Acoustic dataset. Accepted formats: .nc, .zarr,
    directory of compatible .nc files.""",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="""CSV annotation file to create or edit.""",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="""Path to custom config file. This option is for reproducibility.
    To modify user config, we recommend using `echolabel config`,
    then editing the YAML file. User config will be automatically detected.""",
)
@click.option(
    "--freqs-hz",
    "-f",
    multiple=True,
    type=click.FLOAT,
    help="""Frequency of a channel to visualize. The unit is Hz. This argument is
    matched against the frequency_nominal or channel variable in the dataset.
    If your frequency variable is not in Hz, provide scaling factor by specifying
    fvar_unit_scale in user config. E.g. fvar_unit_scale = 1000 for IMOS dataset
    defining channel in kHz. By default, the first channel of the dataset is
    used.""",
)
@click.option("--frame-width", type=int, help="Width of the images in ping axis samples.")
@click.option("--vmin", type=float, help="Minimum bound of color mapping. By default -90 dB.")
@click.option("--vmax", type=float, help="Maximum bound of color mapping. By default -50 dB.")
@click.option(
    "--cmap",
    type=str,
    help="""Colormap. When mapping a single frequency, any matplotlib
    cmap may be used. An additional'RGB' cmap will produce a tri-frequency
    composite echogram. By default 'viridis'.""",
)
@click.option(
    "--reuse-images/--no-reuse-images",
    default=None,
    help="""Reuse cached images dataset. Saves preprocessing time, but prone
    to errors if data or parameters have changed.""",
)
def process(
    source,
    output,
    config,
    freqs_hz,
    frame_width,
    vmin,
    vmax,
    cmap,
    reuse_images,
):
    """
    Run Labelme wrapper

    Examples:

    $ echolabel process -s data.zarr -o regions.csv -f 38000 --cmap magma

    $ echolabel process -s data.zarr -o regions.csv --vmin -60 --vmax -20

    $ echolabel process -s data.zarr -o regions.csv -f 38000 -f 70000 -f 120000 --cmap RGB

    $ echolabel process -s data.zarr -o regions.csv --reuse-images
    """

    # Load configuration with fallback chain
    cfg = Config.load(config)

    # Override with CLI arguments
    cli_overrides = {}
    if freqs_hz:
        cli_overrides["channels_frequency_nominal"] = list(freqs_hz)
    if frame_width is not None:
        cli_overrides["frame_width"] = frame_width
    if vmin is not None:
        cli_overrides["vmin"] = vmin
    if vmax is not None:
        cli_overrides["vmax"] = vmax
    if cmap is not None:
        cli_overrides["echogram_cmap"] = cmap
    if reuse_images is not None:
        cli_overrides["reuse_images"] = reuse_images

    cfg.update(cli_overrides)

    # Run application
    app = EcholabelApp(cfg)

    click.echo(f"Processing {source} → {output}")
    app.run(source, output)
    click.echo("Complete!")


if __name__ == "__main__":
    cli()
