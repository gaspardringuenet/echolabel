from pathlib import Path

import click

from echolabel.app import EcholabelApp
from echolabel.config.config import Config, init_user_config


@click.group()
def cli():
    """Echolabel - Minimalist echogram annotation software"""
    pass


@cli.command()
def init():
    """Initialize user configuration file"""
    config_path = init_user_config()
    click.echo(f"Configuration initialized at: {config_path}")
    click.echo("Edit this file to customize your defaults")


@cli.command()
def cache():
    """Show default cache directory"""
    app = EcholabelApp()
    click.echo(f"Default cache directory - {app.cache.root}")


@cli.command()
@click.option("--source", "-s", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", required=True, type=click.Path(path_type=Path))
@click.option("--config", "-c", type=click.Path(exists=True, path_type=Path))
@click.option("--freqs-hz", multiple=True, type=float)
@click.option("--frame-width", type=int)
@click.option("--vmin", type=float)
@click.option("--vmax", type=float)
@click.option("--cmap", type=str)
@click.option("--reuse-images/--no-reuse-images", default=None)
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
    """Run Labelme wrapper"""

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
    app = EcholabelApp()

    print(cfg.echogram_cmap)

    builder_params = dict(
        source=source,
        output_dir=app.cache.img_dataset,
        freqs_hz=cfg.channels_frequency_nominal,
        frame_width=cfg.frame_width,
        range_samples_slice=cfg.range_samples_slice,
        vmin=cfg.vmin,
        vmax=cfg.vmax,
        echogram_cmap=cfg.echogram_cmap,
    )

    click.echo(f"Processing {source} → {output}")

    app.run(
        output=output,
        reuse_images=cfg.reuse_images,
        **builder_params,
    )

    click.echo("Complete!")


if __name__ == "__main__":
    cli()
