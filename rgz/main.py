"""Processes Radio Galaxy Zoo raw data."""

import logging
from pathlib import Path

import click

import rgz.subjects

logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--in",
    "in_",
    type=click.Path(resolve_path=True, path_type=Path),
    help="Contains one JSON subject per line, exported from RGZ.",
)
@click.option(
    "--out",
    type=click.Path(resolve_path=True, path_type=Path),
    help="JSON file to write reduced subjects to.",
)
@click.option(
    "--cache",
    type=click.Path(
        resolve_path=True, dir_okay=True, file_okay=False, exists=True, path_type=Path
    ),
    help="Where to download files to.",
)
def subjects(in_: Path, out: Path, cache: Path):
    """Processes RGZ subjects."""
    rgz.subjects.process(in_, cache, out)


@cli.command()
@click.option(
    "--in",
    "in_",
    type=click.Path(resolve_path=True, path_type=Path),
    help="Contains one JSON classification per line, exported from RGZ.",
)
@click.option(
    "--out",
    type=click.Path(resolve_path=True, path_type=Path),
    help="JSON file to write reduced classifications to.",
)
@click.option(
    "--cache",
    type=click.Path(
        resolve_path=True, dir_okay=True, file_okay=False, exists=True, path_type=Path
    ),
    help="Where to download files to.",
)
def classifications(in_: Path, out: Path, cache: Path):
    """Processes RGZ classifications."""
    rgz.classifications.process(in_, cache, out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
