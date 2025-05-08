"""Processes Radio Galaxy Zoo raw data."""

from pathlib import Path

import click

import rgz.subjects


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


if __name__ == "__main__":
    cli()
