"""Processes Radio Galaxy Zoo raw data."""

import click


@click.group()
def cli():
    pass


@cli.command()
def subjects():
    """Processes RGZ subjects."""


if __name__ == "__main__":
    cli()
