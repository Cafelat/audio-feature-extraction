"""Command-line interface for dataset generator."""

import click


@click.group()
@click.version_option()
def cli() -> None:
    """Audio feature extraction and dataset generator."""
    pass


@cli.command()
def info() -> None:
    """Display system information."""
    click.echo("Dataset Generator v0.1.0")
    click.echo("Ready for implementation.")


if __name__ == "__main__":
    cli()
