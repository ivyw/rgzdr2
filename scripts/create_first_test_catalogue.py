"""Creates a subset of the FIRST catalogue for testing."""

import json
from pathlib import Path

from astropy.io import ascii
import click
import numpy as np


@click.command()
@click.option(
    "--first",
    "first_path",
    type=click.Path(resolve_path=True, exists=True, path_type=Path),
    help="FIRST CSV catalogue.",
)
@click.option(
    "--out",
    "output_path",
    type=click.Path(resolve_path=True, exists=False, path_type=Path),
    help="Output path for subset catalogue.",
)
@click.option(
    "--subjects",
    "processed_subjects_path",
    type=click.Path(resolve_path=True, exists=True, path_type=Path),
    help="Processed subjects JSON.",
)
def main(first_path: Path, output_path: Path, processed_subjects_path: Path):
    first = ascii.read(first_path, format="csv")
    with open(processed_subjects_path) as f:
        processed_subjects = json.load(f)

    first_ids = set()
    for subject in processed_subjects:
        bboxes = subject["bboxes"]
        for bbox in bboxes:
            first_ids |= set(bbox["first"])

    first_ids = {f[len("FIRST_") :] for f in first_ids if f.startswith("FIRST")}

    mask = [i in first_ids for i in first["FIRST"]]

    first[mask].write(output_path, format="csv")


if __name__ == "__main__":
    main()
