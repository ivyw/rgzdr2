"""This script determines if any subjects have overlapping bboxes."""

import itertools
from pathlib import Path

import click

import rgz.subjects


def overlaps(box_1: rgz.subjects.BBox, box_2: rgz.subjects.BBox) -> bool:
    """Determines if two bounding boxes overlap."""
    xmin_1, ymin_1, xmax_1, ymax_1 = box_1
    xmin_2, ymin_2, xmax_2, ymax_2 = box_2
    return not (
        xmax_1 < xmin_2 or xmax_2 < xmin_1 or ymax_1 < ymin_2 or ymax_2 < ymin_1
    )


def has_overlapping(subject: rgz.subjects.Subject) -> bool:
    """Determines if there are overlapping bboxes in a subject."""
    if len(subject.bboxes) <= 1:
        return False

    for box_1, box_2 in itertools.combinations(subject.bboxes, r=2):
        if overlaps(box_1, box_2):
            return True

    return False


@click.command()
@click.option(
    "--subjects",
    "subjects_path",
    type=click.Path(resolve_path=True, path_type=Path),
    help="Processed subjects JSON.",
    required=True,
)
def main(subjects_path: Path):
    subjects = rgz.subjects.read(subjects_path)
    for subject in subjects:
        if has_overlapping(subject):
            print(subject.zid, "has overlapping bboxes")


if __name__ == "__main__":
    main()
