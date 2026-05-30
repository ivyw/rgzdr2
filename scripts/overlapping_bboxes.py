"""This script determines if any subjects have overlapping bboxes."""

import itertools
from pathlib import Path

import click

import rgz.subjects


def inside(point: tuple[float, float], box: rgz.subjects.BBox) -> bool:
    """Determines if a point is inside a bounding box."""
    # TODO(hzovaro): this *should* work with RA/dec instead of pixel coords
    # but will need to update the input args.
    xmin, ymin, xmax, ymax = box
    x, y = point
    return xmin <= x <= xmax and ymin <= y <= ymax


def overlaps(box_1: rgz.subjects.BBox, box_2: rgz.subjects.BBox) -> bool:
    """Determines if two bounding boxes overlap."""
    xmin_1, ymin_1, _, _ = box_1
    xmin_2, ymin_2, _, _ = box_2
    p1 = (xmin_1, ymin_1)
    p2 = (xmin_2, ymin_2)
    return inside(p1, box_2) or inside(p2, box_1)


def has_overlapping(subject: rgz.subjects.Subject) -> bool:
    """Determines if there are overlapping bboxes in a subject."""
    if len(subject.bboxes) <= 1:
        return False

    # TODO(hzovaro): replace with [ri.bbox for ri in subject.radio_islands]
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
