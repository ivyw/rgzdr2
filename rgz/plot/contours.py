"""Plot contours."""

import json
from pathlib import Path

from rgz import constants
from rgz import subjects


class ContoursNotFoundError(Exception):
    """Raised when the contours list in the raw subject JSON file is empty."""

def get_contours(
    subject: subjects.Subject,
    px_coords: bool = False,
    px_scaling: float = 100 / constants.RADIO_MAX_PX,
    cache: Path = Path("first"),
) -> list[list[tuple]]:
    """Returns the contours of a subject.

    The raw contour data consists of a series of (x, y) coordinate pairs
    relative to the upper-left hand corner of a 132x132 image, where (65, 65)
    represents the centre of the image. If px_coords is True, this function
    applies applies a stretch defined by the px_scaling arg such that
    the coordinates are defined on a

         (px_scaling * RADIO_MAX_PX) x (px_scaling * RADIO_MAX_PX)

    grid. By default, px_scaling is set so that the returned coordinates are
    defined on a 100 x 100 grid to match the dimensions of the FIRST images.

    Note this function expects the contour data to be located in
    cache / f"{subject.id}.json". An exception is raised if the data cannot be
    found.

    Args:
        subject: the subject.
        px_coords: if True, contour coordinates are returned in pixel units
            relative to the upper left-hand corner of the image. If False,
            they are given in degrees as RA/dec pairs.
        px_scaling: Stretch applied to contour coordinates if px_coords is True.
            Ignored if px_coords is False.
        cache: path to contour data.

    Returns:
        A list of lists each representing a radio island, each of which
        contains a list of (x, y) pairs for each contour.

    Raises:
        FileNotFoundError if the file containing the contour data cannot be
        found.

    """
    fname = cache / f"{subject.id}.json"
    try:
        with open(fname) as f:
            islands = json.load(f)["contours"]
    except FileNotFoundError:
        raise FileNotFoundError(
            f"contour data for subject with ID {subject.id} not found!"
        )
    island_contours = []
    for island in islands:
        contours = []
        for contour in island:
            xs = [coord["x"] for coord in contour["arr"]]
            ys = [constants.RADIO_MAX_PX - coord["y"] for coord in contour["arr"]]
            coords = np.stack([xs, ys]).T
            if not px_coords:
                coords = [
                    subjects.transform_coord_radio(
                        coord=c, subject=subject, raw_subject=None, cache=cache
                    )
                    for c in coords
                ]
                coords = [(ra.value, dec.value) for ra, dec in coords]
            else:
                coords = [(x * px_scaling, y * px_scaling) for x, y in coords]
            contours.append(coords)
        island_contours.append(contours)
    return island_contours
