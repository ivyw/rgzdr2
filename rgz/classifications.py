"""Handles RGZ classifications."""

from collections.abc import Generator
import json
import logging
from pathlib import Path
from typing import Any

import attr
import numpy as np
import numpy.typing as npt
from astropy.coordinates import SkyCoord
import astropy.wcs
from astroquery.vizier import Vizier

from rgz import constants
from rgz import rgz
from rgz import subjects
from rgz import units as u


logger = logging.getLogger(__name__)

type BBox = tuple[float, float, float, float]
type JSON = dict[str, Any]


def get_classifications(path: Path) -> Generator[JSON]:
    """Yields classifications from RGZ.

    Args:
        path: Path to RGZ classifications JSON dumped from MongoDB.
            Each line should be a separate JSON document.

    Yields:
        Classification JSON documents.

    Raises:
        ValueError if a classification document is invalid.
    """
    with open(path, encoding="utf-8") as f:
        # Each row is a JSON document.
        for row in f:
            js = json.loads(row)
            for anno in js["annotations"]:
                if "radio" in anno:
                    break
            else:
                raise ValueError(f"Classification has no annotations: {js}")
            yield js


@attr.s
class Classification:
    """A single RGZ classification of a subject by a citizen scientist.

    Attributes:
        cid: Classification MongoDB ID.
        zid: Zooniverse ID of the subject being classified.
        matches: Mapping from IR host name to radio component names.
        username: Username of the citizen scientist.
        notes: Extra tags assigned to this classification, possibly
            during reduction.
    """

    cid: str = attr.ib()
    zid: str = attr.ib()
    matches: list[tuple[str, set[str]]] = attr.ib()
    username: str | None = attr.ib()
    notes: list[str] = attr.ib()


def transform_coord_ir(
    coord: npt.NDArray[np.float64],
    raw_subject: JSON | None = None,
    cache: Path | None = None,
    wcs: astropy.wcs.WCS | None = None,
) -> u.Quantity[u.deg, u.deg]:
    """Transforms a coordinate from raw classifications to physical.

    You can pass a subject and cache, or a WCS.

    Args:
        coord: Coord to transform, in px coordinates (0, IR_MAX_PX).
        raw_subject: Raw JSON subject.
        cache: Path to the subject data location.
        wcs: WCS of the subject being classified.

    Returns:
        Transformed coordinate RA/dec in deg.
    """
    # TODO: We should use subjects here, not raw subjects.
    if not raw_subject and not wcs:
        raise ValueError()
    if raw_subject and not cache:
        raise ValueError()
    if raw_subject:
        assert cache
        assert not wcs
        im = subjects.fetch_first_from_server_or_cache(raw_subject, cache)
        wcs = rgz.get_wcs(im, cache)
    assert wcs
    # Coord in 424x424 -> 100x100
    px_coord = np.array(coord) * 100 / constants.IR_MAX_PX
    # Flip y axis.
    px_coord[1] = 100 - px_coord[1]
    return wcs.all_pix2world([px_coord], 0)[0] * u.deg


def process_classification(
    raw_classification: JSON,
    subject: subjects.Subject,
    wcs: astropy.wcs.WCS,
    defer_ir_lookup: bool = True,
) -> Classification:
    """Reduces a JSON classification into a nice, value-added format.

    Args:
        raw_classification: JSON classification.
        subject: Subject being classified.
        wcs: WCS of the subject image.
        defer_ir_lookup: Whether to defer the IR lookup for later, or
            perform the lookup in Vizier now (slow!).

    Returns:
        Reduced classification.
    """
    cid = raw_classification["_id"]["$oid"]
    zid = raw_classification["subjects"][0]["zooniverse_id"]
    if zid != subject.zid:
        raise ValueError("Mismatched subjects.")
    matches: list[tuple[str, set[str]]] = []  # (wise, first)
    notes: list[str] = []
    for anno in raw_classification["annotations"]:
        if "radio" not in anno:
            continue
        boxes: set[BBox] = set()
        if anno["radio"] == "No Contours":
            # ?????? ignore this
            continue
        for radio in anno["radio"].values():
            box = (
                round(float(radio["xmax"]), 1),
                round(float(radio["ymax"]), 1),
                round(float(radio["xmin"]), 1),
                round(float(radio["ymin"]), 1),
            )
            boxes.add(box)

        if anno["ir"] == "No Sources":
            ir = "NOSOURCE"
        else:
            if len(anno["ir"]) != 1:
                notes.append("MULTISOURCE")
            ir_coord = transform_coord_ir(np.array(
                [float(anno["ir"]["0"]["x"]),
                 float(anno["ir"]["0"]["y"])]), wcs=wcs)
            ir_ra, ir_dec = ir_coord
            ir_coord = SkyCoord(
                ra=ir_ra.value,
                dec=ir_dec.value,
                unit=(ir_ra.unit, ir_dec.unit),
                frame="icrs",
            )
            if not defer_ir_lookup:
                # Query the IR in Vizier.
                q = Vizier.query_region(  # type: ignore[reportAttributeAccessIssue]
                    ir_coord, radius=5 * u.arcsec, catalog=["II/328/allwise"]
                )
                try:
                    ir = q[0][0]["AllWISE"]
                except IndexError:
                    coord_str = rgz.coord_to_string(ir_coord)
                    ir = f'NOMATCH_J{coord_str.replace(" ", "")}'
            else:
                ir = rgz.coord_to_string(ir_coord)
        matches.append((ir, {c for b in boxes for c in subject.bboxes[b]}))
    return Classification(
        cid=cid,
        zid=zid,
        matches=matches,
        username=raw_classification.get("user_name", None),
        notes=notes,
    )


def classification_to_json_serialisable(classification: Classification) -> JSON:
    """Convert a Classification into a JSON-serialisable dictionary."""
    return {
        "id": classification.cid,
        "zid": classification.zid,
        "matches": [
            {"ir": ir, "radio": list(radio)} for ir, radio in classification.matches
        ],
        "username": classification.username or constants.ANONYMOUS_NAME,
        "notes": classification.notes,
    }


def deserialise_classification(classification: JSON) -> Classification:
    """Read a Classification from a JSON dict."""
    return Classification(
        cid=classification["id"],
        zid=classification["zid"],
        username=classification["username"] or None,
        notes=classification["notes"],
        matches=[(m["ir"], set(m["radio"])) for m in classification["matches"]],
    )
