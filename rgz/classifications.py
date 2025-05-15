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

type JSON = dict[str, Any]


def get_classifications(path="radio_classifications.json") -> Generator[JSON]:
    """Yields classifications from RGZ."""
    with open(path, encoding="utf-8") as f:
        # Each row is a JSON document.
        for row in f:
            js = json.loads(row)
            # if js['subject_ids'][0]['$oid'] not in all_subjects:
            #     # sometimes not true??
            #     continue
            for anno in js["annotations"]:
                if "radio" in anno:
                    break
            else:
                # no annotations?
                continue
            yield js


@attr.s
class Classification:
    """A single RGZ classification of a subject by a citizen scientist."""

    cid: str = attr.ib()
    zid: str = attr.ib()
    matches: list[tuple[str, set[str]]] = attr.ib()
    username: str = attr.ib()
    notes: list[str] = attr.ib()


def transform_coord_ir(
    coord: npt.NDArray[np.float64],
    raw_subject: JSON | None = None,
    wcs: astropy.wcs.WCS | None = None,
    cache: Path | None = None,
):
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
    # coord in 424x424 -> 424x424
    px_coord = np.array(coord) * 100 / constants.IR_MAX_PX
    # flip y axis?
    px_coord[1] = 100 - px_coord[1]
    return wcs.all_pix2world([px_coord], 0)[0] * u.deg


def process_classification(
    raw_classification: JSON,
    subject: subjects.Subject,
    wcs: astropy.wcs.WCS,
    defer_ir_lookup: bool = False,
) -> Classification:
    """Converts a raw classification into a Classification."""
    cid = raw_classification["_id"]["$oid"]
    zid = raw_classification["subjects"][0]["zooniverse_id"]
    if zid != subject.zid:
        raise ValueError("Mismatched subjects.")
    matches = []  # (wise, first)
    notes: list[str] = []
    for anno in raw_classification["annotations"]:
        if "radio" not in anno:
            continue
        boxes = set()
        if anno["radio"] == "No Contours":
            # ?????? ignore this
            continue
        for radio in anno["radio"].values():
            box = tuple(
                round(float(radio[corner]), 1)
                for corner in ["xmax", "ymax", "xmin", "ymin"]
            )
            boxes.add(box)

        if anno["ir"] == "No Sources":
            ir = "NOSOURCE"
        else:
            if len(anno["ir"]) != 1:
                notes.append("MULTISOURCE")
            ir_coord = anno["ir"]["0"]["x"], anno["ir"]["0"]["y"]
            ir_coord = np.array([float(i) for i in ir_coord])
            ir_coord = transform_coord_ir(ir_coord, wcs=wcs)
            ir_coord = SkyCoord(
                ra=ir_coord[0].value,
                dec=ir_coord[1].value,
                unit=(ir_coord[0].unit, ir_coord[0].unit),
                frame="icrs",
            )
            if not defer_ir_lookup:
                # query the IR
                q = Vizier.query_region(  # type: ignore[reportAttributeAccessIssue]
                    ir_coord, radius=5 * u.arcsec, catalog=["II/328/allwise"]
                )
                try:
                    ir = q[0][0]["AllWISE"]
                except IndexError:
                    coord_str = rgz.coord_to_string(ir_coord)
                    ir = f'NOMATCH_J{coord_str.replace(" ", "")}'
            else:
                ir = ir_coord.to_string()
        matches.append((ir, [c for b in boxes for c in subject.bboxes[b]]))
    return Classification(
        cid=cid,
        zid=zid,
        matches=matches,
        username=raw_classification.get("user_name", constants.NO_USER_NAME),
        notes=notes,
    )


def classification_to_json_serialisable(classification):
    return {
        "id": classification.cid,
        "zid": classification.zid,
        "matches": [{"ir": ir, "radio": list(radio)} for ir, radio in classification.matches],
        "username": classification.username,
        "notes": classification.notes,
    }


def deserialise_classification(classification):
    return Classification(
        cid=classification["id"],
        zid=classification["zid"],
        username=classification["username"],
        notes=classification["notes"],
        matches=[(m["ir"], set(m["radio"])) for m in classification["matches"]],
    )
