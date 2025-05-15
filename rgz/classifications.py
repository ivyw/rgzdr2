"""Handles RGZ classifications."""

from collections.abc import Generator
import json
import logging
from typing import Any

import attr
import numpy as np
from astropy.coordinates import SkyCoord

import rgz.subjects


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
    notes: str = attr.ib()


def get_wcs(raw_subject):
    im = rgz.subjects.download_first_image(raw_subject)
    header = im[0].header
    # WCS.dropaxis doesn't seem to work on these images
    # drop these: CTYPE3 CRVAL3 CDELT3 CRPIX3 CROTA3
    for key in ["CTYPE", "CRVAL", "CDELT", "CRPIX", "CROTA"]:
        for i in [3, 4]:
            del header[key + str(i)]
    wcs = WCS(header)
    return wcs


def transform_coord_ir(
    coord: tuple[float, float], raw_subject: dict[str, ...] = None, wcs: ... = None
):
    if not raw_subject and not wcs:
        raise ValueError()
    if raw_subject:
        assert not wcs
        wcs = get_wcs(raw_subject)
    # coord in 424x424 -> 424x424
    coord = coord * 100 / 424
    # flip y axis?
    coord[1] = 100 - coord[1]
    c = wcs.all_pix2world([coord], 0)[0] * u.deg
    return c


def process_classification(
    classification: dict[str, ...], subject: rgz.subjects.Subject, wcs: ..., defer_ir_lookup=False
) -> Classification:
    """Converts a raw classification into a Classification."""
    cid = classification["_id"]["$oid"]
    zid = classification["subjects"][0]["zooniverse_id"]
    if zid != subject.zid:
        raise ValueError("Mismatched subjects.")
    matches = []  # (wise, first)
    notes = []
    for anno in classification["annotations"]:
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
            ir_coord = skcoord.SkyCoord(
                ra=ir_coord[0].value,
                dec=ir_coord[1].value,
                unit=(ir_coord[0].unit, ir_coord[0].unit),
                frame="icrs",
            )
            if not defer_ir_lookup:
                # query the IR
                q = Vizier.query_region(
                    ir_coord, radius=5 * u.arcsec, catalog=["II/328/allwise"]
                )
                try:
                    ir = q[0][0]["AllWISE"]
                except IndexError:
                    ir = f'NOMATCH_J{ir_coord.to_string("hmsdms", sep="").replace(" ", "")}'
            else:
                ir = ir_coord.to_string()
        matches.append((ir, [c for b in boxes for c in subject.bboxes[b]]))
    return Classification(
        cid=cid,
        zid=zid,
        matches=matches,
        username=classification.get("user_name", "NO_USER_NAME"),
        notes=notes,
    )


def classification_to_json_serialisable(classification):
    return {
        "id": classification.cid,
        "zid": classification.zid,
        "matches": [{"ir": ir, "radio": radio} for ir, radio in classification.matches],
        "username": classification.username,
        "notes": classification.notes,
    }


def deserialise_classification(classification):
    return Classification(
        cid=classification["id"],
        zid=classification["zid"],
        username=classification["username"],
        notes=classification["notes"],
        matches=[(m["ir"], m["radio"]) for m in classification["matches"]],
    )