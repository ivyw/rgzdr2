"""Handles RGZ subjects."""

from collections.abc import Sequence
import json
import logging
from pathlib import Path
from typing import Self, Iterable

from astropy.coordinates import SkyCoord
from astroquery.image_cutouts.first import First
import astropy.io.ascii
from astropy.io import fits
import astropy.table
from astropy.units import Quantity
from astroquery.vizier import Vizier
from astropy.wcs import WCS
import attr
import backoff
import numpy as np
import numpy.typing as npt
import requests
from tqdm import tqdm

from rgz import constants
from rgz import radioislands  # TODO(hzovaro): consider renaming to first or something similar.
from rgz import rgz
from rgz import units as u

# Indent of output JSON files.
_JSON_INDENT = 2


logger = logging.getLogger(__name__)

type BBox = tuple[float, float, float, float]  # xmin, ymin, xmax, ymax
type HDU = fits.hdu.base.ExtensionHDU
type ZooniverseID = str


@attr.s
class Subject:
    """A Radio Galaxy Zoo subject.

    Attributes:
        id: RGZ MongoDB ID.
        zid: Zooniverse ID.
        coords: Central right ascension, declination, both in degrees.
        bboxes: Bounding boxes for the radio islands in the subject.
                This is defined per Radio Galaxy Zoo, so
                (xmin, ymin, xmax, ymax).
        wcs: astropy World Coordinate System extracted from FIRST image centred
                on this subject.
    """

    # TODO(hzovaro): should we also store the WISE wcs?

    id: str = attr.ib()
    zid: ZooniverseID = attr.ib()
    coords: tuple[float, float] = attr.ib()
    bboxes: dict[BBox, list[radioislands.FIRSTID]] = attr.ib()
    wcs: WCS = attr.ib()

    def to_json(self) -> rgz.JSON:
        """Converts a Subject into a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "zid": self.zid,
            "coords": self.coords,
            "wcs": self.wcs.to_header_string(),
            "bboxes": [{"bbox": list(k), "first": v} for k, v in self.bboxes.items()],
        }

    @classmethod
    def from_json(cls, subject: rgz.JSON) -> Self:
        """Reads a Subject from JSON."""
        return cls(
            subject["id"],
            subject["zid"],
            subject["coords"],
            {tuple(b["bbox"]): b["first"] for b in subject["bboxes"]},
            WCS(subject["wcs"]),
        )


def get_bboxes(
    raw_subject: rgz.JSON,
    cache: Path,
) -> Sequence[BBox]:
    """Fetches the bboxes of a subject from RGZ, caching locally."""
    fname = cache / f'{raw_subject["_id"]["$oid"]}.json'
    try:
        with open(fname) as f:
            js = json.load(f)
    except FileNotFoundError:
        url = raw_subject["location"]["contours"]
        response = requests.get(url)
        if not response.ok:
            if response.status_code == 404:
                raise FileNotFoundError(f"HTTP 404: {url}")
            raise RuntimeError("Error:", response.status_code)
        js = response.json()
        assert abs(js["width"] - constants.RADIO_MAX_PX) <= 1
        with open(fname, "w") as f:
            # Don't indent here to keep the filesize down.
            # These don't need to be human-readable.
            json.dump(js, f)
    bboxes = []
    for contour in js["contours"]:
        assert contour[0]["k"] == 0
        # Bboxes are 1-indexed...
        bboxes.append(tuple([round(c, 1) - 1 for c in contour[0]["bbox"]]))
    return tuple(bboxes)


def process_subject(
    raw_subject: rgz.JSON,
    cache: Path,
    first_tree: radioislands.FIRSTTree,
) -> Subject:
    """Reduces a JSON subject into a nice, value-added format."""
    sid = raw_subject["_id"]["$oid"]
    zid = raw_subject["zooniverse_id"]
    bboxes = get_bboxes(raw_subject, cache)
    # contours = get_contours(...)
    # TODO(hzovaro): also, get contours from the raw_subject. 
    bbox_to_firsts = {}

    with radioislands.fetch_first_image_from_server_or_cache(
        raw_subject=raw_subject, cache=cache
    ) as im:
        wcs = rgz.get_wcs(im)

    # TODO(hzovaro): make radio islands here. 
    # risland_list = []
    # for bbox in bboxes:
    #     risland = radioislands.RadioIsland(
    #         bbox_px=bbox,
    #     )
    #     risland_list.append(radio_island)

    # TODO(hzovaro): move the below to the constructor for RadioIsland.
    for bbox in bboxes:
        firsts = radioislands.get_first_from_bbox(bbox, wcs, first_tree)
        bbox_to_firsts[bbox] = firsts

    return Subject(
        id=sid, zid=zid, coords=raw_subject["coords"], bboxes=bbox_to_firsts, wcs=wcs
    )


def process(subjects_path: Path, cache: Path, output_path: Path):
    """Processes subjects from raw to reduced JSON."""
    first_catalogue = radioislands.fetch_first_catalogue_from_server_or_cache(cache)
    first_tree = radioislands.build_first_tree(first_catalogue)

    subjects = []
    failures = set()
    # Get subject count for progress bar.
    with open(subjects_path, encoding="utf-8") as f:
        n_subjects = len(f.readlines())
    with open(subjects_path, encoding="utf-8") as f:
        # Each row is a JSON document.
        for row in tqdm(f, desc="Processing subjects...", total=n_subjects):
            raw_subject = json.loads(row)
            try:
                subjects.append(process_subject(raw_subject, cache, first_tree))
            except FileNotFoundError as e:
                failures.add(raw_subject["zooniverse_id"])
                continue
            except Exception as e:
                logger.warning(
                    "Error processing {}: {}".format(raw_subject["zooniverse_id"], e)
                )
                failures.add(raw_subject["zooniverse_id"])
    json_subjects = []
    for subject in tqdm(subjects, desc="Serialising subjects..."):
        json_subjects.append(subject.to_json())
    with open(output_path, "w") as f:
        json.dump(json_subjects, f, indent=_JSON_INDENT)
    if failures:
        print("Failures:")
        for subject in failures:
            print(subject)


def read(subjects_path: Path) -> Iterable[Subject]:
    """Reads subjects from a JSON file."""
    with open(subjects_path) as f:
        json_subjects = json.load(f)
    for s in json_subjects:
        yield Subject.from_json(s)
