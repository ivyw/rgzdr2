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


def process_subject(
    raw_subject: rgz.JSON,
    cache: Path,
    first_tree: radioislands.FIRSTTree,
) -> Subject:
    """Reduces a JSON subject into a nice, value-added format."""
    sid = raw_subject["_id"]["$oid"]
    zid = raw_subject["zooniverse_id"]

    # TODO(hzovaro): Everything below is defined on a per-subject basis,
    # not a per-radio-island basis, so maybe it would be better 
    # to store these in subjects.
    radioislands.download_contour_data(raw_subject, cache)
    radioislands.download_first_image(raw_subject, cache)
    bboxes = radioislands.get_bboxes(sid, cache)
    contours = radioislands.get_contours(sid, cache)
    # TODO(hzovaro): get_wcs is specific to FIRST so this should be 
    # where the rest of the FIRST-related utilities are.
    wcs = radioislands.get_wcs(sid, cache)

    # TODO(hzovaro): make radio islands here. 
    # risland_list = []
    # for bbox in bboxes:
    #     risland = radioislands.RadioIsland(
    #         bbox_px=bbox,
    #     )
    #     risland_list.append(radio_island)

    # TODO(hzovaro): move the below to the constructor for RadioIsland.
    bbox_to_firsts = {}
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
