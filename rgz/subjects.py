"""Handles RGZ subjects."""

import json
import logging
from pathlib import Path
from typing import Self, Iterable

from astropy.io import fits
from astropy.wcs import WCS
import attr
from tqdm import tqdm

from rgz import bboxes
from rgz import first
from rgz import radio_islands
from rgz import rgz

# Indent of output JSON files.
_JSON_INDENT = 2


logger = logging.getLogger(__name__)

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
    radio_islands: list = attr.ib()  # TODO why doesn't list[radio_islands.RadioIsland] = attr.ib() work here>?
    wcs: WCS = attr.ib()

    def to_json(self) -> rgz.JSON:
        """Converts a Subject into a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "zid": self.zid,
            "coords": self.coords,
            "radio_islands": [ri.to_json() for ri in self.radio_islands],
            "wcs": self.wcs.to_header_string(),
        }

    @classmethod
    def from_json(cls, subject: rgz.JSON) -> Self:
        """Reads a Subject from JSON."""
        return cls(
            subject["id"],
            subject["zid"],
            subject["coords"],
            [radio_islands.RadioIsland.from_json(ri) for ri in subject["radio_islands"]],
            WCS(subject["wcs"]),
        )
    

def process_subject(
    raw_subject: rgz.JSON,
    cache: Path,
    first_tree: first.FIRSTTree,
) -> Subject:
    """Reduces a JSON subject into a nice, value-added format."""
    sid = raw_subject["_id"]["$oid"]
    zid = raw_subject["zooniverse_id"]

    first.download_contour_data(raw_subject, cache)
    first.download_first_image(raw_subject, cache)
    wcs = first.get_first_wcs(sid, cache)
    bbox_list = bboxes.get_bboxes(sid, wcs=wcs, cache=cache)
    rgzbbox_list = bboxes.get_bboxes(sid, wcs=wcs, cache=cache, units="RGZ")
    contours_list = first.get_contours(sid, wcs=wcs, cache=cache)
    
    risland_list = []
    assert len(bbox_list) == len(rgzbbox_list)
    assert len(contours_list) == len(rgzbbox_list)
    for bbox, rgzbbox, contours in zip(bbox_list, rgzbbox_list, contours_list):
        # TODO(hzovaro): implement some kind of "invalid bbox" flag for dodgy
        # bboxes.
        risland = radio_islands.RadioIsland(
            bbox=bbox,
            rgzbbox=rgzbbox,
            contours=contours[0], # For now just store the zeroth contour
            first_tree=first_tree,
        )
        risland_list.append(risland)
    return Subject(
        id=sid, zid=zid, coords=raw_subject["coords"], 
        radio_islands=risland_list, 
        wcs=wcs
    )


def process(subjects_path: Path, cache: Path, output_path: Path):
    """Processes subjects from raw to reduced JSON."""
    first_catalogue = first.fetch_first_catalogue_from_server_or_cache(cache)
    first_tree = first.build_first_tree(first_catalogue)

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
