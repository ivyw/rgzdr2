"""Handles RGZ subjects."""

import json
import logging
from pathlib import Path
from typing import Self, Iterable

from astroquery.image_cutouts.first import First
from astropy.io import fits
from astroquery.vizier import Vizier
from astropy.wcs import WCS
import attr
from tqdm import tqdm

from rgz import radioislands  # TODO(hzovaro): consider renaming to first or something similar.
from rgz import rgz
from rgz import units as u

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
    radioislands: list = attr.ib()  # TODO why doesn't list[radioislands.RadioIsland] = attr.ib() work here>?
    wcs: WCS = attr.ib()

    def to_json(self) -> rgz.JSON:
        """Converts a Subject into a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "zid": self.zid,
            "coords": self.coords,
            "radioislands": [ri.to_json() for ri in self.radioislands],
            "wcs": self.wcs.to_header_string(),
        }

    @classmethod
    def from_json(cls, subject: rgz.JSON) -> Self:
        """Reads a Subject from JSON."""
        return cls(
            subject["id"],
            subject["zid"],
            subject["coords"],
            # radioislands.from_dict(ri) needs to retrn a radioisland, because 
            # the below line needs to be a list of radioislands.
            [radioislands.RadioIsland.from_json(ri) for ri in subject["radioislands"]],
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
    wcs = radioislands.get_wcs(sid, cache)
    
    # NOTE: both of these are in physical units already.
    bboxes = radioislands.get_bboxes(sid, wcs=wcs, cache=cache)
    rgzbboxes = radioislands.__get_rgzbboxes(sid, wcs=wcs, cache=cache)
    # TODO(hzovaro) there is a bug here - get_contours is only returning a list of len 1 
    # when it should be 2 for subject 52af81007aa69f059a001a84
    contours_list = radioislands.get_contours(sid, wcs=wcs, cache=cache)
    
    # TODO(hzovaro): get_wcs is specific to FIRST so this should be 
    # where the rest of the FIRST-related utilities are.

    risland_list = []
    # TODO(hzovaro): change back to include contours once get_contours is fixed
    # for bbox, rgzbbox, contours in zip(bboxes, rgzbboxes, contours_list):
    for bbox, rgzbbox in zip(bboxes, rgzbboxes):
        risland = radioislands.RadioIsland(
            bbox=bbox,
            rgzbbox=rgzbbox,
            # contours=contours[0], # For now just store the zeroth contour
            contours=[[1,2], [3,4], [5,6]],  # TODO(hzovaro) change back
            first_tree=first_tree,
        )
        risland_list.append(risland)
    # if sid == "52af81007aa69f059a001a84":
    #     # OLD:
    #     # [{'bbox': [70.3, 69.2, 61.4, 61.7], 'first': ['FIRST_J100345.7+102837']}, 
    #     #  {'bbox': [93.0, 113.5, 83.4, 103.5], 'first': ['FIRST_J100343.5+102737']}]
    #     breakpoint()
    return Subject(
        id=sid, zid=zid, coords=raw_subject["coords"], 
        radioislands=risland_list, 
        wcs=wcs
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
