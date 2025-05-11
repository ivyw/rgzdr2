"""Handles RGZ subjects."""

import json
import logging
from pathlib import Path
from typing import Any, Generator
import warnings

from astropy.coordinates import SkyCoord
import astroquery
from astroquery.image_cutouts.first import First
from astropy.io import fits
import astropy.units as u
from astroquery.vizier import Vizier
from astropy.wcs import WCS, FITSFixedWarning
import attr
import backoff
import numpy as np
import requests
from tqdm import tqdm


IR_MAX_PX = 424
RADIO_MAX_PX = 132
IM_WIDTH_ARCMIN = 3

# Max number of retries for fetching data from the internet.
MAX_TRIES = 10

logger = logging.getLogger(__name__)


@attr.s
class Subject:
    id = attr.ib()
    zid = attr.ib()
    coords = attr.ib()
    bboxes = attr.ib()


@backoff.on_exception(
    backoff.expo,
    requests.exceptions.ConnectionError,
    max_tries=MAX_TRIES,
)
def fetch_first(coord: SkyCoord, image_size: u.arcmin) -> fits.HDUList:
    """Fetches a FIRST image from the FIRST server."""
    return First.get_images(coord, image_size=image_size)


def download_first_image(raw_subject: dict[str, Any], cache: Path) -> fits.HDUList:
    """Fetches a FIRST image from the FIRST server or cache."""
    coord = raw_subject["coords"]
    coord = SkyCoord(ra=coord[0], dec=coord[1], unit="deg")
    fname = cache / f'{raw_subject["_id"]["$oid"]}.fits'

    try:
        return fits.open(fname)
    except FileNotFoundError:
        im = fetch_first(coord, image_size=3 * u.arcmin)
        im.writeto(fname)
        return im


def transform_coord_radio(
    coord: tuple[int, int],
    raw_subject: dict[str, Any],
    cache: Path,
) -> u.Quantity:
    """Transforms a radio image pixel coordinate.

    Note that this uses the WCS of the subject image, and can be slow!

    TODO: Speed this up by avoiding the image reload whenever possible, e.g. by passing in the image.
    """
    im = download_first_image(raw_subject, cache)
    header = im[0].header
    # WCS.dropaxis doesn't seem to work on these images
    # drop these: CTYPE3 CRVAL3 CDELT3 CRPIX3 CROTA3
    for key in ["CTYPE", "CRVAL", "CDELT", "CRPIX", "CROTA"]:
        for i in [3, 4]:
            del header[key + str(i)]

    with warnings.catch_warnings(action="ignore", category=FITSFixedWarning):
        wcs = WCS(header)
    # coord in 132x132 -> 100x100
    coord = coord * 100 / 132
    # flip y axis
    c = wcs.all_pix2world([coord], 0)[0] * u.deg
    return c


def transform_bbox(bbox, raw_subject, cache):
    bbox_ = bbox
    bbox = np.array(bbox)
    bbox = np.concatenate(
        [
            transform_coord_radio(bbox[:2], raw_subject, cache),
            transform_coord_radio(bbox[2:], raw_subject, cache),
        ]
    )
    return bbox


def get_first_from_bbox(bbox, raw_subject, cache, verbose=False):
    # TODO: might need to flip horizontally or even vertically...
    bbox = transform_bbox(bbox, raw_subject, cache)
    # find the centre
    centre = (bbox[::2].mean(), bbox[1::2].mean())
    # and the width, height
    width = abs(bbox[2] - bbox[0]).to(u.arcsec)
    height = abs(bbox[3] - bbox[1]).to(u.arcsec)

    # round widths and heights up to nearest arcsec plus two
    width = np.ceil(width.to(u.arcsec)) + 2 * u.arcsec
    height = np.ceil(height.to(u.arcsec)) + 2 * u.arcsec

    if verbose:
        print("get_first_from_bbox:", centre, width, height)
    skc = SkyCoord(
        ra=centre[0].value,
        dec=centre[1].value,
        unit=(centre[0].unit, centre[0].unit),
        frame="icrs",
    )
    # Now we can do a VizieR query.
    # TODO: Manually cache this into the cache directory.
    q = Vizier.query_region(
        skc, width=width, height=height, catalog=["VIII/92/first14"]
    )
    try:
        return list(q[0]["FIRST"])
    except IndexError:
        return [f'NOFIRST_J{skc.to_string("hmsdms", sep="").replace(" ", "")}']


def get_bboxes(
    subject: dict[str, Any],
    cache: Path,
) -> tuple[tuple[float, float, float, float], Any]:
    """Fetches the bboxes of a subject from RGZ, caching locally."""
    fname = cache / f'{subject["_id"]["$oid"]}.json'
    try:
        with open(fname) as f:
            js = json.load(f)
    except FileNotFoundError:
        url = subject["location"]["contours"]
        response = requests.get(url)
        if not response.ok:
            if response.status_code == 404:
                raise FileNotFoundError(f"HTTP 404: {url}")
            raise RuntimeError("Error:", response.status_code)
        js = response.json()
        assert abs(js["width"] - 132) <= 1
        with open(fname, "w") as f:
            json.dump(js, f)
    bboxes = []
    for contour in js["contours"]:
        assert contour[0]["k"] == 0
        bboxes.append(tuple([round(c, 1) for c in contour[0]["bbox"]]))
    return tuple(bboxes)


def subject_to_json_serialisable(subject: Subject) -> dict[str, Any]:
    return {
        "id": subject.id,
        "zid": subject.zid,
        "coords": subject.coords,
        "bboxes": [{"bbox": list(k), "first": v} for k, v in subject.bboxes.items()],
    }


def process_subject(
    raw_subject: dict[str, Any], cache: Path, verbose: bool = False
) -> Subject:
    sid = raw_subject["_id"]["$oid"]
    zid = raw_subject["zooniverse_id"]
    bboxes = get_bboxes(raw_subject, cache)
    bbox_to_firsts = {}
    for bbox in bboxes:
        firsts = get_first_from_bbox(bbox, raw_subject, cache, verbose=verbose)
        bbox_to_firsts[bbox] = firsts
    s = Subject(id=sid, zid=zid, coords=raw_subject["coords"], bboxes=bbox_to_firsts)
    return s


def process(subjects_path: Path, cache: Path, output_path: Path) -> Generator[Subject]:
    """Process subjects from raw to reduced JSON."""
    subjects = []
    # Get subject count for progress bar.
    with open(subjects_path, encoding="utf-8") as f:
        n_subjects = len(f.readlines())
    with open(subjects_path, encoding="utf-8") as f:
        # Each row is a JSON document.
        for row in tqdm(f, desc="Processing subjects...", total=n_subjects):
            try:
                subjects.append(process_subject(json.loads(row), cache))
            except FileNotFoundError as e:
                logger.warning(e)
                continue
    json_subjects = []
    for subject in tqdm(subjects, desc="Serialising subjects..."):
        json_subjects.append(subject_to_json_serialisable(subject))
    with open(output_path, "w") as f:
        json.dump(json_subjects, f)


def deserialise_subject(subject):
    return Subject(
        subject["id"],
        subject["zid"],
        subject["coords"],
        {tuple(b["bbox"]): b["first"] for b in subject["bboxes"]},
    )
