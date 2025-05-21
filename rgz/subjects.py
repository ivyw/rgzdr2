"""Handles RGZ subjects."""

from collections.abc import Sequence
import json
import logging
from pathlib import Path
from typing import Any

from astropy.coordinates import SkyCoord
from astroquery.image_cutouts.first import First
from astropy.io import fits
from astropy.units import Quantity
from astroquery.vizier import Vizier
import attr
import backoff
import numpy as np
import numpy.typing as npt
import requests
from tqdm import tqdm

from rgz import constants
from rgz import rgz
from rgz import units as u

# Max number of retries for fetching data from the internet.
MAX_TRIES = 10

logger = logging.getLogger(__name__)


type BBox = tuple[float, float, float, float]
type JSON = dict[str, Any]
type HDU = fits.hdu.base.ExtensionHDU


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
    """

    id: str = attr.ib()
    zid: str = attr.ib()
    coords: tuple[float, float] = attr.ib()
    bboxes: dict[BBox, list[str]] = attr.ib()


@backoff.on_exception(
    backoff.expo,
    requests.exceptions.ConnectionError,
    max_tries=MAX_TRIES,
)
def download_first(coord: SkyCoord, image_size: Quantity[u.arcmin]) -> fits.HDUList:
    """Downloads a FIRST image from the FIRST server."""
    ims = First.get_images(coord, image_size=image_size)
    if isinstance(ims, fits.HDUList):
        return ims
    # Technically allowed by documentation, but I don't expect it to happen
    # with the files we're opening (i.e. FIRST survey files).
    raise TypeError(f"Expected HDUList; got {type(ims)}")


def read_subject_image_from_file(subject: Subject, cache: Path) -> fits.HDUList:
    """Reads a FIRST image from the cache."""
    fname = cache / f"{subject.id}.fits"
    return fits.open(fname)


def fetch_first_from_server_or_cache(
    raw_subject: JSON,
    cache: Path,
) -> fits.HDUList:
    """Fetches a FIRST image from the FIRST server or cache."""
    coord = raw_subject["coords"]
    coord = SkyCoord(ra=coord[0], dec=coord[1], unit="deg")
    fname = cache / f'{raw_subject["_id"]["$oid"]}.fits'
    try:
        return fits.open(fname)
    except FileNotFoundError:
        logger.debug("Cache miss; downloading %s", fname)
        im = download_first(coord, image_size=3 * u.arcmin)
        im.writeto(fname)
        return im


def transform_coord_radio(
    coord: npt.NDArray[np.float64],
    raw_subject: JSON,
    cache: Path,
) -> Quantity[u.deg, u.deg]:
    """Transforms a radio image pixel coordinate into RA/dec.

    Note that this uses the WCS of the subject image, and can be slow!

    TODO: Speed this up by avoiding the image reload whenever possible, e.g. by passing in the image.
    """
    with fetch_first_from_server_or_cache(raw_subject, cache) as im:
        wcs = rgz.get_wcs(im)

    # Coord in 132x132 -> 100x100.
    coord = coord * 100 / constants.RADIO_MAX_PX

    return wcs.all_pix2world([coord], 0)[0] * u.deg


def transform_bbox_px_to_phys(
    px_bbox: BBox, raw_subject: JSON, cache: Path
) -> npt.NDArray[np.float64]:
    """Transforms a bbox from pixel coordinates to RA/dec."""
    phys_bbox = np.array(px_bbox)
    return np.concatenate(
        [
            transform_coord_radio(phys_bbox[:2], raw_subject, cache),
            transform_coord_radio(phys_bbox[2:], raw_subject, cache),
        ]
    )


def get_first_from_bbox(
    px_bbox: BBox,
    raw_subject: JSON,
    cache: Path,
) -> list[str]:
    """Finds FIRST components within a bounding box."""
    # TODO: Might need to flip horizontally or even vertically...
    phys_bbox = transform_bbox_px_to_phys(px_bbox, raw_subject, cache)
    # Find the centre...
    centre = (phys_bbox[::2].mean(), phys_bbox[1::2].mean())
    # ...and the width and height.
    width = abs(phys_bbox[2] - phys_bbox[0]).to(u.arcsec)
    height = abs(phys_bbox[3] - phys_bbox[1]).to(u.arcsec)

    # Round widths and heights up to nearest arcsec plus two.
    width = np.ceil(width.to(u.arcsec)) + 2 * u.arcsec
    height = np.ceil(height.to(u.arcsec)) + 2 * u.arcsec

    logger.debug("get_first_from_bbox: %s %s %s", centre, width, height)
    skc = SkyCoord(
        ra=centre[0].value,
        dec=centre[1].value,
        unit=(centre[0].unit, centre[0].unit),
        frame="icrs",
    )
    # Now we can do a VizieR query.
    # TODO: Manually cache this into the cache directory.
    q = Vizier.query_region(  # type: ignore[reportAttributeAccessIssue]
        skc, width=width, height=height, catalog=["VIII/92/first14"]
    )
    try:
        return [f"FIRST_{name}" for name in q[0]["FIRST"]]
    except IndexError:
        coord_str = rgz.coord_to_string(skc)
        return [f'NOFIRST_J{coord_str.replace(" ", "")}']


def get_bboxes(
    raw_subject: JSON,
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
        assert abs(js["width"] - 132) <= 1
        with open(fname, "w") as f:
            json.dump(js, f)
    bboxes = []
    for contour in js["contours"]:
        assert contour[0]["k"] == 0
        bboxes.append(tuple([round(c, 1) for c in contour[0]["bbox"]]))
    return tuple(bboxes)


def subject_to_json_serialisable(subject: Subject) -> JSON:
    """Converts a Subject into a JSON-compatible dictionary."""
    return {
        "id": subject.id,
        "zid": subject.zid,
        "coords": subject.coords,
        "bboxes": [{"bbox": list(k), "first": v} for k, v in subject.bboxes.items()],
    }


def process_subject(
    raw_subject: JSON,
    cache: Path,
) -> Subject:
    """Reduces a JSON subject into a nice, value-added format."""
    sid = raw_subject["_id"]["$oid"]
    zid = raw_subject["zooniverse_id"]
    bboxes = get_bboxes(raw_subject, cache)
    bbox_to_firsts = {}
    for bbox in bboxes:
        firsts = get_first_from_bbox(bbox, raw_subject, cache)
        bbox_to_firsts[bbox] = firsts
    s = Subject(id=sid, zid=zid, coords=raw_subject["coords"], bboxes=bbox_to_firsts)
    return s


def process(subjects_path: Path, cache: Path, output_path: Path):
    """Processes subjects from raw to reduced JSON."""
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


def deserialise_subject(subject: dict[str, Any]) -> Subject:
    """Reads a Subject from JSON."""
    return Subject(
        subject["id"],
        subject["zid"],
        subject["coords"],
        {tuple(b["bbox"]): b["first"] for b in subject["bboxes"]},
    )
