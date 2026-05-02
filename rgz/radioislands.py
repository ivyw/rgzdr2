# """Handles RGZ radio islands."""
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

from rgz import constants
from rgz import rgz
from rgz import units as u

# Max number of retries for fetching data from the internet.
MAX_TRIES = 10


logger = logging.getLogger(__name__)


# TODO(hzovaro) to avoid circular imports, move this somewhere else
# TODO(hzovaro) replace with DataClass and update everything that uses bboxes
type BBox = tuple[float, float, float, float]  # xmin, ymin, xmax, ymax
type HDU = fits.hdu.base.ExtensionHDU
type FIRSTID = str


# def get_phys_bbox(self, bbox_px: BBox, wcs: ...) -> BBox_phys:
#     """Returns a BBox_phys corresponding to a bbox_px according to a given World Coordinate System."""

type FIRSTTree = tuple[npt.NDArray[np.float64], list[str]]


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


def fetch_first_image_from_server_or_cache(
    raw_subject: rgz.JSON | None,
    cache: Path,
) -> fits.HDUList:
    """Fetches a FIRST image from the FIRST server or cache given a subject.
    The subject can be specified either via a raw subject (in JSON format) or a
    processed subject (as a Subject object).

    This function looks for an existing FIRST image in the cache directory
    assuming the filename
        cache / f'{raw_subject["_id"]["$oid"]}.fits'
    if a raw subject is specified, or
        cache / f"{subject.id}.fits"
    if a processed subject is specified. If no image can be found, it is
    downloaded using astroquery.image_cutouts.first.First and saved using the
    above filename. The image is returned in the form of an astropy HDUList.

    Args:
        raw_subject: desired raw subject in JSON format. Defaults to None. Must
            be specified if subject is None.
        subject: desired Subject. Defaults to None. Must be specified if
            raw_subject is None.
        cache: path to FIRST images.

    Returns:
        HDUList containing the FIRST image.

    Raises:
        ValueError if neither a raw subject or a subject are specified, or if
        they are both specified.

    """
    # TODO(hzovaro): write tests for this
    ra, dec = raw_subject["coords"]
    coord = SkyCoord(ra=ra, dec=dec, unit="deg")
    fname = cache / f'{raw_subject["_id"]["$oid"]}.fits'
    try:
        return fits.open(fname)
    except FileNotFoundError:
        logger.debug("Cache miss; downloading %s", fname)
        im = download_first(coord, image_size=3 * u.arcmin)
        im.writeto(fname)
        return im


def fetch_first_catalogue_from_server_or_cache(
    cache: Path,
) -> astropy.table.table.Table:
    """Fetches the FIRST catalogue from Vizier or cache."""
    try:
        return astropy.io.ascii.read(
            str(cache / constants.FIRST_CATALOGUE_NAME), guess=False, format="csv"
        )  # type: ignore[reportReturnType]
    except IOError as e:
        logger.info("Cache miss; downloading FIRST table from Vizier")
        download_first_catalogue(cache)
        return fetch_first_catalogue_from_server_or_cache(cache)


def build_first_tree(first_catalogue: astropy.table.table.Table) -> FIRSTTree:
    """Build a spatial index for FIRST component centres."""
    coords = np.stack([first_catalogue["RA_DEG"], first_catalogue["DE_DEG"]]).T  # type: ignore
    return (coords, list(first_catalogue["FIRST"]))  # type: ignore


def download_first_catalogue(cache: Path):
    """Downloads the FIRST catalogue from Vizier."""
    first = Vizier(row_limit=-1).get_catalogs(  # type: ignore[reportAttributeAccessIssue]
        "VIII/92/first14"
    )
    skc = SkyCoord(
        ra=first[0]["RAJ2000"],
        dec=first[0]["DEJ2000"],
        unit=(u.hourangle, u.deg),
    )
    ra = skc.ra.deg  # type: ignore[reportOptionalMemberAccess]
    dec = skc.dec.deg  # type: ignore[reportOptionalMemberAccess]
    first[0]["RA_DEG"] = ra
    first[0]["DE_DEG"] = dec
    first[0].write(cache / constants.FIRST_CATALOGUE_NAME, format="csv")

def get_first_from_bbox(
    px_bbox: BBox,
    wcs: WCS,
    first_tree: FIRSTTree,
) -> list[FIRSTID]:
    """Finds FIRST components within a bounding box."""
    # TODO(MatthewJA): Also use the contours to ensure that they really are within the boxes.
    phys_bbox = transform_bbox_px_to_phys(px_bbox, wcs)
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

    # TODO(MatthewJA): Speed this up using some kind of tree.
    ra, dec = rgz.get_deg(skc)
    width_deg = width.to(u.deg).value
    height_deg = height.to(u.deg).value
    upper_ra = ra + width_deg / 2
    lower_ra = ra - width_deg / 2
    upper_dec = dec + height_deg / 2
    lower_dec = dec - height_deg / 2
    matching_indices = find_points_in_box(
        first_tree[0], lower_ra, upper_ra, lower_dec, upper_dec
    )
    if not matching_indices:
        coord_str = rgz.coord_to_string(skc)
        return [f'NOFIRST_J{coord_str.replace(" ", "")}']

    names = []
    for index in matching_indices:
        names.append("FIRST_" + first_tree[1][index])
    return sorted(names)


def transform_coord_radio(
    coord: npt.NDArray[np.float64],
    wcs: WCS,
) -> Quantity[u.deg, u.deg]:
    """Transforms a radio image pixel coordinate into RA/dec."""
    # Coord in 132x132 -> 100x100.
    # TODO(hzovaro): are coords indexed from 1 or zero? Change the below to
    # reflect this.
    if np.any(coord < 0) or np.any(coord >= constants.RADIO_MAX_PX):
        raise ValueError(
            f"pixel coordinates {coord} "
            "are outside of range "
            f"[0, {constants.RADIO_MAX_PX})!"
        )
    coord = coord * 100 / constants.RADIO_MAX_PX
    return wcs.all_pix2world([coord], 0)[0] * u.deg


def transform_bbox_px_to_phys(
    px_bbox: BBox,
    wcs: WCS,
) -> npt.NDArray[np.float64]:
    """Transforms a bbox from pixel coordinates to RA/dec."""
    xmin, ymin, xmax, ymax = px_bbox
    # Flip vertically.
    phys_bbox = np.array(
        [
            xmin,
            constants.RADIO_MAX_PX - 1 - ymax,
            xmax,
            constants.RADIO_MAX_PX - 1 - ymin,
        ]
    )
    return np.concatenate(
        [
            transform_coord_radio(coord=phys_bbox[:2], wcs=wcs),
            transform_coord_radio(coord=phys_bbox[2:], wcs=wcs),
        ]
    )


def find_points_in_box(
    points: npt.NDArray,
    lower_ra: u.Quantity[u.deg],
    upper_ra: u.Quantity[u.deg],
    lower_dec: u.Quantity[u.deg],
    upper_dec: u.Quantity[u.deg],
) -> list[int]:
    """Finds points that are within a box."""
    if upper_ra < lower_ra:
        # Edge case at RA = 0.
        # Left side:
        return find_points_in_box(
            points, lower_ra, 360.0 * u.deg, lower_dec, upper_dec
        ) + find_points_in_box(points, 0 * u.deg, upper_ra, lower_dec, upper_dec)
    # We need to have <= or we would fail on the boundary.
    mask = (
        (points[:, 0] <= upper_ra)
        & (points[:, 0] >= lower_ra)
        & (points[:, 1] <= upper_dec)
        & (points[:, 1] >= lower_dec)
    )
    return list(mask.nonzero()[0])




# @dataclass
# class BBox:
#     """Class for holding a bounding box."""
#     xmin_px: float 
#     ymin_px: float
#     xmax_px: float 
#     ymax_px: float


# @dataclass
# class BBox_phys:
#     """Class for holding a bounding box."""
#     xmin_phys: Quantity[u.deg]
#     ymin_phys: Quantity[u.deg]
#     xmax_phys: Quantity[u.deg]
#     ymax_phys: Quantity[u.deg]


# class RadioIsland:
#     """A Radio Galaxy Zoo radio island.

#     Radio islands are defined as contiguous radio sources in FIRST.
    
#     Attributes: 
#     - pixel-unit bounding box
#     - physical-unit bounding box 
#         - to make this, need a WCS instance. 
#     - contours 
#         - would this make the instances too big? Probably not if we just show 
#             the zeroth contours. 
#         - Q: do we want to store contours as part of class instances, or just 
#             make a method that gets them dynamically? 

#     Methods:
#     - get_first_from_bbox

#     TODO: 
#     - where would this get made in the code? and at that point, what information
#         do we have? 
#             A: in process_subject.
#     - how and where would this get used in plotting methods?

#     Extracting contours:
#     - contours are accessed in get_bbox. TODO: modify this method to return bbox
#         and the zeroth contour. OR, make a separate method that's basically the 
#         same but it grabs the contours instead. 
#         - NOTE: there is a method in plot/contours.py called get_contours that gets contours. 
#             This essentially reads in the raw data in the same way as get_bboxes in subjects.py, except for
#             that get_contours assumes that the raw subject json file (cache/<subject id>.json)
#             already exists, whereas get_bboxes queries the URL for this data 
#             if the json file doesn't exist. We can easily merge these into 
#             a single function (perhaps splitting off the json-file-checking-and
#             -URL-querying into its own function).
#             We can make the level of contours to return an input arg, so that 
#             you can get it to return just the zeroth contour, or the full set.
#             In the constructor for RadioIsland we can call it with level=0.

#     """
    

#     def __init__(self, 
#                  bbox_px: BBox,
#                  contours: ...,
#                  first_tree: FIRSTTree,
#                  wcs: ...) -> None:
#         """
#         logic: 
#             - bboxes and the contours come from the raw subject file.
#             - the WCS, first IDs, transformed bbox coords are all done in post-processing.
#         """

#         self.bbox_px = bbox_px
#         self.contours = contours 
#         self.wcs = wcs

#         # Postprocessing 
#         self.bbox_phys = get_phys_bbox(bbox_px, wcs) 
#         self.firsts = get_first_from_bbox(bbox_px, wcs, first_tree)
    