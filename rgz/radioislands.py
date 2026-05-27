# """Handles RGZ radio islands."""
from collections.abc import Sequence
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Self

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


type HDU = fits.hdu.base.ExtensionHDU
type FIRSTID = str


# Sticking with a simple tuple instead of a class since we don't really 
# use these anywhere
type RGZBBox = tuple[float, float, float, float]
type FIRSTTree = tuple[npt.NDArray[np.float64], list[str]]

class SpuriousBBoxError(ValueError):
    """Raised when an input bounding box has invalid dimensions."""
    pass


@dataclass(init=True)
class BBox:
    """Class for holding a bounding box in physical coordinates."""
    ra_min: Quantity[u.deg]
    dec_min: Quantity[u.deg]
    ra_max: Quantity[u.deg]
    dec_max: Quantity[u.deg]

    def __post_init__(self):
        if not isinstance(self.ra_min, Quantity):
            raise ValueError("'ra_min' must be an astropy Quantity!")
        if not isinstance(self.ra_max, Quantity):
            raise ValueError("'ra_max' must be an astropy Quantity!")
        if not isinstance(self.dec_min, Quantity):
            raise ValueError("'dec_min' must be an astropy Quantity!")
        if not isinstance(self.dec_max, Quantity):
            raise ValueError("'dec_max' must be an astropy Quantity!")
        if (self.dec_max > 90.0 * u.deg) or (self.dec_max < -90.0 * u.deg):
            raise SpuriousBBoxError("'dec_max' must be between -90 and 90 degrees!")
        if (self.dec_min > 90.0 * u.deg) or (self.dec_min < -90.0 * u.deg):
            raise SpuriousBBoxError("'dec_min' must be between -90 and 90 degrees!")
        if (self.ra_max > 360.0 * u.deg) or (self.ra_max < 0.0 * u.deg):
            raise SpuriousBBoxError("'ra_max' must be between -90 and 90 degrees!")
        if (self.ra_min > 360.0 * u.deg) or (self.ra_min < 0.0 * u.deg):
            raise SpuriousBBoxError("'ra_min' must be between -90 and 90 degrees!")
        if (self.ra_max <= self.ra_min):
            logger.warning("'ra_max' is less than 'ra_min'!")
            # TODO(hzovaro): change back
            # raise SpuriousBBoxError("'ra_max' must be greater than 'ra_min'!")
        if (self.dec_max <= self.dec_min):
            logger.warning("'dec_max' is less than 'dec_min'!")
            # TODO(hzovaro): change back
            # raise SpuriousBBoxError("'dec_max' must be greater than 'dec_min'!")
        
    @property
    def width(self):
        return self.ra_max - self.ra_min
        
    @property
    def height(self):
        return self.dec_max - self.dec_min
        
    @property
    def centre(self):
        return SkyCoord(0.5 * (self.ra_min + self.ra_max),
                        0.5 * (self.dec_min + self.dec_max))

    def to_json(self) -> rgz.JSON:
        return {
            "ra_min": self.ra_min.value,
            "ra_max": self.ra_max.value,
            "dec_min": self.dec_min.value,
            "dec_max": self.dec_max.value,
            "width": self.width.value,
            "height": self.height.value,
            "centre": [self.centre.ra.value, self.centre.dec.value],
        }
    
    @classmethod
    def from_json(cls, bbox_dict: rgz.JSON):
        return cls(
            ra_min=bbox_dict["ra_min"] * u.deg,
            dec_min=bbox_dict["dec_min"] * u.deg,
            ra_max=bbox_dict["ra_max"] * u.deg,
            dec_max=bbox_dict["dec_max"] * u.deg,
        )


@backoff.on_exception(
    backoff.expo,
    requests.exceptions.ConnectionError,
    max_tries=MAX_TRIES,
)
def download_first_image(
    raw_subject: rgz.JSON | None,
    cache: Path,
) -> None:
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
    if Path(fname).exists():
        logger.debug(f"File {fname} already exists! Not re-downloading")
        return
    logger.debug("Cache miss; downloading %s", fname)
    ims = First.get_images(coord, image_size=3 * u.arcmin)
    if not isinstance(ims, fits.HDUList):
        # Technically allowed by documentation, but I don't expect it to happen
        # with the files we're opening (i.e. FIRST survey files).
        raise TypeError(f"Expected HDUList; got {type(ims)}")
    ims.writeto(fname)


def load_first_image(
    sid: str,
    cache: Path,
) -> fits.HDUList:
    """Fetches FIRST image data for a subject.
    """
    # TODO(hzovaro): this should be a method of radioisland
    fname = cache / f'{sid}.fits'
    return fits.open(fname)


def get_wcs(
    sid: str,
    cache: Path,
) -> WCS:
    """Fetches the FIRST WCS of a subject."""
    hdulist = load_first_image(sid, cache)
    return rgz.get_wcs(hdulist)


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
    bbox: BBox,
    first_tree: FIRSTTree,
) -> list[FIRSTID]:
    """Finds FIRST components within a bounding box."""
    # TODO(MatthewJA): Also use the contours to ensure that they really are within the boxes.

    # Round widths and heights up to nearest arcsec plus two.
    width = np.ceil(bbox.width.to(u.arcsec)) + 2 * u.arcsec
    height = np.ceil(bbox.height.to(u.arcsec)) + 2 * u.arcsec
    logger.debug("get_first_from_bbox: %s %s %s", bbox.centre, width, height)

    # TODO(MatthewJA): Speed this up using some kind of tree.
    ra = bbox.centre.ra.value
    dec = bbox.centre.dec.value
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
        coord_str = rgz.coord_to_string(bbox.centre)
        ra_hh, ra_mm, ra_ss, dec_dd, dec_mm, dec_ss = [float(s) for s in coord_str.split(" ")]
        coord_str = f"NOFIRST_J{ra_hh:02.0f}{ra_mm:02.0f}{ra_ss:0.1f}{dec_dd:+02.0f}{dec_mm:02.0f}{dec_ss:02.0f}"
        return [f'NOFIRST_J{coord_str.replace(" ", "")}']

    names = []
    for index in matching_indices:
        names.append("FIRST_" + first_tree[1][index])
    return sorted(names)


def transform_rgzbbox_to_phys(
    bbox: RGZBBox,
    wcs: WCS,
) -> BBox:
    """Transforms a bbox from pixel coordinates to RA/dec.
    NOTE: the order in which the coordinates are read in looks weird, but it
    is correct!!! See here: https://github.com/ivyw/rgzdr2/issues/56
    (noting that since making that comment on the GitHub, I identified a 
    further minor issue where ymin and yax were swapped around, meaning that 
    dec_min was greater than dec_max. This has now been fixed in the below code)
    """
    # Reset origin to zero, flip vertically, and scale.
    xmin_transformed = (bbox[0] - 1) * 100 / constants.RADIO_MAX_PX
    ymin_transformed = (constants.RADIO_MAX_PX - 1 - (bbox[1] - 1)) * 100 / constants.RADIO_MAX_PX
    xmax_transformed = (bbox[2] - 1) * 100 / constants.RADIO_MAX_PX
    ymax_transformed = (constants.RADIO_MAX_PX - 1 - (bbox[3] - 1)) * 100 / constants.RADIO_MAX_PX

    # Transform to RA/Dec.
    ra_min, dec_min = wcs.all_pix2world(np.array([[xmin_transformed, ymin_transformed]]), 0)[0] * u.deg
    ra_max, dec_max = wcs.all_pix2world(np.array([[xmax_transformed, ymax_transformed]]), 0)[0] * u.deg

    return BBox(ra_min=ra_min,
                dec_min=dec_min,
                ra_max=ra_max,
                dec_max=dec_max)


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


def download_contour_data(
    raw_subject: rgz.JSON,
    cache: Path,
):
    """Fetches contour data for a raw subject, caching locally.
    #TODO(hzovaro): what is the appropriate return type?
    """
    # NOTE: this function is what creates the file below.
    # TODO(hzovaro): handle file already exists properly
    fname = cache / f'{raw_subject["_id"]["$oid"]}.json'
    if Path(fname).exists():
        logger.debug(f"File {fname} already exists! Not re-downloading")
        return
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


def load_contour_data(
    sid: str,
    cache: Path,
):
    """Fetches contour data for a raw subject.
    #TODO(hzovaro): what is the appropriate return type?
    """
    fname = cache / f'{sid}.json'
    with open(fname) as f:
        return json.load(f)        


def __get_rgzbboxes(
    sid: str,
    wcs: WCS,
    cache: Path,
) -> Sequence[RGZBBox]:
    """Fetches the RAW RGZ bboxes in weird fucked up coordinates. ONLY TO BE USED FOR DEBUGGING!"""
    js = load_contour_data(sid, cache)
    bboxes = []
    for contour in js["contours"]:
        assert contour[0]["k"] == 0
        # NOTE: Bboxes are 1-indexed.
        bbox = tuple([round(c, 1) for c in contour[0]["bbox"]])
        bboxes.append(bbox)
    return bboxes


def get_bboxes(
    sid: str,
    wcs: WCS,
    cache: Path,
) -> Sequence[BBox]:
    """Fetches the bboxes of a subject in RA/Dec."""
    js = load_contour_data(sid, cache)
    bboxes = []
    for contour in js["contours"]:
        assert contour[0]["k"] == 0
        # NOTE: Bboxes are 1-indexed.
        bbox = tuple([round(c, 1) for c in contour[0]["bbox"]])
        bboxes.append(transform_rgzbbox_to_phys(bbox=bbox, wcs=wcs))
    return bboxes


class ContoursNotFoundError(Exception):
    """Raised when the contours list in the raw subject JSON file is empty."""


def get_contours(
    sid: str,
    wcs: WCS,
    cache: Path,
) -> list[list[tuple]]:  # N contours x N points x 2
    """Returns the contours of a subject in RA/Dec.
    # TODO(hzovaro): this is broken - it should return a list of len
    # N x radio islands x N x contours x 2 but it only returns a list of length 1
    # TODO(hzovaro): what about contour level?
    # TODO(hzovaro): should we round these?

    The raw contour data consists of a series of (x, y) coordinate pairs
    relative to the upper-left hand corner of a 132x132 image, where (65, 65)
    represents the centre of the image. If px_coords is True, this function
    applies applies a stretch defined by the px_scaling arg such that
    the coordinates are defined on a

         (px_scaling * RADIO_MAX_PX) x (px_scaling * RADIO_MAX_PX)

    grid. By default, px_scaling is set so that the returned coordinates are
    defined on a 100 x 100 grid to match the dimensions of the FIRST images.

    Args:
        subject: the subject.
        cache: path to contour data.

    Returns:
        A list of lists each representing a radio island, each of which
        contains a list of (x, y) pairs for each contour.

    Raises:
        FileNotFoundError if the file containing the contour data cannot be
        found.
        
    """
    js = load_contour_data(sid, cache)
    island_contours = []
    for island in js["contours"]:
        contours = []
        for contour in island:
            xs = [(coord["x"] - 1) * 100 / constants.RADIO_MAX_PX for coord in contour["arr"]]
            ys = [(constants.RADIO_MAX_PX - 1 - (coord["y"] - 1)) * 100 / constants.RADIO_MAX_PX for coord in contour["arr"]]
            # transform 
            coords = wcs.all_pix2world(np.vstack([xs, ys]).T, 0) * u.deg
            coords = [(x.value, y.value) for x, y in coords]
            contours.append(coords)
        island_contours.append(contours)
    if len(island_contours) == 0:
        raise ContoursNotFoundError(f"Contour data not found for subject {sid}!")
    return island_contours


class RadioIsland:
    """A Radio Galaxy Zoo radio island.

    Radio islands are defined as contiguous radio sources in FIRST.
    
    Attributes: 
    - physical-unit bounding box 
    - contours 

    Methods:
    - get_first_from_bbox


    """

    def __init__(self, 
                 rgzbbox: RGZBBox,
                 bbox: BBox,
                 contours: list[list[tuple]], # TODO what is this?
                 first_tree: FIRSTTree | None = None,
                 firsts: list[FIRSTID] | None = None,
                 ) -> None:
        """Initialise a RadioIsland instance."""
        # Input validation
        if (first_tree is None) and (firsts is None):
            raise ValueError("first_tree must be specified if firsts is None!")
        if (first_tree is not None) and (firsts is not None):
            raise ValueError("Only one of first_tree and firsts may be specified!")
        if (firsts is not None):
            if len(firsts) == 0:
                raise ValueError("firsts must not be empty!")

        # Initialise instance attributes
        self.rgzbbox = rgzbbox
        self.bbox = bbox
        self.contours = contours  # TODO(hzovaro): input check this
        if firsts is not None:
            self.firsts = firsts
        else:
            self.firsts = get_first_from_bbox(bbox, first_tree)

    def __eq__(self, other) -> bool:
        if not isinstance(other, RadioIsland):
            return ValueError("other must be a BBox!")
        return (self.bbox == other.bbox) &\
            (self.rgzbbox == other.rgzbbox) &\
            (self.contours == other.contours) &\
            (self.firsts == other.firsts)

    def to_json(self) -> rgz.JSON:
        """Converts a RadioIsland into a JSON-compatible dictionary."""
        return {
            "bbox": self.bbox.to_json(), 
            "rgzbbox": self.rgzbbox,
            "contours": self.contours,
            "firsts": self.firsts,
        }
    
    @classmethod
    def from_json(cls, radioisland: rgz.JSON) -> Self:
        """Reads a RadioIsland from JSON."""
        return cls(
            bbox=BBox.from_json(radioisland["bbox"]),
            rgzbbox=radioisland["rgzbbox"],
            contours=radioisland["contours"],
            firsts=radioisland["firsts"],
        )