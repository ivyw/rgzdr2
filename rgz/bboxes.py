"""Handles RGZ bounding boxes."""

from collections.abc import Sequence
from dataclasses import dataclass
import json
import logging
from pathlib import Path

from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from astropy.wcs import WCS
import numpy as np
import numpy.typing as npt

from rgz import constants
from rgz import rgz
from rgz import units as u

logger = logging.getLogger(__name__)


type RGZBBox = tuple[float, float, float, float]


class SpuriousBBoxError(ValueError):
    """Raised when an input bounding box has invalid dimensions."""

    pass


@dataclass(init=True, frozen=True)
class BBox:
    """A bounding box in physical coordinates."""

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
        if self.ra_max == self.ra_min:
            raise SpuriousBBoxError(
                f"'ra_max' ({self.ra_max}) must not be the same as 'ra_min' ({self.ra_min})!"
            )
        elif self.ra_max < self.ra_min:
            # Raise a warning rather than an error because RAs can wrap.
            logger.warning(
                f"'ra_max' {self.ra_max} is less than 'ra_min' {self.ra_min}!"
            )
        if self.dec_max <= self.dec_min:
            logger.warning("'dec_max' is less than 'dec_min'!")
            # TODO(hzovaro): change back
            # raise SpuriousBBoxError("'dec_max' must be greater than 'dec_min'!")

    @property
    def width(self) -> Quantity[u.deg]:
        # Use np.mod here to correctly calculate the width in cases where the
        # BBox crosses an RA of 0.
        return np.mod(self.ra_max.value - self.ra_min.value, 360) * u.deg

    @property
    def height(self) -> Quantity[u.deg]:
        return self.dec_max - self.dec_min

    @property
    def centre(self) -> SkyCoord:
        return SkyCoord(
            0.5 * (self.ra_min + self.ra_max), 0.5 * (self.dec_min + self.dec_max)
        )

    def to_json(self) -> rgz.JSON:
        return {
            "ra_min": self.ra_min.to(u.deg).value,
            "ra_max": self.ra_max.to(u.deg).value,
            "dec_min": self.dec_min.to(u.deg).value,
            "dec_max": self.dec_max.to(u.deg).value,
            "width": self.width.to(u.deg).value,
            "height": self.height.to(u.deg).value,
            "centre": [self.centre.ra.to(u.deg).value, self.centre.dec.to(u.deg).value],
        }

    @classmethod
    def from_json(cls, bbox_dict: rgz.JSON):
        return cls(
            ra_min=bbox_dict["ra_min"] * u.deg,
            dec_min=bbox_dict["dec_min"] * u.deg,
            ra_max=bbox_dict["ra_max"] * u.deg,
            dec_max=bbox_dict["dec_max"] * u.deg,
        )

    def get_points_in_box(
        self,
        points: npt.NDArray,
    ) -> list[int]:
        """Returns the subset of points that lie within the BBox."""
        if self.ra_max > self.ra_min:
            mask = (
                (points[:, 0] <= self.ra_max)
                & (points[:, 0] >= self.ra_min)
                & (points[:, 1] <= self.dec_max)
                & (points[:, 1] >= self.dec_min)
            )
        else:
            mask = (
                (points[:, 0] <= self.ra_max)
                & (points[:, 0] >= (self.ra_min - 360 * u.deg))
                & (points[:, 1] <= self.dec_max)
                & (points[:, 1] >= self.dec_min)
            )
        return list(mask.nonzero()[0])


def get_bboxes(
    sid: str,
    wcs: WCS,
    cache: Path,
    units: str = "physical",
) -> Sequence[BBox] | Sequence[RGZBBox]:
    """Fetches the bboxes of a subject in RA/Dec."""
    fname = cache / f"{sid}.json"
    with open(fname) as f:
        js = json.load(f)
    bbox_list = []
    for contour in js["contours"]:
        assert contour[0]["k"] == 0
        # NOTE: Bboxes are 1-indexed.
        bbox = tuple([round(c, 1) for c in contour[0]["bbox"]])
        if units == "physical":
            bbox_list.append(transform_rgzbbox_to_phys(bbox=bbox, wcs=wcs))
        else:
            bbox_list.append(bbox)
    return bbox_list


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
    ymin_transformed = (
        (constants.RADIO_MAX_PX - 1 - (bbox[1] - 1)) * 100 / constants.RADIO_MAX_PX
    )
    xmax_transformed = (bbox[2] - 1) * 100 / constants.RADIO_MAX_PX
    ymax_transformed = (
        (constants.RADIO_MAX_PX - 1 - (bbox[3] - 1)) * 100 / constants.RADIO_MAX_PX
    )

    # Transform to RA/Dec.
    ra_min, dec_min = (
        wcs.all_pix2world(np.array([[xmin_transformed, ymin_transformed]]), 0)[0]
        * u.deg
    )
    ra_max, dec_max = (
        wcs.all_pix2world(np.array([[xmax_transformed, ymax_transformed]]), 0)[0]
        * u.deg
    )

    return BBox(ra_min=ra_min, dec_min=dec_min, ra_max=ra_max, dec_max=dec_max)
