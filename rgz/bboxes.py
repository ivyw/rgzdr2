"""Handles RGZ bounding boxes."""

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path

from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from astropy.wcs import WCS
import numpy as np

from rgz import constants
from rgz import rgz
from rgz import units as u

_HALF_CIRCLE = 180 * u.deg
_FULL_CIRCLE = 360 * u.deg
_MAX_DEC = 90 * u.deg
_MIN_BBOX_SIZE = 1 * u.arcsec

type RGZBBox = tuple[float, float, float, float]


class SpuriousBBoxError(ValueError):
    """Raised when an input bounding box has invalid dimensions."""


@dataclass(init=True, frozen=True)
class BBox:
    """A bounding box in physical coordinates."""

    ra_min: Quantity[u.deg]
    dec_min: Quantity[u.deg]
    ra_max: Quantity[u.deg]
    dec_max: Quantity[u.deg]

    def __post_init__(self):
        if not isinstance(self.ra_min, Quantity):
            raise ValueError("'ra_min' must be an astropy Quantity.")
        if not isinstance(self.ra_max, Quantity):
            raise ValueError("'ra_max' must be an astropy Quantity.")
        if not isinstance(self.dec_min, Quantity):
            raise ValueError("'dec_min' must be an astropy Quantity.")
        if not isinstance(self.dec_max, Quantity):
            raise ValueError("'dec_max' must be an astropy Quantity.")
        # Convert out of Quantity _once_ before comparisons. Much faster.
        ra_min_deg = self.ra_min.to_value(u.deg)
        ra_max_deg = self.ra_max.to_value(u.deg)
        dec_min_deg = self.dec_min.to_value(u.deg)
        dec_max_deg = self.dec_max.to_value(u.deg)
        max_dec_deg = _MAX_DEC.to_value(u.deg)
        full_circle_deg = _FULL_CIRCLE.to_value(u.deg)
        half_circle_deg = _HALF_CIRCLE.to_value(u.deg)
        min_bbox_size_deg = _MIN_BBOX_SIZE.to_value(u.deg)
        if (dec_max_deg > max_dec_deg) or (dec_max_deg < -max_dec_deg):
            raise SpuriousBBoxError("'dec_max' must be between -90 and 90 degrees.")
        if (dec_min_deg > max_dec_deg) or (dec_min_deg < -max_dec_deg):
            raise SpuriousBBoxError("'dec_min' must be between -90 and 90 degrees.")
        if (ra_max_deg > full_circle_deg) or (ra_max_deg < 0.0):
            raise SpuriousBBoxError("'ra_max' must be between -90 and 90 degrees.")
        if (ra_min_deg > full_circle_deg) or (ra_min_deg < 0.0):
            raise SpuriousBBoxError("'ra_min' must be between -90 and 90 degrees.")
        # Allow ra_max < ra_min only when wrapping across RA = 0 (ra_min - ra_max >= 180 deg).
        # Otherwise, inverted or zero-width boxes wrap to ~360 deg in self.width.
        if ra_max_deg <= ra_min_deg and (ra_min_deg - ra_max_deg) < half_circle_deg:
            raise SpuriousBBoxError(
                f"'ra_max' ({self.ra_max}) must be greater than 'ra_min' ({self.ra_min})."
            )
        if dec_max_deg <= dec_min_deg:
            raise SpuriousBBoxError("'dec_max' must be greater than 'dec_min'.")
        width_deg = (ra_max_deg - ra_min_deg) % full_circle_deg
        height_deg = dec_max_deg - dec_min_deg
        if width_deg < min_bbox_size_deg or height_deg < min_bbox_size_deg:
            raise SpuriousBBoxError(
                f"BBox dimensions ({self.width.to(u.arcsec)}, {self.height.to(u.arcsec)})"
                " must be at least 1 arcsec!"
            )

    @property
    def width(self) -> Quantity[u.deg]:
        # Use % to handle width when BBox crosses RA = 0.
        full_circle_deg = _FULL_CIRCLE.to_value(u.deg)
        return (
            (self.ra_max.to_value(u.deg) - self.ra_min.to_value(u.deg))
            % full_circle_deg
        ) * u.deg

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
        points: Quantity[u.deg],
    ) -> list[int]:
        """Returns the indices of points that lie within the BBox.

        Args:
            points: (N, 2) array of (RA, Dec) coordinates.

        Returns:
            Indices of rows in `points` inside the bounding box.
        """
        # Convert out of Astropy Quantity _before_ comparisons,
        # which saves function calls and a ton of time.
        points_deg = points.to_value(u.deg)
        ra_min = self.ra_min.to_value(u.deg)
        ra_max = self.ra_max.to_value(u.deg)
        dec_min = self.dec_min.to_value(u.deg)
        dec_max = self.dec_max.to_value(u.deg)
        # Check one axis first, rather than all axes at once.
        # This also saves comparisons!
        dec_idxs = (
            (points_deg[:, 1] <= dec_max) & (points_deg[:, 1] >= dec_min)
        ).nonzero()[0]
        ras = points_deg[dec_idxs, 0]
        if ra_max > ra_min:
            ra_mask = (ras <= ra_max) & (ras >= ra_min)
        else:
            ra_mask = (ras <= ra_max) & (ras >= (ra_min - 360.0))
        return list(dec_idxs[ra_mask])


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
