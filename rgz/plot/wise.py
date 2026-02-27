"""WISE plotting utilities."""

import astropy.io.fits
import astropy.wcs
from astropy.coordinates import SkyCoord
import attrs
import matplotlib.axes
import numpy.typing as npt

from rgz import cutouts
from rgz import constants
from rgz import units as u


@attrs.define
class WISEImage:
    data: astropy.io.fits.ImageHDU
    wcs: astropy.wcs.WCS


def get_wise_image(coords: tuple[float, float]) -> WISEImage:
    """Gets a RGZ WISE cutout centred on the given coordinates.

    Args:
        coords: (ra, dec) in deg.

    Return:
        WISE image.
    """
    ra, dec = coords
    coords_wise = SkyCoord(ra, dec, unit="deg")
    hdulist_wise = cutouts.get_allwise_cutout(
        coords=coords_wise,
        size=constants.IM_WIDTH_ARCMIN * u.arcmin,
        save_fits=False,
    )
    hdu_wise: astropy.io.fits.ImageHDU = hdulist_wise[
        0
    ]  # pyright: ignore[reportAssignmentType]
    return WISEImage(
        data=hdu_wise,
        wcs=astropy.wcs.WCS(hdu_wise.header),
    )

def imshow(im: astropy.io.fits.ImageHDU, ax: matplotlib.axes.Axes):
    """Shows a WISE image."""
    data: npt.ArrayLike = im.data # pyright: ignore[reportAssignmentType]
    ax.imshow(data, cmap="gist_heat", vmax=6, vmin=2)
