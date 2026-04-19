"""WISE plotting utilities."""

from pathlib import Path

import astropy.io.fits
import astropy.wcs
from astropy.coordinates import SkyCoord
import attrs
import matplotlib.axes
import numpy.typing as npt

from rgz import cutouts
from rgz import constants
from rgz import units as u

_CACHE_DIR = 'wise_w1'


@attrs.define
class WISEImage:
    data: astropy.io.fits.ImageHDU
    wcs: astropy.wcs.WCS


def get_wise_image(coords: tuple[float, float], cache: Path, subject_name: str | None = None) -> WISEImage:
    """Gets a RGZ WISE cutout centred on the given coordinates.

    Args:
        coords: (ra, dec) in deg.
        cache: Directory to cache WISE FITS files.
        subject_name: Subject name to use as a filename.

    Return:
        WISE image.
    """
    ra, dec = coords
    coords_wise = SkyCoord(ra, dec, unit="deg")
    
    if subject_name:
        cutout_name = subject_name
    else:
        cutout_name = coords_wise.to_string('hmsdms', sep='').replace(' ', '')
    cutout_path = cache / _CACHE_DIR / f"{cutout_name}.fits"
    try:
        hdu_wise: astropy.io.fits.ImageHDU = astropy.io.fits.open(cutout_path)[0]  # pyright: ignore[reportAssignmentType]
        return WISEImage(data=hdu_wise, wcs=astropy.wcs.WCS(hdu_wise.header))
    except FileNotFoundError:
        pass  # continue

    hdulist_wise = cutouts.get_allwise_cutout(
        coords=coords_wise,
        size=constants.IM_WIDTH_ARCMIN * u.arcmin,
        save_fits=(cache is not None),
        cutout_path=cutout_path,
    )
    hdu_wise: astropy.io.fits.ImageHDU = hdulist_wise[0]  # pyright: ignore[reportAssignmentType]
    return WISEImage(
        data=hdu_wise,
        wcs=astropy.wcs.WCS(hdu_wise.header),
    )


def imshow(im: astropy.io.fits.ImageHDU, ax: matplotlib.axes.Axes):
    """Shows a WISE image."""
    data: npt.ArrayLike = im.data  # pyright: ignore[reportAssignmentType]
    ax.imshow(data, cmap="gist_heat", vmax=6, vmin=2)
