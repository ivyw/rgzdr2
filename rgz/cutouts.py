from io import BytesIO
import logging
import numpy as np
from pathlib import Path
from typing import Literal
import urllib.parse

from astropy import units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.table import Table
import pandas as pd
import requests


logger = logging.getLogger(__name__)


class SIAQueryFailError(Exception):
    """Raised when a Simple Image Access Query fails."""


class CutoutNotFoundError(Exception):
    """Raised when no valid cutouts can be found."""


class CutoutDownloadFailError(Exception):
    """Raised when a cutout download fails."""


class NegativeImageSizeError(ValueError):
    """Raised when the requested cutout size is negative."""


class InvalidWISEBandError(ValueError):
    """Raised when the requested band is invalid."""


def get_allwise_image_list(
    ra: u.Quantity[u.deg],
    dec: u.Quantity[u.deg],
    band: Literal["W1", "W2", "W3", "W4"],
) -> pd.DataFrame:
    """Returns a Pandas DataFrame containing a list of AllWISE images containing 
    the specified coordinates.

    This function uses a Simple Image Access Query to obtain a list of
    "science"-quality AllWISE images in the selected band containing the
    specified coordinates within a 0.01 degree radius.

    References:
        https://wise2.ipac.caltech.edu/docs/release/allwise/
        https://irsa.ipac.caltech.edu/ibe/docs/wise/allwise/p3am_cdd/
        https://irsa.ipac.caltech.edu/ibe/sia.html
        https://irsa.ipac.caltech.edu/ibe/cutouts.html

    Args:
        ra_deg: right ascension in degrees.
        dec_deg: declination in degrees.
        band: AllWISE band.

    Returns:
        a Pandas DataFrame containing valid AllWISE image metadata.

    Raises:
        SIAQueryFailError: the Simple Image Access Query failed.
    """
    # NOTE: passing a separate URL params dict to requests.get doesn't work 
    # because of the plus signs
    ra_deg = ra.to(u.deg).value
    dec_deg = dec.to(u.deg).value
    imglist_url = ("https://irsa.ipac.caltech.edu/SIA?"
                    "COLLECTION=wise_allwise&POS=circle+"
                    f"{ra_deg:.5f}+{dec_deg:.5f}+0.01&RESPONSEFORMAT=FITS")
    try:
        r = requests.get(imglist_url)
    except ConnectionError as e:
        raise SIAQueryFailError(
            f"Simple Image Access Query failed with message {e.message}!"
        ) from e
    df = Table(fits.open(BytesIO(r.content))[1].data).to_pandas()

    # Filter by band and science readiness
    cond = df["energy_bandpassname"] == band
    cond &= df["dataproduct_subtype"].str.rstrip() == "science"

    # Return DataFrame with valid rows
    return df.loc[cond]


def get_allwise_cutout(
    coords: SkyCoord,
    size: u.Quantity[u.arcmin] = 3 * u.arcmin,
    band: Literal["W1", "W2", "W3", "W4"] = "W1",
    save_fits: bool = False,
    cutout_path: Path = None,
) -> fits.HDUList | None:
    """Returns a FITS HDUList of an AllWISE cutout, optionally saving to file.

    This function extracts a cutout image from AllWISE via the NASA/IPAC
    Infrared Science Archive (IRSA). This is a two-step process: first, a
    Simple Image Access Query is used to identify the list of AllWISE images
    containing the requested coordinates (see get_allwise_image_list()).
    The image from which to extract the cutout is taken as the first image in
    the list in the requested WISE band that is also of "science" quality.
    Then, a cutout from the image is extracted by constructing a query using
    the corresponding access URL. If the cutout exists, this function returns
    an astropy FITS Header Data Unit List (HDUList) containing the FITS header
    and the image; otherwise an exception is raised.

    References:
        https://wise2.ipac.caltech.edu/docs/release/allwise/
        https://irsa.ipac.caltech.edu/ibe/docs/wise/allwise/p3am_cdd/
        https://irsa.ipac.caltech.edu/ibe/sia.html
        https://irsa.ipac.caltech.edu/ibe/cutouts.html

    Args:
        coords: Requested cutout coordinates.
        size: Requested cutout size in arcminutes.
        band: AllWISE band. Default: W1.
        save_fits: If True, saves the downloaded cutout to path specified by
            cutout_path. Otherwise, the FITS file is saved to a temporary file.
        cutout_path: If save_fits is True, specifies the path where the image
            is saved; this argument is ignored otherwise. If save_fits is True
            and cutout_path is unspecified, the file is saved to
                f"allwise_{band:s}_{ra_deg:.4f}_{dec_deg:.4f}.fits"
            where ra_deg and dec_deg are the RA and dec respectively.

    Returns:
        If a valid cutout is found, returns an astropy.io.fits.HDUList
        containing the FITS header (the zeroth extension) and the image (the
        first extension).

    Raises:
        NegativeImageSizeError: the cutout image size is negative.
        InvalidWISEBandError: the specified band is not a valid WISE band.
        CutoutNotFoundError: no AllWISE cutout could be found for the input
            combination of coordinates and band.
        CutoutDownloadFailError: an error occurred during the download of the
            FITS file.
    """
    # Input validation
    size_arcmin = size.to(u.arcmin).value
    if size_arcmin < 0:
        raise NegativeImageSizeError(f"Cutout size {size_arcmin} < 0!")
    valid_bands = ["W1", "W2", "W3", "W4"]
    if band not in valid_bands:
        raise InvalidWISEBandError(
            f"Band {band} is not a valid WISE band - "
            f"valid values are {', '.join(valid_bands)}"
        )

    # If no filename is supplied, save cutout to 
    # allwise_<band>_<ra_deg>_<dec_deg>.fits
    ra_deg = coords.ra.value
    dec_deg = coords.dec.value
    if cutout_path is None:
        cutout_path = Path(f"allwise_{band:s}_{ra_deg:.4f}_{dec_deg:.4f}.fits")
    else:
        if not save_fits:
            logger.warning(
                "You have specified a cutout_path but I am not saving the "
                "result to file!"
            )
        if not cutout_path.suffix:
            cutout_path = cutout_path.with_suffix(".fits")

    # Get list of AllWISE images containing the target RA/Dec
    df_valid = get_allwise_image_list(ra=coords.ra, 
                                      dec=coords.dec,
                                      band=band)
    if df_valid.shape[0] == 0:
        raise CutoutNotFoundError(
            f"No valid AllWISE cutouts could be found for inputs "
            f"RA = {ra_deg:.4f}, dec = {dec_deg:.4f}, band = {band}"
        )

    # Get the access URL for the image. If there are multiple then just take 
    # the first one
    access_url = df_valid["access_url"].values[0].rstrip()

    # Construct the cutout URL & download
    url_params= urllib.parse.urlencode({
        "center": f"{ra_deg:.5f},{dec_deg:.5f}deg",
        "size": f"{size_arcmin:5f}arcmin",
    }, safe=",")
    cutout_url = f"{access_url}?{url_params}"
    try:
        hdulist = fits.open(cutout_url)
    except Exception as e:
        raise CutoutDownloadFailError(
            f"Cutout download failed with message {e.message}!"
        ) from e

    # Save to file if requested
    if save_fits:
        hdulist.writeto(cutout_path, overwrite=True)

    return hdulist


if __name__ == "__main__":
    from astropy.wcs import WCS
    import matplotlib.pyplot as plt
    import numpy as np

    # Plot the AllWISE cutout for NGC1068
    coords = SkyCoord(
        ra="02:42:40.71", 
        dec="-00:00:47.86", 
        unit=(u.hourangle, u.deg), 
        equinox="J2000"
    )
    hdulist = get_allwise_cutout(coords=coords, size=3.5 * u.arcmin)
    im = hdulist[0].data
    wcs = WCS(hdulist[0].header)
    fig, ax = plt.subplots(subplot_kw=dict(projection=wcs))
    ax.imshow(np.log10(im))
