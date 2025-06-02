import logging
from pathlib import Path 
import shutil
import tempfile
import urllib
import urllib.error

import pandas as pd

from astropy import units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.table import Table

from IPython.core.debugger import set_trace

logger = logging.getLogger(__name__)

def get_allwise_cutout(coords: SkyCoord,
                       size: u.Quantity[u.arcmin] = 3 * u.arcmin,
                       band: str = "W1",
                       save_fits: bool = False,
                       cutout_path: Path = None,
                       ) -> fits.HDUList | None:
    """Returns a FITS HDUList of an AllWISE cutout, optionally saving it to file. 

    This function extracts a cutout image from AllWISE via the NASA/IPAC 
    Infrared Science Archive (IRSA). This is a two-step process: first, a
    Simple Image Access Query is used to identify the list of AllWISE images 
    containing the requested coordinates. The image from which to extract the 
    cutout is taken as the first image in the list in the requested WISE band 
    that is also of "science" quality. Then, a cutout from the image is 
    extracted by constructing a query using the corresponding access URL. 
    If the cutout exists, this function returns an astropy FITS Header Data 
    Unit List (HDUList) containing the FITS header and the image; otherwise it 
    returns None.

    INPUTS
    ---------------------------------------------------------------------------
    coords          astropy.Skycoord
        Requested cutout coordinates. 
    
    size     float 
        Requested cutout size in arcminutes. 
    
    band            str (default: "W1")
        AllWISE band. Default: W1.
    
    save_fits       bool (default: False)
        If True, saves the downloaded cutout to path specified by cutout_path.
        Otherwise, the FITS file is saved to a temporary file. 
    
    cutout_path    Path (default: None)
        If save_fits is True, specifies the path where the image is saved; this
        argument is ignored otherwise.
        If save_fits is True and cutout_path is unspecified, the file is 
        saved to 
            f"allwise_{band:s}_{ra_deg:.4f}_{dec_deg:.4f}.fits"
        where ra_deg and dec_deg are the RA and dec respectively.
    
    RETURNS
    ---------------------------------------------------------------------------
    If a valid cutout is found, returns an astropy.io.fits.HDUList containing
    the FITS header (the zeroth extension) and the image (the first extension).
    If no valid cutout is found, returns None.  
    
    REFERENCES 
    ---------------------------------------------------------------------------
        https://wise2.ipac.caltech.edu/docs/release/allwise/ 
        https://irsa.ipac.caltech.edu/ibe/docs/wise/allwise/p3am_cdd/ 
        https://irsa.ipac.caltech.edu/ibe/sia.html 
        https://irsa.ipac.caltech.edu/ibe/cutouts.html 
    """
    # Input validation
    size_arcmin = size.to(u.arcmin).value
    if size_arcmin < 0:
        raise ValueError(f"Cutout size {size_arcmin} < 0!")
    valid_bands = ["W1", "W2", "W3", "W4"]
    if band not in valid_bands:
        raise ValueError(f"Band {band} is not a valid WISE band - valid values are {','.join(valid_bands)}")

    # If no filename is supplied, save cutout to allwise_<band>_<ra_deg>_<dec_deg>.fits
    ra_deg = coords.ra.value
    dec_deg = coords.dec.value
    if cutout_path is None:
        cutout_path = Path(f"allwise_{band:s}_{ra_deg:.4f}_{dec_deg:.4f}.fits")
    else:
        if save_fits == False:
            logger.warning("You have specified a cutout_path but I am not saving the result to file!")
        if cutout_path.suffix == "":
            cutout_path = cutout_path.with_suffix(".fits")

    # Get list of AllWISE images containing the target RA/Dec, save to a temporary file 
    imglist_url = f"https://irsa.ipac.caltech.edu/SIA?COLLECTION=wise_allwise&POS=circle+{ra_deg:.5f}+{dec_deg:.5f}+0.01&RESPONSEFORMAT=FITS"
    try:
        with urllib.request.urlopen(imglist_url) as response:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                shutil.copyfileobj(response, tmp_file)
    except urllib.error.URLError as e:
        msg = f"Simple Image Access Query failed with message {e.message}!"
        logger.warning(msg)
        return

    t = fits.open(tmp_file.name)
    tab = t[1].data 
    df = Table(tab).to_pandas()

    # Filter by band and science readiness
    cond = df["energy_bandpassname"] == band
    cond &= df["dataproduct_subtype"].str.rstrip() == "science"
    if df.loc[cond].shape[0] == 0:
        msg = f"AllWISE cutout query for RA = {ra_deg:.4f}, dec = {dec_deg:.4f}, band = {band} returned no valid images!"
        logger.warning(msg)
        return 

    # Get the access URL for the image. If there are multiple then just take the first one 
    access_url = df.loc[cond, "access_url"].values[0].rstrip()

    # Construct the cutout URL & download 
    # set_trace()
    query_str = f"center={ra_deg:.5f},{dec_deg:.5f}deg&size={size_arcmin:5f}arcmin"
    cutout_url = f"{access_url}?{query_str}"
    try:
        if save_fits:
            urllib.request.urlretrieve(cutout_url, cutout_path)
            hdulist = fits.open(cutout_path)
        else:
            with urllib.request.urlopen(cutout_url) as response:
                with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
                    shutil.copyfileobj(response, tmp_file)
                    hdulist = fits.open(tmp_file.name)
    except urllib.error.URLError as e:
        msg = f"Cutout download failed with message {e.message}!"
        logger.warning(msg)

    return hdulist


if __name__ == "__main__":

    import numpy as np
    import astropy.units as u
    from astropy.wcs import WCS
    import matplotlib.pyplot as plt 
    
    plt.ion()
    plt.close("all")

    fname = Path("cutouts/ngc1068.fits")
    coords = SkyCoord(ra="02:42:40.71", dec="-00:00:47.86", unit=(u.hourangle, u.deg), equinox="J2000")
    hdulist = get_allwise_cutout(coords=coords,
                                 size=3.5 * u.arcmin)
    im = hdulist[0].data 
    wcs = WCS(hdulist[0].header)
    fig, ax = plt.subplots(subplot_kw=dict(projection=wcs))
    ax.imshow(np.log10(im))

    # Saving to FITS file 
    cutout_path = Path("cutouts/NGC3997.fits")
    coords = SkyCoord(ra="11:57:47.0", dec="+25:16:14.00", unit=(u.hourangle, u.deg), equinox="J2000")
    hdulist = get_allwise_cutout(coords=coords,
                                 size=10 * u.arcmin,
                                 save_fits=True, 
                                 cutout_path=cutout_path)
    im = hdulist[0].data 
    wcs = WCS(hdulist[0].header)
    fig, ax = plt.subplots(subplot_kw=dict(projection=wcs))
    ax.imshow(np.log10(im))