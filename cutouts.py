import urllib.error
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.table import Table
from pathlib import Path 
import pandas as pd
import tempfile
import urllib
import shutil
import sys 

def get_allwise_cutout(coords: SkyCoord,
                       size_arcmin: int = 3,
                       band: str = "W1",
                       cutout_fname: Path = None,
                       ):
    """Downloads a FITS cutout from AllWISE. 
    
    References:
        https://irsa.ipac.caltech.edu/ibe/docs/wise/allwise/p3am_cdd/ 
        https://irsa.ipac.caltech.edu/ibe/sia.html 
        https://irsa.ipac.caltech.edu/ibe/cutouts.html 
    """

    # TODO input ra/deg in SkyCoord

    # Input validation
    if size_arcmin < 0:
        raise ValueError(f"Cutout size size_arcmin {size_arcmin} < 0!")
    valid_bands = ["W1", "W2", "W3", "W4"]
    if band not in valid_bands:
        raise ValueError(f"Band {band} is not a valid WISE band - valid values are {','.join(valid_bands)}")

    # If no filename is supplied, save cutout to allwise_<band>_<ra_deg>_<dec_deg>.fits
    ra_deg = coords.ra.value
    dec_deg = coords.dec.value
    if cutout_fname is None:
        cutout_fname = Path(f"allwise_{band:s}_{ra_deg:.4f}_{dec_deg:.4f}.fits")
    elif cutout_fname.suffix == "":
        cutout_fname = cutout_fname.with_suffix(".fits")

    # Get list of AllWISE images containing the target RA/Dec, save to a temporary file 
    # TODO do we want CALIB=2 here?
    imglist_url = f"https://irsa.ipac.caltech.edu/SIA?COLLECTION=wise_allwise&POS=circle+{ra_deg:.5f}+{dec_deg:.5f}+0.01&RESPONSEFORMAT=FITS"
    try:
        with urllib.request.urlopen(imglist_url) as response:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                shutil.copyfileobj(response, tmp_file)
    except urllib.error.URLError as e:
        # TODO add logger 
        msg = f"Simple Image Access Query failed with message {e.message}!"
        print(msg)
        return

    t = fits.open(tmp_file.name)
    tab = t[1].data 
    df = Table(tab).to_pandas()

    # Filter by band and science readiness
    cond = df["energy_bandpassname"] == band
    cond &= df["dataproduct_subtype"].str.rstrip() == "science"
    if df.loc[cond].shape[0] == 0:
        # TODO add logger 
        msg = f"AllWISE cutout query for RA = {ra_deg:.4f}, dec = {dec_deg:.4f}, band = {band} returned no valid images!"
        print(msg)
        return 

    # Get the access URL for the image. If there are multiple then just take the first one 
    access_url = df.loc[cond, "access_url"].values[0].rstrip()

    # Construct the cutout URL & download 
    query_str = f"center={ra_deg:.5f},{dec_deg:.5f}deg&size={size_arcmin:d}arcmin"
    cutout_url = f"{access_url}?{query_str}"
    try:
        urllib.request.urlretrieve(cutout_url, cutout_fname)
    except urllib.error.URLError as e:
        # TODO add logger
        msg = f"Cutout download failed with message {e.message}!"
        print(msg)

    return 


if __name__ == "__main__":

    import numpy as np
    import astropy.units as u
    from astropy.wcs import WCS
    import matplotlib.pyplot as plt 
    
    plt.ion()
    plt.close("all")

    fname = Path("cutouts/ngc1068.fits")
    coords = SkyCoord(ra="02:42:40.71", dec="-00:00:47.86", unit=(u.hourangle, u.deg), equinox="J2000")
    get_allwise_cutout(coords=coords,
                       size_arcmin=10,
                       cutout_fname=fname,)
    
    hdulist = fits.open(fname)
    im = hdulist[0].data 

    fig, ax = plt.subplots(subplot_kw=dict(projection=WCS(hdulist[0].header)))
    ax.imshow(np.log10(im))



