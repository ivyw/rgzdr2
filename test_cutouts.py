import numpy as np
from pathlib import Path

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
import matplotlib.pyplot as plt 

from cutouts import get_allwise_cutout

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

# TODO how to check that something raises an exception where expected?