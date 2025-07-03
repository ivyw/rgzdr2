from pathlib import Path

from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np

from cutouts import (
    get_allwise_cutout,
    NegativeImageSizeError,
    InvalidWISEBandError,
)


def test_get_cutout():
    coords = SkyCoord(
        ra="02:42:40.71", dec="-00:00:47.86", unit=(u.hourangle, u.deg), equinox="J2000"
    )
    hdulist = get_allwise_cutout(coords=coords, size=3.5 * u.arcmin)


def test_save_cutout():
    # Test that cutout is not saved if save_fits is True
    cutout_path = Path("NGC3997.fits")
    coords = SkyCoord(
        ra="11:57:47.0", dec="+25:16:14.00", unit=(u.hourangle, u.deg), equinox="J2000"
    )
    hdulist = get_allwise_cutout(
        coords=coords, size=10 * u.arcmin, save_fits=True, cutout_path=cutout_path
    )
    assert cutout_path.exists()
    Path.unlink(cutout_path)


def test_save_cutout_default_fname():
    # Test that cutout is saved if save_fits is True and no filename specified
    coords = SkyCoord(
        ra="11:57:47.0", dec="+25:16:14.00", unit=(u.hourangle, u.deg), equinox="J2000"
    )
    hdulist = get_allwise_cutout(coords=coords, size=10 * u.arcmin, save_fits=True)
    default_path = Path(
        f"allwise_{'W1':s}_{coords.ra.value:.4f}_{coords.dec.value:.4f}.fits"
    )
    assert default_path.exists()
    Path.unlink(default_path)


def test_not_save_cutout():
    # Test that cutout is not saved if save_fits is False
    cutout_path = Path("NGC3997.fits")
    Path.unlink(cutout_path, missing_ok=True)
    coords = SkyCoord(
        ra="11:57:47.0", dec="+25:16:14.00", unit=(u.hourangle, u.deg), equinox="J2000"
    )
    hdulist = get_allwise_cutout(
        coords=coords, size=10 * u.arcmin, save_fits=False, cutout_path=cutout_path
    )
    assert not cutout_path.exists()


def test_invalid_cutout_size():
    try:
        coords = SkyCoord(
            ra="00:00:00", dec="00:00:00.0", unit=(u.hourangle, u.deg), equinox="J2000"
        )
        hdulist = get_allwise_cutout(coords=coords, size=-3.5 * u.arcmin)
    except NegativeImageSizeError as e:
        pass


def test_invalid_band():
    try:
        coords = SkyCoord(
            ra="00:00:00", dec="00:00:00.0", unit=(u.hourangle, u.deg), equinox="J2000"
        )
        hdulist = get_allwise_cutout(coords=coords, band="W5", size=3.5 * u.arcmin)
    except InvalidWISEBandError as e:
        pass


if __name__ == "__main__":
    from astropy.wcs import WCS
    import matplotlib.pyplot as plt

    # Plot the AllWISE cutout for NGC1068
    fname = Path("ngc1068.fits")
    coords = SkyCoord(
        ra="02:42:40.71", dec="-00:00:47.86", unit=(u.hourangle, u.deg), equinox="J2000"
    )
    hdulist = get_allwise_cutout(coords=coords, size=3.5 * u.arcmin)
    im = hdulist[0].data
    wcs = WCS(hdulist[0].header)
    fig, ax = plt.subplots(subplot_kw=dict(projection=wcs))
    ax.imshow(np.log10(im))
