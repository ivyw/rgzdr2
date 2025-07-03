from pathlib import Path
import unittest

from astropy.coordinates import SkyCoord
import astropy.units as u


# TODO importing this from rgz causes a circular import error 
import cutouts


class TestCutouts(unittest.TestCase):
    """Tests for rgz.cutouts."""

    def test_get_cutout(self):
        """Test that cutouts.get_allwise_cutout successfully returns a HDUList."""
        coords = SkyCoord(
            ra="02:42:40.71", dec="-00:00:47.86", unit=(u.hourangle, u.deg), equinox="J2000"
        )
        hdulist = cutouts.get_allwise_cutout(coords=coords, size=3.5 * u.arcmin)
        self.assertTrue(len(hdulist) > 0)


    def test_save_cutout(self):
        """Test that cutout is saved if save_fits is True."""
        cutout_path = Path("NGC3997.fits")
        coords = SkyCoord(
            ra="11:57:47.0", dec="+25:16:14.00", unit=(u.hourangle, u.deg), equinox="J2000"
        )
        hdulist = cutouts.get_allwise_cutout(
            coords=coords, size=10 * u.arcmin, save_fits=True, cutout_path=cutout_path
        )
        self.assertTrue(cutout_path.exists())
        Path.unlink(cutout_path)


    def test_save_cutout_default_fname(self):
        """Test that cutout is saved if save_fits is True and no filename specified."""
        coords = SkyCoord(
            ra="11:57:47.0", dec="+25:16:14.00", unit=(u.hourangle, u.deg), equinox="J2000"
        )
        hdulist = cutouts.get_allwise_cutout(coords=coords, size=10 * u.arcmin, save_fits=True)
        default_path = Path(
            f"allwise_{'W1':s}_{coords.ra.value:.4f}_{coords.dec.value:.4f}.fits"
        )
        self.assertTrue(default_path.exists())
        Path.unlink(default_path)


    def test_not_save_cutout(self):
        """Test that cutout is not saved if save_fits is False."""
        cutout_path = Path("NGC3997.fits")
        Path.unlink(cutout_path, missing_ok=True)
        coords = SkyCoord(
            ra="11:57:47.0", dec="+25:16:14.00", unit=(u.hourangle, u.deg), equinox="J2000"
        )
        hdulist = cutouts.get_allwise_cutout(
            coords=coords, size=10 * u.arcmin, save_fits=False, cutout_path=cutout_path
        )
        self.assertFalse(cutout_path.exists())


    def invalid_cutout_size(self):
        """Runs cutouts.get_allwise_cutout with a negative image size."""
        coords = SkyCoord(
                        ra="00:00:00", dec="00:00:00.0", unit=(u.hourangle, u.deg), equinox="J2000"
                    )
        hdulist = cutouts.get_allwise_cutout(coords=coords, size=-3.5 * u.arcmin)


    def test_invalid_cutout_size(self):
        """Tests that passing an invalid cutout size raises cutouts.NegativeImageSizeError."""
        self.assertRaises(cutouts.NegativeImageSizeError, self.invalid_cutout_size)


    def invalid_band(self):
        """Runs cutouts.get_allwise_cutout with an invalid WISE band."""
        coords = SkyCoord(
            ra="00:00:00", dec="00:00:00.0", unit=(u.hourangle, u.deg), equinox="J2000"
        )
        hdulist = cutouts.get_allwise_cutout(coords=coords, band="W5", size=3.5 * u.arcmin)


    def test_invalid_band(self):
        """Tests that passing an invalid band raises cutouts.InvalidWISEBandError."""
        self.assertRaises(cutouts.InvalidWISEBandError, self.invalid_band)


if __name__ == "__main__":
    unittest.main()
