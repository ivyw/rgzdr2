from unittest import mock
from pathlib import Path
import tempfile
import unittest
import io

import attr
import astropy.table
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd
import numpy as np

from rgz import testutils
from rgz import cutouts


@attr.s
class MockContent:
    content: bytes = attr.ib()


_base_fits_open = fits.open


def mock_get(url: str) -> MockContent:
    if url.startswith("https://irsa.ipac.caltech.edu/SIA"):
        return MockContent(testutils.get_wise_sia_file())

    raise NotImplementedError(f"URL: {url!r}")


def mock_open(file, **kwargs):
    if hasattr(file, "read"):
        return _base_fits_open(file)

    if file.startswith("https://irsa.ipac.caltech.edu/ibe/data/wise/allwise/p3am_cdd"):
        return _base_fits_open(io.BytesIO(testutils.get_wise_image_file()))

    raise NotImplementedError(f"file: {file!r}")


@unittest.skip("Needs mocking + IRSA down")
class TestCutouts(unittest.TestCase):
    """Tests for rgz.cutouts."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()

        self.get_patcher = mock.patch("rgz.cutouts.requests.get", side_effect=mock_get)
        self.mock_get = self.get_patcher.start()

        self.open_patcher = mock.patch("rgz.cutouts.fits.open", side_effect=mock_open)
        self.mock_open = self.open_patcher.start()

    def tearDown(self):
        self.get_patcher.stop()
        self.open_patcher.stop()

    def test_get_cutout(self):
        """Test that cutouts.get_allwise_cutout successfully returns a
        HDUList."""
        coords = SkyCoord(
            ra="02:42:40.71",
            dec="-00:00:47.86",
            unit=(u.hourangle, u.deg),
            equinox="J2000",
        )
        hdulist = cutouts.get_allwise_cutout(coords=coords, size=3.5 * u.arcmin)
        self.assertTrue(len(hdulist) > 0)
        self.mock_get.assert_called_once()

    def test_save_cutout(self):
        """Test that cutout is saved if save_fits is True."""
        cutout_path = Path(self.tempdir.name) / "NGC3997.fits"
        coords = SkyCoord(
            ra="11:57:47.0",
            dec="+25:16:14.00",
            unit=(u.hourangle, u.deg),
            equinox="J2000",
        )
        _ = cutouts.get_allwise_cutout(
            coords=coords, size=10 * u.arcmin, save_fits=True, cutout_path=cutout_path
        )
        self.mock_open.assert_called_once()
        self.assertTrue(cutout_path.exists())

    def test_save_cutout_default_fname(self):
        """Test that cutout is saved if save_fits is True and no filename
        specified."""
        coords = SkyCoord(
            ra="11:57:47.0",
            dec="+25:16:14.00",
            unit=(u.hourangle, u.deg),
            equinox="J2000",
        )
        _ = cutouts.get_allwise_cutout(
            coords=coords, size=10 * u.arcmin, save_fits=True
        )
        default_path = Path(
            f"allwise_{'W1':s}_{coords.ra.value:.4f}_" f"{coords.dec.value:.4f}.fits"
        )
        self.assertTrue(default_path.exists())
        Path.unlink(default_path)

    def test_not_save_cutout(self):
        """Test that cutout is not saved if save_fits is False."""
        cutout_path = Path(self.tempdir.name) / "NGC3997.fits"
        Path.unlink(cutout_path, missing_ok=True)
        coords = SkyCoord(
            ra="11:57:47.0",
            dec="+25:16:14.00",
            unit=(u.hourangle, u.deg),
            equinox="J2000",
        )
        hdulist = cutouts.get_allwise_cutout(
            coords=coords, size=10 * u.arcmin, save_fits=False, cutout_path=cutout_path
        )
        self.assertFalse(cutout_path.exists())

    def test_invalid_cutout_size(self):
        """Tests passing an invalid cutout size raises NegativeImageSizeError."""
        coords = SkyCoord(
            ra="00:00:00", dec="00:00:00.0", unit=(u.hourangle, u.deg), equinox="J2000"
        )

        with self.assertRaises(cutouts.NegativeImageSizeError):
            _ = cutouts.get_allwise_cutout(coords=coords, size=-3.5 * u.arcmin)

    def test_invalid_band(self):
        """Tests that passing an invalid band raises InvalidWISEBandError."""
        coords = SkyCoord(
            ra="00:00:00", dec="00:00:00.0", unit=(u.hourangle, u.deg), equinox="J2000"
        )
        with self.assertRaises(cutouts.InvalidWISEBandError):
            _ = cutouts.get_allwise_cutout(
                coords=coords,
                band="W5",  # pyright: ignore[reportArgumentType]
                size=3.5 * u.arcmin,
            )

    def test_invalid_coords(self):
        """Tests passing an RA/Dec that returns no valid AllWISE images."""
        coords = SkyCoord(
            ra="00:00:00", dec="00:00:00.0", unit=(u.hourangle, u.deg), equinox="J2000"
        )
        with self.assertRaises(cutouts.CutoutNotFoundError):
            _ = cutouts.get_allwise_cutout(coords=coords, size=3.5 * u.arcmin)


if __name__ == "__main__":
    unittest.main()
