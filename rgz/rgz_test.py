"""Tests for rgz.rgz utilities."""

import datetime
import logging
import time
import unittest

from astropy.coordinates import SkyCoord
import numpy as np

from rgz import rgz
from rgz import units as u

logger = logging.getLogger(__name__)

_MAX_RUNTIME_PER_COORD_TO_STRING_CALL = datetime.timedelta(microseconds=25)
_MAX_RUNTIME_PER_STRINGS_TO_SKYCOORD_COORD = datetime.timedelta(microseconds=5)


class TestCoordToString(unittest.TestCase):
    """Tests for rgz.coord_to_string and rgz.skycoord_to_string."""

    def test_fuzz_consistency_with_astropy(self):
        """Matches SkyCoord.to_string('hmsdms', sep=' ') across random coordinates."""
        rng = np.random.default_rng(42)
        ras = rng.uniform(0.0, 360.0, 1000)
        decs = rng.uniform(-90.0, 90.0, 1000)

        sky_coords = SkyCoord(ra=ras, dec=decs, unit=(u.deg, u.deg), frame="icrs")
        want_strs = sky_coords.to_string("hmsdms", sep=" ")

        for ra, dec, want in zip(ras, decs, want_strs):
            got = rgz.coord_to_string(ra * u.deg, dec * u.deg)
            self.assertEqual(want, got, f"Mismatch for ra={ra}, dec={dec}")

    def test_edge_cases(self):
        """Matches SkyCoord.to_string('hmsdms', sep=' ') on edge cases."""
        coords = np.array(
            [
                (0.0, 0.0),
                (180.0, -0.0),
                (359.999999999, 90.0),
                (15.0, -90.0),
                (0.0, -0.5),
                (
                    15.0 * (1.0 + 59.0 / 60.0 + 59.999999999 / 3600.0),
                    10.0 + 59.0 / 60.0 + 59.999999999 / 3600.0,
                ),
            ]
        )
        ras = coords[:, 0]
        decs = coords[:, 1]

        sky_coords = SkyCoord(ra=ras, dec=decs, unit=(u.deg, u.deg), frame="icrs")
        want_strs = sky_coords.to_string("hmsdms", sep=" ")

        for ra, dec, want in zip(ras, decs, want_strs):
            got = rgz.coord_to_string(ra * u.deg, dec * u.deg)
            self.assertEqual(want, got, f"Mismatch for ra={ra}, dec={dec}")

    def test_runtime(self):
        """Performance test for coord_to_string."""
        n_calls = 4000
        rng = np.random.default_rng(42)
        coords = [
            (ra * u.deg, dec * u.deg)
            for ra, dec in zip(
                rng.uniform(0.0, 360.0, n_calls),
                rng.uniform(-90.0, 90.0, n_calls),
            )
        ]

        now = time.perf_counter()
        for ra, dec in coords:
            rgz.coord_to_string(ra, dec)
        elapsed = datetime.timedelta(seconds=time.perf_counter() - now)
        elapsed_per_call = elapsed / n_calls
        logger.info(
            "coord_to_string: %.2f us/call (%d calls in %.3fs)",
            elapsed_per_call / datetime.timedelta(microseconds=1),
            n_calls,
            elapsed.total_seconds(),
        )

        self.assertLess(
            elapsed_per_call,
            _MAX_RUNTIME_PER_COORD_TO_STRING_CALL,
            f"coord_to_string took {elapsed_per_call} per call "
            f"(> {_MAX_RUNTIME_PER_COORD_TO_STRING_CALL})",
        )


class TestStringsToSkycoord(unittest.TestCase):
    """Tests for rgz.strings_to_skycoord."""

    def test_fuzz_consistency_with_astropy(self):
        """Matches SkyCoord(coords, unit=('hourangle', 'deg')) across random coordinates."""
        rng = np.random.default_rng(42)
        ras = rng.uniform(0.0, 360.0, 1000)
        decs = rng.uniform(-90.0, 90.0, 1000)
        coord_strs = list(
            SkyCoord(ra=ras, dec=decs, unit=(u.deg, u.deg), frame="icrs").to_string(
                "hmsdms", sep=" "
            )
        )

        want = SkyCoord(coord_strs, unit=("hourangle", "deg"))
        got = rgz.strings_to_skycoord(coord_strs)

        got_ra, got_dec = rgz.get_deg(got)
        want_ra, want_dec = rgz.get_deg(want)
        np.testing.assert_allclose(got_ra, want_ra, rtol=0, atol=1e-12)
        np.testing.assert_allclose(got_dec, want_dec, rtol=0, atol=1e-12)

    def test_edge_cases(self):
        """Matches SkyCoord(coords, unit=('hourangle', 'deg')) on edge cases."""
        coord_strs = [
            "00 00 00 +00 00 00",
            "12 00 00 -00 00 00",
            "00 00 00 -00 30 00",
            "00 00 00 -00 00 00.5",
            "23 59 59.99999999 +90 00 00",
            "01 00 00 -90 00 00",
        ]

        want = SkyCoord(coord_strs, unit=("hourangle", "deg"))
        got = rgz.strings_to_skycoord(coord_strs)

        got_ra, got_dec = rgz.get_deg(got)
        want_ra, want_dec = rgz.get_deg(want)
        np.testing.assert_allclose(got_ra, want_ra, rtol=0, atol=1e-12)
        np.testing.assert_allclose(got_dec, want_dec, rtol=0, atol=1e-12)

    def test_runtime(self):
        """Performance test for strings_to_skycoord."""
        n_coords = 20000
        rng = np.random.default_rng(42)
        ras = rng.uniform(0.0, 360.0, n_coords)
        decs = rng.uniform(-90.0, 90.0, n_coords)
        coord_strs = [
            rgz.coord_to_string(ra * u.deg, dec * u.deg)
            for ra, dec in zip(ras, decs)
        ]

        now = time.perf_counter()
        rgz.strings_to_skycoord(coord_strs)
        elapsed = datetime.timedelta(seconds=time.perf_counter() - now)
        elapsed_per_coord = elapsed / n_coords
        logger.info(
            "strings_to_skycoord: %.2f us/coord (%d coords in %.3fs)",
            elapsed_per_coord / datetime.timedelta(microseconds=1),
            n_coords,
            elapsed.total_seconds(),
        )

        self.assertLess(
            elapsed_per_coord,
            _MAX_RUNTIME_PER_STRINGS_TO_SKYCOORD_COORD,
            f"strings_to_skycoord took {elapsed_per_coord} per coord "
            f"(> {_MAX_RUNTIME_PER_STRINGS_TO_SKYCOORD_COORD})",
        )


if __name__ == "__main__":
    unittest.main()
