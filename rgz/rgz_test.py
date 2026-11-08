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
            f"coord_to_string took {elapsed_per_call} per call (> {_MAX_RUNTIME_PER_COORD_TO_STRING_CALL})",
        )


if __name__ == "__main__":
    unittest.main()
