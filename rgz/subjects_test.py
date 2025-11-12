"""Tests for processing RGZ subjects."""

import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from python.runfiles import Runfiles

import rgz.subjects
import rgz.units as u

# Path to test directory.
_TEST_DIR = Path(os.path.dirname(__file__)) / "testdata/"

# Path to "cache" data.
_TEST_CACHE_DATA_PATH = _TEST_DIR / "first"

# Path to test (raw) subjects JSON.
_TEST_SUBJECTS_PATH = _TEST_DIR / "radio_subjects_test_subset.json"

# Path to test (processed) subjects JSON.
_TEST_SUBJECTS_PROCESSED_PATH = _TEST_DIR / "radio_subjects_test_subset_processed.json"


class TestFindPointsInBox(unittest.TestCase):
    """Tests for rgz.subjects.find_points_in_box."""

    def test_simple(self):
        lower_ra = lower_dec = 0.0 * u.deg
        upper_ra = upper_dec = 1.0 * u.deg
        points = np.array(
            [
                [0.5, 0.5],
                [0.2, 0.9],
                [-0.1, 0.1],
                [0.1, -0.1],
                [-0.1, -0.1],
            ]
        ) * u.deg
        want = [0, 1]
        got = rgz.subjects.find_points_in_box(
            points, lower_ra, upper_ra, lower_dec, upper_dec
        )
        self.assertSetEqual(set(got), set(want))

    def test_ra_boundary(self):
        lower_dec = -1.0 * u.deg
        upper_dec = 1.0 * u.deg
        lower_ra = 359.9 * u.deg
        upper_ra = 0.1 * u.deg
        points = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
            ]
        ) * u.deg
        want = [0]
        got = rgz.subjects.find_points_in_box(
            points, lower_ra, upper_ra, lower_dec, upper_dec
        )
        self.assertSetEqual(set(got), set(want))


class TestProcess(unittest.TestCase):
    """Tests for rgz.subjects.process."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_regression(self):
        """Tests behaviour consistency in processing subjects."""
        output_path = self.temp_dir_path / "out.json"
        rgz.subjects.process(_TEST_SUBJECTS_PATH, _TEST_CACHE_DATA_PATH, output_path)
        with open(output_path) as f:
            got = json.load(f)
        with open(_TEST_SUBJECTS_PROCESSED_PATH) as f:
            want = json.load(f)
        self.assertEqual(want, got)


if __name__ == "__main__":
    unittest.main()
