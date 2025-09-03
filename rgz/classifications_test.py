"""Tests for processing RGZ classifications."""

import json
import os
from pathlib import Path
import tempfile
import unittest

from python.runfiles import Runfiles

import rgz.classifications

# Path to test directory.
_TEST_DIR = Path(os.path.dirname(__file__)) / "testdata/"

# Path to "cache" data.
_TEST_CACHE_DATA_PATH = _TEST_DIR / "first"

# Path to test (processed) subjects JSON.
_TEST_SUBJECTS_PROCESSED_PATH = _TEST_DIR / "radio_subjects_test_subset_processed.json"

# Path to test (raw) classifications JSON.
_TEST_CLASSIFICATIONS_PATH = _TEST_DIR / "radio_classifications_test_subset.json"

# Path to test (processed) classifications JSON.
_TEST_CLASSIFICATIONS_PROCESSED_PATH = (
    _TEST_DIR / "radio_classifications_test_subset_processed.json"
)


class TestProcess(unittest.TestCase):
    """Tests for rgz.classifications.process."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_regression(self):
        """Tests behaviour consistency in processing classifications."""
        output_path = self.temp_dir_path / "out.json"
        rgz.classifications.process(
            _TEST_CLASSIFICATIONS_PATH,
            _TEST_SUBJECTS_PROCESSED_PATH,
            _TEST_CACHE_DATA_PATH,
            output_path,
        )
        with open(output_path) as f:
            got = json.load(f)
        with open(_TEST_CLASSIFICATIONS_PROCESSED_PATH) as f:
            want = json.load(f)
        self.assertEqual(want, got)


if __name__ == "__main__":
    unittest.main()
