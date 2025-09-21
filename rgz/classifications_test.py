"""Tests for processing RGZ classifications."""

import inspect
import json
import os
from pathlib import Path
import tempfile
import unittest

from python.runfiles import Runfiles

import rgz.classifications

# "Cache" data filename.
_TEST_CACHE_DATA_FILENAME = "first"

# Test (processed) subjects JSON filename.
_TEST_SUBJECTS_PROCESSED_FILENAME = "radio_subjects_test_subset_processed.json"

# Test (raw) classifications JSON filename.
_TEST_CLASSIFICATIONS_FILENAME = "radio_classifications_test_subset.json"

# Test (processed) classifications JSON filename.
_TEST_CLASSIFICATIONS_PROCESSED_FILENAME = (
    "radio_classifications_test_subset_processed.json"
)


def get_test_data_dir() -> Path:
    """Gets the directory that test data is held in."""
    current_frame = inspect.currentframe()
    if current_frame is not None:
        current_file_path = inspect.getfile(current_frame)
    else:
        current_file_path = __file__
    return Path(os.path.dirname(current_file_path)) / "testdata"


class TestProcess(unittest.TestCase):
    """Tests for rgz.classifications.process."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)
        self.test_data_path = get_test_data_dir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_regression(self):
        """Tests behaviour consistency in processing classifications."""
        output_path = self.temp_dir_path / "out.json"
        rgz.classifications.process(
            self.test_data_path / _TEST_CLASSIFICATIONS_FILENAME,
            self.test_data_path / _TEST_SUBJECTS_PROCESSED_FILENAME,
            self.test_data_path / _TEST_CACHE_DATA_FILENAME,
            output_path,
        )
        with open(output_path) as f:
            got = json.load(f)
        with open(self.test_data_path / _TEST_CLASSIFICATIONS_PROCESSED_FILENAME) as f:
            want = json.load(f)
        self.assertEqual(want, got)


if __name__ == "__main__":
    unittest.main()
