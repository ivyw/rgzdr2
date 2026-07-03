"""Tests for processing RGZ subjects."""

import json
import logging
import numpy as np
import os
from pathlib import Path
import tempfile
import unittest

import rgz.consensus
import rgz.constants
import rgz.subjects
from rgz import testutils

logger = logging.getLogger(__name__)


# Path to test directory.
_TEST_DIR = Path(os.path.dirname(__file__)) / "testdata/"

# Path to "cache" data.
_TEST_CACHE_DATA_PATH = _TEST_DIR / "first"

# Path to test (raw) subjects JSON.
_TEST_SUBJECTS_PATH = _TEST_DIR / "subjects.json"

# Path to test (processed) subjects JSON.
_TEST_SUBJECTS_PROCESSED_PATH = _TEST_DIR / "subjects_processed.json"


class TestProcess(unittest.TestCase):
    """Tests for rgz.subjects.process."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_regression(self, update: bool = False):
        """Tests behaviour consistency in processing subjects."""
        output_path = self.temp_dir_path / "out.json"
        rgz.subjects.process(_TEST_SUBJECTS_PATH, _TEST_CACHE_DATA_PATH, output_path)
        with open(output_path) as f:
            got = json.load(f)

        if update:
            with open(_TEST_SUBJECTS_PROCESSED_PATH, "w") as f:
                json.dump(got, f)

        with open(_TEST_SUBJECTS_PROCESSED_PATH) as f:
            want = json.load(f)

        self.assertEqual(len(got), len(want))
        for got_, want_ in zip(got, want):
            # Improve error messages by checking each entry individually.
            testutils.assert_json_almost_equal(self, want_, got_)


if __name__ == "__main__":
    unittest.main()
