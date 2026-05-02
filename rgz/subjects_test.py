"""Tests for processing RGZ subjects."""

import json
import os
from pathlib import Path
import tempfile
import unittest

import rgz.consensus
import rgz.constants
import rgz.subjects

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
        self.assertEqual(want, got)


# TODO(hzovaro): add test for serialisation/deserialisation - in particular that
# transforming a WCS from WCS object -> string -> object doesn't result in any
# problematic differences


if __name__ == "__main__":
    unittest.main()
