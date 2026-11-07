"""Tests for processing RGZ subjects."""

import json
import logging
from pathlib import Path
import tempfile
import time
import unittest

import rgz.consensus
import rgz.constants
import rgz.first
import rgz.subjects
from rgz import testutils

logger = logging.getLogger(__name__)


# Path to test directory.
_TEST_DIR = testutils.get_test_data_dir()

# Path to "cache" data.
_TEST_CACHE_DATA_PATH = _TEST_DIR / testutils.CACHE_DATA_FILENAME

# Path to test (raw) subjects JSON.
_TEST_SUBJECTS_PATH = _TEST_DIR / "subjects.json"

# Path to test (processed) subjects JSON.
_TEST_SUBJECTS_PROCESSED_PATH = _TEST_DIR / testutils.SUBJECTS_PROCESSED_FILENAME


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

    def test_runtime(self):
        """Performance test for processing subjects."""
        first_catalogue = rgz.first.fetch_first_catalogue_from_server_or_cache(
            _TEST_CACHE_DATA_PATH
        )
        first_tree = rgz.first.build_first_tree(first_catalogue)

        with open(_TEST_SUBJECTS_PATH, encoding="utf-8") as f:
            raw_subjects = [json.loads(row) for row in f]

        t0 = time.perf_counter()
        for raw_subject in raw_subjects:
            rgz.subjects.process_subject(
                raw_subject, _TEST_CACHE_DATA_PATH, first_tree
            )
        elapsed = time.perf_counter() - t0

        self.assertLess(
            elapsed,
            0.2,
            f"Processing {len(raw_subjects)} subjects took {elapsed:.2f}s (> 0.2s)",
        )


if __name__ == "__main__":
    unittest.main()
