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
        
        self.assertEqual(want, got)
        return 

        subject_ids_old = [s["id"] for s in want]
        subject_ids_new = [s["id"] for s in got]
        self.assertEqual(set(subject_ids_new), set(subject_ids_old))

        # Sort
        want = [want[ii] for ii in np.argsort(subject_ids_old)]
        got = [got[ii] for ii in np.argsort(subject_ids_new)]

        # Check len is the same
        self.assertTrue(len(want), len(got))

        # Check attributes are the same
        ii = 0
        for subject_old, subject_new in zip(want, got):
            self.assertEqual(subject_old["id"], subject_new["id"])
            self.assertEqual(subject_old["zid"], subject_new["zid"])
            self.assertEqual(subject_old["coords"], subject_new["coords"])
            self.assertEqual(subject_old["wcs"], subject_new["wcs"])

            # Check same number of radio_islands
            breakpoint()
            if not (len(subject_old["bboxes"]) == len(subject_new["radio_islands"])):
                breakpoint()
            self.assertEqual(
                len(subject_old["bboxes"]),
                len(subject_new["radio_islands"])
            )

            # Check firsts are the same
            for bbox_old, ri_new in zip(
                subject_old["bboxes"],
                subject_new["radio_islands"],
            ):
                self.assertEqual(
                    bbox_old["bbox"],
                    [c - 1 for c in ri_new["rgzbbox"]]  # NOTE: need to subtract 1 from these since in the code that produced the test data these coords had already been shifted so that the origin was (0, 0).
                )
                # We don't try to compare NOFIRSTs since we've changed the string
                # formatting so the comparison will always fail
                old_firsts = set([f for f in bbox_old["first"] if not f.startswith("NOFIRST")])
                new_firsts = set([f for f in ri_new["firsts"] if not f.startswith("NOFIRST")])
                # print("Old: "); print(old_firsts); print("New: "); print(new_firsts)
                self.assertEqual(old_firsts, new_firsts)

            logger.warning(f"Test passed for subject {ii}!")
            ii += 1



if __name__ == "__main__":
    unittest.main()
