"""Common testing utilities and constants."""

import inspect
import os
from pathlib import Path

# "Cache" data filename.
CACHE_DATA_FILENAME = "first"

# Test (processed) subjects JSON filename.
SUBJECTS_PROCESSED_FILENAME = "radio_subjects_test_subset_processed.json"

# Test (raw) classifications JSON filename.
CLASSIFICATIONS_FILENAME = "radio_classifications_test_subset.json"

# Test (processed) classifications JSON filename.
CLASSIFICATIONS_PROCESSED_FILENAME = "radio_classifications_test_subset_processed.json"

# Test (matched) classifications JSON filename.
CLASSIFICATIONS_MATCHED_FILENAME = "radio_classifications_test_subset_matched.json"

# Test consensus JSON filename.
CONSENSUS_FILENAME = "consensus.json"


def get_test_data_dir() -> Path:
    """Gets the directory that test data is held in."""
    current_frame = inspect.currentframe()
    if current_frame is not None:
        current_file_path = inspect.getfile(current_frame)
    else:
        current_file_path = __file__
    return Path(os.path.dirname(current_file_path)) / "testdata"
