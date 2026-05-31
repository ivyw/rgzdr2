"""Common testing utilities and constants."""

import inspect
import os
from pathlib import Path

# "Cache" data filename.
CACHE_DATA_FILENAME = "first"

# Test (processed) subjects JSON filename.
SUBJECTS_PROCESSED_FILENAME = "subjects_processed.json"

# Test (raw) classifications JSON filename.
CLASSIFICATIONS_FILENAME = "classifications.json"

# Test (processed) classifications JSON filename.
CLASSIFICATIONS_PROCESSED_FILENAME = "classifications_processed.json"

# Test (matched) classifications JSON filename.
CLASSIFICATIONS_MATCHED_FILENAME = "classifications_matched.json"

# Test consensus JSON filename.
CONSENSUS_FILENAME = "consensus.json"

# Test WISE SIA filename.
WISE_SIA_FILENAME = "wise.fits"

# Test WISE image filename.
WISE_IMAGE_FILENAME = "wise_image.fits"


def get_test_data_dir() -> Path:
    """Gets the directory that test data is held in."""
    current_frame = inspect.currentframe()
    if current_frame is not None:
        current_file_path = inspect.getfile(current_frame)
    else:
        current_file_path = __file__
    return Path(os.path.dirname(current_file_path)) / "testdata"


def get_wise_sia_file() -> bytes:
    with open(get_test_data_dir() / WISE_SIA_FILENAME, 'rb') as f:
        return f.read()


def get_wise_image_file() -> bytes:
    with open(get_test_data_dir() / WISE_IMAGE_FILENAME, 'rb') as f:
        return f.read()