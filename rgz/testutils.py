"""Common testing utilities and constants."""

import collections
from collections.abc import Mapping
import inspect
import os
from pathlib import Path
from typing import Any, Hashable
import unittest

import astropy.table

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

# Test WISE (IRSA) query results table filename.
WISE_LOOKUP_FILENAME = "wise_lookup.ecsv"

# Test WISE SIA filename.
WISE_SIA_FILENAME = "wise.fits"

# Test WISE image filename.
WISE_IMAGE_FILENAME = "wise_image.fits"

# Valid types that can deserialise from JSON.
type _JSON_ENTITY = list[_JSON_ENTITY] | Mapping[
    str, _JSON_ENTITY
] | float | int | str | None


def get_test_data_dir() -> Path:
    """Gets the directory that test data is held in."""
    current_frame = inspect.currentframe()
    if current_frame is not None:
        current_file_path = inspect.getfile(current_frame)
    else:
        current_file_path = __file__
    return Path(os.path.dirname(current_file_path)) / "testdata"


def get_wise_sia_file() -> bytes:
    with open(get_test_data_dir() / WISE_SIA_FILENAME, "rb") as f:
        return f.read()


def get_wise_image_file() -> bytes:
    with open(get_test_data_dir() / WISE_IMAGE_FILENAME, "rb") as f:
        return f.read()


def get_wise_irsa_query_result() -> astropy.table.Table:
    return astropy.io.ascii.read(get_test_data_dir() / WISE_LOOKUP_FILENAME)


def _subtractable(a: Any) -> bool:
    return hasattr(a, "__sub__")


def _make_hashable(a: _JSON_ENTITY) -> Hashable:
    if isinstance(a, list):
        return tuple(_make_hashable(v) for v in a)
    if isinstance(a, Mapping):
        return tuple((k, _make_hashable(v)) for k, v in a.items())
    if isinstance(a, float):
        return round(a, 7)
    return a


def assert_json_almost_equal(t: unittest.TestCase, da: _JSON_ENTITY, db: _JSON_ENTITY):
    """Asserts that two JSON-deserialised items are almost equal.

    Handles floating point precision.

    Args:
        da: first item
        db: second item
    """
    try:
        _assert_json_almost_equal(t, da, db)
    except AssertionError as e:
        raise AssertionError(f"{da} != {db}") from e


def _assert_json_almost_equal(t: unittest.TestCase, da: _JSON_ENTITY, db: _JSON_ENTITY):
    # Unfortunately we have to hardcode types here
    # as strings and lists look ~identical in ducktyping.
    # At least we only use this for testing!
    if isinstance(da, dict) and isinstance(db, dict):
        keys_a, keys_b = da.keys(), db.keys()
        t.assertEqual(keys_a, keys_b)
        values_a = [da[k] for k in sorted(keys_a)]
        values_b = [db[k] for k in sorted(keys_b)]
        assert_json_almost_equal(t, values_a, values_b)
        return
    if isinstance(da, list) and isinstance(db, list):
        t.assertEqual(len(da), len(db))
        da_ = [_make_hashable(a) for a in da]
        db_ = [_make_hashable(b) for b in db]
        counter_a = collections.Counter(da_)
        counter_b = collections.Counter(db_)
        for k in counter_a.keys() | counter_b.keys():
            a = counter_a[k]
            b = counter_b[k]
            assert_json_almost_equal(t, a, b)
        return

    if _subtractable(da) and _subtractable(db):
        t.assertAlmostEqual(da, db)  # type: ignore
        return

    t.assertEqual(da, db)
    return
