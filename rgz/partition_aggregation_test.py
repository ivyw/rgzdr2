"""Tests for pure partition aggregation."""

import datetime
import logging
import math
import time
import unittest

from rgz import partition_aggregation

logger = logging.getLogger(__name__)

_MAX_RUNTIME_PER_AGGREGATE_SUBSET_CALL = datetime.timedelta(milliseconds=4)


class TestPartitionAggregation(unittest.TestCase):
    """Tests for rgz.partition_aggregation."""

    def test_aggregate_subset(self):
        """Tests mathematical partition aggregation on radio partitions."""
        block_ab = frozenset({"A", "B"})
        block_c = frozenset({"C"})
        radio_partitions = [
            [block_ab, block_c],
            [block_ab, block_c],
            [frozenset({"A"}), frozenset({"B", "C"})],
        ]

        aggregation = partition_aggregation.aggregate_subset(radio_partitions)
        self.assertEqual(
            aggregation.aggregation, frozenset({block_ab, block_c})
        )
        self.assertIn(block_ab, aggregation.scores)
        self.assertIn(block_c, aggregation.scores)

    def test_large_unanimous_clique_no_overflow(self):
        """Tests that a 10-element unanimous clique does not overflow float64."""
        expected_block = frozenset(f"A{i}" for i in range(10))
        radio_partitions = [[expected_block] for _ in range(5)]

        aggregation = partition_aggregation.aggregate_subset(radio_partitions)
        self.assertEqual(aggregation.aggregation, frozenset({expected_block}))
        self.assertFalse(math.isnan(aggregation.probability))
        self.assertAlmostEqual(aggregation.probability, 1.0)
        self.assertAlmostEqual(aggregation.scores[expected_block], 1.0)

    def test_runtime(self):
        """Performance test for aggregate_subset on an 8-element clique."""
        partition_majority = [
            frozenset({"A", "B", "C"}),
            frozenset({"D", "E"}),
            frozenset({"F", "G", "H"}),
        ]
        partition_minority = [
            frozenset({"A", "B"}),
            frozenset({"C", "D", "E"}),
            frozenset({"F", "G"}),
            frozenset({"H"}),
        ]
        radio_partitions = [
            partition_majority if i % 3 != 0 else partition_minority
            for i in range(20)
        ]

        n_calls = 20
        now = time.perf_counter()
        for _ in range(n_calls):
            partition_aggregation.aggregate_subset(radio_partitions)
        elapsed = datetime.timedelta(seconds=time.perf_counter() - now)
        elapsed_per_call = elapsed / n_calls
        logger.info(
            "aggregate_subset: %.2f ms/call (%d calls in %.3fs)",
            elapsed_per_call / datetime.timedelta(milliseconds=1),
            n_calls,
            elapsed.total_seconds(),
        )

        self.assertLess(
            elapsed_per_call,
            _MAX_RUNTIME_PER_AGGREGATE_SUBSET_CALL,
            f"aggregate_subset took {elapsed_per_call} per call "
            f"(> {_MAX_RUNTIME_PER_AGGREGATE_SUBSET_CALL})",
        )


class TestAggregateHosts(unittest.TestCase):
    """Tests for rgz.partition_aggregation.aggregate_hosts."""

    def test_em_disambiguates_merged_block(self):
        """Tests that EM assigns a merged annotation to the block matching its host."""
        block_a = frozenset({"A1"})
        block_b = frozenset({"B1"})
        partition = frozenset({block_a, block_b})

        ir_matches = [
            [("WISE_A", block_a), ("WISE_B", block_b)],
            [("WISE_A", block_a), ("WISE_B", block_b)],
            [("WISE_A", block_a), ("WISE_B", block_b)],
            # Merged annotation assigned to block_a via WISE_A.
            [("WISE_A", block_a | block_b)],
        ]

        host_consensuses = partition_aggregation.aggregate_hosts(
            partition, ir_matches
        )
        self.assertEqual(host_consensuses[block_a].host_name, "WISE_A")
        self.assertAlmostEqual(host_consensuses[block_a].ir_score, 1.0)
        self.assertEqual(host_consensuses[block_b].host_name, "WISE_B")
        self.assertAlmostEqual(host_consensuses[block_b].ir_score, 1.0)

    def test_partial_visibility(self):
        """Tests that unseen islands in a block do not reduce its Jaccard weight."""
        island_a1 = frozenset({"A1"})
        block_a = island_a1 | frozenset({"A2"})
        block_b = frozenset({"B1"})
        partition = frozenset({block_a, block_b})

        ir_matches = [
            # A2 is unseen, so {A1, B1} weights block_a and block_b equally (1/2 each).
            [("WISE_1", island_a1 | block_b)],
            [("WISE_2", block_a), ("WISE_2", block_b)],
        ]

        host_consensuses = partition_aggregation.aggregate_hosts(
            partition, ir_matches
        )
        self.assertEqual(host_consensuses[block_a].host_name, "WISE_2")
        self.assertEqual(host_consensuses[block_b].host_name, "WISE_2")
        self.assertAlmostEqual(
            host_consensuses[block_a].ir_score,
            host_consensuses[block_b].ir_score,
        )
        self.assertAlmostEqual(host_consensuses[block_a].ir_score, 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
