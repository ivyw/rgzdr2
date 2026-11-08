"""Partition aggregation strategies for classifications."""

from collections.abc import Collection, Iterable, Sequence
import dataclasses
import functools
import itertools
import math

import disjoint_set
import numpy as np

from rgz.classifications import Classification
from rgz.first import FIRSTID

# A big number for situations where everyone agrees.
VERY_BIG = 1e10


def get_all_first_ids(classifications: Sequence[Classification]) -> Collection[FIRSTID]:
    """Gets all FIRST IDs touched by a set of classifications."""
    all_names = set()
    for c in classifications:
        for _, radios in c.coord_matches:
            for r in radios:
                all_names.add(r)
    return all_names


def powerset[T](iterable: Iterable[T]) -> frozenset[frozenset[T]]:
    """Iterates over the powerset."""
    s = list(iterable)
    return frozenset(
        {
            frozenset(s)
            for s in itertools.chain.from_iterable(
                itertools.combinations(s, r) for r in range(len(s) + 1)
            )
        }
    )


def deterministic_powerset[T](iterable: Iterable[T]) -> list[frozenset[T]]:
    """Returns all subsets of an iterable in a deterministic order.

    Args:
        iterable: Elements to generate the powerset from.

    Returns:
        Sorted powerset.
    """
    elements = sorted(iterable)
    return [
        frozenset(subset)
        for r in range(len(elements) + 1)
        for subset in itertools.combinations(elements, r)
    ]


@dataclasses.dataclass
class Aggregation:
    aggregation: frozenset[frozenset[FIRSTID]]
    scores: dict[frozenset[FIRSTID], float]
    probability: float


def aggregate_subset(
    classifications: Sequence[Classification],
) -> Aggregation:
    """Aggregates a subset of classifications using dynamic programming.

    Args:
        classifications: Subset of classifications.

    Returns:
        Consensus.
    """
    # Convert FIRST IDs into indices.
    all_names = sorted(get_all_first_ids(classifications))
    name_to_index = {name: i for i, name in enumerate(all_names)}

    n = len(all_names)
    n_observed_same = np.zeros((n, n))
    n_observed_total = np.zeros((n, n))
    for c in classifications:
        ds = disjoint_set.DisjointSet()
        for _, radios in c.coord_matches:
            for r1, r2 in itertools.product(radios, radios):
                ds.union(name_to_index[r1], name_to_index[r2])

        all_components = {i for component in ds.itersets() for i in component}
        for i, j in itertools.permutations(all_components, 2):
            n_observed_total[i, j] += 1
        for component in ds.itersets():
            for i, j in itertools.permutations(component, 2):
                n_observed_same[i, j] += 1

    # TODO: Figure out how to handle unobserved pairs.
    # It'd be nice to use something other than a zero prior.
    # We could do a prior based on distance, perhaps.
    # This situation will fail for large galaxies where a-b and b-c are
    # seen together, but never a-c. This effectively hard-caps our
    # galaxies at 4.5' long, soft cap at 3'.
    unseen_together = n_observed_total == 0
    n_observed_same[unseen_together] = 0
    n_observed_total[unseen_together] = 1

    p_same_partition = n_observed_same / n_observed_total
    p_one = p_same_partition == 1
    with np.errstate(divide="ignore"):
        log_weights = np.where(
            ~p_one,
            np.log(p_same_partition / (1 - p_same_partition)),
            math.log(VERY_BIG),
        )

    # [Logsumexp, max] for each partitioning in S: (logsumexp = log Z, max = solution)
    #   Sum for each partition in the partitioning:
    #       Sum for each unique pair of elements in the partition:
    #           log_weights[i, j]

    # To find the max for a set S:
    #   Arbitrarily choose an element k from S
    #   For each possible subset T of S \ {k}:
    #       T = T | {k}
    #       Compute the sum for T
    #       Add to the biggest sum of S \ T
    #       Store the biggest

    @functools.cache
    def log_weight_sum(subset: frozenset[FIRSTID]) -> float:
        """∑_{ij} log(w_{ij})"""
        total = 0.0
        for i, j in itertools.combinations(subset, 2):
            total += log_weights[name_to_index[i], name_to_index[j]]
        return float(total)

    @dataclasses.dataclass
    class PartitionProduct:
        max_log_product: float
        log_partition_function: float
        best_partition: frozenset[frozenset[FIRSTID]]

    @functools.cache
    def compute_partition_product(s: frozenset[FIRSTID]) -> PartitionProduct:
        if not s:
            return PartitionProduct(0.0, 0.0, frozenset())

        max_log_product = -math.inf
        log_z_terms: list[float] = []
        maximiser = frozenset()
        element = min(s)
        for subset in deterministic_powerset(s - {element}):
            subset = subset | {element}
            lw = log_weight_sum(subset)
            if lw == -math.inf:
                continue
            pp = compute_partition_product(s - subset)
            log_product = lw + pp.max_log_product
            log_z_terms.append(lw + pp.log_partition_function)
            if log_product > max_log_product:
                max_log_product = log_product
                maximiser = pp.best_partition | {subset}

        if not maximiser:
            raise RuntimeError()

        max_log_z = max(log_z_terms)
        log_partition_function = max_log_z + math.log(
            sum(math.exp(term - max_log_z) for term in log_z_terms)
        )

        return PartitionProduct(max_log_product, log_partition_function, maximiser)

    elements = frozenset(all_names)
    pp = compute_partition_product(elements)

    block_probabilities = {}
    for block in pp.best_partition:
        log_block_prob = (
            log_weight_sum(block)
            + compute_partition_product(elements - block).log_partition_function
            - pp.log_partition_function
        )
        block_probabilities[block] = math.exp(log_block_prob)

    return Aggregation(
        aggregation=pp.best_partition,
        scores=block_probabilities,
        probability=math.exp(pp.max_log_product - pp.log_partition_function),
    )
