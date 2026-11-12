"""Partition aggregation strategies for classifications."""

from collections.abc import Collection, Iterable, Sequence
import dataclasses
import functools
import itertools
import math

import disjoint_set
import numpy as np

from rgz.classifications import ALLWISEID
from rgz.first import FIRSTID

type IslandID = FIRSTID
type Block = frozenset[IslandID]
type Partition = frozenset[Block]

# A big number for situations where everyone agrees.
VERY_BIG = 1e10
# Convergence tolerance for expectation-maximisation.
_EM_CONVERGENCE_TOL = 1e-9
# Number of iterations for expectation-maximisation.
_EM_MAX_ITERATIONS = 1000


def get_all_island_ids(
    radio_partitions: Sequence[Collection[Block]],
) -> set[IslandID]:
    """Gets all island IDs touched by a sequence of radio partitions."""
    return {
        island_id
        for partition in radio_partitions
        for block in partition
        for island_id in block
    }


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
    aggregation: Partition
    scores: dict[Block, float]
    probability: float


@dataclasses.dataclass(frozen=True)
class HostConsensus:
    """Consensus IR host galaxy and agreement score for a radio block."""

    host_name: ALLWISEID | None
    ir_score: float


def aggregate_subset(
    radio_partitions: Sequence[Collection[Block]],
) -> Aggregation:
    """Aggregates a subset of radio partitions using dynamic programming.

    Args:
        radio_partitions: Annotated radio island blocks per classification.

    Returns:
        Consensus.
    """
    # Convert island IDs into indices.
    all_names = sorted(get_all_island_ids(radio_partitions))
    name_to_index = {name: i for i, name in enumerate(all_names)}

    n = len(all_names)
    n_observed_same = np.zeros((n, n))
    n_observed_total = np.zeros((n, n))
    for partition in radio_partitions:
        ds = disjoint_set.DisjointSet()
        for block in partition:
            for r1, r2 in itertools.product(block, block):
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
    def log_weight_sum(subset: Block) -> float:
        """∑_{ij} log(w_{ij})"""
        total = 0.0
        for i, j in itertools.combinations(subset, 2):
            total += log_weights[name_to_index[i], name_to_index[j]]
        return float(total)

    @dataclasses.dataclass
    class PartitionProduct:
        max_log_product: float
        log_partition_function: float
        best_partition: Partition

    @functools.cache
    def compute_partition_product(s: Block) -> PartitionProduct:
        if not s:
            return PartitionProduct(0.0, 0.0, frozenset())

        max_log_product = -math.inf
        log_z_terms: list[float] = []
        maximiser: Partition = frozenset()
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

    block_probabilities: dict[Block, float] = {}
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


def aggregate_hosts(
    partition: Partition,
    ir_matches: Sequence[Sequence[tuple[ALLWISEID, Block]]],
) -> dict[Block, HostConsensus]:
    """Estimates consensus IR host galaxies for a fixed radio partition via EM.

    Args:
        partition: Consensus radio partition `t* = {b_1, ..., b_K}`.
        ir_matches: Annotated `(ir_host, radio_block)` pairs per classification.

    Returns:
        Mapping from each consensus radio block `b_k` to its `HostConsensus`.
    """
    if not partition:
        return {}

    blocks = sorted(partition, key=sorted)
    clique_islands = frozenset().union(*blocks)

    annotations: list[tuple[ALLWISEID, Block, Block]] = []
    for matches in ir_matches:
        pairs = [
            (ir_host, block & clique_islands)
            for ir_host, block in matches
        ]
        pairs = [(ir_host, block) for ir_host, block in pairs if block]
        if pairs:
            visible_islands = frozenset().union(*(block for _, block in pairs))
            for ir_host, block in pairs:
                annotations.append((ir_host, block, visible_islands))

    if not annotations:
        return {
            block: HostConsensus(host_name=None, ir_score=math.nan)
            for block in blocks
        }

    hosts = sorted({ir_host for ir_host, _, _ in annotations})
    host_to_index = {ir_host: idx for idx, ir_host in enumerate(hosts)}
    host_indices = np.array(
        [host_to_index[ir_host] for ir_host, _, _ in annotations]
    )
    host_mask = np.eye(len(hosts), dtype=np.float64)[host_indices].T

    # Jaccard agreement w_k(b') = |b' ∩ b_k| / |b' ∪ (b_k ∩ S')|.
    jaccard_weights = np.array(
        [
            [
                len(annotated_block & consensus_block)
                / len(annotated_block | (consensus_block & visible_islands))
                if annotated_block & consensus_block
                else 0.0
                for consensus_block in blocks
            ]
            for _, annotated_block, visible_islands in annotations
        ],
        dtype=np.float64,
    )

    def m_step(responsibilities: np.ndarray) -> np.ndarray:
        weighted_counts = host_mask @ responsibilities
        column_totals = weighted_counts.sum(axis=0, keepdims=True)
        return np.divide(
            weighted_counts,
            column_totals,
            out=np.zeros_like(weighted_counts),
            where=column_totals > 0,
        )

    host_probs = m_step(jaccard_weights)
    for _ in range(_EM_MAX_ITERATIONS):
        joint_probs = host_probs[host_indices] * jaccard_weights
        row_totals = joint_probs.sum(axis=1, keepdims=True)
        responsibilities = np.divide(
            joint_probs,
            row_totals,
            out=np.zeros_like(joint_probs),
            where=row_totals > 0,
        )
        next_host_probs = m_step(responsibilities)
        if np.max(np.abs(next_host_probs - host_probs)) < _EM_CONVERGENCE_TOL:
            host_probs = next_host_probs
            break
        host_probs = next_host_probs

    consensus_by_block: dict[Block, HostConsensus] = {}
    for block_idx, consensus_block in enumerate(blocks):
        block_probs = host_probs[:, block_idx]
        if block_probs.sum() == 0:
            consensus_by_block[consensus_block] = HostConsensus(
                host_name=None, ir_score=math.nan
            )
        else:
            best_host_idx = int(np.argmax(block_probs))
            consensus_by_block[consensus_block] = HostConsensus(
                host_name=hosts[best_host_idx],
                ir_score=float(block_probs[best_host_idx]),
            )

    return consensus_by_block

