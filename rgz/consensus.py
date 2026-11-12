"""Aggregates classifications."""

import collections
from collections.abc import Iterable, Sequence
import enum
import json
import logging
from pathlib import Path
from typing import Self

import attr
import disjoint_set
from tqdm.auto import tqdm

from rgz import classifications
from rgz import first
from rgz import partition_aggregation
from rgz import rgz
from rgz import subjects

logger = logging.getLogger(__name__)

_JSON_INDENT = 2
_NAN = float("nan")
_MAX_SUBSET_SIZE = 10


class AggregationMethod(enum.StrEnum):
    DR1 = "dr1"
    PARTITION = "partition"


@attr.s
class DR1ConsensusSource:
    """A radio source with a cross-identified IR host galaxy.

    Attributes:
        zid: Zooniverse ID of the subject containing this source.
        components: FIRST components of the source.
        host_name: Name of the AllWISE host if it exists.
        n_radio_agreement: Number of citizen scientists who agreed
            with the radio component combination.
        n_ir_agreement: Number of citizen scientists who agreed that
            the IR host was an IR host of any radio source. You might
            want to use n_joint_agreement instead of this.
        n_joint_agreement: Number of citizen scientists who agreed with
            the IR host galaxy _and_ the radio component combination.
        votes: Number of citizen scientists who voted on this subject.
    """

    zid: subjects.ZooniverseID = attr.ib()
    components: set[first.FIRSTID] = attr.ib(order=sorted)
    host_name: classifications.ALLWISEID | None = attr.ib()
    n_radio_agreement: int = attr.ib()
    n_ir_agreement: int = attr.ib()
    n_joint_agreement: int = attr.ib()
    n_votes: int = attr.ib()

    def radio_agreement_percent(self) -> float:
        """Gets the percentage radio agreement."""
        return self.n_radio_agreement / self.n_votes

    @classmethod
    def from_json(cls, obj: rgz.JSON) -> Self:
        """Reads a JSON dict."""
        return cls(
            zid=obj["zid"],
            components=set(obj["components"]),
            host_name=obj["host_name"],
            n_radio_agreement=obj["n_radio_agreement"],
            n_ir_agreement=obj["n_ir_agreement"],
            n_joint_agreement=obj["n_joint_agreement"],
            n_votes=obj["votes"],
        )

    def to_json(self) -> rgz.JSON:
        """Converts the DR1ConsensusSource to a JSON-compatible dict."""
        return {
            "zid": self.zid,
            "components": sorted(self.components),
            "host_name": self.host_name,
            "n_radio_agreement": self.n_radio_agreement,
            "n_ir_agreement": self.n_ir_agreement,
            "n_joint_agreement": self.n_joint_agreement,
            "votes": self.n_votes,
        }


@attr.s
class PartitionConsensusSource:
    """A radio source with a cross-identified IR host galaxy.

    Attributes:
        zids: Zooniverse IDs of the subjects containing this source.
        components: FIRST components of the source.
        host_name: Name of the AllWISE host if it exists.
        radio_score: Partition probability of the radio partition.
        ir_score: Maximum-likelihood probability of the consensus IR host.
    """

    zids: set[subjects.ZooniverseID] = attr.ib(order=sorted)
    components: set[first.FIRSTID] = attr.ib(order=sorted)
    host_name: classifications.ALLWISEID | None = attr.ib(default=None, order=False)
    radio_score: float = attr.ib(default=_NAN, order=False)
    ir_score: float = attr.ib(default=_NAN, order=False)

    @classmethod
    def from_json(cls, obj: rgz.JSON) -> Self:
        """Reads a JSON dict."""
        return cls(
            zids=set(obj["zids"]),
            components=set(obj["components"]),
            host_name=obj["host_name"],
            radio_score=obj["radio_score"],
            ir_score=obj["ir_score"],
        )

    def to_json(self) -> rgz.JSON:
        """Converts the PartitionConsensusSource to a JSON-compatible dict."""
        return {
            "zids": sorted(self.zids),
            "components": sorted(self.components),
            "host_name": self.host_name,
            "radio_score": self.radio_score,
            "ir_score": self.ir_score,
        }


def aggregate_subject(
    subject: subjects.Subject,
    classifications: list[classifications.CrossMatchedClassification],
) -> list[DR1ConsensusSource]:
    """Aggregates classifications into consensus sources for a single subject."""
    # Confirm every classification is of the subject.
    for classification in classifications:
        if classification.zid != subject.zid:
            raise ValueError(
                f"Classification is not of subject {subject.zid}:"
                f" {classification.cid} ({classification.zid})"
            )
    # Represent each combination of radio objects by something deterministic and hashable.
    radio_component_combinations = [cl.radio_combinations() for cl in classifications]

    # What's the most common combination (consensus)?
    counter = collections.Counter(sorted(radio_component_combinations))
    ((consensus_radio_combination, consensus_radio_count),) = counter.most_common(1)

    # Amongst people who chose this, what IR was most common?
    radio_source_to_ir_options = {
        source: [] for source in consensus_radio_combination.sources()
    }
    for radio_combination, classification in zip(
        radio_component_combinations, classifications
    ):
        if radio_combination == consensus_radio_combination:
            for ir, radio_source in classification.ir_matches:
                radio_source_to_ir_options[radio_source].append(ir)

    # How many people thought _each_ IR object was a host, regardless
    # of their choice of radio?
    n_ir_votes = collections.Counter()
    for classification in classifications:
        for ir, _ in classification.ir_matches:
            n_ir_votes[ir] += 1

    matches = []
    for radio_source, irs in radio_source_to_ir_options.items():
        ((consensus_ir, consensus_ir_count),) = collections.Counter(
            sorted(irs)
        ).most_common(1)
        matches.append(
            DR1ConsensusSource(
                zid=classifications[0].zid,
                components=set(radio_source),
                host_name=consensus_ir,
                n_joint_agreement=consensus_ir_count,
                n_ir_agreement=n_ir_votes[consensus_ir],
                n_radio_agreement=consensus_radio_count,
                n_votes=len(classifications),
            )
        )

    return matches


def _dr1_consensus(
    all_subjects: Sequence[subjects.Subject],
    all_classifications: Sequence[classifications.CrossMatchedClassification],
) -> list[DR1ConsensusSource]:
    """Aggregates classifications across subjects using the DR1 strategy."""
    zid_to_classifications = collections.defaultdict(list)
    for classification in all_classifications:
        zid_to_classifications[classification.zid].append(classification)

    zid_to_subject = {s.zid: s for s in all_subjects}

    consensuses = []
    while zid_to_classifications:
        zid, zid_classifications = zid_to_classifications.popitem()
        subject = zid_to_subject.pop(zid)
        consensuses.extend(aggregate_subject(subject, zid_classifications))

    return consensuses


def map_components_to_islands(
    all_subjects: Iterable[subjects.Subject],
) -> dict[first.FIRSTID, first.FIRSTID]:
    """Finds components appearing in the same radio island.

    Args:
        all_subjects: RGZ subjects.

    Returns:
        Mapping from each component ID to a representative component ID for its island.
    """
    islands = disjoint_set.DisjointSet()
    all_components = set()

    for subject in all_subjects:
        for radio_island in subject.radio_islands:
            for component in radio_island.firsts:
                all_components.add(component)
            if len(radio_island.firsts) > 1:
                first_component, *other_components = radio_island.firsts
                for other_component in other_components:
                    islands.union(first_component, other_component)

    return {
        component: islands.find(component) if component in islands else component
        for component in all_components
    }


def _partition_consensus(
    all_subjects: Sequence[subjects.Subject],
    all_classifications: Sequence[classifications.CrossMatchedClassification],
    max_subset_size: int = _MAX_SUBSET_SIZE,
) -> list[PartitionConsensusSource]:
    """Aggregates classifications across subjects using a partitioning strategy.

    Args:
        all_subjects: RGZ subjects.
        all_classifications: Cross-matched classifications.
        max_subset_size: Maximum allowable radio islands in a connected clique.

    Returns:
        Consensus radio sources with partition agreement scores.

    Raises:
        ValueError: If a connected clique contains more than `max_subset_size` islands.
    """
    component_to_island = map_components_to_islands(all_subjects)
    island_to_components = collections.defaultdict(set)
    for component, island in tqdm(
        component_to_island.items(), desc="Processing components..."
    ):
        island_to_components[island].add(component)

    # Build cliques based on which radio islands were grouped together by
    # any citizen scientist.
    # TODO(MatthewJA): Union only within each `radio_source` (and filter
    # `island_classifications` to `clique_representatives`) rather than across
    # the entire classification, as disjoint radio sources have zero pair
    # weight (`w_ij = 0`) and factorise independently in the partition function.
    cliques = disjoint_set.DisjointSet()
    component_to_zids = collections.defaultdict(set)
    representative_to_classification_indices = collections.defaultdict(set)

    for i, classification in enumerate(
        tqdm(all_classifications, desc="Constructing cliques...")
    ):
        representatives: list[first.FIRSTID] = []
        for _, radio_source in classification.coord_matches:
            for component in radio_source:
                representative = component_to_island.get(component, component)
                representatives.append(representative)
                component_to_zids[component].add(classification.zid)
                representative_to_classification_indices[representative].add(i)

        if representatives:
            first_representative, *other_representatives = representatives
            for other_representative in other_representatives:
                cliques.union(first_representative, other_representative)

    # For each clique, aggregate a consensus with the partition strategy.
    consensuses: list[PartitionConsensusSource] = []
    for clique_representatives in tqdm(
        cliques.itersets(), desc="Processing cliques..."
    ):
        if len(clique_representatives) > max_subset_size:
            raise ValueError(
                f"Too many islands in clique: {len(clique_representatives)} > {max_subset_size}: "
                f"{sorted(clique_representatives)}"
            )

        subset_indices = {
            i
            for representative in clique_representatives
            for i in representative_to_classification_indices[representative]
        }
        # Convert each set of components into a single canonical representative
        # for the corresponding radio island.
        island_ir_matches = []
        island_partitions = []
        for i in subset_indices:
            matches = []
            partition = []
            for ir_host, radio_source in all_classifications[i].ir_matches:
                islands = frozenset(
                    component_to_island.get(c, c) for c in radio_source
                )
                matches.append((ir_host, islands))
                partition.append(islands)
            island_ir_matches.append(matches)
            island_partitions.append(partition)

        aggregation_result = partition_aggregation.aggregate_subset(
            island_partitions
        )
        host_consensuses = partition_aggregation.aggregate_hosts(
            aggregation_result.aggregation, island_ir_matches
        )

        for block in aggregation_result.aggregation:
            block_components = {
                component
                for representative in block
                for component in island_to_components.get(
                    representative, {representative}
                )
            }
            block_zids = {
                zid
                for component in block_components
                for zid in component_to_zids[component]
            }
            host_consensus = host_consensuses[block]
            consensuses.append(
                PartitionConsensusSource(
                    zids=block_zids,
                    components=block_components,
                    host_name=host_consensus.host_name,
                    radio_score=aggregation_result.scores.get(block, 0.0),
                    ir_score=host_consensus.ir_score,
                )
            )

    return consensuses


def _filter_nofirsts[T](
    matches: list[tuple[T, classifications.RadioSource]],
) -> list[tuple[T, classifications.RadioSource]]:
    """Filters out NOFIRST components and subsequently empty sources."""
    filtered = []
    for ir, radio_source in matches:
        first_only = classifications.RadioSource(
            c for c in radio_source if not c.startswith("NOFIRST_")
        )
        if first_only:
            filtered.append((ir, first_only))
    return filtered


def aggregate(
    subjects_path: Path,
    classifications_path: Path,
    out_path: Path,
    method: AggregationMethod = AggregationMethod.DR1,
    max_subset_size: int = _MAX_SUBSET_SIZE,
) -> None:
    """Aggregates classifications into a consensus for each subject.

    Args:
        subjects_path: Path to reduced subjects JSON.
        classifications_path: Path to reduced, cross-matched subjects JSON.
        out_path: Path to output the consensus JSON.
        method: Aggregation strategy to apply.
        max_subset_size: Maximum radio islands allowed in a subset for partition aggregation.
    """
    # Pop raw subjects in-place to avoid having two copies in memory.
    with open(subjects_path, encoding="utf-8") as f:
        raw_subjects = json.load(f)
    raw_subjects.reverse()
    all_subjects = []
    while raw_subjects:
        all_subjects.append(subjects.Subject.from_json(raw_subjects.pop()))

    with open(classifications_path, encoding="utf-8") as f:
        raw_classifications = json.load(f)
    raw_classifications.reverse()
    all_classifications = []
    while raw_classifications:
        classification = classifications.CrossMatchedClassification.from_json(
            raw_classifications.pop()
        )
        classification.coord_matches = _filter_nofirsts(classification.coord_matches)
        classification.ir_matches = _filter_nofirsts(classification.ir_matches)
        all_classifications.append(classification)

    match method:
        case AggregationMethod.DR1:
            consensuses = _dr1_consensus(all_subjects, all_classifications)
        case AggregationMethod.PARTITION:
            consensuses = _partition_consensus(
                all_subjects, all_classifications, max_subset_size=max_subset_size
            )

    # Delete subjects and classifications to avoid having multiple copies in memory.
    del all_subjects
    del all_classifications
    consensuses.sort()
    jsons = [c.to_json() for c in consensuses]
    del consensuses
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(jsons, f, indent=_JSON_INDENT)
