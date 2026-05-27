"""Handles RGZ classifications."""

import collections
from collections.abc import Generator, Iterable
import functools
import json
import logging
import multiprocessing
from pathlib import Path
from typing import Self

import attr
import numpy as np
import numpy.typing as npt
from astropy.coordinates import SkyCoord
import astropy.table
import astropy.wcs
import pyvo
from tqdm import tqdm

from rgz import constants
from rgz import radioislands
from rgz import rgz
from rgz import subjects
from rgz import units as u

# Indent of output JSON files.
_JSON_INDENT = 2

logger = logging.getLogger(__name__)

type ALLWISEID = str


def get_classifications(path: Path) -> Generator[rgz.JSON]:
    """Yields classifications from RGZ.

    Args:
        path: Path to RGZ classifications JSON dumped from MongoDB.
            Each line should be a separate JSON document.

    Yields:
        Classification JSON documents.

    Raises:
        ValueError if a classification document is invalid.
    """
    with open(path, encoding="utf-8") as f:
        # Each row is a JSON document.
        for row in f:
            js = json.loads(row)
            for anno in js["annotations"]:
                if "radio" in anno:
                    break
            else:
                raise ValueError(f"Classification has no annotations: {js}")
            yield js


class RadioSource(tuple[radioislands.FIRSTID]):
    """Represents a unique set of radio components.

    Invariant: Always sorted, unique entries.
    """

    def __new__(cls, radio_source: Iterable[radioislands.FIRSTID]) -> Self:
        return super().__new__(cls, sorted(set(radio_source)))

    def __repr__(self) -> str:
        tuple_repr = super().__repr__()
        return f"RadioSource({tuple_repr})"

    def components(self) -> frozenset[radioislands.FIRSTID]:
        """Gets radio components in this source."""
        return frozenset(self)


class RadioSourceCombination(tuple[RadioSource]):
    """Identifies a unique combination of radio components.

    Invariant: Always sorted, unique entries.
    """

    def __new__(
        cls: type[Self], radio_combinations: Iterable[Iterable[radioislands.FIRSTID]]
    ) -> Self:
        representations = []
        for radios in radio_combinations:
            representations.append(RadioSource(radios))
        return super().__new__(cls, sorted(representations))

    def __repr__(self) -> str:
        tuple_repr = super().__repr__()
        return f"RadioCombination({tuple_repr})"

    def sources(self) -> frozenset[RadioSource]:
        """Gets radio sources in this combination.

        A radio source is a collection of radio components that are all part of
        the same physical object.
        """
        return frozenset(self)


@attr.s
class Classification:
    """A single RGZ classification of a subject by a citizen scientist.

    Attributes:
        cid: Classification MongoDB ID.
        zid: Zooniverse ID of the subject being classified.
        matches: Mapping from IR host name to radio component names.
        username: Username of the citizen scientist.
        notes: Extra tags assigned to this classification, possibly
            during reduction.
    """

    # Pipeline modifications are add-only, i.e. if we modify the
    # data through the pipeline, we will never remove or edit an
    # element. This means we can use any of our utility code on
    # partial runs.

    # Classification ID from RGZ MongoDB.
    cid: str = attr.ib()
    # Zooniverse ID.
    zid: subjects.ZooniverseID = attr.ib()
    # IR coordinate -> radio names. We use str as the key to avoid
    # floating point mismatch in keys.
    coord_matches: list[tuple[rgz.HMSDMS, RadioSource]] = attr.ib()
    # Contributor of the classification.
    username: str | None = attr.ib()
    # Additional notes about this classification accumulated during
    # processing.
    notes: list[str] = attr.ib()
    # IR cross-match -> radio names.
    ir_matches: list[tuple[ALLWISEID, RadioSource]] = attr.ib(
        default=attr.Factory(list)
    )

    def to_json(self) -> rgz.JSON:
        """Converts a Classification into a JSON-serialisable dictionary.

        Unset usernames are replaced by empty string.

        Returns:
            Dict representation of the classification, ready to be serialised to JSON.
        """

        # Sort sets being converted into lists - the order doesn't matter
        # but it should be consistent when serialising.
        def dict_key(d):
            return (d["ir"], d["radio"])

        return {
            "id": self.cid,
            "zid": self.zid,
            "coord_matches": sorted(
                [
                    {"ir": ir, "radio": sorted(radio)}
                    for ir, radio in self.coord_matches
                ],
                key=dict_key,
            ),
            "username": self.username or "",
            "notes": self.notes,
            "ir_matches": sorted(
                [{"ir": ir, "radio": sorted(radio)} for ir, radio in self.ir_matches],
                key=dict_key,
            ),
        }

    @classmethod
    def from_json(cls, classification: rgz.JSON) -> Self:
        """Reads a Classification from a JSON dict.

        Args:
            classification: Classification dict to deserialise.

        Returns:
            Classification.
        """
        return cls(
            cid=classification["id"],
            zid=classification["zid"],
            username=classification["username"] or None,
            notes=classification["notes"],
            coord_matches=[
                (m["ir"], RadioSource(m["radio"]))
                for m in classification["coord_matches"]
            ],
            ir_matches=[
                (m["ir"], RadioSource(m["radio"])) for m in classification["ir_matches"]
            ],
        )

    def radio_combinations(
        self,
    ) -> RadioSourceCombination:
        """Gets the combination of radio sources present in this classification."""
        if self.ir_matches:
            radios = (radio for _, radio in self.ir_matches)
        else:
            radios = (radio for _, radio in self.coord_matches)
        return RadioSourceCombination(radios)


def transform_coord_ir(
    coord: npt.NDArray[np.float64],
    wcs: astropy.wcs.WCS,
) -> u.Quantity[u.deg, u.deg]:
    """Transforms a WISE image pixel coordinate into RA/Dec.

    Args:
        coord: Coord to transform, in px coordinates (0, IR_MAX_PX).
        wcs: WCS of the FIRST image of the subject being classified.

    Returns:
        Transformed coordinate RA/dec in deg.
    """
    # Coord in 424x424 -> 100x100
    px_coord = np.array(coord) * 100 / constants.IR_MAX_PX
    # Flip y axis.
    px_coord[1] = 100 - 1 - px_coord[1]
    return wcs.all_pix2world([px_coord], 0)[0] * u.deg


def process_classification(
    raw_classification: rgz.JSON,
    subject: subjects.Subject,
) -> Classification:
    """Reduces a JSON classification into a nice, value-added format.

    Args:
        raw_classification: JSON classification.
        subject: Subject being classified.

    Returns:
        Reduced classification.
    """
    cid = raw_classification["_id"]["$oid"]
    zid = raw_classification["subjects"][0]["zooniverse_id"]
    if zid != subject.zid:
        raise ValueError("Mismatched subjects.")
    matches: list[tuple[str, set[str]]] = []  # (wise, first)
    notes: list[str] = []
    for anno in raw_classification["annotations"]:
        if "radio" not in anno:
            continue
        boxes: set[subjects.BBox] = set()
        if anno["radio"] == "No Contours":
            # ?????? ignore this
            continue
        for radio in anno["radio"].values():
            box = (
                round(float(radio["xmax"]), 1) - 1,
                round(float(radio["ymax"]), 1) - 1,
                round(float(radio["xmin"]), 1) - 1,
                round(float(radio["ymin"]), 1) - 1,
            )
            box = radioislands.transform_rgzbbox_to_phys(bbox=box, wcs=subject.wcs)
            boxes.add(box)

        if anno["ir"] == "No Sources":
            ir = "NOSOURCE"
        else:
            if len(anno["ir"]) != 1:
                notes.append("MULTISOURCE")
            ir_coord = transform_coord_ir(
                np.array([float(anno["ir"]["0"]["x"]), float(anno["ir"]["0"]["y"])]),
                wcs=subject.wcs,
            )
            ir_ra, ir_dec = ir_coord
            ir_coord = SkyCoord(
                ra=ir_ra.value,
                dec=ir_dec.value,
                unit=(ir_ra.unit, ir_dec.unit),
                frame="icrs",
            )
            ir = rgz.coord_to_string(ir_coord)
        first_ids = [ri.firsts for ri in subject.radioislands if box == ri.bbox]
        matches.append((ir, set(first_ids)))

    return Classification(
        cid=cid,
        zid=zid,
        coord_matches=matches,
        username=raw_classification.get("user_name", None),
        notes=notes,
        # We'll set IR matches later, in a deferred lookup. That way we can
        # easily do the lookup as a batch e.g. with CDS xmatch.
    )


def process(classifications_path: Path, subjects_path: Path, output_path: Path):
    """Processes classifications from raw to reduced JSON."""
    # Get classifications count for progress bar.
    with open(classifications_path, encoding="utf-8") as f:
        n_classifications = len(f.readlines())
    raw_classifications = []
    with open(classifications_path, encoding="utf-8") as f:
        # Each row is a JSON document.
        for row in tqdm(f, desc="Reading classifications...", total=n_classifications):
            raw_classifications.append(json.loads(row))

    # Batch classifications by subject to minimise IO.
    subject_to_classifications = collections.defaultdict(list)
    for classification in raw_classifications:
        sid = classification["subject_ids"][0]["$oid"]
        subject_to_classifications[sid].append(classification)

    # Load all reduced subjects.
    with open(subjects_path, "r") as f:
        subjects_ = [subjects.Subject.from_json(js) for js in json.load(f)]

    classifications = []
    bar = tqdm(total=n_classifications, desc="Processing classifications...")
    for subject in subjects_:
        raw_classifications_for_subject = subject_to_classifications[subject.id]
        for raw_classification in raw_classifications_for_subject:
            classification = process_classification(
                raw_classification,
                subject,
            )
            classifications.append(classification)
            bar.update(1)

    json_classifications = []
    for classification in tqdm(classifications, desc="Serialising subjects..."):
        json_classifications.append(classification.to_json())
    with open(output_path, "w") as f:
        json.dump(json_classifications, f, indent=_JSON_INDENT)


def host_lookup(
    classifications_path: Path,
    output_path: Path,
    radius: u.Quantity[u.deg] = constants.DEFAULT_IR_SEARCH_RADIUS,
):
    """Looks up missing host galaxy locations.

    Populates the ir_matches field.

    Args:
        classifications_path: Path to classifications JSON.
        output_path: Path to output JSON.
        radius: IR search radius.
    """
    with open(classifications_path) as f:
        classifications = [Classification.from_json(c) for c in json.load(f)]

    coordinates_to_lookup = set()
    for c in tqdm(classifications, desc="Processing classifications..."):
        for ir, _ in c.coord_matches:
            if ir == "NOSOURCE":
                continue

            coordinates_to_lookup.add(ir)

    # Sort for determinism.
    coordinates_to_lookup = sorted(coordinates_to_lookup)
    # Run small batches so that queries take a reasonable amount of time.
    batch_size = 20000
    batches = []
    for batch_idx in tqdm(
        range(0, len(coordinates_to_lookup), batch_size), desc="Batching coordinates..."
    ):
        batch = coordinates_to_lookup[batch_idx : batch_idx + batch_size]
        batches.append(batch)
    with multiprocessing.Pool(25) as p:
        all_results = p.starmap(query_irsa, [(radius, b) for b in batches])

    logging.info("Joining tables...")
    results = astropy.table.vstack(all_results)
    sc = SkyCoord(ra=results["ra"], dec=results["dec"])

    # TODO: Batch these SkyCoord lookups - match_to_catalog_sky can work on
    # multiple values at once.
    for c in tqdm(classifications, desc="Reprocessing classifications..."):
        ir_matches = []
        for ir, radio in c.coord_matches:
            if ir == "NOSOURCE":
                ir_matches.append((ir, radio))
                continue

            ir_sc = SkyCoord(ir, unit=("hourangle", "deg"))
            idx, dist, _ = ir_sc.match_to_catalog_sky(sc)
            if dist > radius:
                ir_matches.append((ir, radio))
                continue

            designation = results["designation"][idx]
            ir_matches.append((designation, radio))
        c.ir_matches = ir_matches

    with open(output_path, "w") as f:
        json.dump(
            [c.to_json() for c in classifications],
            f,
            indent=_JSON_INDENT,
        )


def query_irsa(radius, coordinates_to_lookup) -> astropy.table.Table:
    """Queries IRSA for the given coordinates."""
    logging.info("Sorting classifications into SkyCoords...")
    sc = SkyCoord(sorted(coordinates_to_lookup), unit=("hourangle", "deg"))
    logging.info("Building coordinate table...")
    coordinate_table = astropy.table.Table(
        {
            "ra": sc.ra.deg,  # type: ignore[reportOptionalMemberAccess]
            "dec": sc.dec.deg,  # type: ignore[reportOptionalMemberAccess]
        }
    )

    # Batch query IRSA.
    logging.info("Connecting to TAP...")
    irsa = pyvo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP")
    wise = constants.ALLWISE_SOURCE_CATALOGUE
    r = radius.to(u.deg).value
    query = f"""
        SELECT w.designation, w.ra, w.dec FROM {wise} as w
        WHERE CONTAINS(POINT(ra, dec), CIRCLE(TAP_UPLOAD.my_table.ra, TAP_UPLOAD.my_table.dec, {r})) = 1
        """
    logger.info("Querying IRSA...")
    q = irsa.run_async(
        query,
        maxrec=len(coordinates_to_lookup) + 1,
        delete=True,
        uploads={
            "my_table": coordinate_table,
        },
    )
    if q.query_status != "OK":
        raise NotImplementedError(f"Unimplemented query status: {q.query_status}")

    results = astropy.table.unique(q.to_table())
    return results
