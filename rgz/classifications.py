"""Handles RGZ classifications."""

import collections
from collections.abc import Generator
import json
import logging
from pathlib import Path
from typing import Any

import attr
import numpy as np
import numpy.typing as npt
from astropy.coordinates import SkyCoord
import astropy.io.votable
import astropy.table
import astropy.wcs
from astroquery.vizier import Vizier
import pyvo
from tqdm import tqdm

from rgz import constants
from rgz import rgz
from rgz import subjects
from rgz import units as u


logger = logging.getLogger(__name__)

type BBox = tuple[float, float, float, float]
type JSON = dict[str, Any]


def get_classifications(path: Path) -> Generator[JSON]:
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
    zid: str = attr.ib()
    # IR coordinate -> radio names. We use str as the key to avoid
    # floating point mismatch in keys.
    coord_matches: list[tuple[str, set[str]]] = attr.ib()
    # Contributor of the classification.
    username: str | None = attr.ib()
    # Additional notes about this classification accumulated during
    # processing.
    notes: list[str] = attr.ib()
    # IR cross-match -> radio names.
    ir_matches: list[tuple[str, set[str]]] = attr.ib(default=attr.Factory(list))


def transform_coord_ir(
    coord: npt.NDArray[np.float64],
    raw_subject: JSON | None = None,
    cache: Path | None = None,
    wcs: astropy.wcs.WCS | None = None,
) -> u.Quantity[u.deg, u.deg]:
    """Transforms a coordinate from raw classifications to physical.

    You can pass a subject and cache, or a WCS.

    Args:
        coord: Coord to transform, in px coordinates (0, IR_MAX_PX).
        raw_subject: Raw JSON subject.
        cache: Path to the subject data location.
        wcs: WCS of the subject being classified.

    Returns:
        Transformed coordinate RA/dec in deg.
    """
    # TODO: We should use subjects here, not raw subjects.
    if not raw_subject and not wcs:
        raise ValueError()
    if raw_subject and not cache:
        raise ValueError()
    if raw_subject:
        assert cache
        assert not wcs
        im = subjects.fetch_first_image_from_server_or_cache(raw_subject, cache)
        wcs = rgz.get_wcs(im)
    assert wcs
    # Coord in 424x424 -> 100x100
    px_coord = np.array(coord) * 100 / constants.IR_MAX_PX
    # Flip y axis.
    px_coord[1] = 100 - px_coord[1]
    return wcs.all_pix2world([px_coord], 0)[0] * u.deg


def process_classification(
    raw_classification: JSON,
    subject: subjects.Subject,
    wcs: astropy.wcs.WCS,
) -> Classification:
    """Reduces a JSON classification into a nice, value-added format.

    Args:
        raw_classification: JSON classification.
        subject: Subject being classified.
        wcs: WCS of the subject image.

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
        boxes: set[BBox] = set()
        if anno["radio"] == "No Contours":
            # ?????? ignore this
            continue
        for radio in anno["radio"].values():
            box = (
                round(float(radio["xmax"]), 1),
                round(float(radio["ymax"]), 1),
                round(float(radio["xmin"]), 1),
                round(float(radio["ymin"]), 1),
            )
            boxes.add(box)

        if anno["ir"] == "No Sources":
            ir = "NOSOURCE"
        else:
            if len(anno["ir"]) != 1:
                notes.append("MULTISOURCE")
            ir_coord = transform_coord_ir(
                np.array([float(anno["ir"]["0"]["x"]), float(anno["ir"]["0"]["y"])]),
                wcs=wcs,
            )
            ir_ra, ir_dec = ir_coord
            ir_coord = SkyCoord(
                ra=ir_ra.value,
                dec=ir_dec.value,
                unit=(ir_ra.unit, ir_dec.unit),
                frame="icrs",
            )
            ir = rgz.coord_to_string(ir_coord)
        matches.append((ir, {c for b in boxes for c in subject.bboxes[b]}))
    return Classification(
        cid=cid,
        zid=zid,
        coord_matches=matches,
        username=raw_classification.get("user_name", None),
        notes=notes,
        # We'll set IR matches later, in a deferred lookup. That way we can
        # easily do the lookup as a batch e.g. with CDS xmatch.
    )


def classification_to_json_serialisable(classification: Classification) -> JSON:
    """Convert a Classification into a JSON-serialisable dictionary."""
    return {
        "id": classification.cid,
        "zid": classification.zid,
        "coord_matches": [
            {"ir": ir, "radio": list(radio)}
            for ir, radio in classification.coord_matches
        ],
        "username": classification.username or constants.ANONYMOUS_NAME,
        "notes": classification.notes,
        "ir_matches": [
            {"ir": ir, "radio": list(radio)} for ir, radio in classification.ir_matches
        ],
    }


def deserialise_classification(classification: JSON) -> Classification:
    """Read a Classification from a JSON dict."""
    return Classification(
        cid=classification["id"],
        zid=classification["zid"],
        username=classification["username"] or None,
        notes=classification["notes"],
        coord_matches=[
            (m["ir"], set(m["radio"])) for m in classification["coord_matches"]
        ],
        ir_matches=[(m["ir"], set(m["radio"])) for m in classification["ir_matches"]],
    )


def process(
    classifications_path: Path, subjects_path: Path, cache: Path, output_path: Path
):
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
        subjects_ = [subjects.deserialise_subject(js) for js in json.load(f)]

    classifications = []
    bar = tqdm(total=n_classifications, desc="Processing classifications...")
    for subject in subjects_:
        raw_classifications_for_subject = subject_to_classifications[subject.id]
        im = subjects.read_subject_image_from_file(subject, cache)
        wcs = rgz.get_wcs(im)
        for raw_classification in raw_classifications_for_subject:
            classification = process_classification(
                raw_classification,
                subject,
                wcs,
            )
            classifications.append(classification)
            bar.update(1)

    json_classifications = []
    for classification in tqdm(classifications, desc="Serialising subjects..."):
        json_classifications.append(classification_to_json_serialisable(classification))
    with open(output_path, "w") as f:
        json.dump(json_classifications, f)


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
        classifications = [deserialise_classification(c) for c in json.load(f)]

    coordinates_to_lookup = set()
    for c in tqdm(classifications, desc="Processing classifications..."):
        for ir, _ in c.coord_matches:
            if ir == "NOSOURCE":
                continue

            coordinates_to_lookup.add(ir)

    sc = SkyCoord(sorted(coordinates_to_lookup), unit=("hourangle", "deg"))

    coordinate_table = astropy.table.Table(
        {
            "ra": sc.ra.deg,  # type: ignore[reportOptionalMemberAccess]
            "dec": sc.dec.deg,  # type: ignore[reportOptionalMemberAccess]
        }
    )

    # Batch query IRSA.
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
        json.dump([classification_to_json_serialisable(c) for c in classifications], f)
