# rgzdr2

[![Lint](https://github.com/ivyw/rgzdr2/actions/workflows/lint.yml/badge.svg)](https://github.com/ivyw/rgzdr2/actions/workflows/lint.yml)

Code Repository for RGZ DR2 pipeline

Figshare link to FIRST FITS/JSON repo: <https://figshare.com/s/b4e28330635e7861c2b4?file=54481925>

Purpose is to develop the DR2 pipeline that overcomes the limitations of the RGZ DR1 pipeline (<https://github.com/willettk/rgz-analysis>)

## Getting started

Install `uv`. Then:

```bash
uv run rgz
```

## Data dependencies

Most of the data will be downloaded automatically, but you can speed things up by providing them locally. The structure should be:

- data/
  - cache/
    - first_2014Dec17.csv
    - 52af81027aa69f059a003a95.fits
    - 52af81027aa69f059a003a95.json
    - ...
  - radio_subjects.json
  - radio_classifications.json

## Running the pipeline

You need to have the RGZ data dumped as JSON: this is the only input. To save some time, you could use the cache folder from a previous run of the pipeline; it's deterministic wherever possible and the cache stores some of the slower data files.

### Processing RGZ subjects

```bash
uv run rgz subjects --in=data/radio_subjects.json --out=data/radio_subjects_processed.json --cache=data/cache
```

This will:

- download FIRST images from the FIRST server,
- download FIRST catalogue data from Vizier,
- download JSON contours from the Zooniverse server, and
- use the combined information to build a reduced dataset of RGZ subjects.

A reduced RGZ subject is a JSON object with the following schema:

```json
{
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "RGZ subject",
    "type": "object",
    "properties": {
        "id": {
            "type": "string",
            "description": "Zooniverse MongoDB ID."
        },
        "zid": {
            "type": "string",
            "description": "Zooniverse ID."
        },
        "coords": {
            "type": "array",
            "prefixItems": [
                {
                    "type": "number",
                    "description": "Right ascension (deg)."
                },
                {
                    "type": "number",
                    "description": "Declination (deg)."
                }
            ],
            "items": false
        },
        "bboxes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "first": {
                        "type": "string"
                    },
                    "bbox": {
                        "type": "array",
                        "prefixItems": [
                            {
                                "type": "number",
                                "description": "Maximum RA (px)."
                            },
                            {
                                "type": "number",
                                "description": "Maximum dec (px)."
                            },
                            {
                                "type": "number",
                                "description": "Minimum RA (px)."
                            },
                            {
                                "type": "number",
                                "description": "Minimum dec (px)."
                            }
                        ],
                        "items": false
                    }
                }
            }
        }
    }
}
```

For example:

```JSON
{
    "id": "52af7eb58c51f405a600001b",
    "zid": "ARG0002w6r",
    "coords": [151.87758333333332, 12.038472222222222],
    "bboxes": [
        {
            "bbox": [26.4, 50.8, 15.5, 29.1],
            "first": ["NOFIRST_J100734.79897624+120143.45817109"]
        }, {
            "bbox": [71.7, 71.2, 58.0, 58.4],
            "first": ["J100730.6+120218"]
        }
    ]
}
```

### Processing the RGZ classifications

```bash
uv run rgz classifications --in=data/radio_classifications.json --out=data/radio_classifications_processed.json --cache=data/cache --subjects=data/radio_subjects_processed.json
```

This will:

- use the downloaded FIRST images to figure out what radio components citizen scientists selected, and
- use the combined information to build a reduced dataset of RGZ classifications.

A reduced RGZ classification is a JSON object with the following schema: (TODO)

Once the classifications are processed, perform the host lookup. This is separate because it is slow, and we may want to make this asynchronous in future. This will include AllWISE cross-matches.

```bash
uv run rgz host-lookup --classifications=data/radio_classifications_processed.json --out=data/radio_classifications_matched.json
```

### Aggregate classifications into a consensus

```bash
uv run rgz aggregate --subjects=data/radio_subjects_processed.json --classifications=data/radio_classifications_matched.json --out=data/radio_consensus.json
```

This will, for each subject, decide on a consensus between all classifications for that subject. It does not account for duplicates or overlaps.

## Developing

### Dependency management

Dependencies are listed in `pyproject.toml` and managed by `uv`. If you need to, you can sync this with `uv sync`.

### Linting

Run `black`:

```bash
uv run black rgz
```

### Testing

Run tests with `pytest`:

```bash
uv run pytest
```

### Notebooks

To run notebooks, you need to install the source as a kernel.

```bash
.venv/bin/python -m ipykernel install --user --name=rgz --display-name "rgz"
```

Then you can call `jupyter` from `uv`, and choose the `rgz` kernel.

```bash
uv run jupyter notebook
```
