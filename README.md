# rgzdr2

[![Lint](https://github.com/ivyw/rgzdr2/actions/workflows/lint.yml/badge.svg)](https://github.com/ivyw/rgzdr2/actions/workflows/lint.yml)

Code Repository for RGZ DR2 pipeline

Figshare link to FIRST FITS/JSON repo: <https://figshare.com/s/b4e28330635e7861c2b4?file=54481925>

Purpose is to develop the DR2 pipeline that overcomes the limitations of the RGZ DR1 pipeline (<https://github.com/willettk/rgz-analysis>)

## Getting started

Install Bazel. Then:

```bash
bazel build rgz:main
```

This will build the RGZ binary. Then you can run it:

```bash
bazel-bin/rgz/main --help
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
bazel-bin/rgz/main subjects --in=data/radio_subjects.json --out=data/radio_subjects_processed.json --cache=data/cache
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
bazel-bin/rgz/main classifications --in=data/radio_classifications.json --out=data/radio_classifications_processed.json --cache=data/cache --subjects=data/radio_subjects_processed.json
```

This will:

- use the downloaded FIRST images to figure out what radio components citizen scientists selected, and
- use the combined information to build a reduced dataset of RGZ classifications.

A reduced RGZ classification is a JSON object with the following schema: (TODO)

## Developing

### Dependency management

Dependencies are listed in `pyproject.toml`. After updating them here, use `bazel` to update the corresponding requirements file:

```bash
bazel run rgz:requirements.update
```

...and add them as a dependency in the relevant BUILD rules.

### Testing

Run tests with Bazel:

```bash
bazel test rgz:all
```

### Notebooks

To run notebooks, use the `jupyter_server` target from the root directory:

```bash
bazel run notebooks:jupyter_server
```
