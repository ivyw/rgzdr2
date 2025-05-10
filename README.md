# rgzdr2
[![Lint](https://github.com/ivyw/rgzdr2/actions/workflows/lint.yml/badge.svg)](https://github.com/ivyw/rgzdr2/actions/workflows/lint.yml)

Code Repository for RGZ DR2 pipeline

Purpose is to develop the DR2 pipeline that overcomes the limitations of the RGZ DR1 pipeline (https://github.com/willettk/rgz-analysis)

## Getting started

Install Bazel. Then:

```bash
bazel build rgz:main
```

This will build the RGZ binary. Then you can run it:

```bash
bazel-bin/rgz/main --help
```

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

## Developing

### Dependency management

Dependencies are listed in `pyproject.toml`. After updating them here, use `bazel` to update the corresponding requirements file:

```bash
bazel run rgz:requirements.update
```

...and add them as a dependency in the relevant BUILD rules.
