# rgzdr2
Code Repository for RGZ DR2 pipeline

Purpose is to develop the DR2 pipeline that overcomes the limitations of the RGZ DR1 pipeline (https://github.com/willettk/rgz-analysis)

## Getting started

Install Bazel. Then:

```bash
bazel build rgz:main
```

This will build the RGZ binary. Then you can run it:

```bash
./bazel-build/rgz/main --help
```

## Developing

### Dependency management

Dependencies are listed in `pyproject.toml`. After updating them here, use `bazel` to update the corresponding requirements file:

```bash
bazel run rgz:requirements.update
```
