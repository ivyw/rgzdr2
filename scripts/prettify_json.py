#!/usr/bin/env python3

"""Prettifies a JSON file in-place."""

import argparse
import json
from pathlib import Path


def main(path: Path):
    with open(path) as f:
        file = json.load(f)
    with open(path, "w") as f:
        json.dump(file, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    main(args.path)
