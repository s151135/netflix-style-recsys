#!/usr/bin/env python3
"""Build the release demo from the recorded model settings; no hyperparameter search."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recsys.train.datasets import load_movielens_small
from recsys.train.deployment import build_demo_from_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/demo")
    parser.add_argument("--report", default="docs/benchmarks/movielens-validation.json")
    args = parser.parse_args()
    if Path(args.output_dir).exists():
        parser.error("Output already exists; choose a new directory")
    interactions, items = load_movielens_small()
    print("Reproducing the recorded BPR configuration...", flush=True)
    output = build_demo_from_report(interactions, items, args.report, args.output_dir)
    print(f"Demo ready: {output}")


if __name__ == "__main__":
    main()
