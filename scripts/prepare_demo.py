#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recsys.train.datasets import load_movielens_small
from recsys.train.deployment import prepare_demo


def main():
    parser = argparse.ArgumentParser(description="Package the selected model for the demo.")
    parser.add_argument("--experiment-dir", default="artifacts/experiments/bpr_validation")
    parser.add_argument("--output-dir", default="artifacts/demo")
    args = parser.parse_args()
    interactions, items = load_movielens_small()
    output = prepare_demo(interactions, items, args.experiment_dir, args.output_dir)
    print(f"Demo bundle ready: {output}")


if __name__ == "__main__":
    main()
