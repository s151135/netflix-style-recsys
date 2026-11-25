#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recsys.train.datasets import load_movielens_small
from recsys.train.tuning import tune_bpr


def main():
    parser = argparse.ArgumentParser(description="Select BPR on validation, then evaluate test once.")
    parser.add_argument("--output-dir", default="artifacts/experiments/bpr_validation")
    parser.add_argument("--rating-threshold", type=float, default=4.0)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    interactions, _ = load_movielens_small()
    report = tune_bpr(interactions, args.output_dir, k=args.k,
                      rating_threshold=args.rating_threshold,
                      progress=lambda message: print(message, flush=True))
    print(json.dumps(report["test"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
