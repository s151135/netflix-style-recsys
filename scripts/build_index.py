#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recsys.batch.embed_items import export_item_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Export item embeddings for ANN/index builds.")
    parser.add_argument("--model-dir", default="artifacts/local/bpr_champion")
    parser.add_argument("--output", default="artifacts/local/item_embeddings.npz")
    args = parser.parse_args()
    export_item_embeddings(args.model_dir, args.output)


if __name__ == "__main__":
    main()
