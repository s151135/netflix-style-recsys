#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from recsys.data.readers import write_table
from recsys.utils.ids import new_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Create impression rows from materialized recommendations.")
    parser.add_argument("--recommendations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-version", default="local@candidate")
    args = parser.parse_args()
    recs = pd.read_json(args.recommendations)
    rows = []
    for row in recs.itertuples(index=False):
        request_id = new_id("req")
        for rank, item_id in enumerate(row.ranked_item_ids):
            rows.append(
                {
                    "impression_id": new_id("imp"),
                    "request_id": request_id,
                    "user_id": row.user_id,
                    "surface": "homepage",
                    "row_id": None,
                    "item_id": item_id,
                    "model_version": args.model_version,
                    "rank": rank,
                    "score": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
    write_table(pd.DataFrame(rows), args.output)


if __name__ == "__main__":
    main()
