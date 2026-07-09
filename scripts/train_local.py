#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recsys.logging import configure_logging, get_logger
from recsys.train.pipelines import run_local_smoke_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the local recommendation baseline.")
    parser.add_argument("--output-dir", default="artifacts/local")
    args = parser.parse_args()
    configure_logging()
    log = get_logger(__name__)
    result = run_local_smoke_pipeline(Path(args.output_dir))
    log.info("training complete: %s", result.artifact_dir)
    print(json.dumps(result.metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
