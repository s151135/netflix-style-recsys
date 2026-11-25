#!/usr/bin/env python3
"""Export a clean Git snapshot, Python distributions and optional demo bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
__version__ = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include-demo")
    args = parser.parse_args()
    if git("status", "--porcelain"):
        parser.error("Commit the reviewed source before exporting a release")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Choose a new or empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    name = f"netflix-style-recsys-{__version__}"
    source = output / f"{name}-source.zip"
    with ZipFile(source, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in git("ls-files").splitlines():
            path = ROOT / relative
            if path.is_file() and not path.is_symlink():
                archive.write(path, f"{name}/{relative}")
    build_dir = ROOT / "work" / "release-dist"
    build_dir.mkdir(parents=True, exist_ok=True)
    log = ROOT / "work" / "release-build.log"
    with log.open("w") as stream:
        subprocess.run([sys.executable, "-m", "build", "--outdir", str(build_dir)],
                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    assets = [source]
    for suffix in [f"netflix_style_recsys-{__version__}-py3-none-any.whl",
                   f"netflix_style_recsys-{__version__}.tar.gz"]:
        destination = output / suffix
        shutil.copyfile(build_dir / suffix, destination)
        assets.append(destination)
    if args.include_demo:
        bundle = Path(args.include_demo).resolve()
        required = ["bpr_champion/factors.npz", "bpr_champion/metadata.json",
                    "sample_interactions.csv", "sample_items.json", "evaluation.json",
                    "MOVIELENS_README.txt"]
        if not all((bundle / relative).is_file() for relative in required):
            parser.error("Demo bundle is incomplete or lacks the MovieLens notice")
        target = output / f"{name}-demo-bundle.zip"
        with ZipFile(target, "w", ZIP_DEFLATED, compresslevel=9) as archive:
            for relative in required:
                archive.write(bundle / relative, f"artifacts/demo/{relative}")
            archive.write(ROOT / "THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md")
        assets.append(target)
    for path in assets:
        if path.suffix in {".zip", ".whl"}:
            with ZipFile(path) as archive:
                if archive.testzip() is not None:
                    raise ValueError(f"Archive integrity check failed: {path.name}")
    manifest = {"version": __version__, "commit": git("rev-parse", "HEAD"),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "assets": [{"name": path.name, "bytes": path.stat().st_size,
                            "sha256": digest(path)} for path in assets],
                "demo_scope": "Educational MovieLens demo; see included third-party notices"}
    metadata = output / "RELEASE.json"
    metadata.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    assets.append(metadata)
    (output / "SHA256SUMS.txt").write_text("".join(f"{digest(path)}  {path.name}\n" for path in assets))
    print(f"Release {__version__} exported from {manifest['commit'][:12]} to {output}")


if __name__ == "__main__":
    main()
