from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _minimal_yaml(text: str) -> dict[str, Any]:
    """Tiny YAML subset parser used only when PyYAML is not installed."""
    result: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, result)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        line = raw.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if line.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError("List items require a list parent")
            parent.append(_parse_scalar(line[2:]))
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return result


def _parse_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [] if not inner else [_parse_scalar(x.strip()) for x in inner.split(",")]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip('"').strip("'")


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    config_path = Path(path or os.getenv("RECSYS_CONFIG", "configs/local_mac.yaml"))
    if not config_path.exists():
        return {}
    try:
        import yaml

        data = yaml.safe_load(config_path.read_text()) or {}
    except Exception:
        data = _minimal_yaml(config_path.read_text())
    parent = data.pop("extends", None)
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = config_path.parent.parent / parent_path
        return _deep_merge(load_config(parent_path), data)
    return data


@dataclass(frozen=True)
class RuntimeSettings:
    artifact_dir: Path = field(
        default_factory=lambda: Path(os.getenv("RECSYS_ARTIFACT_DIR", "artifacts/local"))
    )
    registry_dir: Path = field(
        default_factory=lambda: Path(os.getenv("RECSYS_MODEL_REGISTRY", "artifacts/registry"))
    )
    config: dict[str, Any] = field(default_factory=load_config)

    def to_json(self) -> str:
        return json.dumps(
            {
                "artifact_dir": str(self.artifact_dir),
                "registry_dir": str(self.registry_dir),
                "config": self.config,
            },
            indent=2,
            sort_keys=True,
        )
