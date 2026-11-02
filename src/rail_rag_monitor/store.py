"""Tiny file-based store: JSONL tables keyed by a field, plus small JSON documents."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Paths relative to the data directory.
PROBE_RUNS = Path("metrics/probe_runs.jsonl")
TRAFFIC_DAILY = Path("metrics/traffic_daily.jsonl")
ALERT_STATE = Path("alerts/state.json")
NOISE = Path("alerts/noise.json")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def upsert_jsonl(path: Path, rows: Iterable[dict], key: str) -> None:
    """Insert rows, replacing existing rows with the same `key` value in place."""
    existing = read_jsonl(path)
    index = {row[key]: i for i, row in enumerate(existing)}
    for row in rows:
        if row[key] in index:
            existing[index[row[key]]] = row
        else:
            index[row[key]] = len(existing)
            existing.append(row)
    _atomic_write(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in existing))


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    _atomic_write(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
