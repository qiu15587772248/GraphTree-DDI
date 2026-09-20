"""Shared paths and small helpers for the DDInter external-validation pipeline."""

from __future__ import annotations

from pathlib import Path
import sys as _sys

def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "graphtree_ddi" / "paths.py").is_file():
            return p
    raise RuntimeError("cannot locate repository root (graphtree_ddi/paths.py)")

_REPO_ROOT = _repo_root()
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"
STATS_DIR = HERE / "stats"
from graphtree_ddi.paths import DATA_PROCESSED, REPO_ROOT  # noqa: E402
PROJECT_DIR = REPO_ROOT
PROCESSED_DIR = DATA_PROCESSED

ATC_CODES = list("ABCDGHJLMNPRSV")
TZ_CST = timezone(timedelta(hours=8))

DDINTER2_CSV = (
    "https://ddinter2.scbdd.com/static/media/download/ddinter_downloads_code_{code}.csv"
)
DDINTER1_CSV = (
    "https://ddinter.scbdd.com/static/media/download/ddinter_downloads_code_{code}.csv"
)
DDINTER2_DETAIL = "https://ddinter2.scbdd.com/server/drug-detail/{ddinter_id}/"
DDINTER1_DETAIL = "https://ddinter.scbdd.com/server/drug-detail/{ddinter_id}/"

USER_AGENT = "GraphTree-DDI-external-eval/1.0"


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def now_cst_iso() -> str:
    return datetime.now(TZ_CST).isoformat(timespec="seconds")


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
