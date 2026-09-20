"""Repository path constants. Never hard-code machine-local drives."""

from __future__ import annotations

from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"
RESOURCES_DIR = REPO_ROOT / "resources"
SPLITS_DIR = REPO_ROOT / "splits"
