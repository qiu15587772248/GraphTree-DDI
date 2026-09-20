"""Merge ATC-batch CSVs, drop cross-file duplicates, and isolate Level conflicts."""

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

from pathlib import Path

import pandas as pd

from graphtree_ddi.external.ddinter._common import (
    ATC_CODES,
    RAW_DIR,
    STATS_DIR,
    configure_stdio,
    ensure_dirs,
    now_cst_iso,
    write_json,
)

HERE = Path(__file__).resolve().parent
CLEAN_CSV = HERE / "ddinter_pairs_clean.csv"
CONFLICTS_CSV = HERE / "conflicts.csv"
LEVEL_DIST_CSV = HERE / "level_distribution.csv"
STATS_JSON = STATS_DIR / "02_clean.json"


def load_raw() -> pd.DataFrame:
    frames = []
    missing = []
    for code in ATC_CODES:
        path = RAW_DIR / f"ddinter_downloads_code_{code}.csv"
        if not path.exists():
            missing.append(code)
            continue
        print(f"[02] reading {path.name}", flush=True)
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
        df.columns = [c.strip() for c in df.columns]
        expected = ["DDInterID_A", "Drug_A", "DDInterID_B", "Drug_B", "Level"]
        if list(df.columns)[:5] != expected:
            raise ValueError(f"{path.name} unexpected columns: {list(df.columns)}")
        df = df[expected].copy()
        df["source_atc"] = code
        frames.append(df)
    if missing:
        raise FileNotFoundError(f"missing ATC files: {missing}")
    return pd.concat(frames, ignore_index=True)


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["DDInterID_A", "Drug_A", "DDInterID_B", "Drug_B", "Level"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    df = df[
        (df["DDInterID_A"] != "")
        & (df["DDInterID_B"] != "")
        & (df["DDInterID_A"] != df["DDInterID_B"])
    ].copy()
    nums_a = pd.to_numeric(df["DDInterID_A"].str.extract(r"(\d+)$", expand=False), errors="coerce")
    nums_b = pd.to_numeric(df["DDInterID_B"].str.extract(r"(\d+)$", expand=False), errors="coerce")
    bad = nums_a.isna() | nums_b.isna()
    if int(bad.sum()) > 0:
        raise ValueError(f"non-numeric DDInter IDs: {int(bad.sum())} rows")
    swap = nums_a > nums_b
    left_id = df["DDInterID_A"].where(~swap, df["DDInterID_B"])
    left_name = df["Drug_A"].where(~swap, df["Drug_B"])
    right_id = df["DDInterID_B"].where(~swap, df["DDInterID_A"])
    right_name = df["Drug_B"].where(~swap, df["Drug_A"])
    df["DDInterID_A"] = left_id
    df["Drug_A"] = left_name
    df["DDInterID_B"] = right_id
    df["Drug_B"] = right_name
    df["pair_key"] = df["DDInterID_A"] + "|" + df["DDInterID_B"]
    return df


def main() -> int:
    configure_stdio()
    ensure_dirs()
    print("[02] loading raw ATC files", flush=True)
    raw = load_raw()
    n_raw = int(len(raw))
    print(f"[02] raw_rows={n_raw:,}", flush=True)
    df = normalize_frame(raw)
    print(f"[02] after basic filter={len(df):,}", flush=True)

    level_nunique = df.groupby("pair_key", sort=False)["Level"].nunique(dropna=False)
    conflict_keys = level_nunique[level_nunique > 1].index
    n_conflict_pairs = int(len(conflict_keys))
    is_conflict = df["pair_key"].isin(conflict_keys)
    conflicts = df.loc[is_conflict].sort_values(["pair_key", "source_atc", "Level"])
    conflicts.to_csv(CONFLICTS_CSV, index=False, encoding="utf-8")
    print(
        f"[02] conflict_pairs={n_conflict_pairs:,} rows={len(conflicts):,} -> {CONFLICTS_CSV.name}",
        flush=True,
    )

    kept = df.loc[~is_conflict]
    # Same unordered pair + same Level across ATC files: keep first row
    # (files are read in ATC letter order A,B,C,...).
    clean = (
        kept.sort_values(["pair_key", "source_atc"])
        .drop_duplicates(subset=["pair_key"], keep="first")
        [["DDInterID_A", "Drug_A", "DDInterID_B", "Drug_B", "Level"]]
        .reset_index(drop=True)
    )
    clean.to_csv(CLEAN_CSV, index=False, encoding="utf-8")

    level_dist = (
        clean["Level"].value_counts(dropna=False).rename_axis("Level").reset_index(name="n")
    )
    level_dist["pct"] = (level_dist["n"] / max(len(clean), 1) * 100).round(4)
    level_dist.to_csv(LEVEL_DIST_CSV, index=False, encoding="utf-8")

    n_unique_drugs = int(
        pd.unique(pd.concat([clean["DDInterID_A"], clean["DDInterID_B"]], ignore_index=True)).size
    )
    stats = {
        "generated_at_utc8": now_cst_iso(),
        "n_raw_rows": n_raw,
        "n_after_basic_filter": int(len(df)),
        "n_conflict_pairs": n_conflict_pairs,
        "n_conflict_rows": int(len(conflicts)),
        "n_clean_unordered_pairs": int(len(clean)),
        "n_unique_ddinter_drugs": n_unique_drugs,
        "n_dropped_as_atc_duplicates": int(len(kept) - len(clean)),
        "level_distribution": {
            str(row.Level): int(row.n) for row in level_dist.itertuples(index=False)
        },
        "clean_csv": str(CLEAN_CSV),
        "conflicts_csv": str(CONFLICTS_CSV),
    }
    write_json(STATS_JSON, stats)

    print(f"[02] clean unordered pairs={len(clean):,} unique drugs={n_unique_drugs:,}", flush=True)
    print("[02] Level distribution:", flush=True)
    for row in level_dist.itertuples(index=False):
        print(f"    {row.Level}: {row.n:,} ({row.pct:.2f}%)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
