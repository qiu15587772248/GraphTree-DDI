#!/usr/bin/env python3
"""Match a user-built pairs table against the published hashed split manifest."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS = _REPO / "splits" / "pair_splits.csv.gz"


def canonical_hash(a: str, b: str) -> str:
    x, y = (str(a), str(b)) if str(a) < str(b) else (str(b), str(a))
    return hashlib.sha256(f"{x}|{y}".encode("ascii")).hexdigest()


def pick_cols(df: pd.DataFrame) -> tuple[str, str]:
    cols = {c.lower(): c for c in df.columns}
    for a, b in (("drug_a", "drug_b"), ("drug1_id", "drug2_id"),
                 ("drug1", "drug2"), ("id_a", "id_b")):
        if a in cols and b in cols:
            return cols[a], cols[b]
    raise SystemExit(
        "Need columns drug_a,drug_b (or drug1_id,drug2_id). "
        f"Got: {list(df.columns)}"
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Verify pair hashes against splits/pair_splits.csv.gz")
    p.add_argument("--pairs", required=True, help="User pairs CSV (DrugBank IDs)")
    p.add_argument("--splits", default=str(DEFAULT_SPLITS))
    p.add_argument("--out", default="", help="Optional per-row match CSV")
    args = p.parse_args(argv)

    pairs = pd.read_csv(args.pairs, dtype=str)
    c1, c2 = pick_cols(pairs)
    hashes = [canonical_hash(a, b) for a, b in zip(pairs[c1], pairs[c2])]

    opener = gzip.open if str(args.splits).endswith(".gz") else open
    with opener(args.splits, "rt", encoding="utf-8") as f:
        manifest = pd.read_csv(f, dtype={"pair_hash": str})
    known = set(manifest["pair_hash"].tolist())
    hit = sum(1 for h in hashes if h in known)
    n = len(hashes)
    rate = hit / n if n else 0.0
    print(f"user_pairs	{n}")
    print(f"manifest_rows	{len(manifest)}")
    print(f"matched	{hit}")
    print(f"unmatched	{n - hit}")
    print(f"match_rate	{rate:.6f}")
    if args.out:
        out = pd.DataFrame({
            c1: pairs[c1],
            c2: pairs[c2],
            "pair_hash": hashes,
            "in_manifest": [h in known for h in hashes],
        })
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False)
        print(f"wrote	{args.out}")
    return 0 if hit == n and n else (0 if n == 0 else 1)


if __name__ == "__main__":
    raise SystemExit(main())
