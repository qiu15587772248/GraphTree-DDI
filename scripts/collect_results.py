"""Merge GPU-A pair results with GPU-B Exp-1 pair results.

Copies or symlinks into models/results/final_v2_pair_merged/ without modifying
the source directories. Validates unique (method, fold, seed) and a total of 75.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
DEFAULT_A = PROJ / "results" / "final_v2_pair"
DEFAULT_B = PROJ / "results" / "final_v2_pair_exp1"
DEFAULT_OUT = PROJ / "results" / "final_v2_pair_merged"
EXPECTED_N = 75
METHODS = ["Exp-1", "Exp-2", "Exp-3", "Exp-4", "Exp-5"]
FOLDS = [1, 2, 3, 4, 5]
SEEDS = [42, 123, 2024]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge final_v2_pair + final_v2_pair_exp1")
    p.add_argument("--pair_dir", type=str, default=str(DEFAULT_A))
    p.add_argument("--exp1_dir", type=str, default=str(DEFAULT_B))
    p.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--copy", action="store_true", help="强制复制预测 npz（默认优先符号链接）")
    return p.parse_args(argv)


def load_state(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"缺少 {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def triple(rec: dict) -> tuple[str, int, int]:
    return str(rec["method"]), int(rec["fold"]), int(rec["seed"])


def link_or_copy(src: Path, dst: Path, force_copy: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if not force_copy:
        try:
            os.symlink(src.resolve(), dst)
            return "symlink"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


def merge_csv(src_paths: list[Path], out_path: Path, runs: dict) -> None:
    try:
        import pandas as pd
    except ImportError:
        out_path.write_text("", encoding="utf-8")
        print(f"[warn] 无 pandas，跳过 CSV 合并 → 空文件 {out_path}")
        return
    frames = []
    for p in src_paths:
        if p.is_file():
            frames.append(pd.read_csv(p))
    if not frames:
        # rebuild a thin table from merged state
        rows = []
        for rec in runs.values():
            if "error" in rec:
                continue
            rows.append({
                "method": rec.get("method"),
                "fold": rec.get("fold"),
                "seed": rec.get("seed"),
                "test_set": rec.get("test_set", "test"),
            })
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[collect] 无源 CSV，已从 state 写薄表 → {out_path}")
        return
    df = pd.concat(frames, ignore_index=True)
    key_cols = ["method", "fold", "seed"]
    for c in key_cols:
        if c not in df.columns:
            raise SystemExit(f"CSV 缺列 {c}: {list(df.columns)}")
    dup = df.duplicated(subset=key_cols, keep=False)
    if dup.any():
        bad = df.loc[dup, key_cols].drop_duplicates().to_dict("records")
        raise SystemExit(f"CSV (method, fold, seed) 重复: {bad[:10]}")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[collect] CSV 行数={len(df)} → {out_path}")


def expected_triples() -> set[tuple[str, int, int]]:
    return {(m, f, s) for m in METHODS for f in FOLDS for s in SEEDS}


def main(argv=None) -> int:
    args = parse_args(argv)
    pair_dir = Path(args.pair_dir)
    exp1_dir = Path(args.exp1_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_out = out_dir / "predictions"
    pred_out.mkdir(parents=True, exist_ok=True)

    state_a = load_state(pair_dir / "final_state.json")
    state_b = load_state(exp1_dir / "final_state.json")
    runs_a = dict(state_a.get("runs") or {})
    runs_b = dict(state_b.get("runs") or {})

    merged_runs: dict = {}
    sources: dict[str, str] = {}
    for label, runs in (("final_v2_pair", runs_a), ("final_v2_pair_exp1", runs_b)):
        for key, rec in runs.items():
            if key in merged_runs:
                t = triple(rec)
                raise SystemExit(
                    f"run key 重复: {key}  triple={t}  已来自 {sources[key]}  再遇 {label}"
                )
            merged_runs[key] = rec
            sources[key] = label

    triples = [triple(rec) for rec in merged_runs.values()]
    uniq = set(triples)
    if len(uniq) != len(triples):
        seen = {}
        dups = []
        for t in triples:
            if t in seen:
                dups.append(t)
            seen[t] = True
        raise SystemExit(f"(method, fold, seed) 重复: {dups}")

    want = expected_triples()
    missing = sorted(want - uniq)
    extra = sorted(uniq - want)
    n_ok = sum(
        1 for rec in merged_runs.values()
        if "macro_f1" in rec and "error" not in rec
    )
    n_err = sum(1 for rec in merged_runs.values() if "error" in rec)

    if len(uniq) != EXPECTED_N:
        raise SystemExit(
            f"(method, fold, seed) 合计 {len(uniq)}，期望 {EXPECTED_N}。"
            f" missing={missing[:15]} extra={extra[:15]}"
        )
    if missing:
        raise SystemExit(f"缺 {len(missing)} 条: {missing[:20]}")
    if extra:
        print(f"[warn] 超出 5×5×3 笛卡尔积的条目: {extra}")

    merged_state = {
        "mode": state_a.get("mode") or state_b.get("mode"),
        "runs": merged_runs,
        "merged_from": {
            "pair_dir": str(pair_dir),
            "exp1_dir": str(exp1_dir),
            "n_from_pair": len(runs_a),
            "n_from_exp1": len(runs_b),
        },
    }
    out_state = out_dir / "final_state.json"
    with open(out_state, "w", encoding="utf-8") as f:
        json.dump(merged_state, f, ensure_ascii=False, indent=2)

    merge_csv(
        [pair_dir / "final_raw_metrics.csv", exp1_dir / "final_raw_metrics.csv"],
        out_dir / "final_raw_metrics.csv",
        merged_runs,
    )

    copied = {"symlink": 0, "copy": 0}
    for src_dir, label in ((pair_dir, "pair"), (exp1_dir, "exp1")):
        pred = src_dir / "predictions"
        if not pred.is_dir():
            print(f"[warn] 无预测目录 {pred}")
            continue
        for npz in sorted(pred.glob("*.npz")):
            dst = pred_out / npz.name
            if dst.exists() or dst.is_symlink():
                raise SystemExit(f"预测文件名冲突: {npz.name} ({label})")
            how = link_or_copy(npz, dst, force_copy=args.copy)
            copied[how] = copied.get(how, 0) + 1

    manifest = {
        "out_dir": str(out_dir),
        "n_runs": len(merged_runs),
        "n_unique_method_fold_seed": len(uniq),
        "n_ok_macro_f1": n_ok,
        "n_error": n_err,
        "expected": EXPECTED_N,
        "predictions": copied,
        "sources": sources,
    }
    (out_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(
        f"[collect] 合并完成  runs={len(merged_runs)}  unique(method,fold,seed)={len(uniq)}  "
        f"ok={n_ok} error={n_err}  pred={copied}  → {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
