"""Compare drug-cold-start and pair splits: pipeline vs baselines.

Does not modify models/run_pipeline_final.py or models/baselines/common.py.
If signatures differ, this file is the adapter. Run from repo root:

  $env:PYTHONIOENCODING='utf-8'
  python models/tests/test_drug_split_equality.py
"""

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
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from graphtree_ddi.paths import DATA_PROCESSED, REPO_ROOT  # noqa: E402
HERE = Path(__file__).resolve()
PROJ = REPO_ROOT
DATA_DIR = DATA_PROCESSED / "v2"
SEEDS = [42, 123, 2024]
N_CLASSES = 5
N_FOLDS = 5


import graphtree_ddi.models.run_pipeline_final as pipe  # noqa: E402
from graphtree_ddi.models.baselines import common as bl  # noqa: E402


def load_pairs_y():
    meta_path = DATA_DIR / "meta.json"
    pairs_path = DATA_DIR / "pairs.csv"
    y_path = DATA_DIR / "y.npy"
    if not pairs_path.is_file() or not y_path.is_file():
        raise FileNotFoundError(f"需要 {pairs_path} 与 {y_path}")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    n = int(meta["n_samples"])
    pairs = pd.read_csv(pairs_path)
    pairs = pairs.iloc[:n].reset_index(drop=True)
    y = np.asarray(np.load(y_path)[:n])
    if len(pairs) != n or len(y) != n:
        raise ValueError(f"长度不一致: pairs={len(pairs)} y={len(y)} n={n}")
    return pairs, y, n


def sorted_ids(values) -> np.ndarray:
    return np.asarray(sorted(values), dtype=object)


def sorted_idx(values) -> np.ndarray:
    return np.sort(np.asarray(values, dtype=np.int64), kind="mergesort")


def label_dist(y: np.ndarray, idx: np.ndarray) -> list[int]:
    if len(idx) == 0:
        return [0] * N_CLASSES
    return np.bincount(y[idx], minlength=N_CLASSES).astype(int).tolist()


def describe_mismatch(name: str, a: np.ndarray, b: np.ndarray) -> str:
    sa = set(a.tolist())
    sb = set(b.tolist())
    only_a = sa - sb
    only_b = sb - sa
    inter = sa & sb
    ex_a = list(sorted(only_a, key=str)[:8])
    ex_b = list(sorted(only_b, key=str)[:8])
    return (
        f"    mismatch {name}: |pipe|={len(a)} |base|={len(b)} "
        f"intersect={len(inter)} only_pipe={len(only_a)} only_base={len(only_b)}\n"
        f"      only_pipe sample={ex_a}\n"
        f"      only_base sample={ex_b}"
    )


# ── adapters (do not change source implementations) ─────────────────────────
def adapt_pipeline_drug(pairs: pd.DataFrame, y: np.ndarray, seed: int) -> dict:
    raw = pipe.make_drug_cold_start_split(pairs, y, seed)
    return {
        "U": sorted_ids(raw["U"]),
        "KK-train": sorted_idx(raw["idx_train"]),
        "KK-val": sorted_idx(raw["idx_val"]),
        "S1": sorted_idx(raw["S1"]),
        "S2": sorted_idx(raw["S2"]),
        "KK": sorted_idx(raw["KK"]),
        "raw": raw,
    }


def adapt_baseline_drug(pairs: pd.DataFrame, y: np.ndarray, seed: int) -> dict:
    indices = np.arange(len(y), dtype=np.int64)
    raw = bl.build_drug_splits(pairs, y, indices, seed)
    return {
        "U": sorted_ids(raw.U),
        "KK-train": sorted_idx(raw.kk_train),
        "KK-val": sorted_idx(raw.kk_val),
        "S1": sorted_idx(raw.s1),
        "S2": sorted_idx(raw.s2),
        "KK": sorted_idx(raw.kk),
        "raw": raw,
    }


def adapt_pipeline_pair(y: np.ndarray) -> dict:
    """Faithful extract of prepare_global_data pair branch (n_folds=5).

    prepare_global_data also builds the KG; that is not needed to compare indices.
    """
    all_idx = np.arange(len(y))
    idx_tv, idx_test = train_test_split(
        all_idx, test_size=0.2, stratify=y, random_state=pipe.SPLIT_RANDOM_STATE,
    )
    skf = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=pipe.SPLIT_RANDOM_STATE,
    )
    cv_splits = []
    for fold_idx, (train_rel, val_rel) in enumerate(
        skf.split(idx_tv, y[idx_tv]), start=1,
    ):
        cv_splits.append((fold_idx, idx_tv[train_rel], idx_tv[val_rel]))
    return {"idx_tv": idx_tv, "idx_test": idx_test, "cv_splits": cv_splits}


def adapt_baseline_pair(y: np.ndarray) -> dict:
    splits = bl.pair_splits_pipeline_final(
        y, n_folds=N_FOLDS, random_state=bl.SPLIT_RANDOM_STATE,
    )
    return {
        "idx_tv": splits.idx_tv,
        "idx_test": splits.idx_test,
        "cv_splits": splits.cv_splits,
    }


def print_row(cols: list[str], widths: list[int]) -> None:
    parts = []
    for c, w in zip(cols, widths):
        parts.append(str(c).ljust(w))
    print("  " + "  ".join(parts))


def compare_drug(pairs: pd.DataFrame, y: np.ndarray) -> tuple[bool, list[dict], list[str]]:
    names = ["U", "KK-train", "KK-val", "S1", "S2"]
    all_ok = True
    rows = []
    diffs = []
    for seed in SEEDS:
        print(f"\n{'=' * 72}")
        print(f"  drug-cold-start seed={seed}")
        print(f"{'=' * 72}")
        p = adapt_pipeline_drug(pairs, y, seed)
        b = adapt_baseline_drug(pairs, y, seed)
        widths = [10, 12, 12, 8, 28, 28]
        print_row(["集合", "pipe_n", "base_n", "相等", "pipe 5类分布", "base 5类分布"], widths)
        for name in names:
            pa, ba = p[name], b[name]
            equal = bool(np.array_equal(pa, ba))
            if name == "U":
                pdists = "(drugs, n/a)"
                bdists = "(drugs, n/a)"
            else:
                pdists = str(label_dist(y, pa))
                bdists = str(label_dist(y, ba))
            mark = "YES" if equal else "NO"
            if not equal:
                all_ok = False
                diffs.append(
                    f"seed={seed} 集合={name}: pipe_n={len(pa)} base_n={len(ba)}\n"
                    + describe_mismatch(name, pa, ba)
                )
            print_row(
                [name, f"{len(pa):,}", f"{len(ba):,}", mark, pdists, bdists],
                widths,
            )
            rows.append({
                "seed": seed, "set": name, "pipe_n": int(len(pa)),
                "base_n": int(len(ba)), "equal": equal,
                "pipe_dist": pdists, "base_dist": bdists,
            })
        # extra: KK union size (not in the required five, for diagnosis)
        print(
            f"  (extra) KK total  pipe={len(p['KK']):,}  base={len(b['KK']):,}  "
            f"equal={np.array_equal(p['KK'], b['KK'])}"
        )
    return all_ok, rows, diffs


def compare_pair(y: np.ndarray) -> tuple[bool, list[dict], list[str]]:
    print(f"\n{'=' * 72}")
    print("  pair 模式：20% 封存测试 + StratifiedKFold(5, seed=42)")
    print(f"{'=' * 72}")
    p = adapt_pipeline_pair(y)
    b = adapt_baseline_pair(y)
    all_ok = True
    rows = []
    diffs = []
    widths = [14, 12, 12, 8]
    print_row(["集合", "pipe_n", "base_n", "相等"], widths)

    def check(name: str, pa: np.ndarray, ba: np.ndarray) -> None:
        nonlocal all_ok
        sa, sb = sorted_idx(pa), sorted_idx(ba)
        equal = bool(np.array_equal(sa, sb))
        mark = "YES" if equal else "NO"
        print_row([name, f"{len(sa):,}", f"{len(sb):,}", mark], widths)
        rows.append({
            "seed": 42, "set": name, "pipe_n": int(len(sa)),
            "base_n": int(len(sb)), "equal": equal,
        })
        if not equal:
            all_ok = False
            diffs.append(
                f"pair 集合={name}: pipe_n={len(sa)} base_n={len(sb)}\n"
                + describe_mismatch(name, sa, sb)
            )

    check("idx_test", p["idx_test"], b["idx_test"])
    p_folds = {f: va for f, _tr, va in p["cv_splits"]}
    b_folds = {f: va for f, _tr, va in b["cv_splits"]}
    for fold in range(1, N_FOLDS + 1):
        check(f"fold{fold}_idx_val", p_folds[fold], b_folds[fold])
    return all_ok, rows, diffs


def suspected_cause(diffs: list[str]) -> str:
    blob = "\n".join(diffs)
    hints = []
    if "集合=U" in blob:
        hints.append("U 集合本身不一致：permutation / 药物宇宙排序可能不同")
    if any(s in blob for s in ("S1", "S2", "KK-train", "KK-val", "KK")):
        hints.append(
            "成员判定：pipeline 用 np.isin(d, np.array(list(U), dtype=object))，"
            "baselines 用 pd.Series.isin(U)；object 数组上的 np.isin 在部分 numpy 版本会误判"
        )
        hints.append(
            "KK 顺序：两边都声称 np.flatnonzero 升序后再 train_test_split；"
            "若 KK 成员已不同，则 KK-train/val 必然不同"
        )
    if "idx_test" in blob or "idx_val" in blob:
        hints.append(
            "pair：pipeline 适配层逐字复现 prepare_global_data 的 "
            "train_test_split + StratifiedKFold；baselines 为 pair_splits_pipeline_final"
        )
    if not hints:
        hints.append("未见明显模式；请对照差异样本手工核对")
    return "；".join(hints)


def main() -> int:
    print(f"[test] PROJ={PROJ}")
    print(f"[test] DATA_DIR={DATA_DIR}")
    print(f"[test] pipeline.make_drug_cold_start_split = {pipe.make_drug_cold_start_split}")
    print(f"[test] baselines.build_drug_splits = {bl.build_drug_splits}")
    print(f"[test] SPLIT_RANDOM_STATE pipe={pipe.SPLIT_RANDOM_STATE} base={bl.SPLIT_RANDOM_STATE}")
    pairs, y, n = load_pairs_y()
    print(
        f"[test] n={n:,}  y dist={np.bincount(y, minlength=N_CLASSES).tolist()}  "
        f"pairs cols={list(pairs.columns)}"
    )

    drug_ok, drug_rows, drug_diffs = compare_drug(pairs, y)
    pair_ok, pair_rows, pair_diffs = compare_pair(y)

    print(f"\n{'=' * 72}")
    print("  汇总")
    print(f"{'=' * 72}")
    print("  集合 × 大小 × 是否相等（drug，三种子）:")
    print("  seed  set        pipe_n     base_n     equal")
    for r in drug_rows:
        print(
            f"  {r['seed']:<4}  {r['set']:<10} {r['pipe_n']:>10,} {r['base_n']:>10,}  "
            f"{'YES' if r['equal'] else 'NO'}"
        )
    print("  pair 划分:")
    for r in pair_rows:
        print(
            f"  {r['set']:<16} {r['pipe_n']:>10,} {r['base_n']:>10,}  "
            f"{'YES' if r['equal'] else 'NO'}"
        )

    diffs = drug_diffs + pair_diffs
    if diffs:
        print("\n  ** 发现不一致（未修改任何实现）**")
        for d in diffs:
            print(d)
        print(f"\n  疑似原因: {suspected_cause(diffs)}")
        print("  drug 总体: FAIL" if not drug_ok else "  drug 总体: PASS")
        print("  pair 总体: FAIL" if not pair_ok else "  pair 总体: PASS")
        return 1

    print("\n  drug 三种子 U / KK-train / KK-val / S1 / S2 全部逐元素相等")
    print("  pair idx_test 与每折 idx_val 全部逐元素相等")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
