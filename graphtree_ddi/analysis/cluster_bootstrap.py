"""Drug-clustered bootstrap of severity-only Macro-F1 (Supplementary Table S3e note).

On one sealed-test run (default: Exp-5 fold 4 seed 2024, the median five-class Macro-F1 run used in
Table 6 / Table S3e, against DistMult on the same fold and seed), drugs that occur in the true Tier >= 1
test pairs are resampled with replacement (B = 1000, seed 0). Each pair is weighted by the product of its
two drugs' multiplicities (a pair with an undrawn endpoint gets weight 0). Severity-only Macro-F1
(labels 1-4; a predicted Tier 0 is an error) is recomputed for both methods from the weighted confusion
matrix. The 2.5th and 97.5th percentiles of the difference are the 95% interval.

Drug endpoints are read from ``pairs.csv`` (and checked against ``y.npy``) in ``--data_dir``.

    python -m graphtree_ddi.analysis.cluster_bootstrap \
        --pred_dir results/final_v2_pair_merged results/baselines_v2_pair \
        --data_dir data/processed/v2 \
        --out_dir results/analysis
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "graphtree_ddi" / "paths.py").is_file():
            return p
    raise RuntimeError("cannot locate repository root (graphtree_ddi/paths.py)")


_REPO_ROOT = _repo_root()
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from graphtree_ddi.analysis import metrics as M  # noqa: E402

SEV = (1, 2, 3, 4)
# Same display names as Supplementary Table S3e; Exp-5 is printed without the main-text parenthetical.
LABELS = dict(M.DEFAULT_METHOD_LABELS)
LABELS["Exp-5"] = "GraphTree-DDI"


def _test_runs(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["test_set"].astype(str).isin(["test", "holdout"])].copy()


def weighted_sev_f1(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    cm = np.bincount(y * 5 + p, weights=w, minlength=25).reshape(5, 5)
    f1 = []
    for c in SEV:
        tp, col, row = cm[c, c], cm[:, c].sum(), cm[c, :].sum()
        prec = tp / col if col > 0 else 0.0
        rec = tp / row if row > 0 else 0.0
        f1.append(2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0)
    return float(np.mean(f1))


def _pick_run(df: pd.DataFrame, method: str, fold: int, seed: int) -> pd.Series:
    sub = df[
        df["method"].astype(str).eq(method)
        & df["fold"].astype(int).eq(int(fold))
        & df["seed"].astype(int).eq(int(seed))
    ]
    if len(sub) != 1:
        raise SystemExit(f"{method} fold {fold} seed {seed}: expected 1 sealed-test run, found {len(sub)}")
    return sub.iloc[0]


def run(df: pd.DataFrame, method_a: str, method_b: str, fold: int, run_seed: int,
        data_dir: Path, n_boot: int, seed: int) -> dict:
    ra = _pick_run(df, method_a, fold, run_seed)
    rb = _pick_run(df, method_b, fold, run_seed)
    ia = np.asarray(ra.test_idx, dtype=np.int64)
    ib = np.asarray(rb.test_idx, dtype=np.int64)
    ya = np.asarray(ra.y_true, dtype=np.int64)
    yb = np.asarray(rb.y_true, dtype=np.int64)
    if not (np.array_equal(ia, ib) and np.array_equal(ya, yb)):
        raise SystemExit(f"{method_a} and {method_b} runs are not aligned")
    y_true, pred_a, pred_b, _, _ = M.align_pair(ra, rb)

    y_all = np.load(data_dir / "y.npy")
    pairs = pd.read_csv(data_dir / "pairs.csv", usecols=["drug1_id", "drug2_id"])
    if len(pairs) != len(y_all) or not np.array_equal(y_all[ia].astype(np.int64), ya):
        raise SystemExit("test_idx does not index pairs.csv / y.npy")

    m = y_true >= 1
    y, a, b = y_true[m], np.asarray(pred_a, dtype=np.int64)[m], np.asarray(pred_b, dtype=np.int64)[m]
    d1 = pairs["drug1_id"].to_numpy()[ia[m]]
    d2 = pairs["drug2_id"].to_numpy()[ia[m]]
    drugs, inv = np.unique(np.concatenate([d1, d2]), return_inverse=True)
    g1, g2 = inv[: len(d1)], inv[len(d1):]
    n_drugs = len(drugs)

    ones = np.ones(len(y))
    obs_a, obs_b = weighted_sev_f1(y, a, ones), weighted_sev_f1(y, b, ones)
    sk_a = f1_score(y, a, labels=list(SEV), average="macro", zero_division=0)
    if abs(obs_a - sk_a) > 1e-12:
        raise SystemExit("weighted F1 does not match sklearn")
    obs = obs_a - obs_b
    print(f"[data] severity-only pairs {len(y):,}; drugs {n_drugs:,}; "
          f"delta = {obs_a:.4f} - {obs_b:.4f} = {obs:.4f}")

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    t0 = time.perf_counter()
    for r in range(n_boot):
        cnt = np.bincount(rng.integers(0, n_drugs, n_drugs), minlength=n_drugs).astype(np.float64)
        w = cnt[g1] * cnt[g2]
        deltas[r] = weighted_sev_f1(y, a, w) - weighted_sev_f1(y, b, w)
    dt = time.perf_counter() - t0
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    share = float((deltas <= 0).mean())
    print(f"[boot] B={n_boot} in {dt:.1f}s; 95% CI [{lo:.4f}, {hi:.4f}]; share <= 0: {share:.4f}")
    return dict(
        run=f"{method_a} f{fold} s{run_seed} vs {method_b} f{fold} s{run_seed}",
        n_pairs_sev=int(len(y)), n_drugs=int(n_drugs), n_boot=n_boot, seed=seed,
        sev_f1_a=obs_a, sev_f1_b=obs_b, delta=obs, ci_low=float(lo), ci_high=float(hi),
        share_le0=share, label_a=M.display_name(method_a, LABELS), label_b=M.display_name(method_b, LABELS),
        fold=fold, run_seed=run_seed,
    )


def write(rec: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["run", "n_pairs_sev", "n_drugs", "n_boot", "seed", "sev_f1_a", "sev_f1_b",
            "delta", "ci_low", "ci_high", "share_le0"]
    pd.DataFrame([{k: rec[k] for k in cols}]).to_csv(out_dir / "cluster_bootstrap.csv", index=False, encoding="utf-8")
    text = "\n".join([
        "# Drug-clustered bootstrap: GraphTree-DDI minus DistMult, severity-only Macro-F1",
        "",
        "Generated by `graphtree_ddi.analysis.cluster_bootstrap`. Representative sealed-test run "
        f"({rec['run'].split(' vs ')[0]}, as in Table 6 / Table S3e). Drugs occurring in true Tier ≥ 1 "
        "test pairs are resampled with replacement; each pair is weighted by the product "
        "of its two drugs' multiplicities.",
        "",
        f"- Severity-only test pairs: {rec['n_pairs_sev']:,}; drugs (clusters): {rec['n_drugs']:,}",
        f"- B = {rec['n_boot']}, seed = {rec['seed']}",
        f"- Δ severity-only Macro-F1 = {rec['delta']:.4f} "
        f"({rec['label_a']} {rec['sev_f1_a']:.4f}, {rec['label_b']} {rec['sev_f1_b']:.4f})",
        f"- Drug-clustered 95% percentile CI: [{rec['ci_low']:.4f}, {rec['ci_high']:.4f}]; "
        f"share of replicates ≤ 0: {rec['share_le0']:.4f}",
        "- Pair-level paired bootstrap on the same run (Table S3e): [0.0026, 0.0041]",
        "",
    ])
    (out_dir / "cluster_bootstrap.md").write_text(text, encoding="utf-8")
    print(f"[write] {out_dir / 'cluster_bootstrap.md'}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pred_dir", nargs="+", required=True, help="result directories with prediction *.npz")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--data_dir", default="", help="processed tables (pairs.csv, y.npy); default data/processed/v2")
    p.add_argument("--method_a", default="Exp-5")
    p.add_argument("--method_b", default="DistMult")
    p.add_argument("--fold", type=int, default=4)
    p.add_argument("--run_seed", type=int, default=2024, help="training seed of the sealed-test run")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    data_dir = Path(args.data_dir) if args.data_dir else (_REPO_ROOT / "data" / "processed" / "v2")
    if not (data_dir / "pairs.csv").is_file() or not (data_dir / "y.npy").is_file():
        raise SystemExit(f"data_dir needs pairs.csv and y.npy: {data_dir}")
    df = _test_runs(M.load_predictions(args.pred_dir))
    df = df[df["method"].astype(str).isin({args.method_a, args.method_b})]
    print(f"[load] {len(df)} sealed-test runs; methods: {M.methods_in(df)}")
    rec = run(df, args.method_a, args.method_b, args.fold, args.run_seed, data_dir, args.n_boot, args.seed)
    write(rec, Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
