"""High-tier operating curve on the sealed pair test set (Supplementary Table S3f).

For each of the 15 sealed-test runs (5 folds x seeds 42/123/2024) a pair is graded high-tier when
p(Tier 3) + p(Tier 4) >= t, for t in {0.5, 0.3, 0.2, 0.1, 0.05}; the argmax decision (predicted tier >= 3,
the Table 4 operating point) is listed first. Two quantities are computed per run and summarized as
mean +/- SD (ddof = 0, as in Table 4):

- HT-Sens.: P(high-tier | true y >= 3)
- false high-tier rate: P(high-tier | true y in {1, 2}); constructed Tier 0 pairs are excluded, so the rate
  is a severity-only quantity.

Only ``--curve_method`` (GraphTree-DDI / Exp-5) is swept over thresholds. R-GCN (DDI-only, Exp-2),
R-GCN (hybrid KG, Exp-3) and DistMult are listed at their argmax operating point. The curve is described
on the sealed test set; no threshold is selected on it.

    python -m graphtree_ddi.analysis.threshold_curve \
        --pred_dir results/final_v2_pair_merged results/baselines_v2_pair \
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from graphtree_ddi.analysis import metrics as M  # noqa: E402

METHODS = ["Exp-5", "Exp-2", "Exp-3", "DistMult"]
THRESHOLDS = (0.5, 0.3, 0.2, 0.1, 0.05)
N_SEALED = 15
# Supplementary Table S3f prints the fusion model without the main-text parenthetical.
LABELS = dict(M.DEFAULT_METHOD_LABELS)
LABELS["Exp-5"] = "GraphTree-DDI"
TABLE4_HTSENS = {"Exp-5": 0.9847, "Exp-2": 0.9932, "Exp-3": 0.9926, "DistMult": 0.9818}


def _test_runs(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["test_set"].astype(str).isin(["test", "holdout"])].copy()


def _rates(y: np.ndarray, high: np.ndarray) -> tuple[float, float]:
    return float(high[y >= 3].mean()), float(high[(y == 1) | (y == 2)].mean())


def run(df: pd.DataFrame, methods: list[str], curve_method: str, thresholds: tuple[float, ...]) -> list[dict]:
    rows: list[dict] = []
    ref_idx = None
    for method in methods:
        sub = df[df["method"].astype(str).eq(method)].copy()
        # Same order as a sorted glob of ``{method}_f*_s*.npz``, so mean/SD match bitwise.
        sub = sub.assign(_name=sub["path"].map(lambda p: Path(p).name)).sort_values("_name")
        if len(sub) != N_SEALED:
            raise SystemExit(f"{method}: expected {N_SEALED} sealed-test runs, found {len(sub)}")
        label = M.display_name(method, LABELS)
        per_point: dict[str, list[tuple[float, float]]] = {"argmax": []}
        if method == curve_method:
            per_point.update({f"t = {t:g}": [] for t in thresholds})
        for _, rec in sub.iterrows():
            idx = np.sort(np.asarray(rec.test_idx, dtype=np.int64))
            y = np.asarray(rec.y_true, dtype=np.int64)
            yp = np.asarray(rec.y_pred, dtype=np.int64)
            pr = np.asarray(rec.y_prob, dtype=np.float64)
            if ref_idx is None:
                ref_idx = idx
            elif not np.array_equal(ref_idx, idx):
                raise SystemExit(f"{method} fold {int(rec.fold)} seed {int(rec.seed)}: sealed test pair set differs")
            if not np.array_equal(pr.argmax(1), yp):
                raise SystemExit(f"{method} fold {int(rec.fold)} seed {int(rec.seed)}: y_pred is not argmax(y_prob)")
            per_point["argmax"].append(_rates(y, yp >= 3))
            if method == curve_method:
                p_hi = pr[:, 3] + pr[:, 4]
                for t in thresholds:
                    per_point[f"t = {t:g}"].append(_rates(y, p_hi >= t))
        for point, vals in per_point.items():
            a = np.asarray(vals)
            rows.append(dict(
                method=method, label=label, operating_point=point, n_runs=len(a),
                ht_sens_mean=a[:, 0].mean(), ht_sens_sd=a[:, 0].std(ddof=0),
                false_hi_t12_mean=a[:, 1].mean(), false_hi_t12_sd=a[:, 1].std(ddof=0),
            ))
        got = rows[-len(per_point)]["ht_sens_mean"]
        if method in TABLE4_HTSENS:
            ok = abs(got - TABLE4_HTSENS[method]) < 5.5e-5
            print(f"[check] {label} argmax HT-Sens {got:.4f} vs Table 4 {TABLE4_HTSENS[method]:.4f}"
                  f" {'OK' if ok else 'MISMATCH'}")
            if not ok:
                raise SystemExit("Table 4 mismatch")
    return rows


def write(rows: list[dict], out_dir: Path, n_pairs: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["method", "label", "operating_point", "n_runs", "ht_sens_mean", "ht_sens_sd",
            "false_hi_t12_mean", "false_hi_t12_sd"]
    pd.DataFrame(rows)[cols].to_csv(out_dir / "threshold_curve.csv", index=False, encoding="utf-8")
    lines = [
        "# High-tier operating curve on the sealed pair test set (Table S3f)",
        "",
        "Generated by `graphtree_ddi.analysis.threshold_curve`. 15 sealed-test runs per method "
        f"(n = {n_pairs:,} pairs each); mean ± SD (ddof = 0). High-tier = predicted Tier ≥ 3 "
        "(argmax) or p(Tier 3) + p(Tier 4) ≥ t. False high-tier rate = share of true "
        "Tier 1–2 interactions graded high-tier (Tier 0 excluded). Described on the "
        "sealed test set; no threshold is selected on it.",
        "",
        "| Method | Operating point | HT-Sens. | True Tier 1–2 graded ≥ 3 |",
        "|---|---|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['operating_point']} | "
            f"{r['ht_sens_mean']:.4f} ± {r['ht_sens_sd']:.4f} | "
            f"{100 * r['false_hi_t12_mean']:.2f}% ± {100 * r['false_hi_t12_sd']:.2f}% |"
        )
    (out_dir / "threshold_curve.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[write] {out_dir / 'threshold_curve.md'}")
    print(f"[write] {out_dir / 'threshold_curve.csv'}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pred_dir", nargs="+", required=True, help="result directories with prediction *.npz")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--methods", nargs="+", default=list(METHODS))
    p.add_argument("--curve_method", default="Exp-5", help="method swept over --thresholds; others stay at argmax")
    p.add_argument("--thresholds", nargs="+", type=float, default=list(THRESHOLDS))
    args = p.parse_args(argv)
    df = _test_runs(M.load_predictions(args.pred_dir))
    df = df[df["method"].astype(str).isin(set(args.methods))]
    print(f"[load] {len(df)} sealed-test runs; methods: {M.methods_in(df)}")
    n_pairs = int(len(df.iloc[0].y_true)) if len(df) else 0
    rows = run(df, list(args.methods), args.curve_method, tuple(args.thresholds))
    write(rows, Path(args.out_dir), n_pairs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
