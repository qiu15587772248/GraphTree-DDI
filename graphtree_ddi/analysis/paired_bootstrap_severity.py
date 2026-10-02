"""Severity-only paired bootstrap (main-text Table 6, Supplementary Table S3e).

Same pairing and percentile-CI protocol as ``paired_bootstrap.py``: the full sealed test set is resampled with
indices shared by the two methods, and the metrics are recomputed on every resample.

- ``macro_f1_sev``: Macro-F1 on the resampled rows with true tier >= 1, labels 1-4 (a predicted Tier 0 is an error).
- ``qwk_sev``: quadratic weighted kappa on the same rows, labels 0-4 (a predicted Tier 0 is penalized).
- ``macro_f1``: five-class Macro-F1, for comparison with ``paired_bootstrap.py``.

Point estimate and CI come from the representative run of method A (five-class Macro-F1 closest to its median over
the matched runs); "CI != 0" counts the matched runs whose 95% CI excludes zero. The drug-propensity prior (B3) is
deterministic, so its single prediction is paired with every run of method A (``--prior``).

    python -m graphtree_ddi.analysis.paired_bootstrap_severity \
        --pred_dir results/final_v2_pair_merged results/baselines_v2_pair results/prior_baseline \
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
from sklearn.metrics import cohen_kappa_score, f1_score  # noqa: E402

from graphtree_ddi.analysis import metrics as M  # noqa: E402
from graphtree_ddi.analysis import paired_bootstrap as PB  # noqa: E402

COMPARATORS = ["DistMult", "ComplEx", "Exp-3", "Exp-2", "MLP", "DDIMDL", "DeepDDI_SSP", "Exp-1", "Exp-4"]
METRIC_KEYS = ("macro_f1_sev", "qwk_sev", "macro_f1")
LABELS = dict(M.DEFAULT_METHOD_LABELS)
LABELS.update({"B3": "Drug-propensity prior (B3)", "DrugPrior": "Drug-propensity prior (B3)"})

_SEV = (1, 2, 3, 4)
_FIVE = (0, 1, 2, 3, 4)
_QWK_W = ((np.arange(5)[:, None] - np.arange(5)[None, :]) ** 2) / 16.0


def _cm5(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = np.clip(np.asarray(y_true, dtype=np.int64), 0, 4)
    yp = np.clip(np.asarray(y_pred, dtype=np.int64), 0, 4)
    return np.bincount(yt * 5 + yp, minlength=25).reshape(5, 5)


def _macro_f1(cm: np.ndarray, labels: tuple[int, ...]) -> float:
    f1 = []
    for c in labels:
        tp, col, row = float(cm[c, c]), float(cm[:, c].sum()), float(cm[c, :].sum())
        p = tp / col if col else 0.0
        r = tp / row if row else 0.0
        f1.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return float(np.mean(f1))


def _qwk(cm: np.ndarray) -> float:
    o = cm.astype(np.float64)
    n = o.sum()
    if n <= 0:
        return float("nan")
    e = np.outer(o.sum(axis=1), o.sum(axis=0)) / n
    den = float((_QWK_W * e).sum())
    return 1.0 if den == 0 else 1.0 - float((_QWK_W * o).sum()) / den


def _bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """One confusion matrix per resample; dropping row 0 restricts it to true tier >= 1."""
    cm = _cm5(y_true, y_pred)
    sev = cm.copy()
    sev[0, :] = 0
    return {"macro_f1": _macro_f1(cm, _FIVE), "macro_f1_sev": _macro_f1(sev, _SEV), "qwk_sev": _qwk(sev)}


def _check_against_sklearn(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    yt, yp = np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)
    m = yt >= 1
    ref = {
        "macro_f1_sev": f1_score(yt[m], yp[m], labels=list(_SEV), average="macro", zero_division=0),
        "qwk_sev": cohen_kappa_score(yt[m], yp[m], weights="quadratic", labels=list(_FIVE)),
        "macro_f1": f1_score(yt, yp, labels=list(_FIVE), average="macro", zero_division=0),
    }
    got = _bundle(yt, yp)
    for k, v in ref.items():
        if abs(got[k] - float(v)) > 1e-12:
            raise SystemExit(f"metric mismatch for {k}: {got[k]} vs sklearn {v}")


def _test_runs(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["test_set"].astype(str).isin(["test", "holdout"])].copy()


def _representative(runs_a: list[pd.Series]) -> tuple:
    f1 = [_macro_f1(_cm5(r.y_true, r.y_pred), _FIVE) for r in runs_a]
    med = float(np.median(f1))
    best = min(range(len(runs_a)), key=lambda i: (abs(f1[i] - med), int(runs_a[i].fold), int(runs_a[i].seed)))
    r = runs_a[best]
    return int(r.fold), int(r.seed), str(r.test_set), str(r.split)


def _pairs(df: pd.DataFrame, method_a: str, method_b: str, prior: str | None) -> list[tuple[pd.Series, pd.Series]]:
    if method_b == prior:
        b = df[df["method"].astype(str).eq(prior)].sort_values(["seed", "fold"])
        if b.empty:
            return []
        rb = b.iloc[0]
        return [(ra, rb) for _, ra in df[df["method"].astype(str).eq(method_a)].iterrows()]
    return PB.match_runs(df, method_a, method_b, test_set="test")


def run(df: pd.DataFrame, method_a: str, comparators: list[str], prior: str | None, n_boot: int, seed: int):
    runs_a = [r for _, r in df[df["method"].astype(str).eq(method_a)].iterrows()]
    if not runs_a:
        raise SystemExit(f"no test-set runs for {method_a}")
    _check_against_sklearn(runs_a[0].y_true, runs_a[0].y_pred)
    rep = _representative(runs_a)
    print(f"[rep] {method_a} representative run: fold {rep[0]}, seed {rep[1]}")
    PB._bundle = _bundle  # paired_bootstrap_once looks the bundle up at call time
    rows = []
    for mb in comparators:
        pairs = _pairs(df, method_a, mb, prior)
        if not pairs:
            print(f"[skip] no matched runs for {mb}")
            continue
        t0 = time.perf_counter()
        for ra, rb in pairs:
            y_true, pa, pb, _, _ = M.align_pair(ra, rb)
            is_rep = (int(ra.fold), int(ra.seed), str(ra.test_set), str(ra.split)) == rep
            for part in PB.paired_bootstrap_once(y_true, pa, pb, n_boot=n_boot, seed=seed,
                                                 metric_keys=list(METRIC_KEYS)):
                rows.append(dict(method_a=method_a, method_b=mb, fold=int(ra.fold), seed=int(ra.seed),
                                 is_representative=is_rep, **part))
        print(f"[run] {method_a} vs {mb}: {len(pairs)} runs, {time.perf_counter() - t0:.0f} s", flush=True)
    return pd.DataFrame(rows)


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    out = []
    for mb, g in detail.groupby("method_b", sort=False):
        rec = {"method_b": mb, "label": M.display_name(mb, LABELS), "n_runs": int(g[["fold", "seed"]].drop_duplicates().shape[0])}
        for mk in METRIC_KEYS:
            sub = g[g["metric"] == mk]
            use = sub[sub["is_representative"]].iloc[0] if sub["is_representative"].any() else sub.iloc[0]
            rec.update({f"{mk}_delta": use["delta"], f"{mk}_ci_low": use["ci_low"], f"{mk}_ci_high": use["ci_high"],
                        f"{mk}_excl0": int(sub["excludes_zero"].astype(bool).sum())})
        out.append(rec)
    return pd.DataFrame(out)


def write(summary: pd.DataFrame, detail: pd.DataFrame, out_dir: Path, n_boot: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "paired_bootstrap_severity.csv", index=False)
    detail.to_csv(out_dir / "paired_bootstrap_severity_runs.csv", index=False)
    f = lambda x: f"{x:.4f}"  # noqa: E731
    lines = [f"Severity-only paired bootstrap (B={n_boot}, seed={seed}); delta = method A minus comparator.", "",
             "| Comparator | Δ severity-only Macro-F1 | 95% CI | CI ≠ 0 | Δ severity-only QWK | 95% CI | CI ≠ 0 |",
             "|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        n = r["n_runs"]
        lines.append(f"| {r['label']} | {f(r['macro_f1_sev_delta'])} | [{f(r['macro_f1_sev_ci_low'])}, "
                     f"{f(r['macro_f1_sev_ci_high'])}] | {r['macro_f1_sev_excl0']}/{n} | {f(r['qwk_sev_delta'])} | "
                     f"[{f(r['qwk_sev_ci_low'])}, {f(r['qwk_sev_ci_high'])}] | {r['qwk_sev_excl0']}/{n} |")
    (out_dir / "paired_bootstrap_severity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[write] {out_dir}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pred_dir", nargs="+", required=True, help="result directories with prediction *.npz")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--method_a", default="Exp-5")
    p.add_argument("--comparators", nargs="*", default=COMPARATORS)
    p.add_argument("--prior", default="B3", help="deterministic prior paired with every run of method A ('' = none)")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    prior = args.prior or None
    comps = list(args.comparators) + ([prior] if prior and prior not in args.comparators else [])
    df = _test_runs(M.load_predictions(args.pred_dir))
    df = df[df["method"].astype(str).isin({args.method_a, *comps})]
    print(f"[load] {len(df)} test-set runs; methods: {M.methods_in(df)}")
    detail = run(df, args.method_a, comps, prior, args.n_boot, args.seed)
    if detail.empty:
        raise SystemExit("no comparator could be paired")
    summary = summarize(detail)
    write(summary, detail, Path(args.out_dir), args.n_boot, args.seed)
    print(summary[["label", "n_runs", "macro_f1_sev_delta", "macro_f1_delta", "qwk_sev_delta"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
