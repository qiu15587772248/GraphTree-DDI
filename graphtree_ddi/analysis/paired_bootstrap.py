"""同一测试集上两方法的配对 bootstrap（共享重采样索引）。

Δ = metric(A) − metric(B)。默认 B=1000、seed=0。
对每个匹配的 (fold, seed[, test_set]) 都算；并标出方法 A 的代表性 run
（macro-F1 距中位数最近的那一次，供正文引用）。
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

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


from graphtree_ddi.analysis import metrics as M  # noqa: E402

DEFAULT_METRICS = list(M.BOOTSTRAP_METRICS)


def _bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": M.metric_value("macro_f1", y_true, y_pred),
        "accuracy": M.metric_value("accuracy", y_true, y_pred),
        "qwk": M.metric_value("qwk", y_true, y_pred),
        "high_risk_sens": M.metric_value("high_risk_sens", y_true, y_pred),
        "mae_ordinal": M.metric_value("mae_ordinal", y_true, y_pred),
    }


def paired_bootstrap_once(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    metric_keys: list[str] | None = None,
) -> list[dict]:
    """对一组对齐后的预测做配对 bootstrap。"""
    y_true = np.asarray(y_true, dtype=np.int64)
    pred_a = np.asarray(pred_a, dtype=np.int64)
    pred_b = np.asarray(pred_b, dtype=np.int64)
    n = int(len(y_true))
    keys = list(metric_keys or DEFAULT_METRICS)
    point_a = _bundle(y_true, pred_a)
    point_b = _bundle(y_true, pred_b)
    rng = np.random.default_rng(int(seed))
    store = {k: np.full(int(n_boot), np.nan, dtype=np.float64) for k in keys}
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n, dtype=np.int64)
        ma = _bundle(y_true[idx], pred_a[idx])
        mb = _bundle(y_true[idx], pred_b[idx])
        for k in keys:
            va, vb = ma[k], mb[k]
            if np.isfinite(va) and np.isfinite(vb):
                store[k][b] = float(va) - float(vb)
    rows = []
    for k in keys:
        deltas = store[k]
        finite = deltas[np.isfinite(deltas)]
        da = point_a[k]
        db = point_b[k]
        delta = float(da - db) if (np.isfinite(da) and np.isfinite(db)) else float("nan")
        if finite.size >= 20:
            lo, hi = np.percentile(finite, [2.5, 97.5])
            lo, hi = float(lo), float(hi)
            excludes0 = bool((lo > 0.0) or (hi < 0.0))
        else:
            lo, hi, excludes0 = float("nan"), float("nan"), False
        rows.append(dict(
            metric=k,
            n=n,
            n_boot=int(n_boot),
            value_a=float(da) if np.isfinite(da) else float("nan"),
            value_b=float(db) if np.isfinite(db) else float("nan"),
            delta=delta,
            ci_low=lo,
            ci_high=hi,
            excludes_zero=excludes0,
            n_valid_boot=int(finite.size),
        ))
    return rows


def match_runs(
    df: pd.DataFrame,
    method_a: str,
    method_b: str,
    test_set: str | None = None,
) -> list[tuple[pd.Series, pd.Series]]:
    a = df[df["method"].astype(str).eq(method_a)]
    b = df[df["method"].astype(str).eq(method_b)]
    if test_set:
        if test_set in ("test", "holdout"):
            a = a[a["test_set"].astype(str).isin(["test", "holdout"])]
            b = b[b["test_set"].astype(str).isin(["test", "holdout"])]
        else:
            a = a[a["test_set"].astype(str).eq(test_set)]
            b = b[b["test_set"].astype(str).eq(test_set)]
    pairs = []
    b_map = {}
    for _, rb in b.iterrows():
        key = (int(rb.fold), int(rb.seed), str(rb.test_set), str(rb.split))
        b_map[key] = rb
        # 也允许 test/holdout 互通
        if str(rb.test_set) in ("test", "holdout"):
            b_map[(int(rb.fold), int(rb.seed), "test", str(rb.split))] = rb
    seen = set()
    for _, ra in a.iterrows():
        keys = [
            (int(ra.fold), int(ra.seed), str(ra.test_set), str(ra.split)),
        ]
        if str(ra.test_set) in ("test", "holdout"):
            keys.append((int(ra.fold), int(ra.seed), "test", str(ra.split)))
        rb = None
        for k in keys:
            if k in b_map:
                rb = b_map[k]
                break
        if rb is None:
            continue
        sig = (int(ra.fold), int(ra.seed), str(ra.test_set), str(ra.split))
        if sig in seen:
            continue
        seen.add(sig)
        pairs.append((ra, rb))
    return pairs


SUMMARY_METRICS = ("macro_f1", "qwk", "high_risk_sens", "mae_ordinal")
SUMMARY_HEAD = {
    "macro_f1": "Δ Macro-F1",
    "qwk": "Δ QWK",
    "high_risk_sens": "Δ HR Sens.",
    "mae_ordinal": "Δ MAE",
}


def run_paired_bootstrap(
    pred_dir,
    method_a: str,
    method_b: str,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    test_set: str | None = None,
    metrics: list[str] | None = None,
    df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if df is None:
        df = M.load_predictions(pred_dir)
    if df.empty:
        raise SystemExit(f"未在 {pred_dir} 读到预测")
    have = set(df["method"].astype(str))
    if method_a not in have or method_b not in have:
        raise SystemExit(
            f"方法缺失: A={method_a} B={method_b}；目录中有 {sorted(have)}"
        )
    pairs = match_runs(df, method_a, method_b, test_set=test_set)
    if not pairs:
        raise SystemExit(f"没有匹配的 (fold, seed) : {method_a} vs {method_b}")

    hold = [(ra, rb) for ra, rb in pairs if str(ra.test_set) in ("test", "holdout")]
    pool = hold if hold else pairs
    f1s = []
    for ra, _rb in pool:
        f1s.append(M.metric_value("macro_f1", ra.y_true, ra.y_pred, ra.y_prob))
    med = float(np.nanmedian(np.asarray(f1s, dtype=np.float64)))
    rep_sig = None
    best = None
    for ra, rb in pool:
        f1 = M.metric_value("macro_f1", ra.y_true, ra.y_pred, ra.y_prob)
        cand = (abs(f1 - med), int(ra.fold), int(ra.seed), str(ra.test_set))
        if best is None or cand < best:
            best = cand
            rep_sig = (int(ra.fold), int(ra.seed), str(ra.test_set), str(ra.split))

    out_rows = []
    n_skip = 0
    for ra, rb in pairs:
        try:
            y_true, pa, pb, _qa, _qb = M.align_pair(ra, rb)
        except ValueError as e:
            n_skip += 1
            print(f"[bootstrap] skip {method_a} vs {method_b} "
                  f"f{int(ra.fold)} s{int(ra.seed)}: {e}")
            continue
        parts = paired_bootstrap_once(
            y_true, pa, pb, n_boot=n_boot, seed=seed, metric_keys=metrics,
        )
        is_rep = (
            int(ra.fold), int(ra.seed), str(ra.test_set), str(ra.split)
        ) == rep_sig
        for part in parts:
            out_rows.append(dict(
                method_a=method_a,
                method_b=method_b,
                fold=int(ra.fold),
                seed=int(ra.seed),
                split=str(ra.split),
                test_set=str(ra.test_set),
                is_representative=bool(is_rep),
                **part,
            ))
    if not out_rows:
        raise SystemExit(
            f"没有可对齐的 run: {method_a} vs {method_b} (skipped={n_skip})"
        )
    return pd.DataFrame(out_rows)


def summarize_vs_many(
    detail: pd.DataFrame,
    labels: dict[str, str] | None = None,
    metric_keys: tuple[str, ...] = SUMMARY_METRICS,
) -> pd.DataFrame:
    """每个 method_b 一行：代表性 run 的点估计+CI，以及全部 run 中 CI 不含 0 的比例。"""
    rows = []
    for mb, g in detail.groupby("method_b", sort=False):
        rec = dict(
            method_a=g["method_a"].iloc[0],
            method_b=mb,
            method_a_label=M.display_name(g["method_a"].iloc[0], labels),
            method_b_label=M.display_name(mb, labels),
        )
        n_runs = int(g[["fold", "seed", "test_set"]].drop_duplicates().shape[0])
        rec["n_runs"] = n_runs
        for mk in metric_keys:
            sub = g[g["metric"] == mk]
            if sub.empty:
                rec[f"{mk}_delta"] = float("nan")
                rec[f"{mk}_ci_low"] = float("nan")
                rec[f"{mk}_ci_high"] = float("nan")
                rec[f"{mk}_excl0_n"] = 0
                rec[f"{mk}_excl0_frac"] = float("nan")
                continue
            rep = sub[sub["is_representative"]]
            use = rep.iloc[0] if len(rep) else sub.iloc[0]
            rec[f"{mk}_delta"] = float(use["delta"])
            rec[f"{mk}_ci_low"] = float(use["ci_low"])
            rec[f"{mk}_ci_high"] = float(use["ci_high"])
            rec[f"{mk}_excl0_n"] = int(sub["excludes_zero"].astype(bool).sum())
            rec[f"{mk}_excl0_frac"] = float(sub["excludes_zero"].astype(bool).mean())
        rows.append(rec)
    out = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(M.METHOD_ORDER)}
    out["_ord"] = out["method_b"].map(lambda m: order.get(m, 999))
    return out.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)


def _fmt4(v) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "---"
    return f"{x:.4f}" if np.isfinite(x) else "---"


def write_merged_tables(summary: pd.DataFrame, path: Path) -> None:
    """一张合并 Markdown / LaTeX 表。"""
    path = Path(path)
    keys = list(SUMMARY_METRICS)
    header = ["Comparator"]
    for mk in keys:
        header += [SUMMARY_HEAD[mk], "95% CI (rep)", "CI≠0"]
    header.append("N runs")
    lines = [
        "# Paired bootstrap summary",
        "",
        f"Δ = {summary['method_a_label'].iloc[0]} − comparator. "
        "Point estimate and 95% CI from the representative run "
        "(median macro-F1 of method A). CI≠0 = fraction of matched runs "
        "whose 95% CI excludes 0.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for _, r in summary.iterrows():
        cells = [str(r["method_b_label"])]
        for mk in keys:
            cells.append(_fmt4(r[f"{mk}_delta"]))
            cells.append(f"[{_fmt4(r[f'{mk}_ci_low'])}, {_fmt4(r[f'{mk}_ci_high'])}]")
            n = int(r["n_runs"])
            k = int(r[f"{mk}_excl0_n"])
            cells.append(f"{k}/{n}")
        cells.append(str(int(r["n_runs"])))
        lines.append("| " + " | ".join(cells) + " |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    n_col = 1 + 3 * len(keys) + 1
    align = "l" + "c" * (n_col - 1)
    L = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Paired bootstrap: GraphTree-DDI vs each comparator "
        r"(representative-run $\Delta$ and 95\% CI; CI$\neq$0 = runs whose CI excludes 0).}",
        r"\label{tab:paired_bootstrap}",
        f"\\begin{{tabular}}{{{align}}}",
        r"\toprule",
    ]
    tex_heads = ["Comparator"]
    for mk in keys:
        tex_heads += [SUMMARY_HEAD[mk], "95\\% CI (rep)", "CI$\\neq$0"]
    tex_heads.append("N")
    L.append(" " + " & ".join(M.latex_escape(h).replace("\\textbackslash{}", "\\")
                              if False else h for h in tex_heads) + r" \\")
    # latex_escape would break $\neq$; write heads as-is (they are controlled)
    L[-1] = " " + " & ".join(tex_heads) + r" \\"
    L.append(r"\midrule")
    for _, r in summary.iterrows():
        cells = [M.latex_escape(str(r["method_b_label"]))]
        for mk in keys:
            cells.append(_fmt4(r[f"{mk}_delta"]))
            cells.append(
                f"[{_fmt4(r[f'{mk}_ci_low'])}, {_fmt4(r[f'{mk}_ci_high'])}]"
            )
            n = int(r["n_runs"])
            k = int(r[f"{mk}_excl0_n"])
            cells.append(f"{k}/{n}")
        cells.append(str(int(r["n_runs"])))
        L.append(" " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.with_suffix(".tex").write_text("\n".join(L), encoding="utf-8")


def run_vs_many(
    pred_dir,
    method_a: str,
    method_bs: list[str],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    test_set: str | None = None,
    labels: dict[str, str] | None = None,
    df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df is None:
        df = M.load_predictions(pred_dir)
    have = set(df["method"].astype(str))
    if method_a not in have:
        raise SystemExit(f"方法 A 缺失: {method_a}；目录中有 {sorted(have)}")
    details = []
    for mb in method_bs:
        if mb == method_a:
            continue
        if mb not in have:
            print(f"[bootstrap] 跳过缺失方法 {mb}")
            continue
        try:
            part = run_paired_bootstrap(
                pred_dir, method_a, mb, n_boot=n_boot, seed=seed,
                test_set=test_set, df=df,
            )
        except SystemExit as e:
            print(f"[bootstrap] 跳过 {method_a} vs {mb}: {e}")
            continue
        details.append(part)
    if not details:
        raise SystemExit("没有成功的配对 bootstrap")
    detail = pd.concat(details, ignore_index=True)
    summary = summarize_vs_many(detail, labels=labels)
    return detail, summary


def _write_md(df: pd.DataFrame, path: Path) -> None:
    if "method_b" in df.columns and df["method_b"].nunique() > 1:
        # 多对照时只写明细；合并表由 write_merged_tables 负责
        df.to_csv(Path(path).with_suffix(".csv"), index=False, encoding="utf-8-sig")
        return
    lines = [
        f"# Paired bootstrap: {df['method_a'].iloc[0]} vs {df['method_b'].iloc[0]}",
        "",
        f"Δ = metric(A) − metric(B). B={int(df['n_boot'].iloc[0])}, "
        "95% percentile CI. Shared resample indices.",
        "",
        "## Representative run (median macro-F1 of method A)",
        "",
    ]
    rep = df[df["is_representative"]]
    cols = ["metric", "value_a", "value_b", "delta", "ci_low", "ci_high", "excludes_zero"]
    if rep.empty:
        lines.append("*none*")
    else:
        r0 = rep.iloc[0]
        lines.append(
            f"fold={int(r0.fold)} seed={int(r0.seed)} split={r0.split} "
            f"test_set={r0.test_set} n={int(r0.n)}"
        )
        lines.append("")
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, r in rep.iterrows():
            cells = []
            for c in cols:
                v = r[c]
                if c == "excludes_zero":
                    cells.append("yes" if bool(v) else "no")
                elif isinstance(v, float):
                    cells.append(f"{v:.4f}" if np.isfinite(v) else "---")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## All matched (fold, seed) runs", ""]
    show = ["fold", "seed", "test_set", "metric", "delta", "ci_low", "ci_high",
            "excludes_zero", "is_representative"]
    lines.append("| " + " | ".join(show) + " |")
    lines.append("|" + "|".join(["---"] * len(show)) + "|")
    for _, r in df.iterrows():
        cells = []
        for c in show:
            v = r[c]
            if c in ("excludes_zero", "is_representative"):
                cells.append("yes" if bool(v) else "no")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}" if np.isfinite(v) else "---")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Paired bootstrap between two or more methods")
    M.add_pred_dir_arg(p)
    p.add_argument("--method_a", default="Exp-5")
    p.add_argument(
        "--method_b", nargs="*", default=None,
        help="一个或多个对照方法；省略则对其余全部方法",
    )
    p.add_argument("--out", required=True, help="明细 CSV；合并表写同名 .md/.tex")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test_set", default="", help="仅该 test_set；空=全部匹配")
    M.add_method_names_arg(p)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ts = args.test_set.strip() or None
    labels = M.parse_method_names(args.method_names)
    df = M.load_predictions(args.pred_dir)
    have = M.methods_in(df)
    method_a = args.method_a
    if method_a not in set(have):
        raise SystemExit(f"method_a={method_a} 不在 {have}")
    if args.method_b:
        mbs = list(args.method_b)
    else:
        mbs = [m for m in have if m != method_a]
    detail, summary = run_vs_many(
        args.pred_dir, method_a, mbs,
        n_boot=args.n_boot, seed=args.seed, test_set=ts,
        labels=labels, df=df,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out, index=False, encoding="utf-8-sig")
    sum_csv = out.with_name(out.stem + "_summary.csv")
    summary.to_csv(sum_csv, index=False, encoding="utf-8-sig")
    write_merged_tables(summary, out.with_suffix(".md"))
    _write_md(detail, out.with_name(out.stem + "_detail.md"))
    print(f"[bootstrap] detail {len(detail)} rows → {out}")
    print(f"[bootstrap] summary {len(summary)} comparators → {sum_csv}")
    return detail, summary


if __name__ == "__main__":
    main()
