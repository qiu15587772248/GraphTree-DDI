"""从 state / npz 汇总 mean±SD，写出 Markdown 与 LaTeX（booktabs）表。"""

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

MAIN_COLS = [
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro-F1"),
    ("weighted_f1", "Weighted-F1"),
    ("macro_auroc", "Macro AUROC"),
    ("macro_ap", "Macro AP"),
]
ORD_COLS = [
    ("adjacent_acc", "Adjacent Acc."),
    ("high_risk_sens", "HR Sens."),
    ("high_risk_spec", "HR Spec."),
    ("severe_underestimation_rate", "Sev. Underest."),
    ("qwk", "QWK"),
    ("mae_ordinal", "Ordinal MAE"),
]
COLD_COLS = [
    ("accuracy", "Acc."),
    ("macro_f1", "Macro-F1"),
    ("qwk", "QWK"),
    ("high_risk_sens", "HR Sens."),
    ("mae_ordinal", "MAE"),
]


def _metrics_frame(pred_dir) -> pd.DataFrame:
    pred = M.load_predictions(pred_dir)
    parts = []
    if not pred.empty:
        parts.append(M.metrics_table_from_runs(pred))
    try:
        st = M.runs_from_state(pred_dir)
    except ValueError as e:
        # state 与 npz 跨目录重复时：只保留 npz 已覆盖的，state 去重
        print(f"[summarize] state 合并冲突，改为按 npz 优先去重: {e}")
        st = pd.DataFrame()
        dirs = M.as_pred_dirs(pred_dir)
        acc: list[dict] = []
        seen = set()
        if not pred.empty:
            for rec in pred.itertuples(index=False):
                seen.add(M.prediction_key(rec.method, rec.fold, rec.seed, rec.test_set))
        for d in dirs:
            for row in M._runs_from_state_one(d):
                k = M.prediction_key(row["method"], row["fold"], row["seed"], row["test_set"])
                if k in seen:
                    continue
                seen.add(k)
                acc.append(row)
        if acc:
            st = pd.DataFrame(acc)
    if not st.empty:
        if parts:
            have = set(
                zip(parts[0]["method"], parts[0]["fold"], parts[0]["seed"],
                    parts[0]["split"], parts[0]["test_set"])
            )
            keep = []
            for rec in st.itertuples(index=False):
                key = (rec.method, rec.fold, rec.seed, rec.split, rec.test_set)
                keep.append(key not in have)
            st = st.loc[np.asarray(keep)].copy()
        if not st.empty:
            parts.append(st)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True, sort=False)


def _agg(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for method, g in df.groupby("method", sort=False):
        rec = dict(method=method, n_runs=int(len(g)))
        for k in keys:
            if k not in g.columns:
                rec[f"{k}_mean"] = float("nan")
                rec[f"{k}_sd"] = float("nan")
                continue
            mu, sd = M.mean_sd(g[k].tolist())
            rec[f"{k}_mean"] = mu
            rec[f"{k}_sd"] = sd
        for i in range(M.N_CLASSES):
            col = f"f1_c{i}"
            if col in g.columns:
                mu, sd = M.mean_sd(g[col].tolist())
            else:
                mu, sd = float("nan"), float("nan")
            rec[f"{col}_mean"] = mu
            rec[f"{col}_sd"] = sd
        rows.append(rec)
    out = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(M.methods_in(df.assign(method=df["method"])))}
    out["_ord"] = out["method"].map(lambda m: order.get(m, 999))
    return out.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)


def _bold_flags(agg: pd.DataFrame, spec: list[tuple[str, str]]) -> dict[str, list[bool]]:
    flags = {}
    for key, _lab in spec:
        mus = agg[f"{key}_mean"].tolist()
        higher = M.HIGHER_IS_BETTER.get(key, True)
        flags[key] = [M.is_best(mus, mu, higher) for mu in mus]
    return flags


def _md_table(agg: pd.DataFrame, spec: list[tuple[str, str]], title: str,
              labels: dict[str, str] | None = None) -> str:
    flags = _bold_flags(agg, spec)
    header = ["Method"] + [lab for _k, lab in spec] + ["N"]
    lines = [f"## {title}", "", "| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for i, rec in agg.iterrows():
        cells = [M.display_name(rec["method"], labels)]
        for key, _lab in spec:
            cells.append(M.format_mean_sd(
                rec[f"{key}_mean"], rec[f"{key}_sd"],
                bold=flags[key][i],
            ))
        cells.append(str(int(rec["n_runs"])))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _tex_table(agg: pd.DataFrame, spec: list[tuple[str, str]], caption: str, label: str,
               labels: dict[str, str] | None = None) -> str:
    flags = _bold_flags(agg, spec)
    n = 1 + len(spec) + 1
    align = "l" + "c" * (n - 1)
    L = [
        f"% {caption}",
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        r"\toprule",
    ]
    heads = ["Method"] + [lab for _k, lab in spec] + ["N"]
    L.append(" " + " & ".join(M.latex_escape(h) for h in heads) + r" \\")
    L.append(r"\midrule")
    for i, rec in agg.iterrows():
        cells = [M.latex_escape(M.display_name(rec["method"], labels))]
        for key, _lab in spec:
            cells.append(M.format_mean_sd_tex(
                rec[f"{key}_mean"], rec[f"{key}_sd"],
                bold=flags[key][i],
            ))
        cells.append(str(int(rec["n_runs"])))
        L.append(" " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


def _md_per_class(agg: pd.DataFrame, labels: dict[str, str] | None = None) -> str:
    spec = [(f"f1_c{i}", M.CLASS_NAMES_EN[i]) for i in range(M.N_CLASSES)]
    header = ["Method"] + [s[1] for s in spec] + ["N"]
    flags = {}
    for key, _ in spec:
        mus = agg[f"{key}_mean"].tolist()
        flags[key] = [M.is_best(mus, mu, True) for mu in mus]
    lines = ["## Per-class F1 (mean ± SD)", "",
             "| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for i, rec in agg.iterrows():
        cells = [M.display_name(rec["method"], labels)]
        for j, (key, _) in enumerate(spec):
            cells.append(M.format_mean_sd(
                rec[f"{key}_mean"], rec[f"{key}_sd"], bold=flags[key][i],
            ))
        cells.append(str(int(rec["n_runs"])))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _tex_per_class(agg: pd.DataFrame, labels: dict[str, str] | None = None) -> str:
    spec = [(f"f1_c{i}", M.CLASS_NAMES_EN[i]) for i in range(M.N_CLASSES)]
    flags = {}
    for key, _ in spec:
        mus = agg[f"{key}_mean"].tolist()
        flags[key] = [M.is_best(mus, mu, True) for mu in mus]
    align = "l" + "c" * (M.N_CLASSES + 1)
    L = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Per-class F1 (mean $\pm$ SD across runs). Best in bold.}",
        r"\label{tab:per_class_f1}",
        f"\\begin{{tabular}}{{{align}}}",
        r"\toprule",
    ]
    heads = ["Method"] + [s[1] for s in spec] + ["N"]
    L.append(" " + " & ".join(M.latex_escape(h) for h in heads) + r" \\")
    L.append(r"\midrule")
    for i, rec in agg.iterrows():
        cells = [M.latex_escape(M.display_name(rec["method"], labels))]
        for key, _ in spec:
            cells.append(M.format_mean_sd_tex(
                rec[f"{key}_mean"], rec[f"{key}_sd"], bold=flags[key][i],
            ))
        cells.append(str(int(rec["n_runs"])))
        L.append(" " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


def _cold_table(df: pd.DataFrame, labels: dict[str, str] | None = None) -> tuple[pd.DataFrame, str, str]:
    sub = df[df["test_set"].astype(str).isin(["S1", "S2"])].copy()
    if sub.empty:
        return pd.DataFrame(), "", ""
    methods = M.methods_in(sub)
    rows = []
    for m in methods:
        rec = dict(method=m)
        for ts in ("S1", "S2"):
            g = sub[(sub["method"] == m) & (sub["test_set"] == ts)]
            rec[f"{ts}_n"] = int(len(g))
            for key, _lab in COLD_COLS:
                mu, sd = M.mean_sd(g[key].tolist() if key in g.columns else [])
                rec[f"{ts}_{key}_mean"] = mu
                rec[f"{ts}_{key}_sd"] = sd
        rows.append(rec)
    agg = pd.DataFrame(rows)
    # markdown
    header = ["Method"]
    for ts in ("S1", "S2"):
        for _k, lab in COLD_COLS:
            header.append(f"{ts} {lab}")
        header.append(f"{ts} N")
    lines = [
        "## Cold-start S1/S2 (mean ± SD across drug-split seeds)",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    flags = {}
    for ts in ("S1", "S2"):
        for key, _lab in COLD_COLS:
            col = f"{ts}_{key}_mean"
            mus = agg[col].tolist()
            flags[(ts, key)] = [M.is_best(mus, mu, M.HIGHER_IS_BETTER.get(key, True)) for mu in mus]
    for i, rec in agg.iterrows():
        cells = [M.display_name(rec["method"], labels)]
        for ts in ("S1", "S2"):
            for key, _lab in COLD_COLS:
                cells.append(M.format_mean_sd(
                    rec[f"{ts}_{key}_mean"], rec[f"{ts}_{key}_sd"],
                    bold=flags[(ts, key)][i],
                ))
            cells.append(str(int(rec[f"{ts}_n"])))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    md = "\n".join(lines)

    n_col = 1 + 2 * (len(COLD_COLS) + 1)
    align = "l" + "c" * (n_col - 1)
    L = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Cold-start S1/S2 metrics (mean $\pm$ SD over drug-split seeds). Best in bold.}",
        r"\label{tab:coldstart}",
        f"\\begin{{tabular}}{{{align}}}",
        r"\toprule",
    ]
    tex_heads = ["Method"]
    for ts in ("S1", "S2"):
        for _k, lab in COLD_COLS:
            tex_heads.append(f"{ts} {lab}")
        tex_heads.append(f"{ts} N")
    L.append(" " + " & ".join(M.latex_escape(h) for h in tex_heads) + r" \\")
    L.append(r"\midrule")
    for i, rec in agg.iterrows():
        cells = [M.latex_escape(M.display_name(rec["method"], labels))]
        for ts in ("S1", "S2"):
            for key, _lab in COLD_COLS:
                cells.append(M.format_mean_sd_tex(
                    rec[f"{ts}_{key}_mean"], rec[f"{ts}_{key}_sd"],
                    bold=flags[(ts, key)][i],
                ))
            cells.append(str(int(rec[f"{ts}_n"])))
        L.append(" " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return agg, md, "\n".join(L)


def write_tables(pred_dir, out_dir: str | Path,
                 labels: dict[str, str] | None = None) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = labels if labels is not None else M.DEFAULT_METHOD_LABELS
    df = _metrics_frame(pred_dir)
    written: dict[str, Path] = {}
    if df.empty:
        (out_dir / "README.txt").write_text(
            "No runs found in pred_dir / state.\n", encoding="utf-8",
        )
        return written

    pair = df[
        (df["split"].astype(str) == "pair")
        & df["test_set"].astype(str).isin(["test", "holdout"])
    ].copy()
    md_blocks = ["# Result tables (mean ± SD across fold × seed)", ""]
    tex_blocks = [
        "% Requires \\usepackage{booktabs}",
        "% Values are mean ± SD; best per column in bold.",
        "",
    ]

    if not pair.empty:
        agg = _agg(pair, [k for k, _ in MAIN_COLS + ORD_COLS])
        agg.to_csv(out_dir / "main_and_ordinal_raw.csv", index=False, encoding="utf-8-sig")
        written["raw_pair"] = out_dir / "main_and_ordinal_raw.csv"
        md_main = _md_table(agg, MAIN_COLS, "Main results (pair holdout)", labels)
        md_ord = _md_table(agg, ORD_COLS, "Ordinal / clinical metrics (pair holdout)", labels)
        md_pc = _md_per_class(agg, labels)
        (out_dir / "main_results.md").write_text(md_main, encoding="utf-8")
        (out_dir / "ordinal_clinical.md").write_text(md_ord, encoding="utf-8")
        (out_dir / "per_class_f1.md").write_text(md_pc, encoding="utf-8")
        tex_main = _tex_table(
            agg, MAIN_COLS,
            "Main classification metrics (mean $\\pm$ SD). Best in bold.",
            "tab:main_results", labels,
        )
        tex_ord = _tex_table(
            agg, ORD_COLS,
            "Ordinal and clinical metrics (mean $\\pm$ SD). Best in bold.",
            "tab:ordinal", labels,
        )
        tex_pc = _tex_per_class(agg, labels)
        (out_dir / "main_results.tex").write_text(tex_main, encoding="utf-8")
        (out_dir / "ordinal_clinical.tex").write_text(tex_ord, encoding="utf-8")
        (out_dir / "per_class_f1.tex").write_text(tex_pc, encoding="utf-8")
        md_blocks += [md_main, md_ord, md_pc]
        tex_blocks += [tex_main, tex_ord, tex_pc]
        written.update(dict(
            main_md=out_dir / "main_results.md",
            main_tex=out_dir / "main_results.tex",
            ordinal_md=out_dir / "ordinal_clinical.md",
            ordinal_tex=out_dir / "ordinal_clinical.tex",
            perclass_md=out_dir / "per_class_f1.md",
            perclass_tex=out_dir / "per_class_f1.tex",
        ))
    else:
        md_blocks.append("*No pair-holdout runs.*\n")

    cold_agg, cold_md, cold_tex = _cold_table(df, labels)
    if cold_md:
        cold_agg.to_csv(out_dir / "coldstart_s1s2_raw.csv", index=False, encoding="utf-8-sig")
        (out_dir / "coldstart_s1s2.md").write_text(cold_md, encoding="utf-8")
        (out_dir / "coldstart_s1s2.tex").write_text(cold_tex, encoding="utf-8")
        md_blocks.append(cold_md)
        tex_blocks.append(cold_tex)
        written.update(dict(
            cold_md=out_dir / "coldstart_s1s2.md",
            cold_tex=out_dir / "coldstart_s1s2.tex",
        ))
    else:
        md_blocks.append("*No S1/S2 cold-start runs.*\n")

    (out_dir / "all_tables.md").write_text("\n".join(md_blocks), encoding="utf-8")
    (out_dir / "all_tables.tex").write_text("\n".join(tex_blocks), encoding="utf-8")
    written["all_md"] = out_dir / "all_tables.md"
    written["all_tex"] = out_dir / "all_tables.tex"
    print(f"[summarize] wrote {len(written)} files under {out_dir}")
    return written


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Summarize runs to Markdown/LaTeX tables")
    M.add_pred_dir_arg(p)
    p.add_argument("--out_dir", required=True)
    M.add_method_names_arg(p)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    write_tables(args.pred_dir, args.out_dir, labels=M.parse_method_names(args.method_names))


if __name__ == "__main__":
    main()
