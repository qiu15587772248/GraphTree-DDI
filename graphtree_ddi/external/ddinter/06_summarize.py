"""Summarize DDInter inference CSV (no plots).

Input CSV columns (from 05_run_inference.py):
  drug1_id, drug2_id, ext_level, y_pred, p0..p4, group, drugbank_label

Writes summary_{subset}_{model}.md and .json next to --pred (or --out_dir).
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
import re
from pathlib import Path

import numpy as np
import pandas as pd

from graphtree_ddi.external.ddinter._common import configure_stdio, now_cst_iso, write_json

LEVELS = ("Major", "Moderate", "Minor", "Unknown")
LEVEL_ORD = {"Major": 3, "Moderate": 2, "Minor": 1}  # Unknown excluded
PRED_CLASSES = (0, 1, 2, 3, 4)
FNAME_RE = re.compile(
    r"ddinter_pred_(absent|negative|positive|all)_(exp1|exp5)\.csv$",
    re.IGNORECASE,
)
GROUP_TO_SUBSET = {
    "absent_in_drugbank": "absent",
    "in_drugbank_negative": "negative",
    "in_drugbank_positive": "positive",
}
SUBSET_TO_GROUP = {v: k for k, v in GROUP_TO_SUBSET.items()}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DDInter 推理结果汇总（无图）")
    p.add_argument("--pred", type=str, required=True, help="05 输出 CSV")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument(
        "--subset",
        choices=["absent", "negative", "positive", "all", "auto"],
        default="auto",
    )
    p.add_argument("--model", choices=["exp1", "exp5", "auto"], default="auto")
    return p.parse_args(argv)


def infer_meta(pred_path: Path, df: pd.DataFrame, subset: str, model: str) -> tuple[str, str]:
    m = FNAME_RE.search(pred_path.name)
    fname_subset = m.group(1).lower() if m else None
    fname_model = m.group(2).lower() if m else None
    groups = sorted({g for g in df["group"].dropna().astype(str).str.strip().unique() if g})
    if subset == "auto":
        if fname_subset:
            subset = fname_subset
        elif len(groups) == 1 and groups[0] in GROUP_TO_SUBSET:
            subset = GROUP_TO_SUBSET[groups[0]]
        else:
            subset = "all"
    if model == "auto":
        model = fname_model or "exp5"
    return subset, model


def expected_risk(df: pd.DataFrame) -> pd.Series:
    acc = None
    for k in PRED_CLASSES:
        pk = pd.to_numeric(df[f"p{k}"], errors="coerce")
        acc = pk * k if acc is None else acc + pk * k
    return acc


def _rate(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return float(num) / float(den)


def pred_dist(series: pd.Series) -> dict[str, int]:
    y = pd.to_numeric(series, errors="coerce")
    out = {str(c): int((y == c).sum()) for c in PRED_CLASSES}
    out["na"] = int(y.isna().sum())
    return out


def group_metrics(sub: pd.DataFrame) -> dict:
    y = pd.to_numeric(sub["y_pred"], errors="coerce")
    inferred = sub.loc[y.notna()].copy()
    yi = pd.to_numeric(inferred["y_pred"], errors="coerce")
    er = expected_risk(inferred) if len(inferred) else pd.Series(dtype=float)
    n = int(len(sub))
    n_inf = int(len(inferred))
    ge1 = int((yi >= 1).sum()) if n_inf else 0
    ge3 = int((yi >= 3).sum()) if n_inf else 0
    er_valid = er.dropna() if len(er) else er
    return {
        "n": n,
        "n_inferred": n_inf,
        "n_skipped": n - n_inf,
        "n_pred_ge1": ge1,
        "n_pred_ge3": ge3,
        "rate_pred_ge1": _rate(ge1, n_inf),
        "rate_pred_ge3": _rate(ge3, n_inf),
        "expected_risk_mean": float(er_valid.mean()) if len(er_valid) else None,
        "expected_risk_median": float(er_valid.median()) if len(er_valid) else None,
        "pred_dist": pred_dist(sub["y_pred"]),
    }


def spearman_vs_level(df: pd.DataFrame) -> dict:
    y = pd.to_numeric(df["y_pred"], errors="coerce")
    er = expected_risk(df)
    level = df["ext_level"].astype(str)
    mask = y.notna() & er.notna() & level.isin(LEVEL_ORD)
    n = int(mask.sum())
    payload = {
        "n": n,
        "mapping": {"Major": 3, "Moderate": 2, "Minor": 1},
        "excluded": "Unknown 及 y_pred/期望风险缺失行",
        "rho": None,
        "pvalue": None,
    }
    if n < 3:
        payload["note"] = "有效样本不足，未计算 Spearman"
        return payload
    x = level.loc[mask].map(LEVEL_ORD).astype(float).to_numpy()
    r = er.loc[mask].astype(float).to_numpy()
    if np.unique(x).size < 2 or np.unique(r).size < 2:
        payload["note"] = "某一侧无变异，Spearman 无定义"
        return payload
    from scipy.stats import spearmanr

    rho, pval = spearmanr(x, r)
    payload["rho"] = float(rho) if rho == rho else None
    payload["pvalue"] = float(pval) if pval == pval else None
    return payload


def auroc_major_vs_minor(df: pd.DataFrame) -> dict:
    y = pd.to_numeric(df["y_pred"], errors="coerce")
    p3 = pd.to_numeric(df["p3"], errors="coerce")
    p4 = pd.to_numeric(df["p4"], errors="coerce")
    score = p3 + p4
    level = df["ext_level"].astype(str)
    mask = y.notna() & score.notna() & level.isin(["Major", "Minor"])
    n_major = int((mask & (level == "Major")).sum())
    n_minor = int((mask & (level == "Minor")).sum())
    payload = {
        "score": "P(>=3)=p3+p4",
        "n_major": n_major,
        "n_minor": n_minor,
        "auroc": None,
    }
    if n_major == 0 or n_minor == 0:
        payload["note"] = "Major 或 Minor 有效样本为 0，未计算 AUROC"
        return payload
    from sklearn.metrics import roc_auc_score

    y_bin = (level.loc[mask] == "Major").astype(int).to_numpy()
    s = score.loc[mask].astype(float).to_numpy()
    payload["auroc"] = float(roc_auc_score(y_bin, s))
    return payload


def looks_like_smoke(pred_path: Path, out_dir: Path) -> tuple[bool, str]:
    stats_path = out_dir / "inference_stats.json"
    if stats_path.exists():
        import json

        try:
            st = json.loads(stats_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            st = {}
        if st.get("smoke_model") or st.get("paper_warning"):
            return True, str(st.get("paper_warning") or "冒烟模型结果不可用于论文")
        if "smoke" in str(st.get("bundle", "")).lower():
            return True, "推理包路径含 smoke：冒烟模型结果不可用于论文"
    blob = (str(pred_path) + " " + str(out_dir)).lower()
    if "smoke" in blob:
        return True, "路径含 smoke：冒烟模型结果不可用于论文"
    return False, ""


def fmt_pct(rate: float | None) -> str:
    if rate is None:
        return "n/a"
    return f"{100.0 * rate:.2f}%"


def fmt_num(x: float | None, nd=4) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "n/a"
    return f"{x:.{nd}f}"


def dist_table(dist: dict[str, int]) -> str:
    lines = ["| 预测等级 | 条数 |", "|---|---|"]
    for c in PRED_CLASSES:
        lines.append(f"| {c} | {dist.get(str(c), 0):,} |")
    lines.append(f"| （空 / 跳过） | {dist.get('na', 0):,} |")
    return "\n".join(lines)


def build_absent_section(df: pd.DataFrame) -> tuple[str, dict]:
    by_level = {}
    rows = [
        "| ext_level | n | 可推理 | 检出有相互作用 (≥1) | 预测 ≥3 | 期望风险均值 | 期望风险中位数 |",
        "|---|---|---|---|---|---|---|",
    ]
    for lv in LEVELS:
        sub = df.loc[df["ext_level"].astype(str) == lv]
        met = group_metrics(sub)
        by_level[lv] = met
        r1 = fmt_pct(met["rate_pred_ge1"])
        r3 = fmt_pct(met["rate_pred_ge3"])
        if met["n_inferred"]:
            r1 = f"{r1} ({met['n_pred_ge1']}/{met['n_inferred']})"
            r3 = f"{r3} ({met['n_pred_ge3']}/{met['n_inferred']})"
        rows.append(
            f"| {lv} | {met['n']:,} | {met['n_inferred']:,} | {r1} | {r3} | "
            f"{fmt_num(met['expected_risk_mean'])} | {fmt_num(met['expected_risk_median'])} |"
        )
    overall = group_metrics(df)
    by_level["overall"] = overall
    r1 = fmt_pct(overall["rate_pred_ge1"])
    r3 = fmt_pct(overall["rate_pred_ge3"])
    if overall["n_inferred"]:
        r1 = f"{r1} ({overall['n_pred_ge1']}/{overall['n_inferred']})"
        r3 = f"{r3} ({overall['n_pred_ge3']}/{overall['n_inferred']})"
    rows.append(
        f"| **合计** | {overall['n']:,} | {overall['n_inferred']:,} | {r1} | {r3} | "
        f"{fmt_num(overall['expected_risk_mean'])} | {fmt_num(overall['expected_risk_median'])} |"
    )
    sp = spearman_vs_level(df)
    au = auroc_major_vs_minor(df)
    md = "\n".join([
        "## absent 子集",
        "",
        "检出有相互作用 = 预测等级 ≥ 1（相对 0=无相互作用）。分母为可推理条数。",
        "",
        "### 按 ext_level",
        "",
        *rows,
        "",
        "### 预测等级分布（全体）",
        "",
        dist_table(overall["pred_dist"]),
        "",
        "### 序相关与 Major–Minor 区分",
        "",
        f"- Spearman ρ（Major=3, Moderate=2, Minor=1 vs 期望风险 Σ k·p_k；排除 Unknown）："
        f" **{fmt_num(sp.get('rho'))}**（p={fmt_num(sp.get('pvalue'), 4)}，n={sp['n']:,}）",
        f"- Major vs Minor AUROC（打分 P(≥3)=p3+p4）："
        f" **{fmt_num(au.get('auroc'))}**（Major n={au['n_major']:,}，Minor n={au['n_minor']:,}）",
    ])
    return md, {"by_ext_level": by_level, "spearman": sp, "auroc_major_vs_minor": au, "overall": overall}


def build_negative_section(df: pd.DataFrame) -> tuple[str, dict]:
    overall = group_metrics(df)
    md = "\n".join([
        "## negative 子集（DrugBank 负样本交叉）",
        "",
        f"- 行数 {overall['n']:,}，可推理 {overall['n_inferred']:,}，跳过 {overall['n_skipped']:,}",
        "",
        "### 预测等级分布",
        "",
        dist_table(overall["pred_dist"]),
    ])
    return md, {"overall": overall, "pred_dist": overall["pred_dist"]}


def build_positive_section(df: pd.DataFrame) -> tuple[str, dict]:
    overall = group_metrics(df)
    md = "\n".join([
        "## positive 子集（DrugBank 正样本交叉）",
        "",
        f"- 行数 {overall['n']:,}，可推理 {overall['n_inferred']:,}，跳过 {overall['n_skipped']:,}",
        "",
        "本脚本对 positive 只给预测等级分布；absent 专用的 Spearman / AUROC 不在此计算。",
        "",
        dist_table(overall["pred_dist"]),
    ])
    return md, {"overall": overall, "pred_dist": overall["pred_dist"]}


def rows_for(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    if subset == "all":
        return df
    g = SUBSET_TO_GROUP[subset]
    if "group" in df.columns:
        hit = df.loc[df["group"].astype(str) == g]
        if len(hit):
            return hit
    return df


def main(argv=None) -> int:
    configure_stdio()
    args = parse_args(argv)
    pred_path = Path(args.pred)
    if not pred_path.exists():
        raise SystemExit(f"pred 不存在: {pred_path}")
    out_dir = Path(args.out_dir) if args.out_dir else pred_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_path, dtype=str, encoding="utf-8")
    need = {"drug1_id", "drug2_id", "ext_level", "y_pred", "p0", "p1", "p2", "p3", "p4"}
    if not need.issubset(df.columns):
        raise SystemExit(f"缺列 {need - set(df.columns)}，实际 {list(df.columns)}")
    if "group" not in df.columns:
        df["group"] = ""
    subset, model = infer_meta(pred_path, df, args.subset, args.model)
    work = rows_for(df, subset).copy()
    smoke, warn = looks_like_smoke(pred_path, out_dir)

    sections_md = []
    payload: dict = {
        "generated_at_utc8": now_cst_iso(),
        "pred": str(pred_path),
        "subset": subset,
        "model": model,
        "n_rows_file": int(len(df)),
        "n_rows_used": int(len(work)),
        "smoke_model": smoke,
        "paper_warning": warn if smoke else "",
        "no_plots": True,
    }

    groups_present = set(work["group"].dropna().astype(str))
    want_absent = subset in {"absent", "all"}
    want_neg = subset in {"negative", "all"}
    want_pos = subset in {"positive", "all"}
    if subset == "all":
        want_absent = "absent_in_drugbank" in groups_present
        want_neg = "in_drugbank_negative" in groups_present
        want_pos = "in_drugbank_positive" in groups_present

    if want_absent:
        sub = work if subset == "absent" else rows_for(work, "absent")
        md, js = build_absent_section(sub)
        sections_md.append(md)
        payload["absent"] = js
    if want_neg:
        sub = work if subset == "negative" else rows_for(work, "negative")
        md, js = build_negative_section(sub)
        sections_md.append(md)
        payload["negative"] = js
    if want_pos:
        sub = work if subset == "positive" else rows_for(work, "positive")
        md, js = build_positive_section(sub)
        sections_md.append(md)
        payload["positive"] = js

    warn_block = ""
    if smoke:
        warn_block = (
            "> **警告：冒烟模型结果不可用于论文。** "
            "本机 `final_smoke_full` 为 v1 数据上 2 epoch / 50 树的流程验证包，"
            "数字只能证明脚本跑通，不能写入论文表格。\n\n"
        )

    head = [
        f"# DDInter 推理汇总（subset={subset}, model={model}）",
        "",
        warn_block.rstrip(),
        "",
        f"- 输入：`{pred_path}`",
        f"- 生成时间（UTC+8）：{payload['generated_at_utc8']}",
        f"- 文件行数：{payload['n_rows_file']:,}；本汇总使用：{payload['n_rows_used']:,}",
        "- 本脚本不画图；图由 `models/analysis/external_eval.py` 负责。",
        "",
    ]
    md_text = "\n".join([ln for ln in head if ln is not None]) + "\n" + "\n\n".join(sections_md) + "\n"
    # collapse accidental triple newlines at top
    while "\n\n\n" in md_text:
        md_text = md_text.replace("\n\n\n", "\n\n")

    stem = f"summary_{subset}_{model}"
    md_path = out_dir / f"{stem}.md"
    js_path = out_dir / f"{stem}.json"
    md_path.write_text(md_text, encoding="utf-8")
    write_json(js_path, payload)
    print(f"[06] subset={subset} model={model} → {md_path}", flush=True)
    print(f"[06] json={js_path}", flush=True)
    if smoke:
        print("[06] WARNING: 冒烟模型结果不可用于论文", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
