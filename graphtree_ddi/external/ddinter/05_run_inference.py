"""Run frozen-bundle inference on DDInter-mapped pairs.

Reuses models/predict_pairs.py (load_bundle / predict_rows). Does not copy
model code. Writes downstream columns expected by models/analysis/external_eval.py
plus group / drugbank_label.

Usage (PowerShell, repo-relative or absolute paths via pathlib):

  $env:PYTHONIOENCODING='utf-8'
  python .\\05_run_inference.py `
    --bundle <final_full> --data_dir <processed> --out_dir <dir> `
    --subset absent --model exp5 [--limit N]
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
import time
from pathlib import Path

import pandas as pd

from graphtree_ddi.external.ddinter._common import configure_stdio, now_cst_iso, write_json

HERE = Path(__file__).resolve().parent
DEFAULT_PAIRS = HERE / "ddinter_mapped_pairs.csv"

SUBSETS = ("absent", "negative", "positive", "all")
MODELS = ("exp5", "exp1", "both")
PRED_COLS = ["drug1_id", "drug2_id", "ext_level", "y_pred", "p0", "p1", "p2", "p3", "p4"]
EXTRA_COLS = ["group", "drugbank_label"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DDInter 映射表 → predict_pairs 推理")
    p.add_argument("--bundle", type=str, required=True, help="final_full 推理包目录")
    p.add_argument("--data_dir", type=str, required=True, help="processed 数据目录")
    p.add_argument("--out_dir", type=str, required=True, help="输出目录")
    p.add_argument("--subset", choices=SUBSETS, default="absent")
    p.add_argument("--model", choices=MODELS, default="exp5")
    p.add_argument("--limit", type=int, default=None, help="子集截断条数（文件顺序）")
    p.add_argument("--pairs_csv", type=str, default=None, help="默认本目录 ddinter_mapped_pairs.csv")
    return p.parse_args(argv)


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    t = series.astype(str).str.strip().str.lower()
    return t.isin(["true", "1", "yes"])


def _positive_label(series: pd.Series) -> pd.Series:
    nums = pd.to_numeric(series, errors="coerce")
    return nums.isin([1, 2, 3, 4])


def membership_group(df: pd.DataFrame) -> pd.Series:
    pos = _positive_label(df["in_drugbank_positive"])
    neg = _truthy(df["in_drugbank_negative"])
    abs_ = _truthy(df["absent_in_drugbank"])
    out = pd.Series("other", index=df.index, dtype=object)
    out = out.mask(abs_, "absent_in_drugbank")
    out = out.mask(neg, "in_drugbank_negative")
    out = out.mask(pos, "in_drugbank_positive")
    return out


def drugbank_label_series(df: pd.DataFrame) -> pd.Series:
    nums = pd.to_numeric(df["in_drugbank_positive"], errors="coerce")
    out = pd.Series("", index=df.index, dtype=object)
    ok = nums.isin([1, 2, 3, 4])
    out.loc[ok] = nums.loc[ok].astype(int).astype(str)
    return out


def filter_subset(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    if subset == "all":
        return df.copy()
    if subset == "positive":
        return df.loc[_positive_label(df["in_drugbank_positive"])].copy()
    if subset == "negative":
        return df.loc[_truthy(df["in_drugbank_negative"])].copy()
    if subset == "absent":
        return df.loc[_truthy(df["absent_in_drugbank"])].copy()
    raise ValueError(subset)


def _import_predict_pairs():
    import graphtree_ddi.models.predict_pairs as pp  # noqa: WPS433

    return pp


def bundle_looks_like_smoke(bundle: Path) -> bool:
    text = bundle.as_posix().lower()
    return "smoke" in text


def models_requested(model: str) -> list[str]:
    if model == "both":
        return ["exp1", "exp5"]
    return [model]


def format_export(
    mapped: pd.DataFrame,
    pred: pd.DataFrame,
    model_key: str,
) -> pd.DataFrame:
    n_cls = 5
    pred_col = f"{model_key}_pred"
    if pred_col not in pred.columns:
        raise KeyError(f"predict_pairs 输出缺少 {pred_col}，实际列: {list(pred.columns)}")
    y = pd.to_numeric(pred[pred_col], errors="coerce")
    out = pd.DataFrame({
        "drug1_id": mapped["drug_a"].astype(str).str.strip().values,
        "drug2_id": mapped["drug_b"].astype(str).str.strip().values,
        "ext_level": mapped["ddinter_level"].astype(str).str.strip().values,
        "y_pred": y.astype("Int64"),
        "group": membership_group(mapped).values,
        "drugbank_label": drugbank_label_series(mapped).values,
    })
    for c in range(n_cls):
        src = f"{model_key}_p{c}"
        if src not in pred.columns:
            out[f"p{c}"] = pd.NA
        else:
            out[f"p{c}"] = pd.to_numeric(pred[src], errors="coerce")
    return out[PRED_COLS + EXTRA_COLS]


def count_inferred(export: pd.DataFrame) -> tuple[int, int]:
    ok = export["y_pred"].notna()
    n_ok = int(ok.sum())
    n_skip = int((~ok).sum())
    return n_ok, n_skip


def main(argv=None) -> int:
    configure_stdio()
    args = parse_args(argv)
    bundle = Path(args.bundle)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    pairs_csv = Path(args.pairs_csv) if args.pairs_csv else DEFAULT_PAIRS
    if not bundle.exists():
        raise SystemExit(f"bundle 不存在: {bundle}")
    if not pairs_csv.exists():
        raise SystemExit(f"映射表不存在: {pairs_csv}")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print(f"[05] pairs_csv={pairs_csv}", flush=True)
    mapped_all = pd.read_csv(pairs_csv, dtype=str, encoding="utf-8")
    need = {"drug_a", "drug_b", "ddinter_level",
            "in_drugbank_positive", "in_drugbank_negative", "absent_in_drugbank"}
    if not need.issubset(mapped_all.columns):
        raise SystemExit(f"映射表缺列 {need - set(mapped_all.columns)}")

    n_source = int(len(mapped_all))
    mapped = filter_subset(mapped_all, args.subset)
    n_subset = int(len(mapped))
    if args.limit is not None:
        if args.limit < 0:
            raise SystemExit("--limit 必须 >= 0")
        mapped = mapped.iloc[: args.limit].copy()
    mapped = mapped.reset_index(drop=True)
    n_query = int(len(mapped))
    print(
        f"[05] subset={args.subset}  source={n_source:,}  "
        f"subset_n={n_subset:,}  query={n_query:,}  model={args.model}",
        flush=True,
    )
    if n_query == 0:
        raise SystemExit("子集为空，无推理行")

    tmp_in = out_dir / f"_tmp_query_{args.subset}.csv"
    query = mapped[["drug_a", "drug_b"]].copy()
    query.to_csv(tmp_in, index=False, encoding="utf-8")

    pp = _import_predict_pairs()
    import torch  # after path setup via predict_pairs side effects is fine

    drugs_csv = pp._resolve_drugs_csv(data_dir, None)
    drugs_df = pd.read_csv(drugs_csv)
    drugs_dict = {row["drugbank_id"]: row.to_dict() for _, row in drugs_df.iterrows()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[05] import predict_pairs OK  device={device}  drugs_csv={drugs_csv}", flush=True)
    print(f"[05] predict_pairs 输出列将含 Exp-1/Exp-3/Exp-5 的 pred 与 p0..p4", flush=True)

    ctx = pp.load_bundle(bundle, device)
    df_in = pd.read_csv(tmp_in, dtype=str)
    pred = pp.predict_rows(df_in, ctx, drugs_dict)
    if len(pred) != n_query:
        raise SystemExit(f"predict_rows 行数 {len(pred)} != 查询 {n_query}")

    elapsed = time.perf_counter() - t0
    smoke = bundle_looks_like_smoke(bundle)
    model_keys = models_requested(args.model)
    by_model = {}
    outputs = []
    for mk in model_keys:
        export = format_export(mapped, pred, mk)
        n_ok, n_skip = count_inferred(export)
        out_csv = out_dir / f"ddinter_pred_{args.subset}_{mk}.csv"
        export.to_csv(out_csv, index=False, encoding="utf-8")
        outputs.append(str(out_csv))
        by_model[mk] = {
            "n_inferred": n_ok,
            "n_skipped": n_skip,
            "output": str(out_csv),
        }
        print(
            f"[05] {mk}: inferred={n_ok:,}  skipped={n_skip:,}  → {out_csv}",
            flush=True,
        )

    # Top-level inferred/skipped: single model, or exp5 if both.
    primary = "exp5" if "exp5" in by_model else model_keys[0]
    stats = {
        "generated_at_utc8": now_cst_iso(),
        "bundle": str(bundle),
        "data_dir": str(data_dir),
        "pairs_csv": str(pairs_csv),
        "subset": args.subset,
        "model": args.model,
        "limit": args.limit,
        "n_mapped_source": n_source,
        "n_subset": n_subset,
        "n_queried": n_query,
        "n_inferred": by_model[primary]["n_inferred"],
        "n_skipped": by_model[primary]["n_skipped"],
        "elapsed_seconds": round(elapsed, 3),
        "device": str(device),
        "by_model": by_model,
        "outputs": outputs,
        "tmp_query": str(tmp_in),
        "smoke_model": smoke,
        "paper_warning": (
            "冒烟模型结果不可用于论文（2 epoch / 50 树，仅验证流程）。"
            if smoke
            else ""
        ),
        "predict_pairs_columns_observed": list(pred.columns),
        "export_columns": PRED_COLS + EXTRA_COLS,
        "skipped_definition": "y_pred 为空（未知药物 ID 或不在索引 / 无法构造特征）",
        "note": (
            "predict_pairs 内部同时计算 Exp-1、Exp-3、Exp-5；"
            "本脚本按 --model 抽取 y_pred 与 p0..p4。"
        ),
    }
    stats_path = out_dir / "inference_stats.json"
    write_json(stats_path, stats)
    print(f"[05] elapsed={elapsed:.1f}s  stats={stats_path}", flush=True)
    if smoke:
        print("[05] WARNING: 冒烟模型结果不可用于论文", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
