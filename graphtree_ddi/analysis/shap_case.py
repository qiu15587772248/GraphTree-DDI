"""XGBoost TreeSHAP：全局 top-20 与单药对 waterfall。

特征名:
  [0:2048]    fp_prod_*
  [2048:4096] fp_diff_*
  [4096:4120] data.preprocess.BIO_FEATURE_NAMES（24 维药理真实名称）
  Exp-5 融合尾部 4096 维（run_pipeline_final.PAIR_OPS_SPEC / pair_ops_embed）:
    concat([h_a, h_b, h_a-h_b, h_a*h_b])，emb_dim=1024
    [4120:5144] gnn_ha_*
    [5144:6168] gnn_hb_*
    [6168:7192] gnn_diff_*
    [7192:8216] gnn_prod_*

--self_test 用 sklearn 随机数据训练小 XGB，验证脚本可跑。
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
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_THIS = Path(__file__).resolve()
_ANALYSIS = _THIS.parent
_PROJ = _THIS.parents[2]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

from graphtree_ddi.analysis.plot_figures import apply_paper_style, save_fig  # noqa: E402

FP_PROD = 2048
FP_DIFF = 2048
N_BIO = 24
XGB_DIM = FP_PROD + FP_DIFF + N_BIO  # 4120
GNN_EMB = 1024
PAIR_OPS_DIM = GNN_EMB * 4  # 4096
FUSION_DIM = XGB_DIM + PAIR_OPS_DIM  # 8216

_FALLBACK_BIO = [
    "CYP底物重叠率", "CYP共享底物(有/无)", "CYP底物-抑制剂交互", "CYP底物-诱导剂交互",
    "CYP抑制剂重叠率", "CYP诱导剂重叠率", "共享CYP酶总数", "CYP任意交互(有/无)",
    "转运体底物重叠率", "转运体共享底物(有/无)", "转运体底物-抑制剂交互", "转运体底物-诱导剂交互",
    "P-gp共享(有/无)", "P-gp底物-抑制剂交互", "OATP1B1共享(有/无)", "共享转运体总数",
    "靶点重叠Jaccard", "靶点共享(有/无)", "靶点激动剂重叠率", "靶点拮抗剂重叠率",
    "靶点抑制剂重叠率", "靶点激动-拮抗冲突", "靶点激动-抑制冲突", "共享靶点总数",
]


def bio_feature_names() -> list[str]:
    try:
        from graphtree_ddi.data.preprocess import BIO_FEATURE_NAMES
        names = list(BIO_FEATURE_NAMES)
        if len(names) == N_BIO:
            return names
    except Exception:  # noqa: BLE001
        pass
    return list(_FALLBACK_BIO)


def feature_names_for_dim(n_features: int) -> list[str]:
    names = [f"fp_prod_{i}" for i in range(FP_PROD)]
    names += [f"fp_diff_{i}" for i in range(FP_DIFF)]
    names += bio_feature_names()
    if n_features == XGB_DIM:
        return names
    if n_features >= FUSION_DIM or n_features == XGB_DIM + PAIR_OPS_DIM:
        names = names[:XGB_DIM]
        names += [f"gnn_ha_{i}" for i in range(GNN_EMB)]
        names += [f"gnn_hb_{i}" for i in range(GNN_EMB)]
        names += [f"gnn_diff_{i}" for i in range(GNN_EMB)]
        names += [f"gnn_prod_{i}" for i in range(GNN_EMB)]
        if n_features > len(names):
            names += [f"extra_{i}" for i in range(n_features - len(names))]
        return names[:n_features]
    if n_features < len(names):
        return [f"f_{i}" for i in range(n_features)]
    names += [f"extra_{i}" for i in range(n_features - len(names))]
    return names


def _enable_cjk_ticks() -> None:
    apply_paper_style()
    plt.rcParams["font.serif"] = [
        "Times New Roman", "Times", "Microsoft YaHei", "SimHei", "SimSun", "DejaVu Serif",
    ]
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def load_booster(model_path: Path):
    from xgboost import XGBClassifier, Booster
    model_path = Path(model_path)
    suf = model_path.suffix.lower()
    if suf in (".json", ".ubj", ".bin", ".model"):
        try:
            clf = XGBClassifier()
            clf.load_model(str(model_path))
            return clf
        except Exception:
            bst = Booster()
            bst.load_model(str(model_path))
            return bst
    raise ValueError(f"不支持的模型后缀: {model_path}")


def _n_features(model) -> int:
    if hasattr(model, "n_features_in_") and model.n_features_in_:
        return int(model.n_features_in_)
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    # xgboost 3: num_feature
    try:
        return int(booster.num_feature())
    except Exception:
        pass
    cfg = json.loads(booster.save_config())
    try:
        return int(cfg["learner"]["learner_model_param"]["num_feature"])
    except Exception as e:
        raise RuntimeError(f"无法解析特征数: {e}") from e


def shap_values_abs_mean(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, object]:
    import shap
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    # list[n_classes] of (n, f)  or  ndarray (n, f, c) / (n, f)
    if isinstance(sv, list):
        arr = np.stack([np.asarray(a) for a in sv], axis=-1)  # (n, f, c)
    else:
        arr = np.asarray(sv)
        if arr.ndim == 2:
            arr = arr[..., None]
    mean_abs = np.mean(np.abs(arr), axis=(0, 2))
    expected = np.asarray(explainer.expected_value).reshape(-1)
    return mean_abs, arr, expected


def _cjk_font():
    from matplotlib.font_manager import FontProperties, fontManager
    for name in ("Microsoft YaHei", "SimHei", "SimSun", "Microsoft JhengHei", "Noto Sans CJK SC"):
        try:
            path = fontManager.findfont(FontProperties(family=name), fallback_to_default=False)
        except (ValueError, Exception):
            continue
        if path and all(x not in path.replace("\\", "/").lower() for x in ("dejavu", "times")):
            return FontProperties(fname=path)
    return FontProperties()


def plot_top20(names: list[str], mean_abs: np.ndarray, stem: Path, title: str) -> list[Path]:
    _enable_cjk_ticks()
    order = np.argsort(mean_abs)[::-1][:20]
    vals = mean_abs[order][::-1]
    labs = [names[i] for i in order][::-1]
    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    ax.barh(range(len(vals)), vals, color="#0072B2")
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labs, fontsize=8, fontproperties=_cjk_font())
    ax.set_xlabel("Mean |SHAP| (averaged over classes)")
    ax.set_title(title)
    fig.tight_layout()
    return save_fig(fig, stem)


def plot_waterfall(
    names: list[str],
    shap_row: np.ndarray,
    base: float,
    stem: Path,
    title: str,
    top_k: int = 18,
) -> list[Path]:
    _enable_cjk_ticks()
    shap_row = np.asarray(shap_row, dtype=np.float64).reshape(-1)
    order = np.argsort(np.abs(shap_row))[::-1]
    top = order[:top_k]
    rest = shap_row[order[top_k:]].sum() if order.size > top_k else 0.0
    vals = list(shap_row[top]) + ([rest] if order.size > top_k else [])
    labs = [names[i] for i in top] + (["other"] if order.size > top_k else [])
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    y = np.arange(len(vals))[::-1]
    left = float(base)
    for yi, v, lab in zip(y, vals, labs):
        color = "#D55E00" if v >= 0 else "#0072B2"
        ax.barh(yi, v, left=left, color=color, height=0.7)
        left += v
    ax.axvline(base, color="#888888", ls="--", lw=0.8, label="E[f(x)]")
    ax.axvline(left, color="black", ls=":", lw=0.8, label="f(x)")
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=8, fontproperties=_cjk_font())
    ax.set_xlabel("SHAP contribution (log-odds / margin)")
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return save_fig(fig, stem)


def train_toy_xgb(out_dir: Path, seed: int = 0):
    from sklearn.datasets import make_classification
    from xgboost import XGBClassifier
    X, y = make_classification(
        n_samples=400, n_features=40, n_informative=16, n_redundant=4,
        n_classes=5, n_clusters_per_class=1, random_state=int(seed),
    )
    X = X.astype(np.float32)
    clf = XGBClassifier(
        n_estimators=30, max_depth=3, learning_rate=0.2,
        subsample=0.9, colsample_bytree=0.9,
        objective="multi:softprob", num_class=5,
        eval_metric="mlogloss", n_jobs=1, random_state=int(seed),
        tree_method="hist",
    )
    clf.fit(X, y)
    model_path = Path(out_dir) / "toy_xgb.json"
    clf.save_model(str(model_path))
    np.save(Path(out_dir) / "toy_X.npy", X)
    np.save(Path(out_dir) / "toy_y.npy", y.astype(np.int64))
    return clf, X, y, model_path


def _pair_caption(pairs_csv: str | Path | None, row: int, drugs_csv: str | Path | None = None) -> str:
    if not pairs_csv:
        return ""
    p = Path(pairs_csv)
    if not p.exists():
        return ""
    import pandas as pd
    df = pd.read_csv(p)
    if row < 0 or row >= len(df):
        return ""
    rec = df.iloc[int(row)]
    na = rec["name_a"] if "name_a" in df.columns else None
    nb = rec["name_b"] if "name_b" in df.columns else None
    da = rec["drug_a"] if "drug_a" in df.columns else rec.get("drug1_id", "")
    db = rec["drug_b"] if "drug_b" in df.columns else rec.get("drug2_id", "")
    if (na is None or str(na) == "nan") or (nb is None or str(nb) == "nan"):
        if drugs_csv and Path(drugs_csv).exists():
            names = pd.read_csv(drugs_csv, usecols=lambda c: c in ("drugbank_id", "name"))
            m = dict(zip(names["drugbank_id"].astype(str), names["name"].astype(str)))
            na = m.get(str(da), str(da))
            nb = m.get(str(db), str(db))
        else:
            na, nb = str(da), str(db)
    return f"{na} + {nb}"


def run_shap(
    model_path: str | Path,
    out_dir: str | Path,
    *,
    X: np.ndarray | None = None,
    row: int = 0,
    n_explain: int = 128,
    title_prefix: str = "",
    pair_caption: str = "",
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_booster(Path(model_path))
    n_feat = _n_features(model)
    names = feature_names_for_dim(n_feat)
    (out_dir / "feature_names.json").write_text(
        json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    if X is None:
        raise ValueError("需要特征矩阵 X")
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != n_feat:
        raise ValueError(f"X 列数 {X.shape[1]} != 模型特征数 {n_feat}")
    n_use = min(int(n_explain), len(X))
    rng = np.random.default_rng(0)
    if n_use < len(X):
        take = np.sort(rng.choice(len(X), size=n_use, replace=False))
        Xs = X[take]
        row_in_x = int(row)
        if row_in_x not in set(take.tolist()):
            # 保证 waterfall 样本在解释集中
            Xs[0] = X[row_in_x]
            local_row = 0
        else:
            local_row = int(np.where(take == row_in_x)[0][0])
    else:
        Xs = X
        local_row = int(np.clip(row, 0, len(X) - 1))

    mean_abs, arr, expected = shap_values_abs_mean(model, Xs)
    written = []
    pref = (title_prefix or "").strip()
    head = f"{pref} " if pref else ""
    written += plot_top20(
        names, mean_abs, out_dir / "shap_global_top20",
        f"{head}TreeSHAP global top-20".strip(),
    )
    # 单样本：取该行预测类别
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    import xgboost as xgb
    drow = xgb.DMatrix(Xs[local_row: local_row + 1])
    prob = booster.predict(drow)
    prob = np.asarray(prob).reshape(-1)
    pred_c = int(np.argmax(prob)) if prob.size else 0
    shap_c = arr[local_row, :, pred_c if arr.shape[-1] > pred_c else 0]
    base = float(expected[pred_c] if pred_c < len(expected) else expected[0])
    written += plot_waterfall(
        names, shap_c, base, out_dir / "shap_waterfall",
        f"{head}Waterfall: {pair_caption or f'row={row}'}  class={pred_c}".strip(),
    )
    print(f"[shap] n_feat={n_feat} n_explain={len(Xs)} pred_class={pred_c} → {out_dir}")
    return written


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="TreeSHAP for XGBoost DDI models")
    p.add_argument("--model", default="", help=".json / .ubj XGBoost 模型")
    p.add_argument("--X", default="", help="特征矩阵 .npy (n, d)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--row", type=int, default=0, help="waterfall 样本行号（相对 X）")
    p.add_argument("--n_explain", type=int, default=128)
    p.add_argument("--pairs_csv", default="", help="fusion_pairs.csv，用于药名标注")
    p.add_argument("--drugs_csv", default="", help="drugbank_drugs.csv（若 pairs 无 name 列）")
    p.add_argument("--title_prefix", default="")
    p.add_argument("--self_test", action="store_true",
                   help="sklearn 随机数据训练小 XGB 并跑 SHAP")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.self_test:
        demo = out_dir / "self_test"
        demo.mkdir(parents=True, exist_ok=True)
        clf, X, y, model_path = train_toy_xgb(demo)
        (demo / "SELF_TEST_ONLY.txt").write_text(
            "Random sklearn data. Not real DDI pairs. Do not use in the manuscript.\n",
            encoding="utf-8",
        )
        run_shap(
            model_path, demo, X=X, row=int(args.row), n_explain=min(80, len(X)),
            title_prefix="[SELF-TEST]",
        )
        return
    if not args.model:
        raise SystemExit("需要 --model 或 --self_test")
    if not args.X:
        raise SystemExit("真实 SHAP 需要 --X 特征 .npy")
    X = np.load(args.X)
    cap = _pair_caption(args.pairs_csv or None, int(args.row), args.drugs_csv or None)
    run_shap(
        args.model, out_dir, X=X, row=args.row, n_explain=args.n_explain,
        title_prefix=args.title_prefix,
        pair_caption=cap,
    )


if __name__ == "__main__":
    main()
