"""外部等级 vs 模型预测：列联表、期望风险、Spearman、Major-vs-Minor AUROC。

CSV 列: drug1_id, drug2_id, ext_level, y_pred, p0..p4
ext_level ∈ {Major, Moderate, Minor, Unknown}

假数据仅用于调通，写入独立目录，不得进入论文 figures/。
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


from graphtree_ddi.analysis import metrics as M  # noqa: E402
from graphtree_ddi.analysis.plot_figures import OKABE_ITO, apply_paper_style, save_fig  # noqa: E402

EXT_ORDER = ["Minor", "Moderate", "Major", "Unknown"]
EXT_ORDINAL = {"Minor": 1, "Moderate": 2, "Major": 3}
PRED_NAMES = list(M.CLASS_NAMES_EN)


def expected_level(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    k = np.arange(p.shape[1], dtype=np.float64)
    return p @ k


def p_high_risk(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return p[:, 3] + p[:, 4] if p.shape[1] >= 5 else p[:, -1]


def _prob_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [f"p{i}" for i in range(M.N_CLASSES)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺列 {missing}")
    return df[cols].to_numpy(dtype=np.float64)


def make_synthetic_csv(path: Path, n: int = 200, seed: int = 0) -> Path:
    rng = np.random.default_rng(int(seed))
    ext = rng.choice(EXT_ORDER, size=n, p=[0.30, 0.28, 0.22, 0.20])
    # 让 Major 更可能被预测为 3/4，Minor 更可能为 0/1
    logits = rng.normal(size=(n, M.N_CLASSES))
    for i, lv in enumerate(ext):
        if lv == "Major":
            logits[i, 3:] += 2.2
        elif lv == "Moderate":
            logits[i, 1:3] += 1.4
        elif lv == "Minor":
            logits[i, :2] += 2.0
        else:
            logits[i] += rng.normal(scale=0.3, size=M.N_CLASSES)
    logits -= logits.max(axis=1, keepdims=True)
    prob = np.exp(logits)
    prob /= prob.sum(axis=1, keepdims=True)
    y_pred = prob.argmax(axis=1)
    df = pd.DataFrame({
        "drug1_id": [f"DB{10000 + i}" for i in range(n)],
        "drug2_id": [f"DB{20000 + i}" for i in range(n)],
        "ext_level": ext,
        "y_pred": y_pred,
    })
    for k in range(M.N_CLASSES):
        df[f"p{k}"] = prob[:, k]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def evaluate_external(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["ext_level"] = df["ext_level"].astype(str)
    df["y_pred"] = df["y_pred"].astype(int)
    P = _prob_matrix(df)
    e_lev = expected_level(P)
    df["e_level"] = e_lev
    df["p_ge3"] = p_high_risk(P)

    ct = pd.crosstab(
        df["y_pred"], df["ext_level"],
        rownames=["y_pred"], colnames=["ext_level"],
    )
    for col in EXT_ORDER:
        if col not in ct.columns:
            ct[col] = 0
    ct = ct.reindex(index=list(range(M.N_CLASSES)), columns=EXT_ORDER, fill_value=0)

    ranked = df[df["ext_level"].isin(EXT_ORDINAL)].copy()
    if len(ranked) >= 3:
        rho = float(ranked["ext_level"].map(EXT_ORDINAL).corr(ranked["e_level"], method="spearman"))
    else:
        rho = float("nan")

    mm = df[df["ext_level"].isin(["Major", "Minor"])].copy()
    if mm["ext_level"].nunique() == 2 and len(mm) >= 4:
        y_bin = (mm["ext_level"] == "Major").astype(int).to_numpy()
        try:
            auroc = float(roc_auc_score(y_bin, mm["p_ge3"].to_numpy()))
        except ValueError:
            auroc = float("nan")
    else:
        auroc = float("nan")

    major = df[df["ext_level"] == "Major"]
    if len(major):
        recall = float(np.mean(major["y_pred"].to_numpy() >= 3))
    else:
        recall = float("nan")

    return dict(
        df=df, crosstab=ct, spearman_rho=rho,
        major_vs_minor_auroc=auroc, major_high_risk_recall=recall,
    )


def plot_crosstab_heatmap(ct: pd.DataFrame, stem: Path, test_banner: bool) -> list[Path]:
    apply_paper_style()
    arr = ct.to_numpy(dtype=np.float64)
    col_sum = arr.sum(axis=0, keepdims=True)
    norm = np.divide(arr, col_sum, out=np.zeros_like(arr), where=col_sum > 0)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    im = ax.imshow(norm, cmap="cividis", vmin=0, vmax=1)
    ax.set_xticks(range(len(ct.columns)))
    ax.set_yticks(range(len(ct.index)))
    ax.set_xticklabels(list(ct.columns))
    ax.set_yticklabels(PRED_NAMES)
    ax.set_xlabel("External level")
    ax.set_ylabel("Predicted level")
    title = "Predicted vs external level (col-normalized)"
    if test_banner:
        title = "[TEST DATA] " + title
    ax.set_title(title)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            ax.text(
                j, i, f"{norm[i, j]:.2f}\n({int(arr[i, j])})",
                ha="center", va="center", fontsize=7,
                color="white" if norm[i, j] > 0.55 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return save_fig(fig, stem)


def plot_elevel_box(df: pd.DataFrame, stem: Path, test_banner: bool) -> list[Path]:
    apply_paper_style()
    data, labels = [], []
    for lv in EXT_ORDER:
        vals = df.loc[df["ext_level"] == lv, "e_level"].to_numpy(dtype=float)
        data.append(vals if len(vals) else np.array([np.nan]))
        labels.append(f"{lv}\n(n={len(vals)})")
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    bp = ax.boxplot(
        data, tick_labels=labels, patch_artist=True, widths=0.6,
        medianprops=dict(color="black", lw=1.2),
    )
    for patch, c in zip(bp["boxes"], OKABE_ITO):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_ylabel("E[level] = Σ k·p_k")
    title = "Expected predicted risk by external level"
    if test_banner:
        title = "[TEST DATA] " + title
    ax.set_title(title)
    fig.tight_layout()
    return save_fig(fig, stem)


def run_external(
    csv_path: str | Path,
    out_dir: str | Path,
    *,
    test_data: bool = False,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    need = {"drug1_id", "drug2_id", "ext_level", "y_pred"} | {f"p{i}" for i in range(M.N_CLASSES)}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"CSV 缺列: {sorted(miss)}")
    res = evaluate_external(df)
    ct: pd.DataFrame = res["crosstab"]
    ct.to_csv(out_dir / "crosstab_counts.csv", encoding="utf-8-sig")
    banner = "TEST DATA (synthetic, n=200). Do not use in the manuscript.\n\n" if test_data else ""
    summary = (
        f"{banner}"
        f"Spearman ρ (external ordinal vs E[level]): "
        f"{res['spearman_rho']:.4f}\n"
        f"Major vs Minor AUROC (score = P(level≥3)): "
        f"{res['major_vs_minor_auroc']:.4f}\n"
        f"Recall of predicted high-risk (≥3) among Major: "
        f"{res['major_high_risk_recall']:.4f}\n"
        f"n={len(df)}\n"
    )
    (out_dir / "external_summary.txt").write_text(summary, encoding="utf-8")
    (out_dir / "external_summary.md").write_text(
        "# External evaluation" + (" (TEST DATA)" if test_data else "") + "\n\n"
        + summary + "\n## Contingency (counts)\n\n" + ct.to_string() + "\n",
        encoding="utf-8",
    )
    plot_crosstab_heatmap(ct, out_dir / "ext_crosstab_heatmap", test_data)
    plot_elevel_box(res["df"], out_dir / "ext_elevel_box", test_data)
    print(f"[external] → {out_dir}")
    print(summary)
    return res


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="External DDI level evaluation")
    p.add_argument("--csv", default="", help="外部评估 CSV")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--demo", action="store_true",
                   help="生成 200 行假数据并评估（写入 out_dir，勿指向论文 figures）")
    p.add_argument("--demo_seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    if args.demo or not args.csv:
        csv_path = out_dir / "synthetic_external_200.csv"
        make_synthetic_csv(csv_path, n=200, seed=args.demo_seed)
        note = out_dir / "THIS_IS_TEST_DATA.txt"
        note.write_text(
            "Synthetic 200-row CSV for script smoke tests only.\n"
            "Do not copy these figures into sci_manuscript/figures.\n",
            encoding="utf-8",
        )
        run_external(csv_path, out_dir, test_data=True)
    else:
        run_external(args.csv, out_dir, test_data=False)


if __name__ == "__main__":
    main()
