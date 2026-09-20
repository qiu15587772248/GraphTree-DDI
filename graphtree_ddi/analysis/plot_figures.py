"""英文论文图：Times/serif、300 dpi、PDF+PNG、色盲友好调色板。"""

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
from sklearn.metrics import (  # noqa: E402
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize  # noqa: E402

_THIS = Path(__file__).resolve()
_ANALYSIS = _THIS.parent
_PROJ = _THIS.parents[2]
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

from graphtree_ddi.analysis import metrics as M  # noqa: E402

# Okabe–Ito + Tol extras (10 colorblind-safe)
OKABE_ITO = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#000000",
    "#F0E442",
    "#882255",
    "#44AA99",
]
CMAP_CM = "cividis"
DPI = 300


def apply_paper_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": DPI,
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    })


def save_fig(fig: plt.Figure, stem: Path) -> list[Path]:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def method_color_map(methods: list[str]) -> dict[str, str]:
    cmap = {}
    i = 0
    for m in M.METHOD_ORDER:
        cmap[m] = OKABE_ITO[i % len(OKABE_ITO)]
        i += 1
    for m in methods:
        if m not in cmap:
            cmap[m] = OKABE_ITO[i % len(OKABE_ITO)]
            i += 1
    return cmap


def _holdout(df: pd.DataFrame) -> pd.DataFrame:
    return df[M.pair_holdout_mask(df)].copy()


def _rep_row(df: pd.DataFrame, method: str) -> pd.Series | None:
    sub = _holdout(df)
    if sub.empty:
        sub = df[df["method"].astype(str).eq(method)]
    else:
        sub = sub[sub["method"].astype(str).eq(method)]
    if sub.empty:
        sub = df[df["method"].astype(str).eq(method)]
    if sub.empty:
        return None
    return M.pick_representative(df, method, test_set=str(sub.iloc[0].test_set))


def plot_confusion_one(y_true, y_pred, title: str, stem: Path) -> list[Path]:
    apply_paper_style()
    cm = confusion_matrix(y_true, y_pred, labels=M.CLASS_LABELS, normalize="true")
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cm, cmap=CMAP_CM, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(M.N_CLASSES))
    ax.set_yticks(range(M.N_CLASSES))
    ax.set_xticklabels(M.CLASS_NAMES_EN, rotation=35, ha="right")
    ax.set_yticklabels(M.CLASS_NAMES_EN)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(M.N_CLASSES):
        for j in range(M.N_CLASSES):
            val = cm[i, j]
            ax.text(
                j, i, f"{val:.2f}", ha="center", va="center",
                color="white" if val > 0.55 else "black", fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return save_fig(fig, stem)


def plot_confusion_panel(rows: list[tuple[str, np.ndarray, np.ndarray]], stem: Path) -> list[Path]:
    apply_paper_style()
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n + 1.1, 4.2), layout="constrained")
    if n == 1:
        axes = [axes]
    last_im = None
    for ax, (name, yt, yp) in zip(axes, rows):
        cm = confusion_matrix(yt, yp, labels=M.CLASS_LABELS, normalize="true")
        last_im = ax.imshow(cm, cmap=CMAP_CM, vmin=0.0, vmax=1.0)
        ax.set_xticks(range(M.N_CLASSES))
        ax.set_yticks(range(M.N_CLASSES))
        ax.set_xticklabels([s.split()[0] for s in M.CLASS_NAMES_EN])
        ax.set_yticklabels([s.split()[0] for s in M.CLASS_NAMES_EN])
        ax.set_xlabel("Predicted")
        ax.set_title(name)
        for i in range(M.N_CLASSES):
            for j in range(M.N_CLASSES):
                val = cm[i, j]
                ax.text(
                    j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val > 0.55 else "black", fontsize=7,
                )
    axes[0].set_ylabel("True")
    if last_im is not None:
        fig.colorbar(last_im, ax=list(axes), shrink=0.86, pad=0.02)
    fig.suptitle("Normalized confusion matrices")
    return save_fig(fig, stem)


def plot_per_class_f1(tbl: pd.DataFrame, stem: Path,
                      labels: dict[str, str] | None = None) -> list[Path]:
    apply_paper_style()
    methods = M.methods_in(tbl)
    x = np.arange(M.N_CLASSES, dtype=float)
    width = min(0.82 / max(len(methods), 1), 0.16)
    colors = method_color_map(methods)
    fig_w = max(8.2, 0.55 * len(methods) + 5.0)
    fig, ax = plt.subplots(figsize=(fig_w, 4.6))
    for i, m in enumerate(methods):
        g = tbl[tbl["method"] == m]
        means, sds = [], []
        for c in range(M.N_CLASSES):
            mu, sd = M.mean_sd(g[f"f1_c{c}"].tolist())
            means.append(mu)
            sds.append(0.0 if not np.isfinite(sd) else sd)
        offset = (i - (len(methods) - 1) / 2) * width
        ax.bar(
            x + offset, means, width=width, label=M.display_name(m, labels),
            color=colors[m],
            yerr=sds, capsize=2, error_kw=dict(lw=0.8, ecolor="#333333"),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(M.CLASS_NAMES_EN, rotation=20, ha="right")
    ax.set_ylabel("F1")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Per-class F1 by method")
    ax.legend(frameon=False, ncol=min(len(methods), 2), fontsize=8)
    fig.tight_layout()
    return save_fig(fig, stem)


def plot_run_boxplot(tbl: pd.DataFrame, metric: str, ylabel: str, stem: Path,
                     labels: dict[str, str] | None = None) -> list[Path]:
    apply_paper_style()
    methods = M.methods_in(tbl)
    colors = method_color_map(methods)
    data = [tbl.loc[tbl["method"] == m, metric].astype(float).to_numpy() for m in methods]
    tick = [M.display_name(m, labels) for m in methods]
    fig, ax = plt.subplots(figsize=(max(7.2, 0.95 * len(methods) + 2.2), 4.6))
    bp = ax.boxplot(
        data, tick_labels=tick, patch_artist=True, widths=0.55,
        medianprops=dict(color="black", lw=1.2),
        whiskerprops=dict(color="#333333"),
        capprops=dict(color="#333333"),
        flierprops=dict(marker="o", markersize=3, markerfacecolor="#333333"),
    )
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(colors[m])
        patch.set_alpha(0.75)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} across runs")
    ax.tick_params(axis="x", rotation=28, labelsize=8)
    fig.tight_layout()
    return save_fig(fig, stem)


def plot_coldstart(tbl: pd.DataFrame, stem: Path, metric: str = "macro_f1",
                   labels: dict[str, str] | None = None) -> list[Path]:
    apply_paper_style()
    sub = tbl[tbl["test_set"].astype(str).isin(["S1", "S2"])].copy()
    if sub.empty:
        return []
    methods = M.methods_in(sub)
    colors = method_color_map(["S1", "S2"])
    x = np.arange(len(methods), dtype=float)
    width = 0.36
    means = {ts: [] for ts in ("S1", "S2")}
    sds = {ts: [] for ts in ("S1", "S2")}
    for m in methods:
        for ts in ("S1", "S2"):
            g = sub[(sub["method"] == m) & (sub["test_set"] == ts)]
            mu, sd = M.mean_sd(g[metric].tolist() if metric in g.columns else [])
            means[ts].append(0.0 if not np.isfinite(mu) else mu)
            sds[ts].append(0.0 if not np.isfinite(sd) else sd)
    fig, ax = plt.subplots(figsize=(max(6.0, 1.1 * len(methods) + 2), 4.2))
    ax.bar(
        x - width / 2, means["S1"], width, yerr=sds["S1"], label="S1 (one unseen drug)",
        color=colors["S1"], capsize=3, error_kw=dict(lw=0.8),
    )
    ax.bar(
        x + width / 2, means["S2"], width, yerr=sds["S2"], label="S2 (both unseen)",
        color=colors["S2"], capsize=3, error_kw=dict(lw=0.8),
    )
    ax.set_xticks(x)
    ax.set_xticklabels([M.display_name(m, labels) for m in methods], rotation=28, ha="right")
    ylab = "Macro-F1" if metric == "macro_f1" else metric
    ax.set_ylabel(ylab)
    ax.set_title("Cold-start S1 / S2")
    ax.legend(frameon=False)
    fig.tight_layout()
    return save_fig(fig, stem)


def _macro_roc(y_true: np.ndarray, y_prob: np.ndarray):
    y_bin = label_binarize(np.asarray(y_true, dtype=int), classes=M.CLASS_LABELS)
    if y_bin.ndim == 1:
        y_bin = np.eye(M.N_CLASSES, dtype=y_bin.dtype)[np.asarray(y_true, dtype=int)]
    grid = np.linspace(0.0, 1.0, 101)
    tprs, aucs = [], []
    for i in range(M.N_CLASSES):
        if y_bin[:, i].sum() <= 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        aucs.append(auc(fpr, tpr))
        tprs.append(np.interp(grid, fpr, tpr))
        tprs[-1][0] = 0.0
    if not tprs:
        return grid, np.full_like(grid, np.nan), float("nan")
    mean_tpr = np.mean(np.stack(tprs, axis=0), axis=0)
    mean_tpr[-1] = 1.0
    return grid, mean_tpr, float(np.mean(aucs))


def _macro_pr(y_true: np.ndarray, y_prob: np.ndarray):
    y_bin = label_binarize(np.asarray(y_true, dtype=int), classes=M.CLASS_LABELS)
    if y_bin.ndim == 1:
        y_bin = np.eye(M.N_CLASSES, dtype=y_bin.dtype)[np.asarray(y_true, dtype=int)]
    grid = np.linspace(0.0, 1.0, 101)
    precs, aps = [], []
    for i in range(M.N_CLASSES):
        if y_bin[:, i].sum() <= 0:
            continue
        p, r, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        precs.append(np.interp(grid, r[::-1], p[::-1]))
        aps.append(float(average_precision_score(y_bin[:, i], y_prob[:, i])))
    if not precs:
        return grid, np.full_like(grid, np.nan), float("nan")
    return grid, np.mean(np.stack(precs, axis=0), axis=0), float(np.mean(aps))


def plot_macro_roc(df: pd.DataFrame, stem: Path,
                   labels: dict[str, str] | None = None) -> list[Path]:
    apply_paper_style()
    methods = M.methods_in(df)
    colors = method_color_map(methods)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="#888888", label="Chance")
    for m in methods:
        rec = _rep_row(df, m)
        if rec is None:
            continue
        fpr, tpr, a = _macro_roc(rec.y_true, rec.y_prob)
        ax.plot(fpr, tpr, color=colors[m], lw=1.8,
                label=f"{M.display_name(m, labels)} (AUC={a:.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Macro-average ROC (one-vs-rest)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower right", fontsize=7.5)
    fig.tight_layout()
    return save_fig(fig, stem)


def plot_macro_pr(df: pd.DataFrame, stem: Path,
                  labels: dict[str, str] | None = None) -> list[Path]:
    apply_paper_style()
    methods = M.methods_in(df)
    colors = method_color_map(methods)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for m in methods:
        rec = _rep_row(df, m)
        if rec is None:
            continue
        rec_g, prec, ap = _macro_pr(rec.y_true, rec.y_prob)
        ax.plot(rec_g, prec, color=colors[m], lw=1.8,
                label=f"{M.display_name(m, labels)} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Macro-average precision–recall")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="best", fontsize=7.5)
    fig.tight_layout()
    return save_fig(fig, stem)


def plot_label_distribution(y_path: Path, stem: Path) -> list[Path]:
    apply_paper_style()
    y = np.load(y_path)
    y = np.asarray(y).astype(np.int64).ravel()
    counts = np.bincount(y, minlength=M.N_CLASSES)[:M.N_CLASSES]
    colors = OKABE_ITO[:M.N_CLASSES]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bars = ax.bar(range(M.N_CLASSES), counts, color=colors, edgecolor="black", lw=0.4)
    ax.set_xticks(range(M.N_CLASSES))
    ax.set_xticklabels(M.CLASS_NAMES_EN, rotation=20, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Label distribution (processed pairs)")
    total = int(counts.sum()) or 1
    for b, c in zip(bars, counts):
        ax.text(
            b.get_x() + b.get_width() / 2, b.get_height(),
            f"{c:,}\n({100.0 * c / total:.1f}%)",
            ha="center", va="bottom", fontsize=8,
        )
    ax.set_ylim(0, max(counts) * 1.22)
    fig.tight_layout()
    return save_fig(fig, stem)


def plot_all(
    pred_dir,
    out_dir: str | Path,
    *,
    data_dir: str | Path | None = None,
    panel_methods: list[str] | None = None,
    y_path: str | Path | None = None,
    labels: dict[str, str] | None = None,
) -> list[Path]:
    apply_paper_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = labels if labels is not None else M.DEFAULT_METHOD_LABELS
    df = M.load_predictions(pred_dir)
    written: list[Path] = []
    if df.empty:
        print("[plots] no npz predictions; skip prediction figures")
    else:
        tbl = M.metrics_table_from_runs(df)
        methods = M.methods_in(df)
        for m in methods:
            rec = _rep_row(df, m)
            if rec is None:
                continue
            stem = out_dir / f"cm_{_safe(m)}"
            written += plot_confusion_one(
                rec.y_true, rec.y_pred,
                f"{M.display_name(m, labels)} (normalized)", stem,
            )
        panel = panel_methods or M.default_panel_methods(df, 3)
        panel = [m for m in panel if m in methods][:3]
        triples = []
        for m in panel:
            rec = _rep_row(df, m)
            if rec is not None:
                triples.append((M.display_name(m, labels), rec.y_true, rec.y_pred))
        if triples:
            written += plot_confusion_panel(triples, out_dir / "cm_panel3")

        pair_tbl = tbl[(tbl["split"] == "pair") & tbl["test_set"].isin(["test", "holdout"])]
        f1_src = pair_tbl if not pair_tbl.empty else tbl
        written += plot_per_class_f1(f1_src, out_dir / "per_class_f1", labels=labels)
        box_src = pair_tbl if not pair_tbl.empty else tbl
        if not box_src.empty:
            written += plot_run_boxplot(
                box_src, "macro_f1", "Macro-F1", out_dir / "box_macro_f1", labels=labels,
            )
            written += plot_run_boxplot(
                box_src, "qwk", "Quadratic weighted κ", out_dir / "box_qwk", labels=labels,
            )
        cold = plot_coldstart(tbl, out_dir / "coldstart_s1s2", labels=labels)
        written += cold
        written += plot_macro_roc(df, out_dir / "roc_macro", labels=labels)
        written += plot_macro_pr(df, out_dir / "pr_macro", labels=labels)

    yp = Path(y_path) if y_path else None
    if yp is None:
        dd = Path(data_dir) if data_dir else (_PROJ / "data" / "processed" / "v2")
        yp = dd / "y.npy"
    if yp.exists():
        written += plot_label_distribution(yp, out_dir / "label_distribution")
    else:
        print(f"[plots] y.npy not found: {yp}")
    print(f"[plots] {len(written)} files → {out_dir}")
    return written


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Paper figures (PDF+PNG)")
    M.add_pred_dir_arg(p)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--data_dir", default="", help="含 y.npy 的 processed 目录")
    p.add_argument("--y_path", default="")
    p.add_argument("--panel_methods", nargs="*", default=None)
    M.add_method_names_arg(p)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    plot_all(
        args.pred_dir, args.out_dir,
        data_dir=args.data_dir or None,
        panel_methods=args.panel_methods,
        y_path=args.y_path or None,
        labels=M.parse_method_names(args.method_names),
    )


if __name__ == "__main__":
    main()
