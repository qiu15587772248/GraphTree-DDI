"""
可视化工具模块（viz_utils.py）
══════════════════════════════════════════════════════════════════════════════

提供:
  1. CJK 字体检测与注册（支持 Windows/Linux，无中文字体时 graceful fallback）
  2. 双语文本字典 TX
  3. _both_langs() 装饰器，对 'cn'/'en' 各调用一次绘图
  4. 通用可视化函数（CV 箱线图、方法对比条形图、混淆矩阵、ROC、PR 等）

所有函数均按 baseline_xgboost.py 的双语模式对齐。
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


import os
import warnings
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════
# 一、字体检测与注册
# ═══════════════════════════════════════════════════════════════
_CJK_FONT_CANDIDATES = [
    # Linux 优先
    "Noto Sans CJK SC", "Noto Sans CJK TC", "Noto Sans CJK JP",
    "Source Han Sans SC", "Source Han Sans CN",
    "WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "WenQuanYi Zen Hei Mono",
    "Droid Sans Fallback", "AR PL UKai CN", "AR PL UMing CN",
    # Windows / 通用
    "Microsoft YaHei", "微软雅黑",
    "SimHei", "黑体", "SimSun", "宋体",
    "STHeiti", "STXihei",
    # macOS
    "PingFang SC", "PingFang TC", "Heiti SC", "Hiragino Sans GB",
]


def register_cjk_fonts() -> tuple[bool, str | None]:
    """扫描系统字体，找到 CJK 字体则注册并返回 (True, 字体名)；
    否则返回 (False, None) 并给出建议。

    扫描顺序：
      1. matplotlib 已缓存的 fontManager
      2. 文件名包含 CJK 关键词的系统字体文件
      3. Linux 常见 CJK 字体文件路径
    """
    # 1) 检查 matplotlib 已注册字体
    available = {f.name for f in fm.fontManager.ttflist}
    for name in _CJK_FONT_CANDIDATES:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True, name

    # 2) 扫描系统字体文件中名字含 CJK 关键词的
    keywords = ("yahei", "simhei", "msyh", "heiti",
                "wqy", "wenquanyi", "notosanscjk", "notosans-cjk",
                "sourcehan", "source-han",
                "droidsansfallback", "droid-sans-fallback",
                "arphic", "ukai", "uming",
                "pingfang", "hiragino", "cjk")
    system_fonts = fm.findSystemFonts()
    for p in system_fonts:
        low = Path(p).name.lower()
        if any(k in low for k in keywords):
            try:
                fe = fm.FontEntry(fname=p, name="CustomCJK")
                fm.fontManager.ttflist.insert(0, fe)
                matplotlib.rcParams["font.sans-serif"] = ["CustomCJK", "DejaVu Sans"]
                matplotlib.rcParams["axes.unicode_minus"] = False
                return True, f"CustomCJK (from {Path(p).name})"
            except Exception:
                continue

    # 3) Linux 常见 CJK 字体路径兜底
    linux_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ]
    for p in linux_paths:
        if os.path.exists(p):
            try:
                fe = fm.FontEntry(fname=p, name="CustomCJK_Fallback")
                fm.fontManager.ttflist.insert(0, fe)
                matplotlib.rcParams["font.sans-serif"] = ["CustomCJK_Fallback", "DejaVu Sans"]
                matplotlib.rcParams["axes.unicode_minus"] = False
                return True, f"CustomCJK_Fallback (from {Path(p).name})"
            except Exception:
                continue

    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return False, None


def print_font_install_hint() -> None:
    """未找到 CJK 字体时打印安装建议（只在 Linux 上提示）。"""
    print("=" * 72)
    print("  [WARNING] 未检测到中文字体，中文图表会出现豆腐块（□）。")
    print("  在 Ubuntu/Debian 服务器上，请执行以下命令安装字体：")
    print("")
    print("    sudo apt-get update")
    print("    sudo apt-get install -y fonts-wqy-microhei fonts-wqy-zenhei fonts-noto-cjk")
    print("    rm -rf ~/.cache/matplotlib")
    print("")
    print("  安装后重新运行本脚本即可。")
    print("=" * 72)


# 模块加载时立即扫描一次
_HAS_CJK, _CJK_FONT_NAME = register_cjk_fonts()
if _HAS_CJK:
    print(f"[viz_utils] 已加载 CJK 字体: {_CJK_FONT_NAME}")
else:
    print_font_install_hint()


def has_cjk_font() -> bool:
    return _HAS_CJK


def cjk_font_name() -> str | None:
    return _CJK_FONT_NAME


# ═══════════════════════════════════════════════════════════════
# 二、双语文本字典
# ═══════════════════════════════════════════════════════════════
TX: dict[str, tuple[str, str]] = {
    # 通用
    "methods":     ("方法", "Method"),
    "accuracy":    ("准确率", "Accuracy"),
    "macro_f1":    ("Macro-F1", "Macro-F1"),
    "weighted_f1": ("Weighted-F1", "Weighted-F1"),
    "macro_auroc": ("宏平均 AUROC", "Macro AUROC"),
    "macro_ap":    ("宏平均 AP", "Macro AP"),
    "class":       ("类别", "Class"),
    "fold":        ("折编号", "Fold"),
    "seed":        ("种子", "Seed"),
    "predicted":   ("预测类别", "Predicted Label"),
    "actual":      ("真实类别", "True Label"),
    "metric_val":  ("指标值", "Metric Value"),
    "sample_cnt":  ("样本数量", "Sample Count"),
    "proportion":  ("占比", "Proportion"),
    "score":       ("分数", "Score"),
    "f1":          ("F1 分数", "F1 Score"),
    "mean":        ("均值", "Mean"),
    "std":         ("标准差", "Std"),
    # 损失曲线
    "train_loss":  ("训练损失", "Training Loss"),
    "val_loss":    ("验证损失", "Validation Loss"),
    "best_iter":   ("最优轮次", "Best Iteration"),
    "iter_label":  ("训练轮次", "Training Iteration"),
    "loss_label":  ("损失", "Loss"),
    # ROC / PR
    "random_guess":("随机猜测 (AUC=0.50)", "Random Guess (AUC=0.50)"),
    "fpr":         ("假正例率 (FPR)", "False Positive Rate (FPR)"),
    "tpr":         ("真正例率 (TPR / 召回率)", "True Positive Rate (Recall)"),
    "recall":      ("召回率 (Recall)", "Recall"),
    "precision":   ("精确率 (Precision)", "Precision"),
    # 混淆矩阵
    "cm_abs":      ("混淆矩阵 (样本数)", "Confusion Matrix (Counts)"),
    "cm_norm":     ("混淆矩阵 (归一化比例)", "Confusion Matrix (Normalized)"),
    # CV 相关
    "cv_title":    ("5 折交叉验证结果 (跨 3 seed)", "5-Fold CV Results (across 3 seeds)"),
    "per_class_cv":("各类别 F1 (5 折 × 3 seed)", "Per-Class F1 (5-fold × 3 seeds)"),
    "cmp_methods": ("方法对比 (均值 ± 标准差)", "Method Comparison (mean ± std)"),
    "dist_title":  ("数据集类别分布 (训练/验证/测试)",
                    "Dataset Class Distribution (Train/Val/Test)"),
    "dist_xlabel": ("数据集", "Split"),
    "dist_ylabel": ("样本数量", "Sample Count"),
    # 实验名称
    "exp1":        ("Exp-1 XGBoost v2", "Exp-1 XGBoost v2"),
    "exp2":        ("Exp-2 R-GCN(DDI-only)", "Exp-2 R-GCN(DDI-only)"),
    "exp3":        ("Exp-3 R-GCN(Hybrid)", "Exp-3 R-GCN(Hybrid)"),
    "exp4":        ("Exp-4 R-GAT(Hybrid)", "Exp-4 R-GAT(Hybrid)"),
    "exp5":        ("Exp-5 Fusion (R-GCN+XGB)", "Exp-5 Fusion (R-GCN+XGB)"),
    # 风险等级（5 级）
    "risk_cn": (
        ["安全(无交互)", "一般(PK/PD)", "中度(具体轻症)",
         "严重(需干预)", "危重(生命威胁)"],
        None,  # placeholder
    ),
    "risk_en": (
        None,
        ["Safe (No Interaction)",
         "General (PK/PD)",
         "Moderate (Specific Symptom)",
         "Serious (Requires Intervention)",
         "Critical (Life-Threatening)"],
    ),
}

RISK_LABELS_CN = TX["risk_cn"][0]
RISK_LABELS_EN = TX["risk_en"][1]
RISK_COLORS = ["#4CAF50", "#8BC34A", "#FFC107", "#FF5722", "#F44336"]

# 方法配色（5 个方法）
METHOD_COLORS = {
    "Exp-1": "#94a3b8",  # 灰（XGBoost 基线）
    "Exp-2": "#22c55e",  # 绿（R-GCN DDI）
    "Exp-3": "#3b82f6",  # 蓝（R-GCN Hybrid 主候选）
    "Exp-4": "#f59e0b",  # 橙（R-GAT 对照）
    "Exp-5": "#ef4444",  # 红（融合，最终模型）
}


def t(key: str, lang: str) -> str:
    """取双语文本。"""
    v = TX.get(key)
    if v is None:
        return key
    # risk_cn / risk_en 是 list 型，特殊处理
    if isinstance(v[0], list) or isinstance(v[1], list):
        return key  # 不该直接用 t() 取风险标签
    return v[0] if lang == "cn" else v[1]


def labels_for_lang(lang: str) -> list[str]:
    return RISK_LABELS_CN if lang == "cn" else RISK_LABELS_EN


def _reapply_font(lang: str) -> None:
    """按语言重新应用字体 rcParams（防止中途被改）。"""
    if lang == "cn":
        if _HAS_CJK:
            matplotlib.rcParams["font.sans-serif"] = [_CJK_FONT_NAME, "DejaVu Sans"]
        else:
            matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    else:
        matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def both_langs(plot_fn, *args, **kwargs):
    """对 cn 和 en 各调用一次。绘图函数需支持 lang= 关键字参数。"""
    for lang in ("cn", "en"):
        _reapply_font(lang)
        plot_fn(*args, lang=lang, **kwargs)
    _reapply_font("cn")


# ═══════════════════════════════════════════════════════════════
# 三、通用可视化（传入各实验聚合数据后直接画）
# ═══════════════════════════════════════════════════════════════
def _fmt_pm(vals: list[float]) -> str:
    """格式化为 mean ± std。"""
    if not vals:
        return "—"
    m = float(np.mean(vals))
    s = float(np.std(vals))
    return f"{m:.4f} ± {s:.4f}"


def plot_method_comparison_bars(
    agg_metrics: dict,
    method_order: list[str],
    metrics_keys: list[str],
    out_path_cn: Path,
    out_path_en: Path,
    lang: str = "cn",
    y_min: float | None = None,
    y_max: float = 1.06,
) -> None:
    """多方法 × 多指标 均值对比条形图（带误差栏）。
    
    agg_metrics[method_name][metric_key] = list of values (across fold × seed)
    
    Args:
        y_min: 自适应数据下限时取 mean-std 最小值 - 0.03，且对齐到 0.05 网格；
               传入值时覆盖自适应。
    """
    fig, ax = plt.subplots(figsize=(max(10, len(metrics_keys) * 2), 6))
    n_methods = len(method_order)
    n_metrics = len(metrics_keys)
    width = 0.8 / max(n_methods, 1)
    x = np.arange(n_metrics)

    all_lows = []
    for i, method in enumerate(method_order):
        means = []
        stds = []
        for mk in metrics_keys:
            vals = agg_metrics.get(method, {}).get(mk, [])
            if vals:
                m = float(np.mean(vals))
                s = float(np.std(vals))
                means.append(m)
                stds.append(s)
                all_lows.append(m - s)
            else:
                means.append(0.0)
                stds.append(0.0)
        offset = (i - n_methods / 2 + 0.5) * width
        color = METHOD_COLORS.get(method, "#6366f1")
        bars = ax.bar(x + offset, means, width * 0.9, yerr=stds,
                      label=method, color=color, alpha=0.88,
                      capsize=3, error_kw=dict(lw=1, ecolor="#1f2937"))
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=45)

    # 自适应 ylim：取 mean-std 最小值 - 0.03，对齐 0.05 网格，不低于 0
    if y_min is None:
        if all_lows:
            lo = min(all_lows) - 0.03
            lo = max(0.0, (np.floor(lo * 20) / 20))  # 向下取 0.05
        else:
            lo = 0.5
    else:
        lo = y_min

    metric_labels = [t(k, lang) for k in metrics_keys]
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel(t("score", lang), fontsize=12)
    ax.set_ylim(lo, y_max)
    ax.set_title(t("cmp_methods", lang), fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", ncol=min(n_methods, 5))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = out_path_cn if lang == "cn" else out_path_en
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_class_f1_comparison(
    per_class_f1_by_method: dict,
    method_order: list[str],
    out_path_cn: Path,
    out_path_en: Path,
    lang: str = "cn",
    y_min: float | None = None,
    y_max: float = 1.05,
) -> None:
    """每个方法的每类 F1 均值对比（5 类 × N 方法，带误差栏）。
    
    per_class_f1_by_method[method_name] = list of [c0,c1,c2,c3,c4] 列表（跨 fold × seed）
    """
    n_cls = 5
    labels = labels_for_lang(lang)[:n_cls]
    fig, ax = plt.subplots(figsize=(max(12, n_cls * 2), 6))
    n_methods = len(method_order)
    width = 0.8 / max(n_methods, 1)
    x = np.arange(n_cls)

    all_lows = []
    for i, method in enumerate(method_order):
        runs = per_class_f1_by_method.get(method, [])
        if not runs:
            means = [0.0] * n_cls
            stds = [0.0] * n_cls
        else:
            arr = np.asarray(runs, dtype=float)  # (n_runs, 5)
            means = arr.mean(axis=0).tolist()
            stds = arr.std(axis=0).tolist()
        for m, s in zip(means, stds):
            if m > 0:
                all_lows.append(m - s)
        offset = (i - n_methods / 2 + 0.5) * width
        color = METHOD_COLORS.get(method, "#6366f1")
        bars = ax.bar(x + offset, means, width * 0.9, yerr=stds,
                      label=method, color=color, alpha=0.88,
                      capsize=3, error_kw=dict(lw=1, ecolor="#1f2937"))
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=7, rotation=0)

    if y_min is None:
        if all_lows:
            lo = min(all_lows) - 0.03
            lo = max(0.0, (np.floor(lo * 20) / 20))
        else:
            lo = 0.4
    else:
        lo = y_min

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15)
    ax.set_ylabel(t("f1", lang), fontsize=12)
    ax.set_ylim(lo, y_max)
    ax.set_title(t("per_class_cv", lang), fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left", ncol=min(n_methods, 5))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = out_path_cn if lang == "cn" else out_path_en
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cv_box(
    agg_metrics: dict,
    method_order: list[str],
    metric_key: str,
    out_path_cn: Path,
    out_path_en: Path,
    lang: str = "cn",
    include_methods: list[str] | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    title_suffix: str | tuple[str, str] = "",
) -> None:
    """某指标在 N 方法 × (fold × seed) 下的箱线图。
    
    Args:
        include_methods: 指定只画哪些方法（用于生成 stable-only 版本）。
                         None = 画全部 method_order 中的方法。
        y_min / y_max:   覆盖自适应 ylim（默认 matplotlib 自动计算）。
        title_suffix:    str 或 (cn_suffix, en_suffix)。tuple 时按 lang 选择。
    """
    if isinstance(title_suffix, tuple):
        suffix = title_suffix[0] if lang == "cn" else title_suffix[1]
    else:
        suffix = title_suffix
    if include_methods is not None:
        methods = [m for m in method_order if m in include_methods]
    else:
        methods = method_order

    # annotate 模式下图更高给文本留空间
    h = 6.5 if False else 5  # placeholder, replaced below
    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.5), 5))
    data = [agg_metrics.get(m, {}).get(metric_key, []) for m in methods]
    colors = [METHOD_COLORS.get(m, "#6366f1") for m in methods]
    bp = ax.boxplot(data, labels=methods, patch_artist=True,
                    medianprops=dict(color="#1f2937", linewidth=1.5),
                    widths=0.6)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    # 叠加散点
    for i, d in enumerate(data):
        if d:
            jitter = np.random.uniform(-0.1, 0.1, size=len(d))
            ax.scatter(np.ones(len(d)) * (i + 1) + jitter, d,
                       s=14, c="#1f2937", alpha=0.6, zorder=3)

    title = f"{t(metric_key, lang)} — {t('cv_title', lang)}{suffix}"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(t(metric_key, lang), fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    if y_min is not None or y_max is not None:
        cur = ax.get_ylim()
        ax.set_ylim(y_min if y_min is not None else cur[0],
                    y_max if y_max is not None else cur[1])
    plt.tight_layout()
    out = out_path_cn if lang == "cn" else out_path_en
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cv_box_annotated(
    agg_metrics: dict,
    method_order: list[str],
    metric_key: str,
    out_path_cn: Path,
    out_path_en: Path,
    lang: str = "cn",
    include_methods: list[str] | None = None,
    title_suffix: str | tuple[str, str] = "",
) -> None:
    """与 plot_cv_box 类似，但在每个箱子顶部标注 "mean ± std" 文本，
    即使箱子扁（std 很小）读者也能一眼看到准确值。
    """
    if isinstance(title_suffix, tuple):
        suffix = title_suffix[0] if lang == "cn" else title_suffix[1]
    else:
        suffix = title_suffix
    if include_methods is not None:
        methods = [m for m in method_order if m in include_methods]
    else:
        methods = method_order

    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.6), 6.5))
    data = [agg_metrics.get(m, {}).get(metric_key, []) for m in methods]
    colors = [METHOD_COLORS.get(m, "#6366f1") for m in methods]
    bp = ax.boxplot(data, labels=methods, patch_artist=True,
                    medianprops=dict(color="#1f2937", linewidth=1.8),
                    widths=0.55)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.72)
    for i, d in enumerate(data):
        if d:
            jitter = np.random.uniform(-0.08, 0.08, size=len(d))
            ax.scatter(np.ones(len(d)) * (i + 1) + jitter, d,
                       s=20, c="#1f2937", alpha=0.7, zorder=3)

    title = f"{t(metric_key, lang)} — {t('cv_title', lang)}{suffix}"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(t(metric_key, lang), fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    # 顶部留空间给标注
    y_lo, y_hi = ax.get_ylim()
    span = y_hi - y_lo
    ax.set_ylim(y_lo, y_hi + span * 0.14)
    y_hi_new = ax.get_ylim()[1]

    for i, d in enumerate(data):
        if not d:
            continue
        m = float(np.mean(d))
        s = float(np.std(d))
        txt = f"μ={m:.4f}\nσ={s:.4f}"
        ax.text(i + 1, y_hi_new - span * 0.02, txt,
                ha="center", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.28",
                          facecolor="white", edgecolor="#9ca3af",
                          alpha=0.92, linewidth=0.8))

    plt.tight_layout()
    out = out_path_cn if lang == "cn" else out_path_en
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cv_box_faceted(
    agg_metrics: dict,
    method_order: list[str],
    metric_key: str,
    out_path_cn: Path,
    out_path_en: Path,
    lang: str = "cn",
    include_methods: list[str] | None = None,
    title_suffix: str | tuple[str, str] = "",
) -> None:
    """面板式箱线图：每个方法一个独立子图，y 轴独立自适应到该方法的数据范围。

    适用场景：方法之间 std 差异巨大（如 Exp-1/5 std=0.0007 vs Exp-2/3 std=0.004），
    让每个方法的箱子在自己的坐标系里充分展开，不互相压制。
    每个子图标题显示方法名 + μ ± σ。
    """
    if isinstance(title_suffix, tuple):
        suffix = title_suffix[0] if lang == "cn" else title_suffix[1]
    else:
        suffix = title_suffix
    if include_methods is not None:
        methods = [m for m in method_order if m in include_methods]
    else:
        methods = method_order

    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(max(10, n * 2.3), 5.8),
                             sharey=False)
    if n == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        d = agg_metrics.get(method, {}).get(metric_key, [])
        c = METHOD_COLORS.get(method, "#6366f1")
        if not d:
            ax.text(0.5, 0.5, "—", transform=ax.transAxes,
                    ha="center", va="center", fontsize=20)
            ax.set_xticks([])
            ax.set_title(method, fontsize=11, fontweight="bold")
            continue
        bp = ax.boxplot([d], labels=[method], patch_artist=True,
                         medianprops=dict(color="#1f2937", linewidth=1.8),
                         widths=0.55)
        for patch in bp["boxes"]:
            patch.set_facecolor(c)
            patch.set_alpha(0.72)
        jitter = np.random.uniform(-0.08, 0.08, size=len(d))
        ax.scatter(np.ones(len(d)) + jitter, d,
                   s=26, c="#1f2937", alpha=0.75, zorder=3)
        mean = float(np.mean(d))
        std = float(np.std(d))
        # 自适应 ylim：扩到 ±4σ 或 ±0.003 保底
        pad = max(std * 4, 0.003)
        ax.set_ylim(mean - pad, mean + pad * 1.3)
        ax.set_title(f"{method}\nμ={mean:.4f}  σ={std:.4f}",
                     fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="y", labelsize=8)

    global_title = f"{t(metric_key, lang)} — {t('cv_title', lang)}{suffix}"
    fig.suptitle(global_title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = out_path_cn if lang == "cn" else out_path_en
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_pair(
    y_true: np.ndarray, y_pred: np.ndarray,
    title_key: str,
    out_path_cn: Path, out_path_en: Path,
    lang: str = "cn",
) -> None:
    """混淆矩阵双图（绝对 + 归一化）。"""
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    n_cls = len(np.unique(y_true))
    labels = labels_for_lang(lang)[:n_cls]
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".1%"],
        [t("cm_abs", lang), t("cm_norm", lang)],
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    linewidths=0.5, annot_kws={"size": 10})
        ax.set_xlabel(t("predicted", lang), fontsize=11)
        ax.set_ylabel(t("actual", lang), fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=15)
        ax.tick_params(axis="y", rotation=0)
    if title_key:
        plt.suptitle(title_key, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = out_path_cn if lang == "cn" else out_path_en
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_multi(
    y_true: np.ndarray, y_prob: np.ndarray,
    title_key: str,
    out_path_cn: Path, out_path_en: Path,
    lang: str = "cn",
) -> None:
    from sklearn.metrics import roc_curve, roc_auc_score
    from sklearn.preprocessing import label_binarize
    n_cls = y_prob.shape[1]
    labels = labels_for_lang(lang)[:n_cls]
    y_bin = label_binarize(y_true, classes=list(range(n_cls)))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label=t("random_guess", lang))
    for i, (lbl, color) in enumerate(zip(labels, RISK_COLORS)):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{lbl}  (AUC={auc:.3f})")
    ax.set_xlabel(t("fpr", lang), fontsize=12)
    ax.set_ylabel(t("tpr", lang), fontsize=12)
    ax.set_title(title_key, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = out_path_cn if lang == "cn" else out_path_en
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pr_multi(
    y_true: np.ndarray, y_prob: np.ndarray,
    title_key: str,
    out_path_cn: Path, out_path_en: Path,
    lang: str = "cn",
) -> None:
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize
    n_cls = y_prob.shape[1]
    labels = labels_for_lang(lang)[:n_cls]
    y_bin = label_binarize(y_true, classes=list(range(n_cls)))
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, (lbl, color) in enumerate(zip(labels, RISK_COLORS)):
        if y_bin[:, i].sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_bin[:, i], y_prob[:, i])
        ax.plot(rec, prec, color=color, lw=2.2, label=f"{lbl}  (AP={ap:.3f})")
    ax.set_xlabel(t("recall", lang), fontsize=12)
    ax.set_ylabel(t("precision", lang), fontsize=12)
    ax.set_title(title_key, fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = out_path_cn if lang == "cn" else out_path_en
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def plot_class_distribution(
    y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
    out_path_cn: Path, out_path_en: Path,
    lang: str = "cn",
) -> None:
    n_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))
    lbs = labels_for_lang(lang)[:n_classes]
    splits_cn = ["训练集", "验证集", "测试集"]
    splits_en = ["Train", "Val", "Test"]
    splits = splits_cn if lang == "cn" else splits_en
    data = {
        splits[0]: np.bincount(y_train, minlength=n_classes),
        splits[1]: np.bincount(y_val, minlength=n_classes),
        splits[2]: np.bincount(y_test, minlength=n_classes),
    }
    n_cls = len(lbs)
    width = min(0.14, 0.8 / n_cls)
    offsets = (np.arange(n_cls) - (n_cls - 1) / 2) * width
    fig, ax = plt.subplots(figsize=(max(10, n_cls * 2), 6))
    for i, (lbl, color) in enumerate(zip(lbs, RISK_COLORS)):
        x = np.arange(len(splits))
        vals = [data[s][i] for s in splits]
        bars = ax.bar(x + offsets[i], vals, width, label=lbl, color=color, alpha=0.85)
        ax.bar_label(bars, fmt="%d", padding=2, fontsize=7.5)
    ax.set_xlabel(t("dist_xlabel", lang), fontsize=12)
    ax.set_ylabel(t("dist_ylabel", lang), fontsize=12)
    ax.set_title(t("dist_title", lang), fontsize=13, fontweight="bold")
    ax.set_xticks(np.arange(len(splits)))
    ax.set_xticklabels(splits, fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = out_path_cn if lang == "cn" else out_path_en
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cv_line(
    folds_metrics: list[dict],
    out_path_cn: Path, out_path_en: Path,
    lang: str = "cn",
    y_min: float | None = None,
    y_max: float = 1.04,
) -> None:
    """单方法的 CV 折内对比（Macro-F1、Accuracy 各折 bars + 均值线）。

    folds_metrics: list of dict(fold, seed, macro_f1, accuracy, per_class_f1)
    y_min:  None=自适应（取数据最小值-0.05 并对齐 0.05 网格）
    """
    if not folds_metrics:
        return
    # 按 fold 聚合
    fold_ids = sorted({r["fold"] for r in folds_metrics})
    f1_by_fold = {f: [r["macro_f1"] for r in folds_metrics if r["fold"] == f] for f in fold_ids}
    acc_by_fold = {f: [r["accuracy"] for r in folds_metrics if r["fold"] == f] for f in fold_ids}

    f1_means = [float(np.mean(f1_by_fold[f])) for f in fold_ids]
    acc_means = [float(np.mean(acc_by_fold[f])) for f in fold_ids]
    f1_stds = [float(np.std(f1_by_fold[f])) for f in fold_ids]
    acc_stds = [float(np.std(acc_by_fold[f])) for f in fold_ids]

    fig, ax = plt.subplots(figsize=(max(8, len(fold_ids) * 1.5), 5))
    x = np.arange(len(fold_ids))
    w = 0.38
    ax.bar(x - w / 2, f1_means, w, yerr=f1_stds,
           label=t("macro_f1", lang), color="#6366f1", alpha=0.85,
           capsize=3, error_kw=dict(lw=1))
    ax.bar(x + w / 2, acc_means, w, yerr=acc_stds,
           label=t("accuracy", lang), color="#22c55e", alpha=0.85,
           capsize=3, error_kw=dict(lw=1))
    overall_f1 = float(np.mean([r["macro_f1"] for r in folds_metrics]))
    overall_f1_std = float(np.std([r["macro_f1"] for r in folds_metrics]))
    ax.axhline(overall_f1, color="#ef4444", lw=1.5, ls="--",
               label=f"{t('mean', lang)} {t('macro_f1', lang)} = "
                     f"{overall_f1:.4f} ± {overall_f1_std:.4f}")
    for xi, fm, am in zip(x, f1_means, acc_means):
        ax.text(xi - w / 2, fm + 0.004, f"{fm:.4f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + w / 2, am + 0.004, f"{am:.4f}", ha="center", va="bottom", fontsize=8)

    # 自适应 y 轴下限：考虑 mean-std 和 mean 两条下边界
    if y_min is None:
        lows = []
        for m, s in zip(f1_means, f1_stds):
            lows.append(m - s)
            lows.append(m)
        for m, s in zip(acc_means, acc_stds):
            lows.append(m - s)
            lows.append(m)
        if lows:
            lo = min(lows) - 0.04
            lo = max(0.0, (np.floor(lo * 20) / 20))
        else:
            lo = 0.5
    else:
        lo = y_min

    ax.set_xticks(x)
    if lang == "cn":
        ax.set_xticklabels([f"第{f}折" for f in fold_ids], fontsize=10)
    else:
        ax.set_xticklabels([f"Fold {f}" for f in fold_ids], fontsize=10)
    ax.set_ylim(lo, y_max)
    ax.set_ylabel(t("metric_val", lang), fontsize=12)
    ax.set_title(t("cv_title", lang), fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = out_path_cn if lang == "cn" else out_path_en
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
