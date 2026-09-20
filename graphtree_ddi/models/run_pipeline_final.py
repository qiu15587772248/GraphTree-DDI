"""
GNN DDI 论文级终极实验（run_pipeline_final.py）
══════════════════════════════════════════════════════════════════════════════

规格（顶会/SCI 1 区对标）:
  数据划分:  20% 测试集（封存） + 80% train+val → 在 80% 上做 5-fold CV
  随机性:    每折内跑 3 seed（42, 123, 2024），共 75 次训练
  指标:      mean ± std（across 5 fold × 3 seed = 15 runs/method）
  方法:      Exp-1/2/3/4/5 全套
  可视化:    双语图表（cn/en），仿 baseline_xgboost.py 的 TX 字典模式

子实验与最优配置（基于 pipeline1/2/3 已验证）——超参与结构不得改动:
  Exp-1  XGBoost v2         4120 维手工特征
                            n_estimators=200000, early_stopping=200
                            max_depth=6, lr=0.05, subsample=0.8, colsample_bytree=0.6
  Exp-2  R-GCN(DDI-only)    hidden_dim=2048, num_bases=10, focal_gamma=2.0
                            patience=150, max_epochs=1500
  Exp-3  R-GCN(Hybrid) ⭐   hidden_dim=2048, num_bases=8, focal_gamma=2.0 (主模型)
                            patience=150, max_epochs=1500
  Exp-4  R-GAT(Hybrid)      hidden_dim=768, focal_gamma=0.0, decoder=mlp
                            patience=150, max_epochs=1500
  Exp-5  Fusion             Exp-3 嵌入(1024/药物) + pair_ops(concat+diff+prod=4096)
                            + XGBoost v2 原特征(4120) = 8216 维 → 训练 XGBoost

产出:
  <out_dir>/
    ├─ final_state.json
    ├─ final_report.md
    ├─ final_raw_metrics.csv
    ├─ ckpts/  predictions/  cn/  en/
    └─ final_full/           （--final_full）

Usage: python -m graphtree_ddi.models.run_pipeline_final --help
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
import gc
import json
import pickle
import sys
import time
import traceback
from pathlib import Path


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, average_precision_score,
    classification_report, cohen_kappa_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402
from sklearn.preprocessing import label_binarize  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402

from graphtree_ddi.models import viz_utils  # noqa: E402
from graphtree_ddi.models.viz_utils import (  # noqa: E402
    METHOD_COLORS,
    RISK_LABELS_CN,
    RISK_LABELS_EN,
    both_langs,
    has_cjk_font,
    labels_for_lang,
    plot_class_distribution,
    plot_confusion_pair,
    plot_cv_box,
    plot_cv_box_annotated,
    plot_cv_box_faceted,
    plot_cv_line,
    plot_method_comparison_bars,
    plot_per_class_f1_comparison,
    plot_pr_multi,
    plot_roc_multi,
    t,
)

from graphtree_ddi.models.gnn import (  # noqa: E402
    NUM_RELATIONS,
    PROCESSED_DIR,
    RESULTS_DIR,
    RGAT_DDI,
    RGCN_DDI,
    _clear_gpu,
    build_knowledge_graph,
    load_ddi_pairs,
    set_reproducibility,
    train_model,
)

try:
    from graphtree_ddi.data.preprocess import BIO_FEATURE_NAMES, FEAT_DIM, N_BIO
except Exception:  # noqa: BLE001
    BIO_FEATURE_NAMES = [f"bio_{i}" for i in range(24)]
    N_BIO, FEAT_DIM = 24, 4120


# ─────────────────────────────────────────────────────────────
# 配置（超参与结构与历史实验一致，禁止改动）
# ─────────────────────────────────────────────────────────────
SEEDS = [42, 123, 2024]
N_FOLDS = 5
SPLIT_RANDOM_STATE = 42
DRUG_SPLIT_SEEDS_DEFAULT = [42, 123, 2024]
DRUG_DEFAULT_METHODS = ["Exp-1", "Exp-3", "Exp-5"]

METHODS_ORDER = ["Exp-1", "Exp-2", "Exp-3", "Exp-4", "Exp-5"]
METHOD_LABELS_CN = {
    "Exp-1": "Exp-1 XGBoost v2",
    "Exp-2": "Exp-2 R-GCN(DDI-only)",
    "Exp-3": "Exp-3 R-GCN(Hybrid)",
    "Exp-4": "Exp-4 R-GAT(Hybrid)",
    "Exp-5": "Exp-5 Fusion (R-GCN+XGB)",
}

# 来自 v1 全量 final_state.json 的按方法 elapsed_sec 均值，用于首个 run 的 ETA
HIST_MEAN_SEC = {
    "Exp-1": 2395.0,
    "Exp-2": 722.7,
    "Exp-3": 654.9,
    "Exp-4": 410.4,
    "Exp-5": 2096.0,
}

ORDINAL_METRICS = (
    "adjacent_acc", "high_risk_sens", "high_risk_spec",
    "severe_underestimation_rate", "qwk", "mae_ordinal",
)
CORE_METRICS = (
    "accuracy", "macro_f1", "weighted_f1", "macro_auroc", "macro_ap",
)

GNN_COMMON_CFG = dict(
    lr=5e-4, weight_decay=1e-4,
    batch_size=4096, warmup_epochs=10,
    label_smoothing=0.1,
    plateau_factor=0.5, plateau_patience=15,
    use_amp=True, drop_edge=0.0, grad_clip=1.0,
)

EXP_CONFIGS = {
    "Exp-2": dict(
        graph="ddi_only", model="rgcn",
        hidden_dim=2048, emb_dim=1024, num_bases=10,
        focal_gamma=2.0, decoder_type="mlp", norm_type="batch",
    ),
    "Exp-3": dict(
        graph="hybrid", model="rgcn",
        hidden_dim=2048, emb_dim=1024, num_bases=8,
        focal_gamma=2.0, decoder_type="mlp", norm_type="batch",
    ),
    "Exp-4": dict(
        graph="hybrid", model="rgat",
        hidden_dim=768, emb_dim=384,
        edge_emb_dim=64, heads=4,
        focal_gamma=0.0, decoder_type="mlp", norm_type="batch",
        self_loop_fill="zero",
    ),
}

PAIR_OPS_SPEC = {
    "name": "concat_diff_prod",
    "formula": "concat([h_a, h_b, h_a - h_b, h_a * h_b], axis=-1)",
    "ops": ["concat(h_a)", "concat(h_b)", "diff(h_a-h_b)", "prod(h_a*h_b)"],
    "emb_dim": 1024,
    "output_dim": 4096,
    "note": "与 PairMLP 输入及 Exp-5 融合尾部 4096 维一致；推理前须把 drug_a < drug_b。",
}


class RunPaths:
    """data_dir / out_dir 运行期路径。禁止在 import 时写死结果目录。"""

    def __init__(self, data_dir: Path, out_dir: Path):
        self.data_dir = Path(data_dir)
        self.out_dir = Path(out_dir)
        self.ckpt_dir = self.out_dir / "ckpts"
        self.pred_dir = self.out_dir / "predictions"
        self.out_cn = self.out_dir / "cn"
        self.out_en = self.out_dir / "en"
        self.state_path = self.out_dir / "final_state.json"
        self.report_path = self.out_dir / "final_report.md"
        self.raw_csv_path = self.out_dir / "final_raw_metrics.csv"
        self.final_full_dir = self.out_dir / "final_full"
        for d in (self.out_dir, self.ckpt_dir, self.pred_dir, self.out_cn, self.out_en):
            d.mkdir(parents=True, exist_ok=True)


PATHS: RunPaths | None = None


def P() -> RunPaths:
    if PATHS is None:
        raise RuntimeError("路径未初始化，请先 configure_paths()")
    return PATHS


def configure_paths(data_dir: Path | str, out_dir: Path | str) -> RunPaths:
    global PATHS
    PATHS = RunPaths(Path(data_dir), Path(out_dir))
    return PATHS


# ═══════════════════════════════════════════════════════════════
# 序数 / 临床指标 + pair_ops
# ═══════════════════════════════════════════════════════════════
def pair_ops_embed(h_a: np.ndarray, h_b: np.ndarray) -> np.ndarray:
    """concat + diff + prod。h_a/h_b: (N, D) 或 (D,)。"""
    h_a = np.asarray(h_a)
    h_b = np.asarray(h_b)
    squeeze = h_a.ndim == 1
    if squeeze:
        h_a = h_a[None, :]
        h_b = h_b[None, :]
    out = np.concatenate([h_a, h_b, h_a - h_b, h_a * h_b], axis=1).astype(np.float32)
    return out[0] if squeeze else out


def _evaluate_predictions(y_test: np.ndarray, y_pred: np.ndarray,
                          y_prob: np.ndarray, n_classes: int = 5) -> dict:
    """计算全部指标（含序数/临床），统一返回格式。"""
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    acc = float(accuracy_score(y_test, y_pred))
    f1_mac = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_wt = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0).tolist()
    per_class_prec = precision_score(y_test, y_pred, average=None, zero_division=0).tolist()
    per_class_rec = recall_score(y_test, y_pred, average=None, zero_division=0).tolist()
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))
    if y_bin.ndim == 1:
        y_bin = np.eye(n_classes, dtype=y_bin.dtype)[y_test]
    per_class_auroc, per_class_ap = [], []
    for i in range(n_classes):
        if i < y_bin.shape[1] and y_bin[:, i].sum() > 0:
            per_class_auroc.append(float(roc_auc_score(y_bin[:, i], y_prob[:, i])))
            per_class_ap.append(float(average_precision_score(y_bin[:, i], y_prob[:, i])))
        else:
            per_class_auroc.append(float("nan"))
            per_class_ap.append(float("nan"))
    macro_auroc = float(np.nanmean(per_class_auroc))
    macro_ap = float(np.nanmean(per_class_ap))

    diff = np.abs(y_pred.astype(np.int64) - y_test.astype(np.int64))
    adjacent_acc = float(np.mean(diff <= 1)) if len(diff) else float("nan")
    mae_ordinal = float(np.mean(diff)) if len(diff) else float("nan")
    true_hr = y_test >= 3
    if true_hr.any():
        high_risk_sens = float(np.mean(y_pred[true_hr] >= 3))
        severe_underestimation_rate = float(np.mean(y_pred[true_hr] <= 1))
    else:
        high_risk_sens = float("nan")
        severe_underestimation_rate = float("nan")
    true_lr = y_test < 3
    high_risk_spec = float(np.mean(y_pred[true_lr] < 3)) if true_lr.any() else float("nan")
    try:
        qwk = float(cohen_kappa_score(y_test, y_pred, weights="quadratic"))
    except Exception:  # noqa: BLE001
        qwk = float("nan")

    return dict(
        accuracy=acc,
        macro_f1=f1_mac,
        weighted_f1=f1_wt,
        macro_auroc=macro_auroc,
        macro_ap=macro_ap,
        per_class_f1=per_class_f1,
        per_class_precision=per_class_prec,
        per_class_recall=per_class_rec,
        per_class_auroc=per_class_auroc,
        per_class_ap=per_class_ap,
        adjacent_acc=adjacent_acc,
        high_risk_sens=high_risk_sens,
        high_risk_spec=high_risk_spec,
        severe_underestimation_rate=severe_underestimation_rate,
        qwk=qwk,
        mae_ordinal=mae_ordinal,
        n_eval=int(len(y_test)),
    )


# ═══════════════════════════════════════════════════════════════
# 冷启动药物划分（定义必须与基线执行者逐字一致）
# ═══════════════════════════════════════════════════════════════
def make_drug_cold_start_split(
    pairs: pd.DataFrame,
    y: np.ndarray,
    drug_split_seed: int,
) -> dict:
    """按药物冷启动划分 KK / S1 / S2，以及 KK 上的 train/val。

    定义（禁止改动）:
      drugs = sorted(set(pairs.drug1_id) | set(pairs.drug2_id))
      rng = np.random.default_rng(drug_split_seed)
      perm = rng.permutation(len(drugs))
      n_u = int(round(0.2 * len(drugs)))
      U = {drugs[i] for i in perm[:n_u]}，其余为 K
      KK = 两药均在 K；S1 = 恰一药在 U；S2 = 两药均在 U
      对 KK: train_test_split(test_size=0.10, stratify=y_KK,
                              random_state=drug_split_seed)
      → 10% 验证（早停），其余训练
    """
    drugs = sorted(set(pairs.drug1_id) | set(pairs.drug2_id))
    rng = np.random.default_rng(drug_split_seed)
    perm = rng.permutation(len(drugs))
    n_u = int(round(0.2 * len(drugs)))
    U = {drugs[i] for i in perm[:n_u]}
    K = set(drugs) - U

    d1 = pairs["drug1_id"].to_numpy()
    d2 = pairs["drug2_id"].to_numpy()
    u_list = np.array(list(U), dtype=object)
    in_u1 = np.isin(d1, u_list)
    in_u2 = np.isin(d2, u_list)
    n_u_pair = in_u1.astype(np.int8) + in_u2.astype(np.int8)
    KK = np.flatnonzero(n_u_pair == 0)
    S1 = np.flatnonzero(n_u_pair == 1)
    S2 = np.flatnonzero(n_u_pair == 2)

    idx_train, idx_val = _safe_stratified_split(
        KK, y, test_size=0.10, random_state=drug_split_seed,
    )
    print(f"\n[drug-split seed={drug_split_seed}]")
    print(f"  drugs={len(drugs):,}  |U|={len(U):,}  |K|={len(K):,}")
    print(f"  KK={len(KK):,}  S1={len(S1):,}  S2={len(S2):,}")
    print(f"  train={len(idx_train):,}  val={len(idx_val):,}")
    return dict(
        U=U, K=K, KK=KK, S1=S1, S2=S2,
        idx_train=idx_train, idx_val=idx_val,
        drug_split_seed=drug_split_seed,
    )


def _safe_stratified_split(idx: np.ndarray, y: np.ndarray,
                           test_size, random_state: int):
    """严格按 stratify 划分；仅当 sklearn 无法分层（极小/冒烟）时退回无分层。"""
    idx = np.asarray(idx)
    if len(idx) < 2:
        return idx, idx[:0]
    y_sub = y[idx]
    try:
        return train_test_split(
            idx, test_size=test_size, stratify=y_sub, random_state=random_state,
        )
    except ValueError as e:
        print(f"  [警告] stratify 失败 ({e})，改为非分层划分（仅冒烟/极端分布）")
        return train_test_split(
            idx, test_size=test_size, random_state=random_state,
        )


# ═══════════════════════════════════════════════════════════════
# 数据准备
# ═══════════════════════════════════════════════════════════════
def _stratified_subsample(y: np.ndarray, frac: float = 0.01,
                          random_state: int = SPLIT_RANDOM_STATE) -> np.ndarray:
    n = max(int(round(len(y) * frac)), 1)
    n = min(n, len(y))
    idx = np.arange(len(y))
    try:
        keep, _ = train_test_split(
            idx, train_size=n, stratify=y, random_state=random_state,
        )
    except ValueError:
        print("  [警告] 1% 分层抽样失败，改用均匀随机抽样")
        keep = np.random.default_rng(random_state).choice(idx, size=n, replace=False)
    return np.sort(keep)


def prepare_global_data(smoke: bool, n_folds: int):
    """加载全量数据、KG 静态缓存、固定划分出 test / train+val。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[global] 设备: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    data_dir = P().data_dir
    print(f"\n[global] data_dir = {data_dir}")
    print(f"[global] out_dir  = {P().out_dir}")

    print("\n[global] 构建实体目录（药物/酶/转运体/靶点）...")
    _, drug_fp_raw, kg_meta = build_knowledge_graph(
        include_bio=True, ddi_pairs=None,
    )
    kg_static = getattr(build_knowledge_graph, "last_cache", None)
    n_drugs = kg_meta["n_drugs"]
    n_bio = kg_meta["n_enz"] + kg_meta["n_tp"] + kg_meta["n_tgt"]

    print("\n[global] 加载 DDI 对数据...")
    pair_idx, y, valid = load_ddi_pairs(kg_meta["drug_id_to_idx"], data_dir=data_dir)
    orig_idx = np.flatnonzero(valid)
    pairs_full = pd.read_csv(data_dir / "pairs.csv")
    with open(data_dir / "meta.json", encoding="utf-8") as f:
        data_meta = json.load(f)
    n_s = int(data_meta["n_samples"])
    pairs_full = pairs_full.iloc[:n_s]
    pairs = pairs_full.iloc[orig_idx].reset_index(drop=True)

    if smoke:
        print("\n[smoke] 分层抽取 1% 样本（不改变全量划分随机种子定义，仅缩小本地验证集）")
        keep = _stratified_subsample(y, frac=0.01, random_state=SPLIT_RANDOM_STATE)
        pair_idx = pair_idx[keep]
        y = y[keep]
        orig_idx = orig_idx[keep]
        pairs = pairs.iloc[keep].reset_index(drop=True)
        print(f"  smoke n={len(y):,}  分布={np.bincount(y, minlength=5).tolist()}")

    all_idx = np.arange(len(y))
    idx_tv, idx_test = train_test_split(
        all_idx, test_size=0.2, stratify=y, random_state=SPLIT_RANDOM_STATE,
    )
    print("\n[global] 数据划分（全局一次, pair 模式）:")
    print(f"  train+val: {len(idx_tv):>8,}  ({len(idx_tv)/len(y)*100:.1f}%)")
    print(f"  test:      {len(idx_test):>8,}  ({len(idx_test)/len(y)*100:.1f}%)")

    cv_splits = []
    if n_folds == 1:
        tr, va = _safe_stratified_split(
            idx_tv, y, test_size=0.2, random_state=SPLIT_RANDOM_STATE,
        )
        cv_splits.append((1, tr, va))
        print("  [n_folds=1] 在 train+val 上做一次 80/20 分层划分（等价于一折的 val 比例）")
    else:
        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=SPLIT_RANDOM_STATE,
        )
        for fold_idx, (train_rel, val_rel) in enumerate(skf.split(idx_tv, y[idx_tv]), start=1):
            cv_splits.append((fold_idx, idx_tv[train_rel], idx_tv[val_rel]))
        print(f"\n[global] {n_folds}-fold CV (在 train+val 上)")

    y_test_arr = y[np.sort(idx_test)]
    return dict(
        device=device,
        drug_fp=drug_fp_raw,
        kg_meta=kg_meta,
        kg_static=kg_static,
        n_drugs=n_drugs,
        n_bio=n_bio,
        pair_idx=pair_idx,
        y=y,
        orig_idx=orig_idx,
        pairs=pairs,
        data_meta=data_meta,
        idx_tv=idx_tv,
        idx_test=idx_test,
        cv_splits=cv_splits,
        y_test_fixed=y_test_arr,
        smoke=smoke,
    )


def build_graphs_for_fold(ctx: dict, idx_train_f: np.ndarray):
    """基于当前训练边集合构建 仅DDI图 和 混合图。

    只加入 idx_train_f 中的 DDI 边（训练折 / KK-train）。U 药作为节点保留其特征；
    Hybrid 的药物–属性静态边（酶/转运体/靶点，含反向）全部保留，但 U 药的任何
    DDI 边都不会进入图（因为它们不在 idx_train_f 里）。
    """
    train_pairs = ctx["pair_idx"][idx_train_f]
    train_labels = ctx["y"][idx_train_f]
    cache = ctx.get("kg_static")
    graph_ddi, _, _ = build_knowledge_graph(
        ddi_pairs=train_pairs, ddi_labels=train_labels, include_bio=False,
        static_cache=cache,
    )
    graph_hybrid, _, _ = build_knowledge_graph(
        ddi_pairs=train_pairs, ddi_labels=train_labels, include_bio=True,
        static_cache=cache,
    )
    return graph_ddi, graph_hybrid


# ═══════════════════════════════════════════════════════════════
# 训练辅助
# ═══════════════════════════════════════════════════════════════
def _save_xgb(model, json_path: Path, pkl_path: Path) -> None:
    model.save_model(str(json_path))
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  [保存] {json_path.name} + {pkl_path.name}")


def _build_gnn_model(method: str, ctx: dict) -> torch.nn.Module:
    cfg = EXP_CONFIGS[method]
    common = dict(
        n_drugs=ctx["n_drugs"], n_bio=ctx["n_bio"],
        hidden_dim=cfg["hidden_dim"], emb_dim=cfg["emb_dim"],
        num_rels=NUM_RELATIONS, dropout=0.3,
        norm_type=cfg["norm_type"], decoder_type=cfg["decoder_type"],
    )
    if cfg["model"] == "rgcn":
        return RGCN_DDI(num_bases=cfg["num_bases"], **common)
    return RGAT_DDI(edge_emb_dim=cfg["edge_emb_dim"], heads=cfg["heads"],
                    self_loop_fill=cfg["self_loop_fill"], **common)


def _make_gnn_train_config(method: str, smoke: bool) -> dict:
    fg = EXP_CONFIGS[method]["focal_gamma"]
    if smoke:
        return {**GNN_COMMON_CFG, "focal_gamma": fg,
                "max_epochs": 2, "patience": 2, "min_epochs": 1,
                "warmup_epochs": 1}
    return {**GNN_COMMON_CFG, "focal_gamma": fg,
            "max_epochs": 1500, "patience": 150, "min_epochs": 80}


def _xgb_n_est_es(smoke: bool) -> tuple[int, int]:
    if smoke:
        return 50, 10
    return 200000, 200


def _load_X_mm():
    mm_path = P().data_dir / "X_full.dat"
    x_path = P().data_dir / "X.npy"
    meta_path = P().data_dir / "meta.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    n_s, f_d = meta["n_samples"], meta["feature_dim"]
    if mm_path.exists():
        X_mm = np.memmap(mm_path, dtype="float32", mode="r", shape=(n_s, f_d))
    elif x_path.exists():
        X_mm = np.load(x_path, mmap_mode="r")
    else:
        raise FileNotFoundError(f"未找到 {mm_path} 或 {x_path}")
    return X_mm, n_s, f_d


def _load_xy_for_indices(ctx: dict, X_mm, local_idx: np.ndarray):
    """按 orig_idx（pairs.csv 行号）顺序读 memmap，y/pair 与之对齐。"""
    local_idx = np.asarray(local_idx)
    mem = ctx["orig_idx"][local_idx]
    order = np.argsort(mem, kind="mergesort")
    mem_s = mem[order]
    loc_s = local_idx[order]
    X = np.array(X_mm[mem_s], dtype=np.float32)
    y = ctx["y"][loc_s]
    pairs = ctx["pair_idx"][loc_s]
    return X, y, pairs, loc_s, mem_s


def _predictions_cache_path(method: str, fold: int, seed: int,
                            test_set: str | None = None) -> Path:
    name = f"{method}_f{fold}_s{seed}"
    if test_set and test_set not in ("test", "holdout"):
        name += f"_{test_set}"
    return P().pred_dir / f"{name}.npz"


def _save_predictions(method: str, fold: int, seed: int,
                      test_idx: np.ndarray, y_true: np.ndarray,
                      y_pred: np.ndarray, y_prob: np.ndarray,
                      test_set: str | None = None) -> None:
    """每个 (method, fold, seed[, test_set]) 都保存。test_idx 相对 pairs.csv 行。"""
    p = _predictions_cache_path(method, fold, seed, test_set)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        test_idx=np.asarray(test_idx, dtype=np.int64),
        y_true=np.asarray(y_true, dtype=np.int8),
        y_pred=np.asarray(y_pred, dtype=np.int8),
        y_prob=np.asarray(y_prob, dtype=np.float32),
    )
    print(f"  [保存] 预测 → {p.name}")


def _gnn_evaluate(model, graph, drug_fp, pair_idx, y, idx_test, device):
    model.eval()
    graph = graph.to(device)
    drug_fp = drug_fp.to(device)
    ei, et = graph.edge_index, graph.edge_type
    test_pairs = pair_idx[idx_test]
    y_test = y[idx_test]
    if len(idx_test) == 0:
        empty_prob = np.zeros((0, 5), dtype=np.float32)
        metrics = _evaluate_predictions(
            np.zeros(0, dtype=int), np.zeros(0, dtype=int), empty_prob, n_classes=5,
        )
        return y_test, np.zeros(0, dtype=int), empty_prob, metrics
    test_pairs_t = torch.from_numpy(test_pairs).to(device, dtype=torch.long, non_blocking=True)
    with torch.no_grad():
        h = model.encode(drug_fp, ei, et)
    all_logits = []
    with torch.no_grad():
        for start in range(0, test_pairs_t.size(0), 8192):
            bp = test_pairs_t[start: start + 8192]
            h_a = h[bp[:, 0]]
            h_b = h[bp[:, 1]]
            logits = model.pair_mlp(h_a, h_b)
            all_logits.append(logits.cpu())
    all_logits = torch.cat(all_logits, dim=0)
    y_prob = torch.nn.functional.softmax(all_logits, dim=1).numpy()
    y_pred = all_logits.argmax(1).numpy()
    metrics = _evaluate_predictions(y_test, y_pred, y_prob, n_classes=5)
    return y_test, y_pred, y_prob, metrics


def _result_record(method, fold, seed, metrics, elapsed, n_params=0,
                   test_set="test") -> dict:
    return dict(
        method=method, fold=fold, seed=seed, test_set=test_set,
        n_params=int(n_params), elapsed_sec=round(elapsed, 1), **metrics,
    )


def run_exp_gnn(method: str, ctx: dict, graph, fold: int, seed: int,
                idx_train_f: np.ndarray, idx_val_f: np.ndarray,
                idx_test, smoke: bool, save_ckpt: bool,
                test_set: str = "test",
                save_bundle_dir: Path | None = None) -> list[dict]:
    """训练一个 GNN。idx_test 可以是 ndarray，或 {'S1': idx, 'S2': idx}。"""
    exp_id = int(method.split("-")[-1])
    set_reproducibility(seed + (exp_id - 2) * 1_000_003, deterministic_cuda=True)
    _clear_gpu()
    model = _build_gnn_model(method, ctx)
    n_params = sum(p.numel() for p in model.parameters())
    train_cfg = _make_gnn_train_config(method, smoke)

    print(f"\n  {'-' * 60}")
    print(f"  {method} | fold={fold} seed={seed} | 参数量 {n_params:,}")
    print(f"  {'-' * 60}")

    t0 = time.time()
    model, hist = train_model(
        model, graph, ctx["drug_fp"],
        ctx["pair_idx"], ctx["y"],
        idx_train_f, idx_val_f, ctx["device"], train_cfg,
    )
    if save_ckpt:
        ckpt_path = P().ckpt_dir / f"{method}_f{fold}_s{seed}.pt"
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, ckpt_path)
        print(f"  [保存] 权重 → {ckpt_path}")
        if save_bundle_dir is not None:
            dst = Path(save_bundle_dir) / "exp3_rgcn.pt"
            dst.write_bytes(ckpt_path.read_bytes())

    test_map = idx_test if isinstance(idx_test, dict) else {test_set: idx_test}
    records = []
    for ts, tidx in test_map.items():
        y_test, y_pred, y_prob, metrics = _gnn_evaluate(
            model, graph, ctx["drug_fp"],
            ctx["pair_idx"], ctx["y"], np.asarray(tidx), ctx["device"],
        )
        test_idx_csv = ctx["orig_idx"][np.asarray(tidx)]
        _save_predictions(method, fold, seed, test_idx_csv, y_test, y_pred, y_prob, ts)
        elapsed = time.time() - t0
        records.append(_result_record(method, fold, seed, metrics, elapsed, n_params, ts))
    del model
    _clear_gpu()
    gc.collect()
    return records


def run_exp1_xgboost(ctx: dict, fold: int, seed: int,
                     idx_train_f: np.ndarray, idx_val_f: np.ndarray,
                     idx_test, smoke: bool, test_set: str = "test",
                     save_bundle_dir: Path | None = None) -> list[dict]:
    from xgboost import XGBClassifier

    X_mm, n_s, f_d = _load_X_mm()
    print(f"\n  [Exp-1 fold={fold} seed={seed}] 加载 X_mm (排序后顺序读)...")
    X_train, y_train, _, _, _ = _load_xy_for_indices(ctx, X_mm, idx_train_f)
    X_val, y_val, _, _, _ = _load_xy_for_indices(ctx, X_mm, idx_val_f)

    n_est, es = _xgb_n_est_es(smoke)
    n_classes = len(np.unique(ctx["y"]))
    xgb = XGBClassifier(
        n_estimators=n_est, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.6,
        eval_metric=["mlogloss", "merror"],
        random_state=seed,
        device="cuda", tree_method="hist", n_jobs=1,
        early_stopping_rounds=es,
        objective="multi:softprob",
        num_class=n_classes,
    )
    sw = compute_sample_weight("balanced", y_train)
    n_eval_tr = min(100_000, len(X_train))
    eval_idx = np.random.default_rng(seed).choice(
        len(X_train), n_eval_tr, replace=False,
    )

    print(f"  [Exp-1] 训练 XGBoost (n_est={n_est}, es={es})...")
    t0 = time.time()
    xgb.fit(
        X_train, y_train, sample_weight=sw,
        eval_set=[(X_train[eval_idx], y_train[eval_idx]), (X_val, y_val)],
        verbose=100,
    )
    train_elapsed = time.time() - t0
    if save_bundle_dir is not None:
        _save_xgb(xgb, Path(save_bundle_dir) / "exp1_xgboost.json",
                  Path(save_bundle_dir) / "exp1_xgboost.pkl")

    test_map = idx_test if isinstance(idx_test, dict) else {test_set: idx_test}
    records = []
    for ts, tidx in test_map.items():
        X_test, y_test, _, loc_s, mem_s = _load_xy_for_indices(ctx, X_mm, np.asarray(tidx))
        if len(y_test) == 0:
            y_pred = np.zeros(0, dtype=int)
            y_prob = np.zeros((0, n_classes), dtype=np.float32)
        else:
            y_pred = xgb.predict(X_test)
            y_prob = xgb.predict_proba(X_test)
        metrics = _evaluate_predictions(y_test, y_pred, y_prob, n_classes=n_classes)
        _save_predictions("Exp-1", fold, seed, mem_s, y_test, y_pred, y_prob, ts)
        records.append(_result_record("Exp-1", fold, seed, metrics, train_elapsed, 0, ts))
        del X_test

    del X_train, X_val, X_mm, xgb
    gc.collect()
    return records


def run_exp5_fusion(ctx: dict, fold: int, seed: int,
                    idx_train_f: np.ndarray, idx_val_f: np.ndarray,
                    graph_hybrid, idx_test, smoke: bool,
                    test_set: str = "test",
                    save_bundle_dir: Path | None = None) -> list[dict]:
    from xgboost import XGBClassifier

    ckpt_path = P().ckpt_dir / f"Exp-3_f{fold}_s{seed}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"未找到 Exp-3 权重 {ckpt_path}，必须先跑 Exp-3。")

    print(f"\n  [Exp-5 fold={fold} seed={seed}] 加载 Exp-3 权重 {ckpt_path.name}")
    device = ctx["device"]
    set_reproducibility(seed + 3 * 1_000_003, deterministic_cuda=True)
    rgcn = _build_gnn_model("Exp-3", ctx)
    state = torch.load(ckpt_path, map_location=device)
    rgcn.load_state_dict(state)
    rgcn.to(device)
    rgcn.eval()

    graph_hybrid_d = graph_hybrid.to(device)
    drug_fp_d = ctx["drug_fp"].to(device)
    with torch.no_grad():
        drug_emb = rgcn.get_drug_embeddings(
            drug_fp_d, graph_hybrid_d.edge_index, graph_hybrid_d.edge_type,
        ).cpu().numpy()
    emb_dim = drug_emb.shape[1]
    print(f"  [Exp-5] 药物嵌入维度: {emb_dim}")
    if save_bundle_dir is not None:
        np.save(Path(save_bundle_dir) / "drug_embeddings.npy",
                drug_emb.astype(np.float32))

    def _pair_emb(pairs):
        return pair_ops_embed(drug_emb[pairs[:, 0]], drug_emb[pairs[:, 1]])

    X_mm, n_s, f_d = _load_X_mm()
    print("  [Exp-5] 加载 X_mm (排序后顺序读)...")
    X_tr, y_train, pairs_tr, _, _ = _load_xy_for_indices(ctx, X_mm, idx_train_f)
    X_vl, y_val, pairs_vl, _, _ = _load_xy_for_indices(ctx, X_mm, idx_val_f)
    gnn_tr = _pair_emb(pairs_tr)
    gnn_vl = _pair_emb(pairs_vl)
    print("  [Exp-5] 拼接融合特征...")
    X_train = np.concatenate([X_tr, gnn_tr], axis=1)
    X_val = np.concatenate([X_vl, gnn_vl], axis=1)
    print(f"  [Exp-5] 融合特征维度: {X_train.shape[1]} "
          f"({f_d} XGBoost + {gnn_tr.shape[1]} GNN pair)")
    del X_tr, X_vl, gnn_tr, gnn_vl, rgcn
    _clear_gpu()
    gc.collect()

    n_est, es = _xgb_n_est_es(smoke)
    n_classes = len(np.unique(ctx["y"]))
    xgb = XGBClassifier(
        n_estimators=n_est, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.6,
        eval_metric=["mlogloss", "merror"],
        random_state=seed,
        device="cuda", tree_method="hist", n_jobs=1,
        early_stopping_rounds=es,
        objective="multi:softprob",
        num_class=n_classes,
    )
    sw = compute_sample_weight("balanced", y_train)
    n_eval_tr = min(100_000, len(X_train))
    eval_idx = np.random.default_rng(seed).choice(
        len(X_train), n_eval_tr, replace=False,
    )
    print(f"  [Exp-5] 训练融合 XGBoost (n_est={n_est}, es={es})...")
    t0 = time.time()
    xgb.fit(
        X_train, y_train, sample_weight=sw,
        eval_set=[(X_train[eval_idx], y_train[eval_idx]), (X_val, y_val)],
        verbose=100,
    )
    train_elapsed = time.time() - t0
    if save_bundle_dir is not None:
        _save_xgb(xgb, Path(save_bundle_dir) / "exp5_xgboost.json",
                  Path(save_bundle_dir) / "exp5_xgboost.pkl")
    del X_train, X_val
    gc.collect()

    test_map = idx_test if isinstance(idx_test, dict) else {test_set: idx_test}
    records = []
    for ts, tidx in test_map.items():
        X_te, y_test, pairs_te, loc_s, mem_s = _load_xy_for_indices(
            ctx, X_mm, np.asarray(tidx),
        )
        if len(y_test) == 0:
            y_pred = np.zeros(0, dtype=int)
            y_prob = np.zeros((0, n_classes), dtype=np.float32)
        else:
            X_test = np.concatenate([X_te, _pair_emb(pairs_te)], axis=1)
            y_pred = xgb.predict(X_test)
            y_prob = xgb.predict_proba(X_test)
            del X_test
        metrics = _evaluate_predictions(y_test, y_pred, y_prob, n_classes=n_classes)
        _save_predictions("Exp-5", fold, seed, mem_s, y_test, y_pred, y_prob, ts)
        records.append(_result_record("Exp-5", fold, seed, metrics, train_elapsed, 0, ts))
        del X_te

    del X_mm, xgb, drug_emb
    gc.collect()
    return records


# ═══════════════════════════════════════════════════════════════
# 状态 / ETA
# ═══════════════════════════════════════════════════════════════
def load_state() -> dict:
    if P().state_path.exists():
        try:
            with open(P().state_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"runs": {}, "mode": None}


def save_state(state: dict) -> None:
    with open(P().state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def ensure_mode(state: dict, smoke: bool, split_mode: str,
                force_reset: bool) -> dict:
    cur = f"{'smoke' if smoke else 'full'}:{split_mode}"
    prev = state.get("mode")
    if prev is not None and prev != cur:
        msg = (f"[pipeline_final] out_dir 已有 mode={prev}，当前={cur}。")
        if force_reset:
            print(msg + " --force_reset，清空 state。")
            state = {"runs": {}, "mode": cur}
            save_state(state)
            return state
        raise SystemExit(
            msg + " 请换 --out_dir，或加 --force_reset（会清空该目录下的 state）。",
        )
    state["mode"] = cur
    save_state(state)
    return state


def run_key(method: str, fold: int, seed: int, test_set: str | None = None) -> str:
    k = f"{method}_f{fold}_s{seed}"
    if test_set and test_set not in ("test", "holdout"):
        k += f"_{test_set}"
    return k


def _is_run_done(state: dict, key: str) -> bool:
    r = state.get("runs", {}).get(key)
    if r is None or "error" in r:
        return False
    return "macro_f1" in r


def _records_done(state: dict, method: str, fold: int, seed: int,
                  test_sets: list[str]) -> bool:
    return all(_is_run_done(state, run_key(method, fold, seed, ts)) for ts in test_sets)


class EtaTracker:
    def __init__(self, total_jobs: int, smoke: bool):
        self.total = max(total_jobs, 1)
        self.done = 0
        self.smoke = smoke
        self.by_method: dict[str, list[float]] = {m: [] for m in METHODS_ORDER}
        self.remaining_labels: list[str] = []

    def set_remaining(self, labels: list[str]) -> None:
        self.remaining_labels = list(labels)

    def _prior(self, method: str) -> float:
        scale = 0.04 if self.smoke else 1.0
        return HIST_MEAN_SEC.get(method, 600.0) * scale

    def after(self, method: str, elapsed: float) -> None:
        self.done += 1
        self.by_method.setdefault(method, []).append(float(elapsed))
        rem_sec = 0.0
        for lab in self.remaining_labels:
            m = lab.split("|")[0]
            hist = self.by_method.get(m) or []
            rem_sec += float(np.mean(hist)) if hist else self._prior(m)
        print(f"  [进度] {self.done}/{self.total}  本run {elapsed:.1f}s  "
              f"预计剩余 {rem_sec/60:.1f} min ({rem_sec/3600:.2f} h)")


def _store_records(state: dict, records: list[dict]) -> None:
    for r in records:
        state["runs"][run_key(r["method"], r["fold"], r["seed"], r.get("test_set"))] = r
    save_state(state)


# ═══════════════════════════════════════════════════════════════
# 主循环
# ═══════════════════════════════════════════════════════════════
def _run_method_block(state, ctx, methods, fold, seed, idx_train, idx_val,
                      idx_test, graph_ddi, graph_hybrid, smoke, test_sets,
                      eta: EtaTracker, remaining: list[str]) -> None:
    def _try(method: str, runner):
        if _records_done(state, method, fold, seed, test_sets):
            print(f"  [跳过-已完成] {method}_f{fold}_s{seed}")
            eta.done += 1
            eta.set_remaining(remaining)
            eta.after(method, 0.0)
            return
        try:
            recs = runner()
            _store_records(state, recs)
            elapsed = recs[0].get("elapsed_sec", 0.0) if recs else 0.0
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {method}_f{fold}_s{seed}: {e}")
            traceback.print_exc()
            for ts in test_sets:
                state["runs"][run_key(method, fold, seed, ts)] = dict(
                    method=method, fold=fold, seed=seed, test_set=ts, error=str(e),
                )
            save_state(state)
            _clear_gpu()
            elapsed = 0.0
        eta.set_remaining(remaining)
        eta.after(method, float(elapsed))

    if "Exp-2" in methods and graph_ddi is not None:
        _try("Exp-2", lambda: run_exp_gnn(
            "Exp-2", ctx, graph_ddi, fold, seed,
            idx_train, idx_val, idx_test, smoke, save_ckpt=False))
        if remaining:
            remaining.pop(0)

    if "Exp-3" in methods and graph_hybrid is not None:
        _try("Exp-3", lambda: run_exp_gnn(
            "Exp-3", ctx, graph_hybrid, fold, seed,
            idx_train, idx_val, idx_test, smoke, save_ckpt=True))
        if remaining:
            remaining.pop(0)

    if "Exp-4" in methods and graph_hybrid is not None:
        _try("Exp-4", lambda: run_exp_gnn(
            "Exp-4", ctx, graph_hybrid, fold, seed,
            idx_train, idx_val, idx_test, smoke, save_ckpt=False))
        if remaining:
            remaining.pop(0)

    if "Exp-5" in methods and graph_hybrid is not None:
        exp3_ckpt = P().ckpt_dir / f"Exp-3_f{fold}_s{seed}.pt"
        if not exp3_ckpt.exists():
            print(f"  [跳过-Exp-5] 对应 Exp-3 权重不存在: {exp3_ckpt.name}")
            for ts in test_sets:
                k = run_key("Exp-5", fold, seed, ts)
                if k not in state["runs"]:
                    state["runs"][k] = dict(
                        method="Exp-5", fold=fold, seed=seed, test_set=ts,
                        error="Exp-3 ckpt missing",
                    )
            save_state(state)
            eta.done += 1
        else:
            _try("Exp-5", lambda: run_exp5_fusion(
                ctx, fold, seed, idx_train, idx_val, graph_hybrid, idx_test, smoke))
        if remaining:
            remaining.pop(0)

    if "Exp-1" in methods:
        _try("Exp-1", lambda: run_exp1_xgboost(
            ctx, fold, seed, idx_train, idx_val, idx_test, smoke))
        if remaining:
            remaining.pop(0)


def main_loop_pair(ctx: dict, smoke: bool, methods: list[str],
                   n_folds: int, seeds: list[int],
                   force_reset: bool = False) -> dict:
    state = load_state()
    state = ensure_mode(state, smoke, "pair", force_reset=force_reset)
    need_graph = any(m in methods for m in ("Exp-2", "Exp-3", "Exp-4", "Exp-5"))
    jobs = []
    for fold, _, _ in ctx["cv_splits"]:
        for seed in seeds:
            for m in _exec_order(methods):
                jobs.append(f"{m}|f{fold}|s{seed}")
    eta = EtaTracker(len(jobs), smoke)
    remaining = list(jobs)

    for fold, idx_train_f, idx_val_f in ctx["cv_splits"]:
        if need_graph:
            print(f"\n{'█' * 72}")
            print(f"  Fold {fold}/{n_folds}  |  构建图...")
            print(f"{'█' * 72}")
            graph_ddi, graph_hybrid = build_graphs_for_fold(ctx, idx_train_f)
        else:
            graph_ddi, graph_hybrid = None, None
        for seed in seeds:
            print(f"\n{'━' * 72}")
            print(f"  Fold {fold}/{n_folds}  Seed {seed}")
            print(f"{'━' * 72}")
            _run_method_block(
                state, ctx, methods, fold, seed, idx_train_f, idx_val_f,
                ctx["idx_test"], graph_ddi, graph_hybrid, smoke, ["test"],
                eta, remaining,
            )
        if need_graph:
            del graph_ddi, graph_hybrid
            _clear_gpu()
            gc.collect()
    return state


def main_loop_drug(ctx: dict, smoke: bool, methods: list[str],
                   drug_split_seeds: list[int],
                   force_reset: bool = False) -> dict:
    state = load_state()
    state = ensure_mode(state, smoke, "drug", force_reset=force_reset)
    need_graph = any(m in methods for m in ("Exp-2", "Exp-3", "Exp-4", "Exp-5"))
    jobs = []
    for seed in drug_split_seeds:
        for m in _exec_order(methods):
            jobs.append(f"{m}|f0|s{seed}")
    eta = EtaTracker(len(jobs), smoke)
    remaining = list(jobs)

    for seed in drug_split_seeds:
        split = make_drug_cold_start_split(ctx["pairs"], ctx["y"], seed)
        idx_train, idx_val = split["idx_train"], split["idx_val"]
        idx_test = {"S1": split["S1"], "S2": split["S2"]}
        # 图：只加入训练集中 KK 的 DDI 边
        if need_graph:
            print(f"\n{'█' * 72}")
            print(f"  Drug-split seed={seed}  |  构建图（仅 KK-train DDI 边）...")
            print(f"{'█' * 72}")
            graph_ddi, graph_hybrid = build_graphs_for_fold(ctx, idx_train)
        else:
            graph_ddi, graph_hybrid = None, None
        print(f"\n{'━' * 72}")
        print(f"  Drug-split seed={seed}  (模型种子 = 划分种子)")
        print(f"{'━' * 72}")
        _run_method_block(
            state, ctx, methods, 0, seed, idx_train, idx_val,
            idx_test, graph_ddi, graph_hybrid, smoke, ["S1", "S2"],
            eta, remaining,
        )
        if need_graph:
            del graph_ddi, graph_hybrid
            _clear_gpu()
            gc.collect()
    return state


def _exec_order(methods: list[str]) -> list[str]:
    """GNN 先于 Fusion，Exp-1 最后（与历史流水线一致）。"""
    order = ["Exp-2", "Exp-3", "Exp-4", "Exp-5", "Exp-1"]
    return [m for m in order if m in methods]


# ═══════════════════════════════════════════════════════════════
# --final_full
# ═══════════════════════════════════════════════════════════════
def run_final_full(ctx: dict, smoke: bool) -> None:
    """在 80% 开发集上训练 Exp-3 / Exp-5 / Exp-1，写出推理包。"""
    out = P().final_full_dir
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'█' * 72}")
    print("  --final_full：开发集训练 Exp-3 / Exp-5 / Exp-1")
    print(f"  输出: {out}")
    print(f"{'█' * 72}")

    y = ctx["y"]
    idx_tv, idx_test = ctx["idx_tv"], ctx["idx_test"]
    idx_train, idx_val = _safe_stratified_split(
        idx_tv, y, test_size=0.05, random_state=SPLIT_RANDOM_STATE,
    )
    print(f"  开发集 {len(idx_tv):,} → train {len(idx_train):,} + val(5%) {len(idx_val):,}")
    print(f"  封存测试 {len(idx_test):,}（不进入图、不进入损失）")

    seed = SPLIT_RANDOM_STATE
    graph_ddi, graph_hybrid = build_graphs_for_fold(ctx, idx_train)

    rec3 = run_exp_gnn(
        "Exp-3", ctx, graph_hybrid, fold=0, seed=seed,
        idx_train_f=idx_train, idx_val_f=idx_val, idx_test=idx_test,
        smoke=smoke, save_ckpt=True, test_set="test", save_bundle_dir=out,
    )[0]

    exp3_cfg = dict(
        hidden_dim=2048, emb_dim=1024, num_bases=8,
        num_rels=NUM_RELATIONS, dropout=0.3,
        norm_type="batch", decoder_type="mlp", fp_dim=2048,
        n_drugs=ctx["n_drugs"], n_bio=ctx["n_bio"],
    )
    with open(out / "exp3_config.json", "w", encoding="utf-8") as f:
        json.dump(exp3_cfg, f, ensure_ascii=False, indent=2)

    np.save(out / "graph_edge_index.npy", graph_hybrid.edge_index.cpu().numpy())
    np.save(out / "graph_edge_type.npy", graph_hybrid.edge_type.cpu().numpy())
    np.save(out / "drug_fingerprints.npy", ctx["drug_fp"].cpu().numpy().astype(np.float32))
    with open(out / "drug_id_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(ctx["kg_meta"]["drug_id_to_idx"], f, ensure_ascii=False, indent=2)
    kg_save = {
        "n_drugs": ctx["kg_meta"]["n_drugs"],
        "n_enz": ctx["kg_meta"]["n_enz"],
        "n_tp": ctx["kg_meta"]["n_tp"],
        "n_tgt": ctx["kg_meta"]["n_tgt"],
        "num_relations": ctx["kg_meta"]["num_relations"],
        "enzyme_list": ctx["kg_meta"]["enzyme_list"],
        "transporter_list": ctx["kg_meta"]["transporter_list"],
        "target_list": ctx["kg_meta"]["target_list"],
        "drug_ids": ctx["kg_meta"]["drug_ids"],
        "static_edge_types": {
            "forward": {
                "0-2": "drug→enzyme substrate/inhibitor/inducer",
                "9": "drug→enzyme other",
                "3-5": "drug→transporter substrate/inhibitor/inducer",
                "10": "drug→transporter other",
                "6-8": "drug→target agonist/antagonist/inhibitor",
                "11": "drug→target other",
                "12-16": "drug↔drug DDI risk 0-4 (training split only)",
            },
            "reverse_offset": 17,
            "note": "U 药节点及其药物–属性静态边可保留；DDI 边仅来自开发集训练集。",
        },
    }
    with open(out / "kg_meta.json", "w", encoding="utf-8") as f:
        json.dump(kg_save, f, ensure_ascii=False, indent=2)

    np.save(out / "train_idx.npy", ctx["orig_idx"][idx_train])
    np.save(out / "val_idx.npy", ctx["orig_idx"][idx_val])
    np.save(out / "test_idx.npy", ctx["orig_idx"][idx_test])

    rec5 = run_exp5_fusion(
        ctx, fold=0, seed=seed, idx_train_f=idx_train, idx_val_f=idx_val,
        graph_hybrid=graph_hybrid, idx_test=idx_test, smoke=smoke,
        test_set="test", save_bundle_dir=out,
    )[0]
    rec1 = run_exp1_xgboost(
        ctx, fold=0, seed=seed, idx_train_f=idx_train, idx_val_f=idx_val,
        idx_test=idx_test, smoke=smoke, test_set="test", save_bundle_dir=out,
    )[0]

    feature_layout = {
        "xgb_v2": {
            "dim": int(FEAT_DIM),
            "slices": {"fp_product": [0, 2048], "fp_diff": [2048, 4096],
                       "bio_24": [4096, 4096 + int(N_BIO)]},
            "bio_names": list(BIO_FEATURE_NAMES),
        },
        "gnn_pair_ops": dict(PAIR_OPS_SPEC),
        "fusion": {
            "dim": int(FEAT_DIM) + int(PAIR_OPS_SPEC["output_dim"]),
            "slices": {"xgb_v2": [0, int(FEAT_DIM)],
                       "gnn_pair": [int(FEAT_DIM), int(FEAT_DIM) + int(PAIR_OPS_SPEC["output_dim"])]},
        },
        "canonical_order": "drug_a < drug_b (DrugBank ID 字符串比较)，与 v2 pairs 一致",
    }
    with open(out / "feature_layout.json", "w", encoding="utf-8") as f:
        json.dump(feature_layout, f, ensure_ascii=False, indent=2)
    with open(out / "pair_ops.json", "w", encoding="utf-8") as f:
        json.dump(PAIR_OPS_SPEC, f, ensure_ascii=False, indent=2)

    label_names = ctx.get("data_meta", {}).get("label_names") or {
        str(i): RISK_LABELS_CN[i] for i in range(5)
    }
    classes = {"n_classes": 5, "label_names": label_names,
               "risk_labels_cn": RISK_LABELS_CN, "risk_labels_en": RISK_LABELS_EN}
    with open(out / "classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    metrics = {"Exp-1": rec1, "Exp-3": rec3, "Exp-5": rec5, "smoke": smoke}
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": str(P().data_dir),
        "graph_ddi_source": "training portion of development set (95% of 80%); val/test DDI 边不入图",
        "models": ["Exp-1", "Exp-3", "Exp-5"],
        "files": sorted(p.name for p in out.iterdir() if p.is_file()),
        "predict": "python -u models/predict_pairs.py --bundle <this_dir> --data_dir <processed> --input pairs.csv --out pred.csv",
    }
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"  [final_full] 完成 → {out}")
    del graph_ddi, graph_hybrid
    _clear_gpu()
    gc.collect()


# ═══════════════════════════════════════════════════════════════
# 聚合与报告
# ═══════════════════════════════════════════════════════════════
def _row_method_key(r: dict) -> str:
    ts = r.get("test_set") or "test"
    if ts in ("test", "holdout"):
        return r["method"]
    return f"{r['method']}/{ts}"


def aggregate_by_method(state: dict) -> dict:
    keys_seen: list[str] = []
    agg: dict = {}
    metric_names = list(CORE_METRICS) + list(ORDINAL_METRICS)
    for r in state.get("runs", {}).values():
        if "error" in r:
            continue
        m = _row_method_key(r)
        if m not in agg:
            agg[m] = {k: [] for k in metric_names}
            agg[m]["per_class_f1"] = []
            agg[m]["runs"] = []
            keys_seen.append(m)
        for mk in metric_names:
            if mk in r and r[mk] is not None:
                agg[m][mk].append(float(r[mk]))
        if "per_class_f1" in r:
            agg[m]["per_class_f1"].append(r["per_class_f1"])
        agg[m]["runs"].append(r)
    agg["_order"] = keys_seen
    return agg


def _ms(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return (float("nan"), float("nan"))
    return float(np.mean(vals)), float(np.std(vals))


def _paired_ttest(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    try:
        from scipy.stats import ttest_rel
        return float(ttest_rel(a, b).pvalue)
    except Exception:
        return float("nan")


def save_raw_csv(state: dict) -> None:
    rows = []
    extra = list(ORDINAL_METRICS)
    for k, r in state.get("runs", {}).items():
        if "error" in r:
            continue
        base = dict(
            method=r["method"], fold=r["fold"], seed=r["seed"],
            test_set=r.get("test_set", "test"),
            elapsed_sec=r.get("elapsed_sec", None),
            n_eval=r.get("n_eval", None),
        )
        for mk in CORE_METRICS:
            base[mk] = r.get(mk, None)
        for mk in extra:
            base[mk] = r.get(mk, None)
        pc = r.get("per_class_f1", [np.nan] * 5)
        for i in range(5):
            base[f"f1_c{i}"] = pc[i] if i < len(pc) else np.nan
        rows.append(base)
    df = pd.DataFrame(rows)
    df.to_csv(P().raw_csv_path, index=False, encoding="utf-8-sig")
    print(f"[report] 原始 CSV → {P().raw_csv_path}")


def _method_order_from_agg(agg: dict) -> list[str]:
    seen = agg.get("_order") or []
    preferred = []
    for m in METHODS_ORDER:
        preferred.extend([k for k in seen if k == m or k.startswith(m + "/")])
    rest = [k for k in seen if k not in preferred and k != "_order"]
    return [k for k in preferred + rest if k in agg]


def generate_report(state: dict, agg: dict) -> None:
    L: list[str] = []
    L.append("# GNN DDI 论文主实验报告")
    L.append("")
    L.append(f"*生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*")
    L.append(f"*mode: {state.get('mode')}*")
    L.append("")
    runs_ok = [r for r in state["runs"].values() if "error" not in r]
    runs_err = [r for r in state["runs"].values() if "error" in r]
    L.append(f"**完成**: {len(runs_ok)} 次 | **失败**: {len(runs_err)} 次")
    L.append("")
    order = _method_order_from_agg(agg)
    L.append("## 一、主结果 (mean ± std)")
    L.append("")
    L.append("| 方法 | Accuracy | Macro-F1 | Weighted-F1 | Macro AUROC | Macro AP | N |")
    L.append("|---|---|---|---|---|---|---|")
    for m in order:
        row = agg[m]
        n = len(row.get("accuracy", []))
        label = METHOD_LABELS_CN.get(m, m)
        if n == 0:
            L.append(f"| {label} | — | — | — | — | — | 0 |")
            continue
        cells = []
        for mk in CORE_METRICS:
            mu, sd = _ms(row.get(mk, []))
            txt = f"{mu:.4f} ± {sd:.4f}"
            if mk == "macro_f1":
                txt = f"**{txt}**"
            cells.append(txt)
        L.append(f"| {label} | " + " | ".join(cells) + f" | {n} |")
    L.append("")
    L.append("## 一b、序数 / 临床指标 (mean ± std)")
    L.append("")
    L.append("| 方法 | adjacent_acc | high_risk_sens | high_risk_spec | severe_underest | QWK | MAE |")
    L.append("|---|---|---|---|---|---|---|")
    for m in order:
        row = agg[m]
        label = METHOD_LABELS_CN.get(m, m)
        if not row.get("adjacent_acc"):
            L.append(f"| {label} | — | — | — | — | — | — |")
            continue
        cells = []
        for mk in ORDINAL_METRICS:
            mu, sd = _ms(row.get(mk, []))
            cells.append(f"{mu:.4f} ± {sd:.4f}")
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("定义: `adjacent_acc`=mean(|pred−true|≤1); "
             "`high_risk_sens`=P(pred≥3|true≥3); "
             "`high_risk_spec`=P(pred<3|true<3); "
             "`severe_underestimation_rate`=P(pred≤1|true≥3); "
             "`qwk`=quadratic weighted Cohen's κ; "
             "`mae_ordinal`=mean(|pred−true|)。")
    L.append("")
    L.append("## 二、每类 F1 (mean ± std)")
    L.append("")
    L.append("| 方法 | 类0 安全 | 类1 一般 | 类2 中度 | 类3 严重 | 类4 危重 |")
    L.append("|---|---|---|---|---|---|")
    for m in order:
        rows = agg[m]["per_class_f1"]
        label = METHOD_LABELS_CN.get(m, m)
        if not rows:
            L.append(f"| {label} | — | — | — | — | — |")
            continue
        arr = np.asarray(rows)
        cells = [f"{arr[:, i].mean():.4f} ± {arr[:, i].std():.4f}" for i in range(min(5, arr.shape[1]))]
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## 三、Exp-3 主候选 vs 其它方法的配对 t-test (Macro-F1)")
    L.append("")
    base_key = "Exp-3" if "Exp-3" in agg else next((k for k in order if k.startswith("Exp-3")), None)
    if base_key and agg[base_key]["runs"]:
        L.append("| 对比 | Exp-3 Macro-F1 (μ±σ) | 其它方法 (μ±σ) | Δ | p-value |")
        L.append("|---|---|---|---|---|")
        runs_by_fs = {(r["fold"], r["seed"], r.get("test_set", "test")): r
                      for r in agg[base_key]["runs"]}
        for m in order:
            if m == base_key:
                continue
            other_by_fs = {(r["fold"], r["seed"], r.get("test_set", "test")): r
                           for r in agg[m]["runs"]}
            common = sorted(set(runs_by_fs) & set(other_by_fs))
            if not common:
                continue
            a = [runs_by_fs[fs]["macro_f1"] for fs in common]
            b = [other_by_fs[fs]["macro_f1"] for fs in common]
            pval = _paired_ttest(a, b)
            dm = np.mean(b) - np.mean(a)
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            L.append(f"| {base_key} vs {METHOD_LABELS_CN.get(m, m)} | "
                     f"{np.mean(a):.4f} ± {np.std(a):.4f} | "
                     f"{np.mean(b):.4f} ± {np.std(b):.4f} | "
                     f"{dm:+.4f} | {pval:.4g} {sig} |")
    L.append("")
    L.append("---")
    L.append("*完整原始指标见 `final_raw_metrics.csv`；可视化图表见 `cn/` 和 `en/`。*")
    with open(P().report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[report] 报告 → {P().report_path}")


def make_all_plots(state: dict, agg: dict, ctx: dict) -> None:
    print("\n[plots] 生成双语图表...")
    order = _method_order_from_agg(agg)
    if not order:
        print("[plots] 无可用方法，跳过")
        return
    metrics_keys = list(CORE_METRICS)
    both_langs(
        plot_method_comparison_bars,
        agg, order, metrics_keys,
        P().out_cn / "methods_comparison.png",
        P().out_en / "methods_comparison.png",
    )
    per_cls_dict = {m: agg[m]["per_class_f1"] for m in order}
    both_langs(
        plot_per_class_f1_comparison,
        per_cls_dict, order,
        P().out_cn / "per_class_f1_comparison.png",
        P().out_en / "per_class_f1_comparison.png",
    )
    for mk in ("macro_f1", "accuracy"):
        both_langs(
            plot_cv_box, agg, order, mk,
            P().out_cn / f"cv_box_{mk}.png",
            P().out_en / f"cv_box_{mk}.png",
        )
        stable_methods = [m for m in order if not str(m).startswith("Exp-4")]
        if stable_methods:
            both_langs(
                plot_cv_box, agg, order, mk,
                P().out_cn / f"cv_box_{mk}_stable.png",
                P().out_en / f"cv_box_{mk}_stable.png",
                include_methods=stable_methods,
                title_suffix=(" (排除 Exp-4, 稳定方法放大)",
                              " (excl. Exp-4, stable methods zoom)"),
            )
            both_langs(
                plot_cv_box_annotated, agg, order, mk,
                P().out_cn / f"cv_box_{mk}_stable_annotated.png",
                P().out_en / f"cv_box_{mk}_stable_annotated.png",
                include_methods=stable_methods,
                title_suffix=(" (排除 Exp-4, 带 μ±σ 标注)",
                              " (excl. Exp-4, with μ±σ labels)"),
            )
            both_langs(
                plot_cv_box_faceted, agg, order, mk,
                P().out_cn / f"cv_box_{mk}_stable_faceted.png",
                P().out_en / f"cv_box_{mk}_stable_faceted.png",
                include_methods=stable_methods,
                title_suffix=(" (排除 Exp-4, 分面板独立纵轴)",
                              " (excl. Exp-4, faceted independent y-axis)"),
            )
    for m in order:
        runs = agg[m]["runs"]
        if not runs:
            continue
        both_langs(
            plot_cv_line, runs,
            P().out_cn / f"cv_line_{m.replace('/', '_')}.png",
            P().out_en / f"cv_line_{m.replace('/', '_')}.png",
        )

    pred_dir = P().pred_dir
    if pred_dir.exists():
        for m in METHODS_ORDER:
            p = pred_dir / f"{m}_f1_s{SEEDS[0]}.npz"
            if not p.exists():
                p = pred_dir / f"{m}_f0_s{SEEDS[0]}_S1.npz"
            if not p.exists():
                continue
            with np.load(p) as nz:
                files = set(nz.files)
                y_test = (nz["y_true"] if "y_true" in files else nz["y_test"]).astype(int)
                y_pred = nz["y_pred"].astype(int)
                y_prob = nz["y_prob"].astype(np.float32)
            cm_title_key = METHOD_LABELS_CN[m]
            both_langs(
                plot_confusion_pair, y_test, y_pred, cm_title_key,
                P().out_cn / f"confusion_{m}.png",
                P().out_en / f"confusion_{m}.png",
            )
            both_langs(
                plot_roc_multi, y_test, y_prob, f"{METHOD_LABELS_CN[m]} - ROC",
                P().out_cn / f"roc_{m}.png",
                P().out_en / f"roc_{m}.png",
            )
            both_langs(
                plot_pr_multi, y_test, y_prob, f"{METHOD_LABELS_CN[m]} - PR",
                P().out_cn / f"pr_{m}.png",
                P().out_en / f"pr_{m}.png",
            )

    y = ctx["y"]
    if ctx.get("cv_splits"):
        idx_train_f1 = ctx["cv_splits"][0][1]
        idx_val_f1 = ctx["cv_splits"][0][2]
        y_train = y[np.sort(idx_train_f1)]
        y_val = y[np.sort(idx_val_f1)]
        y_test = ctx.get("y_test_fixed", y[np.sort(ctx["idx_test"])])
        both_langs(
            plot_class_distribution, y_train, y_val, y_test,
            P().out_cn / "class_distribution.png",
            P().out_en / "class_distribution.png",
        )
    print("[plots] 完成。")


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="GNN DDI 论文主实验")
    parser.add_argument("--data_dir", type=str, default=str(PROCESSED_DIR),
                        help="含 X_full.dat/y.npy/pairs.csv/meta.json 的目录（默认 v1 processed）")
    parser.add_argument("--out_dir", type=str, default=str(RESULTS_DIR / "final"),
                        help="结果输出目录（ckpt/predictions/state）")
    parser.add_argument("--smoke", action="store_true",
                        help="冒烟：分层 1% 样本，GNN 2 epoch，XGBoost 50 树")
    parser.add_argument("--skip_methods", type=str, nargs="*", default=[],
                        help="跳过的方法，例如 --skip_methods Exp-1 Exp-4")
    parser.add_argument("--methods", type=str, nargs="*", default=None,
                        help="只跑这些方法；drug 模式默认 Exp-1 Exp-3 Exp-5")
    parser.add_argument("--n_folds", type=int, default=N_FOLDS,
                        help="pair 模式折数（1 表示 train+val 上一次 80/20）")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="pair 模式训练种子，默认 42 123 2024")
    parser.add_argument("--split_mode", type=str, default="pair",
                        choices=["pair", "drug"],
                        help="pair=现有对级划分；drug=冷启动药物划分")
    parser.add_argument("--drug_split_seed", type=int, nargs="*", default=None,
                        help="drug 模式划分种子（可多个），默认 42 123 2024；模型种子=划分种子")
    parser.add_argument("--final_full", action="store_true",
                        help="在 80%% 开发集上训练 Exp-3/5/1 并写出 out_dir/final_full/")
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑（自动从 final_state.json 恢复）")
    parser.add_argument("--force_reset", action="store_true",
                        help="mode 不一致时清空 state（危险）")
    parser.add_argument("--report_only", action="store_true",
                        help="只生成报告，不跑新实验")
    return parser.parse_args(argv)


def resolve_methods(args) -> list[str]:
    if args.methods:
        methods = list(args.methods)
    elif args.split_mode == "drug" and not args.final_full:
        methods = list(DRUG_DEFAULT_METHODS)
    else:
        methods = list(METHODS_ORDER)
    skip = set(args.skip_methods or [])
    invalid = (set(methods) | skip) - set(METHODS_ORDER)
    if invalid:
        raise SystemExit(f"无效方法: {invalid}，合法值 {METHODS_ORDER}")
    return [m for m in METHODS_ORDER if m in methods and m not in skip]


def main(argv=None):
    args = parse_args(argv)
    configure_paths(args.data_dir, args.out_dir)
    methods = resolve_methods(args)
    seeds = list(args.seeds) if args.seeds else list(SEEDS)
    drug_seeds = list(args.drug_split_seed) if args.drug_split_seed else list(DRUG_SPLIT_SEEDS_DEFAULT)
    n_folds = int(args.n_folds)

    t_start = time.time()
    if not has_cjk_font():
        print("[pipeline_final] 未检测到 CJK 字体，cn/ 图表可能出现豆腐块。")

    if args.report_only:
        state = load_state()
        agg = aggregate_by_method(state)
        save_raw_csv(state)
        generate_report(state, agg)
        try:
            ctx = prepare_global_data(smoke=False, n_folds=n_folds)
            make_all_plots(state, agg, ctx)
        except Exception as e:
            print(f"[plots] 跳过可视化: {e}")
        return

    ctx = prepare_global_data(smoke=args.smoke, n_folds=n_folds)
    ctx["split_mode"] = args.split_mode

    if args.final_full:
        run_final_full(ctx, args.smoke)
        print(f"\n  final_full 完成，耗时 {(time.time()-t_start)/60:.1f} min")
        print(f"  包目录: {P().final_full_dir}")
        return

    if args.split_mode == "drug":
        state = main_loop_drug(ctx, args.smoke, methods, drug_seeds,
                               force_reset=args.force_reset)
    else:
        state = main_loop_pair(ctx, args.smoke, methods, n_folds, seeds,
                               force_reset=args.force_reset)

    agg = aggregate_by_method(state)
    save_raw_csv(state)
    generate_report(state, agg)
    try:
        make_all_plots(state, agg, ctx)
    except Exception as e:  # noqa: BLE001
        print(f"[plots] 可视化出错（已保存报告，不影响数据）: {e}")
        traceback.print_exc()

    dur_h = (time.time() - t_start) / 3600
    print(f"\n{'█' * 72}")
    print(f"  全部完成！总耗时: {dur_h:.1f} 小时")
    print(f"  报告: {P().report_path}")
    print(f"  图表: {P().out_cn}  |  {P().out_en}")
    print(f"{'█' * 72}")


if __name__ == "__main__":
    main()
