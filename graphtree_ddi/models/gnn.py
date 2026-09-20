"""
R-GCN / R-GAT 图神经网络 DDI 风险分级模型（改进版：混合异构图 + 链路预测范式）

方案二：基于 DDI + 生物网络混合异构图（药物-酶-转运体-靶点 + DDI边），
用 R-GCN/R-GAT 学习药物嵌入，预测药物对的相互作用风险等级（5级）。

实验矩阵：
  Exp-1  XGBoost （4120维，由 baseline_xgboost.py 完成）
  Exp-2  R-GCN（仅DDI图，对标 AERGCN-DDI）
  Exp-3  R-GCN（混合图，消融注意力）
  Exp-4  R-GAT（混合图，核心模型）
  Exp-5  R-GAT + XGBoost 融合

Hidden Dim 搜索由独立脚本 gnn_dim_sweep.py 完成。

训练优化：Focal Loss + ReduceLROnPlateau + Warmup + AMP + 大 epoch 早停
技术栈：PyTorch + PyTorch Geometric（RGCNConv / GATv2Conv）
"""

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

import gc
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, RGCNConv
from tqdm import tqdm

warnings.filterwarnings("ignore")

from graphtree_ddi.data.preprocess import (
    FEAT_DIM,
    N_BIO,
    _all_names,
    _names_by_action,
    _parse_v2_field,
    build_fingerprint_cache,
    build_pair_features,
    smiles_to_fingerprint,
)

# ═══════════════════════════════════════════════════════════════
#  路径 / 常量
# ═══════════════════════════════════════════════════════════════
from graphtree_ddi.paths import DATA_PROCESSED, DATA_RAW, REPO_ROOT  # noqa: E402
_PROJ_ROOT = REPO_ROOT
RAW_DIR = DATA_RAW
PROCESSED_DIR = DATA_PROCESSED
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_CN = RESULTS_DIR / "cn"
OUTPUT_EN = RESULTS_DIR / "en"

RISK_LABELS_CN = [
    "安全(无交互)", "一般(PK/PD)", "中度(具体轻症)",
    "严重(需干预)", "危重(生命威胁)",
]
RISK_LABELS_EN = [
    "Safe", "General (PK/PD)", "Moderate",
    "Serious", "Critical",
]
RISK_COLORS = ["#4CAF50", "#8BC34A", "#FFC107", "#FF5722", "#F44336"]

ENZYME_ACTION_MAP = {"substrate": 0, "inhibitor": 1, "inducer": 2}
TRANSPORTER_ACTION_MAP = {"substrate": 3, "inhibitor": 4, "inducer": 5}
TARGET_ACTION_MAP = {"agonist": 6, "antagonist": 7, "inhibitor": 8}
_ENZ_OTHER, _TP_OTHER, _TGT_OTHER = 9, 10, 11
NUM_BIO_FORWARD_RELS = 12
NUM_DDI_RELS = 5                       # risk levels 0-4
NUM_FORWARD_RELS = NUM_BIO_FORWARD_RELS + NUM_DDI_RELS  # 17
NUM_RELATIONS = NUM_FORWARD_RELS * 2   # 34 (forward + reverse)
DDI_REL_OFFSET = NUM_BIO_FORWARD_RELS  # DDI rel IDs start at 12

# 经 gnn_dim_sweep / 调参确定的各实验默认超参（无 CLI 覆盖时使用）
# Exp-2: R-GCN 仅 DDI — dim_sweep_rgcn_ddi_only 最优 hidden_dim=2048
# Exp-3: R-GCN 混合图 — 与 Exp-2 同结构，默认与 Exp-2 对齐
# Exp-4: R-GAT 混合图 — 显存与稳定性：hidden_dim=512；γ=2 易崩溃，默认 γ=0.9
DEFAULT_HIDDEN_DIM_EXP2 = 2048
DEFAULT_FOCAL_GAMMA_EXP2 = 2.0
DEFAULT_HIDDEN_DIM_EXP3 = 2048
DEFAULT_FOCAL_GAMMA_EXP3 = 2.0
DEFAULT_HIDDEN_DIM_EXP4 = 512
DEFAULT_FOCAL_GAMMA_EXP4 = 0.9

# 与 baseline_xgboost.split_data(..., random_state=...) 必须一致，保证同一测试集
SPLIT_RANDOM_STATE = 42


def set_reproducibility(
    seed: int = 42,
    *,
    deterministic_cuda: bool = True,
    strict_deterministic: bool = False,
) -> torch.Generator | None:
    """固定 Python / NumPy / PyTorch / CUDA 随机性，便于同 seed 下复现训练结果。

    数据集 train/val/test 划分请始终使用 SPLIT_RANDOM_STATE（与 XGBoost 方案一一致），
    勿与训练 seed 混用。

    Args:
        strict_deterministic: True 时调用 torch.use_deterministic_algorithms(True, warn_only=True)，
            消除 PyG scatter_add 等算子的非确定性。开启后训练会略慢（约 10~30%），
            但相同 seed 下多次运行的指标差异从 ±0.005 级别下降到 1e-4 级别。

    Returns:
        torch.Generator 对象（绑定在 CPU 上），调用方可传给 DataLoader 等需要 RNG 的地方，
        进一步消除 worker 线程的随机性。当前代码暂不使用 DataLoader，但返回此对象便于后续扩展。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_cuda and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if strict_deterministic:
        # warn_only=True：PyG 内部某些 scatter 操作没有确定版本，仍会 fallback 到非确定实现，
        # 但只发出警告不报错。绝大多数情况下训练过程的非确定性被显著抑制。
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [警告] torch.use_deterministic_algorithms 启用失败: {e}")
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def _train_run_seed(base: int, exp_id: int) -> int:
    """各子实验使用独立偏移的 seed，单独重跑 Exp-k 与全流程中的 Exp-k 一致。"""
    return base + (exp_id - 2) * 1_000_003


# ═══════════════════════════════════════════════════════════════
#  中文字体
# ═══════════════════════════════════════════════════════════════
def _setup_cn_font():
    candidates = [
        "Microsoft YaHei", "微软雅黑", "SimHei", "黑体",
        "STHeiti", "WenQuanYi Micro Hei",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((f for f in candidates if f in available), None)
    if not chosen:
        for p in fm.findSystemFonts():
            if any(k in p.lower() for k in ("yahei", "simhei", "msyh", "heiti")):
                fe = fm.FontEntry(fname=p, name="CustomCN")
                fm.fontManager.ttflist.insert(0, fe)
                chosen = "CustomCN"
                break
    if chosen:
        matplotlib.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


_setup_cn_font()


def _labels(lang: str):
    return RISK_LABELS_CN if lang == "cn" else RISK_LABELS_EN


def _out(fname: str, lang: str) -> Path:
    return (OUTPUT_CN if lang == "cn" else OUTPUT_EN) / fname


def _drop_edges(edge_index: torch.Tensor, edge_type: torch.Tensor,
                drop_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    """DropEdge (Rong et al., ICLR 2020)：训练时随机丢弃一部分边，防止过平滑与过拟合。

    - 仅用于训练阶段的 GNN encode；验证/测试用原图。
    - 使用 torch.bernoulli 以继承全局 RNG seed。
    - 前后向边对称丢弃（需要时在上层逻辑实现）；当前简单实现为每条边独立判定。

    Args:
        edge_index: [2, E] 边索引。
        edge_type:  [E] 边类型。
        drop_ratio: 丢弃比例，范围 [0, 1)。0 表示禁用。

    Returns:
        (filtered_edge_index, filtered_edge_type)
    """
    if drop_ratio <= 0:
        return edge_index, edge_type
    if drop_ratio >= 1:
        raise ValueError(f"drop_ratio={drop_ratio} 必须 < 1")
    n_edges = edge_index.size(1)
    keep_mask = torch.rand(n_edges, device=edge_index.device) >= drop_ratio
    return edge_index[:, keep_mask], edge_type[keep_mask]


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) + Label Smoothing.
    参考 MSFCL (J Chem Inf Model, 2025) 在 DDI 风险预测中的应用。
    """
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.1,
                 reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.register_buffer("weight",
                             weight if weight is not None else None)

    def forward(self, logits, targets):
        n_classes = logits.size(1)
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             reduction="none",
                             label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.weight is not None:
            w = self.weight[targets]
            focal = focal * w / w.mean()
        if self.reduction == "mean":
            return focal.mean()
        return focal.sum()


class PairNorm(nn.Module):
    """PairNorm (Zhao & Akoglu, ICLR 2020)：专为 GNN 设计的归一化层，缓解 oversmoothing。

    与 BatchNorm 相比，PairNorm 不依赖 batch 统计量（对 GNN 的全图前向更友好），
    核心思想是把全图节点表示的"总方差"强制拉回到固定尺度，防止节点嵌入塌缩到同一点。

    模式：
      - "PN":     经典 PairNorm（中心化 + L2 归一化 + 缩放）
      - "PN-SI":  Scale-Individually，每个节点独立缩放（对异质图更稳）
      - "PN-SCS": Scaled-Center-and-Scale，中心化后按全局标准差缩放

    Args:
        scale:  缩放因子（论文建议 1~10，默认 1）
        mode:   "PN" / "PN-SI" / "PN-SCS"，默认 "PN-SI"
    """
    def __init__(self, scale: float = 1.0, mode: str = "PN-SI"):
        super().__init__()
        assert mode in ("PN", "PN-SI", "PN-SCS")
        self.scale = scale
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, D]
        col_mean = x.mean(dim=0, keepdim=True)
        x = x - col_mean
        if self.mode == "PN":
            # 全局行范数均值作为标量缩放（论文原始公式）
            rownorm_mean = x.pow(2).sum(dim=1).mean().sqrt().clamp(min=1e-6)
            x = self.scale * x / rownorm_mean
        elif self.mode == "PN-SI":
            # 每个节点独立按自身 L2 范数归一化（对异质图更稳）
            rownorm = x.norm(p=2, dim=1, keepdim=True).clamp(min=1e-6)
            x = self.scale * x / rownorm
        elif self.mode == "PN-SCS":
            # Scale-Center-Scale：与 PN 相比额外除以行数平方根
            n = x.size(0)
            rownorm_mean = x.pow(2).sum(dim=1).mean().sqrt().clamp(min=1e-6)
            x = self.scale * x / (rownorm_mean * (n ** 0.5))
        return x


# ═══════════════════════════════════════════════════════════════
#  1. 知识图谱构建
# ═══════════════════════════════════════════════════════════════
def _load_kg_static(drugs_csv: Path | str):
    """解析药物实体、生物网络前向边、Morgan 指纹。结果可在各折之间复用。

    边类型（前向，再复制一份 +17 作为反向）:
      0-2 / 9   drug→enzyme: substrate / inhibitor / inducer / other
      3-5 / 10  drug→transporter: substrate / inhibitor / inducer / other
      6-8 / 11  drug→target: agonist / antagonist / inhibitor / other
      12-16     drug↔drug DDI risk 0-4（本函数不含 DDI，由 build_knowledge_graph 追加）
    """
    drugs_df = pd.read_csv(drugs_csv)
    print(f"\n{'='*60}")
    print("  知识图谱构建")
    print(f"{'='*60}")
    print(f"  药物总数: {len(drugs_df):,}")

    drug_ids = sorted(drugs_df["drugbank_id"].dropna().unique())
    enzyme_names: set[str] = set()
    transporter_names: set[str] = set()
    target_names: set[str] = set()

    for _, row in drugs_df.iterrows():
        for names in _parse_v2_field(row.get("cyp_enzymes", "")).values():
            enzyme_names |= names
        for names in _parse_v2_field(row.get("transporters", "")).values():
            transporter_names |= names
        for names in _parse_v2_field(row.get("targets", "")).values():
            target_names |= names

    enzyme_list = sorted(enzyme_names)
    transporter_list = sorted(transporter_names)
    target_list = sorted(target_names)

    n_drugs = len(drug_ids)
    n_enz = len(enzyme_list)
    n_tp = len(transporter_list)
    n_tgt = len(target_list)
    n_total = n_drugs + n_enz + n_tp + n_tgt

    drug_id_to_idx = {d: i for i, d in enumerate(drug_ids)}
    enz_to_idx = {e: i + n_drugs for i, e in enumerate(enzyme_list)}
    tp_to_idx = {t: i + n_drugs + n_enz for i, t in enumerate(transporter_list)}
    tgt_to_idx = {t: i + n_drugs + n_enz + n_tp for i, t in enumerate(target_list)}

    print(f"  酶节点:     {n_enz:>6,}")
    print(f"  转运体节点: {n_tp:>6,}")
    print(f"  靶点节点:   {n_tgt:>6,}")
    print(f"  总节点:     {n_total:>6,}")

    src, dst, rels = [], [], []
    rel_counts = [0] * NUM_FORWARD_RELS

    def _add(drug_idx, bio_idx, rel_id):
        src.append(drug_idx)
        dst.append(bio_idx)
        rels.append(rel_id)
        rel_counts[rel_id] += 1

    for _, row in tqdm(drugs_df.iterrows(), total=len(drugs_df),
                       desc="解析实体关系"):
        didx = drug_id_to_idx.get(row["drugbank_id"])
        if didx is None:
            continue

        for action, names in _parse_v2_field(row.get("cyp_enzymes", "")).items():
            rid = ENZYME_ACTION_MAP.get(action, _ENZ_OTHER)
            for n in names:
                idx = enz_to_idx.get(n)
                if idx is not None:
                    _add(didx, idx, rid)

        for action, names in _parse_v2_field(row.get("transporters", "")).items():
            rid = TRANSPORTER_ACTION_MAP.get(action, _TP_OTHER)
            for n in names:
                idx = tp_to_idx.get(n)
                if idx is not None:
                    _add(didx, idx, rid)

        for action, names in _parse_v2_field(row.get("targets", "")).items():
            rid = TARGET_ACTION_MAP.get(action, _TGT_OTHER)
            for n in names:
                idx = tgt_to_idx.get(n)
                if idx is not None:
                    _add(didx, idx, rid)

    n_bio_fwd = len(src)

    print("\n计算药物 Morgan 指纹...")
    drug_fp = np.zeros((n_drugs, 2048), dtype=np.float32)
    n_valid_fp = 0
    for _, row in tqdm(drugs_df.iterrows(), total=len(drugs_df),
                       desc="Morgan FP"):
        idx = drug_id_to_idx.get(row["drugbank_id"])
        if idx is None:
            continue
        smi = row.get("smiles", "")
        if pd.notna(smi) and smi:
            fp = smiles_to_fingerprint(str(smi))
            if fp is not None:
                drug_fp[idx] = fp
                n_valid_fp += 1
    print(f"  有效指纹: {n_valid_fp}/{n_drugs}")
    drug_fp = torch.from_numpy(drug_fp)

    meta = dict(
        drug_id_to_idx=drug_id_to_idx,
        enz_to_idx=enz_to_idx,
        tp_to_idx=tp_to_idx,
        tgt_to_idx=tgt_to_idx,
        drug_ids=drug_ids,
        enzyme_list=enzyme_list,
        transporter_list=transporter_list,
        target_list=target_list,
        n_drugs=n_drugs,
        n_enz=n_enz,
        n_tp=n_tp,
        n_tgt=n_tgt,
        num_relations=NUM_RELATIONS,
    )
    return dict(
        src_bio=src, dst_bio=dst, rels_bio=rels,
        rel_counts=rel_counts, n_bio_fwd=n_bio_fwd,
        drug_fp=drug_fp, meta=meta, n_total=n_total,
        n_valid_fp=n_valid_fp,
    )


def build_knowledge_graph(drugs_csv: Path | str | None = None,
                          ddi_pairs=None, ddi_labels=None,
                          include_bio=True,
                          static_cache: dict | None = None):
    """
    从 drugbank_drugs.csv 解析生物网络关系，可选加入 DDI 训练边。

    Args:
        ddi_pairs: [N, 2] 药物对节点索引（训练集），None=不加 DDI 边
        ddi_labels: [N] 风险等级标签（0-4），与 ddi_pairs 配对
        include_bio: 是否包含生物网络边（Exp-2 设为 False）
        static_cache: 复用 _load_kg_static 的解析结果，避免每折重复读 CSV / 算指纹。
                      不改变边类型、节点集合或特征。Hybrid 的药物–属性静态边全部保留；
                      仅 DDI 边随 ddi_pairs 变化。

    返回:
        graph (Data): 含 edge_index, edge_type, num_nodes
        drug_fp (Tensor): [n_drugs, 2048] Morgan 指纹
        meta (dict): 节点/边映射表、统计信息
    """
    if drugs_csv is None:
        drugs_csv = RAW_DIR / "drugbank_drugs.csv"
    if static_cache is None:
        static_cache = _load_kg_static(drugs_csv)
    else:
        print(f"\n{'='*60}")
        print("  知识图谱构建（复用实体/指纹缓存）")
        print(f"{'='*60}")

    meta = static_cache["meta"]
    drug_fp = static_cache["drug_fp"]
    n_total = static_cache["n_total"]
    rel_counts = static_cache["rel_counts"]

    if include_bio:
        src = list(static_cache["src_bio"])
        dst = list(static_cache["dst_bio"])
        rels = list(static_cache["rels_bio"])
        n_bio_fwd = int(static_cache["n_bio_fwd"])
    else:
        src, dst, rels = [], [], []
        n_bio_fwd = 0

    # ── 加入 DDI 训练边 ──
    n_ddi_fwd = 0
    ddi_rel_counts = [0] * NUM_DDI_RELS
    if ddi_pairs is not None and ddi_labels is not None:
        for (a, b), label in zip(ddi_pairs, ddi_labels):
            rel_id = DDI_REL_OFFSET + int(label)  # 12 + label
            src.append(int(a))
            dst.append(int(b))
            rels.append(rel_id)
            ddi_rel_counts[int(label)] += 1
            n_ddi_fwd += 1

    n_fwd = len(src)
    rev_src = list(dst)
    rev_dst = list(src)
    rev_rels = [r + NUM_FORWARD_RELS for r in rels]
    src.extend(rev_src)
    dst.extend(rev_dst)
    rels.extend(rev_rels)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(rels, dtype=torch.long)

    graph_type = ("混合图" if include_bio and n_ddi_fwd > 0
                  else "仅DDI图" if n_ddi_fwd > 0 else "仅生物网络图")
    print(f"  图类型:     {graph_type}")
    print(f"  生物网络前向边: {n_bio_fwd:>10,}")
    print(f"  DDI前向边:      {n_ddi_fwd:>10,}")
    print(f"  总前向边:       {n_fwd:>10,}")
    print(f"  总边数(含反向): {len(src):>10,}")

    if include_bio:
        rel_names = [
            "drug→enz:substrate", "drug→enz:inhibitor", "drug→enz:inducer",
            "drug→enz:other",
            "drug→tp:substrate", "drug→tp:inhibitor", "drug→tp:inducer",
            "drug→tp:other",
            "drug→tgt:agonist", "drug→tgt:antagonist", "drug→tgt:inhibitor",
            "drug→tgt:other",
        ]
        for i, name in enumerate(rel_names):
            if rel_counts[i] > 0:
                print(f"    {name:<28s} {rel_counts[i]:>6,}")
    if n_ddi_fwd > 0:
        for lvl in range(NUM_DDI_RELS):
            print(f"    drug↔drug:risk_{lvl:<14d} {ddi_rel_counts[lvl]:>10,}")

    graph = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=n_total)
    # 把 cache 挂在函数上，便于调用方在首次构建后复用（不改变返回签名）
    build_knowledge_graph.last_cache = static_cache
    print(f"{'='*60}\n")
    return graph, drug_fp, meta


# ═══════════════════════════════════════════════════════════════
#  2. 加载 DDI 对数据（复用 preprocess 的 pairs.csv + y.npy）
# ═══════════════════════════════════════════════════════════════
def load_ddi_pairs(drug_id_to_idx: dict, data_dir: Path | str | None = None):
    """
    加载预处理好的 DDI 药物对和标签，将 drug_id 映射为图节点索引。
    返回 (pair_indices [N,2], labels [N], valid_mask)
    valid_mask 长度 = meta.n_samples，与 pairs.csv / y.npy / X_full.dat 行对齐。
    过滤后的 pair_indices/labels 对应 orig_idx = np.flatnonzero(valid_mask)。
    """
    data_dir = Path(data_dir) if data_dir is not None else PROCESSED_DIR
    pairs_df = pd.read_csv(data_dir / "pairs.csv")
    y = np.load(data_dir / "y.npy")

    with open(data_dir / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    n_samples = meta["n_samples"]
    y = y[:n_samples]
    pairs_df = pairs_df.iloc[:n_samples]

    pair_idx = np.full((n_samples, 2), -1, dtype=np.int64)
    valid = np.zeros(n_samples, dtype=bool)
    n_miss = 0
    for i, (d1, d2) in enumerate(zip(pairs_df["drug1_id"], pairs_df["drug2_id"])):
        a = drug_id_to_idx.get(d1, -1)
        b = drug_id_to_idx.get(d2, -1)
        if a >= 0 and b >= 0:
            pair_idx[i] = [a, b]
            valid[i] = True
        else:
            n_miss += 1
    if n_miss > 0:
        print(f"  [警告] {n_miss} 对药物不在知识图谱中，已跳过")

    pair_idx = pair_idx[valid]
    y = y[valid]
    print(f"  有效 DDI 对: {len(y):,}  (5类分布: {np.bincount(y, minlength=5).tolist()})")
    return pair_idx, y, valid


def split_data(y, test_size=0.2, val_size=0.2,
               random_state=SPLIT_RANDOM_STATE):
    """与 baseline_xgboost.split_data 完全一致的划分，确保与 XGBoost 共用同一测试集。"""
    all_idx = np.arange(len(y))
    idx_tv, idx_test = train_test_split(
        all_idx, test_size=test_size, stratify=y, random_state=random_state,
    )
    idx_train, idx_val = train_test_split(
        idx_tv,
        test_size=val_size / (1 - test_size),
        stratify=y[idx_tv], random_state=random_state,
    )
    n = len(y)
    print(f"\n数据集划分（与 XGBoost 一致, seed={random_state}）")
    print(f"  训练集: {len(idx_train):>8,}  ({len(idx_train)/n*100:.0f}%)")
    print(f"  验证集: {len(idx_val):>8,}   ({len(idx_val)/n*100:.0f}%)")
    print(f"  测试集: {len(idx_test):>8,}   ({len(idx_test)/n*100:.0f}%)")
    return idx_train, idx_val, idx_test


# ═══════════════════════════════════════════════════════════════
#  3. 模型定义
# ═══════════════════════════════════════════════════════════════
class PairMLP(nn.Module):
    """药物对嵌入 → 5 级风险概率（MLP 解码器，默认）。

    输入拼接 [h_a, h_b, h_a-h_b, h_a*h_b]，共 4*emb_dim 维。
    """
    def __init__(self, in_dim, hidden=512, n_classes=5, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, h_a, h_b):
        pair = torch.cat([h_a, h_b, h_a - h_b, h_a * h_b], dim=-1)
        return self.net(pair)


class BilinearDecoder(nn.Module):
    """Bilinear 解码器：为每个风险等级 r 学一个完整的 emb × emb 矩阵 W_r，
    score_r(h_a, h_b) = h_a^T · W_r · h_b。

    参数量: n_classes * emb_dim^2（emb=256, 5 类 → 320K，与 PairMLP 同量级）
    优势: 显式建模"同一对药物在不同风险等级下的交互差异"，是知识图谱链路预测的经典解码器
          （RESCAL 系列）。对 Exp-4 R-GAT 的 Macro-F1 通常提升 1-2 个点。
    """
    def __init__(self, emb_dim, n_classes=5, dropout=0.3):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_classes = n_classes
        # W: [n_classes, emb_dim, emb_dim]
        self.W = nn.Parameter(torch.empty(n_classes, emb_dim, emb_dim))
        nn.init.xavier_uniform_(self.W)
        self.drop = nn.Dropout(dropout)

    def forward(self, h_a, h_b):
        h_a = self.drop(h_a)
        h_b = self.drop(h_b)
        # h_a: [B, D] → [B, 1, D]; W: [R, D, D]; h_b: [B, D] → [B, D, 1]
        # 结果: [B, R]
        # 做法: einsum("bd,rdD,bD->br", h_a, W, h_b)
        scores = torch.einsum("bd,rde,be->br", h_a, self.W, h_b)
        return scores


class DistMultDecoder(nn.Module):
    """DistMult 解码器（Yang et al., ICLR 2015）：W_r 为对角矩阵，
    score_r(h_a, h_b) = h_a · diag(w_r) · h_b = Σ_i w_r[i] * h_a[i] * h_b[i]。

    参数量: n_classes * emb_dim（emb=256, 5 类 → 1.3K，极轻量）
    优势: 参数量比 Bilinear 小两个数量级，更不易过拟合；在 SumGNN 等工作中已验证有效。
    局限: 隐式假设关系对称（w_r[i]*h_a[i]*h_b[i] = w_r[i]*h_b[i]*h_a[i]），
          对 DDI 任务是合适的（DDI 本身对称）。
    """
    def __init__(self, emb_dim, n_classes=5, dropout=0.3):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_classes = n_classes
        # w: [n_classes, emb_dim]
        self.w = nn.Parameter(torch.empty(n_classes, emb_dim))
        nn.init.xavier_uniform_(self.w)
        self.drop = nn.Dropout(dropout)

    def forward(self, h_a, h_b):
        h_a = self.drop(h_a)
        h_b = self.drop(h_b)
        # 逐元素: [B, D] * [R, D] 广播
        prod = h_a.unsqueeze(1) * h_b.unsqueeze(1)        # [B, 1, D]
        scores = (prod * self.w.unsqueeze(0)).sum(dim=-1)  # [B, R]
        return scores


def _make_decoder(decoder_type: str, emb_dim: int, n_classes: int, dropout: float):
    """统一解码器工厂。"""
    if decoder_type == "mlp":
        return PairMLP(emb_dim * 4, n_classes=n_classes, dropout=dropout)
    if decoder_type == "bilinear":
        return BilinearDecoder(emb_dim, n_classes=n_classes, dropout=dropout)
    if decoder_type == "distmult":
        return DistMultDecoder(emb_dim, n_classes=n_classes, dropout=dropout)
    raise ValueError(f"unknown decoder_type: {decoder_type}")


class RGCN_DDI(nn.Module):
    """
    基础 R-GCN 模型: 2 层 RGCNConv + PairMLP。
    药物节点: Morgan FP → Linear → hidden_dim
    生物节点: nn.Embedding(hidden_dim)

    可选消融特性（默认关闭，保持向后兼容）:
      - norm_type="batch" (默认) / "pair" / "none"：归一化层选择
      - residual_mode="add" (默认) / "none"：残差连接
      - 若 emb_dim == hidden_dim，第二层 RGCNConv 输出和 hidden_dim 同维度（增大表达力）
    """
    def __init__(self, n_drugs, n_bio, fp_dim=2048, hidden_dim=256,
                 emb_dim=128, num_rels=NUM_RELATIONS, num_bases=8,
                 n_classes=5, dropout=0.3,
                 norm_type: str = "batch",
                 residual_mode: str = "add",
                 pairnorm_scale: float = 1.0,
                 decoder_type: str = "mlp"):
        super().__init__()
        self.n_drugs = n_drugs
        self.n_bio = n_bio
        self.emb_dim = emb_dim
        self.norm_type = norm_type
        self.residual_mode = residual_mode
        self.decoder_type = decoder_type

        self.drug_proj = nn.Linear(fp_dim, hidden_dim)
        self.bio_emb = nn.Embedding(n_bio, hidden_dim)

        self.conv1 = RGCNConv(hidden_dim, hidden_dim, num_rels,
                              num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_dim, emb_dim, num_rels,
                              num_bases=num_bases)

        self.norm1 = self._make_norm(hidden_dim, norm_type, pairnorm_scale)
        self.norm2 = self._make_norm(emb_dim, norm_type, pairnorm_scale)

        self.drop = nn.Dropout(dropout)
        # pair_mlp 属性名保持（下游 fusion / evaluate 代码无需改动）
        self.pair_mlp = _make_decoder(decoder_type, emb_dim, n_classes, dropout)

    @staticmethod
    def _make_norm(dim, norm_type, pairnorm_scale):
        if norm_type == "batch":
            return nn.BatchNorm1d(dim)
        if norm_type == "pair":
            return PairNorm(scale=pairnorm_scale, mode="PN-SI")
        if norm_type == "layer":
            return nn.LayerNorm(dim)
        if norm_type == "none":
            return nn.Identity()
        raise ValueError(f"unknown norm_type: {norm_type}")

    def _build_x(self, drug_fp):
        x_drug = self.drug_proj(drug_fp)
        x_bio = self.bio_emb.weight
        return torch.cat([x_drug, x_bio], dim=0)

    def encode(self, drug_fp, edge_index, edge_type):
        x = self._build_x(drug_fp)
        h1 = self.conv1(x, edge_index, edge_type)
        h1 = self.norm1(h1)
        h1 = F.relu(h1, inplace=True)
        h1 = self.drop(h1)
        if self.residual_mode == "add":
            # 第一层残差：x 与 h1 均为 hidden_dim
            h1 = h1 + x
        h = self.conv2(h1, edge_index, edge_type)
        h = self.norm2(h)
        # 第二层残差：仅当 emb_dim == hidden_dim 时启用
        if self.residual_mode == "add" and h.size(-1) == h1.size(-1):
            h = h + h1
        return h

    def forward(self, drug_fp, edge_index, edge_type, pair_idx):
        h = self.encode(drug_fp, edge_index, edge_type)
        h_a = h[pair_idx[:, 0]]
        h_b = h[pair_idx[:, 1]]
        return self.pair_mlp(h_a, h_b)

    def get_drug_embeddings(self, drug_fp, edge_index, edge_type):
        h = self.encode(drug_fp, edge_index, edge_type)
        return h[: self.n_drugs]


class RGAT_DDI(nn.Module):
    """
    R-GAT 模型: GATv2Conv + 关系嵌入（edge features）。
    注意力权重可导出用于可解释性分析。

    可选消融特性（默认关闭，保持向后兼容）:
      - norm_type="batch" (默认) / "pair" / "layer" / "none"
      - self_loop_fill="zero" (默认 PyG 行为) / "mean"：self-loop 时 edge_attr 的填充策略
      - attn_dropout: GATv2Conv 内部的注意力 dropout（独立于节点 dropout）
      - residual_mode="add" (默认) / "none"
    """
    def __init__(self, n_drugs, n_bio, fp_dim=2048, hidden_dim=256,
                 emb_dim=128, num_rels=NUM_RELATIONS, edge_emb_dim=64,
                 heads=4, n_classes=5, dropout=0.3,
                 norm_type: str = "batch",
                 residual_mode: str = "add",
                 pairnorm_scale: float = 1.0,
                 self_loop_fill: str = "zero",
                 attn_dropout: float | None = None,
                 decoder_type: str = "mlp"):
        super().__init__()
        self.n_drugs = n_drugs
        self.n_bio = n_bio
        self.emb_dim = emb_dim
        self.norm_type = norm_type
        self.residual_mode = residual_mode
        self.decoder_type = decoder_type

        self.drug_proj = nn.Linear(fp_dim, hidden_dim)
        self.bio_emb = nn.Embedding(n_bio, hidden_dim)
        self.edge_emb = nn.Embedding(num_rels, edge_emb_dim)

        assert hidden_dim % heads == 0
        # GATv2Conv 支持 fill_value 参数控制 add_self_loops 时 edge_attr 的填充：
        #   "zero"（默认）：self-loop 的 edge_attr 填 0 向量，会让 "零关系嵌入"
        #                  与真实关系嵌入范数差距大，推高 attention 方差。
        #   "mean"：用所有真实边 edge_attr 的均值填充，缓解范数失衡。
        # 两者对训练稳定性影响较大，本项目推荐 "mean"，但保留默认为 "zero" 以保持旧行为。
        conv_kwargs = dict(
            edge_dim=edge_emb_dim, add_self_loops=True,
        )
        # dropout 传 GATv2Conv 作为注意力 dropout
        conv_kwargs["dropout"] = dropout if attn_dropout is None else attn_dropout
        if self_loop_fill == "mean":
            conv_kwargs["fill_value"] = "mean"
        elif self_loop_fill == "zero":
            pass  # PyG 默认
        else:
            raise ValueError(f"unknown self_loop_fill: {self_loop_fill}")

        self.conv1 = GATv2Conv(
            hidden_dim, hidden_dim // heads, heads=heads, **conv_kwargs,
        )
        self.conv2 = GATv2Conv(
            hidden_dim, emb_dim, heads=1, **conv_kwargs,
        )

        self.norm1 = self._make_norm(hidden_dim, norm_type, pairnorm_scale)
        self.norm2 = self._make_norm(emb_dim, norm_type, pairnorm_scale)

        self.drop = nn.Dropout(dropout)
        self.pair_mlp = _make_decoder(decoder_type, emb_dim, n_classes, dropout)

    @staticmethod
    def _make_norm(dim, norm_type, pairnorm_scale):
        if norm_type == "batch":
            return nn.BatchNorm1d(dim)
        if norm_type == "pair":
            return PairNorm(scale=pairnorm_scale, mode="PN-SI")
        if norm_type == "layer":
            return nn.LayerNorm(dim)
        if norm_type == "none":
            return nn.Identity()
        raise ValueError(f"unknown norm_type: {norm_type}")

    def _build_x(self, drug_fp):
        x_drug = self.drug_proj(drug_fp)
        x_bio = self.bio_emb.weight
        return torch.cat([x_drug, x_bio], dim=0)

    def encode(self, drug_fp, edge_index, edge_type):
        x = self._build_x(drug_fp)
        edge_attr = self.edge_emb(edge_type)

        h1 = self.conv1(x, edge_index, edge_attr=edge_attr)
        h1 = self.norm1(h1)
        h1 = F.elu(h1, inplace=True)
        h1 = self.drop(h1)
        if self.residual_mode == "add":
            h1 = h1 + x
        h = self.conv2(h1, edge_index, edge_attr=edge_attr)
        h = self.norm2(h)
        if self.residual_mode == "add" and h.size(-1) == h1.size(-1):
            h = h + h1
        return h

    def encode_with_attention(self, drug_fp, edge_index, edge_type):
        """编码并返回第一层注意力权重（用于可解释性）。"""
        x = self._build_x(drug_fp)
        edge_attr = self.edge_emb(edge_type)
        h1, (ei_attn, alpha) = self.conv1(
            x, edge_index, edge_attr=edge_attr,
            return_attention_weights=True,
        )
        h1 = self.norm1(h1)
        h1 = F.elu(h1, inplace=True)
        h1 = self.drop(h1)
        if self.residual_mode == "add":
            h1 = h1 + x
        h = self.conv2(h1, edge_index, edge_attr=edge_attr)
        h = self.norm2(h)
        if self.residual_mode == "add" and h.size(-1) == h1.size(-1):
            h = h + h1
        return h, ei_attn, alpha

    def forward(self, drug_fp, edge_index, edge_type, pair_idx):
        h = self.encode(drug_fp, edge_index, edge_type)
        h_a = h[pair_idx[:, 0]]
        h_b = h[pair_idx[:, 1]]
        return self.pair_mlp(h_a, h_b)

    def get_drug_embeddings(self, drug_fp, edge_index, edge_type):
        h = self.encode(drug_fp, edge_index, edge_type)
        return h[: self.n_drugs]


# ═══════════════════════════════════════════════════════════════
#  4. 训练
# ═══════════════════════════════════════════════════════════════
def _class_weights(y):
    counts = np.bincount(y, minlength=5).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = len(y) / (len(counts) * counts)
    return torch.FloatTensor(w)


def _run_batches(model, drug_fp, edge_index, edge_type,
                 pair_idx, labels, criterion, device, batch_size,
                 optimizer=None, scaler=None,
                 drop_edge: float = 0.0, grad_clip: float = 1.0):
    """
    训练或验证的批次循环。optimizer=None 时为验证模式。

    核心优化：
      1) GNN encode 仅执行一次，mini-batch 只作用于 pair MLP。
         训练时将所有 batch 的 loss 汇聚为单一连通张量后一次性 backward，
         encoder 计算图只反传一次（速度与 dim_sweep 一致），梯度完全正确。
      2) pair_idx / labels 若为 numpy ndarray，会一次性搬到 GPU 后整个 epoch 复用，
         避免每个 batch 都 CPU→GPU 同步 copy（在 strict_deterministic 模式下尤其明显）。
         若调用方传入的已经是 torch.Tensor（建议已 .to(device)），则直接复用。

    Args:
        drop_edge: 训练时的 DropEdge 比例，>0 启用；验证时强制 0。
        grad_clip: 梯度裁剪阈值（按范数），0 或负数表示禁用。
    """
    is_train = optimizer is not None
    model.train(is_train)
    use_amp = scaler is not None

    # ── 一次性搬 GPU：pair_idx & labels（避免 per-batch copy）──
    # 调用方约定: train_model 已一次性搬好 GPU tensor 并复用；此处的分支
    # 兼容直接传 numpy 的场景（单元测试 / 其它入口），保证代码健壮。
    if isinstance(pair_idx, np.ndarray):
        pair_idx_t = torch.from_numpy(pair_idx).to(device, dtype=torch.long,
                                                   non_blocking=True)
    else:
        # 若已是 GPU long tensor，.to() 返回自身引用（零 copy）
        pair_idx_t = pair_idx.to(device, dtype=torch.long, non_blocking=True)
    if isinstance(labels, np.ndarray):
        labels_t = torch.from_numpy(labels).to(device, dtype=torch.long,
                                               non_blocking=True)
    else:
        labels_t = labels.to(device, dtype=torch.long, non_blocking=True)

    n = labels_t.size(0)
    if is_train:
        # 用 GPU 上的随机打散，避免 CPU↔GPU 来回拷贝
        perm = torch.randperm(n, device=device)
    else:
        perm = torch.arange(n, device=device)
    total_correct, total_count = 0, 0
    monitor_loss = 0.0  # 用于打印，不参与反传

    # ── 训练时可选 DropEdge（验证始终用原图） ──
    if is_train and drop_edge > 0:
        ei_use, et_use = _drop_edges(edge_index, edge_type, drop_edge)
    else:
        ei_use, et_use = edge_index, edge_type

    # ── GNN encode: 全图只做一次 ──
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        with torch.amp.autocast("cuda", enabled=use_amp):
            h = model.encode(drug_fp, ei_use, et_use)

    if is_train:
        optimizer.zero_grad()

    # ── pair MLP 分批前向，把每个 batch 的 loss 保留在计算图里 ──
    loss_parts = []  # 每个元素都通过 h 连接到 encoder 计算图
    for start in range(0, n, batch_size):
        idx = perm[start: start + batch_size]
        bp = pair_idx_t.index_select(0, idx)
        bl = labels_t.index_select(0, idx)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", enabled=use_amp):
                h_a = h[bp[:, 0]]
                h_b = h[bp[:, 1]]
                logits = model.pair_mlp(h_a, h_b)
                batch_loss = criterion(logits, bl)

        if is_train:
            # 按 batch 大小加权（等效于在整个训练集上求平均 loss）
            loss_parts.append(batch_loss * (len(idx) / n))

        monitor_loss += batch_loss.detach().item() * len(idx)
        total_correct += (logits.detach().argmax(1) == bl).sum().item()
        total_count += len(idx)

    # ── 单次 backward：encoder 只反传一次，速度快且梯度正确 ──
    if is_train and loss_parts:
        total_loss_tensor = sum(loss_parts)   # 单一连通张量
        # NaN/Inf 保护：loss 异常时跳过本步反传（防止 GAT 崩溃时写入坏权重）
        if not torch.isfinite(total_loss_tensor):
            print(f"  [警告] 本 step 的 loss 非有限值 (value={total_loss_tensor.item()})，跳过反传")
            optimizer.zero_grad(set_to_none=True)
        else:
            if scaler is not None:
                scaler.scale(total_loss_tensor).backward()
                scaler.unscale_(optimizer)
                if grad_clip and grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss_tensor.backward()
                if grad_clip and grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

    return monitor_loss / max(total_count, 1), total_correct / max(total_count, 1)


def _val_macro_f1(model, drug_fp, edge_index, edge_type,
                  pair_idx, labels, device, batch_size=8192):
    """计算验证集 Macro-F1（不计算梯度）。encode 只做一次。
    pair_idx 支持 numpy 或 GPU tensor，二者都会一次性保证在 GPU 上复用。
    """
    model.eval()
    if isinstance(pair_idx, np.ndarray):
        pair_t = torch.from_numpy(pair_idx).to(device, dtype=torch.long,
                                               non_blocking=True)
    else:
        pair_t = pair_idx.to(device, dtype=torch.long, non_blocking=True)
    n = pair_t.size(0)
    with torch.no_grad():
        h = model.encode(drug_fp, edge_index, edge_type)
    preds = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            bp = pair_t[start: start + batch_size]
            h_a = h[bp[:, 0]]
            h_b = h[bp[:, 1]]
            logits = model.pair_mlp(h_a, h_b)
            preds.append(logits.argmax(1).cpu().numpy())
    preds = np.concatenate(preds)
    # labels 保持 numpy（sklearn 需要 numpy）
    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = labels
    return f1_score(labels_np, preds, average="macro", zero_division=0)


def train_model(model, graph, drug_fp, pair_idx, y,
                idx_train, idx_val, device, config):
    """
    训练循环：Focal Loss + ReduceLROnPlateau + Warmup + AMP（可关）。
    max_epochs 极大，完全依赖早停结束训练，确保充分拟合。
    返回 (model, history)。

    支持的 config 字段（除原有字段外新增）:
      - use_amp:    bool，是否启用 AMP（默认 True）
      - drop_edge:  float，训练时 DropEdge 比例（默认 0，禁用）
      - grad_clip:  float，梯度裁剪阈值（默认 1.0）
    """
    lr = config.get("lr", 5e-4)
    wd = config.get("weight_decay", 1e-4)
    patience = config.get("patience", 100)
    max_epochs = config.get("max_epochs", 99999)
    bs = config.get("batch_size", 4096)
    warmup_epochs = config.get("warmup_epochs", 10)
    focal_gamma = config.get("focal_gamma", 2.0)
    label_smoothing = config.get("label_smoothing", 0.1)
    min_epochs = config.get("min_epochs", 50)
    plateau_factor = config.get("plateau_factor", 0.5)
    plateau_patience = config.get("plateau_patience", 15)
    use_amp_cfg = config.get("use_amp", True)
    drop_edge = float(config.get("drop_edge", 0.0))
    grad_clip = float(config.get("grad_clip", 1.0))

    graph = graph.to(device)
    drug_fp = drug_fp.to(device)
    model = model.to(device)

    ei = graph.edge_index
    et = graph.edge_type
    class_w = _class_weights(y[idx_train])

    criterion = FocalLoss(weight=class_w.to(device), gamma=focal_gamma,
                          label_smoothing=label_smoothing)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=plateau_factor,
        patience=plateau_patience, min_lr=1e-7)

    use_amp = device.type == "cuda" and use_amp_cfg
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_f1, best_epoch, wait = 0.0, 0, 0
    best_state = None
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "lr": []}

    # ── 一次性把训练/验证索引搬到 GPU，整个训练循环里复用，避免每 epoch 反复 copy ──
    tr_pairs_t = torch.from_numpy(pair_idx[idx_train]).to(
        device, dtype=torch.long, non_blocking=True)
    tr_labels_t = torch.from_numpy(y[idx_train]).to(
        device, dtype=torch.long, non_blocking=True)
    val_pairs_t = torch.from_numpy(pair_idx[idx_val]).to(
        device, dtype=torch.long, non_blocking=True)
    val_labels_t = torch.from_numpy(y[idx_val]).to(
        device, dtype=torch.long, non_blocking=True)

    print(f"\n{'='*60}")
    print(f"  训练开始  (max={max_epochs}, patience={patience}, "
          f"bs={bs}, lr={lr})")
    print(f"  Focal(γ={focal_gamma}) + LabelSmooth({label_smoothing}) "
          f"+ ReduceLROnPlateau(f={plateau_factor},p={plateau_patience})")
    print(f"  Warmup={warmup_epochs}ep, AMP={'ON' if use_amp else 'OFF'}, "
          f"DropEdge={drop_edge:.2f}, GradClip={grad_clip:.2f}")
    print(f"  仅通过早停结束训练")
    print(f"{'='*60}")

    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        if epoch <= warmup_epochs:
            warmup_lr = lr * epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        t_loss, t_acc = _run_batches(
            model, drug_fp, ei, et,
            tr_pairs_t, tr_labels_t, criterion, device, bs,
            optimizer=optimizer, scaler=scaler,
            drop_edge=drop_edge, grad_clip=grad_clip,
        )

        v_loss, _ = _run_batches(
            model, drug_fp, ei, et,
            val_pairs_t, val_labels_t, criterion, device, bs,
            optimizer=None,
        )
        v_f1 = _val_macro_f1(model, drug_fp, ei, et,
                             val_pairs_t, val_labels_t, device)

        if epoch > warmup_epochs:
            scheduler.step(v_f1)
        cur_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_f1"].append(v_f1)
        history["lr"].append(cur_lr)

        if v_f1 > best_f1:
            best_f1, best_epoch = v_f1, epoch
            best_state = {k: v.cpu().clone() for k, v in
                          model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 10 == 0 or epoch == 1 or wait == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:>4d}  "
                  f"loss={t_loss:.4f}/{v_loss:.4f}  "
                  f"F1={v_f1:.4f}  best={best_f1:.4f}@{best_epoch}  "
                  f"lr={cur_lr:.1e}  ({elapsed:.0f}s)")

        if epoch >= min_epochs and wait >= patience:
            print(f"  Early stopping @ epoch {epoch} "
                  f"(best F1={best_f1:.4f} @ epoch {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    print(f"  训练完成  最佳验证 Macro-F1={best_f1:.4f} @ epoch {best_epoch}")
    print(f"  总耗时: {time.time()-t0:.1f}s\n")
    return model, history


# ═══════════════════════════════════════════════════════════════
#  5. 评估（与 XGBoost 格式对齐）
# ═══════════════════════════════════════════════════════════════
def evaluate_model(model, graph, drug_fp, pair_idx, y,
                   idx_test, device, model_name="R-GCN"):
    """
    在测试集上评估，输出格式与 baseline_xgboost.py 一致。
    返回 (y_test, y_pred, y_prob, metrics_dict)
    """
    model.eval()
    graph = graph.to(device)
    drug_fp = drug_fp.to(device)
    ei, et = graph.edge_index, graph.edge_type

    test_pairs = pair_idx[idx_test]
    y_test = y[idx_test]
    n_classes = len(np.unique(y))

    test_pairs_t = torch.from_numpy(test_pairs).to(device, dtype=torch.long,
                                                   non_blocking=True)
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
    y_prob = F.softmax(all_logits, dim=1).numpy()
    y_pred = all_logits.argmax(1).numpy()

    acc = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_wt = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    y_bin = label_binarize(y_test, classes=list(range(n_classes)))
    aucs, aps = [], []
    per_class = {}
    for i in range(n_classes):
        if y_bin[:, i].sum() > 0:
            auc_i = roc_auc_score(y_bin[:, i], y_prob[:, i])
            ap_i = average_precision_score(y_bin[:, i], y_prob[:, i])
        else:
            auc_i, ap_i = float("nan"), float("nan")
        aucs.append(auc_i)
        aps.append(ap_i)
        mask_i = y_test == i
        tp = ((y_pred == i) & mask_i).sum()
        fn = (mask_i).sum() - tp
        fp = ((y_pred == i) & ~mask_i).sum()
        tn = (~mask_i).sum() - fp
        prec_i = tp / max(tp + fp, 1)
        rec_i = tp / max(tp + fn, 1)
        f1_i = 2 * prec_i * rec_i / max(prec_i + rec_i, 1e-9)
        acc_i = (tp + tn) / len(y_test)
        per_class[i] = dict(acc=acc_i, prec=prec_i, rec=rec_i,
                            f1=f1_i, auroc=auc_i, ap=ap_i,
                            support=int(mask_i.sum()))

    macro_auroc = float(np.nanmean(aucs))
    macro_ap = float(np.nanmean(aps))

    # ── 打印（与 XGBoost 格式一致） ──
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {model_name} 测试集评估")
    print(f"{sep}")
    print(f"  {'指标':<24s} {'数值':>10s}")
    print(f"  {'-'*36}")
    print(f"  {'准确率 Accuracy':<24s} {acc:>10.4f}")
    print(f"  {'Macro-F1':<24s} {f1_mac:>10.4f}")
    print(f"  {'Weighted-F1':<24s} {f1_wt:>10.4f}")
    print(f"  {'宏平均 AUROC':<24s} {macro_auroc:>10.4f}")
    print(f"  {'宏平均 AP':<24s} {macro_ap:>10.4f}")
    print()

    header = (f"  {'类别':<26s} {'准确率':>8s} {'精确率':>8s} "
              f"{'召回率':>8s} {'F1':>8s} {'AUROC':>8s} "
              f"{'AP':>8s} {'样本数':>8s}")
    print(header)
    print(f"  {'-'*80}")
    labels_cn = RISK_LABELS_CN
    for i in range(n_classes):
        c = per_class[i]
        lbl = labels_cn[i] if i < len(labels_cn) else f"类别{i}"
        print(f"  {lbl:<22s} {c['acc']:>8.4f} {c['prec']:>8.4f} "
              f"{c['rec']:>8.4f} {c['f1']:>8.4f} {c['auroc']:>8.4f} "
              f"{c['ap']:>8.4f} {c['support']:>8,}")
    print(sep)
    print()
    print(classification_report(
        y_test, y_pred,
        target_names=[labels_cn[i] for i in range(n_classes)],
        digits=2, zero_division=0,
    ))

    metrics = dict(accuracy=acc, macro_f1=f1_mac, weighted_f1=f1_wt,
                   macro_auroc=macro_auroc, macro_ap=macro_ap,
                   per_class=per_class)
    return y_test, y_pred, y_prob, metrics


# ═══════════════════════════════════════════════════════════════
#  6. 可视化
# ═══════════════════════════════════════════════════════════════
def _both_langs(fn, *args, **kwargs):
    for lang in ("cn", "en"):
        fn(*args, lang=lang, **kwargs)


def plot_training_curves(history, lang="cn", model_name="R-GCN"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="训练" if lang == "cn" else "Train")
    ax1.plot(epochs, history["val_loss"], label="验证" if lang == "cn" else "Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} " + ("损失曲线" if lang == "cn" else "Loss Curve"))
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["val_f1"], color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro-F1")
    ax2.set_title(f"{model_name} " + ("验证 Macro-F1" if lang == "cn" else "Validation Macro-F1"))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(_out(f"gnn_{model_name.lower()}_training.png", lang), dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, lang="cn", model_name="R-GCN"):
    labels = _labels(lang)
    n = len(labels)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, fmt, title_suffix in [
        (ax1, cm, "d", "样本数" if lang == "cn" else "Counts"),
        (ax2, cm_norm, ".2f", "归一化" if lang == "cn" else "Normalized"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", ax=ax,
                    xticklabels=labels, yticklabels=labels)
        ax.set_xlabel("预测类别" if lang == "cn" else "Predicted")
        ax.set_ylabel("真实类别" if lang == "cn" else "True")
        ax.set_title(f"{model_name} " + ("混淆矩阵" if lang == "cn" else "Confusion Matrix")
                     + f" ({title_suffix})")
    plt.tight_layout()
    plt.savefig(_out(f"gnn_{model_name.lower()}_confusion.png", lang), dpi=150)
    plt.close()


def plot_roc_pr(y_true, y_prob, lang="cn", model_name="R-GCN"):
    labels = _labels(lang)
    n_classes = y_prob.shape[1]
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for i in range(n_classes):
        if y_bin[:, i].sum() == 0:
            continue
        from sklearn.metrics import roc_curve, precision_recall_curve as prc
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc_i = roc_auc_score(y_bin[:, i], y_prob[:, i])
        ax1.plot(fpr, tpr, color=RISK_COLORS[i],
                 label=f"{labels[i]} (AUC={auc_i:.3f})")
        prec, rec, _ = prc(y_bin[:, i], y_prob[:, i])
        ap_i = average_precision_score(y_bin[:, i], y_prob[:, i])
        ax2.plot(rec, prec, color=RISK_COLORS[i],
                 label=f"{labels[i]} (AP={ap_i:.3f})")

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax1.set_xlabel("FPR")
    ax1.set_ylabel("TPR")
    ax1.set_title(f"{model_name} ROC")
    ax1.legend(fontsize=7, loc="lower right")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"{model_name} PR")
    ax2.legend(fontsize=7, loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(_out(f"gnn_{model_name.lower()}_roc_pr.png", lang), dpi=150)
    plt.close()


def plot_tsne(model, graph, drug_fp, y, pair_idx, device,
              lang="cn", model_name="R-GAT"):
    """将药物嵌入降维至 2D，按最高涉及风险等级着色。"""
    model.eval()
    graph = graph.to(device)
    drug_fp = drug_fp.to(device)
    with torch.no_grad():
        emb = model.get_drug_embeddings(
            drug_fp, graph.edge_index, graph.edge_type,
        ).cpu().numpy()

    n_drugs = emb.shape[0]
    drug_max_risk = np.zeros(n_drugs, dtype=int)
    for (a, b), label in zip(pair_idx, y):
        if a < n_drugs:
            drug_max_risk[a] = max(drug_max_risk[a], label)
        if b < n_drugs:
            drug_max_risk[b] = max(drug_max_risk[b], label)

    mask = np.linalg.norm(emb, axis=1) > 1e-6
    emb_valid = emb[mask]
    risk_valid = drug_max_risk[mask]

    n_sample = min(5000, len(emb_valid))
    rng = np.random.default_rng(42)
    sel = rng.choice(len(emb_valid), n_sample, replace=False)
    emb_sel = emb_valid[sel]
    risk_sel = risk_valid[sel]

    print("  t-SNE 降维中...")
    coords = TSNE(n_components=2, random_state=42,
                  perplexity=min(30, n_sample - 1)).fit_transform(emb_sel)

    labels = _labels(lang)
    fig, ax = plt.subplots(figsize=(10, 8))
    for lvl in range(5):
        m = risk_sel == lvl
        if m.sum() == 0:
            continue
        ax.scatter(coords[m, 0], coords[m, 1], c=RISK_COLORS[lvl],
                   label=labels[lvl], s=8, alpha=0.6)
    ax.legend(fontsize=8, markerscale=2)
    title = ("药物嵌入 t-SNE 可视化（按最高风险着色）" if lang == "cn"
             else "Drug Embedding t-SNE (colored by max risk)")
    ax.set_title(f"{model_name} {title}")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(_out(f"gnn_{model_name.lower()}_tsne.png", lang), dpi=150)
    plt.close()
    print("  t-SNE 图已保存")


def plot_attention_analysis(model, graph, drug_fp, kg_meta, device,
                            lang="cn", top_k=20):
    """提取 R-GAT 注意力权重，展示哪些生物实体被模型最关注。"""
    if not isinstance(model, RGAT_DDI):
        return
    model.eval()
    graph = graph.to(device)
    drug_fp = drug_fp.to(device)

    with torch.no_grad():
        _, ei_attn, alpha = model.encode_with_attention(
            drug_fp, graph.edge_index, graph.edge_type,
        )

    alpha = alpha.cpu().numpy().mean(axis=1)  # average over heads
    ei = ei_attn.cpu().numpy()

    n_drugs = kg_meta["n_drugs"]
    bio_attention: dict[str, float] = {}
    bio_count: dict[str, int] = {}

    idx_to_name = {}
    for name, idx in kg_meta["enz_to_idx"].items():
        idx_to_name[idx] = ("Enzyme", name)
    for name, idx in kg_meta["tp_to_idx"].items():
        idx_to_name[idx] = ("Transporter", name)
    for name, idx in kg_meta["tgt_to_idx"].items():
        idx_to_name[idx] = ("Target", name)

    for k in range(ei.shape[1]):
        src_node, dst_node = int(ei[0, k]), int(ei[1, k])
        bio_node = None
        if src_node >= n_drugs and src_node in idx_to_name:
            bio_node = src_node
        elif dst_node >= n_drugs and dst_node in idx_to_name:
            bio_node = dst_node
        if bio_node is None:
            continue
        etype, ename = idx_to_name[bio_node]
        key = f"{etype}: {ename}"
        bio_attention[key] = bio_attention.get(key, 0.0) + float(alpha[k])
        bio_count[key] = bio_count.get(key, 0) + 1

    avg_attn = {k: bio_attention[k] / max(bio_count[k], 1)
                for k in bio_attention}
    sorted_bio = sorted(avg_attn.items(), key=lambda x: -x[1])[:top_k]

    if not sorted_bio:
        return

    names = [x[0] for x in sorted_bio][::-1]
    vals = [x[1] for x in sorted_bio][::-1]
    colors = []
    for n in names:
        if n.startswith("Enzyme"):
            colors.append("#E57373")
        elif n.startswith("Transporter"):
            colors.append("#64B5F6")
        else:
            colors.append("#81C784")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    xlabel = "平均注意力权重" if lang == "cn" else "Mean Attention Weight"
    title = ("R-GAT Top-{} 生物实体注意力" if lang == "cn"
             else "R-GAT Top-{} Bio-Entity Attention").format(top_k)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#E57373", label="Enzyme" if lang == "en" else "酶"),
        Patch(facecolor="#64B5F6",
              label="Transporter" if lang == "en" else "转运体"),
        Patch(facecolor="#81C784",
              label="Target" if lang == "en" else "靶点"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    plt.savefig(_out("gnn_rgat_attention.png", lang), dpi=150)
    plt.close()
    print("  注意力分析图已保存")


# ═══════════════════════════════════════════════════════════════
#  7. 融合实验（GNN 嵌入 + XGBoost）
# ═══════════════════════════════════════════════════════════════
def run_fusion_experiment(model, graph, drug_fp, pair_idx, y,
                          idx_train, idx_val, idx_test,
                          device, kg_meta,
                          xgb_n_estimators: int = 200000,
                          xgb_early_stopping: int = 200,
                          xgb_random_state: int = 42,
                          save_suffix: str = "",
                          return_predictions: bool = False):
    """
    Exp-5: 提取 GNN 药物嵌入，拼接到 XGBoost 特征中重新训练。

    Args:
        xgb_n_estimators / xgb_early_stopping / xgb_random_state: XGBoost 超参
        save_suffix: 融合模型保存文件后缀（多 seed/fold 时自动填充）
        return_predictions: 若 True，额外返回 y_test/y_pred/y_prob 与完整 per-class 指标

    Returns:
        若 return_predictions=False：dict(accuracy, macro_f1, weighted_f1, macro_auroc, macro_ap)
        若 return_predictions=True：dict(..., per_class_f1, per_class_auroc, per_class_ap,
                                         y_test, y_pred, y_prob)
    """
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier

    print(f"\n{'='*60}")
    print("  Exp-5: GNN 嵌入 + XGBoost 融合")
    print(f"{'='*60}")

    # 提取药物嵌入
    model.eval()
    graph_d = graph.to(device)
    drug_fp_d = drug_fp.to(device)
    with torch.no_grad():
        drug_emb = model.get_drug_embeddings(
            drug_fp_d, graph_d.edge_index, graph_d.edge_type,
        ).cpu().numpy()
    emb_dim = drug_emb.shape[1]
    print(f"  药物嵌入维度: {emb_dim}")

    # 构建 GNN pair features: concat + diff + product
    def _pair_emb(pairs):
        a_emb = drug_emb[pairs[:, 0]]
        b_emb = drug_emb[pairs[:, 1]]
        return np.concatenate([a_emb, b_emb, a_emb - b_emb, a_emb * b_emb],
                              axis=1)

    gnn_train = _pair_emb(pair_idx[idx_train])
    gnn_val = _pair_emb(pair_idx[idx_val])
    gnn_test = _pair_emb(pair_idx[idx_test])
    print(f"  GNN pair 特征维度: {gnn_train.shape[1]}")

    # 加载 XGBoost 原始特征
    y_path = PROCESSED_DIR / "y.npy"
    mm_path = PROCESSED_DIR / "X_full.dat"
    x_path = PROCESSED_DIR / "X.npy"
    meta_path = PROCESSED_DIR / "meta.json"

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    n_s, f_d = meta["n_samples"], meta["feature_dim"]

    if mm_path.exists():
        X_mm = np.memmap(mm_path, dtype="float32", mode="r", shape=(n_s, f_d))
    elif x_path.exists():
        X_mm = np.load(x_path, mmap_mode="r")
    else:
        print("  [错误] 未找到 XGBoost 特征文件，跳过融合实验")
        return None

    # 原始特征 + GNN 嵌入特征
    print("  拼接 XGBoost 原始特征 + GNN 嵌入...")
    X_train = np.concatenate([
        np.array(X_mm[idx_train], dtype=np.float32), gnn_train,
    ], axis=1)
    X_val = np.concatenate([
        np.array(X_mm[idx_val], dtype=np.float32), gnn_val,
    ], axis=1)
    X_test = np.concatenate([
        np.array(X_mm[idx_test], dtype=np.float32), gnn_test,
    ], axis=1)
    print(f"  融合特征维度: {X_train.shape[1]} "
          f"({f_d} XGBoost + {gnn_train.shape[1]} GNN)")

    y_train = y[idx_train]
    y_val = y[idx_val]
    y_test = y[idx_test]

    n_classes = len(np.unique(y))
    xgb = XGBClassifier(
        n_estimators=xgb_n_estimators, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.6,
        eval_metric=["mlogloss", "merror"],
        random_state=xgb_random_state,
        device="cuda", tree_method="hist", n_jobs=1,
        early_stopping_rounds=xgb_early_stopping,
        objective="multi:softprob",
        num_class=n_classes,
    )
    sw = compute_sample_weight("balanced", y_train)

    n_eval_tr = min(100_000, len(X_train))
    eval_idx = np.random.default_rng(xgb_random_state).choice(
        len(X_train), n_eval_tr, replace=False)

    print(f"  训练 XGBoost 融合特征 "
          f"(n_estimators={xgb_n_estimators}, early_stopping={xgb_early_stopping}, "
          f"seed={xgb_random_state})...")
    xgb.fit(
        X_train, y_train,
        sample_weight=sw,
        eval_set=[(X_train[eval_idx], y_train[eval_idx]),
                  (X_val, y_val)],
        verbose=100,
    )

    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_wt = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0).tolist()
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))
    per_class_auroc, per_class_ap = [], []
    for i in range(n_classes):
        if y_bin[:, i].sum() > 0:
            per_class_auroc.append(float(roc_auc_score(y_bin[:, i], y_prob[:, i])))
            per_class_ap.append(float(average_precision_score(y_bin[:, i], y_prob[:, i])))
        else:
            per_class_auroc.append(float("nan"))
            per_class_ap.append(float("nan"))

    print(f"\n  {'='*50}")
    print(f"  XGBoost + GNN 融合 测试集评估")
    print(f"  {'='*50}")
    print(f"  {'准确率 Accuracy':<24s} {acc:>10.4f}")
    print(f"  {'Macro-F1':<24s} {f1_mac:>10.4f}")
    print(f"  {'Weighted-F1':<24s} {f1_wt:>10.4f}")
    print(f"  {'宏平均 AUROC':<24s} {float(np.nanmean(per_class_auroc)):>10.4f}")
    print(f"  {'宏平均 AP':<24s} {float(np.nanmean(per_class_ap)):>10.4f}")
    print()
    print(classification_report(
        y_test, y_pred,
        target_names=[RISK_LABELS_CN[i] for i in range(n_classes)],
        digits=3, zero_division=0,
    ))

    fusion_metrics = dict(
        accuracy=acc, macro_f1=f1_mac, weighted_f1=f1_wt,
        macro_auroc=float(np.nanmean(per_class_auroc)),
        macro_ap=float(np.nanmean(per_class_ap)),
        per_class_f1=per_class_f1,
        per_class_auroc=per_class_auroc,
        per_class_ap=per_class_ap,
    )
    if return_predictions:
        fusion_metrics["y_test"] = y_test
        fusion_metrics["y_pred"] = y_pred
        fusion_metrics["y_prob"] = y_prob

    # 保存融合模型
    import pickle
    fusion_path = RESULTS_DIR / f"xgboost_gnn_fusion{save_suffix}.pkl"
    with open(fusion_path, "wb") as f:
        pickle.dump(xgb, f)
    print(f"  融合模型已保存: {fusion_path}")

    del X_mm, X_train, X_val, X_test
    gc.collect()
    return fusion_metrics


# ═══════════════════════════════════════════════════════════════
#  8. 对比实验汇总
# ═══════════════════════════════════════════════════════════════
def print_comparison(results: dict, suffix: str = ""):
    """打印所有实验的对比表，并生成 metrics_summary 对比条形图。

    Args:
        suffix: 结果文件后缀（多 seed 时自动填充，如 "_seed42"）
    """
    valid = {k: v for k, v in results.items() if v is not None}
    if not valid:
        print("  无可用实验结果")
        return

    print(f"\n{'='*72}")
    print("  方案一 vs 方案二：完整对比")
    print(f"{'='*72}")
    header = (f"  {'模型':<28s} {'Accuracy':>10s} {'Macro-F1':>10s} "
              f"{'W-F1':>10s} {'AUROC':>10s} {'AP':>10s}")
    print(header)
    print(f"  {'-'*68}")
    for name, m in valid.items():
        print(f"  {name:<28s} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f} "
              f"{m['weighted_f1']:>10.4f} {m['macro_auroc']:>10.4f} "
              f"{m['macro_ap']:>10.4f}")
    print(f"{'='*72}\n")

    out = {}
    for name, m in valid.items():
        out[name] = {k: v for k, v in m.items() if k != "per_class"}
    out_path = RESULTS_DIR / f"gnn_comparison{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  对比结果已保存: {out_path}")

    # 只在非后缀模式下画 summary 图，避免多 seed 时每次覆盖
    if not suffix:
        _both_langs(plot_metrics_summary, valid)


def plot_metrics_summary(results: dict, lang="cn"):
    """对比条形图（与 baseline_xgboost.py 的 metrics_summary.png 格式对齐）。"""
    metrics_keys = ["accuracy", "macro_f1", "weighted_f1", "macro_auroc", "macro_ap"]
    labels_map = {
        "cn": ["准确率", "Macro-F1", "Weighted-F1", "宏AUROC", "宏AP"],
        "en": ["Accuracy", "Macro-F1", "Weighted-F1", "Macro AUROC", "Macro AP"],
    }
    metric_labels = labels_map.get(lang, labels_map["en"])

    names = list(results.keys())
    n_models = len(names)
    n_metrics = len(metrics_keys)
    x = np.arange(n_metrics)
    width = 0.8 / max(n_models, 1)

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0",
              "#00BCD4", "#795548"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, name in enumerate(names):
        m = results[name]
        vals = [m.get(k, 0) for k in metrics_keys]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9,
                      label=name, color=colors[i % len(colors)], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    title = "全部实验指标对比" if lang == "cn" else "All Experiments Metrics Comparison"
    ax.set_title(title, fontsize=13)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(_out("metrics_summary.png", lang), dpi=150)
    plt.close()
    print(f"  指标对比图已保存: {_out('metrics_summary.png', lang)}")


# ═══════════════════════════════════════════════════════════════
#  9. 主实验入口
# ═══════════════════════════════════════════════════════════════
def _clear_gpu():
    """强制清理 GPU 显存碎片，实验切换时调用。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_gnn_experiments(
    hidden_dim: int | None = None,
    num_bases: int | None = None,
    edge_emb_dim: int = 64,
    heads: int = 4,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 4096,
    max_epochs: int = 99999,
    patience: int = 100,
    warmup_epochs: int = 10,
    focal_gamma: float | None = None,
    label_smoothing: float = 0.1,
    dropout: float = 0.3,
    plateau_factor: float = 0.5,
    plateau_patience: int = 15,
    skip_fusion: bool = False,
    start_from: int = 2,
    stop_at: int = 5,
    seed: int = 42,
    deterministic_cuda: bool = True,
    # —— 新增特性开关（默认全部关闭，保持向后兼容） ——
    strict_deterministic: bool = False,
    use_amp: bool = True,
    norm_type: str = "batch",       # "batch" / "pair" / "layer" / "none"
    pairnorm_scale: float = 1.0,
    drop_edge_exp4: float = 0.0,    # 仅 Exp-4 R-GAT 使用，其它保持 0
    drop_edge_all: float = 0.0,     # 若 >0 则对所有实验生效（会覆盖 Exp-4 专用值）
    self_loop_fill: str = "zero",   # Exp-4 GAT 的 self-loop 填充策略
    grad_clip: float = 1.0,
    emb_dim_mode: str = "half",     # "half" (hidden//2) / "full" (hidden) / "quarter" (hidden//4)
    decoder_type: str = "mlp",      # "mlp" / "bilinear" / "distmult"
    results_suffix: str = "",       # 结果文件后缀，多 seed 时自动填充
):
    """
    运行全部 GNN 实验。默认按实验使用文件顶部常量中的 hidden_dim / focal_gamma；
    若传入 hidden_dim 或 focal_gamma（非 None），则对所有子实验统一覆盖，便于扫维/消融。

      Exp-2: R-GCN (仅DDI图, 对标 AERGCN-DDI)
      Exp-3: R-GCN (混合图, 消融注意力)
      Exp-4: R-GAT (混合图)
      Exp-5: R-GAT + XGBoost 融合（保留入口，但默认叙事改为 R-GCN(DDI-only) 嵌入融合）

    max_epochs 极大（99999），完全依赖早停结束训练。

    train/val/test 划分固定为 SPLIT_RANDOM_STATE（默认 42），与 baseline_xgboost 一致。
    各子实验训练前会调用 set_reproducibility(seed + (exp_id-2)*1_000_003)。

    num_bases 默认行为：
      - Exp-2 (仅 DDI 图，10 种关系)：num_bases=8（足够）
      - Exp-3 (混合图，34 种关系)：num_bases=17（= 前向关系数，避免 basis 不足）
      - 若通过 CLI 显式传入 --num_bases 则覆盖上述默认

    start_from: 从第几个实验开始（2/3/4/5），用于重跑失败的实验。
    已完成实验的结果会从 gnn_comparison.json 中自动读取并合并到对比表。

    results_suffix: 传入非空字符串时，输出文件会改为 gnn_comparison{suffix}.json，
        用于多 seed 运行时避免互相覆盖。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    vram_total = 0.0
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        vram_free = torch.cuda.mem_get_info()[0] / 1e9
        print(f"  显存总量: {vram_total:.1f} GB  可用: {vram_free:.1f} GB")

    det_msg = "开（可复现）" if deterministic_cuda else "关（--fast_cuda，略快）"
    strict_msg = "开（use_deterministic_algorithms）" if strict_deterministic else "关（默认）"
    print(f"\n可复现性: 训练 seed 基={seed}；子实验 seed = 基 + (实验号−2)×1_000_003；"
          f"数据划分 random_state={SPLIT_RANDOM_STATE}（与 XGBoost 一致）；"
          f"cudnn deterministic: {det_msg}；严格确定性: {strict_msg}；"
          f"AMP: {'ON' if use_amp else 'OFF'}")
    if norm_type != "batch" or drop_edge_all > 0 or drop_edge_exp4 > 0 or self_loop_fill != "zero":
        print(f"【消融特性】norm_type={norm_type}, drop_edge_all={drop_edge_all}, "
              f"drop_edge_exp4={drop_edge_exp4}, self_loop_fill={self_loop_fill}")

    use_uniform_h = hidden_dim is not None
    use_uniform_gamma = focal_gamma is not None
    use_uniform_bases = num_bases is not None
    if use_uniform_h or use_uniform_gamma or use_uniform_bases:
        cov = []
        if use_uniform_h: cov.append(f"hidden_dim={hidden_dim}")
        if use_uniform_gamma: cov.append(f"focal_gamma={focal_gamma}")
        if use_uniform_bases: cov.append(f"num_bases={num_bases}")
        print("\n【统一超参覆盖】 " + ", ".join(cov))
    else:
        print("\n【各实验默认超参】"
              f"  Exp-2: H={DEFAULT_HIDDEN_DIM_EXP2}, γ={DEFAULT_FOCAL_GAMMA_EXP2}, bases=8 | "
              f"Exp-3: H={DEFAULT_HIDDEN_DIM_EXP3}, γ={DEFAULT_FOCAL_GAMMA_EXP3}, bases=17 | "
              f"Exp-4: H={DEFAULT_HIDDEN_DIM_EXP4}, γ={DEFAULT_FOCAL_GAMMA_EXP4}, bases=17")

    def _resolve_h_g(exp_id: int):
        h_def = {
            2: DEFAULT_HIDDEN_DIM_EXP2,
            3: DEFAULT_HIDDEN_DIM_EXP3,
            4: DEFAULT_HIDDEN_DIM_EXP4,
        }[exp_id]
        g_def = {
            2: DEFAULT_FOCAL_GAMMA_EXP2,
            3: DEFAULT_FOCAL_GAMMA_EXP3,
            4: DEFAULT_FOCAL_GAMMA_EXP4,
        }[exp_id]
        h = hidden_dim if use_uniform_h else h_def
        g = focal_gamma if use_uniform_gamma else g_def
        # emb_dim 策略：
        #   "half" (默认): emb = hidden//2，与历史行为一致
        #   "full":        emb = hidden，PairMLP 表达力更强（第二层残差连接也会启用）
        #   "quarter":     emb = hidden//4，更小的嵌入，显存紧张时用
        if emb_dim_mode == "full":
            e = h
        elif emb_dim_mode == "quarter":
            e = h // 4
        else:  # half
            e = h // 2
        return h, g, e

    def _resolve_bases(exp_id: int) -> int:
        """Exp-2 用 8（10 种关系已足），Exp-3/4 用 17（17 种前向关系，避免 basis 不足）。"""
        if use_uniform_bases:
            return num_bases
        return 8 if exp_id == 2 else 17

    def _resolve_drop_edge(exp_id: int) -> float:
        if drop_edge_all > 0:
            return drop_edge_all
        if exp_id == 4:
            return drop_edge_exp4
        return 0.0

    def _make_config(fg: float, de: float):
        return dict(
            lr=lr, weight_decay=weight_decay, patience=patience,
            max_epochs=max_epochs, batch_size=batch_size,
            warmup_epochs=warmup_epochs,
            focal_gamma=fg, label_smoothing=label_smoothing,
            plateau_factor=plateau_factor, plateau_patience=plateau_patience,
            use_amp=use_amp, drop_edge=de, grad_clip=grad_clip,
        )

    # ── Step 1: 基础信息加载 ──
    _, drug_fp_raw, kg_meta = build_knowledge_graph(
        include_bio=True, ddi_pairs=None)
    n_drugs = kg_meta["n_drugs"]
    n_bio = kg_meta["n_enz"] + kg_meta["n_tp"] + kg_meta["n_tgt"]

    print("加载 DDI 对数据...")
    pair_idx, y, valid_mask = load_ddi_pairs(kg_meta["drug_id_to_idx"])
    idx_train, idx_val, idx_test = split_data(y)
    train_pairs = pair_idx[idx_train]
    train_labels = y[idx_train]

    # ── 加载已有结果（始终保留未被重跑的实验结果） ──
    results = {}
    comparison_fname = f"gnn_comparison{results_suffix}.json"
    prev_path = RESULTS_DIR / comparison_fname
    if prev_path.exists():
        try:
            with open(prev_path, encoding="utf-8") as f:
                results = json.load(f)
            if results:
                print(f"\n已加载已有实验结果 ({comparison_fname}): {list(results.keys())}")
                print(f"  本次将重跑 Exp-{start_from}~{stop_at}，其余结果保留。")
        except Exception:
            results = {}

    # ── Step 2: 构建两种图 ──
    need_ddi = start_from <= 2 <= stop_at
    need_hybrid = start_from <= 4 and stop_at >= 3

    if need_ddi:
        print("构建仅DDI图 (Exp-2)...")
    graph_ddi, _, _ = build_knowledge_graph(
        ddi_pairs=train_pairs, ddi_labels=train_labels,
        include_bio=False) if need_ddi else (None, None, None)

    if need_hybrid:
        print("构建混合图 (Exp-3/4)...")
    graph_hybrid, _, _ = build_knowledge_graph(
        ddi_pairs=train_pairs, ddi_labels=train_labels,
        include_bio=True) if need_hybrid else (None, None, None)

    drug_fp = drug_fp_raw

    # ── Exp-2: R-GCN 仅DDI图 (对标 AERGCN-DDI) ──
    if start_from <= 2 <= stop_at:
        set_reproducibility(_train_run_seed(seed, 2),
                            deterministic_cuda=deterministic_cuda,
                            strict_deterministic=strict_deterministic)
        h2, g2, e2 = _resolve_h_g(2)
        b2 = _resolve_bases(2)
        de2 = _resolve_drop_edge(2)
        config = _make_config(g2, de2)
        print(f"\n{'#'*60}")
        print(f"#  Exp-2: R-GCN 仅DDI图 (对标 AERGCN-DDI)")
        print(f"#  hidden_dim={h2}, emb_dim={e2}, focal_gamma={g2}, num_bases={b2}")
        print(f"#  norm_type={norm_type}, drop_edge={de2}")
        print(f"{'#'*60}")
        _clear_gpu()
        rgcn_ddi = RGCN_DDI(
            n_drugs=n_drugs, n_bio=n_bio, hidden_dim=h2,
            emb_dim=e2, num_rels=NUM_RELATIONS, num_bases=b2,
            dropout=dropout,
            norm_type=norm_type, pairnorm_scale=pairnorm_scale,
            decoder_type=decoder_type,
        )
        n_p = sum(p.numel() for p in rgcn_ddi.parameters())
        print(f"  参数量: {n_p:,}")

        rgcn_ddi, hist = train_model(
            rgcn_ddi, graph_ddi, drug_fp, pair_idx, y,
            idx_train, idx_val, device, config)
        yt, yp, yprob, m = evaluate_model(
            rgcn_ddi, graph_ddi, drug_fp, pair_idx, y, idx_test, device,
            "Exp-2 R-GCN(DDI-only)")
        results["Exp-2 R-GCN(DDI-only)"] = m
        _both_langs(plot_training_curves, hist, model_name="Exp2-R-GCN-DDI")
        _both_langs(plot_confusion_matrix, yt, yp, model_name="Exp2-R-GCN-DDI")
        _both_langs(plot_roc_pr, yt, yprob, model_name="Exp2-R-GCN-DDI")

        torch.save(rgcn_ddi.state_dict(),
                   RESULTS_DIR / f"rgcn_ddi_only_model{results_suffix}.pt")
        del rgcn_ddi
        _clear_gpu()

    # ── Exp-3: R-GCN 混合图 (消融: GCN vs GAT) ──
    if start_from <= 3 <= stop_at:
        set_reproducibility(_train_run_seed(seed, 3),
                            deterministic_cuda=deterministic_cuda,
                            strict_deterministic=strict_deterministic)
        h3, g3, e3 = _resolve_h_g(3)
        b3 = _resolve_bases(3)
        de3 = _resolve_drop_edge(3)
        config = _make_config(g3, de3)
        print(f"\n{'#'*60}")
        print(f"#  Exp-3: R-GCN 混合图 (消融: GCN vs GAT)")
        print(f"#  hidden_dim={h3}, emb_dim={e3}, focal_gamma={g3}, num_bases={b3}")
        print(f"#  norm_type={norm_type}, drop_edge={de3}")
        print(f"{'#'*60}")
        _clear_gpu()
        rgcn_h = RGCN_DDI(
            n_drugs=n_drugs, n_bio=n_bio, hidden_dim=h3,
            emb_dim=e3, num_rels=NUM_RELATIONS, num_bases=b3,
            dropout=dropout,
            norm_type=norm_type, pairnorm_scale=pairnorm_scale,
            decoder_type=decoder_type,
        )
        n_p = sum(p.numel() for p in rgcn_h.parameters())
        print(f"  参数量: {n_p:,}")

        rgcn_h, hist = train_model(
            rgcn_h, graph_hybrid, drug_fp, pair_idx, y,
            idx_train, idx_val, device, config)
        yt, yp, yprob, m = evaluate_model(
            rgcn_h, graph_hybrid, drug_fp, pair_idx, y, idx_test, device,
            "Exp-3 R-GCN(Hybrid)")
        results["Exp-3 R-GCN(Hybrid)"] = m
        _both_langs(plot_training_curves, hist, model_name="Exp3-R-GCN-Hybrid")
        _both_langs(plot_confusion_matrix, yt, yp, model_name="Exp3-R-GCN-Hybrid")
        _both_langs(plot_roc_pr, yt, yprob, model_name="Exp3-R-GCN-Hybrid")

        torch.save(rgcn_h.state_dict(),
                   RESULTS_DIR / f"rgcn_hybrid_model{results_suffix}.pt")
        del rgcn_h
        _clear_gpu()

    # ── Exp-4: R-GAT 混合图 (GAT 对照组) ──
    if start_from <= 4 <= stop_at:
        set_reproducibility(_train_run_seed(seed, 4),
                            deterministic_cuda=deterministic_cuda,
                            strict_deterministic=strict_deterministic)
        h4, g4, e4 = _resolve_h_g(4)
        de4 = _resolve_drop_edge(4)
        config = _make_config(g4, de4)
        print(f"\n{'#'*60}")
        print(f"#  Exp-4: R-GAT 混合图 (GAT 对照组)")
        print(f"#  hidden_dim={h4}, emb_dim={e4}, focal_gamma={g4}")
        print(f"#  norm_type={norm_type}, drop_edge={de4}, self_loop_fill={self_loop_fill}")
        print(f"{'#'*60}")
        _clear_gpu()
        rgat = RGAT_DDI(
            n_drugs=n_drugs, n_bio=n_bio, hidden_dim=h4,
            emb_dim=e4, num_rels=NUM_RELATIONS,
            edge_emb_dim=edge_emb_dim, heads=heads, dropout=dropout,
            norm_type=norm_type, pairnorm_scale=pairnorm_scale,
            self_loop_fill=self_loop_fill,
            decoder_type=decoder_type,
        )
        n_p = sum(p.numel() for p in rgat.parameters())
        print(f"  参数量: {n_p:,}")

        rgat, hist = train_model(
            rgat, graph_hybrid, drug_fp, pair_idx, y,
            idx_train, idx_val, device, config)
        yt, yp, yprob, m = evaluate_model(
            rgat, graph_hybrid, drug_fp, pair_idx, y, idx_test, device,
            "Exp-4 R-GAT(Hybrid)")
        results["Exp-4 R-GAT(Hybrid)"] = m
        _both_langs(plot_training_curves, hist, model_name="Exp4-R-GAT-Hybrid")
        _both_langs(plot_confusion_matrix, yt, yp, model_name="Exp4-R-GAT-Hybrid")
        _both_langs(plot_roc_pr, yt, yprob, model_name="Exp4-R-GAT-Hybrid")

        _both_langs(plot_tsne, rgat, graph_hybrid, drug_fp, y, pair_idx, device,
                    model_name="Exp4-R-GAT-Hybrid")
        _both_langs(plot_attention_analysis, rgat, graph_hybrid, drug_fp, kg_meta,
                    device)

        torch.save(rgat.state_dict(),
                   RESULTS_DIR / f"rgat_hybrid_model{results_suffix}.pt")

        # ── Exp-5: R-GAT + XGBoost 融合（保留兼容入口） ──
        if not skip_fusion:
            set_reproducibility(_train_run_seed(seed, 5),
                                deterministic_cuda=deterministic_cuda,
                                strict_deterministic=strict_deterministic)
            fusion_metrics = run_fusion_experiment(
                rgat, graph_hybrid, drug_fp, pair_idx, y,
                idx_train, idx_val, idx_test, device, kg_meta)
            results["Exp-5 R-GAT+XGBoost"] = fusion_metrics

        del rgat
        _clear_gpu()

    # ── 对比汇总 ──
    xgb_cv_path = RESULTS_DIR / "cv_results.json"
    if xgb_cv_path.exists():
        with open(xgb_cv_path, encoding="utf-8") as f:
            xgb_cv = json.load(f)
        if "test_metrics" in xgb_cv:
            tm = xgb_cv["test_metrics"]
            results["Exp-1 XGBoost v2"] = dict(
                accuracy=tm.get("accuracy", 0),
                macro_f1=tm.get("macro_f1", 0),
                weighted_f1=tm.get("weighted_f1", 0),
                macro_auroc=tm.get("macro_auroc", 0),
                macro_ap=tm.get("macro_ap", 0),
            )

    print_comparison(results, suffix=results_suffix)
    print("全部 GNN 实验完成！")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GNN DDI 实验")
    # ── 基础超参 ──
    parser.add_argument(
        "--hidden_dim", type=int, default=None,
        help="统一覆盖各实验的隐藏层维度；省略则 Exp-2/3/4 使用文件内默认常量")
    parser.add_argument(
        "--num_bases", type=int, default=None,
        help="RGCN basis 数；省略则 Exp-2 用 8，Exp-3/4 用 17")
    parser.add_argument(
        "--focal_gamma", type=float, default=None,
        help="统一覆盖各实验的 Focal γ；省略则 Exp-2/3 用 2.0，Exp-4 用 0.9")
    # ── 实验范围 ──
    parser.add_argument("--skip_fusion", action="store_true",
                        help="跳过 Exp-5 融合实验")
    parser.add_argument("--start_from", type=int, default=2,
                        choices=[2, 3, 4, 5],
                        help="从第几个实验开始（2/3/4/5），已完成结果从"
                             " gnn_comparison.json 自动读取合并")
    parser.add_argument("--stop_at", type=int, default=5,
                        choices=[2, 3, 4, 5],
                        help="运行到第几个实验就停（含）。配合 start_from"
                             " 可只跑某一个实验，例如只跑Exp-2: "
                             "--start_from 2 --stop_at 2")
    # ── 可复现性 ──
    parser.add_argument("--seed", type=int, default=42,
                        help="训练随机种子基（子实验自动加偏移）；数据划分固定为 42 与 XGBoost 一致")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="多 seed 批量运行。例 --seeds 42 123 2024 7 99，"
                             "每个 seed 跑一次完整 start_from~stop_at，"
                             "结果分别保存为 gnn_comparison_seed{N}.json；"
                             "与 --seed 互斥（给了 --seeds 就忽略 --seed）")
    parser.add_argument("--fast_cuda", action="store_true",
                        help="关闭 cudnn deterministic（更快，数值可能略有差异）")
    parser.add_argument("--strict_deterministic", action="store_true",
                        help="开启 torch.use_deterministic_algorithms(True, warn_only=True)，"
                             "消除 PyG scatter 的非确定性。训练略慢但相同 seed 下波动 <1e-4")
    parser.add_argument("--no_amp", action="store_true",
                        help="关闭 AMP 混合精度，全程 fp32。训练慢 30-50%%，但最稳定可复现")
    # ── 消融开关（默认全部关闭） ──
    parser.add_argument("--norm_type", type=str, default="batch",
                        choices=["batch", "pair", "layer", "none"],
                        help="归一化层类型。'pair' 专为 GNN 设计，缓解 oversmoothing（对 GAT 尤其有效）")
    parser.add_argument("--pairnorm_scale", type=float, default=1.0,
                        help="PairNorm 的 scale 参数（仅 norm_type=pair 生效）")
    parser.add_argument("--drop_edge_exp4", type=float, default=0.0,
                        help="Exp-4 R-GAT 的 DropEdge 比例，推荐 0.2~0.3")
    parser.add_argument("--drop_edge_all", type=float, default=0.0,
                        help="对所有实验统一应用 DropEdge（会覆盖 --drop_edge_exp4）")
    parser.add_argument("--self_loop_fill", type=str, default="zero",
                        choices=["zero", "mean"],
                        help="Exp-4 GATv2 的 self-loop edge_attr 填充策略。"
                             "'mean' 缓解零关系嵌入的范数失衡")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="梯度裁剪阈值（按范数），<=0 禁用")
    parser.add_argument("--emb_dim_mode", type=str, default="half",
                        choices=["half", "full", "quarter"],
                        help="emb_dim 相对于 hidden_dim 的比例："
                             "half=hidden//2 (默认/历史行为)、full=hidden、quarter=hidden//4")
    parser.add_argument("--decoder", type=str, default="mlp",
                        choices=["mlp", "bilinear", "distmult"],
                        help="药物对解码器类型。'mlp' 为默认 PairMLP，"
                             "'bilinear' 为关系特定全矩阵 W_r，'distmult' 为对角 W_r")

    args = parser.parse_args()

    common_kwargs = dict(
        hidden_dim=args.hidden_dim,
        num_bases=args.num_bases,
        skip_fusion=args.skip_fusion,
        start_from=args.start_from,
        stop_at=args.stop_at,
        focal_gamma=args.focal_gamma,
        deterministic_cuda=not args.fast_cuda,
        strict_deterministic=args.strict_deterministic,
        use_amp=not args.no_amp,
        norm_type=args.norm_type,
        pairnorm_scale=args.pairnorm_scale,
        drop_edge_exp4=args.drop_edge_exp4,
        drop_edge_all=args.drop_edge_all,
        self_loop_fill=args.self_loop_fill,
        grad_clip=args.grad_clip,
        emb_dim_mode=args.emb_dim_mode,
        decoder_type=args.decoder,
    )

    if args.seeds:
        print(f"\n{'#'*72}")
        print(f"#  多 seed 模式: seeds = {args.seeds}")
        print(f"{'#'*72}")
        all_runs = {}
        for s in args.seeds:
            print(f"\n{'=' * 72}\n  开始 seed={s} 的完整运行\n{'=' * 72}")
            suffix = f"_seed{s}"
            res = run_gnn_experiments(seed=s, results_suffix=suffix, **common_kwargs)
            all_runs[f"seed{s}"] = res
        # 汇总多 seed 结果到 gnn_multi_seed.json
        multi_path = RESULTS_DIR / "gnn_multi_seed.json"
        out = {k: {name: {kk: vv for kk, vv in m.items() if kk != "per_class"}
                   for name, m in res.items() if m is not None}
               for k, res in all_runs.items()}
        with open(multi_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n多 seed 汇总结果已保存: {multi_path}")
    else:
        run_gnn_experiments(seed=args.seed, **common_kwargs)
