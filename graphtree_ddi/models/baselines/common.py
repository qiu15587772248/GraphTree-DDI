"""基线共用：数据加载、划分协议、指标、memmap 分批、预测/状态保存。

特征布局以 data/preprocess.py 的 build_pair_features 为准（float32，4120 维）：
  [0:2048]     fp_product（两药 Morgan 指纹逐位乘积）
  [2048:4096]  fp_diff（两药指纹绝对值差）
  [4096:4104]  CYP 酶 8 维
  [4104:4112]  转运体 8 维
  [4112:4120]  靶点 8 维

注意：X 不是 [drug_a 2060 ‖ drug_b 2060]。药对级药理块已经是交互特征，
无法从一行 X 唯一还原两药各自的 2048 位指纹。DeepDDI_SSP / DistMult / ComplEx
所需的单药指纹从 SMILES 按 preprocess.smiles_to_fingerprint 重算。
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

import ctypes
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight


N_CLASSES = 5
FEAT_DIM = 4120  # 2048*2 + 24，与 preprocess.FEAT_DIM 一致
SPLIT_RANDOM_STATE = 42  # 与 run_pipeline_final.SPLIT_RANDOM_STATE 一致
N_FOLDS_DEFAULT = 5
SEEDS_DEFAULT = [42, 123, 2024]
SMOKE_FRACTION = 0.01

# 与 data/preprocess.py::build_pair_features 输出下标一一对应
SLICE_FP_PRODUCT = slice(0, 2048)
SLICE_FP_DIFF = slice(2048, 4096)
SLICE_FINGERPRINT = slice(0, 4096)
SLICE_CYP = slice(4096, 4104)
SLICE_TRANSPORTER = slice(4104, 4112)
SLICE_TARGET = slice(4112, 4120)

MODALITY_SLICES: dict[str, slice] = {
    "fingerprint": SLICE_FINGERPRINT,
    "cyp": SLICE_CYP,
    "transporter": SLICE_TRANSPORTER,
    "target": SLICE_TARGET,
}

PIPELINE_METRIC_KEYS = (
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "macro_auroc",
    "macro_ap",
)
EXTRA_METRIC_KEYS = (
    "adjacent_acc",
    "high_risk_sens",
    "high_risk_spec",
    "severe_underestimation_rate",
    "qwk",
    "mae_ordinal",
)
ALL_SCALAR_METRIC_KEYS = PIPELINE_METRIC_KEYS + EXTRA_METRIC_KEYS


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except Exception:
        pass


def _as_float(x) -> float:
    v = float(x)
    return v


@dataclass
class LoadedData:
    data_dir: Path
    meta: dict
    n_samples: int
    feat_dim: int
    y: np.ndarray
    pairs: pd.DataFrame
    drug1: np.ndarray
    drug2: np.ndarray
    X_mm: np.memmap | np.ndarray
    drugs_csv: Path
    x_file: Path | None = None
    X_ram: np.ndarray | None = None
    X_gpu: object | None = None
    gather_map: np.ndarray | None = None
    preload_requested: str = "memmap"
    preload_actual: str = "memmap"
    ram_avail_gb: float | None = None
    ram_total_gb: float | None = None
    x_bytes: int = 0


def load_processed(
    data_dir: Path,
    drugs_csv: Path | None = None,
) -> LoadedData:
    data_dir = Path(data_dir)
    meta_path = data_dir / "meta.json"
    y_path = data_dir / "y.npy"
    pairs_path = data_dir / "pairs.csv"
    mm_path = data_dir / "X_full.dat"
    x_path = data_dir / "X.npy"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    n_samples = int(meta["n_samples"])
    feat_dim = int(meta["feature_dim"])
    if feat_dim != FEAT_DIM:
        print(f"[warn] meta.feature_dim={feat_dim}，preprocess 常量 FEAT_DIM={FEAT_DIM}")
    y = np.load(y_path)
    y = np.asarray(y[:n_samples])
    pairs = pd.read_csv(pairs_path)
    pairs = pairs.iloc[:n_samples].reset_index(drop=True)
    if len(y) != n_samples or len(pairs) != n_samples:
        raise ValueError(
            f"y/pairs 长度与 meta.n_samples 不一致: y={len(y)} pairs={len(pairs)} n={n_samples}"
        )
    if mm_path.exists():
        X_mm = np.memmap(mm_path, dtype="float32", mode="r", shape=(n_samples, feat_dim))
        src = mm_path
    elif x_path.exists():
        X_mm = np.load(x_path, mmap_mode="r")
        src = x_path
    else:
        raise FileNotFoundError(f"未找到 {mm_path} 或 {x_path}")
    x_bytes = int(n_samples) * int(feat_dim) * 4
    print(f"[data] {src.name} dtype={getattr(X_mm, 'dtype', None)} "
          f"shape=({n_samples}, {feat_dim})  {x_bytes / 1e9:.2f} GB")
    # 只读第 0 行做布局冒烟，避免把 25GB 拉进内存
    row0 = np.asarray(X_mm[0], dtype=np.float32)
    print(
        f"[data] row0 指纹积范数={float(np.linalg.norm(row0[SLICE_FP_PRODUCT])):.4f} "
        f"指纹差范数={float(np.linalg.norm(row0[SLICE_FP_DIFF])):.4f} "
        f"CYP={row0[SLICE_CYP].tolist()} "
        f"转运体={row0[SLICE_TRANSPORTER].tolist()} "
        f"靶点={row0[SLICE_TARGET].tolist()}"
    )
    drug1 = pairs["drug1_id"].astype(str).to_numpy()
    drug2 = pairs["drug2_id"].astype(str).to_numpy()
    if drugs_csv is None:
        drugs_csv = _PROJ / "data" / "raw" / "drugbank_drugs.csv"
    else:
        drugs_csv = Path(drugs_csv)
    return LoadedData(
        data_dir=data_dir,
        meta=meta,
        n_samples=n_samples,
        feat_dim=feat_dim,
        y=y.astype(np.int64, copy=False),
        pairs=pairs,
        drug1=drug1,
        drug2=drug2,
        X_mm=X_mm,
        drugs_csv=drugs_csv,
        x_file=src,
        x_bytes=x_bytes,
    )


def physical_ram_bytes() -> tuple[int, int]:
    """返回 (avail_phys, total_phys)，单位字节。跨 Windows / Linux。"""
    if sys.platform.startswith("win"):
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            raise OSError("GlobalMemoryStatusEx 失败")
        return int(stat.ullAvailPhys), int(stat.ullTotalPhys)
    try:
        page = int(os.sysconf("SC_PAGE_SIZE"))
        total = int(os.sysconf("SC_PHYS_PAGES")) * page
        avail = int(os.sysconf("SC_AVPHYS_PAGES")) * page
        return avail, total
    except (ValueError, OSError, AttributeError):
        pass
    info = Path("/proc/meminfo")
    if info.exists():
        kv = {}
        for line in info.read_text(encoding="utf-8").splitlines():
            parts = line.replace(":", " ").split()
            if len(parts) >= 2:
                kv[parts[0]] = int(parts[1]) * 1024
        avail = int(kv.get("MemAvailable", kv.get("MemFree", 0)))
        total = int(kv.get("MemTotal", 0))
        return avail, total
    return 0, 0


def ram_gather_ready(data: LoadedData) -> bool:
    return data.X_ram is not None or data.X_gpu is not None


def _fromfile_x(path: Path, n: int, d: int) -> np.ndarray:
    t0 = time.time()
    arr = np.fromfile(path, dtype=np.float32, count=n * d)
    if arr.size != n * d:
        raise ValueError(f"fromfile 读到 {arr.size} 个数，期望 {n * d}")
    arr = arr.reshape(n, d)
    print(f"[preload] np.fromfile {path.name} → {arr.shape}  {time.time() - t0:.1f}s")
    return arr


def compact_working_x(data: LoadedData, work_idx: np.ndarray) -> None:
    """把工作子集行按升序一次读入 RAM，gather_map 映射原始行号。"""
    work = np.sort(np.asarray(work_idx, dtype=np.int64))
    t0 = time.time()
    data.X_ram = np.ascontiguousarray(data.X_mm[work], dtype=np.float32)
    gmap = np.full(data.n_samples, -1, dtype=np.int32)
    gmap[work] = np.arange(len(work), dtype=np.int32)
    data.gather_map = gmap
    data.preload_actual = "cpu_subset"
    gb = data.X_ram.nbytes / 1e9
    print(
        f"[preload] 工作子集 {len(work):,} 行载入 RAM ({gb:.2f} GB)，"
        f"顺序读 {time.time() - t0:.1f}s（全量文件未驻留）"
    )


def apply_preload(
    data: LoadedData,
    mode: str,
    device,
    work_idx: np.ndarray | None = None,
    ram_margin_gb: float = 1.0,
    gpu_reserve_gb: float = 6.0,
) -> LoadedData:
    """按 --preload 把 X 放到 memmap / CPU RAM / GPU。不足则降级。"""
    mode = (mode or "cpu").lower()
    if mode not in ("memmap", "cpu", "gpu"):
        raise ValueError(f"未知 preload: {mode}")
    data.preload_requested = mode
    avail, total = physical_ram_bytes()
    data.ram_avail_gb = avail / 1e9
    data.ram_total_gb = total / 1e9
    need = int(data.x_bytes)
    print(
        f"[preload] request={mode}  X={need / 1e9:.2f} GB  "
        f"RAM avail={data.ram_avail_gb:.2f} GB total={data.ram_total_gb:.2f} GB"
    )
    if data.x_file is None:
        data.preload_actual = "memmap"
        return data

    if mode == "gpu":
        try:
            import torch
            if getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
                free_b, tot_b = torch.cuda.mem_get_info()
                need_gpu = need + int(gpu_reserve_gb * (1024 ** 3))
                print(
                    f"[preload] GPU free={free_b / 1e9:.2f} GB total={tot_b / 1e9:.2f} GB  "
                    f"need X+{gpu_reserve_gb:.0f}GB={need_gpu / 1e9:.2f} GB"
                )
                if free_b >= need_gpu and avail >= need + ram_margin_gb * 1e9:
                    X = _fromfile_x(data.x_file, data.n_samples, data.feat_dim)
                    data.X_gpu = torch.from_numpy(X).to(device, dtype=torch.float32)
                    data.X_ram = None
                    data.gather_map = None
                    data.preload_actual = "gpu"
                    print(f"[preload] 实际=gpu  X 在 {device}")
                    return data
                print("[warn] GPU 余量不足（需 X 字节 + 6 GB），回退 cpu")
            else:
                print("[warn] --preload gpu 但 CUDA 不可用，回退 cpu")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] GPU 预载失败 ({e})，回退 cpu")
        mode = "cpu"

    if mode == "cpu":
        if avail >= need + ram_margin_gb * 1e9:
            try:
                data.X_ram = _fromfile_x(data.x_file, data.n_samples, data.feat_dim)
                data.gather_map = None
                data.X_gpu = None
                data.preload_actual = "cpu"
                print("[preload] 实际=cpu 全量驻留 RAM")
                return data
            except MemoryError:
                print("[warn] MemoryError，回退 memmap")
        else:
            print(
                f"[warn] 物理内存不足以预载全量 X "
                f"({data.ram_avail_gb:.2f} GB 可用 < {need / 1e9:.2f}+{ram_margin_gb:.0f} GB)，回退 memmap"
            )
        if work_idx is not None and len(work_idx) < data.n_samples:
            compact_working_x(data, work_idx)
            return data
        data.preload_actual = "memmap"
        print("[preload] 实际=memmap")
        return data

    data.preload_actual = "memmap"
    print("[preload] 实际=memmap")
    return data


def gather_x(data: LoadedData, rows: np.ndarray, device):
    """按原始 pairs 行号取特征，返回 device 上的 float32 张量。"""
    import torch

    rows = np.asarray(rows, dtype=np.int64)
    if data.X_gpu is not None:
        if data.gather_map is not None:
            loc = data.gather_map[rows]
            if np.any(loc < 0):
                raise IndexError("gather_map 含未物化的行")
            idx = torch.from_numpy(loc.astype(np.int64, copy=False)).to(device)
        else:
            idx = torch.from_numpy(rows).to(device)
        return data.X_gpu.index_select(0, idx)
    if data.X_ram is not None:
        if data.gather_map is not None:
            loc = data.gather_map[rows]
            if np.any(loc < 0):
                raise IndexError("gather_map 含未物化的行")
            x = np.ascontiguousarray(data.X_ram[loc], dtype=np.float32)
        else:
            x = np.ascontiguousarray(data.X_ram[rows], dtype=np.float32)
        t = torch.from_numpy(x)
        if device.type == "cuda":
            return t.pin_memory().to(device, non_blocking=True)
        return t.to(device)
    x = np.ascontiguousarray(data.X_mm[rows], dtype=np.float32)
    t = torch.from_numpy(x)
    if device.type == "cuda":
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


def smoke_subset_indices(y: np.ndarray, fraction: float = SMOKE_FRACTION, seed: int = 42) -> np.ndarray:
    """分层抽取 fraction 的行索引，返回升序原始下标。"""
    all_idx = np.arange(len(y), dtype=np.int64)
    if fraction >= 1.0:
        return all_idx
    n_take = max(int(round(len(y) * fraction)), N_CLASSES * 20)
    n_take = min(n_take, len(y))
    test_size = n_take / float(len(y))
    try:
        _, sub = train_test_split(
            all_idx, test_size=test_size, stratify=y, random_state=seed,
        )
    except ValueError:
        _, sub = train_test_split(
            all_idx, test_size=test_size, stratify=None, random_state=seed,
        )
    return np.sort(np.asarray(sub, dtype=np.int64))


@dataclass
class PairSplits:
    idx_tv: np.ndarray
    idx_test: np.ndarray
    cv_splits: list[tuple[int, np.ndarray, np.ndarray]]  # (fold, train, val)


def pair_splits_pipeline_final(
    y: np.ndarray,
    n_folds: int = N_FOLDS_DEFAULT,
    random_state: int = SPLIT_RANDOM_STATE,
) -> PairSplits:
    """逐字复现 models/run_pipeline_final.py::prepare_global_data 的划分协议。

    对应源码（random_state=42, n_splits=5）::
        all_idx = np.arange(len(y))
        idx_tv, idx_test = train_test_split(
            all_idx, test_size=0.2, stratify=y, random_state=SPLIT_RANDOM_STATE,
        )
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SPLIT_RANDOM_STATE)
        for fold_idx, (train_rel, val_rel) in enumerate(skf.split(idx_tv, y[idx_tv]), start=1):
            idx_train_f = idx_tv[train_rel]
            idx_val_f = idx_tv[val_rel]
    """
    all_idx = np.arange(len(y))
    idx_tv, idx_test = train_test_split(
        all_idx, test_size=0.2, stratify=y, random_state=random_state,
    )
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    cv_splits: list[tuple[int, np.ndarray, np.ndarray]] = []
    for fold_idx, (train_rel, val_rel) in enumerate(skf.split(idx_tv, y[idx_tv]), start=1):
        idx_train_f = idx_tv[train_rel]
        idx_val_f = idx_tv[val_rel]
        cv_splits.append((fold_idx, idx_train_f, idx_val_f))
    return PairSplits(idx_tv=idx_tv, idx_test=idx_test, cv_splits=cv_splits)


def _pair_splits_independent(
    y: np.ndarray,
    n_folds: int = N_FOLDS_DEFAULT,
    random_state: int = SPLIT_RANDOM_STATE,
) -> PairSplits:
    """换变量名的独立重算，用于与 pipeline 复现函数比对索引。"""
    n = len(y)
    indices = np.arange(n)
    dev, heldout = train_test_split(
        indices, test_size=0.2, stratify=y, random_state=random_state,
    )
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds: list[tuple[int, np.ndarray, np.ndarray]] = []
    for i, (tr_rel, va_rel) in enumerate(kfold.split(dev, y[dev]), start=1):
        folds.append((i, dev[tr_rel], dev[va_rel]))
    return PairSplits(idx_tv=dev, idx_test=heldout, cv_splits=folds)


def assert_pair_splits_match_pipeline(y: np.ndarray, splits: PairSplits) -> None:
    alt = _pair_splits_independent(y, n_folds=len(splits.cv_splits), random_state=SPLIT_RANDOM_STATE)
    if not np.array_equal(splits.idx_test, alt.idx_test):
        raise AssertionError("pair 测试集索引与 run_pipeline_final 协议不一致")
    if not np.array_equal(splits.idx_tv, alt.idx_tv):
        raise AssertionError("pair train+val 索引与 run_pipeline_final 协议不一致")
    if len(splits.cv_splits) != len(alt.cv_splits):
        raise AssertionError("CV 折数不一致")
    for (f1, tr1, va1), (f2, tr2, va2) in zip(splits.cv_splits, alt.cv_splits):
        if f1 != f2 or (not np.array_equal(tr1, tr2)) or (not np.array_equal(va1, va2)):
            raise AssertionError(f"fold={f1} 的 train/val 索引与 pipeline 协议不一致")
    print(
        f"[split-check] pair 模式与 run_pipeline_final.py 协议完全一致 | "
        f"n_test={len(splits.idx_test):,} sum(idx_test)={int(np.sum(splits.idx_test))} "
        f"head={splits.idx_test[:5].tolist()}"
    )


def remap_splits_to_original(local: PairSplits, original_idx: np.ndarray) -> PairSplits:
    """将相对子集的划分映射回 pairs.csv 行号。"""
    return PairSplits(
        idx_tv=original_idx[local.idx_tv],
        idx_test=original_idx[local.idx_test],
        cv_splits=[
            (f, original_idx[tr], original_idx[va]) for f, tr, va in local.cv_splits
        ],
    )


@dataclass
class DrugSplits:
    drugs: list
    U: set
    K: set
    kk_train: np.ndarray
    kk_val: np.ndarray
    s1: np.ndarray
    s2: np.ndarray
    kk: np.ndarray
    stratified: bool
    n_u: int


def build_drug_splits(
    pairs: pd.DataFrame,
    y: np.ndarray,
    indices: np.ndarray,
    drug_split_seed: int,
) -> DrugSplits:
    """冷启动药物划分。定义必须与流水线执行者逐字一致：

    drugs = sorted(set(pairs.drug1_id) | set(pairs.drug2_id))
    rng = np.random.default_rng(drug_split_seed)
    perm = rng.permutation(len(drugs))
    n_u = int(round(0.2 * len(drugs)))
    U = {drugs[i] for i in perm[:n_u]}   # 其余为 K
    KK = 两药均在 K；S1 = 恰一药在 U；S2 = 两药均在 U
    验证集: train_test_split(KK, test_size=0.10, stratify=y_KK, random_state=drug_split_seed)

    当 indices 不是全集时（冒烟 1%），drugs / KK / S1 / S2 只在该工作子集上计算，
    返回的下标仍是 pairs.csv 的原始行号。KK 转成 np.flatnonzero 顺序（原始行号升序）
    再交给 train_test_split，避免 set 无序。
    """
    work = np.asarray(indices, dtype=np.int64)
    sub = pairs.iloc[work]
    # 工作子集上的药物宇宙（全集时等价于全表）
    drugs = sorted(set(sub["drug1_id"]) | set(sub["drug2_id"]))
    rng = np.random.default_rng(drug_split_seed)
    perm = rng.permutation(len(drugs))
    n_u = int(round(0.2 * len(drugs)))
    U = {drugs[i] for i in perm[:n_u]}
    K = {drugs[i] for i in perm[n_u:]}
    d1 = sub["drug1_id"].to_numpy()
    d2 = sub["drug2_id"].to_numpy()
    u1 = pd.Series(d1, copy=False).isin(U).to_numpy()
    u2 = pd.Series(d2, copy=False).isin(U).to_numpy()
    kk_local = np.flatnonzero((~u1) & (~u2))
    s1_local = np.flatnonzero(u1 ^ u2)
    s2_local = np.flatnonzero(u1 & u2)
    kk_idx = work[kk_local]
    s1_idx = work[s1_local]
    s2_idx = work[s2_local]
    y_KK = y[kk_idx]
    stratified = True
    try:
        kk_train, kk_val = train_test_split(
            kk_idx, test_size=0.10, stratify=y_KK, random_state=drug_split_seed,
        )
    except ValueError:
        stratified = False
        kk_train, kk_val = train_test_split(
            kk_idx, test_size=0.10, stratify=None, random_state=drug_split_seed,
        )
        print("[warn] KK 分层 90/10 失败（某类样本过少），已退化为非分层")
    print(
        f"[split] drug seed={drug_split_seed} | drugs={len(drugs)} U={len(U)} K={len(K)} "
        f"| KK={len(kk_idx):,} (train={len(kk_train):,} val={len(kk_val):,}) "
        f"S1={len(s1_idx):,} S2={len(s2_idx):,} stratified={stratified}"
    )
    return DrugSplits(
        drugs=drugs, U=U, K=K,
        kk_train=np.asarray(kk_train, dtype=np.int64),
        kk_val=np.asarray(kk_val, dtype=np.int64),
        s1=s1_idx, s2=s2_idx, kk=kk_idx,
        stratified=stratified, n_u=n_u,
    )


def evaluate_predictions(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_classes: int = N_CLASSES,
) -> dict:
    """复现 run_pipeline_final._evaluate_predictions，并追加序数/临床指标。"""
    y_test = np.asarray(y_test, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_prob.ndim != 2 or y_prob.shape[1] != n_classes:
        raise ValueError(f"y_prob 期望 (n,{n_classes})，得到 {y_prob.shape}")
    labels = list(range(n_classes))
    acc = float(accuracy_score(y_test, y_pred))
    f1_mac = float(f1_score(y_test, y_pred, average="macro", zero_division=0, labels=labels))
    f1_wt = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    per_class_f1 = f1_score(
        y_test, y_pred, average=None, zero_division=0, labels=labels,
    ).tolist()
    per_class_prec = precision_score(
        y_test, y_pred, average=None, zero_division=0, labels=labels,
    ).tolist()
    per_class_rec = recall_score(
        y_test, y_pred, average=None, zero_division=0, labels=labels,
    ).tolist()
    y_bin = label_binarize(y_test, classes=labels)
    if y_bin.ndim == 1:
        y_bin = np.eye(n_classes, dtype=y_bin.dtype)[y_test]
    per_class_auroc, per_class_ap = [], []
    for i in range(n_classes):
        if y_bin[:, i].sum() > 0:
            per_class_auroc.append(float(roc_auc_score(y_bin[:, i], y_prob[:, i])))
            per_class_ap.append(float(average_precision_score(y_bin[:, i], y_prob[:, i])))
        else:
            per_class_auroc.append(float("nan"))
            per_class_ap.append(float("nan"))
    macro_auroc = float(np.nanmean(per_class_auroc))
    macro_ap = float(np.nanmean(per_class_ap))

    adjacent_acc = float(np.mean(np.abs(y_pred - y_test) <= 1))
    high = y_test >= 3
    low = ~high
    high_risk_sens = (
        float(np.mean(y_pred[high] >= 3)) if np.any(high) else float("nan")
    )
    high_risk_spec = (
        float(np.mean(y_pred[low] < 3)) if np.any(low) else float("nan")
    )
    severe_underestimation_rate = (
        float(np.mean(y_pred[high] <= 1)) if np.any(high) else float("nan")
    )
    try:
        qwk = float(cohen_kappa_score(y_test, y_pred, weights="quadratic"))
    except Exception:
        qwk = float("nan")
    mae_ordinal = float(np.mean(np.abs(y_pred.astype(np.float64) - y_test.astype(np.float64))))

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
    )


def balanced_class_weights(y_train: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    y_train = np.asarray(y_train, dtype=np.int64)
    present = np.unique(y_train)
    w = np.ones(n_classes, dtype=np.float32)
    if len(present) == 0:
        return w
    computed = compute_class_weight("balanced", classes=present, y=y_train)
    for c, wi in zip(present, computed):
        w[int(c)] = float(wi)
    return w


class IndexBatcher:
    """memmap：升序切块、只打乱块顺序。RAM/GPU：真正的样本级 shuffle。"""

    def __init__(
        self,
        indices: np.ndarray,
        batch_size: int,
        shuffle: bool,
        seed: int,
        drop_last: bool = False,
        sample_shuffle: bool = False,
    ):
        idx = np.asarray(indices, dtype=np.int64)
        self.sample_shuffle = bool(sample_shuffle)
        self.base_idx = idx.copy() if self.sample_shuffle else np.sort(idx)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(int(seed))
        self.drop_last = bool(drop_last)

    def __len__(self) -> int:
        n = len(self.base_idx)
        if n == 0:
            return 0
        n_full = n // self.batch_size
        extra = 0 if (self.drop_last or n % self.batch_size == 0) else 1
        n_batches = n_full + extra
        if n_batches == 0 and n > 0:
            return 1
        return n_batches

    def __iter__(self) -> Iterator[np.ndarray]:
        n = len(self.base_idx)
        if n == 0:
            return
        if self.sample_shuffle:
            idx = self.base_idx.copy()
            if self.shuffle and n > 1:
                self.rng.shuffle(idx)
            if self.drop_last:
                n_use = (n // self.batch_size) * self.batch_size
                if n_use == 0:
                    n_use = n
                idx = idx[:n_use]
            for s in range(0, len(idx), self.batch_size):
                sl = idx[s: s + self.batch_size]
                if len(sl) == 0:
                    continue
                yield sl
            return
        starts = np.arange(0, n, self.batch_size)
        if self.drop_last:
            starts = np.array([s for s in starts if s + self.batch_size <= n], dtype=np.int64)
            if len(starts) == 0:
                starts = np.array([0], dtype=np.int64)
        if self.shuffle and len(starts) > 1:
            self.rng.shuffle(starts)
        for s in starts:
            sl = self.base_idx[s: s + self.batch_size]
            if len(sl) == 0:
                continue
            yield sl


# 兼容旧名
SortedIndexBatcher = IndexBatcher


def load_morgan_fingerprints(
    drug_ids: list[str],
    drugs_csv: Path,
) -> tuple[np.ndarray, dict[str, int]]:
    """为给定 DrugBank ID 计算 2048 位 Morgan 指纹（与 preprocess / gnn 相同生成器）。"""
    from graphtree_ddi.data.preprocess import smiles_to_fingerprint

    from tqdm import tqdm

    uniq = list(drug_ids)
    id_to_idx = {d: i for i, d in enumerate(uniq)}
    fp = np.zeros((len(uniq), 2048), dtype=np.float32)
    if not Path(drugs_csv).exists():
        print(f"[warn] 未找到 {drugs_csv}，单药指纹全 0")
        return fp, id_to_idx
    df = pd.read_csv(drugs_csv, usecols=["drugbank_id", "smiles"])
    wanted = set(uniq)
    df = df[df["drugbank_id"].astype(str).isin(wanted)]
    n_ok = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Morgan FP"):
        did = str(row.get("drugbank_id", ""))
        j = id_to_idx.get(did)
        if j is None:
            continue
        smi = row.get("smiles", "")
        if smi is None or (isinstance(smi, float) and np.isnan(smi)) or str(smi).strip() == "":
            continue
        arr = smiles_to_fingerprint(str(smi))
        if arr is None:
            continue
        fp[j] = np.asarray(arr, dtype=np.float32)
        n_ok += 1
    n_missing = len(uniq) - n_ok
    print(f"[fp] Morgan 指纹 {n_ok}/{len(uniq)} 有效（缺 {n_missing}）<- {Path(drugs_csv).name}")
    return fp, id_to_idx


def pair_entity_indices(
    drug1: np.ndarray,
    drug2: np.ndarray,
    id_to_idx: dict[str, int],
) -> np.ndarray:
    p1 = pd.Series(drug1, copy=False).map(id_to_idx)
    p2 = pd.Series(drug2, copy=False).map(id_to_idx)
    if p1.isna().any() or p2.isna().any():
        n_miss = int(p1.isna().sum() + p2.isna().sum())
        raise ValueError(f"有 {n_miss} 个药物 ID 不在指纹表中")
    return np.stack([p1.to_numpy(dtype=np.int32), p2.to_numpy(dtype=np.int32)], axis=1)


def tanimoto_ssp(fp: np.ndarray, ref_fp: np.ndarray) -> np.ndarray:
    """fp: [n, d] 0/1，ref_fp: [m, d] → [n, m] Tanimoto。"""
    a = np.asarray(fp, dtype=np.float32)
    b = np.asarray(ref_fp, dtype=np.float32)
    inter = a @ b.T
    a_sum = a.sum(axis=1, keepdims=True)
    b_sum = b.sum(axis=1, keepdims=True).T
    union = a_sum + b_sum - inter
    return inter / np.maximum(union, 1e-8)


def fit_ssp_pca(
    fp_matrix: np.ndarray,
    pair_idx: np.ndarray,
    train_indices: np.ndarray,
    n_components: int = 50,
    pca_random_state: int = 0,
) -> tuple[np.ndarray, int]:
    """训练集出现的药物为参照，SSP + PCA。返回每个实体的 PCA 向量 [n_drugs, k]。"""
    from sklearn.decomposition import PCA

    train_ent = np.unique(pair_idx[np.asarray(train_indices, dtype=np.int64)].reshape(-1))
    ref_fp = fp_matrix[train_ent]
    ssp_all = tanimoto_ssp(fp_matrix, ref_fp)
    ssp_train_drugs = ssp_all[train_ent]
    k = int(min(n_components, ssp_train_drugs.shape[0], ssp_train_drugs.shape[1]))
    k = max(k, 1)
    pca = PCA(n_components=k, random_state=pca_random_state, svd_solver="randomized")
    pca.fit(ssp_train_drugs)
    drug_pca = pca.transform(ssp_all).astype(np.float32)
    print(
        f"[ssp] 参照药物={len(train_ent)}  SSP={ssp_all.shape}  PCA->{k}  "
        f"explained_var_sum={float(np.sum(pca.explained_variance_ratio_)):.4f}"
    )
    return drug_pca, k


def prediction_path(
    out_dir: Path,
    method: str,
    fold: int,
    seed: int,
    test_set: str | None,
) -> Path:
    name = f"{method}_f{fold}_s{seed}"
    if test_set in ("S1", "S2"):
        name += f"_{test_set}"
    return Path(out_dir) / "predictions" / f"{name}.npz"


def save_predictions(
    out_dir: Path,
    method: str,
    fold: int,
    seed: int,
    test_idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    test_set: str | None = None,
) -> Path:
    path = prediction_path(out_dir, method, fold, seed, test_set)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        test_idx=np.asarray(test_idx, dtype=np.int64),
        y_true=np.asarray(y_true, dtype=np.int8),
        y_pred=np.asarray(y_pred, dtype=np.int8),
        y_prob=np.asarray(y_prob, dtype=np.float32),
    )
    return path


def run_key(method: str, fold: int, seed: int, split_mode: str, test_set: str) -> str:
    return f"{method}_f{fold}_s{seed}_{split_mode}_{test_set}"


def load_state(path: Path) -> dict:
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"runs": {}, "mode": None}


def save_state(path: Path, state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, allow_nan=True)
    tmp.replace(path)


def ensure_mode(state: dict, smoke: bool, path: Path) -> dict:
    cur = "smoke" if smoke else "full"
    prev = state.get("mode")
    if prev is not None and prev != cur:
        print(f"[state] 模式切换 {prev}→{cur}，重置 baselines_state.json")
        state = {"runs": {}, "mode": cur}
        save_state(path, state)
    else:
        state["mode"] = cur
        save_state(path, state)
    return state


def is_run_done(state: dict, key: str) -> bool:
    r = state.get("runs", {}).get(key)
    if r is None or "error" in r:
        return False
    return "macro_f1" in r


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if not np.isfinite(v):
            return None
        return v
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def save_raw_csv(state: dict, path: Path) -> None:
    rows = []
    for r in state.get("runs", {}).values():
        if "error" in r or "macro_f1" not in r:
            continue
        row = dict(
            method=r.get("method"),
            fold=r.get("fold"),
            seed=r.get("seed"),
            split_mode=r.get("split_mode"),
            test_set=r.get("test_set"),
            kge_entity=r.get("kge_entity"),
            preload=r.get("preload"),
            elapsed_sec=r.get("elapsed_sec"),
            best_epoch=r.get("best_epoch"),
            epoch_train_sec_mean=r.get("epoch_train_sec_mean"),
            epoch_n_seen_mean=r.get("epoch_n_seen_mean"),
        )
        for k in ALL_SCALAR_METRIC_KEYS:
            row[k] = r.get(k)
        pc = r.get("per_class_f1") or [None] * N_CLASSES
        for i in range(N_CLASSES):
            row[f"f1_c{i}"] = pc[i] if i < len(pc) else None
        rows.append(row)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
