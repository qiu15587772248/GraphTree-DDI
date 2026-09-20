"""
Hidden Dim 搜索脚本（独立运行）

支持两种模式：
  R-GCN + DDI-only 图（Exp-2 核心实验，默认）
  R-GAT + 混合图   （Exp-4 原始搜索）

用法:
    # 测试 Exp-2 R-GCN DDI-only 的最优 hidden_dim（默认）
    python scripts/ablation/gnn_dim_sweep.py

    # 测试特定维度
    python scripts/ablation/gnn_dim_sweep.py --dims 256 512 1024 2048

    # 测试 R-GAT 混合图（原始 Exp-4 搜索）
    python scripts/ablation/gnn_dim_sweep.py --model rgat --graph hybrid
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
import sys
import time
from pathlib import Path

import numpy as np
import torch


from graphtree_ddi.models.gnn import (
    RGAT_DDI,
    RGCN_DDI,
    NUM_RELATIONS,
    RESULTS_DIR,
    FocalLoss,
    SPLIT_RANDOM_STATE,
    _class_weights,
    _run_batches,
    _val_macro_f1,
    build_knowledge_graph,
    evaluate_model,
    load_ddi_pairs,
    set_reproducibility,
    split_data,
)


def _warmup_lr(optimizer, epoch, warmup_epochs, target_lr):
    if epoch <= warmup_epochs:
        lr = target_lr * epoch / max(warmup_epochs, 1)
        for pg in optimizer.param_groups:
            pg["lr"] = lr


def _load_existing_results(out_path: Path):
    """读取已有 dim_sweep_results.json，便于断点续跑与结果合并。"""
    if not out_path.exists():
        return {}
    try:
        with open(out_path, encoding="utf-8") as f:
            data = json.load(f)
        raw_results = data.get("results", {})
        parsed = {}
        for k, v in raw_results.items():
            try:
                parsed[int(k)] = v
            except (TypeError, ValueError):
                continue
        if parsed:
            print(f"检测到已有结果文件: {out_path}")
            print(f"  已有维度: {sorted(parsed.keys())}")
        return parsed
    except Exception as e:
        print(f"  [警告] 读取已有结果失败，忽略旧文件: {e}")
        return {}


def train_sweep(model, graph, drug_fp, pair_idx, y,
                idx_train, idx_val, device, config):
    """
    Dim sweep 专用训练循环。
    ReduceLROnPlateau + Warmup + AMP + 早停。
    max_epochs 较大，完全依赖早停结束。
    """
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    lr = config.get("lr", 5e-4)
    wd = config.get("weight_decay", 1e-4)
    patience = config.get("patience", 80)
    max_epochs = config.get("max_epochs", 99999)
    bs = config.get("batch_size", 4096)
    warmup_epochs = config.get("warmup_epochs", 10)
    focal_gamma = config.get("focal_gamma", 2.0)
    label_smoothing = config.get("label_smoothing", 0.1)
    min_epochs = config.get("min_epochs", 50)

    graph = graph.to(device)
    drug_fp = drug_fp.to(device)
    model = model.to(device)

    ei = graph.edge_index
    et = graph.edge_type
    class_w = _class_weights(y[idx_train])

    criterion = FocalLoss(weight=class_w.to(device), gamma=focal_gamma,
                          label_smoothing=label_smoothing)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                  patience=15, min_lr=1e-7)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_f1, best_epoch, wait = 0.0, 0, 0
    best_state = None
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "lr": []}

    print(f"\n{'='*60}")
    print(f"  训练  max={max_epochs}, patience={patience}, bs={bs}, lr={lr}")
    print(f"  Focal(γ={focal_gamma}) + LabelSmooth({label_smoothing}) "
          f"+ ReduceLROnPlateau")
    print(f"  Warmup={warmup_epochs}ep, AMP={'ON' if use_amp else 'OFF'}")
    print(f"{'='*60}")

    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        _warmup_lr(optimizer, epoch, warmup_epochs, lr)

        t_loss, _ = _run_batches(
            model, drug_fp, ei, et,
            pair_idx[idx_train], y[idx_train], criterion, device, bs,
            optimizer=optimizer, scaler=scaler)

        v_loss, _ = _run_batches(
            model, drug_fp, ei, et,
            pair_idx[idx_val], y[idx_val], criterion, device, bs,
            optimizer=None)
        v_f1 = _val_macro_f1(model, drug_fp, ei, et,
                             pair_idx[idx_val], y[idx_val], device)

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
    print(f"  训练完成  最佳 Macro-F1={best_f1:.4f} @ epoch {best_epoch}")
    print(f"  总耗时: {time.time()-t0:.1f}s\n")
    return model, history


def _estimate_vram_gb(hidden_dim: int, n_edges: int,
                      model_type: str = "rgat") -> float:
    """粗略估算 hidden_dim 所需显存（GB）。"""
    if model_type == "rgat":
        # GATv2Conv：每条边需存 hidden_dim 维注意力中间张量
        act_gb = 1.5 * n_edges * hidden_dim * 2 * 2 / 1e9
        param_gb = hidden_dim * hidden_dim * 3 * 4 / 1e9 * 3
    else:  # rgcn
        # RGCNConv：按节点聚合，无边级中间张量
        n_nodes = 20620  # 估算
        act_gb = n_nodes * hidden_dim * 2 * 2 / 1e9  # 前向+反向节点特征
        param_gb = hidden_dim * hidden_dim * 8 * 4 / 1e9  # num_bases=8
    return act_gb + param_gb + 2.0


def run_dim_sweep(dims=(256, 512, 1024, 2048), resume_existing=False,
                  model_type: str = "rgcn", graph_type: str = "ddi_only",
                  focal_gamma: float = 2.0, seed: int = 42,
                  deterministic_cuda: bool = True):
    include_bio = (graph_type == "hybrid")
    model_label = f"R-{'GAT' if model_type == 'rgat' else 'GCN'}"
    graph_label = "混合图" if include_bio else "仅DDI图"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"模型: {model_label}   图类型: {graph_label}   focal_gamma: {focal_gamma}")
    vram_total = 0.0
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        vram_free = torch.cuda.mem_get_info()[0] / 1e9
        print(f"  显存总量: {vram_total:.1f} GB  可用: {vram_free:.1f} GB")
    print(f"本次计划测试维度: {tuple(dims)}")
    det_msg = "开" if deterministic_cuda else "关（--fast_cuda）"
    print(f"可复现性: 训练 seed 基={seed}，每档维度用 seed+hidden_dim；"
          f"划分 random_state={SPLIT_RANDOM_STATE}（与 XGBoost 一致）；cudnn deterministic: {det_msg}")

    config = dict(
        lr=5e-4, weight_decay=1e-4,
        patience=80, max_epochs=99999,
        batch_size=4096, warmup_epochs=10,
        focal_gamma=focal_gamma, label_smoothing=0.1, min_epochs=50,
    )

    # 加载基础数据
    print("\n加载基础数据...")
    _, drug_fp, kg_meta = build_knowledge_graph(
        include_bio=True, ddi_pairs=None)
    n_drugs = kg_meta["n_drugs"]
    n_bio = kg_meta["n_enz"] + kg_meta["n_tp"] + kg_meta["n_tgt"]

    print("加载 DDI 对数据...")
    pair_idx, y, _ = load_ddi_pairs(kg_meta["drug_id_to_idx"])
    idx_train, idx_val, idx_test = split_data(y, random_state=SPLIT_RANDOM_STATE)
    train_pairs = pair_idx[idx_train]
    train_labels = y[idx_train]

    # 结果文件按模型+图类型区分，避免覆盖
    out_fname = f"dim_sweep_{model_type}_{graph_type}.json"
    out_path = RESULTS_DIR / out_fname
    sweep_results = (_load_existing_results(out_path)
                     if resume_existing else {})

    # 构建图（所有维度共用同一图）
    print(f"构建{graph_label}（所有维度共用同一图）...")
    graph_h, _, _ = build_knowledge_graph(
        ddi_pairs=train_pairs, ddi_labels=train_labels,
        include_bio=include_bio)
    n_edges = graph_h.edge_index.shape[1]
    print(f"  总边数: {n_edges:,}")

    for hdim in dims:
        edim = hdim // 2
        print(f"\n{'#'*60}")
        print(f"#  {model_label} + {graph_label}  Hidden Dim = {hdim}, Emb Dim = {edim}")
        print(f"{'#'*60}")

        if vram_total > 0:
            est_gb = _estimate_vram_gb(hdim, n_edges, model_type)
            print(f"  预估显存需求: {est_gb:.1f} GB / 可用: {vram_total:.1f} GB")

        try:
            set_reproducibility(seed + hdim, deterministic_cuda=deterministic_cuda)
            if model_type == "rgcn":
                model = RGCN_DDI(
                    n_drugs=n_drugs, n_bio=n_bio, hidden_dim=hdim,
                    emb_dim=edim, num_rels=NUM_RELATIONS, num_bases=8)
            else:
                model = RGAT_DDI(
                    n_drugs=n_drugs, n_bio=n_bio, hidden_dim=hdim,
                    emb_dim=edim, num_rels=NUM_RELATIONS)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  参数量: {n_params:,}")

            model, hist = train_sweep(
                model, graph_h, drug_fp, pair_idx, y,
                idx_train, idx_val, device, config)

            if device.type == "cuda":
                alloc = torch.cuda.max_memory_allocated() / 1e9
                reserv = torch.cuda.max_memory_reserved() / 1e9
                print(f"  峰值显存: 已分配={alloc:.1f}GB  已预留={reserv:.1f}GB")
                torch.cuda.reset_peak_memory_stats()

            tag = f"{model_label}(H={hdim})"
            _, _, _, metrics = evaluate_model(
                model, graph_h, drug_fp, pair_idx, y, idx_test, device, tag)

            best_epoch = int(np.argmax(hist["val_f1"])) + 1
            sweep_results[hdim] = {
                "model": model_label,
                "graph": graph_label,
                "emb_dim": edim,
                "n_params": n_params,
                "best_val_f1": float(max(hist["val_f1"])),
                "best_epoch": best_epoch,
                "total_epochs": len(hist["val_f1"]),
                "test_macro_f1": metrics["macro_f1"],
                "test_accuracy": metrics["accuracy"],
                "test_weighted_f1": metrics["weighted_f1"],
                "test_macro_auroc": metrics["macro_auroc"],
                "test_macro_ap": metrics["macro_ap"],
            }

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ⚠ OOM: hidden_dim={hdim} 超出显存，跳过")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                sweep_results[hdim] = {"error": "OOM"}
            else:
                raise

    # 汇总
    valid = {k: v for k, v in sweep_results.items() if "error" not in v}
    if not valid:
        print("\n所有维度均 OOM，请减小 dims 或增加显存")
        return

    best_dim = max(valid, key=lambda k: valid[k]["test_macro_f1"])

    print(f"\n{'='*72}")
    print(f"  Dim Sweep 结果汇总")
    print(f"{'='*72}")
    print(f"  {'H_dim':<8s} {'Params':>10s} {'Epochs':>8s} "
          f"{'ValF1':>8s} {'TestF1':>8s} {'TestAcc':>8s} "
          f"{'AUROC':>8s} {'AP':>8s}")
    print(f"  {'-'*66}")
    for hdim in sorted(sweep_results.keys()):
        r = sweep_results.get(hdim, {})
        if "error" in r:
            print(f"  {hdim:<8d} {'OOM':>10s}")
            continue
        marker = " ★" if hdim == best_dim else ""
        print(f"  {hdim:<8d} {r['n_params']:>10,} {r['total_epochs']:>8d} "
              f"{r['best_val_f1']:>8.4f} {r['test_macro_f1']:>8.4f} "
              f"{r['test_accuracy']:>8.4f} "
              f"{r['test_macro_auroc']:>8.4f} {r['test_macro_ap']:>8.4f}"
              f"{marker}")
    print(f"\n  >>> 最优 hidden_dim = {best_dim} <<<")
    print(f"{'='*72}\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model": model_label, "graph": graph_label,
                   "best_dim": best_dim,
                   "results": {str(k): v for k, v in sweep_results.items()}},
                  f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {out_path}")
    print(f"\n下一步: 在 gnn.py 中设置 hidden_dim={best_dim} 运行实验")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hidden Dim 搜索（支持 R-GCN / R-GAT，DDI-only / Hybrid 图）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 搜索 Exp-2 R-GCN DDI-only 最优维度（默认）
  python scripts/ablation/gnn_dim_sweep.py

  # 搜索 Exp-4 R-GAT 混合图最优维度
  python scripts/ablation/gnn_dim_sweep.py --model rgat --graph hybrid

  # 只测 1024 和 2048
  python scripts/ablation/gnn_dim_sweep.py --dims 1024 2048
        """,
    )
    parser.add_argument("--dims", type=int, nargs="+",
                        default=[256, 512, 1024, 2048],
                        help="测试的 hidden_dim 列表（默认: 256 512 1024 2048）")
    parser.add_argument("--model", choices=["rgcn", "rgat"], default="rgcn",
                        help="模型类型：rgcn（默认，Exp-2/3）或 rgat（Exp-4）")
    parser.add_argument("--graph", choices=["ddi_only", "hybrid"],
                        default="ddi_only",
                        help="图类型：ddi_only（默认，Exp-2）或 hybrid（Exp-3/4）")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal Loss gamma（R-GCN 用 2.0，R-GAT 用 0.9）")
    parser.add_argument("--resume", action="store_true",
                        help="读取已有结果，跳过已完成的维度")
    parser.add_argument("--seed", type=int, default=42,
                        help="训练随机种子基；每档 hidden_dim 使用 seed+dim")
    parser.add_argument("--fast_cuda", action="store_true",
                        help="关闭 cudnn deterministic（更快，数值可能略有差异）")
    args = parser.parse_args()
    run_dim_sweep(dims=tuple(args.dims), resume_existing=args.resume,
                  model_type=args.model, graph_type=args.graph,
                  focal_gamma=args.focal_gamma, seed=args.seed,
                  deterministic_cuda=not args.fast_cuda)
