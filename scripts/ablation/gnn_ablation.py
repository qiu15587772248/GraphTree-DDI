"""
GNN 消融验证脚本（gnn_ablation.py）
══════════════════════════════════════════════════════════════════════════════

专门用于验证以下优化建议的有效性：

  A. Exp-3 num_bases: 8 vs 17 vs 34（R-GCN basis 数对混合图的影响）
  B. norm_type: batch vs pair（PairNorm 是否改善 Exp-4 R-GAT 的稳定性）
  C. Exp-4 DropEdge: 0.0 / 0.2 / 0.4（对 R-GAT 的稳定化作用）
  D. Exp-4 self_loop_fill: zero vs mean（GATv2 self-loop 填充策略）
  E. 复现性验证：同 seed 连跑 2 次，看差异是否 <1e-3

每组子实验只跑一次（seed=42），目的是快速看"方向是否正确"，
不做多 seed 统计。统计验证留给 gnn.py 的 --seeds 主流程。

用法示例
--------
全部消融:
    python scripts/ablation/gnn_ablation.py --all

只跑某一个:
    python scripts/ablation/gnn_ablation.py --ablation A   # num_bases
    python scripts/ablation/gnn_ablation.py --ablation B   # norm_type
    python scripts/ablation/gnn_ablation.py --ablation C   # drop_edge
    python scripts/ablation/gnn_ablation.py --ablation D   # self_loop_fill
    python scripts/ablation/gnn_ablation.py --ablation E   # reproducibility

可用 --fast 降低 max_epochs 与 patience 做快速冒烟测试（不用于正式结果）。
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
import time
from pathlib import Path
from typing import Any


# 主脚本中的通用工具
from graphtree_ddi.models.gnn import (  # noqa: E402
    NUM_RELATIONS,
    RGCN_DDI,
    RGAT_DDI,
    RESULTS_DIR,
    build_knowledge_graph,
    load_ddi_pairs,
    split_data,
    set_reproducibility,
    train_model,
    evaluate_model,
    _clear_gpu,
    _train_run_seed,
)

import numpy as np  # noqa: E402
import torch  # noqa: E402


ABLATION_OUT = RESULTS_DIR / "ablation_results.json"


# ═══════════════════════════════════════════════════════════════
# 通用运行器：构造模型 + 训练 + 评估，返回指标 dict
# ═══════════════════════════════════════════════════════════════
def _run_one(
    *,
    label: str,
    model_cls: type[torch.nn.Module],
    model_kwargs: dict[str, Any],
    graph,
    drug_fp,
    pair_idx,
    y,
    idx_train,
    idx_val,
    idx_test,
    device,
    config: dict[str, Any],
    seed: int,
    exp_id: int = 4,
) -> dict[str, Any]:
    """跑单个配置的 Exp-like 实验并返回指标。"""
    print(f"\n{'★' * 72}")
    print(f"  运行配置: {label}")
    print(f"{'★' * 72}")
    set_reproducibility(_train_run_seed(seed, exp_id),
                        deterministic_cuda=True,
                        strict_deterministic=config.get("strict_deterministic", False))
    _clear_gpu()
    model = model_cls(**model_kwargs)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {n_p:,}")

    t0 = time.time()
    model, hist = train_model(
        model, graph, drug_fp, pair_idx, y,
        idx_train, idx_val, device, config,
    )
    _, _, _, m = evaluate_model(
        model, graph, drug_fp, pair_idx, y, idx_test, device, label,
    )
    elapsed = time.time() - t0
    del model
    _clear_gpu()
    return dict(
        label=label,
        accuracy=m["accuracy"],
        macro_f1=m["macro_f1"],
        weighted_f1=m["weighted_f1"],
        macro_auroc=m["macro_auroc"],
        macro_ap=m["macro_ap"],
        per_class_f1=[m["per_class"][i]["f1"] for i in range(5)],
        elapsed_sec=round(elapsed, 1),
        config_snapshot={
            "model_kwargs": {k: v for k, v in model_kwargs.items()
                             if k not in ("n_drugs", "n_bio")},
            "config": {k: v for k, v in config.items()},
        },
    )


# ═══════════════════════════════════════════════════════════════
# 载入数据（所有消融共享）
# ═══════════════════════════════════════════════════════════════
def _prepare(need_hybrid: bool = True, need_ddi: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, drug_fp, kg_meta = build_knowledge_graph(include_bio=True, ddi_pairs=None)
    n_drugs = kg_meta["n_drugs"]
    n_bio = kg_meta["n_enz"] + kg_meta["n_tp"] + kg_meta["n_tgt"]
    pair_idx, y, _ = load_ddi_pairs(kg_meta["drug_id_to_idx"])
    idx_train, idx_val, idx_test = split_data(y)

    graph_ddi, graph_hybrid = None, None
    if need_ddi:
        print("\n[ablation] 构建仅DDI图...")
        graph_ddi, _, _ = build_knowledge_graph(
            ddi_pairs=pair_idx[idx_train], ddi_labels=y[idx_train],
            include_bio=False)
    if need_hybrid:
        print("\n[ablation] 构建混合图...")
        graph_hybrid, _, _ = build_knowledge_graph(
            ddi_pairs=pair_idx[idx_train], ddi_labels=y[idx_train],
            include_bio=True)

    return dict(
        device=device, drug_fp=drug_fp, kg_meta=kg_meta,
        n_drugs=n_drugs, n_bio=n_bio,
        pair_idx=pair_idx, y=y,
        idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,
        graph_ddi=graph_ddi, graph_hybrid=graph_hybrid,
    )


# ═══════════════════════════════════════════════════════════════
# 统一训练配置（消融共享）
# ═══════════════════════════════════════════════════════════════
def _base_config(fast: bool = False) -> dict[str, Any]:
    """fast=True 时缩短训练，仅用于冒烟验证逻辑是否正常。"""
    if fast:
        return dict(
            lr=5e-4, weight_decay=1e-4,
            max_epochs=30, patience=10, min_epochs=5,
            batch_size=4096, warmup_epochs=3,
            label_smoothing=0.1,
            plateau_factor=0.5, plateau_patience=5,
            use_amp=True, drop_edge=0.0, grad_clip=1.0,
        )
    return dict(
        lr=5e-4, weight_decay=1e-4,
        max_epochs=1000, patience=50, min_epochs=30,
        batch_size=4096, warmup_epochs=10,
        label_smoothing=0.1,
        plateau_factor=0.5, plateau_patience=15,
        use_amp=True, drop_edge=0.0, grad_clip=1.0,
    )


# ═══════════════════════════════════════════════════════════════
# A. num_bases 消融（Exp-3 R-GCN 混合图）
# ═══════════════════════════════════════════════════════════════
def ablation_A_num_bases(ctx: dict, fast: bool = False) -> list[dict]:
    """测试 R-GCN 在 34 种关系的混合图上 basis 数对性能的影响。"""
    print(f"\n{'═' * 72}")
    print(f"  消融 A: Exp-3 R-GCN(Hybrid) num_bases ∈ {{8, 17, 34}}")
    print(f"  假设：num_bases 不足时 R-GCN 难以区分 34 种关系，导致生物网络信号被'压缩'丢失")
    print(f"{'═' * 72}")
    results = []
    HIDDEN = 2048
    EMB = HIDDEN // 2
    cfg = {**_base_config(fast), "focal_gamma": 2.0}
    for nb in (8, 17, 34):
        mk = dict(
            n_drugs=ctx["n_drugs"], n_bio=ctx["n_bio"],
            hidden_dim=HIDDEN, emb_dim=EMB,
            num_rels=NUM_RELATIONS, num_bases=nb,
            dropout=0.3, norm_type="batch",
        )
        r = _run_one(
            label=f"Exp-3 R-GCN(Hybrid) num_bases={nb}",
            model_cls=RGCN_DDI, model_kwargs=mk,
            graph=ctx["graph_hybrid"], drug_fp=ctx["drug_fp"],
            pair_idx=ctx["pair_idx"], y=ctx["y"],
            idx_train=ctx["idx_train"], idx_val=ctx["idx_val"],
            idx_test=ctx["idx_test"], device=ctx["device"],
            config=cfg, seed=42, exp_id=3,
        )
        results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════
# B. norm_type 消融（Exp-4 R-GAT）
# ═══════════════════════════════════════════════════════════════
def ablation_B_norm_type(ctx: dict, fast: bool = False) -> list[dict]:
    """BatchNorm vs PairNorm：PairNorm 是否能缓解 R-GAT 在密集图上的不稳定。"""
    print(f"\n{'═' * 72}")
    print(f"  消融 B: Exp-4 R-GAT(Hybrid) norm_type ∈ {{batch, pair, layer}}")
    print(f"  假设：BatchNorm 在 GNN 全图前向时把稀有生物节点压到与药物节点同量级，削弱区分度；")
    print(f"       PairNorm 缓解 oversmoothing")
    print(f"{'═' * 72}")
    results = []
    HIDDEN = 512
    EMB = HIDDEN // 2
    cfg = {**_base_config(fast), "focal_gamma": 0.9}
    for nt in ("batch", "pair", "layer"):
        mk = dict(
            n_drugs=ctx["n_drugs"], n_bio=ctx["n_bio"],
            hidden_dim=HIDDEN, emb_dim=EMB,
            num_rels=NUM_RELATIONS, edge_emb_dim=64, heads=4,
            dropout=0.3, norm_type=nt, pairnorm_scale=1.0,
            self_loop_fill="zero",
        )
        r = _run_one(
            label=f"Exp-4 R-GAT(Hybrid) norm={nt}",
            model_cls=RGAT_DDI, model_kwargs=mk,
            graph=ctx["graph_hybrid"], drug_fp=ctx["drug_fp"],
            pair_idx=ctx["pair_idx"], y=ctx["y"],
            idx_train=ctx["idx_train"], idx_val=ctx["idx_val"],
            idx_test=ctx["idx_test"], device=ctx["device"],
            config=cfg, seed=42, exp_id=4,
        )
        results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════
# C. DropEdge 消融（Exp-4 R-GAT）
# ═══════════════════════════════════════════════════════════════
def ablation_C_drop_edge(ctx: dict, fast: bool = False) -> list[dict]:
    """DropEdge 是否稳定 R-GAT 训练、提升 Macro-F1。"""
    print(f"\n{'═' * 72}")
    print(f"  消融 C: Exp-4 R-GAT(Hybrid) drop_edge ∈ {{0.0, 0.2, 0.4}}")
    print(f"  假设：随机丢边可降低 GAT 在密集邻居上的 attention softmax 崩塌")
    print(f"{'═' * 72}")
    results = []
    HIDDEN = 512
    EMB = HIDDEN // 2
    for de in (0.0, 0.2, 0.4):
        cfg = {**_base_config(fast), "focal_gamma": 0.9, "drop_edge": de}
        mk = dict(
            n_drugs=ctx["n_drugs"], n_bio=ctx["n_bio"],
            hidden_dim=HIDDEN, emb_dim=EMB,
            num_rels=NUM_RELATIONS, edge_emb_dim=64, heads=4,
            dropout=0.3, norm_type="batch",
            self_loop_fill="zero",
        )
        r = _run_one(
            label=f"Exp-4 R-GAT(Hybrid) drop_edge={de}",
            model_cls=RGAT_DDI, model_kwargs=mk,
            graph=ctx["graph_hybrid"], drug_fp=ctx["drug_fp"],
            pair_idx=ctx["pair_idx"], y=ctx["y"],
            idx_train=ctx["idx_train"], idx_val=ctx["idx_val"],
            idx_test=ctx["idx_test"], device=ctx["device"],
            config=cfg, seed=42, exp_id=4,
        )
        results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════
# D. self_loop_fill 消融（Exp-4 R-GAT）
# ═══════════════════════════════════════════════════════════════
def ablation_D_self_loop_fill(ctx: dict, fast: bool = False) -> list[dict]:
    """GATv2 self-loop 的 edge_attr 填 0 vs 填均值，哪种更稳。"""
    print(f"\n{'═' * 72}")
    print(f"  消融 D: Exp-4 R-GAT(Hybrid) self_loop_fill ∈ {{zero, mean}}")
    print(f"  假设：fill=zero 时 self-loop 的 edge_attr 范数远小于真实边，破坏 attention 均衡")
    print(f"{'═' * 72}")
    results = []
    HIDDEN = 512
    EMB = HIDDEN // 2
    cfg = {**_base_config(fast), "focal_gamma": 0.9}
    for slf in ("zero", "mean"):
        mk = dict(
            n_drugs=ctx["n_drugs"], n_bio=ctx["n_bio"],
            hidden_dim=HIDDEN, emb_dim=EMB,
            num_rels=NUM_RELATIONS, edge_emb_dim=64, heads=4,
            dropout=0.3, norm_type="batch",
            self_loop_fill=slf,
        )
        r = _run_one(
            label=f"Exp-4 R-GAT(Hybrid) self_loop={slf}",
            model_cls=RGAT_DDI, model_kwargs=mk,
            graph=ctx["graph_hybrid"], drug_fp=ctx["drug_fp"],
            pair_idx=ctx["pair_idx"], y=ctx["y"],
            idx_train=ctx["idx_train"], idx_val=ctx["idx_val"],
            idx_test=ctx["idx_test"], device=ctx["device"],
            config=cfg, seed=42, exp_id=4,
        )
        results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════
# F. Exp-4 精细超参扫描：hidden_dim ∈ {256,384,512,768} × γ ∈ {0.5,0.9,1.5}
# ═══════════════════════════════════════════════════════════════
def ablation_F_exp4_grid(ctx: dict, fast: bool = False) -> list[dict]:
    """Exp-4 R-GAT 的 hidden_dim 和 focal_gamma 精细网格搜索。

    已知 H>=1024 会崩溃，这里只扫 256~768 的甜蜜区间。
    """
    print(f"\n{'═' * 72}")
    print(f"  消融 F: Exp-4 R-GAT 精细网格 (hidden_dim × focal_gamma)")
    print(f"  扫描: H ∈ {{256, 384, 512, 768}} × γ ∈ {{0.5, 0.9, 1.5}}")
    print(f"{'═' * 72}")
    results = []
    H_grid = (256, 384, 512, 768)
    gamma_grid = (0.5, 0.9, 1.5)
    base_cfg = _base_config(fast)
    for H in H_grid:
        # H 必须被 heads=4 整除
        if H % 4 != 0:
            continue
        EMB = H // 2
        for gm in gamma_grid:
            cfg = {**base_cfg, "focal_gamma": gm}
            mk = dict(
                n_drugs=ctx["n_drugs"], n_bio=ctx["n_bio"],
                hidden_dim=H, emb_dim=EMB,
                num_rels=NUM_RELATIONS, edge_emb_dim=64, heads=4,
                dropout=0.3, norm_type="batch",
                self_loop_fill="zero",
            )
            try:
                r = _run_one(
                    label=f"Exp-4 R-GAT H={H} γ={gm}",
                    model_cls=RGAT_DDI, model_kwargs=mk,
                    graph=ctx["graph_hybrid"], drug_fp=ctx["drug_fp"],
                    pair_idx=ctx["pair_idx"], y=ctx["y"],
                    idx_train=ctx["idx_train"], idx_val=ctx["idx_val"],
                    idx_test=ctx["idx_test"], device=ctx["device"],
                    config=cfg, seed=42, exp_id=4,
                )
                results.append(r)
            except torch.cuda.OutOfMemoryError as e:
                print(f"  [OOM] H={H} γ={gm} 显存不足，跳过: {e}")
                _clear_gpu()
    return results


# ═══════════════════════════════════════════════════════════════
# G. 解码器消融：MLP vs Bilinear vs DistMult（在 Exp-4 R-GAT 上对比）
# ═══════════════════════════════════════════════════════════════
def ablation_G_decoder(ctx: dict, fast: bool = False) -> list[dict]:
    """比较三种解码器在 Exp-4 R-GAT 上的表现。"""
    print(f"\n{'═' * 72}")
    print(f"  消融 G: Exp-4 R-GAT 解码器 ∈ {{mlp, bilinear, distmult}}")
    print(f"  假设: 关系特定的 Bilinear/DistMult 比拼接 MLP 更适合多级风险分类")
    print(f"{'═' * 72}")
    results = []
    HIDDEN = 512
    EMB = HIDDEN // 2
    cfg = {**_base_config(fast), "focal_gamma": 0.9}
    for dec in ("mlp", "bilinear", "distmult"):
        mk = dict(
            n_drugs=ctx["n_drugs"], n_bio=ctx["n_bio"],
            hidden_dim=HIDDEN, emb_dim=EMB,
            num_rels=NUM_RELATIONS, edge_emb_dim=64, heads=4,
            dropout=0.3, norm_type="batch", self_loop_fill="zero",
            decoder_type=dec,
        )
        r = _run_one(
            label=f"Exp-4 R-GAT decoder={dec}",
            model_cls=RGAT_DDI, model_kwargs=mk,
            graph=ctx["graph_hybrid"], drug_fp=ctx["drug_fp"],
            pair_idx=ctx["pair_idx"], y=ctx["y"],
            idx_train=ctx["idx_train"], idx_val=ctx["idx_val"],
            idx_test=ctx["idx_test"], device=ctx["device"],
            config=cfg, seed=42, exp_id=4,
        )
        results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════
# H. emb_dim_mode 消融：half vs full （Exp-3 R-GCN 混合图）
# ═══════════════════════════════════════════════════════════════
def ablation_H_emb_dim(ctx: dict, fast: bool = False) -> list[dict]:
    """emb_dim = hidden//2 (默认) vs emb_dim = hidden (第二层残差启用)。"""
    print(f"\n{'═' * 72}")
    print(f"  消融 H: Exp-3 R-GCN(Hybrid) emb_dim_mode ∈ {{half, full}}")
    print(f"  假设: full 模式下第二层 conv 输出维度不收缩, PairMLP 输入 4H 维更丰富")
    print(f"{'═' * 72}")
    results = []
    HIDDEN = 2048
    cfg = {**_base_config(fast), "focal_gamma": 2.0}
    for mode, emb in (("half", HIDDEN // 2), ("full", HIDDEN)):
        mk = dict(
            n_drugs=ctx["n_drugs"], n_bio=ctx["n_bio"],
            hidden_dim=HIDDEN, emb_dim=emb,
            num_rels=NUM_RELATIONS, num_bases=17,
            dropout=0.3, norm_type="batch",
        )
        try:
            r = _run_one(
                label=f"Exp-3 R-GCN emb_dim={mode}(={emb})",
                model_cls=RGCN_DDI, model_kwargs=mk,
                graph=ctx["graph_hybrid"], drug_fp=ctx["drug_fp"],
                pair_idx=ctx["pair_idx"], y=ctx["y"],
                idx_train=ctx["idx_train"], idx_val=ctx["idx_val"],
                idx_test=ctx["idx_test"], device=ctx["device"],
                config=cfg, seed=42, exp_id=3,
            )
            results.append(r)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  [OOM] emb_dim={emb} 显存不足，跳过: {e}")
            _clear_gpu()
    return results


# ═══════════════════════════════════════════════════════════════
# E. 复现性验证：同 seed 连跑两次（关闭 AMP + strict_deterministic）
# ═══════════════════════════════════════════════════════════════
def ablation_E_reproducibility(ctx: dict, fast: bool = False) -> list[dict]:
    """检查"seed=42 + no_amp + strict_deterministic"的组合能否达到 <1e-3 的差异。"""
    print(f"\n{'═' * 72}")
    print(f"  消融 E: 复现性验证 Exp-4 R-GAT，同 seed 连跑两次")
    print(f"  配置：no_amp + strict_deterministic + seed=42")
    print(f"  目标：两次的 Macro-F1 差异 < 0.001")
    print(f"{'═' * 72}")
    results = []
    HIDDEN = 512
    EMB = HIDDEN // 2
    # 使用更短训练以控制时间（反正只看差异）
    cfg_base = _base_config(fast=True if not fast else fast)
    cfg = {**cfg_base, "focal_gamma": 0.9, "use_amp": False,
           "strict_deterministic": True}
    mk = dict(
        n_drugs=ctx["n_drugs"], n_bio=ctx["n_bio"],
        hidden_dim=HIDDEN, emb_dim=EMB,
        num_rels=NUM_RELATIONS, edge_emb_dim=64, heads=4,
        dropout=0.3, norm_type="batch", self_loop_fill="zero",
    )
    for run_i in (1, 2):
        r = _run_one(
            label=f"Exp-4 repro run {run_i}",
            model_cls=RGAT_DDI, model_kwargs=mk,
            graph=ctx["graph_hybrid"], drug_fp=ctx["drug_fp"],
            pair_idx=ctx["pair_idx"], y=ctx["y"],
            idx_train=ctx["idx_train"], idx_val=ctx["idx_val"],
            idx_test=ctx["idx_test"], device=ctx["device"],
            config=cfg, seed=42, exp_id=4,
        )
        results.append(r)
    diff = abs(results[0]["macro_f1"] - results[1]["macro_f1"])
    print(f"\n  [消融 E 结果] Macro-F1 差异: {diff:.6f}  "
          f"{'✓ 合格 (<1e-3)' if diff < 1e-3 else '✗ 仍有波动'}")
    return results


# ═══════════════════════════════════════════════════════════════
# 打印与保存
# ═══════════════════════════════════════════════════════════════
def _print_table(title: str, results: list[dict]):
    print(f"\n{'─' * 72}")
    print(f"  [{title}] 结果汇总")
    print(f"{'─' * 72}")
    print(f"  {'配置':<46s} {'Macro-F1':>10s} {'Acc':>8s} {'AUROC':>8s}")
    for r in results:
        print(f"  {r['label']:<46s} {r['macro_f1']:>10.4f} "
              f"{r['accuracy']:>8.4f} {r['macro_auroc']:>8.4f}")
    print(f"{'─' * 72}\n")


def _save_all(all_results: dict):
    out = {}
    # 已有结果累积（不覆盖其它消融）
    if ABLATION_OUT.exists():
        try:
            with open(ABLATION_OUT, encoding="utf-8") as f:
                out = json.load(f)
        except Exception:
            out = {}
    out.update(all_results)
    ABLATION_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(ABLATION_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n所有消融结果已保存: {ABLATION_OUT}")


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="GNN 消融验证脚本")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["A", "B", "C", "D", "E", "F", "G", "H"],
                        help="只跑某一组消融；省略则需要 --all")
    parser.add_argument("--all", action="store_true",
                        help="跑全部八组消融（A~H，注意 F 最耗时 12 次训练）")
    parser.add_argument("--fast", action="store_true",
                        help="缩短训练（max_epochs=30, patience=10），仅用于冒烟测试")
    args = parser.parse_args()

    if not args.all and not args.ablation:
        parser.error("必须指定 --all 或 --ablation A/B/C/D/E/F/G/H")

    need_ddi = False
    need_hybrid = True  # 所有消融都用混合图
    ctx = _prepare(need_hybrid=need_hybrid, need_ddi=need_ddi)

    all_results = {}
    if args.all or args.ablation == "A":
        r = ablation_A_num_bases(ctx, args.fast)
        _print_table("A: num_bases 消融", r)
        all_results["A_num_bases"] = r
    if args.all or args.ablation == "B":
        r = ablation_B_norm_type(ctx, args.fast)
        _print_table("B: norm_type 消融", r)
        all_results["B_norm_type"] = r
    if args.all or args.ablation == "C":
        r = ablation_C_drop_edge(ctx, args.fast)
        _print_table("C: drop_edge 消融", r)
        all_results["C_drop_edge"] = r
    if args.all or args.ablation == "D":
        r = ablation_D_self_loop_fill(ctx, args.fast)
        _print_table("D: self_loop_fill 消融", r)
        all_results["D_self_loop_fill"] = r
    if args.all or args.ablation == "E":
        r = ablation_E_reproducibility(ctx, args.fast)
        _print_table("E: 复现性验证", r)
        all_results["E_reproducibility"] = r
    if args.all or args.ablation == "F":
        r = ablation_F_exp4_grid(ctx, args.fast)
        _print_table("F: Exp-4 精细网格", r)
        all_results["F_exp4_grid"] = r
    if args.all or args.ablation == "G":
        r = ablation_G_decoder(ctx, args.fast)
        _print_table("G: 解码器消融", r)
        all_results["G_decoder"] = r
    if args.all or args.ablation == "H":
        r = ablation_H_emb_dim(ctx, args.fast)
        _print_table("H: emb_dim 消融", r)
        all_results["H_emb_dim"] = r

    _save_all(all_results)


if __name__ == "__main__":
    main()
