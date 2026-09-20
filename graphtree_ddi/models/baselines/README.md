# 代表性基线（`graphtree_ddi/models/baselines/`）

五级 DDI 风险分类（0–4）的对照方法。只在本目录增改文件，不修改 `run_pipeline_final.py` / `gnn.py` / `baseline_xgboost.py`。

| 文件 | 作用 |
|---|---|
| `run_baselines.py` | CLI 主入口、训练/早停/断点续跑 |
| `common.py` | memmap 加载、pair/drug 划分、指标、预测与 state 保存 |
| `models.py` | MLP / DDIMDL-style / DeepDDI_SSP / DistMult / ComplEx |
| `README.md` | 本说明 |

数据目录（v1 默认 `data/processed/`，v2 换 `--data_dir data/processed/v2`）需含：`X_full.dat`、`y.npy`、`pairs.csv`、`meta.json`。单药 Morgan 指纹来自 `--drugs_csv`（默认 `data/raw/drugbank_drugs.csv`）。

---

## 1. 特征布局（以 `graphtree_ddi/data/preprocess.py` 为准）

`X_full.dat` 是 `float32` memmap，形状 `(n_samples, 4120)`，`n_samples` 以 `meta.json` 为准（v2 为 **1,332,174**，约 21.95 GB）。

**不是** `[drug_a 2060 ‖ drug_b 2060]`。`build_pair_features` 的实际布局：

| 切片 | 维数 | 含义 |
|---|---|---|
| `[0:2048]` | 2048 | `fp_product`：两药 Morgan（r=2, 2048 bit）逐位乘积 |
| `[2048:4096]` | 2048 | `fp_diff`：两药指纹绝对值差 |
| `[4096:4104]` | 8 | CYP 酶交互特征 |
| `[4104:4112]` | 8 | 转运体交互特征 |
| `[4112:4120]` | 8 | 靶点交互特征 |

因此无法从一行 X 唯一还原两药各自的位向量。`DeepDDI_SSP` / `DistMult` / `ComplEx` 所需的单药 2048 维指纹，用与预处理相同的 `smiles_to_fingerprint` 从 SMILES 重算。

---

## 2. 划分协议

### pair（与 `run_pipeline_final.py::prepare_global_data` 一致）

```
all_idx = np.arange(len(y))
idx_tv, idx_test = train_test_split(all_idx, test_size=0.2, stratify=y, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# 每折: idx_train_f = idx_tv[train_rel], idx_val_f = idx_tv[val_rel]
```

每次启动会在**全量 y**上重算该划分，并与独立复现函数比对索引（冒烟时全量 `n_test=266,455`，`sum(idx_test)=1,256,723,714`）。训练若加 `--smoke`，再对 1% 分层子集套用同一协议（子集索引会映射回 `pairs.csv` 行号）。

默认种子 `[42, 123, 2024]`。评估始终在封存测试集（pair）或 S1/S2（drug），早停看验证折 macro-F1。

### drug（与流水线执行者逐字一致）

```
drugs = sorted(set(pairs.drug1_id) | set(pairs.drug2_id))
rng = np.random.default_rng(drug_split_seed)
perm = rng.permutation(len(drugs))
n_u = int(round(0.2 * len(drugs)))
U = {drugs[i] for i in perm[:n_u]}          # 其余为 K
KK = 两药均在 K;  S1 = 恰一药在 U;  S2 = 两药均在 U
train_test_split(KK, test_size=0.10, stratify=y_KK, random_state=drug_split_seed)
```

`KK` 先按原始行号升序（`np.flatnonzero`）再交给 `train_test_split`，避免 `set` 无序。drug 模式**模型种子 = `drug_split_seed`**（忽略 `--seeds`）。同一套权重分别在 S1、S2 评估并分文件保存。

冒烟时药物宇宙取 1% 子集中出现的药物，否则 S2 极易为空。全量运行时 `indices` 为全集，与上述定义等价。

---

## 3. 方法定义与论文差异

通用：5 类 softmax + **类别加权交叉熵**；AdamW `lr=1e-3` `wd=1e-4`；grad clip 1.0；验证集 macro-F1 早停（全量默认 `epochs=50` `patience=10`，冒烟 2/2）。

`--preload`（默认 `cpu`）：

| 值 | 行为 |
|---|---|
| `cpu` | 启动时 `np.fromfile` 把整份 `X_full.dat` 读进 RAM。可用物理内存 < X+1 GB 则警告并回退 memmap；若当前工作集是子集，则把该子集顺序读进 RAM（`cpu_subset`）。训练按**样本级 shuffle** 从 RAM gather。 |
| `gpu` | 先看 `torch.cuda.mem_get_info()`：空闲需 ≥ X 字节 + 6 GB。不足则回退 `cpu`。每批 `index_select`，特征不经 PCIe。 |
| `memmap` | 不预载。按升序切块、只打乱块顺序，减轻随机磁盘读。 |

云端请用 `--preload cpu`（机器内存建议 ≥ 64 GB）。本机 34 GB 在可用内存 ≥ 22 GB 时可以全量预载。

### MLP

4120 维药对特征 → `1024-BN-ReLU-Dropout → 512-BN-ReLU-Dropout → 256-BN-ReLU-Dropout → 5`。无对应单篇「MLP 基线」论文，作为全连接对照。

### DDIMDL（**DDIMDL 式多模态 DNN**，不是逐字复现）

- 论文：Deng et al., *Bioinformatics* 2020。原文对化学/酶/通路/靶点 **Jaccard 相似度矩阵做 PCA**，每个模态一个子 DNN，再拼隐向量进联合 DNN；任务多为多类型 DDI。
- 这里：四个模态直接用 preprocess 的药对块（指纹积+差 4096、CYP 8、转运体 8、靶点 8），各子 DNN 后融合到 5 类。没有相似度矩阵、没有通路模态、不是 86 类型多标签。

### DeepDDI_SSP（仿 Ryu et al. 2018 *PNAS*）

- 原文：全库药物为参照算 Tanimoto SSP，PCA→50，药对拼接 100 维，较深 DNN，**86 类多标签** DDI。
- 这里：参照集 **仅训练集出现的药物**（避免泄漏）；指纹为 Morgan 2048；PCA 50（维数随参照数下调）；`512-256-128` DNN；**5 类单标签**。冷启动药物仍对训练集参照算 SSP 再 `PCA.transform`。

### DistMult / ComplEx（自实现，不用 PyKEEN）

- `--kge_entity lookup`（**pair 默认**）：`nn.Embedding(n_drugs, dim)` 按 `drug_index` 查表，这是文献 DistMult/ComplEx 的本义。维度 200；ComplEx 的 embedding 输出 `2×200`（Re‖Im）。
- `--kge_entity proj`（**drug 默认**）：`emb = W x_drug`，`x_drug` 为 2048 维 Morgan。U 药没有训练过的查找向量，必须用特征投影。
- `--kge_entity auto`：pair→lookup，drug→proj。实际用的值写入每条 `baselines_state.json` 的 `kge_entity`。
- DistMult：`sum_d h_d r_d t_d`。ComplEx：`Re(<h, r, conjugate(t)>)`。
- **不做负采样**（类 0 已是显式标签）。药对方向与 `pairs.csv` 的 `(drug1, drug2)` 一致。
- pair 的 lookup 对「只出现在测试对里的药物」仍有随机初始化、未更新的行（pair 划分不是按药切断）。drug 模式不要用 lookup。

---

## 4. 指标与产出

复用/对齐流水线：`accuracy`、`macro_f1`、`weighted_f1`、`macro AUROC (ovr)`、`macro AP`、每类 F1。额外：

| 键 | 定义 |
|---|---|
| `adjacent_acc` | mean(\|pred−true\| ≤ 1) |
| `high_risk_sens` | P(pred≥3 \| true≥3) |
| `high_risk_spec` | P(pred<3 \| true<3) |
| `severe_underestimation_rate` | P(pred≤1 \| true≥3) |
| `qwk` | `cohen_kappa_score(..., weights='quadratic')` |
| `mae_ordinal` | mean(\|pred−true\|) |

产出：

- `out_dir/predictions/{method}_f{fold}_s{seed}[_S1|_S2].npz`  
  键：`test_idx, y_true, y_pred, y_prob`（`test_idx` 为 `pairs.csv` 行号）
- `out_dir/baselines_state.json`（断点续跑；含 `method,fold,seed,split_mode,test_set,kge_entity,preload,elapsed_sec,best_epoch,epoch_stats` 及全部指标）
- `out_dir/baselines_raw_metrics.csv`

pair 预测文件无 `_S1/_S2` 后缀；drug 分别写 `_S1`、`_S2`。

---

## 5. 冒烟（本机 RTX 3060 6GB，仅此）

v2 数据、`--preload cpu`。两次分开跑，确认 lookup / proj：

```powershell
$env:PYTHONIOENCODING='utf-8'
# pair + lookup
conda run -n pytorch --no-capture-output python -u graphtree_ddi/models/baselines/run_baselines.py `
  --smoke --split_mode pair --kge_entity lookup --preload cpu `
  --data_dir data/processed/v2 `
  --out_dir models/results/baselines_smoke_lookup --device cuda --batch_size 512

# drug + proj
conda run -n pytorch --no-capture-output python -u graphtree_ddi/models/baselines/run_baselines.py `
  --smoke --split_mode drug --kge_entity proj --preload cpu `
  --data_dir data/processed/v2 `
  --out_dir models/results/baselines_smoke_proj --device cuda --batch_size 512
```

`--smoke`：分层 1%（v2 上 13,322 / 1,332,174）、2 epoch、1 折 1 种子。不要在本机跑全量 50 epoch。

实测（2026-09-17）：lookup pair 5/5 ok，墙钟 59.9 s，`kge_entity=lookup`；proj drug 10/10 ok，墙钟 29.4 s，`kge_entity=proj`。第一次冒烟时可用 RAM 17.59 GB，全量预载失败，改把 1% 子集载入 RAM；第二次可用 24.14 GB，成功 `np.fromfile` 全量 21.95 GB。2 epoch 指标不能当论文数字。

---

## 6. 云端完整运行

Linux，`PYTHONIOENCODING=utf-8`。内存建议 ≥ 64 GB，使用 `--preload cpu`。v2：

```bash
export PYTHONIOENCODING=utf-8
cd ddi_prediction

# pair：5 方法 × 5 fold × 3 seed = 75 次（KGE 默认 lookup）
python -u graphtree_ddi/models/baselines/run_baselines.py \
  --data_dir data/processed/v2 \
  --out_dir models/results/baselines_pair \
  --split_mode pair --kge_entity lookup --preload cpu \
  --methods MLP DDIMDL DeepDDI_SSP DistMult ComplEx \
  --folds 1 2 3 4 5 --seeds 42 123 2024 \
  --epochs 50 --patience 10 --batch_size 2048 --device cuda

# drug：3 个划分 × 5 方法 = 15 次训练（KGE 默认 proj；每方法评 S1+S2）
python -u graphtree_ddi/models/baselines/run_baselines.py \
  --data_dir data/processed/v2 \
  --out_dir models/results/baselines_drug \
  --split_mode drug --kge_entity proj --preload cpu \
  --methods MLP DDIMDL DeepDDI_SSP DistMult ComplEx \
  --drug_split_seed 42 123 2024 \
  --epochs 50 --patience 10 --batch_size 2048 --device cuda
```

中断后对同一 `--out_dir` 再跑会跳过已有 `macro_f1` 的 run。`--smoke` 与全量的 state `mode` 不同，混用会重置。

若云 GPU 显存 ≥ X+6 GB（约 28 GB 空闲），可改 `--preload gpu`。依赖：Python 3.10+、`torch`（CUDA）、`numpy`、`pandas`、`scikit-learn`、`rdkit`。不需要 PyG / PyKEEN。

---

## 7. 时长：v2 上 10% 子集实测吞吐（不是 1% 线性外推）

条件（2026-09-17，本机 RTX 3060 laptop）：`--data_dir data/processed/v2`，分层 10%（133,217 对），`--preload cpu` **成功把全量 21.95 GB 读进 RAM**（当时可用物理内存 24.46 GB / 总 34.14 GB），`batch_size 2048`，3 epoch，pair 折 1。KGE 为 `lookup`。

本机 RAM **在空闲时够 22 GB 预载**；可用内存偏低时会回退（见第 5 节）。云端不要依赖 32 GB 机器的余量，用 ≥ 64 GB。

v2 全量划分规模（只算索引、未训练）：pair 折训练 **852,591** 行；drug seed=42 的 KK-train **757,411** 行。

10% 折 1 训练 85,258 行，`drop_last` 后每 epoch 实际 **n_seen=83,968**。下面 `train_sec` 只含训练前向+反向（`cuda.synchronize`），不含验证。

| 方法 | ep1 train_s / n | ep2 | ep3 | 三轮 mean train_s | mean val_s |
|---|---|---|---|---|---|
| MLP | 1.597 / 83968 | 1.074 | 1.064 | 1.2450 | 0.2143 |
| DDIMDL | 0.885 | 0.867 | 0.863 | 0.8717 | 0.1827 |
| DeepDDI_SSP | 0.159 | 0.139 | 0.149 | 0.1490 | 0.0160 |
| DistMult | 0.185 | 0.118 | 0.110 | 0.1377 | 0.0070 |
| ComplEx | 0.299 | 0.248 | 0.252 | 0.2663 | 0.0103 |

外推：全量 drop_last 后 pair 每 epoch 处理 851,968 行（×10.146× 本次 n_seen）；drug 755,712 行（×9）。验证集按 213,148（pair）/ 84,157（drug）相对本次 21,315 行缩放。单次 run = 50 × (train_epoch + val_epoch)，这是 **50 epoch 上限**（早停只会更短）。

### 方法 × 单次 run × 总 run 数 = 总 GPU·h

| 方法 | pair 每 epoch (s) | pair 单次 50ep (h) | ×75 = pair GPU·h | drug 每 epoch (s) | drug 单次 50ep (h) | ×15 = drug GPU·h |
|---|---|---|---|---|---|---|
| MLP | 14.776 | 0.2052 | **15.391** | 12.051 | 0.1674 | 2.511 |
| DDIMDL | 10.671 | 0.1482 | **11.115** | 8.566 | 0.1190 | 1.785 |
| DeepDDI_SSP | 1.672 | 0.0232 | 1.741 | 1.404 | 0.0195 | 0.293 |
| DistMult | 1.467 | 0.0204 | 1.528 | 1.267 | 0.0176 | 0.264 |
| ComplEx | 2.806 | 0.0390 | 2.923 | 2.438 | 0.0339 | 0.508 |
| **合计** | | | **32.70** | | | **5.36** |

pair+drug **50 epoch 上限合计 38.1 GPU·h**。早停若在 ~15 epoch 触发，大约乘 0.3 → 约 11 GPU·h。MLP/DDIMDL 占绝大部分；KGE/SSP 可以忽略。drug 的 DistMult/ComplEx 生产配置是 `proj`，上表 drug 列用的是 pair 上测到的 lookup 每样本时间，数量级相同。

吞吐测量命令：

```bash
python -u graphtree_ddi/models/baselines/run_baselines.py \
  --data_dir data/processed/v2 --out_dir models/results/baselines_throughput10 \
  --split_mode pair --preload cpu --kge_entity lookup \
  --subset_frac 0.1 --folds 1 --seeds 42 --epochs 3 --batch_size 2048 --device cuda
```

---

## 8. CLI

```
--data_dir --out_dir --drugs_csv
--methods MLP DDIMDL DeepDDI_SSP DistMult ComplEx
--folds 1 2 3 4 5
--seeds 42 123 2024          # 仅 pair；drug 模式下忽略
--drug_split_seed 42 123 2024
--split_mode pair|drug
--preload {memmap,cpu,gpu}   # 默认 cpu
--kge_entity {auto,lookup,proj}  # auto: pair=lookup, drug=proj
--subset_frac 0.1            # 分层工作集；--smoke 默认 0.01
--device auto|cuda|cpu
--epochs --batch_size --patience --lr --dropout
--smoke
```
