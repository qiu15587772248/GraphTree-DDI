# DDInter 外部验证数据准备

本目录存放「冻结的 DrugBank 模型 → DDInter 跨数据库外部验证」所需的**数据准备产物**。此处**不做模型推理**，只完成下载、清洗、ID 映射与集合关系标注。

生成时间（UTC+8）：2026-09-17T02:47:54+08:00

## 1. 来源与许可

- **数据库**：DDInter 2.0 官方按 ATC 分类提供的相互作用 CSV 批量下载（[ddinter2.scbdd.com](https://ddinter2.scbdd.com/)）。
- **直链格式**：`https://ddinter2.scbdd.com/static/media/download/ddinter_downloads_code_{X}.csv`，`X ∈ {A,B,C,D,G,H,J,L,M,N,P,R,S,V}`（共 14 个）。下载页未全部列出，但其余路径实测可访问。
- **回退**：若 2.0 直链失败，改用 1.0 站点同名路径 `https://ddinter.scbdd.com/static/media/download/...`。先前实测 1.0 与 2.0 文件哈希一致。
- **许可**：CC BY-NC-SA 4.0（非商业、署名、相同方式共享）。论文与仓库使用时需保留署名与许可声明。
- **表头**：`DDInterID_A, Drug_A, DDInterID_B, Drug_B, Level`；`Level ∈ {Major, Moderate, Minor, Unknown}`。文件**不含** DrugBank ID。
- **药物详情页**（例如 `https://ddinter2.scbdd.com/server/drug-detail/DDInter1/`）在 Useful Links 中外链到 `https://go.drugbank.com/drugs/DBxxxxx`。

### 1.1 下载记录

- 请求文件数：14
- 成功（含本地已存在）：14
- 改用 1.0 回退：0
- 失败：0
- 14 个文件数据行合计（含 ATC 交叉重复，未去重）：507,655

明细见 `raw/download_manifest.csv`。

| 文件 | ATC | 来源 | SHA-256 | 数据行 | 下载时间 (UTC+8) | 状态 |
|---|---|---|---|---|---|---|
| ddinter_downloads_code_A.csv | A | ddinter2 | `a22ca451d2b755ca2331886f7e00540c86f555f9f55704a59a19c691251f52e0` | 56,367 | 2026-09-17T02:34:38+08:00 | ok |
| ddinter_downloads_code_B.csv | B | ddinter2 | `76de5115a55587f0e822e1096b684fd3ddde058fbefcbb19896df58820ace130` | 15,140 | 2026-09-17T02:34:44+08:00 | ok |
| ddinter_downloads_code_C.csv | C | ddinter2 | `0885978959af84e183cfc86c8bcb48536f690a4c7e951eff06607a5616883aeb` | 60,454 | 2026-09-17T02:34:46+08:00 | ok |
| ddinter_downloads_code_D.csv | D | ddinter2 | `c0627ec39965dbe27829e4934cd20d71eb079f4e373287678737bba9423a306a` | 25,681 | 2026-09-17T02:34:53+08:00 | ok |
| ddinter_downloads_code_G.csv | G | ddinter2 | `6a4d8d1eaac6da1ffc54bd23a625f9c5e28e3ca37bd26b3bca4d2cc26505fb99` | 34,282 | 2026-09-17T02:34:56+08:00 | ok |
| ddinter_downloads_code_H.csv | H | ddinter2 | `f0f925f0ba1ee68c4668d3e7a6732719b34623b22a0f5d017f9090e59f63c88e` | 11,727 | 2026-09-17T02:35:00+08:00 | ok |
| ddinter_downloads_code_J.csv | J | ddinter2 | `e5dbb8e5cb3179905b426ee4a2ee5c93756c38cdd54215a55c5b8255a5b14e04` | 44,414 | 2026-09-17T02:35:02+08:00 | ok |
| ddinter_downloads_code_L.csv | L | ddinter2 | `f54f4486cc00344f8c86508c31c2ca3fad6d1d37ec3af5bcf609b4bbda5507ef` | 65,389 | 2026-09-17T02:35:10+08:00 | ok |
| ddinter_downloads_code_M.csv | M | ddinter2 | `2a57eb78818803b4662116490023c20c8bbfe145e60c35680a0efb8ac635a0ad` | 22,097 | 2026-09-17T02:35:18+08:00 | ok |
| ddinter_downloads_code_N.csv | N | ddinter2 | `854733ff382210b030bcece135113f01ca3f6924d3803feca2ff71c1d939c797` | 91,595 | 2026-09-17T02:35:21+08:00 | ok |
| ddinter_downloads_code_P.csv | P | ddinter2 | `83f8c0edb20d09ef29b8500b48f9623e208379525688ade9e70a7df2d8749d55` | 5,492 | 2026-09-17T02:35:30+08:00 | ok |
| ddinter_downloads_code_R.csv | R | ddinter2 | `1c39c3d4a6e41659b7a988538d7f363867592abc6f41a815de8746f6e2150574` | 30,563 | 2026-09-17T02:35:31+08:00 | ok |
| ddinter_downloads_code_S.csv | S | ddinter2 | `c3888df996d97ebf1346b11a98df0b334cfa3c150a89a31150ffd354eebcbfe5` | 32,430 | 2026-09-17T02:35:35+08:00 | ok |
| ddinter_downloads_code_V.csv | V | ddinter2 | `353973fb300453946aea95733e8fb52338e4b59426d2ce6ca318363d4f84f10a` | 12,024 | 2026-09-17T02:35:39+08:00 | ok |

## 2. 清洗规则

输入：`raw/ddinter_downloads_code_{A–V}.csv`。

1. 去掉两端 ID 缺失或 `DDInterID_A == DDInterID_B` 的行。
2. 将无序对规范为数值 ID 较小者在前（`DDInterID_A` 的数字部分 `<` `DDInterID_B`）。
3. **ATC 交叉重复**：同一无序对出现在多个 ATC 文件且 `Level` 相同，只保留一条（按 ATC 字母序 A→V 取首次出现的药名写法）。
4. **Level 冲突（保守）**：同一无序对出现不同 `Level` 时，整对写入 `conflicts.csv`，**不进入**主表 `ddinter_pairs_clean.csv`。
5. 不根据药名排序，不以 DrugBank ID 作为本步键。

清洗结果：

- 原始行数（14 文件合计）：507,655
- Level 冲突无序对：0（涉及行 0）
- 主表无序对数：234,981
- 主表唯一 DDInter 药物数：1,971

Level 分布（主表）：

| Level | 条数 |
|---|---|
| Major | 39,082 |
| Moderate | 143,748 |
| Minor | 9,736 |
| Unknown | 42,415 |

说明：官方 2.0 论文数字为 302,516 条；本批官方 CSV 去重后约 234,981 条无序对，**接近 DDInter 1.0 公开规模（约 23.5 万）**，不能写成“已使用 DDInter 2.0 全量 302,516”。

## 3. 药名 → DrugBank ID 映射规则

项目对照表（**不随仓库分发**；须用你自己许可的 DrugBank 5.1.15 XML 生成）：

- `data/processed/drugbank_name_aliases.csv`（`drugbank_id, primary_name, alias, alias_normalized, atc_codes`）
- `data/processed/drugbank_name_i18n.csv`（`drugbank_id, name_en, name_cn`）

从 XML 的 `<synonym>` 与 `<international-brands>/<international-brand>/<name>`
导出（主名写入 `name_en`；含 CJK 的别名写入 `name_cn`）：

```bash
python -m graphtree_ddi.data.download_data --export-name-tables
```

也可按同一列格式自行从 synonyms / international-brands 建表。这些表含 DrugBank
名称，受 DrugBank 学术许可约束，请勿再分发。

匹配顺序（大小写不敏感；一名称对应多个 DrugBank ID 则视为歧义，不自动选取）：

1. **精确主名** `exact_primary`：与 `primary_name` 的大小写折叠 / NFKC 规范化形式完全一致。置信 1.00。
2. **精确 i18n** `exact_i18n`：与 `name_en` / `name_cn` 完全一致。置信 1.00。
3. **精确别名** `exact_alias`：与 `alias` 完全一致。置信 0.95。
4. **规范化别名** `exact_alias_normalized`：与 `alias_normalized`（去标点、连字符改空格后的小写串）完全一致。置信 0.90。
5. **盐形式后缀剥离**（仅当以上均失败）：仅在大小写不敏感精确匹配失败后，对规范化英文名的末尾盐/水合物/反离子单词做剥离：逐个去掉末尾属于预定盐词表的 token（如 hydrochloride、sodium、mesylate、hydrate 等），要求剩余词干长度≥4 且词干本身不是盐词；不剥离词中段、不做模糊匹配、不启用编辑距离。 词干再对主名 / 别名做精确查找。`salt_stripped_primary` 置信 0.80；`salt_stripped_alias` 置信 0.75。
6. **详情页** `detail_page`：对仍未匹配或名称歧义的药物，访问 DDInter 详情页，用正则提取 `go.drugbank.com/drugs/DBxxxxx`。节流 **1 请求/秒**，失败重试 2 次（共 3 次），先 2.0 再 1.0；总墙钟预算 **40 分钟**。置信 0.99。结果缓存于 `raw/detail_page_cache.csv`，可重复运行。

不使用模糊匹配、不使用编辑距离、不凭 ATC 自动消歧。

### 3.1 药物级覆盖率

- DDInter 唯一药物：1,971
- 映射到 DrugBank ID：1,864
- 未映射：107
- **覆盖率：94.5713%**
- 名称阶段即命中（抓取前）：1,769
- 抓取前未匹配 / 歧义：201 / 1
- 本次新发详情页请求 / 由此解决：202 / 95
- 详情页抓取在时限内完成（含缓存命中，未新发请求亦视为完成）。

| method | 药物数 |
|---|---|
| exact_primary | 1,756 |
| detail_page_no_link | 107 |
| detail_page | 95 |
| exact_alias | 8 |
| salt_stripped_primary | 5 |

未映射的 107 个药物均已访问详情页，但页面无 DrugBank 外链（`detail_page_no_link`）。药名多为带给药途径括号的条目，例如 `(ophthalmic)` / `(nasal)` / `(topical)`，本流水线未把途径括号当作可剥离后缀（避免把局部制剂自动并入全身给药条目）。

完整表：`drug_mapping.csv`（列：`DDInterID, Drug, drugbank_id, method, confidence`）。未映射清单：`unmatched_drugs.csv`。

## 4. 药物对级映射与三组集合关系

项目药物对：`data/processed/pairs.csv`（与 `y.npy` 行对齐）。标签 0 为负样本，1–4 为正样本风险等级。项目药物集合 = `pairs.csv` 中出现过的全部 DrugBank ID。

纳入 `ddinter_mapped_pairs.csv` 的条件（同时满足）：

- 两端 DDInter 药物均映射到 DrugBank ID；
- 两个 DrugBank ID 均属于项目药物集合；
- 两端不是同一 DrugBank ID。

列：

- `drug_a`, `drug_b`：字典序 `drug_a < drug_b`
- `ddinter_level`
- `in_drugbank_positive`：若该无序对在 `pairs.csv` 中为正样本，填写其标签 1–4，否则空
- `in_drugbank_negative`：是否为本项目负样本（`True`/`False`）
- `absent_in_drugbank`：两者皆非（`True`/`False`）

同一 DrugBank 无序对若由多条 DDInter 记录映射而来且 Level 不同，整对剔除并写入 `mapped_pair_level_conflicts.csv`（保守）。

### 4.1 三组数量

- 两端均可映射且均在项目药物集合中的无序对：186,273
- 其中因 DrugBank 端 Level 冲突剔除：493
- **in_drugbank_positive**：129,729
- **in_drugbank_negative**：59
- **absent_in_drugbank**：56,485

项目药物集合大小：14,615；`pairs.csv` 行数：1,332,273；无序对去重后：1,332,174（多出 99 行是相同无序对的重复记录，标签冲突 0 对）。

### 4.2 三组 × DDInter Level 交叉表

| 组别 \ Level | Major | Moderate | Minor | Unknown | 合计 |
|---|---|---|---|---|---|
| in_drugbank_positive | 25168 | 79636 | 4680 | 20245 | 129,729 |
| in_drugbank_negative | 12 | 23 | 1 | 23 | 59 |
| absent_in_drugbank | 4720 | 27891 | 2579 | 21295 | 56,485 |

正样本子集中，DDInter Level 与本项目 1–4 标签交叉：

| DDInter Level \ DrugBank 标签 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Major | 17537 | 353 | 5510 | 1768 |
| Moderate | 58643 | 2792 | 14897 | 3304 |
| Minor | 3804 | 211 | 579 | 86 |
| Unknown | 16641 | 796 | 1496 | 1312 |

## 5. 可重复运行

在仓库根下不要改 `data/`、`models/` 或论文文件。于本目录执行（PowerShell）：

```powershell
$env:PYTHONIOENCODING='utf-8'
Set-Location "."
python .\01_download.py
python .\02_clean.py
python .\03_map_drugs.py
python .\04_map_pairs.py
```

- 已存在的 `raw/*.csv` 默认跳过下载（只重算哈希）；强制重下：`$env:DDINTER_FORCE_DOWNLOAD='1'`。
- 详情页缓存：`raw/detail_page_cache.csv`。

## 6. 论文中应采用的谨慎措辞

1. **不要写“DDInter 2.0 全量 302,516 条”作为本实验实际样本量。** 官方 ATC 批量 CSV 去重后的无序对数为 234,981，与 1.0 版本公开规模（约 23.5 万）同量级；2.0 站点文件与 1.0 同名文件先前实测哈希一致。应写为「使用 DDInter 官方 ATC 分类 CSV（CC BY-NC-SA 4.0），去重后 N 条无序对」并给出本目录统计。
2. **这是跨数据库外部验证，不是独立前瞻临床验证。** DDInter 与 DrugBank 都大量引用药品说明书、标签和已发表文献，上游可能重叠。重叠对上的一致不能解释为“全新临床队列上的泛化”。
3. 映射依赖药名精确匹配与详情页 DrugBank 外链，存在未映射药物与盐形式漏匹配；报告必须同时给出**药物级覆盖率**和**对级纳入数**，不能把 DDInter 全库当作已全部对齐到本模型药物空间。
4. `absent_in_drugbank` 只表示「两端药物都在本项目药物集合中，但该无序对既不是训练用正样本也不是所构建负样本」，**不能**解释为临床上已证实安全。
5. 本项目负样本由结构远离等规则采样，不是随机于全空间，也不是经临床确认的“无相互作用”。与 DDInter 的交叉只能描述集合关系。
6. Level 体系不同：DDInter 为 Major/Moderate/Minor/Unknown，本项目为基于 DrugBank 描述文本的 0–4 规则分级。二者**不是**同一序数尺度，交叉表只作描述，不宜直接当金标准校准曲线。

## 7. 本目录文件

| 路径 | 说明 |
|---|---|
| `01_download.py` … `04_map_pairs.py` | 可重复脚本 |
| `_common.py` | 路径与时间辅助 |
| `raw/ddinter_downloads_code_*.csv` | 14 个原始 ATC 文件 |
| `raw/download_manifest.csv` | SHA-256、行数、下载时间 |
| `raw/detail_page_cache.csv` | 详情页抓取缓存 |
| `ddinter_pairs_clean.csv` | 去重且剔除 Level 冲突后的主表 |
| `conflicts.csv` | Level 冲突对 |
| `level_distribution.csv` | 主表 Level 分布 |
| `drug_mapping.csv` | 药物级映射 |
| `unmatched_drugs.csv` | 未映射药物 |
| `ddinter_mapped_pairs.csv` | 对级映射 + 三组标注 |
| `mapped_pairs_crosstab.csv` | 三组 × Level |
| `mapped_pair_level_conflicts.csv` | 映射后 Level 仍冲突的 DrugBank 对 |
| `stats/*.json` | 各步机器可读统计 |

## 8. 推理与汇总

本节为手工追加（`05_run_inference.py` / `06_summarize.py`）。**不要改 `models/`、`data/`。** 重跑 `04_map_pairs.py` 会整文件重写本 README，需再追加本节。

上游推理器 `models/predict_pairs.py` 对每个药对**同时**写出 Exp-1、Exp-3、Exp-5：

- 标识：`drug_a, drug_b, drug_a_canon, drug_b_canon, in_index`
- 预测等级：`exp1_pred, exp3_pred, exp5_pred`
- 五类概率：`exp1_p0..p4`、`exp3_p0..p4`、`exp5_p0..p4`
- 未知 DrugBank ID：`in_index=0`，对应模型列为空值

`05_run_inference.py` **import 复用** `load_bundle` / `predict_rows`（不复制模型代码），按 `--model exp5|exp1|both` 抽取其中一套，写成下游统计所需列。

### 8.1 输出列

`out_dir/ddinter_pred_{subset}_{model}.csv`：

| 列 | 含义 |
|---|---|
| `drug1_id`, `drug2_id` | DrugBank ID（与映射表一致，字典序 `drug_a < drug_b`） |
| `ext_level` | DDInter Level：`Major` / `Moderate` / `Minor` / `Unknown` |
| `y_pred` | 所选模型预测等级 0–4；无法推理则为空 |
| `p0` … `p4` | 五类概率 |
| `group` | `in_drugbank_positive` / `in_drugbank_negative` / `absent_in_drugbank` |
| `drugbank_label` | 正样本组填项目标签 1–4，否则空 |

`inference_stats.json`：查询条数、可推理/跳过条数、耗时。**同一 `out_dir` 再跑会覆盖该文件**，建议按 subset（及模型）分目录。

`06_summarize.py` 读上述 CSV，写 `summary_{subset}_{model}.md` 与同名 JSON（不画图；图由 `models/analysis/external_eval.py` 负责）：

- **absent**：按 `ext_level` 的检出率（`y_pred≥1`）、`y_pred≥3` 比例、期望风险 `Σ k·p_k` 均值/中位数、预测等级分布；Spearman ρ（Major=3, Moderate=2, Minor=1 vs 期望风险，排除 Unknown）；Major vs Minor AUROC（打分 `P(≥3)=p3+p4`）
- **negative**：预测等级分布

### 8.2 命令（PowerShell）

在本目录执行。论文用**正式** `final_full` 包，不要用冒烟包。

```powershell
$env:PYTHONIOENCODING='utf-8'
Set-Location "."
$bundle = "results/final_v2_full/final_full"   # 正式包；勿用 final_smoke_full
$data = "data/processed"

& $py .\05_run_inference.py --bundle $bundle --data_dir $data --out_dir .\out\absent_exp5 --subset absent --model exp5
& $py .\06_summarize.py --pred .\out\absent_exp5\ddinter_pred_absent_exp5.csv

& $py .\05_run_inference.py --bundle $bundle --data_dir $data --out_dir .\out\negative_exp5 --subset negative --model exp5
& $py .\06_summarize.py --pred .\out\negative_exp5\ddinter_pred_negative_exp5.csv
```

其它参数：`--subset positive|all`；`--model exp1|both`（`both` 写两份 CSV）；`--limit N` 按映射表文件顺序截断（仅调试）。

### 8.3 本机冒烟验证（不可用于论文）

`models/results/final_smoke_full/final_full/` 是 v1 `data/processed` 上 **2 epoch / 50 树** 的流程包。已用其对 **absent `--limit 300`** 与 **negative 全部 59** 跑通 `05`+`06`，产物在 `out_smoke/absent_exp5/`、`out_smoke/negative_exp5/`。

**冒烟数字不得写入论文。** 正式外部验证需用完整训练的 `final_full` 对 absent 全量（56,485）与 negative（59）重跑。
