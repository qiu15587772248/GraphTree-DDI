"""Map cleaned DDInter pairs onto this project's DrugBank pair set and label membership."""

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

from pathlib import Path

import numpy as np
import pandas as pd

from graphtree_ddi.external.ddinter._common import (
    PROCESSED_DIR,
    STATS_DIR,
    configure_stdio,
    ensure_dirs,
    now_cst_iso,
    read_json,
    write_json,
)

HERE = Path(__file__).resolve().parent
CLEAN_CSV = HERE / "ddinter_pairs_clean.csv"
MAPPING_CSV = HERE / "drug_mapping.csv"
OUT_CSV = HERE / "ddinter_mapped_pairs.csv"
CROSS_CSV = HERE / "mapped_pairs_crosstab.csv"
MAPPED_CONFLICTS_CSV = HERE / "mapped_pair_level_conflicts.csv"
STATS_JSON = STATS_DIR / "04_map_pairs.json"
README_MD = HERE / "README.md"

PAIRS_CSV = PROCESSED_DIR / "pairs.csv"
Y_NPY = PROCESSED_DIR / "y.npy"


def pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def load_project_pairs() -> tuple[set[str], dict[tuple[str, str], int], dict]:
    pairs = pd.read_csv(PAIRS_CSV, dtype=str)
    y = np.load(Y_NPY)
    n_rows = min(len(pairs), len(y))
    if len(pairs) != len(y):
        print(f"[04] warning: pairs.csv rows={len(pairs)} y.npy={len(y)}; using n={n_rows}")
    pairs = pairs.iloc[:n_rows]
    y = y[:n_rows]
    drug_ids = set(pairs["drug1_id"].astype(str)) | set(pairs["drug2_id"].astype(str))
    labels: dict[tuple[str, str], int] = {}
    n_dup_extra = 0
    n_dup_label_conflict = 0
    for d1, d2, lab in zip(pairs["drug1_id"], pairs["drug2_id"], y):
        key = pair_key(str(d1), str(d2))
        lab_i = int(lab)
        if key in labels:
            n_dup_extra += 1
            if labels[key] != lab_i:
                n_dup_label_conflict += 1
        else:
            labels[key] = lab_i
    meta = {
        "n_project_pair_rows": int(n_rows),
        "n_project_unique_pairs": int(len(labels)),
        "n_project_duplicate_extra_rows": int(n_dup_extra),
        "n_project_duplicate_label_conflicts": int(n_dup_label_conflict),
    }
    return drug_ids, labels, meta


def group_flags(label: int | None) -> tuple[str, str, str]:
    """Return (in_drugbank_positive, in_drugbank_negative, absent_in_drugbank)."""
    if label is None:
        return "", "False", "True"
    if label == 0:
        return "", "True", "False"
    return str(label), "False", "False"


def membership_name(pos: str, neg: str, absent: str) -> str:
    if pos:
        return "in_drugbank_positive"
    if neg == "True":
        return "in_drugbank_negative"
    if absent == "True":
        return "absent_in_drugbank"
    return "other"


def fmt_int(value: object) -> str:
    if isinstance(value, bool) or value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{int(value):,}"
    if isinstance(value, str) and value.isdigit():
        return f"{int(value):,}"
    return str(value) if value not in {"", "n/a"} else "n/a"


def write_readme(pair_stats: dict) -> None:
    dl = read_json(STATS_DIR / "01_download.json") if (STATS_DIR / "01_download.json").exists() else {}
    cl = read_json(STATS_DIR / "02_clean.json") if (STATS_DIR / "02_clean.json").exists() else {}
    mp = read_json(STATS_DIR / "03_map_drugs.json") if (STATS_DIR / "03_map_drugs.json").exists() else {}

    def fmt_level_dist(dist: dict) -> str:
        if not dist:
            return "（无）"
        lines = ["| Level | 条数 |", "|---|---|"]
        for k in ["Major", "Moderate", "Minor", "Unknown"]:
            if k in dist:
                lines.append(f"| {k} | {dist[k]:,} |")
        for k, v in dist.items():
            if k not in {"Major", "Moderate", "Minor", "Unknown"}:
                lines.append(f"| {k} | {v:,} |")
        return "\n".join(lines)

    fallback_files = [
        f["file"] for f in dl.get("files", []) if f.get("fallback_used")
    ]
    failed_files = [
        f["file"] for f in dl.get("files", []) if f.get("status") == "failed"
    ]
    manifest_lines = [
        "| 文件 | ATC | 来源 | SHA-256 | 数据行 | 下载时间 (UTC+8) | 状态 |",
        "|---|---|---|---|---|---|---|",
    ]
    for f in dl.get("files", []):
        sha = f.get("sha256") or ""
        manifest_lines.append(
            f"| {f.get('file','')} | {f.get('atc_code','')} | {f.get('source_site','')} "
            f"| `{sha}` | {int(f.get('n_data_rows') or 0):,} | {f.get('downloaded_at_utc8','')} "
            f"| {f.get('status','')} |"
        )

    method_lines = ["| method | 药物数 |", "|---|---|"]
    for k, v in sorted(mp.get("method_counts", {}).items(), key=lambda kv: (-kv[1], kv[0])):
        method_lines.append(f"| {k} | {v:,} |")

    cross = pair_stats.get("crosstab", {})
    levels = pair_stats.get("level_order", ["Major", "Moderate", "Minor", "Unknown"])
    groups = ["in_drugbank_positive", "in_drugbank_negative", "absent_in_drugbank"]
    cross_lines = ["| 组别 \\ Level | " + " | ".join(levels) + " | 合计 |", "|---|" + "---|" * (len(levels) + 1)]
    for g in groups:
        row = cross.get(g, {})
        cells = [str(int(row.get(lv, 0))) for lv in levels]
        total = int(sum(int(row.get(lv, 0)) for lv in levels))
        cross_lines.append(f"| {g} | " + " | ".join(cells) + f" | {total:,} |")

    pos_dist = pair_stats.get("positive_label_by_ddinter_level", {})
    pos_lines = ["| DDInter Level \\ DrugBank 标签 | 1 | 2 | 3 | 4 |", "|---|---|---|---|---|"]
    for lv in levels:
        row = pos_dist.get(lv, {})
        pos_lines.append(
            f"| {lv} | {int(row.get('1',0))} | {int(row.get('2',0))} | {int(row.get('3',0))} | {int(row.get('4',0))} |"
        )

    scrape_note = ""
    if mp.get("scrape_timed_out"):
        scrape_note = (
            f"详情页抓取在 40 分钟预算内停止，仍有 "
            f"{mp.get('n_unmapped', '?')} 个 DDInter 药物未映射。"
        )
    else:
        scrape_note = "详情页抓取在时限内完成（含缓存命中，未新发请求亦视为完成）。"

    md = f"""# DDInter 外部验证数据准备

本目录存放「冻结的 DrugBank 模型 → DDInter 跨数据库外部验证」所需的**数据准备产物**。此处**不做模型推理**，只完成下载、清洗、ID 映射与集合关系标注。

生成时间（UTC+8）：{pair_stats.get("generated_at_utc8", now_cst_iso())}

## 1. 来源与许可

- **数据库**：DDInter 2.0 官方按 ATC 分类提供的相互作用 CSV 批量下载（[ddinter2.scbdd.com](https://ddinter2.scbdd.com/)）。
- **直链格式**：`https://ddinter2.scbdd.com/static/media/download/ddinter_downloads_code_{{X}}.csv`，`X ∈ {{A,B,C,D,G,H,J,L,M,N,P,R,S,V}}`（共 14 个）。下载页未全部列出，但其余路径实测可访问。
- **回退**：若 2.0 直链失败，改用 1.0 站点同名路径 `https://ddinter.scbdd.com/static/media/download/...`。先前实测 1.0 与 2.0 文件哈希一致。
- **许可**：CC BY-NC-SA 4.0（非商业、署名、相同方式共享）。论文与仓库使用时需保留署名与许可声明。
- **表头**：`DDInterID_A, Drug_A, DDInterID_B, Drug_B, Level`；`Level ∈ {{Major, Moderate, Minor, Unknown}}`。文件**不含** DrugBank ID。
- **药物详情页**（例如 `https://ddinter2.scbdd.com/server/drug-detail/DDInter1/`）在 Useful Links 中外链到 `https://go.drugbank.com/drugs/DBxxxxx`。

### 1.1 下载记录

- 请求文件数：{dl.get("n_files_requested", "n/a")}
- 成功（含本地已存在）：{dl.get("n_ok", "n/a")}
- 改用 1.0 回退：{dl.get("n_fallback_to_v1", 0)}{("（" + ", ".join(fallback_files) + "）") if fallback_files else ""}
- 失败：{dl.get("n_failed", 0)}{("（" + ", ".join(failed_files) + "）") if failed_files else ""}
- 14 个文件数据行合计（含 ATC 交叉重复，未去重）：{fmt_int(dl.get("total_data_rows_sum"))}

明细见 `raw/download_manifest.csv`。

{chr(10).join(manifest_lines)}

## 2. 清洗规则

输入：`raw/ddinter_downloads_code_{{A–V}}.csv`。

1. 去掉两端 ID 缺失或 `DDInterID_A == DDInterID_B` 的行。
2. 将无序对规范为数值 ID 较小者在前（`DDInterID_A` 的数字部分 `<` `DDInterID_B`）。
3. **ATC 交叉重复**：同一无序对出现在多个 ATC 文件且 `Level` 相同，只保留一条（按 ATC 字母序 A→V 取首次出现的药名写法）。
4. **Level 冲突（保守）**：同一无序对出现不同 `Level` 时，整对写入 `conflicts.csv`，**不进入**主表 `ddinter_pairs_clean.csv`。
5. 不根据药名排序，不以 DrugBank ID 作为本步键。

清洗结果：

- 原始行数（14 文件合计）：{fmt_int(cl.get("n_raw_rows"))}
- Level 冲突无序对：{fmt_int(cl.get("n_conflict_pairs"))}（涉及行 {fmt_int(cl.get("n_conflict_rows"))}）
- 主表无序对数：{fmt_int(cl.get("n_clean_unordered_pairs"))}
- 主表唯一 DDInter 药物数：{fmt_int(cl.get("n_unique_ddinter_drugs"))}

Level 分布（主表）：

{fmt_level_dist(cl.get("level_distribution", {}))}

说明：官方 2.0 论文数字为 302,516 条；本批官方 CSV 去重后约 {fmt_int(cl.get("n_clean_unordered_pairs"))} 条无序对，**接近 DDInter 1.0 公开规模（约 23.5 万）**，不能写成“已使用 DDInter 2.0 全量 302,516”。

## 3. 药名 → DrugBank ID 映射规则

项目对照表（只读）：

- `data/processed/drugbank_name_aliases.csv`（`drugbank_id, primary_name, alias, alias_normalized, atc_codes`）
- `data/processed/drugbank_name_i18n.csv`（`drugbank_id, name_en, name_cn`）

匹配顺序（大小写不敏感；一名称对应多个 DrugBank ID 则视为歧义，不自动选取）：

1. **精确主名** `exact_primary`：与 `primary_name` 的大小写折叠 / NFKC 规范化形式完全一致。置信 1.00。
2. **精确 i18n** `exact_i18n`：与 `name_en` / `name_cn` 完全一致。置信 1.00。
3. **精确别名** `exact_alias`：与 `alias` 完全一致。置信 0.95。
4. **规范化别名** `exact_alias_normalized`：与 `alias_normalized`（去标点、连字符改空格后的小写串）完全一致。置信 0.90。
5. **盐形式后缀剥离**（仅当以上均失败）：{mp.get("salt_rule", "")} 词干再对主名 / 别名做精确查找。`salt_stripped_primary` 置信 0.80；`salt_stripped_alias` 置信 0.75。
6. **详情页** `detail_page`：对仍未匹配或名称歧义的药物，访问 DDInter 详情页，用正则提取 `go.drugbank.com/drugs/DBxxxxx`。节流 **1 请求/秒**，失败重试 2 次（共 3 次），先 2.0 再 1.0；总墙钟预算 **40 分钟**。置信 0.99。结果缓存于 `raw/detail_page_cache.csv`，可重复运行。

不使用模糊匹配、不使用编辑距离、不凭 ATC 自动消歧。

### 3.1 药物级覆盖率

- DDInter 唯一药物：{fmt_int(mp.get("n_ddinter_drugs"))}
- 映射到 DrugBank ID：{fmt_int(mp.get("n_mapped_to_drugbank"))}
- 未映射：{fmt_int(mp.get("n_unmapped"))}
- **覆盖率：{mp.get("drug_level_coverage_pct", "n/a")}%**
- 名称阶段即命中（抓取前）：{fmt_int(mp.get("n_name_match_before_scrape"))}
- 抓取前未匹配 / 歧义：{fmt_int(mp.get("n_unmatched_before_scrape"))} / {fmt_int(mp.get("n_ambiguous_before_scrape"))}
- 本次新发详情页请求 / 由此解决：{fmt_int(mp.get("n_scrape_requests"))} / {fmt_int(mp.get("n_scrape_resolved"))}
- {scrape_note}

{chr(10).join(method_lines)}

未映射的 {fmt_int(mp.get("n_unmapped"))} 个药物均已访问详情页，但页面无 DrugBank 外链（`detail_page_no_link`）。药名多为带给药途径括号的条目，例如 `(ophthalmic)` / `(nasal)` / `(topical)`，本流水线未把途径括号当作可剥离后缀（避免把局部制剂自动并入全身给药条目）。

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

- 两端均可映射且均在项目药物集合中的无序对：{fmt_int(pair_stats.get("n_mapped_pairs"))}
- 其中因 DrugBank 端 Level 冲突剔除：{fmt_int(pair_stats.get("n_mapped_level_conflict_pairs"))}
- **in_drugbank_positive**：{fmt_int(pair_stats.get("n_in_drugbank_positive"))}
- **in_drugbank_negative**：{fmt_int(pair_stats.get("n_in_drugbank_negative"))}
- **absent_in_drugbank**：{fmt_int(pair_stats.get("n_absent_in_drugbank"))}

项目药物集合大小：{fmt_int(pair_stats.get("n_project_drugs"))}；`pairs.csv` 行数：{fmt_int(pair_stats.get("n_project_pair_rows"))}；无序对去重后：{fmt_int(pair_stats.get("n_project_unique_pairs"))}（多出 {fmt_int(pair_stats.get("n_project_duplicate_extra_rows"))} 行是相同无序对的重复记录，标签冲突 {fmt_int(pair_stats.get("n_project_duplicate_label_conflicts"))} 对）。

### 4.2 三组 × DDInter Level 交叉表

{chr(10).join(cross_lines)}

正样本子集中，DDInter Level 与本项目 1–4 标签交叉：

{chr(10).join(pos_lines)}

## 5. 可重复运行

在仓库根下不要改 `data/`、`models/` 或论文文件。于本目录执行（PowerShell）：

```powershell
$env:PYTHONIOENCODING='utf-8'
Set-Location "."
python .\\01_download.py
python .\\02_clean.py
python .\\03_map_drugs.py
python .\\04_map_pairs.py
```

- 已存在的 `raw/*.csv` 默认跳过下载（只重算哈希）；强制重下：`$env:DDINTER_FORCE_DOWNLOAD='1'`。
- 详情页缓存：`raw/detail_page_cache.csv`。

## 6. 论文中应采用的谨慎措辞

1. **不要写“DDInter 2.0 全量 302,516 条”作为本实验实际样本量。** 官方 ATC 批量 CSV 去重后的无序对数为 {fmt_int(cl.get("n_clean_unordered_pairs"))}，与 1.0 版本公开规模（约 23.5 万）同量级；2.0 站点文件与 1.0 同名文件先前实测哈希一致。应写为「使用 DDInter 官方 ATC 分类 CSV（CC BY-NC-SA 4.0），去重后 N 条无序对」并给出本目录统计。
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
"""
    README_MD.write_text(md, encoding="utf-8")
    print(f"[04] wrote {README_MD}")


def main() -> int:
    configure_stdio()
    ensure_dirs()
    clean = pd.read_csv(CLEAN_CSV, dtype=str, encoding="utf-8").fillna("")
    mapping = pd.read_csv(MAPPING_CSV, dtype=str, encoding="utf-8").fillna("")
    mapped_ids = mapping[mapping["drugbank_id"].ne("")].copy()
    id_map = dict(zip(mapped_ids["DDInterID"], mapped_ids["drugbank_id"]))

    project_drugs, project_labels, pair_meta = load_project_pairs()
    print(
        f"[04] project drugs={len(project_drugs):,} "
        f"pair_rows={pair_meta['n_project_pair_rows']:,} "
        f"unique_pairs={pair_meta['n_project_unique_pairs']:,} "
        f"dup_extra={pair_meta['n_project_duplicate_extra_rows']:,}",
        flush=True,
    )

    records = []
    n_skip_unmapped = 0
    n_skip_not_in_project = 0
    n_skip_self = 0
    for row in clean.itertuples(index=False):
        db_a = id_map.get(str(row.DDInterID_A), "")
        db_b = id_map.get(str(row.DDInterID_B), "")
        if not db_a or not db_b:
            n_skip_unmapped += 1
            continue
        if db_a not in project_drugs or db_b not in project_drugs:
            n_skip_not_in_project += 1
            continue
        if db_a == db_b:
            n_skip_self += 1
            continue
        a, b = pair_key(db_a, db_b)
        records.append(
            {
                "drug_a": a,
                "drug_b": b,
                "ddinter_level": str(row.Level),
                "ddinter_id_a": str(row.DDInterID_A),
                "ddinter_id_b": str(row.DDInterID_B),
            }
        )

    raw_mapped = pd.DataFrame.from_records(records)
    n_before_db_dedup = int(len(raw_mapped))
    n_conflict_pairs = 0
    if raw_mapped.empty:
        out = pd.DataFrame(
            columns=[
                "drug_a",
                "drug_b",
                "ddinter_level",
                "in_drugbank_positive",
                "in_drugbank_negative",
                "absent_in_drugbank",
            ]
        )
        conflicts = raw_mapped
    else:
        level_nunique = raw_mapped.groupby(["drug_a", "drug_b"], sort=False)[
            "ddinter_level"
        ].nunique()
        conflict_idx = level_nunique[level_nunique > 1].index
        n_conflict_pairs = int(len(conflict_idx))
        if n_conflict_pairs:
            conflicts = raw_mapped.set_index(["drug_a", "drug_b"]).loc[conflict_idx].reset_index()
            conflicts.to_csv(MAPPED_CONFLICTS_CSV, index=False, encoding="utf-8")
            keep = raw_mapped.set_index(["drug_a", "drug_b"]).drop(index=conflict_idx).reset_index()
        else:
            conflicts = raw_mapped.iloc[0:0]
            MAPPED_CONFLICTS_CSV.write_text(
                "drug_a,drug_b,ddinter_level,ddinter_id_a,ddinter_id_b\n",
                encoding="utf-8",
            )
            keep = raw_mapped
        keep = (
            keep.groupby(["drug_a", "drug_b"], sort=False)
            .agg(ddinter_level=("ddinter_level", "first"))
            .reset_index()
        )
        pos_col, neg_col, abs_col = [], [], []
        for a, b in zip(keep["drug_a"], keep["drug_b"]):
            lab = project_labels.get(pair_key(a, b))
            p, n, ab = group_flags(lab)
            pos_col.append(p)
            neg_col.append(n)
            abs_col.append(ab)
        keep["in_drugbank_positive"] = pos_col
        keep["in_drugbank_negative"] = neg_col
        keep["absent_in_drugbank"] = abs_col
        out = keep[
            [
                "drug_a",
                "drug_b",
                "ddinter_level",
                "in_drugbank_positive",
                "in_drugbank_negative",
                "absent_in_drugbank",
            ]
        ].sort_values(["drug_a", "drug_b"])

    out.to_csv(OUT_CSV, index=False, encoding="utf-8")

    out = out.copy()
    out["membership"] = [
        membership_name(p, n, a)
        for p, n, a in zip(
            out["in_drugbank_positive"],
            out["in_drugbank_negative"],
            out["absent_in_drugbank"],
        )
    ]
    level_order = ["Major", "Moderate", "Minor", "Unknown"]
    present_levels = [lv for lv in level_order if lv in set(out["ddinter_level"])]
    extra_levels = [lv for lv in sorted(set(out["ddinter_level"])) if lv not in level_order]
    levels = present_levels + extra_levels

    crosstab_df = (
        pd.crosstab(out["membership"], out["ddinter_level"])
        if len(out)
        else pd.DataFrame()
    )
    for lv in levels:
        if lv not in crosstab_df.columns:
            crosstab_df[lv] = 0
    for g in ["in_drugbank_positive", "in_drugbank_negative", "absent_in_drugbank"]:
        if g not in crosstab_df.index:
            crosstab_df.loc[g] = 0
    if len(crosstab_df):
        crosstab_df = crosstab_df.reindex(
            index=["in_drugbank_positive", "in_drugbank_negative", "absent_in_drugbank"],
            columns=levels,
        ).fillna(0).astype(int)
        crosstab_df["total"] = crosstab_df.sum(axis=1)
        crosstab_df.to_csv(CROSS_CSV, encoding="utf-8")
    else:
        pd.DataFrame(
            {"membership": ["in_drugbank_positive", "in_drugbank_negative", "absent_in_drugbank"]}
        ).to_csv(CROSS_CSV, index=False, encoding="utf-8")

    n_pos = int((out["membership"] == "in_drugbank_positive").sum()) if len(out) else 0
    n_neg = int((out["membership"] == "in_drugbank_negative").sum()) if len(out) else 0
    n_abs = int((out["membership"] == "absent_in_drugbank").sum()) if len(out) else 0

    pos_cross: dict[str, dict[str, int]] = {}
    if n_pos:
        sub = out[out["membership"] == "in_drugbank_positive"]
        ct = pd.crosstab(sub["ddinter_level"], sub["in_drugbank_positive"])
        for lv in levels:
            pos_cross[lv] = {
                str(lab): int(ct.loc[lv, lab]) if (lv in ct.index and lab in ct.columns) else 0
                for lab in ["1", "2", "3", "4"]
            }

    crosstab_dict = {}
    if len(out):
        for g in ["in_drugbank_positive", "in_drugbank_negative", "absent_in_drugbank"]:
            crosstab_dict[g] = {
                lv: int(crosstab_df.loc[g, lv]) if lv in crosstab_df.columns else 0
                for lv in levels
            }

    stats = {
        "generated_at_utc8": now_cst_iso(),
        "n_project_drugs": int(len(project_drugs)),
        "n_project_pairs": int(len(project_labels)),
        **pair_meta,
        "n_skip_unmapped_endpoint": int(n_skip_unmapped),
        "n_skip_endpoint_not_in_project": int(n_skip_not_in_project),
        "n_skip_self_pair": int(n_skip_self),
        "n_raw_mapped_rows": n_before_db_dedup,
        "n_mapped_level_conflict_pairs": n_conflict_pairs,
        "n_mapped_pairs": int(len(out)),
        "n_in_drugbank_positive": n_pos,
        "n_in_drugbank_negative": n_neg,
        "n_absent_in_drugbank": n_abs,
        "crosstab": crosstab_dict,
        "level_order": levels,
        "positive_label_by_ddinter_level": pos_cross,
        "mapped_csv": str(OUT_CSV),
    }
    write_json(STATS_JSON, stats)
    write_readme(stats)

    print(f"[04] skip unmapped={n_skip_unmapped:,} not_in_project={n_skip_not_in_project:,} self={n_skip_self:,}")
    print(f"[04] mapped pairs={len(out):,} (level conflicts dropped={n_conflict_pairs:,})")
    print(f"[04] positive={n_pos:,} negative={n_neg:,} absent={n_abs:,}")
    if len(out):
        print("[04] crosstab:")
        print(crosstab_df.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
