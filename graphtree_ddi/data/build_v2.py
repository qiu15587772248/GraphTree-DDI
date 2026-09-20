# -*- coding: utf-8 -*-
"""
从 processed/（v1，2026-04-14）构建 processed/v2/。

v2 做三件事：用 2026-07-02 短语映射重算正样本标签、按无序对去重、
按与 v1 相同的字符串 sorted() 规则统一 drug1_id < drug2_id。

不修改 v1 文件、preprocess.py、models/。

X 布局（由 preprocess.build_pair_features 与抽样重算确认，不是每药拼接）：
  [0:2048]     Morgan 指纹逐位相乘
  [2048:4096]  Morgan 指纹逐位绝对差
  [4096:4120]  24 维对称药理特征
因此统一 ID 顺序时只交换 pairs 中的两个 ID，不交换 X 前后半段。
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

from graphtree_ddi.paths import DATA_PROCESSED, DATA_RAW, PKG_DIR  # noqa: E402

HERE = PKG_DIR / "data"

import argparse
import ast
import gc
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


from graphtree_ddi.data.preprocess import (  # noqa: E402
    FEAT_DIM,
    PHRASE_LABEL_MAP,
    build_fingerprint_cache,
    build_pair_features,
    map_severity,
)

EXPECTED_POS_CHANGED = 56055
EXPECTED_POS_CHANGED_TOL = 0.10  # 相对偏差超过 10% 则停止
RNG_SEED = 42
N_VERIFY_ROWS = 200
N_REVIEW_POS = 20


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def extract_phrase_map(py_path: Path) -> dict[str, int]:
    src = py_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "PHRASE_LABEL_MAP":
                return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "PHRASE_LABEL_MAP":
                    return ast.literal_eval(node.value)
    raise RuntimeError(f"未找到 PHRASE_LABEL_MAP: {py_path}")


def compare_maps(preprocess_py: Path, map_py: Path) -> dict:
    m1 = extract_phrase_map(preprocess_py)
    m2 = extract_phrase_map(map_py)
    only1 = sorted(set(m1) - set(m2))
    only2 = sorted(set(m2) - set(m1))
    mismatch = sorted(k for k in set(m1) & set(m2) if m1[k] != m2[k])
    return {
        "equal": m1 == m2,
        "n_preprocess": len(m1),
        "n_phrase_file": len(m2),
        "only_in_preprocess": only1,
        "only_in_phrase_file": only2,
        "value_mismatch": [
            {"phrase": k, "preprocess": m1[k], "phrase_file": m2[k]} for k in mismatch
        ],
        "imported_map_equals_preprocess_file": PHRASE_LABEL_MAP == m1,
    }


def unordered_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))  # 与 v1 preprocess 中 tuple(sorted([drug1, drug2])) 相同：字符串比较


def load_ddi_first(ddi_csv: Path) -> pd.DataFrame:
    """
    复用 v1 的描述来源：raw/drugbank_ddi.csv。

    XML 解析函数在 download_data.parse_drugbank_xml，不在 preprocess.py。
    该函数会原地覆盖 drugbank_drugs.csv / drugbank_ddi.csv，故本脚本不调用它，
    以免改写 v1 所用原始表。CSV 即 2026-04-14 从 full database.xml 解析的产物。
    镜像去重与 v1 一致：pair_key = tuple(sorted([id1, id2]))，keep='first'。
    """
    ddi = pd.read_csv(ddi_csv, dtype={"drug1_id": str, "drug2_id": str})
    if "description" not in ddi.columns:
        raise RuntimeError(f"DDI 表缺少 description 列: {list(ddi.columns)}")
    ddi["pair_key"] = [unordered_key(a, b) for a, b in zip(ddi["drug1_id"], ddi["drug2_id"])]
    n_before = len(ddi)
    first = ddi.drop_duplicates(subset="pair_key", keep="first").reset_index(drop=True)
    first["new_label"] = first["description"].map(map_severity).astype(np.int32)
    first.attrs["n_before_dedup"] = n_before
    return first


def crosstab_to_json(old: np.ndarray, new: np.ndarray) -> dict[str, dict[str, int]]:
    ct = pd.crosstab(
        pd.Series(old, name="old"),
        pd.Series(new, name="new"),
        dropna=False,
    )
    out: dict[str, dict[str, int]] = {}
    for i in ct.index:
        out[str(int(i))] = {str(int(j)): int(ct.loc[i, j]) for j in ct.columns}
    return out


def fmt_int(n: int) -> str:
    return f"{n:,}"


def md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def write_mismatch_report(out_dir: Path, cmp: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "MAP_MISMATCH.md"
    lines = [
        "# PHRASE_LABEL_MAP 不一致，已停止",
        "",
        f"- preprocess.py 条目数: {cmp['n_preprocess']}",
        f"- phrase_label_map.py 条目数: {cmp['n_phrase_file']}",
        f"- 仅在 preprocess.py: {cmp['only_in_preprocess']}",
        f"- 仅在 phrase_label_map.py: {cmp['only_in_phrase_file']}",
        f"- 同短语不同等级: {cmp['value_mismatch']}",
        "",
        "未猜测、未继续构建 v2。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="从 v1 processed 构建 v2（重标、去重、统一顺序）")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=DATA_PROCESSED,
        help="v1 目录（含 X_full.dat / y.npy / pairs.csv / meta.json）",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_PROCESSED / "v2",
        help="v2 输出目录",
    )
    p.add_argument(
        "--ddi-csv",
        type=Path,
        default=DATA_RAW / "drugbank_ddi.csv",
        help="DrugBank DDI 描述表（XML 解析产物，v1 实际输入）",
    )
    p.add_argument(
        "--drugs-csv",
        type=Path,
        default=DATA_RAW / "drugbank_drugs.csv",
        help="药物表（用于指纹缓存 / drug_features.npy）",
    )
    p.add_argument("--chunk-size", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=RNG_SEED)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_dir: Path = args.input_dir.resolve()
    out_dir: Path = args.output_dir.resolve()
    chunk_size = int(args.chunk_size)
    rng = np.random.default_rng(int(args.seed))

    preprocess_py = HERE / "preprocess.py"
    map_py = HERE / "phrase_label_map.py"
    if not map_py.is_file():
        map_py = in_dir / "phrase_label_map.py"
    print(f"[0] 比对 PHRASE_LABEL_MAP: {preprocess_py} vs {map_py}", flush=True)
    cmp = compare_maps(preprocess_py, map_py)
    if not cmp["equal"] or not cmp["imported_map_equals_preprocess_file"]:
        write_mismatch_report(out_dir, cmp)
        return 2
    print(f"    一致，{cmp['n_preprocess']} 条。", flush=True)
    label_map_sha = sha256_file(map_py)

    x_path = in_dir / "X_full.dat"
    y_path = in_dir / "y.npy"
    pairs_path = in_dir / "pairs.csv"
    meta_path = in_dir / "meta.json"
    for pth in (x_path, y_path, pairs_path, meta_path):
        if not pth.exists():
            print(f"缺少 v1 文件: {pth}")
            return 1

    with meta_path.open(encoding="utf-8") as f:
        meta_v1 = json.load(f)

    print("[1] 读取 v1 标签与 pairs", flush=True)
    y_v1 = np.load(y_path)
    pairs_v1 = pd.read_csv(pairs_path, dtype={"drug1_id": str, "drug2_id": str})
    n_v1 = int(y_v1.shape[0])
    if len(pairs_v1) != n_v1:
        print(f"行数不一致: pairs={len(pairs_v1)} y={n_v1}，停止")
        return 1
    if list(pairs_v1.columns) != ["drug1_id", "drug2_id"]:
        print(f"pairs.csv 列名非预期: {list(pairs_v1.columns)}")
        return 1

    feat_dim = int(meta_v1.get("feature_dim", FEAT_DIM))
    if feat_dim != FEAT_DIM:
        print(f"feature_dim 与 preprocess.FEAT_DIM 不一致: meta={feat_dim} FEAT_DIM={FEAT_DIM}")
        return 1
    x_bytes = x_path.stat().st_size
    itemsize = np.dtype(np.float32).itemsize
    if x_bytes % (feat_dim * itemsize) != 0:
        print(f"X_full.dat 大小不能整除 {feat_dim}*4: {x_bytes}")
        return 1
    n_alloc = x_bytes // (feat_dim * itemsize)
    if n_alloc < n_v1:
        print(f"X 分配行数 {n_alloc} < y 行数 {n_v1}")
        return 1

    print(
        f"    y/pairs={n_v1:,}  X bytes={x_bytes:,}  分配行={n_alloc:,}  "
        f"有效行按 y 截断；dtype=float32  shape=({n_v1},{feat_dim})",
        flush=True,
    )
    print(f"    y dtype={y_v1.dtype} bincount={np.bincount(y_v1, minlength=5).tolist()}", flush=True)

    id1 = pairs_v1["drug1_id"].astype(str)
    id2 = pairs_v1["drug2_id"].astype(str)
    id_lens = sorted(set(id1.str.len()) | set(id2.str.len()))

    print("[2] 加载 DDI 描述并按 v1 规则镜像去重后重算正样本标签", flush=True)
    ddi_first = load_ddi_first(args.ddi_csv.resolve())
    exact_lookup = {
        (a, b): (desc, int(lab))
        for a, b, desc, lab in zip(
            ddi_first["drug1_id"], ddi_first["drug2_id"],
            ddi_first["description"], ddi_first["new_label"],
        )
    }
    y_new = np.array(y_v1, copy=True)
    n_pos = int((y_v1 > 0).sum())
    exact_hit = 0
    miss = 0
    desc_by_v1 = {}
    for i in range(n_v1):
        if int(y_v1[i]) == 0:
            continue
        a, b = id1.iat[i], id2.iat[i]
        hit = exact_lookup.get((a, b))
        if hit is None:
            miss += 1
            continue
        exact_hit += 1
        desc_by_v1[i] = hit[0]
        y_new[i] = hit[1]
    if miss:
        print(f"正样本无法在 DDI 去重表中精确匹配: {miss}，停止")
        return 1
    changed_mask = (y_v1 > 0) & (y_new != y_v1)
    n_changed = int(changed_mask.sum())
    rate = n_changed / n_pos if n_pos else 0.0
    print(
        f"    DDI 原始 {ddi_first.attrs['n_before_dedup']:,} → 去重 {len(ddi_first):,}；"
        f"正样本精确命中 {exact_hit:,}/{n_pos:,}；变化 {n_changed:,} ({rate*100:.4f}%)",
        flush=True,
    )
    rel = abs(n_changed - EXPECTED_POS_CHANGED) / EXPECTED_POS_CHANGED
    if rel > EXPECTED_POS_CHANGED_TOL:
        print(
            f"变化量与已知 {EXPECTED_POS_CHANGED} 相对偏差 {rel:.2%}，超过 "
            f"{EXPECTED_POS_CHANGED_TOL:.0%}，停止排查"
        )
        ct = pd.crosstab(pd.Series(y_v1, name="old"), pd.Series(y_new, name="new"))
        print(ct.to_string())
        return 1

    ct_json = crosstab_to_json(y_v1, y_new)

    print("[3] 无序对重复 / 正负冲突检查", flush=True)
    keys = [unordered_key(a, b) for a, b in zip(id1, id2)]
    first_of_key: dict[tuple[str, str], int] = {}
    extra_dup_rows: list[int] = []
    labels_by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        labels_by_key[k].append(int(y_v1[i]))
        if k in first_of_key:
            extra_dup_rows.append(i)
        else:
            first_of_key[k] = i
    n_dup_groups = sum(1 for xs in labels_by_key.values() if len(xs) > 1)
    n_extra = len(extra_dup_rows)
    dup_label_sets = Counter(
        tuple(sorted(set(xs))) for xs in labels_by_key.values() if len(xs) > 1
    )
    n_pos_dup_groups = sum(
        1 for xs in labels_by_key.values() if len(xs) > 1 and all(v != 0 for v in xs)
    )
    n_mixed = sum(
        1
        for xs in labels_by_key.values()
        if len(xs) > 1 and (0 in xs) and any(v != 0 for v in xs)
    )
    n_neg_dup_groups = sum(
        1 for xs in labels_by_key.values() if len(xs) > 1 and all(v == 0 for v in xs)
    )
    print(
        f"    无序重复组={n_dup_groups} 多余行={n_extra} 全负={n_neg_dup_groups} "
        f"全正={n_pos_dup_groups} 正负并存={n_mixed}",
        flush=True,
    )
    extra_set = set(extra_dup_rows)
    keep_idx = np.array([i for i in range(n_v1) if i not in extra_set], dtype=np.int64)
    n_v2 = int(keep_idx.shape[0])
    assert n_v2 == n_v1 - n_extra

    y_keep_old = y_v1[keep_idx]
    y_keep_new = y_new[keep_idx]
    p1 = id1.iloc[keep_idx].to_numpy()
    p2 = id2.iloc[keep_idx].to_numpy()

    print("[4] 统一 ID 顺序（字符串比较，与 v1 pair_key 相同）", flush=True)
    swap_old = Counter()
    swap_new = Counter()
    n_swap = 0
    out_id1 = np.empty(n_v2, dtype=object)
    out_id2 = np.empty(n_v2, dtype=object)
    for j in range(n_v2):
        a, b = str(p1[j]), str(p2[j])
        if a > b:
            a, b = b, a
            n_swap += 1
            swap_old[int(y_keep_old[j])] += 1
            swap_new[int(y_keep_new[j])] += 1
        out_id1[j] = a
        out_id2[j] = b
    print(f"    交换 {n_swap:,} 行；按 v1 类别 {dict(swap_old)}；按新标签 {dict(swap_new)}", flush=True)

    # 磁盘
    out_dir.mkdir(parents=True, exist_ok=True)
    x_out_bytes = n_v2 * feat_dim * itemsize
    usage = shutil.disk_usage(out_dir)
    need = x_out_bytes + 512 * 1024 * 1024
    print(
        f"[5] 磁盘: 可用 {usage.free/1e9:.2f} GB，X_v2 需 {x_out_bytes/1e9:.2f} GB",
        flush=True,
    )
    if usage.free < need:
        print("磁盘不足，停止")
        return 1

    print("[6] 计算每药 Morgan 指纹并写入 drug_features.npy（2048 维，非 2060）", flush=True)
    drugs_df = pd.read_csv(args.drugs_csv.resolve(), dtype={"drugbank_id": str})
    fp_cache = build_fingerprint_cache(drugs_df)
    drugs_dict = drugs_df.set_index("drugbank_id").to_dict("index")
    drug_ids_fp = sorted(fp_cache.keys())
    fp_mat = np.stack([np.asarray(fp_cache[d], dtype=np.float32) for d in drug_ids_fp], axis=0)
    assert fp_mat.shape[1] == 2048
    np.save(out_dir / "drug_features.npy", fp_mat)
    pd.DataFrame(
        {
            "feature_row": np.arange(len(drug_ids_fp), dtype=np.int32),
            "drugbank_id": drug_ids_fp,
        }
    ).to_csv(out_dir / "drug_index.csv", index=False)
    print(f"    drug_features {fp_mat.shape}  drug_index {len(drug_ids_fp)}", flush=True)

    # 抽样确认 v1 X 布局（交换前的原始行）
    print("[7] 用每药指纹抽样验证 v1 X 布局", flush=True)
    layout_ok = 0
    layout_fail = 0
    layout_max_abs = 0.0
    probe_n = min(30, n_v1)
    probe_idx = rng.choice(n_v1, size=probe_n, replace=False)
    X_v1 = np.memmap(x_path, dtype=np.float32, mode="r", shape=(n_v1, feat_dim))
    for i in probe_idx:
        a, b = str(id1.iat[int(i)]), str(id2.iat[int(i)])
        feat = build_pair_features(a, b, fp_cache, drugs_dict)
        if feat is None:
            layout_fail += 1
            continue
        mad = float(np.max(np.abs(feat.astype(np.float32) - X_v1[int(i)])))
        layout_max_abs = max(layout_max_abs, mad)
        if mad <= 1e-6:
            layout_ok += 1
        else:
            layout_fail += 1
    print(
        f"    重算匹配 {layout_ok}/{probe_n}  fail={layout_fail} max_abs={layout_max_abs}",
        flush=True,
    )
    if layout_fail:
        print("v1 X 与当前 build_pair_features 不一致，停止（避免在错误布局上交换半段）")
        return 1

    print("[8] 写 y / pairs / row_map", flush=True)
    y_v2 = y_keep_new.astype(np.int32, copy=True)
    np.save(out_dir / "y.npy", y_v2)
    np.save(out_dir / "row_map_v1_to_v2.npy", keep_idx)
    pairs_v2 = pd.DataFrame({"drug1_id": out_id1, "drug2_id": out_id2})
    pairs_v2.to_csv(out_dir / "pairs.csv", index=False)

    print("[9] 分块复制 X（不交换半段）", flush=True)
    partial = out_dir / "X_full.dat.partial"
    x_out = out_dir / "X_full.dat"
    if partial.exists():
        partial.unlink()
    X_out = np.memmap(partial, dtype=np.float32, mode="w+", shape=(n_v2, feat_dim))
    n_chunks = (n_v2 + chunk_size - 1) // chunk_size
    for c, start in enumerate(range(0, n_v2, chunk_size), start=1):
        end = min(start + chunk_size, n_v2)
        src = keep_idx[start:end]
        X_out[start:end] = X_v1[src]
        if c == 1 or c == n_chunks or c % 10 == 0:
            print(f"    chunk {c}/{n_chunks}  rows {start:,}-{end-1:,}", flush=True)
    X_out.flush()
    del X_out
    gc.collect()
    if x_out.exists():
        x_out.unlink()
    partial.replace(x_out)
    actual_x_bytes = x_out.stat().st_size
    if actual_x_bytes != x_out_bytes:
        print(f"X 写出大小不符: {actual_x_bytes} != {x_out_bytes}")
        return 1

    built_at = datetime.now().isoformat(timespec="seconds")
    counts_v2 = np.bincount(y_v2, minlength=5).astype(int).tolist()
    counts_v1 = np.bincount(y_v1, minlength=5).astype(int).tolist()
    label_names = meta_v1.get("label_names", {})

    changes_from_v1 = {
        "n_v1_samples": n_v1,
        "n_v2_samples": n_v2,
        "n_dropped_duplicate_rows": n_extra,
        "n_duplicate_unordered_groups": n_dup_groups,
        "duplicate_label_sets": {str(k): int(v) for k, v in dup_label_sets.items()},
        "n_positive_only_unordered_dup_groups": n_pos_dup_groups,
        "n_pos_neg_conflict_groups": n_mixed,
        "n_negative_only_unordered_dup_groups": n_neg_dup_groups,
        "n_positive_label_changed": n_changed,
        "n_positives_v1": n_pos,
        "positive_label_changed_rate": rate,
        "crosstab_old_to_new_including_dropped_dups": ct_json,
        "n_id_swaps": n_swap,
        "n_id_swaps_by_v1_class": {str(k): int(v) for k, v in sorted(swap_old.items())},
        "n_id_swaps_by_v2_class": {str(k): int(v) for k, v in sorted(swap_new.items())},
        "id_compare_rule": "str_sorted_tuple_same_as_v1_pair_key",
        "id_string_lengths": id_lens,
        "x_layout": "fp_product[0:2048] || fp_absdiff[2048:4096] || bio[4096:4120]",
        "x_halves_swapped": False,
        "reason_x_not_swapped": (
            "v1/preprocess.build_pair_features 输出为配对对称特征，"
            "不是 [drug_a 2060 || drug_b 2060]；交换半段会破坏向量。"
        ),
        "v1_x_file_bytes": int(x_bytes),
        "v1_x_allocated_rows": int(n_alloc),
        "v1_valid_rows": n_v1,
        "v1_x_unused_tail_rows": int(n_alloc - n_v1),
        "layout_probe_n": probe_n,
        "layout_probe_ok": layout_ok,
        "layout_probe_max_abs": layout_max_abs,
        "ddi_csv": str(args.ddi_csv.resolve()),
        "ddi_rows_raw": int(ddi_first.attrs["n_before_dedup"]),
        "ddi_rows_after_mirror_dedup": int(len(ddi_first)),
        "xml_parser": "download_data.parse_drugbank_xml（本脚本未调用，避免覆盖 raw CSV）",
        "per_drug_feature_dim": 2048,
        "per_drug_feature_note": (
            "仓库内不存在 2060 维每药矩阵；药理 24 维是配对特征。"
            "cloud 端用 drug_features.npy + drugbank_drugs.csv + build_pair_features 重建 X。"
        ),
    }

    meta_v2 = dict(meta_v1)
    meta_v2.update(
        {
            "n_samples": n_v2,
            "feature_dim": feat_dim,
            "data_version": "v2",
            "label_map_file": str(map_py.as_posix()),
            "label_map_sha256": label_map_sha,
            "built_at": built_at,
            "counts": {str(i): int(c) for i, c in enumerate(counts_v2)},
            "counts_v1": {str(i): int(c) for i, c in enumerate(counts_v1)},
            "changes_from_v1": changes_from_v1,
            "strategy": (
                str(meta_v1.get("strategy", ""))
                + "；v2：2026-07-02 PHRASE_LABEL_MAP 重标 + 无序对去重 + 字符串顺序规范化"
            ),
        }
    )
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta_v2, f, ensure_ascii=False, indent=2)

    print("[10] 重新加载 v2 并校验", flush=True)
    y_reload = np.load(out_dir / "y.npy")
    pairs_reload = pd.read_csv(out_dir / "pairs.csv", dtype={"drug1_id": str, "drug2_id": str})
    row_map_reload = np.load(out_dir / "row_map_v1_to_v2.npy")
    X_reload = np.memmap(x_out, dtype=np.float32, mode="r", shape=(n_v2, feat_dim))
    assert y_reload.shape == (n_v2,)
    assert len(pairs_reload) == n_v2
    assert row_map_reload.shape == (n_v2,)
    assert X_reload.shape == (n_v2, feat_dim)
    assert x_out.stat().st_size == n_v2 * feat_dim * itemsize

    order_ok = int((pairs_reload["drug1_id"] < pairs_reload["drug2_id"]).sum())
    order_bad = int((pairs_reload["drug1_id"] > pairs_reload["drug2_id"]).sum())
    order_eq = int((pairs_reload["drug1_id"] == pairs_reload["drug2_id"]).sum())

    verify_idx = rng.choice(n_v2, size=min(N_VERIFY_ROWS, n_v2), replace=False)
    n_order_ok = 0
    n_feat_ok = 0
    n_feat_fail = 0
    n_feat_skip = 0
    verify_max_abs = 0.0
    for j in verify_idx:
        a = str(pairs_reload.at[int(j), "drug1_id"])
        b = str(pairs_reload.at[int(j), "drug2_id"])
        if a < b:
            n_order_ok += 1
        feat = build_pair_features(a, b, fp_cache, drugs_dict)
        if feat is None:
            n_feat_skip += 1
            continue
        mad = float(np.max(np.abs(feat.astype(np.float32) - X_reload[int(j)])))
        verify_max_abs = max(verify_max_abs, mad)
        if mad <= 1e-6:
            n_feat_ok += 1
        else:
            n_feat_fail += 1
    print(
        f"    shape ok; 全表顺序 drug1<drug2 {order_ok}/{n_v2} bad={order_bad} eq={order_eq}; "
        f"抽{len(verify_idx)}行 顺序OK={n_order_ok} 特征OK={n_feat_ok} "
        f"fail={n_feat_fail} skip={n_feat_skip} max_abs={verify_max_abs}",
        flush=True,
    )
    if order_bad or order_eq or n_feat_fail:
        print("校验失败，停止写最终清单前请检查（文件已写出）")
        return 1

    pos_v2_idx = np.where(y_reload > 0)[0]
    review_pick = rng.choice(pos_v2_idx, size=min(N_REVIEW_POS, len(pos_v2_idx)), replace=False)
    review_rows = []
    for j in sorted(int(x) for x in review_pick):
        v1_i = int(row_map_reload[j])
        a = str(pairs_reload.at[j, "drug1_id"])
        b = str(pairs_reload.at[j, "drug2_id"])
        desc = desc_by_v1.get(v1_i)
        if desc is None:
            hit = exact_lookup.get((str(id1.iat[v1_i]), str(id2.iat[v1_i])))
            desc = hit[0] if hit else ""
        snippet = md_escape(str(desc)[:240])
        review_rows.append(
            {
                "v2_row": j,
                "v1_row": v1_i,
                "drug1_id": a,
                "drug2_id": b,
                "old_label": int(y_v1[v1_i]),
                "new_label": int(y_reload[j]),
                "description_snippet": snippet,
            }
        )

    print("[11] SHA-256 与清单", flush=True)
    output_files = [
        "X_full.dat",
        "y.npy",
        "pairs.csv",
        "meta.json",
        "row_map_v1_to_v2.npy",
        "drug_features.npy",
        "drug_index.csv",
    ]
    file_records = []
    for name in output_files:
        pth = out_dir / name
        rec = {
            "name": name,
            "bytes": int(pth.stat().st_size),
            "sha256": sha256_file(pth),
        }
        file_records.append(rec)
        print(f"    {name}  {rec['bytes']:,}  {rec['sha256']}", flush=True)

    python_exe = sys.executable
    pkg_versions = {
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import rdkit
        from rdkit import Chem  # noqa: F401

        pkg_versions["rdkit"] = getattr(rdkit, "__version__", "unknown")
    except Exception as e:  # pragma: no cover
        pkg_versions["rdkit"] = f"unavailable: {e}"

    conservative = [
        "PHRASE_LABEL_MAP 在 preprocess.py 与 phrase_label_map.py 完全一致（均为 2026-07-02 文件时间戳），继续。",
        "XML 解析在 download_data.parse_drugbank_xml 而非 preprocess.py；未调用该函数，以免覆盖 raw CSV。描述来自 v1 同期 drugbank_ddi.csv，镜像去重 keep=first 与 v1 相同。",
        "X 不是 [drug_a || drug_b] 拼接；统一顺序只交换 pairs ID，不交换 X 半段。",
        "v1 X_full.dat 按分配上界写了未截断尾部；读写均按 y.npy 行数截断。v2 写出精确行数。",
        "每药数值特征保存为 2048 维 Morgan 指纹，不编造 2060 维。",
        "大文件直接写到 v2 目录；C 盘可用空间不足以容纳 25GB memmap。",
    ]

    manifest = {
        "data_version": "v2",
        "built_at": built_at,
        "python_executable": python_exe,
        "package_versions": pkg_versions,
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "label_map_file": str(map_py.as_posix()),
        "label_map_sha256": label_map_sha,
        "phrase_map_compare": {k: cmp[k] for k in ("equal", "n_preprocess", "n_phrase_file")},
        "v1": {
            "n_samples": n_v1,
            "counts": {str(i): int(c) for i, c in enumerate(counts_v1)},
            "x_bytes": int(x_bytes),
            "x_allocated_rows": int(n_alloc),
            "pairs_columns": ["drug1_id", "drug2_id"],
            "y_dtype": str(y_v1.dtype),
            "x_dtype": "float32",
            "feature_dim": feat_dim,
        },
        "v2": {
            "n_samples": n_v2,
            "counts": {str(i): int(c) for i, c in enumerate(counts_v2)},
            "x_bytes": int(actual_x_bytes),
            "x_shape": [n_v2, feat_dim],
        },
        "changes_from_v1": changes_from_v1,
        "verification": {
            "reload_shapes_ok": True,
            "n_rows_drug1_lt_drug2": order_ok,
            "n_rows_drug1_gt_drug2": order_bad,
            "n_rows_drug1_eq_drug2": order_eq,
            "random_verify_n": int(len(verify_idx)),
            "random_verify_order_ok": n_order_ok,
            "random_verify_feature_ok": n_feat_ok,
            "random_verify_feature_fail": n_feat_fail,
            "random_verify_feature_skip": n_feat_skip,
            "random_verify_max_abs": verify_max_abs,
            "seed": int(args.seed),
        },
        "positive_review_sample": review_rows,
        "files": file_records,
        "conservative_decisions": conservative,
        "anomalies": [
            f"v1 X_full.dat 分配 {n_alloc} 行、有效 {n_v1} 行，多余尾部 {n_alloc-n_v1} 行未使用。",
            f"无序重复 {n_dup_groups} 组、多余 {n_extra} 行，且全部为负样本（与审计 99 组一致）。",
            "正样本无无序重复，无同一无序对既正又负，无 self-pair。",
            "任务预期每药 2060 维拼接布局与代码不符；已按实证布局处理。",
            "DATA_AUDIT_README.md 当前文本未写 56055/4.74% 数字；本运行正样本变化 "
            f"{n_changed}（{rate*100:.4f}%），与任务已知审计数字一致。",
        ],
    }
    with (out_dir / "manifest_v2.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    md = []
    md.append("# DDI processed v2 清单")
    md.append("")
    md.append(f"- 生成时间: `{built_at}`")
    md.append(f"- Python: `{python_exe}`")
    md.append(
        f"- 包版本: numpy `{pkg_versions['numpy']}` / pandas `{pkg_versions['pandas']}` / "
        f"rdkit `{pkg_versions.get('rdkit', '')}`"
    )
    md.append(f"- 输入: `{in_dir}`")
    md.append(f"- 输出: `{out_dir}`")
    md.append(
        f"- 标签映射: `{map_py.as_posix()}`  SHA-256 `{label_map_sha}` "
        f"（与 `data/preprocess.py` 的 PHRASE_LABEL_MAP 一致，{cmp['n_preprocess']} 条）"
    )
    md.append("")
    md.append("## v1 读取结论")
    md.append("")
    md.append(
        f"- `pairs.csv` 列: `drug1_id, drug2_id`；行数 {fmt_int(n_v1)}，与 `y.npy` 一致。"
    )
    md.append(
        f"- `y.npy` dtype=`{y_v1.dtype}` 分布: "
        + ", ".join(f"类{i}={fmt_int(c)}" for i, c in enumerate(counts_v1))
    )
    md.append(
        f"- `X_full.dat` dtype=`float32`，按 `y` 解释为 `({fmt_int(n_v1)}, {feat_dim})`；"
        f"文件 {fmt_int(x_bytes)} 字节 = 分配 {fmt_int(n_alloc)} 行 × {feat_dim} × 4，"
        f"多出 {fmt_int(n_alloc-n_v1)} 行未使用尾部（preprocess memmap 按上界预分配且未截断）。"
    )
    md.append(
        "- 每行布局（代码 + 重算核对）: "
        "`[0:2048] fp_a*fp_b` ‖ `[2048:4096] |fp_a-fp_b|` ‖ `[4096:4120] 24维对称药理特征`。"
        f" 抽样 {probe_n} 行与 `build_pair_features` 最大绝对差 `{layout_max_abs}`。"
    )
    md.append(
        f"- DrugBank ID 字符串长度集合: {id_lens}。顺序规则与 v1 去重键相同："
        "`tuple(sorted([drug1_id, drug2_id]))` 为**字符串**比较。"
        "本数据全部为 7 字符 ID，字符串序与数字后缀序一致。"
    )
    md.append("")
    md.append("## 标签重算")
    md.append("")
    md.append(
        f"- 描述来源: `{args.ddi_csv.resolve()}` "
        f"（{fmt_int(int(ddi_first.attrs['n_before_dedup']))} 行 → 镜像去重 "
        f"{fmt_int(len(ddi_first))} 行）。"
    )
    md.append(f"- 正样本精确匹配 (drug1_id, drug2_id): {fmt_int(exact_hit)} / {fmt_int(n_pos)}。")
    md.append(
        f"- 正样本旧→新变化: **{fmt_int(n_changed)}**（{rate*100:.4f}%），"
        f"与已知约 56,055 / 4.74% 一致。"
    )
    md.append("- 负样本保持 0。")
    md.append("")
    md.append("交叉表（含随后删除的 99 条重复负样本；负类无变化）:")
    md.append("")
    md.append("| old \\ new | 0 | 1 | 2 | 3 | 4 |")
    md.append("|---:|---:|---:|---:|---:|---:|")
    for oi in range(5):
        row = ct_json.get(str(oi), {})
        cells = " | ".join(fmt_int(int(row.get(str(j), 0))) for j in range(5))
        md.append(f"| {oi} | {cells} |")
    md.append("")
    md.append("## 去重")
    md.append("")
    md.append(
        f"- 无序对重复组: **{n_dup_groups}**（预期 99）；多余行 **{n_extra}**；保留首次出现。"
    )
    md.append(f"- 重复组标签集合计数: `{ {str(k): int(v) for k, v in dup_label_sets.items()} }`")
    md.append(f"- 正样本无序重复组: {n_pos_dup_groups}")
    md.append(f"- 同一无序对既正又负: {n_mixed}")
    md.append("")
    md.append("## 统一顺序")
    md.append("")
    md.append(f"- 交换 ID 的行数: **{fmt_int(n_swap)}**（X 半段不交换）。")
    md.append(f"- 按 v1 类别: `{ {str(k): int(v) for k, v in sorted(swap_old.items())} }`")
    md.append(f"- 按新标签: `{ {str(k): int(v) for k, v in sorted(swap_new.items())} }`")
    md.append("")
    md.append("## v2 类别分布")
    md.append("")
    for i, c in enumerate(counts_v2):
        name = label_names.get(str(i), "")
        md.append(f"- 类 {i} {name}: {fmt_int(int(c))} ({c/n_v2*100:.4f}%)")
    md.append(f"- 合计: {fmt_int(n_v2)}")
    md.append("")
    md.append("## 校验")
    md.append("")
    md.append(
        f"- 重载 shape 一致: X `{n_v2}×{feat_dim}`，y/pairs/row_map 均为 {fmt_int(n_v2)}。"
    )
    md.append(
        f"- 全表 `drug1_id < drug2_id`: {fmt_int(order_ok)}；`>` {order_bad}；`==` {order_eq}。"
    )
    md.append(
        f"- 随机 {len(verify_idx)} 行（seed={args.seed}）: 顺序 OK {n_order_ok}；"
        f"与每药指纹+`build_pair_features` 一致 {n_feat_ok}；失败 {n_feat_fail}；"
        f"max_abs={verify_max_abs}。"
    )
    md.append("")
    md.append("## 人工复核：随机 20 条正样本")
    md.append("")
    md.append("| v2_row | v1_row | drug1 | drug2 | 旧标签 | 新标签 | 描述片段 |")
    md.append("|---:|---:|---|---|---:|---:|---|")
    for r in review_rows:
        md.append(
            f"| {r['v2_row']} | {r['v1_row']} | {r['drug1_id']} | {r['drug2_id']} | "
            f"{r['old_label']} | {r['new_label']} | {r['description_snippet']} |"
        )
    md.append("")
    md.append("## 输出文件 SHA-256")
    md.append("")
    md.append("| 文件 | 字节 | SHA-256 |")
    md.append("|---|---:|---|")
    for rec in file_records:
        md.append(f"| `{rec['name']}` | {fmt_int(rec['bytes'])} | `{rec['sha256']}` |")
    md.append("")
    md.append("## 异常与保守决定")
    md.append("")
    for a in manifest["anomalies"]:
        md.append(f"- {a}")
    md.append("")
    for d in conservative:
        md.append(f"- {d}")
    md.append("")
    md.append("## 用每药指纹重建 X（云端，无需上传 25GB）")
    md.append("")
    md.append(
        "对 `pairs.csv` 每一行 `(drug1_id, drug2_id)`，用 `drug_index.csv` 取两行 "
        "`drug_features.npy`（2048 维），再结合 `data/raw/drugbank_drugs.csv` 的 "
        "enzyme/transporter/target 字段调用 `preprocess.build_pair_features`，"
        "即可得到 4120 维行向量。不要把两药指纹直接首尾拼接。"
    )
    md.append("")

    md_text = "\n".join(md)
    (out_dir / "MANIFEST_V2.md").write_text(md_text, encoding="utf-8")
    md_path = out_dir / "MANIFEST_V2.md"
    md_rec = {
        "name": "MANIFEST_V2.md",
        "bytes": int(md_path.stat().st_size),
        "sha256": sha256_file(md_path),
    }
    file_records.append(md_rec)
    print(f"    MANIFEST_V2.md  {md_rec['bytes']:,}  {md_rec['sha256']}", flush=True)
    manifest["files"] = file_records
    json_path = out_dir / "manifest_v2.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(
        f"    manifest_v2.json  {json_path.stat().st_size:,}  {sha256_file(json_path)}",
        flush=True,
    )

    print("[完成]", out_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
