"""从 final_full 推理包构造 Exp-5 的 8216 维融合特征（只 import predict_pairs，不复制）。

用法:
  python -u models/analysis/build_fusion_features.py \\
      --bundle models/results/final_smoke_full/final_full \\
      --data_dir data/processed/v2 \\
      --from_test_idx models/results/final_smoke_full/final_full/test_idx.npy \\
      --n 200 --seed 0 \\
      --out_dir models/results/analysis_smoke/shap_smoke
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
from pathlib import Path

import numpy as np
import pandas as pd
import torch


from graphtree_ddi.data.preprocess import FEAT_DIM, build_pair_features  # noqa: E402
from graphtree_ddi.models.predict_pairs import (  # noqa: E402
    _canonical,
    _resolve_drugs_csv,
    load_bundle,
    pair_ops_embed,
)


def _load_indices(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as z:
            if "test_idx" not in z.files:
                raise KeyError(f"{path.name} 不含 test_idx")
            return np.asarray(z["test_idx"], dtype=np.int64).ravel()
    return np.asarray(np.load(path), dtype=np.int64).ravel()


def _drug_name(drugs_dict: dict, did: str) -> str:
    rec = drugs_dict.get(str(did)) or {}
    name = rec.get("name")
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return str(did)
    s = str(name).strip()
    return s if s else str(did)


def build_fusion_matrix(
    bundle: Path,
    data_dir: Path,
    pairs: pd.DataFrame,
    *,
    drugs_csv: Path | None = None,
    device: torch.device | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """pairs 需含 drug_a, drug_b；可选 y_true / test_idx。"""
    need = {"drug_a", "drug_b"}
    if not need.issubset(pairs.columns):
        raise ValueError(f"pairs 必须含 {need}，实际 {list(pairs.columns)}")
    drugs_csv = _resolve_drugs_csv(data_dir, None if drugs_csv is None else str(drugs_csv))
    drugs_df = pd.read_csv(drugs_csv)
    drugs_dict = {str(row["drugbank_id"]): row.to_dict() for _, row in drugs_df.iterrows()}
    print(f"[fusion] drugs_csv={drugs_csv}  n={len(drugs_dict):,}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[fusion] load_bundle {bundle} on {device}")
    ctx = load_bundle(Path(bundle), device)
    idmap = ctx["drug_id_to_idx"]
    fp_cache = ctx["fp_cache"]
    drug_emb = np.asarray(ctx["drug_emb"], dtype=np.float32)
    layout = ctx.get("feature_layout") or {}
    fusion_dim = int(layout.get("fusion", {}).get("dim", FEAT_DIM + 4096))

    xs: list[np.ndarray] = []
    meta_rows: list[dict] = []
    n_skip = 0
    for i, rec in pairs.iterrows():
        raw_a = str(rec["drug_a"]).strip()
        raw_b = str(rec["drug_b"]).strip()
        a, b = _canonical(raw_a, raw_b)
        in_index = a in idmap and b in idmap
        feat = build_pair_features(a, b, fp_cache, drugs_dict) if in_index else None
        if (not in_index) or feat is None:
            n_skip += 1
            continue
        ia, ib = int(idmap[a]), int(idmap[b])
        gnn = pair_ops_embed(drug_emb[ia], drug_emb[ib]).reshape(-1)
        xgb = np.asarray(feat, dtype=np.float32).reshape(-1)
        full = np.concatenate([xgb, gnn], axis=0).astype(np.float32)
        if full.shape[0] != fusion_dim:
            # 仍写出，但告警（冒烟模型可能维度一致）
            if i == pairs.index[0]:
                print(f"[warn] 融合维 {full.shape[0]} vs layout {fusion_dim}")
        xs.append(full)
        meta = dict(
            row=len(xs) - 1,
            drug_a=raw_a,
            drug_b=raw_b,
            drug_a_canon=a,
            drug_b_canon=b,
            name_a=_drug_name(drugs_dict, a),
            name_b=_drug_name(drugs_dict, b),
            in_index=1,
        )
        if "y_true" in rec.index and rec["y_true"] == rec["y_true"]:
            meta["y_true"] = int(rec["y_true"])
        if "test_idx" in rec.index and rec["test_idx"] == rec["test_idx"]:
            meta["test_idx"] = int(rec["test_idx"])
        meta_rows.append(meta)

    if not xs:
        raise SystemExit("没有任何可构造的药对（均不在药物索引或特征失败）")
    X = np.stack(xs, axis=0)
    meta = pd.DataFrame(meta_rows)
    print(
        f"[fusion] X={X.shape}  skipped={n_skip}  "
        f"xgb={FEAT_DIM} + gnn_pair={X.shape[1] - FEAT_DIM}"
    )
    return X, meta


def pairs_from_test_idx(
    data_dir: Path,
    idx: np.ndarray,
    n: int | None,
    seed: int,
) -> pd.DataFrame:
    pairs_path = Path(data_dir) / "pairs.csv"
    y_path = Path(data_dir) / "y.npy"
    pairs = pd.read_csv(pairs_path)
    y = np.load(y_path) if y_path.exists() else None
    idx = np.asarray(idx, dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < len(pairs))]
    if n is not None and int(n) > 0 and int(n) < len(idx):
        rng = np.random.default_rng(int(seed))
        take = np.sort(rng.choice(len(idx), size=int(n), replace=False))
        idx = idx[take]
    d1 = pairs["drug1_id"].astype(str).to_numpy()
    d2 = pairs["drug2_id"].astype(str).to_numpy()
    out = pd.DataFrame({
        "drug_a": d1[idx],
        "drug_b": d2[idx],
        "test_idx": idx,
    })
    if y is not None:
        yy = np.asarray(y).ravel()
        ok = idx[idx < len(yy)]
        out = out.iloc[: len(ok)].copy()
        out["y_true"] = yy[out["test_idx"].to_numpy()]
    print(f"[fusion] sampled {len(out)} pairs from {pairs_path.name}")
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build Exp-5 8216-d fusion features")
    p.add_argument("--bundle", required=True, help="final_full 目录")
    p.add_argument("--data_dir", required=True, help="processed 目录（pairs.csv / y.npy）")
    p.add_argument("--pairs_csv", default="", help="列 drug_a,drug_b（可选 y_true）")
    p.add_argument("--from_test_idx", default="",
                   help="npz（键 test_idx）或 npy 索引，从封存测试集抽样")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--drugs_csv", default="")
    p.add_argument("--mark_smoke", action="store_true",
                   help="写入不可用于论文的标注文件")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    bundle = Path(args.bundle)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pairs_csv:
        pairs = pd.read_csv(args.pairs_csv)
        if "drug_a" not in pairs.columns and {"drug1_id", "drug2_id"} <= set(pairs.columns):
            pairs = pairs.rename(columns={"drug1_id": "drug_a", "drug2_id": "drug_b"})
    elif args.from_test_idx:
        idx = _load_indices(Path(args.from_test_idx))
        pairs = pairs_from_test_idx(data_dir, idx, args.n, args.seed)
    else:
        fallback = bundle / "test_idx.npy"
        if not fallback.exists():
            raise SystemExit("需要 --pairs_csv 或 --from_test_idx（或 bundle/test_idx.npy）")
        idx = _load_indices(fallback)
        pairs = pairs_from_test_idx(data_dir, idx, args.n, args.seed)

    X, meta = build_fusion_matrix(
        bundle, data_dir, pairs,
        drugs_csv=Path(args.drugs_csv) if args.drugs_csv else None,
    )
    x_path = out_dir / "fusion_X.npy"
    csv_path = out_dir / "fusion_pairs.csv"
    np.save(x_path, X)
    meta.to_csv(csv_path, index=False, encoding="utf-8-sig")
    info = dict(
        X=str(x_path), pairs=str(csv_path),
        n=int(X.shape[0]), dim=int(X.shape[1]),
        bundle=str(bundle), data_dir=str(data_dir),
        layout="[0:4120] xgb_v2  [4120:8216] gnn pair_ops concat/diff/prod",
    )
    (out_dir / "fusion_manifest.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    if args.mark_smoke:
        (out_dir / "NOT_FOR_MANUSCRIPT.txt").write_text(
            "Smoke fusion features / SHAP. Do not use in the paper.\n",
            encoding="utf-8",
        )
    print(f"[fusion] wrote {x_path}  {csv_path}")


if __name__ == "__main__":
    main()
