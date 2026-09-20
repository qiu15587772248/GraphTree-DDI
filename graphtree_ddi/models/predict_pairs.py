"""任意药物对推理：Exp-1 / Exp-3 / Exp-5。

输入 CSV 列: drug_a, drug_b（DrugBank ID）。内部按 drug_a < drug_b
规范顺序后再构造特征，与 processed v2 一致。

不在药物索引中的 ID：该行对应模型输出为空值，并在 stderr 统计。

用法:
  python -u models/predict_pairs.py \\
      --bundle models/results/final/final_full \\
      --data_dir data/processed \\
      --input pairs_query.csv \\
      --out pred.csv
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


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

from graphtree_ddi.data.preprocess import build_pair_features  # noqa: E402
from graphtree_ddi.models.gnn import NUM_RELATIONS, RGCN_DDI  # noqa: E402


def pair_ops_embed(h_a: np.ndarray, h_b: np.ndarray) -> np.ndarray:
    h_a = np.asarray(h_a)
    h_b = np.asarray(h_b)
    squeeze = h_a.ndim == 1
    if squeeze:
        h_a = h_a[None, :]
        h_b = h_b[None, :]
    out = np.concatenate([h_a, h_b, h_a - h_b, h_a * h_b], axis=1).astype(np.float32)
    return out[0] if squeeze else out


def _resolve_drugs_csv(data_dir: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    cands = [
        data_dir / "drugbank_drugs.csv",
        data_dir.parent / "raw" / "drugbank_drugs.csv",
        data_dir.parent.parent / "raw" / "drugbank_drugs.csv",
        _REPO_ROOT / "data" / "raw" / "drugbank_drugs.csv",
    ]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(
        "未找到 drugbank_drugs.csv，请传 --drugs_csv。"
        f" 已尝试: {[str(c) for c in cands]}",
    )


def _load_xgb(bundle: Path, stem: str):
    from xgboost import XGBClassifier
    json_p = bundle / f"{stem}.json"
    pkl_p = bundle / f"{stem}.pkl"
    clf = XGBClassifier()
    if json_p.exists():
        clf.load_model(str(json_p))
        return clf
    if pkl_p.exists():
        import pickle
        with open(pkl_p, "rb") as f:
            return pickle.load(f)
    return None


def _canonical(a: str, b: str) -> tuple[str, str]:
    a = str(a).strip()
    b = str(b).strip()
    return (a, b) if a < b else (b, a)


def load_bundle(bundle: Path, device: torch.device) -> dict:
    with open(bundle / "exp3_config.json", encoding="utf-8") as f:
        exp3_cfg = json.load(f)
    with open(bundle / "drug_id_to_idx.json", encoding="utf-8") as f:
        drug_id_to_idx = json.load(f)
    with open(bundle / "kg_meta.json", encoding="utf-8") as f:
        kg_meta = json.load(f)
    with open(bundle / "classes.json", encoding="utf-8") as f:
        classes = json.load(f)
    with open(bundle / "pair_ops.json", encoding="utf-8") as f:
        pair_ops = json.load(f)
    with open(bundle / "feature_layout.json", encoding="utf-8") as f:
        feature_layout = json.load(f)

    fps = np.load(bundle / "drug_fingerprints.npy")
    drug_ids = kg_meta["drug_ids"]
    fp_cache = {did: fps[i] for i, did in enumerate(drug_ids)}

    ei = torch.from_numpy(np.load(bundle / "graph_edge_index.npy").astype(np.int64))
    et = torch.from_numpy(np.load(bundle / "graph_edge_type.npy").astype(np.int64))
    n_nodes = int(exp3_cfg["n_drugs"]) + int(exp3_cfg["n_bio"])
    graph = Data(edge_index=ei, edge_type=et, num_nodes=n_nodes)

    model = RGCN_DDI(
        n_drugs=int(exp3_cfg["n_drugs"]),
        n_bio=int(exp3_cfg["n_bio"]),
        hidden_dim=int(exp3_cfg["hidden_dim"]),
        emb_dim=int(exp3_cfg["emb_dim"]),
        num_rels=int(exp3_cfg.get("num_rels", NUM_RELATIONS)),
        num_bases=int(exp3_cfg["num_bases"]),
        dropout=float(exp3_cfg.get("dropout", 0.3)),
        norm_type=exp3_cfg.get("norm_type", "batch"),
        decoder_type=exp3_cfg.get("decoder_type", "mlp"),
        fp_dim=int(exp3_cfg.get("fp_dim", 2048)),
    )
    state = torch.load(bundle / "exp3_rgcn.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    drug_fp = torch.from_numpy(fps.astype(np.float32)).to(device)
    graph = graph.to(device)
    with torch.no_grad():
        h_all = model.encode(drug_fp, graph.edge_index, graph.edge_type)
        drug_emb = h_all[: model.n_drugs].detach().cpu().numpy()

    emb_path = bundle / "drug_embeddings.npy"
    if emb_path.exists():
        saved = np.load(emb_path)
        if saved.shape == drug_emb.shape:
            drug_emb = saved.astype(np.float32)

    return dict(
        device=device, model=model, h_all=h_all, drug_emb=drug_emb,
        drug_id_to_idx=drug_id_to_idx, fp_cache=fp_cache,
        exp1=_load_xgb(bundle, "exp1_xgboost"),
        exp5=_load_xgb(bundle, "exp5_xgboost"),
        classes=classes, pair_ops=pair_ops, feature_layout=feature_layout,
        n_classes=int(classes.get("n_classes", 5)),
    )


def _nan_probs(n: int) -> tuple[float, list]:
    return float("nan"), [float("nan")] * n


def predict_rows(df: pd.DataFrame, bundle_ctx: dict, drugs_dict: dict) -> pd.DataFrame:
    n_cls = bundle_ctx["n_classes"]
    idmap = bundle_ctx["drug_id_to_idx"]
    fp_cache = bundle_ctx["fp_cache"]
    model = bundle_ctx["model"]
    h_all = bundle_ctx["h_all"]
    drug_emb = bundle_ctx["drug_emb"]
    device = bundle_ctx["device"]
    unknown = []
    rows = []

    for i, rec in df.iterrows():
        raw_a = str(rec["drug_a"]).strip()
        raw_b = str(rec["drug_b"]).strip()
        a, b = _canonical(raw_a, raw_b)
        in_a = a in idmap
        in_b = b in idmap
        in_index = bool(in_a and in_b)
        if not in_a:
            unknown.append(a)
        if not in_b:
            unknown.append(b)

        out = {
            "drug_a": raw_a, "drug_b": raw_b,
            "drug_a_canon": a, "drug_b_canon": b,
            "in_index": int(in_index),
        }

        e1_pred, e1_p = _nan_probs(n_cls)
        e3_pred, e3_p = _nan_probs(n_cls)
        e5_pred, e5_p = _nan_probs(n_cls)

        if in_index:
            ia, ib = int(idmap[a]), int(idmap[b])
            # Exp-3: PairMLP on encoded nodes
            with torch.no_grad():
                logits = model.pair_mlp(h_all[ia][None, :], h_all[ib][None, :])
                prob3 = torch.softmax(logits, dim=1).cpu().numpy()[0]
            e3_p = [float(x) for x in prob3]
            e3_pred = int(np.argmax(prob3))

            feat = build_pair_features(a, b, fp_cache, drugs_dict)
            if feat is not None:
                feat = np.asarray(feat, dtype=np.float32).reshape(1, -1)
                if bundle_ctx["exp1"] is not None:
                    p1 = bundle_ctx["exp1"].predict_proba(feat)[0]
                    e1_p = [float(x) for x in p1]
                    e1_pred = int(np.argmax(p1))
                if bundle_ctx["exp5"] is not None:
                    gnn_p = pair_ops_embed(drug_emb[ia], drug_emb[ib]).reshape(1, -1)
                    full = np.concatenate([feat, gnn_p], axis=1)
                    p5 = bundle_ctx["exp5"].predict_proba(full)[0]
                    e5_p = [float(x) for x in p5]
                    e5_pred = int(np.argmax(p5))

        out["exp1_pred"] = e1_pred
        out["exp3_pred"] = e3_pred
        out["exp5_pred"] = e5_pred
        for k, probs in (("exp1", e1_p), ("exp3", e3_p), ("exp5", e5_p)):
            for c, pv in enumerate(probs):
                out[f"{k}_p{c}"] = pv
        rows.append(out)

    unknown_unique = sorted(set(unknown))
    print(f"[predict_pairs] 行数={len(df)}  未知 ID 出现次数={len(unknown)}  "
          f"唯一未知 ID={len(unknown_unique)}")
    if unknown_unique:
        show = unknown_unique[:20]
        print(f"  未知 ID 示例: {show}{' ...' if len(unknown_unique) > 20 else ''}")
    return pd.DataFrame(rows)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="DDI 任意药物对推理（Exp-1/3/5）")
    p.add_argument("--bundle", type=str, required=True,
                   help="final_full 目录")
    p.add_argument("--data_dir", type=str, required=True,
                   help="processed 数据目录（用于定位 raw/drugbank_drugs.csv）")
    p.add_argument("--input", type=str, required=True,
                   help="查询 CSV，列 drug_a,drug_b")
    p.add_argument("--out", type=str, default="pair_predictions.csv")
    p.add_argument("--drugs_csv", type=str, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    bundle = Path(args.bundle)
    data_dir = Path(args.data_dir)
    if not bundle.exists():
        raise SystemExit(f"bundle 不存在: {bundle}")
    inp = Path(args.input)
    df = pd.read_csv(inp)
    need = {"drug_a", "drug_b"}
    if not need.issubset(df.columns):
        raise SystemExit(f"输入 CSV 必须含列 {need}，实际 {list(df.columns)}")

    drugs_csv = _resolve_drugs_csv(data_dir, args.drugs_csv)
    drugs_df = pd.read_csv(drugs_csv)
    drugs_dict = {row["drugbank_id"]: row.to_dict() for _, row in drugs_df.iterrows()}
    print(f"[predict_pairs] bundle={bundle}")
    print(f"[predict_pairs] drugs_csv={drugs_csv}  n_drugs_table={len(drugs_dict):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = load_bundle(bundle, device)
    out_df = predict_rows(df, ctx, drugs_dict)
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_p, index=False, encoding="utf-8-sig")
    n_ok = int(out_df["in_index"].sum())
    print(f"[predict_pairs] 可推理 {n_ok}/{len(out_df)} 对 → {out_p}")


if __name__ == "__main__":
    main()
