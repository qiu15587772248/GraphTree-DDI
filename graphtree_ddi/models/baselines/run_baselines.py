"""代表性基线主入口。

只依赖 models/baselines/ 内文件；划分协议复现 run_pipeline_final.py。
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
import sys
import time
import traceback
from pathlib import Path


import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.optim import AdamW  # noqa: E402

from graphtree_ddi.models.baselines import common as C  # noqa: E402
from graphtree_ddi.models.baselines.models import build_model, n_parameters  # noqa: E402

METHODS_ALL = ["MLP", "DDIMDL", "DeepDDI_SSP", "DistMult", "ComplEx"]
NEED_X = {"MLP", "DDIMDL"}
NEED_FP = {"DeepDDI_SSP", "DistMult", "ComplEx"}
NEED_SSP = {"DeepDDI_SSP"}
KGE_METHODS = {"DistMult", "ComplEx"}


def parse_methods(raw: list[str] | None) -> list[str]:
    aliases = {
        "DeepDDI": "DeepDDI_SSP",
        "SSP": "DeepDDI_SSP",
        "deepddi": "DeepDDI_SSP",
        "deepddi_ssp": "DeepDDI_SSP",
    }
    if not raw:
        return list(METHODS_ALL)
    out: list[str] = []
    for item in raw:
        for tok in str(item).replace(",", " ").split():
            name = aliases.get(tok, tok)
            if name not in METHODS_ALL:
                raise SystemExit(f"未知 --methods {tok}，可选 {METHODS_ALL}")
            if name not in out:
                out.append(name)
    return out


def pick_device(spec: str) -> torch.device:
    spec = (spec or "auto").lower()
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] 请求 CUDA 但不可用，改用 CPU")
        return torch.device("cpu")
    return torch.device(spec)


def resolve_kge_entity(spec: str, split_mode: str) -> str:
    spec = (spec or "auto").lower()
    if spec in ("lookup", "proj"):
        return spec
    return "lookup" if split_mode == "pair" else "proj"


def _to_device(arr: np.ndarray, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(arr))
    if dtype is not None:
        t = t.to(dtype)
    if device.type == "cuda":
        t = t.pin_memory().to(device, non_blocking=True)
    else:
        t = t.to(device)
    return t


def _clear_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


class FoldFeatures:
    """每个 fold 可能变化的 SSP-PCA（参照药物=训练集）。"""

    def __init__(self):
        self.drug_pca: np.ndarray | None = None
        self.pca_dim: int = 50
        self.train_key: tuple | None = None


def logits_from_batch(
    model: torch.nn.Module,
    method: str,
    rows: np.ndarray,
    *,
    data: C.LoadedData,
    pair_ent: np.ndarray | None,
    fp_matrix: np.ndarray | None,
    feat: FoldFeatures,
    device: torch.device,
    kge_entity: str = "proj",
) -> torch.Tensor:
    if method in NEED_X:
        return model(C.gather_x(data, rows, device))
    if method == "DeepDDI_SSP":
        assert feat.drug_pca is not None and pair_ent is not None
        ea = feat.drug_pca[pair_ent[rows, 0]]
        eb = feat.drug_pca[pair_ent[rows, 1]]
        x = np.concatenate([ea, eb], axis=1).astype(np.float32, copy=False)
        return model(_to_device(x, device))
    if method in KGE_METHODS:
        assert pair_ent is not None
        if kge_entity == "lookup":
            a = pair_ent[rows, 0]
            b = pair_ent[rows, 1]
            return model(
                _to_device(a, device, dtype=torch.long),
                _to_device(b, device, dtype=torch.long),
            )
        assert fp_matrix is not None
        xa = fp_matrix[pair_ent[rows, 0]]
        xb = fp_matrix[pair_ent[rows, 1]]
        return model(_to_device(xa, device), _to_device(xb, device))
    raise ValueError(method)


def _batcher(indices, batch_size, shuffle, seed, drop_last, data: C.LoadedData):
    return C.IndexBatcher(
        indices, batch_size, shuffle=shuffle, seed=seed, drop_last=drop_last,
        sample_shuffle=C.ram_gather_ready(data) and shuffle,
    )


@torch.no_grad()
def infer_split(
    model: torch.nn.Module,
    method: str,
    indices: np.ndarray,
    *,
    data: C.LoadedData,
    pair_ent: np.ndarray | None,
    fp_matrix: np.ndarray | None,
    feat: FoldFeatures,
    device: torch.device,
    batch_size: int,
    kge_entity: str = "proj",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    batcher = _batcher(indices, batch_size, shuffle=False, seed=0, drop_last=False, data=data)
    ys, preds, probs, idxs = [], [], [], []
    for rows in batcher:
        logits = logits_from_batch(
            model, method, rows, data=data, pair_ent=pair_ent,
            fp_matrix=fp_matrix, feat=feat, device=device, kge_entity=kge_entity,
        )
        prob = F.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32)
        pred = logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
        ys.append(data.y[rows])
        preds.append(pred)
        probs.append(prob)
        idxs.append(rows)
    y_true = np.concatenate(ys) if ys else np.zeros((0,), dtype=np.int64)
    y_pred = np.concatenate(preds) if preds else np.zeros((0,), dtype=np.int64)
    y_prob = np.concatenate(probs) if probs else np.zeros((0, C.N_CLASSES), dtype=np.float32)
    test_idx = np.concatenate(idxs) if idxs else np.zeros((0,), dtype=np.int64)
    return test_idx, y_true, y_pred, y_prob


def train_one(
    method: str,
    model: torch.nn.Module,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    data: C.LoadedData,
    pair_ent: np.ndarray | None,
    fp_matrix: np.ndarray | None,
    feat: FoldFeatures,
    device: torch.device,
    epochs: int,
    batch_size: int,
    patience: int,
    lr: float,
    seed: int,
    min_epochs: int = 1,
    kge_entity: str = "proj",
) -> tuple[torch.nn.Module, int, float, list[dict]]:
    model.to(device)
    y_tr = data.y[train_idx]
    weights = torch.tensor(C.balanced_class_weights(y_tr), dtype=torch.float32, device=device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    drop_last = len(train_idx) > batch_size
    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    bad = 0
    epoch_stats: list[dict] = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        batcher = _batcher(
            train_idx, batch_size, shuffle=True, seed=seed + epoch,
            drop_last=drop_last, data=data,
        )
        total_loss, n_seen = 0.0, 0
        _sync(device)
        t_tr = time.time()
        for rows in batcher:
            logits = logits_from_batch(
                model, method, rows, data=data, pair_ent=pair_ent,
                fp_matrix=fp_matrix, feat=feat, device=device, kge_entity=kge_entity,
            )
            target = _to_device(data.y[rows], device, dtype=torch.long)
            loss = F.cross_entropy(logits, target, weight=weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = int(rows.shape[0])
            total_loss += float(loss.detach().cpu()) * bs
            n_seen += bs
        _sync(device)
        train_sec = time.time() - t_tr
        t_va = time.time()
        _, y_true, y_pred, y_prob = infer_split(
            model, method, val_idx, data=data, pair_ent=pair_ent,
            fp_matrix=fp_matrix, feat=feat, device=device, batch_size=batch_size,
            kge_entity=kge_entity,
        )
        _sync(device)
        val_sec = time.time() - t_va
        val_metrics = C.evaluate_predictions(y_true, y_pred, y_prob)
        val_f1 = val_metrics["macro_f1"]
        if not np.isfinite(val_f1):
            val_f1 = -1.0
        mean_loss = total_loss / max(n_seen, 1)
        sps = n_seen / train_sec if train_sec > 0 else 0.0
        print(
            f"    epoch {epoch:03d}/{epochs}  loss={mean_loss:.4f}  "
            f"val_macro_f1={val_f1:.4f}  train={train_sec:.3f}s n={n_seen} "
            f"({sps:.0f} samp/s)  val={val_sec:.3f}s"
        )
        epoch_stats.append(dict(
            epoch=epoch, n_seen=int(n_seen), train_sec=float(train_sec),
            val_sec=float(val_sec), samples_per_sec=float(sps),
        ))
        improved = val_f1 > best_f1 + 1e-6
        if best_state is None or improved:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if epoch >= min_epochs and bad >= patience:
                print(f"    early stop @ epoch {epoch} (best={best_epoch}, val_macro_f1={best_f1:.4f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    elapsed = time.time() - t0
    return model, best_epoch, elapsed, epoch_stats


def maybe_prepare_ssp(
    method: str,
    train_idx: np.ndarray,
    feat: FoldFeatures,
    fp_matrix: np.ndarray | None,
    pair_ent: np.ndarray | None,
) -> None:
    if method != "DeepDDI_SSP":
        return
    key = (int(train_idx[0]) if len(train_idx) else -1, int(len(train_idx)), int(train_idx.sum()))
    if feat.train_key == key and feat.drug_pca is not None:
        return
    assert fp_matrix is not None and pair_ent is not None
    drug_pca, k = C.fit_ssp_pca(fp_matrix, pair_ent, train_idx, n_components=50, pca_random_state=0)
    feat.drug_pca = drug_pca
    feat.pca_dim = k
    feat.train_key = key


def evaluate_and_record(
    *,
    state: dict,
    state_path: Path,
    csv_path: Path,
    out_dir: Path,
    method: str,
    fold: int,
    seed: int,
    split_mode: str,
    test_set: str,
    model: torch.nn.Module,
    test_idx: np.ndarray,
    data: C.LoadedData,
    pair_ent: np.ndarray | None,
    fp_matrix: np.ndarray | None,
    feat: FoldFeatures,
    device: torch.device,
    batch_size: int,
    elapsed_sec: float,
    best_epoch: int,
    extra: dict | None = None,
    kge_entity: str = "proj",
) -> dict:
    tidx, y_true, y_pred, y_prob = infer_split(
        model, method, test_idx, data=data, pair_ent=pair_ent,
        fp_matrix=fp_matrix, feat=feat, device=device, batch_size=batch_size,
        kge_entity=kge_entity,
    )
    metrics = C.evaluate_predictions(y_true, y_pred, y_prob)
    C.save_predictions(
        out_dir, method, fold, seed, tidx, y_true, y_pred, y_prob,
        test_set=test_set if test_set in ("S1", "S2") else None,
    )
    rec = dict(
        method=method, fold=int(fold), seed=int(seed),
        split_mode=split_mode, test_set=test_set,
        elapsed_sec=round(float(elapsed_sec), 3),
        best_epoch=int(best_epoch),
        n_test=int(len(tidx)),
        **metrics,
    )
    if extra:
        rec.update(extra)
    rec = C.json_safe(rec)
    key = C.run_key(method, fold, seed, split_mode, test_set)
    state["runs"][key] = rec
    C.save_state(state_path, state)
    C.save_raw_csv(state, csv_path)
    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) and np.isfinite(v) else "nan"
    f1s = rec.get("per_class_f1") or []
    f1_txt = ",".join(_fmt(x) if x is not None else "nan" for x in f1s)
    print(
        f"  [{key}] acc={_fmt(rec.get('accuracy'))} macroF1={_fmt(rec.get('macro_f1'))} "
        f"wF1={_fmt(rec.get('weighted_f1'))} AUROC={_fmt(rec.get('macro_auroc'))} "
        f"AP={_fmt(rec.get('macro_ap'))} adj={_fmt(rec.get('adjacent_acc'))} "
        f"HRsens={_fmt(rec.get('high_risk_sens'))} HRspec={_fmt(rec.get('high_risk_spec'))} "
        f"under={_fmt(rec.get('severe_underestimation_rate'))} "
        f"qwk={_fmt(rec.get('qwk'))} mae={_fmt(rec.get('mae_ordinal'))} "
        f"f1=[{f1_txt}]  {elapsed_sec:.1f}s best_ep={best_epoch}"
    )
    return rec


def run_method_on_split(
    method: str,
    fold: int,
    seed: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    eval_sets: list[tuple[str, np.ndarray]],
    *,
    split_mode: str,
    data: C.LoadedData,
    pair_ent: np.ndarray | None,
    fp_matrix: np.ndarray | None,
    feat: FoldFeatures,
    device: torch.device,
    args,
    state: dict,
    state_path: Path,
    csv_path: Path,
    kge_entity: str,
    n_drugs: int | None,
) -> None:
    pending = []
    for test_set, _idx in eval_sets:
        key = C.run_key(method, fold, seed, split_mode, test_set)
        if C.is_run_done(state, key):
            print(f"  [跳过-已完成] {key}")
        else:
            pending.append(test_set)
    if not pending:
        return

    C.set_seed(seed)
    maybe_prepare_ssp(method, train_idx, feat, fp_matrix, pair_ent)
    kwargs = dict(dropout=args.dropout, in_dim=data.feat_dim, fp_dim=2048, emb_dim=200)
    if method == "DeepDDI_SSP":
        kwargs["in_dim"] = int(feat.pca_dim * 2)
    if method in KGE_METHODS:
        kwargs["entity_mode"] = kge_entity
        kwargs["n_drugs"] = n_drugs
    model = build_model(method, **kwargs)
    n_par = n_parameters(model)
    print(
        f"\n  {method} | mode={split_mode} fold={fold} seed={seed} | "
        f"kge={kge_entity if method in KGE_METHODS else '-'} "
        f"preload={data.preload_actual} | "
        f"params={n_par:,} train={len(train_idx):,} val={len(val_idx):,}"
    )
    try:
        model, best_epoch, elapsed, epoch_stats = train_one(
            method, model, train_idx, val_idx,
            data=data, pair_ent=pair_ent, fp_matrix=fp_matrix, feat=feat,
            device=device, epochs=args.epochs, batch_size=args.batch_size,
            patience=args.patience, lr=args.lr, seed=seed,
            min_epochs=1 if args.smoke else 3,
            kge_entity=kge_entity,
        )
        tr_secs = [e["train_sec"] for e in epoch_stats] or [0.0]
        n_seens = [e["n_seen"] for e in epoch_stats] or [0]
        extra = {
            "n_params": n_par,
            "kge_entity": kge_entity if method in KGE_METHODS else None,
            "preload": data.preload_actual,
            "preload_requested": data.preload_requested,
            "epoch_stats": epoch_stats,
            "epoch_train_sec_mean": float(np.mean(tr_secs)),
            "epoch_n_seen_mean": float(np.mean(n_seens)),
            "n_train": int(len(train_idx)),
        }
        for test_set, tidx in eval_sets:
            if test_set not in pending and C.is_run_done(
                state, C.run_key(method, fold, seed, split_mode, test_set)
            ):
                continue
            if len(tidx) == 0:
                print(f"  [跳过-空集] {method} {test_set}")
                rec = C.json_safe(dict(
                    method=method, fold=fold, seed=seed, split_mode=split_mode,
                    test_set=test_set, elapsed_sec=round(elapsed, 3),
                    best_epoch=best_epoch, error="empty test set",
                ))
                state["runs"][C.run_key(method, fold, seed, split_mode, test_set)] = rec
                C.save_state(state_path, state)
                continue
            evaluate_and_record(
                state=state, state_path=state_path, csv_path=csv_path,
                out_dir=Path(args.out_dir), method=method, fold=fold, seed=seed,
                split_mode=split_mode, test_set=test_set, model=model,
                test_idx=tidx, data=data, pair_ent=pair_ent, fp_matrix=fp_matrix,
                feat=feat, device=device, batch_size=args.batch_size,
                elapsed_sec=elapsed, best_epoch=best_epoch, extra=extra,
                kge_entity=kge_entity,
            )
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] {method} f{fold} s{seed} {split_mode}: {e}")
        traceback.print_exc()
        for test_set, _ in eval_sets:
            state["runs"][C.run_key(method, fold, seed, split_mode, test_set)] = dict(
                method=method, fold=fold, seed=seed, split_mode=split_mode,
                test_set=test_set, error=str(e),
            )
        C.save_state(state_path, state)
    finally:
        del model
        _clear_device(device)


def print_split_summary(y: np.ndarray, idx: np.ndarray, name: str) -> None:
    yy = y[idx]
    bc = np.bincount(yy, minlength=C.N_CLASSES)
    print(f"  {name}: n={len(idx):,}  class={bc.tolist()}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="DDI 代表性基线（MLP/DDIMDL/DeepDDI_SSP/DistMult/ComplEx）")
    parser.add_argument("--data_dir", type=str, default=str(_PROJ / "data" / "processed"))
    parser.add_argument("--out_dir", type=str, default=str(_PROJ / "models" / "results" / "baselines"))
    parser.add_argument("--drugs_csv", type=str, default=str(_PROJ / "data" / "raw" / "drugbank_drugs.csv"))
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--folds", type=int, nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--split_mode", nargs="+", default=["pair"], choices=["pair", "drug"])
    parser.add_argument("--drug_split_seed", type=int, nargs="*", default=None)
    parser.add_argument("--preload", type=str, default="cpu", choices=["memmap", "cpu", "gpu"])
    parser.add_argument("--kge_entity", type=str, default="auto",
                        choices=["auto", "lookup", "proj"],
                        help="auto: pair→lookup, drug→proj")
    parser.add_argument("--subset_frac", type=float, default=None,
                        help="分层抽取该比例作为工作集（--smoke 默认 0.01）")
    args = parser.parse_args(argv)

    if args.epochs is None:
        args.epochs = 2 if args.smoke else 50
    if args.patience is None:
        args.patience = 2 if args.smoke else 10
    if args.folds is None:
        args.folds = [1] if (args.smoke or args.subset_frac) else [1, 2, 3, 4, 5]
    if args.seeds is None:
        args.seeds = [42] if (args.smoke or args.subset_frac) else list(C.SEEDS_DEFAULT)
    if args.drug_split_seed is None:
        args.drug_split_seed = [42]
    args.methods = parse_methods(args.methods)
    args.out_dir = str(Path(args.out_dir))
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    print(f"[env] device={device}")
    if device.type == "cuda":
        print(f"  GPU={torch.cuda.get_device_name(0)}  "
              f"mem={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    data = C.load_processed(Path(args.data_dir), drugs_csv=Path(args.drugs_csv))
    work_idx = np.arange(data.n_samples, dtype=np.int64)
    frac = args.subset_frac
    if args.smoke and frac is None:
        frac = C.SMOKE_FRACTION
    if frac is not None and frac < 1.0:
        work_idx = C.smoke_subset_indices(data.y, float(frac), seed=42)
        print(
            f"[subset] 分层 {float(frac):.1%} → {len(work_idx):,}/{data.n_samples:,} 对  "
            f"class={np.bincount(data.y[work_idx], minlength=5).tolist()}  "
            f"epochs={args.epochs} patience={args.patience} smoke={bool(args.smoke)}"
        )

    data = C.apply_preload(data, args.preload, device, work_idx=work_idx)

    # 即使用冒烟子集训练，也先在全量 y 上核对 pipeline 划分协议（只动索引，不读 X）
    full_pair = C.pair_splits_pipeline_final(data.y, n_folds=5, random_state=C.SPLIT_RANDOM_STATE)
    C.assert_pair_splits_match_pipeline(data.y, full_pair)

    fp_matrix = None
    pair_ent = None
    n_drugs = None
    kge_modes_needed = {resolve_kge_entity(args.kge_entity, sm) for sm in args.split_mode}
    need_ent = any(m in (NEED_SSP | KGE_METHODS) for m in args.methods)
    need_fp = any(m in NEED_SSP for m in args.methods) or (
        any(m in KGE_METHODS for m in args.methods) and ("proj" in kge_modes_needed)
    )
    if need_ent:
        drugs_sorted = sorted(set(data.drug1[work_idx]) | set(data.drug2[work_idx]))
        n_drugs = len(drugs_sorted)
        id_to_idx = {d: i for i, d in enumerate(drugs_sorted)}
        if need_fp:
            fp_matrix, id_to_idx = C.load_morgan_fingerprints(drugs_sorted, Path(args.drugs_csv))
        pair_ent = np.zeros((data.n_samples, 2), dtype=np.int32)
        pair_ent[work_idx] = C.pair_entity_indices(
            data.drug1[work_idx], data.drug2[work_idx], id_to_idx,
        )
        print(f"[ent] n_drugs={n_drugs}  fp={'yes' if fp_matrix is not None else 'no'}  "
              f"kge_modes={sorted(kge_modes_needed)}")

    state_path = Path(args.out_dir) / "baselines_state.json"
    csv_path = Path(args.out_dir) / "baselines_raw_metrics.csv"
    state = C.ensure_mode(C.load_state(state_path), args.smoke, state_path)
    state["preload_requested"] = data.preload_requested
    state["preload_actual"] = data.preload_actual
    state["ram_avail_gb"] = data.ram_avail_gb
    state["ram_total_gb"] = data.ram_total_gb
    state["kge_entity_cli"] = args.kge_entity
    C.save_state(state_path, state)

    t_all = time.time()
    for split_mode in args.split_mode:
        print(f"\n{'=' * 72}\n  split_mode={split_mode}\n{'=' * 72}")
        if split_mode == "pair":
            y_work = data.y[work_idx]
            # 必须始终 5-fold（与 pipeline 一致），再只跑 --folds 指定折
            local = C.pair_splits_pipeline_final(
                y_work, n_folds=5, random_state=C.SPLIT_RANDOM_STATE,
            )
            # 子集上同样用独立重算核对
            C.assert_pair_splits_match_pipeline(y_work, local)
            splits = C.remap_splits_to_original(local, work_idx)
            print_split_summary(data.y, splits.idx_test, "pair/test")
            fold_map = {f: (tr, va) for f, tr, va in splits.cv_splits}
            for fold in args.folds:
                if fold not in fold_map:
                    raise SystemExit(f"--folds {fold} 超出当前 CV（1..{len(splits.cv_splits)}）")
                idx_train, idx_val = fold_map[fold]
                print_split_summary(data.y, idx_train, f"pair/fold{fold}/train")
                print_split_summary(data.y, idx_val, f"pair/fold{fold}/val")
                feat = FoldFeatures()
                for seed in args.seeds:
                    kge_ent = resolve_kge_entity(args.kge_entity, "pair")
                    for method in args.methods:
                        run_method_on_split(
                            method, fold, seed, idx_train, idx_val,
                            [("test", splits.idx_test)],
                            split_mode="pair", data=data, pair_ent=pair_ent,
                            fp_matrix=fp_matrix, feat=feat, device=device,
                            args=args, state=state, state_path=state_path,
                            csv_path=csv_path, kge_entity=kge_ent, n_drugs=n_drugs,
                        )
        else:
            # drug 模式：模型种子 = 划分种子（与流水线执行者一致）
            if args.seeds != [42] and args.seeds != list(C.SEEDS_DEFAULT) and not args.smoke:
                print("[note] drug 模式忽略 --seeds，改用 --drug_split_seed 作为模型种子")
            for dseed in args.drug_split_seed:
                dsplit = C.build_drug_splits(data.pairs, data.y, work_idx, dseed)
                print_split_summary(data.y, dsplit.kk_train, "drug/KK-train")
                print_split_summary(data.y, dsplit.kk_val, "drug/KK-val")
                print_split_summary(data.y, dsplit.s1, "drug/S1")
                print_split_summary(data.y, dsplit.s2, "drug/S2")
                feat = FoldFeatures()
                kge_ent = resolve_kge_entity(args.kge_entity, "drug")
                if kge_ent == "lookup":
                    print("[warn] drug 模式使用 kge_entity=lookup：U 药嵌入未经训练，冷启动会退化")
                for method in args.methods:
                    run_method_on_split(
                        method, fold=1, seed=dseed,
                        train_idx=dsplit.kk_train, val_idx=dsplit.kk_val,
                        eval_sets=[("S1", dsplit.s1), ("S2", dsplit.s2)],
                        split_mode="drug", data=data, pair_ent=pair_ent,
                        fp_matrix=fp_matrix, feat=feat, device=device,
                        args=args, state=state, state_path=state_path,
                        csv_path=csv_path, kge_entity=kge_ent, n_drugs=n_drugs,
                    )

    C.save_raw_csv(state, csv_path)
    dt = time.time() - t_all
    n_ok = sum(1 for r in state["runs"].values() if "macro_f1" in r)
    n_err = sum(1 for r in state["runs"].values() if "error" in r)
    print(f"\n{'=' * 72}")
    print(f"完成  ok={n_ok}  error={n_err}  墙钟={dt:.1f}s")
    print(f"preload requested={data.preload_requested} actual={data.preload_actual}")
    print(f"state: {state_path}")
    print(f"csv:   {csv_path}")
    print(f"pred:  {Path(args.out_dir) / 'predictions'}")
    # 吞吐摘要：每个 method 的 mean train_sec / n_seen
    seen = {}
    for r in state.get("runs", {}).values():
        if "epoch_stats" not in r or "macro_f1" not in r:
            continue
        m = r["method"]
        if m in seen:
            continue
        seen[m] = r
    if seen:
        print("吞吐（首次出现的该 method 的 epoch_stats）:")
        for m, r in seen.items():
            for es in r.get("epoch_stats") or []:
                print(
                    f"  {m}  epoch={es.get('epoch')}  n_seen={es.get('n_seen')}  "
                    f"train_sec={es.get('train_sec'):.3f}  "
                    f"samp/s={es.get('samples_per_sec'):.1f}  val_sec={es.get('val_sec'):.3f}"
                )
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
