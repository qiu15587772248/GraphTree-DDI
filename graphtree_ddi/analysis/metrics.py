"""指标定义（与 models/baselines/common.py 一致）及预测兼容加载器。

load_predictions(dir) 把基线 npz 与主流水线 npz 统一成 DataFrame。
每一行是一次 run（一份 npz），列：
  method, fold, seed, split, test_set, y_true, y_pred, y_prob
另附 test_idx / n / path 便于对齐与溯源。
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

import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


from graphtree_ddi.models.baselines import common as _C  # noqa: E402

N_CLASSES = int(_C.N_CLASSES)
SCALAR_METRIC_KEYS = tuple(_C.ALL_SCALAR_METRIC_KEYS)
ALL_SCALAR_METRIC_KEYS = SCALAR_METRIC_KEYS
PIPELINE_METRIC_KEYS = tuple(_C.PIPELINE_METRIC_KEYS)
EXTRA_METRIC_KEYS = tuple(_C.EXTRA_METRIC_KEYS)
CLASS_LABELS = list(range(N_CLASSES))

# 与 common.evaluate_predictions 逐项一致；bootstrap 只抽这些标量
BOOTSTRAP_METRICS = (
    "macro_f1",
    "accuracy",
    "qwk",
    "high_risk_sens",
    "mae_ordinal",
)

HIGHER_IS_BETTER = {
    "accuracy": True,
    "macro_f1": True,
    "weighted_f1": True,
    "macro_auroc": True,
    "macro_ap": True,
    "adjacent_acc": True,
    "high_risk_sens": True,
    "high_risk_spec": True,
    "qwk": True,
    "severe_underestimation_rate": False,
    "mae_ordinal": False,
}

CLASS_NAMES_EN = (
    "0 Safe",
    "1 General",
    "2 Moderate",
    "3 Serious",
    "4 Critical",
)

# 论文显示名；可用 --method_names JSON 覆盖
DEFAULT_METHOD_LABELS: dict[str, str] = {
    "Exp-1": "XGBoost (handcrafted)",
    "Exp-2": "R-GCN (DDI-only)",
    "Exp-3": "R-GCN (hybrid KG)",
    "Exp-4": "R-GAT (hybrid KG)",
    "Exp-5": "GraphTree-DDI (ours)",
    "MLP": "MLP",
    "DDIMDL": "DDIMDL-style",
    "DeepDDI_SSP": "DeepDDI-style (SSP)",
    "DistMult": "DistMult",
    "ComplEx": "ComplEx",
}

# 基线 5 → Exp-1…Exp-5
METHOD_ORDER = [
    "MLP", "DDIMDL", "DeepDDI_SSP", "DistMult", "ComplEx",
    "Exp-1", "Exp-2", "Exp-3", "Exp-4", "Exp-5",
]

_NPZ_NAME = re.compile(
    r"^(?P<method>.+)_f(?P<fold>\d+)_s(?P<seed>\d+)"
    r"(?:_(?P<tag>S1|S2|s1|s2|test|holdout|KK|kk))?$",
)

_STATE_NAMES = (
    "baselines_state.json",
    "final_state.json",
    "state.json",
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_classes: int = N_CLASSES,
) -> dict:
    """直接复用 baselines.common.evaluate_predictions。"""
    return _C.evaluate_predictions(y_true, y_pred, y_prob, n_classes=n_classes)


def scalar_from_metrics(metrics: dict, key: str) -> float:
    v = metrics.get(key, float("nan"))
    if v is None:
        return float("nan")
    return float(v)


def metric_value(
    key: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    n_classes: int = N_CLASSES,
) -> float:
    """单指标。bootstrap 五件套可不传 y_prob。"""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    key = str(key)
    if key in ("mae", "MAE"):
        key = "mae_ordinal"
    if key in ("acc",):
        key = "accuracy"
    if key in ("f1", "macro-f1", "macro_F1"):
        key = "macro_f1"
    if key in ("QWK", "kappa"):
        key = "qwk"
    if key == "accuracy":
        return float(np.mean(y_true == y_pred)) if len(y_true) else float("nan")
    if key == "mae_ordinal":
        if len(y_true) == 0:
            return float("nan")
        return float(np.mean(np.abs(y_true.astype(np.float64) - y_pred.astype(np.float64))))
    if key == "high_risk_sens":
        high = y_true >= 3
        if not np.any(high):
            return float("nan")
        return float(np.mean(y_pred[high] >= 3))
    if key in ("macro_f1", "weighted_f1", "qwk", "adjacent_acc",
               "high_risk_spec", "severe_underestimation_rate",
               "macro_auroc", "macro_ap"):
        if y_prob is None:
            n = len(y_true)
            y_prob = np.zeros((n, n_classes), dtype=np.float64)
            if n:
                y_prob[np.arange(n), np.clip(y_pred, 0, n_classes - 1)] = 1.0
        return scalar_from_metrics(compute_metrics(y_true, y_pred, y_prob, n_classes), key)
    if y_prob is None:
        raise ValueError(f"指标 {key} 需要 y_prob")
    return scalar_from_metrics(compute_metrics(y_true, y_pred, y_prob, n_classes), key)


def _as_prob(y_prob: np.ndarray, n: int, n_classes: int = N_CLASSES) -> np.ndarray:
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_prob.ndim == 1:
        y_prob = y_prob.reshape(-1, n_classes)
    if y_prob.ndim != 2:
        raise ValueError(f"y_prob 维数错误: {y_prob.shape}")
    if y_prob.shape[0] != n:
        raise ValueError(f"y_prob 行数 {y_prob.shape[0]} != n={n}")
    if y_prob.shape[1] != n_classes:
        pad = np.zeros((n, n_classes), dtype=np.float64)
        k = min(n_classes, y_prob.shape[1])
        pad[:, :k] = y_prob[:, :k]
        y_prob = pad
    return y_prob


def read_npz(path: Path) -> dict[str, np.ndarray]:
    """读取一份预测 npz，兼容 y_true / y_test 旧键。"""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        files = set(z.files)
        if "y_true" in files:
            y_true = np.asarray(z["y_true"])
        elif "y_test" in files:
            y_true = np.asarray(z["y_test"])
        else:
            raise KeyError(f"{path.name} 缺少 y_true/y_test")
        if "y_pred" not in files:
            raise KeyError(f"{path.name} 缺少 y_pred")
        y_pred = np.asarray(z["y_pred"])
        if "y_prob" in files:
            y_prob = np.asarray(z["y_prob"])
        else:
            y_prob = np.zeros((len(y_true), N_CLASSES), dtype=np.float32)
            if len(y_true):
                y_prob[np.arange(len(y_true)), np.clip(y_pred.astype(int), 0, N_CLASSES - 1)] = 1.0
        if "test_idx" in files:
            test_idx = np.asarray(z["test_idx"], dtype=np.int64)
        else:
            test_idx = np.arange(len(y_true), dtype=np.int64)
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if len(y_true) != len(y_pred):
        raise ValueError(f"{path.name} y_true/y_pred 长度不一致")
    y_prob = _as_prob(y_prob, len(y_true))
    if len(test_idx) != len(y_true):
        test_idx = np.arange(len(y_true), dtype=np.int64)
    return dict(test_idx=test_idx, y_true=y_true, y_pred=y_pred, y_prob=y_prob.astype(np.float32))


def parse_npz_stem(stem: str) -> dict[str, Any] | None:
    m = _NPZ_NAME.match(stem)
    if not m:
        return None
    tag = m.group("tag")
    tag_n = tag.upper() if tag else None
    if tag_n in ("TEST", "HOLDOUT"):
        test_set = "test"
        split = "pair"
    elif tag_n in ("S1", "S2"):
        test_set = tag_n
        split = "drug"
    elif tag_n == "KK":
        test_set = "KK"
        split = "drug"
    else:
        test_set = "test"
        split = "pair"
    return dict(
        method=m.group("method"),
        fold=int(m.group("fold")),
        seed=int(m.group("seed")),
        split=split,
        test_set=test_set,
        tag=tag_n,
    )


def resolve_pred_dir(pred_dir: str | Path) -> Path:
    """接受 results 根目录或其中的 predictions/。"""
    d = Path(pred_dir)
    if not d.exists():
        raise FileNotFoundError(d)
    if d.is_file() and d.suffix == ".npz":
        return d.parent
    sub = d / "predictions"
    if sub.is_dir() and any(sub.glob("*.npz")):
        return sub
    return d


def find_state_paths(pred_dir: str | Path) -> list[Path]:
    d = Path(pred_dir)
    npz_dir = resolve_pred_dir(d) if d.exists() else d
    roots = []
    for cand in (d, npz_dir, npz_dir.parent, d.parent):
        if cand not in roots:
            roots.append(cand)
    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for name in _STATE_NAMES:
            p = root / name
            if p.exists() and p not in seen:
                out.append(p)
                seen.add(p)
        for p in root.glob("*_state.json"):
            if p not in seen:
                out.append(p)
                seen.add(p)
    return out


def load_state_dicts(pred_dir: str | Path) -> list[dict]:
    states = []
    for p in find_state_paths(pred_dir):
        try:
            with open(p, encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                obj["_state_path"] = str(p)
                states.append(obj)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 无法读 state {p}: {e}")
    return states


def _lookup_run_meta(states: list[dict], method: str, fold: int, seed: int, test_set: str) -> dict:
    """从 baselines / pipeline 两种 run key 里取 split_mode 等。"""
    candidates = [
        f"{method}_f{fold}_s{seed}_{test_set}",
        f"{method}_f{fold}_s{seed}",
        f"{method}_f{fold}_s{seed}_pair_{test_set}",
        f"{method}_f{fold}_s{seed}_drug_{test_set}",
        f"{method}_f{fold}_s{seed}_pair_test",
        f"{method}_f{fold}_s{seed}_drug_{test_set}",
    ]
    if test_set in ("test", "holdout"):
        candidates.extend([
            f"{method}_f{fold}_s{seed}_pair_test",
            f"{method}_f{fold}_s{seed}_pair_holdout",
        ])
    for st in states:
        runs = st.get("runs") or {}
        mode = str(st.get("mode") or "")
        for key in candidates:
            rec = runs.get(key)
            if not isinstance(rec, dict):
                continue
            rec = dict(rec)
            rec["_state_key"] = key
            rec["_state_mode"] = mode
            return rec
        for key, rec in runs.items():
            if not isinstance(rec, dict):
                continue
            if rec.get("method") != method:
                continue
            if int(rec.get("fold", -1)) != int(fold):
                continue
            if int(rec.get("seed", -1)) != int(seed):
                continue
            ts = rec.get("test_set") or "test"
            if str(ts) != str(test_set) and not (
                test_set == "test" and ts in ("test", "holdout")
            ):
                continue
            rec = dict(rec)
            rec["_state_key"] = key
            rec["_state_mode"] = mode
            return rec
    return {}


def _split_from_meta(parsed: dict, rec: dict) -> str:
    sm = rec.get("split_mode")
    if sm in ("pair", "drug"):
        return str(sm)
    mode = str(rec.get("_state_mode") or "")
    if ":drug" in mode:
        return "drug"
    if ":pair" in mode:
        return "pair"
    ts = parsed.get("test_set")
    if ts in ("S1", "S2", "KK"):
        return "drug"
    return str(parsed.get("split") or "pair")


def list_npz_files(pred_dir: str | Path) -> list[Path]:
    d = resolve_pred_dir(pred_dir)
    files = sorted(d.glob("*.npz"))
    return [p for p in files if parse_npz_stem(p.stem) is not None]


def load_prediction_runs(pred_dir: str | Path) -> list[dict]:
    """返回 list[dict]，每个元素是一次 run 的数组与元数据。"""
    pred_dir = Path(pred_dir)
    files = list_npz_files(pred_dir)
    states = load_state_dicts(pred_dir)
    rows: list[dict] = []
    for path in files:
        parsed = parse_npz_stem(path.stem)
        if parsed is None:
            continue
        try:
            arrays = read_npz(path)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 跳过 {path.name}: {e}")
            continue
        rec = _lookup_run_meta(
            states, parsed["method"], parsed["fold"], parsed["seed"], parsed["test_set"],
        )
        split = _split_from_meta(parsed, rec)
        test_set = parsed["test_set"]
        if rec.get("test_set") in ("S1", "S2", "test", "holdout", "KK"):
            if parsed.get("tag") in (None, "TEST", "HOLDOUT"):
                test_set = "test" if rec.get("test_set") in (None, "test", "holdout") else rec["test_set"]
        rows.append(dict(
            method=str(parsed["method"]),
            fold=int(parsed["fold"]),
            seed=int(parsed["seed"]),
            split=str(split),
            test_set=str(test_set),
            test_idx=arrays["test_idx"],
            y_true=arrays["y_true"],
            y_pred=arrays["y_pred"],
            y_prob=arrays["y_prob"],
            n=int(len(arrays["y_true"])),
            path=str(path),
        ))
    return rows


def as_pred_dirs(pred_dir: str | Path | Iterable[str | Path]) -> list[Path]:
    """把 --pred_dir 的一个或多个路径收成 list[Path]。"""
    if pred_dir is None:
        raise ValueError("pred_dir 为空")
    if isinstance(pred_dir, (str, Path)):
        items: list[str | Path] = [pred_dir]
    else:
        items = list(pred_dir)
    if not items:
        raise ValueError("pred_dir 为空")
    out: list[Path] = []
    for p in items:
        d = Path(p)
        if not d.exists():
            raise FileNotFoundError(d)
        out.append(d)
    return out


def prediction_key(method, fold, seed, test_set) -> tuple:
    ts = str(test_set)
    if ts.lower() in ("test", "holdout", "", "none"):
        ts = "test"
    return (str(method), int(fold), int(seed), ts)


def _merge_rows(rows: list[dict], *, kind: str) -> list[dict]:
    seen: dict[tuple, str] = {}
    out: list[dict] = []
    for r in rows:
        k = prediction_key(r["method"], r["fold"], r["seed"], r["test_set"])
        src = str(r.get("path") or r.get("source_key") or r.get("source_dir") or "")
        if k in seen:
            raise ValueError(
                f"重复预测 key {k}（{kind}）：\n  已有: {seen[k]}\n  冲突: {src}"
            )
        seen[k] = src
        out.append(r)
    return out


def load_predictions(pred_dir: str | Path | Iterable[str | Path]) -> pd.DataFrame:
    """兼容加载器：可传入多个结果目录，按 (method, fold, seed, test_set) 合并。

    重复 key 直接报错。y_* 为每个 run 的 numpy 数组。
    """
    dirs = as_pred_dirs(pred_dir)
    rows: list[dict] = []
    for d in dirs:
        part = load_prediction_runs(d)
        for r in part:
            r["source_dir"] = str(d)
        rows.extend(part)
    if not rows:
        print(f"[warn] {dirs} 下没有可解析的预测 npz")
        return pd.DataFrame(columns=[
            "method", "fold", "seed", "split", "test_set",
            "y_true", "y_pred", "y_prob", "test_idx", "n", "path",
        ])
    rows = _merge_rows(rows, kind="npz")
    df = pd.DataFrame(rows)
    df = df.sort_values(["split", "test_set", "method", "fold", "seed"]).reset_index(drop=True)
    print(f"[load] {len(df)} runs from {len(dirs)} dir(s); methods={methods_in(df)}")
    return df


def _runs_from_state_one(pred_dir: str | Path) -> list[dict]:
    rows = []
    for st in load_state_dicts(pred_dir):
        mode = str(st.get("mode") or "")
        src = str(st.get("_state_path") or pred_dir)
        for key, rec in (st.get("runs") or {}).items():
            if not isinstance(rec, dict) or "error" in rec:
                continue
            if "macro_f1" not in rec and "accuracy" not in rec:
                continue
            method = rec.get("method")
            if method is None:
                continue
            split = rec.get("split_mode")
            if split not in ("pair", "drug"):
                if ":drug" in mode:
                    split = "drug"
                else:
                    split = "pair"
            ts = rec.get("test_set") or "test"
            row = dict(
                method=str(method),
                fold=int(rec.get("fold", 0)),
                seed=int(rec.get("seed", 0)),
                split=str(split),
                test_set=str(ts),
                n=int(rec.get("n_eval") or rec.get("n_test") or 0),
                source_key=str(key),
                path=src,
                source_dir=str(pred_dir),
            )
            for k in SCALAR_METRIC_KEYS:
                row[k] = rec.get(k)
            pc = rec.get("per_class_f1") or [None] * N_CLASSES
            for i in range(N_CLASSES):
                row[f"f1_c{i}"] = pc[i] if i < len(pc) else None
            rows.append(row)
    return rows


def runs_from_state(pred_dir: str | Path | Iterable[str | Path]) -> pd.DataFrame:
    """仅从 *state.json 汇总标量（无 y_pred 数组）。npz 缺失时供制表。"""
    dirs = as_pred_dirs(pred_dir)
    rows: list[dict] = []
    for d in dirs:
        rows.extend(_runs_from_state_one(d))
    if not rows:
        return pd.DataFrame()
    rows = _merge_rows(rows, kind="state")
    return pd.DataFrame(rows).reset_index(drop=True)


def metrics_table_from_runs(df: pd.DataFrame) -> pd.DataFrame:
    """对 load_predictions 的每一行重算全部指标。"""
    rows = []
    for rec in df.itertuples(index=False):
        m = compute_metrics(rec.y_true, rec.y_pred, rec.y_prob)
        row = dict(
            method=rec.method, fold=int(rec.fold), seed=int(rec.seed),
            split=rec.split, test_set=rec.test_set, n=int(rec.n),
        )
        for k in SCALAR_METRIC_KEYS:
            row[k] = m.get(k)
        pc = m.get("per_class_f1") or [float("nan")] * N_CLASSES
        for i in range(N_CLASSES):
            row[f"f1_c{i}"] = pc[i] if i < len(pc) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def align_pair(
    rec_a: dict | pd.Series,
    rec_b: dict | pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按 test_idx 对齐两方法，返回 y_true, pred_a, pred_b, prob_a, prob_b。"""
    def _get(obj, name):
        if isinstance(obj, dict):
            return obj[name]
        return getattr(obj, name)

    ia = np.asarray(_get(rec_a, "test_idx"), dtype=np.int64)
    ib = np.asarray(_get(rec_b, "test_idx"), dtype=np.int64)
    ya = np.asarray(_get(rec_a, "y_true"), dtype=np.int64)
    yb = np.asarray(_get(rec_b, "y_true"), dtype=np.int64)
    pa = np.asarray(_get(rec_a, "y_pred"), dtype=np.int64)
    pb = np.asarray(_get(rec_b, "y_pred"), dtype=np.int64)
    qa = np.asarray(_get(rec_a, "y_prob"), dtype=np.float64)
    qb = np.asarray(_get(rec_b, "y_prob"), dtype=np.float64)
    if ia.shape == ib.shape and np.array_equal(ia, ib):
        if not np.array_equal(ya, yb):
            print("[warn] 相同 test_idx 但 y_true 不一致，以方法 A 为准")
        return ya, pa, pb, qa, qb
    order_b = {int(v): j for j, v in enumerate(ib)}
    keep_a = [i for i, v in enumerate(ia) if int(v) in order_b]
    if not keep_a:
        raise ValueError("两方法 test_idx 无交集，无法配对")
    idx_a = np.asarray(keep_a, dtype=np.int64)
    idx_b = np.asarray([order_b[int(ia[i])] for i in idx_a], dtype=np.int64)
    y_true = ya[idx_a]
    y_true_b = yb[idx_b]
    if not np.array_equal(y_true, y_true_b):
        n_diff = int(np.sum(y_true != y_true_b))
        print(f"[warn] 对齐后仍有 {n_diff} 条 y_true 不一致，以方法 A 为准")
    return y_true, pa[idx_a], pb[idx_b], qa[idx_a], qb[idx_b]


def pair_holdout_mask(df: pd.DataFrame) -> pd.Series:
    ts = df["test_set"].astype(str)
    sp = df["split"].astype(str)
    return (sp == "pair") & ts.isin(["test", "holdout"])


def drug_mask(df: pd.DataFrame, test_set: str | None = None) -> pd.Series:
    m = df["split"].astype(str).eq("drug") | df["test_set"].astype(str).isin(["S1", "S2"])
    if test_set is not None:
        m = m & df["test_set"].astype(str).eq(str(test_set))
    return m


def parse_method_names(spec: str | Path | dict | None) -> dict[str, str]:
    """内置映射 + 可选 JSON 文件 / JSON 字符串 / dict 覆盖。"""
    labels = dict(DEFAULT_METHOD_LABELS)
    if spec is None or spec == "":
        return labels
    if isinstance(spec, dict):
        labels.update({str(k): str(v) for k, v in spec.items()})
        return labels
    raw = str(spec).strip()
    if not raw:
        return labels
    p = Path(raw)
    if p.exists() and p.is_file():
        extra = json.loads(p.read_text(encoding="utf-8"))
    else:
        extra = json.loads(raw)
    if not isinstance(extra, dict):
        raise ValueError("--method_names 必须是 JSON 对象")
    labels.update({str(k): str(v) for k, v in extra.items()})
    return labels


def display_name(method: str, labels: dict[str, str] | None = None) -> str:
    lab = labels if labels is not None else DEFAULT_METHOD_LABELS
    return str(lab.get(str(method), method))


def methods_in(df: pd.DataFrame) -> list[str]:
    seen = list(dict.fromkeys(df["method"].astype(str).tolist()))
    ordered = [m for m in METHOD_ORDER if m in seen]
    ordered += [m for m in seen if m not in ordered]
    return ordered


def add_pred_dir_arg(parser, required: bool = True) -> None:
    parser.add_argument(
        "--pred_dir", nargs="+", required=required,
        help="一个或多个结果目录（基线 / 主流水线 / 冷启动），按 "
             "(method, fold, seed, test_set) 合并；重复 key 报错",
    )


def add_method_names_arg(parser) -> None:
    parser.add_argument(
        "--method_names", default="",
        help="覆盖显示名：JSON 文件路径或 JSON 对象字符串",
    )


def pick_representative(
    df: pd.DataFrame,
    method: str,
    test_set: str = "test",
    metric: str = "macro_f1",
) -> pd.Series | None:
    """方法 A 在该 test_set 上各 run 的 metric 取中位数对应的那一行。"""
    sub = df[df["method"].astype(str).eq(method)].copy()
    if test_set in ("test", "holdout"):
        sub = sub[sub["test_set"].astype(str).isin(["test", "holdout"])]
    else:
        sub = sub[sub["test_set"].astype(str).eq(test_set)]
    if sub.empty:
        return None
    scores = []
    for i, rec in sub.iterrows():
        val = metric_value(metric, rec.y_true, rec.y_pred, rec.y_prob)
        scores.append((i, val, int(rec.fold), int(rec.seed)))
    arr = np.asarray([s[1] for s in scores], dtype=np.float64)
    med = float(np.nanmedian(arr))
    # 距中位数最近；并列取较小 fold, seed
    best = min(scores, key=lambda s: (abs(s[1] - med), s[2], s[3]))
    row = sub.loc[best[0]]
    return row


def default_panel_methods(df: pd.DataFrame, k: int = 3) -> list[str]:
    ms = methods_in(df)
    want = [m for m in ("Exp-5", "Exp-1", "Exp-3") if m in ms]
    if len(want) >= k:
        return want[:k]
    rest = [m for m in ms if m not in want]
    return (want + rest)[:k]


def format_mean_sd(mu: float, sd: float, digits: int = 4, bold: bool = False) -> str:
    if mu is None or (isinstance(mu, float) and not np.isfinite(mu)):
        return "---"
    if sd is None or (isinstance(sd, float) and not np.isfinite(sd)):
        txt = f"{mu:.{digits}f}"
    else:
        txt = f"{mu:.{digits}f} ± {sd:.{digits}f}"
    return f"**{txt}**" if bold else txt


def format_mean_sd_tex(mu: float, sd: float, digits: int = 4, bold: bool = False) -> str:
    if mu is None or (isinstance(mu, float) and not np.isfinite(mu)):
        return "---"
    if sd is None or (isinstance(sd, float) and not np.isfinite(sd)):
        txt = f"{mu:.{digits}f}"
    else:
        txt = f"{mu:.{digits}f} {{\\pm}} {sd:.{digits}f}"
    return f"\\textbf{{{txt}}}" if bold else txt


def latex_escape(s: str) -> str:
    return (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def mean_sd(vals: Iterable[float]) -> tuple[float, float]:
    a = np.asarray(list(vals), dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(a)), float(np.std(a, ddof=0))


def is_best(values: list[float], this: float, higher: bool) -> bool:
    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(this) or not np.any(np.isfinite(arr)):
        return False
    if higher:
        return bool(np.isclose(this, np.nanmax(arr), rtol=0, atol=1e-12))
    return bool(np.isclose(this, np.nanmin(arr), rtol=0, atol=1e-12))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="加载预测并打印概览")
    add_pred_dir_arg(ap)
    add_method_names_arg(ap)
    args = ap.parse_args()
    df = load_predictions(args.pred_dir)
    names = parse_method_names(args.method_names)
    show = df[["method", "fold", "seed", "split", "test_set", "n"]].copy()
    show["label"] = show["method"].map(lambda m: display_name(m, names))
    print(show.to_string(index=False))
    print(f"n_runs={len(df)}  methods={methods_in(df)}")
