"""Mondrian split-conformal pharmacist-referral policy on saved DDI predictions.

Score r(x) = p(Tier 3) + p(Tier 4), the predicted probability of a severe tier
(Tier >= 3). On a calibration split, class-conditional (Mondrian) thresholds are

- lam_lo: the k-th smallest severe score, k = floor(alpha * (n_severe + 1))
  (lam_lo = -inf if k < 1);
- lam_hi: the k2-th smallest non-severe score,
  k2 = ceil((1 - beta) * (n_nonsevere + 1)) (lam_hi = +inf if k2 > n_nonsevere).

A test pair is ALERT if r > lam_hi, otherwise PASSIVE if r < lam_lo, and
otherwise REVIEW. REVIEW is exactly the band lam_lo <= r <= lam_hi, which is
empty when the thresholds cross. PASSIVE is a subset of {r < lam_lo} and ALERT
is a subset of {r > lam_hi}, so the Mondrian guarantees still hold:

    P(r(X) < lam_lo | Y >= 3) <= alpha,
    P(r(X) > lam_hi | Y <= 2) <= beta.

Because lam_lo is non-decreasing in alpha and lam_hi is non-increasing in beta,
the review band can only shrink, so the review fraction is non-increasing in
each budget when the other is held fixed.

Transductive runs (15 per method: folds 1-5, seeds 42/123/2024) use a random
50/50 pair split of the sealed test set (floor(n/2) calibration, the rest
evaluation). Inductive runs shuffle the held-out drug set U into halves A/B.
For S1 (exactly one drug in U) the primary calibration is drug-level conformal
risk control (Angelopoulos, Bates, Fisch, Lei, Schuster, ICLR 2024), with the
held-out drug as the exchangeable unit. Pair-level Mondrian calibration on the
same A/B split is kept as the ablation 'pair-level (held-out drugs)'. S2
(both drugs in U) stays pair-level on the drug-grouped split ('S2 pair-level').
A pooled diagnostic applies transductive fold-1 thresholds to the inductive
evaluation half. It is not a route-conditional or drug-level guarantee.

Repeats use np.random.default_rng(MASTER_SEED + run_id * 1000 + rep) with
MASTER_SEED = 20261012. Transductive run_id = method_index * 15 + (fold - 1) * 3
+ seed_index. Inductive run_id = 1000 + method_index * 6 + seed_index * 2
+ partition_index. seed_index is 42, 123, 2024 -> 0, 1, 2. No model is trained.

Writes into ``--out_dir`` (file names unchanged): conformal_transductive.csv,
conformal_inductive.csv, conformal_summary.md, conformal_raw.json, and
fig_conformal.png / fig_conformal.pdf. Prediction ``*.npz`` are read from each
``--pred_dir`` (or its ``predictions/`` subdirectory). ``y.npy`` and ``pairs.csv``
come from ``--labels`` / ``--pairs`` or from ``--data_dir``.

    python -m graphtree_ddi.analysis.conformal_referral \
        --pred_dir results/final_v2_pair_merged results/baselines_v2_pair \
                   results/prior_baseline results/final_v2_drug \
                   results/baselines_v2_drug \
        --data_dir data/processed/v2 \
        --out_dir results/analysis
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from math import ceil, floor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.text import Text  # noqa: E402

def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "graphtree_ddi" / "paths.py").is_file():
            return p
    raise RuntimeError("cannot locate repository root (graphtree_ddi/paths.py)")


_REPO_ROOT = _repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Assigned from the CLI in main() before any file is read or written.
TAB: Path | None = None
FIG: Path | None = None
Y_PATH: Path | None = None
PAIRS_PATH: Path | None = None
NAMES_PATH: Path | None = None
PRED_DIRS: list[Path] = []

MASTER_SEED = 20261012
SEEDS = (42, 123, 2024)
SEED_INDEX = {42: 0, 123: 1, 2024: 2}
N_TEST = 266_435
# (nominal level, numerator, denominator). alpha = beta grid, exact rationals.
LEVELS = (
    (0.005, 1, 200),
    (0.01, 1, 100),
    (0.02, 1, 50),
    (0.05, 1, 20),
)
ALPHAS = tuple(level for level, _, _ in LEVELS)
BETAS = ALPHAS
LEVEL_NUM = np.array([num for _, num, _ in LEVELS], dtype=np.int64)
LEVEL_DEN = np.array([den for _, _, den in LEVELS], dtype=np.int64)

METRIC_NAMES = (
    "miss_sev",
    "false_alert",
    "review_frac",
    "passive_frac",
    "alert_frac",
    "review_frac_sev",
    "review_frac_tier0",
    "review_frac_sevonly",
    "auto_acc",
    "severe_sens",
)
REF_NAMES = (
    "ref_argmax_miss",
    "ref_argmax_false_alert",
    "ref_thr05_miss",
    "ref_thr05_false_alert",
)
EXTRA_METRICS = ("lam_lo", "lam_hi", "exceed_miss", "exceed_false_alert")

CAL_DRUG = "drug-level CRC"
CAL_PAIR_S1 = "pair-level (held-out drugs)"
CAL_S2 = "S2 pair-level"
CAL_POOLED = "pooled (catalogued-pair) calibration"
CAL_REF = "reference"

# Method order is the run_id index. Files are discovered from --pred_dir.
TRANSDUCTIVE_SPECS = (
    ("Exp-1", None),
    ("Exp-2", None),
    ("Exp-3", None),
    ("Exp-4", None),
    ("Exp-5", None),
    ("MLP", None),
    ("DDIMDL", None),
    ("DeepDDI_SSP", None),
    ("DistMult", None),
    ("ComplEx", None),
    ("DrugPrior", None),
)
INDUCTIVE_SPECS = (
    ("Exp-1", None),
    ("Exp-3", None),
    ("Exp-5", None),
    ("MLP", None),
    ("DDIMDL", None),
    ("DeepDDI_SSP", None),
    ("DistMult", None),
    ("ComplEx", None),
)

# Okabe–Ito, same hex codes as scripts/fig_style.py.
OKABE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "vermillion": "#D55E00",
    "black": "#000000",
}
DPI = 300
SD_DDOF = 0
N_WORKERS = min(8, os.cpu_count() or 1)


def _utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass


def display_name(names: dict, method: str) -> str:
    if method not in names:
        raise SystemExit(f"method_names_11.json has no entry for {method}")
    return str(names[method])


def trans_run_id(method_index: int, fold: int, seed: int) -> int:
    return method_index * 15 + (fold - 1) * 3 + SEED_INDEX[seed]


def ind_run_id(method_index: int, seed: int, part: str) -> int:
    return 1000 + method_index * 6 + SEED_INDEX[seed] * 2 + (0 if part == "S1" else 1)


def split_rng(run_id: int, rep: int) -> np.random.Generator:
    return np.random.default_rng(MASTER_SEED + run_id * 1000 + rep)


def decide_masks(r: np.ndarray, lam_lo: np.ndarray, lam_hi: np.ndarray):
    """ALERT if r > lam_hi, else PASSIVE if r < lam_lo, else REVIEW."""
    r = np.asarray(r, dtype=np.float64)
    lam_lo = np.atleast_1d(np.asarray(lam_lo, dtype=np.float64))
    lam_hi = np.atleast_1d(np.asarray(lam_hi, dtype=np.float64))
    rr = r[None, None, :]
    lo = lam_lo[:, None, None]
    hi = lam_hi[None, :, None]
    alert = rr > hi
    passive = (~alert) & (rr < lo)
    review = ~(alert | passive)
    return passive, alert, review


def fit_thresholds(r: np.ndarray, y: np.ndarray):
    """Class-conditional Mondrian thresholds on one calibration set.

    Returns lam_lo[alpha_index], lam_hi[beta_index]. The two arrays are aligned
    with LEVELS but a threshold for alpha does not depend on beta.
    """
    y = np.asarray(y)
    r = np.asarray(r, dtype=np.float64)
    s = np.sort(r[y >= 3])
    t = np.sort(r[y <= 2])
    n_s = int(s.size)
    n_ns = int(t.size)
    lam_lo = np.empty(len(LEVELS), dtype=np.float64)
    lam_hi = np.empty(len(LEVELS), dtype=np.float64)
    for i, (_, num, den) in enumerate(LEVELS):
        k = (num * (n_s + 1)) // den
        if k < 1:
            lam_lo[i] = -np.inf
        elif k > n_s:
            raise RuntimeError(f"lam_lo k={k} exceeds n_s={n_s}")
        else:
            lam_lo[i] = s[k - 1]
        k2 = ((den - num) * (n_ns + 1) + den - 1) // den
        if k2 > n_ns:
            lam_hi[i] = np.inf
        elif k2 < 1:
            raise RuntimeError(f"lam_hi k2={k2} for n_ns={n_ns}")
        else:
            lam_hi[i] = t[k2 - 1]
    return lam_lo, lam_hi


def _distinct_weighted(scores: np.ndarray, weights: np.ndarray):
    """Distinct ascending scores and the summed weight at each value."""
    if scores.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty
    order = np.argsort(scores, kind="mergesort")
    ordered = np.asarray(scores, dtype=np.float64)[order]
    weight = np.asarray(weights, dtype=np.float64)[order]
    change = np.empty(ordered.size, dtype=bool)
    change[0] = True
    change[1:] = ordered[1:] != ordered[:-1]
    starts = np.flatnonzero(change)
    return ordered[starts], np.add.reduceat(weight, starts)


def _crc_lam_lo(values: np.ndarray, cumulative: np.ndarray, n_units: int, num: int, den: int) -> float:
    """Largest passive threshold whose drug-averaged calibration loss stays within budget.

    cumulative[u] = G_{u+1}. lam = v_{u*+1} with u* = max{u : G_u <= b}, b = alpha*(n+1)-1.
    """
    if n_units <= 0 or values.size == 0:
        return -np.inf
    numer = num * (n_units + 1) - den
    if numer < 0:
        return -np.inf
    scaled = cumulative * float(den)
    u_star = int(np.searchsorted(scaled, float(numer), side="right"))
    if u_star >= values.size:
        return np.inf
    return float(values[u_star])


def _crc_lam_hi(values: np.ndarray, cumulative: np.ndarray, n_units: int, num: int, den: int) -> float:
    """Smallest alert threshold whose drug-averaged false-alert loss stays within budget."""
    if n_units <= 0 or values.size == 0:
        return np.inf
    numer = num * (n_units + 1) - den
    if numer < 0:
        return np.inf
    scaled = cumulative * float(den)
    # G_u * den >= n_units * den - numer  iff weight strictly above v_u is <= b.
    target = float(n_units * den - numer)
    idx = int(np.searchsorted(scaled, target, side="left"))
    if idx >= values.size:
        return np.inf
    return float(values[idx])


def fit_crc(scores_sev, drug_sev, scores_non, drug_non, n_units: int):
    """Drug-level conformal risk control thresholds for one calibration half.

    Each calibration drug with at least one pair of the relevant class has total
    weight 1, split equally across those pairs. Returns lam_lo, lam_hi, n_s, n_ns.
    """
    n_units = int(n_units)
    scores_sev = np.asarray(scores_sev, dtype=np.float64)
    scores_non = np.asarray(scores_non, dtype=np.float64)
    if scores_sev.size:
        drug_sev = np.asarray(drug_sev, dtype=np.int64)
        counts = np.bincount(drug_sev, minlength=n_units)
        n_s = int(np.count_nonzero(counts))
        values_s, grouped_s = _distinct_weighted(scores_sev, 1.0 / counts[drug_sev])
        cumulative_s = np.cumsum(grouped_s)
    else:
        n_s = 0
        values_s = cumulative_s = np.empty(0, dtype=np.float64)
    if scores_non.size:
        drug_non = np.asarray(drug_non, dtype=np.int64)
        counts = np.bincount(drug_non, minlength=n_units)
        n_ns = int(np.count_nonzero(counts))
        values_n, grouped_n = _distinct_weighted(scores_non, 1.0 / counts[drug_non])
        cumulative_n = np.cumsum(grouped_n)
    else:
        n_ns = 0
        values_n = cumulative_n = np.empty(0, dtype=np.float64)
    lam_lo = np.empty(len(LEVELS), dtype=np.float64)
    lam_hi = np.empty(len(LEVELS), dtype=np.float64)
    for i, (_, num, den) in enumerate(LEVELS):
        lam_lo[i] = _crc_lam_lo(values_s, cumulative_s, n_s, num, den)
        lam_hi[i] = _crc_lam_hi(values_n, cumulative_n, n_ns, num, den)
    return lam_lo, lam_hi, n_s, n_ns


def drug_averaged_rates(r, y, drug, lam_lo, lam_hi, n_units: int):
    """Drug-equal averages of per-drug miss, false-alert, and review fractions."""
    r = np.asarray(r, dtype=np.float64)
    y = np.asarray(y)
    drug = np.asarray(drug, dtype=np.int64)
    lam_lo = np.asarray(lam_lo, dtype=np.float64)
    lam_hi = np.asarray(lam_hi, dtype=np.float64)
    n_units = int(n_units)
    n_level = len(lam_lo)
    drug_miss = np.full(n_level, np.nan)
    drug_fa = np.full(n_level, np.nan)
    drug_review = np.full((n_level, n_level), np.nan)
    exceed_miss = np.full(n_level, np.nan)
    exceed_fa = np.full(n_level, np.nan)
    severe = y >= 3
    nonsevere = y <= 2
    if np.any(severe):
        grouped = drug[severe]
        scores = r[severe]
        counts = np.bincount(grouped, minlength=n_units).astype(np.float64)
        valid = counts > 0
        denom = counts[valid]
        for i, lo in enumerate(lam_lo):
            hits = np.bincount(grouped, weights=(scores < lo).astype(np.float64), minlength=n_units)[valid]
            value = float(np.mean(hits / denom))
            drug_miss[i] = value
            exceed_miss[i] = float(value * float(LEVEL_DEN[i]) > float(LEVEL_NUM[i]) + 1e-9)
    if np.any(nonsevere):
        grouped = drug[nonsevere]
        scores = r[nonsevere]
        counts = np.bincount(grouped, minlength=n_units).astype(np.float64)
        valid = counts > 0
        denom = counts[valid]
        for j, hi in enumerate(lam_hi):
            hits = np.bincount(grouped, weights=(scores > hi).astype(np.float64), minlength=n_units)[valid]
            value = float(np.mean(hits / denom))
            drug_fa[j] = value
            exceed_fa[j] = float(value * float(LEVEL_DEN[j]) > float(LEVEL_NUM[j]) + 1e-9)
    counts = np.bincount(drug, minlength=n_units).astype(np.float64)
    valid = counts > 0
    if np.any(valid):
        denom = counts[valid]
        for i, lo in enumerate(lam_lo):
            at_least = r >= lo
            for j, hi in enumerate(lam_hi):
                review = at_least & (r <= hi)
                hits = np.bincount(drug, weights=review.astype(np.float64), minlength=n_units)[valid]
                drug_review[i, j] = float(np.mean(hits / denom))
    return drug_miss, drug_fa, drug_review, exceed_miss, exceed_fa


def _prefix(mask: np.ndarray) -> np.ndarray:
    out = np.empty(mask.size + 1, dtype=np.int64)
    out[0] = 0
    np.cumsum(mask, out=out[1:])
    return out


def eval_grid(r: np.ndarray, y: np.ndarray, argmax: np.ndarray, lam_lo, lam_hi):
    """Metrics, exceed_miss, exceed_false_alert on one evaluation set.

    Scores are sorted once. Every alpha/beta pair is then a pair of order-statistic
    lookups, including ties: r < lam_lo is searchsorted(..., side='left') and
    r <= lam_hi is searchsorted(..., side='right'). exceed_* is 1 when the
    empirical rate is strictly above the nominal budget, compared in integers
    (count * denominator > numerator * n).
    """
    lam_lo = np.asarray(lam_lo, dtype=np.float64)
    lam_hi = np.asarray(lam_hi, dtype=np.float64)
    if lam_lo.shape != (len(LEVELS),) or lam_hi.shape != (len(LEVELS),):
        raise RuntimeError("eval_grid expects one threshold per grid level")
    y = np.asarray(y)
    r = np.asarray(r, dtype=np.float64)
    argmax = np.asarray(argmax)
    n = int(r.size)
    n_level = len(LEVELS)
    if n == 0:
        empty = np.full((n_level, n_level, len(METRIC_NAMES)), np.nan)
        return empty, np.full((n_level, n_level), np.nan), np.full((n_level, n_level), np.nan)

    order = np.argsort(r, kind="quicksort")
    y_s = y[order]
    r_s = r[order]
    p_sev = _prefix(y_s >= 3)
    p_non = _prefix(y_s <= 2)
    p_t0 = _prefix(y_s == 0)
    p_ge = _prefix(y_s >= 1)
    p_cor = _prefix(argmax[order] == y_s)
    n_sev = int(p_sev[n])
    n_non = int(p_non[n])
    n_t0 = int(p_t0[n])
    n_ge = int(p_ge[n])
    n_cor = int(p_cor[n])
    idx_lo = np.searchsorted(r_s, lam_lo, side="left")
    idx_hi = np.searchsorted(r_s, lam_hi, side="right")

    metrics = np.empty((n_level, n_level, len(METRIC_NAMES)), dtype=np.float64)
    exceed_miss = np.full((n_level, n_level), np.nan)
    exceed_fa = np.full((n_level, n_level), np.nan)
    for i in range(n_level):
        i_lo = int(idx_lo[i])
        lo = float(lam_lo[i])
        for j in range(n_level):
            i_hi = int(idx_hi[j])
            hi = float(lam_hi[j])
            # ALERT is always r > lam_hi. PASSIVE is r < lam_lo among the rest.
            # When lam_lo > lam_hi that rest is r <= lam_hi, and REVIEW is empty.
            n_alert = n - i_hi
            c_alert_sev = int(p_sev[n] - p_sev[i_hi])
            c_alert_non = int(p_non[n] - p_non[i_hi])
            c_alert_t0 = int(p_t0[n] - p_t0[i_hi])
            c_alert_ge = int(p_ge[n] - p_ge[i_hi])
            if lo > hi:
                n_pass = i_hi
                n_hit = n_cor
                c_pass_sev = int(p_sev[i_hi])
                c_pass_t0 = int(p_t0[i_hi])
                c_pass_ge = int(p_ge[i_hi])
            else:
                n_pass = i_lo
                n_hit = int(p_cor[i_lo] + (n_cor - p_cor[i_hi]))
                c_pass_sev = int(p_sev[i_lo])
                c_pass_t0 = int(p_t0[i_lo])
                c_pass_ge = int(p_ge[i_lo])
            n_review = n - n_pass - n_alert
            if n_review < 0:
                raise RuntimeError("PASSIVE and ALERT overlap")
            if n_sev:
                miss = c_pass_sev / n_sev
                review_sev = (n_sev - c_pass_sev - c_alert_sev) / n_sev
                exceed_miss[i, j] = float(c_pass_sev * int(LEVEL_DEN[i]) > int(LEVEL_NUM[i]) * n_sev)
            else:
                miss = np.nan
                review_sev = np.nan
            if n_non:
                false_alert = c_alert_non / n_non
                exceed_fa[i, j] = float(c_alert_non * int(LEVEL_DEN[j]) > int(LEVEL_NUM[j]) * n_non)
            else:
                false_alert = np.nan
            review_t0 = (n_t0 - c_pass_t0 - c_alert_t0) / n_t0 if n_t0 else np.nan
            review_ge1 = (n_ge - c_pass_ge - c_alert_ge) / n_ge if n_ge else np.nan
            n_auto = n_pass + n_alert
            auto_acc = n_hit / n_auto if n_auto else np.nan
            metrics[i, j, 0] = miss
            metrics[i, j, 1] = false_alert
            metrics[i, j, 2] = n_review / n
            metrics[i, j, 3] = n_pass / n
            metrics[i, j, 4] = n_alert / n
            metrics[i, j, 5] = review_sev
            metrics[i, j, 6] = review_t0
            metrics[i, j, 7] = review_ge1
            metrics[i, j, 8] = auto_acc
            metrics[i, j, 9] = 1.0 - miss
    return metrics, exceed_miss, exceed_fa


def eval_reference(r: np.ndarray, y: np.ndarray, argmax: np.ndarray) -> np.ndarray:
    """Argmax and fixed r >= 0.5 policies on the same evaluation pairs."""
    y = np.asarray(y)
    r = np.asarray(r, dtype=np.float64)
    argmax = np.asarray(argmax)
    sev = y >= 3
    non = y <= 2
    out = np.full(4, np.nan, dtype=np.float64)
    if np.any(sev):
        out[0] = float(np.mean(argmax[sev] <= 2))
        out[2] = float(np.mean(r[sev] < 0.5))
    if np.any(non):
        out[1] = float(np.mean(argmax[non] >= 3))
        out[3] = float(np.mean(r[non] >= 0.5))
    return out


def _thresholds_close(left: np.ndarray, right: np.ndarray) -> bool:
    same = np.isfinite(left) & np.isfinite(right)
    if np.any(same) and np.max(np.abs(left[same] - right[same])) > 0:
        return False
    if np.any(np.isneginf(left) != np.isneginf(right)):
        return False
    if np.any(np.isposinf(left) != np.isposinf(right)):
        return False
    return True


def _assert_crc_matches_pair_level(rng: np.random.Generator) -> None:
    """One pair per drug must reproduce the Mondrian order statistics, including ties."""
    cases = []
    for n in (1, 2, 3, 5, 8, 17, 40, 100, 250, 1000):
        scores = rng.random(n)
        labels = rng.integers(0, 5, size=n)
        cases.append((scores, labels))
        tied = rng.choice(rng.random(max(2, n // 5)), size=n)
        cases.append((tied, labels))
    # Force a class to be empty and a class to be tiny.
    scores = rng.random(30)
    labels = np.zeros(30, dtype=np.int64)
    labels[:2] = 3
    cases.append((scores, labels))
    mismatches = 0
    for scores, labels in cases:
        drugs = np.arange(scores.size)
        pair_lo, pair_hi = fit_thresholds(scores, labels)
        severe = labels >= 3
        nonsevere = labels <= 2
        crc_lo, crc_hi, _, _ = fit_crc(scores[severe], drugs[severe], scores[nonsevere], drugs[nonsevere], scores.size)
        if not (_thresholds_close(pair_lo, crc_lo) and _thresholds_close(pair_hi, crc_hi)):
            mismatches += 1
            print(
                f"[crc-equiv] mismatch n={scores.size} pair_lo={pair_lo} crc_lo={crc_lo} "
                f"pair_hi={pair_hi} crc_hi={crc_hi}",
                flush=True,
            )
    print(f"[crc-equiv] unit-weight thresholds vs pair-level order statistics: "
          f"{len(cases) - mismatches}/{len(cases)} exact", flush=True)
    if mismatches:
        raise SystemExit("drug-level CRC does not match pair-level thresholds when every unit has one pair")


def run_clustered_crc(n_groups: int = 1500, repeats: int = 200):
    """Exchangeable groups with heavy-tailed sizes. The guarantee is drug-averaged."""
    rng = np.random.default_rng(MASTER_SEED + 11)
    sigma = 0.9
    mu = np.log(150.0) - 0.5 * sigma * sigma
    sizes = np.clip(np.rint(rng.lognormal(mu, sigma, size=n_groups)), 1, 4000).astype(np.int64)
    drug = np.repeat(np.arange(n_groups), sizes)
    shift = rng.normal(0.0, 0.45, size=n_groups)
    latent = shift[drug] + rng.normal(0.0, 0.3, size=drug.size)
    severe_prob = 0.08 + 0.35 / (1.0 + np.exp(-1.4 * shift[drug]))
    is_severe = rng.random(drug.size) < severe_prob
    y = np.empty(drug.size, dtype=np.int64)
    y[is_severe] = rng.integers(3, 5, size=int(is_severe.sum()))
    y[~is_severe] = rng.integers(0, 3, size=int((~is_severe).sum()))
    score = 1.0 / (1.0 + np.exp(-(latent + np.where(is_severe, 0.55, -0.25))))
    print(
        f"[crc-cluster] groups={n_groups} pairs={drug.size} "
        f"size mean={sizes.mean():.1f} median={np.median(sizes):.0f} max={sizes.max()}",
        flush=True,
    )
    n_level = len(LEVELS)
    drug_miss = np.empty((repeats, n_level, n_level))
    drug_fa = np.empty_like(drug_miss)
    pair_miss = np.empty_like(drug_miss)
    pair_fa = np.empty_like(drug_miss)
    t0 = time.perf_counter()
    n_half = n_groups // 2
    for rep in range(repeats):
        perm = split_rng(2_000_000, rep).permutation(n_groups)
        in_a = np.zeros(n_groups, dtype=bool)
        in_a[perm[:n_half]] = True
        cal = in_a[drug]
        ev = ~cal
        severe = y >= 3
        nonsevere = y <= 2
        lo, hi, _, _ = fit_crc(score[cal & severe], drug[cal & severe], score[cal & nonsevere], drug[cal & nonsevere], n_groups)
        metrics, _, _ = eval_grid(score[ev], y[ev], y[ev], lo, hi)
        d_miss, d_fa, _, _, _ = drug_averaged_rates(score[ev], y[ev], drug[ev], lo, hi, n_groups)
        pair_miss[rep] = metrics[:, :, 0]
        pair_fa[rep] = metrics[:, :, 1]
        drug_miss[rep] = d_miss[:, None]
        drug_fa[rep] = d_fa[None, :]
    dt = time.perf_counter() - t0
    rows = []
    ok_all = True
    for i, alpha in enumerate(ALPHAS):
        for j, beta in enumerate(BETAS):
            m = drug_miss[:, i, j]
            f = drug_fa[:, i, j]
            mean_m = float(m.mean())
            mean_f = float(f.mean())
            se_m = float(m.std(ddof=1) / np.sqrt(repeats))
            se_f = float(f.std(ddof=1) / np.sqrt(repeats))
            ok_m = mean_m <= alpha + 3.0 * se_m
            ok_f = mean_f <= beta + 3.0 * se_f
            ok = bool(ok_m and ok_f)
            ok_all = ok_all and ok
            rows.append(
                {
                    "alpha": alpha,
                    "beta": beta,
                    "mean_drug_miss": mean_m,
                    "se_drug_miss": se_m,
                    "mean_drug_false_alert": mean_f,
                    "se_drug_false_alert": se_f,
                    "mean_pair_miss": float(pair_miss[:, i, j].mean()),
                    "mean_pair_false_alert": float(pair_fa[:, i, j].mean()),
                    "ok": ok,
                }
            )
            print(
                f"[crc-cluster] alpha={alpha:.3f} beta={beta:.3f} "
                f"drug_miss={mean_m:.6f} se={se_m:.6f} drug_fa={mean_f:.6f} se={se_f:.6f} "
                f"pair_miss={rows[-1]['mean_pair_miss']:.6f} pair_fa={rows[-1]['mean_pair_false_alert']:.6f} "
                f"{'PASS' if ok else 'FAIL'}",
                flush=True,
            )
    print(f"[crc-cluster] R={repeats} in {dt:.1f}s -> {'PASS' if ok_all else 'FAIL'}", flush=True)
    if not ok_all:
        raise SystemExit("clustered drug-level conformal risk control self-check failed")
    return rows, dt


def run_unit_checks() -> None:
    samples = [0, 1, 2, 19, 20, 99, 100, 199, 200, 1000, 2923, 32068, 101137, 133217, 266435]
    for n in samples:
        for _, num, den in LEVELS:
            assert (num * (n + 1)) // den == floor(Fraction(num, den) * (n + 1))
            assert ((den - num) * (n + 1) + den - 1) // den == ceil(
                (1 - Fraction(num, den)) * (n + 1)
            )
    disagree = 0
    for n in range(0, 200_000, 17):
        for level, num, den in LEVELS:
            exact = (num * (n + 1)) // den
            approx = int(np.floor(level * (n + 1) + 1e-12))
            k2_exact = ((den - num) * (n + 1) + den - 1) // den
            k2_approx = int(np.ceil((1.0 - level) * (n + 1) - 1e-12))
            disagree += int(exact != approx) + int(k2_exact != k2_approx)
    print(f"[check] float floor/ceil disagreements on a stride-17 grid: {disagree}", flush=True)

    r = np.array([0.1, 0.2, 0.5, 0.8, 0.9])
    expected = {
        (0.2, 0.8): [0, 2, 2, 2, 1],
        (0.8, 0.2): [0, 0, 1, 1, 1],
        (-np.inf, 0.8): [2, 2, 2, 2, 1],
        (0.2, np.inf): [0, 2, 2, 2, 2],
    }
    for (lo, hi), lab in expected.items():
        passive, alert, review = decide_masks(r, [lo], [hi])
        got = np.where(passive[0, 0], 0, np.where(alert[0, 0], 1, 2))
        if list(got) != lab:
            raise SystemExit(f"decision rule mismatch for {(lo, hi)}: {list(got)} != {lab}")

    rng = np.random.default_rng(0)
    scores = rng.random(1000)
    labels = np.array([3] * 400 + [0] * 600)
    lo, hi = fit_thresholds(scores, labels)
    severe = np.sort(scores[:400])
    nonsevere = np.sort(scores[400:])
    k = (1 * 401) // 100
    k2 = ((100 - 1) * 601 + 99) // 100
    if k2 != 595 or lo[1] != severe[k - 1] or hi[1] != nonsevere[k2 - 1]:
        raise SystemExit("fit_thresholds off-by-one against sorted order statistics")

    r = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
    y = np.array([0, 0, 3, 3, 1, 4])
    argmax = np.array([0, 0, 3, 2, 1, 4])
    # Broadcast the hand threshold across the grid; metrics must not depend on it
    # being a fitted quantile. Check the (0, 0) cell only.
    metrics, exceed_m, exceed_f = eval_grid(r, y, argmax, np.full(4, 0.25), np.full(4, 0.85))
    got = metrics[0, 0]
    target = np.array([0.0, 1.0 / 3.0, 2.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0, 2.0 / 3.0, 0.0, 0.5, 1.0, 1.0])
    if np.max(np.abs(got - target)) > 1e-12:
        raise SystemExit(f"eval_grid hand example mismatch: {got} != {target}")
    if exceed_m[0, 0] != 0.0 or exceed_f[0, 0] != 1.0:
        raise SystemExit("exceedance flags mismatch on the hand example")
    ref = eval_reference(r, y, argmax)
    if np.max(np.abs(ref - np.array([1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0]))) > 1e-12:
        raise SystemExit(f"reference policy mismatch: {ref}")
    # Ties at the threshold are REVIEW. Crossed thresholds use the other side of the rule.
    tied = eval_grid(
        np.array([0.2, 0.2, 0.5, 0.8, 0.8]),
        np.array([0, 3, 1, 2, 4]),
        np.array([0, 3, 1, 2, 4]),
        np.full(4, 0.5),
        np.full(4, 0.5),
    )[0][0, 0]
    tied_target = np.array([0.5, 1.0 / 3.0, 0.2, 0.4, 0.4, 0.0, 0.0, 0.25, 1.0, 0.5])
    if np.max(np.abs(tied - tied_target)) > 1e-12:
        raise SystemExit(f"tie-threshold metrics mismatch: {tied}")
    crossed = eval_grid(
        np.array([0.1, 0.4, 0.9]),
        np.array([0, 3, 1]),
        np.array([0, 3, 1]),
        np.full(4, 0.8),
        np.full(4, 0.2),
    )[0][0, 0]
    crossed_target = np.array([0.0, 0.5, 0.0, 1.0 / 3.0, 2.0 / 3.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    if np.max(np.abs(crossed - crossed_target)) > 1e-12:
        raise SystemExit(f"crossed-threshold metrics mismatch: {crossed}")
    rng = np.random.default_rng(1)
    sample_r = rng.random(800)
    sample_y = rng.integers(0, 5, 800)
    sample_lo, sample_hi = fit_thresholds(sample_r[:400], sample_y[:400])
    sample_review = eval_grid(sample_r[400:], sample_y[400:], sample_y[400:], sample_lo, sample_hi)[0][:, :, 2]
    if np.any(sample_review[1:] > sample_review[:-1] + 1e-12) or np.any(
        sample_review[:, 1:] > sample_review[:, :-1] + 1e-12
    ):
        raise SystemExit("pair-level review is not monotone in alpha and beta on one split")
    _assert_crc_matches_pair_level(rng)
    print("[check] order-statistic arithmetic, decisions, and metric definitions OK", flush=True)


def run_synthetic(n: int = 200_000, repeats: int = 200):
    """Exchangeable Dirichlet scores. Validity is E[error] <= budget, checked as mean <= budget + 3 SE."""
    rng = np.random.default_rng(MASTER_SEED)
    probs = rng.dirichlet(np.ones(5), size=n)
    cdf = np.cumsum(probs, axis=1)
    cdf[:, -1] = 1.0
    y = (cdf < rng.random(n)[:, None]).sum(axis=1).astype(np.int64)
    if int(y.min()) < 0 or int(y.max()) > 4:
        raise SystemExit(f"synthetic labels out of range [{y.min()}, {y.max()}]")
    r = probs[:, 3] + probs[:, 4]
    # Class index is unused by miss/false-alert; argmax is filled so auto_acc is defined.
    argmax_full = probs.argmax(axis=1).astype(np.int64)
    miss = np.empty((repeats, len(LEVELS), len(LEVELS)))
    false_alert = np.empty_like(miss)
    t0 = time.perf_counter()
    n_cal = n // 2
    for rep in range(repeats):
        perm = split_rng(1_000_000, rep).permutation(n)
        cal = perm[:n_cal]
        ev = perm[n_cal:]
        lo, hi = fit_thresholds(r[cal], y[cal])
        metrics, _, _ = eval_grid(r[ev], y[ev], argmax_full[ev], lo, hi)
        miss[rep] = metrics[:, :, 0]
        false_alert[rep] = metrics[:, :, 1]
    dt = time.perf_counter() - t0
    rows = []
    ok_all = True
    for i, alpha in enumerate(ALPHAS):
        for j, beta in enumerate(BETAS):
            m = miss[:, i, j]
            f = false_alert[:, i, j]
            mean_m = float(m.mean())
            mean_f = float(f.mean())
            se_m = float(m.std(ddof=1) / np.sqrt(repeats))
            se_f = float(f.std(ddof=1) / np.sqrt(repeats))
            ok_m = mean_m <= alpha + 3.0 * se_m
            ok_f = mean_f <= beta + 3.0 * se_f
            ok = bool(ok_m and ok_f)
            ok_all = ok_all and ok
            rows.append(
                {
                    "alpha": alpha,
                    "beta": beta,
                    "mean_miss": mean_m,
                    "se_miss": se_m,
                    "bound_miss": alpha + 3.0 * se_m,
                    "ok_miss": ok_m,
                    "mean_false_alert": mean_f,
                    "se_false_alert": se_f,
                    "bound_false_alert": beta + 3.0 * se_f,
                    "ok_false_alert": ok_f,
                }
            )
            print(
                f"[synthetic] alpha={alpha:.3f} beta={beta:.3f} "
                f"miss={mean_m:.6f} se={se_m:.6f} bound={alpha + 3.0 * se_m:.6f} "
                f"fa={mean_f:.6f} se={se_f:.6f} bound={beta + 3.0 * se_f:.6f} "
                f"{'PASS' if ok else 'FAIL'}",
                flush=True,
            )
    print(f"[synthetic] R={repeats} n={n} in {dt:.1f}s -> {'PASS' if ok_all else 'FAIL'}", flush=True)
    if not ok_all:
        raise SystemExit("synthetic conformal self-check failed")
    return rows, dt


def _resolve_pred_dir(pred_dir: str | Path) -> Path:
    """Accept a results directory or its predictions/ subdirectory."""
    d = Path(pred_dir)
    if not d.is_dir():
        raise SystemExit(f"prediction directory not found: {d}")
    sub = d / "predictions"
    if sub.is_dir() and any(sub.glob("*.npz")):
        return sub
    return d


def _search_dirs(directory: Path | None) -> list[Path]:
    if directory is not None:
        return [directory]
    if not PRED_DIRS:
        raise SystemExit("no prediction directories configured (--pred_dir)")
    return list(PRED_DIRS)


def _gather_npz(directory: Path | None, pattern: str) -> list[Path]:
    found: dict[Path, Path] = {}
    for folder in _search_dirs(directory):
        for path in folder.glob(pattern):
            found.setdefault(path.resolve(), path)
    return list(found.values())


def list_transductive(method: str, directory: Path | None = None):
    rx = re.compile(rf"^{re.escape(method)}_f(\d+)_s(\d+)\.npz$")
    raw = [
        p
        for p in _gather_npz(directory, f"{method}_f*_s*.npz")
        if not p.name.endswith(("_S1.npz", "_S2.npz"))
    ]
    parsed = []
    bad = []
    for path in raw:
        match = rx.match(path.name)
        if match is None:
            bad.append(path.name)
            continue
        parsed.append((int(match.group(1)), int(match.group(2)), path))
    if bad:
        raise SystemExit(f"{method}: unexpected prediction files {bad}")
    parsed.sort(key=lambda item: (item[0], item[1]))
    if len(parsed) != 15:
        where = directory if directory is not None else PRED_DIRS
        raise SystemExit(f"{method}: expected 15 transductive runs, found {len(parsed)} in {where}")
    folds = {item[0] for item in parsed}
    seeds = {item[1] for item in parsed}
    if folds != {1, 2, 3, 4, 5} or seeds != set(SEEDS):
        raise SystemExit(f"{method}: unexpected folds {sorted(folds)} seeds {sorted(seeds)}")
    return parsed


def list_inductive(method: str, directory: Path | None = None):
    rx = re.compile(rf"^{re.escape(method)}_f(\d+)_s(\d+)_(S[12])\.npz$")
    raw = _gather_npz(directory, f"{method}_f*_s*_S[12].npz")
    parsed = []
    bad = []
    for path in raw:
        match = rx.match(path.name)
        if match is None:
            bad.append(path.name)
            continue
        parsed.append((int(match.group(1)), int(match.group(2)), match.group(3), path))
    if bad:
        raise SystemExit(f"{method}: unexpected inductive files {bad}")
    if len(parsed) != 6:
        raise SystemExit(f"{method}: expected 6 inductive runs (3 seeds x S1/S2), found {len(parsed)}")
    folds = {item[0] for item in parsed}
    seeds = {item[1] for item in parsed}
    parts = {item[2] for item in parsed}
    if len(folds) != 1 or seeds != set(SEEDS) or parts != {"S1", "S2"}:
        raise SystemExit(
            f"{method}: unexpected inductive folds {sorted(folds)} seeds {sorted(seeds)} parts {sorted(parts)}"
        )
    parsed.sort(key=lambda item: (item[1], item[2]))
    return parsed


def load_npz(path: Path, y_all: np.ndarray):
    with np.load(path) as zipped:
        idx = np.array(zipped["test_idx"], dtype=np.int64, copy=True)
        y = np.array(zipped["y_true"], dtype=np.int64, copy=True)
        y_pred = np.array(zipped["y_pred"], dtype=np.int64, copy=True)
        prob = np.array(zipped["y_prob"], dtype=np.float32, copy=True)
    if prob.ndim != 2 or prob.shape[1] != 5 or prob.shape[0] != idx.shape[0] or y.shape[0] != idx.shape[0]:
        raise SystemExit(f"{path.name}: unexpected array shapes")
    if not np.isfinite(prob).all():
        raise SystemExit(f"{path.name}: non-finite y_prob")
    dev = float(np.max(np.abs(prob.sum(axis=1, dtype=np.float64) - 1.0)))
    r = prob[:, 3].astype(np.float64) + prob[:, 4].astype(np.float64)
    if not np.isfinite(r).all():
        raise SystemExit(f"{path.name}: non-finite severe score")
    argmax = np.argmax(prob, axis=1).astype(np.int64)
    n_mismatch = int(np.count_nonzero(argmax != y_pred))
    if idx.size == 0 or int(idx.min()) < 0 or int(idx.max()) >= len(y_all):
        raise SystemExit(f"{path.name}: test_idx out of range of y.npy")
    if len(np.unique(idx)) != len(idx):
        raise SystemExit(f"{path.name}: duplicate test_idx")
    if not np.array_equal(y, np.asarray(y_all[idx], dtype=np.int64)):
        raise SystemExit(f"{path.name}: y_true != y.npy[test_idx]. Stopping.")
    if int(y.min()) < 0 or int(y.max()) > 4:
        raise SystemExit(f"{path.name}: y_true outside 0..4")
    return idx, y, r, argmax, dev, n_mismatch


def load_pair_universe():
    y_all = np.load(Y_PATH)
    pairs = pd.read_csv(PAIRS_PATH, usecols=["drug1_id", "drug2_id"])
    if len(pairs) != len(y_all):
        raise SystemExit(f"pairs.csv has {len(pairs)} rows but y.npy has {len(y_all)}")
    drugs = sorted(set(pairs.drug1_id) | set(pairs.drug2_id))
    d1 = pairs["drug1_id"].to_numpy()
    d2 = pairs["drug2_id"].to_numpy()
    n_u = int(round(0.2 * len(drugs)))
    print(
        f"[data] pairs={len(y_all)} drugs={len(drugs)} n_u={n_u} "
        f"|A|={n_u // 2} |B|={n_u - n_u // 2}",
        flush=True,
    )
    split_by_seed = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(drugs))
        heldout = {drugs[int(i)] for i in perm[:n_u]}
        if len(heldout) != n_u:
            raise SystemExit(f"seed {seed}: held-out drug set has {len(heldout)} drugs, expected {n_u}")
        held_sorted = np.array(sorted(heldout))
        in1 = np.isin(d1, held_sorted)
        in2 = np.isin(d2, held_sorted)
        s1 = np.flatnonzero(in1 ^ in2)
        s2 = np.flatnonzero(in1 & in2)
        mask1 = np.zeros(len(y_all), dtype=bool)
        mask2 = np.zeros(len(y_all), dtype=bool)
        mask1[s1] = True
        mask2[s2] = True
        split_by_seed[seed] = {
            "U_index": pd.Index(held_sorted.tolist()),
            "S1": mask1,
            "S2": mask2,
            "n_s1": int(s1.size),
            "n_s2": int(s2.size),
            "n_u": n_u,
        }
        print(f"[data] seed {seed}: reconstructed S1={s1.size} S2={s2.size}", flush=True)
    return y_all, d1, d2, split_by_seed


def _positions(index: pd.Index, drugs: np.ndarray, path: Path) -> np.ndarray:
    pos = index.get_indexer(drugs)
    if np.any(pos < 0):
        raise SystemExit(f"{path.name}: a pair drug is missing from the reconstructed held-out set U")
    return pos.astype(np.int16, copy=False)


def prepare_inductive(y_all, d1, d2, split_by_seed):
    jobs = []
    seen_ids = set()
    for method_index, (method, directory) in enumerate(INDUCTIVE_SPECS):
        for fold, seed, part, path in list_inductive(method, directory):
            idx, y, r, argmax, dev, n_mismatch = load_npz(path, y_all)
            expected = split_by_seed[seed][part]
            got = np.zeros(expected.shape, dtype=bool)
            got[idx] = True
            if int(got.sum()) != len(idx) or not np.array_equal(got, expected):
                inter = int(np.count_nonzero(got & expected))
                raise SystemExit(
                    f"ASSERTION FAILED: {path.name} test_idx does not match reconstructed {part} "
                    f"for drug_split_seed {seed}. n_file={len(idx)} n_recon={int(expected.sum())} "
                    f"n_intersection={inter}. Stopping without guessing."
                )
            d1r = d1[idx]
            d2r = d2[idx]
            in1 = split_by_seed[seed]["U_index"].get_indexer(d1r) >= 0
            in2 = split_by_seed[seed]["U_index"].get_indexer(d2r) >= 0
            if part == "S1":
                if not np.all(in1 ^ in2):
                    raise SystemExit(f"{path.name}: S1 rows are not exactly one drug in U")
                u_drug = np.where(in1, d1r, d2r)
                positions = _positions(split_by_seed[seed]["U_index"], u_drug, path)
            else:
                if not np.all(in1 & in2):
                    raise SystemExit(f"{path.name}: S2 rows are not both drugs in U")
                positions = (
                    _positions(split_by_seed[seed]["U_index"], d1r, path),
                    _positions(split_by_seed[seed]["U_index"], d2r, path),
                )
            run_id = ind_run_id(method_index, seed, part)
            if run_id in seen_ids:
                raise SystemExit(f"duplicate inductive run_id {run_id}")
            seen_ids.add(run_id)
            jobs.append(
                {
                    "method": method,
                    "method_index": method_index,
                    "fold": fold,
                    "seed": seed,
                    "part": part,
                    "path": path,
                    "y": np.asarray(y, dtype=np.int8),
                    "r": r,
                    "argmax": np.asarray(argmax, dtype=np.int8),
                    "positions": positions,
                    "n_u": split_by_seed[seed]["n_u"],
                    "run_id": run_id,
                    "prob_dev": dev,
                    "n_mismatch": n_mismatch,
                }
            )
            print(
                f"[check] {path.name}: test_idx matches reconstructed {part} "
                f"(n={len(idx)}, prob |sum-1|={dev:.3e})",
                flush=True,
            )
    return jobs


def _accumulate_eval(r, y, argmax, lo, hi, sums):
    metrics, exceed_m, exceed_f = eval_grid(r, y, argmax, lo, hi)
    ref = eval_reference(r, y, argmax)
    sums["metrics"] += metrics
    sums["exceed_miss"] += exceed_m
    sums["exceed_fa"] += exceed_f
    sums["lam_lo"] += lo
    sums["lam_hi"] += hi
    sums["reference"] += ref
    sums["n"] += 1
    return metrics


def _empty_sums():
    n_level = len(LEVELS)
    return {
        "metrics": np.zeros((n_level, n_level, len(METRIC_NAMES))),
        "exceed_miss": np.zeros((n_level, n_level)),
        "exceed_fa": np.zeros((n_level, n_level)),
        "lam_lo": np.zeros(n_level),
        "lam_hi": np.zeros(n_level),
        "reference": np.zeros(len(REF_NAMES)),
        "n": 0,
    }


def _mean_sums(sums):
    n = sums["n"]
    if n <= 0:
        raise RuntimeError("no repeats accumulated")
    return {key: (value / n if key != "n" else n) for key, value in sums.items()}


def _pair_split_means(r, y, argmax, run_id: int, repeats: int, store_lam: bool, store_mono: bool = False):
    sums = _empty_sums()
    lo_reps = np.empty((repeats, len(LEVELS))) if store_lam else None
    hi_reps = np.empty((repeats, len(LEVELS))) if store_lam else None
    mono = np.empty((repeats, len(LEVELS), len(LEVELS))) if store_mono else None
    n = int(r.size)
    n_cal = n // 2
    for rep in range(repeats):
        perm = split_rng(run_id, rep).permutation(n)
        cal = perm[:n_cal]
        ev = perm[n_cal:]
        lo, hi = fit_thresholds(r[cal], y[cal])
        metrics = _accumulate_eval(r[ev], y[ev], argmax[ev], lo, hi, sums)
        if store_lam:
            lo_reps[rep] = lo
            hi_reps[rep] = hi
        if store_mono:
            mono[rep] = metrics[:, :, 2]
    return _mean_sums(sums), lo_reps, hi_reps, mono


def _transductive_worker(task):
    path = task["path"]
    t0 = time.perf_counter()
    try:
        idx, y, r, argmax, dev, mismatch = load_npz(path, task["y_all"])
    except SystemExit as exc:
        raise RuntimeError(str(exc)) from exc
    if len(idx) != N_TEST:
        raise RuntimeError(f"{path.name}: expected n={N_TEST}, found {len(idx)}")
    mean, lo_reps, hi_reps, mono = _pair_split_means(
        r, y, argmax, task["run_id"], task["repeats"], task["store_lam"], task["store_mono"]
    )
    return {
        "method": task["method"],
        "file": path.name,
        "fold": task["fold"],
        "seed": task["seed"],
        "run_id": task["run_id"],
        "prob_dev": dev,
        "n_mismatch": mismatch,
        "seconds": time.perf_counter() - t0,
        "idx": idx,
        "lo_reps": lo_reps,
        "hi_reps": hi_reps,
        "mono_review": mono,
        **mean,
    }


def _stack_runs(runs):
    return {
        "files": [item["file"] for item in runs],
        "folds": [item["fold"] for item in runs],
        "seeds": [item["seed"] for item in runs],
        "run_ids": [item["run_id"] for item in runs],
        "prob_dev": [item["prob_dev"] for item in runs],
        "n_mismatch": [item["n_mismatch"] for item in runs],
        "metrics": np.stack([item["metrics"] for item in runs]),
        "exceed_miss": np.stack([item["exceed_miss"] for item in runs]),
        "exceed_fa": np.stack([item["exceed_fa"] for item in runs]),
        "lam_lo": np.stack([item["lam_lo"] for item in runs]),
        "lam_hi": np.stack([item["lam_hi"] for item in runs]),
        "reference": np.stack([item["reference"] for item in runs]),
    }


def run_transductive(y_all, repeats: int):
    tasks = []
    seen_ids = set()
    for method_index, (method, directory) in enumerate(TRANSDUCTIVE_SPECS):
        for fold, seed, path in list_transductive(method, directory):
            run_id = trans_run_id(method_index, fold, seed)
            if run_id in seen_ids:
                raise SystemExit(f"duplicate transductive run_id {run_id}")
            seen_ids.add(run_id)
            tasks.append(
                {
                    "method": method,
                    "path": path,
                    "fold": fold,
                    "seed": seed,
                    "run_id": run_id,
                    "repeats": repeats,
                    "store_lam": method in ("Exp-1", "Exp-5") and fold == 1,
                    "store_mono": method in ("Exp-1", "Exp-5", "DrugPrior") and fold == 1 and seed == 42,
                    "y_all": y_all,
                }
            )
    print(f"[transductive] {len(tasks)} runs on {N_WORKERS} threads", flush=True)
    t_all = time.perf_counter()
    results = []
    fold_sorted = {}
    order_sigs = {}
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(_transductive_worker, task) for task in tasks]
        for fut in as_completed(futures):
            try:
                result = fut.result()
            except Exception as exc:
                raise SystemExit(str(exc)) from exc
            idx = result.pop("idx")
            fold = result["fold"]
            ordered = np.sort(idx)
            previous = fold_sorted.get(fold)
            if previous is None:
                fold_sorted[fold] = ordered
            elif not np.array_equal(previous, ordered):
                raise SystemExit(
                    f"{result['file']}: test_idx set differs from another run of fold {fold}. Stopping."
                )
            signature = (
                int(idx[0]),
                int(idx[len(idx) // 2]),
                int(idx[-1]),
                int(np.bitwise_xor.reduce(idx.view(np.uint64))),
            )
            order_sigs.setdefault(signature, []).append(result["file"])
            results.append(result)
            print(
                f"[transductive] {result['file']} run_id={result['run_id']} {result['seconds']:.2f}s "
                f"prob|sum-1|={result['prob_dev']:.3e}",
                flush=True,
            )
    same = all(np.array_equal(fold_sorted[1], fold_sorted[fold]) for fold in fold_sorted)
    modal = max(order_sigs, key=lambda key: len(order_sigs[key]))
    order_mismatch = sorted(name for sig, names in order_sigs.items() if sig != modal for name in names)
    print(f"[check] all transductive folds share one test_idx set: {same}", flush=True)
    print(
        f"[check] test_idx row order differs for {len(order_mismatch)} files "
        f"(same pair set; each file keeps its own row order): {order_mismatch}",
        flush=True,
    )
    print(f"[transductive] all runs in {time.perf_counter() - t_all:.1f}s", flush=True)
    by_method = {method: [] for method, _directory in TRANSDUCTIVE_SPECS}
    cache = {}
    prob_devs = []
    n_mismatch = 0
    for result in results:
        by_method[result["method"]].append(result)
        prob_devs.append((result["file"], result["prob_dev"]))
        n_mismatch += result["n_mismatch"]
        if result["lo_reps"] is not None:
            cache[(result["method"], result["seed"])] = (result["lo_reps"], result["hi_reps"])
    store = {}
    mono = {}
    for method, _directory in TRANSDUCTIVE_SPECS:
        runs = sorted(by_method[method], key=lambda item: (item["fold"], item["seed"]))
        if len(runs) != 15:
            raise SystemExit(f"{method}: expected 15 finished runs, found {len(runs)}")
        for run in runs:
            if run["mono_review"] is not None:
                mono[run["file"]] = run["mono_review"]
        store[method] = _stack_runs(runs)
    for method in ("Exp-1", "Exp-5"):
        for seed in SEEDS:
            if (method, seed) not in cache:
                raise SystemExit(f"missing fold-1 thresholds for pooled diagnostic: {method} s{seed}")
    return store, cache, same, prob_devs, n_mismatch, order_mismatch, mono


def _drug_masks(job, in_a: np.ndarray):
    if job["part"] == "S1":
        cal = in_a[job["positions"]]
        ev = ~cal
        n_mixed = 0
    else:
        pos1, pos2 = job["positions"]
        cal = in_a[pos1] & in_a[pos2]
        ev = (~in_a)[pos1] & (~in_a)[pos2]
        n_mixed = int(pos1.size - int(cal.sum()) - int(ev.sum()))
    return cal, ev, n_mixed


def _attach_drug_sums(sums):
    n_level = len(LEVELS)
    sums["drug_miss"] = np.zeros(n_level)
    sums["drug_fa"] = np.zeros(n_level)
    sums["drug_review"] = np.zeros((n_level, n_level))
    sums["exceed_drug_miss"] = np.zeros(n_level)
    sums["exceed_drug_fa"] = np.zeros(n_level)
    sums["n_s_units"] = 0.0
    sums["n_ns_units"] = 0.0
    return sums


def _add_drug(sums, drug_miss, drug_fa, drug_review, exceed_miss, exceed_fa, n_s, n_ns):
    sums["drug_miss"] += drug_miss
    sums["drug_fa"] += drug_fa
    sums["drug_review"] += drug_review
    sums["exceed_drug_miss"] += exceed_miss
    sums["exceed_drug_fa"] += exceed_fa
    sums["n_s_units"] += n_s
    sums["n_ns_units"] += n_ns


def _inductive_worker(payload):
    job, repeats, cache_lo, cache_hi = payload
    t_file = time.perf_counter()
    pair_sums = _empty_sums()
    is_s1 = job["part"] == "S1"
    crc_sums = _attach_drug_sums(_empty_sums()) if is_s1 else None
    pooled_sums = _attach_drug_sums(_empty_sums()) if cache_lo is not None and is_s1 else (
        _empty_sums() if cache_lo is not None else None
    )
    n_cal = n_eval = n_mixed = 0
    n_u = job["n_u"]
    n_half = n_u // 2
    r = job["r"]
    y = job["y"]
    argmax = job["argmax"]
    drug = job["positions"] if is_s1 else None
    store_mono = is_s1 and job["method"] == "Exp-1" and job["seed"] == 42
    mono_pair = np.empty((repeats, len(LEVELS), len(LEVELS))) if store_mono else None
    mono_crc = np.empty((repeats, len(LEVELS), len(LEVELS))) if store_mono else None
    mono_drug = np.empty((repeats, len(LEVELS), len(LEVELS))) if store_mono else None
    for rep in range(repeats):
        perm = split_rng(job["run_id"], rep).permutation(n_u)
        in_a = np.zeros(n_u, dtype=bool)
        in_a[perm[:n_half]] = True
        cal, ev, mixed = _drug_masks(job, in_a)
        n_cal_rep = int(cal.sum())
        n_eval_rep = int(ev.sum())
        if n_cal_rep == 0 or n_eval_rep == 0:
            raise RuntimeError(f"{job['path'].name} rep {rep}: empty calibration or evaluation half")
        lo, hi = fit_thresholds(r[cal], y[cal])
        metrics = _accumulate_eval(r[ev], y[ev], argmax[ev], lo, hi, pair_sums)
        if store_mono:
            mono_pair[rep] = metrics[:, :, 2]
        if is_s1:
            severe = y >= 3
            nonsevere = y <= 2
            lo_c, hi_c, n_s, n_ns = fit_crc(
                r[cal & severe], drug[cal & severe], r[cal & nonsevere], drug[cal & nonsevere], n_u
            )
            crc_metrics = _accumulate_eval(r[ev], y[ev], argmax[ev], lo_c, hi_c, crc_sums)
            d_miss, d_fa, d_review, e_miss, e_fa = drug_averaged_rates(
                r[ev], y[ev], drug[ev], lo_c, hi_c, n_u
            )
            _add_drug(crc_sums, d_miss, d_fa, d_review, e_miss, e_fa, n_s, n_ns)
            if store_mono:
                mono_crc[rep] = crc_metrics[:, :, 2]
                mono_drug[rep] = d_review
        if pooled_sums is not None:
            _accumulate_eval(r[ev], y[ev], argmax[ev], cache_lo[rep], cache_hi[rep], pooled_sums)
            if is_s1:
                d_miss, d_fa, d_review, e_miss, e_fa = drug_averaged_rates(
                    r[ev], y[ev], drug[ev], cache_lo[rep], cache_hi[rep], n_u
                )
                _add_drug(pooled_sums, d_miss, d_fa, d_review, e_miss, e_fa, 0, 0)
        n_cal += n_cal_rep
        n_eval += n_eval_rep
        n_mixed += mixed
    pooled_mean = None
    if pooled_sums is not None:
        if is_s1:
            pooled_sums.pop("n_s_units", None)
            pooled_sums.pop("n_ns_units", None)
        pooled_mean = _mean_sums(pooled_sums)
    return {
        "method": job["method"],
        "part": job["part"],
        "file": job["path"].name,
        "seed": job["seed"],
        "run_id": job["run_id"],
        "fold": job["fold"],
        "prob_dev": job["prob_dev"],
        "n_mismatch": job["n_mismatch"],
        "seconds": time.perf_counter() - t_file,
        "mean_cal": n_cal / repeats,
        "mean_eval": n_eval / repeats,
        "mean_mixed": n_mixed / repeats,
        "pair_level": _mean_sums(pair_sums),
        "crc": None if crc_sums is None else _mean_sums(crc_sums),
        "pooled": pooled_mean,
        "mono_pair": mono_pair,
        "mono_crc": mono_crc,
        "mono_drug": mono_drug,
    }


def run_inductive(jobs, cache, repeats: int):
    payloads = []
    for job in jobs:
        cache_lo = cache_hi = None
        if job["method"] in ("Exp-1", "Exp-5"):
            cache_key = (job["method"], job["seed"])
            if cache_key not in cache:
                raise SystemExit(
                    f"no catalogued-route thresholds for {job['method']} fold 1 seed {job['seed']}"
                )
            cache_lo, cache_hi = cache[cache_key]
            expected_shape = (repeats, len(LEVELS))
            if cache_lo.shape != expected_shape or cache_hi.shape != expected_shape:
                raise SystemExit(
                    f"pooled thresholds for {cache_key} have shapes {cache_lo.shape} and "
                    f"{cache_hi.shape}, expected {expected_shape}"
                )
        payloads.append((job, repeats, cache_lo, cache_hi))
    print(f"[inductive] {len(payloads)} runs on {N_WORKERS} threads", flush=True)
    finished = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(_inductive_worker, payload) for payload in payloads]
        for fut in as_completed(futures):
            try:
                result = fut.result()
            except Exception as exc:
                raise SystemExit(str(exc)) from exc
            finished.append(result)
            print(
                f"[inductive] {result['file']} run_id={result['run_id']} {result['seconds']:.2f}s "
                f"mean_cal={result['mean_cal']:.1f} mean_eval={result['mean_eval']:.1f} "
                f"mean_mixed={result['mean_mixed']:.1f}",
                flush=True,
            )
    grouped = {}
    counts = {}
    prob_devs = []
    n_mismatch = 0
    mono = {}
    for result in sorted(finished, key=lambda item: (item["method"], item["part"], item["seed"])):
        method = result["method"]
        part = result["part"]
        bucket = grouped.setdefault(method, {}).setdefault(part, {})
        pair_name = CAL_PAIR_S1 if part == "S1" else CAL_S2
        bucket.setdefault(pair_name, []).append(
            {
                "file": result["file"],
                "seed": result["seed"],
                "run_id": result["run_id"],
                "fold": result["fold"],
                **result["pair_level"],
            }
        )
        if result["crc"] is not None:
            bucket.setdefault(CAL_DRUG, []).append(
                {
                    "file": result["file"],
                    "seed": result["seed"],
                    "run_id": result["run_id"],
                    "fold": result["fold"],
                    **result["crc"],
                }
            )
        if result["pooled"] is not None:
            bucket.setdefault(CAL_POOLED, []).append(
                {
                    "file": result["file"],
                    "seed": result["seed"],
                    "run_id": result["run_id"],
                    "fold": result["fold"],
                    "threshold_source": f"{method}_f1_s{result['seed']}.npz",
                    **result["pooled"],
                }
            )
        count_slot = counts.setdefault(method, {}).setdefault(
            part, {"cal": [], "eval": [], "mixed": [], "seeds": [], "files": [], "n_s": [], "n_ns": []}
        )
        count_slot["cal"].append(result["mean_cal"])
        count_slot["eval"].append(result["mean_eval"])
        count_slot["mixed"].append(result["mean_mixed"])
        count_slot["seeds"].append(result["seed"])
        count_slot["files"].append(result["file"])
        if result["crc"] is not None:
            count_slot["n_s"].append(float(result["crc"]["n_s_units"]))
            count_slot["n_ns"].append(float(result["crc"]["n_ns_units"]))
        prob_devs.append((result["file"], result["prob_dev"]))
        n_mismatch += result["n_mismatch"]
        if result["mono_pair"] is not None:
            mono[f"{result['file']} pair-level"] = result["mono_pair"]
            mono[f"{result['file']} drug-level CRC"] = result["mono_crc"]
            mono[f"{result['file']} drug-averaged review"] = result["mono_drug"]

    stacked = {}
    for method, parts in grouped.items():
        stacked[method] = {}
        for part, cals in parts.items():
            stacked[method][part] = {}
            for cal, runs in cals.items():
                if len(runs) != 3:
                    raise SystemExit(f"{method} {part} {cal}: expected 3 seeds, found {len(runs)}")
                block = {
                    "files": [item["file"] for item in runs],
                    "seeds": [item["seed"] for item in runs],
                    "run_ids": [item["run_id"] for item in runs],
                    "folds": [item["fold"] for item in runs],
                    "threshold_source": [item.get("threshold_source") for item in runs],
                    "metrics": np.stack([item["metrics"] for item in runs]),
                    "exceed_miss": np.stack([item["exceed_miss"] for item in runs]),
                    "exceed_fa": np.stack([item["exceed_fa"] for item in runs]),
                    "lam_lo": np.stack([item["lam_lo"] for item in runs]),
                    "lam_hi": np.stack([item["lam_hi"] for item in runs]),
                    "reference": np.stack([item["reference"] for item in runs]),
                }
                for key in (
                    "drug_miss",
                    "drug_fa",
                    "drug_review",
                    "exceed_drug_miss",
                    "exceed_drug_fa",
                    "n_s_units",
                    "n_ns_units",
                ):
                    if key in runs[0]:
                        block[key] = np.stack([np.asarray(item[key], dtype=np.float64) for item in runs])
                stacked[method][part][cal] = block
    return stacked, counts, prob_devs, n_mismatch, mono


def _mean_sd(values: np.ndarray):
    return values.mean(axis=0), values.std(axis=0, ddof=SD_DDOF)


def _row(base, alpha, beta, metric, mean, sd, n_runs):
    record = dict(base)
    record.update(
        alpha=alpha,
        beta=beta,
        metric=metric,
        mean=float(mean),
        sd=float(sd),
        n_runs=int(n_runs),
    )
    return record


def rows_from_block(base, block, *, with_reference: bool):
    rows = []
    n_runs = int(block["metrics"].shape[0])
    mean, sd = _mean_sd(block["metrics"])
    exceed_m, exceed_m_sd = _mean_sd(block["exceed_miss"])
    exceed_f, exceed_f_sd = _mean_sd(block["exceed_fa"])
    lam_lo, lam_lo_sd = _mean_sd(block["lam_lo"])
    lam_hi, lam_hi_sd = _mean_sd(block["lam_hi"])
    for i, alpha in enumerate(ALPHAS):
        for j, beta in enumerate(BETAS):
            for k, name in enumerate(METRIC_NAMES):
                rows.append(_row(base, alpha, beta, name, mean[i, j, k], sd[i, j, k], n_runs))
            extras = (
                ("lam_lo", lam_lo[i], lam_lo_sd[i]),
                ("lam_hi", lam_hi[j], lam_hi_sd[j]),
                ("exceed_miss", exceed_m[i, j], exceed_m_sd[i, j]),
                ("exceed_false_alert", exceed_f[i, j], exceed_f_sd[i, j]),
            )
            for name, mu, sg in extras:
                rows.append(_row(base, alpha, beta, name, mu, sg, n_runs))
    if "drug_miss" in block:
        drug_miss, drug_miss_sd = _mean_sd(block["drug_miss"])
        drug_fa, drug_fa_sd = _mean_sd(block["drug_fa"])
        drug_review, drug_review_sd = _mean_sd(block["drug_review"])
        exceed_drug_m, exceed_drug_m_sd = _mean_sd(block["exceed_drug_miss"])
        exceed_drug_f, exceed_drug_f_sd = _mean_sd(block["exceed_drug_fa"])
        for i, alpha in enumerate(ALPHAS):
            for j, beta in enumerate(BETAS):
                drug_rows = (
                    ("drug_miss_sev", drug_miss[i], drug_miss_sd[i]),
                    ("drug_false_alert", drug_fa[j], drug_fa_sd[j]),
                    ("drug_review_frac", drug_review[i, j], drug_review_sd[i, j]),
                    ("exceed_drug_miss", exceed_drug_m[i], exceed_drug_m_sd[i]),
                    ("exceed_drug_fa", exceed_drug_f[j], exceed_drug_f_sd[j]),
                )
                for name, mu, sg in drug_rows:
                    rows.append(_row(base, alpha, beta, name, mu, sg, n_runs))
        if "n_s_units" in block:
            n_s, n_s_sd = _mean_sd(block["n_s_units"])
            n_ns, n_ns_sd = _mean_sd(block["n_ns_units"])
            for alpha in ALPHAS:
                for beta in BETAS:
                    rows.append(_row(base, alpha, beta, "n_s_units", n_s, n_s_sd, n_runs))
                    rows.append(_row(base, alpha, beta, "n_ns_units", n_ns, n_ns_sd, n_runs))
    if with_reference:
        ref_base = dict(base)
        if "calibration" in ref_base:
            ref_base["calibration"] = CAL_REF
        ref_mean, ref_sd = _mean_sd(block["reference"])
        for k, name in enumerate(REF_NAMES):
            rows.append(_row(ref_base, np.nan, np.nan, name, ref_mean[k], ref_sd[k], n_runs))
    return rows


def assemble_tables(names, trans, inductive):
    if len(trans) != 11:
        raise SystemExit(f"expected 11 transductive methods, found {len(trans)}")
    rows_t = []
    for method, _directory in TRANSDUCTIVE_SPECS:
        block = trans[method]
        if block["metrics"].shape[0] != 15:
            raise SystemExit(f"{method}: expected 15 runs, found {block['metrics'].shape[0]}")
        if not np.isfinite(block["metrics"]).all():
            raise SystemExit(f"{method}: non-finite transductive metric")
        rows_t.extend(
            rows_from_block(
                {"method": method, "display_name": display_name(names, method)},
                block,
                with_reference=True,
            )
        )
    rows_i = []
    for method, _directory in INDUCTIVE_SPECS:
        if method not in inductive:
            raise SystemExit(f"missing inductive method {method}")
        for part in ("S1", "S2"):
            primary_name = CAL_DRUG if part == "S1" else CAL_S2
            primary = inductive[method][part][primary_name]
            if primary["metrics"].shape[0] != 3:
                raise SystemExit(f"{method} {part}: expected 3 inductive runs")
            if not np.isfinite(primary["metrics"]).all():
                raise SystemExit(f"{method} {part}: non-finite inductive metric")
            base = {
                "method": method,
                "display_name": display_name(names, method),
                "partition": part,
                "calibration": primary_name,
            }
            rows_i.extend(rows_from_block(base, primary, with_reference=True))
            if part == "S1":
                ablation = inductive[method][part][CAL_PAIR_S1]
                ablation_base = dict(base)
                ablation_base["calibration"] = CAL_PAIR_S1
                if np.max(np.abs(ablation["reference"] - primary["reference"])) > 1e-12:
                    raise SystemExit(f"{method} S1: pair-level and drug-level references differ")
                rows_i.extend(rows_from_block(ablation_base, ablation, with_reference=False))
            if method in ("Exp-1", "Exp-5"):
                pooled = inductive[method][part][CAL_POOLED]
                if np.max(np.abs(pooled["reference"] - primary["reference"])) > 1e-12:
                    raise SystemExit(f"{method} {part}: pooled and primary eval references differ")
                pooled_base = dict(base)
                pooled_base["calibration"] = CAL_POOLED
                rows_i.extend(rows_from_block(pooled_base, pooled, with_reference=False))
    df_t = pd.DataFrame(rows_t)
    df_i = pd.DataFrame(rows_i)
    return df_t, df_i


def cell(df: pd.DataFrame, metric: str, alpha, beta, **eq):
    mask = (df["metric"] == metric).to_numpy()
    alpha_v = df["alpha"].to_numpy(dtype=float)
    beta_v = df["beta"].to_numpy(dtype=float)
    if alpha is None:
        mask &= np.isnan(alpha_v)
    else:
        mask &= np.isfinite(alpha_v) & np.isclose(alpha_v, float(alpha))
    if beta is None:
        mask &= np.isnan(beta_v)
    else:
        mask &= np.isfinite(beta_v) & np.isclose(beta_v, float(beta))
    for key, value in eq.items():
        mask &= (df[key] == value).to_numpy()
    idx = np.flatnonzero(mask)
    if idx.size != 1:
        raise RuntimeError(f"lookup {metric} alpha={alpha} beta={beta} {eq}: found {idx.size} rows")
    row = df.iloc[int(idx[0])]
    return float(row["mean"]), float(row["sd"])


def pm(mean, sd, digits=4) -> str:
    if not np.isfinite(mean):
        return "NA"
    if not np.isfinite(sd):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def md_table(headers, body) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" if i == 0 else "---:" for i in range(len(headers))) + " |",
    ]
    # first column is left-aligned; rewrite the separator properly
    sep = ["---"] + ["---:"] * (len(headers) - 1)
    lines[1] = "| " + " | ".join(sep) + " |"
    for row in body:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def monotonicity_violations(df: pd.DataFrame, group_cols: list[str], metric: str = "review_frac"):
    work = df.loc[df["metric"] == metric].copy()
    work = work[np.isfinite(work["alpha"].to_numpy(dtype=float)) & np.isfinite(work["beta"].to_numpy(dtype=float))]
    violations = []
    if work.empty:
        return violations
    tolerance = 1e-12
    for key, sub in work.groupby(group_cols, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        grid = np.empty((len(ALPHAS), len(BETAS)))
        for i, alpha in enumerate(ALPHAS):
            for j, beta in enumerate(BETAS):
                hit = sub[np.isclose(sub["alpha"], alpha) & np.isclose(sub["beta"], beta)]
                if len(hit) != 1:
                    raise RuntimeError(f"monotonicity lookup {key} {metric} alpha={alpha} beta={beta}: {len(hit)}")
                grid[i, j] = float(hit["mean"].iloc[0])
        for j, beta in enumerate(BETAS):
            for i in range(len(ALPHAS) - 1):
                if grid[i + 1, j] > grid[i, j] + tolerance:
                    record = {col: key[jcol] for jcol, col in enumerate(group_cols)}
                    record.update(
                        metric=metric,
                        direction="alpha",
                        beta=float(beta),
                        alpha_lo=float(ALPHAS[i]),
                        alpha_hi=float(ALPHAS[i + 1]),
                        review_lo=float(grid[i, j]),
                        review_hi=float(grid[i + 1, j]),
                        delta=float(grid[i + 1, j] - grid[i, j]),
                    )
                    violations.append(record)
        for i, alpha in enumerate(ALPHAS):
            for j in range(len(BETAS) - 1):
                if grid[i, j + 1] > grid[i, j] + tolerance:
                    record = {col: key[jcol] for jcol, col in enumerate(group_cols)}
                    record.update(
                        metric=metric,
                        direction="beta",
                        alpha=float(alpha),
                        beta_lo=float(BETAS[j]),
                        beta_hi=float(BETAS[j + 1]),
                        review_lo=float(grid[i, j]),
                        review_hi=float(grid[i, j + 1]),
                        delta=float(grid[i, j + 1] - grid[i, j]),
                    )
                    violations.append(record)
    return violations


def assert_split_monotone(grids: dict) -> int:
    """Each saved (run, split) review grid must be exactly non-increasing in both budgets."""
    violations = 0
    for name, review in grids.items():
        if review is None:
            continue
        bad = 0
        for grid in review:
            if np.any(grid[1:] > grid[:-1] + 1e-12) or np.any(grid[:, 1:] > grid[:, :-1] + 1e-12):
                bad += 1
        violations += bad
        print(f"[mono-split] {name}: {bad} / {len(review)} splits increased", flush=True)
    if violations:
        raise SystemExit(f"per-split review monotonicity failed on {violations} splits")
    return violations


def block_json(block):
    payload = {
        "files": block["files"],
        "run_ids": [int(x) for x in block["run_ids"]],
        "seeds": [int(x) for x in block["seeds"]],
        "folds": [int(x) for x in block["folds"]],
        "metrics": {
            name: block["metrics"][:, :, :, k] for k, name in enumerate(METRIC_NAMES)
        },
        "lam_lo": block["lam_lo"],
        "lam_hi": block["lam_hi"],
        "exceed_miss": block["exceed_miss"],
        "exceed_false_alert": block["exceed_fa"],
        "reference": {name: block["reference"][:, k] for k, name in enumerate(REF_NAMES)},
    }
    if "threshold_source" in block:
        payload["threshold_source"] = block["threshold_source"]
    return payload


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            return None
        return number
    if value is None:
        return None
    return value


def build_raw(names, trans, inductive, counts, meta):
    raw = {
        "config": {
            "master_seed": MASTER_SEED,
            "R": meta["repeats"],
            "sd_ddof": SD_DDOF,
            "alphas": list(ALPHAS),
            "betas": list(BETAS),
            "metric_names": list(METRIC_NAMES),
            "metric_index": "metrics[run][alpha_index][beta_index]",
            "transductive_run_id": "method_index * 15 + (fold - 1) * 3 + seed_index",
            "inductive_run_id": "1000 + method_index * 6 + seed_index * 2 + partition_index",
            "seed_index": {str(seed): index for seed, index in SEED_INDEX.items()},
            "method_order_transductive": [method for method, _ in TRANSDUCTIVE_SPECS],
            "method_order_inductive": [method for method, _ in INDUCTIVE_SPECS],
            "calibration_note": (
                "Transductive calibration is the first floor(n/2) pairs of a permutation. "
                "Inductive A is the first floor(|U|/2) drugs of a permutation of U; "
                "B is the complement. Pooled thresholds come from transductive fold 1 "
                "with the same seed, repeat-aligned, and are applied to the inductive "
                "evaluation half. Pooled calibration is a diagnostic and is not a "
                "route-conditional guarantee."
            ),
        },
        "self_checks": meta["self_checks"],
        "transductive": {},
        "inductive": {},
    }
    for method, _directory in TRANSDUCTIVE_SPECS:
        raw["transductive"][method] = {
            "display_name": display_name(names, method),
            **block_json(trans[method]),
        }
    for method, _directory in INDUCTIVE_SPECS:
        raw["inductive"][method] = {"display_name": display_name(names, method), "partitions": {}}
        for part in ("S1", "S2"):
            raw["inductive"][method]["partitions"][part] = {
                cal: block_json(inductive[method][part][cal]) for cal in inductive[method][part]
            }
            count = counts[method][part]
            raw["inductive"][method]["partitions"][part]["split_counts_mean_over_R"] = {
                "seeds": count["seeds"],
                "files": count["files"],
                "cal": count["cal"],
                "eval": count["eval"],
                "mixed_dropped": count["mixed"],
            }
    return raw


def build_markdown(df_t, df_i, meta) -> str:
    order = []
    for method in df_t["method"].unique():
        review, _sd = cell(df_t, "review_frac", 0.01, 0.01, method=method)
        order.append((review, method))
    order.sort()
    methods = [method for _review, method in order]

    def grab(df, method, metric, alpha, beta, **eq):
        return pm(*cell(df, metric, alpha, beta, method=method, **eq))

    operating = [
        "miss_sev",
        "false_alert",
        "review_frac",
        "passive_frac",
        "alert_frac",
        "auto_acc",
    ]
    lines = [
        "# Conformal pharmacist-referral policy",
        "",
        "Generated by `graphtree_ddi.analysis.conformal_referral`. No model is trained. "
        f"Each run is split R = {meta['repeats']} times "
        f"(master seed {MASTER_SEED}; generator seed = {MASTER_SEED} + run_id × 1000 + rep). "
        "Per run, metrics are the mean over those splits. The tables report the mean and "
        f"SD across runs (ddof = {SD_DDOF}, the same convention as Table 4). "
        "Transductive n = 15 (folds 1–5 × seeds 42/123/2024). Inductive n = 3 seeds.",
        "",
        "Score r = p(Tier 3) + p(Tier 4). Pair-level lam_lo is the "
        "floor(alpha × (n_severe + 1))-th smallest severe score (or −∞ if that index is < 1) "
        "and lam_hi is the ceil((1 − beta) × (n_nonsevere + 1))-th smallest non-severe score "
        "(or +∞ if that index exceeds the calibration count). The decision is ALERT if "
        "r > lam_hi, otherwise PASSIVE if r < lam_lo, otherwise REVIEW. REVIEW is exactly "
        "lam_lo ≤ r ≤ lam_hi and is empty when the thresholds cross. "
        "PASSIVE ⊆ {r < lam_lo} and ALERT ⊆ {r > lam_hi}, so "
        "P(r < lam_lo | Y ≥ 3) ≤ alpha and P(r > lam_hi | Y ≤ 2) ≤ beta still hold. "
        "`exceed_miss` / `exceed_false_alert` are the fraction of (run, split) evaluations "
        "whose realized pair-level rate is strictly above the budget.",
        "",
        "`review_frac_sevonly` is P(REVIEW | Y ≥ 1), not P(REVIEW | Y ≥ 3) "
        "(`review_frac_sev` is the latter). `severe_sens` = 1 − miss_sev. "
        "Reference policies are scored on the same evaluation halves and do not use the "
        "conformal thresholds: argmax treats a predicted tier ≥ 3 as an alert; the fixed "
        "rule alerts when r ≥ 0.5.",
        "",
        "Held-out drugs U are split into halves A (calibration) and B (evaluation). "
        "For S1, each pair has one uncatalogued drug, and that drug is the exchangeable "
        "unit of drug-level conformal risk control (Angelopoulos, Bates, Fisch, Lei and "
        "Schuster, ICLR 2024). A severe calibration drug with n_k severe pairs gives each "
        "of those pairs weight 1/n_k. The budget is b = alpha × (n_s + 1) − 1, and lam_lo "
        "is the next distinct severe score after the largest cumulative weight that is "
        "still ≤ b (or −∞ if b < 0, +∞ if every score is inside the budget). lam_hi is "
        "the smallest distinct non-severe score whose strictly heavier tail has total "
        "weight ≤ beta × (n_ns + 1) − 1. The guaranteed quantities are the drug-averaged "
        "severe-miss and false-alert rates on evaluation drugs. "
        "'pair-level (held-out drugs)' refits ordinary Mondrian thresholds on the same "
        "S1 calibration pairs; it is a design ablation, not the primary uncatalogued policy. "
        "S2 keeps pair-level Mondrian calibration on pairs with both drugs in A, evaluated "
        "on pairs with both drugs in B. Pooled (catalogued-pair) calibration copies "
        "transductive fold-1 thresholds onto the inductive evaluation half and does not "
        "inherit either guarantee.",
        "",
        "## Self-checks",
        "",
        f"- Order-statistic integer arithmetic and the decision/metric hand example passed. "
        f"Float floor/ceil disagreements on a stride-17 grid of n: {meta['float_disagreements']}.",
        f"- Synthetic exchangeable check (Dirichlet probabilities, labels drawn from them, "
        f"n = 200,000, R = 200, {meta['synthetic_sec']:.1f}s): "
        f"{'PASS' if meta['synthetic_pass'] else 'FAIL'} on all 16 (alpha, beta) points. "
        "Criterion: mean severe-miss ≤ alpha + 3 SE and mean false alert ≤ beta + 3 SE "
        "(SE = sample SD / sqrt(R)).",
        f"- Smallest severe-miss slack (alpha + 3 SE − mean): {meta['min_miss_slack']:.6f}. "
        f"Smallest false-alert slack: {meta['min_fa_slack']:.6f}.",
        f"- Unit-weight drug-level CRC matched pair-level order statistics on "
        f"{meta['crc_equiv_cases']} random samples, including ties.",
        f"- Clustered drug-level CRC (1,500 lognormal groups, mean size about 150, "
        f"R = 200, {meta['crc_cluster_sec']:.1f}s): "
        f"{'PASS' if meta['crc_cluster_pass'] else 'FAIL'} on all 16 grid points for the "
        "drug-averaged severe-miss and false-alert rates. "
        f"Smallest drug-miss slack: {meta['crc_min_miss_slack']:.6f}; "
        f"smallest drug-false-alert slack: {meta['crc_min_fa_slack']:.6f}.",
        f"- y_prob rows were checked on all {meta['n_prob_files']} prediction files. "
        f"Max |row sum − 1| = {meta['prob_max']:.3e} ({meta['prob_worst']}).",
        f"- y_pred versus argmax(y_prob) mismatches across all files: {meta['n_mismatch']}.",
        f"- All transductive folds share one sealed test_idx set: {meta['fold_same']}. "
        f"Row order differs for {len(meta['order_mismatch'])} files "
        f"({', '.join(meta['order_mismatch']) if meta['order_mismatch'] else 'none'}); "
        "labels were checked against y.npy[test_idx] in each file's own order.",
        f"- Cross-run mean review fraction is non-increasing in alpha (beta fixed) and in "
        f"beta (alpha fixed), tolerance 1e-12, for pair-level review and, where defined, "
        f"drug-averaged review: {len(meta['violations'])} violation(s). "
        f"Per-split checks on saved runs: {meta['split_mono_violations']} increases.",
        "",
    ]
    if meta["violations"]:
        lines.append("Monotonicity assertion failed. Increases larger than 1e-12:")
        lines.append("")
        lines.extend(
            md_table(
                ["Group", "beta", "alpha", "review (smaller alpha)", "review (larger alpha)", "delta"],
                [
                    [
                        ", ".join(f"{key}={item[key]}" for key in item if key in (
                            "method", "partition", "calibration"
                        )),
                        f"{item.get('beta', item.get('beta_lo', float('nan'))):.3f}",
                        (
                            f"{item['alpha_lo']:.3f} → {item['alpha_hi']:.3f}"
                            if item.get("direction") == "alpha"
                            else f"beta {item.get('beta_lo', float('nan')):.3f} → {item.get('beta_hi', float('nan')):.3f}"
                        ),
                        f"{item['review_lo']:.4f}",
                        f"{item['review_hi']:.4f}",
                        f"{item['delta']:.4f}",
                    ]
                    for item in meta["violations"]
                ],
            )
        )
        lines.append("")
    else:
        lines.append("Cross-run and per-split review fractions did not increase with either budget (tolerance 1e-12).")
        lines.append("")

    lines.extend(
        [
            "## Transductive results at alpha = beta = 0.01",
            "",
            "Sorted by review fraction (low to high). Severe-miss is P(PASSIVE | Y ≥ 3). "
            "False alert is P(ALERT | Y ≤ 2). Auto accuracy is argmax accuracy on PASSIVE or ALERT pairs.",
            "",
        ]
    )
    body = []
    for method in methods:
        name = df_t.loc[df_t["method"] == method, "display_name"].iloc[0]
        body.append([name] + [grab(df_t, method, metric, 0.01, 0.01) for metric in operating])
    lines.extend(
        md_table(
            ["Method", "Severe-miss", "False alert", "Review", "Passive", "Alert", "Auto accuracy"],
            body,
        )
    )
    lines.extend(["", "Thresholds and budget exceedance at the same operating point. "
                  "lam_lo depends only on alpha; lam_hi depends only on beta.", ""])
    body = []
    for method in methods:
        name = df_t.loc[df_t["method"] == method, "display_name"].iloc[0]
        body.append(
            [
                name,
                grab(df_t, method, "lam_lo", 0.01, 0.01),
                grab(df_t, method, "lam_hi", 0.01, 0.01),
                grab(df_t, method, "exceed_miss", 0.01, 0.01),
                grab(df_t, method, "exceed_false_alert", 0.01, 0.01),
                grab(df_t, method, "review_frac_sev", 0.01, 0.01),
                grab(df_t, method, "review_frac_tier0", 0.01, 0.01),
                grab(df_t, method, "review_frac_sevonly", 0.01, 0.01),
                grab(df_t, method, "severe_sens", 0.01, 0.01),
            ]
        )
    lines.extend(
        md_table(
            [
                "Method",
                "lam_lo",
                "lam_hi",
                "P(miss > alpha)",
                "P(false alert > beta)",
                "Review | severe",
                "Review | tier 0",
                "Review | Y ≥ 1",
                "Severe sensitivity",
            ],
            body,
        )
    )

    lines.extend(["", "## GraphTree-DDI across alpha at beta = 0.01", ""])
    body = []
    for alpha in ALPHAS:
        body.append(
            [f"{alpha:.3f}"]
            + [grab(df_t, "Exp-5", metric, alpha, 0.01) for metric in operating]
            + [
                grab(df_t, "Exp-5", "lam_lo", alpha, 0.01),
                grab(df_t, "Exp-5", "lam_hi", alpha, 0.01),
                grab(df_t, "Exp-5", "exceed_miss", alpha, 0.01),
                grab(df_t, "Exp-5", "exceed_false_alert", alpha, 0.01),
            ]
        )
    lines.extend(
        md_table(
            [
                "alpha",
                "Severe-miss",
                "False alert",
                "Review",
                "Passive",
                "Alert",
                "Auto accuracy",
                "lam_lo",
                "lam_hi",
                "P(miss > alpha)",
                "P(false alert > beta)",
            ],
            body,
        )
    )

    lines.extend(["", "## GraphTree-DDI across beta at alpha = 0.01", ""])
    body = []
    for beta in BETAS:
        body.append(
            [f"{beta:.3f}"]
            + [grab(df_t, "Exp-5", metric, 0.01, beta) for metric in operating]
            + [
                grab(df_t, "Exp-5", "lam_lo", 0.01, beta),
                grab(df_t, "Exp-5", "lam_hi", 0.01, beta),
                grab(df_t, "Exp-5", "exceed_miss", 0.01, beta),
                grab(df_t, "Exp-5", "exceed_false_alert", 0.01, beta),
            ]
        )
    lines.extend(
        md_table(
            [
                "beta",
                "Severe-miss",
                "False alert",
                "Review",
                "Passive",
                "Alert",
                "Auto accuracy",
                "lam_lo",
                "lam_hi",
                "P(miss > alpha)",
                "P(false alert > beta)",
            ],
            body,
        )
    )

    lines.extend(
        [
            "",
            "## Reference policies on the transductive evaluation halves",
            "",
            "Same run order as the alpha = beta = 0.01 table.",
            "",
        ]
    )
    body = []
    for method in methods:
        name = df_t.loc[df_t["method"] == method, "display_name"].iloc[0]
        body.append(
            [
                name,
                grab(df_t, method, "ref_argmax_miss", None, None),
                grab(df_t, method, "ref_argmax_false_alert", None, None),
                grab(df_t, method, "ref_thr05_miss", None, None),
                grab(df_t, method, "ref_thr05_false_alert", None, None),
            ]
        )
    lines.extend(
        md_table(
            ["Method", "Argmax severe-miss", "Argmax false alert", "r ≥ 0.5 severe-miss", "r ≥ 0.5 false alert"],
            body,
        )
    )

    s1_metrics = [
        "drug_miss_sev",
        "drug_false_alert",
        "miss_sev",
        "false_alert",
        "review_frac",
        "drug_review_frac",
        "passive_frac",
        "alert_frac",
        "auto_acc",
    ]
    settings = ((0.01, 0.01), (0.05, 0.01), (0.05, 0.05))
    for alpha, beta in settings:
        lines.extend(
            [
                "",
                f"## S1 drug-level conformal risk control at alpha = {alpha:.3f}, beta = {beta:.3f}",
                "",
                "Drug-averaged severe-miss is the mean, over evaluation drugs with at least one "
                "severe pair, of that drug's severe-miss fraction. It is the quantity controlled "
                "by drug-level CRC. Pair-level rates use the same thresholds but weight pairs equally.",
                "",
            ]
        )
        body = []
        for method, _directory in INDUCTIVE_SPECS:
            name = df_i.loc[df_i["method"] == method, "display_name"].iloc[0]
            row = [name] + [
                grab(df_i, method, metric, alpha, beta, partition="S1", calibration=CAL_DRUG)
                for metric in s1_metrics
            ]
            row.append(grab(df_i, method, "n_s_units", alpha, beta, partition="S1", calibration=CAL_DRUG))
            row.append(grab(df_i, method, "n_ns_units", alpha, beta, partition="S1", calibration=CAL_DRUG))
            body.append(row)
        lines.extend(
            md_table(
                [
                    "Method",
                    "Drug-avg miss",
                    "Drug-avg false alert",
                    "Pair miss",
                    "Pair false alert",
                    "Pair review",
                    "Drug-avg review",
                    "Passive",
                    "Alert",
                    "Auto accuracy",
                    "n_s",
                    "n_ns",
                ],
                body,
            )
        )

    lines.extend(
        [
            "",
            "## S1 pooled catalogued-pair thresholds",
            "",
            "Thresholds fit on the calibration half of transductive fold 1 (same seed) and applied "
            "to the S1 evaluation drugs. Drug-averaged and pair-level rates are both shown. "
            "This diagnostic does not inherit the drug-level guarantee.",
            "",
        ]
    )
    for alpha, beta in settings:
        lines.append(f"alpha = {alpha:.3f}, beta = {beta:.3f}")
        lines.append("")
        body = []
        for method in ("Exp-1", "Exp-5"):
            name = df_i.loc[df_i["method"] == method, "display_name"].iloc[0]
            body.append(
                [name]
                + [
                    grab(df_i, method, metric, alpha, beta, partition="S1", calibration=CAL_POOLED)
                    for metric in s1_metrics
                ]
            )
        lines.extend(
            md_table(
                [
                    "Method",
                    "Drug-avg miss",
                    "Drug-avg false alert",
                    "Pair miss",
                    "Pair false alert",
                    "Pair review",
                    "Drug-avg review",
                    "Passive",
                    "Alert",
                    "Auto accuracy",
                ],
                body,
            )
        )
        lines.append("")

    lines.extend(["", "## S1 pair-level ablation on held-out drugs", ""])
    lines.append(
        "Ordinary Mondrian thresholds on the same S1 calibration pairs, with the new decision rule. "
        "Pairs that share a held-out drug are not exchangeable, so this is not the primary guarantee."
    )
    lines.append("")
    for alpha, beta in settings:
        lines.append(f"alpha = {alpha:.3f}, beta = {beta:.3f}")
        lines.append("")
        body = []
        for method, _directory in INDUCTIVE_SPECS:
            name = df_i.loc[df_i["method"] == method, "display_name"].iloc[0]
            body.append(
                [name]
                + [
                    grab(df_i, method, metric, alpha, beta, partition="S1", calibration=CAL_PAIR_S1)
                    for metric in operating
                ]
            )
        lines.extend(
            md_table(
                ["Method", "Severe-miss", "False alert", "Review", "Passive", "Alert", "Auto accuracy"],
                body,
            )
        )
        lines.append("")

    lines.extend(
        [
            "## S2 pair-level calibration",
            "",
            "Both drugs are uncatalogued. Calibration pairs have both drugs in A and evaluation "
            "pairs have both drugs in B. Mixed pairs are dropped. There is no drug-level CRC for S2.",
            "",
        ]
    )
    for alpha, beta in settings:
        lines.append(f"alpha = {alpha:.3f}, beta = {beta:.3f}")
        lines.append("")
        body = []
        for method, _directory in INDUCTIVE_SPECS:
            name = df_i.loc[df_i["method"] == method, "display_name"].iloc[0]
            body.append(
                [name]
                + [
                    grab(df_i, method, metric, alpha, beta, partition="S2", calibration=CAL_S2)
                    for metric in operating
                ]
            )
        lines.extend(
            md_table(
                ["Method", "Severe-miss", "False alert", "Review", "Passive", "Alert", "Auto accuracy"],
                body,
            )
        )
        lines.append("")
    lines.extend(["Pooled catalogued-pair thresholds on the S2 evaluation half (Exp-1 and Exp-5):", ""])
    for alpha, beta in ((0.01, 0.01), (0.05, 0.05)):
        lines.append(f"alpha = {alpha:.3f}, beta = {beta:.3f}")
        lines.append("")
        body = []
        for method in ("Exp-1", "Exp-5"):
            name = df_i.loc[df_i["method"] == method, "display_name"].iloc[0]
            body.append(
                [name]
                + [
                    grab(df_i, method, metric, alpha, beta, partition="S2", calibration=CAL_POOLED)
                    for metric in operating
                ]
            )
        lines.extend(
            md_table(
                ["Method", "Severe-miss", "False alert", "Review", "Passive", "Alert", "Auto accuracy"],
                body,
            )
        )
        lines.append("")

    lines.extend(["## Inductive reference policies", ""])
    lines.append(
        "Computed on the evaluation half of the primary calibration (drug-level CRC for S1, "
        "pair-level for S2). Pooled and the S1 pair-level ablation use that same half."
    )
    lines.append("")
    body = []
    for method, _directory in INDUCTIVE_SPECS:
        name = df_i.loc[df_i["method"] == method, "display_name"].iloc[0]
        for part in ("S1", "S2"):
            body.append(
                [
                    name,
                    part,
                    grab(df_i, method, "ref_argmax_miss", None, None, partition=part, calibration=CAL_REF),
                    grab(df_i, method, "ref_argmax_false_alert", None, None, partition=part, calibration=CAL_REF),
                    grab(df_i, method, "ref_thr05_miss", None, None, partition=part, calibration=CAL_REF),
                    grab(df_i, method, "ref_thr05_false_alert", None, None, partition=part, calibration=CAL_REF),
                ]
            )
    lines.extend(
        md_table(
            ["Method", "Partition", "Argmax severe-miss", "Argmax false alert", "r ≥ 0.5 severe-miss", "r ≥ 0.5 false alert"],
            body,
        )
    )

    lines.extend(
        [
            "",
            "## Drug-grouped split sizes",
            "",
            "Mean over R splits within a seed, then mean ± SD across the 3 seeds. "
            "S2 mixed pairs (one drug in A and one in B) are dropped and are in neither half. "
            "n_s and n_ns are the numbers of calibration drugs with at least one severe or "
            "non-severe S1 pair, the units of drug-level CRC.",
            "",
        ]
    )
    body = []
    for method, _directory in INDUCTIVE_SPECS:
        name = df_i.loc[df_i["method"] == method, "display_name"].iloc[0]
        for part in ("S1", "S2"):
            count = meta["counts"][method][part]
            n_s = count.get("n_s") or []
            n_ns = count.get("n_ns") or []
            body.append(
                [
                    name,
                    part,
                    pm(float(np.mean(count["cal"])), float(np.std(count["cal"], ddof=SD_DDOF)), digits=1),
                    pm(float(np.mean(count["eval"])), float(np.std(count["eval"], ddof=SD_DDOF)), digits=1),
                    pm(float(np.mean(count["mixed"])), float(np.std(count["mixed"], ddof=SD_DDOF)), digits=1),
                    pm(float(np.mean(n_s)), float(np.std(n_s, ddof=SD_DDOF)), digits=1) if n_s else "—",
                    pm(float(np.mean(n_ns)), float(np.std(n_ns, ddof=SD_DDOF)), digits=1) if n_ns else "—",
                ]
            )
    lines.extend(
        md_table(
            ["Method", "Partition", "Calibration pairs", "Evaluation pairs", "Mixed dropped", "n_s", "n_ns"],
            body,
        )
    )
    lines.extend(
        [
            "",
            f"Runtime: computation {meta['compute_sec']:.1f}s, total {meta['total_sec']:.1f}s "
            f"(includes the synthetic check, {meta['synthetic_sec']:.1f}s).",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def apply_figure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavusans",
            "axes.linewidth": 0.6,
        }
    )


def curve(df, metric, beta, **eq):
    ys, ss = [], []
    for alpha in ALPHAS:
        mu, sd = cell(df, metric, alpha, beta, **eq)
        ys.append(mu)
        ss.append(sd)
    return np.asarray(ys, dtype=float), np.asarray(ss, dtype=float)


def legend_below(ax, ncol: int, anchor_y: float):
    handles, labels = ax.get_legend_handles_labels()
    if labels and labels[0] == "$y = x$":
        handles = handles[1:] + handles[:1]
        labels = labels[1:] + labels[:1]
    legend = ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, anchor_y),
        ncol=ncol,
        frameon=False,
        fontsize=7,
        handlelength=1.8,
        columnspacing=0.8,
        labelspacing=0.25,
        borderaxespad=0.0,
        handletextpad=0.4,
    )
    legend.set_clip_on(False)
    return legend


def audit_text(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_bb = fig.bbox
    texts = []
    for text in fig.findobj(match=Text):
        if text.get_visible() is not True:
            continue
        raw = text.get_text()
        if raw is None or str(raw).strip() == "":
            continue
        box = text.get_window_extent(renderer=renderer)
        if box.width <= 0 and box.height <= 0:
            continue
        texts.append((str(raw), box))
    outside = []
    for raw, box in texts:
        if (
            box.x0 < fig_bb.x0 - 1.0
            or box.y0 < fig_bb.y0 - 1.0
            or box.x1 > fig_bb.x1 + 1.0
            or box.y1 > fig_bb.y1 + 1.0
        ):
            outside.append(
                (
                    raw,
                    tuple(round(v, 2) for v in (box.x0, box.y0, box.x1, box.y1)),
                    tuple(round(v, 2) for v in (fig_bb.x0, fig_bb.y0, fig_bb.x1, fig_bb.y1)),
                )
            )
    overlaps = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            left, right = texts[i][1], texts[j][1]
            ix0 = max(left.x0, right.x0)
            iy0 = max(left.y0, right.y0)
            ix1 = min(left.x1, right.x1)
            iy1 = min(left.y1, right.y1)
            if ix1 - ix0 > 0.8 and iy1 - iy0 > 0.8:
                overlaps.append((texts[i][0], texts[j][0], round(ix1 - ix0, 2), round(iy1 - iy0, 2)))
    return outside, overlaps


def _log_yerr(ys, ss):
    ys = np.asarray(ys, dtype=float)
    ss = np.asarray(ss, dtype=float)
    lower = np.minimum(ss, np.where(ys > 0, ys * 0.999, 0.0))
    return np.vstack([lower, ss])


def _plain_log_formatter():
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    def _format(value, _pos):
        if value <= 0:
            return ""
        if value >= 1:
            return f"{value:.0f}" if abs(value - round(value)) < 1e-8 else f"{value:g}"
        return f"{value:g}"

    return FuncFormatter(_format), LogLocator(base=10), NullFormatter()


def _panel_tag(ax, tag, title):
    ax.text(
        0.0, 1.03, tag, transform=ax.transAxes, fontweight="bold", fontsize=7,
        ha="left", va="bottom", clip_on=False,
    )
    ax.text(
        0.16, 1.03, title, transform=ax.transAxes, fontsize=7,
        ha="left", va="bottom", clip_on=False,
    )


def make_figure(df_t: pd.DataFrame, df_i: pd.DataFrame):
    apply_figure_style()
    fig, axes = plt.subplots(1, 3, figsize=(16.0 / 2.54, 6.0 / 2.54))
    xs = np.asarray(ALPHAS, dtype=float)
    tick_labels = ["0.005", "0.01", "0.02", "0.05"]
    styles = (
        (OKABE["blue"], "o", "-"),
        (OKABE["orange"], "s", "--"),
        (OKABE["green"], "D", "-."),
        (OKABE["sky"], "^", ":"),
        (OKABE["purple"], "v", (0, (3, 1, 1, 1))),
        (OKABE["vermillion"], "P", (0, (1, 1))),
    )

    ax = axes[0]
    panel_a = (
        ("Exp-5", "GraphTree-DDI"),
        ("Exp-3", "R-GCN (hybrid KG)"),
        ("DistMult", "DistMult"),
        ("MLP", "MLP"),
        ("DrugPrior", "Drug-prior (B3)"),
        ("Exp-1", "XGBoost"),
    )
    for (method, label), (color, marker, style) in zip(panel_a, styles):
        ys, err = curve(df_t, "review_frac", 0.01, method=method)
        ax.errorbar(
            xs, np.where(ys > 0, ys, np.nan), yerr=_log_yerr(ys, err),
            color=color, marker=marker, linestyle=style, linewidth=0.9, markersize=3.0,
            capsize=1.4, elinewidth=0.6, capthick=0.6, label=label,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(0.0038, 0.07)
    positive = []
    for method, _label in panel_a:
        ys, err = curve(df_t, "review_frac", 0.01, method=method)
        keep = ys > 0
        positive.extend((ys[keep] - err[keep]).tolist())
        positive.extend((ys[keep] + err[keep]).tolist())
    positive = np.asarray([v for v in positive if v > 0], dtype=float)
    ax.set_ylim(float(positive.min()) / 1.8, 0.45)
    formatter, locator, minor = _plain_log_formatter()
    ax.yaxis.set_major_locator(locator)
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_minor_formatter(minor)
    ax.set_xlabel(r"Severe-miss budget $\alpha$")
    ax.set_ylabel("Pharmacist review fraction")
    _panel_tag(ax, "(a)", "Catalogued route")

    ax = axes[1]
    panel_b = (
        ("Exp-1", "XGBoost (routed)"),
        ("MLP", "MLP"),
        ("DDIMDL", "DDIMDL-style"),
        ("DistMult", "DistMult"),
        ("Exp-5", "GraphTree-DDI"),
    )
    for (method, label), (color, marker, style) in zip(panel_b, styles):
        ys, err = curve(df_i, "review_frac", 0.01, method=method, partition="S1", calibration=CAL_DRUG)
        ax.errorbar(
            xs, ys, yerr=err, color=color, marker=marker, linestyle=style,
            linewidth=0.9, markersize=3.0, capsize=1.4, elinewidth=0.6, capthick=0.6, label=label,
        )
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(0.0038, 0.07)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"Severe-miss budget $\alpha$")
    ax.set_ylabel("Pharmacist review fraction")
    _panel_tag(ax, "(b)", "Uncatalogued S1")

    ax = axes[2]
    panel_c = (
        (df_t, {"method": "Exp-5"}, "miss_sev", "GraphTree-DDI catalogued", styles[0]),
        (df_i, {"method": "Exp-1", "partition": "S1", "calibration": CAL_DRUG}, "drug_miss_sev", "XGBoost S1 CRC", styles[5]),
        (df_i, {"method": "Exp-1", "partition": "S1", "calibration": CAL_POOLED}, "drug_miss_sev", "XGBoost S1 pooled", styles[1]),
        (df_i, {"method": "Exp-5", "partition": "S1", "calibration": CAL_POOLED}, "drug_miss_sev", "GraphTree-DDI S1 pooled", styles[2]),
    )
    collected = []
    for frame, eq, metric, label, (color, marker, style) in panel_c:
        ys, err = curve(frame, metric, 0.01, **eq)
        collected.append(ys)
        collected.append(ys + err)
        ax.errorbar(
            xs, np.where(ys > 0, ys, np.nan), yerr=_log_yerr(ys, err),
            color=color, marker=marker, linestyle=style, linewidth=0.9, markersize=3.0,
            capsize=1.4, elinewidth=0.6, capthick=0.6, label=label, zorder=3,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(0.0038, 0.07)
    ax.set_ylim(0.0025, 2.2)
    formatter, locator, minor = _plain_log_formatter()
    ax.yaxis.set_major_locator(locator)
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_minor_formatter(minor)
    ax.autoscale(False)
    ax.plot([0.003, 2.2], [0.003, 2.2], color="#4d4d4d", linestyle="--", linewidth=0.7, zorder=1, label=r"$y = x$")
    ax.set_xlabel(r"Severe-miss budget $\alpha$")
    ax.set_ylabel("Realized severe-miss rate")
    _panel_tag(ax, "(c)", "Calibration")

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(length=2.2, width=0.5, pad=1.2)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.78, bottom=0.48, wspace=0.50)
    legend_below(axes[0], ncol=2, anchor_y=-0.42)
    legend_below(axes[1], ncol=2, anchor_y=-0.42)
    legend_below(axes[2], ncol=1, anchor_y=-0.42)

    FIG.mkdir(parents=True, exist_ok=True)
    png = FIG / "fig_conformal.png"
    pdf = FIG / "fig_conformal.pdf"
    fig.savefig(png, dpi=DPI)
    fig.savefig(pdf)
    outside, overlaps = audit_text(fig)
    renderer = fig.canvas.get_renderer()
    for axis in axes:
        legend = axis.get_legend()
        if legend is None:
            continue
        legend_box = legend.get_window_extent(renderer)
        axes_box = axis.get_window_extent(renderer)
        overlap = min(legend_box.y1, axes_box.y1) - max(legend_box.y0, axes_box.y0)
        if overlap > 1.5 and legend_box.x1 > axes_box.x0 and legend_box.x0 < axes_box.x1:
            overlaps.append(("legend", axis.get_title() or "panel", round(overlap, 2), 0))
    width_cm, height_cm = (size * 2.54 for size in fig.get_size_inches())
    plt.close(fig)
    print(f"[figure] {width_cm:.2f} x {height_cm:.2f} cm; text outside={len(outside)} overlaps={len(overlaps)}", flush=True)
    for item in outside:
        print(f"[figure] outside: {item}", flush=True)
    for item in overlaps:
        print(f"[figure] overlap: {item}", flush=True)
    print(f"[write] {png}", flush=True)
    print(f"[write] {pdf}", flush=True)
    return outside, overlaps

def _bind_paths(args) -> None:
    global TAB, FIG, Y_PATH, PAIRS_PATH, NAMES_PATH, PRED_DIRS
    out = Path(args.out_dir)
    TAB = out
    FIG = out
    NAMES_PATH = (
        Path(args.method_names)
        if str(args.method_names).strip()
        else (_REPO_ROOT / "resources" / "method_names_11.json")
    )
    if args.figure_only:
        return
    if not args.pred_dir:
        raise SystemExit("--pred_dir is required unless --figure-only")
    PRED_DIRS = [_resolve_pred_dir(p) for p in args.pred_dir]
    data_dir = Path(args.data_dir) if str(args.data_dir).strip() else (_REPO_ROOT / "data" / "processed" / "v2")
    Y_PATH = Path(args.labels) if str(args.labels).strip() else (data_dir / "y.npy")
    PAIRS_PATH = Path(args.pairs) if str(args.pairs).strip() else (data_dir / "pairs.csv")
    missing = [str(p) for p in (Y_PATH, PAIRS_PATH, NAMES_PATH) if not p.is_file()]
    if missing:
        raise SystemExit(f"missing input files: {missing}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Mondrian conformal referral analysis on saved predictions.")
    parser.add_argument(
        "--pred_dir", nargs="+", default=None,
        help="result directories with prediction *.npz (a predictions/ subdirectory is used when present)",
    )
    parser.add_argument("--out_dir", required=True, help="directory for conformal tables and fig_conformal")
    parser.add_argument(
        "--data_dir", default="",
        help="processed tables containing y.npy and pairs.csv (default: data/processed/v2)",
    )
    parser.add_argument("--labels", default="", help="y.npy (default: <data_dir>/y.npy)")
    parser.add_argument("--pairs", default="", help="pairs.csv (default: <data_dir>/pairs.csv)")
    parser.add_argument(
        "--method_names", default="",
        help="JSON display-name map (default: resources/method_names_11.json)",
    )
    parser.add_argument("--R", type=int, default=None, help="Calibration/evaluation repeats (default 100).")
    parser.add_argument("--quick", action="store_true", help="Use R=5 unless --R is also set.")
    parser.add_argument("--figure-only", action="store_true", help="Redraw fig_conformal from the saved tables.")
    return parser.parse_args(argv)


def resolve_repeats(args) -> int:
    if args.R is not None:
        if args.quick:
            print("[config] --R overrides --quick", flush=True)
        if args.R < 1:
            raise SystemExit("--R must be positive")
        return int(args.R)
    if args.quick:
        return 5
    return 100


def main(argv=None) -> int:
    _utf8_stdio()
    args = parse_args(argv)
    _bind_paths(args)
    if args.figure_only:
        df_t = pd.read_csv(TAB / "conformal_transductive.csv")
        df_i = pd.read_csv(TAB / "conformal_inductive.csv")
        outside, overlaps = make_figure(df_t, df_i)
        return 0 if not outside and not overlaps else 2

    t_all = time.perf_counter()
    run_unit_checks()
    # The disagreement count is printed inside the unit check; recompute the flag for the markdown.
    disagree = 0
    for n in range(0, 200_000, 17):
        for level, num, den in LEVELS:
            exact = (num * (n + 1)) // den
            approx = int(np.floor(level * (n + 1) + 1e-12))
            k2_exact = ((den - num) * (n + 1) + den - 1) // den
            k2_approx = int(np.ceil((1.0 - level) * (n + 1) - 1e-12))
            disagree += int(exact != approx) + int(k2_exact != k2_approx)
    syn_rows, syn_dt = run_synthetic()
    crc_rows, crc_dt = run_clustered_crc()
    repeats = resolve_repeats(args)
    print(f"[config] R={repeats} master_seed={MASTER_SEED} sd_ddof={SD_DDOF}", flush=True)
    names = json.loads(NAMES_PATH.read_text(encoding="utf-8"))
    for method, _directory in tuple(TRANSDUCTIVE_SPECS) + tuple(INDUCTIVE_SPECS):
        display_name(names, method)

    y_all, d1, d2, split_by_seed = load_pair_universe()
    jobs = prepare_inductive(y_all, d1, d2, split_by_seed)
    print(f"[check] inductive index assertions passed for {len(jobs)} files", flush=True)

    t_comp = time.perf_counter()
    trans, cache, fold_same, prob_t, mis_t, order_mismatch, mono_trans = run_transductive(y_all, repeats)
    inductive, counts, prob_i, mis_i, mono_ind = run_inductive(jobs, cache, repeats)
    compute_sec = time.perf_counter() - t_comp
    split_mono_violations = assert_split_monotone({**mono_trans, **mono_ind})

    df_t, df_i = assemble_tables(names, trans, inductive)
    violations = []
    for metric in ("review_frac", "drug_review_frac"):
        violations.extend(monotonicity_violations(df_t, ["method"], metric))
        violations.extend(monotonicity_violations(df_i, ["method", "partition", "calibration"], metric))
    print(f"[mono] cross-run review increases: {len(violations)}", flush=True)
    prob_devs = prob_t + prob_i
    worst_name, worst_dev = max(prob_devs, key=lambda item: item[1])
    min_miss_slack = min(row["bound_miss"] - row["mean_miss"] for row in syn_rows)
    min_fa_slack = min(row["bound_false_alert"] - row["mean_false_alert"] for row in syn_rows)
    total_sec = time.perf_counter() - t_all
    meta = {
        "repeats": repeats,
        "synthetic_sec": syn_dt,
        "synthetic_pass": all(row["ok_miss"] and row["ok_false_alert"] for row in syn_rows),
        "min_miss_slack": min_miss_slack,
        "min_fa_slack": min_fa_slack,
        "crc_equiv_cases": 21,
        "crc_cluster_sec": crc_dt,
        "crc_cluster_pass": all(row["ok"] for row in crc_rows),
        "crc_min_miss_slack": min(row["alpha"] + 3.0 * row["se_drug_miss"] - row["mean_drug_miss"] for row in crc_rows),
        "crc_min_fa_slack": min(
            row["beta"] + 3.0 * row["se_drug_false_alert"] - row["mean_drug_false_alert"] for row in crc_rows
        ),
        "float_disagreements": disagree,
        "n_prob_files": len(prob_devs),
        "prob_max": worst_dev,
        "prob_worst": worst_name,
        "n_mismatch": int(mis_t + mis_i),
        "fold_same": bool(fold_same),
        "order_mismatch": order_mismatch,
        "violations": violations,
        "split_mono_violations": split_mono_violations,
        "counts": counts,
        "compute_sec": compute_sec,
        "total_sec": total_sec,
    }

    self_checks = {
        "synthetic": syn_rows,
        "synthetic_seconds": syn_dt,
        "float_floor_ceil_disagreements": disagree,
        "prob_row_sum_max_abs": worst_dev,
        "prob_row_sum_worst_file": worst_name,
        "prob_row_sum_abs_by_file": {name: dev for name, dev in prob_devs},
        "y_pred_argmax_mismatches": int(mis_t + mis_i),
        "transductive_folds_share_test_idx": bool(fold_same),
        "monotonicity_violations": violations,
        "split_monotonicity_violations": split_mono_violations,
        "crc_cluster": crc_rows,
    }
    raw = build_raw(
        names,
        trans,
        inductive,
        counts,
        {"repeats": repeats, "self_checks": self_checks},
    )
    raw["runtime_sec"] = {"compute": compute_sec, "total": total_sec, "synthetic": syn_dt}

    TAB.mkdir(parents=True, exist_ok=True)
    df_t.to_csv(TAB / "conformal_transductive.csv", index=False, encoding="utf-8", float_format="%.16g")
    df_i.to_csv(TAB / "conformal_inductive.csv", index=False, encoding="utf-8", float_format="%.16g")
    (TAB / "conformal_raw.json").write_text(
        json.dumps(to_jsonable(raw), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    markdown = build_markdown(df_t, df_i, meta)
    (TAB / "conformal_summary.md").write_text(markdown, encoding="utf-8")
    if violations or split_mono_violations:
        print(markdown)
        raise SystemExit(
            f"monotonicity assertion failed: {len(violations)} cross-run increases, "
            f"{split_mono_violations} per-split increases"
        )
    print(markdown)
    print(f"[write] {TAB / 'conformal_transductive.csv'}", flush=True)
    print(f"[write] {TAB / 'conformal_inductive.csv'}", flush=True)
    print(f"[write] {TAB / 'conformal_summary.md'}", flush=True)
    print(f"[write] {TAB / 'conformal_raw.json'}", flush=True)

    outside, overlaps = make_figure(df_t, df_i)
    print(f"[done] total {time.perf_counter() - t_all:.1f}s", flush=True)
    if outside or overlaps:
        print("[figure] text audit failed", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
