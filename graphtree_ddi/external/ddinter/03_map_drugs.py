"""Map DDInter drug names to DrugBank IDs (exact/alias/salt, then detail-page scrape)."""

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

import csv
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import requests

from graphtree_ddi.external.ddinter._common import (
    DDINTER1_DETAIL,
    DDINTER2_DETAIL,
    PROCESSED_DIR,
    RAW_DIR,
    STATS_DIR,
    USER_AGENT,
    configure_stdio,
    ensure_dirs,
    now_cst_iso,
    write_json,
)

HERE = Path(__file__).resolve().parent
CLEAN_CSV = HERE / "ddinter_pairs_clean.csv"
MAPPING_CSV = HERE / "drug_mapping.csv"
UNMATCHED_CSV = HERE / "unmatched_drugs.csv"
SCRAPE_CACHE_CSV = RAW_DIR / "detail_page_cache.csv"
STATS_JSON = STATS_DIR / "03_map_drugs.json"

ALIASES_CSV = PROCESSED_DIR / "drugbank_name_aliases.csv"
I18N_CSV = PROCESSED_DIR / "drugbank_name_i18n.csv"

DB_ID_RE = re.compile(r"go\.drugbank\.com/drugs/(DB\d+)", re.IGNORECASE)
SCRAPE_INTERVAL_SEC = 1.0
SCRAPE_BUDGET_SEC = 40 * 60
SCRAPE_ATTEMPTS = 3  # 1 try + 2 retries
REQUEST_TIMEOUT = 30

# Trailing salt / hydrate / counter-ion tokens. Applied only after NFKC+casefold
# tokenization; never stripped if the remaining stem is itself a salt token.
SALT_TOKENS = {
    "hydrochloride",
    "dihydrochloride",
    "monohydrochloride",
    "trihydrochloride",
    "hcl",
    "hydrobromide",
    "dihydrobromide",
    "hbr",
    "hydroiodide",
    "hi",
    "sodium",
    "potassium",
    "calcium",
    "magnesium",
    "zinc",
    "mesylate",
    "methanesulfonate",
    "methanesulphonate",
    "besylate",
    "tosylate",
    "sulfate",
    "sulphate",
    "disulfate",
    "phosphate",
    "diphosphate",
    "acetate",
    "diacetate",
    "citrate",
    "maleate",
    "fumarate",
    "tartrate",
    "bitartrate",
    "succinate",
    "lactate",
    "nitrate",
    "bromide",
    "chloride",
    "iodide",
    "hydrate",
    "monohydrate",
    "dihydrate",
    "trihydrate",
    "tetrahydrate",
    "hemihydrate",
    "anhydrous",
    "pamoate",
    "embonate",
    "napsylate",
    "esylate",
    "edisylate",
    "gluconate",
    "benzoate",
    "carbonate",
    "bicarbonate",
    "propionate",
    "valerate",
    "enanthate",
    "decanoate",
    "stearate",
    "oxalate",
    "malate",
    "aspartate",
    "besilate",
    "tosilate",
    "camsylate",
}

SALT_RULE_TEXT = (
    "仅在大小写不敏感精确匹配失败后，对规范化英文名的末尾盐/水合物/反离子单词做剥离："
    "逐个去掉末尾属于预定盐词表的 token（如 hydrochloride、sodium、mesylate、hydrate 等），"
    "要求剩余词干长度≥4 且词干本身不是盐词；不剥离词中段、不做模糊匹配、不启用编辑距离。"
)


def normalize_en(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = re.sub(r"[®™]", "", text)
    text = re.sub(r"[\(\)\[\]\{\},.;:]", " ", text)
    text = re.sub(r"[-_/]+", " ", text)
    return " ".join(text.split())


def strip_salt_suffix(normalized: str) -> str | None:
    tokens = normalized.split()
    changed = False
    while len(tokens) > 1 and tokens[-1] in SALT_TOKENS:
        tokens.pop()
        changed = True
    if not changed:
        return None
    stem = " ".join(tokens)
    if not stem or stem in SALT_TOKENS or len(stem) < 4:
        return None
    return stem


class NameIndex:
    def __init__(self) -> None:
        self.primary: dict[str, set[str]] = defaultdict(set)
        self.alias_raw: dict[str, set[str]] = defaultdict(set)
        self.alias_norm: dict[str, set[str]] = defaultdict(set)
        self.i18n: dict[str, set[str]] = defaultdict(set)
        self.id_to_primary: dict[str, str] = {}

    def add(self, bucket: dict[str, set[str]], key: str, dbid: str) -> None:
        if key:
            bucket[key].add(dbid)


def build_index() -> NameIndex:
    idx = NameIndex()
    aliases = pd.read_csv(ALIASES_CSV, dtype=str, encoding="utf-8-sig").fillna("")
    for row in aliases.itertuples(index=False):
        dbid = str(row.drugbank_id).strip()
        primary = str(row.primary_name).strip()
        alias = str(row.alias).strip()
        alias_norm = str(row.alias_normalized).strip() or normalize_en(alias)
        if not dbid:
            continue
        if primary:
            idx.id_to_primary[dbid] = primary
            idx.add(idx.primary, primary.casefold(), dbid)
            idx.add(idx.primary, normalize_en(primary), dbid)
        if alias:
            idx.add(idx.alias_raw, alias.casefold(), dbid)
            idx.add(idx.alias_raw, normalize_en(alias), dbid)
        if alias_norm:
            idx.add(idx.alias_norm, alias_norm, dbid)
    i18n = pd.read_csv(I18N_CSV, dtype=str, encoding="utf-8-sig").fillna("")
    for row in i18n.itertuples(index=False):
        dbid = str(row.drugbank_id).strip()
        if not dbid:
            continue
        for col in ("name_en", "name_cn"):
            val = str(getattr(row, col, "")).strip()
            if val:
                idx.add(idx.i18n, val.casefold(), dbid)
                idx.add(idx.i18n, normalize_en(val), dbid)
    return idx


def lookup(bucket: dict[str, set[str]], *keys: str) -> set[str]:
    found: set[str] = set()
    for key in keys:
        if key:
            found |= bucket.get(key, set())
    return found


def match_name(name: str, idx: NameIndex) -> tuple[set[str], str, str]:
    """Return (ids, method, confidence). Empty ids => unmatched for this name."""
    raw = (name or "").strip()
    if not raw:
        return set(), "unmatched", ""
    folded = raw.casefold()
    norm = normalize_en(raw)

    ids = lookup(idx.primary, folded, norm)
    if len(ids) == 1:
        return ids, "exact_primary", "1.00"
    if len(ids) > 1:
        return ids, "ambiguous_primary", "0.00"

    ids = lookup(idx.i18n, folded, norm)
    if len(ids) == 1:
        return ids, "exact_i18n", "1.00"
    if len(ids) > 1:
        return ids, "ambiguous_i18n", "0.00"

    ids = lookup(idx.alias_raw, folded, norm)
    if len(ids) == 1:
        return ids, "exact_alias", "0.95"
    if len(ids) > 1:
        return ids, "ambiguous_alias", "0.00"

    ids = lookup(idx.alias_norm, norm)
    if len(ids) == 1:
        return ids, "exact_alias_normalized", "0.90"
    if len(ids) > 1:
        return ids, "ambiguous_alias_normalized", "0.00"

    stem = strip_salt_suffix(norm)
    if stem:
        ids = lookup(idx.primary, stem)
        if len(ids) == 1:
            return ids, "salt_stripped_primary", "0.80"
        if len(ids) > 1:
            return ids, "ambiguous_salt_primary", "0.00"
        ids = lookup(idx.alias_raw, stem) | lookup(idx.alias_norm, stem)
        if len(ids) == 1:
            return ids, "salt_stripped_alias", "0.75"
        if len(ids) > 1:
            return ids, "ambiguous_salt_alias", "0.00"
    return set(), "unmatched", ""


def collect_ddinter_drugs(clean: pd.DataFrame) -> pd.DataFrame:
    records: dict[str, Counter] = defaultdict(Counter)
    for a, na, b, nb in zip(
        clean["DDInterID_A"], clean["Drug_A"], clean["DDInterID_B"], clean["Drug_B"]
    ):
        if a:
            records[str(a).strip()][str(na).strip()] += 1
        if b:
            records[str(b).strip()][str(nb).strip()] += 1
    rows = []
    for dd_id, names in records.items():
        canonical = names.most_common(1)[0][0] if names else ""
        alt = [n for n, _ in names.most_common() if n and n != canonical]
        rows.append(
            {
                "ddinter_id": dd_id,
                "drug_name": canonical,
                "alt_names": "|".join(alt),
            }
        )
    return pd.DataFrame(rows).sort_values("ddinter_id").reset_index(drop=True)


def name_match_row(row: pd.Series, idx: NameIndex) -> dict:
    names = [row["drug_name"]]
    if row["alt_names"]:
        names.extend(row["alt_names"].split("|"))
    names = [n for n in names if n]
    methods_hit: list[tuple[set[str], str, str]] = []
    for name in names:
        ids, method, conf = match_name(name, idx)
        if ids:
            methods_hit.append((ids, method, conf))
    if not methods_hit:
        return {
            "ddinter_id": row["ddinter_id"],
            "drug_name": row["drug_name"],
            "drugbank_id": "",
            "method": "unmatched",
            "confidence": "",
            "candidates": "",
        }
    union: set[str] = set()
    for ids, _, _ in methods_hit:
        union |= ids
    # pick the highest-priority successful method (list is already in name order;
    # re-rank by method priority)
    rank = {
        "exact_primary": 0,
        "exact_i18n": 1,
        "exact_alias": 2,
        "exact_alias_normalized": 3,
        "salt_stripped_primary": 4,
        "salt_stripped_alias": 5,
    }
    methods_hit.sort(key=lambda t: rank.get(t[1], 50))
    best_ids, best_method, best_conf = methods_hit[0]
    if len(union) == 1:
        dbid = next(iter(union))
        return {
            "ddinter_id": row["ddinter_id"],
            "drug_name": row["drug_name"],
            "drugbank_id": dbid,
            "method": best_method if len(best_ids) == 1 else methods_hit[0][1],
            "confidence": best_conf if len(union) == 1 else "1.00",
            "candidates": dbid,
        }
    return {
        "ddinter_id": row["ddinter_id"],
        "drug_name": row["drug_name"],
        "drugbank_id": "",
        "method": "ambiguous",
        "confidence": "",
        "candidates": "|".join(sorted(union)),
    }


def load_scrape_cache() -> dict[str, dict]:
    if not SCRAPE_CACHE_CSV.exists():
        return {}
    df = pd.read_csv(SCRAPE_CACHE_CSV, dtype=str, encoding="utf-8").fillna("")
    cache = {}
    for rec in df.to_dict("records"):
        key = str(rec.get("ddinter_id") or "").strip()
        if key:
            cache[key] = rec
    return cache


def append_scrape_cache(row: dict) -> None:
    exists = SCRAPE_CACHE_CSV.exists()
    with SCRAPE_CACHE_CSV.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "ddinter_id",
                "drugbank_id",
                "http_status",
                "url",
                "source_site",
                "fetched_at_utc8",
                "note",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def extract_drugbank_id(html: str) -> str:
    found = DB_ID_RE.findall(html or "")
    if not found:
        return ""
    hit = found[0].strip()
    digits = re.sub(r"\D", "", hit)
    return f"DB{digits}" if digits else ""


def fetch_detail(ddinter_id: str) -> dict:
    urls = [
        ("ddinter2", DDINTER2_DETAIL.format(ddinter_id=ddinter_id)),
        ("ddinter1", DDINTER1_DETAIL.format(ddinter_id=ddinter_id)),
    ]
    last_note = ""
    for site, url in urls:
        last_exc = None
        for attempt in range(1, SCRAPE_ATTEMPTS + 1):
            try:
                resp = requests.get(
                    url,
                    headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
                    timeout=REQUEST_TIMEOUT,
                )
                if resp.status_code == 200:
                    dbid = extract_drugbank_id(resp.text)
                    return {
                        "ddinter_id": ddinter_id,
                        "drugbank_id": dbid,
                        "http_status": str(resp.status_code),
                        "url": url,
                        "source_site": site,
                        "fetched_at_utc8": now_cst_iso(),
                        "note": "" if dbid else "page_ok_no_drugbank_link",
                    }
                last_exc = RuntimeError(f"HTTP {resp.status_code}")
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
            last_note = f"{site} attempt {attempt}: {last_exc}"
            if attempt < SCRAPE_ATTEMPTS:
                time.sleep(SCRAPE_INTERVAL_SEC)
        # try 1.0 site next
    return {
        "ddinter_id": ddinter_id,
        "drugbank_id": "",
        "http_status": "failed",
        "url": urls[0][1],
        "source_site": "both_failed",
        "fetched_at_utc8": now_cst_iso(),
        "note": last_note,
    }


def scrape_unresolved(rows: list[dict]) -> tuple[int, int, bool]:
    """Fill unmatched/ambiguous rows via detail pages. Returns (scraped, resolved, timed_out)."""
    cache = load_scrape_cache()
    need = [
        r
        for r in rows
        if r["method"] in {"unmatched", "ambiguous"} or not r["drugbank_id"]
    ]
    scraped = 0
    resolved = 0
    timed_out = False
    deadline = time.monotonic() + SCRAPE_BUDGET_SEC
    last_request_at = 0.0

    for row in need:
        dd_id = row["ddinter_id"]
        cached = cache.get(dd_id)
        cache_ok = bool(
            cached
            and (
                str(cached.get("drugbank_id") or "").strip()
                or str(cached.get("note") or "").startswith("page_ok")
            )
        )
        if not cache_ok:
            if time.monotonic() >= deadline:
                timed_out = True
                break
            wait = SCRAPE_INTERVAL_SEC - (time.monotonic() - last_request_at)
            if wait > 0:
                # If waiting would exceed budget, stop rather than overshoot much.
                if time.monotonic() + wait >= deadline:
                    timed_out = True
                    break
                time.sleep(wait)
            cached = fetch_detail(dd_id)
            last_request_at = time.monotonic()
            append_scrape_cache(cached)
            cache[dd_id] = cached
            scraped += 1
            print(
                f"  [scrape] {dd_id} -> {cached.get('drugbank_id') or cached.get('note') or cached.get('http_status')}"
            )
        dbid = str(cached.get("drugbank_id") or "").strip()
        if dbid:
            row["drugbank_id"] = dbid
            row["method"] = "detail_page"
            row["confidence"] = "0.99"
            row["candidates"] = dbid
            resolved += 1
        elif row["method"] == "ambiguous":
            row["method"] = "ambiguous"
        else:
            note = str(cached.get("note") or "")
            if "no_drugbank_link" in note:
                row["method"] = "detail_page_no_link"
            else:
                row["method"] = "detail_page_failed"
    return scraped, resolved, timed_out


def main() -> int:
    configure_stdio()
    ensure_dirs()
    if not CLEAN_CSV.exists():
        raise FileNotFoundError(CLEAN_CSV)

    clean = pd.read_csv(CLEAN_CSV, dtype=str, encoding="utf-8")
    drugs = collect_ddinter_drugs(clean)
    print(f"[03] unique DDInter drugs: {len(drugs):,}")
    print("[03] building DrugBank name index...")
    idx = build_index()
    print(
        f"[03] index sizes primary={len(idx.primary):,} "
        f"alias_raw={len(idx.alias_raw):,} alias_norm={len(idx.alias_norm):,} "
        f"i18n={len(idx.i18n):,}"
    )

    mapped = [name_match_row(row, idx) for row in drugs.to_dict("records")]
    n_name_ok = sum(1 for r in mapped if r["drugbank_id"])
    n_unmatched = sum(1 for r in mapped if r["method"] == "unmatched")
    n_ambiguous = sum(1 for r in mapped if r["method"] == "ambiguous")
    n_need_scrape = sum(1 for r in mapped if not r["drugbank_id"])
    print(
        f"[03] name-match unique-ID hits={n_name_ok:,} "
        f"unmatched={n_unmatched:,} ambiguous={n_ambiguous:,} "
        f"need_scrape={n_need_scrape:,}",
        flush=True,
    )

    scraped, scrape_resolved, timed_out = scrape_unresolved(mapped)
    remaining_unmatched = sum(1 for r in mapped if not r["drugbank_id"])
    print(
        f"[03] scrape new_requests={scraped} newly_resolved={scrape_resolved} "
        f"timed_out={timed_out} remaining_unmapped={remaining_unmatched}"
    )

    out_df = pd.DataFrame(mapped)
    out_public = out_df[
        ["ddinter_id", "drug_name", "drugbank_id", "method", "confidence"]
    ].rename(
        columns={
            "ddinter_id": "DDInterID",
            "drug_name": "Drug",
            "drugbank_id": "drugbank_id",
            "method": "method",
            "confidence": "confidence",
        }
    )
    out_public.to_csv(MAPPING_CSV, index=False, encoding="utf-8")
    unmatched_df = out_df[out_df["drugbank_id"].eq("")].copy()
    unmatched_df.to_csv(UNMATCHED_CSV, index=False, encoding="utf-8")

    method_counts = out_df["method"].value_counts().to_dict()
    n_drugs = int(len(out_df))
    n_mapped = int(out_df["drugbank_id"].ne("").sum())
    coverage = round(n_mapped / n_drugs * 100, 4) if n_drugs else 0.0
    stats = {
        "generated_at_utc8": now_cst_iso(),
        "n_ddinter_drugs": n_drugs,
        "n_mapped_to_drugbank": n_mapped,
        "n_unmapped": n_drugs - n_mapped,
        "drug_level_coverage_pct": coverage,
        "n_name_match_before_scrape": int(n_name_ok),
        "n_unmatched_before_scrape": int(n_unmatched),
        "n_ambiguous_before_scrape": int(n_ambiguous),
        "n_scrape_requests": int(scraped),
        "n_scrape_resolved": int(scrape_resolved),
        "scrape_timed_out": bool(timed_out),
        "method_counts": {str(k): int(v) for k, v in method_counts.items()},
        "salt_rule": SALT_RULE_TEXT,
        "mapping_csv": str(MAPPING_CSV),
    }
    write_json(STATS_JSON, stats)
    print(f"[03] coverage {n_mapped}/{n_drugs} = {coverage:.2f}%")
    print(f"[03] wrote {MAPPING_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
