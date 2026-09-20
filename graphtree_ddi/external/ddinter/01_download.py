"""Download 14 DDInter ATC-batch CSV files into raw/ and record checksums."""

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
import hashlib
import os
import time
from pathlib import Path

import requests

from graphtree_ddi.external.ddinter._common import (
    ATC_CODES,
    DDINTER1_CSV,
    DDINTER2_CSV,
    RAW_DIR,
    STATS_DIR,
    USER_AGENT,
    configure_stdio,
    ensure_dirs,
    now_cst_iso,
    write_json,
)

MANIFEST_CSV = RAW_DIR / "download_manifest.csv"
STATS_JSON = STATS_DIR / "01_download.json"
TIMEOUT_SEC = 120
MAX_ATTEMPTS = 3
FORCE = os.environ.get("DDINTER_FORCE_DOWNLOAD", "").strip() in {"1", "true", "True", "yes"}

MANIFEST_FIELDS = [
    "file",
    "atc_code",
    "url",
    "source_site",
    "fallback_used",
    "sha256",
    "nbytes",
    "n_rows",
    "n_data_rows",
    "downloaded_at_utc8",
    "status",
    "note",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_csv_rows(path: Path) -> tuple[int, int]:
    """Return (total_physical_rows_including_header, data_rows)."""
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        n = sum(1 for _ in reader)
    data_rows = max(n - 1, 0) if n else 0
    return n, data_rows


def looks_like_ddinter_csv(content: bytes) -> bool:
    head = content[:200].lstrip(b"\xef\xbb\xbf")
    return head.startswith(b"DDInterID_A") or b"DDInterID_A" in head[:80]


def http_get(url: str) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": USER_AGENT, "Accept": "text/csv,*/*"},
                timeout=TIMEOUT_SEC,
            )
            if resp.status_code == 200 and looks_like_ddinter_csv(resp.content):
                return resp
            last_exc = RuntimeError(
                f"HTTP {resp.status_code}, bytes={len(resp.content)}"
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
        if attempt < MAX_ATTEMPTS:
            time.sleep(2 * attempt)
    raise RuntimeError(f"failed {url}: {last_exc}")


def download_one(code: str) -> dict:
    filename = f"ddinter_downloads_code_{code}.csv"
    dest = RAW_DIR / filename
    downloaded_at = now_cst_iso()
    note = ""
    url_used = DDINTER2_CSV.format(code=code)
    source_site = "ddinter2"
    fallback_used = False

    if dest.exists() and dest.stat().st_size > 100 and not FORCE:
        n_rows, n_data = count_csv_rows(dest)
        return {
            "file": filename,
            "atc_code": code,
            "url": url_used,
            "source_site": "local_cache",
            "fallback_used": False,
            "sha256": sha256_file(dest),
            "nbytes": dest.stat().st_size,
            "n_rows": n_rows,
            "n_data_rows": n_data,
            "downloaded_at_utc8": downloaded_at,
            "status": "skipped_existing",
            "note": "file already present; checksum recomputed",
        }

    try:
        try:
            resp = http_get(url_used)
        except Exception as exc_v2:
            url_used = DDINTER1_CSV.format(code=code)
            source_site = "ddinter1"
            fallback_used = True
            note = f"ddinter2 failed ({exc_v2}); used 1.0 site"
            resp = http_get(url_used)
        dest.write_bytes(resp.content)
        n_rows, n_data = count_csv_rows(dest)
        return {
            "file": filename,
            "atc_code": code,
            "url": url_used,
            "source_site": source_site,
            "fallback_used": fallback_used,
            "sha256": sha256_file(dest),
            "nbytes": dest.stat().st_size,
            "n_rows": n_rows,
            "n_data_rows": n_data,
            "downloaded_at_utc8": downloaded_at,
            "status": "ok",
            "note": note,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "file": filename,
            "atc_code": code,
            "url": url_used,
            "source_site": source_site,
            "fallback_used": fallback_used,
            "sha256": "",
            "nbytes": dest.stat().st_size if dest.exists() else 0,
            "n_rows": 0,
            "n_data_rows": 0,
            "downloaded_at_utc8": downloaded_at,
            "status": "failed",
            "note": str(exc),
        }


def main() -> int:
    configure_stdio()
    ensure_dirs()
    rows = [download_one(code) for code in ATC_CODES]
    with MANIFEST_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    ok = [r for r in rows if r["status"] in {"ok", "skipped_existing"}]
    failed = [r for r in rows if r["status"] == "failed"]
    fallback = [r for r in ok if r["fallback_used"]]
    stats = {
        "generated_at_utc8": now_cst_iso(),
        "n_files_requested": len(ATC_CODES),
        "n_ok": len(ok),
        "n_failed": len(failed),
        "n_fallback_to_v1": len(fallback),
        "n_skipped_existing": sum(1 for r in rows if r["status"] == "skipped_existing"),
        "total_data_rows_sum": int(sum(int(r["n_data_rows"]) for r in ok)),
        "files": rows,
        "manifest_csv": str(MANIFEST_CSV),
    }
    write_json(STATS_JSON, stats)

    print(f"[01] wrote {MANIFEST_CSV}")
    print(f"[01] ok={len(ok)} failed={len(failed)} fallback_v1={len(fallback)}")
    for row in rows:
        print(
            f"  {row['atc_code']}: {row['status']} "
            f"rows={row['n_data_rows']} sha256={row['sha256'][:12] if row['sha256'] else '-'} "
            f"src={row['source_site']}"
        )
        if row["note"]:
            print(f"      note: {row['note']}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
