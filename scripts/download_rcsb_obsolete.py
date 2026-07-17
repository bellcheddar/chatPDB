#!/usr/bin/env python3
"""
download_rcsb_obsolete.py — pull the obsolete/superseded PDB entry mapping into
data/corpus/obsolete/ for RAG ingestion and SFT generation.

New finding from round 4 research, not originally in scope: confirmed live that
data/corpus/rcsb/pdb_entries_enriched.csv's `status_code` field is uniformly "REL" -- obsolete
entries were never in our corpus pull to begin with (RCSB's search API returns only released
entries by default). Experts care about this ("use 6XYZ, 1ABC was superseded") and chatPDB
currently has no way to warn a user off a stale ID or point them to its replacement.

Access confirmed live 2026-07-17 (round 4 research):
- Bulk list: `https://data.rcsb.org/rest/v1/holdings/removed/entry_ids` -- JSON array of all
  obsolete IDs (~4,968 confirmed).
- Per-entry detail: `https://data.rcsb.org/rest/v1/holdings/removed/{id}` -- includes
  `rcsb_repository_holdings_removed.id_codes_replaced_by` (empty list if no replacement was ever
  issued for that ID).

Usage:
    python scripts/download_rcsb_obsolete.py
    python scripts/download_rcsb_obsolete.py --limit 200   # smoke test
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import pandas as pd
import requests

OUT = Path("data/corpus/obsolete")
BULK_URL = "https://data.rcsb.org/rest/v1/holdings/removed/entry_ids"
DETAIL_URL = "https://data.rcsb.org/rest/v1/holdings/removed"
HEADERS = {"User-Agent": "chatPDB/1.0 (protein-structure-rag; marc@marcdeller.com)", "Connection": "close"}

_lock = Lock()
_stats = {"ok": 0, "failed": 0, "done": 0}


def _fetch_one(obsolete_id: str) -> dict | None:
    try:
        r = requests.get(f"{DETAIL_URL}/{obsolete_id}", headers=HEADERS, timeout=(10, 20))
        if r.status_code != 200:
            with _lock:
                _stats["failed"] += 1
            return None
        holdings = r.json().get("rcsb_repository_holdings_removed", {})
        with _lock:
            _stats["ok"] += 1
        return {
            "obsolete_id": obsolete_id,
            "removed_date": holdings.get("deposit_date") or holdings.get("audit_authors"),
            "superseded_by": ",".join(holdings.get("id_codes_replaced_by", []) or []),
            "title": holdings.get("title"),
        }
    except requests.RequestException:
        with _lock:
            _stats["failed"] += 1
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print("[1/2] Fetching bulk obsolete-entry ID list ...")
    r = requests.get(BULK_URL, headers=HEADERS, timeout=(10, 30))
    r.raise_for_status()
    obsolete_ids = r.json()
    print(f"  {len(obsolete_ids):,} obsolete entries")
    if args.limit:
        obsolete_ids = obsolete_ids[:args.limit]

    print("\n[2/2] Fetching per-entry replacement detail ...")
    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_fetch_one, oid): oid for oid in obsolete_ids}
        for fut in as_completed(futures):
            result = fut.result()
            if result:
                rows.append(result)
            with _lock:
                _stats["done"] += 1
                done = _stats["done"]
            if done % 1000 == 0:
                print(f"  {done:,}/{len(obsolete_ids):,} (ok {_stats['ok']:,}, failed {_stats['failed']:,})",
                      flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "obsolete_entries.csv", index=False)
    has_replacement = (df["superseded_by"] != "").sum() if not df.empty else 0
    print(f"\nDone in {(time.time()-t0)/60:.1f}m: {len(df):,} rows, {has_replacement:,} with a "
          f"documented replacement ID -> obsolete_entries.csv")


if __name__ == "__main__":
    main()
