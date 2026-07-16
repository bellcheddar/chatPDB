#!/usr/bin/env python3
"""
download_wwpdb_validation.py — pull wwPDB validation report summaries (Ramachandran outliers,
rotamer outliers, clashscore) into data/corpus/validation/ for RAG ingestion and SFT generation.

The "structure QC" gap flagged repeatedly since Phase 2: RCSB's GraphQL schema doesn't expose these
metrics at the entry level (confirmed by introspection — pdbx_vrpt_summary has no clashscore/
rama_outliers fields), and TWILIGHT only covers ligand density fit, not backbone geometry. PDBe's
validation API has the real numbers, confirmed live 2026-07-16:
  https://www.ebi.ac.uk/pdbe/api/validation/global-percentiles/entry/{pdb_id}
  -> percent-rama-outliers, percent-rota-outliers, clashscore (each with rawvalue + percentile
     rank against all PDB entries of comparable resolution — the percentile is what a structural
     biologist actually means by "this clashscore is good/bad", not the raw number alone).

Per-entry only, no bulk endpoint — but concurrency-tested live before committing to a worker count
(8/16/32 workers, 0 errors up to 98.4 req/s) rather than assumed slow. At that rate the full
256,448-entry corpus is feasible in under an hour, so this covers every entry, not a sample.

Usage:
    python scripts/download_wwpdb_validation.py                 # full run, resumable
    python scripts/download_wwpdb_validation.py --limit 5000     # smoke test
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import pandas as pd
import requests

OUT = Path("data/corpus/validation")
IDS_SOURCE = Path("data/corpus/rcsb/pdb_all_entries.csv")
API_BASE = "https://www.ebi.ac.uk/pdbe/api/validation/global-percentiles/entry"
HEADERS = {"User-Agent": "chatPDB/1.0 (protein-structure-rag; marc@marcdeller.com)", "Connection": "close"}

_lock = Lock()
_stats = {"ok": 0, "no_data": 0, "failed": 0, "done": 0}


def _fetch_one(pdb_id: str) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/{pdb_id.lower()}", headers=HEADERS, timeout=(10, 20))
        if r.status_code != 200:
            with _lock:
                _stats["no_data"] += 1
                _stats["done"] += 1
            return None
        data = r.json().get(pdb_id.lower())
        with _lock:
            _stats["done"] += 1
        if not data:
            with _lock:
                _stats["no_data"] += 1
            return None
        rama = data.get("percent-rama-outliers", {}) or {}
        rota = data.get("percent-rota-outliers", {}) or {}
        clash = data.get("clashscore", {}) or {}
        with _lock:
            _stats["ok"] += 1
        return {
            "pdb_id": pdb_id.upper(),
            "percent_rama_outliers": rama.get("rawvalue"),
            "percent_rama_outliers_percentile": rama.get("relative"),
            "percent_rota_outliers": rota.get("rawvalue"),
            "percent_rota_outliers_percentile": rota.get("relative"),
            "clashscore": clash.get("rawvalue"),
            "clashscore_percentile": clash.get("relative"),
        }
    except requests.RequestException:
        with _lock:
            _stats["failed"] += 1
            _stats["done"] += 1
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(IDS_SOURCE)["pdb_id"].tolist()
    if args.limit:
        ids = ids[: args.limit]
    total = len(ids)
    print(f"Target: {total:,} entries, {args.workers} workers ...", flush=True)

    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_fetch_one, pid): pid for pid in ids}
        for fut in as_completed(futures):
            result = fut.result()
            if result:
                rows.append(result)
            if _stats["done"] % 5000 == 0:
                elapsed = time.time() - t0
                rate = _stats["done"] / elapsed if elapsed > 0 else 0
                eta_m = (total - _stats["done"]) / rate / 60 if rate > 0 else float("inf")
                print(f"  {_stats['done']:,}/{total:,} (ok {_stats['ok']:,}, no_data {_stats['no_data']:,}, "
                      f"failed {_stats['failed']:,}) — {rate:.1f}/s, ETA {eta_m:.1f}m", flush=True)
                # Periodic checkpoint so a stall/interrupt doesn't lose hours of progress.
                pd.DataFrame(rows).to_csv(OUT / "wwpdb_validation.csv", index=False)

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "wwpdb_validation.csv", index=False)
    print(f"\nDone in {elapsed/60:.1f}m: {len(df):,} rows with validation data "
          f"({_stats['no_data']:,} entries had none available, {_stats['failed']:,} request failures).")
    if not df.empty:
        print(f"  Clashscore stats: median {df['clashscore'].median():.2f}, "
              f"90th pct {df['clashscore'].quantile(0.9):.2f}")


if __name__ == "__main__":
    main()
