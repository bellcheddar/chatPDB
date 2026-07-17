#!/usr/bin/env python3
"""
download_pdbredo.py — pull PDB-REDO re-refinement metadata into data/corpus/pdbredo/ for RAG
ingestion and SFT generation.

PDB-REDO (pdb-redo.eu) automatically re-refines and re-builds X-ray structures; each entry's
`data.json` records the original deposited R-factor/R-free alongside PDB-REDO's own re-refined
values plus ~190 validation Z-scores. This is metadata-only, HTTP per-entry -- not the re-refined
coordinate files, which would rival the 353 GB mmCIF pool in size.

Access confirmed live 2026-07-17 (round 4 research): `https://pdb-redo.eu/db/{pdbid}/data.json`
(lowercase ID) works per entry, verified live for 6lu7/1cbs. Not every legacy/small entry has been
reprocessed by PDB-REDO (404/500 for some, e.g. very old NMR-adjacent entries) -- expected, handled
as a skip, not an error.

**Revision note (2026-07-17, same round):** originally attempted via rsync
(`rsync://rsync.pdb-redo.eu/pdb-redo/`, dry-run tested and confirmed correct), but the real
full-tree pull stalled -- an ESTABLISHED TCP connection with zero files written after 25+ minutes,
the same silent-hang shape this project has repeatedly hit with long-lived pooled connections
(previously `requests.Session()`, here rsync's own protocol enumerating a huge filtered directory
tree). Switched to HTTP per-entry fetches against chatPDB's own known 256,448 PDB IDs instead of
having rsync enumerate PDB-REDO's tree itself -- same concurrent-fetch pattern already proven fast
elsewhere this round (SCOP2, MobiDB).

License (pdb-redo.eu/license): free of copyright, commercial and non-commercial use permitted,
requires attributing PDB-REDO and the original structure authors.

Usage:
    python scripts/download_pdbredo.py
    python scripts/download_pdbredo.py --limit 500 --workers 8   # smoke test
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import pandas as pd
import requests

OUT = Path("data/corpus/pdbredo")
IDS_SOURCE = Path("data/corpus/rcsb/pdb_entries_enriched.csv")
API = "https://pdb-redo.eu/db"
HEADERS = {"User-Agent": "chatPDB/1.0 (protein-structure-rag; marc@marcdeller.com)", "Connection": "close"}

# The handful of fields worth promoting to real CSV columns -- the rest of each data.json's ~200
# fields are mostly per-Z-score geometry diagnostics not worth a column each.
KEEP_FIELDS = [
    "RFACT", "RFREE",              # original deposited R-work / R-free
    "RFFIN", "RFFINUNB", "RFFINZ", "SIGRFFIN",  # PDB-REDO's re-refined free-R + uncertainty
    "DATARESH", "SPACEGROUP", "WAVELENGTH", "VERSION", "BNET",
]

_lock = Lock()
_stats = {"ok": 0, "no_data": 0, "failed": 0, "done": 0}


def _fetch_one(pdb_id: str) -> dict | None:
    try:
        r = requests.get(f"{API}/{pdb_id.lower()}/data.json", headers=HEADERS, timeout=(10, 20))
        if r.status_code != 200:
            with _lock:
                _stats["no_data"] += 1
            return None
        data = r.json()
        props = data.get("properties", {})
        row = {"pdb_id": pdb_id.upper()}
        for field in KEEP_FIELDS:
            row[field.lower()] = props.get(field)
        with _lock:
            _stats["ok"] += 1
        return row
    except requests.RequestException:
        with _lock:
            _stats["failed"] += 1
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    pdb_ids = sorted(pd.read_csv(IDS_SOURCE, usecols=["pdb_id"])["pdb_id"].dropna().unique())
    if args.limit:
        pdb_ids = pdb_ids[:args.limit]
    print(f"  {len(pdb_ids):,} PDB IDs to query, {args.workers} workers")

    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_fetch_one, pid): pid for pid in pdb_ids}
        for fut in as_completed(futures):
            result = fut.result()
            if result:
                rows.append(result)
            with _lock:
                _stats["done"] += 1
                done = _stats["done"]
            if done % 10_000 == 0:
                rate = done / (time.time() - t0)
                eta = (len(pdb_ids) - done) / rate / 60 if rate > 0 else 0
                print(f"  {done:,}/{len(pdb_ids):,} (ok {_stats['ok']:,}, no_data {_stats['no_data']:,}, "
                      f"failed {_stats['failed']:,}) -- {rate:.1f}/s, ETA {eta:.1f}m", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "pdbredo_metadata.csv", index=False)
    print(f"\nDone in {(time.time()-t0)/60:.1f}m: {len(df):,} rows -> pdbredo_metadata.csv")

    for col in ("rfree", "rffin"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    delta = (df["rfree"] - df["rffin"]).dropna()
    if not delta.empty:
        print(f"  R-free delta (deposited - PDB-REDO): median {delta.median():+.4f}, "
              f"{(delta > 0.01).sum():,} entries improved by >0.01, "
              f"{(delta < -0.01).sum():,} entries got worse by >0.01")


if __name__ == "__main__":
    main()
