#!/usr/bin/env python3
"""
download_mobidb.py — pull MobiDB intrinsic-disorder annotations into data/corpus/mobidb/ for RAG
ingestion and SFT generation.

Answers "why is this region missing from the crystal structure" -- a genuinely disordered region
predicted/curated by MobiDB is a real, biologically meaningful reason for missing density, distinct
from a poorly-diffracting or poorly-modelled region.

Access confirmed live 2026-07-17 (round 4 research): `https://mobidb.org/api/download?acc={acc}&
format=json` works per-accession. Comma-separated batch queries (`acc=A,B`) were tested and silently
return only the first accession with no error -- there is no working batch/bulk endpoint reachable
without a browser-rendered Swagger session, so this is genuinely one request per accession. Scoped
to the 73,910 PDB-cross-referenced accessions already in data/corpus/uniprot/uniprot_entries.csv.
Prefers `curated-disorder-priority` (DisProt/IDEAL-backed, when it exists) and falls back to
`prediction-disorder-priority` (a consensus of several predictors) for the great majority of
accessions that have no curated annotation.

Usage:
    python scripts/download_mobidb.py
    python scripts/download_mobidb.py --limit 500 --workers 8   # smoke test
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import pandas as pd
import requests

OUT = Path("data/corpus/mobidb")
UNIPROT_ENTRIES = Path("data/corpus/uniprot/uniprot_entries.csv")
API = "https://mobidb.org/api/download"
HEADERS = {"User-Agent": "chatPDB/1.0 (protein-structure-rag; marc@marcdeller.com)", "Connection": "close"}

_lock = Lock()
_stats = {"curated": 0, "prediction": 0, "no_data": 0, "failed": 0, "done": 0}


def _fetch_one(acc: str) -> dict | None:
    try:
        r = requests.get(API, params={"acc": acc, "format": "json"}, headers=HEADERS, timeout=(10, 20))
        if r.status_code != 200 or not r.text.strip():
            with _lock:
                _stats["no_data"] += 1
            return None
        data = r.json()
        block = data.get("curated-disorder-priority")
        source = "curated"
        if not block:
            block = data.get("prediction-disorder-priority")
            source = "prediction"
        if not block:
            with _lock:
                _stats["no_data"] += 1
            return None
        with _lock:
            _stats[source] += 1
        return {
            "accession": acc,
            "length": data.get("length"),
            "source": source,
            "content_fraction": block.get("content_fraction"),
            "content_count": block.get("content_count"),
            "regions": ";".join(f"{a}-{b}" for a, b in block.get("regions", [])),
        }
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
    accessions = sorted(pd.read_csv(UNIPROT_ENTRIES, usecols=["accession"])["accession"].dropna().unique())
    if args.limit:
        accessions = accessions[:args.limit]
    print(f"  {len(accessions):,} accessions to query, {args.workers} workers")

    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_fetch_one, acc): acc for acc in accessions}
        for fut in as_completed(futures):
            result = fut.result()
            if result:
                rows.append(result)
            with _lock:
                _stats["done"] += 1
                done = _stats["done"]
            if done % 5000 == 0:
                rate = done / (time.time() - t0)
                eta = (len(accessions) - done) / rate / 60 if rate > 0 else 0
                print(f"  {done:,}/{len(accessions):,} (curated {_stats['curated']:,}, "
                      f"prediction {_stats['prediction']:,}, no_data {_stats['no_data']:,}, "
                      f"failed {_stats['failed']:,}) -- {rate:.1f}/s, ETA {eta:.1f}m", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "mobidb_disorder.csv", index=False)
    print(f"\nDone in {(time.time()-t0)/60:.1f}m: {len(df):,} rows -> mobidb_disorder.csv")


if __name__ == "__main__":
    main()
