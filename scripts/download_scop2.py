#!/usr/bin/env python3
"""
download_scop2.py — pull SCOP2 fold/superfamily/family descriptions into data/corpus/scop2/ for
RAG ingestion and SFT generation.

We already have SIFTS PDB->SCOP2 domain-ID mappings (data/corpus/rcsb/sifts_pdb_scop2.csv,
SF_DOMID/FA_DOMID columns) but no semantic descriptions for those domain IDs -- the classification
names live one level up, on the parent superfamily/family node, not on the domain instance ID
itself.

Access confirmed live 2026-07-17 (round 4 research): SCOP2's own domain (scop.mrc-lmb.cam.ac.uk)
has NO working scripted API today -- its documented REST API 404s and the site is a JS SPA with no
scrapeable static pages (an earlier "confirmed" call that a new API existed was itself wrong on
recheck). The real working path is EBI's PDBe mappings API,
`https://www.ebi.ac.uk/pdbe/api/mappings/scop2/{pdb_id}`, queried per PDB ID (not per node -- SIFTS'
SF_DOMID/FA_DOMID turn out to be near-unique per-chain domain *instance* IDs, 36,898 distinct out of
36,915 rows, not reusable classification node IDs). Scoped to the 27,530 distinct PDB IDs already
present in sifts_pdb_scop2.csv rather than all 256k corpus entries.

Usage:
    python scripts/download_scop2.py
    python scripts/download_scop2.py --limit 200 --workers 8   # smoke test
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import pandas as pd
import requests

OUT = Path("data/corpus/scop2")
SIFTS_SCOP2 = Path("data/corpus/rcsb/sifts_pdb_scop2.csv")
API = "https://www.ebi.ac.uk/pdbe/api/mappings/scop2"
HEADERS = {"User-Agent": "chatPDB/1.0 (protein-structure-rag; marc@marcdeller.com)", "Connection": "close"}

_lock = Lock()
_stats = {"ok": 0, "no_data": 0, "failed": 0, "done": 0}


def _fetch_one(pdb_id: str) -> list[dict]:
    rows = []
    try:
        r = requests.get(f"{API}/{pdb_id.lower()}", headers=HEADERS, timeout=(10, 20))
        if r.status_code != 200:
            with _lock:
                _stats["no_data"] += 1
            return rows
        nodes = r.json().get(pdb_id.lower(), {}).get("SCOP2", {}).get("nodes", {})
        class_name = nodes["classes"][0]["name"].strip() if nodes.get("classes") else None
        fold_name = nodes["folds"][0]["name"].strip() if nodes.get("folds") else None
        for level, key in [("superfamily", "superfamilies"), ("family", "families")]:
            for node in nodes.get(key, []):
                node_name = node.get("name", "").strip()
                for mapping in node.get("mappings", []):
                    rows.append({
                        "domain_id": mapping.get("domain_id"),
                        "pdb_id": pdb_id.upper(),
                        "chain": mapping.get("chain_id"),
                        "level": level,
                        "node_id": node.get("id"),
                        "node_name": node_name,
                        "fold_name": fold_name,
                        "class_name": class_name,
                    })
        with _lock:
            _stats["ok"] += 1
    except requests.RequestException:
        with _lock:
            _stats["failed"] += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    pdb_ids = sorted(pd.read_csv(SIFTS_SCOP2, usecols=["PDB"], dtype=str)["PDB"].dropna().unique())
    if args.limit:
        pdb_ids = pdb_ids[:args.limit]
    print(f"  {len(pdb_ids):,} distinct PDB IDs to query, {args.workers} workers")

    all_rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_fetch_one, pid): pid for pid in pdb_ids}
        for fut in as_completed(futures):
            all_rows.extend(fut.result())
            with _lock:
                _stats["done"] += 1
                done = _stats["done"]
            if done % 2000 == 0:
                rate = done / (time.time() - t0)
                eta = (len(pdb_ids) - done) / rate / 60 if rate > 0 else 0
                print(f"  {done:,}/{len(pdb_ids):,} (ok {_stats['ok']:,}, no_data {_stats['no_data']:,}, "
                      f"failed {_stats['failed']:,}) -- {rate:.1f}/s, ETA {eta:.1f}m", flush=True)

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["domain_id", "level"])
    df.to_csv(OUT / "scop2_domain_names.csv", index=False)
    print(f"\nDone in {(time.time()-t0)/60:.1f}m: {len(df):,} domain-level rows "
          f"({_stats['ok']:,} PDB entries with SCOP2 data, {_stats['no_data']:,} without, "
          f"{_stats['failed']:,} request failures) -> scop2_domain_names.csv")


if __name__ == "__main__":
    main()
