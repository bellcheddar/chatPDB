#!/usr/bin/env python3
"""
download_emdb.py — pull EMDB (Electron Microscopy Data Bank) per-entry map metadata into
data/corpus/emdb/ for RAG ingestion and SFT generation.

The 353 GB mmCIF pool has coordinates for every EM-method PDB entry (35,330 of them, confirmed
live 2026-07-17) but zero map-level metadata -- resolution-determination method (e.g. "FSC 0.143
cut-off"), contour level, pixel spacing, symmetry. This closes that gap for the now-dominant
experimental method, metadata-only (no map volumes, which run into the GB-per-entry range).

**Revision note (2026-07-17, same round):** originally used `https://www.ebi.ac.uk/emdb/api/search/*`
paginated in blocks of 100, on the assumption it returned one document per EMDB entry. Wrong --
confirmed live partway through a multi-hour run that the search index returns far more documents
than the real 59,608-entry total (still-valid, non-duplicate-looking EMDB IDs kept appearing past
start=350,000), producing massively duplicated (pdb_id, emdb_id) rows with no natural termination in
sight. Switched to the documented bulk index instead:
`https://ftp.ebi.ac.uk/pub/databases/emdb/status/latest/emdb_released_holdings.json` (confirmed
live, exactly 59,608 entries) gives the definitive ID list, then `https://www.ebi.ac.uk/emdb/api/
entry/{EMD-ID}` (confirmed live) is queried per-ID, concurrently -- same pattern as every other
per-entry downloader this round, and returns the identical nested schema `_parse_entry()` below was
already written against.

Usage:
    python scripts/download_emdb.py
    python scripts/download_emdb.py --limit 200 --workers 8   # smoke test
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import pandas as pd
import requests

OUT = Path("data/corpus/emdb")
IDS_SOURCE = Path("data/corpus/rcsb/pdb_entries_enriched.csv")
HOLDINGS_URL = "https://ftp.ebi.ac.uk/pub/databases/emdb/status/latest/emdb_released_holdings.json"
ENTRY_URL = "https://www.ebi.ac.uk/emdb/api/entry"
HEADERS = {"User-Agent": "chatPDB/1.0 (protein-structure-rag; marc@marcdeller.com)", "Connection": "close"}

_lock = Lock()
_stats = {"matched": 0, "no_pdb_link": 0, "failed": 0, "done": 0}


def _get(path, default=None):
    """Walk a dotted path through nested dicts/lists, returning `default` on any miss -- EMDB's
    schema has several genuinely optional branches (not every entry has been fitted to a PDB model,
    not every method has the same image_processing shape), so defensive walking beats a KeyError."""
    def walk(obj, parts):
        # Collapse a list to its first element at every step (not just mid-path) -- EMDB wraps
        # several singleton branches (structure_determination, image_processing...) in a list even
        # when there's exactly one, including at the very end of a path.
        if isinstance(obj, list):
            obj = obj[0] if obj else None
        if obj is None or not parts:
            return obj
        key = parts[0]
        if isinstance(obj, dict):
            return walk(obj.get(key), parts[1:])
        return default
    result = walk(path[0], path[1:])
    return result if result is not None else default


def _parse_entry(entry: dict) -> dict:
    emdb_id = entry.get("emdb_id", "")
    # Not routed through _get(): pdb_reference is a genuine list (an entry can fit >1 PDB model)
    # and _get()'s list-collapsing (needed for the *singleton* wrapper lists elsewhere in this
    # schema, e.g. structure_determination) would incorrectly collapse it before indexing.
    pdb_list = entry.get("crossreferences", {}).get("pdb_list", {})
    pdb_refs = pdb_list.get("pdb_reference", []) if isinstance(pdb_list, dict) else []
    if isinstance(pdb_refs, dict):  # EMDB renders a single reference as a bare dict, not a 1-list
        pdb_refs = [pdb_refs]
    pdb_id = (pdb_refs[0].get("pdb_id", "").upper() if pdb_refs else None)

    map_ = entry.get("map", {})
    contour = _get([map_, "contour_list", "contour"])
    dims = map_.get("dimensions", {})
    pixel = map_.get("pixel_spacing", {})

    sd = _get([entry, "structure_determination_list", "structure_determination"], {})
    method = sd.get("method")
    final_recon = _get([sd, "image_processing", "final_reconstruction"], {})

    return {
        "emdb_id": emdb_id,
        "pdb_id": pdb_id,
        "method": method,
        "resolution_A": _get([final_recon, "resolution", "valueOf_"]),
        "resolution_method": final_recon.get("resolution_method"),
        "contour_level": contour.get("level") if isinstance(contour, dict) else None,
        "pixel_spacing_x_A": _get([pixel, "x", "valueOf_"]),
        "space_group": _get([map_, "symmetry", "space_group"]),
        "dim_col": dims.get("col"), "dim_row": dims.get("row"), "dim_sec": dims.get("sec"),
    }


def _fetch_one(emdb_id: str, valid_pdb_ids: set[str]) -> dict | None:
    try:
        r = requests.get(f"{ENTRY_URL}/{emdb_id}", headers=HEADERS, timeout=(10, 20))
        if r.status_code != 200:
            with _lock:
                _stats["failed"] += 1
            return None
        entry = r.json()
        parsed = _parse_entry(entry)
        if not parsed["pdb_id"]:
            with _lock:
                _stats["no_pdb_link"] += 1
            return None
        if parsed["pdb_id"] not in valid_pdb_ids:
            with _lock:
                _stats["no_pdb_link"] += 1
            return None
        with _lock:
            _stats["matched"] += 1
        return parsed
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
    valid_pdb_ids = set(pd.read_csv(IDS_SOURCE, usecols=["pdb_id"])["pdb_id"].str.upper())
    print(f"  {len(valid_pdb_ids):,} corpus PDB IDs loaded for filtering")

    print("[1/2] Fetching bulk EMDB holdings index ...")
    r = requests.get(HOLDINGS_URL, headers=HEADERS, timeout=(10, 60))
    r.raise_for_status()
    emdb_ids = sorted(r.json().keys())
    print(f"  {len(emdb_ids):,} EMDB entries total")
    if args.limit:
        emdb_ids = emdb_ids[:args.limit]

    print(f"\n[2/2] Fetching per-entry map metadata ({args.workers} workers) ...")
    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_fetch_one, eid, valid_pdb_ids): eid for eid in emdb_ids}
        for fut in as_completed(futures):
            result = fut.result()
            if result:
                rows.append(result)
            with _lock:
                _stats["done"] += 1
                done = _stats["done"]
            if done % 5000 == 0:
                rate = done / (time.time() - t0)
                eta = (len(emdb_ids) - done) / rate / 60 if rate > 0 else 0
                print(f"  {done:,}/{len(emdb_ids):,} (matched {_stats['matched']:,}, "
                      f"no_pdb_link {_stats['no_pdb_link']:,}, failed {_stats['failed']:,}) -- "
                      f"{rate:.1f}/s, ETA {eta:.1f}m", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "emdb_map_metadata.csv", index=False)
    print(f"\nDone in {(time.time()-t0)/60:.1f}m: {len(df):,} rows matched to corpus PDB entries "
          f"-> emdb_map_metadata.csv")


if __name__ == "__main__":
    main()
