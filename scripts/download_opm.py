#!/usr/bin/env python3
"""
download_opm.py — pull OPM (Orientations of Proteins in Membranes) data into data/corpus/opm/ for
RAG ingestion and SFT generation.

Adds a structural category chatPDB currently can't reason about at all: is this a membrane protein,
and where does the bilayer sit relative to the deposited coordinates.

Access confirmed live 2026-07-17 (round 4 research): opm.phar.umich.edu itself is a JS SPA with no
scrapeable static API, but its backing storage is a publicly listable Google Cloud Storage bucket
(`https://storage.googleapis.com/opm-assets?prefix=pdb/`, S3-style XML listing, paginated via
`&marker=`). Each `pdb/{id}.pdb` file carries OPM's membrane placement as a REMARK line
("1/2 of bilayer thickness: N.N") plus HETATM "dummy atom" records marking the membrane boundary
plane -- this is metadata embedded in a small coordinate-adjacent file, not a bulk map/volume, so
disk cost is trivial (OPM is a curated set, low tens of thousands of entries at most, not 256k).
Scoped to entries whose PDB ID is already in this corpus.

Usage:
    python scripts/download_opm.py
    python scripts/download_opm.py --limit 200   # smoke test
"""

import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from xml.etree import ElementTree

import pandas as pd
import requests

OUT = Path("data/corpus/opm")
IDS_SOURCE = Path("data/corpus/rcsb/pdb_entries_enriched.csv")
BUCKET_LIST_URL = "https://storage.googleapis.com/opm-assets"
FILE_BASE = "https://opm-assets.storage.googleapis.com/pdb"
HEADERS = {"User-Agent": "chatPDB/1.0 (protein-structure-rag; marc@marcdeller.com)", "Connection": "close"}
_NS = {"s3": "http://doc.s3.amazonaws.com/2006-03-01"}
_THICKNESS_RE = re.compile(r"1/2 of bilayer thickness:\s*([\d.]+)")


def list_bucket_pdb_ids() -> list[str]:
    """Paginate the public GCS bucket listing to enumerate every OPM PDB ID -- there's no bulk
    index file, this listing IS the index."""
    ids = []
    marker = ""
    while True:
        params = {"prefix": "pdb/", "max-keys": 1000}
        if marker:
            params["marker"] = marker
        r = requests.get(BUCKET_LIST_URL, params=params, headers=HEADERS, timeout=(10, 30))
        r.raise_for_status()
        root = ElementTree.fromstring(r.text)
        keys = [el.text for el in root.findall(".//s3:Contents/s3:Key", _NS)]
        for key in keys:
            if key.endswith(".pdb"):
                ids.append(Path(key).stem.upper())
        next_marker_el = root.find("s3:NextMarker", _NS)
        is_truncated = root.findtext("s3:IsTruncated", default="false", namespaces=_NS)
        if is_truncated != "true" or next_marker_el is None:
            break
        marker = next_marker_el.text
    return ids


_lock = Lock()
_stats = {"done": 0}


def _fetch_one(pdb_id: str) -> dict | None:
    try:
        r = requests.get(f"{FILE_BASE}/{pdb_id.lower()}.pdb", headers=HEADERS, timeout=(10, 20))
        if r.status_code != 200:
            return None
        m = _THICKNESS_RE.search(r.text)
        has_dummy_atoms = " DUM " in r.text
        return {
            "pdb_id": pdb_id,
            "half_bilayer_thickness_A": float(m.group(1)) if m else None,
            "has_membrane_dummy_atoms": has_dummy_atoms,
        }
    except requests.RequestException:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    valid_pdb_ids = set(pd.read_csv(IDS_SOURCE, usecols=["pdb_id"])["pdb_id"].str.upper())

    print("[1/2] Listing OPM's bucket ...")
    all_ids = list_bucket_pdb_ids()
    matched_ids = sorted(set(all_ids) & valid_pdb_ids)
    print(f"  {len(all_ids):,} OPM entries total, {len(matched_ids):,} match this corpus")
    if args.limit:
        matched_ids = matched_ids[:args.limit]

    # Originally sequential with a per-file sleep -- observed ~1.8s/request against this GCS-backed
    # bucket (each Connection:close request pays a fresh TLS handshake), which would have taken
    # ~7-8h for the full 15k set. Concurrent fetch, same pattern as every other multi-request
    # downloader this round.
    print(f"\n[2/2] Fetching + parsing membrane-placement REMARK per matched entry ({args.workers} workers) ...")
    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_fetch_one, pid): pid for pid in matched_ids}
        for fut in as_completed(futures):
            result = fut.result()
            if result:
                rows.append(result)
            with _lock:
                _stats["done"] += 1
                done = _stats["done"]
            if done % 1000 == 0:
                rate = done / (time.time() - t0)
                eta = (len(matched_ids) - done) / rate / 60 if rate > 0 else 0
                print(f"  {done:,}/{len(matched_ids):,} -- {rate:.1f}/s, ETA {eta:.1f}m", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "opm_membrane_placement.csv", index=False)
    print(f"\nDone in {(time.time()-t0)/60:.1f}m: {len(df):,} rows -> opm_membrane_placement.csv")


if __name__ == "__main__":
    main()
