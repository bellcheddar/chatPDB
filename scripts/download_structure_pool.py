#!/usr/bin/env python3
"""
download_structure_pool.py — download real PDB coordinate files into data/structures/ for
Phase 3 SFT ground-truth generation.

This is NOT a RAG corpus source (nothing here gets embedded/ingested) — it's the local file pool
scripts/build_dataset.py runs Biopython/gemmi/DSSP against to generate genuinely execution-verified
tool-calling examples (secondary structure assignment in particular has no shortcut: it has to be
computed from real coordinates, unlike resolution/R-free/UniProt-mapping questions which can be
grounded directly in already-verified corpus metadata).

Selection: a stratified sample (data/structures_pool_ids.csv, built ad hoc for this phase) of 850
entries — 600 X-ray, 150 solution NMR, 100 EM — capped to 50-4,000 atoms so parsing/DSSP stay fast.

Usage:
    python scripts/download_structure_pool.py
"""

import time
from pathlib import Path

import pandas as pd
import requests

OUT = Path("data/structures")
IDS_FILE = Path("data/structures_pool_ids.csv")
PAUSE = 0.15


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(IDS_FILE)["pdb_id"].tolist()
    print(f"Downloading {len(ids):,} structure files ...")

    session = requests.Session()
    session.headers["User-Agent"] = "chatPDB/1.0 (protein-structure-rag; marc@marcdeller.com)"
    ok, failed = 0, []
    for i, pdb_id in enumerate(ids, 1):
        dest = OUT / f"{pdb_id.lower()}.pdb"
        if dest.exists():
            ok += 1
            continue
        try:
            r = session.get(f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb", timeout=30)
            if r.status_code == 200 and r.text.strip():
                dest.write_text(r.text)
                ok += 1
            else:
                failed.append(pdb_id)
        except requests.RequestException:
            failed.append(pdb_id)
        if i % 50 == 0:
            print(f"  {i:,}/{len(ids):,} ({ok:,} ok, {len(failed):,} failed)", flush=True)
        time.sleep(PAUSE)

    print(f"\nDone: {ok:,} downloaded, {len(failed):,} failed.")
    if failed:
        print(f"  Failed (likely no legacy PDB format available, large/complex entries): {failed[:20]}")


if __name__ == "__main__":
    main()
