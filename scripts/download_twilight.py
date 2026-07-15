#!/usr/bin/env python3
"""
download_twilight.py — pull TWILIGHT ligand electron-density quality data into
data/corpus/twilight/ for RAG ingestion.

TWILIGHT (Rupp lab, ruppweb.org) is a per-ligand-instance quality assessment of every
crystallographic ligand in the PDB: RSCC (real-space correlation coefficient — how well the
modelled ligand actually fits the electron density), OWAB (occupancy-weighted average B-factor),
and the entry's resolution/R-work/R-free at time of that ligand's deposition. This is exactly the
"structure QC" data PROJECT_PLAN.md section 4 flagged as a gap (wwPDB's own per-entry validation
reports aren't reachable via the RCSB GraphQL path used in scripts/download_rcsb.py) — TWILIGHT
covers the ligand-fit side of that gap directly, at residue-instance granularity.

Source (confirmed live 2026-07-15): https://www.ruppweb.org/twilight/ligands-2020-01-15.tsv.bz2
A single bulk bzip2-compressed TSV, ~870k rows (one row per ligand instance across all PDB
entries with a bound heteroatom group as of the file's 2020-01-15 snapshot date — the file is a
fixed historical snapshot, not live-updated; newer depositions since then won't appear here).

Usage:
    python scripts/download_twilight.py
"""

import bz2
from pathlib import Path

import pandas as pd
import requests

OUT = Path("data/corpus/twilight")
URL = "https://www.ruppweb.org/twilight/ligands-2020-01-15.tsv.bz2"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("\n[1/1] TWILIGHT ligand density-quality data ...")
    print(f"  Downloading {URL} ...")
    r = requests.get(URL, timeout=300)
    r.raise_for_status()

    text = bz2.decompress(r.content).decode("utf-8", errors="replace")
    df = pd.read_csv(pd.io.common.StringIO(text), sep="\t")
    df.columns = [c.strip() for c in df.columns]
    print(f"  Parsed {len(df):,} ligand-instance rows")

    df.to_csv(OUT / "twilight_ligands.csv", index=False)
    print(f"  Saved {len(df):,} rows -> twilight_ligands.csv")
    if "RSCC" in df.columns:
        rscc = pd.to_numeric(df["RSCC"], errors="coerce")
        print(f"  RSCC range: {rscc.min():.3f}-{rscc.max():.3f}, median {rscc.median():.3f}")


if __name__ == "__main__":
    main()
