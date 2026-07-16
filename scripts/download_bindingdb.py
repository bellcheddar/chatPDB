#!/usr/bin/env python3
"""
download_bindingdb.py — pull real protein-ligand binding affinity data (Ki/IC50/Kd/EC50) cross-
referenced to PDB structures into data/corpus/bindingdb/ for RAG ingestion and SFT generation.

TWILIGHT (already ingested) answers "does this ligand's modelled *pose* fit the density?" but says
nothing about *potency* — how tightly the ligand actually binds. BindingDB closes that gap: real,
experimentally measured affinity data, a large fraction of it tied directly to specific PDB entries
via co-crystal structures.

Source (confirmed live 2026-07-16): https://www.bindingdb.org/rwd/bind/downloads/ bulk TSV,
BindingDB_All (~9 GB uncompressed, 3.2M+ measurements, updated monthly). No API needed for this —
one bulk download, then a streaming filter (never loads the full file into memory: it's read row by
row directly out of the zip) down to just the rows whose "PDB ID(s) for Ligand-Target Complex"
field contains at least one PDB ID already in this project's corpus.

Usage:
    python scripts/download_bindingdb.py                  # download (if needed) + filter
    python scripts/download_bindingdb.py --skip-download   # reuse an existing zip
"""

import argparse
import csv
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

OUT = Path("data/corpus/bindingdb")
ZIP_PATH = OUT / "BindingDB_All.zip"
BULK_URL = "https://www.bindingdb.org/rwd/bind/downloads/BindingDB_All_202607_tsv.zip"
IDS_SOURCE = Path("data/corpus/rcsb/pdb_all_entries.csv")

KEEP_COLUMNS = {
    "Ligand SMILES": "ligand_smiles",
    "Ligand HET ID in PDB": "ligand_het_id",
    "BindingDB Ligand Name": "ligand_name",
    "Target Name": "target_name",
    "Target Source Organism According to Curator or DataSource": "target_organism",
    "Ki (nM)": "ki_nM",
    "IC50 (nM)": "ic50_nM",
    "Kd (nM)": "kd_nM",
    "EC50 (nM)": "ec50_nM",
    "pH": "assay_pH",
    "Temp (C)": "assay_temp_C",
    "Article DOI": "article_doi",
    "PMID": "pmid",
    "PDB ID(s) for Ligand-Target Complex": "pdb_ids",
    "UniProt (SwissProt) Primary ID of Target Chain 1": "uniprot_primary",
}


def download_bulk() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 100_000_000:
        print(f"  {ZIP_PATH} already present ({ZIP_PATH.stat().st_size / 1e6:.0f} MB), skipping download")
        return
    print(f"  Downloading {BULK_URL} (~600 MB) ...")
    r = requests.get(BULK_URL, stream=True, timeout=600)
    r.raise_for_status()
    with open(ZIP_PATH, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    print(f"  Saved {ZIP_PATH.stat().st_size / 1e6:.0f} MB")


def filter_to_pdb_corpus() -> None:
    valid_ids = set(pd.read_csv(IDS_SOURCE)["pdb_id"].str.upper())
    print(f"  Filtering against {len(valid_ids):,} corpus PDB IDs ...")

    zf = zipfile.ZipFile(ZIP_PATH)
    rows = []
    total_seen = 0
    with zf.open("BindingDB_All.tsv") as raw:
        reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", errors="replace"), delimiter="\t")
        for row in reader:
            total_seen += 1
            pdb_field = (row.get("PDB ID(s) for Ligand-Target Complex") or "").strip()
            if not pdb_field:
                continue
            matched = [pid for pid in (p.strip() for p in pdb_field.split(",")) if pid.upper() in valid_ids]
            if not matched:
                continue
            out_row = {new: row.get(old, "") for old, new in KEEP_COLUMNS.items()}
            out_row["pdb_ids"] = ",".join(matched)  # only the IDs that are actually in our corpus
            rows.append(out_row)
            if total_seen % 500_000 == 0:
                print(f"    scanned {total_seen:,} rows, {len(rows):,} matched so far", flush=True)

    print(f"  Scanned {total_seen:,} total BindingDB rows, {len(rows):,} matched a corpus PDB entry")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bindingdb_pdb_affinities.csv", index=False)
    print(f"  Saved {len(df):,} rows -> bindingdb_pdb_affinities.csv")
    ZIP_PATH.unlink(missing_ok=True)
    print(f"  Removed staging zip ({BULK_URL.rsplit('/', 1)[-1]}) — filtered CSV is the corpus artefact")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    print("\n[1/2] BindingDB bulk download ...")
    if not args.skip_download:
        download_bulk()
    print("\n[2/2] Filtering to corpus PDB entries ...")
    filter_to_pdb_corpus()


if __name__ == "__main__":
    main()
