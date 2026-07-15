#!/usr/bin/env python3
"""
download_uniprot.py — pull UniProtKB/Swiss-Prot entries into data/corpus/uniprot/ for RAG
ingestion.

chatPDB already has PDB-chain -> UniProt-accession mappings via SIFTS (sifts_pdb_uniprot.csv from
scripts/download_rcsb.py), but not what those accessions actually *are* — protein name, function,
organism, and the standardised keyword vocabulary UniProt annotates every entry with. This script
fetches both.

Full UniProtKB/Swiss-Prot is ~570k reviewed entries (and TrEMBL is 250M+ unreviewed) — pulling all
of it is out of scope. Same pattern as scripts/download_pharos.py: scope to the accessions that
actually matter for chatPDB, i.e. every unique UniProt accession already cross-referenced to a PDB
structure (73,910 of them in sifts_pdb_uniprot.csv) — the intersection of "has a solved structure"
and "has curated UniProt annotation" is exactly what a structure-focused assistant needs.

Sources (confirmed live 2026-07-15):
  - UniProtKB REST batch endpoint: https://rest.uniprot.org/uniprotkb/accessions
    Accepts a comma-separated accession list (confirmed working at 100/request), TSV output.
  - UniProt controlled-vocabulary keywords: https://rest.uniprot.org/keywords/stream
    Full list in one request, 1,201 keywords with category + definition — the vocabulary that
    UniProt's own "Keywords" column (pulled per-entry below) draws from.

Usage:
    python scripts/download_uniprot.py                # full run: keywords + all 73,910 accessions
    python scripts/download_uniprot.py --limit 500     # smoke test
"""

import argparse
import io
import time
from pathlib import Path

import pandas as pd
import requests

OUT = Path("data/corpus/uniprot")
SIFTS_UNIPROT = Path("data/corpus/rcsb/sifts_pdb_uniprot.csv")
ENTRY_URL = "https://rest.uniprot.org/uniprotkb/accessions"
KEYWORDS_URL = "https://rest.uniprot.org/keywords/stream"
BATCH_SIZE = 100
PAUSE = 0.2

ENTRY_FIELDS = "accession,protein_name,gene_names,organism_name,cc_function,keyword"
# UniProt function/keyword text can run very long for large multi-domain polyproteins (a
# SARS-CoV-2 replicase entry's function text alone ran past 8,000 characters) — truncating
# keeps ingest_rag.py's chunk-size math from being skewed by rare pathological outliers, same
# defensive reasoning as the MAX_ROWS_PER_CSV sampling in ingest_rag.py.
MAX_FUNCTION_CHARS = 1500


def fetch_keywords() -> None:
    print("\n[1/2] UniProt controlled-vocabulary keywords ...")
    r = requests.get(KEYWORDS_URL, params={"query": "*", "format": "tsv",
                                            "fields": "id,name,category,definition"}, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), sep="\t")
    df.to_csv(OUT / "uniprot_keywords.csv", index=False)
    print(f"  Saved {len(df):,} rows -> uniprot_keywords.csv")


def top_accessions(limit: int | None) -> list[str]:
    df = pd.read_csv(SIFTS_UNIPROT, usecols=["SP_PRIMARY"], dtype=str)
    accs = df["SP_PRIMARY"].dropna().unique().tolist()
    return accs[:limit] if limit else accs


def fetch_entries(accessions: list[str]) -> None:
    print(f"\n[2/2] UniProtKB/Swiss-Prot entries for {len(accessions):,} PDB-cross-referenced accessions ...")
    rows = []
    session = requests.Session()
    session.headers["User-Agent"] = "chatPDB/1.0 (protein-structure-rag; marc@marcdeller.com)"
    for i in range(0, len(accessions), BATCH_SIZE):
        batch = accessions[i : i + BATCH_SIZE]
        try:
            r = session.get(ENTRY_URL, params={"accessions": ",".join(batch), "format": "tsv",
                                                "fields": ENTRY_FIELDS}, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), sep="\t")
        except (requests.RequestException, pd.errors.ParserError) as e:
            print(f"    [warn] batch at {i}: {e}")
            time.sleep(2)
            continue
        for _, row in df.iterrows():
            func = str(row.get("Function [CC]", "") or "")
            if len(func) > MAX_FUNCTION_CHARS:
                func = func[:MAX_FUNCTION_CHARS] + "... [truncated]"
            rows.append({
                "accession": row.get("Entry", ""),
                "protein_name": row.get("Protein names", ""),
                "gene_names": row.get("Gene Names", ""),
                "organism": row.get("Organism", ""),
                "function": func,
                "keywords": row.get("Keywords", ""),
            })
        if (i // BATCH_SIZE) % 20 == 0:
            print(f"  UniProt: {len(rows):,}/{len(accessions):,}", flush=True)
        time.sleep(PAUSE)

    print(f"  UniProt: {len(rows):,}/{len(accessions):,} entries fetched          ")
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "uniprot_entries.csv", index=False)
    print(f"  Saved {len(df_out):,} rows -> uniprot_entries.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="stop after this many accessions (smoke test)")
    parser.add_argument("--skip-keywords", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if not args.skip_keywords:
        fetch_keywords()
    accessions = top_accessions(args.limit)
    fetch_entries(accessions)


if __name__ == "__main__":
    main()
