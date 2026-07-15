#!/usr/bin/env python3
"""
download_pharos.py — pull Pharos (Illuminating the Druggable Genome) target data into
data/corpus/pharos/ for RAG ingestion.

Ties chatPDB's structural corpus to target druggability/study-status context: Pharos data joins
directly to the UniProt accessions already in sifts_pdb_uniprot.csv, so a question like "how well
studied is the target behind PDB entry X" can chain PDB -> (SIFTS) -> UniProt -> (Pharos) -> target
development level / disease associations.

TDL (target development level) meaning, for reference in generated Q&A later:
  Tclin — target of an approved drug
  Tchem — target of a potent small molecule, not yet an approved drug
  Tbio  — well-studied biologically but no known potent small-molecule ligand
  Tdark — understudied; little functional annotation

Source: Pharos GraphQL API, https://pharos-api.ncats.io/graphql

IMPORTANT — bulk pagination is broken on this API (confirmed 2026-07-15, both via variables and
inline literal values): `targets(top, skip)` silently ignores both arguments and always returns
the same first 10 targets regardless of what's requested. Confirmed via schema introspection that
`skip`/`top` are real, correctly-typed Int arguments — this isn't a naming mistake, the backend (or
a caching layer in front of it) just doesn't honour them. A full ~20,412-target dump therefore isn't
reliably achievable through this endpoint; the "correct" way to get the full dataset is importing
the underlying TCRD MySQL dump (http://juniper.health.unm.edu/tcrd/), which is out of scope for now
(new infra dependency for one source).

Per-target lookup via `target(q:{uniprot:...})` DOES work correctly (verified against EGFR/P00533).
So instead of a broken bulk dump, this script prioritises: rank every UniProt accession already
cross-referenced to a PDB structure (sifts_pdb_uniprot.csv, 73,910 unique accessions — querying all
of those individually would take ~6+ hours) by how many PDB entries reference it, and pulls Pharos
data for the top N most-studied-in-the-PDB targets. That's the intersection that actually matters
for chatPDB: targets with solved structures, prioritised by structural coverage.

Usage:
    python scripts/download_pharos.py                  # top 2000 UniProt accessions by PDB coverage
    python scripts/download_pharos.py --top-n 500       # smaller run
"""

import argparse
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

OUT = Path("data/corpus/pharos")
SIFTS_UNIPROT = Path("data/corpus/rcsb/sifts_pdb_uniprot.csv")
GRAPHQL_URL = "https://pharos-api.ncats.io/graphql"
PAUSE = 0.2

QUERY = """
query GetTarget($uniprot: String!) {
  target(q: { uniprot: $uniprot }) {
    name
    sym
    tdl
    fam
    uniprot
    novelty
    diseaseCounts { name }
  }
}
"""


def top_uniprot_accessions(n: int) -> list[str]:
    if not SIFTS_UNIPROT.exists():
        raise SystemExit(f"{SIFTS_UNIPROT} not found — run scripts/download_rcsb.py first.")
    df = pd.read_csv(SIFTS_UNIPROT, usecols=["SP_PRIMARY"], dtype=str)
    counts = Counter(df["SP_PRIMARY"].dropna())
    return [acc for acc, _ in counts.most_common(n)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top-n", type=int, default=2000,
                         help="how many top-PDB-coverage UniProt accessions to look up in Pharos")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "chatPDB/1.0 (protein-structure-rag; marc@marcdeller.com)"

    print(f"\n[1/1] Pharos targets for top {args.top_n} PDB-cross-referenced UniProt accessions ...")
    accessions = top_uniprot_accessions(args.top_n)
    print(f"  {len(accessions):,} accessions to look up (ranked by PDB structure count)")

    rows = []
    for i, acc in enumerate(accessions, 1):
        try:
            resp = session.post(GRAPHQL_URL, json={"query": QUERY, "variables": {"uniprot": acc}}, timeout=30)
            resp.raise_for_status()
            t = resp.json().get("data", {}).get("target")
        except requests.RequestException as e:
            print(f"    [warn] {acc}: {e}")
            t = None
        if t:
            diseases = " | ".join(d["name"] for d in (t.get("diseaseCounts") or [])[:10])
            rows.append({
                "uniprot": acc,
                "name": t.get("name", ""),
                "symbol": t.get("sym", ""),
                "tdl": t.get("tdl", ""),
                "family": t.get("fam", ""),
                "novelty": t.get("novelty"),
                "top_diseases": diseases,
            })
        if i % 100 == 0:
            print(f"  Pharos: {i:,}/{len(accessions):,} ({len(rows):,} matched)", end="\r")
        time.sleep(PAUSE)

    print(f"  Pharos: {len(rows):,}/{len(accessions):,} accessions matched a target          ")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "pharos_targets.csv", index=False)
    print(f"  Saved {len(df):,} rows -> pharos_targets.csv")
    if not df.empty:
        print(f"  TDL breakdown: {df['tdl'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
