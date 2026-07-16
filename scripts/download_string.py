#!/usr/bin/env python3
"""
download_string.py — pull STRING protein-protein interaction network data into
data/corpus/string/ for RAG ingestion and SFT generation.

Scope decision: STRING's per-organism coverage is uneven — excellent for well-studied model
organisms (human, mouse, yeast, E. coli), patchy or absent for the thousands of other organisms
represented in the PDB. Rather than query every organism in the corpus (most queries would return
nothing, for real biological reasons, not API failure), this scopes to human (taxonomy ID 9606,
71,711 of this corpus's entries, confirmed live 2026-07-16) — the organism with both the deepest
STRING coverage and the most PDB-cross-referenced accessions, so this is the single highest-value,
most tractable population to cover well rather than covering everything thinly.

Source (confirmed live 2026-07-16): https://string-db.org/api/tsv/network — accepts a UniProt
accession directly for the query itself. Returns a small interconnected sub-network around the
query protein, NOT a star graph: many of the returned edges are between two *other* proteins in
that neighbourhood, not between the query protein and a partner (confirmed by inspecting real
output — most edges for a B2M query didn't mention B2M at all). A "who does protein X interact
with" fact needs edges that actually touch X, so this resolves the query accession's STRING
preferred name first (get_string_ids), then keeps only edges where that name appears on either
side of the pair, before ranking by confidence score.

Usage:
    python scripts/download_string.py                # top 3,000 human PDB-cross-referenced accessions
    python scripts/download_string.py --top-n 200     # smoke test
"""

import argparse
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

OUT = Path("data/corpus/string")
SIFTS_UNIPROT = Path("data/corpus/rcsb/sifts_pdb_uniprot.csv")
UNIPROT_ENTRIES = Path("data/corpus/uniprot/uniprot_entries.csv")
NETWORK_URL = "https://string-db.org/api/tsv/network"
RESOLVE_URL = "https://string-db.org/api/tsv/get_string_ids"
HEADERS = {"User-Agent": "chatPDB/1.0 (protein-structure-rag; marc@marcdeller.com)", "Connection": "close"}
HUMAN_TAXID = 9606
TOP_PARTNERS = 8


def top_human_accessions(n: int) -> list[str]:
    uniprot = pd.read_csv(UNIPROT_ENTRIES, usecols=["accession", "organism"])
    human_accs = set(uniprot[uniprot["organism"].str.contains("Homo sapiens", na=False, case=False)]["accession"])
    sifts = pd.read_csv(SIFTS_UNIPROT, usecols=["SP_PRIMARY"], dtype=str)
    counts = Counter(a for a in sifts["SP_PRIMARY"].dropna() if a in human_accs)
    return [acc for acc, _ in counts.most_common(n)]


def _resolve_preferred_name(acc: str) -> str | None:
    try:
        r = requests.get(RESOLVE_URL, params={"identifiers": acc, "species": HUMAN_TAXID},
                          headers=HEADERS, timeout=(10, 20))
        if r.status_code != 200 or not r.text.strip():
            return None
        lines = r.text.strip().split("\n")
        if len(lines) < 2:  # header only — accession not resolvable in STRING for this species
            return None
        parts = lines[1].split("\t")
        if len(parts) >= 5:
            return parts[4]  # preferredName
    except requests.RequestException:
        pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top-n", type=int, default=3000)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"\n[1/1] STRING interaction networks for top {args.top_n} human PDB-cross-referenced accessions ...")
    accessions = top_human_accessions(args.top_n)
    print(f"  {len(accessions):,} accessions to query")

    rows = []
    for i, acc in enumerate(accessions, 1):
        preferred_name = _resolve_preferred_name(acc)
        if not preferred_name:
            continue
        try:
            r = requests.get(NETWORK_URL, params={"identifiers": acc, "species": HUMAN_TAXID},
                              headers=HEADERS, timeout=(10, 20))
            if r.status_code == 200 and r.text.strip():
                lines = r.text.strip().split("\n")[1:]  # skip header
                partners = []
                for line in lines:
                    parts = line.split("\t")
                    if len(parts) < 6:
                        continue
                    name_a, name_b, score = parts[2], parts[3], float(parts[5])
                    # Keep only edges that actually touch the queried protein — STRING's network
                    # endpoint returns a small interconnected neighbourhood, not a star graph, so
                    # most raw edges are between two *other* proteins (confirmed empirically).
                    if preferred_name not in (name_a, name_b):
                        continue
                    partner = name_b if name_a == preferred_name else name_a
                    partners.append((partner, score))
                partners.sort(key=lambda p: -p[1])
                for partner, score in partners[:TOP_PARTNERS]:
                    rows.append({"uniprot": acc, "protein_name": preferred_name,
                                 "partner_name": partner, "combined_score": score})
        except requests.RequestException:
            pass
        if i % 200 == 0:
            print(f"  STRING: {i:,}/{len(accessions):,} ({len(rows):,} edges collected)", flush=True)
        time.sleep(0.1)

    print(f"  STRING: done, {len(rows):,} interaction edges from {len(accessions):,} accessions queried")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "string_interactions.csv", index=False)
    print(f"  Saved {len(df):,} rows -> string_interactions.csv")


if __name__ == "__main__":
    main()
