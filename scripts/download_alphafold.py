#!/usr/bin/env python3
"""
download_alphafold.py — pull AlphaFold DB predictions into data/corpus/alphafold/ for RAG
ingestion and SFT generation.

This is the single biggest gap from the original corpus brainstorm (PROJECT_PLAN.md section 4):
chatPDB's entire identity rests on "reasons about real structures, is not a structure predictor" —
but until now there was no predicted-structure data in the corpus at all to contrast against, so
that boundary could only be trained as a refusal, never as an actual comparison ("here is what
AlphaFold predicts for this region, here is what the real structure shows"). This script closes
that gap.

Source (confirmed live 2026-07-16): https://alphafold.ebi.ac.uk/api/prediction/{uniprot_accession}
Per-accession REST API, rich response: per-region pLDDT confidence breakdown
(fractionPlddtVeryLow/Low/Confident/VeryHigh), overall confidence (globalMetricValue), model
version/creation date, and file URLs (mmCIF/PDB/PAE).

Scope: not all ~214M AlphaFold predictions, and not even all 73,910 UniProt accessions already
cross-referenced to a PDB structure (sifts_pdb_uniprot.csv) — that would take ~5-6 hours for
marginal extra value. Same pattern as scripts/download_pharos.py: ranked by how many PDB structures
already reference each accession (the intersection of "has a real structure" and "is well-studied
enough to be worth a predicted/experimental comparison" is exactly what matters here), capped to a
generous top N.

Usage:
    python scripts/download_alphafold.py                  # top 15,000 by PDB structure count
    python scripts/download_alphafold.py --top-n 500       # smoke test
"""

import argparse
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

OUT = Path("data/corpus/alphafold")
SIFTS_UNIPROT = Path("data/corpus/rcsb/sifts_pdb_uniprot.csv")
API_BASE = "https://alphafold.ebi.ac.uk/api/prediction"
HEADERS = {"User-Agent": "chatPDB/1.0 (protein-structure-rag; marc@marcdeller.com)", "Connection": "close"}


def top_uniprot_accessions(n: int) -> list[str]:
    df = pd.read_csv(SIFTS_UNIPROT, usecols=["SP_PRIMARY"], dtype=str)
    counts = Counter(df["SP_PRIMARY"].dropna())
    return [acc for acc, _ in counts.most_common(n)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top-n", type=int, default=15000)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"\n[1/1] AlphaFold DB predictions for top {args.top_n} PDB-cross-referenced UniProt accessions ...")
    accessions = top_uniprot_accessions(args.top_n)
    print(f"  {len(accessions):,} accessions to query")

    rows = []
    for i, acc in enumerate(accessions, 1):
        # No requests.Session(): confirmed pattern on this project now — pooled connections can
        # silently hang against any host on a long-running loop (EBI, then data.rcsb.org).
        try:
            r = requests.get(f"{API_BASE}/{acc}", headers=HEADERS, timeout=(10, 20))
            if r.status_code == 200:
                data = r.json()
                if data:
                    d = data[0]
                    rows.append({
                        "uniprot": acc,
                        "af_entry_id": d.get("entryId", ""),
                        "global_plddt": d.get("globalMetricValue"),
                        "fraction_plddt_very_low": d.get("fractionPlddtVeryLow"),
                        "fraction_plddt_low": d.get("fractionPlddtLow"),
                        "fraction_plddt_confident": d.get("fractionPlddtConfident"),
                        "fraction_plddt_very_high": d.get("fractionPlddtVeryHigh"),
                        "sequence_length": (d.get("sequenceEnd") or 0) - (d.get("sequenceStart") or 0) + 1,
                        "model_created_date": d.get("modelCreatedDate", ""),
                        "latest_version": d.get("latestVersion"),
                        "gene": d.get("gene", ""),
                        "organism": d.get("organismScientificName", ""),
                        "is_reviewed": d.get("isUniProtReviewed"),
                        "cif_url": d.get("cifUrl", ""),
                        "pae_image_url": d.get("paeImageUrl", ""),
                    })
        except requests.RequestException:
            pass
        if i % 500 == 0:
            print(f"  AlphaFold: {i:,}/{len(accessions):,} ({len(rows):,} matched)", flush=True)
        time.sleep(0.1)

    print(f"  AlphaFold: {len(rows):,}/{len(accessions):,} accessions matched a prediction          ")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "alphafold_predictions.csv", index=False)
    print(f"  Saved {len(df):,} rows -> alphafold_predictions.csv")
    if not df.empty:
        high_conf = (df["global_plddt"] >= 90).sum()
        low_conf = (df["global_plddt"] < 50).sum()
        print(f"  Confidence spread: {high_conf:,} very-high (pLDDT>=90), {low_conf:,} low (pLDDT<50)")


if __name__ == "__main__":
    main()
