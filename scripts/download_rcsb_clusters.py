#!/usr/bin/env python3
"""
download_rcsb_clusters.py — pull RCSB's precomputed sequence-identity cluster files into
data/corpus/clusters/ for RAG ingestion and SFT generation.

Without this, chatPDB can't answer "how many genuinely distinct structures of this protein exist"
or "is this a unique fold or the 400th lysozyme" -- questions an expert asks reflexively. RCSB
publishes weekly DIAMOND-based clustering across all PDB polymer entities at several identity
thresholds; a chain's cluster membership at 100% (near-identical sequence) distinguishes a genuine
re-determination from a routine re-deposit, and at 30% distinguishes real structural novelty from a
close homolog.

Access confirmed live 2026-07-17 (round 4 research): bulk flat files at
`https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-{N}.txt` for N in
30/40/50/70/90/95/100 (percent identity). One cluster per line, members space-separated as
`PDBID_entityNum` (polymer-entity IDs, not bare PDB IDs or chain IDs). Not documented on RCSB's own
clustering docs page -- confirmed only by direct fetch, so this URL pattern could move; check before
assuming it still works in a future round.

Usage:
    python scripts/download_rcsb_clusters.py
    python scripts/download_rcsb_clusters.py --thresholds 30 100   # smoke test / subset
"""

import argparse
from pathlib import Path

import pandas as pd
import requests

OUT = Path("data/corpus/clusters")
URL_TEMPLATE = "https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-{n}.txt"
HEADERS = {"User-Agent": "chatPDB/1.0 (protein-structure-rag; marc@marcdeller.com)", "Connection": "close"}
ALL_THRESHOLDS = [30, 40, 50, 70, 90, 95, 100]


def fetch_threshold(n: int) -> pd.DataFrame:
    r = requests.get(URL_TEMPLATE.format(n=n), headers=HEADERS, timeout=(10, 120))
    r.raise_for_status()
    rows = []
    for cluster_id, line in enumerate(r.text.strip().split("\n")):
        members = line.split()
        for member in members:
            pdb_id, _, entity = member.partition("_")
            rows.append({"pdb_id": pdb_id.upper(), "entity": entity, "cluster_id": cluster_id,
                          "cluster_size": len(members)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--thresholds", type=int, nargs="+", default=ALL_THRESHOLDS)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for n in args.thresholds:
        print(f"  fetching {n}% identity clusters ...")
        df = fetch_threshold(n)
        df.to_csv(OUT / f"clusters_{n}pct.csv", index=False)
        n_clusters = df["cluster_id"].nunique()
        n_singletons = (df.groupby("cluster_id")["pdb_id"].transform("size") == 1).sum()
        print(f"    {len(df):,} entity rows, {n_clusters:,} clusters, {n_singletons:,} singletons "
              f"-> clusters_{n}pct.csv")


if __name__ == "__main__":
    main()
