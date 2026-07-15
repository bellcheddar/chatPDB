#!/usr/bin/env python3
"""
download_cath.py — pull the CATH structural domain classification hierarchy into
data/corpus/cath/ for RAG ingestion.

chatPDB already has PDB-chain -> CATH-domain-ID mappings via SIFTS (sifts_pdb_cath.csv from
scripts/download_rcsb.py), but not what those domain IDs actually *mean* — the Class/
Architecture/Topology/Homology descriptions ("Orthogonal Bundle", "Beta Barrel", etc.) that make
CATH useful for answering "what fold is this?" This script fetches that classification hierarchy
and joins it to a per-domain CATH code.

Source: CATH's own HTTPS mirror of its FTP releases (confirmed live 2026-07-15):
  https://download.cathdb.info/cath/releases/latest-release/cath-classification-data/
    cath-names.txt        — CATH code -> text description (e.g. "1.10" -> "Orthogonal Bundle")
    cath-domain-list.txt  — CATH domain ID -> numeric Class.Architecture.Topology.Homology code
                             (CATH List File format 2.0; columns confirmed against a live sample)

Usage:
    python scripts/download_cath.py
"""

from pathlib import Path

import pandas as pd
import requests

OUT = Path("data/corpus/cath")
CATH_BASE = "https://download.cathdb.info/cath/releases/latest-release/cath-classification-data"

S = requests.Session()
S.headers["User-Agent"] = "chatPDB/1.0 (protein-structure-rag; marc@marcdeller.com)"


def download_text(url: str, label: str) -> str:
    print(f"  Downloading {label} ...")
    r = S.get(url, timeout=120)
    r.raise_for_status()
    return r.text


def parse_names(text: str) -> dict[str, str]:
    """cath-names.txt: '<code>    <example_domain>    :<description>' per line."""
    names: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        left, desc = line.split(":", 1)
        code = left.split()[0].strip()
        names[code] = desc.strip()
    return names


def parse_domain_list(text: str) -> pd.DataFrame:
    """cath-domain-list.txt (CLF format 2.0): domain_id class arch topology homology
    s35 s60 s95 s100 s100count length resolution — confirmed against a live sample."""
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 11:
            continue
        rows.append({
            "domain_id": parts[0],
            "cath_class": parts[1],
            "cath_architecture": parts[2],
            "cath_topology": parts[3],
            "cath_homology": parts[4],
            "length": parts[10],
            "resolution_A": parts[11] if len(parts) > 11 else "",
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("\n[1/1] CATH classification hierarchy ...")
    names_text = download_text(f"{CATH_BASE}/cath-names.txt", "cath-names.txt")
    domains_text = download_text(f"{CATH_BASE}/cath-domain-list.txt", "cath-domain-list.txt")

    names = parse_names(names_text)
    df = parse_domain_list(domains_text)
    print(f"  Parsed {len(df):,} classified domains, {len(names):,} named CATH codes")

    def code(row, depth) -> str:
        levels = [row["cath_class"], row["cath_architecture"], row["cath_topology"], row["cath_homology"]]
        return ".".join(levels[:depth])

    df["class_desc"] = df.apply(lambda r: names.get(code(r, 1), ""), axis=1)
    df["architecture_desc"] = df.apply(lambda r: names.get(code(r, 2), ""), axis=1)
    df["topology_desc"] = df.apply(lambda r: names.get(code(r, 3), ""), axis=1)
    df["homology_desc"] = df.apply(lambda r: names.get(code(r, 4), ""), axis=1)
    df["cath_code"] = df.apply(lambda r: code(r, 4), axis=1)

    out_cols = ["domain_id", "cath_code", "class_desc", "architecture_desc", "topology_desc",
                "homology_desc", "length", "resolution_A"]
    df[out_cols].to_csv(OUT / "cath_classification.csv", index=False)
    print(f"  Saved {len(df):,} rows -> cath_classification.csv")

    # A small standalone code->description table is useful on its own (e.g. "what does
    # architecture 1.10 mean?") without joining through a specific domain.
    df_names = pd.DataFrame([{"cath_code": k, "description": v} for k, v in names.items()])
    df_names.to_csv(OUT / "cath_code_descriptions.csv", index=False)
    print(f"  Saved {len(df_names):,} rows -> cath_code_descriptions.csv")


if __name__ == "__main__":
    main()
