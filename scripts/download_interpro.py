#!/usr/bin/env python3
"""
download_interpro.py — pull InterPro entry metadata into data/corpus/interpro/ for RAG ingestion.

chatPDB already has PDB-chain -> InterPro-ID mappings via SIFTS (sifts_pdb_interpro.csv from
scripts/download_rcsb.py), but not what those InterPro IDs actually *are* — domain/family name,
type, GO terms, member database cross-references. This script fetches that entry-level metadata.

Source: InterPro REST API (confirmed live 2026-07-15): https://www.ebi.ac.uk/interpro/api/entry/interpro/
Cursor-paginated, page_size=200 confirmed to work (54,190 total entries -> ~271 pages). Full prose
descriptions require a separate per-entry call each (54k+ requests — too expensive for a bulk pull);
this script fetches the list-level metadata (name, type, GO terms, member-database cross-refs)
which covers every entry cheaply and is enough to answer "what is InterPro domain IPR000971"-style
questions. Per-entry prose descriptions are a documented future enhancement (see PROJECT_PLAN.md
section 4), scoped down to just the ~31k InterPro IDs actually referenced in the PDB corpus if
pursued, not all 54k.

Usage:
    python scripts/download_interpro.py
    python scripts/download_interpro.py --limit 1000   # smoke test
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

OUT = Path("data/corpus/interpro")
BASE_URL = "https://www.ebi.ac.uk/interpro/api/entry/interpro/"
PAUSE = 0.2


def flatten_member_dbs(member_databases: dict | None) -> str:
    if not member_databases:
        return ""
    parts = []
    for db, entries in member_databases.items():
        for acc, name in (entries or {}).items():
            parts.append(f"{db}:{acc} ({name})")
    return " | ".join(parts)


def flatten_go_terms(go_terms: list | None) -> str:
    if not go_terms:
        return ""
    return " | ".join(f"{g['identifier']} {g['name']} [{g['category']['code']}]" for g in go_terms)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="stop after this many entries (smoke test)")
    parser.add_argument("--page-size", type=int, default=200)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "chatPDB/1.0 (protein-structure-rag; marc@marcdeller.com)", "Connection": "close"}

    print("\n[1/1] InterPro entries ...", flush=True)
    url = f"{BASE_URL}?page_size={args.page_size}"
    rows = []
    total = None
    page_num = 0
    while url:
        # No requests.Session() / connection pooling: a pooled keep-alive connection to this
        # host was observed to silently go dead mid-run (ESTABLISHED, 0% CPU, no data, for
        # 16+ minutes — well past any configured timeout) with no exception raised to trigger
        # a retry. A plain per-request GET with Connection: close sidesteps reusing a dead
        # socket entirely, at the cost of a new TCP+TLS handshake per page (acceptable for
        # ~271 pages).
        for attempt in range(3):
            try:
                r = requests.get(url, headers=headers, timeout=(10, 30))
                r.raise_for_status()
                data = r.json()
                break
            except requests.RequestException as e:
                print(f"    [warn] page {page_num} attempt {attempt + 1}: {e}", flush=True)
                time.sleep(2)
        else:
            print(f"    [warn] giving up on page {page_num} after 3 attempts, stopping", flush=True)
            break
        page_num += 1
        total = total or data.get("count")
        for item in data.get("results", []):
            meta = item.get("metadata", {})
            rows.append({
                "accession": meta.get("accession", ""),
                "name": meta.get("name", ""),
                "type": meta.get("type", ""),
                "source_database": meta.get("source_database", ""),
                "integrated": meta.get("integrated") or "",
                "member_databases": flatten_member_dbs(meta.get("member_databases")),
                "go_terms": flatten_go_terms(meta.get("go_terms")),
            })
        url = data.get("next")
        print(f"  InterPro: page {page_num}, {len(rows):,}/{total:,}", flush=True)
        if page_num % 20 == 0:
            pd.DataFrame(rows).to_csv(OUT / "interpro_entries.csv", index=False)
            print(f"    (checkpoint saved at {len(rows):,} rows)", flush=True)
        if args.limit and len(rows) >= args.limit:
            break
        time.sleep(PAUSE)

    print(f"  InterPro: {len(rows):,} entries fetched          ")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "interpro_entries.csv", index=False)
    print(f"  Saved {len(df):,} rows -> interpro_entries.csv")


if __name__ == "__main__":
    main()
