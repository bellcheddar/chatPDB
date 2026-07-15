#!/usr/bin/env python3
"""
download_rcsb.py — pull RCSB PDB + SIFTS data into data/corpus/rcsb/ for RAG ingestion.

Downloads:
  1. All PDB entry metadata (~220k structures: header, compound, source, deposition/release dates,
     resolution) from the wwPDB derived-data index files — fast, no GraphQL needed.
  2. Structural-method enrichment for those entries via RCSB GraphQL: experimental method, R-free/
     R-work, EM map resolution, polymer/nucleic-acid entity counts, structure determination
     methodology. This is the data chatPDB's "experimental method interpretation" behaviour class
     is grounded in.
  3. The full Chemical Component Dictionary (CCD) — every ligand/monomer definition with SMILES/
     InChI, parsed directly from wwPDB's bulk components.cif.gz with gemmi (all ~50k blocks in a
     few seconds). Not a drug-like subset — chatPDB needs the whole structural dictionary; chem_sage's
     download_pdb.py already covers a drug-like slice for its own chemistry-focused corpus.
     (An earlier version of this step enumerated IDs from a wwPDB index file and round-tripped each
     through RCSB GraphQL; that index file (cc-counts.tdd) no longer exists and the Search API
     fallback's query shape has also gone stale. Parsing the bulk CIF directly sidesteps both.)
  4. SIFTS cross-references (from EBI, columns kept exactly as published): PDB -> UniProt chain
     mapping, PDB -> Pfam, PDB -> CATH, PDB -> SCOP2, PDB -> EC number, PDB -> GO terms,
     PDB -> InterPro. Feeds chatPDB's "database cross-referencing" behaviour class. (chem_sage's
     download_pdb.py used a combined pdb_chain_cath_scop.csv.gz that EBI has since retired in favour
     of separate CATH/SCOP2 files — verified against the live directory listing 2026-07-15.)

APIs used:
  - RCSB PDB GraphQL      https://data.rcsb.org/graphql
  - wwPDB derived data    https://files.wwpdb.org/pub/pdb/derived_data/
  - wwPDB CCD bulk file   https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz
  - EBI SIFTS             https://ftp.ebi.ac.uk/pub/databases/msd/sifts/

RCSB rate limit: ~10 requests/s. Script sleeps 0.15s between GraphQL requests, same as chem_sage's
download_pdb.py (whose helpers this ports directly).

Validation-report metrics (clashscore, Ramachandran outliers) are NOT available through this
GraphQL path (PdbxVrptSummary doesn't expose them at the entry level, confirmed by introspection
2026-07-15) — that's the separate "wwPDB validation reports" per-entry XML source noted in
PROJECT_PLAN.md section 4, not bundled here.

Usage:
    python scripts/download_rcsb.py                  # full pull (entries + CCD + SIFTS)
    python scripts/download_rcsb.py --limit 500       # smoke test: only enrich first 500 entries
    python scripts/download_rcsb.py --skip-ccd --skip-sifts   # entries only, fast
"""

from __future__ import annotations

import argparse
import gzip
import io
import time
from pathlib import Path

import gemmi
import pandas as pd
import requests

RCSB_GQL = "https://data.rcsb.org/graphql"
WWPDB = "https://files.wwpdb.org/pub/pdb/derived_data"
WWPDB_MONOMERS = "https://files.wwpdb.org/pub/pdb/data/monomers"
SIFTS_BASE = "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv"

S = requests.Session()
S.headers["User-Agent"] = "chatPDB/1.0 (protein-structure-rag; marc@marcdeller.com)"
PAUSE = 0.15


# ---------------------------------------------------------------------------
# Helpers (ported from chem_sage/scripts/download_pdb.py)
# ---------------------------------------------------------------------------

def download_text(url: str, label: str, timeout: int = 300) -> str:
    print(f"  Downloading {label} ...")
    try:
        r = S.get(url, timeout=timeout, stream=True)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"    [warn] {label}: {e}")
        return ""


def download_gz_csv(url: str, label: str, comment_char: str | None = None) -> pd.DataFrame:
    """Download a gzip-compressed CSV from SIFTS/EBI."""
    print(f"  Downloading {label} ...")
    try:
        r = S.get(url, timeout=600, stream=True)
        r.raise_for_status()
        buf = io.BytesIO(r.content)
        with gzip.open(buf) as gz:
            df = pd.read_csv(gz, comment=comment_char, low_memory=False)
        print(f"  {label}: {len(df):,} rows raw")
        return df
    except Exception as e:
        print(f"    [warn] {label}: {e}")
        return pd.DataFrame()


def graphql(query: str, variables: dict | None = None) -> dict:
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    try:
        r = S.post(RCSB_GQL, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
        r.raise_for_status()
        data = r.json()
        if data.get("errors"):
            print(f"    [warn] GraphQL errors: {data['errors'][:1]}")
        return data.get("data", {})
    except Exception as e:
        print(f"    [warn] GraphQL: {e}")
        return {}


# ---------------------------------------------------------------------------
# GraphQL: entry structural-method enrichment.
# Field names confirmed live against the RCSB schema 2026-07-15 (1CRN/6VXX/1UBQ smoke test):
# refine, em_3d_reconstruction, and the rcsb_entry_info/polymer_entity_count_* fields below all
# resolve; pdbx_vrpt_summary does NOT expose clashscore/rama_outliers at this level (introspected).
# ---------------------------------------------------------------------------

ENTRY_ENRICH_GQL = """
query GetEntries($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    struct { title }
    exptl { method }
    struct_keywords { pdbx_keywords text }
    rcsb_entry_info {
      resolution_combined
      deposited_atom_count
      deposited_polymer_entity_instance_count
      deposited_nonpolymer_entity_instance_count
      structure_determination_methodology
      polymer_entity_count_protein
      polymer_entity_count_nucleic_acid
    }
    refine {
      ls_R_factor_R_free
      ls_R_factor_R_work
      ls_d_res_high
    }
    em_3d_reconstruction { resolution }
    pdbx_database_status { recvd_initial_deposition_date status_code }
  }
}
"""

def flatten_entry(entry: dict) -> dict:
    exptl = (entry.get("exptl") or [{}])
    exptl0 = exptl[0] if exptl else {}
    info = entry.get("rcsb_entry_info") or {}
    refine = entry.get("refine") or [{}]
    refine0 = refine[0] if refine else {}
    em = entry.get("em_3d_reconstruction") or [{}]
    em0 = em[0] if em else {}
    status = entry.get("pdbx_database_status") or {}
    kw = entry.get("struct_keywords") or {}
    return {
        "pdb_id": entry.get("rcsb_id", ""),
        "title": (entry.get("struct") or {}).get("title", ""),
        "method": exptl0.get("method", ""),
        "keywords": kw.get("text", ""),
        "resolution_A": (info.get("resolution_combined") or [None])[0],
        "em_resolution_A": em0.get("resolution"),
        "r_free": refine0.get("ls_R_factor_R_free"),
        "r_work": refine0.get("ls_R_factor_R_work"),
        "refine_res_high_A": refine0.get("ls_d_res_high"),
        "atom_count": info.get("deposited_atom_count"),
        "polymer_instance_count": info.get("deposited_polymer_entity_instance_count"),
        "nonpolymer_instance_count": info.get("deposited_nonpolymer_entity_instance_count"),
        "protein_entity_count": info.get("polymer_entity_count_protein"),
        "nucleic_acid_entity_count": info.get("polymer_entity_count_nucleic_acid"),
        "determination_methodology": info.get("structure_determination_methodology", ""),
        "deposition_date": status.get("recvd_initial_deposition_date", ""),
        "status_code": status.get("status_code", ""),
    }


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step1_entry_index(out: Path) -> list[str]:
    print("\n[1/4] All PDB entry metadata (derived data index files) ...")
    entries_txt = download_text(f"{WWPDB}/index/entries.idx", "entries.idx")
    resolu_txt = download_text(f"{WWPDB}/index/resolu.idx", "resolu.idx")

    # Column layout confirmed live 2026-07-15 (curl the raw file and read the header row before
    # trusting this — wwPDB has changed it before): IDCODE, HEADER, ACCESSION DATE, COMPOUND,
    # SOURCE, AUTHOR LIST, RESOLUTION, EXPERIMENT TYPE (empty for X-ray). Different from, and
    # NOT interchangeable with, chem_sage's download_pdb.py column assumption (IDCODE HEADER
    # COMPOUND SOURCE AUTHOR DEPOSITION RELEASE REVTYPE) — that layout produced silently
    # misaligned fields when ported as-is; caught by cross-checking against the GraphQL-sourced
    # pdb_entries_enriched.csv for a known entry (102M) during Phase 2 corpus QA.
    rows = []
    for line in entries_txt.splitlines():
        if not line.strip() or line.startswith("IDCODE") or line.startswith("---"):
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        if parts:
            pdb_id = parts[0].strip().upper()
            if len(pdb_id) == 4:
                rows.append({
                    "pdb_id": pdb_id,
                    "header": parts[1].strip() if len(parts) > 1 else "",
                    "deposition_date": parts[2].strip() if len(parts) > 2 else "",
                    "compound": parts[3].strip() if len(parts) > 3 else "",
                    "source": parts[4].strip() if len(parts) > 4 else "",
                    "author_list": parts[5].strip() if len(parts) > 5 else "",
                    "experiment_type": parts[7].strip() if len(parts) > 7 else "",
                })

    resolu_map = {}
    for line in resolu_txt.splitlines():
        if not line.strip() or line.startswith("IDCODE") or line.startswith("--"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                resolu_map[parts[0].upper()] = float(parts[-1])
            except ValueError:
                pass

    df = pd.DataFrame(rows)
    if df.empty:
        print("  [warn] entries.idx parse failed, falling back to RCSB holdings endpoint ...")
        try:
            ids = S.get("https://data.rcsb.org/rest/v1/holdings/current/entry_ids", timeout=120).json()
        except Exception as e:
            print(f"    [warn] holdings fallback failed: {e}")
            ids = []
        df = pd.DataFrame({"pdb_id": ids})
    else:
        df["resolution_A"] = df["pdb_id"].map(resolu_map)

    df.to_csv(out / "pdb_all_entries.csv", index=False)
    print(f"  Saved {len(df):,} entries -> pdb_all_entries.csv")
    return df["pdb_id"].dropna().tolist()


def step2_entry_enrichment(out: Path, pdb_ids: list[str], batch_size: int = 50) -> None:
    print(f"\n[2/4] Structural-method enrichment for {len(pdb_ids):,} entries via GraphQL ...")
    rows = []
    for i in range(0, len(pdb_ids), batch_size):
        batch = pdb_ids[i : i + batch_size]
        data = graphql(ENTRY_ENRICH_GQL, {"ids": batch})
        for entry in data.get("entries") or []:
            if entry:
                rows.append(flatten_entry(entry))
        done = min(i + batch_size, len(pdb_ids))
        if done % 2000 < batch_size:
            print(f"  Enrichment: {done:,}/{len(pdb_ids):,} ({len(rows):,} records)", end="\r")
        time.sleep(PAUSE)
    print(f"  Enrichment: {len(rows):,} records fetched          ")
    df = pd.DataFrame(rows)
    df.to_csv(out / "pdb_entries_enriched.csv", index=False)
    print(f"  Saved {len(df):,} rows -> pdb_entries_enriched.csv")


def _best_smiles(desc_rows: list[tuple]) -> tuple[str, str, str]:
    """Pick one SMILES/InChI/InChIKey out of a component's descriptor rows.
    Each row is (comp_id, type, program, descriptor). Prefer OpenEye's canonical
    SMILES (wwPDB's own curated toolkit), then any SMILES_CANONICAL, then any SMILES."""
    by_type: dict[str, list[tuple]] = {}
    for row in desc_rows:
        _, dtype, program, descriptor = row
        by_type.setdefault(dtype, []).append((program, descriptor.strip('"')))

    smiles = ""
    for candidates in (by_type.get("SMILES_CANONICAL", []), by_type.get("SMILES", [])):
        if not candidates:
            continue
        openeye = next((d for p, d in candidates if "OpenEye" in p), None)
        smiles = openeye or candidates[0][1]
        if smiles:
            break
    inchi = (by_type.get("InChI", [("", "")])[0][1]) if by_type.get("InChI") else ""
    inchikey = (by_type.get("InChIKey", [("", "")])[0][1]) if by_type.get("InChIKey") else ""
    return smiles, inchi, inchikey


def step3_full_ccd(out: Path) -> None:
    """Full Chemical Component Dictionary, parsed directly from wwPDB's bulk
    components.cif.gz (every ligand/monomer definition, not a drug-like subset).
    gemmi parses all ~50k blocks of this file in a few seconds — far more reliable
    than enumerating IDs from an index file and round-tripping each through GraphQL."""
    print("\n[3/4] Full Chemical Component Dictionary (all ligands/monomers, not just drug-like) ...")
    local_path = out / "_components.cif.gz"
    print("  Downloading components.cif.gz (~115 MB) ...")
    try:
        r = S.get(f"{WWPDB_MONOMERS}/components.cif.gz", timeout=600, stream=True)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk_bytes in r.iter_content(chunk_size=1 << 20):
                f.write(chunk_bytes)
    except Exception as e:
        print(f"    [warn] components.cif.gz: {e}")
        return

    print("  Parsing with gemmi ...")
    doc = gemmi.cif.read(str(local_path))
    rows = []
    for block in doc:
        comp_id = block.name
        name = block.find_pair("_chem_comp.name")
        formula = block.find_pair("_chem_comp.formula")
        fw = block.find_pair("_chem_comp.formula_weight")
        ctype = block.find_pair("_chem_comp.type")
        status = block.find_pair("_chem_comp.pdbx_release_status")
        desc_rows = list(block.find("_pdbx_chem_comp_descriptor.", ["comp_id", "type", "program", "descriptor"]))
        smiles, inchi, inchikey = _best_smiles([tuple(r) for r in desc_rows])
        rows.append({
            "comp_id": comp_id,
            "name": (name[1] if name else "").strip('"'),
            "formula": (formula[1] if formula else "").strip('"'),
            "formula_weight": (fw[1] if fw else None),
            "type": (ctype[1] if ctype else "").strip('"'),
            "release_status": (status[1] if status else ""),
            "smiles": smiles,
            "inchi": inchi,
            "inchikey": inchikey,
        })

    local_path.unlink(missing_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "pdb_ccd_full.csv", index=False)
    print(f"  Saved {len(df):,} CCD entries -> pdb_ccd_full.csv (full dictionary, no drug-like filter)")


def _fetch_sifts(out: Path, filename: str, out_name: str, dedup_subset: list[str] | None = None) -> None:
    """Download one SIFTS flatfile as-is (columns kept exactly as EBI publishes them —
    verified live against the current flatfiles/csv/ listing 2026-07-15, since several
    of these have been renamed/split since chem_sage's download_pdb.py was written)."""
    df = download_gz_csv(f"{SIFTS_BASE}/{filename}.csv.gz", f"SIFTS {filename}", comment_char="#")
    if df.empty:
        return
    if dedup_subset:
        df = df.drop_duplicates(subset=[c for c in dedup_subset if c in df.columns])
    df.to_csv(out / f"{out_name}.csv", index=False)
    print(f"  Saved {len(df):,} rows -> {out_name}.csv")


def step4_sifts(out: Path) -> None:
    print("\n[4/4] SIFTS PDB cross-references (UniProt, Pfam, CATH, SCOP2, EC, GO, InterPro) ...")
    _fetch_sifts(out, "pdb_chain_uniprot", "sifts_pdb_uniprot", ["PDB", "CHAIN", "SP_PRIMARY"])
    _fetch_sifts(out, "pdb_chain_pfam", "sifts_pdb_pfam")
    # pdb_chain_cath_scop.csv.gz was retired; CATH and SCOP2 are now separate files.
    _fetch_sifts(out, "pdb_chain_cath_uniprot", "sifts_pdb_cath", ["PDB", "CHAIN", "CATH_ID"])
    _fetch_sifts(out, "pdb_chain_scop2_uniprot", "sifts_pdb_scop2")
    # Bonus cross-references, same file shape/helper, directly matching the "database
    # cross-referencing" behaviour class: EC number (enzyme function), GO terms, InterPro domains.
    _fetch_sifts(out, "pdb_chain_enzyme", "sifts_pdb_enzyme", ["PDB", "CHAIN", "ACCESSION", "EC_NUMBER"])
    _fetch_sifts(out, "pdb_chain_go", "sifts_pdb_go")
    _fetch_sifts(out, "pdb_chain_interpro", "sifts_pdb_interpro")


def print_summary(out: Path) -> None:
    print("\n=== RCSB corpus summary ===")
    total_kb = 0.0
    for f in sorted(out.iterdir()):
        if f.is_file():
            kb = f.stat().st_size / 1024
            total_kb += kb
            try:
                n = sum(1 for _ in open(f)) - 1
                row_hint = f"  ({n:,} rows)"
            except Exception:
                row_hint = ""
            print(f"  {f.name:35s}  {kb:10.1f} KB{row_hint}")
    print(f"\n  Total: {total_kb / 1024:.1f} MB")
    print("\nNext: python scripts/ingest_rag.py --corpus data/corpus --store .chroma --reset")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("data/corpus/rcsb"))
    parser.add_argument("--limit", type=int, default=None,
                         help="only enrich the first N entries (smoke test); CCD/SIFTS still run in full")
    parser.add_argument("--skip-enrichment", action="store_true")
    parser.add_argument("--skip-ccd", action="store_true")
    parser.add_argument("--skip-sifts", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pdb_ids = step1_entry_index(args.out_dir)
    if args.limit:
        pdb_ids = pdb_ids[: args.limit]

    if not args.skip_enrichment:
        step2_entry_enrichment(args.out_dir, pdb_ids)
    if not args.skip_ccd:
        step3_full_ccd(args.out_dir)
    if not args.skip_sifts:
        step4_sifts(args.out_dir)

    print_summary(args.out_dir)


if __name__ == "__main__":
    main()
