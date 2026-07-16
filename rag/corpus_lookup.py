#!/usr/bin/env python3
"""
corpus_lookup.py — deterministic, non-embedding lookup over the chatPDB corpus.

Same rationale as chem_sage's rag/corpus_lookup.py: dense embedding search over chunks that
each concatenate ~50-500 table rows is good at finding topically-similar chunks, but bad at
pinpointing one exact row inside a chunk. A query like "what R-free was 102M solved at" embeds
close to *any* chunk about resolution/R-free, not specifically the chunk containing row 102M —
confirmed empirically 2026-07-15 (see PROJECT_PLAN.md Phase 2 notes). Exact-ID questions (a PDB
ID, a CCD comp_id) should go through this deterministic path instead of rag/retrieve.py.

Usage (Python API):
    from rag.corpus_lookup import lookup
    print(lookup("what resolution and R-free was 102M solved at?"))

CLI:
    python -m rag.corpus_lookup "102M"
"""

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CORPUS_ROOT = Path("data/corpus")

# 4-character PDB ID: digit followed by 3 alphanumeric characters (standard wwPDB format).
_PDB_ID_RE = re.compile(r"\b[0-9][A-Za-z0-9]{3}\b")
# UniProt accession, common 6-character form (e.g. P02185, P00533, Q9Y6K9).
_UNIPROT_RE = re.compile(r"\b[A-Z][0-9][A-Z0-9]{3}[0-9]\b")
# InterPro accession (e.g. IPR000971).
_INTERPRO_RE = re.compile(r"\bIPR\d{6}\b")

# Common query words that happen to also be valid (usually irrelevant) CCD codes — "PDB" itself
# is a real, if obscure, ligand entry. Same rationale as chem_sage's corpus_lookup.py
# _CAPS_STOPWORDS: don't let ordinary question vocabulary masquerade as a database ID.
_CAPS_STOPWORDS = {"PDB", "RCSB", "DNA", "RNA", "SIFTS", "CIF", "API", "ID", "EC", "GO", "CCD"}


@dataclass
class CorpusFile:
    filename: str
    key_col: str
    display_cols: list[str]
    subdir: str = "rcsb"
    case_insensitive: bool = True


# Registry of corpus files safe to load fully for exact-match lookup (all comfortably small).
# The two huge SIFTS files (GO, InterPro chain mappings; tens of millions of rows) are
# deliberately excluded — loading them in full for a single lookup would be slow; they're
# covered by semantic retrieval over the sampled/chunked copy in rag/retrieve.py instead.
REGISTRY: list[CorpusFile] = [
    CorpusFile("pdb_entries_enriched.csv", "pdb_id",
               ["title", "method", "resolution_A", "em_resolution_A", "r_free", "r_work",
                "atom_count", "polymer_instance_count", "nonpolymer_instance_count",
                "determination_methodology", "deposition_date", "space_group", "cell_a", "cell_b",
                "cell_c", "cell_alpha", "cell_beta", "cell_gamma", "crystallization_pH",
                "crystallization_temp_K", "diffraction_wavelength_A", "citation_title",
                "citation_journal", "citation_year", "citation_doi", "citation_pubmed_id",
                "organism", "taxonomy_id", "primary_sequence_length", "assembly_count"]),
    CorpusFile("pdb_all_entries.csv", "pdb_id",
               ["header", "compound", "source", "author_list", "deposition_date",
                "experiment_type", "resolution_A"]),
    CorpusFile("sifts_pdb_uniprot.csv", "PDB",
               ["CHAIN", "SP_PRIMARY", "PDB_BEG", "PDB_END", "SP_BEG", "SP_END"]),
    CorpusFile("sifts_pdb_pfam.csv", "PDB",
               ["CHAIN", "SP_PRIMARY", "PFAM_ID", "COVERAGE"]),
    CorpusFile("sifts_pdb_cath.csv", "PDB", ["CHAIN", "CATH_ID"]),
    CorpusFile("sifts_pdb_scop2.csv", "PDB", ["CHAIN", "SF_DOMID", "FA_DOMID"]),
    CorpusFile("sifts_pdb_enzyme.csv", "PDB", ["CHAIN", "ACCESSION", "EC_NUMBER"]),
    CorpusFile("pdb_ccd_full.csv", "comp_id",
               ["name", "formula", "formula_weight", "type", "smiles", "inchikey"],
               case_insensitive=False),
    # CATH domain classification is keyed by domain_id (e.g. "102mA00"), not a bare PDB ID, so
    # it's not reached by a direct PDB-ID lookup — lookup() chains it via sifts_pdb_cath's
    # CATH_ID column instead (see the two-hop join below).
    CorpusFile("cath_classification.csv", "domain_id",
               ["cath_code", "class_desc", "architecture_desc", "topology_desc", "homology_desc"],
               subdir="cath"),
    CorpusFile("interpro_entries.csv", "accession",
               ["name", "type", "member_databases", "go_terms"],
               subdir="interpro", case_insensitive=False),
    CorpusFile("pharos_targets.csv", "uniprot",
               ["name", "symbol", "tdl", "family", "top_diseases"],
               subdir="pharos", case_insensitive=False),
    CorpusFile("uniprot_entries.csv", "accession",
               ["protein_name", "gene_names", "organism", "function", "keywords"],
               subdir="uniprot", case_insensitive=False),
    # TWILIGHT is keyed by PDB ID but has one row per bound ligand instance, so a single PDB
    # entry with several ligands returns several rows here — this is correct, not a bug.
    CorpusFile("twilight_ligands.csv", "PDBID",
               ["LigNm", "ResNr", "RSCC", "OWAB", "Resol", "Rwork", "Rfree", "Valid"],
               subdir="twilight"),
]

_cache: dict[str, pd.DataFrame] = {}


def _load(cf: CorpusFile) -> pd.DataFrame | None:
    path = CORPUS_ROOT / cf.subdir / cf.filename
    if cf.filename in _cache:
        return _cache[cf.filename]
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype=str, low_memory=False).fillna("")
    _cache[cf.filename] = df
    return df


def extract_ids(query: str) -> list[str]:
    """Pull candidate database IDs (PDB, UniProt, InterPro, CCD comp) out of a query."""
    pdb_ids = _PDB_ID_RE.findall(query)
    interpro_ids = _INTERPRO_RE.findall(query)
    uniprot_ids = [w for w in _UNIPROT_RE.findall(query) if w not in pdb_ids]
    # All-caps standalone tokens (e.g. "ATP", "HEM") are likely CCD component IDs.
    caps = [w for w in re.findall(r"\b[A-Z0-9]{2,5}\b", query)
            if w not in pdb_ids and w not in _CAPS_STOPWORDS]
    seen: list[str] = []
    for candidate in pdb_ids + interpro_ids + uniprot_ids + caps:
        if candidate not in seen:
            seen.append(candidate)
    return seen


def lookup_id(entry_id: str) -> dict[str, list[dict]]:
    """Look up one ID across every registered corpus file. Returns {filename: [matching rows]}.

    Includes one chained (two-hop) join: a PDB-ID match against sifts_pdb_cath.csv yields a
    CATH_ID (e.g. "102mA00"), which is then looked up in cath_classification.csv to attach the
    actual fold description — CATH domain IDs aren't PDB IDs themselves, so a direct lookup
    would never reach that file."""
    results: dict[str, list[dict]] = {}
    for cf in REGISTRY:
        df = _load(cf)
        if df is None or cf.key_col not in df.columns:
            continue
        if cf.case_insensitive:
            mask = df[cf.key_col].str.lower() == entry_id.lower()
        else:
            mask = df[cf.key_col] == entry_id
        matches = df[mask]
        if not matches.empty:
            cols = [cf.key_col] + [c for c in cf.display_cols if c in matches.columns]
            results[cf.filename] = matches[cols].to_dict(orient="records")

    cath_hits = results.get("sifts_pdb_cath.csv")
    if cath_hits:
        cath_cf = next(cf for cf in REGISTRY if cf.filename == "cath_classification.csv")
        cath_df = _load(cath_cf)
        if cath_df is not None:
            domain_ids = {row["CATH_ID"] for row in cath_hits if row.get("CATH_ID")}
            joined = cath_df[cath_df["domain_id"].isin(domain_ids)]
            if not joined.empty:
                cols = [cath_cf.key_col] + cath_cf.display_cols
                results["cath_classification.csv"] = joined[cols].to_dict(orient="records")
    return results


def lookup(query: str) -> str:
    """High-level entry point: extract candidate IDs from a free-text query, look each up,
    and render a plain-text grounded answer with source file attribution — or an explicit
    'not found' rather than staying silent, so a caller never mistakes an empty result for
    'not looked up'."""
    ids = extract_ids(query)
    if not ids:
        return "No PDB ID or CCD component ID found in the query."

    blocks = []
    for entry_id in ids:
        results = lookup_id(entry_id)
        if not results:
            continue
        for filename, rows in results.items():
            for row in rows:
                fields = "\n".join(f"    {k}: {v}" for k, v in row.items()
                                   if k not in ("pdb_id", "comp_id", "PDB", "domain_id", "accession", "uniprot"))
                blocks.append(f"[Source: {filename}, exact match on '{entry_id}']\n{fields}")
    if not blocks:
        return f"No corpus match for: {', '.join(ids)}"
    return "\n\n".join(blocks)


if __name__ == "__main__":
    import sys
    print(lookup(" ".join(sys.argv[1:]) or "102M"))
