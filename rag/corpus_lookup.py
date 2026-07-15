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

CORPUS_DIR = Path("data/corpus/rcsb")

# 4-character PDB ID: digit followed by 3 alphanumeric characters (standard wwPDB format).
_PDB_ID_RE = re.compile(r"\b[0-9][A-Za-z0-9]{3}\b")
# CCD component ID: 1-5 alphanumeric characters, all upper-case as typically written/queried.
_COMP_ID_RE = re.compile(r"\b[A-Z0-9]{1,5}\b")

# Common query words that happen to also be valid (usually irrelevant) CCD codes — "PDB" itself
# is a real, if obscure, ligand entry. Same rationale as chem_sage's corpus_lookup.py
# _CAPS_STOPWORDS: don't let ordinary question vocabulary masquerade as a database ID.
_CAPS_STOPWORDS = {"PDB", "RCSB", "DNA", "RNA", "SIFTS", "CIF", "API", "ID", "EC", "GO", "CCD"}


@dataclass
class CorpusFile:
    filename: str
    key_col: str
    display_cols: list[str]
    case_insensitive: bool = True


# Registry of corpus files safe to load fully for exact-match lookup (all comfortably small).
# The two huge SIFTS files (GO, InterPro; tens of millions of rows) are deliberately excluded —
# loading them in full for a single lookup would be slow; they're covered by semantic retrieval
# over the sampled/chunked copy in rag/retrieve.py instead. A future pass could add an indexed
# (e.g. sqlite) lookup for those two if exact GO/InterPro lookups turn out to matter in practice.
REGISTRY: list[CorpusFile] = [
    CorpusFile("pdb_entries_enriched.csv", "pdb_id",
               ["title", "method", "resolution_A", "em_resolution_A", "r_free", "r_work",
                "atom_count", "polymer_instance_count", "nonpolymer_instance_count",
                "determination_methodology", "deposition_date"]),
    CorpusFile("pdb_all_entries.csv", "pdb_id",
               ["header", "compound", "source", "author_list", "deposition_date",
                "experiment_type", "resolution_A"]),
    CorpusFile("sifts_pdb_uniprot.csv", "PDB",
               ["CHAIN", "SP_PRIMARY", "PDB_BEG", "PDB_END", "SP_BEG", "SP_END"]),
    CorpusFile("sifts_pdb_pfam.csv", "PDB",
               ["CHAIN", "PFAM_ID", "PFAM_NAME", "PDB_BEG", "PDB_END"]),
    CorpusFile("sifts_pdb_cath.csv", "PDB", ["CHAIN", "CATH_ID"]),
    CorpusFile("sifts_pdb_scop2.csv", "PDB", ["CHAIN", "SF_DOMID", "FA_DOMID"]),
    CorpusFile("sifts_pdb_enzyme.csv", "PDB", ["CHAIN", "ACCESSION", "EC_NUMBER"]),
    CorpusFile("pdb_ccd_full.csv", "comp_id",
               ["name", "formula", "formula_weight", "type", "smiles", "inchikey"],
               case_insensitive=False),
]

_cache: dict[str, pd.DataFrame] = {}


def _load(cf: CorpusFile) -> pd.DataFrame | None:
    path = CORPUS_DIR / cf.filename
    if cf.filename in _cache:
        return _cache[cf.filename]
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype=str, low_memory=False).fillna("")
    _cache[cf.filename] = df
    return df


def extract_ids(query: str) -> list[str]:
    """Pull candidate PDB IDs (and bare CCD comp_ids for all-caps tokens) out of a query."""
    ids = _PDB_ID_RE.findall(query)
    # All-caps standalone tokens (e.g. "ATP", "HEM") are likely CCD component IDs.
    caps = [w for w in re.findall(r"\b[A-Z0-9]{2,5}\b", query)
            if w not in ids and w not in _CAPS_STOPWORDS]
    seen: list[str] = []
    for candidate in ids + caps:
        if candidate not in seen:
            seen.append(candidate)
    return seen


def lookup_id(entry_id: str) -> dict[str, list[dict]]:
    """Look up one ID across every registered corpus file. Returns {filename: [matching rows]}."""
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
                fields = "\n".join(f"    {k}: {v}" for k, v in row.items() if k not in ("pdb_id", "comp_id", "PDB"))
                blocks.append(f"[Source: {filename}, exact match on '{entry_id}']\n{fields}")
    if not blocks:
        return f"No corpus match for: {', '.join(ids)}"
    return "\n\n".join(blocks)


if __name__ == "__main__":
    import sys
    print(lookup(" ".join(sys.argv[1:]) or "102M"))
