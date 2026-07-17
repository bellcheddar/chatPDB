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
    # BindingDB's pdb_ids column holds a comma-separated list of PDB IDs per row (one binding
    # measurement can be co-crystallized in several depositions) — an exact-match mask would never
    # hit those rows, so this flags key_col for a split-and-contains match instead.
    multi_value: bool = False


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
    # Round 3 sources: AlphaFold DB, BindingDB, wwPDB validation (via PDBe), STRING.
    CorpusFile("alphafold_predictions.csv", "uniprot",
               ["af_entry_id", "global_plddt", "fraction_plddt_very_low", "fraction_plddt_low",
                "fraction_plddt_confident", "fraction_plddt_very_high", "sequence_length",
                "model_created_date", "gene", "organism", "is_reviewed", "cif_url"],
               subdir="alphafold", case_insensitive=False),
    CorpusFile("bindingdb_pdb_affinities.csv", "pdb_ids",
               ["ligand_name", "ligand_het_id", "target_name", "target_organism", "ki_nM",
                "ic50_nM", "kd_nM", "ec50_nM", "assay_pH", "assay_temp_C", "article_doi", "pmid",
                "uniprot_primary"],
               subdir="bindingdb", multi_value=True),
    CorpusFile("wwpdb_validation.csv", "pdb_id",
               ["percent_rama_outliers", "percent_rama_outliers_percentile", "percent_rota_outliers",
                "percent_rota_outliers_percentile", "clashscore", "clashscore_percentile"],
               subdir="validation"),
    CorpusFile("string_interactions.csv", "uniprot",
               ["protein_name", "partner_name", "combined_score"],
               subdir="string", case_insensitive=False),
    # Round 4 sources: PDB-REDO, EMDB, SCOP2, MobiDB, OPM, obsolete entries, AlphaFraud
    # (staged/partial), citation verification.
    CorpusFile("pdbredo_metadata.csv", "pdb_id",
               ["rfact", "rfree", "rffin", "rffinunb", "rffinz", "sigrffin", "dataresh",
                "spacegroup", "wavelength", "version", "bnet"],
               subdir="pdbredo"),
    CorpusFile("emdb_map_metadata.csv", "pdb_id",
               ["emdb_id", "method", "resolution_A", "resolution_method", "contour_level",
                "pixel_spacing_x_A", "space_group", "dim_col", "dim_row", "dim_sec"],
               subdir="emdb"),
    CorpusFile("scop2_domain_names.csv", "pdb_id",
               ["chain", "level", "node_id", "node_name", "fold_name", "class_name"],
               subdir="scop2"),
    CorpusFile("mobidb_disorder.csv", "accession",
               ["length", "source", "content_fraction", "content_count", "regions"],
               subdir="mobidb", case_insensitive=False),
    CorpusFile("opm_membrane_placement.csv", "pdb_id",
               ["half_bilayer_thickness_A", "has_membrane_dummy_atoms"],
               subdir="opm"),
    CorpusFile("obsolete_entries.csv", "obsolete_id",
               ["removed_date", "superseded_by", "title"],
               subdir="obsolete"),
    # AlphaFraud comparisons are keyed by PDB entry_id, staged/partial (backfill still running).
    CorpusFile("alphafraud_comparisons.csv", "entry_id",
               ["uniprot", "uniprot_name", "resolution", "method", "tm_by_experiment", "lddt",
                "gdt_ts", "ca_rmsd", "fraud_score", "confidently_wrong", "mean_plddt", "is_novel"],
               subdir="alphafraud"),
    CorpusFile("citation_verification.csv", "doi",
               ["bucket", "crossref_title", "title_similarity", "pmid_doi_match"],
               subdir="citations", case_insensitive=False),
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
        if cf.multi_value:
            needle = entry_id.lower() if cf.case_insensitive else entry_id
            def _contains(cell: str, _needle=needle, _ci=cf.case_insensitive) -> bool:
                parts = [p.strip() for p in cell.split(",")]
                if _ci:
                    parts = [p.lower() for p in parts]
                return _needle in parts
            mask = df[cf.key_col].apply(_contains)
        elif cf.case_insensitive:
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
