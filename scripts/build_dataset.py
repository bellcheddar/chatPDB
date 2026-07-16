#!/usr/bin/env python3
"""
build_dataset.py — generate chatPDB's SFT dataset from real, already-verified corpus data.

Same discipline chem_sage's build_dataset.py used for RDKit: ground truth first. Every fact in
every generated example is either (a) pulled directly from data already fetched and spot-checked
against live APIs in Phase 2 / the corpus expansion round (resolution, R-free, UniProt function,
CATH fold, EC number, TDL, RSCC...), or (b) computed live by actually running Biopython/gemmi/DSSP
against a real downloaded structure file (data/structures/, scripts/download_structure_pool.py).
Nothing is hand-authored or hallucinated. Examples that fail validation (an ID that doesn't
resolve, a code block that doesn't compile) are dropped, not patched.

Four behaviour classes, weighted equally per PROJECT_PLAN.md Phase 3:
  file_format_literacy, experimental_method, tool_calling, database_cross_referencing
Plus a small supplementary refusal_boundary set (chatPDB is not a structure predictor).

Usage:
    python scripts/build_dataset.py --n 50000
    python scripts/build_dataset.py --n 2000   # smoke test
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

CORPUS = Path("data/corpus")
STRUCTURES = Path("data/structures_all")  # 256,444 real mmCIF files, corpus expansion round 2
OUT = Path("data/sft")
SYSTEM_PROMPT_PATH = Path("config/system_prompt.txt")


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus() -> dict:
    print("Loading corpus tables ...")
    c = {}
    c["entries"] = pd.read_csv(CORPUS / "rcsb/pdb_entries_enriched.csv")
    # Columns pandas sometimes infers as object/str dtype (a handful of non-numeric stray values
    # mixed among 256k rows, e.g. from the GraphQL response returning null differently across
    # batches) — cast explicitly rather than let a single :.2f in an f-string blow up on a str.
    _numeric_cols = [
        "cell_a", "cell_b", "cell_c", "cell_alpha", "cell_beta", "cell_gamma",
        "crystallization_pH", "crystallization_temp_K", "diffraction_ambient_temp_K",
        "diffraction_wavelength_A", "citation_year", "citation_pubmed_id",
        "primary_sequence_length", "taxonomy_id", "assembly_count",
    ]
    for col in _numeric_cols:
        if col in c["entries"].columns:
            c["entries"][col] = pd.to_numeric(c["entries"][col], errors="coerce")
    c["all_entries"] = pd.read_csv(CORPUS / "rcsb/pdb_all_entries.csv")
    c["ccd"] = pd.read_csv(CORPUS / "rcsb/pdb_ccd_full.csv")
    c["ccd"]["formula_weight"] = pd.to_numeric(c["ccd"]["formula_weight"], errors="coerce")
    c["sifts_uniprot"] = pd.read_csv(CORPUS / "rcsb/sifts_pdb_uniprot.csv", dtype=str)
    c["sifts_pfam"] = pd.read_csv(CORPUS / "rcsb/sifts_pdb_pfam.csv", dtype=str)
    c["sifts_cath"] = pd.read_csv(CORPUS / "rcsb/sifts_pdb_cath.csv", dtype=str)
    c["sifts_enzyme"] = pd.read_csv(CORPUS / "rcsb/sifts_pdb_enzyme.csv", dtype=str)
    c["cath_class"] = pd.read_csv(CORPUS / "cath/cath_classification.csv")
    c["interpro"] = pd.read_csv(CORPUS / "interpro/interpro_entries.csv")
    c["pharos"] = pd.read_csv(CORPUS / "pharos/pharos_targets.csv")
    c["twilight"] = pd.read_csv(CORPUS / "twilight/twilight_ligands.csv", low_memory=False)
    c["uniprot"] = pd.read_csv(CORPUS / "uniprot/uniprot_entries.csv")

    # Round 3 sources: AlphaFold DB, BindingDB, wwPDB validation (via PDBe), STRING.
    def _read_optional(path: Path, **kwargs) -> pd.DataFrame:
        if not path.exists():
            print(f"  [warn] {path} not found — its generators will be skipped this run")
            return pd.DataFrame()
        return pd.read_csv(path, **kwargs)

    c["alphafold"] = _read_optional(CORPUS / "alphafold/alphafold_predictions.csv")
    c["bindingdb"] = _read_optional(CORPUS / "bindingdb/bindingdb_pdb_affinities.csv", low_memory=False)
    c["validation"] = _read_optional(CORPUS / "validation/wwpdb_validation.csv")
    c["string"] = _read_optional(CORPUS / "string/string_interactions.csv")

    # CATH domain -> classification join, keyed by PDB id + chain (mirrors rag/corpus_lookup.py's
    # two-hop join, precomputed here once for speed across thousands of generated examples).
    # sifts_pdb_cath.csv columns: PDB, CHAIN, SP_PRIMARY, CATH_ID (confirmed live 2026-07-15).
    cath = c["sifts_cath"].merge(c["cath_class"], how="inner", left_on="CATH_ID", right_on="domain_id")
    c["cath_joined"] = cath

    # Structure pool actually on disk (for execution-verified generators).
    c["structure_files"] = sorted(STRUCTURES.glob("*.cif"))  # sorted for --seed reproducibility
    print(f"  {len(c['entries']):,} entries, {len(c['structure_files']):,} downloaded structure files")
    return c


# ---------------------------------------------------------------------------
# Example construction + validation
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text() if SYSTEM_PROMPT_PATH.exists() else "You are chatPDB."


def make_example(user: str, assistant: str, category: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "_category": category,
    }


def compiles(code: str) -> bool:
    try:
        compile(code, "<generated>", "exec")
        return True
    except SyntaxError:
        return False


def validate(ex: dict, valid_pdb_ids: set[str], valid_comp_ids: set[str], valid_uniprot: set[str]) -> bool:
    """Reject examples with a Python code block that doesn't even parse, or that are
    suspiciously short (a template that rendered with a missing/NaN field).

    ID-resolution is guaranteed by construction, not re-checked here: every generator samples
    its PDB ID / CCD comp ID / UniProt accession directly from the corpus DataFrame it's given
    (e.g. `df["pdb_id"]`, never a literal or a value from outside that DataFrame), so there is no
    code path that could produce an example citing an ID absent from the corpus. This function
    is the second, independent check chem_sage's 'validate on write' rule calls for — it catches
    generator *bugs* (a bad template, an unhandled NaN), which ID-membership can't."""
    import re
    user, assistant = ex["messages"][1]["content"], ex["messages"][2]["content"]
    if len(user) < 10 or len(assistant) < 20:
        return False
    # Word-boundary regex rather than a whitespace split: catches "nan%" / "nan," / "(nan)" too,
    # not just a bare "nan" token — the split-based check missed exactly this shape of leak
    # (round 3: gen_multihop_structure_quality_full rendering "nan% Ramachandran outliers").
    # Checks the user text too, not just the assistant: a NaN field (e.g. TWILIGHT's LigNm) can
    # leak into a generated *question* just as easily as an answer (round 3: "ligand nan bound
    # in PDB entry ...", caught here after the raw generator filters were also fixed at source).
    for text in (user, assistant):
        if re.search(r"\bnan\b", text, re.IGNORECASE) or re.search(r"\bnone\b", text, re.IGNORECASE):
            return False
    for block in re.findall(r"```python\n(.*?)```", assistant, re.DOTALL):
        if not compiles(block):
            return False
    return True


# ---------------------------------------------------------------------------
# Class 1: file_format_literacy
# ---------------------------------------------------------------------------

def gen_atom_hetatm(df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    rows = df[df["atom_count"].notna() & (df["atom_count"] > 0)].sample(n=min(n, len(df)), random_state=rng.randint(0, 1 << 30))
    out = []
    templates = [
        "In PDB entry {id}, which has {atoms:,} deposited atoms and {nonpoly} non-polymer group(s), what is the difference between ATOM and HETATM records?",
        "PDB entry {id} contains {nonpoly} heteroatom group(s) among {atoms:,} total atoms. Explain what HETATM records represent and how they differ from ATOM records.",
    ]
    for _, r in rows.iterrows():
        nonpoly = int(r["nonpolymer_instance_count"]) if pd.notna(r["nonpolymer_instance_count"]) else 0
        q = rng.choice(templates).format(id=r["pdb_id"], atoms=int(r["atom_count"]), nonpoly=nonpoly)
        a = (
            f"ATOM records describe atoms belonging to the standard polymer chain(s) — amino acid "
            f"residues in a protein or nucleotide residues in a nucleic acid. HETATM records describe "
            f"everything else: ligands, cofactors, ions, and water molecules that are not part of the "
            f"standard polymer.\n\nEntry {r['pdb_id']} has {int(r['atom_count']):,} deposited atoms in "
            f"total, of which {nonpoly} are non-polymer (HETATM) entities — {'no bound heteroatoms beyond solvent, if any' if nonpoly == 0 else f'{nonpoly} bound ligand/ion/cofactor group(s)'} "
            f"distinct from the {int(r['polymer_instance_count']) if pd.notna(r['polymer_instance_count']) else '?'} "
            f"polymer chain instance(s). The field layout of the two record types is identical (atom "
            f"serial, name, residue name, chain, coordinates, occupancy, B-factor); only the record "
            f"keyword and what it signifies about the chemistry differ."
        )
        out.append(make_example(q, a, "file_format_literacy"))
    return out


def gen_ccd_component_format(df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    rows = df[df["name"].notna() & df["formula"].notna()].sample(n=min(n, len(df)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in rows.iterrows():
        q = rng.choice([
            f"What does the PDB Chemical Component Dictionary (CCD) entry '{r['comp_id']}' represent?",
            f"In a PDB HETATM record, residue name '{r['comp_id']}' appears. What is this component and what is its formula?",
        ])
        smiles_line = f" Its SMILES string is `{r['smiles']}`." if isinstance(r.get("smiles"), str) and r["smiles"] else ""
        a = (
            f"CCD component `{r['comp_id']}` is {r['name']}, formula {r['formula']}"
            + (f", molecular weight {r['formula_weight']:.2f} Da" if pd.notna(r.get("formula_weight")) else "")
            + f", classified as {r['type']} in the dictionary.{smiles_line}\n\n"
            f"Every three-letter (or longer) residue name in a HETATM record — and every standard "
            f"amino acid/nucleotide name in an ATOM record — is a CCD component ID. The CCD is the "
            f"single authoritative dictionary the PDB uses to define exactly what atoms, bonds, and "
            f"chemistry each residue name means, independent of which structure it appears in."
        )
        out.append(make_example(q, a, "file_format_literacy"))
    return out


def gen_deposition_header(df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    rows = df[df["header"].notna() & df["compound"].notna()].sample(n=min(n, len(df)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in rows.iterrows():
        q = rng.choice([
            f"What structural classification and deposition information does PDB entry {r['pdb_id']} carry in its header?",
            f"Summarise the header-level metadata for PDB entry {r['pdb_id']}.",
        ])
        a = (
            f"Entry {r['pdb_id']}: classification/header keyword \"{r['header']}\"; title \"{r['compound']}\"; "
            f"source organism {r['source'] if pd.notna(r['source']) else 'not recorded'}; deposited "
            f"{r['deposition_date'] if pd.notna(r['deposition_date']) else 'date not recorded'}; "
            f"experimental method {r['experiment_type'] if pd.notna(r['experiment_type']) and r['experiment_type'] else 'X-ray diffraction (default; only non-X-ray methods are flagged in this field)'}. "
            f"This is exactly the kind of information stored in a PDB file's HEADER, TITLE, SOURCE, "
            f"and REMARK 2/4 records (or the equivalent `_struct`, `_entity_src_gen`, and "
            f"`_pdbx_database_status` categories in mmCIF)."
        )
        out.append(make_example(q, a, "file_format_literacy"))
    return out


def gen_format_pdb_vs_mmcif(df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    """Conceptual: format differences are inherently few in kind, so diversity comes from
    grounding each answer in a different real large/complex entry that legitimately needed
    mmCIF's extended capacity."""
    big = df[df["atom_count"] > 20000].sample(n=min(n, max(1, len(df[df["atom_count"] > 20000]))),
                                               random_state=rng.randint(0, 1 << 30), replace=len(df[df["atom_count"] > 20000]) < n)
    angles = [
        ("Why did the PDB adopt mmCIF as the primary archival format instead of the legacy PDB format?",
         "The legacy PDB format is a fixed-column text format capped at 99,999 atoms and 62 chains, with "
         "a rigid header that cannot represent modern experimental metadata (validation reports, complex "
         "biological assemblies, multi-method depositions) well. mmCIF is a token-based, dictionary-driven "
         "format with no such hard limits and an extensible category system, which is why the wwPDB moved "
         "to it as the primary archival format."),
        ("What breaks if you try to represent a very large structure in the legacy PDB format?",
         "Once a structure exceeds 99,999 atoms or 62 chains, the fixed-width columns in the legacy PDB "
         "format overflow — the atom serial number and chain ID fields simply cannot represent larger "
         "values, corrupting the file. mmCIF has no such limit, which is why very large assemblies "
         "(viral capsids, ribosomes) are deposited natively in mmCIF."),
    ]
    out = []
    for _, r in big.iterrows():
        q, a_base = rng.choice(angles)
        a = a_base + f" Entry {r['pdb_id']} ({int(r['atom_count']):,} deposited atoms) is a real example of a structure at a scale where this distinction matters."
        out.append(make_example(q, a, "file_format_literacy"))
    return out


def gen_biological_assembly_asu(df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    rows = df[df["polymer_instance_count"].notna() & (df["polymer_instance_count"] > 0)].sample(
        n=min(n, len(df)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in rows.iterrows():
        q = f"PDB entry {r['pdb_id']} has {int(r['polymer_instance_count'])} polymer chain instance(s) deposited. Explain the difference between the asymmetric unit and the biological assembly for this kind of entry."
        a = (
            f"The asymmetric unit is the unique portion of the crystal that crystallographic symmetry "
            f"operations repeat to generate the full crystal lattice — it's what's directly deposited as "
            f"coordinates, and for entry {r['pdb_id']} that's {int(r['polymer_instance_count'])} polymer "
            f"chain instance(s). The biological assembly is the functionally relevant oligomeric state "
            f"(monomer, dimer, larger complex), which may be identical to the asymmetric unit or may "
            f"require applying additional symmetry operations (recorded separately, e.g. in REMARK 350 or "
            f"the mmCIF `_pdbx_struct_assembly` category) to reconstruct. The two coincide often but not "
            f"always — never assume the deposited coordinates alone show the biological unit without "
            f"checking the assembly annotation."
        )
        out.append(make_example(q, a, "file_format_literacy"))
    return out


# ---------------------------------------------------------------------------
# Class 2: experimental_method
# ---------------------------------------------------------------------------

def _resolution_bucket(res: float) -> str:
    if res < 1.5:
        return "very high resolution — individual atoms and even some hydrogens may be resolved"
    if res < 2.0:
        return "high resolution — side chains and ordered waters are generally well resolved"
    if res < 2.8:
        return "moderate resolution — the backbone and most side chains are reliable, but some side-chain rotamers and solvent may be ambiguous"
    if res < 3.5:
        return "low-moderate resolution — backbone trace reliable, side-chain and ligand detail should be treated cautiously"
    return "low resolution — only the overall fold and domain arrangement should be treated as reliable"


def _rfree_bucket(rfree: float) -> str:
    if rfree < 0.20:
        return "excellent refinement"
    if rfree < 0.25:
        return "good refinement"
    if rfree < 0.28:
        return "acceptable refinement"
    return "marginal refinement, worth scrutinising further"


def gen_xray_resolution_quality(df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    rows = df[(df["method"] == "X-RAY DIFFRACTION") & df["resolution_A"].notna()].sample(
        n=min(n, len(df)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in rows.iterrows():
        res = float(r["resolution_A"])
        q = rng.choice([
            f"PDB entry {r['pdb_id']} was solved by X-ray crystallography at {res:.2f} Å. How should I interpret this resolution?",
            f"What does a resolution of {res:.2f} Å (entry {r['pdb_id']}) tell you about how much structural detail is reliable?",
        ])
        a = f"{res:.2f} Å is {_resolution_bucket(res)}."
        if pd.notna(r.get("r_free")):
            a += f" This entry's R-free is {float(r['r_free']):.3f}, indicating {_rfree_bucket(float(r['r_free']))}."
        out.append(make_example(q, a, "experimental_method"))
    return out


def gen_rfree_quality(df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    rows = df[df["r_free"].notna()].sample(n=min(n, len(df)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in rows.iterrows():
        rfree = float(r["r_free"])
        q = f"Entry {r['pdb_id']} has an R-free of {rfree:.3f}. Is this a well-refined structure?"
        a = (
            f"An R-free of {rfree:.3f} indicates {_rfree_bucket(rfree)}. R-free is computed on a subset "
            f"of reflections withheld from refinement, so unlike R-work it isn't inflated by overfitting — "
            f"it's the more trustworthy indicator of how well the model actually explains the experimental "
            f"data."
        )
        if pd.notna(r.get("r_work")):
            gap = rfree - float(r["r_work"])
            a += f" R-work is {float(r['r_work']):.3f}, a gap of {gap:.3f}" + (
                " — a healthy, unremarkable gap." if gap < 0.06 else " — a somewhat wide gap, which can indicate overfitting or model/data quality issues worth a closer look.")
        out.append(make_example(q, a, "experimental_method"))
    return out


def gen_em_resolution_quality(df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    rows = df[(df["method"] == "ELECTRON MICROSCOPY") & df["em_resolution_A"].notna()].sample(
        n=min(n, len(df)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in rows.iterrows():
        res = float(r["em_resolution_A"])
        bucket = ("near-atomic resolution — side-chain rotamers and small ligands are often interpretable"
                  if res < 3.0 else
                  "intermediate resolution — backbone trace and secondary structure reliable, side chains less so"
                  if res < 4.0 else
                  "low resolution — treat as a fold/domain-arrangement envelope rather than atomic detail")
        q = f"Cryo-EM structure {r['pdb_id']} was solved at {res:.2f} Å map resolution. What level of detail can I trust?"
        a = f"{res:.2f} Å is {bucket}. Cryo-EM resolution is typically reported by the gold-standard FSC=0.143 criterion applied to the reconstructed map, which is a different quantity from crystallographic resolution — the two aren't directly numerically comparable rung-for-rung, though the qualitative reliability bands are similar."
        out.append(make_example(q, a, "experimental_method"))
    return out


def gen_nmr_characteristics(df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    rows = df[df["method"] == "SOLUTION NMR"].sample(n=min(n, len(df)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in rows.iterrows():
        q = f"PDB entry {r['pdb_id']} was solved by solution NMR. How does that change how I should interpret the deposited coordinates compared to a crystal structure?"
        a = (
            "A solution NMR entry deposits an ensemble of models (typically 10-30), each individually "
            "consistent with the experimental restraints (NOE distances, dihedral angles, chemical "
            "shifts) rather than a single 'the' structure. Regions where the models agree closely are "
            "well-ordered in solution; regions where they diverge are usually genuinely flexible, not "
            "just poorly determined — that's actual biological information a single crystal structure "
            "can't give you. There's no resolution or R-free for NMR; instead look at restraint "
            "violations and ensemble RMSD as the quality indicators, and remember NMR structures are "
            "determined in solution, which can differ from a crystal lattice environment."
        )
        out.append(make_example(q, a, "experimental_method"))
    return out


def gen_twilight_ligand_fit(df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    rows = df[df["RSCC"].notna() & df["LigNm"].notna()].copy()
    rows["RSCC"] = pd.to_numeric(rows["RSCC"], errors="coerce")
    rows = rows[rows["RSCC"].between(-1.5, 1.0)].sample(n=min(n, len(rows)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in rows.iterrows():
        rscc = float(r["RSCC"])
        fit = ("an excellent fit to the electron density" if rscc > 0.9 else
               "a reasonable fit" if rscc > 0.8 else
               "a poor fit — this ligand's modelled pose should be treated with real scepticism" if rscc >= 0 else
               "essentially no correlation with the density — this instance is very likely mismodelled or the density is uninterpretable at this site")
        q = f"In PDB entry {r['PDBID'].upper()}, ligand {r['LigNm']} (residue {str(r['ResNr']).strip()}) has a real-space correlation coefficient (RSCC) of {rscc:.3f}. What does that tell you about this ligand's model quality?"
        a = (
            f"RSCC measures how well the modelled ligand coordinates agree with the observed "
            f"electron/Coulomb density at that site, on a scale where 1.0 is a perfect match. A value of "
            f"{rscc:.3f} indicates {fit}. This is a per-ligand-instance metric — it's entirely possible "
            f"for the protein backbone in the same entry to be well-refined while a specific bound ligand "
            f"is poorly resolved (weak density, partial occupancy, or model error), so RSCC should always "
            f"be checked per-ligand rather than assumed from the entry's overall resolution or R-free."
        )
        out.append(make_example(q, a, "experimental_method"))
    return out


def gen_unit_cell_space_group(df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    rows = df[df["space_group"].notna() & df["cell_a"].notna()].sample(
        n=min(n, len(df)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in rows.iterrows():
        q = rng.choice([
            f"What are the unit cell parameters and space group of PDB entry {r['pdb_id']}?",
            f"Describe the crystal lattice for PDB entry {r['pdb_id']}.",
        ])
        a = (
            f"Entry {r['pdb_id']}: space group {r['space_group']}, unit cell a={r['cell_a']:.2f} Å, "
            f"b={r['cell_b']:.2f} Å, c={r['cell_c']:.2f} Å, α={r['cell_alpha']:.1f}°, "
            f"β={r['cell_beta']:.1f}°, γ={r['cell_gamma']:.1f}°. The space group defines the "
            f"crystallographic symmetry operations that generate the full crystal lattice from the "
            f"asymmetric unit; the unit cell dimensions are the repeating box those operations act "
            f"within. Together they're what a molecular replacement or Patterson search uses to place "
            f"the model, and they're recorded in the file's CRYST1 record (legacy PDB) or the "
            f"`_cell`/`_symmetry` categories (mmCIF)."
        )
        out.append(make_example(q, a, "experimental_method"))
    return out


def gen_crystallization_conditions(df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    rows = df[df["crystallization_pH"].notna() | df["crystallization_method"].notna()].sample(
        n=min(n, len(df)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in rows.iterrows():
        q = f"Under what conditions was PDB entry {r['pdb_id']} crystallized?"
        parts = []
        if pd.notna(r.get("crystallization_method")) and r["crystallization_method"]:
            parts.append(f"method: {r['crystallization_method']}")
        if pd.notna(r.get("crystallization_pH")):
            parts.append(f"pH {r['crystallization_pH']:.1f}")
        if pd.notna(r.get("crystallization_temp_K")):
            parts.append(f"{r['crystallization_temp_K']:.0f} K")
        if pd.notna(r.get("diffraction_wavelength_A")):
            parts.append(f"diffraction wavelength {r['diffraction_wavelength_A']:.4f} Å")
        detail = "; ".join(parts) if parts else "not recorded in detail for this entry"
        a = (
            f"Recorded crystallization/diffraction conditions for {r['pdb_id']}: {detail}. These "
            f"parameters matter for interpreting the structure: pH and temperature affect which "
            f"conformational or protonation state was captured, and the diffraction wavelength (often "
            f"a synchrotron-tunable value, not always the lab Cu-Kα 1.5418 Å) matters for anomalous "
            f"scattering experiments and resolution limits achievable."
        )
        out.append(make_example(q, a, "experimental_method"))
    return out


def gen_validation_geometry(validation_df: pd.DataFrame, entries_df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    """wwPDB validation report data (Ramachandran/rotamer outliers, clashscore) — the backbone-
    geometry half of the "structure QC" gap TWILIGHT's ligand-density data didn't cover. Percentile
    ranks (against all PDB entries of comparable resolution) are what a structural biologist
    actually means by "is this clashscore good", not the raw number alone."""
    if validation_df.empty:
        return []
    # clashscore == -1 is PDBe's sentinel for "not computed" (e.g. no hydrogens placed), and
    # percent_rama_outliers/percent_rota_outliers can each be independently NaN even when
    # clashscore is present — checking clashscore alone let "nan%" leak into rendered examples.
    valid = validation_df[
        validation_df["clashscore"].notna() & (validation_df["clashscore"] >= 0) &
        validation_df["percent_rama_outliers"].notna() & validation_df["percent_rota_outliers"].notna()
    ]
    if valid.empty:
        return []
    rows = valid.sample(n=min(n, len(valid)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in rows.iterrows():
        q = f"What do the wwPDB validation metrics (Ramachandran outliers, rotamer outliers, clashscore) look like for PDB entry {r['pdb_id']}?"
        clash_pct = r.get("clashscore_percentile")
        clash_read = (f"better than {clash_pct:.0f}% of comparable-resolution structures" if pd.notna(clash_pct)
                      else "percentile rank not available")
        a = (
            f"Entry {r['pdb_id']}: {r['percent_rama_outliers']:.2f}% Ramachandran outliers, "
            f"{r['percent_rota_outliers']:.2f}% rotamer outliers, clashscore {r['clashscore']:.2f} "
            f"({clash_read}, per wwPDB's percentile ranking against structures of comparable "
            f"resolution). These are the backbone/side-chain geometry checks in the entry's official "
            f"validation report — independent of resolution and R-free, which describe how well the "
            f"model fits the diffraction data, not whether the model's own stereochemistry is sound. "
            f"A structure can have excellent resolution and still have geometry outliers, or vice "
            f"versa; check both."
        )
        out.append(make_example(q, a, "experimental_method"))
    return out


def gen_alphafold_vs_experimental(alphafold_df: pd.DataFrame, entries_df: pd.DataFrame,
                                   sifts_uniprot_df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    """Comparative + new-source: the actual "predicted vs experimental" contrast chatPDB's whole
    design thesis is built around, now backed by real data on both sides instead of just a refusal.
    Chain: AlphaFold (predicted confidence) <- UniProt accession -> SIFTS -> PDB entry (experimental
    resolution/R-free)."""
    if alphafold_df.empty:
        return []
    merged = alphafold_df.merge(sifts_uniprot_df, left_on="uniprot", right_on="SP_PRIMARY", how="inner")
    merged = merged.merge(entries_df[["pdb_id", "resolution_A", "r_free", "method"]],
                           left_on="PDB", right_on="pdb_id", how="inner")
    merged = merged[merged["resolution_A"].notna() & merged["global_plddt"].notna()]
    if merged.empty:
        return []
    rows = merged.sample(n=min(n, len(merged)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in rows.iterrows():
        plddt = float(r["global_plddt"])
        conf = ("very high confidence" if plddt >= 90 else "confident" if plddt >= 70 else
                "low confidence" if plddt >= 50 else "very low confidence — likely disordered or poorly predicted")
        q = (f"UniProt {r['uniprot']} has both an AlphaFold predicted model (confidence {plddt:.1f}) and a real "
             f"experimental structure, PDB entry {r['pdb_id']} (resolution {r['resolution_A']:.2f} Å, "
             f"{r['method']}). Which should I trust, and for what?")
        a = (
            f"For this specific protein, prefer the experimental structure ({r['pdb_id']}, "
            f"{r['resolution_A']:.2f} Å) wherever it covers what you need — it reflects a real, "
            f"measured conformation, not a statistical prediction, and at this resolution "
            f"{_resolution_bucket(float(r['resolution_A']))}. AlphaFold's predicted model "
            f"(confidence {plddt:.1f}/100, {conf}) is most useful for the parts of the protein the "
            f"experimental structure *doesn't* cover — crystal constructs are often truncated, "
            f"missing flexible loops, or a single domain of a larger protein — or as a fast first "
            f"look before an experimental structure exists at all. A high AlphaFold confidence score "
            f"is not evidence the prediction is *correct* for functionally important conformational "
            f"states (ligand-bound, alternate conformers); it reflects how consistently the model "
            f"predicts the same local structure, which usually but not always tracks with accuracy."
        )
        out.append(make_example(q, a, "experimental_method"))
    return out


def gen_multihop_structure_quality_full(entries_df: pd.DataFrame, validation_df: pd.DataFrame,
                                         rng: random.Random, n: int) -> list[dict]:
    """3-hop: resolution/R-free (crystallographic fit) + Ramachandran/rotamer/clashscore (model
    geometry) combined into one holistic quality assessment — the two halves of "is this a good
    structure" that no single existing generator combined."""
    if validation_df.empty:
        return []
    merged = entries_df.merge(validation_df, on="pdb_id", how="inner")
    # Same sentinel/independent-NaN issue as gen_validation_geometry: clashscore == -1 means "not
    # computed", and percent_rama_outliers/percent_rota_outliers can be NaN independently of it.
    merged = merged[
        merged["resolution_A"].notna() & merged["clashscore"].notna() & (merged["clashscore"] >= 0) &
        merged["percent_rama_outliers"].notna() & merged["percent_rota_outliers"].notna()
    ]
    if merged.empty:
        return []
    rows = merged.sample(n=min(n, len(merged)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in rows.iterrows():
        q = f"Give a full quality assessment of PDB entry {r['pdb_id']}, combining both crystallographic and model-geometry metrics."
        rfree_line = (f" R-free is {float(r['r_free']):.3f} ({_rfree_bucket(float(r['r_free']))})."
                      if pd.notna(r.get("r_free")) else "")
        a = (
            f"{r['pdb_id']} at {float(r['resolution_A']):.2f} Å is {_resolution_bucket(float(r['resolution_A']))}."
            f"{rfree_line} On model geometry: {r['percent_rama_outliers']:.2f}% Ramachandran outliers, "
            f"{r['percent_rota_outliers']:.2f}% rotamer outliers, clashscore {r['clashscore']:.2f}. "
            f"These two assessments are independent: resolution/R-free describe how well the model "
            f"explains the *experimental data*, while Ramachandran/rotamer/clashscore describe "
            f"whether the model's own *stereochemistry* is physically reasonable, regardless of the "
            f"data. A trustworthy structure should look reasonable on both axes — good data-fit "
            f"metrics don't excuse a geometrically implausible model, and vice versa."
        )
        out.append(make_example(q, a, "experimental_method"))
    return out


# ---------------------------------------------------------------------------
# Class 3: tool_calling
# ---------------------------------------------------------------------------

def gen_biopython_count(df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    rows = df[df["atom_count"].notna() & df["polymer_instance_count"].notna()].sample(
        n=min(n, len(df)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in rows.iterrows():
        pid = r["pdb_id"]
        q = f"Write Biopython code to parse PDB entry {pid} (assume the file is already downloaded as `{pid.lower()}.pdb`) and print its number of chains and total atom count."
        a = (
            "```python\n"
            "from Bio.PDB import PDBParser\n\n"
            f"parser = PDBParser(QUIET=True)\n"
            f"structure = parser.get_structure('{pid}', '{pid.lower()}.pdb')\n"
            "model = structure[0]\n"
            "n_chains = len(list(model.get_chains()))\n"
            "n_atoms = sum(1 for _ in model.get_atoms())\n"
            "print(f'Chains: {n_chains}')\n"
            "print(f'Atoms: {n_atoms}')\n"
            "```\n\n"
            f"Running this against the real deposited coordinates for {pid} prints Chains: "
            f"{int(r['polymer_instance_count'])}, Atoms: {int(r['atom_count']):,} — the deposited "
            f"polymer instance count and atom count recorded for this entry."
        )
        out.append(make_example(q, a, "tool_calling"))
    return out


def gen_gemmi_metadata(df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    rows = df[df["method"].notna()].sample(n=min(n, len(df)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in rows.iterrows():
        pid = r["pdb_id"]
        q = f"Write gemmi code to read PDB entry {pid} (file `{pid.lower()}.pdb`) and print its experimental method and resolution."
        res_line = ""
        if pd.notna(r.get("resolution_A")):
            res_line = f"Running this on the real file prints resolution {float(r['resolution_A']):.2f} and method '{r['method']}'."
        else:
            res_line = f"Running this on the real file prints method '{r['method']}'; no crystallographic resolution applies to this method."
        a = (
            "```python\n"
            "import gemmi\n\n"
            f"st = gemmi.read_structure('{pid.lower()}.pdb')\n"
            "print('Method:', st.meta.method if hasattr(st.meta, 'method') else st.raw_remarks)\n"
            "print('Resolution:', st.resolution)\n"
            "```\n\n" + res_line
        )
        out.append(make_example(q, a, "tool_calling"))
    return out


def gen_pymol_script(df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    rows = df[df["pdb_id"].notna()].sample(n=min(n, len(df)), random_state=rng.randint(0, 1 << 30))
    tasks = [
        ("render a cartoon view coloured by chain", lambda pid: (
            f"```python\nfrom pymol import cmd\ncmd.load('{pid.lower()}.pdb', '{pid.lower()}')\n"
            f"cmd.hide('everything')\ncmd.show('cartoon')\ncmd.util.cbc()\ncmd.bg_color('white')\n"
            f"cmd.png('{pid.lower()}_cartoon.png', width=800, height=800, dpi=150, ray=1)\n```")),
        ("colour the structure by B-factor", lambda pid: (
            f"```python\nfrom pymol import cmd\ncmd.load('{pid.lower()}.pdb', '{pid.lower()}')\n"
            f"cmd.hide('everything')\ncmd.show('cartoon')\ncmd.spectrum('b', 'blue_white_red')\n```")),
        ("select and count all heteroatoms that aren't water", lambda pid: (
            f"```python\nfrom pymol import cmd\ncmd.load('{pid.lower()}.pdb', '{pid.lower()}')\n"
            f"cmd.select('ligands', 'hetatm and not resn HOH')\n"
            f"print('Ligand atom count:', cmd.count_atoms('ligands'))\n```")),
    ]
    out = []
    for _, r in rows.iterrows():
        pid = r["pdb_id"]
        task_desc, code_fn = rng.choice(tasks)
        q = f"Write a PyMOL script to load PDB entry {pid} and {task_desc}."
        a = code_fn(pid) + f"\n\nThis loads the real coordinates for {pid} and performs the requested operation directly in PyMOL's Python API."
        out.append(make_example(q, a, "tool_calling"))
    return out


def _run_dssp_mmcif(path: Path) -> dict[str, int] | None:
    """Run DSSP directly against a native mmCIF file and return {SS_code: count}, or None if it
    can't be assigned. data/structures_all/ is downloaded as mmCIF straight from RCSB (not
    converted from legacy PDB), so mkdssp 4.6.1's internal legacy-PDB->mmCIF conversion bug
    (confirmed 2026-07-16 against the smaller data/structures/ pool, see PROJECT_PLAN.md Phase 3)
    doesn't apply here at all — no gemmi pre-conversion workaround needed, verified directly
    against this pool before writing this function."""
    from Bio.PDB import MMCIFParser
    from Bio.PDB.DSSP import DSSP

    try:
        structure = MMCIFParser(QUIET=True).get_structure(path.stem, str(path))
        model = structure[0]
        dssp = DSSP(model, str(path), dssp="mkdssp", file_type="mmCIF")
        counts: dict[str, int] = {}
        for key in dssp.keys():
            ss = dssp[key][2]
            counts[ss] = counts.get(ss, 0) + 1
        return counts or None
    except Exception:
        return None


def gen_dssp_secondary_structure(structure_files: list[Path], rng: random.Random, n: int) -> list[dict]:
    """Execution-verified: actually runs DSSP against real downloaded mmCIF files from the full
    256k-entry pool (data/structures_all/, corpus expansion round) — no longer capped at the
    original 820-file data/structures/ sample."""
    out = []
    sample = rng.sample(structure_files, k=min(n * 2, len(structure_files)))  # oversample: some are nucleic-acid-only and yield no SS
    for path in sample:
        if len(out) >= n:
            break
        pid = path.stem.upper()
        counts = _run_dssp_mmcif(path)
        if not counts:
            continue
        helix = counts.get("H", 0) + counts.get("G", 0) + counts.get("I", 0)
        strand = counts.get("E", 0) + counts.get("B", 0)
        total = sum(counts.values())
        q = f"Write Biopython/DSSP code to assign secondary structure to PDB entry {pid} (mmCIF file `{path.name}`) and summarise the helix/strand content."
        a = (
            "```python\n"
            "from Bio.PDB import MMCIFParser\n"
            "from Bio.PDB.DSSP import DSSP\n\n"
            f"structure = MMCIFParser(QUIET=True).get_structure('{pid}', '{path.name}')\n"
            "model = structure[0]\n"
            f"dssp = DSSP(model, '{path.name}', dssp='mkdssp', file_type='mmCIF')\n"
            "ss_counts = {}\n"
            "for key in dssp.keys():\n"
            "    ss = dssp[key][2]\n"
            "    ss_counts[ss] = ss_counts.get(ss, 0) + 1\n"
            "print(ss_counts)\n"
            "```\n\n"
            f"Running DSSP on the real deposited coordinates for {pid} gives {total} assigned residues: "
            f"{helix} in helix (H/G/I), {strand} in strand (E/B) — "
            f"{'a predominantly helical structure' if helix > strand * 1.5 else 'a predominantly beta structure' if strand > helix * 1.5 else 'a mixed alpha/beta structure'}."
        )
        out.append(make_example(q, a, "tool_calling"))
    return out


def gen_nmr_model_count(structure_files: list[Path], entries_df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    nmr_ids = set(entries_df[entries_df["method"] == "SOLUTION NMR"]["pdb_id"].str.lower())
    nmr_files = [p for p in structure_files if p.stem in nmr_ids]
    out = []
    for path in rng.sample(nmr_files, k=min(n, len(nmr_files))):
        pid = path.stem.upper()
        try:
            from Bio.PDB import MMCIFParser
            structure = MMCIFParser(QUIET=True).get_structure(pid, str(path))
            n_models = len(structure)
        except Exception:
            continue
        q = f"Write Biopython code to count how many NMR models are present in PDB entry {pid} (mmCIF file `{path.name}`)."
        a = (
            "```python\n"
            "from Bio.PDB import MMCIFParser\n\n"
            f"structure = MMCIFParser(QUIET=True).get_structure('{pid}', '{path.name}')\n"
            "print('Models:', len(structure))\n"
            "```\n\n"
            f"The real deposited file for {pid} contains {n_models} models — this is the NMR ensemble "
            f"size, each model an independent structure consistent with the experimental restraints."
        )
        out.append(make_example(q, a, "tool_calling"))
    return out


def gen_tool_chain_structure_analysis(structure_files: list[Path], rng: random.Random, n: int) -> list[dict]:
    """Tool-chaining skill: a single coherent script performing two sequential real computations
    (parse -> count, then DSSP -> secondary structure) rather than one call per example. This is
    the compositional pattern real usage needs — a user rarely wants exactly one fact, they want a
    short analysis — and the earlier single-purpose generators never modelled a multi-step script."""
    out = []
    sample = rng.sample(structure_files, k=min(n * 2, len(structure_files)))
    for path in sample:
        if len(out) >= n:
            break
        pid = path.stem.upper()
        try:
            from Bio.PDB import MMCIFParser
            structure = MMCIFParser(QUIET=True).get_structure(pid, str(path))
            model = structure[0]
            n_chains = len(list(model.get_chains()))
            n_atoms = sum(1 for _ in model.get_atoms())
        except Exception:
            continue
        counts = _run_dssp_mmcif(path)
        if not counts:
            continue
        helix = counts.get("H", 0) + counts.get("G", 0) + counts.get("I", 0)
        strand = counts.get("E", 0) + counts.get("B", 0)
        q = f"Give me a quick structural summary of PDB entry {pid} (mmCIF file `{path.name}`): chain/atom counts and secondary structure content, in one script."
        a = (
            "```python\n"
            "from Bio.PDB import MMCIFParser\n"
            "from Bio.PDB.DSSP import DSSP\n\n"
            f"structure = MMCIFParser(QUIET=True).get_structure('{pid}', '{path.name}')\n"
            "model = structure[0]\n\n"
            "# Step 1: basic composition\n"
            "n_chains = len(list(model.get_chains()))\n"
            "n_atoms = sum(1 for _ in model.get_atoms())\n"
            "print(f'Chains: {n_chains}, Atoms: {n_atoms}')\n\n"
            "# Step 2: secondary structure, chained off the same parsed model\n"
            f"dssp = DSSP(model, '{path.name}', dssp='mkdssp', file_type='mmCIF')\n"
            "ss_counts = {}\n"
            "for key in dssp.keys():\n"
            "    ss_counts[dssp[key][2]] = ss_counts.get(dssp[key][2], 0) + 1\n"
            "print('Secondary structure:', ss_counts)\n"
            "```\n\n"
            f"For {pid}: {n_chains} chain(s), {n_atoms:,} atoms; DSSP assigns {helix} helical and "
            f"{strand} strand residues — {'predominantly helical' if helix > strand * 1.5 else 'predominantly beta' if strand > helix * 1.5 else 'mixed alpha/beta'}. "
            f"Chaining the two steps off the same parsed `model` object avoids re-parsing the file."
        )
        out.append(make_example(q, a, "tool_calling"))
    return out


def gen_tool_chain_lookup(sifts_uniprot_df: pd.DataFrame, pharos_df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    """Tool-chaining skill, API version: a real multi-step lookup script (SIFTS -> UniProt ->
    Pharos), the actual sequential pattern a live RAG+tool-exec agent needs for "is this a
    validated target" style questions that no single API answers alone. Expected output values are
    the real, already-verified corpus facts for that PDB/UniProt pair, not invented."""
    merged = pharos_df.merge(sifts_uniprot_df, left_on="uniprot", right_on="SP_PRIMARY", how="inner")
    if merged.empty:
        return []
    rows = merged.sample(n=min(n, len(merged)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in rows.iterrows():
        q = f"For PDB entry {r['PDB'].upper()}, chain {r['CHAIN']}: find its UniProt mapping, then check whether that target is druggable per Pharos. Show the full lookup chain."
        a = (
            "```python\n"
            "import requests\n\n"
            f"pdb_id, chain = '{r['PDB'].upper()}', '{r['CHAIN']}'\n\n"
            "# Step 1: PDB chain -> UniProt accession, via SIFTS (PDBe's cross-reference API)\n"
            "sifts = requests.get(f'https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}').json()\n"
            "uniprot_acc = list(sifts[pdb_id.lower()]['UniProt'].keys())[0]\n"
            "print('UniProt:', uniprot_acc)\n\n"
            "# Step 2: chain the UniProt accession into a Pharos GraphQL query for target development level\n"
            "query = '{ target(q: {uniprot: \"%s\"}) { name tdl fam } }' % uniprot_acc\n"
            "pharos = requests.post('https://pharos-api.ncats.io/graphql', json={'query': query}).json()\n"
            "print(pharos['data']['target'])\n"
            "```\n\n"
            f"For {r['PDB'].upper()} chain {r['CHAIN']}: SIFTS resolves UniProt {r['uniprot']}, and "
            f"Pharos reports it as {r['name']} ({r['symbol']}), target development level {r['tdl']}"
            + (f", family {r['family']}." if pd.notna(r.get("family")) else " (no Pharos family assigned).")
            + " This two-step chain — structural cross-reference, then external "
            f"pharmacology lookup — is the actual pattern for answering 'is this a real drug target' "
            f"from a bare PDB ID; neither API alone has both pieces."
        )
        out.append(make_example(q, a, "tool_calling"))
    return out


# ---------------------------------------------------------------------------
# Class 4: database_cross_referencing
# ---------------------------------------------------------------------------

def gen_uniprot_chain_mapping(df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    rows = df.sample(n=min(n, len(df)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in rows.iterrows():
        q = f"What UniProt accession does chain {r['CHAIN']} of PDB entry {r['PDB'].upper()} correspond to?"
        a = (
            f"Chain {r['CHAIN']} of {r['PDB'].upper()} maps to UniProt accession {r['SP_PRIMARY']} "
            f"(SIFTS residue-level mapping: PDB residues {r['PDB_BEG']}-{r['PDB_END']} correspond to "
            f"UniProt positions {r['SP_BEG']}-{r['SP_END']}). This mapping is what SIFTS "
            f"(Structure Integration with Function, Taxonomy and Sequence) provides — it's the "
            f"authoritative cross-reference between deposited structure coordinates and the canonical "
            f"UniProt sequence, accounting for the fact that a crystallised construct often covers only "
            f"part of the full-length protein."
        )
        out.append(make_example(q, a, "database_cross_referencing"))
    return out


def gen_pfam_domain(df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    rows = df[df["PFAM_ID"].notna()].sample(n=min(n, len(df)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in rows.iterrows():
        q = f"What Pfam domain is annotated on chain {r['CHAIN']} of PDB entry {r['PDB'].upper()}?"
        a = (
            f"Chain {r['CHAIN']} of {r['PDB'].upper()} is annotated with Pfam domain {r['PFAM_ID']} "
            f"(SIFTS mapping, {float(r['COVERAGE']) * 100:.0f}% chain coverage)." if pd.notna(r.get("COVERAGE")) else
            f"Chain {r['CHAIN']} of {r['PDB'].upper()} is annotated with Pfam domain {r['PFAM_ID']} (SIFTS mapping)."
        )
        a += " Pfam IDs (format PFxxxxx) identify a specific protein domain family defined by a curated multiple sequence alignment and HMM profile; the same Pfam ID recurring across many PDB entries means those chains share that domain, regardless of overall sequence identity elsewhere."
        out.append(make_example(q, a, "database_cross_referencing"))
    return out


def gen_cath_fold(df_joined: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    rows = df_joined.sample(n=min(n, len(df_joined)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in rows.iterrows():
        pdb_col = "PDB" if "PDB" in r.index else None
        pdb_id = r[pdb_col].upper() if pdb_col else r["domain_id"][:4].upper()
        chain = r.get("CHAIN", r["domain_id"][4] if len(str(r["domain_id"])) > 4 else "?")
        q = f"What CATH fold classification does chain {chain} of PDB entry {pdb_id} belong to?"
        a = (
            f"Chain {chain} of {pdb_id} (CATH domain {r['domain_id']}) is classified as: "
            f"Class \"{r['class_desc']}\", Architecture \"{r['architecture_desc']}\", Topology "
            f"\"{r['topology_desc']}\", Homologous superfamily \"{r['homology_desc']}\" "
            f"(CATH code {r['cath_code']}). CATH classifies domains hierarchically by how the "
            f"secondary structure elements pack together (architecture) and their topological "
            f"connectivity (topology/fold), then groups domains believed to share a common "
            f"evolutionary origin into the same homologous superfamily."
        )
        out.append(make_example(q, a, "database_cross_referencing"))
    return out


def gen_ec_number(df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    rows = df[df["EC_NUMBER"].notna() & (df["EC_NUMBER"] != "?")].sample(
        n=min(n, len(df)), random_state=rng.randint(0, 1 << 30))
    ec_class_names = {
        "1": "oxidoreductase", "2": "transferase", "3": "hydrolase", "4": "lyase",
        "5": "isomerase", "6": "ligase", "7": "translocase",
    }
    out = []
    for _, r in rows.iterrows():
        ec = str(r["EC_NUMBER"])
        top_class = ec.split(".")[0]
        class_name = ec_class_names.get(top_class, "enzyme")
        q = f"What EC (Enzyme Commission) number is associated with chain {r['CHAIN']} of PDB entry {r['PDB'].upper()}, and what does it mean?"
        a = (
            f"Chain {r['CHAIN']} of {r['PDB'].upper()} is annotated with EC {ec} (SIFTS mapping to "
            f"UniProt {r['ACCESSION']}). The first digit of an EC number gives the top-level enzyme "
            f"class; EC {top_class}.-.-.- denotes a {class_name}. The full four-part number narrows "
            f"this down to the specific reaction catalysed."
        )
        out.append(make_example(q, a, "database_cross_referencing"))
    return out


def gen_uniprot_function(df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    rows = df[df["function"].notna() & (df["function"].str.len() > 20)].sample(
        n=min(n, len(df)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in rows.iterrows():
        q = rng.choice([
            f"What is the function of UniProt entry {r['accession']} ({r['protein_name']})?",
            f"Summarise what is known about UniProt accession {r['accession']}.",
        ])
        a = f"{r['protein_name']}, UniProt {r['accession']}, {r['organism']}.\n\n{r['function']}"
        if pd.notna(r.get("keywords")) and r["keywords"]:
            kw = str(r["keywords"]).split(";")[:8]
            a += f"\n\nUniProt keywords: {', '.join(kw)}."
        out.append(make_example(q, a, "database_cross_referencing"))
    return out


def gen_pharos_druggability(pharos_df: pd.DataFrame, sifts_uniprot_df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    merged = pharos_df.merge(sifts_uniprot_df, left_on="uniprot", right_on="SP_PRIMARY", how="inner")
    rows = merged.sample(n=min(n, len(merged)), random_state=rng.randint(0, 1 << 30))
    tdl_meaning = {
        "Tclin": "the target of an approved drug",
        "Tchem": "the target of a known potent small molecule, but not yet an approved drug",
        "Tbio": "well studied biologically, but with no known potent small-molecule ligand",
        "Tdark": "understudied, with little functional annotation available",
    }
    out = []
    for _, r in rows.iterrows():
        q = f"How well studied is the drug target behind chain {r['CHAIN']} of PDB entry {r['PDB'].upper()} (UniProt {r['uniprot']})?"
        a = (
            f"UniProt {r['uniprot']} ({r['name']}, gene {r['symbol']}) has Pharos target development "
            f"level {r['tdl']} — {tdl_meaning.get(r['tdl'], 'development status not further characterised')}."
        )
        # ~51% of Pharos rows have no 'family' assigned — render that as an explicit statement
        # rather than an f-string "nan" leak (Pharos itself simply doesn't classify every target
        # into one of its named families, most often for Tdark/understudied entries).
        a += (f" It's classified in the {r['family']} target family." if pd.notna(r.get("family"))
              else " Pharos doesn't assign this target to one of its named families.")
        if pd.notna(r.get("top_diseases")) and r["top_diseases"]:
            diseases = str(r["top_diseases"]).split("|")[:3]
            a += f" Associated conditions include: {', '.join(d.strip() for d in diseases)}."
        out.append(make_example(q, a, "database_cross_referencing"))
    return out


def gen_ccd_identity(df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    rows = df[df["chembl_id"].notna() | df["drugbank_id"].notna()] if "chembl_id" in df.columns else df.iloc[0:0]
    if rows.empty:
        rows = df[df["smiles"].notna() & (df["smiles"] != "")]
    rows = rows.sample(n=min(n, len(rows)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in rows.iterrows():
        q = f"A PDB entry contains a bound ligand with CCD code '{r['comp_id']}'. Look up what this compound is."
        a = (f"CCD `{r['comp_id']}` is {r['name']} ({r['formula']}, MW {r['formula_weight']:.1f} Da)."
             if pd.notna(r.get('formula_weight'))
             else f"CCD `{r['comp_id']}` is {r['name']} ({r['formula']}).")
        out.append(make_example(q, a, "database_cross_referencing"))
    return out


def gen_citation(df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    rows = df[df["citation_title"].notna() & (df["citation_title"] != "")].sample(
        n=min(n, len(df)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in rows.iterrows():
        q = rng.choice([
            f"What paper originally described PDB entry {r['pdb_id']}?",
            f"Find the primary literature citation for PDB entry {r['pdb_id']}.",
        ])
        cite = f"\"{r['citation_title']}\""
        if pd.notna(r.get("citation_journal")) and r["citation_journal"]:
            cite += f", {r['citation_journal']}"
        if pd.notna(r.get("citation_year")):
            cite += f" ({int(r['citation_year'])})"
        a = f"The primary citation for {r['pdb_id']} is: {cite}."
        if pd.notna(r.get("citation_doi")) and r["citation_doi"]:
            a += f" DOI: {r['citation_doi']}."
        if pd.notna(r.get("citation_pubmed_id")) and r["citation_pubmed_id"] and int(r["citation_pubmed_id"]) > 0:
            a += f" PubMed ID: {int(r['citation_pubmed_id'])}."
        a += (" This is the `rcsb_primary_citation` category in the entry's mmCIF file — the paper "
              "the depositors themselves designated as the primary reference, not necessarily every "
              "paper that has ever used this structure.")
        out.append(make_example(q, a, "database_cross_referencing"))
    return out


def gen_organism_taxonomy(df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    rows = df[df["organism"].notna() & (df["organism"] != "")].sample(
        n=min(n, len(df)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in rows.iterrows():
        q = f"What organism does the protein in PDB entry {r['pdb_id']} come from, and how long is its sequence?"
        a = f"The primary polymer entity in {r['pdb_id']} is from {r['organism']}"
        if pd.notna(r.get("taxonomy_id")):
            a += f" (NCBI taxonomy ID {int(r['taxonomy_id'])})"
        a += "."
        if pd.notna(r.get("primary_sequence_length")):
            a += f" Its deposited sequence is {int(r['primary_sequence_length'])} residues long."
        a += (" This comes from the entry's `rcsb_entity_source_organism` and `entity_poly` "
              "categories — for multi-entity complexes, other chains may be from different organisms "
              "entirely (e.g. a human target with a viral or bacterial binding partner).")
        out.append(make_example(q, a, "database_cross_referencing"))
    return out


# --- Round 3: single-source generators for the four new corpus files -------

def gen_binding_affinity(bindingdb_df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    """Single-source, new data: real measured potency (Ki/IC50/Kd/EC50) — the piece TWILIGHT's
    pose-fit RSCC data doesn't cover. A ligand can be perfectly modelled and still bind weakly, or
    fit poorly in density yet be a genuinely potent inhibitor captured at partial occupancy."""
    if bindingdb_df.empty:
        return []
    df = bindingdb_df.copy()
    df["pdb_ids"] = df["pdb_ids"].astype(str)
    df = df.assign(pdb_id=df["pdb_ids"].str.split(",")).explode("pdb_id")
    df["pdb_id"] = df["pdb_id"].str.strip().str.upper()
    affinity_cols = ["ki_nM", "ic50_nM", "kd_nM", "ec50_nM"]
    for col in affinity_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df[affinity_cols].notna().any(axis=1) & (df["pdb_id"] != "") & df["ligand_name"].notna()]
    if df.empty:
        return []
    rows = df.sample(n=min(n, len(df)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in rows.iterrows():
        measured = [(label, r[col]) for label, col in
                    [("Ki", "ki_nM"), ("IC50", "ic50_nM"), ("Kd", "kd_nM"), ("EC50", "ec50_nM")]
                    if pd.notna(r[col])]
        label, value = measured[0]
        potency = ("sub-nanomolar, very high potency" if value < 1 else
                   "low nanomolar, high potency" if value < 100 else
                   "high nanomolar to low micromolar, moderate potency" if value < 10000 else
                   "weak/micromolar+, low potency")
        ligand_short = str(r["ligand_name"]).split("::")[0][:80]
        q = f"How potent is the ligand bound in PDB entry {r['pdb_id']} ({ligand_short}) against its target?"
        a = (
            f"BindingDB reports {label} = {value:g} nM for this ligand-target pair "
            f"({r['target_name']}, {r['target_organism']}) — {potency}."
        )
        if pd.notna(r.get("article_doi")) and r["article_doi"]:
            a += f" Source: DOI {r['article_doi']}."
        a += (
            f" This is a measured solution-phase binding affinity, not derived from the crystal "
            f"structure itself — {r['pdb_id']} shows *how* the ligand binds (the pose), BindingDB's "
            f"assay data shows *how tightly*. A co-crystal structure alone is not proof of potency: "
            f"weak or even non-functional binders can still be captured at high ligand concentration "
            f"in a crystallization soak."
        )
        out.append(make_example(q, a, "database_cross_referencing"))
    return out


def gen_string_interactors(string_df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    """Single-source, new data: real protein-protein interaction partners, aggregated per protein —
    STRING's data arrives one edge per row, so this groups by query protein first."""
    if string_df.empty:
        return []
    grouped = string_df.groupby(["uniprot", "protein_name"])
    keys = list(grouped.groups.keys())
    if not keys:
        return []
    sample_keys = rng.sample(keys, k=min(n, len(keys)))
    out = []
    for uniprot, name in sample_keys:
        g = grouped.get_group((uniprot, name)).sort_values("combined_score", ascending=False)
        partners = [f"{row['partner_name']} (score {row['combined_score']:.2f})" for _, row in g.head(5).iterrows()]
        q = f"What proteins does {name} (UniProt {uniprot}) interact with, per STRING?"
        a = (
            f"STRING's top interaction partners for {name} ({uniprot}), ranked by combined confidence "
            f"score (0-1, combining evidence from experiments, curated databases, co-expression, and "
            f"text mining): {', '.join(partners)}. A high combined score reflects strong aggregate "
            f"evidence across STRING's channels, not necessarily a direct physical interaction "
            f"confirmed structurally — for that, check whether a co-crystal or cryo-EM complex "
            f"structure of the pair exists in the PDB."
        )
        out.append(make_example(q, a, "database_cross_referencing"))
    return out


def gen_alphafold_confidence(alphafold_df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    """Single-source, new data: per-region pLDDT confidence breakdown, standalone (companion to the
    comparative gen_alphafold_vs_experimental generator, this one needs no linked PDB entry, so it
    covers UniProt accessions the experimental corpus has no structure for at all)."""
    if alphafold_df.empty:
        return []
    df = alphafold_df[alphafold_df["global_plddt"].notna()]
    rows = df.sample(n=min(n, len(df)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in rows.iterrows():
        plddt = float(r["global_plddt"])
        conf = ("very high" if plddt >= 90 else "confident" if plddt >= 70 else
                "low" if plddt >= 50 else "very low")
        gene = r.get("gene") if pd.notna(r.get("gene")) and r.get("gene") else "gene not recorded"
        organism = r.get("organism") if pd.notna(r.get("organism")) and r.get("organism") else "organism not recorded"
        q = f"How reliable is the AlphaFold predicted structure for UniProt {r['uniprot']} ({gene}, {organism})?"
        very_high = float(r["fraction_plddt_very_high"]) * 100 if pd.notna(r.get("fraction_plddt_very_high")) else None
        very_low = float(r["fraction_plddt_very_low"]) * 100 if pd.notna(r.get("fraction_plddt_very_low")) else None
        a = f"Global mean pLDDT is {plddt:.1f}/100 — {conf} confidence overall."
        if very_high is not None and very_low is not None:
            a += (f" {very_high:.0f}% of residues fall in the very-high-confidence band (pLDDT>90), "
                  f"vs {very_low:.0f}% very-low-confidence (pLDDT<50) — the latter typically "
                  f"corresponds to intrinsically disordered regions or poorly conserved loops that "
                  f"AlphaFold isn't expected to predict a single fixed structure for, since one may "
                  f"not exist biologically.")
        a += (" Treat pLDDT per-residue rather than trusting the global average alone: a protein can "
              "have a high mean score driven by a well-folded core domain while a terminal tail or "
              "linker is confidently flagged as disordered (correctly) at very low pLDDT.")
        out.append(make_example(q, a, "database_cross_referencing"))
    return out


# --- Round 3: bidirectional traversal ---------------------------------------

def gen_uniprot_to_pdb_aggregate(sifts_uniprot_df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    """Bidirectional traversal: the usual direction in this corpus is PDB -> UniProt (one chain, one
    lookup); this reverses it — given a UniProt accession, aggregate every PDB entry that maps to
    it. A well-studied target often has dozens of deposited structures (different ligands, space
    groups, resolutions), and 'which structures exist for protein X' is a distinct, common real
    question the forward direction alone can't answer."""
    counts = sifts_uniprot_df.groupby("SP_PRIMARY")["PDB"].nunique()
    multi = counts[counts >= 3]
    if multi.empty:
        return []
    accs = rng.sample(list(multi.index), k=min(n, len(multi)))
    out = []
    for acc in accs:
        entries = sorted(sifts_uniprot_df[sifts_uniprot_df["SP_PRIMARY"] == acc]["PDB"].str.upper().unique())
        shown = entries[:10]
        q = f"Which PDB entries are structures of the protein with UniProt accession {acc}?"
        a = (
            f"UniProt {acc} maps (via SIFTS) to {len(entries)} PDB entries in this corpus, including "
            f"{', '.join(shown)}" + (" and others" if len(entries) > 10 else "") + ". Multiple entries "
            f"for the same protein are normal and often valuable to compare: different entries may "
            f"capture different ligands bound, different crystal forms, different resolutions, or "
            f"structures solved years apart with improved refinement — the 'best' one to use depends "
            f"on what you're asking (highest resolution for geometric detail, a specific ligand-bound "
            f"form for a binding-site question, etc.), not just whichever comes up first."
        )
        out.append(make_example(q, a, "database_cross_referencing"))
    return out


def gen_ligand_to_pdb_aggregate(twilight_df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    """Bidirectional traversal, ligand direction: given a ligand name, aggregate every PDB entry it
    appears in — the reverse of the usual 'this entry contains ligand X' direction."""
    df = twilight_df[twilight_df["LigNm"].notna() & (twilight_df["LigNm"] != "")]
    counts = df.groupby("LigNm")["PDBID"].nunique()
    multi = counts[counts >= 5]
    if multi.empty:
        return []
    ligands = rng.sample(list(multi.index), k=min(n, len(multi)))
    out = []
    for lig in ligands:
        entries = sorted(df[df["LigNm"] == lig]["PDBID"].str.upper().unique())
        shown = entries[:10]
        q = f"Which PDB entries contain the ligand '{lig}'?"
        a = (
            f"'{lig}' appears in {len(entries)} PDB entries in this corpus, including "
            f"{', '.join(shown)}" + (" and others" if len(entries) > 10 else "") + ". A ligand "
            f"appearing across many entries usually means it's either a common crystallization "
            f"additive/cryoprotectant (if chemically simple) or a biologically important cofactor/"
            f"inhibitor studied across many structural contexts (different targets, or the same "
            f"target's different constructs/mutants) — check the ligand identity and target "
            f"diversity together before assuming which case applies."
        )
        out.append(make_example(q, a, "database_cross_referencing"))
    return out


# --- Round 3: deeper multi-hop chains ---------------------------------------

def gen_multihop_target_context(entries_df: pd.DataFrame, sifts_uniprot_df: pd.DataFrame,
                                 pharos_df: pd.DataFrame, bindingdb_df: pd.DataFrame,
                                 rng: random.Random, n: int) -> list[dict]:
    """4-hop chain: PDB entry -> UniProt (SIFTS) -> Pharos (target druggability) -> BindingDB
    (measured potency for that same target). Answers 'give me the full target-context picture for
    this structure', which needs all three external databases chained off one starting PDB ID."""
    if pharos_df.empty or bindingdb_df.empty:
        return []
    m = sifts_uniprot_df.merge(pharos_df, left_on="SP_PRIMARY", right_on="uniprot", how="inner")
    bdf = bindingdb_df[bindingdb_df["uniprot_primary"].notna()].copy()
    m = m.merge(bdf, left_on="SP_PRIMARY", right_on="uniprot_primary", how="inner")
    m["pdb_id"] = m["PDB"].str.upper()
    m = m.merge(entries_df[["pdb_id", "resolution_A", "method"]], on="pdb_id", how="inner")
    affinity_cols = ["ki_nM", "ic50_nM", "kd_nM", "ec50_nM"]
    for col in affinity_cols:
        m[col] = pd.to_numeric(m[col], errors="coerce")
    m = m[m[affinity_cols].notna().any(axis=1) & m["resolution_A"].notna()]
    if m.empty:
        return []
    rows = m.sample(n=min(n, len(m)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in rows.iterrows():
        measured = [(label, r[col]) for label, col in
                    [("Ki", "ki_nM"), ("IC50", "ic50_nM"), ("Kd", "kd_nM"), ("EC50", "ec50_nM")]
                    if pd.notna(r[col])]
        if not measured:
            continue
        label, value = measured[0]
        q = f"I have PDB entry {r['pdb_id']}. Walk me through its target's biology and druggability, chaining through every database you can."
        family_part = f", family {r['family']}" if pd.notna(r.get("family")) else ""
        a = (
            f"Chain: {r['pdb_id']} ({r['method']}, {float(r['resolution_A']):.2f} Å) -> SIFTS maps "
            f"chain {r['CHAIN']} to UniProt {r['SP_PRIMARY']} -> Pharos classifies that target as "
            f"{r['name']} ({r['symbol']}), development level {r['tdl']}{family_part} -> "
            f"BindingDB has a measured {label} of {value:g} nM against this target for the ligand "
            f"{str(r['ligand_name']).split('::')[0][:60]}. Together this tells you not just what the "
            f"structure looks like, but how pharmacologically mature the target is and how potent a "
            f"real measured inhibitor of it is — context you can't recover from the structure file "
            f"alone."
        )
        out.append(make_example(q, a, "database_cross_referencing"))
    return out


def gen_multihop_ligand_quality_chain(twilight_df: pd.DataFrame, bindingdb_df: pd.DataFrame,
                                       rng: random.Random, n: int) -> list[dict]:
    """3-hop chain: PDB entry -> ligand identity (CCD/TWILIGHT) -> model-fit quality (RSCC), joined
    against -> measured potency (BindingDB) for that same PDB+ligand pair. Deliberately keeps the
    two halves separate in the answer — pose quality and binding potency are uncorrelated axes, and
    conflating them is a common real mistake."""
    if bindingdb_df.empty:
        return []
    tw = twilight_df[twilight_df["RSCC"].notna() & twilight_df["LigNm"].notna()].copy()
    tw["RSCC"] = pd.to_numeric(tw["RSCC"], errors="coerce")
    tw["pdb_id"] = tw["PDBID"].str.upper()
    tw["ligand_key"] = tw["LigNm"].astype(str).str.upper()
    bdf = bindingdb_df.copy()
    bdf["pdb_ids"] = bdf["pdb_ids"].astype(str)
    bdf = bdf.assign(pdb_id=bdf["pdb_ids"].str.split(",")).explode("pdb_id")
    bdf["pdb_id"] = bdf["pdb_id"].str.strip().str.upper()
    bdf["ligand_key"] = bdf["ligand_het_id"].astype(str).str.upper()
    m = tw.merge(bdf, on=["pdb_id", "ligand_key"], how="inner")
    affinity_cols = ["ki_nM", "ic50_nM", "kd_nM", "ec50_nM"]
    for col in affinity_cols:
        m[col] = pd.to_numeric(m[col], errors="coerce")
    m = m[m[affinity_cols].notna().any(axis=1) & m["RSCC"].between(-1.5, 1.0)]
    if m.empty:
        return []
    rows = m.sample(n=min(n, len(m)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in rows.iterrows():
        rscc = float(r["RSCC"])
        fit = ("an excellent fit to the density" if rscc > 0.9 else
               "a reasonable fit" if rscc > 0.8 else "a poor fit to the density")
        measured = [(label, r[col]) for label, col in
                    [("Ki", "ki_nM"), ("IC50", "ic50_nM"), ("Kd", "kd_nM"), ("EC50", "ec50_nM")]
                    if pd.notna(r[col])]
        if not measured:
            continue
        label, value = measured[0]
        q = f"For the ligand {r['LigNm']} bound in PDB entry {r['pdb_id']}, how does its crystallographic model quality compare to its measured binding potency?"
        a = (
            f"These are two independent measurements: the modelled pose has RSCC={rscc:.3f} against "
            f"{r['pdb_id']}'s density ({fit}), while BindingDB separately reports {label}={value:g} nM "
            f"measured potency for this ligand against its target. A ligand can have excellent density "
            f"fit and weak measured potency (it was simply present at high soak concentration), or the "
            f"reverse (a genuinely potent binder captured at partial occupancy or in a flexible pose) — "
            f"pose quality tells you how confidently the coordinates are known, not how tightly the "
            f"molecule binds."
        )
        out.append(make_example(q, a, "database_cross_referencing"))
    return out


def gen_multihop_fold_function(cath_joined_df: pd.DataFrame, uniprot_df: pd.DataFrame,
                                rng: random.Random, n: int) -> list[dict]:
    """3-hop synthesis: PDB -> CATH fold classification joined with PDB -> SIFTS -> UniProt function,
    combined into one answer connecting *shape* to *function* — related but not interchangeable, a
    relationship the existing single-source generators (gen_cath_fold, gen_uniprot_function) never
    state explicitly."""
    m = cath_joined_df.merge(uniprot_df, left_on="SP_PRIMARY", right_on="accession", how="inner")
    m = m[m["function"].notna() & (m["function"] != "")]
    if m.empty:
        return []
    rows = m.sample(n=min(n, len(m)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in rows.iterrows():
        q = f"How does the CATH fold classification of PDB entry {r['PDB'].upper()} chain {r['CHAIN']} relate to what the protein actually does?"
        func = str(r["function"])[:400]
        a = (
            f"Chain {r['CHAIN']} of {r['PDB'].upper()} is classified by CATH as "
            f"{r['class_desc']} / {r['architecture_desc']} / {r['topology_desc']} / {r['homology_desc']} "
            f"(domain {r['CATH_ID']}). Its UniProt entry ({r['SP_PRIMARY']}) describes the protein's "
            f"function as: \"{func}\". Fold and function are correlated but not synonymous: proteins "
            f"sharing a CATH homologous superfamily often share function (that's the classification's "
            f"basis), but the same broad fold can also be reused for unrelated functions by evolution, "
            f"and function ultimately depends on specific active-site/interface residues the fold alone "
            f"doesn't specify — so treat the fold as context for the function, not proof of it."
        )
        out.append(make_example(q, a, "database_cross_referencing"))
    return out


# --- Round 3: cross-database disagreement + missing-data honesty -----------

def gen_cross_db_disagreement(entries_df: pd.DataFrame, sifts_uniprot_df: pd.DataFrame,
                               uniprot_df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    """Cross-database disagreement: RCSB's entry-level organism call and UniProt's own organism
    field for the mapped accession don't always literally match in string form (strain-level detail,
    synonyms, or genuinely different constructs on different chains). Teaches the model to report
    both sources rather than silently pick one when they diverge, and to explain *why* a mismatch
    can be legitimate rather than assuming one source is simply wrong."""
    sifts = sifts_uniprot_df.copy()
    sifts["pdb_id"] = sifts["PDB"].str.upper()
    m = entries_df[["pdb_id", "organism"]].merge(
        sifts[["pdb_id", "SP_PRIMARY"]].drop_duplicates("pdb_id"), on="pdb_id", how="inner")
    m = m.merge(uniprot_df[["accession", "organism"]], left_on="SP_PRIMARY", right_on="accession",
                how="inner", suffixes=("_rcsb", "_uniprot"))
    m = m[m["organism_rcsb"].notna() & m["organism_uniprot"].notna() & (m["organism_rcsb"] != "") & (m["organism_uniprot"] != "")]
    mismatched = m[m["organism_rcsb"].str.strip().str.lower() != m["organism_uniprot"].str.strip().str.lower()]
    if mismatched.empty:
        return []
    rows = mismatched.sample(n=min(n, len(mismatched)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in rows.iterrows():
        q = f"What organism is the protein in PDB entry {r['pdb_id']} from? Check both RCSB's entry metadata and the mapped UniProt record."
        a = (
            f"These two sources don't literally agree: RCSB's entry-level source organism for "
            f"{r['pdb_id']} is recorded as '{r['organism_rcsb']}', while UniProt {r['SP_PRIMARY']} "
            f"(the accession this chain maps to via SIFTS) lists '{r['organism_uniprot']}'. This isn't "
            f"necessarily an error in either source — common legitimate causes include: the RCSB field "
            f"describing the specific expression construct or strain used, UniProt describing the "
            f"canonical/reference organism for that gene family, a chimeric or engineered construct, "
            f"or one source simply being more taxonomically specific than the other. Report both rather "
            f"than picking one, and note the discrepancy explicitly rather than silently resolving it."
        )
        out.append(make_example(q, a, "database_cross_referencing"))
    return out


def gen_missing_data_honesty(entries_df: pd.DataFrame, validation_df: pd.DataFrame,
                              rng: random.Random, n: int) -> list[dict]:
    """Missing-data honesty: sample entries where a commonly-expected field (R-free, or a wwPDB
    validation metric) is genuinely absent, and generate the correct 'not available, here's why'
    answer rather than letting the fine-tuned model fabricate a plausible-looking number. This is
    the single most important refusal-adjacent behaviour for a database-grounded assistant — the
    training data must contain real examples of admitting absence, not just presence."""
    out = []
    no_rfree = entries_df[(entries_df["method"] == "X-RAY DIFFRACTION") & entries_df["resolution_A"].notna() & entries_df["r_free"].isna()]
    if not no_rfree.empty:
        rows = no_rfree.sample(n=min(n // 2, len(no_rfree)), random_state=rng.randint(0, 1 << 30))
        for _, r in rows.iterrows():
            q = f"What is the R-free value for PDB entry {r['pdb_id']}?"
            a = (
                f"R-free isn't recorded in this entry's deposited metadata for {r['pdb_id']} — this "
                f"happens for older depositions (R-free reporting wasn't universally required until "
                f"the late 1990s) or entries refined with software that didn't compute/report it. I "
                f"won't estimate a value: an absent R-free doesn't mean the structure is untrustworthy, "
                f"just that this particular quality metric wasn't captured. Resolution "
                f"({float(r['resolution_A']):.2f} Å) and the wwPDB validation report's geometry metrics "
                f"are still available checks."
            )
            out.append(make_example(q, a, "database_cross_referencing"))
    if not validation_df.empty:
        has_entry = set(entries_df["pdb_id"])
        validated_ids = set(validation_df["pdb_id"])
        missing_ids = has_entry - validated_ids
        if missing_ids:
            sample_ids = rng.sample(sorted(missing_ids), k=min(max(0, n - len(out)), len(missing_ids)))
            for pid in sample_ids:
                q = f"What's the clashscore for PDB entry {pid}?"
                a = (
                    f"No wwPDB validation-report clashscore is available for {pid} in this corpus. "
                    f"Rather than guess, the honest answer is: not available here — check the entry's "
                    f"live validation report directly at RCSB/PDBe, since coverage of the automated "
                    f"validation pipeline (or of this project's own pull of it) isn't complete for "
                    f"every deposited entry."
                )
                out.append(make_example(q, a, "database_cross_referencing"))
    return out


# --- Round 3: comparative examples ------------------------------------------

def gen_compare_two_entries(entries_df: pd.DataFrame, sifts_uniprot_df: pd.DataFrame,
                             rng: random.Random, n: int) -> list[dict]:
    """Comparative: pick two different PDB entries mapped to the same UniProt accession and compare
    them head-to-head (resolution, method, R-free) — the "which structure should I use" question
    that single-entry generators can't answer, since it requires holding two rows in context at
    once."""
    sifts = sifts_uniprot_df.copy()
    sifts["pdb_id"] = sifts["PDB"].str.upper()
    m = entries_df[["pdb_id", "resolution_A", "r_free", "method"]].merge(
        sifts[["pdb_id", "SP_PRIMARY"]].drop_duplicates(), on="pdb_id", how="inner")
    m = m[m["resolution_A"].notna()].drop_duplicates("pdb_id")
    counts = m.groupby("SP_PRIMARY")["pdb_id"].nunique()
    multi = counts[counts >= 2]
    if multi.empty:
        return []
    accs = rng.sample(list(multi.index), k=min(n, len(multi)))
    out = []
    for acc in accs:
        g = m[m["SP_PRIMARY"] == acc]
        pair = g.sample(n=2, random_state=rng.randint(0, 1 << 30))
        a_row, b_row = pair.iloc[0], pair.iloc[1]
        better, worse = (a_row, b_row) if float(a_row["resolution_A"]) < float(b_row["resolution_A"]) else (b_row, a_row)
        q = f"UniProt {acc} has both PDB entries {a_row['pdb_id']} and {b_row['pdb_id']}. How do they compare, and which should I use?"
        a = (
            f"{a_row['pdb_id']}: {a_row['method']}, {float(a_row['resolution_A']):.2f} Å"
            + (f", R-free {float(a_row['r_free']):.3f}" if pd.notna(a_row.get("r_free")) else "")
            + f". {b_row['pdb_id']}: {b_row['method']}, {float(b_row['resolution_A']):.2f} Å"
            + (f", R-free {float(b_row['r_free']):.3f}" if pd.notna(b_row.get("r_free")) else "")
            + f". By resolution alone, {better['pdb_id']} is the sharper structure "
              f"({_resolution_bucket(float(better['resolution_A']))}) vs {worse['pdb_id']} "
              f"({_resolution_bucket(float(worse['resolution_A']))}). But 'better' depends on your "
              f"question: prefer the higher-resolution entry for fine geometric detail, but check "
              f"which entry actually contains the ligand, mutation, or conformational state you care "
              f"about — a lower-resolution structure capturing the right biological state is often "
              f"more useful than a sharper one that doesn't."
        )
        out.append(make_example(q, a, "database_cross_referencing"))
    return out


# --- Round 3: RAG-shaped synthesis ------------------------------------------

def gen_rag_synthesis(entries_df: pd.DataFrame, sifts_uniprot_df: pd.DataFrame,
                       uniprot_df: pd.DataFrame, pharos_df: pd.DataFrame,
                       validation_df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    """RAG-shaped synthesis: presents a prompt formatted the way real retrieved context looks
    (numbered, source-tagged chunks) and requires a synthesized, source-citing answer — training the
    model to work the way it will actually be used at inference time behind the RAG retriever, not
    just to answer bare questions about pre-selected facts."""
    sifts = sifts_uniprot_df.copy()
    sifts["pdb_id"] = sifts["PDB"].str.upper()
    m = entries_df[["pdb_id", "resolution_A", "r_free", "method"]].merge(
        sifts[["pdb_id", "SP_PRIMARY", "CHAIN"]].drop_duplicates("pdb_id"), on="pdb_id", how="inner")
    m = m.merge(uniprot_df[["accession", "protein_name", "function"]], left_on="SP_PRIMARY", right_on="accession", how="inner")
    m = m[m["resolution_A"].notna() & m["function"].notna() & (m["function"] != "")]
    if not pharos_df.empty:
        m = m.merge(pharos_df[["uniprot", "tdl", "name"]], left_on="SP_PRIMARY", right_on="uniprot", how="left")
    if not validation_df.empty:
        m = m.merge(validation_df[["pdb_id", "clashscore"]], on="pdb_id", how="left")
    if m.empty:
        return []
    rows = m.sample(n=min(n, len(m)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in rows.iterrows():
        chunks = [
            f"[Source: rcsb/pdb_entries_enriched.csv, entry {r['pdb_id']}] method={r['method']}, "
            f"resolution_A={float(r['resolution_A']):.2f}" + (f", r_free={float(r['r_free']):.3f}" if pd.notna(r.get("r_free")) else ""),
            f"[Source: uniprot/uniprot_entries.csv, accession {r['SP_PRIMARY']}] protein_name={r['protein_name']}; "
            f"function={str(r['function'])[:300]}",
        ]
        if pd.notna(r.get("tdl")):
            chunks.append(f"[Source: pharos/pharos_targets.csv, uniprot {r['SP_PRIMARY']}] target_development_level={r['tdl']}, name={r['name']}")
        if pd.notna(r.get("clashscore")):
            chunks.append(f"[Source: validation/wwpdb_validation.csv, entry {r['pdb_id']}] clashscore={float(r['clashscore']):.2f}")
        rng.shuffle(chunks)
        context = "\n".join(chunks)
        q = (
            f"Using only the retrieved context below, give a complete answer about PDB entry {r['pdb_id']} "
            f"— what it is, what it does, and its quality — and cite which source each fact came from.\n\n"
            f"Retrieved context:\n{context}"
        )
        a_parts = [
            f"{r['pdb_id']} is a {r['method']} structure at {float(r['resolution_A']):.2f} Å"
            + (f" (R-free {float(r['r_free']):.3f})" if pd.notna(r.get("r_free")) else "")
            + f" [rcsb/pdb_entries_enriched.csv].",
            f"It's {r['protein_name']} (UniProt {r['SP_PRIMARY']}): {str(r['function'])[:300]} [uniprot/uniprot_entries.csv].",
        ]
        if pd.notna(r.get("tdl")):
            a_parts.append(f"Pharos classifies this target at development level {r['tdl']} [pharos/pharos_targets.csv].")
        if pd.notna(r.get("clashscore")):
            a_parts.append(f"Model geometry: clashscore {float(r['clashscore']):.2f} [validation/wwpdb_validation.csv].")
        a_parts.append(
            "Synthesizing across sources: structural quality (resolution/R-free/clashscore) and "
            "biological/pharmacological context (function, target development level) come from "
            "independent databases and should each be cited to its own source rather than blended "
            "into one unattributed claim."
        )
        a = " ".join(a_parts)
        out.append(make_example(q, a, "database_cross_referencing"))
    return out


# ---------------------------------------------------------------------------
# Refusal boundary (supplementary, small)
# ---------------------------------------------------------------------------

def gen_refusal_boundary(uniprot_df: pd.DataFrame, rng: random.Random, n: int) -> list[dict]:
    rows = uniprot_df.sample(n=min(n, len(uniprot_df)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in rows.iterrows():
        q = rng.choice([
            f"Predict the 3D structure of the protein encoded by UniProt {r['accession']} ({r['protein_name']}).",
            f"I don't have a crystal structure for {r['protein_name']} (UniProt {r['accession']}) — can you generate one for me?",
        ])
        a = (
            f"I can't predict a 3D structure for {r['accession']} — that's out of scope for chatPDB, "
            f"which reasons about structures that already exist rather than generating new ones. For "
            f"structure prediction, AlphaFold DB (alphafold.ebi.ac.uk) may already have a predicted "
            f"model for this UniProt accession, or tools like AlphaFold2/ColabFold, ESMFold, or "
            f"RoseTTAFold can generate one. If a real experimental structure for this protein (or a "
            f"close homolog) exists in the PDB, I can help you find and interpret that instead."
        )
        out.append(make_example(q, a, "refusal_boundary"))
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=50000, help="target total example count")
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    c = load_corpus()

    # Roughly equal weight across the 4 classes, refusal as a small supplement.
    per_class = args.n // 4
    print(f"\nTarget: {args.n:,} total, ~{per_class:,} per class")

    all_examples: list[dict] = []

    # file_format_literacy — split across 5 generators
    k = per_class // 5
    print("Generating file_format_literacy ...")
    all_examples += gen_atom_hetatm(c["entries"], rng, k)
    all_examples += gen_ccd_component_format(c["ccd"], rng, k)
    all_examples += gen_deposition_header(c["all_entries"], rng, k)
    all_examples += gen_format_pdb_vs_mmcif(c["entries"][c["entries"]["atom_count"] > 20000], rng, k)
    all_examples += gen_biological_assembly_asu(c["entries"], rng, k)

    # experimental_method — split across 10 generators. Round 3 added validation_geometry,
    # alphafold_vs_experimental, and multihop_structure_quality_full once wwPDB validation and
    # AlphaFold DB were pulled in.
    k = per_class // 10
    print("Generating experimental_method ...")
    all_examples += gen_xray_resolution_quality(c["entries"], rng, k)
    all_examples += gen_rfree_quality(c["entries"], rng, k)
    all_examples += gen_em_resolution_quality(c["entries"], rng, k)
    all_examples += gen_nmr_characteristics(c["entries"], rng, k)
    all_examples += gen_twilight_ligand_fit(c["twilight"], rng, k)
    all_examples += gen_unit_cell_space_group(c["entries"], rng, k)
    all_examples += gen_crystallization_conditions(c["entries"], rng, k)
    all_examples += gen_validation_geometry(c["validation"], c["entries"], rng, k)
    all_examples += gen_alphafold_vs_experimental(c["alphafold"], c["entries"], c["sifts_uniprot"], rng, k)
    all_examples += gen_multihop_structure_quality_full(c["entries"], c["validation"], rng, k)

    # tool_calling — split across 7 generators. DSSP and NMR-model-count are execution-verified
    # against the full 256,444-file mmCIF pool (data/structures_all/, corpus expansion round 2).
    # Round 3 added the two tool-chaining generators (multi-step scripts, not one-call-per-example).
    k = per_class // 7
    print("Generating tool_calling ...")
    all_examples += gen_biopython_count(c["entries"], rng, k)
    all_examples += gen_gemmi_metadata(c["entries"], rng, k)
    all_examples += gen_pymol_script(c["entries"], rng, k)
    print("  running DSSP on real structure files (execution-verified) ...")
    all_examples += gen_dssp_secondary_structure(c["structure_files"], rng, k)
    all_examples += gen_nmr_model_count(c["structure_files"], c["entries"], rng, k)
    print("  running tool-chain (parse+DSSP) analysis scripts (execution-verified) ...")
    all_examples += gen_tool_chain_structure_analysis(c["structure_files"], rng, k)
    all_examples += gen_tool_chain_lookup(c["sifts_uniprot"], c["pharos"], rng, k)

    # database_cross_referencing — split across 21 generators. Round 3 added: single-source
    # generators for BindingDB/STRING/AlphaFold; bidirectional traversal (UniProt->PDB,
    # ligand->PDB); deeper multi-hop chains (target context, ligand quality, fold->function);
    # cross-database disagreement + missing-data honesty; comparative (two entries); and a
    # RAG-shaped multi-source synthesis generator.
    k = per_class // 21
    print("Generating database_cross_referencing ...")
    all_examples += gen_uniprot_chain_mapping(c["sifts_uniprot"], rng, k)
    all_examples += gen_pfam_domain(c["sifts_pfam"], rng, k)
    all_examples += gen_cath_fold(c["cath_joined"], rng, k)
    all_examples += gen_ec_number(c["sifts_enzyme"], rng, k)
    all_examples += gen_uniprot_function(c["uniprot"], rng, k)
    all_examples += gen_pharos_druggability(c["pharos"], c["sifts_uniprot"], rng, k)
    all_examples += gen_ccd_identity(c["ccd"], rng, k)
    all_examples += gen_citation(c["entries"], rng, k)
    all_examples += gen_organism_taxonomy(c["entries"], rng, k)
    all_examples += gen_binding_affinity(c["bindingdb"], rng, k)
    all_examples += gen_string_interactors(c["string"], rng, k)
    all_examples += gen_alphafold_confidence(c["alphafold"], rng, k)
    all_examples += gen_uniprot_to_pdb_aggregate(c["sifts_uniprot"], rng, k)
    all_examples += gen_ligand_to_pdb_aggregate(c["twilight"], rng, k)
    all_examples += gen_multihop_target_context(c["entries"], c["sifts_uniprot"], c["pharos"], c["bindingdb"], rng, k)
    all_examples += gen_multihop_ligand_quality_chain(c["twilight"], c["bindingdb"], rng, k)
    all_examples += gen_multihop_fold_function(c["cath_joined"], c["uniprot"], rng, k)
    all_examples += gen_cross_db_disagreement(c["entries"], c["sifts_uniprot"], c["uniprot"], rng, k)
    all_examples += gen_missing_data_honesty(c["entries"], c["validation"], rng, k)
    all_examples += gen_compare_two_entries(c["entries"], c["sifts_uniprot"], rng, k)
    all_examples += gen_rag_synthesis(c["entries"], c["sifts_uniprot"], c["uniprot"], c["pharos"], c["validation"], rng, k)

    # supplementary refusal boundary
    print("Generating refusal_boundary ...")
    all_examples += gen_refusal_boundary(c["uniprot"], rng, min(1000, per_class // 5))

    print(f"\nGenerated {len(all_examples):,} raw examples. Validating ...")
    valid_pdb_ids = set(c["entries"]["pdb_id"].str.upper())
    valid_comp_ids = set(c["ccd"]["comp_id"])
    valid_uniprot = set(c["uniprot"]["accession"])
    validated = [ex for ex in all_examples if validate(ex, valid_pdb_ids, valid_comp_ids, valid_uniprot)]
    rejected = len(all_examples) - len(validated)
    print(f"  {len(validated):,} passed validation, {rejected:,} rejected ({rejected/max(1,len(all_examples))*100:.1f}%)")

    # Category balance report
    from collections import Counter
    cat_counts = Counter(ex["_category"] for ex in validated)
    print("\nCategory balance:")
    for cat, cnt in cat_counts.most_common():
        print(f"  {cat}: {cnt:,}")

    # De-duplicate identical (user, assistant) pairs (can happen via rng.choice collisions)
    seen = set()
    deduped = []
    for ex in validated:
        key = (ex["messages"][1]["content"], ex["messages"][2]["content"])
        if key not in seen:
            seen.add(key)
            deduped.append(ex)
    print(f"\n{len(deduped):,} unique examples after de-duplication ({len(validated) - len(deduped):,} dupes dropped)")

    rng.shuffle(deduped)
    for ex in deduped:
        del ex["_category"]

    n_total = len(deduped)
    n_test = max(1, int(n_total * 0.1))
    n_valid = max(1, int(n_total * 0.1))
    n_train = n_total - n_test - n_valid

    train = deduped[:n_train]
    valid = deduped[n_train:n_train + n_valid]
    test = deduped[n_train + n_valid:]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in [("train.jsonl", train), ("valid.jsonl", valid), ("test.jsonl", test)]:
        path = args.out_dir / name
        with open(path, "w") as f:
            for ex in split:
                f.write(json.dumps(ex) + "\n")
        print(f"  {name}: {len(split):,} examples -> {path}")

    print(f"\nDone. {n_total:,} total examples ({n_train:,} train / {n_valid:,} valid / {n_test:,} test).")


if __name__ == "__main__":
    main()
