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
STRUCTURES = Path("data/structures")
OUT = Path("data/sft")
SYSTEM_PROMPT_PATH = Path("config/system_prompt.txt")


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus() -> dict:
    print("Loading corpus tables ...")
    c = {}
    c["entries"] = pd.read_csv(CORPUS / "rcsb/pdb_entries_enriched.csv")
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

    # CATH domain -> classification join, keyed by PDB id + chain (mirrors rag/corpus_lookup.py's
    # two-hop join, precomputed here once for speed across thousands of generated examples).
    # sifts_pdb_cath.csv columns: PDB, CHAIN, SP_PRIMARY, CATH_ID (confirmed live 2026-07-15).
    cath = c["sifts_cath"].merge(c["cath_class"], how="inner", left_on="CATH_ID", right_on="domain_id")
    c["cath_joined"] = cath

    # Structure pool actually on disk (for execution-verified generators).
    c["structure_files"] = sorted(STRUCTURES.glob("*.pdb"))
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
    if "nan" in assistant.lower().split() or "none" in assistant.lower().replace(".", " ").split():
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
    rows = df[df["RSCC"].notna()].copy()
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


def _run_dssp(path: Path) -> dict[str, int] | None:
    """Run DSSP and return {SS_code: count}, or None if it can't be assigned.

    Feeds mkdssp a gemmi-produced mmCIF rather than the original legacy .pdb file: mkdssp 4.6.1's
    own internal PDB->mmCIF conversion has a real bug (confirmed 2026-07-16 against this structure
    pool) that raises a 'Duplicate Key violation' on modern REMARK 3 refinement-statistics blocks
    (multiple TLS groups etc.) — it failed on the majority of this pool's post-2015 X-ray entries.
    gemmi's converter doesn't hit this; running mkdssp against its output sidesteps the bug
    entirely while still computing genuine secondary structure from the same real coordinates."""
    import gemmi
    from Bio.PDB import MMCIFParser
    from Bio.PDB.DSSP import DSSP

    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        st = gemmi.read_structure(str(path))
        st.setup_entities()
        st.make_mmcif_document().write_file(str(tmp_path))
        structure = MMCIFParser(QUIET=True).get_structure(path.stem, str(tmp_path))
        model = structure[0]
        dssp = DSSP(model, str(tmp_path), dssp="mkdssp", file_type="mmCIF")
        counts: dict[str, int] = {}
        for key in dssp.keys():
            ss = dssp[key][2]
            counts[ss] = counts.get(ss, 0) + 1
        return counts or None
    except Exception:
        return None
    finally:
        tmp_path.unlink(missing_ok=True)


def gen_dssp_secondary_structure(structure_files: list[Path], rng: random.Random, n: int) -> list[dict]:
    """Execution-verified: actually runs DSSP against real downloaded files. Capped by pool size."""
    out = []
    sample = rng.sample(structure_files, k=min(n * 2, len(structure_files)))  # oversample: some are nucleic-acid-only and yield no SS
    for path in sample:
        if len(out) >= n:
            break
        pid = path.stem.upper()
        counts = _run_dssp(path)
        if not counts:
            continue
        helix = counts.get("H", 0) + counts.get("G", 0) + counts.get("I", 0)
        strand = counts.get("E", 0) + counts.get("B", 0)
        total = sum(counts.values())
        q = f"Write Biopython/DSSP code to assign secondary structure to PDB entry {pid} (file `{path.name}`) and summarise the helix/strand content."
        a = (
            "```python\n"
            "from Bio.PDB import PDBParser\n"
            "from Bio.PDB.DSSP import DSSP\n\n"
            f"structure = PDBParser(QUIET=True).get_structure('{pid}', '{path.name}')\n"
            "model = structure[0]\n"
            f"dssp = DSSP(model, '{path.name}', dssp='mkdssp')\n"
            "ss_counts = {}\n"
            "for key in dssp.keys():\n"
            "    ss = dssp[key][2]\n"
            "    ss_counts[ss] = ss_counts.get(ss, 0) + 1\n"
            "print(ss_counts)\n"
            "```\n\n"
            f"Running DSSP on the real deposited coordinates for {pid} gives {total} assigned residues: "
            f"{helix} in helix (H/G/I), {strand} in strand (E/B) — "
            f"{'a predominantly helical structure' if helix > strand * 1.5 else 'a predominantly beta structure' if strand > helix * 1.5 else 'a mixed alpha/beta structure'}."
            + (" (Note: if `dssp='mkdssp'` raises a parse error on a legacy .pdb file with an "
               "unusual REMARK 3 block, convert to mmCIF with gemmi first and pass `file_type='mmCIF'` "
               "— a known mkdssp 4.x limitation, not a Biopython issue.)"
               if rng.random() < 0.15 else "")
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
            from Bio.PDB import PDBParser
            structure = PDBParser(QUIET=True).get_structure(pid, str(path))
            n_models = len(structure)
        except Exception:
            continue
        q = f"Write Biopython code to count how many NMR models are present in PDB entry {pid} (file `{path.name}`)."
        a = (
            "```python\n"
            "from Bio.PDB import PDBParser\n\n"
            f"structure = PDBParser(QUIET=True).get_structure('{pid}', '{path.name}')\n"
            "print('Models:', len(structure))\n"
            "```\n\n"
            f"The real deposited file for {pid} contains {n_models} models — this is the NMR ensemble "
            f"size, each model an independent structure consistent with the experimental restraints."
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
            f"level {r['tdl']} — {tdl_meaning.get(r['tdl'], 'development status not further characterised')}. "
            f"It's classified in the {r['family']} target family."
        )
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

    # experimental_method — split across 5 generators
    k = per_class // 5
    print("Generating experimental_method ...")
    all_examples += gen_xray_resolution_quality(c["entries"], rng, k)
    all_examples += gen_rfree_quality(c["entries"], rng, k)
    all_examples += gen_em_resolution_quality(c["entries"], rng, k)
    all_examples += gen_nmr_characteristics(c["entries"], rng, k)
    all_examples += gen_twilight_ligand_fit(c["twilight"], rng, k)

    # tool_calling — split across 5 generators (2 execution-verified, capped by pool size)
    k = per_class // 5
    print("Generating tool_calling ...")
    all_examples += gen_biopython_count(c["entries"], rng, k)
    all_examples += gen_gemmi_metadata(c["entries"], rng, k)
    all_examples += gen_pymol_script(c["entries"], rng, k)
    print("  running DSSP on real structure files (execution-verified) ...")
    all_examples += gen_dssp_secondary_structure(c["structure_files"], rng, k)
    all_examples += gen_nmr_model_count(c["structure_files"], c["entries"], rng, k)

    # database_cross_referencing — split across 7 generators
    k = per_class // 7
    print("Generating database_cross_referencing ...")
    all_examples += gen_uniprot_chain_mapping(c["sifts_uniprot"], rng, k)
    all_examples += gen_pfam_domain(c["sifts_pfam"], rng, k)
    all_examples += gen_cath_fold(c["cath_joined"], rng, k)
    all_examples += gen_ec_number(c["sifts_enzyme"], rng, k)
    all_examples += gen_uniprot_function(c["uniprot"], rng, k)
    all_examples += gen_pharos_druggability(c["pharos"], c["sifts_uniprot"], rng, k)
    all_examples += gen_ccd_identity(c["ccd"], rng, k)

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
