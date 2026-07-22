#!/usr/bin/env python3
"""
metrics.py — shared eval metric functions for chatPDB (Phase 7).

Single source of truth for both eval/eval_pdb.py (single-model) and
eval/compare/eval_compare.py (multi-round) — unlike chem_sage's eval_chem.py /
eval_compare.py, which keep two independently-maintained copies of the same logic.

Ground-truth sources, both real (never hand-authored):
  - Corpus tables loaded via scripts/build_dataset.py::load_corpus() — real wwPDB/RCSB/SIFTS data.
  - Live tool execution via rag/tool_exec.py — real Biopython recomputation in a sandbox.

Metrics (PROJECT_PLAN.md Phase 7 scope):
  - pdb_id_validity        every stated PDB ID must exist in the real corpus (not format-only —
                            chem_sage's own pdb_id_validity never checks real membership)
  - cross_reference_accuracy   stated PDB<->UniProt/CATH/EC cross-references must match real SIFTS
  - tool_executability     every emitted Biopython block must run without error
  - numerical_fidelity     stated resolution / R-free / chain counts must match the real corpus
  - refusal_accuracy       structure-prediction requests correctly declined as out of scope
  - degeneration_score     no repetition-collapse (chem_sage's trigram-repeat check, ported as-is)

Every function returns a chem_sage-style (ok, total) tuple; (0, 0) means "not applicable to this
response" (nothing to check), not a failure — matches chem_sage's n/a convention exactly.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.tool_exec import (  # noqa: E402
    extract_code, run_sandboxed, _is_biopython_block, _not_yet_enabled_reason,
)

# ---------------------------------------------------------------------------
# Compiled regexes
# ---------------------------------------------------------------------------

CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)

# PDB ID: 4 chars, digit-first, alphanumeric (real PDB ID shape)
_PDB_ID_RE = re.compile(r"\b([0-9][A-Za-z0-9]{3})\b")
_PDB_ID_BLOCKLIST = frozenset({"1000", "2000", "3000", "4000", "5000"})

# Real UniProt accession shape (UniProt's own published regex, simplified)
_UNIPROT_RE = re.compile(
    r"\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9](?:[A-Z][A-Z0-9]{2}[0-9])?)\b"
)
_CATH_RE = re.compile(r"\bCATH\s+(\d+\.\d+\.\d+\.\d+)\b", re.IGNORECASE)
_EC_RE = re.compile(r"\bEC\s+(\d+\.\d+\.\d+\.\d+)\b", re.IGNORECASE)

# Fact-tracking scan: PDB ID tokens and stated resolution/R-free/chain-count values, in document
# order, so each fact can be associated with the PDB ID that most recently preceded it — matches
# chatPDB's real generation pattern ("6R88: X-RAY DIFFRACTION, 1.60 Å, R-free 0.181. 6R8A: ...").
_TRACK_RE = re.compile(
    r"(?P<pdbid>\b[0-9][A-Za-z0-9]{3}\b)"
    r"|(?P<res>\d+\.\d+)\s*Å"
    r"|R-?free\s*[:=]?\s*(?P<rfree>0?\.\d+)"
    r"|(?P<chains>\d+)\s+polymer chain instance"
)

# Phrases that signal a refusal response (out-of-scope structure-prediction requests, etc.)
_REFUSAL_PHRASES = (
    "i cannot", "i can't", "i'm not able", "i am not able",
    "i don't have", "i do not have", "outside my scope",
    "outside the scope", "i'm not designed", "i'm unable",
    "i am unable", "i won't ", "can't help with",
    "cannot help with", "not able to help", "i'd recommend consulting",
    "please consult", "beyond my", "not something i can",
    "that falls outside", "i'm not equipped", "i should not",
    "i'm not in a position", "out of scope for chatpdb",
    "not a structure prediction tool",
)


def _is_refusal(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in _REFUSAL_PHRASES)


# ---------------------------------------------------------------------------
# Corpus-derived lookup tables (build once per eval run, pass into metric calls)
# ---------------------------------------------------------------------------

def build_lookup(corpus: dict) -> dict:
    """Precompute fast lookup sets/dicts from load_corpus()'s real corpus tables.

    Call once per eval run (corpus loading is the expensive part); the returned dict is passed
    into pdb_id_validity / cross_reference_accuracy / numerical_fidelity for every example.
    """
    entries = corpus["entries"]
    valid_pdb_ids = set(entries["pdb_id"].astype(str).str.upper())

    entries_by_id: dict[str, dict] = {}
    for row in entries.itertuples(index=False):
        entries_by_id[row.pdb_id.upper()] = {
            "resolution_A": getattr(row, "resolution_A", None),
            "r_free": getattr(row, "r_free", None),
            "polymer_instance_count": getattr(row, "polymer_instance_count", None),
        }

    def _pairs(df, id_col: str, val_col: str) -> set[tuple[str, str]]:
        if df is None or df.empty:
            return set()
        return set(
            zip(df[id_col].astype(str).str.upper(), df[val_col].astype(str).str.upper())
        )

    return {
        "valid_pdb_ids": valid_pdb_ids,
        "entries_by_id": entries_by_id,
        "sifts_uniprot_pairs": _pairs(corpus.get("sifts_uniprot"), "PDB", "SP_PRIMARY"),
        "sifts_cath_pairs": _pairs(corpus.get("sifts_cath"), "PDB", "CATH_ID"),
        "sifts_enzyme_pairs": _pairs(corpus.get("sifts_enzyme"), "PDB", "EC_NUMBER"),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def pdb_id_validity(output: str, lookup: dict) -> tuple[int, int]:
    """Every stated PDB-ID-shaped token must be a real entry in the corpus.

    Upgrade over chem_sage's own pdb_id_validity (format-only regex, never checked real
    membership) — chatPDB has the real full ID set loaded, so use it.
    """
    candidates = [
        m.group(1).upper() for m in _PDB_ID_RE.finditer(output)
        if m.group(1).upper() not in _PDB_ID_BLOCKLIST
    ]
    if not candidates:
        return 0, 0
    ok = sum(1 for c in candidates if c in lookup["valid_pdb_ids"])
    return ok, len(candidates)


def cross_reference_accuracy(output: str, lookup: dict) -> tuple[int, int]:
    """Stated PDB<->UniProt/CATH/EC cross-references must match the real SIFTS tables.

    New metric — chem_sage has no equivalent (no relational cross-reference data). chatPDB's real
    generation pattern states several PDB IDs alongside one cross-reference value in the same
    response (e.g. "UniProt P03227 has 3 structures ... 9BYP ... 9BYQ ... 9BYR"), so every stated
    (PDB, cross-ref) combination found in a response is checked — not just adjacent pairs.
    """
    pdb_ids = {
        m.group(1).upper() for m in _PDB_ID_RE.finditer(output)
        if m.group(1).upper() not in _PDB_ID_BLOCKLIST and m.group(1).upper() in lookup["valid_pdb_ids"]
    }
    if not pdb_ids:
        return 0, 0

    uniprot_ids = set(_UNIPROT_RE.findall(output))
    cath_ids = set(_CATH_RE.findall(output))
    ec_ids = set(_EC_RE.findall(output))

    checked = ok = 0
    for pdb in pdb_ids:
        for up in uniprot_ids:
            checked += 1
            if (pdb, up.upper()) in lookup["sifts_uniprot_pairs"]:
                ok += 1
        for cath in cath_ids:
            checked += 1
            if (pdb, cath.upper()) in lookup["sifts_cath_pairs"]:
                ok += 1
        for ec in ec_ids:
            checked += 1
            if (pdb, ec.upper()) in lookup["sifts_enzyme_pairs"]:
                ok += 1
    return ok, checked


def tool_executability(output: str) -> tuple[int, int]:
    """Count Biopython ```python``` blocks that run without error (rag/tool_exec.py, Phase 6).

    Excludes gemmi/DSSP/PyMOL/ChimeraX blocks (matches rag/tool_exec.py::execute()'s own
    not-yet-enabled skip) — the model correctly emitting a DSSP call it was taught is not a tool
    failure just because Stage 1's sandbox hasn't enabled DSSP execution yet.
    """
    bio_blocks = [
        b for b in extract_code(output)
        if _is_biopython_block(b) and _not_yet_enabled_reason(b) is None
    ]
    if not bio_blocks:
        return 0, 0
    ok = sum(1 for b in bio_blocks if not run_sandboxed(b).startswith(("[error]", "[blocked]")))
    return ok, len(bio_blocks)


def numerical_fidelity(output: str, lookup: dict) -> tuple[int, int]:
    """Stated resolution / R-free / chain-instance-count values must match the real corpus.

    Walks the response in document order, tracking the most recently mentioned valid PDB ID, and
    checks each subsequent fact against that PDB's real row in pdb_entries_enriched.csv — matches
    chatPDB's real per-entry listing generation pattern ("6R88: ..., 1.60 Å, R-free 0.181.
    6R8A: ...").
    """
    prose = CODE_BLOCK_RE.sub("", output)
    tol = {"res": 0.05, "rfree": 0.005}

    last_pdb: str | None = None
    checked = ok = 0
    for m in _TRACK_RE.finditer(prose):
        if m.group("pdbid"):
            candidate = m.group("pdbid").upper()
            if candidate in lookup["valid_pdb_ids"]:
                last_pdb = candidate
            continue
        if last_pdb is None:
            continue
        row = lookup["entries_by_id"].get(last_pdb)
        if row is None:
            continue

        if m.group("res"):
            truth = row.get("resolution_A")
            if truth is None or (isinstance(truth, float) and truth != truth):  # NaN check
                continue
            checked += 1
            if abs(float(m.group("res")) - float(truth)) <= tol["res"]:
                ok += 1
        elif m.group("rfree"):
            truth = row.get("r_free")
            if truth is None or (isinstance(truth, float) and truth != truth):
                continue
            checked += 1
            if abs(float(m.group("rfree")) - float(truth)) <= tol["rfree"]:
                ok += 1
        elif m.group("chains"):
            truth = row.get("polymer_instance_count")
            if truth is None or (isinstance(truth, float) and truth != truth):
                continue
            checked += 1
            if int(m.group("chains")) == int(truth):
                ok += 1

    return ok, checked


def refusal_accuracy(output: str, expected: str) -> tuple[int, int]:
    """For examples whose ground-truth response is a refusal, check the model also refused.

    Returns (0, 0) when the ground truth is NOT a refusal — automatically n/a for the vast
    majority of the test set (only the refusal_boundary behaviour class triggers this).
    """
    if not _is_refusal(expected):
        return 0, 0
    return (1, 1) if _is_refusal(output) else (0, 1)


def degeneration_score(output: str) -> tuple[int, int]:
    """1=clean output, 0=repetition collapse (any 3-token sequence repeated >5x).

    Ported as-is from chem_sage — fully generic, no chemistry dependency to rework.
    """
    tokens = output.split()
    if len(tokens) < 12:
        return 1, 1  # too short to degenerate
    counts: dict[tuple, int] = {}
    for i in range(len(tokens) - 2):
        tg = (tokens[i], tokens[i + 1], tokens[i + 2])
        counts[tg] = counts.get(tg, 0) + 1
    max_repeat = max(counts.values()) if counts else 0
    return (0 if max_repeat > 5 else 1), 1


# ---------------------------------------------------------------------------
# Registry + scorer
# ---------------------------------------------------------------------------

# Ordered metric registry: (id, label, needs_lookup, needs_expected)
METRICS = [
    ("pdb_id_validity", "PDB ID Validity", True, False),
    ("cross_reference_accuracy", "Cross-Reference Accuracy", True, False),
    ("tool_executability", "Tool Executability", False, False),
    ("numerical_fidelity", "Numerical Fidelity", True, False),
    ("refusal_accuracy", "Refusal Accuracy", False, True),
    ("degeneration_score", "Degeneration-Free", False, False),
]

_FN = {
    "pdb_id_validity": pdb_id_validity,
    "cross_reference_accuracy": cross_reference_accuracy,
    "tool_executability": tool_executability,
    "numerical_fidelity": numerical_fidelity,
    "refusal_accuracy": refusal_accuracy,
    "degeneration_score": degeneration_score,
}


def score_all(output: str, expected: str, lookup: dict) -> dict[str, tuple[int, int]]:
    scores: dict[str, tuple[int, int]] = {}
    for key, _, needs_lookup, needs_expected in METRICS:
        fn = _FN[key]
        if needs_expected:
            scores[key] = fn(output, expected)
        elif needs_lookup:
            scores[key] = fn(output, lookup)
        else:
            scores[key] = fn(output)
    return scores
