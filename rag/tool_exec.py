#!/usr/bin/env python3
"""
tool_exec.py — sandboxed execution shim for the structural-biology tools the model emits (Phase 6).

Detects Biopython code blocks in a model response, runs each in a restricted subprocess against
real structure files, and returns the stdout so the assistant can reason over real computed values
rather than the raw completion's (potentially hallucinated) claims -- this is the concrete fix for
the gap Phase 5 testing found: mlx_lm.generate alone emits correct code but can state wrong summary
numbers, since nothing actually executes the code at raw-generation time.

Staged rollout (same caution chem_sage applied to RDKit->PyMOL, see PROJECT_PLAN.md Phase 6):
  Stage 1 (this file, initial scope): Biopython only. gemmi/DSSP and PyMOL are detected and
  flagged as not-yet-enabled, matching chem_sage's own PyMOL-skip pattern -- not executed until
  each tool's isolation story is worked through individually.

Security posture (best-effort on macOS, no root required):
  - Static blocklist: obvious network/filesystem/subprocess import patterns rejected before
    execution (subprocess itself is blocked at this stage -- DSSP needs it and is not yet enabled).
  - Isolated working directory: a fresh temp dir per run. Referenced structure files are *copied*
    in from data/structures_all/ by name (never symlinked, never run with cwd inside the real
    353k-file corpus) so the subprocess has no path back to the real corpus to read beyond what it
    explicitly asked for, and no ability to write/delete anything there even if a blocked pattern
    slipped through.
  - Clean environment: subprocess inherits only PATH/PYTHONPATH/HOME/VIRTUAL_ENV.
  - Hard timeout: 20s per block; process killed on expiry.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CODE_BLOCK = re.compile(r"```python\s*(.*?)```", re.DOTALL)
CIF_FILENAME = re.compile(r"""['"]([a-zA-Z0-9]{4}\.cif)['"]""")
TIMEOUT_S = 20

STRUCTURES_DIR = Path(__file__).resolve().parent.parent / "data" / "structures_all"

# Patterns that signal network, destructive filesystem access, or subprocess escape.
# Rejection is a static text check; it is not a substitute for OS-level isolation.
_BLOCKED = frozenset([
    "import socket",
    "import http",
    "import urllib",
    "import requests",
    "import ftplib",
    "import smtplib",
    "import subprocess",  # DSSP/PyMOL need this -- not yet enabled at this stage
    "os.system(",
    "os.popen(",
    "shutil.rmtree",
    "shutil.copy",
    "__import__(",
    "open(",  # file writes; structure reads go through Bio.PDB's own parsers, not raw open()
])

# Markers for "is this a stage-1 (Biopython-only) block?"
_BIOPYTHON_MARKERS = frozenset(["from Bio", "import Bio"])
# Markers for tools detected but not yet enabled -- flagged, not executed.
_NOT_YET_ENABLED_MARKERS = frozenset([
    "import gemmi", "from gemmi",
    "DSSP(", "dssp=",
    "from pymol", "import pymol",
    "chimerax", "ChimeraX",
])


def extract_code(model_output: str) -> list[str]:
    """Return all ```python ... ``` blocks from the model response."""
    return [m.strip() for m in CODE_BLOCK.findall(model_output)]


def _is_biopython_block(code: str) -> bool:
    return any(marker in code for marker in _BIOPYTHON_MARKERS)


def _not_yet_enabled_reason(code: str) -> str | None:
    for marker in _NOT_YET_ENABLED_MARKERS:
        if marker in code:
            return f"gemmi/DSSP/PyMOL/ChimeraX (matched '{marker}') -- not yet enabled, run manually"
    return None


def _static_check(code: str) -> str | None:
    """Return a block reason string if the code contains a disallowed pattern, else None."""
    for pattern in _BLOCKED:
        if pattern in code:
            return f"[blocked] disallowed pattern in code: '{pattern}'"
    return None


def run_sandboxed(code: str) -> str:
    """Execute one Biopython code block in an isolated subprocess against real structure files."""
    blocked = _static_check(code)
    if blocked:
        return blocked

    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        **{k: v for k, v in os.environ.items()
           if k.startswith(("CONDA_", "VIRTUAL_ENV", "HOME"))},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Copy in only the specific real structure files the code references, by name -- never
        # give the subprocess a path back into the real 353k-file corpus.
        for cif_name in set(CIF_FILENAME.findall(code)):
            src = STRUCTURES_DIR / cif_name.lower()
            if src.is_file():
                shutil.copy2(src, tmpdir_path / cif_name)

        script = tmpdir_path / "bio_block.py"
        script.write_text(code)
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_S,
                cwd=tmpdir,
                env=env,
            )
            if proc.returncode == 0:
                return proc.stdout.strip() or "[ok, no output]"
            return f"[error]\n{proc.stderr.strip()}"
        except subprocess.TimeoutExpired:
            return f"[error] execution timed out after {TIMEOUT_S}s"
        except Exception as e:
            return f"[error] {e}"


def execute(model_output: str) -> str:
    """Run every Biopython block in the model output; return concatenated real results.

    gemmi/DSSP/PyMOL/ChimeraX blocks are flagged but not executed at this stage.
    """
    blocks = extract_code(model_output)
    if not blocks:
        return "[no executable tool calls found]"

    parts = []
    for i, block in enumerate(blocks, 1):
        not_yet = _not_yet_enabled_reason(block)
        if not_yet:
            parts.append(f"[block {i}] [skipped: {not_yet}]")
            continue
        if not _is_biopython_block(block):
            parts.append(f"[block {i}] [skipped: not a recognised Biopython block]")
            continue
        result = run_sandboxed(block)
        parts.append(f"[block {i}]\n{result}")

    return "\n\n".join(parts)


if __name__ == "__main__":
    _demo = (
        "```python\n"
        "from Bio.PDB import MMCIFParser\n"
        "structure = MMCIFParser(QUIET=True).get_structure('4RE2', '4re2.cif')\n"
        "print('chains:', [c.id for c in structure[0]])\n"
        "print('atoms:', sum(1 for _ in structure.get_atoms()))\n"
        "```"
    )
    print(execute(_demo))
