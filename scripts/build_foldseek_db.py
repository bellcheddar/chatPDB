#!/usr/bin/env python3
"""
build_foldseek_db.py — build a local Foldseek database over data/structures_all/ so chatPDB can
answer "what known structures is this fold most similar to?" against its own corpus, offline, with
no external API call.

Foldseek binary: precompiled universal (x86_64+arm64) macOS build from
github.com/steineggerlab/foldseek releases (10-941cd33), not available via Homebrew/conda on this
machine — downloaded straight from the GitHub release asset into tools/foldseek/.

Smoke-tested on a 200-file subset before committing to the full pool (per this round's plan):
createdb took 0.57s for 200 files (742% CPU, all cores) -- scales to an estimated ~12-15 minutes for
the full 256,444-file pool, dramatically faster than the multi-hour DSSP-execution phases elsewhere
in this project, since Foldseek's structure encoding is far cheaper per-file than DSSP's full
secondary-structure assignment. `createdb` takes the directory path directly (not a shell glob --
256k files blow past ARG_MAX for `ls *.cif`, confirmed the hard way).

Usage:
    python scripts/build_foldseek_db.py                 # createdb + createindex, full pool
    python scripts/build_foldseek_db.py --limit-copy N  # smoke test on a small subset first
"""

import argparse
import subprocess
import time
from pathlib import Path

FOLDSEEK = Path("tools/foldseek/bin/foldseek")
STRUCTURES = Path("data/structures_all")
DB_DIR = Path("tools/foldseek_db")
DB_PATH = DB_DIR / "db"
TMP_DIR = Path("tools/foldseek_tmp")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args = parser.parse_args()

    if not FOLDSEEK.exists():
        raise SystemExit(f"{FOLDSEEK} not found -- download the release tarball first (see module docstring).")

    DB_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/2] createdb over {STRUCTURES} ({sum(1 for _ in STRUCTURES.glob('*.cif')):,} files) ...")
    t0 = time.time()
    subprocess.run([str(FOLDSEEK), "createdb", str(STRUCTURES), str(DB_PATH)], check=True)
    print(f"  done in {(time.time()-t0)/60:.1f}m")

    print("\n[2/2] createindex (builds the search index for fast querying) ...")
    t0 = time.time()
    subprocess.run([str(FOLDSEEK), "createindex", str(DB_PATH), str(TMP_DIR)], check=True)
    print(f"  done in {(time.time()-t0)/60:.1f}m")

    print(f"\nDone. Query with:\n"
          f"  {FOLDSEEK} easy-search <query.cif> {DB_PATH} <out.m8> tools/foldseek_query_tmp")


if __name__ == "__main__":
    main()
