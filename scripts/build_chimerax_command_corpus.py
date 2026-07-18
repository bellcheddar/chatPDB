#!/usr/bin/env python3
"""
build_chimerax_command_corpus.py — introspect the real, installed ChimeraX command set as ground
truth for the new gen_chimerax_script generator (round 5). ChimeraX has no importable Python module
outside its own bundled interpreter, so this spawns ChimeraX headless
(`--nogui --silent --exit --script`) and has it dump its own command registry
(chimerax.core.commands.cli.registered_commands + cli.usage per command) to a JSON file, which this
wrapper then reads back and converts to CSV — same "ask the real tool what it can do" pattern as
build_pymol_command_corpus.py.

Usage:
    python scripts/build_chimerax_command_corpus.py
"""
from __future__ import annotations

import csv
import json
import subprocess
import tempfile
from pathlib import Path

CHIMERAX_BIN = Path("/Applications/ChimeraX-1.10.1.app/Contents/MacOS/ChimeraX")
OUT = Path("data/corpus/chimerax/chimerax_commands.csv")

INNER_SCRIPT = """
import json
from chimerax.core.commands import cli

names = sorted(cli.registered_commands(multiword=True))
rows = []
for name in names:
    try:
        usage = cli.usage(session, name)
    except Exception:
        usage = ""
    rows.append({"command": name, "usage": usage})

with open(r"__OUT_PATH__", "w") as f:
    json.dump(rows, f)
"""


def main() -> None:
    if not CHIMERAX_BIN.exists():
        print(f"[warn] ChimeraX not found at {CHIMERAX_BIN} — skipping")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_out = Path(tmpdir) / "chimerax_commands.json"
        script_path = Path(tmpdir) / "introspect.py"
        script_path.write_text(INNER_SCRIPT.replace("__OUT_PATH__", str(tmp_out)))

        print("Spawning headless ChimeraX to introspect its own command registry ...")
        result = subprocess.run(
            [str(CHIMERAX_BIN), "--nogui", "--silent", "--exit", "--script", str(script_path)],
            capture_output=True, text=True, timeout=120,
        )
        if not tmp_out.exists():
            print("[error] ChimeraX introspection did not produce output")
            print(result.stdout[-2000:])
            print(result.stderr[-2000:])
            return
        rows = json.loads(tmp_out.read_text())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["command", "usage"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  {len(rows):,} real ChimeraX commands introspected via chimerax.core.commands.cli")
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
