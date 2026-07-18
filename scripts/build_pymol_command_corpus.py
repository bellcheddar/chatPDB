#!/usr/bin/env python3
"""
build_pymol_command_corpus.py — introspect the real, installed PyMOL API for its complete command
set, as ground truth for gen_pymol_script (round 5). Replaces the previous 3-hardcoded-template
approach: rather than hand-picking a handful of commands to teach, enumerate every public `cmd.*`
function PyMOL 3.1.0 actually exposes, with its real docstring, so the SFT generator can sample
from (and execution-verify against) the genuine command surface.

Usage:
    python scripts/build_pymol_command_corpus.py
"""
from __future__ import annotations

import csv
import inspect
from pathlib import Path

OUT = Path("data/corpus/pymol/pymol_commands.csv")

# Commands that only make sense in an interactive GUI session (mouse/window management, editing
# undo stacks, dialogs) — real PyMOL commands, but not meaningfully scriptable/execution-verifiable
# in a headless `pymol -cq` batch run. Flagged, not excluded, so the generator can still teach their
# existence via a templated (non-executed) example.
GUI_ONLY_PREFIXES = (
    "button", "cls", "dialog", "splash", "full_screen", "gesture", "wizard", "viewport",
    "mouse", "mask", "menu", "config_mouse", "edit_mode", "editing", "undo", "redo",
    "get_wizard", "set_wizard", "rebuild", "refresh",
)


def main() -> None:
    from pymol import cmd

    print("Introspecting real PyMOL cmd API ...")
    rows = []
    for name in dir(cmd):
        if name.startswith("_"):
            continue
        if not name[0].islower():
            continue  # real PyMOL commands are lowercase; CapWords names are internal helper classes
        attr = getattr(cmd, name)
        if not callable(attr):
            continue
        doc = inspect.getdoc(attr) or ""
        try:
            real_sig = inspect.signature(attr)
            # drop the internal `_self` module-reference param -- an implementation detail
            # (PyMOL's own dispatch mechanism), never something a caller passes explicitly.
            params = [p for pname, p in real_sig.parameters.items() if pname != "_self"]
            sig = str(real_sig.replace(parameters=params))
        except (ValueError, TypeError):
            sig = ""
        gui_only = name.startswith(GUI_ONLY_PREFIXES)
        rows.append({
            "command": name,
            "signature": sig,
            "docstring": doc.replace("\n", " ").strip(),
            "gui_only": gui_only,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["command", "signature", "docstring", "gui_only"])
        writer.writeheader()
        writer.writerows(rows)

    n_documented = sum(1 for r in rows if r["docstring"])
    n_gui_only = sum(1 for r in rows if r["gui_only"])
    print(f"  {len(rows):,} real PyMOL commands introspected via dir(cmd)")
    print(f"  {n_documented:,} have a real docstring, {n_gui_only:,} flagged GUI-only")
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
