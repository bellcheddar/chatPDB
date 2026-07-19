#!/usr/bin/env python3
"""
build_py3dmol_command_corpus.py — scrape 3Dmol.js's own official GLViewer API reference for
py3Dmol/3Dmol.js command awareness (round 6). Unlike PyMOL/ChimeraX, py3Dmol's Python API is a
blind `__getattr__` proxy (any attribute name becomes a JS call string with zero local validation
-- confirmed live by reading py3Dmol/__init__.py), so there's no local Python-side introspection
target. The real command surface lives in 3Dmol.js itself; this scrapes the real, official,
published API reference instead of a local `dir()`/registry call.

Usage:
    python scripts/build_py3dmol_command_corpus.py
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import requests

OUT = Path("data/corpus/py3dmol/py3dmol_commands.csv")
DOC_URL = "https://3dmol.org/doc/GLViewer.html"


def main() -> None:
    print(f"Fetching real 3Dmol.js GLViewer API reference from {DOC_URL} ...")
    resp = requests.get(DOC_URL, timeout=30)
    resp.raise_for_status()
    html = resp.text

    # Real structure confirmed live 2026-07-18 by fetching the raw page (not an LLM-summarized
    # paraphrase of it): each method is
    #   <h3 class="name has-anchor" id="METHOD">...METHOD<span class="signature">(PARAMS)</span>
    #   ...</h3><div class="description"><p>DESCRIPTION</p></div>
    pattern = re.compile(
        r'<h3 class="name has-anchor" id="([a-zA-Z_$][a-zA-Z0-9_$]*)">'
        r'.*?<span class="signature">\((.*?)\)</span>.*?</h3>'
        r'<div class="description"><p>(.*?)</p></div>',
        re.DOTALL,
    )
    rows = []
    seen = set()
    for name, sig_raw, desc_raw in pattern.findall(html):
        if name.startswith("_") or name in seen:
            continue
        seen.add(name)
        desc = re.sub(r"<[^>]+>", "", desc_raw).strip()
        desc = re.sub(r"\s+", " ", desc)
        sig = re.sub(r"<[^>]+>", "", sig_raw).strip()
        rows.append({"method": name, "signature": f"({sig})", "description": desc})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "signature", "description"])
        writer.writeheader()
        writer.writerows(rows)

    n_documented = sum(1 for r in rows if r["description"])
    print(f"  {len(rows):,} real GLViewer methods scraped from the official 3Dmol.js API docs")
    print(f"  {n_documented:,} have a real description")
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
