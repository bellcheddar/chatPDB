#!/usr/bin/env python3
"""
eval_rescore.py — re-score an existing raw_results.json with updated metric functions.

Loads cached model outputs and re-applies eval/metrics.py's current (possibly fixed/tuned) logic
without re-running any model servers. Ported near-verbatim from chem_sage's real eval_rescore.py —
genuinely useful for iterating on regex/tolerance tuning cheaply.

Usage:
    cd chatPDB/
    python eval/compare/eval_rescore.py
    python eval/compare/eval_rescore.py --input eval/compare/results/raw_results.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics import METRICS, build_lookup, score_all  # noqa: E402
from eval.compare.eval_compare import html_report, md_report  # noqa: E402
from scripts.build_dataset import load_corpus  # noqa: E402


def rescore(raw: dict, lookup: dict) -> dict:
    for mid, model_data in raw.items():
        if model_data.get("skipped"):
            continue
        results = model_data.get("results", [])
        for res in results:
            res["scores"] = score_all(res.get("output", ""), res.get("expected", ""), lookup)

        agg: dict[str, tuple[int, int]] = {}
        for key, *_ in METRICS:
            agg[key] = (
                sum(r["scores"][key][0] for r in results),
                sum(r["scores"][key][1] for r in results),
            )
        model_data["aggregate"] = agg
    return raw


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-score chatPDB eval from cached outputs")
    ap.add_argument("--input", type=Path, default=Path(__file__).parent / "results" / "raw_results.json")
    ap.add_argument("--config", type=Path, default=Path(__file__).parent / "models.yaml")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "results")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"raw_results.json not found: {args.input}")

    cfg_raw = yaml.safe_load(args.config.read_text())
    settings = cfg_raw["settings"]
    all_cfgs = cfg_raw["models"]

    raw = json.loads(args.input.read_text())
    print(f"Loaded cached results for: {[k for k in raw if not raw[k].get('skipped')]}")
    print("Loading corpus tables for ground-truth lookup…")
    lookup = build_lookup(load_corpus())

    print("Re-scoring with updated metric functions…")
    rescored = rescore(raw, lookup)

    print(f"\n{'Model':<22} {'ID Valid':>9} {'Exec':>8} {'Fidelity':>10}")
    print("─" * 54)
    for cfg in all_cfgs:
        mid = cfg["id"]
        er = rescored.get(mid)
        if not er or er.get("skipped"):
            continue
        agg = er["aggregate"]

        def pct(k):
            ok, tot = agg[k]
            return f"{ok/tot:.0%}" if tot else "n/a"

        print(f"  {cfg['display_name']:<20} {pct('pdb_id_validity'):>9} "
              f"{pct('tool_executability'):>8} {pct('numerical_fidelity'):>10}")
    print()

    rescored_path = args.out_dir / "raw_results_rescored.json"
    rescored_path.write_text(json.dumps(rescored, indent=2, default=str))
    print(f"Rescored JSON → {rescored_path}")

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    html_path = args.out_dir / f"compare_{timestamp}.html"
    md_path = args.out_dir / f"compare_{timestamp}.md"

    html_path.write_text(html_report(all_cfgs, rescored, settings, generated_at))
    md_path.write_text(md_report(all_cfgs, rescored, settings, generated_at))
    print(f"HTML  → {html_path}")
    print(f"MD    → {md_path}")


if __name__ == "__main__":
    main()
