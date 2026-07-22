#!/usr/bin/env python3
"""
eval_pdb.py — chatPDB single-model evaluation harness (Phase 7).

Queries an OpenAI-compatible endpoint (mlx_lm.server) against a seeded sample of the frozen test
split and scores every response with eval/metrics.py's real, corpus-grounded metrics. Ported from
chem_sage's real eval/eval_chem.py structure (model invocation, test loading, HTML report style).

Metrics (see eval/metrics.py for real logic — every ground truth is either live corpus lookup or
live tool re-execution, nothing hand-authored):
  - PDB ID validity          every stated PDB ID resolves to a real corpus entry
  - Cross-reference accuracy stated PDB<->UniProt/CATH/EC mappings match real SIFTS data
  - Tool executability       every emitted Biopython block runs without error
  - Numerical fidelity       stated resolution/R-free/chain counts match the real corpus
  - Refusal accuracy         out-of-scope structure-prediction requests correctly declined
  - Degeneration-free        no repetition-collapse

Usage:
    mlx_lm.server --model models/chatpdb_32b_v1 --port 8080   # separate terminal
    python eval/eval_pdb.py --n 20                             # smoke test
    python eval/eval_pdb.py --n 200 --out eval/results/scorecard.html
"""

import argparse
import json
import random
import sys
import textwrap
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics import METRICS, build_lookup, score_all  # noqa: E402
from scripts.build_dataset import load_corpus  # noqa: E402

_EVAL_SYSTEM = Path(PROJECT_ROOT / "config" / "system_prompt.txt").read_text()


# ---------------------------------------------------------------------------
# Model query
# ---------------------------------------------------------------------------

def query_model(base_url: str, model: str, messages: list[dict]) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    try:
        resp = requests.post(
            url,
            json={"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 1024},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        return f"[error] cannot connect to {url} — is mlx_lm.server running?"
    except Exception as e:
        return f"[error] {e}"


# ---------------------------------------------------------------------------
# HTML scorecard (marcdeller.com palette, matching chem_sage's report style)
# ---------------------------------------------------------------------------

def _pct(num: int, denom: int) -> str:
    return f"{num / denom:.0%}" if denom else "n/a"


def _card(val: str, label: str, sub: str = "") -> str:
    body = f"<div class='val'>{val}</div><div class='lbl'>{label}"
    body += f"<br><small>{sub}</small>" if sub else ""
    body += "</div>"
    return f"<div class='card'>{body}</div>"


def _html_scorecard(results: list[dict], model: str, base_url: str, n_test: int) -> str:
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in results:
        for key, *_ in METRICS:
            ok, tot = r["scores"][key]
            agg[key][0] += ok
            agg[key][1] += tot

    lats = [r["latency"] for r in results if r.get("latency", 0) > 0]
    mean_lat = f"{sum(lats)/len(lats):.1f}s" if lats else "n/a"

    cards = "".join(
        _card(_pct(*agg[key]), label, f"{agg[key][0]}/{agg[key][1]}")
        for key, label, *_ in METRICS
    )

    rows = ""
    for i, r in enumerate(results[:100], 1):
        q_display = (r["prompt"][:80] + "…") if len(r["prompt"]) > 80 else r["prompt"]
        resp_safe = r["output"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        resp_prev = (resp_safe[:400] + "…") if len(resp_safe) > 400 else resp_safe
        metric_cells = "".join(
            f"<td class='num'>{r['scores'][key][0]}/{r['scores'][key][1]}</td>"
            for key, *_ in METRICS
        )
        rows += (
            f"<tr><td class='num'>{i}</td><td class='q'>{q_display}</td>"
            f"{metric_cells}"
            f"<td><details><summary>view</summary>"
            f"<div class='resp'>{resp_prev}</div></details></td></tr>\n"
        )

    metric_headers = "".join(f"<th>{label}</th>" for _, label, *_ in METRICS)

    return textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>chatPDB Eval — {datetime.now():%Y-%m-%d %H:%M}</title>
        <style>
          :root {{
            --navy: #1C244B; --accent: #467FF7; --bg: #f5f7fb;
            --text: #1a1a2e; --card: #ffffff; --border: #e4e8f0;
          }}
          * {{ box-sizing: border-box; margin: 0; padding: 0; }}
          body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg); color: var(--text);
            padding: 2.5rem 2rem; max-width: 1300px; margin: auto;
          }}
          h1 {{ color: var(--navy); font-size: 1.7rem; margin-bottom: .25rem; }}
          .sub {{ color: #666; font-size: .82rem; margin-bottom: 2rem; }}
          .cards {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }}
          .card {{
            background: var(--card); border-radius: 12px; border-top: 3px solid var(--accent);
            padding: 1.1rem 1.4rem; box-shadow: 0 2px 10px rgba(0,0,0,.06);
            min-width: 150px; text-align: center; flex: 1;
          }}
          .card .val {{ font-size: 2rem; font-weight: 700; color: var(--accent); }}
          .card .lbl {{ font-size: .76rem; color: #666; margin-top: .3rem; line-height: 1.5; }}
          .card .lbl small {{ font-size: .68rem; color: #999; }}
          table {{
            width: 100%; border-collapse: collapse;
            background: var(--card); border-radius: 12px;
            overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,.06);
            margin-bottom: 1.5rem; font-size: .77rem;
          }}
          th {{
            background: var(--navy); color: #fff;
            text-align: left; padding: .6rem .9rem; font-size: .75rem; font-weight: 600;
          }}
          td {{ padding: .45rem .9rem; border-bottom: 1px solid var(--border); }}
          td.q {{ max-width: 280px; word-break: break-word; }}
          td.num {{ text-align: center; white-space: nowrap; }}
          tr:last-child td {{ border-bottom: none; }}
          tr:nth-child(even) {{ background: #f9fafd; }}
          details summary {{ cursor: pointer; color: var(--accent); font-size: .72rem; }}
          .resp {{
            background: #f0f4ff; border-radius: 6px; padding: .5rem .7rem;
            font-size: .7rem; max-height: 220px; overflow-y: auto;
            margin-top: .35rem; white-space: pre-wrap; word-break: break-word;
            font-family: monospace;
          }}
          .footer {{
            margin-top: 2rem; font-size: .72rem; color: #aaa;
            border-top: 1px solid var(--border); padding-top: 1rem;
          }}
          a {{ color: var(--accent); text-decoration: none; }}
          a:hover {{ text-decoration: underline; }}
        </style>
        </head>
        <body>

        <h1>chatPDB Evaluation Scorecard</h1>
        <div class="sub">
          Model: <b>{model}</b> &nbsp;·&nbsp; {base_url}
          &nbsp;·&nbsp; {datetime.now():%Y-%m-%d %H:%M}
          &nbsp;·&nbsp; {n_test} test examples
          &nbsp;·&nbsp; mean latency {mean_lat}
        </div>

        <div class="cards">{cards}</div>

        <h2 style="color:var(--navy);font-size:1.1rem;margin:1.5rem 0 .8rem;">
          Per-example results <span style="font-weight:400;font-size:.85rem;color:#999;">(first 100 shown)</span>
        </h2>
        <table>
          <thead><tr><th>#</th><th>Prompt</th>{metric_headers}<th>Response</th></tr></thead>
          <tbody>
        {rows}  </tbody>
        </table>

        <div class="footer">
          Generated by chatPDB eval harness &nbsp;·&nbsp;
          <a href="https://marcdeller.com">Marc C. Deller, D.Phil.</a> &nbsp;·&nbsp;
          <a href="mailto:marc@marcdeller.com">marc@marcdeller.com</a>
        </div>
        </body>
        </html>
    """)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="chatPDB evaluation harness — runs against mlx_lm.server")
    ap.add_argument("--base-url", default="http://localhost:8080/v1")
    ap.add_argument(
        "--model", default=str(PROJECT_ROOT / "models" / "chatpdb_32b_v1"),
        help="Model id as mlx_lm.server resolves it — must exactly match the path passed to "
             "`mlx_lm.server --model ...` (check http://localhost:8080/v1/models if unsure); "
             "the server does NOT accept an arbitrary display name.",
    )
    ap.add_argument("--test", type=Path, default=Path("data/sft/test.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("eval/results/scorecard.html"))
    ap.add_argument("--n", type=int, default=200, help="Number of examples to sample (default: 200)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.test.exists():
        raise SystemExit(f"Test set not found: {args.test}. Run scripts/build_dataset.py first.")

    all_examples = [json.loads(l) for l in args.test.read_text().splitlines() if l.strip()]
    examples = (
        all_examples if len(all_examples) <= args.n
        else random.Random(args.seed).sample(all_examples, args.n)
    )
    if not examples:
        raise SystemExit("Test file is empty.")

    print(f"Sampled {len(examples)} of {len(all_examples)} examples (seed={args.seed})")

    corpus = load_corpus()
    lookup = build_lookup(corpus)

    from tqdm import tqdm

    print(f"Evaluating {len(examples)} examples against {args.base_url} ({args.model})")

    results: list[dict] = []
    errors = 0
    running: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    pbar = tqdm(examples, unit="ex", dynamic_ncols=True)
    for ex in pbar:
        messages = [m for m in ex["messages"] if m["role"] in ("system", "user")]
        if not any(m["role"] == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": _EVAL_SYSTEM})

        prompt = next((m["content"] for m in messages if m["role"] == "user"), "")
        expected = next((m["content"] for m in ex["messages"] if m["role"] == "assistant"), "")

        t0 = time.time()
        output = query_model(args.base_url, args.model, messages)
        latency = time.time() - t0

        if output.startswith("[error]"):
            errors += 1

        scores = score_all(output, expected, lookup)
        for key, *_ in METRICS:
            ok, tot = scores[key]
            running[key][0] += ok
            running[key][1] += tot

        results.append({
            "prompt": prompt, "output": output, "expected": expected,
            "latency": latency, "scores": scores,
        })

        pbar.set_postfix_str(
            "  ".join(f"{label} {running[key][0]}/{running[key][1]}" for key, label, *_ in METRICS)
            + (f"  err={errors}" if errors else ""),
            refresh=True,
        )

    html = _html_scorecard(results, args.model, args.base_url, len(examples))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)

    W = 62
    print(f"\n{'─' * W}")
    print(f"  {'Metric':<28} {'Score':>8}   Detail")
    print(f"{'─' * W}")
    for key, label, *_ in METRICS:
        ok, tot = running[key]
        suffix = "" if tot else "  [n/a — not triggered by this sample]"
        print(f"  {label:<28} {_pct(ok, tot):>8}   ({ok}/{tot}){suffix}")
    print(f"{'─' * W}")
    if errors:
        print(f"  ⚠  {errors} query error{'s' if errors != 1 else ''} — check server connection.")
    print(f"\nScorecard → {args.out}\n")


if __name__ == "__main__":
    main()
