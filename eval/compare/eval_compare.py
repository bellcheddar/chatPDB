#!/usr/bin/env python3
"""
eval_compare.py — side-by-side multi-round comparison eval for chatPDB (Phase 7).

Spawns one mlx_lm.server per model, runs N shared examples from the frozen test set, aggregates
every metric in eval/metrics.py, and writes .html + .md reports. Ported wholesale from chem_sage's
real eval/compare/eval_compare.py: server orchestration, ResourceMonitor (psutil CPU/RSS sampling),
--resume via cached raw_results.json, HTML+MD reports with a Plotly val-loss curve.

Starts with a single model (chatPDB has shipped one training round) — models.yaml is designed for
append-only extension, same convention as the top-level README's Models/Training tables.

Usage:
    cd chatPDB/
    python eval/compare/eval_compare.py
    python eval/compare/eval_compare.py --resume          # skip already-run models
    python eval/compare/eval_compare.py --models chatpdb_32b_v1
    python eval/compare/eval_compare.py --limit 10        # quick smoke-test
"""

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import textwrap
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
import yaml

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics import METRICS, build_lookup, score_all  # noqa: E402
from scripts.build_dataset import load_corpus  # noqa: E402

EVAL_SYSTEM = (PROJECT_ROOT / "config" / "system_prompt.txt").read_text()

# ─── resource monitor ──────────────────────────────────────────────────────────

class ResourceMonitor:
    """Samples CPU % and RSS of a server PID every 3 seconds in a daemon thread.

    Ported as-is from chem_sage — fully generic, no chemistry/protein dependency.
    """

    def __init__(self, pid: int):
        self._pid = pid
        self.cpu_samples: list[float] = []
        self.rss_samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        if _HAS_PSUTIL:
            self._thread.start()

    def _run(self):
        try:
            proc = psutil.Process(self._pid)
            proc.cpu_percent(interval=None)   # prime the counter
            while not self._stop.wait(3.0):
                try:
                    children = proc.children(recursive=True)
                    cpu = proc.cpu_percent(interval=None) + sum(
                        c.cpu_percent(interval=None) for c in children
                    )
                    rss = proc.memory_info().rss + sum(
                        c.memory_info().rss for c in children
                    )
                    self.cpu_samples.append(cpu)
                    self.rss_samples.append(rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
        except Exception:
            pass

    def stop(self) -> dict:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        _m = lambda xs: round(statistics.mean(xs), 1) if xs else None
        _x = lambda xs: round(max(xs), 1) if xs else None
        return {
            "cpu_mean_pct": _m(self.cpu_samples),
            "cpu_peak_pct": _x(self.cpu_samples),
            "rss_mean_gb": round(statistics.mean(self.rss_samples) / 1e9, 2) if self.rss_samples else None,
            "rss_peak_gb": round(max(self.rss_samples) / 1e9, 2) if self.rss_samples else None,
            "n_samples": len(self.cpu_samples),
        }


# ─── server management ─────────────────────────────────────────────────────────

def _start_server(model_path: str, port: int) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "mlx_lm.server", "--model", model_path, "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )


def _wait_for_server(port: int, timeout: int) -> bool:
    url = f"http://localhost:{port}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        except Exception:
            pass
        time.sleep(3)
    return False


def _stop_server(proc: subprocess.Popen):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()


# ─── test set sampling ─────────────────────────────────────────────────────────

def sample_examples(test_file: Path, n: int, seed: int) -> list[dict]:
    import random
    lines = [json.loads(l) for l in test_file.read_text().splitlines() if l.strip()]
    if len(lines) <= n:
        return lines
    return random.Random(seed).sample(lines, n)


# ─── model query ───────────────────────────────────────────────────────────────

def query_model(port: int, messages: list[dict], timeout: int = 120, model_path: str = "chatpdb") -> tuple[str, float]:
    url = f"http://localhost:{port}/v1/chat/completions"
    t0 = time.time()
    try:
        resp = requests.post(url, json={
            "model": model_path,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1024,
        }, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"], time.time() - t0
    except Exception as e:
        return f"[error] {e}", time.time() - t0


# ─── per-model eval loop ───────────────────────────────────────────────────────

def run_model_eval(
    model_cfg: dict,
    examples: list[dict],
    settings: dict,
    lookup: dict,
    partial_file: Optional[Path] = None,
) -> dict:
    from tqdm import tqdm

    port = settings["port"]
    timeout = settings.get("query_timeout", 120)
    su_timeout = settings.get("server_startup_timeout", 200)

    model_path = str(PROJECT_ROOT / model_cfg["path"])
    print(f"\n{'─'*62}")
    print(f"  {model_cfg['display_name']}  ·  {model_cfg['path']}")
    print(f"{'─'*62}")

    if not Path(model_path).exists():
        print(f"  ⚠  Model path not found: {model_path}  — skipping.")
        return {"model_id": model_cfg["id"], "skipped": True, "reason": "path not found"}

    proc = _start_server(model_path, port)
    print(f"  ⏳ Waiting for server on :{port} (up to {su_timeout}s)…", end="", flush=True)
    ready = _wait_for_server(port, timeout=su_timeout)
    if not ready:
        _stop_server(proc)
        return {"model_id": model_cfg["id"], "skipped": True, "reason": "server startup timeout"}
    print(" ready.")

    monitor = ResourceMonitor(proc.pid)
    monitor.start()

    results = []
    t_start = time.time()
    errors = 0

    pbar = tqdm(examples, desc=f"  {model_cfg['display_name'][:22]:<22}", unit="ex")
    for ex in pbar:
        messages = [m for m in ex.get("messages", []) if m["role"] in ("system", "user")]
        if not any(m["role"] == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": EVAL_SYSTEM})

        prompt = next((m["content"] for m in messages if m["role"] == "user"), "")
        expected = next(
            (m["content"] for m in reversed(ex.get("messages", [])) if m["role"] == "assistant"), "",
        )

        output, latency = query_model(port, messages, timeout=timeout, model_path=model_path)
        if output.startswith("[error]"):
            errors += 1

        scores = score_all(output, expected, lookup)
        results.append({
            "prompt": prompt,
            "expected": expected,
            "output": output,
            "latency": round(latency, 2),
            "scores": scores,
        })

        idv_ok = sum(r["scores"]["pdb_id_validity"][0] for r in results)
        idv_tot = sum(r["scores"]["pdb_id_validity"][1] for r in results)
        exe_ok = sum(r["scores"]["tool_executability"][0] for r in results)
        exe_tot = sum(r["scores"]["tool_executability"][1] for r in results)
        pbar.set_postfix_str(
            f"ID {idv_ok}/{idv_tot}  Exec {exe_ok}/{exe_tot}"
            + (f"  err={errors}" if errors else ""),
            refresh=True,
        )

        if partial_file is not None:
            try:
                partial_agg = {
                    key: (
                        sum(r["scores"][key][0] for r in results),
                        sum(r["scores"][key][1] for r in results),
                    )
                    for key, *_ in METRICS
                }
                partial_file.write_text(json.dumps({
                    "model_id": model_cfg["id"],
                    "n": len(results),
                    "aggregate": partial_agg,
                }, default=str))
            except Exception:
                pass

    total_secs = time.time() - t_start

    agg: dict[str, tuple[int, int]] = {}
    for key, *_ in METRICS:
        agg[key] = (
            sum(r["scores"][key][0] for r in results),
            sum(r["scores"][key][1] for r in results),
        )

    latencies = [r["latency"] for r in results if not r["output"].startswith("[error]")]
    word_count = sum(len(r["output"].split()) for r in results)
    mean_latency = round(statistics.mean(latencies), 1) if latencies else 0
    tok_per_sec = round(word_count / total_secs, 1) if total_secs else 0

    resource_stats = monitor.stop()
    _stop_server(proc)

    return {
        "model_id": model_cfg["id"],
        "skipped": False,
        "results": results,
        "aggregate": agg,
        "resource": resource_stats,
        "mean_latency": mean_latency,
        "tok_per_sec": tok_per_sec,
        "total_min": round(total_secs / 60, 1),
        "n": len(results),
        "errors": errors,
    }


# ─── loss curve (Plotly, matches chem_sage's real default — SVG is dead-code fallback only) ────

def _loss_curve_plotly(model_configs: list[dict]) -> str:
    try:
        import plotly.graph_objects as go
    except ImportError:
        return "<p><em>plotly not installed — loss curve unavailable</em></p>"

    fig = go.Figure()
    for cfg in model_configs:
        curve = cfg.get("val_loss_curve", [])
        pts = [(p[0], p[1]) for p in curve if p[1] is not None]
        if not pts:
            continue
        iters, losses = zip(*pts)
        fig.add_trace(go.Scatter(
            x=list(iters), y=list(losses),
            name=cfg.get("display_name", "?"), mode="lines",
            line=dict(color=cfg.get("color", "#888")),
        ))
    fig.update_layout(
        xaxis_title="Iteration", yaxis_title="Val Loss",
        template="plotly_white", height=380, margin=dict(l=50, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})


# ─── reports ────────────────────────────────────────────────────────────────────

def _pct(ok: int, tot: int, na: str = "n/a") -> str:
    return f"{ok/tot:.0%}" if tot else na


def html_report(model_configs: list[dict], eval_results: dict, settings: dict, generated_at: str) -> str:
    metric_rows = ""
    for key, label, *_ in METRICS:
        cells = ""
        for cfg in model_configs:
            er = eval_results.get(cfg["id"])
            if er and not er.get("skipped"):
                ok, tot = er["aggregate"].get(key, (0, 0))
                cells += f"<td class='num'>{_pct(ok, tot)}<br><small>{ok}/{tot}</small></td>"
            else:
                cells += "<td class='num'>—</td>"
        metric_rows += f"<tr><td>{label}</td>{cells}</tr>\n"

    runtime_rows = ""
    for label, fn in (
        ("Mean latency (s)", lambda e: str(e["mean_latency"])),
        ("Throughput (words/s)", lambda e: str(e["tok_per_sec"])),
        ("Eval time (min)", lambda e: str(e["total_min"])),
        ("Server CPU mean %", lambda e: str(e["resource"].get("cpu_mean_pct", "—"))),
        ("Server RSS peak GB", lambda e: str(e["resource"].get("rss_peak_gb", "—"))),
    ):
        cells = ""
        for cfg in model_configs:
            er = eval_results.get(cfg["id"])
            cells += f"<td class='num'>{fn(er)}</td>" if er and not er.get("skipped") else "<td class='num'>—</td>"
        runtime_rows += f"<tr><td>{label}</td>{cells}</tr>\n"

    headers = "".join(f"<th>{cfg['display_name']}</th>" for cfg in model_configs)
    loss_curve_html = _loss_curve_plotly(model_configs)

    return textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>chatPDB Comparative Eval — {generated_at}</title>
        <style>
          :root {{ --navy: #1C244B; --accent: #467FF7; --bg: #f5f7fb; --text: #1a1a2e; --card: #fff; --border: #e4e8f0; }}
          * {{ box-sizing: border-box; margin: 0; padding: 0; }}
          body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text);
                  padding: 2.5rem 2rem; max-width: 1200px; margin: auto; }}
          h1 {{ color: var(--navy); font-size: 1.7rem; margin-bottom: .25rem; }}
          h2 {{ color: var(--navy); font-size: 1.1rem; margin: 2rem 0 .8rem; }}
          .sub {{ color: #666; font-size: .82rem; margin-bottom: 2rem; }}
          table {{ width: 100%; border-collapse: collapse; background: var(--card); border-radius: 12px;
                    overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,.06); margin-bottom: 1.5rem; font-size: .8rem; }}
          th {{ background: var(--navy); color: #fff; text-align: left; padding: .6rem .9rem; font-size: .78rem; }}
          td {{ padding: .5rem .9rem; border-bottom: 1px solid var(--border); }}
          td.num {{ text-align: center; }}
          td.num small {{ color: #999; }}
          tr:last-child td {{ border-bottom: none; }}
          .card {{ background: var(--card); border-radius: 12px; padding: 1rem 1.4rem;
                    box-shadow: 0 2px 10px rgba(0,0,0,.06); margin-bottom: 1.5rem; }}
          .footer {{ margin-top: 2rem; font-size: .72rem; color: #aaa; border-top: 1px solid var(--border); padding-top: 1rem; }}
          a {{ color: var(--accent); text-decoration: none; }}
        </style>
        </head>
        <body>

        <h1>chatPDB Comparative Evaluation</h1>
        <div class="sub">Generated {generated_at} &nbsp;·&nbsp; {settings.get('n_examples')} examples/model (seed={settings.get('sample_seed')})</div>

        <h2>Metrics</h2>
        <table>
          <thead><tr><th>Metric</th>{headers}</tr></thead>
          <tbody>{metric_rows}</tbody>
        </table>

        <h2>Val loss</h2>
        <div class="card">{loss_curve_html}</div>

        <h2>Eval runtime stats</h2>
        <table>
          <thead><tr><th></th>{headers}</tr></thead>
          <tbody>{runtime_rows}</tbody>
        </table>

        <div class="footer">
          Generated by chatPDB comparative eval harness &nbsp;·&nbsp;
          <a href="https://marcdeller.com">Marc C. Deller, D.Phil.</a> &nbsp;·&nbsp;
          <a href="mailto:marc@marcdeller.com">marc@marcdeller.com</a>
        </div>
        </body>
        </html>
    """)


def md_report(model_configs: list[dict], eval_results: dict, settings: dict, generated_at: str) -> str:
    lines = [
        "# chatPDB Comparative Evaluation",
        "",
        f"Generated {generated_at} · {settings.get('n_examples')} examples/model (seed={settings.get('sample_seed')})",
        "",
        "## Metrics",
        "",
    ]
    header = "| Metric | " + " | ".join(c["display_name"] for c in model_configs) + " |"
    sep = "|---|" + "---|" * len(model_configs)
    lines += [header, sep]
    for key, label, *_ in METRICS:
        row = f"| {label} |"
        for cfg in model_configs:
            er = eval_results.get(cfg["id"])
            if er and not er.get("skipped"):
                ok, tot = er["aggregate"].get(key, (0, 0))
                row += f" {_pct(ok, tot)} ({ok}/{tot}) |"
            else:
                row += " — |"
        lines.append(row)

    lines += ["", "## Eval runtime stats", ""]
    rt_header = "| | " + " | ".join(c["display_name"] for c in model_configs) + " |"
    lines += [rt_header, sep]
    for label, fn in (
        ("Mean latency (s)", lambda e: str(e["mean_latency"])),
        ("Throughput (words/s)", lambda e: str(e["tok_per_sec"])),
        ("Eval time (min)", lambda e: str(e["total_min"])),
        ("Server CPU mean %", lambda e: str(e["resource"].get("cpu_mean_pct", "—"))),
        ("Server RSS peak GB", lambda e: str(e["resource"].get("rss_peak_gb", "—"))),
    ):
        row = f"| {label} |"
        for cfg in model_configs:
            er = eval_results.get(cfg["id"])
            row += f" {fn(er)} |" if er and not er.get("skipped") else " — |"
        lines.append(row)

    lines += ["", "---", "", "*Generated by chatPDB comparative eval harness — [marcdeller.com](https://marcdeller.com)*"]
    return "\n".join(lines)


# ─── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="chatPDB multi-round comparative eval")
    ap.add_argument("--config", type=Path, default=Path(__file__).parent / "models.yaml")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "results")
    ap.add_argument("--resume", action="store_true", help="Load cached raw_results.json and skip already-run models")
    ap.add_argument("--models", nargs="*", help="Subset of model ids to run (default: all)")
    ap.add_argument("--limit", type=int, help="Evaluate only the first N examples (smoke-test)")
    args = ap.parse_args()

    cfg_raw = yaml.safe_load(args.config.read_text())
    settings = cfg_raw["settings"]
    all_cfgs = cfg_raw["models"]

    if args.models:
        all_cfgs = [c for c in all_cfgs if c["id"] in args.models]
        if not all_cfgs:
            raise SystemExit(f"No models matched: {args.models}")

    test_file = PROJECT_ROOT / settings["test_file"]
    if not test_file.exists():
        raise SystemExit(f"Test file not found: {test_file}")

    n = args.limit if args.limit else settings["n_examples"]
    examples = sample_examples(test_file, n, settings["sample_seed"])
    print(f"Sampled {len(examples)} examples (seed={settings['sample_seed']})")

    print("Loading corpus tables for ground-truth lookup…")
    lookup = build_lookup(load_corpus())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = args.out_dir / "raw_results.json"
    partial_file = args.out_dir / "partial_results.json"

    eval_results: dict = {}
    if args.resume and cache_file.exists():
        eval_results = json.loads(cache_file.read_text())
        print(f"Loaded cached results for: {list(eval_results.keys())}")

    for cfg in all_cfgs:
        mid = cfg["id"]
        if mid in eval_results:
            print(f"  ↩  Skipping {cfg['display_name']} (cached)")
            continue
        result = run_model_eval(cfg, examples, settings, lookup, partial_file=partial_file)
        eval_results[mid] = result
        cache_file.write_text(json.dumps(eval_results, indent=2, default=str))
        partial_file.unlink(missing_ok=True)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    html_path = args.out_dir / f"compare_{timestamp}.html"
    md_path = args.out_dir / f"compare_{timestamp}.md"

    html_path.write_text(html_report(all_cfgs, eval_results, settings, generated_at))
    md_path.write_text(md_report(all_cfgs, eval_results, settings, generated_at))

    print(f"\n{'═'*62}")
    print(f"  HTML report → {html_path}")
    print(f"  MD report   → {md_path}")
    print(f"  Raw results → {cache_file}")
    print(f"{'═'*62}\n")

    print(f"\n{'─'*62}")
    print(f"  {'Model':<22} {'ID Valid':>9} {'Exec':>8} {'Fidelity':>10} {'Val Loss':>10}")
    print(f"{'─'*62}")
    for cfg in all_cfgs:
        er = eval_results.get(cfg["id"])
        star = " ★" if cfg.get("is_baseline") else "  "
        if er and not er.get("skipped"):
            idv = _pct(*er["aggregate"]["pdb_id_validity"])
            exe = _pct(*er["aggregate"]["tool_executability"])
            fid = _pct(*er["aggregate"]["numerical_fidelity"])
        else:
            idv = exe = fid = "—"
        bvl = cfg.get("best_val_loss", "?")
        print(f"  {cfg['display_name'] + star:<22} {idv:>9} {exe:>8} {fid:>10} {str(bvl):>10}")
    print(f"{'─'*62}\n")


if __name__ == "__main__":
    main()
