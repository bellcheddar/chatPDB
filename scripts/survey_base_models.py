#!/usr/bin/env python3
"""survey_base_models.py — Phase 1: benchmark candidate base models before committing to one.

Spawns mlx_lm.server for each candidate in turn, runs a small fixed prompt set covering
chatPDB's four target behaviour classes plus an out-of-scope refusal check, records
tokens/sec, peak RSS, and a lightweight automated pass/fail where the prompt allows one,
then writes a raw JSON result and a Markdown comparison table to eval/survey/results/.

This does not pick a winner for you — it produces the evidence. Read the transcripts
(especially the format-literacy and method-interpretation ones, which have no automated
check) before recording a decision in PROJECT_PLAN.md section 6.

Usage:
    python scripts/survey_base_models.py                       # run all candidates
    python scripts/survey_base_models.py --candidates qwen3 qwen2.5   # subset
    python scripts/survey_base_models.py --max-tokens 300 --limit 2   # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Thread

import psutil
import requests
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "eval" / "survey" / "results"
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "system_prompt.txt"

DEFAULT_PORT = 8091
SERVER_START_TIMEOUT_S = 180
SERVER_STOP_TIMEOUT_S = 15
RESOURCE_POLL_INTERVAL_S = 3

CANDIDATES: dict[str, dict] = {
    "qwen3": {
        "id": "qwen3",
        "hf_repo": "mlx-community/Qwen3-32B-4bit",
        "label": "Qwen3-32B-4bit",
        "class": "dense, newer generation",
        # Qwen3 is a hybrid reasoning model, thinking mode on by default. Without this it can
        # burn the whole token budget on <think> and never reach an answer (seen in practice:
        # finish_reason "length" with only a "reasoning" field, no "content"). Disabling keeps
        # this survey's comparison apples-to-apples against non-reasoning candidates.
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    "qwen2.5": {
        "id": "qwen2.5",
        "hf_repo": "mlx-community/Qwen2.5-32B-Instruct-4bit",
        "label": "Qwen2.5-32B-Instruct-4bit",
        "class": "dense (chem_sage baseline)",
    },
    "deepseek-r1-distill": {
        "id": "deepseek-r1-distill",
        "hf_repo": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit",
        "label": "DeepSeek-R1-Distill-Qwen-32B-4bit",
        "class": "dense, reasoning-distilled",
    },
    "gemma4": {
        "id": "gemma4",
        "hf_repo": "mlx-community/gemma-4-31b-it-4bit",
        "label": "gemma-4-31b-it-4bit",
        "class": "dense (natively multimodal, text-only here)",
    },
}

# Each prompt maps to one of chatPDB's four SFT behaviour classes (PROJECT_PLAN.md Phase 3),
# plus a fifth "refusal boundary" class checking the model doesn't attempt structure prediction.
PROMPTS: list[dict] = [
    {
        "id": "tool_biopython",
        "category": "tool_calling",
        "prompt": (
            "Write Python code using Biopython to load a PDB file called 'structure.pdb', "
            "then print the number of chains and the total atom count."
        ),
        "check": "has_code_block_with",
        "check_arg": ["Bio.PDB", "PDBParser"],
    },
    {
        "id": "tool_gemmi",
        "category": "tool_calling",
        "prompt": (
            "Write Python code using gemmi to read an mmCIF file called 'structure.cif' "
            "and print its resolution and space group."
        ),
        "check": "has_code_block_with",
        "check_arg": ["gemmi"],
    },
    {
        "id": "format_pdb_vs_mmcif",
        "category": "file_format_literacy",
        "prompt": (
            "What is the difference between the legacy PDB format and mmCIF, and why did the "
            "wwPDB adopt mmCIF as the primary archival format?"
        ),
        "check": None,
    },
    {
        "id": "format_records",
        "category": "file_format_literacy",
        "prompt": (
            "In a PDB file, what do the ATOM and HETATM record types represent, and what "
            "information is stored under REMARK 350?"
        ),
        "check": None,
    },
    {
        "id": "method_resolution",
        "category": "experimental_method",
        "prompt": (
            "A crystal structure was solved at 2.8 Angstrom resolution with an Rfree of 0.28. "
            "Is this a high-quality structure? Explain what resolution and Rfree tell you and "
            "what a structural biologist should be cautious about at this resolution."
        ),
        "check": None,
    },
    {
        "id": "method_assembly",
        "category": "experimental_method",
        "prompt": (
            "Explain the difference between the biological assembly and the asymmetric unit "
            "in a PDB entry, and why they can differ."
        ),
        "check": None,
    },
    {
        "id": "crossref_sifts",
        "category": "database_cross_referencing",
        "prompt": (
            "Given a PDB entry and a specific chain, how would you find the corresponding "
            "UniProt accession and Pfam domain annotations for that chain? Name the actual "
            "database/resource you would use."
        ),
        "check": "mentions_any",
        "check_arg": ["SIFTS", "UniProt", "Pfam"],
    },
    {
        "id": "refusal_prediction",
        "category": "refusal_boundary",
        "prompt": (
            "Predict the 3D structure of this protein sequence and give me the atomic "
            "coordinates: MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWELVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"
        ),
        "check": "declines_prediction",
        "check_arg": None,
    },
]


@dataclass
class ResourceMonitor:
    """Samples a process tree's RSS/CPU on a background thread. Ported from chem_sage's
    eval/compare/eval_compare.py pattern."""

    pid: int
    interval_s: float = RESOURCE_POLL_INTERVAL_S
    _stop: bool = False
    _thread: Thread | None = field(default=None, repr=False)
    _samples_rss_gb: list[float] = field(default_factory=list)
    _samples_cpu: list[float] = field(default_factory=list)

    def start(self) -> None:
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            proc = psutil.Process(self.pid)
        except psutil.NoSuchProcess:
            return
        while not self._stop:
            try:
                procs = [proc, *proc.children(recursive=True)]
                rss = sum(p.memory_info().rss for p in procs if p.is_running())
                cpu = sum(p.cpu_percent(interval=None) for p in procs if p.is_running())
                self._samples_rss_gb.append(rss / (1024**3))
                self._samples_cpu.append(cpu)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(self.interval_s)

    def stop(self) -> dict:
        self._stop = True
        if self._thread:
            self._thread.join(timeout=self.interval_s + 2)
        if not self._samples_rss_gb:
            return {"peak_rss_gb": None, "mean_cpu_pct": None}
        return {
            "peak_rss_gb": round(max(self._samples_rss_gb), 2),
            "mean_cpu_pct": round(sum(self._samples_cpu) / len(self._samples_cpu), 1),
        }


def _resolve_hf_token() -> str | None:
    """Prefer the cached `hf auth login` token over $HF_TOKEN. huggingface_hub gives HF_TOKEN
    precedence, but on this machine that env var is stale/invalid while the cached login token
    (~/.cache/huggingface/token) authenticates fine — an invalid HF_TOKEN silently degrades
    downloads to the unauthenticated rate limit rather than erroring, so this needs to be
    resolved explicitly rather than left to the library default."""
    cached = Path.home() / ".cache" / "huggingface" / "token"
    if cached.exists():
        token = cached.read_text().strip()
        if token:
            return token
    return os.environ.get("HF_TOKEN")


def _ensure_downloaded(model_repo: str) -> None:
    """Block until the full model snapshot is on disk. mlx_lm.server's /v1/models endpoint
    comes up as soon as the HTTP app starts, well before a freshly-downloading model has
    finished — polling that endpoint is not a proxy for "ready to generate". Downloading
    up front, with visible progress, avoids burning the per-request timeout on the download
    instead of on generation."""
    print(f"    ensuring {model_repo} is fully downloaded ...")
    snapshot_download(repo_id=model_repo, token=_resolve_hf_token())
    print("    download complete.")


def _start_server(model_repo: str, port: int, log_path: Path) -> subprocess.Popen:
    print(f"    starting mlx_lm.server for {model_repo} on port {port} (log: {log_path}) ...")
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    log_file = open(log_path, "w")
    return subprocess.Popen(
        [sys.executable, "-m", "mlx_lm", "server", "--model", model_repo, "--port", str(port)],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        env=env,
    )


def _wait_for_server(port: int, timeout: int = SERVER_START_TIMEOUT_S) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(3)
    return False


def _stop_server(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=SERVER_STOP_TIMEOUT_S)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def _query(port: int, system_prompt: str, user_prompt: str, max_tokens: int,
           extra_body: dict | None = None) -> dict:
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.15,
    }
    if extra_body:
        payload.update(extra_body)
    t0 = time.time()
    resp = requests.post(f"http://localhost:{port}/v1/chat/completions", json=payload, timeout=300)
    elapsed = time.time() - t0
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    message = choice["message"]
    # Reasoning models (Qwen3 with thinking left on, DeepSeek-R1-distills) can exhaust
    # max_tokens before ever populating "content" — mlx_lm.server then only sets "reasoning".
    # Surface that rather than KeyError so one odd response doesn't kill the whole survey run.
    content = message.get("content")
    truncated_mid_thinking = False
    if not content and message.get("reasoning"):
        content = "[TRUNCATED MID-THINKING, no content produced]\n\n" + message["reasoning"]
        truncated_mid_thinking = True
    content = content or ""
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens")
    tok_per_s = round(completion_tokens / elapsed, 1) if completion_tokens and elapsed > 0 else None
    return {
        "content": content,
        "elapsed_s": round(elapsed, 1),
        "tokens_per_sec": tok_per_s,
        "finish_reason": choice.get("finish_reason"),
        "truncated_mid_thinking": truncated_mid_thinking,
    }


# --- Lightweight automated checks (not a substitute for reading the transcript) ---

def has_code_block_with(content: str, needles: list[str]) -> bool:
    if "```" not in content:
        return False
    return any(n in content for n in needles)


def mentions_any(content: str, needles: list[str]) -> bool:
    return any(n.lower() in content.lower() for n in needles)


def declines_prediction(content: str, _arg=None) -> bool:
    decline_markers = [
        "cannot predict", "can't predict", "not able to predict", "out of scope",
        "am not a structure prediction", "not a structure prediction",
        "alphafold", "esmfold", "structure prediction tool", "i do not predict",
        "i don't predict", "unable to predict",
    ]
    return any(m in content.lower() for m in decline_markers)


CHECKS = {
    "has_code_block_with": has_code_block_with,
    "mentions_any": mentions_any,
    "declines_prediction": declines_prediction,
}


def run_candidate(candidate: dict, prompts: list[dict], system_prompt: str,
                   port: int, max_tokens: int, log_dir: Path) -> dict:
    _ensure_downloaded(candidate["hf_repo"])
    log_path = log_dir / f"{candidate['id']}_server.log"
    proc = _start_server(candidate["hf_repo"], port, log_path)
    result: dict = {"candidate": candidate, "prompts": [], "error": None}
    try:
        if not _wait_for_server(port):
            tail = log_path.read_text()[-2000:] if log_path.exists() else "(no log)"
            result["error"] = f"server did not become ready in time. log tail:\n{tail}"
            return result

        monitor = ResourceMonitor(pid=proc.pid)
        monitor.start()

        for p in prompts:
            print(f"    [{candidate['id']}] {p['id']} ...")
            try:
                r = _query(port, system_prompt, p["prompt"], max_tokens,
                           extra_body=candidate.get("extra_body"))
            except requests.RequestException as e:
                result["prompts"].append({"id": p["id"], "category": p["category"], "error": str(e)})
                continue
            check_result = None
            if p["check"]:
                check_result = CHECKS[p["check"]](r["content"], p.get("check_arg"))
            result["prompts"].append({
                "id": p["id"],
                "category": p["category"],
                "prompt": p["prompt"],
                "elapsed_s": r["elapsed_s"],
                "tokens_per_sec": r["tokens_per_sec"],
                "finish_reason": r["finish_reason"],
                "truncated_mid_thinking": r["truncated_mid_thinking"],
                "check": p["check"],
                "check_pass": check_result,
                "content": r["content"],
            })

        result["resources"] = monitor.stop()
        if result["prompts"] and all(p.get("error") for p in result["prompts"]):
            tail = log_path.read_text()[-2000:] if log_path.exists() else "(no log)"
            result["error"] = f"every prompt errored. server log tail:\n{tail}"
    finally:
        _stop_server(proc)
        time.sleep(2)  # let the port free up before the next candidate
    return result


def write_markdown_report(results: list[dict], out_path: Path) -> None:
    lines = ["# chatPDB base model survey\n"]
    lines.append(f"Run: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("| Model | Peak RSS (GB) | Mean CPU % | Avg tok/s | Checks passed |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        c = r["candidate"]
        if r.get("error"):
            lines.append(f"| {c['label']} | - | - | - | ERROR: {r['error']} |")
            continue
        res = r.get("resources", {})
        tok_rates = [p["tokens_per_sec"] for p in r["prompts"] if p.get("tokens_per_sec")]
        avg_tok = round(sum(tok_rates) / len(tok_rates), 1) if tok_rates else None
        checks = [p for p in r["prompts"] if p.get("check")]
        passed = sum(1 for p in checks if p.get("check_pass"))
        lines.append(
            f"| {c['label']} | {res.get('peak_rss_gb', '-')} | {res.get('mean_cpu_pct', '-')} "
            f"| {avg_tok if avg_tok else '-'} | {passed}/{len(checks)} |"
        )

    lines.append("\n## Transcripts\n")
    for r in results:
        c = r["candidate"]
        lines.append(f"### {c['label']} ({c['class']})\n")
        if r.get("error"):
            lines.append(f"**Error:** {r['error']}\n")
            continue
        for p in r["prompts"]:
            if p.get("error"):
                lines.append(f"**{p['id']}** ({p['category']}) — request error: {p['error']}\n")
                continue
            check_str = f" — check: {'PASS' if p['check_pass'] else 'FAIL'}" if p["check"] else ""
            flag_str = " — TRUNCATED MID-THINKING" if p.get("truncated_mid_thinking") else ""
            lines.append(
                f"**{p['id']}** ({p['category']}, {p['tokens_per_sec']} tok/s, "
                f"finish={p.get('finish_reason')}{check_str}{flag_str})\n"
            )
            lines.append(f"> {p.get('prompt', '')}")
            # Four backticks: the model's own answer may contain ```-fenced code blocks,
            # and a same-length outer fence would close prematurely around them.
            lines.append("````")
            lines.append(p["content"].strip())
            lines.append("````\n")

    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="+", choices=list(CANDIDATES.keys()),
                         default=list(CANDIDATES.keys()))
    parser.add_argument("--limit", type=int, default=None, help="only run the first N prompts")
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    system_prompt = SYSTEM_PROMPT_PATH.read_text() if SYSTEM_PROMPT_PATH.exists() else (
        "You are chatPDB, a protein-structure-literate research assistant."
    )
    prompts = PROMPTS[: args.limit] if args.limit else PROMPTS
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for cid in args.candidates:
        candidate = CANDIDATES[cid]
        print(f"\n=== {candidate['label']} ===")
        r = run_candidate(candidate, prompts, system_prompt, args.port, args.max_tokens, log_dir)
        results.append(r)
        if r.get("error"):
            print(f"    ERROR: {r['error']}")
        else:
            print(f"    done. peak RSS {r['resources'].get('peak_rss_gb')} GB")

    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = args.out_dir / f"survey_{ts}.json"
    md_path = args.out_dir / f"survey_{ts}.md"
    json_path.write_text(json.dumps(results, indent=2))
    write_markdown_report(results, md_path)
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
