"""
chat_remote.py — drop-in launcher for web sessions.

Patches stream_generate / generate in mlx_lm to call the remote HF Space API,
then imports and runs the real chat.py unchanged.
All Rich output, corpus tables, slash commands, and CLI behaviour are identical.

Real fix relative to chem_sage's own version of this file (found + fixed 2026-07-22, chatPDB
Phase 9): the original _fake_load() returned (model, tokenizer=None) -- chat.py's own
_do_generate() calls tokenizer.apply_chat_template(...) directly (uncaught), which would crash
immediately on a None tokenizer. Fixed by loading a REAL tokenizer against the bundled tokenizer/
directory (~11.5MB: tokenizer.json, tokenizer_config.json, chat_template.jinja, config.json -- no
model weights needed). Uses plain transformers.AutoTokenizer, NOT mlx_lm.tokenizer_utils -- MLX is
Apple-Silicon-only and this file runs on Marc's Linux droplet, where mlx_lm can't even be
installed; transformers is portable and already a dependency here via sentence-transformers (RAG).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

# -- locate the chatPDB repo root (two levels up from this file) --
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Real bug found + fixed 2026-07-22 (a fourth one, beyond tokenizer/sample_utils/__main__ above):
# chat.py's own RAG modules (rag/corpus_lookup.py's CORPUS_ROOT, rag/retrieve.py's CHROMA_STORE)
# use paths relative to the process's *current working directory*, not to the repo root -- this
# only ever worked in testing because chat.py/chat_remote.py happened to be run from the repo
# root by convention. app.py (Flask, living in web/flask_app/) spawns this file as a subprocess
# WITHOUT setting cwd, so it inherits app.py's own cwd -- confirmed live: without this chdir, the
# corpus fast-path silently finds no data/corpus/ and returns "not found", and the RAG retriever
# fails to find .chroma/ and degrades to "Retriever unavailable" after a real ~50s timeout, both
# silently rather than erroring loudly. chdir before chat.py imports anything RAG-related.
os.chdir(REPO_ROOT)

HF_SPACE_URL = os.environ.get("HF_SPACE_URL", "").rstrip("/")
TOKENIZER_DIR = Path(__file__).resolve().parent / "tokenizer"

# ---------------------------------------------------------------------------
# Fake mlx_lm that calls the remote API instead of a local GPU
# ---------------------------------------------------------------------------

def _remote_stream_generate(model, tokenizer, *, prompt: str, max_tokens: int = 512,
                             sampler=None, logits_processors=None, **kwargs):
    """Yield text chunks from the HF Space streaming endpoint.

    Wraps the whole request in try/except: chat.py's own _do_generate() has no error handling
    around its stream_generate() call, so an uncaught exception here (HF Space cold-start,
    network blip, ZeroGPU quota) would kill the entire PTY subprocess -- a much worse failure mode
    in the hosted-demo context (every browser session dies) than the local-CLI context this
    function's counterpart runs in normally. Yield a single informative chunk instead of crashing.
    """
    import urllib.error
    import urllib.request

    if not HF_SPACE_URL:
        yield types.SimpleNamespace(
            text="[chatPDB backend not configured -- HF_SPACE_URL is unset. This session cannot "
                 "reach the model.]"
        )
        return

    body = json.dumps({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.15,
        "repeat_penalty": 1.15,
    }).encode()
    req = urllib.request.Request(
        f"{HF_SPACE_URL}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    token = json.loads(payload).get("token", "")
                except json.JSONDecodeError:
                    continue
                if token:
                    # Yield a fake chunk object that chat.py expects
                    yield types.SimpleNamespace(text=token)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        yield types.SimpleNamespace(
            text=f"[chatPDB backend unreachable: {exc}. The HF Space may be cold-starting "
                 f"(can take 1-2 min) or temporarily down -- try again shortly.]"
        )


def _remote_generate(model, tokenizer, *, prompt: str, max_tokens: int = 512, **kwargs) -> str:
    """Blocking version — collect all tokens and return."""
    return "".join(chunk.text for chunk in _remote_stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens))


def _load_real_tokenizer():
    """Load a real tokenizer from the bundled tokenizer/ dir -- no model weights needed.

    Confirmed live (2026-07-22): transformers.AutoTokenizer.from_pretrained() against just
    tokenizer.json / tokenizer_config.json / chat_template.jinja / config.json produces a working
    tokenizer whose apply_chat_template(enable_thinking=False) and encode() both work exactly as
    chat.py expects, with none of the ~18GB of weight files present.
    """
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))


def _fake_load(model_path, *args, **kwargs):
    """Return (dummy model, real tokenizer); the actual model weights live on the HF Space."""
    return types.SimpleNamespace(_is_remote=True), _load_real_tokenizer()


# Inject a fake mlx_lm module before chat.py imports it
_mlx_lm_fake = types.ModuleType("mlx_lm")
_mlx_lm_fake.load = _fake_load
_mlx_lm_fake.stream_generate = _remote_stream_generate
_mlx_lm_fake.generate = _remote_generate

# Also fake out sub-modules chat.py imports from mlx_lm
for _sub in ("utils", "generate", "sample_utils"):
    _m = types.ModuleType(f"mlx_lm.{_sub}")
    sys.modules[f"mlx_lm.{_sub}"] = _m

_mlx_lm_fake.utils = sys.modules["mlx_lm.utils"]
_mlx_lm_fake.generate_mod = sys.modules["mlx_lm.generate"]

# chat.py does `from mlx_lm.sample_utils import make_sampler, make_logits_processors` at the top
# of chat_loop() -- the fake sample_utils module above is otherwise empty, which would raise
# ImportError the moment chat_loop() starts (a second real bug found in chem_sage's own version of
# this file, beyond the tokenizer=None one). The remote generator ignores whatever these return
# (temperature/repeat_penalty for the actual remote call are hardcoded in
# _remote_stream_generate/_remote_generate above), so simple no-op stubs are enough.
sys.modules["mlx_lm.sample_utils"].make_sampler = lambda *a, **k: None
sys.modules["mlx_lm.sample_utils"].make_logits_processors = lambda *a, **k: None

sys.modules["mlx_lm"] = _mlx_lm_fake

# NOTE: chem_sage's own version of this file also fakes `mlx`/`mlx.core` here ("used for
# mx.set_wired_limit etc."), but scripts/chat.py never imports `mlx` or calls `mx.` anything
# directly -- and faking sys.modules["mlx"] as a bare types.ModuleType (no __spec__) actively
# breaks things: transformers' own is_mlx_available() feature probe calls
# importlib.util.find_spec("mlx"), which raises ValueError on a module with __spec__ = None,
# crashing sentence_transformers' import chain (rag/retrieve.py) the moment RAG tries to load.
# Confirmed live 2026-07-22 -- deliberately NOT ported.

# ---------------------------------------------------------------------------
# Run the real chat.py
# ---------------------------------------------------------------------------

chat_path = REPO_ROOT / "scripts" / "chat.py"

# Inject --model flag (chat.py requires it even though we don't load locally)
if "--model" not in sys.argv:
    sys.argv += ["--model", "chatpdb_32b_v1"]

## Real bug found + fixed 2026-07-22 (a third one, beyond the tokenizer and sample_utils issues
## above): chem_sage's own chat_remote.py loads chat.py under module name "chat", so chat.py's own
## `if __name__ == "__main__": main()` guard never evaluates true and main() is never called --
## the whole script silently does nothing. Loading under "__main__" instead fixes it.
spec = importlib.util.spec_from_file_location("__main__", str(chat_path))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
