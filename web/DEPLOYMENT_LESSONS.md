# Web Deployment Lessons — learned from ChemSage (2026-07-22/23)

This file captures every gotcha hit during the ChemSage deployment so chatPDB avoids
them first time. The architecture is identical: Flask PTY on a DigitalOcean droplet,
inference on a HuggingFace ZeroGPU Space, `chat_remote.py` patching the local import.

---

## 1. ZeroGPU anonymous quota is tiny — always pass the HF token

**What happened:** The droplet called the HF Space without authentication. The anonymous
daily ZeroGPU quota (~85 s GPU-time) was exhausted within a few test calls. Every
subsequent request silently returned `event: error` from the SSE stream, which the
parser skipped, yielding 0 tokens in ~0.2 s. The user saw nothing after typing a query.

**Fix:**
- Store `HF_TOKEN=hf_...` in `/opt/<app>/.env` (same file as `HF_SPACE_URL`).
- In `chat_remote.py`, read `HF_TOKEN = os.environ.get("HF_TOKEN", "")` at module level.
- Pass `Authorization: Bearer {HF_TOKEN}` in the headers of **both** the POST (submit job)
  and the GET (poll SSE stream) requests to the Space.
- Verify the token is valid before deploying:
  ```bash
  curl -s https://huggingface.co/api/whoami-v2 -H "Authorization: Bearer hf_..." | python3 -c "import sys,json; print(json.load(sys.stdin)['name'])"
  ```

**Why silent:** The SSE parser only yielded on `data: ["text"]` (bare list) or
`data: {"msg": "process_completed", ...}`. The quota error arrives as
`data: {"error": "...", "title": "ZeroGPU quota exceeded"}` — a dict without `msg` —
so the loop continued to end-of-stream without yielding anything. Add explicit error
detection:
```python
if isinstance(data, dict) and data.get("error"):
    raise RuntimeError(f"ZeroGPU error: {data['error']}")
```

---

## 2. ZeroGPU hardware is Blackwell (CUDA 13.0) — use the cu130 wheel

**What happened:** `llama-cpp-python` installed from the cu121 wheel fails on ZeroGPU
Blackwell hardware because `libggml-cuda.so` has `DT_NEEDED: libcudart.so.12` but the
system only provides `libcudart.so.13`. LD_LIBRARY_PATH does not help (DT_RPATH
takes priority). Binary soname patching works for libcudart but fails for libcublas
due to ELF GNU version symbol mismatch.

**Fix:** Use the native CUDA 13.0 wheel — no shims needed:
```
# requirements.txt
--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu130
llama-cpp-python
```

Confirm hardware before building:
```python
import torch; print(torch.version.cuda)  # should print 13.0 on zero-a10g
```

---

## 3. rsync from Mac preserves 0600 dellboy:staff ownership — service user can't read

**What happened:** `deploy.sh` rsynced `scripts/` → `/opt/chem_sage_scripts/` and
`rag/` → `/opt/chem_sage_rag/`. The Mac files were `0600 dellboy:staff`. The service
user (`chemsage`) could not read them. The error appeared only at runtime:
```
PermissionError: [Errno 13] Permission denied: '/opt/chem_sage_scripts/chat.py'
```

**Fix:** After every rsync, explicitly chown ALL synced dirs to the service user:
```bash
chown -R <service_user>:<service_user> /opt/<app>/
chown -R <service_user>:<service_user> /opt/<app>_scripts/
chown -R <service_user>:<service_user> /opt/<app>_rag/
```
Add this to the `deploy.sh` remote block so it runs on every deploy, not just provisioning.

---

## 4. HF Space SDK: use Gradio, not Docker — ZeroGPU requires `@spaces.GPU`

Docker-based Spaces cannot use ZeroGPU. The Space must use `sdk: gradio` and the
`spaces` Python package so the `@spaces.GPU` decorator can acquire the GPU slot.

```yaml
# README.md frontmatter (the Space card)
sdk: gradio
hardware: zero-gpu
app_file: app.py
```

```python
import spaces

@spaces.GPU(duration=60)
def generate(...) -> str:
    ...
```

`duration` is the max GPU wall-time per call. 60 s is enough for a loaded model + short
inference. Cold start (model loading on first call) takes ~20–40 s for a 20 GB GGUF.

---

## 5. Model loading: download outside `@spaces.GPU`, cache the loaded model inside

```python
# At module level — runs once when the Space starts (no GPU needed for download)
from huggingface_hub import hf_hub_download
_model_path = hf_hub_download(repo_id="Dellboy/...-GGUF", filename="....gguf")

_llm = None  # cached across calls in the same ZeroGPU worker (worker lives ~48 h)

@spaces.GPU(duration=60)
def generate(prompt, max_tokens, temperature, repeat_penalty):
    global _llm
    if _llm is None:
        from llama_cpp import Llama
        _llm = Llama(model_path=_model_path, n_gpu_layers=-1, n_ctx=2048, verbose=False)
    return "".join(
        chunk["choices"][0]["text"]
        for chunk in _llm(prompt, max_tokens=int(max_tokens),
                          temperature=temperature, repeat_penalty=repeat_penalty, stream=True)
        if chunk["choices"][0]["text"]
    )
```

---

## 6. Gradio SSE call protocol (not the `/api/predict` shortcut)

ZeroGPU requires the two-step Gradio call API:
```
POST /gradio_api/call/{fn_name}
  Body: {"data": [arg1, arg2, ...]}
  → {"event_id": "..."}

GET /gradio_api/call/{fn_name}/{event_id}
  → SSE stream
  → event: complete
     data: ["result"]       ← bare list (Gradio 4.x compat format)
```

Parse BOTH response formats in `chat_remote.py`:
```python
if isinstance(data, dict) and data.get("msg") == "process_completed":
    text = (data.get("output", {}).get("data") or [""])[0]
elif isinstance(data, dict) and data.get("error"):
    raise RuntimeError(data["error"])   # quota exceeded, timeout, etc.
elif isinstance(data, list) and data:
    text = data[0] if isinstance(data[0], str) else ""
```

---

## 7. Two Spaces, one typo — name them carefully

During ChemSage, two Spaces existed: `Dellboy/chem-sage-api` (hyphen, the running one)
and `Dellboy/chem_sage-api` (underscore, a leftover). The wrong URL was in the deploy
scripts for several sessions. Before deploying chatPDB:

- Create the Space with a clear, final name.
- Write the exact URL into `deploy.sh`, `provision.sh`, and `.env` at creation time.
- Verify with: `curl -s https://<space-url>/info` or check `GET /v1/models`.

---

## 8. Confirm the Space URL by checking `/gradio_api/info` before wiring

```bash
curl -s https://<subdomain>.hf.space/gradio_api/info | python3 -m json.tool | head -30
```
This returns the API schema — confirms the Space is RUNNING and the function name/arg
order matches what `chat_remote.py` sends.

---

## 9. Don't fake the corpus lookup — it's lighter than you think

`corpus_lookup.py` only needs **pandas + CSV files**. It has no ChromaDB or
sentence-transformers dependency. If `chat_remote.py` fakes it with a no-op, the
entire keyword fast-path (the table results for "list EGFR structures", "show approved
drugs for kinase X", etc.) silently disappears.

Fix: pre-inject fakes **only** for `rag.retrieve` and `rag.tool_exec`; let
`rag.corpus_lookup` import for real. Sync the corpus CSVs to the droplet at deploy time.

The fake `rag.retrieve` is correct — `BAAI/bge-base-en-v1.5` is ~440 MB in RAM and the
`.chroma` store is ~1.4 GB; both are too heavy for a 3.8 GB droplet also running other
services. `--no-rag` is the right flag for the droplet; the corpus fast-path covers the
most common structured data queries without any ML overhead.

```bash
# deploy.sh — create the dir first, then rsync
ssh "$DROPLET" "mkdir -p $APP_DIR/data/corpus"
rsync -az --delete "$REPO_ROOT/data/corpus/" "$DROPLET:$APP_DIR/data/corpus/"
# chown it along with the rest
chown -R <service_user>:<service_user> "${APP_DIR}/data/" 2>/dev/null || true
```

The paths in `corpus_lookup.py` (`Path("data/corpus/...")`) resolve relative to the
process cwd. Make sure `WorkingDirectory=` in the systemd service matches where the
corpus is synced (`/opt/<app>/` → corpus at `/opt/<app>/data/corpus/`).

---

## 10. Test order for deployment verification

1. **HF Space direct** (from your Mac with HF token) — confirms CUDA, model load, inference.
2. **Droplet → HF Space** (SSH in, run the Python snippet from lesson 1) — confirms network + token.
3. **Flask service health** (`curl -o /dev/null -w "%{http_code}" https://<app>.mdeller.com/`) — confirms nginx + gunicorn.
4. **Browser WebSocket** (open the site, type a short query) — full end-to-end.

Do not skip step 2. It catches the quota/token issue before you wonder why the browser shows nothing.
