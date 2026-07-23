"""
chatPDB inference API — HuggingFace Space (Gradio SDK, ZeroGPU)

Exposes a Gradio API endpoint (api_name="generate") consumed by the Flask PTY app on the droplet
via Gradio's own REST protocol (POST /call/generate -> event_id, then GET /call/generate/<id> for
an SSE stream) -- NOT a hand-rolled custom FastAPI route.

Real fix history (found + fixed 2026-07-23, in order):
  1. First attempt added a custom route via `@demo.app.post("/generate")` -- confirmed live via
     curl that this silently doesn't take effect (405, `allow: GET`); Gradio's own catch-all SPA
     route claims the path instead.
  2. Second attempt used `gr.mount_gradio_app()` to mount Gradio under a FastAPI app that owns the
     real routes -- this DID fix routing, but broke ZeroGPU entirely: the Space failed to start
     with "No @spaces.GPU function detected during startup". Confirmed via web research (HF forum
     threads on this exact error) that ZeroGPU's detection hook requires Gradio's own `demo.launch()`
     to be the actual serving entrypoint -- a custom-FastAPI-primary architecture is not supported,
     regardless of whether spaces.GPU is imported/used.
  3. This version: `demo.launch()` is the real entrypoint again (restoring ZeroGPU detection), and
     the /generate contract is exposed via Gradio's own native API mechanism instead of a custom
     route -- a hidden Textbox pair wired to a `.click()` handler with `api_name="generate"`, which
     Gradio automatically exposes as `POST /call/generate` + `GET /call/generate/<event_id>` (SSE).
     This stays entirely within Gradio's own request-handling pipeline, which is what ZeroGPU's
     detection actually requires.

Cold-start note: the 19.8GB GGUF is downloaded once at container boot (module import,
below), NOT inside the @spaces.GPU-decorated function. Real bug found + fixed 2026-07-23:
the download was originally triggered lazily from inside _generate_tokens(), which meant it
ran *inside* the ZeroGPU lease window and burned GPU-seconds on a plain network transfer --
confirmed via real Space logs showing a live download stuck at 81%/16.0GB after 5m24s
wall-clock, well past the `@spaces.GPU(duration=180)` cap, with no further log lines ever
written (the lease was silently reclaimed mid-download). Downloading eagerly at import time
means the file is already local on the Space's persistent container disk by the time any
real request arrives, so the GPU lease only has to cover model load + generation.
"""
from __future__ import annotations

import json

import gradio as gr
import spaces
from huggingface_hub import hf_hub_download

REPO_ID = "Dellboy/chatpdb_32b_v1-GGUF"
FILENAME = "chatpdb_32b_v1_q4km.gguf"
N_CTX = 1536   # matches chatPDB's real training max_seq_length (config/train_config.yaml)
N_GPU_LAYERS = -1  # offload all layers to GPU

print(f"[chatPDB] downloading {FILENAME} from {REPO_ID} (one-time, outside GPU lease)...")
_MODEL_PATH = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
print(f"[chatPDB] model ready at {_MODEL_PATH}")


@spaces.GPU(duration=180)
def _generate_tokens(
    prompt: str,
    max_tokens: int,
    temperature: float,
    repeat_penalty: float,
) -> str:
    """Run inside the ZeroGPU lease; collect all tokens and return as one joined string.

    ZeroGPU-decorated functions run in a bounded lease and can't hold a live generator open
    across the function boundary, so tokens are collected here rather than streamed token-by-token
    -- the client sees compute-then-deliver rather than true live latency, a real trade-off of the
    ZeroGPU model.

    Real fix (found + fixed 2026-07-23): the CUDA-built llama-cpp-python wheel failed to load on
    the real ZeroGPU worker with `OSError: libcudart.so.12: cannot open shared object file` --
    confirmed via the Space's actual runtime logs (fetch_space_logs), not guessed. The
    `nvidia-cuda-runtime-cu12`/`nvidia-cublas-cu12` pip packages bundle the needed .so files (real
    fix pattern confirmed via github.com/abetlen/llama-cpp-python#1460), but llama-cpp-python loads
    its library via a raw `ctypes.CDLL()` call that only searches the system's standard library
    path / LD_LIBRARY_PATH -- it doesn't know to look inside a pip package's install directory
    (unlike PyTorch's own wheels, which have special-cased loader logic for exactly this). The real
    subpath is `<package>.__path__[0]/lib/*.so*` (a `lib/` subdirectory, not a Python submodule).
    Pre-loading every nvidia-*-cu12 package's .so files explicitly via ctypes here, before
    llama_cpp's own import triggers its CDLL() call, makes it resolve to the already-loaded
    libraries instead of searching (and failing) on its own -- done generically across every
    installed `nvidia.*` package rather than hardcoding exact library names, since llama.cpp's
    CUDA backend may need more than just cudart+cublas depending on the exact build.
    """
    import ctypes
    import glob
    import importlib
    import pkgutil

    try:
        import nvidia
        for _finder, _pkg_name, _ in pkgutil.iter_modules(nvidia.__path__):
            try:
                _pkg = importlib.import_module(f"nvidia.{_pkg_name}")
                for _so in glob.glob(f"{_pkg.__path__[0]}/lib/*.so*"):
                    ctypes.CDLL(_so, mode=ctypes.RTLD_GLOBAL)
            except Exception:
                continue
    except Exception:
        pass  # fall through -- if this fails, llama_cpp's own import error will surface as before

    from llama_cpp import Llama

    llm = Llama(
        model_path=_MODEL_PATH,
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
    )
    parts: list[str] = []
    for chunk in llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        repeat_penalty=repeat_penalty,
        stream=True,
    ):
        tok = chunk["choices"][0]["text"]
        if tok:
            parts.append(tok)
    return "".join(parts)


def _generate_api(prompt: str, max_tokens: int, temperature: float, repeat_penalty: float) -> str:
    """Thin wrapper: chat_remote.py sends max_tokens/temperature/repeat_penalty as a JSON string
    packed into one field isn't needed -- Gradio's API takes positional args directly, matching
    the .click() inputs list below."""
    return _generate_tokens(prompt, int(max_tokens), float(temperature), float(repeat_penalty))


with gr.Blocks(title="chatPDB API") as demo:
    gr.Markdown(
        "## 🧬 chatPDB Inference API\n\n"
        "Internal endpoint for [chatpdb.mdeller.com](https://chatpdb.mdeller.com). "
        "Consumed via Gradio's own API: `POST /gradio_api/call/generate` then "
        "`GET /gradio_api/call/generate/<event_id>` (SSE).\n\n"
        "**Cold start:** model download happens once at container boot, outside the GPU "
        "lease. First real request per GPU lease takes ~10-30 s to load the GGUF onto the GPU."
    )

    # Hidden inputs/outputs purely to register a real Gradio event with api_name="generate" --
    # ZeroGPU's startup detection scans Gradio's own event/dependency graph, so the GPU-decorated
    # function must be reachable through a real .click()/.submit() binding, not just referenced
    # from a custom route.
    with gr.Row(visible=False):
        prompt_in = gr.Textbox()
        max_tokens_in = gr.Number(value=512)
        temperature_in = gr.Number(value=0.15)
        repeat_penalty_in = gr.Number(value=1.15)
        text_out = gr.Textbox()
        trigger = gr.Button()

    trigger.click(
        fn=_generate_api,
        inputs=[prompt_in, max_tokens_in, temperature_in, repeat_penalty_in],
        outputs=text_out,
        api_name="generate",
    )


if __name__ == "__main__":
    demo.launch()
