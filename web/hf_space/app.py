"""
chatPDB inference API — HuggingFace Space (ZeroGPU)

Serves a streaming /generate endpoint backed by llama-cpp-python.
The GGUF is pulled from the Hub on first request and cached for the session.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Generator

import spaces
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

REPO_ID = "Dellboy/chatpdb_32b_v1-GGUF"
FILENAME = "chatpdb_32b_v1_q4km.gguf"
MODEL_PATH: Path | None = None   # set after first download

N_CTX = 1536   # matches chatPDB's real training max_seq_length (config/train_config.yaml)
N_GPU_LAYERS = -1  # offload all layers to GPU

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="chatPDB API")


def _get_model():
    """Download GGUF on first call, return cached Llama instance."""
    global MODEL_PATH
    from llama_cpp import Llama

    if MODEL_PATH is None:
        MODEL_PATH = Path(hf_hub_download(repo_id=REPO_ID, filename=FILENAME))

    return Llama(
        model_path=str(MODEL_PATH),
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
    )


@spaces.GPU
def _generate_tokens(
    prompt: str,
    max_tokens: int,
    temperature: float,
    repeat_penalty: float,
) -> Generator[str, None, None]:
    """Run inference inside the ZeroGPU lease."""
    llm = _get_model()
    stream = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        repeat_penalty=repeat_penalty,
        stream=True,
    )
    for chunk in stream:
        token = chunk["choices"][0]["text"]
        if token:
            yield token


@app.post("/generate")
async def generate(request: dict):
    """
    POST /generate
    Body: { "prompt": "...", "max_tokens": 512, "temperature": 0.15, "repeat_penalty": 1.15 }
    Returns: text/event-stream of token strings
    """
    prompt = request.get("prompt", "")
    max_tokens = int(request.get("max_tokens", 512))
    temperature = float(request.get("temperature", 0.15))
    repeat_penalty = float(request.get("repeat_penalty", 1.15))

    def event_stream():
        for token in _generate_tokens(prompt, max_tokens, temperature, repeat_penalty):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {"status": "ok"}
