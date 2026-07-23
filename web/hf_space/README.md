---
title: chatPDB API
emoji: 🧬
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# chatPDB Inference API

ZeroGPU-backed inference endpoint for chatPDB 32B v1 (Q4_K_M GGUF).

Consumed by the Flask PTY app at [chatpdb.mdeller.com](https://chatpdb.mdeller.com).

**Endpoint:** `POST /generate` — returns `text/event-stream` of token chunks.

**Cold start:** first request after idle downloads the ~18.4 GB GGUF and allocates the ZeroGPU
A10G (~60-120 s). This is a portfolio demo; availability is best-effort.
