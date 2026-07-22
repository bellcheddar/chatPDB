---
title: chatPDB API
emoji: 🧬
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# chatPDB Inference API

ZeroGPU-backed streaming inference endpoint for chatPDB 32B v1 (Q4_K_M GGUF).

Endpoint: `POST /generate` — returns `text/event-stream` of token chunks.
