# 🧬 chatPDB

> **A protein-structure-literate AI assistant for the Protein Data Bank: RAG for facts, a QLoRA-tuned model for behaviour, and live Biopython/gemmi/DSSP/PyMOL tool calls for structural truth. Fine-tuned and served locally on Apple Silicon with MLX-LM.**

![MLX-LM](https://img.shields.io/badge/MLX--LM-Apple%20Silicon-000000?logo=apple&logoColor=white) ![fine-tune](https://img.shields.io/badge/fine--tune-QLoRA-467FF7) ![base](https://img.shields.io/badge/base-Qwen3--32B-00897B) ![RAG](https://img.shields.io/badge/RAG-ChromaDB%20+%2042%20sources-9b51e0) ![models](https://img.shields.io/badge/models-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black) ![author](https://img.shields.io/badge/author-Marc%20C.%20Deller%2C%20D.Phil.-1C244B)

<table>
<tr>
<td>🌐 <b>Website</b></td><td><a href="https://marcdeller.com" target="_blank" rel="noopener noreferrer">marcdeller.com</a></td>
<td>✉️ <b>Contact</b></td><td><a href="mailto:marc@marcdeller.com">marc@marcdeller.com</a></td>
<td>🐙 <b>GitHub</b></td><td><a href="https://github.com/bellcheddar/chatPDB" target="_blank" rel="noopener noreferrer">bellcheddar/chatPDB</a></td>
</tr>
</table>

---

chatPDB is a locally hosted, open-source assistant that reasons about protein structures (file
formats, experimental methods, cross-reference databases, structure-manipulation tools), drives
the structural-biology tools a working structural biologist actually uses (Biopython, gemmi, DSSP,
PyMOL, ChimeraX, and more), and grounds every factual claim in the real wwPDB/RCSB/SIFTS/UniProt
corpus rather than memory.

**Why it matters:** off-the-shelf models invent structural facts (wrong resolutions, hallucinated
PDB IDs, fabricated secondary-structure content) and have no way to compute a real value rather
than guess one. chatPDB fixes both by separating concerns: volatile knowledge lives in retrieval,
stable behaviour lives in the fine-tuned weights, and deterministic structural facts (atom counts,
secondary structure, chain composition, geometry) are computed live by real tools, executed in a
sandbox, rather than asserted from memory. It is useful for: interrogating deposited PDB structures
and their cross-references, generating correct Biopython/gemmi/DSSP/PyMOL scripts, honestly
declining to answer when an ID doesn't exist or a value isn't known, and reasoning like a
structural biologist about experimental method, quality, and biological assembly rather than
treating a structure as a bare coordinate file.

Sibling project to [ChemSage](https://github.com/bellcheddar/ChemSage), reusing its proven
MLX-LM/QLoRA/RAG process on a fresh domain: protein structures instead of small-molecule chemistry.

**Project links:**

- [GitHub](https://github.com/bellcheddar/chatPDB)
- [Models on Hugging Face](https://huggingface.co/Dellboy)

## 🤗 Models on HuggingFace Hub

| Model | HuggingFace | Base | SFT dataset | Round |
|---|---|---|---|---|
| **chatPDB 32B v1** ⭐ | [**Dellboy/chatpdb_32b_v1**](https://huggingface.co/Dellboy/chatpdb_32b_v1) | Qwen3-32B-4bit | v7 (97,272 examples) | **Round 1 (current)** |
| chatPDB 32B v1 (GGUF, Q4_K_M) | [Dellboy/chatpdb_32b_v1-GGUF](https://huggingface.co/Dellboy/chatpdb_32b_v1-GGUF) | de-quantized fp16 export of the above | — (quantized, not retrained) | Serves [`Dellboy/chatpdb-api`](https://huggingface.co/spaces/Dellboy/chatpdb-api) |

Add a row here for every future fused model (never overwrite a previous row) — see
[🎓 Training](#-training) below for how each round's real numbers get recorded.

## ⚙️ The hybrid in one line

RAG keeps it truthful today; the fine-tune makes it consistent tomorrow; Biopython/gemmi/DSSP/PyMOL
make it correct always.

## 🚀 Quick start

See **[`PROJECT_PLAN.md`](PROJECT_PLAN.md)** for the full step-by-step build (phases 0 to 10). Short version:

| Step | What | Script / command |
|---|---|---|
| 0 | Env: MLX-LM + Biopython + structural tools | (`requirements.txt`) |
| 1 | Base model (Qwen3-32B, MLX 4-bit) | `config/train_config.yaml` |
| 2 | **RAG first** (ship this alone) | `scripts/ingest_rag.py` |
| 3 | Build + validate SFT dataset | `scripts/build_dataset.py` |
| 4 | QLoRA fine-tune | `python scripts/train_launch.py --config config/train_config.yaml` |
| 5 | Fuse + serve | `python scripts/merge_export.py` → `mlx_lm.server --port 8080` |
| 6 | Close the hybrid loop | `rag/tool_exec.py` (Biopython sandbox; gemmi/DSSP/PyMOL staged next) |
| 7 | Structural evaluation | `python eval/eval_pdb.py` · `python eval/compare/eval_compare.py` |
| 8 | Local CLI + RAG | `python scripts/chat.py` |
| 9 | Hosted demo (API live, droplet pending) | `web/hf_space/` (HF Space API, live) + `web/flask_app/` (droplet terminal UI, pending deploy) |

## ▶️ How to run

**Interactive chat** (loads the model in-process — no server needed, unlike the eval scripts below):
```bash
cd /Users/dellboy/Documents/Vibe_Coding/chatPDB
source .venv/bin/activate
python scripts/chat.py --model models/chatpdb_32b_v1
```
`/help` for slash commands, `/info` for session details, `quit`/`exit`/`q` to leave. `--no-rag` runs
model-only (no retrieval).

**Evaluate the model** (these two DO need `mlx_lm.server`; `eval_compare.py` manages its own):
```bash
# Terminal 1
python scripts/preflight.sh   # flush memory, check for background-process contention
mlx_lm.server --model models/chatpdb_32b_v1 --port 8080

# Terminal 2, once Terminal 1 shows "HTTP server listening"
python eval/eval_pdb.py --n 200                 # single-model scorecard, needs Terminal 1 running
python eval/compare/eval_compare.py              # multi-round comparison, starts its own server
```

## 🎓 Training

Real numbers only, recorded per round — extend this table (never edit a previous round's row) as
new training rounds ship.

| | Round 1 (32B v1) |
|---|---|
| Base model | mlx-community/Qwen3-32B-4bit |
| SFT dataset | v7 — 97,272 examples (77,818 train / 9,727 valid / 9,727 test) |
| LoRA config | rank 32, scale 64, RSLoRA, num_layers 32/64, dropout 0.05 |
| max_seq_length | 1,536 (real-measured: covers 100% of train.jsonl, p99.9=1,190, max=1,973) |
| Iterations trained | ~803 of a planned 1,420 (stopped early — see below) |
| Best checkpoint | true iter 600, val_loss 0.176 |
| Stop reason | real val-loss plateau: block-averaged val loss stopped improving from ~iter 300 onward (0.202 best-block avg at iters 200-300, no further downward trend through iter 800) — not overfitting, just diminishing returns on the remaining budget |
| RAG corpus | 42 source files, 105,463 indexed chunks (ChromaDB + BAAI/bge-base-en-v1.5) |
| Real incidents this round | crashed mid-run at true iter ~630, resumed from the true-iter-600 checkpoint (weights-only resume — iteration counter and LR schedule do not carry over, see `PROJECT_PLAN.md`); a real `mlx_lm` bug (val-loss/train-loss step misalignment) found and patched locally |

Full narrative for every round, including every real bug found and fixed along the way, lives in
[`PROJECT_PLAN.md`](PROJECT_PLAN.md).

## 📊 Evaluation

`eval/eval_pdb.py` (single-model harness) and `eval/compare/eval_compare.py` (multi-round
comparison — auto-managed `mlx_lm.server`, `ResourceMonitor`, `--resume`, HTML + Markdown reports)
score every response against **real, corpus-grounded ground truth** — PDB ID validity checks real
corpus membership (not just format), cross-reference accuracy checks real SIFTS UniProt/CATH/EC
mappings, numerical fidelity checks real resolution/R-free/chain counts, tool executability runs
real Biopython blocks in the Phase 6 sandbox. No metric relies on hand-authored ground truth.

| Metric | Round 1 (32B v1) smoke-test sample |
|---|---|
| PDB ID validity | 89–100% (`--n 20` / `--limit 10` runs) |
| Cross-reference accuracy | 50% |
| Tool executability | n/a in the 10-example sample (0/1 in the 20-example sample — the one attempt was a DSSP block, correctly excluded as not-yet-enabled per Phase 6, not a failure) |
| Numerical fidelity | 33–40% |
| Refusal accuracy | 100% |
| Degeneration-free | 100% |

These are small smoke-test samples (`--n 20`/`--limit 10`), run to verify the harness end to end —
not a full evaluation pass. Run `python eval/eval_pdb.py --n 200` for a real-sized, seeded sample
once a model server is up (see **How to run** below).

## 🌐 Hosted demo

Two services, mirroring chem_sage's architecture: an HF Space
([`Dellboy/chatpdb-api`](https://huggingface.co/spaces/Dellboy/chatpdb-api), Gradio SDK +
`llama-cpp-python`, ZeroGPU) serves the model from a Q4_K_M GGUF; a Flask app on the droplet
(`web/flask_app/`) spawns `scripts/chat.py` in a pseudo-terminal per browser tab and streams it to
an xterm.js terminal — the web page looks exactly like the local CLI.

Real numbers from the conversion pipeline (`scripts/merge_export.py --de-quantize` → llama.cpp
GGUF conversion → `llama-quantize`): fp16 export 61GB → Q4_K_M GGUF **18.4GB (4.82 bits/weight)**,
sanity-checked locally (loads correctly, generates coherent real answers) before uploading to
[`Dellboy/chatpdb_32b_v1-GGUF`](https://huggingface.co/Dellboy/chatpdb_32b_v1-GGUF).

The HF Space backend is **live and confirmed working end to end** — real POST → SSE round-trip
against ZeroGPU hardware returning coherent generated text. Thirteen real bugs were found and fixed
while porting and live-testing chem_sage's own (previously untested) version of this architecture,
including three ZeroGPU-specific ones only findable by live deployment: Docker SDK is incompatible
with ZeroGPU hardware; a custom FastAPI route silently loses to Gradio's own catch-all route, but
ZeroGPU's startup detection requires `demo.launch()` as the real entrypoint — solved by exposing the
endpoint through Gradio's own native `api_name="generate"` REST mechanism instead; and the model
download must happen at container boot, not inside the `@spaces.GPU`-decorated function, or it burns
the bounded GPU lease on a plain network transfer. Full list in `PROJECT_PLAN.md` Phase 9.

Still pending, held for explicit confirmation before each step (shared, externally-visible
infrastructure): push `web/flask_app/` to the droplet, provision `chatpdb.mdeller.com`
nginx/certbot, add the `mdeller-landing` entry.

## 📚 Corpus

wwPDB/RCSB structures and metadata, SIFTS UniProt/Pfam/CATH/GO/EC/InterPro mappings, TWILIGHT
ligand-fit scores, PDB-REDO refinement deltas, EMDB map metadata, OPM membrane placement, MobiDB
disorder predictions, SCOP2 fold classification, sequence-redundancy clusters, obsolete-entry
records, BindingDB affinities, STRING interactions, Pharos druggability, AlphaFold predictions,
AlphaFraud PDB-vs-AlphaFold disagreement data, PubMed/CrossRef citation verification, and full
PyMOL/ChimeraX/py3Dmol command references — see [`data/README.md`](data/README.md) for the
complete dataset-version history and construction rules.

## 🧱 Stack

MLX-LM (QLoRA fine-tuning + serving), Biopython/gemmi/DSSP/PyMOL/ChimeraX/MDAnalysis/ProDy (tool
calls), ChromaDB + BAAI/bge-base-en-v1.5 (RAG retrieval), pandas/plotly/bio3d (analysis tools
taught to the model), Weights & Biases (training observability).

---

## 👤 Author

**Marc C. Deller, D.Phil.**
Structural biologist & drug discovery scientist

<table>
<tr>
<td>🌐</td><td><a href="https://marcdeller.com" target="_blank" rel="noopener noreferrer">marcdeller.com</a></td>
<td>✉️</td><td><a href="mailto:marc@marcdeller.com">marc@marcdeller.com</a></td>
<td>🐙</td><td><a href="https://github.com/bellcheddar/chatPDB" target="_blank" rel="noopener noreferrer">github.com/bellcheddar/chatPDB</a></td>
</tr>
</table>
