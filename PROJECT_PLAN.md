# chatPDB: a protein-structure-aware LLM for the Protein Data Bank

**Mission:** build a locally hosted, open-source assistant that is fluent in everything about the
Protein Data Bank: file formats and specs (PDB, mmCIF, PDBML), experimental techniques (X-ray,
cryo-EM, NMR, solution scattering), entry metadata and IDs, cross-reference databases, ligand and
binding data, and the real tools structural biologists use to read, manipulate, and QC structure
files. **chatPDB is not a structure prediction tool.** It never guesses a fold or a coordinate — it
reasons about structures that already exist, their provenance, and how to query and manipulate them
with real tools.

**Author:** Marc C. Deller, D.Phil. ([marcdeller.com](https://marcdeller.com))
**Status:** Planning complete (2026-07-14). Repo scaffolded. Phase 0 (environment) not yet started.
**Fine-tune stack:** MLX-LM on Apple Silicon (committed — same choice chem_sage validated over five
rounds; see section 3).
**Sibling project:** [chem_sage](https://github.com/bellcheddar/ChemSage) — a QLoRA-tuned chemistry
LLM built with the same process on the same hardware. chatPDB reuses chem_sage's *process* (memory
management, checkpoint/resume, RAG-first sequencing, eval harness shape) but is a fresh design: new
corpus, new tool ecosystem, new architecture decisions where the domain calls for it.
**Hand-off:** this document is the build brief for whichever Claude Code session picks up each
phase. Phases are ordered so each is independently testable and delivers value before the next
begins — the same discipline chem_sage used.

---

## 1. Design thesis (read this before writing any code)

The same two failure modes that chem_sage was designed around apply here, restated for structural
biology:

**(a) The model invents structural facts.** A fine-tuned LLM cannot reliably count atoms, compute
an RMSD, assign secondary structure, or recite a space group from memory — and it will confidently
produce wrong numbers if asked to. The fix is the same as chem_sage's: make the model **emit tool
calls** (Biopython, gemmi, DSSP, PyMOL) and reason over the returned values, rather than
hallucinating them.

**(b) People conflate behaviour with facts.** Put **volatile knowledge in retrieval** and **stable
behaviour in the weights**:

| Concern | Where it lives | Mechanism |
|---|---|---|
| Bulk structure/sequence/annotation data — millions of PDB entries, UniProt records, SIFTS mappings, CATH/SCOP classifications | external corpus | **RAG** |
| House behaviour: how to read a PDB/mmCIF header, when to reach for Biopython/gemmi/PyMOL/DSSP, how to interpret resolution/R-free/FSC, house answer format | model weights | **QLoRA fine-tune** |
| Deterministic structural facts (atom counts, chain composition, secondary structure, distances, B-factor stats, space group) | neither: computed live | **tool calls the model emits** |

chatPDB is therefore the same **hybrid** shape as chem_sage: a QLoRA-tuned base model (tuned for
tool-emitting, structurally literate *behaviour*) served behind a RAG layer (for *facts*), with a
tool-execution shim that runs the Biopython/gemmi/DSSP/PyMOL code the model produces and feeds
results back. "Protein-structure aware" = tuned to reach for the right tool and the right database,
not tuned to memorise coordinates.

---

## 2. Architecture

```
                         ┌──────────────────────────────────────────┐
   user query  ──────▶   │        chatPDB CLI (Rich / prompt_toolkit) │
                         └───────────────┬──────────────────────────┘
                                         │
                         ┌───────────────▼──────────────┐
                         │   RAG retriever (vector store) │  ◀── ingested corpus
                         │   Chroma + local embeddings    │      (PDB, UniProt, AlphaFold, ...)
                         └───────────────┬──────────────┘
                                         │  query + retrieved context
                         ┌───────────────▼──────────────┐
                         │   mlx_lm.server  →  fused model │  (QLoRA-tuned, MLX, OpenAI-compatible)
                         └───────────────┬──────────────┘
                                         │  may emit a tool call
                         ┌───────────────▼──────────────┐
                         │   tool-exec shim (sandboxed)   │  Biopython / gemmi / DSSP / PyMOL
                         └───────────────┬──────────────┘
                                         │  validated results (parsed structure, SS, distances, ...)
                                         ▼
                                   grounded answer
```

Same shape as chem_sage's architecture, with the tool-exec shim swapped from RDKit/PyMOL to a
structural-biology toolset (section 5), and the corpus swapped from chemistry literature/SAR tables
to PDB/UniProt/AlphaFold/OpenBind data (section 4).

---

## 3. Hardware and stack (committed)

Everything runs locally on Apple Silicon, same commitment chem_sage made and validated over five
rounds: fine-tuning uses **MLX-LM** (`mlx_lm.lora`), no discrete GPU, no VRAM wall, unified memory.

| Mac unified memory | Comfortable model size | Rough fine-tune time |
|---|---|---|
| 16 GB | ~8B | about an hour |
| 32 GB | ~14B | a few hours |
| 64 GB | ~32B | longer, but feasible |

**Record:** Mac memory 64 GB. Base model: **to be determined by the Phase 1 survey** (section 6) —
do not default to chem_sage's Qwen2.5-32B-Instruct-4bit without benchmarking current alternatives
first.

---

## 4. Data source brainstorm (corpus)

Not a complete list — a starting map, organized by category, with the access method to use in
`scripts/download_<source>.py`. All of this lands in `data/corpus/`, gitignored.

### Core wwPDB family (primary structures + metadata)
- **RCSB PDB** — `data.rcsb.org` REST + GraphQL, `search.rcsb.org` Search API v2, `files.wwpdb.org`
  derived data. Use the maintained **`rcsb-api`** Python package (`pip install rcsb-api`,
  github.com/rcsb/py-rcsb-api) rather than hand-rolled requests where it covers the query.
- **PDBe** (EMBL-EBI) — `www.ebi.ac.uk/pdbe/api`, separate modules for PDB/EMDB/SIFTS/PISA/validation.
- **PDBj** (Japan) — the third wwPDB partner; include for completeness and cross-checking.
- **BMRB** — Biological Magnetic Resonance Data Bank (wwPDB member): NMR restraints/chemical shifts.
- **EMDB** — cryo-EM map metadata, resolution, FSC curves.
- **wwPDB Chemical Component Dictionary (CCD) / BIRD** — every ligand/monomer definition. chem_sage's
  `scripts/download_pdb.py` already pulls a drug-like subset for its own corpus; chatPDB needs the
  **full** structural CCD, not just the drug-like slice.
- **wwPDB validation reports** — per-entry XML/PDF validation (Ramachandran outliers, clashscore,
  R-free gap). Primary source for "structure QC" training examples.
- **PDB-REDO** — re-refined/re-built X-ray structures; a good source of "what changed and why" pairs.

### Predicted / computational structures
- **AlphaFold DB** (`alphafold.ebi.ac.uk`, bulk via Google Cloud/FTP, REST API) — per-UniProt
  predicted structures with pLDDT/PAE.
- **ESM Metagenomic Atlas** — large-scale predicted structures; useful as an explicit "predicted vs.
  experimentally determined" contrast case (this is where the "not a structure predictor" boundary
  gets trained in).
- **OpenBind** (openbind.uk) — Diamond Light Source-led UK consortium. Confirmed live as of this
  writing: first public release May 2026, protein-ligand structure-affinity data for the EV-A71 2A
  protease, access noted under `/documents-and-tools`. **Caveat:** small/early dataset (one target
  so far), no confirmed bulk API yet — verify the actual access mechanism in Phase 2 before writing
  a dedicated downloader; treat it as a growing source to poll periodically, not a one-shot pull.

### Sequence, domain & fold classification
- **UniProtKB** (Swiss-Prot + TrEMBL) — REST + SPARQL, canonical sequence/function annotation and
  PDB cross-references.
- **Pfam / InterPro** — InterPro now hosts Pfam; domain family annotation.
- **CATH** and **SCOP2** — evolutionary/structural domain classification.
- **Gene3D** — CATH-based domain assignment at proteome scale.

### Cross-reference & interaction annotation
- **SIFTS** (EBI) — residue-level PDB↔UniProt↔Pfam↔CATH↔IntEnz mapping. chem_sage already has a
  working downloader pattern for this in `download_pdb.py` (SIFTS section) worth porting directly.
- **PDBe-KB** — aggregated per-entry knowledge base (functional sites, ligand interactions, conservation).
- **PDBsum** — pictorial per-structure summaries, interaction diagrams.
- **PISA** (EBI) — quaternary structure/interface/assembly analysis (biological assembly vs.
  asymmetric unit — a distinction chatPDB should be fluent in explaining).
- **ProtCID** — protein interface comparison across the PDB.

### Ligand & binding data
- **PDBbind** — measured binding affinities for protein-ligand complexes. chem_sage already has a
  subset; chatPDB wants the *structural* side of the same data, not just affinity numbers.
- **BindingDB** — broader affinity data cross-referenced to PDB entries.
- **PLIP** — protein-ligand interaction fingerprints. This is a **tool**, not a corpus source: run at
  Phase 3 dataset-generation time, not downloaded in bulk.

### Membrane, disorder, nucleic acid, scattering
- **OPM** (Orientations of Proteins in Membranes) and **PDBTM** — membrane protein topology annotation.
- **DisProt** and **MobiDB** — intrinsic disorder annotation.
- **NDB** (Nucleic Acid Database) — DNA/RNA and protein-nucleic-acid complex structures within the PDB.
- **SASBDB** — small-angle scattering data, the solution-state complement to crystal structures.

### Benchmarks, provenance, networks
- **CASP** targets/results — structure-prediction benchmark history. Used as context/QA material
  ("what is CASP, how is prediction accuracy scored") — **never** as training signal for the model
  to attempt prediction itself.
- **STRING** — protein-protein interaction network, cross-referenced to PDB structures.
- **RCSB PDB-101 / Proteopedia** — plain-language educational explainers; a good source of
  well-written prose for the "explain this concept" behaviour class.

### Structure search/comparison tools (used for tool-calling examples, not bulk downloads)
- **FoldSeek**, **Dali server**, RCSB **1D Coordinate Server** / **Sequence Coordinates API**.

**Access protocols summary:** REST/JSON covers most of the above; GraphQL for RCSB's cross-level
queries; bulk FTP/rsync for wwPDB derived data, AlphaFold DB, and EMDB archives; SPARQL optionally
for UniProt; plain HTTP pulls for CCD/mmCIF archives. `requests` plus `rcsb-api` cover nearly
everything; only bulk archive mirroring needs `rsync`/`curl`.

---

## 5. Tool ecosystem (for tool-calling SFT examples and the sandboxed shim)

chem_sage's shim ran RDKit (and PyMOL for rendering). chatPDB's equivalent, structure-first toolset:

- **Biopython** (`Bio.PDB`) — parse PDB/mmCIF, structure objects, superposition, sequence extraction.
- **gemmi** — fast, modern mmCIF/PDB/CCD parsing (handles edge cases Biopython chokes on), symmetry
  operations, space-group handling.
- **PyMOL** — visualization scripting: load, select, colour, measure, render. Same tool chem_sage
  already proved a working sandboxed-execution pattern for.
- **DSSP** (via Biopython's DSSP wrapper or `mkdssp` directly) — secondary structure assignment.
- **MDAnalysis** / **ProDy** — ensemble/trajectory analysis; useful for NMR ensembles and
  descriptive (not predictive) flexibility questions.
- **BioPandas** — tabular PDB/mmCIF wrangling for bulk questions.

---

## 6. Base model survey (Phase 1 — survey before committing)

Marc's explicit call: do not default to chem_sage's Qwen2.5-32B-Instruct-4bit. Build
`scripts/survey_base_models.py` to run a small, fixed prompt set — structural-bio tool-emission,
format-literacy Q&A, general instruction-following — against each candidate served via
`mlx_lm.server`, recording tok/s, peak RSS, and a qualitative pass/fail per prompt.

Candidates as of the current (July 2026) landscape:

| Model | Class | Notes |
|---|---|---|
| `mlx-community/Qwen3-32B-4bit` | dense, newer generation | Confirmed live on mlx-community. Same architecture family/config shape chem_sage already validated for MLX-LM LoRA. Leading candidate. |
| `mlx-community/Qwen2.5-32B-Instruct-4bit` | dense | chem_sage's proven baseline — zero unknowns on this exact Mac. Serves as the survey's control/floor. |
| `DeepSeek-R1-Distill-Qwen-32B` (4-bit MLX build) | dense, reasoning-distilled | Worth testing specifically for method-interpretation/QC reasoning tasks. |
| Dense Gemma 4 31B (4-bit MLX, if mirrored) | dense | Only include if an mlx-community 4-bit build actually exists at survey time — verify before adding to the run. |

**Exit test:** a short written comparison appended to this document (§ survey results, to be added),
one model selected and recorded, before Phase 2 begins.

---

## 7. Phase-by-phase build plan

### Phase 0 — Environment (0.5 day)
1. Create a venv (Python 3.11+), install `requirements.txt`.
2. Confirm MLX-LM works: `mlx_lm.generate --model mlx-community/Qwen2.5-7B-Instruct-4bit --prompt "hello"`.
3. Confirm Biopython, gemmi, DSSP, PyMOL each round-trip on a real PDB entry (e.g. fetch `1CRN`,
   parse it, assign secondary structure, render a quick image).

**Exit test:** all four tools parse/process the same real structure without error.

### Phase 1 — Base model survey (0.5–1 day)
Run `scripts/survey_base_models.py` against the candidates in section 6. Pick one, record it in
section 3 and section 6 of this document.

### Phase 2 — RAG pipeline first (1–2 days)
Same sequencing discipline chem_sage used: retrieval before training, because it delivers value
immediately and is reversible.
1. **Corpus staging** (`data/corpus/`) — populated by `scripts/download_<source>.py` per section 4.
2. **Ingest** (`scripts/ingest_rag.py`) — chunk, embed (local sentence-transformer), write to Chroma.
   Port chem_sage's `PersistentClient(Settings(allow_reset=True, anonymized_telemetry=False))`
   pattern and its pre-chunked-table loader for large relational CSVs (SIFTS mappings especially).
3. Verify OpenBind's actual access mechanism (API vs. manual download) before writing its downloader.

**Exit test:** ask a question only the corpus can answer, get a grounded answer with the source
chunk shown.

### Phase 3 — SFT dataset (the real work, 3–5 days)
All four behaviour classes weighted equally (Marc's call):
1. **File/format literacy** — PDB vs. mmCIF vs. PDBML, header records, CCD components, format
   conversion, common parsing pitfalls.
2. **Experimental-method interpretation** — X-ray (resolution, R/Rfree, space groups), cryo-EM (map
   resolution, FSC), NMR (ensembles, restraints); reading wwPDB validation reports.
3. **Tool-calling for structure manipulation** — emit correct Biopython/gemmi/DSSP/PyMOL code (load,
   parse, select, superpose, measure, assign secondary structure, render).
4. **Database navigation & cross-referencing** — SIFTS PDB↔UniProt↔Pfam↔CATH mapping, ID resolution,
   AlphaFold vs. experimental comparison, finding related entries.

**Construction strategy:** ground-truth-first, same discipline as chem_sage's RDKit approach — run
Biopython/gemmi/DSSP first to get the real value, then template the Q and worked-solution A around
it. Never hand-author a number. Output MLX layout: `train.jsonl` + `valid.jsonl` in `data/sft/`, plus
a frozen `test.jsonl`, chat format (`{"messages": [...]}`) matching chem_sage's schema exactly.

**Exit test:** every assistant code block runs; every PDB/UniProt ID referenced resolves to a real
entry; a spot-read of 20 examples reads like something Marc would have written.

### Phase 4 — QLoRA fine-tune with MLX-LM (0.5–1 day of compute)
`config/train_config.yaml` seeded from chem_sage's validated field names and values (rank, RSLoRA,
`steps_per_report == steps_per_eval` from round one — chem_sage had to learn this the hard way,
chatPDB starts correct).

`scripts/train_launch.py` ports chem_sage's memory-fraction tuning
(`mx.set_cache_limit`/`mx.set_memory_limit` against `mx.device_info()["max_recommended_working_set_size"]`)
and adds two things chem_sage didn't have from round one:
- **`wandb.init()`** logging — train/val loss, tokens/sec, memory, LR schedule — from round one.
  chatPDB is a fresh project, so there's no reason to wait the way chem_sage's own roadmap did.
- **Checkpoint auto-resume** — scan `adapters/<name>/*_adapters.safetensors` for the highest iter
  and offer a `--resume` flag that wires into `mlx_lm.lora`'s native `--resume-adapter-file`.

**Critical:** watch train/val loss together; climbing validation loss means stop early, same rule
chem_sage lived by.

### Phase 5 — Fuse and serve (0.5 day)
Route A, same as chem_sage: `mlx_lm.fuse` → `mlx_lm.server --port 8080`, OpenAI-compatible endpoint.

### Phase 6 — Close the hybrid loop (0.5 day)
Wire `rag/tool_exec.py`: Biopython sandbox first (restricted subprocess, no filesystem/network), add
gemmi/DSSP once stable, PyMOL last — the same staged-caution order chem_sage applied to RDKit→PyMOL.

### Phase 7 — Evaluation (1 day, then ongoing)
`eval/eval_pdb.py`, metrics analogous to chem_sage's `eval_chem.py`:
- **ID validity** — every PDB/UniProt ID the model states must resolve to a real entry.
- **Tool executability** — every emitted Biopython/gemmi/DSSP/PyMOL block must run.
- **Cross-reference accuracy** — stated SIFTS-style mappings must match the real mapping.
- **Numerical fidelity** — stated resolution/R-free/chain counts must match live recompute.
- **Refusal accuracy** — structure-prediction requests ("predict the fold of this sequence")
  correctly declined as out of scope.
- **Degeneration-free** — same repetition-collapse check chem_sage's harness added in later rounds.

`eval/compare/eval_compare.py` ports the multi-round comparison pattern wholesale: auto-managed
`mlx_lm.server` per model, `ResourceMonitor` (psutil CPU/RSS sampling), `--resume` on cached results,
HTML + Markdown reports with hand-rolled SVG loss curves.

### Phase 8 — Local CLI + RAG (0.5–1 day)
`scripts/chat.py` ports chem_sage's proven skeleton: `rich` themed console, `prompt_toolkit` session
with bottom toolbar, `pyfiglet` banner (graceful plain-text fallback), streaming generation via
`mlx_lm.stream_generate` into a `rich.live.Live` display, slash commands (`/help`, `/clear`,
`/reset`, `/history`, `/save`, `/info`, `/retry`), and a deterministic **corpus fast-path**
(`rag/corpus_lookup.py`, ported pattern) that bypasses the LLM entirely for bulk-enumeration queries
("list all PDB entries for X") — a strong anti-hallucination pattern worth keeping as-is.

Rebranded: ASCII banner **"chatPDB"**, marcdeller.com / marc@marcdeller.com credit line, own colour
theme (doesn't need to reuse chem_sage's navy/`#467FF7` — pick something distinct at build time).

**Exit test:** single Python entry point (`python scripts/chat.py`) launches the full CLI experience —
banner, model load with spinners, RAG-grounded chat, tool-call execution — end to end.

### Phase 9 — Hosted demo (new phase, per Marc's hosting decision)
`scripts/merge_export.py --de-quantize` produces standard HF-format fp16 safetensors (MLX is
Apple-only; de-quantized fuse output is portable, ordinary `transformers`-loadable weights). Deploy
via **Hugging Face Spaces** (Gradio `ChatInterface`, ZeroGPU hardware tier) as the default hosted
option, with **Modal** or **Replicate** noted as fallbacks if a 32B model doesn't fit Spaces' free
GPU tier.

### Phase 10 — Iterate (ongoing)
Same design-build-test-learn loop as chem_sage: eval failures become new Phase 3 training examples;
new PDB releases and OpenBind updates get ingested into RAG continuously; re-tune only for genuine
behavioural gaps retrieval can't fix.

---

## 8. Repository structure

```
chatPDB/
├── PROJECT_PLAN.md          # this file
├── README.md                # placeholder now; real version via marcs-vibe-coding skill once Phase 2+ ships
├── requirements.txt
├── .gitignore
├── config/
│   ├── train_config.yaml    # native MLX-LM QLoRA config (added Phase 4)
│   └── system_prompt.txt    # the chatPDB house system prompt
├── data/
│   ├── README.md            # SFT schema
│   ├── corpus/               # RAG source documents (gitignored, populated Phase 2)
│   └── sft/                  # train.jsonl + valid.jsonl + test.jsonl (added Phase 3)
├── scripts/
│   ├── preflight.sh / postflight.sh   # ported from chem_sage + iCloud dataless-file check
│   ├── monitor_training.sh
│   ├── download_<source>.py           # one per data source in section 4 (added Phase 2)
│   ├── survey_base_models.py          # Phase 1 candidate benchmarking
│   ├── build_dataset.py               # SFT generator (Phase 3)
│   ├── ingest_rag.py                  # (Phase 2)
│   ├── train_launch.py                # memory-aware wrapper + wandb + auto-resume (Phase 4)
│   ├── merge_export.py                # fuse + serve + de-quantize export (Phase 5 / 9)
│   └── chat.py                        # CLI entry point (Phase 8)
├── rag/
│   ├── retrieve.py
│   ├── corpus_lookup.py               # deterministic fast-path lookups
│   └── tool_exec.py                   # sandboxed Biopython/gemmi/DSSP/PyMOL runner
├── eval/
│   ├── eval_pdb.py
│   └── compare/
│       ├── models.yaml
│       └── eval_compare.py
├── models/                    # chatpdb_<size>b_v<round> fused model dirs
├── adapters/                  # LoRA checkpoints, <name>/<iter:07d>_adapters.safetensors
└── logs/                      # wandb local run mirror + monitor logs
```

---

## 9. Hand-off notes / hard rules

- Never let a stated structural fact stand unverified: route through Biopython/gemmi/DSSP.
- Never train on unvalidated SFT examples: a bad label is worse than a missing one.
- The tool-exec shim is sandboxed (restricted subprocess) before PyMOL execution is enabled.
- Fine-tune stack is MLX-LM only; do not introduce an NVIDIA/Unsloth path.
- `steps_per_report == steps_per_eval` in every train config from round one.
- Run `preflight.sh` (iCloud dataless-file check included) before every training/eval launch on this
  Mac — Documents-folder files get evicted by Optimize Mac Storage and silently break runs otherwise.
- Model dirs named `chatpdb_<size>b_v<round>`, kept under `models/`.
- The real README is written with the `marcs-vibe-coding` skill once there's substance to document —
  never hand-written plain markdown. If it links anywhere JS-heavy, use GitHub Pages, not
  htmlpreview.github.io.
- chatPDB never trains toward structure prediction. Any dataset example that asks the model to
  predict a fold or coordinate from sequence alone is out of scope and should be rejected at
  dataset-build time, not just softened in the system prompt.

---

## 10. Key references

- MLX-LM: `mlx_lm.lora`, `mlx_lm.fuse`, `mlx_lm.server` (github.com/ml-explore/mlx-lm).
- RCSB PDB APIs: `data.rcsb.org` (REST + GraphQL), `search.rcsb.org` (Search API v2), `rcsb-api`
  Python package (github.com/rcsb/py-rcsb-api).
- PDBe API: `www.ebi.ac.uk/pdbe/api` (PDB/EMDB/SIFTS/PISA/validation modules).
- AlphaFold DB: `alphafold.ebi.ac.uk`.
- OpenBind: `openbind.uk` (Diamond Light Source consortium; verify access method at Phase 2).
- Biopython (`Bio.PDB`), gemmi, DSSP, PyMOL, MDAnalysis, ProDy, BioPandas.
- chem_sage (sibling project, process reference): `/Users/dellboy/Documents/Vibe_Coding/chem_sage/PROJECT_PLAN.md`.

---

*Built by Marc C. Deller, D.Phil. · [marcdeller.com](https://marcdeller.com) · marc@marcdeller.com*
