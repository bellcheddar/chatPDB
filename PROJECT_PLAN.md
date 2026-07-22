# chatPDB: a protein-structure-aware LLM for the Protein Data Bank

**Mission:** build a locally hosted, open-source assistant that is fluent in everything about the
Protein Data Bank: file formats and specs (PDB, mmCIF, PDBML), experimental techniques (X-ray,
cryo-EM, NMR, solution scattering), entry metadata and IDs, cross-reference databases, ligand and
binding data, and the real tools structural biologists use to read, manipulate, and QC structure
files. **chatPDB is not a structure prediction tool.** It never guesses a fold or a coordinate — it
reasons about structures that already exist, their provenance, and how to query and manipulate them
with real tools.

**Author:** Marc C. Deller, D.Phil. ([marcdeller.com](https://marcdeller.com))
**Status:** Phase 3 (SFT dataset) complete through round 6 (2026-07-19): round 5 was a full
visualization/rendering/simulation tool review — full PyMOL (436 commands) and ChimeraX (547
commands) command awareness via live introspection, sequence alignment, WebLogo, biotite plots,
py3Dmol, pdb-tools, a 2D topology schematic, electrostatics prep, molecular dynamics (OpenMM +
GROMACS), crystallography (CCP4 + PHENIX), and AutoDock Vina docking. Round 6 wired in MDAnalysis/
ProDy (installed since round 1, unused until now), added bio3d/R, plotly, full py3Dmol command
awareness (108 methods, documentation-grounded), pandas as a taught skill, and pulled in
AlphaFraud's full backfill (gated on live confirmation the backfill service had genuinely finished,
then a real API-completeness bug caught and fixed via direct database export). Full narrative in
§7, after the round 4 entry.
Base model: `mlx-community/Qwen3-32B-4bit`. RAG corpus: 40 files / 105,463 chunks (up from
39 / 102,873 — py3dmol_commands.csv 16 chunks new, alphafraud_comparisons.csv grown from 7 to
2,581 chunks). SFT dataset v6: 95,884 examples
(76,708 train / 9,588 valid / 9,588 test), 2.7% rejection rate, full regeneration ran ~11h clean
(after the AlphaFraud database-export fix — see §7's round 6 section and `data/README.md`'s v6
entry).
Phase 4 (QLoRA fine-tune) not yet started.
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

**Record:** Mac memory 64 GB. Base model: **`mlx-community/Qwen3-32B-4bit`**, selected via the
Phase 1 survey (section 6) over chem_sage's Qwen2.5-32B-Instruct-4bit baseline, DeepSeek-R1-Distill-
Qwen-32B-4bit, and gemma-4-31b-it-4bit — best transcript accuracy/completeness of the four, at the
cost of roughly a third the token-generation speed of the Qwen2.5 baseline. Thinking mode must be
disabled at inference time (`chat_template_kwargs: {"enable_thinking": false}` on the OpenAI-
compatible endpoint) or it burns its token budget on `<think>` and never reaches an answer.

---

## 4. Data source brainstorm (corpus)

Not a complete list — a starting map, organized by category, with the access method to use in
`scripts/download_<source>.py`. All of this lands in `data/corpus/`, gitignored.

### Core wwPDB family (primary structures + metadata)
- **RCSB PDB** ✅ **implemented** (`scripts/download_rcsb.py`, Phase 2) — `data.rcsb.org` REST +
  GraphQL, `files.wwpdb.org` derived data. (The `rcsb-api` package was evaluated but the hand-rolled
  GraphQL queries below already cover what's needed; not currently used.)
- **PDBe** (EMBL-EBI) — `www.ebi.ac.uk/pdbe/api`, separate modules for PDB/EMDB/SIFTS/PISA/validation.
- **PDBj** (Japan) — the third wwPDB partner; include for completeness and cross-checking.
- **BMRB** — Biological Magnetic Resonance Data Bank (wwPDB member): NMR restraints/chemical shifts.
- **EMDB** ✅ **implemented** (`scripts/download_emdb.py`, round 4) — cryo-EM map metadata
  (resolution + determination method, contour level, pixel spacing, symmetry). Initially attempted
  via `api/search/*` pagination assuming one document per entry; wrong — the search index returns
  far more documents than the real 59,608-entry total (confirmed live: still-valid, non-duplicate
  EMDB IDs kept appearing past a 350,000 offset), producing unbounded duplicate matches. Fixed by
  switching to the bulk holdings index
  (`ftp.ebi.ac.uk/pub/databases/emdb/status/latest/emdb_released_holdings.json`, confirmed exactly
  59,608 entries) for the definitive ID list, then per-entry REST calls
  (`www.ebi.ac.uk/emdb/api/entry/{id}`), concurrent. Result: 34,856 entries matched to a PDB entry
  already in this corpus.
- **wwPDB Chemical Component Dictionary (CCD) / BIRD** ✅ **implemented** — full 53,417-entry
  dictionary parsed from the bulk `components.cif.gz` with gemmi (Phase 2), not the drug-like subset.
- **wwPDB validation reports** ✅ **implemented** (`scripts/download_wwpdb_validation.py`, round 3) —
  RCSB GraphQL's `pdbx_vrpt_summary` doesn't expose clashscore/Rama-outliers at entry level
  (confirmed by introspection, Phase 2), but PDBe's `global-percentiles/entry` REST endpoint does:
  Ramachandran outliers, rotamer outliers, and clashscore with percentile ranks, pulled for the full
  256,448-entry corpus (24 concurrent workers, 252,756 entries returned data). **TWILIGHT**
  (ruppweb.org) ✅ **implemented** (corpus expansion round) covers the complementary ligand-fit half
  of structure QC — per-ligand-instance RSCC/OWAB density-quality scores, 870k rows. Together these
  cover both halves of "is this a good structure": data-fit (resolution/R-free) and model-geometry
  (Ramachandran/rotamer/clashscore) and ligand-pose-fit (RSCC).
- **PDB-REDO** ✅ **implemented** (`scripts/download_pdbredo.py`, round 4) — re-refined/re-built
  X-ray structures, real R-free before/after deltas ("deposited != optimal" as a trained judgment,
  not just a metadata field). Initially attempted via `rsync://rsync.pdb-redo.eu/pdb-redo/`
  (metadata-only filter dry-run tested and confirmed correct), but the real full-tree pull stalled
  silently — an ESTABLISHED TCP connection with zero files written after 25+ minutes, the same
  silent-hang shape this project has repeatedly hit with long-lived connections. Switched to HTTP
  per-entry fetches (`pdb-redo.eu/db/{pdbid}/data.json`, confirmed live) against this corpus's own
  known 256,448 PDB IDs instead of having rsync enumerate PDB-REDO's tree. Result: 195,194 entries
  with PDB-REDO data (median R-free delta +0.01, 95,847 entries improved by >0.01 on re-refinement).

### Predicted / computational structures
- **AlphaFold DB** ✅ **implemented** (`scripts/download_alphafold.py`, round 3) —
  `alphafold.ebi.ac.uk` per-UniProt REST API (`/api/prediction/{accession}`), scoped to the top
  15,000 UniProt accessions by PDB cross-reference count (not all ~214M AlphaFold predictions —
  bulk FTP/GCS mirroring the full set isn't warranted when the goal is linking predictions to real
  experimental structures this corpus already has). 13,754 accessions had a prediction available.
  Global pLDDT, per-band confidence fractions (very-low/low/confident/very-high), and model metadata
  pulled per entry.
- **ESM Metagenomic Atlas** — large-scale predicted structures; useful as an explicit "predicted vs.
  experimentally determined" contrast case (this is where the "not a structure predictor" boundary
  gets trained in).
- **OpenBind** (openbind.uk) — Diamond Light Source-led UK consortium. Confirmed live as of this
  writing: first public release May 2026, protein-ligand structure-affinity data for the EV-A71 2A
  protease, access noted under `/documents-and-tools`. **Caveat:** small/early dataset (one target
  so far), no confirmed bulk API yet — verify the actual access mechanism in Phase 2 before writing
  a dedicated downloader; treat it as a growing source to poll periodically, not a one-shot pull.

### Sequence, domain & fold classification
- **UniProtKB/Swiss-Prot** ✅ **implemented** (`scripts/download_uniprot.py`, corpus expansion round)
  — REST batch endpoint, scoped to the 73,910 accessions cross-referenced to a PDB structure (not
  all of Swiss-Prot/TrEMBL). Also pulled: the 1,201-entry controlled-vocabulary keyword list.
- **Pfam** ✅ (via SIFTS mapping, Phase 2) / **InterPro** ✅ **implemented**
  (`scripts/download_interpro.py`) — full 54,190-entry dictionary (name/type/GO terms), not just the
  PDB-referenced subset.
- **CATH** ✅ **implemented** (`scripts/download_cath.py`) — full classification hierarchy
  (601,328 domains, 8,151 named codes), joined to the SIFTS PDB→CATH-domain-ID mapping.
  **SCOP2** ✅ **implemented** (`scripts/download_scop2.py`, round 4) — fold/superfamily/family
  descriptions. SCOP2's own domain (scop.mrc-lmb.cam.ac.uk) genuinely still has no working scripted
  API (its documented REST API 404s live, confirmed again this round — the earlier "deferred" call
  was correct, not stale). The real working path turned out to be EBI's own PDBe mappings API
  (`ebi.ac.uk/pdbe/api/mappings/scop2/{pdb_id}`, confirmed live), queried per PDB ID (SIFTS'
  SF_DOMID/FA_DOMID turned out to be near-unique per-chain domain *instance* IDs, not reusable
  classification node IDs as first assumed — 36,898 distinct out of 36,915 rows). Scoped to the
  27,530 distinct PDB IDs already in `sifts_pdb_scop2.csv`. Result: 67,083 domain-level rows.
- **Gene3D** — not yet pursued.

### Cross-reference & interaction annotation
- **SIFTS** (EBI) ✅ **implemented** (`scripts/download_rcsb.py`, Phase 2) — UniProt, Pfam, CATH,
  SCOP2, EC/enzyme, GO, InterPro chain mappings (7 files). Two of chem_sage's original file names
  (`cc-counts.tdd`, `pdb_chain_cath_scop.csv.gz`) have been retired upstream since it was written;
  verify the live directory listing before trusting any inherited filename.
- **PDBe-KB** — deferred: real, working per-entry REST API, but no bulk endpoint (256k requests to
  cover the full corpus is impractical in one pass); RCSB's GraphQL enrichment already covers most
  of the same ground.
- **PDBsum** — deferred, and not really a corpus source: `PDBsum1` (a specific tool Marc pointed at,
  github.com/RomanLas/PDBsum1) is a local install-and-run generator with no bulk data or API: it
  processes a user-supplied PDB file into interaction diagrams. Better fit as a Phase 6 tool-calling
  candidate alongside PyMOL, not a Phase 2 download target.
- **PISA** (EBI) — still deferred as a bulk *download*: real, working per-assembly REST API
  (`ebi.ac.uk/pdbe/api/pisa/assembly/:pdbid/:assemblyid`), no bulk endpoint, same 256k-request
  problem as PDBe-KB. **Practical substitute implemented instead (round 4):** FreeSASA
  (`gen_freesasa_interface`/`gen_assembly_biography` in `build_dataset.py`) computes real buried
  interface area locally (complex SASA vs. sum of isolated-chain SASA) on demand against the
  downloaded mmCIF pool — sidesteps the 256k-request wall entirely rather than working around it.
- **ProtCID** — not yet pursued.
- **Pharos** (pharos.nih.gov, not in the original brainstorm — added on request) ✅ **implemented**
  (`scripts/download_pharos.py`) — target druggability/development level (TDL), joined via UniProt
  accession. Bulk pagination (`targets(top,skip)`) is broken server-side (confirmed via schema
  introspection — silently returns the same 10 targets regardless of arguments); per-target lookup
  works, so this is scoped to the top 2,000 PDB-cross-referenced UniProt accessions by structure
  count rather than all ~20k human targets.

### Ligand & binding data
- **PDBbind** — measured binding affinities for protein-ligand complexes; largely superseded for
  this project's purposes by the BindingDB pull below, which already cross-references PDB directly.
  Not separately pursued.
- **BindingDB** ✅ **implemented** (`scripts/download_bindingdb.py`, round 3) — bulk TSV download
  (~9 GB uncompressed, 3.2M+ measurements), streamed row-by-row out of the zip (never loaded fully
  into memory) and filtered to the 113,366 rows whose PDB cross-reference field matches this
  corpus. Real measured Ki/IC50/Kd/EC50 potency data — the piece TWILIGHT's pose-fit RSCC doesn't
  cover (a ligand can be perfectly modelled and still bind weakly, or vice versa).
- **PLIP** ✅ **implemented as a tool** (round 4, `pip install plip`, `gen_plip_interactions`) —
  real protein-ligand interaction fingerprints (H-bonds, hydrophobic contacts, π-stacking), run at
  dataset-generation time against real bound ligands, not downloaded in bulk. Needed `swig` (via
  brew) as an undocumented build dependency for its openbabel Python bindings, and only the openbabel
  wheel pinned to the exact version matching the brew-installed native library (3.2.1, not the
  default-resolved 3.1.1) actually imports without an `AttributeError` on this machine — confirmed
  the hard way. PLIP only accepts legacy PDB format, converted via gemmi first (same pattern DSSP
  already used).

### Membrane, disorder, nucleic acid, scattering
- **OPM** ✅ **implemented** (`scripts/download_opm.py`, round 4) — membrane protein orientation/
  bilayer placement. No formal scripted API (JS SPA), but its backing storage is a publicly
  listable Google Cloud Storage bucket (`storage.googleapis.com/opm-assets`, confirmed live,
  S3-style XML listing) — enumerable without a documented API. Each entry's `.pdb` file carries the
  membrane placement as a REMARK line ("1/2 of bilayer thickness"). Originally pulled sequentially
  with a per-file delay (~1.8s/request against this GCS-backed host, would have taken ~8h for the
  full 15k-entry set); switched to concurrent fetching, same pattern as every other multi-request
  downloader this round. Result: 15,013 entries. **PDBTM** — not pursued.
- **MobiDB** ✅ **implemented** (`scripts/download_mobidb.py`, round 4) — intrinsic disorder
  regions (curated where available, else a consensus prediction), a real biological reason a region
  can be missing from a crystal structure. Confirmed live per-accession API works
  (`mobidb.org/api/download?acc={acc}&format=json`); confirmed comma-separated batch queries
  silently return only the first accession (no error) — genuinely one request per accession, no
  working bulk/batch path found. Real coverage is partial (~40% of queried accessions have any
  disorder data — most TrEMBL-tier accessions simply aren't computed), reported honestly rather
  than padded. Result: 29,190 accessions with data out of 73,910 queried. **DisProt** — not
  separately pursued (MobiDB already aggregates DisProt-curated data where it exists).
- **NDB** (Nucleic Acid Database) — DNA/RNA and protein-nucleic-acid complex structures within the PDB.
- **SASBDB** — small-angle scattering data, the solution-state complement to crystal structures.
- **RCSB sequence-identity clusters** ✅ **implemented** (`scripts/download_rcsb_clusters.py`,
  round 4, not in the original brainstorm) — precomputed clustering at 30/40/50/70/90/95/100%
  identity thresholds (`cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-{N}.txt`,
  confirmed live, not documented on RCSB's own clustering docs page — found only by direct fetch).
  Answers "how many genuinely distinct structures of this protein exist," a question no single-entry
  metadata field can answer.
- **Obsolete/superseded PDB entries** ✅ **implemented** (`scripts/download_rcsb_obsolete.py`,
  round 4, not in the original brainstorm) — confirmed live that this corpus's `status_code` field
  is uniformly `REL`; obsolete entries were never pulled at all (RCSB's search API returns only
  released entries by default). `data.rcsb.org/rest/v1/holdings/removed/entry_ids` (bulk list,
  confirmed live) + per-entry `.../removed/{id}` for the replacement ID. Result: 6,103 obsolete
  entries, most with a documented successor.

### Benchmarks, provenance, networks
- **CASP** targets/results — structure-prediction benchmark history. Used as context/QA material
  ("what is CASP, how is prediction accuracy scored") — **never** as training signal for the model
  to attempt prediction itself.
- **STRING** ✅ **implemented** (`scripts/download_string.py`, round 3) — protein-protein interaction
  network. STRING's per-organism coverage is uneven (excellent for model organisms, patchy
  elsewhere), so scoped to the top 3,000 human PDB-cross-referenced UniProt accessions (human has
  both the deepest STRING coverage and the most PDB cross-references) rather than querying every
  organism in the corpus thinly. 23,377 interaction edges, top 8 partners per protein by combined
  confidence score. Two bugs found and fixed during the build: an `IndexError` on accessions with no
  STRING mapping, and a correctness bug (caught by inspecting sample output, not an exception) where
  the network endpoint returns edges across a small neighbourhood rather than a star graph centered
  on the query protein — fixed by resolving the query's STRING preferred name and filtering to edges
  that actually touch it.
- **RCSB PDB-101 / Proteopedia** — plain-language educational explainers; a good source of
  well-written prose for the "explain this concept" behaviour class.
- **AlphaFraud** ✅ **implemented, staged** (`scripts/download_alphafraud.py`, round 4, Marc's own
  sibling project at alphafraud.mdeller.com, not in the original brainstorm) — real computed
  TM-score/GDT-TS/lDDT/CA-RMSD and a FRAUD score / "confidently wrong" flag comparing AlphaFold
  predictions against real post-training-cutoff experimental structures, replacing the round-3
  `gen_alphafold_vs_experimental`'s thin pLDDT-only comparison. **Staged, not blocking:**
  AlphaFraud's own historical backfill is still running server-side (confirmed live: its `/archive`
  only has run labels through late 2022 plus a handful of 2026 weekly ones — 2023-2025 not yet
  backfilled), so this round's pull only captured 8 real comparison rows. Pulls via the public
  `/api/week/{label}` API (not a DB/SSH pull), tagged `pulled_at`; re-run with default args (no
  `--limit-weeks`) once AlphaFraud's backfill completes for full coverage. Found and fixed a real
  bug along the way: an initial "timeout" fix reused one `ThreadPoolExecutor(max_workers=1)` across
  the whole loop, so `future.result(timeout=...)` stopped *waiting* on a stuck request but never
  freed the pool's only worker thread — every subsequent week silently queued behind it forever
  (looked identical to a hang: ESTABLISHED socket, ~0 CPU, no progress for over an hour). Real fix:
  spin up a throwaway single-worker executor per request and abandon (never join) it on timeout.
- **DOI/citation verification** ✅ **implemented** (`scripts/verify_citations.py`, round 4, not in
  the original brainstorm — this is the mechanism behind Marc's "supported by real DOI checked
  references" ask). Every citation-bearing entry's deposited DOI is independently checked against
  CrossRef (exact-DOI lookup + title/year confirmation, not fuzzy discovery — much higher precision)
  and cross-checked against the deposited PubMed ID via NCBI eutils' `esummary` (not `esearch`,
  which had a real backend outage mid-development this round, independent of this script).
  Deduplicated to the 102,285 distinct DOIs across `pdb_entries_enriched.csv` + BindingDB's
  `article_doi`. Result: 97,283 verified / 4,893 mismatched / 108 unresolvable / 1 rate-limited out
  of 109,542. **Two real bugs found and fixed:** (1) `year is None`/`if title` checks against
  pandas-sourced values — pandas represents a missing numeric/string field as float `NaN`, not
  `None`, and `NaN` is truthy in Python, so both checks silently missed the NaN case and crashed
  the run partway through (70 real entries have a DOI but no citation_year) — fixed with
  `pd.isna()`/`pd.notna()`, the same discipline `build_dataset.py` already uses everywhere. (2) A
  much more consequential bug caught by noticing an implausible ~44% "unresolvable" rate mid-run:
  CrossRef was rate-limiting (HTTP 429) under the original 24-worker concurrency despite the polite
  pool `mailto` param, and the code treated *any* non-200 response as "unresolvable" — silently
  mislabeling thousands of real, valid DOIs as fake. Fixed with real retry-with-backoff on 429 (a
  separate `rate_limited` bucket for genuine post-retry failures, never conflated with
  "unresolvable") and empirically-tuned concurrency (tested 10/14/18 workers live, settled on 16 —
  the true post-fix unresolvable rate is 0.1%, not 44%).

### Structure search/comparison tools (used for tool-calling examples, not bulk downloads)
- **Foldseek** ✅ **implemented as a local tool** (round 4, `gen_foldseek_neighbors`) — precompiled
  universal macOS binary from GitHub releases (not available via brew/conda on this machine). A
  local database was built once over the full 256,444-file mmCIF pool (`scripts/
  build_foldseek_db.py`) — `createdb` took ~8 minutes for the full pool (742% CPU, all cores; a
  200-file smoke test first suggested ~12-15 min, so this landed faster than estimated), giving
  genuine offline structural-neighbor search against chatPDB's own corpus, not an external API call.
- **US-align** ✅ **implemented as a local tool** (round 4, `gen_usalign_pairwise`) — installed via
  `brew install brewsci/bio/usalign` (confirmed live and current; no precompiled binary or GitHub
  releases exist upstream, brew was the only real option). Accurate pairwise TM-score/RMSD,
  complementing Foldseek's fast corpus-wide search.
- **fpocket** ✅ **implemented as a local tool** (round 4, `gen_fpocket_druggability`) — installed
  via `brew install brewsci/bio/fpocket` (confirmed active, non-deprecated tap). Accepts mmCIF
  natively (no gemmi conversion needed, unlike FreeSASA/PLIP), real pocket detection + druggability
  scoring.
- **cctbx/MolProbity** — attempted (round 4, per Marc's explicit decision to take on the install
  friction) for independent local Ramachandran/rotamer recomputation
  (`gen_geometry_recompute_disagreement`), scoped via `bootstrap.py --builder=molprobity` (the
  correct scoped builder, not the full cctbx/phenix suite). **Abandoned after 3 attempts** — every
  attempt failed on the same persistent error (`RPC failed; curl 92 HTTP/2 stream 5 was not closed
  cleanly: CANCEL`) partway through cloning `cctbx_project`/`cbflib`, including after forcing
  `git config --global http.version HTTP/1.1` (the standard workaround for this exact error class).
  This looks like a genuine, unresolved network-transport issue on this machine/connection for
  large git clones specifically, not a cctbx-side problem — worth revisiting from a different
  network in a future round. The generator gracefully returns `[]` without it; nothing else in this
  round depended on it.
- **Dali server**, RCSB **1D Coordinate Server** / **Sequence Coordinates API** — not pursued
  (Foldseek + US-align cover the structural-comparison need locally and offline).

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

**Round 5 additions** (full tool-ecosystem review, Marc's explicit ask — see round 5 narrative
below for the full story):
- **PyMOL — full command awareness**: `scripts/build_pymol_command_corpus.py` introspects the real
  installed PyMOL 3.1.0 API (`dir(cmd)` + `inspect.getdoc`/`inspect.signature`) for its complete
  436-command surface, replacing the previous 3-hardcoded-template approach.
- **ChimeraX** — headless-scriptable (`ChimeraX --nogui --silent --exit --script`), full command
  awareness via `scripts/build_chimerax_command_corpus.py` introspecting
  `chimerax.core.commands.cli.registered_commands()`/`cli.usage()` (547 real commands).
- **MAFFT** (brew) + `Bio.Align.PairwiseAligner` — real pairwise and multiple sequence alignment.
- **logomaker** / **weblogo** — real WebLogo-style sequence logos, downstream of a real MAFFT MSA.
- **biotite** — native mmCIF phi/psi (Ramachandran), Cα-Cα contact maps, per-residue B-factor
  extraction, all computed directly from the deposited structure, no PDB conversion needed.
- **py3Dmol** — self-contained interactive HTML structure viewers (complements PyMOL/ChimeraX's
  static ray-traced PNGs).
- **pdb-tools** — ~50 small, real PDB-manipulation CLI utilities (`pdb_selchain`, `pdb_delhetatm`,
  `pdb_tidy`, etc.).
- **pdb2pqr** + ChimeraX's `coulombic` command — real electrostatics preprocessing (protonation
  states, partial charges/radii) and calculation, without taking on APBS.
- **OpenMM** (pip) + **GROMACS** (brew, `gmx`) — real energy-minimization pipelines (Python-native
  and CLI-file-based respectively), scoped to minimization rather than production MD trajectories
  for corpus-generation runtime reasons.
- **CCP4** (`/Applications/ccp4-9`) + **PHENIX** (`/Applications/phenix-2.1-6048`) — both real,
  pre-installed local suites (not pip packages, each needs its own env script sourced). Real
  `cif2mtz`/`ctruncate`/`refmac5` (CCP4) and `phenix.refine`/`phenix.molprobity` (PHENIX)
  refinement/validation against real deposited reflection data. PHENIX's bundled cctbx build also
  revives round 4's abandoned standalone-cctbx/MolProbity item.
- **AutoDock Vina** (pip, non-trivial Boost-version install — see `requirements.txt`) + OpenBabel
  (already installed) for receptor/ligand PDBQT prep — real redocking of deposited ligands back
  into their own binding pockets.

**Round 6 additions** (R/plotly/pandas audit, Marc's explicit ask — see round 6 narrative below):
- **MDAnalysis** and **ProDy** — both installed since round 1, neither ever wired into a generator
  until now. MDAnalysis: real per-residue RMSF across NMR ensemble models (conformational
  variability). ProDy: real ANM (elastic network model) normal mode analysis, predicting
  flexibility from a single structure's geometry alone.
- **bio3d** (R package, `install.packages("bio3d", ...)`, not pip) — real R-based structural
  analysis (parsing, atom selection, B-factor extraction, normal mode analysis) via headless
  `Rscript`. R itself was already installed (chem_sage-era: ggplot2/dplyr/rcdk), just never used
  for structural biology.
- **plotly** — interactive Ramachandran/contact-map/B-factor charts, reusing round 5's exact real
  biotite computations under a new interactive rendering backend, complementing py3Dmol's
  interactive 3D view.
- **py3Dmol — full command awareness**: 108 real `GLViewer` methods scraped from 3Dmol.js's own
  official API reference (`scripts/build_py3dmol_command_corpus.py`). Unlike PyMOL/ChimeraX,
  py3Dmol's Python API is a blind `__getattr__` proxy with no local introspection target, so this
  tier is documentation-grounded, not execution-verified — no headless browser/JS engine exists in
  this project to confirm a call renders correctly.
- **pandas** — already a hard dependency used throughout corpus construction, now also taught as a
  skill the model writes itself (`groupby`/`sort_values`/`merge` against real corpus CSVs).

---

## 6. Base model survey (Phase 1 — survey before committing)

Marc's explicit call: do not default to chem_sage's Qwen2.5-32B-Instruct-4bit. Build
`scripts/survey_base_models.py` to run a small, fixed prompt set — structural-bio tool-emission,
format-literacy Q&A, general instruction-following — against each candidate served via
`mlx_lm.server`, recording tok/s, peak RSS, and a qualitative pass/fail per prompt.

Candidates, verified live on Hugging Face via the hub API (2026-07-14):

| Model | Class | Notes |
|---|---|---|
| `mlx-community/Qwen3-32B-4bit` | dense, newer generation | Apache 2.0, 14.4K downloads. Same architecture family/config shape chem_sage already validated for MLX-LM LoRA. Leading candidate. |
| `mlx-community/Qwen2.5-32B-Instruct-4bit` | dense | Apache 2.0, 295K downloads. chem_sage's proven baseline — zero unknowns on this exact Mac. Serves as the survey's control/floor. |
| `mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit` | dense, reasoning-distilled | 7.4K downloads. Worth testing specifically for method-interpretation/QC reasoning tasks. |
| `mlx-community/gemma-4-31b-it-4bit` | dense | Apache 2.0 (Gemma licence terms apply), 38.7K downloads. Natively multimodal (image-text-to-text) but usable text-only; only pure-text prompts used in this survey. |

**Exit test:** a short written comparison appended to this document (§ survey results, to be added),
one model selected and recorded, before Phase 2 begins.

### Survey results (run 2026-07-15)

All four candidates run to completion (8 prompts each, `config/system_prompt.txt` as system
message, temperature 0.15). Raw transcripts: `eval/survey/results/survey_20260714_200040.md`
(Qwen2.5), `survey_20260715_005930.md` (DeepSeek-R1-Distill, gemma4), `survey_20260715_012739.md`
(Qwen3). Full JSON alongside each.

| Model | Avg tok/s | Truncated (finish=length) | Automated checks | Notable |
|---|---|---|---|---|
| Qwen3-32B-4bit (thinking off) | 4.3 | 0/8 | 4/4 | Best accuracy and completeness of the four; correctly self-identified as chatPDB; clean, well-organised refusal on the out-of-scope prediction prompt. Markedly slower generation. |
| Qwen2.5-32B-Instruct-4bit | ~12 (11.0–14.4) | 2/8 | 4/4 | Fast, accurate, chem_sage's proven baseline on this exact hardware/software stack. Ran at the original 500-token budget (raised to 800 for the other three after this run, to give reasoning models room) — its two truncations are a budget artifact, not a distinct weakness. |
| DeepSeek-R1-Distill-Qwen-32B-4bit | ~13.6 | 5/8 | 4/4 | Fast but verbose, truncating most non-code answers even at 800 tokens. Mischaracterised a 2.8 Å structure as "high-resolution" (Qwen2.5 and Qwen3 both correctly called it moderate/medium) — a real domain-convention slip, notable given chatPDB's hard rule against unverified structural claims. |
| gemma-4-31b-it-4bit | ~6.2, highly variable (0.6–11.0) | 5/8 | 4/4 | Sharp, specific detail when it completed (e.g. exact legacy-PDB atom/chain limits), but unstable generation speed and heavy truncation. Gemma licence (not pure Apache 2.0) and natively-multimodal overhead unused in a text-only deployment are additional minor costs. |

**Reading:** the automated checks (code-block presence, SIFTS/UniProt/Pfam mention, refusal-phrase
detection) are floor-level competence and all four pass them — they don't discriminate. The real
signal is in the transcripts: Qwen3 is the clear quality leader (complete, accurate, well-calibrated
on structural-QC judgement calls like the resolution question) but roughly a third the speed of the
others; Qwen2.5 is the fast, zero-operational-risk incumbent; DeepSeek-R1-Distill and gemma4 both
struggled with truncation at equal or greater token budgets and gemma4's speed was additionally
unpredictable, making them harder to recommend for an interactive local CLI regardless of raw tok/s.

**Decision (2026-07-15): `mlx-community/Qwen3-32B-4bit`.** Marc's call, choosing transcript quality
over raw speed. The slower generation is a known, accepted tradeoff going into Phase 2 onward —
worth remembering when tuning `max_tokens`/timeouts in the CLI (Phase 8) and when estimating
wall-clock time for the eval harness (Phase 7) and the fine-tune itself (Phase 4). Thinking mode
must stay disabled at inference (see section 3) except where a future round deliberately wants to
test reasoning-mode behaviour on QC/judgement-call prompts.

---

## 7. Phase-by-phase build plan

### Phase 0 — Environment (0.5 day)
1. Create a venv (Python 3.11+), install `requirements.txt`.
2. Confirm MLX-LM works: `mlx_lm.generate --model mlx-community/Qwen2.5-7B-Instruct-4bit --prompt "hello"`.
3. Confirm Biopython, gemmi, DSSP, PyMOL each round-trip on a real PDB entry (e.g. fetch `1CRN`,
   parse it, assign secondary structure, render a quick image).

**Exit test:** all four tools parse/process the same real structure without error.

**Done (2026-07-14).** venv created with Homebrew Python 3.14.6, `--system-site-packages` (so the
already brew-installed PyMOL module is importable without a separate pip install). `mkdssp` installed
via `brew tap brewsci/bio && brew install brewsci/bio/dssp`. All of `requirements.txt` installs
clean. Two environment issues hit and fixed, worth knowing about before recreating this venv
elsewhere:
- **Xcode Command Line Tools were broken** (libc++ headers stripped to ~11 files, missing `<cmath>`
  etc.), which failed `prody`'s native build. Fixed with
  `sudo rm -rf /Library/Developer/CommandLineTools && sudo softwareupdate -i "Command Line Tools for
  Xcode 26.6-26.6"` (the GUI `xcode-select --install` path hung waiting for a click with no display
  attached — the headless `softwareupdate -i` install is the reliable path on this machine).
- **`KMP_DUPLICATE_LIB_OK=TRUE` is required** — torch and mlx each bundle their own `libomp.dylib`,
  and importing both in one process aborts (`OMP: Error #15`) without it. Set as an `export` at the
  bottom of `.venv/bin/activate` (gitignored, so re-add it if the venv is ever recreated).
- **Bio.PDB's DSSP wrapper infers file type from the path's extension** — `PDBList.retrieve_pdb_file`
  saves files as `pdbXXXX.ent`, which DSSP's file-type sniffing rejects. Copy/rename to a `.pdb`
  extension before handing the path to `Bio.PDB.DSSP.DSSP()`.
- Non-fatal: PyMOL prints a `ModuleNotFoundError: No module named 'chatmol'` on launch from this
  venv — it's executing Marc's personal `~/.pymolrc.py`, which imports a module from an unrelated
  project. Doesn't block loading/rendering, but will show up in every `tool_exec.py` PyMOL call's
  stderr later; worth a `pymol.pymol_argv` no-startup-script flag when Phase 6 builds the shim.

Verified end to end: `mlx_lm generate` produced text at ~74 tok/s on `Qwen2.5-7B-Instruct-4bit`
(4.4GB peak memory). Fetched real entry `1CRN` and round-tripped it through all four tools: Biopython
parsed it (1 chain, 46 residues, 327 atoms), gemmi parsed the same file independently and agreed,
DSSP assigned secondary structure (19 helix, 4 strand, 3 3-10-helix, 3 turn, 6 bend, 11 coil
residues), and PyMOL loaded it and rendered a cartoon PNG confirming the fold visually.

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

**Done (2026-07-15).** `scripts/download_rcsb.py` pulls four sources into `data/corpus/rcsb/`
(1.1 GB total, gitignored, regenerate with the script):
- `pdb_all_entries.csv` / `pdb_entries_enriched.csv` — 256,448 entries each, the second GraphQL-
  enriched with method/resolution/R-free/R-work/EM-resolution/entity counts (verified live against
  the RCSB schema before writing the query — `refine` and `em_3d_reconstruction` resolve correctly;
  `pdbx_vrpt_summary` does not expose clashscore/Rama-outliers at entry level, confirmed by
  introspection, so those stay a separate future "wwPDB validation reports" source per section 4).
- `pdb_ccd_full.csv` — the full 53,417-entry Chemical Component Dictionary, parsed directly from
  wwPDB's bulk `components.cif.gz` with **gemmi** (all ~50k blocks in 3.6s) rather than enumerating
  IDs from an index file and round-tripping each through GraphQL — the index file this repo's first
  draft relied on (`cc-counts.tdd`) has been retired, and the RCSB Search API fallback's query shape
  had also gone stale. Parsing the authoritative bulk CIF sidesteps both problems and is more
  reliable regardless.
- Seven SIFTS cross-reference files (UniProt, Pfam, CATH, SCOP2, EC/enzyme, GO, InterPro) — the
  combined `pdb_chain_cath_scop.csv.gz` chem_sage's downloader used has also been retired by EBI in
  favour of separate files; verified the live directory listing before wiring up filenames. GO and
  InterPro are enormous (16.3M and 4.3M rows) — `ingest_rag.py` samples both down to 150k rows
  before chunking (see below) rather than let chunk sizes balloon.

**Bug caught and fixed during this pass:** the entries-index parser silently produced misaligned
columns — wwPDB has changed `entries.idx`'s layout since chem_sage's `download_pdb.py` was written
against it (now `IDCODE, HEADER, ACCESSION DATE, COMPOUND, SOURCE, AUTHOR LIST, RESOLUTION,
EXPERIMENT TYPE`, ported code assumed a different, older 8th-column layout). Caught by cross-
checking a known entry (102M) against the independently-sourced `pdb_entries_enriched.csv` and
finding the title/organism/date fields in the wrong columns. Fixed by reading the live file's header
row directly rather than trusting the inherited assumption — a reminder that every ported chem_sage
data-fetching pattern needs its column/field assumptions re-verified live, not just its code reused.

`scripts/ingest_rag.py` and `rag/retrieve.py` are ported near-verbatim from chem_sage (same
chunking/embedding/Chroma pattern). One real scaling gap found and fixed: chem_sage's
`MAX_CHUNKS_PER_CSV` cap-driven chunking assumes files up to ~1M rows; at 16.3M rows,
`sifts_pdb_go.csv` under the existing cap would have produced ~192,000-character chunks (far past
what a 512-token embedder encodes). Added `MAX_ROWS_PER_CSV`, an even-stride sample applied before
chunking for the two outsized files (GO, InterPro capped to 150k rows each) — a representative
sample, not full coverage, which is an accepted tradeoff since exact lookups on this kind of
low-diversity relational data belong in `rag/corpus_lookup.py`, not semantic search, anyway.

**Finding that shaped the architecture:** dense embedding search over chunks that each concatenate
dozens to hundreds of table rows is good at finding topically-similar chunks but bad at pinpointing
one exact row inside a chunk — confirmed empirically: "what R-free was 102M solved at" retrieved
chunks *about* resolution/R-free generally, not specifically the chunk containing row 102M. This is
exactly the limitation chem_sage's `rag/corpus_lookup.py` exists to solve, so chatPDB built the same
component: `rag/corpus_lookup.py` extracts PDB IDs / CCD comp IDs from a query via regex (with a
stopword guard — "PDB" itself is coincidentally also a real, irrelevant CCD code) and looks them up
directly with pandas across eight registered corpus files, bypassing embeddings entirely for exact
identifier questions. The two huge SIFTS files (GO, InterPro) are deliberately excluded from this
registry (full-file pandas loads would be slow for a single lookup) and remain semantic-search-only
for now.

**Verified end to end:** `corpus_lookup.lookup("102M")` returns exact, correct rows from all eight
registered files (resolution 1.84 Å, R-free 0.203, R-work 0.159, UniProt P02185, Pfam PF00042, CATH
102mA00, EC 1.11.1.-/1.7.-.-). Fed as grounding context into a live `mlx_lm.server` running
Qwen3-32B-4bit (thinking disabled), the model correctly answered "resolution of 1.84 Å... R-free
value of 0.203... Chain A... maps to the UniProt accession P02185" and cited the corpus context —
the actual grounded-answer loop, not just retrieval in isolation.

### Corpus expansion round (2026-07-15/16)

Before starting Phase 3, Marc asked to flesh out the corpus further: named sources (PDBeChem, SCOP,
CATH, InterPro, Pharos, PDBePISA, PDBsum1, TWILIGHT, UniProtKB/Swiss-Prot, ExPASy) plus a broader
search for anything still missing. Every source below was verified live (curl/introspection) before
writing a downloader — three separate stale-API assumptions had already bitten Phase 2 (retired
index files, a changed column layout), so nothing went in on the strength of documentation alone.

**Implemented and ingested** (5 new downloaders, `scripts/download_{cath,interpro,pharos,twilight,
uniprot}.py`, all ported to the same session/GraphQL/pagination patterns as `download_rcsb.py`):

| Source | What it adds | Scale | Access method |
|---|---|---|---|
| **CATH** | Domain fold classification (Class/Architecture/Topology/Homology descriptions), joined to the existing SIFTS PDB→CATH-domain-ID mapping | 601,328 domains, 8,151 named codes | Bulk HTTPS mirror of CATH's FTP release (`download.cathdb.info`) — two flat files, gemmi not needed, plain text parse |
| **InterPro** | Entry name/type/GO terms/member-database cross-refs for all InterPro domains (not just the ones referenced in PDB — full dictionary) | 54,190 entries | REST API, cursor-paginated, 200/page (~271 requests) |
| **Pharos** | Target druggability (TDL: Tclin/Tchem/Tbio/Tdark), family, disease associations — joins via UniProt accession to the existing SIFTS PDB→UniProt mapping | 722 targets matched (top 2,000 PDB-cross-referenced UniProt accessions by structure count, ranked) | GraphQL, per-target lookup (bulk `targets(top,skip)` pagination is broken server-side — see below) |
| **TWILIGHT** | Per-ligand-instance electron-density fit quality (RSCC, OWAB) — the "structure QC" data Phase 2's original RCSB pull couldn't reach (`pdbx_vrpt_summary` doesn't expose it) | 870,386 ligand instances | Single bulk bzip2 TSV snapshot (2020-01-15; not live-updated) |
| **UniProtKB/Swiss-Prot** | Protein name, function (curated prose), organism, keywords for every UniProt accession cross-referenced to a PDB structure, plus the full 1,201-entry controlled-vocabulary keyword list | 73,910 entries (100% of unique accessions in `sifts_pdb_uniprot.csv`) + 1,201 keywords | REST batch endpoint (`/uniprotkb/accessions`, 100/request, ~740 requests) + `/keywords/stream` (single request) |

Full corpus now: **18 files, 65,811 RAG chunks** (up from 10 files / 27,484 chunks after Phase 2's
initial RCSB/SIFTS pull). `rag/corpus_lookup.py`'s registry grew from 8 to 11 files, plus a new
two-hop join (PDB → SIFTS CATH-domain-ID → CATH classification description) and two new ID
patterns (UniProt accession, InterPro accession) so the exact-lookup fast path covers all of it, not
just the original RCSB/SIFTS set.

**Deliberately deferred** (researched, access method identified, not pulled — reasons below so this
doesn't get silently reattempted or silently forgotten):

- **PDBePISA** — real, working REST API (`ebi.ac.uk/pdbe/api/pisa/assembly/:pdbid/:assemblyid`), but
  per-entry only, no bulk endpoint. 256k entries × a request each is impractical in one pass.
- **PDBeChem enrichment** — RCSB's bulk CCD (already ingested via `pdb_ccd_full.csv`) covers
  SMILES/formula/name; PDBeChem's REST API adds cross-refs to BRENDA/Probes-and-Drugs/systematic
  names, but only per-compound (`ebi.ac.uk/pdbe/api/pdb/compound/summary/:id`) — 50k+ requests for
  marginal, chemistry-adjacent (not structural) value.
- **PDBe-KB entry/funpdbe annotations** — real, working, per-entry only (confirmed via
  `ebi.ac.uk/pdbe/api/pdb/entry/summary/:id`); RCSB's GraphQL enrichment already covers most of the
  same ground (assembly/entity composition) that would justify a 256k-request pull.
- **SCOP2 full classification hierarchy** — SCOP has migrated fully under PDBe (redirects to
  `ebi.ac.uk/pdbe/scop/`, a JS SPA); could not find a working public API endpoint after exhausting
  reasonable discovery (tried `rest.scop.mrc-lmb.cam.ac.uk`, `supfam.mrc-lmb.cam.ac.uk`, HTML
  comment hints — none resolved). SIFTS's PDB→SCOP2-domain-ID mapping (`sifts_pdb_scop2.csv`,
  already ingested) still gives partial cross-referencing without the fold descriptions.
- **PhosphoSitePlus** — bulk download requires a paid/registered license (commercial tiers
  $5k–$20k/yr; academic terms require direct application) — not something to script around.
- **PDBsum1** (github.com/RomanLas/PDBsum1) — not a data source at all: a local install-and-run
  tool that processes a user-supplied PDB file into interaction diagrams. No bulk data, nothing to
  download. Worth revisiting as a **tool-calling candidate** for Phase 6 (`rag/tool_exec.py`)
  instead, alongside PyMOL — noted for that phase, not pursued now.

**Bugs found and fixed while implementing the above** (beyond the two already logged in the main
Phase 2 section):

1. **Pharos bulk pagination is broken.** `targets(top, skip)` silently ignores both arguments
   (confirmed via schema introspection — they're real, correctly-typed `Int` args, not a naming
   mistake) and always returns the same first 10 targets regardless of what's requested, with
   inline literal values too (not a variable-passing bug). First attempt appeared to succeed
   (20,420 rows written) but was 100% duplicate data (10 unique symbols total) — caught by checking
   `nunique()` on the output, not by any error. Fixed by switching to per-target lookup
   (`target(q:{uniprot:...})`, confirmed working) against a prioritised subset instead of the
   broken bulk path.
2. **`scripts/download_interpro.py` hung indefinitely when run via a backgrounded/long-lived
   process**, specifically — a bare synchronous `requests.get()` to the identical URL always
   succeeded in seconds, and `curl` never had any trouble either. Reproduced repeatedly: CPU time
   frozen at effectively zero for 15+ minutes, an ESTABLISHED-but-silent TCP connection to
   `pg-www.ebi.ac.uk`. Root-caused to `requests.Session()` connection pooling reusing a socket that
   had gone dead without the OS or urllib3 noticing — the client-side read timeout should have
   caught this but didn't. Fixed by dropping `Session()` entirely in favour of one-off
   `requests.get(..., headers={"Connection": "close"})` calls per page, plus per-page disk
   checkpointing every 20 pages so a future stall doesn't lose all prior progress. Same fix applied
   preventatively to `download_uniprot.py`, which ran its full 740-request pull cleanly afterward.

**Row-count caps added to `ingest_rag.py`** for the new large files: `cath_classification`
(8,000-chunk cap, 601k rows), `interpro_entries` (8,000, 54k rows), `uniprot_entries` (20,000 — high
on purpose so the natural per-row chunk sizing dominates over grouping, since function/keyword text
per row is already substantial), `twilight_ligands` (3,000, 870k rows). None needed the
pre-chunking row-sampling `sifts_pdb_go`/`sifts_pdb_interpro` required — none are large enough
relative to their cap to blow past a reasonable chunk size the way a 16M-row file would.

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

**Done (2026-07-16).** Marc's brief: aim for as many examples as possible, target 50,000, to beat
chem_sage's largest round (20,000, R5). Landed at **45,502** (36,402 train / 4,550 valid / 4,550
test) — short of 50,000 honestly, not padded to hit the number (see below), still more than double
chem_sage's R5.

`scripts/build_dataset.py` uses two ground-truth tiers, both real:
- **Metadata-grounded** (bulk, no download needed): ~20 generator functions across the four classes,
  each sampling a real row from an already-ingested, already-verified corpus DataFrame (resolution,
  R-free, UniProt function, CATH fold, EC number, Pharos TDL, TWILIGHT RSCC, CCD identity...) and
  templating the Q/A around that real value. The PDB ID / accession / comp ID in every example is
  read directly from the DataFrame, never a literal — there is no code path that could cite an ID
  the corpus doesn't contain.
- **Execution-verified** (`scripts/download_structure_pool.py`, 820 real structure files stratified
  600 X-ray / 150 NMR / 100 EM, 50–4,000 atoms, capped for fast parsing): DSSP secondary-structure
  assignment and NMR model counts are computed **live** against real downloaded coordinates — no
  metadata shortcut exists for these, so unlike the templated tier they're hard-capped by pool size
  (820 files), not by target count. This is why `tool_calling` landed at 8,010 instead of ~12,500:
  its three metadata-grounded generators (Biopython atom/chain counts, gemmi resolution/method,
  PyMOL scripts) scale to 256k entries, but the two execution-verified ones can't exceed the pool.
  **This is the honest reason the 50,000 target wasn't hit** — the alternative (duplicate pool files
  with reworded questions, or fabricate DSSP output) would have violated "never hand-author a
  number," so the shortfall was accepted rather than the rule bent.

**A real bug caught and fixed during this pass:** `mkdssp` 4.6.1 has a genuine bug in its internal
legacy-PDB→mmCIF conversion — it raised `Duplicate Key violation` on modern REMARK 3
refinement-statistics blocks (multiple TLS groups etc.), failing on roughly 40% of the structure
pool's post-2015 X-ray entries (confirmed via direct `mkdssp` CLI testing, not just the Biopython
wrapper). Root-caused and fixed by converting each file to mmCIF with **gemmi** first (a converter
that doesn't hit the bug) and running `mkdssp` against that instead — `_run_dssp()` in
`build_dataset.py`. Recorded in the DSSP generator's own examples too (a 15% chance of an added
note about the gemmi-conversion fallback), so the model learns the real-world caveat, not just the
textbook-clean code path.

Class balance and token-length stats (Qwen3-32B-4bit tokenizer, p50=549, p99=816, max=1,549 — all
comfortable under any `max_seq_length` chem_sage used) are in `data/README.md`. 20-example spot-read
done manually; every example read as something Marc would have written, no fabricated numbers, no
hedge-text template leaks (one found and fixed — a literal "if applicable" leaking into prose
regardless of whether the value was actually present).

**Round 2 (2026-07-16): "let's add ALL PDB files and ALL fields from the PDB for each entry to
SFT."** A deliberate escalation from round 1's scoped-down approach (820-file stratified sample,
limited GraphQL field set) — Marc asked for full literal coverage instead. Given the resource
implications (150–400 GB, 8–25+ hours for a literal-all file download), this was clarified with
Marc first rather than just launched: confirmed literal all 256,448 entries, mmCIF over legacy PDB
(universal coverage, and it's the format the DSSP fix below already made the robust choice), and
expanding the metadata query as an independent, much cheaper step.

**`scripts/download_all_structures.py`** — every entry as native mmCIF. Sequential download at any
sensible pause would have taken the better part of a day; empirically tested concurrency (8 → 24 →
32 → 64 → 96 parallel workers) against the real endpoint before committing to a setting, since
`files.rcsb.org` is a static-file CDN, not the rate-limited Search/Data API the commonly-quoted
~10 req/s guideline is documented for. Throughput plateaued around 64 workers (~7.3 files/s in
testing; **13.0 files/s sustained** in the actual full run) — diminishing returns beyond that, so 64
was kept as a courtesy ceiling rather than pushed further for a run that would finish in ~5.5h either
way. Fully resumable (skips existing files, logs failures separately for `--retry-failed`).
**Result: 256,444/256,448 entries downloaded (99.998%) in 5.52h, 353 GB, only 5 genuine failures.**

**Expanded `ENTRY_ENRICH_GQL`** (in `download_rcsb.py`) — added, all verified live before writing
the bigger query: unit cell (`cell`), space group (`symmetry`), crystallization conditions
(`exptl_crystal_grow`: pH/temp/method), diffraction (`diffrn`, `diffrn_source`: wavelength),
primary citation (`rcsb_primary_citation`: title/journal/year/DOI/PubMed ID), and per polymer
entity: full sequence, sequence length, source organism, NCBI taxonomy ID, plus `assembly_count`.
Re-ran the full 256,448-entry enrichment with the bigger query.

**A second connection-pooling stall**, same failure mode as `download_interpro.py`'s (Phase 2):
the re-enrichment run showed "completed" only 42 seconds of real CPU time after 5h22m elapsed, a
connection stuck in `CLOSE_WAIT`. `download_rcsb.py`'s module-level `requests.Session()` was the
cause here too, just against `data.rcsb.org` instead of EBI — confirming this is a general pattern
worth watching for on this machine (long-lived pooled connections against *any* host can go
silently dead), not a one-off EBI quirk. Fixed identically: dropped the Session object for one-off
`Connection: close` requests, added retry-with-backoff and checkpointing every 400 batches to
`graphql()`/`step2_entry_enrichment()`. Re-ran cleanly: 256,448/256,448 entries in well under an
hour, steady progress confirmed by CPU-time monitoring before trusting it to run unattended.

**DSSP generator simplified, not just rescaled.** Since `data/structures_all/` is native mmCIF
(never converted from legacy PDB), round 1's gemmi-pre-conversion workaround for mkdssp's
legacy-PDB→mmCIF bug doesn't apply — confirmed directly (`mkdssp` ran cleanly against a sample
native `.cif` with no `Duplicate Key violation`) before simplifying `_run_dssp()` into
`_run_dssp_mmcif()`, which runs `mkdssp` straight against the downloaded file. A different, milder
version of the same class of dictionary-validation error still appears on a small fraction of
*native* RCSB mmCIF files (a handful of `refine_ls_shell`/`pdbx_database_related` duplicate-key
cases) — the generator's existing oversample-and-skip tolerance absorbed this without needing a
further fix; the DSSP sub-generator still hit its full target.

**Result: `tool_calling` reaches its complete 12,500 target for the first time** (round 1: 8,010,
capped by the 820-file pool). Four new generators added from the expanded metadata
(`gen_citation`, `gen_unit_cell_space_group`, `gen_crystallization_conditions`,
`gen_organism_taxonomy`). Full re-run: **50,233 examples** (40,187 train / 5,023 valid / 5,023
test), the original 50,000 target hit cleanly. `corpus_lookup.py`'s registry and the RAG corpus were
also updated/re-ingested with the expanded `pdb_entries_enriched.csv` fields. Full narrative,
class balance, and token stats in `data/README.md`'s v2 entry.

**Round 3 (2026-07-16): "get AlphaFolddb, PDBbind/BindingDB, wwPDB validation reports, and STRING...
make it all thorough as I want the best possible data before we start training."** Explicit
pre-Phase-4 work: four new corpus sources (§4 above — AlphaFold DB, BindingDB, wwPDB validation,
STRING, all ✅ implemented) plus six requested techniques implemented as ~19 new generator functions:
tool-chaining scripts, deeper multi-hop chains (up to 4 hops: PDB→SIFTS→Pharos→BindingDB), bidirectional
traversal (UniProt→PDB and ligand→PDB, reversing every existing generator's direction),
cross-database disagreement + missing-data honesty, comparative entry-vs-entry examples, and a
RAG-shaped multi-source synthesis generator that presents numbered, source-tagged context chunks and
requires a cited answer — training the model for the shape it actually sees at inference time behind
the retriever.

Two classes of real bugs were caught by this project's standing smoke-test-before-full-run discipline
(2,000-example runs, checked after every generator change): (1) NaN-handling gaps in the two new
wwPDB-validation generators (PDBe uses `clashscore == -1` as an unchecked "not computed" sentinel,
and `percent_rama_outliers`/`percent_rota_outliers` can be independently NaN) and a pre-existing gap
inherited from round 1's `gen_twilight_ligand_fit` (TWILIGHT's `LigNm` can be NaN, leaking "ligand
nan" into generated *questions* — `validate()` had only ever checked the assistant's answer text, not
the user's question, so this had been silently leaking since round 1). (2) A more instructive bug:
one new generator's template literally contained the word "none" in ordinary English ("...none of
which is recoverable...") — `validate()`'s leak-detection check (matching the word "none" anywhere)
correctly but overzealously flagged every single example that generator produced, silently rejecting
100% of its real, valid output. Combined with a second, independent, ~51%-of-rows issue (Pharos's
`family` field is genuinely absent for about half its targets, rendering as literal "family nan" in
three generators, one of them present since round 1), this meant the affected generators were
producing near-zero net yield despite generating correctly-grounded raw examples underneath — a
sharp reminder that a validator with zero rejections isn't proof of correctness, and a validator
rejecting *everything* from one generator is easy to miss in an aggregate rejection-rate number.
Fixed the template wording, added `pd.notna()` guards throughout, and hardened `validate()` to a
word-boundary regex checking both message roles.

**Result: 97,241 examples** (77,793 train / 9,724 valid / 9,724 test) — target doubled to 100,000 to
match the larger generator roster (database_cross_referencing alone grew from 9 to 21 generators),
landing at 97% of target with the lowest rejection rate of any round (1.2%). The full run took
~4.5 hours (execution-verified DSSP calls against the unfiltered 256k-file structure pool dominate —
confirmed via `sample`-based process profiling that it was genuinely executing `mkdssp` subprocesses
throughout, not hung; some structures in the full pool are far larger than round 1's size-capped
sample). `corpus_lookup.py`'s registry and the RAG corpus (round 3: 22 files / 79,235 chunks, up from
18 / 65,811) were updated with all four new source files. Full narrative, class balance, and token
stats in `data/README.md`'s v3 entry.

**Round 4 (2026-07-17): a Fable 5 brainstorm ("what's still missing to make this a genuine
protein-structure expert"), implemented in full, plus AlphaFraud integration and a ChemSage
corpus-merge decision.** Explicit pre-Phase-4 work, same framing as round 3. Full source detail in
§4 above; summary of what shipped:

- **Seven new corpus sources**: PDB-REDO (195,194 entries, real re-refinement R-free deltas), EMDB
  (34,856 entries matched, map-level metadata for the now-dominant cryo-EM method), SCOP2 (67,083
  domain-level rows, fold/superfamily/family descriptions), MobiDB (29,190 accessions with real
  disorder data), OPM (15,013 entries, membrane bilayer placement), RCSB sequence-identity clusters
  (all 7 thresholds, 30–100%), and obsolete/superseded entry mapping (6,103 entries) — the last two
  weren't in the original brainstorm, found while scoping "structural biography."
- **AlphaFraud integration, staged**: real TM-score/GDT-TS/lDDT/CA-RMSD/FRAUD-score comparison data
  from Marc's sibling project, replacing round 3's thin pLDDT-only comparison. AlphaFraud's own
  historical backfill is still running server-side, so this round's pull captured only 8 rows —
  re-run with default args once it completes.
- **Independent DOI/citation verification**: every citation-bearing entry checked against CrossRef
  + PubMed at build time, not trusted as deposited — directly answers Marc's "supported by real DOI
  checked references" ask. 109,542 citations verified: 97,283 confirmed, 4,893 flagged as
  mismatched, 108 genuinely unresolvable.
- **Five new local tool-exec integrations**: FreeSASA (real buried-interface-area computation, the
  practical substitute for the still-impractical-to-bulk-download PISA API), fpocket (real pocket
  detection + druggability scoring), Foldseek (a local database built over the full 256,444-file
  mmCIF pool — genuine offline structural-neighbor search against chatPDB's own corpus), US-align
  (accurate pairwise TM-score/RMSD), and PLIP (real protein-ligand interaction fingerprints).
  cctbx/MolProbity was attempted per Marc's explicit "install it anyway" decision but abandoned
  after 3 failed attempts, all on the same persistent network error mid-clone — the corresponding
  generator gracefully returns `[]` without it.
- **~34 new generators**: the house "structure report card" format (the cheapest, highest
  perceived-expertise item of the round), family/homolog-level reasoning, structural biography
  (a UniProt's full PDB history over time), assembly biography (real FreeSASA-backed biological-vs-
  crystallographic interface judgment), self-consistency checks, mutation/variant-effect refusal
  reinforcement, and a small (12-target) disease→target→structures→ligands→clinical-relevance chain
  built by calling the ClinicalTrials MCP tool directly during this session (Open Targets was
  rate-limited throughout the session, so Pharos's existing disease associations were used instead).
- **ChemSage corpus merge — decided against.** Marc asked whether merging in ChemSage's corpus
  would make chatPDB "the mother of all corpuses." No: ChemSage's corpus is molecule-centric
  (SMILES/Lipinski/reaction-enumeration/ChEMBL/PubChem), genuinely orthogonal to protein-structure
  expertise, and training on it would dilute chatPDB's focused identity for no structure-relevant
  benefit — a real multi-task fine-tuning tradeoff, not just a data-hygiene preference. The one
  genuinely relevant piece, PLIP, was pulled in on its own merit above, not as part of a corpus copy.

**Six real bugs found and fixed this round**, three of them serious enough to have corrupted data
if shipped:
1. **PDB-REDO rsync stall.** The original approach (`rsync://rsync.pdb-redo.eu/`) dry-run tested
   correctly but the real full-tree pull silently stalled — an ESTABLISHED TCP connection, ~0 CPU,
   zero files written after 25+ minutes. Switched to HTTP per-entry fetches against this corpus's
   own known PDB IDs instead of having rsync enumerate PDB-REDO's tree.
2. **AlphaFraud's "timeout" fix didn't actually fix anything.** A `ThreadPoolExecutor(max_workers=1)`
   reused across a loop with `future.result(timeout=...)` stops the *caller* waiting on a stuck
   request but never frees the pool's only worker thread — every subsequent submission queued
   behind it forever (looked exactly like a hang: ESTABLISHED socket, ~0 CPU, no progress for over
   an hour). Real fix: a throwaway single-worker executor per request, abandoned (not joined) on
   timeout.
3. **EMDB's search-based pull had no real termination condition.** `api/search/*` pagination was
   assumed to return one document per entry; live mid-run evidence (still-valid, non-duplicate EMDB
   IDs appearing past a 350,000 offset against a real total of 59,608) proved the search index
   returns far more documents than that, producing unbounded duplicate matches. Fixed by switching
   to the bulk holdings index for the definitive ID list, then per-entry REST calls.
4. **`verify_citations.py` crashed on a NaN citation_year**, the same "pandas NaN is not `None`"
   bug class this project has hit before, this time in a file that hadn't yet absorbed the lesson —
   `year is None` (and the parallel `if title` truthiness check, since a NaN float is truthy in
   Python) both silently missed real NaN values and crashed the run partway through. This is also
   the real explanation for a run that "looked stuck" at the same checkpoint for over an hour: it
   had actually crashed early, and stale monitoring kept re-reading an unchanging log.
5. **CrossRef rate-limiting silently corrupted citation-verification data.** Caught only by
   noticing an implausible ~44% "unresolvable" rate mid-run — CrossRef was returning HTTP 429 under
   24-worker concurrency despite the polite-pool `mailto` param, and the code treated *any*
   non-200 response as "this DOI doesn't exist," which would have taught the model that huge
   fractions of real PDB citations are fake. Fixed with real retry-with-backoff on 429, a separate
   `rate_limited` bucket never conflated with genuine unresolvability, and empirically-tuned
   concurrency (10/14/18 workers tested live, settled on 16 — true post-fix unresolvable rate: 0.1%).
6. **OPM's original sequential puller would have taken ~8 hours** for 15,014 entries (~1.8s/request
   against a GCS-backed host with per-request TLS handshake overhead) — switched to concurrent
   fetching, same pattern as every other multi-request downloader, finished in 24 minutes.

**Result: 93,725 examples** (74,981 train / 9,372 valid / 9,372 test), 2.7% rejection rate. The
full run took **~8.5 hours** — confirmed via repeated `sample`-based process profiling and direct
child-process inspection (caught `fpocket` and other real subprocesses mid-execution, not hung)
that this was genuine compute, not a stall: the new execution-verified tool-calling generators
(especially fpocket, whose exhaustive Voronoi-based pocket search has a real long tail on larger
structures) dominate the runtime far more than round 3's DSSP-only tool_calling class did. Landed
short of the 100,000 target (94%) — reported honestly rather than padded, same discipline as every
round. Flagged the realistic timeline mid-run and confirmed with Marc to let it run to full
completion rather than truncate. `corpus_lookup.py`'s registry and the RAG corpus (now includes all
round 4 source files) were updated. Full narrative, class balance, and token stats in
`data/README.md`'s v4 entry.

**Round 5 (2026-07-17): a full visualization/rendering/simulation tool review, Marc's explicit
ask.** Named requirements: full PDB text/graphical generate-parse-manipulate-view capability via
PyMOL and ChimeraX with "full awareness of *ALL* pymol commands"; sequence alignments; DSSP plots;
WebLogo plots; 2D topology plots; an open brainstorm for anything else an expert structural
biologist would expect. Mid-planning, extended twice further: molecular dynamics (OpenMM,
GROMACS) and in-depth crystallography (CCP4, PHENIX, MTZ files). Every item below was live-verified
against the real installed tool before being written into a generator — several initial
assumptions turned out wrong and were caught this way, not shipped:

- **Full PyMOL command awareness**: `scripts/build_pymol_command_corpus.py` introspects the real
  installed PyMOL 3.1.0 API (`dir(cmd)` + `inspect.getdoc`/`inspect.signature`) — 436 real commands,
  346 with real docstrings. Replaces the previous 3-hardcoded-template `gen_pymol_script` with ~19
  execution-verified task templates (each actually run headless via `pymol -cq` against a real
  structure before being kept) plus `gen_pymol_command_reference`, a broad docstring-grounded Q&A
  generator covering the full 436-command surface.
- **Full ChimeraX command awareness**: `scripts/build_chimerax_command_corpus.py` spawns ChimeraX
  headless (`--nogui --silent --exit --script`) and has it introspect its own command registry
  (`chimerax.core.commands.cli.registered_commands()`/`cli.usage()`) — 547 real commands. Mirrors
  the PyMOL treatment: `gen_chimerax_script` (execution-verified `.cxc` scripts) +
  `gen_chimerax_command_reference` (broad command Q&A).
- **Sequence alignment**: `gen_pairwise_alignment` (real `Bio.Align.PairwiseAligner`, same
  SIFTS-pairing pattern as round 4's `gen_usalign_pairwise`) and `gen_msa_family` (real MAFFT MSA
  over RCSB's own 30%-identity sequence clusters, round 4's `clusters_30pct.csv`).
- **WebLogo sequence logos**: `gen_sequence_logo`, chained off the same real MAFFT MSA, rendered
  with `logomaker` — real position-frequency-matrix logo images, the technique
  weblogo.threeplusone.com's own tool is built on.
- **biotite structural plots**: `gen_dssp_plot` (visual SSE track from the already-verified DSSP
  wrapper), `gen_ramachandran_plot` and `gen_contact_map` and `gen_bfactor_plot` (all computed
  directly from native mmCIF via biotite — no legacy-PDB conversion needed for these three).
- **py3Dmol** (`gen_py3dmol_view`, self-contained interactive HTML viewers) and **pdb-tools**
  (`gen_pdbtools_manipulation`, real `pdb_selchain`/`pdb_delhetatm`/`pdb_delresname`/`pdb_tidy`/
  `pdb_selres` invocations).
- **2D topology diagrams — reduced scope, documented honestly.** FlatProt, the one real current
  tool, requires Python <3.14; chatPDB's venv runs 3.14.6 (confirmed via `pip install flatprot
  --dry-run`) — genuinely blocked, not worked around. `gen_topology_schematic` builds a real linear
  SSE schematic (helices as boxes, strands as arrows, real sequence order) from the same DSSP
  assignment, explicitly smaller in scope than a full spatial 2D fold diagram with strand-crossing
  connectivity — the example text says so, rather than overclaiming.
- **Electrostatics**: `gen_pdb2pqr_prep` (real PDB2PQR protonation/charge/radius assignment);
  ChimeraX's `coulombic` command is covered "for free" by the ChimeraX command corpus above. APBS
  skipped (heavier binary, uncertain 2026 maintenance status).
- **Molecular dynamics**: `gen_openmm_script` (real OpenMM energy minimization, implicit solvent,
  AMBER14 — real potential energy before/after) and `gen_gromacs_pipeline` (real
  `pdb2gmx→editconf→solvate→grompp→mdrun` pipeline, explicit SPC/E water, real final potential
  energy). Both scoped to minimization, not production MD trajectories, for corpus-generation
  runtime reasons — both execution-verified, both protein-only (a real bug caught here: the
  structure-size filter initially didn't exclude nucleic-acid entries, and amber99sb-ildn has no
  DNA/RNA residue templates — `pdb2gmx` genuinely rejected mixed-type chains).
- **Crystallography — CCP4 + PHENIX, both real and already installed.** Corrected mid-planning:
  Marc initially believed CCP4 was installed and PHENIX wasn't; live verification found the
  opposite was closer to true, then a direct filesystem search (Spotlight's query missed it) found
  **both** genuinely installed and working — `/Applications/ccp4-9` (`refmac5` 5.8.0431 confirmed)
  and `/Applications/phenix-2.1-6048` (`phenix.refine` 2.1-6048 confirmed). Real deposited
  structure-factor files fetched from RCSB (`-sf.cif.gz`), converted to MTZ via CCP4's `cif2mtz`,
  with `ctruncate` run first for the ~half of deposits that are intensities (I/SIGI) rather than
  amplitudes (F/SIGF) — column names/types vary genuinely per entry and are detected live, never
  assumed. Four generators: `gen_mtz_manipulation` (real gemmi.Mtz summaries), `gen_ccp4_refmac_
  script` (real refmac5 refinement, real R-factor/R-free before/after), `gen_phenix_refine_script`
  (real phenix.refine), and `gen_phenix_molprobity` (real MolProbity validation via PHENIX's
  bundled cctbx — **revives round 4's abandoned standalone-cctbx/MolProbity item**, no git clone
  needed). Documented limitation: deposited SF data is merged/scaled, not raw diffraction images,
  so `aimless`/`pointless` (unmerged-data scaling) genuinely can't be exercised from this data
  source — out of scope, not silently skipped.
- **AutoDock Vina docking**: `gen_autodock_vina_docking`, real redocking of a structure's own
  deposited ligand back into its own deposited pocket (small search box centred on the real
  ligand position, low exhaustiveness — a redocking sanity check, not a blind search), real
  binding-affinity scores from Vina's scoring function.
- **Boltz2 — investigated and dropped.** A stray Spotlight hit turned out to be test fixtures from
  the unrelated BoltzMaker project, confirmed by Marc. Also out of scope on principle: Boltz2 is a
  structure *predictor*, and chatPDB's explicit design boundary is reasoning about *existing*
  structures, never predicting new ones.

**Real bugs found and fixed this round**, all caught by the same "actually run it" discipline
every generator in this file follows:
1. **GROMACS writes its run summary to stderr, not stdout.** `gen_gromacs_pipeline`'s first version
   parsed `mdrun`'s stdout for the "Potential Energy" line and silently produced zero examples —
   confirmed by direct comparison against a manual terminal run, where the exact same text appeared
   on stderr instead.
2. **Relative cache paths broke once the subprocess `cwd` changed.** `_prepare_mtz`'s cached
   `pdb_path`/`mtz_path` were relative to the project root; every crystallography generator spawns
   its subprocess with `cwd` set to a fresh temp directory, so the same relative path silently
   resolved to a nonexistent location and every refmac5/phenix.refine call failed with zero
   examples produced. Fixed by resolving to absolute paths at the cache-read/write boundary.
2b. **The four crystallography generators each independently rebuilt their own candidate pool**,
   rather than sharing one — caught mid-smoke-test via `sample`-based profiling (the same
   diagnostic tool round 4's hang-investigation playbook established) showing genuine, ongoing
   HTTPS I/O to RCSB long after a pool should have been warm. Refactored to build the pool once in
   `main()` and pass it to all three MTZ-based generators.
3. **`vina`'s pip build failed against the default Homebrew Boost (1.90)** — a real C++
   `std::type_traits` incompatibility between Boost 1.90's headers and vina 1.2.7's code, not a
   local misconfiguration. Fixed with `boost@1.85` (keg-only), `-mt`-suffixed libs symlinked to
   unsuffixed names, and `CONDA_DEFAULT_ENV`/`CONDA_PREFIX` spoofed to point vina's setup.py (which
   only searches a conda env, `/usr/local/include`, or `/usr/include` — never Homebrew's
   `/opt/homebrew`) at the right Boost. Documented in full in `requirements.txt` since a fresh venv
   rebuild would hit the identical wall.
4. **meeko's receptor preparation hit a reproducible internal error** (`RuntimeError: Updated N H
   positions but deleted M`) on multiple real deposited structures in this environment — not an
   altloc or hydrogen issue (both ruled out live). Substituted OpenBabel's AutoDock plugin
   (`obabel -xr`) for receptor PDBQT prep, which required its own fix: Vina's PDBQT parser only
   accepts a strict record-type whitelist (`ROOT`/`ATOM`/`BRANCH`/`TORSDOF`/etc.) and rejects real
   PDB header lines OpenBabel carries through from the input file — filtered before handing off to
   Vina.
5. **A real large-assembly structure with a >1-character chain name crashed the entire full-scale
   run three hours in.** `gemmi.write_pdb()` correctly raises `RuntimeError: chain name too long
   for the PDB format: AAA` (the legacy PDB format only supports single-character chain IDs;
   modern mmCIF doesn't have that limit, and large assemblies genuinely need >26 chains) —
   `gen_pdbtools_manipulation`'s except clause only caught subprocess errors, not this, so the
   exception propagated all the way to `main()` and killed the process. Every hour of work already
   done was lost, since `train.jsonl`/`valid.jsonl`/`test.jsonl` are only written once, at the very
   end. Audited every other `_gemmi_to_pdb`/`write_pdb` call site in the file for the same gap
   (found one more, `gen_pdb2pqr_prep`) and fixed both, then added a structural backstop: `_safe_gen()`
   now wraps all ~65 generator call sites in `main()`, catching and logging any exception a
   generator doesn't already handle internally rather than letting one bad structure end the whole
   multi-hour build. A real, costly gap the round-4 architecture never surfaced at that smaller
   tool/generator count — worth remembering for any future round that adds more execution-verified
   generators touching real, messy deposited structures.

**Result: 94,376 examples** (75,502 train / 9,437 valid / 9,437 test), 2.8% rejection rate — 94% of
the 100,000 target, in line with v3/v4's landing pattern. Class balance:
file_format_literacy 25,000 (exact), database_cross_referencing 23,269, tool_calling 22,338,
experimental_method 21,875, refusal_boundary 2,000 (exact) — no class badly skewed. Zero
`_safe_gen` backstop triggers on the successful run (the two real bugs above were both fixed before
this run started, not papered over by the backstop catching them silently). Full run took ~15h
across two attempts (the fatal crash above ~3h into attempt 1, then a clean ~11h completion on
attempt 2 after the fix) — longer than round 4's ~8.5h as expected, the new molecular-dynamics/
crystallography/docking generators' PHENIX/GROMACS/ChimeraX/PyMOL per-call process-startup
overhead now dominating more than fpocket's long tail alone did in round 4. `corpus_lookup.py`'s
registry and the RAG corpus were updated with the two new command-corpus files. Full narrative,
class balance, and token stats in `data/README.md`'s v5 entry.

**Round 6 (2026-07-19): MDAnalysis/ProDy actually wired in, bio3d/R, plotly, full py3Dmol command
awareness, pandas as a taught skill, and AlphaFraud's full backfill.** Marc asked whether R,
plotly, and pandas were part of the tool ecosystem, and what else was obviously missing. Live audit
found `MDAnalysis` and `ProDy` installed since round 1 but never used in a single generator; `plotly`
and R's `bio3d` absent entirely; pandas used everywhere internally but never taught. Also asked
whether py3Dmol got the same full-command-awareness treatment PyMOL/ChimeraX did in round 5 — it
hadn't. Six new execution-verified generators (five execution-verified against real structures, one
documentation-grounded):
- `gen_mdanalysis_rmsf` — real per-residue RMSF across NMR ensemble models (conformational
  variability, distinct from round 5's crystallographic B-factor plot).
- `gen_prody_anm` — real Anisotropic Network Model normal mode analysis, predicting flexibility
  from a single structure's geometry alone (elastic network, no ensemble needed).
- `gen_bio3d_script` — real R/bio3d scripts (parsing, atom selection, B-factor extraction, normal
  mode analysis) via headless `Rscript`, the direct answer to "do we have R."
- `gen_plotly_view` — interactive Ramachandran/contact-map/B-factor charts, reusing round 5's
  exact real biotite computations under a new interactive rendering backend.
- `gen_py3dmol_command_reference` — 108 real `GLViewer` methods scraped from 3Dmol.js's own
  official API docs (`scripts/build_py3dmol_command_corpus.py`). py3Dmol's Python API is a blind
  `__getattr__` proxy with zero local introspection target (confirmed by reading its source), so
  unlike PyMOL/ChimeraX this tier is explicitly documentation-grounded, not execution-verified —
  no headless browser/JS engine exists in this project to confirm a call renders correctly.
- `gen_pandas_analysis` — real pandas `groupby`/`sort_values`/`merge` code run against real
  corpus CSVs, teaching the skill the model had only ever seen used internally.

**AlphaFraud, gated on Marc's explicit instruction: "only start once the backfill is confirmed 100%
complete."** Live SSH access (`ssh alphafraud.mdeller.com`) found `alphafraud-backfill.service`
still `active`, processing 2025-08 depositions — the public `/archive` page's 90 labels spanning
2018–2026 looked complete but wasn't. Watched the service (`systemctl is-active`, per the unit
file's own documented behavior: clean completion leaves it `inactive`, only a crash restarts it)
across ~13 real hours until it finished: "Full backfill done: 32,728 screened, 2,968 fully
analysed, 4,898 skipped."

**Two real, serious errors caught only by not trusting the first "done" signal** — Marc pushed back
twice on numbers presented as final, both times correctly:
1. My live sample during planning (raw entity counts per `/api/week/{label}`) suggested ~65,000+
   real comparisons; the actual re-pull got 6,799. First correction: the raw counts were
   screened+compared combined, not just `status=='compared'` (170 of 3,608 for one sampled week) —
   a real miscounting on my part, not a bug.
2. Marc pushed back again with AlphaFraud's own admin tally showing **15,482** "fully compared"
   entities — still 2.3x what the corrected 6,799 pull got. This one *was* a real bug: a direct
   read-only SQL query against `/opt/alphafraud/alphafraud.db` confirmed 15,482 real `compared`
   rows exist, but `/api/week/{label}` itself under-serves months that got reprocessed across
   multiple historical backfill runs (2019 almost entirely missing — e.g. 2019-04 has 931 real rows,
   the API serves 5; also 2026-01 through 2026-03) — confirmed live that the API endpoint is the
   bottleneck, not this project's download script. Fixed by killing the in-progress rebuild (only
   24 minutes in — cheap to redo), exporting all 15,482 rows directly from the database over SSH
   (bypassing the buggy endpoint entirely), and documenting the finding + the direct-SQL fallback
   recipe in `download_alphafraud.py`'s own docstring for any future re-run.

**Result: 95,884 examples** (76,708 train / 9,588 valid / 9,588 test), 2.7% rejection rate, 96% of
the 100,000 target. Class balance: file_format_literacy 25,000 (exact), experimental_method 23,430
(up from round 5's 21,875 — the richer AlphaFraud data let `gen_alphafraud_rich_comparison` hit
close to its real target for the first time in the project's history), database_cross_referencing
23,226, tool_calling 22,309, refusal_boundary 2,000 (exact). Zero `_safe_gen` backstop triggers.
Full rebuild took ~11h (clean single run, after the AlphaFraud fix above) — meaningfully less added
runtime than round 5 despite 6 more generators, confirming the plan's own prediction: none of
round 6's new tools have PHENIX/GROMACS/ChimeraX-style per-call process-startup overhead.
`corpus_lookup.py`'s registry and the RAG corpus were updated with `py3dmol_commands.csv` and the
much larger `alphafraud_comparisons.csv`. Full narrative, class balance, and token stats in
`data/README.md`'s v6 entry.

**Round 7 (2026-07-19): closing the robustness/edge-case gaps chem_sage's own eval work
surfaced.** Marc asked whether chatPDB has enough edge-case/bad-example/failure-case coverage to
make the fine-tuned model good, explicitly referencing a chem_sage learning. Reading chem_sage's
actual `eval_chem.py`/`PROJECT_PLAN.md` (not memory) found it grew from 3 to 13 metrics after early
rounds' models showed real failures — hallucinated numbers that didn't match tool output, no
refusal, repetition collapse — and chem_sage responded with dedicated training generators for
exactly those failure modes: `raft_distractor` (a correct value shown next to a wrong one, training
explicit rejection), `refusal`, and `pyexec_drill` (code must actually execute or the example is
rejected). Auditing chatPDB's 88 existing generators against this found solid coverage already
(missing-data honesty, cross-db disagreement, self-consistency checks, citation honesty, obsolete-
entry warnings, structure-prediction refusal) but three genuine gaps, each mirroring a chem_sage
lesson directly:
- **`gen_invalid_pdb_id_refusal`** — zero prior coverage of a nonexistent/invalid PDB ID.
  `validate()`'s own docstring confirmed *"there is no code path that could produce an example
  citing an ID absent from the corpus"* — every other generator samples real IDs by design, so the
  model had never once seen "the user asked about a made-up ID, here's the honest response."
  Live-verified the ~61% real-ID collision rate in the 4-character ID space (17/20 random samples
  hit real entries) before writing the retry-until-confirmed-absent generation loop.
- **`gen_distractor_value_correction`** — chatPDB's answer to `raft_distractor`. Presents a real
  value from a *different* real entry as if it belonged to the one being asked about (resolution,
  R-free, or clashscore), and trains the correction. Never an invented float — both the distractor
  and the correction are real, verifiable numbers from the corpus, just deliberately misattributed.
- **`gen_tool_failure_honesty`** — every existing `tool_calling` generator only keeps successful
  runs, so the corpus had zero examples of a tool genuinely failing and the model saying so
  honestly. Three real, execution-verified failure sub-cases: DSSP on nucleic-acid-only entries
  (live-confirmed `_run_dssp_mmcif()` returns a genuine empty result for entry `101D`), selecting a
  chain that doesn't exist in a specific real entry (PyMOL cleanly returns 0 atoms, no crash),
  and `gemmi.write_pdb()`'s real `RuntimeError` on multi-character chain IDs in large assemblies
  (the same failure mode that crashed round 5's first full run, reused here deliberately). All
  three are captured by actually running the tool against real structures, not fabricated text.

Design principle held throughout: deliberately-constructed "bad" examples still never fabricate —
every wrong value is real, just misattributed; every failure is a real tool genuinely failing.
`validate()`'s leak-detection regex (`\bnan\b`/`\bnone\b`) required care in Phase C's wording (e.g.
"empty result" rather than "returns None" for DSSP's no-protein case).

**Result: 97,272 examples** (77,818 train / 9,727 valid / 9,727 test), 2.6% rejection rate, 97% of
the 100,000 target. Class balance: file_format_literacy 25,000 (exact), experimental_method 23,520,
database_cross_referencing 23,221, tool_calling 22,611, refusal_boundary 3,000 (exact — up from
round 6's 2,000, now split across 3 generators instead of 2, confirming
`gen_invalid_pdb_id_refusal` landed at its full target share). Full clean run took ~11h24m (10:09
start to 21:33 finish), no `_safe_gen` backstop triggers, no corpus source changes so RAG was not
re-ingested this round.

### Phase 4 — QLoRA fine-tune with MLX-LM (0.5–1 day of compute)
`config/train_config.yaml` seeded from chem_sage's validated field names and values (rank, RSLoRA,
`steps_per_report == steps_per_eval` from round one — chem_sage had to learn this the hard way,
chatPDB starts correct).

Tooling audit (2026-07-19, live-checked against current `mlx-lm` docs and chem_sage's actual
scripts, not memory) found the current `mlx-lm` release has moved past several things chem_sage had
to hand-roll:
- **`--report-to wandb`** — training-metric logging is now a native `mlx_lm.lora` flag (also
  `--report-to swanlab`), no `wandb.init()` wrapper needed. Checking chem_sage's real
  `train_launch.py`/`train_qlora.py` found the wrapper was only ever planned in memory, never
  actually implemented — this closes that gap for free rather than porting dead code.
- **`--mask-prompt`** — computes loss only on the completion turn, not the system+user prompt.
  chem_sage never used this. Worth adopting: chatPDB's system prompt is long and shouldn't be
  optimised against.
- **`mx.set_wired_limit()`** (macOS 15+) — caps how much memory MLX wires so paging behaves
  sanely instead of risking the unbounded-wired-growth kernel-panic path some `mlx-lm` users have
  hit. More surgical than chem_sage's `preflight.sh` (kill background apps, hope), used alongside
  it, not instead of it — `scripts/train_launch.py` should call this ahead of
  `mx.set_cache_limit`/`mx.set_memory_limit` (chem_sage's existing memory-fraction tuning against
  `mx.device_info()["max_recommended_working_set_size"]`, still ported as-is).
- **`--grad-checkpoint`** / **`--grad-accumulation-steps`** — native fallbacks if 64 GB unified
  memory isn't enough headroom at the target rank, trading compute for memory more cleanly than
  chem_sage's only lever (reducing `--num-layers`).
- **Sequence-length check before training** — `mlx-lm`'s own docs recommend splitting long
  examples into smaller sequences to cut memory use; worth a token-length pass over
  `data/sft/*.jsonl` before Phase 4 starts, since chatPDB's tool-output code blocks can run long.
- **`asitop`/`mactop`** (third-party, `pip`/`brew` install) — live GPU/CPU/power/RAM+swap in a
  second terminal window during the run, on top of W&B's training-loop-only view. Cheap, optional,
  the thing that would have shown chem_sage's R5 swap thrashing as it happened rather than after.

`scripts/train_launch.py` also adds **checkpoint auto-resume** (scan `adapters/<name>/*_adapters.safetensors`
for the highest iter, offer a `--resume` flag wiring into `mlx_lm.lora`'s native
`--resume-adapter-file`) — something chem_sage never had from round one.

**Critical:** watch train/val loss together; climbing validation loss means stop early, same rule
chem_sage lived by.

**Phase 4 launch (2026-07-20): built, calibrated, found a real corruption bug, launched.**
Built `config/train_config.yaml`, `scripts/train_launch.py`, `scripts/preflight.sh`/`postflight.sh`,
`scripts/check_token_lengths.py`. Real measurements, not assumptions, drove the config:
- Confirmed machine is an **M1 Max, 32 GPU cores, 64 GB** (same tier chem_sage used) via
  `system_profiler` — Marc's "64GB Mac M1" shorthand undersold it.
- Real token-length pass against the actual Qwen3 tokenizer over the full 77,818-example
  `train.jsonl`: p50=581, p99=860, p99.9=1190, max=1973. `max_seq_length: 2048` (later tightened
  to 1536, see below) covers effectively all real data.
- Started conservative on LoRA config: **rank=32, scale=64** (chem_sage's R4-validated baseline,
  not their more-tuned R5 rank=64/scale=90) — chatPDB's first round, no tuning history yet to
  justify the more aggressive config on a different model and ~5x larger dataset.
- First clean calibration (20 iters): **77.00 s/iter**, peak memory 27.9 GB against a 46.7 GB
  limit — looked like ~18 GB of free headroom.

**A genuine bug, not just a disappointing result:** tried stacking three speed levers
(`grad_checkpoint: false`, `max_seq_length: 1536`, `batch_size: 8→4`) based on that headroom.
The combined run produced garbage: learning-rate values like `6.3e+32` (a value that's purely a
function of iteration number and should never vary like that), tokens/sec in the tens of millions
and sometimes negative, one loss value of `-4.04e+30`, and peak memory climbing to **209.76 GB** —
over 3x the physical machine — without crashing. Isolated each lever individually against the known-
good baseline to find the cause:
- `grad_checkpoint: false` alone reproduced the exact same corruption (peak mem to 104 GB, real
  swap usage). **MLX/Metal on this hardware/mlx-lm version does not fail cleanly when real memory
  usage exceeds the recommended working set with checkpointing off — it silently corrupts
  computation instead of erroring.** `grad_checkpoint: true` is load-bearing at this rank/layer
  count on this machine, not just cheap insurance; the "~18 GB of headroom" estimate from peak-
  memory-at-checkpointed was not a safe signal for how much margin exists with checkpointing
  removed. Reverted, kept `true`.
- `max_seq_length: 1536` alone was clean (same loss trajectory, same 27.9 GB peak, 76.03 s/iter —
  no measurable difference from baseline in this sample, since most real batches are far shorter
  than either cap). Kept — safe, free, marginal benefit in the long tail.
- `batch_size: 8` alone was also clean (peak mem only 37.1 GB) but gave **no net wall-clock
  benefit** — per-iter time roughly doubled (162.38 s/iter) to match the doubled batch, i.e. the
  same total compute in fewer, bigger steps. Reverted to 4.

Net result: none of the three speed tricks delivered a real win once properly isolated — the
validated rate stayed at ~76–77 s/iter. Reaching 30% epoch coverage at that rate would need
~5,834 iters (~124h / ~5.2 days). Presented the real trade-off to Marc, who chose **1420 iters
(~30h)**, matching chem_sage's own largest single-round precedent (R5) rather than chasing epoch
coverage on a dataset 5x the size — covers ~7.3% of one epoch (~5,680 of 77,818 examples), a
deliberately scoped first round to get a real checkpoint to evaluate before committing further.

W&B wasn't logged in on this machine; `wandb login` needs an interactive prompt that doesn't work
over a non-TTY shell, so Marc ran `wandb login <key>` himself (the key-argument form, W&B's
supported non-interactive path) via the `!` prefix. Verified working with a real 5-iter calibration
run before committing to the full launch — real run appeared at `wandb.ai/dellboy-none/chatpdb`.

**First launch 2026-07-20 17:41 — killed and restarted at 20:10 after a real cost miscalculation.**
The first launch used `steps_per_report: steps_per_eval: 50` (naively copied from chem_sage's
"align reporting" convention — a lesson about keeping a results *table* readable, not about live-
monitoring cost). Every calibration run had used `--val-batches 0` to isolate pure training-step
timing, so the ~30h estimate never accounted for real validation cost. Once running for real,
`Iter 1: Val loss 2.544, Val took 2128.993s` (35.5 min for `val_batches: 50`) revealed the actual
cost: ~30 eval events across the run × 35.5 min ≈ **17.7h of validation alone**, on top of ~30.4h
of real training — a real total of ~48h, not the ~30h Marc had approved. Caught only because Marc
asked "what would the best step size be" after noticing no loss curves in W&B (itself just the
sparse `steps_per_report=50` cadence working as configured, not a bug — but the question surfaced
the deeper cost error underneath it).

Reworked the reporting/eval split: `steps_per_report: 1` (train-loss reporting is ~free — printing
a number already computed during the normal step — no reason to hold it back), `steps_per_eval: 10`
/ `val_batches: 5` (measured 42.58s/val-batch; ~143 eval events × ~3.55 min ≈ 8.5h validation, real
total ~38.9h — Marc chose tighter early-stopping resolution over minimising wall-clock further).
Killed the first run (PID 47198/47201, ~2.5h in, before any real training-loss data point had even
logged) and relaunched clean from iteration 0 rather than resume, since the sunk cost was small and
a mid-run reporting-cadence change isn't cleanly resumable.

**Relaunched 2026-07-20 20:10**, log at `/tmp/chatpdb_train_v1.log`, W&B run
`wandb.ai/dellboy-none/chatpdb/runs/8bnvprdf`. Final config: rank=32/scale=64, num_layers=32,
max_seq_length=1536, mask_prompt=true, grad_checkpoint=true, batch_size=4, iters=1420,
steps_per_report=1, steps_per_eval=10, val_batches=5, save_every=100 (~14 checkpoints across the
run — resume-safety margin for an unattended run, see `feedback_iter_offset` on R3's battery-
interruption lesson). Expected completion ~2026-07-22 ~11:00 (~38.9h from relaunch).

**Two more mid-run incidents, 2026-07-21/22:**
- **A real bug in `mlx_lm` itself, fixed locally.** `trainer.py`'s val-loss dict logged
  `"iteration": it - 1` while the train-loss dict logged `"iteration": it` — with
  `steps_per_report=1`, val's off-by-one always collided with the *previous* iteration's already-
  logged W&B step, so `val_loss` showed as real in only ~1-in-10 rows and `NaN` elsewhere,
  preventing W&B from rendering a clean curve. Patched the installed
  `mlx_lm/tuner/trainer.py` (`it - 1` → `it`), killed and relaunched clean from iteration 0 a
  second time (W&B run `390dnfgb`) — confirmed fixed via the W&B API directly (train_loss and
  val_loss now share the same step). Also traced a recurring multi-hour CPU-starvation pattern
  (Spotlight's daemon family: `corespotlightd`/`spotlightknowledged`/`mediaanalysisd`/
  `duetexpertd`) to chatPDB's own 256,444-file `data/structures_all/` corpus (chem_sage's much
  smaller corpus, 79 files, never triggered this) — full detail in `feedback_preflight`.
- **Machine crashed at true iter ~630** (last checkpoint: iter 600, 22:20 UTC+1). Resumed via
  `mlx_lm.lora --resume-adapter-file` — confirmed this only reloads weights, not the iteration
  counter/optimizer state/LR schedule position (`Iter 1: Val loss 0.137` correctly matched the
  real iter-600 model, but `Learning Rate` logged as `0.000e+00`, schedule restarted from warmup)
  — the exact same limitation that bit chem_sage's own R3 resume (`feedback_iter_offset`).
  Following chem_sage's precedent rather than fighting the tool: killed the first (wrongly-
  configured, `iters` still 1420) resume attempt immediately, set `iters: 820` (remaining budget:
  1420 − 600) so total real work still matches the original 1420, relaunched clean (W&B run
  `wpv76v0y`). **Logged iter N in this final segment of the run = true iter (N + 600)** — apply
  this offset in the final results table below.

**Stopped early at true iter ~803 (2026-07-22 17:09), by Marc's explicit call after reviewing the
real val-loss trend.** Pulled the full history from both W&B runs (390dnfgb pre-crash,
wpv76v0y post-crash) rather than relying on individual noisy points, and computed block averages
across ~100-true-iter windows:

| True iter range | Avg val loss |
|---|---|
| 1-100 | 1.14 |
| 100-200 | 0.265 |
| 200-300 | **0.202** (best block) |
| 300-400 | 0.218 |
| 400-500 | 0.243 |
| 500-600 | 0.222 |
| 600-700 | 0.273 |
| 700-800 | 0.304 |

Real, substantial improvement happened in the first ~200-300 iterations; from roughly true iter 300
onward, val loss was flat-to-noisy in the 0.2-0.3 range for ~500 further iterations (35% of the
budget) with no further downward trend — a genuine plateau, not overfitting (no sustained climb
either). Continuing to the originally-planned true iter 1420 was judged unlikely to meaningfully
improve the model for the remaining ~10-12h of compute. **Best available checkpoint by val loss:
true iter 600 (val=0.176)** — the single best individual point (0.105 at true iter 530) has no
saved checkpoint at that exact iteration, since `save_every=100` only saves at multiples of 100.

**Final checkpoint inventory** (`adapters/chatpdb_32b_v1_lora/`, renamed to `true_iter_NNNN_
adapters.safetensors` after the run to eliminate ambiguity from the post-crash counter restart —
the raw `0000100`/`0000200` filenames briefly held true iters 700/800 due to the resumed run's own
counter colliding with the original filenames, see the crash-recovery note above):

| True iter | Val loss | File |
|---|---|---|
| 100 | 0.332 | **lost** — overwritten before the precrash backup was made |
| 200 | 0.184 | `true_iter_0000200_adapters.safetensors` |
| 300 | 0.327 | `true_iter_0000300_adapters.safetensors` |
| 400 | 0.313 | `true_iter_0000400_adapters.safetensors` |
| 500 | 0.293 | `true_iter_0000500_adapters.safetensors` |
| **600** | **0.176 (best available)** | `true_iter_0000600_adapters.safetensors` |
| 700 | 0.396 | `true_iter_0000700_adapters.safetensors` |
| 800 (final) | 0.211 | `true_iter_0000800_adapters.safetensors` / `adapters.safetensors` |

Recommend fusing from `true_iter_0000600_adapters.safetensors` (best available val loss) rather
than the final checkpoint, for Phase 5.

### Phase 5 — Fuse and serve (0.5 day)
Route A, same as chem_sage: `mlx_lm.fuse` → `mlx_lm.server --port 8080`, OpenAI-compatible endpoint.

### Phase 6 — Close the hybrid loop (0.5 day)
Wire `rag/tool_exec.py`: Biopython sandbox first (restricted subprocess, no filesystem/network), add
gemmi/DSSP once stable, PyMOL last — the same staged-caution order chem_sage applied to RDKit→PyMOL.

**Done, 2026-07-22.** Directly motivated by a real gap Phase 5 testing found: `mlx_lm.generate`
alone emits structurally-correct real tool-call code, but nothing executes it at raw-generation
time, so the model's own stated summary numbers can be flatly wrong even when the code is right
(confirmed live: asked the fused model to DSSP-summarise entry 4RE2, it emitted correct Biopython/
DSSP code but claimed "14 helical residues, 1 strand" — the real computed result is H=183, E=71 out
of 482 residues). `rag/tool_exec.py` is the fix: detects `python` code blocks in a model response,
copies only the specific real structure file(s) the code references (by name, out of
`data/structures_all/`) into an isolated temp directory, and actually runs the block there, so the
assistant can be grounded in the real computed value rather than the raw completion's guess.

Ported chem_sage's real `tool_exec.py` (static import/network/filesystem blocklist, clean env,
20s timeout, isolated per-run temp dir) and adapted the two real differences: (1) chatPDB's blocks
reference real structure files by relative filename (`'4re2.cif'`), which chem_sage's pure-SMILES
RDKit blocks never needed — added a `.cif`-filename scanner that copies matching real files in
before execution, and blocked bare `open()` so the only file-read path is through the file that was
explicitly copied in, not an arbitrary read elsewhere; (2) staged to Biopython only for this first
pass (matching chem_sage's own RDKit→PyMOL caution) — gemmi/DSSP/PyMOL/ChimeraX blocks are detected
and explicitly flagged as not-yet-enabled rather than silently skipped or (worse) trusted, so a
DSSP-summarising response like the one above is now clearly marked "run manually" instead of
silently accepted. Verified live: a real Biopython block (chain/atom count) executes and returns
the correct real numbers; the real hallucinated-DSSP block from the Phase 5 test is correctly
flagged, not executed; a network-access block is correctly blocked.

The full `chat.py` interactive CLI (streaming generation + `tool_exec` output panel wired together)
is Phase 8's scope, not this phase's — `rag/tool_exec.py` alone is what Phase 6 asked for.

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
HTML + Markdown reports with a Plotly val-loss curve (chem_sage's actual default — its own
`_loss_curve_svg()` turned out to be dead-code fallback-only, not the live default the earlier plan
text above implied).

**Shipped, real-verified:** `eval/metrics.py` (shared metric logic — unlike chem_sage, which keeps
two diverging copies across its two eval files), `eval/eval_pdb.py`, `eval/compare/eval_compare.py`
+ `models.yaml`, `eval/compare/eval_rescore.py` (re-scores cached results without re-querying the
model). Two real upgrades over chem_sage's own precedent: **PDB ID validity now checks real corpus
membership** (chem_sage's own `pdb_id_validity` is format-only regex, never real membership — chatPDB
already has the full ID set loaded via `load_corpus()`, so use it), and **cross-reference accuracy is
new** (chem_sage has no equivalent — checks stated PDB↔UniProt/CATH/EC mappings against the real
SIFTS tables). Numerical fidelity checks stated resolution/R-free/chain-instance-counts against
`pdb_entries_enriched.csv` directly (a real, grounded ground truth) rather than requiring a live
recompute inside the metric itself.

Live-verified end to end against the real `models/chatpdb_32b_v1` server (`--n 20` / `--limit 10`
smoke tests, 2026-07-22): PDB ID validity 89–100%, cross-reference accuracy 50%, refusal accuracy
100%, degeneration-free 100%, numerical fidelity 33–40%. One real bug found and fixed during this
verification: `tool_executability` was penalising a model-emitted DSSP block (`DSSP(...,
dssp='mkdssp')`) as a failure — DSSP is explicitly staged "not yet enabled" per Phase 6's own
`rag/tool_exec.py::execute()`, but the metric function was calling `run_sandboxed()` directly and
skipping that not-yet-enabled filter. Fixed by reusing `_not_yet_enabled_reason()` the same way
`execute()` does. Also found: this installed `mlx_lm.server` (0.31.3) requires the request's
`"model"` field to exactly match the server's resolved absolute model path, not an arbitrary display
name (returns a 404 "Repository Not Found" HF-hub-resolution error otherwise) — `eval_pdb.py`'s
`--model` default now resolves to the real absolute path; `eval_compare.py` already did this
correctly via its per-model `model_path` construction.

A full `--n 200` `eval_pdb.py` pass (real numbers, not just the smoke-test sample above) is the
natural next step whenever the model is evaluated for real — not run as part of this build pass,
since it's a genuine multi-hour-scale live-inference commitment against a 32B model.

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
│   ├── download_rcsb.py                # RCSB entries + CCD + SIFTS (Phase 2)
│   ├── download_cath.py                # CATH classification hierarchy
│   ├── download_interpro.py            # InterPro entry dictionary
│   ├── download_pharos.py              # target druggability, joined via UniProt
│   ├── download_twilight.py            # per-ligand density-quality (RSCC/OWAB)
│   ├── download_uniprot.py             # Swiss-Prot entries + keyword vocabulary
│   ├── download_structure_pool.py      # 820-file PDB-format sample (Phase 3 round 1)
│   ├── download_all_structures.py      # all 256,444 entries as mmCIF, 353 GB (Phase 3 round 2)
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
