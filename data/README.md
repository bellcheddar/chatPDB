# chatPDB dataset layout

MLX-LM expects a data directory containing `train.jsonl` and `valid.jsonl` (note: `valid`, not
`val`). `test.jsonl` is held out and used only by `eval/eval_pdb.py`.

```
data/
├── corpus/            # RAG sources: RCSB/SIFTS/CATH/InterPro/Pharos/TWILIGHT/UniProt/AlphaFold/
│                        # BindingDB/wwPDB-validation/STRING/PDB-REDO/EMDB/SCOP2/MobiDB/OPM/
│                        # clusters/obsolete/AlphaFraud/citations/disease_context/pymol/chimerax
│                        # (gitignored, Phase 2 + corpus expansion + round 3/4/5 sources)
├── cache/crystallography/ # round 5: cached real MTZ conversions (RCSB structure-factor pulls,
│                        # cif2mtz/ctruncate output) keyed by PDB ID, built once and reused across
│                        # gen_mtz_manipulation/gen_ccp4_refmac_script/gen_phenix_refine_script
├── structures/         # 820-file PDB-format sample, superseded by structures_all/ below but kept
│                        # (small, fast for smoke-testing) — gitignored, scripts/download_structure_pool.py
├── structures_all/      # ALL 256,444 entries as native mmCIF, 353 GB (gitignored, "let's add ALL PDB
│                         # files" round, 2026-07-16) — scripts/download_all_structures.py
└── sft/               # SFT data (MLX --data directory), v5 populated 2026-07-18
    ├── train.jsonl     # 75,502 examples
    ├── valid.jsonl     # 9,437 examples
    └── test.jsonl      # 9,437 examples, frozen, eval only
```

## SFT format

One JSON object per line, chat format (`messages`), matching chem_sage's schema exactly so
`mlx_lm.lora` can consume it directly. Every assistant turn that states a structural fact does so by
emitting a tool (Biopython/gemmi/DSSP/PyMOL) code block, never by asserting a number from memory.

### Worked example (real, pulled from `train.jsonl` v2 — DSSP execution-verified against the real
native mmCIF coordinates for 4RE2, one of `data/structures_all/`'s 256,444 downloaded files; system
prompt shortened here for readability, full text is in `config/system_prompt.txt`)

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are chatPDB, a protein-structure-literate research assistant... [full text in config/system_prompt.txt]"
    },
    {
      "role": "user",
      "content": "Write Biopython/DSSP code to assign secondary structure to PDB entry 4RE2 (mmCIF file `4re2.cif`) and summarise the helix/strand content."
    },
    {
      "role": "assistant",
      "content": "```python\nfrom Bio.PDB import MMCIFParser\nfrom Bio.PDB.DSSP import DSSP\n\nstructure = MMCIFParser(QUIET=True).get_structure('4RE2', '4re2.cif')\nmodel = structure[0]\ndssp = DSSP(model, '4re2.cif', dssp='mkdssp', file_type='mmCIF')\nss_counts = {}\nfor key in dssp.keys():\n    ss = dssp[key][2]\n    ss_counts[ss] = ss_counts.get(ss, 0) + 1\nprint(ss_counts)\n```\n\nRunning DSSP on the real deposited coordinates for 4RE2 gives 482 assigned residues: 216 in helix (H/G/I), 76 in strand (E/B) — a predominantly helical structure."
    }
  ]
}
```

## Construction rules (enforced by `scripts/build_dataset.py`)

1. **Ground truth first.** Two tiers, both real: (a) templated around a value already fetched and
   spot-checked against a live API in Phase 2 / the corpus expansion round (resolution, R-free,
   UniProt function, CATH fold, EC number, Pharos TDL, TWILIGHT RSCC, unit cell/space group,
   crystallization conditions, primary citation, source organism...) — this is what `pdb_id`,
   `comp_id`, `accession` etc. are always sampled *from* a corpus DataFrame, never a literal; or
   (b) computed live by actually running Biopython/gemmi/DSSP against a real file in
   `data/structures_all/` (secondary structure assignment has no metadata shortcut — it has to be
   computed). Never hand-authored.
2. **Validate on write.** Every generated Python code block must actually `compile()`; examples
   under a length floor or containing a literal `nan`/`none` leak from an unhandled missing field
   are dropped. ID-resolution is guaranteed by construction (see above), not re-checked by regex.
3. **Balance the four behaviour classes equally** (see `PROJECT_PLAN.md` Phase 3): file/format
   literacy, experimental-method interpretation, tool-calling, database cross-referencing — plus a
   small supplementary `refusal_boundary` class (chatPDB is not a structure predictor).
4. **Never train toward structure prediction.** The `refusal_boundary` class trains the model to
   decline exactly this; no generator asks the model to predict a fold or coordinates from sequence
   alone as if it were in scope.
5. **Split** 80/10/10 train/valid/test; the test split is frozen and never seen during training.

## Dataset versions

| Version | Round | Examples | Rejection rate | Command | Notes |
|---|---|---|---|---|---|
| v1 | R1 | 45,502 (36,402 train) | 2.2% | `build_dataset.py --n 50000 --seed 51` | Target was 50,000; `tool_calling`'s two execution-verified generators (DSSP, NMR model count) were hard-capped by an 820-file structure pool, not padded. Superseded by v2. |
| v2 | R2 | 50,233 (40,187 train) | 1.5% | `build_dataset.py --n 50000 --seed 51` (after "let's add ALL PDB files and ALL fields" — full mmCIF pool + expanded RCSB metadata) | Full 50,000 target hit; `tool_calling` reaches its complete 12,500 target for the first time (previously capped at 8,010) now that the structure pool is 256,444 files instead of 820. Added 4 new generators (citation, unit cell/space group, crystallization conditions, organism/taxonomy) from the expanded metadata pull. Superseded by v3. |
| v3 | R3 | 97,241 (77,793 train) | 1.2% | `build_dataset.py --n 100000 --seed 51` (after "get AlphaFolddb, PDBbind/BindingDB, wwPDB validation reports, and STRING... make it all thorough") | 4 new corpus sources + ~19 new generators across multi-hop chains, bidirectional traversal, cross-database disagreement, missing-data honesty, comparative examples, tool-chaining, and RAG-shaped synthesis. Target doubled to 100,000 to match the larger generator roster; landed at 97,241 (97% of target) with the lowest rejection rate yet. Superseded by v4. |
| v4 | R4 | 93,725 (74,981 train) | 2.7% | `build_dataset.py --n 100000 --seed 51` (after a Fable 5 brainstorm on remaining expert-depth gaps, implemented in full) | 7 new corpus sources + AlphaFraud (staged) + independent CrossRef/PubMed citation verification + 5 new local tool-exec integrations (FreeSASA/fpocket/Foldseek/US-align/PLIP) + ~34 new generators. Landed at 93,725 (94% of target); rejection rate ticked up from v3's 1.2% since several round-4 generators haven't had multiple tuning passes yet. Full run took ~8.5h, dominated by the new execution-verified tool-calling generators' long subprocess tail (fpocket especially). Superseded by v5. |
| v5 | R5 | 94,376 (75,502 train) | 2.8% | `build_dataset.py --n 100000 --seed 51` (a full visualization/rendering/simulation tool review — full PyMOL/ChimeraX command awareness, sequence alignment, WebLogo, biotite plots, py3Dmol, pdb-tools, a 2D topology schematic, MD (OpenMM/GROMACS), crystallography (CCP4/PHENIX), AutoDock Vina docking) | 21 new execution-verified generators (see `PROJECT_PLAN.md`'s round 5 section for the full per-generator writeup). Landed at 94,376 (94% of target). Two real bugs caught only at full scale, both fixed before the numbers above: (1) a legitimate large-assembly structure with a >1-character chain name crashed `gemmi.write_pdb()` with a `RuntimeError` three hours into the first full run — `gen_pdbtools_manipulation`'s except clause only caught subprocess errors, not this, and lost all accumulated work since output is only written once at the end; (2) added `_safe_gen()`, a backstop wrapper around every one of the ~65 generator call sites in `main()`, so no single generator's unexpected exception can crash the whole multi-hour build again — a real robustness gap the round-4 architecture never needed to close at this scale/tool-count before. Full run took ~15h (22:26 run 1's fatal crash + investigation + fix + ~11h clean re-run), the new molecular-dynamics/crystallography/docking generators' PHENIX/GROMACS/ChimeraX/PyMOL process-startup overhead now dominating more than fpocket alone did in v4. |

v4 class balance: `file_format_literacy` 25,000, `database_cross_referencing` 23,201,
`experimental_method` 21,875, `tool_calling` 21,653, `refusal_boundary` 2,000 (supplementary, not
counted toward the four-class equal split — includes the new mutation/variant-effect refusal
variant alongside the original bare-structure-prediction one).

v4 token length (Qwen3-32B-4bit tokenizer, full chat-template-rendered example, n=2,000 sample):
p50=582, p90=683, p95=729, p99=891, max=1,431 — comfortable margin under any `max_seq_length`
chem_sage used (2048–3072); no examples needed truncation.

v3 token length (Qwen3-32B-4bit tokenizer, full chat-template-rendered example, n=2,000 sample):
p50=578, p90=704, p95=734, p99=940, max=1,969 — comfortable margin under any `max_seq_length`
chem_sage used (2048–3072); no examples needed truncation. (For reference, v2: p50=550, p90=620,
p95=638, p99=789, max=1,503 — v3's examples run somewhat longer on average, mainly the multi-hop-chain
and RAG-synthesis generators, which cite several sources per answer.)

**Comparison to chem_sage:** chem_sage grew its SFT dataset over 5 rounds — 1,500 (R1) → 5,000 (R3)
→ 8,000 (R4) → 20,000 (R5) examples. chatPDB's round 2 (50,233) is more than double chem_sage's
largest round in two generation passes rather than five — a consequence of chatPDB's corpus being
large and richly structured enough (256k enriched entries with 30+ fields each, 975k SIFTS-UniProt
rows, 870k TWILIGHT ligand-QC rows, 256,444 real downloaded structure files...) to parameterise
thousands of unique, individually-grounded instances per generator, the same scaling trick chem_sage
used per-molecule with RDKit.

**What changed between v1 and v2:** Marc asked to add *all* PDB structure files (not a stratified
820-file sample) and *all* available RCSB fields (not just the resolution/method/R-free subset).
This meant: (1) downloading mmCIF for all 256,448 entries (`scripts/download_all_structures.py`,
353 GB, ~5.5h at 64 parallel workers — files.rcsb.org's static CDN tolerates far more concurrency
than the ~10 req/s guideline documented for the Search/Data API); (2) expanding the RCSB GraphQL
entry-enrichment query with unit cell, space group, crystallization conditions, diffraction
wavelength, primary citation (title/journal/year/DOI/PubMed), primary sequence + length, source
organism/taxonomy, and assembly count (all verified live before writing the bigger query, same
discipline as every other corpus pull); (3) removing `tool_calling`'s DSSP/NMR-model-count bottleneck
now that the structure pool is 300x larger; (4) four new generators exploiting the new fields.

A second connection-pooling stall (same failure mode as `download_interpro.py`'s, see PROJECT_PLAN.md
Phase 2) hit the metadata re-enrichment mid-run — 5h22m elapsed, 42 seconds of actual CPU time,
against `data.rcsb.org` this time rather than EBI. Fixed the same way: dropped `download_rcsb.py`'s
module-level `requests.Session()` for one-off `Connection: close` requests, plus added checkpointing
every 400 batches. Also found that mkdssp can still raise `Duplicate Key violation` on a small
fraction of *native* mmCIF files straight from RCSB (not just legacy-PDB-converted ones) — the
generator's existing oversample-and-skip tolerance absorbed this without needing a fix; DSSP still
hit its full 2,500-example sub-target.

**Round 3 (2026-07-16): "get AlphaFolddb, PDBbind/BindingDB, wwPDB validation reports, and STRING...
make it all thorough as I want the best possible data before we start training."** Four new corpus
sources plus six requested "techniques" layered on top: deeper multi-hop chains, bidirectional
traversal, cross-database disagreement, comparative examples, tool-chaining skills, and RAG-shaped
synthesis examples. This is explicitly pre-Phase-4 work — Phase 4 (QLoRA fine-tune) has still not
started.

**New corpus sources**, all downloaded and verified live before building generators against them:
- `scripts/download_alphafold.py` — AlphaFold DB predicted structures for the top 15,000 UniProt
  accessions by PDB cross-reference count (not all ~214M AlphaFold predictions — scoped to the
  population this corpus can actually link to real experimental structures). **13,754 accessions**
  pulled (some of the top 15,000 have no AlphaFold prediction available).
- `scripts/download_bindingdb.py` — BindingDB's full bulk TSV (~9 GB, 3.2M+ measurements), streamed
  and filtered down to rows whose PDB cross-reference field matches this corpus. **113,366 rows**
  matched out of 3,228,554 scanned.
- `scripts/download_wwpdb_validation.py` — PDBe's validation-report percentiles (Ramachandran
  outliers, rotamer outliers, clashscore) for the full 256,448-entry corpus, 24 concurrent workers
  (empirically tuned: 8→16→32 workers tested, ~98 req/s at 32, settled on 24 as a respectful pace).
  **252,756 entries** returned validation data (3,672 had none available — mostly NMR/very old
  entries the automated pipeline doesn't score the same way).
- `scripts/download_string.py` — STRING protein-protein interaction edges, scoped to the top 3,000
  human PDB-cross-referenced UniProt accessions (STRING's coverage is excellent for model organisms,
  patchy elsewhere — covering human well beats covering everything thinly). **23,377 edges.**

**Two real bugs in `download_string.py`**, both caught before generators were built against the
data, not after: (1) an `IndexError` when an accession's `get_string_ids` response had only a header
row (no match in STRING for that species) — fixed with a length check; (2) a more consequential
correctness bug found only by manually inspecting sample output, not by an exception — STRING's
`network` endpoint returns edges across a small interconnected *neighbourhood*, not a star graph
centered on the query protein, so most raw edges didn't even mention the queried protein (a B2M
query returned an "HLA-F, LILRB1" edge, neither of which is B2M). Fixed by resolving the query
accession's STRING preferred name first, then filtering to edges that actually touch it.

**~19 new generator functions** across all four classes implementing the requested techniques:
- *Tool-chaining* (`tool_calling`): `gen_tool_chain_structure_analysis` (a single script combining
  Biopython parsing and DSSP secondary-structure assignment, chained off one parsed structure object
  rather than one call per fact) and `gen_tool_chain_lookup` (a real sequential API-lookup script:
  SIFTS PDB→UniProt, then that UniProt accession chained into a Pharos druggability query).
- *Comparative + new-source* (`experimental_method`): `gen_alphafold_vs_experimental` (the actual
  predicted-vs-experimental contrast chatPDB's design thesis is built around, now backed by real
  data on both sides — AlphaFold confidence chained through SIFTS to a real PDB entry's
  resolution/R-free — instead of just a refusal), `gen_validation_geometry` (real Ramachandran/
  rotamer/clashscore data with percentile context), `gen_multihop_structure_quality_full` (crystallo-
  graphic fit and model geometry combined into one holistic assessment, explicit that they're
  independent axes).
- *Single-source new-data* (`database_cross_referencing`): `gen_binding_affinity` (BindingDB
  potency), `gen_string_interactors` (STRING partners, aggregated per protein), `gen_alphafold_confidence`
  (per-region pLDDT breakdown, standalone).
- *Bidirectional traversal*: `gen_uniprot_to_pdb_aggregate` and `gen_ligand_to_pdb_aggregate` — the
  reverse direction from every existing generator (which PDB entries does *this* UniProt accession
  or *this* ligand appear in, not the other way round).
- *Deeper multi-hop chains*: `gen_multihop_target_context` (PDB → SIFTS → Pharos → BindingDB, 4 hops),
  `gen_multihop_ligand_quality_chain` (PDB → CCD/TWILIGHT pose-fit joined against BindingDB potency
  for the same ligand), `gen_multihop_fold_function` (CATH fold classification joined with UniProt
  function, explicit that fold correlates with but doesn't determine function).
- *Cross-database disagreement / honesty*: `gen_cross_db_disagreement` (RCSB's and UniProt's organism
  fields for the same chain don't always literally agree — teaches reporting both rather than
  silently picking one) and `gen_missing_data_honesty` (entries with a genuinely absent R-free or
  validation record get the correct "not available, here's why" answer, not a fabricated number —
  the single most important refusal-adjacent behaviour for a database-grounded assistant).
- *Comparative*: `gen_compare_two_entries` (two different PDB entries of the same UniProt accession,
  compared head-to-head — "which structure should I use" needs two rows in context at once, which no
  single-entry generator can answer).
- *RAG-shaped synthesis*: `gen_rag_synthesis` — presents a prompt formatted like real retrieved RAG
  context (numbered, source-tagged chunks from up to 4 different corpus files, shuffled) and requires
  a synthesized, per-fact-cited answer. This trains the model for the shape it will actually see at
  inference time behind the retriever, not just bare questions about pre-selected facts.

**Bugs caught by smoke-testing before the full run** (2,000-example test runs at each fix, per this
project's standing discipline — see `PROJECT_PLAN.md` §9): (1) `gen_validation_geometry` and
`gen_multihop_structure_quality_full` only checked `clashscore.notna()`, missing that
`percent_rama_outliers`/`percent_rota_outliers` can be independently NaN and that PDBe uses
`clashscore == -1` as a sentinel for "not computed" — both now filter all three fields properly;
(2) a pre-existing gap (present since round 1) in `gen_twilight_ligand_fit` and inherited by the new
`gen_multihop_ligand_quality_chain`: TWILIGHT's `LigNm` field can be NaN, which rendered as the
literal string "ligand nan bound in PDB entry..." in the generated *question* — `validate()` only
ever checked the assistant's answer text, never the user's question, so this leaked straight through;
both the generators (now filter `LigNm.notna()`) and `validate()` (now checks both message roles)
were fixed; (3) the single most consequential bug: `gen_multihop_target_context`'s template literally
contained the English phrase "none of which is recoverable" — `validate()`'s NaN/None leak check
(a whitespace-split word match) correctly flagged every "none" appearing anywhere as a suspected
leak, meaning **100% of this generator's output was silently rejected regardless of the data**,
independent of a second, real, ~51%-of-rows issue where Pharos's `family` field is genuinely absent
and was rendering as a literal "family nan" (present in `gen_pharos_druggability` since round 1 too,
newly inherited by two round-3 generators). Fixed the template wording, added `pd.notna()` guards for
`family` in all three affected generators, and hardened `validate()`'s leak check to a word-boundary
regex (`\bnan\b`/`\bnone\b`) so "nanomolar" and similar real words don't false-positive while
"nan%"/"nan,"/"(nan)" shapes still get caught.

**Result: 97,241 examples** (77,793 train / 9,724 valid / 9,724 test) — up from v2's 50,233, on a
target doubled to 100,000 to match the larger generator roster, landing at 97% of target with the
lowest rejection rate of any round (1.2%, down from v2's 1.5%). `corpus_lookup.py`'s registry and the
RAG corpus (round 3: 22 files / 79,235 chunks, up from 18 / 65,811) were updated with all four new
source files.

**Round 4 (2026-07-17): Fable 5 expert-depth brainstorm, implemented in full.** Marc asked Fable 5
what's still missing to make chatPDB a genuine protein-structure expert, then asked for all of it:
seven new corpus sources (PDB-REDO, EMDB, SCOP2, MobiDB, OPM, sequence-redundancy clusters,
obsolete-entry mapping), a staged AlphaFraud integration, independent CrossRef/PubMed citation
verification, five new local tool-exec integrations (FreeSASA, fpocket, Foldseek, US-align, PLIP),
and ~34 new generators. Full source-by-source detail is in `PROJECT_PLAN.md`'s round 4 section;
this entry focuses on the dataset-generation outcome.

**Construction rule additions this round:**
- **House "structure report card" format** (`_structure_report_card()`): a consistent,
  scannable template (resolution + bucket, R-free + gap, clashscore + percentile, Rama/rotamer
  outliers, one-line verdict) reused across new `experimental_method` generators — the single
  cheapest, highest perceived-expertise change of the round.
- **Calibrated citation trust, not blind trust.** Every citation-bearing example now routes through
  `scripts/verify_citations.py`'s independently-verified bucket (verified / mismatched /
  unresolvable) rather than repeating the deposited DOI/title string as fact.
- **Bidirectional and multi-hop patterns extended further**: family/homolog-level reasoning (CATH
  superfamily → member set, not just single-entry facts), structural biography (a UniProt's full
  PDB timeline), and assembly biography (real FreeSASA-backed interface area, not just the
  `assembly_count` metadata field).

**Six real bugs found and fixed** (three of them serious enough to have corrupted data if shipped
unfixed) — full detail in `PROJECT_PLAN.md`'s round 4 section, summarized here:
1. PDB-REDO's rsync pull silently stalled (ESTABLISHED connection, ~0 CPU, zero progress) — switched
   to HTTP per-entry fetches.
2. AlphaFraud's first "timeout" fix didn't free the stuck worker thread, so it wasn't actually a
   fix — every subsequent request queued behind the stuck one forever. Real fix: a throwaway
   executor per request, abandoned (not joined) on timeout.
3. EMDB's search-based pull had no real termination condition (returned far more documents than the
   real 59,608-entry total, unbounded duplicates) — switched to the bulk holdings index + per-entry
   REST calls.
4. `verify_citations.py` crashed on a NaN `citation_year` — the same "pandas NaN is not `None`" bug
   class this project has hit before, in a file that hadn't yet absorbed the lesson. This was also
   the real explanation for a run that looked "stuck" at one checkpoint for over an hour: it had
   silently crashed early, and monitoring kept re-reading a stale log.
5. **CrossRef rate-limiting (HTTP 429) was silently misclassified as "this DOI doesn't exist"** —
   caught only by noticing an implausible ~44% unresolvable rate mid-run. Would have taught the
   model that huge fractions of real PDB citations are fake had it shipped. Fixed with real
   retry-with-backoff, a separate `rate_limited` bucket, and empirically-tuned concurrency (true
   post-fix unresolvable rate: 0.1%).
6. OPM's original sequential puller would have taken ~8 hours for 15,014 entries — switched to
   concurrent fetching, finished in 24 minutes.

**Timeline:** the full generation run took **~8.5 hours**, confirmed genuine (not a hang) via
repeated `sample`-based process profiling and direct child-process inspection catching real
`fpocket`/`mkdssp`/`foldseek`/`USalign` subprocesses actively executing. The new execution-verified
tool-calling generators — especially fpocket, whose exhaustive Voronoi-based pocket search has a
real long tail on larger structures — dominate this round's runtime far more than round 3's
DSSP-only `tool_calling` class did. Flagged the realistic timeline mid-run and confirmed with Marc
to let it run to completion rather than truncate or reduce scope.

**Result: 93,725 examples** (74,981 train / 9,372 valid / 9,372 test) — landed at 94% of the
100,000 target, reported honestly rather than padded, same discipline as every round. Rejection
rate (2.7%) is higher than v3's 1.2%, expected for a round this size on its first full-scale pass —
several new generators haven't yet had the multiple smoke-test tuning passes the round-3 generators
accumulated over their lifetime. `corpus_lookup.py`'s registry and the RAG corpus (now 37 files /
102,163 chunks, up from 22 / 79,235) were updated with all round-4 source files.
