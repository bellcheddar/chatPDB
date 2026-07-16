# chatPDB dataset layout

MLX-LM expects a data directory containing `train.jsonl` and `valid.jsonl` (note: `valid`, not
`val`). `test.jsonl` is held out and used only by `eval/eval_pdb.py`.

```
data/
├── corpus/            # RAG sources: RCSB/SIFTS/CATH/InterPro/Pharos/TWILIGHT/UniProt (gitignored, Phase 2 + corpus expansion)
├── structures/         # 820-file PDB-format sample, superseded by structures_all/ below but kept
│                        # (small, fast for smoke-testing) — gitignored, scripts/download_structure_pool.py
├── structures_all/      # ALL 256,444 entries as native mmCIF, 353 GB (gitignored, "let's add ALL PDB
│                         # files" round, 2026-07-16) — scripts/download_all_structures.py
└── sft/               # SFT data (MLX --data directory), v2 populated 2026-07-16
    ├── train.jsonl     # 40,187 examples
    ├── valid.jsonl     # 5,023 examples
    └── test.jsonl      # 5,023 examples, frozen, eval only
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
| v2 | R2 | 50,233 (40,187 train) | 1.5% | `build_dataset.py --n 50000 --seed 51` (after "let's add ALL PDB files and ALL fields" — full mmCIF pool + expanded RCSB metadata) | Full 50,000 target hit; `tool_calling` reaches its complete 12,500 target for the first time (previously capped at 8,010) now that the structure pool is 256,444 files instead of 820. Added 4 new generators (citation, unit cell/space group, crystallization conditions, organism/taxonomy) from the expanded metadata pull. |

v2 class balance: `file_format_literacy` 12,500, `tool_calling` 12,500, `experimental_method`
12,495, `database_cross_referencing` 11,738, `refusal_boundary` 1,000 (supplementary, not counted
toward the four-class equal split).

v2 token length (Qwen3-32B-4bit tokenizer, full chat-template-rendered example, n=2,000 sample):
p50=550, p90=620, p95=638, p99=789, max=1,503 — comfortable margin under any `max_seq_length`
chem_sage used (2048–3072); no examples needed truncation.

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
