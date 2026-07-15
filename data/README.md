# chatPDB dataset layout

MLX-LM expects a data directory containing `train.jsonl` and `valid.jsonl` (note: `valid`, not
`val`). `test.jsonl` is held out and used only by `eval/eval_pdb.py`.

```
data/
├── corpus/          # RAG sources: RCSB/SIFTS/CATH/InterPro/Pharos/TWILIGHT/UniProt (gitignored, Phase 2 + corpus expansion)
├── structures/       # 820 real downloaded PDB coordinate files, ground truth for execution-verified
│                      # tool_calling examples (gitignored, scripts/download_structure_pool.py)
└── sft/             # SFT data (MLX --data directory), populated Phase 3 (2026-07-16)
    ├── train.jsonl   # 36,402 examples
    ├── valid.jsonl   # 4,550 examples
    └── test.jsonl    # 4,550 examples, frozen, eval only
```

## SFT format

One JSON object per line, chat format (`messages`), matching chem_sage's schema exactly so
`mlx_lm.lora` can consume it directly. Every assistant turn that states a structural fact does so by
emitting a tool (Biopython/gemmi/DSSP/PyMOL) code block, never by asserting a number from memory.

### Worked example (real, pulled from `train.jsonl` v1 — DSSP execution-verified against the real
deposited coordinates for 1JOI, one of `data/structures/`'s 820 downloaded files; system prompt
shortened here for readability, full text is in `config/system_prompt.txt`)

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are chatPDB, a protein-structure-literate research assistant... [full text in config/system_prompt.txt]"
    },
    {
      "role": "user",
      "content": "Write Biopython/DSSP code to assign secondary structure to PDB entry 1JOI (file `1joi.pdb`) and summarise the helix/strand content."
    },
    {
      "role": "assistant",
      "content": "```python\nfrom Bio.PDB import PDBParser\nfrom Bio.PDB.DSSP import DSSP\n\nstructure = PDBParser(QUIET=True).get_structure('1JOI', '1joi.pdb')\nmodel = structure[0]\ndssp = DSSP(model, '1joi.pdb', dssp='mkdssp')\nss_counts = {}\nfor key in dssp.keys():\n    ss = dssp[key][2]\n    ss_counts[ss] = ss_counts.get(ss, 0) + 1\nprint(ss_counts)\n```\n\nRunning DSSP on the real deposited coordinates for 1JOI gives 128 assigned residues: 27 in helix (H/G/I), 44 in strand (E/B) — a predominantly beta structure."
    }
  ]
}
```

## Construction rules (enforced by `scripts/build_dataset.py`)

1. **Ground truth first.** Two tiers, both real: (a) templated around a value already fetched and
   spot-checked against a live API in Phase 2 / the corpus expansion round (resolution, R-free,
   UniProt function, CATH fold, EC number, Pharos TDL, TWILIGHT RSCC...) — this is what `pdb_id`,
   `comp_id`, `accession` etc. are always sampled *from* a corpus DataFrame, never a literal; or
   (b) computed live by actually running Biopython/gemmi/DSSP against a real file in
   `data/structures/` (secondary structure assignment has no metadata shortcut — it has to be
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
| v1 | R1 | 45,502 (36,402 train) | 2.2% | `build_dataset.py --n 50000 --seed 51` | Target was 50,000; `tool_calling`'s two execution-verified generators (DSSP, NMR model count) are hard-capped by the 820-file structure pool, not padded — see class balance below. |

v1 class balance: `file_format_literacy` 12,500, `experimental_method` 12,500,
`database_cross_referencing` 11,492, `tool_calling` 8,010, `refusal_boundary` 1,000 (supplementary,
not counted toward the four-class equal split).

v1 token length (Qwen3-32B-4bit tokenizer, full chat-template-rendered example, n=2,000 sample):
p50=549, p90=612, p95=646, p99=816, max=1,549 — comfortable margin under any `max_seq_length`
chem_sage used (2048–3072); no examples needed truncation.

**Comparison to chem_sage:** chem_sage grew its SFT dataset over 5 rounds — 1,500 (R1) → 5,000 (R3)
→ 8,000 (R4) → 20,000 (R5) examples. chatPDB's round 1 (45,502) is already more than double
chem_sage's largest round, generated in a single pass rather than five — a consequence of chatPDB's
corpus being large and richly structured enough (256k enriched entries, 975k SIFTS-UniProt rows,
870k TWILIGHT ligand-QC rows...) to parameterise thousands of unique, individually-grounded
instances per generator, the same scaling trick chem_sage used per-molecule with RDKit.
