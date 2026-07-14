# chatPDB dataset layout (placeholder — populated in Phase 3)

MLX-LM expects a data directory containing `train.jsonl` and `valid.jsonl` (note: `valid`, not
`val`). `test.jsonl` is held out and used only by `eval/eval_pdb.py`.

```
data/
├── corpus/          # RAG sources: PDB/PDBe/EMDB metadata, UniProt, SIFTS, AlphaFold, OpenBind, ... (gitignored, populated Phase 2)
└── sft/             # SFT data (MLX --data directory), populated Phase 3
    ├── train.jsonl
    ├── valid.jsonl
    └── test.jsonl   # frozen, eval only
```

## SFT format

One JSON object per line, chat format (`messages`), matching chem_sage's schema exactly so
`mlx_lm.lora` can consume it directly. Every assistant turn that states a structural fact does so by
emitting a tool (Biopython/gemmi/DSSP/PyMOL) code block, never by asserting a number from memory.

### Worked example (illustrative — not yet generated)

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are chatPDB, a protein-structure-literate research assistant."
    },
    {
      "role": "user",
      "content": "How many chains does PDB entry 1CRN have, and what secondary structure dominates it?"
    },
    {
      "role": "assistant",
      "content": "I will parse the structure and assign secondary structure with DSSP rather than recall it.\n\n```python\nfrom Bio.PDB import PDBList, PDBParser, DSSP\npdbl = PDBList()\nfname = pdbl.retrieve_pdb_file('1crn', file_format='pdb', pdir='/tmp')\nstructure = PDBParser(QUIET=True).get_structure('1CRN', fname)\nmodel = structure[0]\nprint('chains:', [c.id for c in model])\ndssp = DSSP(model, fname)\nss_counts = {}\nfor key in dssp.keys():\n    ss = dssp[key][2]\n    ss_counts[ss] = ss_counts.get(ss, 0) + 1\nprint('SS counts:', ss_counts)\n```\n\nReturned: one chain (A), secondary structure dominated by beta strand and coil, with a short alpha-helical segment. 1CRN (crambin) is a small, mostly-beta plant seed protein."
    }
  ]
}
```

## Construction rules (to be enforced by `scripts/build_dataset.py`)

1. **Ground truth first.** Generate the answer by *running* Biopython/gemmi/DSSP against a real
   downloaded structure, then template the question and worked solution around the real output.
   Never hand-author a numeric or categorical fact.
2. **Validate on write.** Reject any example whose assistant code block fails to execute, or whose
   PDB/UniProt ID does not resolve to a real entry in the corpus.
3. **Balance the four behaviour classes equally** (see `PROJECT_PLAN.md` Phase 3): file/format
   literacy, experimental-method interpretation, tool-calling, database cross-referencing.
4. **Never train toward structure prediction.** Any generated example that asks the model to
   predict a fold or coordinates from sequence alone must be rejected at generation time.
5. **Split** 80/10/10 train/valid/test; the test split is frozen and never seen during training.

## Dataset versions

None yet — Phase 3 has not started. This table will track each round the way chem_sage's does
(version, round, example count, class count, generator command, data dir, rejection rate, token
length percentiles, class balance).
