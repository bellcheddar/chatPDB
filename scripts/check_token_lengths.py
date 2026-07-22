"""
Measure token-length distribution of data/sft/<split>.jsonl against the real chat template.

Used to validate max_seq_length before training -- ported from chem_sage's real
check_token_lengths.py. Run against the full split, not a sample, since a single rare long
outlier above max_seq_length gets silently truncated by mlx_lm.lora otherwise.

Usage:
    python scripts/check_token_lengths.py [--data data/sft] [--split train]
    python scripts/check_token_lengths.py --model mlx-community/Qwen3-32B-4bit
"""
import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/sft"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="mlx-community/Qwen3-32B-4bit",
                    help="HuggingFace tokenizer repo to use")
    ap.add_argument("--max-seq-length", type=int, default=None,
                    help="If set, also report the count/fraction of examples exceeding this")
    args = ap.parse_args()

    jsonl = args.data / f"{args.split}.jsonl"
    if not jsonl.exists():
        raise SystemExit(f"Not found: {jsonl}")

    print(f"Loading tokenizer: {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)

    lengths = []
    with open(jsonl) as f:
        for line in f:
            ex = json.loads(line)
            text = tok.apply_chat_template(ex["messages"], tokenize=False)
            lengths.append(len(tok.encode(text)))
    lengths = np.array(lengths)

    print(f"\n{jsonl}: n={len(lengths):,}")
    for p in (50, 90, 95, 99, 99.5, 99.9, 100):
        print(f"  p{p}: {np.percentile(lengths, p):.0f} tokens")
    print(f"  max: {lengths.max()}")

    if args.max_seq_length:
        over = int((lengths > args.max_seq_length).sum())
        frac = over / len(lengths)
        print(f"\n  examples exceeding max_seq_length={args.max_seq_length}: {over:,} "
              f"({frac:.2%}) -- these would be silently truncated during training")


if __name__ == "__main__":
    main()
