#!/usr/bin/env python3
"""
merge_export.py — fuse the LoRA adapter into the base model (Phase 5), MLX-LM on Apple Silicon.

Fuses the adapter into the base model and saves an MLX model directory ready for
mlx_lm.server or a chat script.

Usage:
    python scripts/merge_export.py --adapter-path adapters/chatpdb_32b_v1_best --save-path models/chatpdb_32b_v1

--de-quantize (Phase 9, hosted demo): saves real fp16 HF-format safetensors instead of the MLX
4-bit format, via mlx_lm.fuse's own native --dequantize flag (confirmed live: `python -m mlx_lm
fuse --help` lists --dequantize and --export-gguf; --export-gguf itself only supports
model_type in ("llama", "mixtral", "mistral") -- chatPDB's Qwen3 isn't in that list, so GGUF
conversion still goes through llama.cpp's convert_hf_to_gguf.py as a separate step against this
fp16 output, not mlx_lm's own --export-gguf).

WARNING: --de-quantize needs ~65.6GB free disk for a 32B model (32.8e9 params x 2 bytes) -- check
real free space (`df -h /`) before running; this is real, not a formality, chatPDB has hit tight
disk margins before (see PROJECT_PLAN.md Phase 4 / feedback_icloud_eviction).
"""

import argparse
import subprocess
import sys
import yaml
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("config/train_config.yaml"))
    ap.add_argument("--adapter-path", default=None,
                    help="Override the adapter directory (defaults to the config's adapter_path)")
    ap.add_argument("--save-path", default="models/chatpdb_32b_v1")
    ap.add_argument("--de-quantize", action="store_true",
                    help="Save real fp16 HF-format safetensors instead of MLX 4-bit (Phase 9 "
                         "hosted-demo GGUF pipeline). Needs ~65.6GB free disk for a 32B model.")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    base = cfg["model"]
    adapter = args.adapter_path or cfg.get("adapter_path", "adapters/chatpdb_32b_v1_lora")

    cmd = [sys.executable, "-m", "mlx_lm", "fuse", "--model", base, "--adapter-path", adapter, "--save-path", args.save_path]
    if args.de_quantize:
        cmd.append("--dequantize")
    print("Fusing:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise SystemExit("mlx_lm.fuse not found. Install with: pip install mlx-lm")

    print(f"\nFused model saved to ./{args.save_path}")
    if args.de_quantize:
        print("De-quantized fp16 export ready for GGUF conversion (see PROJECT_PLAN.md Phase 9).")
    else:
        print(f"Serve: mlx_lm.server --model {args.save_path} --port 8080")


if __name__ == "__main__":
    main()
