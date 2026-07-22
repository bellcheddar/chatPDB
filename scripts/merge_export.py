#!/usr/bin/env python3
"""
merge_export.py — fuse the LoRA adapter into the base model (Phase 5), MLX-LM on Apple Silicon.

Fuses the adapter into the base model and saves an MLX model directory ready for
mlx_lm.server or a chat script.

Usage:
    python scripts/merge_export.py --adapter-path adapters/chatpdb_32b_v1_best --save-path models/chatpdb_32b_v1
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
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    base = cfg["model"]
    adapter = args.adapter_path or cfg.get("adapter_path", "adapters/chatpdb_32b_v1_lora")

    cmd = [sys.executable, "-m", "mlx_lm", "fuse", "--model", base, "--adapter-path", adapter, "--save-path", args.save_path]
    print("Fusing:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise SystemExit("mlx_lm.fuse not found. Install with: pip install mlx-lm")

    print(f"\nFused model saved to ./{args.save_path}")
    print(f"Serve: mlx_lm.server --model {args.save_path} --port 8080")


if __name__ == "__main__":
    main()
