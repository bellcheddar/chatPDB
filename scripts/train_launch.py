#!/usr/bin/env python3
"""
train_launch.py — launch QLoRA training with MLX memory limits to prevent swap thrashing.

Ports chem_sage's train_launch.py pattern (disk/swap checks, cache/memory limits against
mx.device_info()["max_recommended_working_set_size"]) and adds:
  - mx.set_wired_limit(), ahead of the cache/memory limit calls, to cap how much memory MLX
    wires so paging behaves sanely (macOS 15+; chem_sage didn't have this).
  - Checkpoint auto-resume: scans adapter_path for the highest-iter checkpoint and offers
    --resume to wire it into mlx_lm.lora's native --resume-adapter-file.
  - --calibrate N: runs N iters only, reports real measured s/iter, and exits before
    committing to the full run. Use this before ever launching a real multi-hour run.

Usage:
    python scripts/train_launch.py --config config/train_config.yaml --calibrate 20
    python scripts/train_launch.py --config config/train_config.yaml
    python scripts/train_launch.py --config config/train_config.yaml --resume

Pre-requisite: run scripts/preflight.sh first to flush background processes.
"""

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


def _check_disk(min_gb: float) -> None:
    usage = shutil.disk_usage("/")
    free_gb = usage.free / (1024**3)
    print(f"  Disk free:           {free_gb:.1f} GB", end="")
    if free_gb < min_gb:
        print(f"  WARNING: only {free_gb:.1f} GB free (need >= {min_gb:.0f} GB for checkpoints)")
        ans = input("     Continue anyway? (y/n) ").strip().lower()
        if ans != "y":
            raise SystemExit("Aborted. Free disk space and re-run.")
    else:
        print("  OK")


def _check_swap() -> None:
    try:
        r = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True)
        line = r.stdout.strip()
        print(f"  Swap:                {line}")
        m = re.search(r"used\s*=\s*([\d.]+)([MG])", line)
        if m:
            val, unit = float(m.group(1)), m.group(2)
            val_mb = val if unit == "M" else val * 1024
            if val_mb > 500:
                print(f"  WARNING: {val_mb:.0f} MB swap in use -- consider running scripts/preflight.sh first")
            else:
                print("  OK: swap clean")
    except Exception:
        pass


def _find_latest_checkpoint(adapter_path: Path) -> Path | None:
    if not adapter_path.exists():
        return None
    checkpoints = list(adapter_path.glob("*_adapters.safetensors"))
    if not checkpoints:
        return None

    def _iter_num(p: Path) -> int:
        m = re.match(r"(\d+)_adapters\.safetensors", p.name)
        return int(m.group(1)) if m else -1

    return max(checkpoints, key=_iter_num)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("config/train_config.yaml"))
    ap.add_argument("--cache-fraction", type=float, default=0.30,
                    help="Cap MLX cache at this fraction of max working set (default: 0.30)")
    ap.add_argument("--memory-fraction", type=float, default=0.90,
                    help="Cap MLX total memory at this fraction (default: 0.90)")
    ap.add_argument("--wired-fraction", type=float, default=0.70,
                    help="Cap MLX wired memory at this fraction of max working set (default: 0.70)")
    ap.add_argument("--min-disk-gb", type=float, default=20.0,
                    help="Abort if free disk is below this GB (default: 20)")
    ap.add_argument("--skip-disk-check", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="Auto-resume from the highest-iter checkpoint under adapter_path")
    ap.add_argument("--calibrate", type=int, default=None, metavar="N",
                    help="Run only N iters to measure real s/iter on this machine, then exit")
    ap.add_argument("--no-report", action="store_true",
                    help="Skip W&B/SwanLab reporting for this run (e.g. before `wandb login` is set up)")
    args = ap.parse_args()

    try:
        import mlx.core as mx
    except ImportError:
        raise SystemExit("MLX not found. Install with: pip install mlx")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    info = mx.device_info()
    ws = info["max_recommended_working_set_size"]
    ws_gb = ws / (1024**3)

    wired_limit = int(ws * args.wired_fraction)
    cache_limit = int(ws * args.cache_fraction)
    memory_limit = int(ws * args.memory_fraction)

    print("=== MLX Memory Configuration ===")
    print(f"  Device:              {info.get('device_name', '?')}")
    print(f"  Device working set:  {ws_gb:.1f} GB")
    print(f"  Wired limit:         {wired_limit / (1024**3):.1f} GB ({args.wired_fraction:.0%})")
    print(f"  Cache limit:         {cache_limit / (1024**3):.1f} GB ({args.cache_fraction:.0%})")
    print(f"  Memory limit:        {memory_limit / (1024**3):.1f} GB ({args.memory_fraction:.0%})")

    if not args.skip_disk_check:
        _check_disk(args.min_disk_gb)
    _check_swap()

    print()

    mx.set_wired_limit(wired_limit)
    mx.set_cache_limit(cache_limit)
    mx.set_memory_limit(memory_limit)

    cmd = [sys.executable, "-m", "mlx_lm", "lora", "--config", str(args.config), "--train"]

    if args.no_report:
        cmd += ["--report-to", ""]

    if args.resume:
        adapter_path = Path(cfg.get("adapter_path", "adapters"))
        latest = _find_latest_checkpoint(adapter_path)
        if latest is None:
            print(f"  --resume requested but no checkpoint found under {adapter_path}; starting fresh.")
        else:
            print(f"  Resuming from: {latest}")
            cmd += ["--resume-adapter-file", str(latest)]

    if args.calibrate is not None:
        print(f"=== Calibration run: {args.calibrate} iters ===")
        # mlx_lm.lora always evals at it==1 and it==iters regardless of steps_per_eval (see
        # tuner/trainer.py: "it == 1 or it % args.steps_per_eval == 0 or it == args.iters") --
        # val_batches=0 keeps those eval calls near-instant so they don't contaminate the
        # measured s/iter with two full validation passes.
        cmd += ["--iters", str(args.calibrate), "--steps-per-report", "1", "--val-batches", "0"]
        start = time.monotonic()
        proc = subprocess.run(cmd)
        elapsed = time.monotonic() - start
        print()
        if proc.returncode == 0:
            s_per_iter = elapsed / args.calibrate
            print(f"=== Calibration complete: {elapsed:.1f}s for {args.calibrate} iters "
                  f"({s_per_iter:.2f} s/iter) ===")
            print(f"  Projected time for N iters: N * {s_per_iter:.2f}s")
            for target_h in (10, 20, 30):
                target_iters = int(target_h * 3600 / s_per_iter)
                print(f"    ~{target_h}h budget -> ~{target_iters} iters")
        sys.exit(proc.returncode)

    print("=== Launching training ===")
    print(f"  Config: {args.config}")
    print()

    try:
        proc = subprocess.run(cmd)
        sys.exit(proc.returncode)
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
