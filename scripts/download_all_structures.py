#!/usr/bin/env python3
"""
download_all_structures.py — download mmCIF coordinate files for every entry in the corpus into
data/structures_all/, for full-scale SFT ground-truth generation.

Supersedes the 820-file stratified pool in data/structures/ (scripts/download_structure_pool.py,
kept as-is — build_dataset.py's Phase 3 generators still reference it and don't need to change).
mmCIF chosen over legacy PDB format: universal coverage (no 99,999-atom/62-chain ceiling that
excludes ~1-5% of entries), and it's already the robust path DSSP was fixed to use in Phase 3
(mkdssp 4.6.1's own legacy-PDB->mmCIF conversion has a real bug — see PROJECT_PLAN.md Phase 3 —
so downloading mmCIF natively sidesteps that class of problem entirely rather than working around
it after the fact).

Scale: ~256,448 entries. At files.rcsb.org's per-file CDN endpoint, sequential download with any
meaningful pause would take the better part of a day; this uses a thread pool (RCSB's ~10 req/s
guidance is documented for the Search/Data API, not the static file CDN, but max_workers is kept
modest as a courtesy) and is fully resumable — safe to Ctrl-C and rerun, already-downloaded files
are skipped by default.

Usage:
    python scripts/download_all_structures.py                    # full run, resumable
    python scripts/download_all_structures.py --limit 5000        # smoke test / partial run
    python scripts/download_all_structures.py --workers 12        # tune concurrency
    python scripts/download_all_structures.py --retry-failed      # only retry data/structures_all/_failed.txt
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import pandas as pd
import requests

OUT = Path("data/structures_all")
FAILED_LOG = OUT / "_failed.txt"
IDS_SOURCE = Path("data/corpus/rcsb/pdb_all_entries.csv")

_progress_lock = Lock()
_stats = {"ok": 0, "skipped": 0, "failed": 0, "done": 0}


def _download_one(pdb_id: str, session: requests.Session, max_retries: int = 3) -> tuple[str, bool]:
    dest = OUT / f"{pdb_id.lower()}.cif"
    if dest.exists() and dest.stat().st_size > 0:
        with _progress_lock:
            _stats["skipped"] += 1
            _stats["done"] += 1
        return pdb_id, True

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
    for attempt in range(max_retries):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 200 and r.content:
                dest.write_bytes(r.content)
                with _progress_lock:
                    _stats["ok"] += 1
                    _stats["done"] += 1
                return pdb_id, True
            if r.status_code == 404:
                break  # entry genuinely has no mmCIF (very rare) - don't retry
        except requests.RequestException:
            pass
        time.sleep(0.5 * (attempt + 1))  # backoff on failure/retry

    with _progress_lock:
        _stats["failed"] += 1
        _stats["done"] += 1
    return pdb_id, False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if args.retry_failed:
        if not FAILED_LOG.exists():
            print("No failed-download log found, nothing to retry.")
            return
        ids = [line.strip() for line in FAILED_LOG.read_text().splitlines() if line.strip()]
    else:
        ids = pd.read_csv(IDS_SOURCE)["pdb_id"].tolist()
        if args.limit:
            ids = ids[: args.limit]

    total = len(ids)
    print(f"Target: {total:,} entries -> {OUT}/ (mmCIF, {args.workers} workers, resumable)", flush=True)

    failed_ids: list[str] = []
    t0 = time.time()
    session = requests.Session()
    session.headers["User-Agent"] = "chatPDB/1.0 (protein-structure-rag; marc@marcdeller.com)"

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_download_one, pid, session): pid for pid in ids}
        for fut in as_completed(futures):
            pid, ok = fut.result()
            if not ok:
                failed_ids.append(pid)
            if _stats["done"] % 1000 == 0:
                elapsed = time.time() - t0
                rate = _stats["done"] / elapsed if elapsed > 0 else 0
                eta_h = (total - _stats["done"]) / rate / 3600 if rate > 0 else float("inf")
                print(f"  {_stats['done']:,}/{total:,} (ok {_stats['ok']:,}, skipped {_stats['skipped']:,}, "
                      f"failed {_stats['failed']:,}) — {rate:.1f}/s, ETA {eta_h:.1f}h", flush=True)

    if failed_ids:
        FAILED_LOG.write_text("\n".join(failed_ids) + "\n")
        print(f"\n{len(failed_ids):,} failed — logged to {FAILED_LOG} (rerun with --retry-failed)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/3600:.2f}h: {_stats['ok']:,} downloaded, {_stats['skipped']:,} already present, "
          f"{_stats['failed']:,} failed.")


if __name__ == "__main__":
    main()
