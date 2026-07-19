#!/usr/bin/env python3
"""
download_alphafraud.py — pull real AlphaFold-vs-experimental comparison data from Marc's sibling
project AlphaFraud (alphafraud.mdeller.com) into data/corpus/alphafraud/ for RAG ingestion and SFT
generation.

Replaces chatPDB's previously-thin "predicted vs experimental" comparison (global pLDDT vs
resolution/R-free only) with real computed TM-score, GDT-TS, lDDT, CA-RMSD, and a FRAUD score /
"confidently wrong" flag (high pLDDT, low TM-score) for structures AlphaFraud has actually evaluated.

STAGED, NOT BLOCKING (confirmed with Marc 2026-07-17): AlphaFraud's historical backfill is still
running -- confirmed live that its /archive only has run labels through late 2022 plus a handful of
recent 2026 weekly runs (the 2023-2025 gap hasn't been backfilled yet). This script is a normal
re-runnable download_*.py: it pulls whatever's computed *right now* and should be re-run with
`--refresh` once AlphaFraud's backfill completes, to pick up full coverage. Nothing in this round's
build blocks on that completion.

Access confirmed live 2026-07-17: no bulk "all entities" API exists. `/api/leaderboard` is
hard-capped at 500 (a biased top-fraud-score sample, not representative). The real path is scraping
`/archive` for the full list of weekly/monthly run labels, then `/api/week/{label}` per label
(confirmed real field set matches AlphaFraud's `entities` DB table exactly) -- this is the only way
to get every entity AlphaFraud has actually scored, not just the most anomalous 500. The large
per-residue JSON blobs (heatmaps_json/per_residue_json/domains_json/metrics_json) are dropped here;
they're not useful for text-generation SFT examples and roughly triple response size. Also pulls
`/api/analysis`, the cached cumulative snapshot (per-superfamily scorecards, conformational
heterogeneity, enrichment stats).

**KNOWN GAP, found round 6 (2026-07-18): `/api/week/{label}` under-serves months that got
reprocessed across multiple backfill runs.** After the full backfill completed (32,728 screened,
15,482 fully `compared`), this script's API-based pull only captured 6,799 of those 15,482 real
rows -- confirmed via a direct read-only SQL query against `/opt/alphafraud/alphafraud.db` over SSH
(`SELECT COUNT(*) FROM entities WHERE status='compared'`). The gap is concentrated in specific
months (2019 almost entirely missing -- e.g. 2019-04 has 931 real rows, the API served 5; also
2026-01 through 2026-03), not spread evenly, and a live re-check of `/api/week/2019-04-01` right
now still only returns 7 total entities -- the API endpoint itself is the bottleneck, not this
script's parsing. **If a future re-run of this script yields a suspiciously low total again (well
under whatever AlphaFraud's own `/api/analysis` snapshot or admin tally reports), don't trust the
API path -- pull directly from the database instead:**
    ssh alphafraud.mdeller.com "/opt/alphafraud/.venv/bin/python3 -c \"
    import sqlite3, csv
    conn = sqlite3.connect('file:/opt/alphafraud/alphafraud.db?mode=ro', uri=True)
    cur = conn.cursor()
    fields = [...]  # see KEEP_FIELDS below
    cur.execute(f\\\"SELECT {','.join(fields)} FROM entities WHERE status='compared'\\\")
    ...\""
(requires the backfill service to be `inactive`, not `active` -- SQLite refuses even read-only
connections while the writer process holds an open transaction, confirmed live both ways.)

Usage:
    python scripts/download_alphafraud.py
    python scripts/download_alphafraud.py --limit-weeks 5   # smoke test
"""

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

OUT = Path("data/corpus/alphafraud")
BASE = "https://alphafraud.mdeller.com"
HEADERS = {"User-Agent": "chatPDB/1.0 (protein-structure-rag; marc@marcdeller.com)", "Connection": "close"}

# The promoted headline columns + a handful of context fields -- deliberately excludes the four
# large per-residue/heatmap JSON blob columns (see module docstring).
KEEP_FIELDS = [
    "entry_id", "chain", "uniprot", "uniprot_name", "description", "deposit_date", "release_date",
    "resolution", "method", "novelty_identity", "is_novel", "closest_pre_cutoff",
    "af_entry_id", "af_model_version", "mean_plddt",
    "tm_by_experiment", "lddt", "gdt_ts", "ca_rmsd", "fraud_score", "confidently_wrong",
    "status", "skip_reason",
]


def list_week_labels() -> list[str]:
    r = requests.get(f"{BASE}/archive", headers=HEADERS, timeout=(10, 30))
    r.raise_for_status()
    labels = sorted(set(re.findall(r"week/([0-9-]+)", r.text)))
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit-weeks", type=int, default=None)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print("[1/3] Listing run labels from /archive ...")
    labels = list_week_labels()
    if args.limit_weeks:
        labels = labels[-args.limit_weeks:]
    print(f"  {len(labels):,} run labels found")

    print("\n[2/3] Fetching per-week entities ...")
    # A large week's response embeds heavy per-residue/heatmap JSON for every entity (tens of MB
    # possible) and can trickle in slowly enough that no single gap between chunks ever breaches
    # requests' (connect, read) timeout, even though the *total* transfer effectively never
    # finishes -- the same silent-stall shape this project has repeatedly hit with long-lived
    # connections (round 3: requests.Session() against EBI/RCSB; this round: PDB-REDO's rsync).
    #
    # First attempt at a fix used ONE reused ThreadPoolExecutor(max_workers=1) with
    # future.result(timeout=...) per call -- wrong: result()'s timeout only stops the *caller* from
    # waiting, it doesn't cancel or free the worker thread, so a single genuinely-hung request
    # permanently occupies the pool's only worker and every subsequent submission queues up behind
    # it forever, silently defeating the "move to the next week" intent (confirmed the hard way: the
    # process sat at 0 progress with ~0 CPU for over an hour). Real fix: spin up a throwaway
    # single-worker executor per request and never call .shutdown(wait=True)/use it as a context
    # manager on timeout -- let a stuck thread be abandoned (leaked until process exit, at most 60
    # of them, trivial) rather than ever blocking on it again.
    HARD_DEADLINE = 90

    def _fetch_week(label: str):
        r = requests.get(f"{BASE}/api/week/{label}", headers=HEADERS, timeout=(15, 60))
        r.raise_for_status()
        return r.json()

    rows = []
    for i, label in enumerate(labels, 1):
        ex = ThreadPoolExecutor(max_workers=1)
        try:
            entities = ex.submit(_fetch_week, label).result(timeout=HARD_DEADLINE)
            ex.shutdown(wait=False)
        except FutureTimeoutError:
            ex.shutdown(wait=False)  # abandon the stuck worker thread, do not join it
            print(f"  [warn] week {label} exceeded {HARD_DEADLINE}s wall-clock deadline, skipping "
                  f"(will still be picked up on the next --refresh)")
            continue
        except (requests.RequestException, json.JSONDecodeError) as e:
            ex.shutdown(wait=False)
            print(f"  [warn] week {label} failed: {e}")
            continue

        for e in entities:
            if e.get("status") != "compared":
                continue
            rows.append({k: e.get(k) for k in KEEP_FIELDS} | {"run_label": label})
        if i % 10 == 0:
            print(f"  {i:,}/{len(labels):,} weeks fetched, {len(rows):,} compared entities so far", flush=True)
        time.sleep(0.05)

    df = pd.DataFrame(rows)
    df["pulled_at"] = datetime.now(timezone.utc).isoformat()
    df.to_csv(OUT / "alphafraud_comparisons.csv", index=False)
    print(f"  {len(df):,} compared entities -> alphafraud_comparisons.csv")
    if not df.empty:
        print(f"  confidently_wrong: {int(df['confidently_wrong'].sum()):,} of {len(df):,}")

    print("\n[3/3] Fetching cumulative analysis snapshot ...")
    try:
        r = requests.get(f"{BASE}/api/analysis", headers=HEADERS, timeout=(15, 60))
        r.raise_for_status()
        snapshot = r.json()
        snapshot["_pulled_at"] = datetime.now(timezone.utc).isoformat()
        (OUT / "alphafraud_analysis_snapshot.json").write_text(json.dumps(snapshot, indent=1))
        print(f"  saved analysis snapshot ({len(json.dumps(snapshot)):,} bytes) -> "
              f"alphafraud_analysis_snapshot.json")
    except requests.RequestException as e:
        print(f"  [warn] analysis snapshot fetch failed: {e}")

    print(f"\nDone. Re-run with default args (no --limit-weeks) once AlphaFraud's backfill "
          f"completes for full historical coverage -- see PROJECT_PLAN.md round 4 notes.")


if __name__ == "__main__":
    main()
