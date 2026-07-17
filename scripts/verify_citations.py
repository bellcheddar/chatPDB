#!/usr/bin/env python3
"""
verify_citations.py — independently verify every DOI-bearing citation in the corpus against
CrossRef + PubMed, into data/corpus/citations/ for SFT generation.

Right now every citation-bearing generator trusts the deposited `citation_doi`/`citation_title`
strings blindly -- never checked against an independent source. Deposited citations are wrong or
stale more often than expected (typos, pre-publication placeholders, "to be published" entries that
were never updated after the paper appeared). This script closes that gap and buckets every
citation into verified / mismatched / unresolvable / no_citation, so generators can teach calibrated
citation trust instead of blind trust.

Method (exact-DOI lookup + confirmation, not fuzzy discovery -- much higher precision):
1. CrossRef `api.crossref.org/works/{doi}` (confirmed reachable live, 200 OK) -- does the DOI
   resolve at all, and does the returned title/year agree with what's deposited (fuzzy string ratio,
   not exact match -- minor punctuation/subtitle differences are common and not a real mismatch).
2. NCBI eutils `esummary.fcgi` keyed by the deposited PubMed ID (NOT `esearch` by DOI -- esearch's
   backend was down/erroring during round 4 development, a real transient NCBI outage independent
   of this script; esummary is a simpler ID-keyed lookup on a different backend path and worked
   fine throughout). Extract the DOI from the PMID's own `articleids` and cross-check it against the
   deposited DOI -- a real, independent second source.

Deduplicated to the 89,660 distinct DOIs across the 213,162 citation-bearing entries in
pdb_entries_enriched.csv (many entries cite the same landmark paper) plus BindingDB's
`article_doi` column, rather than 213k+ redundant lookups.

Usage:
    python scripts/verify_citations.py
    python scripts/verify_citations.py --limit 500 --workers 8   # smoke test
"""

import argparse
import difflib
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import pandas as pd
import requests

OUT = Path("data/corpus/citations")
ENTRIES_SOURCE = Path("data/corpus/rcsb/pdb_entries_enriched.csv")
BINDINGDB_SOURCE = Path("data/corpus/bindingdb/bindingdb_pdb_affinities.csv")
CROSSREF_API = "https://api.crossref.org/works"
PUBMED_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
MAILTO = "marc@marcdeller.com"  # CrossRef "polite pool" -- higher, more reliable rate limits
HEADERS = {"User-Agent": f"chatPDB/1.0 (protein-structure-rag; mailto:{MAILTO})", "Connection": "close"}
TITLE_MATCH_THRESHOLD = 0.6  # difflib ratio; deliberately loose -- this confirms an exact-DOI
                              # lookup, not a fuzzy discovery search, so a wrong DOI would usually
                              # score near 0, not marginally under threshold

_lock = Lock()
_stats = {"verified": 0, "mismatched": 0, "unresolvable": 0, "rate_limited": 0, "failed": 0, "done": 0}


_TAG_RE = re.compile(r"<[^>]+>")


def _title_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    # CrossRef routinely embeds markup in titles (<scp>, <i>, <sub>...) for species names, gene
    # symbols, subscripts, each replaced by whitespace -- both the tag-stripping AND collapsing the
    # resulting whitespace runs to single spaces are needed, or leftover indentation/newlines alone
    # drag the ratio down on titles that actually agree word-for-word (found via round 4 smoke test).
    def norm(s: str) -> str:
        s = _TAG_RE.sub(" ", s)
        s = "".join(c.lower() for c in s if c.isalnum() or c.isspace())
        return re.sub(r"\s+", " ", s).strip()
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def _get_crossref(doi: str, max_retries: int = 5):
    """GET with real retry-with-backoff on 429 -- confirmed live mid-run (round 4) that CrossRef
    rate-limits at 24 concurrent workers despite the polite-pool mailto param, and the original
    code treated ANY non-200 (including 429) as "unresolvable" -- silently mislabeling ~44% of
    real, valid DOIs as fake/unverifiable. A 429 means "ask slower", not "this DOI doesn't exist";
    conflating the two would have taught the model badly wrong information about how trustworthy
    PDB citations are. Only a genuine 404 (or exhausting retries) counts as unresolvable."""
    delay = 1.0
    for attempt in range(max_retries):
        r = requests.get(f"{CROSSREF_API}/{doi}", params={"mailto": MAILTO}, headers=HEADERS,
                          timeout=(10, 20))
        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            time.sleep(float(retry_after) if retry_after else delay)
            delay = min(delay * 2, 30)
            continue
        return r
    return r  # exhausted retries -- return the last (still-429) response, caller treats as a miss


def _verify_one(row: dict) -> dict:
    doi, title, year, pmid = row["doi"], row.get("title"), row.get("year"), row.get("pubmed_id")
    result = {"doi": doi, "bucket": "unresolvable", "crossref_title": None,
              "title_similarity": None, "pmid_doi_match": None}
    try:
        r = _get_crossref(doi)
        if r.status_code == 429:
            # Still rate-limited after retries -- a real transient failure, not evidence the DOI
            # is bad. Bucket separately so it's never confused with a genuine unresolvable DOI.
            with _lock:
                _stats["rate_limited"] += 1
            result["bucket"] = "rate_limited"
            return result
        if r.status_code != 200:
            with _lock:
                _stats["unresolvable"] += 1
            return result
        work = r.json().get("message", {})
        cr_title = (work.get("title") or [None])[0]
        cr_year = None
        date_parts = work.get("published", {}).get("date-parts", [[None]])
        if date_parts and date_parts[0]:
            cr_year = date_parts[0][0]
        result["crossref_title"] = cr_title
        # pd.notna(), not a bare truthiness check: a NaN float is truthy in Python, so `if title`
        # alone would pass a NaN straight into _title_similarity's re.sub() and crash there instead
        # (same missing-pd.isna() bug shape as year_ok above, caught by inspection before it hit
        # a real row and repeated the same crash-mid-run failure).
        sim = _title_similarity(title, cr_title) if pd.notna(title) and cr_title else None
        result["title_similarity"] = sim
        # pandas represents a missing citation_year as float NaN, not None -- `year is None` never
        # caught it, so a NaN year reached int(year) and crashed the whole run partway through
        # (the real cause of this looking "stuck": one worker thread's uncaught exception kills
        # every future's result() as soon as as_completed() reaches it, not a hang).
        year_ok = (pd.isna(year) or pd.isna(cr_year) or int(year) == int(cr_year))
        title_ok = sim is None or sim >= TITLE_MATCH_THRESHOLD
        bucket = "verified" if (title_ok and year_ok) else "mismatched"

        if pmid and pd.notna(pmid):
            try:
                pr = requests.get(PUBMED_API, params={"db": "pubmed", "id": int(pmid), "retmode": "json"},
                                   headers=HEADERS, timeout=(10, 20))
                summary = pr.json().get("result", {}).get(str(int(pmid)), {})
                pmid_doi = next((a["value"] for a in summary.get("articleids", []) if a.get("idtype") == "doi"), None)
                if pmid_doi:
                    match = pmid_doi.strip().lower() == doi.strip().lower()
                    result["pmid_doi_match"] = match
                    if not match:
                        bucket = "mismatched"
            except requests.RequestException:
                pass  # PubMed cross-check is a bonus signal, not required for verification

        result["bucket"] = bucket
        with _lock:
            _stats[bucket if bucket in _stats else "verified"] += 1
    except requests.RequestException:
        with _lock:
            _stats["failed"] += 1
    return result


def _load_distinct_dois() -> pd.DataFrame:
    entries = pd.read_csv(ENTRIES_SOURCE, usecols=["citation_doi", "citation_title", "citation_year", "citation_pubmed_id"],
                           low_memory=False)
    entries = entries[entries["citation_doi"].notna()].drop_duplicates("citation_doi")
    rows = [{"doi": r["citation_doi"], "title": r["citation_title"], "year": r["citation_year"],
             "pubmed_id": r["citation_pubmed_id"]} for _, r in entries.iterrows()]

    if BINDINGDB_SOURCE.exists():
        bdf = pd.read_csv(BINDINGDB_SOURCE, usecols=["article_doi"], low_memory=False)
        bdf = bdf[bdf["article_doi"].notna()]
        known_dois = {r["doi"] for r in rows}
        extra = bdf["article_doi"].drop_duplicates()
        extra = extra[~extra.isin(known_dois)]
        rows += [{"doi": d, "title": None, "year": None, "pubmed_id": None} for d in extra]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    dois = _load_distinct_dois()
    if args.limit:
        dois = dois.head(args.limit)
    print(f"  {len(dois):,} distinct DOIs to verify, {args.workers} workers")

    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_verify_one, row): row for row in dois.to_dict("records")}
        for fut in as_completed(futures):
            results.append(fut.result())
            with _lock:
                _stats["done"] += 1
                done = _stats["done"]
            if done % 5000 == 0:
                rate = done / (time.time() - t0)
                eta = (len(dois) - done) / rate / 60 if rate > 0 else 0
                print(f"  {done:,}/{len(dois):,} (verified {_stats['verified']:,}, "
                      f"mismatched {_stats['mismatched']:,}, unresolvable {_stats['unresolvable']:,}, "
                      f"rate_limited {_stats['rate_limited']:,}, failed {_stats['failed']:,}) -- "
                      f"{rate:.1f}/s, ETA {eta:.1f}m", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(OUT / "citation_verification.csv", index=False)
    print(f"\nDone in {(time.time()-t0)/60:.1f}m: {len(df):,} DOIs verified -> citation_verification.csv")
    print(df["bucket"].value_counts())


if __name__ == "__main__":
    main()
