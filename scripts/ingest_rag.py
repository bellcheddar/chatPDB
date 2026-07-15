#!/usr/bin/env python3
"""
ingest_rag.py — chunk, embed, and store the chatPDB corpus.

Phase 2 of PROJECT_PLAN.md. Ship this before any training: a working RAG layer on the
base model is already a usable product. Ported from chem_sage/scripts/ingest_rag.py —
same chunking/embedding/storage pattern, adapted for chatPDB's corpus file names.

Usage:
    python scripts/ingest_rag.py --corpus data/corpus --store .chroma
    python scripts/ingest_rag.py --corpus data/corpus --store .chroma --reset
"""

import argparse
import hashlib
import math
from pathlib import Path

import chromadb
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
UPSERT_BATCH = 512

# Maximum ChromaDB chunks to produce per CSV/TSV file.
# Files with more rows than this cap are grouped: multiple rows are concatenated
# into one chunk so that the total chunk count stays within the limit.
# This keeps retrieval useful — huge relational files (SIFTS, entry index) are
# still represented without overwhelming the index. Matches chem_sage's approach.
MAX_CHUNKS_PER_CSV = {
    # mapping / relational files — low semantic diversity per row, group heavily
    "pdb_all_entries": 3_000,
    "sifts_pdb_uniprot": 2_000,
    "sifts_pdb_pfam": 2_000,
    "sifts_pdb_cath": 1_000,
    "sifts_pdb_scop2": 1_000,
    "sifts_pdb_enzyme": 2_000,
    "sifts_pdb_go": 3_000,
    "sifts_pdb_interpro": 3_000,
    # richer per-row data — method, resolution, R-free, title — keep more chunks
    "pdb_entries_enriched": 6_000,
    "pdb_ccd_full": 5_000,
    # default cap for any CSV not listed above
    "__default__": 5_000,
}

# Row-count cap applied BEFORE chunking (even-stride sample), for files whose row count is so
# large relative to MAX_CHUNKS_PER_CSV that the cap-driven rows_per_chunk would blow chunks up
# to many thousands of characters (way past what an embedding model meaningfully encodes).
# sifts_pdb_go.csv is the motivating case: 16.3M rows at ~59 chars/row would need ~3,250
# rows/chunk to hit a 5,000-chunk cap — ~192,000 chars/chunk. Sampling to a bounded row count
# first keeps chunks in the low-thousands-of-characters range, which the embedder can still see
# past its truncation limit. GO/InterPro chain mappings are also extremely repetitive per PDB
# entry, so a representative sample loses little that full coverage would have added — exact
# lookups belong in rag/corpus_lookup.py's deterministic fast path, not embedding search anyway.
MAX_ROWS_PER_CSV = {
    "sifts_pdb_go": 150_000,
    "sifts_pdb_interpro": 150_000,
}


def load_documents(corpus_dir: Path):
    """Yield (source_path, text, pre_chunked) for every document in the corpus.

    PDFs / TXT / MD: pre_chunked=False -> sliding-window chunker applied in main().
    CSV/TSV (relational tables): pre_chunked=True -> already split into final-size
      groups; main() must NOT apply the sliding-window chunker a second time.
    """
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        raise SystemExit(f"Corpus directory not found: {corpus_dir}. Run scripts/download_*.py first.")

    found = 0
    for path in sorted(corpus_dir.rglob("*")):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            text = _load_pdf(path)
            if text.strip():
                yield path, text, False
                found += 1
        elif suffix in (".csv", ".tsv"):
            for row_chunk in _load_table(path):
                yield path, row_chunk, True
                found += 1
        elif suffix in (".txt", ".md"):
            text = path.read_text(errors="replace")
            if text.strip():
                yield path, text, False
                found += 1

    if found == 0:
        print(f"Warning: no documents found in {corpus_dir}. "
              "Run scripts/download_rcsb.py (and future download_*.py scripts) first.")


def _load_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print(f"Warning: could not parse PDF {path.name}: {e}")
        return ""


def _load_table(path: Path):
    """Yield fixed-size text chunks for a relational CSV/TSV.

    Two-stage sizing:
    1. Estimate how many rows fit in CHUNK_SIZE chars (using the first 50 rows).
    2. Apply the per-file cap from MAX_CHUNKS_PER_CSV: if the natural row count
       would produce more chunks than the cap, increase rows_per_chunk to stay
       within the cap.

    Each yielded string is <= CHUNK_SIZE chars so main() does NOT apply the
    sliding-window chunker again (pre_chunked=True).
    """
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        df = pd.read_csv(path, sep=sep, dtype=str, low_memory=False).fillna("")
    except Exception as e:
        print(f"Warning: could not parse table {path.name}: {e}")
        return
    if df.empty:
        return

    stem = path.stem.lower()
    row_cap = next((v for k, v in MAX_ROWS_PER_CSV.items() if k in stem), None)
    if row_cap and len(df) > row_cap:
        stride = max(1, len(df) // row_cap)
        original_len = len(df)
        df = df.iloc[::stride].reset_index(drop=True)
        print(f"  {path.name}: sampled {original_len:,} -> {len(df):,} rows (every {stride}th) before chunking")

    header = " | ".join(df.columns.tolist())
    prefix = f"[Table: {path.name}] Columns: {header}\n"
    prefix_len = len(prefix)

    sample = df.head(50)
    avg_row_chars = max(1, int(sample.apply(lambda r: len(" | ".join(r)), axis=1).mean()))
    chars_for_rows = max(200, CHUNK_SIZE - prefix_len)
    rows_by_size = max(1, chars_for_rows // avg_row_chars)

    cap = next((v for k, v in MAX_CHUNKS_PER_CSV.items() if k in stem), MAX_CHUNKS_PER_CSV["__default__"])
    rows_by_cap = max(1, math.ceil(len(df) / cap))
    rows_per_chunk = max(rows_by_size, rows_by_cap)

    for start in range(0, len(df), rows_per_chunk):
        group = df.iloc[start : start + rows_per_chunk]
        rows_text = "\n".join(" | ".join(r) for r in group.itertuples(index=False, name=None))
        yield prefix + rows_text


def chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Character-level sliding window chunker."""
    step = size - overlap
    return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size].strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--store", type=Path, default=Path(".chroma"))
    ap.add_argument("--collection", default="chatpdb")
    ap.add_argument("--embed-model", default=EMBED_MODEL)
    ap.add_argument("--reset", action="store_true", help="Delete and rebuild the collection from scratch")
    args = ap.parse_args()

    print(f"Loading embedder: {args.embed_model}")
    embedder = SentenceTransformer(args.embed_model)

    client = chromadb.PersistentClient(
        path=str(args.store),
        settings=chromadb.Settings(allow_reset=True, anonymized_telemetry=False),
    )
    if args.reset:
        try:
            client.delete_collection(args.collection)
            print(f"Deleted collection '{args.collection}'")
        except Exception:
            pass
    collection = client.get_or_create_collection(args.collection, metadata={"hnsw:space": "cosine"})

    ids, texts, metas = [], [], []
    current_file, file_chunk_count, global_idx = None, 0, 0
    for source, text, pre_chunked in load_documents(args.corpus):
        if source != current_file:
            if current_file is not None:
                print(f"  {current_file.name}: {file_chunk_count} chunks")
            current_file = source
            file_chunk_count = 0
        pieces = [text] if pre_chunked else chunk(text)
        file_chunk_count += len(pieces)
        for piece in pieces:
            uid = hashlib.sha256(f"{global_idx}:{source}:{piece[:40]}".encode()).hexdigest()[:16]
            ids.append(uid)
            texts.append(piece)
            metas.append({"source": str(source), "chunk": global_idx})
            global_idx += 1
    if current_file is not None:
        print(f"  {current_file.name}: {file_chunk_count} chunks")

    if not ids:
        print("No chunks to embed. Run scripts/download_rcsb.py first.")
        return

    print(f"Embedding {len(ids):,} chunks with {args.embed_model} ...")
    all_vecs = []
    embed_batch = 128
    for start in range(0, len(texts), embed_batch):
        batch_vecs = embedder.encode(texts[start : start + embed_batch], show_progress_bar=False).tolist()
        all_vecs.extend(batch_vecs)
        done = min(start + embed_batch, len(texts))
        if done % 5000 < embed_batch or done == len(texts):
            print(f"  embedded {done:,}/{len(texts):,}", end="\r")

    print(f"\nUpserting into Chroma collection '{args.collection}' ...")
    for start in range(0, len(ids), UPSERT_BATCH):
        collection.upsert(
            ids=ids[start : start + UPSERT_BATCH],
            embeddings=all_vecs[start : start + UPSERT_BATCH],
            documents=texts[start : start + UPSERT_BATCH],
            metadatas=metas[start : start + UPSERT_BATCH],
        )

    print(f"\nDone. {len(ids)} chunks in '{args.store}/{args.collection}'.")
    print("Exit test: query a corpus-only fact and confirm a grounded, sourced answer.")
    print("\nQuick query snippet:")
    print("  from rag.retrieve import Retriever, format_context")
    print("  r = Retriever()")
    print("  print(format_context(r.retrieve('your question')))")


if __name__ == "__main__":
    main()
