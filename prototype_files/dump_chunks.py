# prototype_files/dump_chunks.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import random

import chromadb

DEFAULT_CHROMA_DIRNAME = ".chromadb"
DEFAULT_COLLECTION_NAME = "rag_chunks"


def dump_sample_chunks(
    repo_root: Path,
    out_path: Path,
    n_samples: int = 8,
) -> None:
    """
    Dump a small, representative sample of chunks from ChromaDB
    to a text file for human inspection.
    """
    chroma_dir = repo_root / DEFAULT_CHROMA_DIRNAME
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(DEFAULT_COLLECTION_NAME)

    # Pull all metadata + documents (small repo, so OK)
    results = collection.get(include=["documents", "metadatas"])

    docs: List[str] = results.get("documents", [])
    metas: List[Dict[str, Any]] = results.get("metadatas", [])
    ids: List[str] = results.get("ids", [])

    total = len(docs)
    if total == 0:
        raise RuntimeError("No chunks found in Chroma collection.")

    # Random but deterministic-ish sample for reproducibility
    random.seed(42)
    sample_indices = random.sample(range(total), min(n_samples, total))

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"Total chunks in index: {total}\n")
        f.write(f"Sampled chunks: {len(sample_indices)}\n")
        f.write("=" * 100 + "\n\n")

        for i, idx in enumerate(sample_indices, start=1):
            meta = metas[idx] or {}
            doc = docs[idx]
            cid = ids[idx] if idx < len(ids) else "<no-id>"

            f.write(f"[Chunk {i}]\n")
            f.write(f"ID: {cid}\n")
            f.write(f"Parser: {meta.get('parser', '?')}\n")
            f.write(f"Chunk type: {meta.get('chunk_type', '?')}\n")
            f.write(f"Symbol: {meta.get('symbol_name', '')}\n")
            f.write(f"Source path: {meta.get('source_path', '')}\n")
            f.write(
                f"Location: {meta.get('loc_kind')} "
                f"{meta.get('loc_start')}–{meta.get('loc_end')}\n"
            )
            f.write(
                f"Doc type: {meta.get('doc_type')} | "
                f"Doc dir: {meta.get('doc_dir')} | "
                f"Rank hint: {meta.get('doc_rank_hint')}\n"
            )
            f.write("-" * 80 + "\n")
            f.write(doc[:1200].rstrip())  # truncate long chunks
            if len(doc) > 1200:
                f.write("\n... [TRUNCATED]\n")
            f.write("\n" + "=" * 100 + "\n\n")

    print(f"Wrote chunk sample to: {out_path}")


def main():
    repo_root = Path(__file__).resolve().parent.parent

    # Name the output file based on branch / parser if you like
    out_path = repo_root / "chunk_sample.txt"

    dump_sample_chunks(
        repo_root=repo_root,
        out_path=out_path,
        n_samples=8,
    )


if __name__ == "__main__":
    main()
