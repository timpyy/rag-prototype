# prototype_files/query.py
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_CHROMA_DIRNAME = ".chromadb"
DEFAULT_COLLECTION_NAME = "rag_chunks"


def _get_client(persist_directory: Path) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(persist_directory))


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    # Cache the model so repeated queries don’t re-load weights/log spam
    return SentenceTransformer(MODEL_NAME)


def search(
    query: str,
    repo_root: Path,
    top_k: int = 5,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Search Chroma using an embedding of the query.
    Returns list of (score, item) where score is distance (smaller is better).
    """
    chroma_dir = repo_root / DEFAULT_CHROMA_DIRNAME
    client = _get_client(chroma_dir)
    collection = client.get_collection(collection_name)

    model = _get_model()
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    q_list = q_emb.astype(float).tolist()

    results = collection.query(
        query_embeddings=q_list,
        n_results=top_k,
        include=["metadatas", "documents", "distances"],
    )

    out: List[Tuple[float, Dict[str, Any]]] = []
    if not results or "distances" not in results or len(results["distances"]) == 0:
        return out

    dists = results["distances"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    ids = results.get("ids", [[]])[0]  # may be missing in some versions

    for idx, (score, doc, meta) in enumerate(zip(dists, docs, metas)):
        _id = ids[idx] if idx < len(ids) else (meta.get("id") if isinstance(meta, dict) else None)
        item = {"id": _id, "text": doc, "metadata": meta}
        out.append((float(score), item))

    return out


def pretty_print(results: List[Tuple[float, Dict[str, Any]]]) -> None:
    for score, item in results:
        meta = item.get("metadata", {}) or {}

        src = meta.get("source_path", "<unknown>")
        chunk_type = meta.get("chunk_type", "unknown")
        doc_type = meta.get("doc_type", "?")
        doc_dir = meta.get("doc_dir", "?")
        symbol = meta.get("symbol_name", "")

        loc_kind = meta.get("loc_kind", "none")
        loc_start = meta.get("loc_start")
        loc_end = meta.get("loc_end")

        loc = ""
        if loc_kind == "lines" and loc_start is not None:
            loc = f":{loc_start}-{loc_end}"
        elif loc_kind == "cell" and loc_start is not None:
            loc = f" [cell {loc_start}]"

        print("\n" + "=" * 80)
        sym_part = f" | symbol={symbol}" if symbol else ""
        print(f"{chunk_type}{sym_part} | score={score:.4f}")
        print(f"{src}{loc}  (doc_type={doc_type}, doc_dir={doc_dir})")
        if meta:
            print(f"meta: {meta}")
        print("-" * 80)
        print(item["text"][:1500])
