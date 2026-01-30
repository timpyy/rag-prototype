from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

from .config import Chunk

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_CHROMA_DIRNAME = ".chromadb"
DEFAULT_COLLECTION_NAME = "rag_chunks"

def _get_client(persist_directory: Path) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(persist_directory))

def search(
    query: str,
    repo_root: Path,
    top_k: int = 5,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> List[Tuple[float, dict]]:
    """
    Search the Chroma collection using an embedding of the query.
    Returns list of (score, metadata+document) tuples. Score is distance (smaller better)
    but Chroma often returns similarity-like distances depending on config; we pass them through.
    """
    chroma_dir = repo_root / DEFAULT_CHROMA_DIRNAME
    client = _get_client(chroma_dir)

    collection = client.get_collection(collection_name)

    model = SentenceTransformer(MODEL_NAME)
    q_emb = model.encode([query], convert_to_numpy=True)
    # normalize
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    q_list = q_emb.astype(float).tolist()

    results = collection.query(
    query_embeddings=q_list,
    n_results=top_k,
    include=["metadatas", "documents", "distances"],
)

    out = []
    # results format: dict with lists under keys: ids, documents, metadatas, distances
    if not results or "distances" not in results or len(results["distances"]) == 0:
        return out

    # results["distances"][0] list of floats; documents/metadatas similarly lists
    dists = results["distances"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    ids = results.get("ids", [[]])[0]  # may be missing in some versions

    for idx, (score, doc, meta) in enumerate(zip(dists, docs, metas)):
        # Prefer returned ids, fall back to the id stored in metadata, else a placeholder
        _id = ids[idx] if idx < len(ids) else (meta.get("id") if isinstance(meta, dict) else None)

        item = {
            "id": _id,
            "text": doc,
            "metadata": meta,
        }
        out.append((float(score), item))

        return out

def pretty_print(results):
    for score, item in results:
        meta = item.get("metadata", {})
        src = meta.get("source_path", "<unknown>")
        chunk_type = meta.get("chunk_type", "unknown")
        start_line = meta.get("start_line")
        end_line = meta.get("end_line")
        cell_index = meta.get("cell_index")

        loc = ""
        if start_line is not None:
            loc = f":{start_line}-{end_line}"
        elif cell_index is not None:
            loc = f" [cell {cell_index}]"

        print("\n" + "=" * 80)
        print(f"{chunk_type} | score={score:.4f}")
        print(f"{src}{loc}")
        if meta:
            print(f"meta: {meta}")
        print("-" * 80)
        print(item["text"][:1500])
