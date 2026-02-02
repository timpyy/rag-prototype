# prototype_files/index.py
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

from .config import Chunk

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_CHROMA_DIRNAME = ".chromadb"


def get_chroma_client(persist_directory: Path) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(persist_directory))


def build_embeddings(chunks: List[Chunk], model_name: str = MODEL_NAME) -> List[List[float]]:
    """
    Compute embeddings for a list of chunk texts using sentence-transformers.
    Returns a list of python float lists.
    """
    model = SentenceTransformer(model_name)
    texts = [c.text for c in chunks]

    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Normalize to unit length (cosine similarity works better)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms

    return embs.astype(float).tolist()


def create_collection_and_index(
    chunks: List[Chunk],
    chroma_dir: Path,
    collection_name: str = "rag_chunks",
) -> None:
    """
    Create a Chroma collection and index chunks with embeddings and metadata.
    Persists to chroma_dir.
    """
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = get_chroma_client(chroma_dir)

    # Recreate collection fresh each run (simple + avoids mixing schemas)
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        try:
            client.delete_collection(collection_name)
        except Exception:
            # Older/newer chroma versions sometimes differ; fall back
            try:
                col = client.get_collection(collection_name)
                col.delete()
            except Exception:
                pass

    collection = client.create_collection(name=collection_name)

    print(f"Computing embeddings for {len(chunks)} chunks using {MODEL_NAME} ...")
    embeddings = build_embeddings(chunks, model_name=MODEL_NAME)

    ids = [c.id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [c.to_metadata() for c in chunks]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
