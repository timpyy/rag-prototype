# rag_prototype/index.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np

from sentence_transformers import SentenceTransformer
import chromadb

from .config import Chunk

# Embedding model
MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good for prototypes

# Default Chroma persistent directory (under repo root)
DEFAULT_CHROMA_DIRNAME = ".chromadb"

def get_chroma_client(persist_directory: Path) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(persist_directory))

def build_embeddings(chunks: List[Chunk], model_name: str = MODEL_NAME) -> List[List[float]]:
    """
    Compute embeddings for a list of chunk texts using sentence-transformers.
    Returns a list of lists (float).
    """
    model = SentenceTransformer(model_name)
    texts = [c.text for c in chunks]
    # convert_to_numpy so we can normalize and convert to python lists
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
    Create (or connect to) a Chroma collection and index chunks with embeddings and metadata.
    This will persist to chroma_dir.
    """
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = get_chroma_client(chroma_dir)

    # create or get collection
    if collection_name in [c.name for c in client.list_collections()]:
        collection = client.get_collection(collection_name)
        # optionally clear existing items to reindex fresh
        try:
            collection.delete()  # remove old collection
        except Exception:
            pass

    collection = client.create_collection(name=collection_name)

    # create embeddings
    print(f"Computing embeddings for {len(chunks)} chunks using {MODEL_NAME} ...")
    embeddings = build_embeddings(chunks, model_name=MODEL_NAME)

    ids = [c.id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [c.to_metadata() for c in chunks]

    # add to collection (Chroma will store embeddings + docs + metadata)
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
