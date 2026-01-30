# rag_prototype/build_index.py
from __future__ import annotations
from pathlib import Path
from typing import List

from .ingest import iter_repo_files
from .parse_py import parse_python_file
from .parse_ipynb import parse_notebook
from .index import create_collection_and_index
from .config import Chunk

def main():
    repo_root = Path(__file__).resolve().parent.parent
    chroma_dir = repo_root / ".chromadb"

    chunks: List[Chunk] = []
    for path in iter_repo_files(repo_root):
        suffix = path.suffix.lower()
        if suffix == ".py":
            chunks.extend(parse_python_file(path))
        elif suffix == ".ipynb":
            chunks.extend(parse_notebook(path))
        elif suffix == ".md":
            text = path.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                chunks.append(Chunk(
                    id=f"{path}::md",
                    text=text,
                    source_path=str(path),
                    chunk_type="md_section",
                    metadata={"filename": path.name},
                ))

    print(f"Collected {len(chunks)} chunks. Building Chroma index…")
    create_collection_and_index(chunks, chroma_dir, collection_name="rag_chunks")
    print(f"Index built and persisted to: {chroma_dir}")

if __name__ == "__main__":
    main()
