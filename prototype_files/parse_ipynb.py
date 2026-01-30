# rag_prototype/parse_ipynb.py
from __future__ import annotations
from pathlib import Path
from typing import List
import nbformat
from .config import Chunk

def parse_notebook(path: Path) -> List[Chunk]:
    """
    Parse .ipynb into cell-level chunks.
    Keeps both code and markdown cells (non-empty).
    """
    try:
        nb = nbformat.read(path, as_version=4)
    except Exception:
        # fallback: treat whole file as raw text chunk
        txt = path.read_text(encoding="utf-8", errors="ignore")
        return [Chunk(
            id=f"{path}::notebook_raw",
            text=txt,
            source_path=str(path),
            chunk_type="ipynb_cell",
            metadata={"note": "nbformat_read_failed", "filename": path.name},
        )]

    chunks: List[Chunk] = []
    for i, cell in enumerate(nb.cells):
        cell_type = cell.get("cell_type", "unknown")
        source = cell.get("source", "")
        if not source or not str(source).strip():
            continue
        chunks.append(Chunk(
            id=f"{path}::cell::{i}",
            text=str(source),
            source_path=str(path),
            chunk_type="ipynb_cell",
            cell_index=i,
            metadata={"cell_type": cell_type},
        ))
    return chunks
