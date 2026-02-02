# prototype_files/parse_ipynb.py
from __future__ import annotations

from pathlib import Path
from typing import List

import nbformat

from .config import Chunk, infer_doc_fields


def parse_notebook(path: Path) -> List[Chunk]:
    """
    Parse .ipynb into cell-level chunks (code + markdown, non-empty).
    Uses unified metadata fields and loc_kind="cell".
    """
    repo_root = Path(__file__).resolve().parents[1]
    doc_type, doc_dir, doc_rank_hint = infer_doc_fields(repo_root, path)

    try:
        nb = nbformat.read(path, as_version=4)
    except Exception:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        return [Chunk(
            id=f"{path}::notebook_raw",
            text=txt,
            source_path=str(path),
            chunk_type="ipynb_cell",
            doc_type=doc_type,
            doc_dir=doc_dir,
            doc_rank_hint=doc_rank_hint,
            symbol_name="",
            loc_kind="none",
            metadata={"note": "nbformat_read_failed"},
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
            doc_type=doc_type,
            doc_dir=doc_dir,
            doc_rank_hint=doc_rank_hint,
            symbol_name="",
            loc_kind="cell",
            loc_start=i,
            loc_end=i,
            metadata={"cell_type": cell_type},
        ))

    return chunks
