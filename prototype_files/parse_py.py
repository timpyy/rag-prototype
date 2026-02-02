# prototype_files/parse_py.py
from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Optional

from .config import Chunk, infer_doc_fields


def _get_source_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _get_end_lineno(node: ast.AST, default: int) -> int:
    end = getattr(node, "end_lineno", None)
    if isinstance(end, int) and end >= default:
        return end
    return default


def parse_python_file(path: Path) -> List[Chunk]:
    """
    Parse a .py file into chunks (AST-based):
      - module (whole file)
      - top-level functions
      - top-level classes

    Emits unified metadata fields:
      doc_type, doc_dir, doc_rank_hint, symbol_name
      loc_kind, loc_start, loc_end
    """
    repo_root = Path(__file__).resolve().parents[1]
    doc_type, doc_dir, doc_rank_hint = infer_doc_fields(repo_root, path)

    src = _get_source_text(path)
    lines = src.splitlines()
    n_lines = len(lines) if lines else 1

    try:
        tree = ast.parse(src)
    except SyntaxError:
        # On parse failure, index the whole file as one module chunk
        return [Chunk(
            id=f"{path}::module",
            text=src,
            source_path=str(path),
            chunk_type="py_module",
            doc_type=doc_type,
            doc_dir=doc_dir,
            doc_rank_hint=doc_rank_hint,
            symbol_name="",
            loc_kind="lines",
            loc_start=1,
            loc_end=n_lines,
            metadata={"parser": "ast", "note": "syntax_error_fallback"},
        )]

    chunks: List[Chunk] = []

    # Module chunk (whole file)
    chunks.append(Chunk(
        id=f"{path}::module",
        text=src,
        source_path=str(path),
        chunk_type="py_module",
        doc_type=doc_type,
        doc_dir=doc_dir,
        doc_rank_hint=doc_rank_hint,
        symbol_name="",
        loc_kind="lines",
        loc_start=1,
        loc_end=n_lines,
        metadata={"parser": "ast", "filename": path.name},
    ))

    # Top-level defs
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = getattr(node, "lineno", 1) or 1
            end = _get_end_lineno(node, start)
            snippet = "\n".join(lines[start - 1:end]) if lines else ""

            chunks.append(Chunk(
                id=f"{path}::func::{node.name}::{start}-{end}",
                text=snippet,
                source_path=str(path),
                chunk_type="py_function",
                doc_type=doc_type,
                doc_dir=doc_dir,
                doc_rank_hint=doc_rank_hint,
                symbol_name=node.name,
                loc_kind="lines",
                loc_start=start,
                loc_end=end,
                metadata={"parser": "ast"},
            ))

        elif isinstance(node, ast.ClassDef):
            start = getattr(node, "lineno", 1) or 1
            end = _get_end_lineno(node, start)
            snippet = "\n".join(lines[start - 1:end]) if lines else ""

            chunks.append(Chunk(
                id=f"{path}::class::{node.name}::{start}-{end}",
                text=snippet,
                source_path=str(path),
                chunk_type="py_class",
                doc_type=doc_type,
                doc_dir=doc_dir,
                doc_rank_hint=doc_rank_hint,
                symbol_name=node.name,
                loc_kind="lines",
                loc_start=start,
                loc_end=end,
                metadata={"parser": "ast"},
            ))

    return chunks
