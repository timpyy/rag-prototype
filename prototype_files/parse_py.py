from __future__ import annotations
import ast
from pathlib import Path
from typing import List
from .config import Chunk

def _get_source_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()

def parse_python_file(path: Path) -> List[Chunk]:
    lines = _get_source_lines(path)
    src = "\n".join(lines)

    try:
        # Parse into an AST (Abstract Syntax Tree)
        # This is Python's built-in parser. It builds a tree representing the structure of the code
        tree = ast.parse(src)
    except SyntaxError:
        # If parsing fails, treat whole file as one chunk
        return [Chunk(
            id=f"{path}::module",
            text=src,
            source_path=str(path),
            chunk_type="py_module",
            start_line=1,
            end_line=len(lines),
            metadata={"note": "syntax_error_fallback"},
        )]

    chunks: List[Chunk] = []

    chunks.append(Chunk(
        id=f"{path}::module",
        text=src,
        source_path=str(path),
        chunk_type="py_module",
        start_line=1,
        end_line=len(lines),
        metadata={"filename": path.name},
    ))

    # Walk through the top-level nodes and extract functions/classes
    # Create a Chunk object for each extracted unit
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = getattr(node, "lineno", 1)
            end = getattr(node, "end_lineno", start)
            text = "\n".join(lines[start-1:end])
            chunks.append(Chunk(
                id=f"{path}::func::{node.name}::{start}-{end}",
                text=text,
                source_path=str(path),
                chunk_type="py_function",
                start_line=start,
                end_line=end,
                metadata={"symbol": node.name},
            ))

        elif isinstance(node, ast.ClassDef):
            start = getattr(node, "lineno", 1)
            end = getattr(node, "end_lineno", start)
            text = "\n".join(lines[start-1:end])
            chunks.append(Chunk(
                id=f"{path}::class::{node.name}::{start}-{end}",
                text=text,
                source_path=str(path),
                chunk_type="py_class",
                start_line=start,
                end_line=end,
                metadata={"symbol": node.name},
            ))

        return chunks
