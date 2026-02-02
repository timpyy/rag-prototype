# prototype_files/parse_py.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from tree_sitter import Language, Parser
import tree_sitter_python  # pip install tree-sitter-python

from .config import Chunk

PY_FUNC_NODE = "function_definition"
PY_CLASS_NODE = "class_definition"


def _get_parser() -> Parser:
    """
    Create a Tree-sitter Parser for Python using the prebuilt tree-sitter-python package.
    This avoids compiling a shared library manually.
    """
    # tree_sitter_python.language() returns a PyCapsule / pointer to the language
    py_lang = Language(tree_sitter_python.language())
    parser = Parser(py_lang)
    return parser


def _slice_source_by_points(source: str, start_row: int, end_row: int) -> str:
    """
    Slice source lines using 0-based row indices from Tree-sitter start_point/end_point.
    Include full lines from start_row to end_row inclusive for readability/context.
    """
    lines = source.splitlines()
    if not lines:
        return ""
    start_row = max(0, start_row)
    end_row = min(len(lines) - 1, end_row)
    if start_row > end_row:
        return ""
    return "\n".join(lines[start_row : end_row + 1])


def _identifier_from_node(source: str, node) -> Optional[str]:
    """
    Extract the first identifier within a node (function/class name).
    Tree-sitter-python usually has an identifier field, but we keep it robust.
    """
    # Prefer field lookup if available
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return source[name_node.start_byte:name_node.end_byte].strip() or None

    # Fallback: search direct children for identifier
    for child in node.children:
        if child.type == "identifier":
            return source[child.start_byte:child.end_byte].strip() or None

    return None


def parse_python_file(path: Path) -> List[Chunk]:
    """
    Parse a .py file into chunks using Tree-sitter:
      - module chunk (whole file)
      - top-level functions
      - top-level classes

    Keeps the output schema consistent with AST/LibCST versions.
    """
    source = path.read_text(encoding="utf-8", errors="ignore")
    lines = source.splitlines()

    # Always include module chunk
    chunks: List[Chunk] = [
        Chunk(
            id=f"{path}::module",
            text=source,
            source_path=str(path),
            chunk_type="py_module",
            start_line=1,
            end_line=len(lines) if lines else 1,
            metadata={"filename": path.name, "parser": "tree-sitter"},
        )
    ]

    if not source.strip():
        return chunks

    parser = _get_parser()
    tree = parser.parse(source.encode("utf-8"))
    root = tree.root_node  # typically "module"

    # We only chunk top-level defs: direct children of the module
    for child in root.children:
        if child.type == PY_FUNC_NODE:
            name = _identifier_from_node(source, child) or "<anonymous>"
            start_line = child.start_point[0] + 1
            end_line = child.end_point[0] + 1
            snippet = _slice_source_by_points(source, child.start_point[0], child.end_point[0])

            chunks.append(
                Chunk(
                    id=f"{path}::func::{name}::{start_line}-{end_line}",
                    text=snippet,
                    source_path=str(path),
                    chunk_type="py_function",
                    start_line=start_line,
                    end_line=end_line,
                    metadata={"symbol": name, "parser": "tree-sitter"},
                )
            )

        elif child.type == PY_CLASS_NODE:
            name = _identifier_from_node(source, child) or "<anonymous>"
            start_line = child.start_point[0] + 1
            end_line = child.end_point[0] + 1
            snippet = _slice_source_by_points(source, child.start_point[0], child.end_point[0])

            chunks.append(
                Chunk(
                    id=f"{path}::class::{name}::{start_line}-{end_line}",
                    text=snippet,
                    source_path=str(path),
                    chunk_type="py_class",
                    start_line=start_line,
                    end_line=end_line,
                    metadata={"symbol": name, "parser": "tree-sitter"},
                )
            )

    return chunks
