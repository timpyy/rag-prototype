# prototype_files/parse_py.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider, CodeRange

from .config import Chunk


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _slice_source_by_range(source: str, code_range: CodeRange) -> str:
    """
    Extract exact source snippet using LibCST CodeRange (1-based line, 0-based column).
    We do a line-based slice because it is robust and good enough for chunking.
    """
    lines = source.splitlines()
    start_line = max(1, code_range.start.line)
    end_line = min(len(lines), code_range.end.line)

    # LibCST end position is inclusive in terms of "this node ends on this line",
    # but columns are not used here; we include whole lines for context.
    return "\n".join(lines[start_line - 1 : end_line])


@dataclass
class _DefRecord:
    kind: str  # "py_function" or "py_class"
    name: str
    start_line: int
    end_line: int
    text: str


class _TopLevelDefCollector(cst.CSTVisitor):
    """
    Collect top-level function and class definitions with accurate positions.

    Important: We only collect definitions that are direct children of the module body.
    This avoids collecting nested defs inside functions/classes.
    """
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, source: str) -> None:
        self.source = source
        self.records: List[_DefRecord] = []
        self._depth = 0  # module body depth

    def visit_Module(self, node: cst.Module) -> Optional[bool]:
        self._depth = 0
        return True

    def visit_IndentedBlock(self, node: cst.IndentedBlock) -> Optional[bool]:
        # entering a block increases depth; top-level defs are only at depth 0
        self._depth += 1
        return True

    def leave_IndentedBlock(self, node: cst.IndentedBlock) -> None:
        self._depth -= 1

    def visit_SimpleStatementSuite(self, node: cst.SimpleStatementSuite) -> Optional[bool]:
        # one-line suite also increases depth
        self._depth += 1
        return True

    def leave_SimpleStatementSuite(self, node: cst.SimpleStatementSuite) -> None:
        self._depth -= 1

    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        if self._depth == 0:
            r = self.get_metadata(PositionProvider, node)
            text = _slice_source_by_range(self.source, r)
            self.records.append(_DefRecord(
                kind="py_function",
                name=node.name.value,
                start_line=r.start.line,
                end_line=r.end.line,
                text=text,
            ))
        # Still traverse children? Not needed for top-level collection; returning False is faster.
        return False

    def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
        if self._depth == 0:
            r = self.get_metadata(PositionProvider, node)
            text = _slice_source_by_range(self.source, r)
            self.records.append(_DefRecord(
                kind="py_class",
                name=node.name.value,
                start_line=r.start.line,
                end_line=r.end.line,
                text=text,
            ))
        return False


def parse_python_file(path: Path) -> List[Chunk]:
    """
    LibCST-based parser:
      - Preserves formatting + comments via exact source slicing using positions.
      - Extracts: module chunk + top-level function chunks + top-level class chunks.
    """
    source = _read_text(path)
    lines = source.splitlines()

    # Always include module chunk (useful for broad questions & top-level script logic)
    chunks: List[Chunk] = [
        Chunk(
            id=f"{path}::module",
            text=source,
            source_path=str(path),
            chunk_type="py_module",
            start_line=1,
            end_line=len(lines) if lines else 1,
            metadata={"filename": path.name, "parser": "libcst"},
        )
    ]

    try:
        module = cst.parse_module(source)
        wrapper = MetadataWrapper(module)
        collector = _TopLevelDefCollector(source=source)
        wrapper.visit(collector)

        for rec in collector.records:
            if rec.kind == "py_function":
                chunks.append(Chunk(
                    id=f"{path}::func::{rec.name}::{rec.start_line}-{rec.end_line}",
                    text=rec.text,
                    source_path=str(path),
                    chunk_type="py_function",
                    start_line=rec.start_line,
                    end_line=rec.end_line,
                    metadata={"symbol": rec.name, "parser": "libcst"},
                ))
            elif rec.kind == "py_class":
                chunks.append(Chunk(
                    id=f"{path}::class::{rec.name}::{rec.start_line}-{rec.end_line}",
                    text=rec.text,
                    source_path=str(path),
                    chunk_type="py_class",
                    start_line=rec.start_line,
                    end_line=rec.end_line,
                    metadata={"symbol": rec.name, "parser": "libcst"},
                ))

        return chunks

    except Exception:
        # Fallback: if LibCST fails for any reason, return module chunk only
        chunks[0] = Chunk(
            id=f"{path}::module",
            text=source,
            source_path=str(path),
            chunk_type="py_module",
            start_line=1,
            end_line=len(lines) if lines else 1,
            metadata={"filename": path.name, "parser": "libcst", "note": "parse_failed_fallback"},
        )
        return chunks
