# prototype_files/parse_py.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider, CodeRange

from .config import Chunk, infer_doc_fields


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
    return "\n".join(lines[start_line - 1 : end_line])


@dataclass
class _DefRecord:
    kind: str  # "py_function" or "py_class"
    name: str
    start_line: int
    end_line: int
    text: str


class _TopLevelDefCollector(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, source: str) -> None:
        self.source = source
        self.records: List[_DefRecord] = []
        self._depth = 0

    def visit_Module(self, node: cst.Module) -> Optional[bool]:
        self._depth = 0
        return True

    def visit_IndentedBlock(self, node: cst.IndentedBlock) -> Optional[bool]:
        self._depth += 1
        return True

    def leave_IndentedBlock(self, node: cst.IndentedBlock) -> None:
        self._depth -= 1

    def visit_SimpleStatementSuite(self, node: cst.SimpleStatementSuite) -> Optional[bool]:
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
    LibCST-based Python parser producing:
      - module chunk
      - top-level function chunks
      - top-level class chunks

    Uses unified metadata fields: doc_type/doc_dir/doc_rank_hint/symbol_name and loc_*.
    """
    repo_root = Path(__file__).resolve().parents[1]
    doc_type, doc_dir, doc_rank_hint = infer_doc_fields(repo_root, path)

    source = _read_text(path)
    lines = source.splitlines()

    chunks: List[Chunk] = [
        Chunk(
            id=f"{path}::module",
            text=source,
            source_path=str(path),
            chunk_type="py_module",
            doc_type=doc_type,
            doc_dir=doc_dir,
            doc_rank_hint=doc_rank_hint,
            symbol_name="",
            loc_kind="lines",
            loc_start=1,
            loc_end=len(lines) if lines else 1,
            metadata={"parser": "libcst"},
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
                    doc_type=doc_type,
                    doc_dir=doc_dir,
                    doc_rank_hint=doc_rank_hint,
                    symbol_name=rec.name,
                    loc_kind="lines",
                    loc_start=rec.start_line,
                    loc_end=rec.end_line,
                    metadata={"parser": "libcst"},
                ))
            elif rec.kind == "py_class":
                chunks.append(Chunk(
                    id=f"{path}::class::{rec.name}::{rec.start_line}-{rec.end_line}",
                    text=rec.text,
                    source_path=str(path),
                    chunk_type="py_class",
                    doc_type=doc_type,
                    doc_dir=doc_dir,
                    doc_rank_hint=doc_rank_hint,
                    symbol_name=rec.name,
                    loc_kind="lines",
                    loc_start=rec.start_line,
                    loc_end=rec.end_line,
                    metadata={"parser": "libcst"},
                ))

        return chunks

    except Exception:
        # Fallback: module only
        return [Chunk(
            id=f"{path}::module",
            text=source,
            source_path=str(path),
            chunk_type="py_module",
            doc_type=doc_type,
            doc_dir=doc_dir,
            doc_rank_hint=doc_rank_hint,
            symbol_name="",
            loc_kind="lines",
            loc_start=1,
            loc_end=len(lines) if lines else 1,
            metadata={"parser": "libcst", "note": "parse_failed_fallback"},
        )]
