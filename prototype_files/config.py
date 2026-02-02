# prototype_files/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

ChunkType = Literal["py_function", "py_class", "py_module", "ipynb_cell", "md_section", "text"]
LocKind = Literal["lines", "cell", "none"]

Scalar = Union[str, int, float, bool]


def _is_scalar(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool))


@dataclass(frozen=True)
class Chunk:
    # Core identity / payload
    id: str
    text: str
    source_path: str
    chunk_type: ChunkType

    # New doc metadata fields
    doc_type: str                 # e.g., "py", "ipynb", "md"
    doc_dir: str                  # e.g., "src", "notebooks", "docs", "."
    doc_rank_hint: int            # 0 (best) -> higher is less preferred
    symbol_name: str = ""         # function/class name, or "" for module/cell

    # New unified location fields
    loc_kind: LocKind = "none"    # "lines" for .py, "cell" for notebooks
    loc_start: Optional[int] = None
    loc_end: Optional[int] = None

    # Additional optional metadata (must flatten into scalars for Chroma)
    metadata: Optional[Dict[str, Any]] = None

    def to_metadata(self) -> Dict[str, Scalar]:
        """
        Convert to a Chroma-friendly metadata dict.
        Chroma requires a flat dict of scalar values (no dict/list/None).
        """
        m: Dict[str, Scalar] = {
            "id": self.id,
            "source_path": self.source_path,
            "chunk_type": self.chunk_type,
            "doc_type": self.doc_type,
            "doc_dir": self.doc_dir,
            "doc_rank_hint": int(self.doc_rank_hint),
            "symbol_name": self.symbol_name,
            "loc_kind": self.loc_kind,
        }

        # Only include loc_start/loc_end if not None (Chroma can choke on None)
        if self.loc_start is not None:
            m["loc_start"] = int(self.loc_start)
        if self.loc_end is not None:
            m["loc_end"] = int(self.loc_end)

        # Flatten any extra metadata keys; stringify non-scalars; skip None
        if self.metadata:
            for k, v in self.metadata.items():
                key = k if k not in m else f"meta_{k}"

                if v is None:
                    continue
                if _is_scalar(v):
                    m[key] = v  # type: ignore[assignment]
                else:
                    m[key] = str(v)

        return m


def infer_doc_fields(repo_root: Path, file_path: Path) -> tuple[str, str, int]:
    """
    Infer doc_type, doc_dir, doc_rank_hint based on path and extension.

    doc_dir = first path component under repo_root (or "." if file at root)
    doc_rank_hint:
      0 -> code (.py under src/ or similar)
      1 -> notebooks (.ipynb)
      2 -> docs/markdown (.md)
      3 -> everything else
    """
    rel = file_path
    try:
        rel = file_path.resolve().relative_to(repo_root.resolve())
    except Exception:
        # best-effort: fall back to name/parent
        rel = file_path

    parts = rel.parts
    doc_dir = parts[0] if len(parts) > 1 else "."

    ext = file_path.suffix.lower().lstrip(".")
    doc_type = ext if ext else "unknown"

    if doc_type == "py":
        # Prefer src/ style paths
        doc_rank_hint = 0 if doc_dir in {"src", "rag_prototype", "prototype_files"} else 1
    elif doc_type == "ipynb":
        doc_rank_hint = 1
    elif doc_type in {"md", "rst", "txt"}:
        doc_rank_hint = 2
    else:
        doc_rank_hint = 3

    return doc_type, doc_dir, doc_rank_hint
