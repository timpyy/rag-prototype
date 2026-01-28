from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Dict, Any

#Define a simple "Chunk" Data Model

ChunkType = Literal["py_function", "py_class", "py_module", "ipynb_cell", "md_section", "text"]

@dataclass(frozen = True)
class Chunk:
    id: str
    text: str
    source_path: str
    chunk_type: ChunkType
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    cell_index: Optional[int] = None
    metadata: Dict[str, Any] = None
