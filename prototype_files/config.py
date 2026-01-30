from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any, Union

ChunkType = Literal["py_function", "py_class", "py_module", "ipynb_cell", "md_section", "text"]

Scalar = Union[str, int, float, bool, None]

@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source_path: str
    chunk_type: ChunkType
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    cell_index: Optional[int] = None
    metadata: Dict[str, Any] = None

    def to_metadata(self) -> Dict[str, Any]:
        m = {
            "id": self.id,
            "source_path": self.source_path,
            "chunk_type": self.chunk_type,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "cell_index": self.cell_index,
        }

        if self.metadata:
            for k, v in self.metadata.items():
                key = k if k not in m else f"meta_{k}"
                if isinstance(v, (str, int, float, bool)):
                    m[key] = v
                else:
                    if v is None:
                        continue
                    m[key] = str(v)

        m = {k: v for k, v in m.items() if v is not None}

        return m
