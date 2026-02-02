# rag_prototype/ingest.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".idea",
    ".pytest_cache",
    "data",
    ".chromadb",
    "prototype_files",
    ".tree-sitter-build",
    "tree-sitter-python",
    "bindings",
    "tests",
    "Evaluation_Results"
}
ALLOW_EXT = {".py", ".ipynb", ".md"}

def iter_repo_files(repo_root: Path) -> Iterable[Path]:
    """
    Walk repository and yield candidate files to index.
    Skips typical virtualenv / git / caches and the chromadb folder.
    """
    for p in repo_root.rglob("*"):
        if not p.is_file():
            continue
        # skip hidden/system folders
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() in ALLOW_EXT:
            yield p
