from __future__ import annotations
from pathlib import Path
from typing import Iterable

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".idea", ".pytest_cache", "data"} #Skipping big data directories
ALLOW_EXT = {".py", ".ipynb", ".md"}

def iter_repo_files(repo_root: Path) -> Iterable[Path]:
    for p in repo_root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() in ALLOW_EXT:
            yield p