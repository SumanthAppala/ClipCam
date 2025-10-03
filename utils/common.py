import re
from pathlib import Path
from typing import Iterable, List, TypeVar

T = TypeVar("T")

def slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def chunks(lst: List[T], n: int) -> Iterable[List[T]]:
    for i in range(0, len(lst), n):
        yield lst[i:i+n]
