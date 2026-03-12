from __future__ import annotations

from array import array
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Sequence


def compute_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def iter_image_files(root: Path, extensions: Sequence[str]) -> Iterable[Path]:
    allowed = {ext.lower() for ext in extensions}
    for candidate in sorted(root.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in allowed:
            yield candidate


def l2_normalize(values: Sequence[float]) -> list[float]:
    squared = sum(float(value) * float(value) for value in values)
    if squared <= 0.0:
        return [0.0 for _ in values]
    norm = squared ** 0.5
    return [float(value) / norm for value in values]


def dot_product(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def embedding_to_blob(values: Sequence[float]) -> bytes:
    data = array("f", [float(value) for value in values])
    return data.tobytes()


def blob_to_embedding(blob: bytes) -> list[float]:
    data = array("f")
    data.frombytes(blob)
    return list(data)


def json_safe_path(path: Path) -> str:
    return str(path.expanduser().resolve())
