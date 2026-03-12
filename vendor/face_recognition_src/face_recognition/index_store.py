from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .utils import dot_product, ensure_parent_dir


class SearchMatch:
    def __init__(self, score: Optional[float] = None, faiss_id: Optional[int] = None) -> None:
        self.score = score
        self.faiss_id = faiss_id


class SimilarityIndexStore:
    def __init__(self, index_path: Path, dimension: int = 512) -> None:
        self.index_path = index_path
        self.dimension = dimension
        ensure_parent_dir(self.index_path)
        self._backend = self._create_backend(self.index_path, dimension)

    @property
    def committed_count(self) -> int:
        return self._backend.ntotal

    def ensure_consistent(self, metadata_store: object) -> None:
        expected = metadata_store.count_embeddings()
        if self._backend.ntotal != expected:
            self.rebuild_from_store(metadata_store)

    def rebuild_from_store(self, metadata_store: object) -> None:
        backend = self._create_empty_backend(self.dimension)
        vectors: List[Sequence[float]] = []
        for _, embedding in metadata_store.iter_embeddings_ordered():
            vectors.append(embedding)
        if vectors:
            backend.add(vectors)
        backend.save(self.index_path)
        self._backend = backend

    def search(
        self,
        embedding: Sequence[float],
        pending_embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> SearchMatch:
        best = SearchMatch()
        scores, ids = self._backend.search(embedding, 1)
        if scores and ids and ids[0] >= 0:
            best = SearchMatch(score=scores[0], faiss_id=ids[0])

        base_id = self._backend.ntotal
        for offset, pending in enumerate(pending_embeddings or ()):
            score = dot_product(embedding, pending)
            if best.score is None or score > best.score:
                best = SearchMatch(score=score, faiss_id=base_id + offset)
        return best

    def persist_pending(self, pending_embeddings: Sequence[Sequence[float]]) -> None:
        if not pending_embeddings:
            return
        self._backend.add(pending_embeddings)
        self._backend.save(self.index_path)

    def _create_backend(self, index_path: Path, dimension: int) -> "_BackendBase":
        backend = self._create_empty_backend(dimension)
        if not index_path.exists():
            return backend
        try:
            return backend.load(index_path, dimension)
        except Exception:
            return self._create_empty_backend(dimension)

    def _create_empty_backend(self, dimension: int) -> "_BackendBase":
        faiss_backend = _FaissBackend.try_create(dimension)
        if faiss_backend is not None:
            return faiss_backend
        return _PythonIndexBackend(dimension)


class _BackendBase:
    ntotal: int

    def add(self, embeddings: Sequence[Sequence[float]]) -> None:
        raise NotImplementedError

    def load(self, path: Path, dimension: int) -> "_BackendBase":
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError

    def search(self, embedding: Sequence[float], top_k: int) -> Tuple[List[float], List[int]]:
        raise NotImplementedError


class _PythonIndexBackend(_BackendBase):
    def __init__(self, dimension: int, vectors: Optional[Sequence[Sequence[float]]] = None) -> None:
        self.dimension = dimension
        self.vectors = [tuple(float(value) for value in vector) for vector in (vectors or ())]

    @property
    def ntotal(self) -> int:
        return len(self.vectors)

    def add(self, embeddings: Sequence[Sequence[float]]) -> None:
        for embedding in embeddings:
            vector = tuple(float(value) for value in embedding)
            if len(vector) != self.dimension:
                raise ValueError(
                    "expected embedding dimension {}, got {}".format(
                        self.dimension, len(vector)
                    )
                )
            self.vectors.append(vector)

    def load(self, path: Path, dimension: int) -> "_PythonIndexBackend":
        payload = json.loads(path.read_text(encoding="utf-8"))
        vectors = payload.get("vectors", [])
        stored_dimension = int(payload.get("dimension", dimension))
        return _PythonIndexBackend(stored_dimension, vectors=vectors)

    def save(self, path: Path) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps({"dimension": self.dimension, "vectors": self.vectors}),
            encoding="utf-8",
        )
        tmp_path.replace(path)

    def search(self, embedding: Sequence[float], top_k: int) -> Tuple[List[float], List[int]]:
        if not self.vectors:
            return [], []
        query = tuple(float(value) for value in embedding)
        scored = sorted(
            ((dot_product(query, vector), index) for index, vector in enumerate(self.vectors)),
            key=lambda item: item[0],
            reverse=True,
        )
        top = scored[:top_k]
        return [score for score, _ in top], [index for _, index in top]


class _FaissBackend(_BackendBase):
    def __init__(self, dimension: int, faiss_module: object, index: object) -> None:
        self.dimension = dimension
        self._faiss = faiss_module
        self._index = index

    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal)

    @classmethod
    def try_create(cls, dimension: int) -> Optional["_FaissBackend"]:
        try:
            import faiss  # type: ignore
            import numpy as np  # type: ignore
        except ImportError:
            return None

        index = faiss.IndexFlatIP(dimension)
        backend = cls(dimension, faiss, index)
        backend._np = np
        return backend

    def add(self, embeddings: Sequence[Sequence[float]]) -> None:
        if not embeddings:
            return
        matrix = self._np.asarray(embeddings, dtype="float32")
        self._index.add(matrix)

    def load(self, path: Path, dimension: int) -> "_FaissBackend":
        index = self._faiss.read_index(str(path))
        backend = _FaissBackend(dimension, self._faiss, index)
        backend._np = self._np
        return backend

    def save(self, path: Path) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        self._faiss.write_index(self._index, str(tmp_path))
        tmp_path.replace(path)

    def search(self, embedding: Sequence[float], top_k: int) -> Tuple[List[float], List[int]]:
        if self.ntotal == 0:
            return [], []
        query = self._np.asarray([embedding], dtype="float32")
        scores, ids = self._index.search(query, top_k)
        return scores[0].tolist(), ids[0].tolist()
