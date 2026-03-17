"""Shared embedding helpers for graph nodes, query recall, and evidence segments."""

from __future__ import annotations

import hashlib
import importlib.util
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from config import (
    GEMINI_API_KEY,
    MEMORY_EMBEDDING_MODEL,
    MEMORY_EMBEDDING_PROVIDER,
    MEMORY_EMBEDDING_TIMEOUT_SECONDS,
    MEMORY_EMBEDDING_VERSION,
    MEMORY_MILVUS_VECTOR_DIM,
    MEMORY_REAL_EMBEDDINGS_ENABLED,
    MODEL_PROVIDER,
    OPENROUTER_API_KEY,
    OPENROUTER_APP_NAME,
    OPENROUTER_BASE_URL,
    OPENROUTER_SITE_URL,
)


DEFAULT_EMBEDDING_MODEL = "textual_stub_v1"
DEFAULT_EMBEDDING_VERSION = "v1"
DEFAULT_EMBEDDING_SOURCE = "textual_stub"

DEFAULT_OPENROUTER_EMBEDDING_MODEL = "openai/text-embedding-3-small"
DEFAULT_GEMINI_EMBEDDING_MODEL = "text-embedding-004"
DEFAULT_FASTEMBED_MODEL = "BAAI/bge-small-zh-v1.5"


def deterministic_vector(text: str, dim: int) -> List[float]:
    if dim <= 0:
        return []
    if not text:
        return [0.0] * dim

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values: List[float] = []
    state = digest
    while len(values) < dim:
        for byte in state:
            values.append((byte / 127.5) - 1.0)
            if len(values) >= dim:
                break
        state = hashlib.sha256(state).digest()

    return _normalize_vector(values[:dim])


def cosine_similarity(left: List[float], right: List[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
    right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
    return dot / (left_norm * right_norm)


def embedding_text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _normalize_vector(values: List[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [round(float(value) / norm, 6) for value in values]


def _fit_vector(values: List[float], dim: int) -> List[float]:
    if dim <= 0:
        return []
    if not values:
        return [0.0] * dim
    if len(values) == dim:
        return _normalize_vector([float(item) for item in values])

    bucketed = [0.0] * dim
    for index, value in enumerate(values):
        bucketed[index % dim] += float(value)
    return _normalize_vector(bucketed)


class EmbeddingProvider:
    """Configurable embedding provider with deterministic fallback."""

    def __init__(
        self,
        *,
        enabled: bool,
        provider: str,
        model: str,
        dim: int,
        version: str,
        timeout_seconds: float,
        openrouter_api_key: str,
        openrouter_base_url: str,
        openrouter_site_url: str,
        openrouter_app_name: str,
        gemini_api_key: str,
    ) -> None:
        self.enabled = enabled
        self.provider = provider
        self.model = model
        self.dim = dim
        self.version = version
        self.timeout_seconds = timeout_seconds
        self.openrouter_api_key = openrouter_api_key
        self.openrouter_base_url = openrouter_base_url.rstrip("/")
        self.openrouter_site_url = openrouter_site_url
        self.openrouter_app_name = openrouter_app_name
        self.gemini_api_key = gemini_api_key
        self._cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._gemini_client: Any = None
        self._fastembed_model: Any = None

    @classmethod
    def from_config(cls, *, dim: Optional[int] = None) -> "EmbeddingProvider":
        resolved_dim = int(dim or MEMORY_MILVUS_VECTOR_DIM)
        provider = cls._resolve_provider()
        model = cls._resolve_model(provider)
        return cls(
            enabled=MEMORY_REAL_EMBEDDINGS_ENABLED,
            provider=provider,
            model=model,
            dim=resolved_dim,
            version=MEMORY_EMBEDDING_VERSION or DEFAULT_EMBEDDING_VERSION,
            timeout_seconds=MEMORY_EMBEDDING_TIMEOUT_SECONDS,
            openrouter_api_key=OPENROUTER_API_KEY or GEMINI_API_KEY,
            openrouter_base_url=OPENROUTER_BASE_URL,
            openrouter_site_url=OPENROUTER_SITE_URL,
            openrouter_app_name=OPENROUTER_APP_NAME,
            gemini_api_key=GEMINI_API_KEY,
        )

    @staticmethod
    def _resolve_provider() -> str:
        explicit = MEMORY_EMBEDDING_PROVIDER
        if explicit in {"stub", "gemini", "openrouter", "fastembed"}:
            return explicit
        if not MEMORY_REAL_EMBEDDINGS_ENABLED:
            return "stub"
        if importlib.util.find_spec("fastembed") is not None:
            return "fastembed"
        if MODEL_PROVIDER == "openrouter" or OPENROUTER_API_KEY or GEMINI_API_KEY.startswith("sk-"):
            return "openrouter"
        if GEMINI_API_KEY:
            return "gemini"
        return "stub"

    @staticmethod
    def _resolve_model(provider: str) -> str:
        configured = MEMORY_EMBEDDING_MODEL.strip()
        if configured:
            return configured
        if provider == "openrouter":
            return DEFAULT_OPENROUTER_EMBEDDING_MODEL
        if provider == "gemini":
            return DEFAULT_GEMINI_EMBEDDING_MODEL
        if provider == "fastembed":
            return DEFAULT_FASTEMBED_MODEL
        return DEFAULT_EMBEDDING_MODEL

    def build_payload(self, text: str, *, task_type: str = "document") -> Dict[str, object]:
        normalized = str(text or "").strip()
        vector, source, model = self.embed_text(normalized, task_type=task_type)
        return {
            "search_text": normalized,
            "embedding": vector,
            "embedding_model": model,
            "embedding_dim": self.dim,
            "embedding_version": self.version,
            "embedding_text_hash": embedding_text_hash(normalized),
            "embedding_updated_at": datetime.now(timezone.utc).isoformat(),
            "embedding_source": source,
        }

    def embed_query(self, text: str) -> List[float]:
        vector, _, _ = self.embed_text(text, task_type="query")
        return vector

    def embed_text(self, text: str, *, task_type: str = "document") -> Tuple[List[float], str, str]:
        normalized = str(text or "").strip()
        cache_key = (task_type, normalized)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return list(cached["embedding"]), str(cached["embedding_source"]), str(cached["embedding_model"])

        if not normalized:
            payload = {
                "embedding": [0.0] * self.dim,
                "embedding_source": DEFAULT_EMBEDDING_SOURCE,
                "embedding_model": DEFAULT_EMBEDDING_MODEL,
            }
            self._cache[cache_key] = payload
            return list(payload["embedding"]), str(payload["embedding_source"]), str(payload["embedding_model"])

        if self.enabled and self.provider == "fastembed":
            try:
                vector = self._embed_fastembed(normalized, task_type=task_type)
                payload = {
                    "embedding": vector,
                    "embedding_source": f"fastembed:{self.model}",
                    "embedding_model": self.model,
                }
                self._cache[cache_key] = payload
                return list(vector), str(payload["embedding_source"]), self.model
            except Exception:
                pass

        if self.enabled and self.provider == "openrouter" and self.openrouter_api_key:
            try:
                vector = self._embed_openrouter(normalized)
                payload = {
                    "embedding": vector,
                    "embedding_source": f"openrouter:{self.model}",
                    "embedding_model": self.model,
                }
                self._cache[cache_key] = payload
                return list(vector), str(payload["embedding_source"]), self.model
            except Exception:
                pass

        if self.enabled and self.provider == "gemini" and self.gemini_api_key:
            try:
                vector = self._embed_gemini(normalized, task_type=task_type)
                payload = {
                    "embedding": vector,
                    "embedding_source": f"gemini:{self.model}",
                    "embedding_model": self.model,
                }
                self._cache[cache_key] = payload
                return list(vector), str(payload["embedding_source"]), self.model
            except Exception:
                pass

        fallback = {
            "embedding": deterministic_vector(normalized, self.dim),
            "embedding_source": DEFAULT_EMBEDDING_SOURCE,
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
        }
        self._cache[cache_key] = fallback
        return list(fallback["embedding"]), str(fallback["embedding_source"]), str(fallback["embedding_model"])

    def _embed_openrouter(self, text: str) -> List[float]:
        import requests

        response = requests.post(
            f"{self.openrouter_base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.openrouter_site_url,
                "X-Title": self.openrouter_app_name,
            },
            json={
                "model": self.model,
                "input": text,
                "dimensions": self.dim,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") or []
        if not data:
            raise RuntimeError("OpenRouter embeddings response missing data")
        embedding = data[0].get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError("OpenRouter embeddings response missing embedding vector")
        return _fit_vector([float(item) for item in embedding], self.dim)

    def _embed_gemini(self, text: str, *, task_type: str) -> List[float]:
        from google import genai
        from google.genai import types

        if self._gemini_client is None:
            self._gemini_client = genai.Client(api_key=self.gemini_api_key)

        response = self._gemini_client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY" if task_type == "query" else "RETRIEVAL_DOCUMENT",
                output_dimensionality=self.dim,
                auto_truncate=True,
            ),
        )

        candidates = []
        if hasattr(response, "embeddings") and response.embeddings:
            candidates = response.embeddings
        elif hasattr(response, "embedding") and response.embedding:
            candidates = [response.embedding]

        if not candidates:
            raise RuntimeError("Gemini embeddings response missing embeddings")

        values = getattr(candidates[0], "values", None)
        if values is None and isinstance(candidates[0], dict):
            values = candidates[0].get("values")
        if not isinstance(values, list):
            raise RuntimeError("Gemini embeddings response missing vector values")
        return _fit_vector([float(item) for item in values], self.dim)

    def _embed_fastembed(self, text: str, *, task_type: str) -> List[float]:
        from fastembed import TextEmbedding

        if self._fastembed_model is None:
            self._fastembed_model = TextEmbedding(model_name=self.model)

        if task_type == "query":
            vectors = list(self._fastembed_model.query_embed([text]))
        else:
            vectors = list(self._fastembed_model.embed([text]))

        if not vectors:
            raise RuntimeError("FastEmbed response missing vectors")
        return _fit_vector(vectors[0].tolist(), self.dim)


def build_embedding_payload(
    text: str,
    *,
    dim: int,
    model: str = DEFAULT_EMBEDDING_MODEL,
    version: str = DEFAULT_EMBEDDING_VERSION,
    embedder: Optional[EmbeddingProvider] = None,
    task_type: str = "document",
) -> Dict[str, object]:
    normalized = str(text or "").strip()
    if embedder is None:
        return {
            "search_text": normalized,
            "embedding": deterministic_vector(normalized, dim),
            "embedding_model": model,
            "embedding_dim": dim,
            "embedding_version": version,
            "embedding_text_hash": embedding_text_hash(normalized),
            "embedding_updated_at": datetime.now(timezone.utc).isoformat(),
            "embedding_source": DEFAULT_EMBEDDING_SOURCE,
        }
    payload = embedder.build_payload(normalized, task_type=task_type)
    payload["embedding_dim"] = dim
    payload["embedding_version"] = version
    return payload
