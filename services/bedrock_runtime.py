"""
Bedrock runtime helpers for Bedrock-native VLM/LLM calls.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import os
from typing import Any, Dict, List, Optional

from config import BEDROCK_REGION, BEDROCK_REQUEST_TIMEOUT_SECONDS


def build_bedrock_client(region_name: Optional[str] = None):
    import boto3
    from botocore.config import Config as BotoConfig

    bearer_token = (
        os.getenv("AWS_BEARER_TOKEN_BEDROCK")
        or os.getenv("BEDROCK_API_KEY")
        or os.getenv("bedrock_key")
    )
    if bearer_token and not os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = bearer_token

    timeout = max(BEDROCK_REQUEST_TIMEOUT_SECONDS, 30)
    return boto3.client(
        "bedrock-runtime",
        region_name=region_name or BEDROCK_REGION,
        config=BotoConfig(
            read_timeout=timeout,
            connect_timeout=min(timeout, 30),
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )


def build_bedrock_management_client(region_name: Optional[str] = None):
    import boto3
    from botocore.config import Config as BotoConfig

    bearer_token = (
        os.getenv("AWS_BEARER_TOKEN_BEDROCK")
        or os.getenv("BEDROCK_API_KEY")
        or os.getenv("bedrock_key")
    )
    if bearer_token and not os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = bearer_token

    timeout = max(BEDROCK_REQUEST_TIMEOUT_SECONDS, 30)
    return boto3.client(
        "bedrock",
        region_name=region_name or BEDROCK_REGION,
        config=BotoConfig(
            read_timeout=timeout,
            connect_timeout=min(timeout, 30),
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )


def _has_bedrock_credentials() -> bool:
    return bool(
        os.getenv("AWS_BEARER_TOKEN_BEDROCK")
        or os.getenv("BEDROCK_API_KEY")
        or os.getenv("bedrock_key")
        or os.getenv("AWS_ACCESS_KEY_ID")
        or os.getenv("AWS_PROFILE")
    )


def _parse_foundation_model_id(model_arn: str) -> Optional[str]:
    if not model_arn or "foundation-model/" not in model_arn:
        return None
    return model_arn.split("foundation-model/", 1)[1].strip() or None


def _profile_priority(profile_id: str) -> tuple[int, str]:
    if profile_id.startswith("global."):
        return (0, profile_id)
    if profile_id.startswith("apac."):
        return (1, profile_id)
    return (2, profile_id)


@lru_cache(maxsize=8)
def _bedrock_catalog(region_name: str) -> Dict[str, Any]:
    empty_catalog: Dict[str, Any] = {
        "model_ids": set(),
        "profiles_by_model": {},
        "profile_ids": set(),
    }
    if not _has_bedrock_credentials():
        return empty_catalog

    try:
        client = build_bedrock_management_client(region_name)
        model_ids = {
            str(item.get("modelId") or "").strip()
            for item in client.list_foundation_models().get("modelSummaries", [])
            if str(item.get("modelId") or "").strip()
        }
        profiles_by_model: Dict[str, List[str]] = defaultdict(list)
        profile_ids = set()
        for item in client.list_inference_profiles().get("inferenceProfileSummaries", []):
            profile_id = str(item.get("inferenceProfileId") or "").strip()
            if not profile_id:
                continue
            profile_ids.add(profile_id)
            for model in item.get("models", []) or []:
                model_id = _parse_foundation_model_id(str(model.get("modelArn") or ""))
                if model_id:
                    profiles_by_model[model_id].append(profile_id)
        for model_id, profile_list in profiles_by_model.items():
            profiles_by_model[model_id] = sorted(set(profile_list), key=_profile_priority)
        return {
            "model_ids": model_ids,
            "profiles_by_model": dict(profiles_by_model),
            "profile_ids": profile_ids,
        }
    except Exception:
        return empty_catalog


def resolve_bedrock_model_candidates(
    requested_model_ids: List[str],
    region_name: Optional[str] = None,
) -> List[str]:
    normalized_requested = [str(value or "").strip() for value in requested_model_ids if str(value or "").strip()]
    if not normalized_requested:
        return []

    catalog = _bedrock_catalog(region_name or BEDROCK_REGION)
    model_ids = catalog.get("model_ids", set())
    profiles_by_model = catalog.get("profiles_by_model", {})
    profile_ids = catalog.get("profile_ids", set())

    candidates: List[str] = []
    for index, requested in enumerate(normalized_requested):
        resolved: List[str] = []
        if requested in profile_ids:
            resolved.append(requested)
        else:
            profile_matches = list(profiles_by_model.get(requested, []) or [])
            if profile_matches:
                resolved.extend(profile_matches)
            if requested in model_ids:
                resolved.append(requested)
            elif not profile_matches and len(normalized_requested) == 1 and index == 0:
                # If there is only one requested model and nothing else to try,
                # keep it so callers fail loudly instead of silently dropping it.
                resolved.append(requested)
        for candidate in resolved:
            if candidate not in candidates:
                candidates.append(candidate)

    if candidates:
        return candidates
    return normalized_requested


def should_try_next_bedrock_model(exc: Exception) -> bool:
    message = str(exc).lower()
    retryable_markers = (
        "model identifier is invalid",
        "on-demand throughput isn’t supported",
        "inference profile",
        "unsupported countries, regions, or territories",
        "access denied",
        "not allowed",
    )
    return any(marker in message for marker in retryable_markers)


def extract_text_from_converse_response(response: Dict[str, Any]) -> str:
    content = (
        response.get("output", {})
        .get("message", {})
        .get("content", [])
    )
    parts: List[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n".join(parts).strip()


def build_text_message(prompt: str) -> List[Dict[str, Any]]:
    return [{"role": "user", "content": [{"text": prompt}]}]


def build_image_message(prompt: str, image_data: bytes, mime_type: str) -> List[Dict[str, Any]]:
    image_format = (mime_type or "image/jpeg").split("/")[-1].lower()
    if image_format == "jpg":
        image_format = "jpeg"
    return [
        {
            "role": "user",
            "content": [
                {"text": prompt},
                {
                    "image": {
                        "format": image_format,
                        "source": {"bytes": image_data},
                    }
                },
            ],
        }
    ]


def build_inference_config(*, temperature: float, max_tokens: int, top_p: float | None = 0.9) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "temperature": temperature,
        "maxTokens": max_tokens,
    }
    if top_p is not None:
        config["topP"] = top_p
    return config
