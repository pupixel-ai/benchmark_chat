"""
VLM analysis module.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import re
import time
from typing import Any, Dict, List, Optional

from config import (
    API_PROXY_KEY,
    API_PROXY_MODEL,
    API_PROXY_URL,
    BEDROCK_LLM_MAX_OUTPUT_TOKENS,
    BEDROCK_REGION,
    BEDROCK_VLM_MODEL_FALLBACK,
    BEDROCK_VLM_MAX_OUTPUT_TOKENS,
    BEDROCK_VLM_MODEL_POLICY,
    BEDROCK_VLM_MODEL_PRIMARY,
    GEMINI_API_KEY,
    GEMINI_BASE_URL,
    GEMINI_MODEL,
    MAX_RETRIES,
    OPENROUTER_API_KEY,
    OPENROUTER_APP_NAME,
    OPENROUTER_BASE_URL,
    OPENROUTER_SITE_URL,
    OPENROUTER_VLM_MODEL,
    TASK_VERSION_V0323,
    TASK_VERSION_V0325,
    TASK_VERSION_V0327_DB,
    TASK_VERSION_V0327_DB_QUERY,
    TASK_VERSION_V0327_EXP,
    RETRY_DELAY,
    TASK_VERSION_V0317_HEAVY,
    V0323_OPENROUTER_MODEL,
    V0325_OPENROUTER_VLM_MODEL,
    VLM_PROVIDER,
    VLM_CACHE_PATH,
    VLM_MODEL,
)
from models import Photo
from services.bedrock_runtime import (
    build_bedrock_client,
    build_image_message,
    build_inference_config,
    extract_text_from_converse_response,
    resolve_bedrock_model_candidates,
    should_try_next_bedrock_model,
)
from utils import load_json, save_json


class VLMCallError(RuntimeError):
    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.details = dict(details or {})


class VLMAnalyzer:
    """VLM analyzer with Bedrock, Gemini proxy, and OpenRouter support."""

    _CONTACT_TYPE_ENUM = {
        "kiss",
        "hug",
        "holding_hands",
        "arm_in_arm",
        "selfie_together",
        "shoulder_lean",
        "sitting_close",
        "standing_near",
        "no_contact",
    }
    _SOURCE_TYPE_ALIASES = {
        "camera_photo": "camera_photo",
        "live_photo": "camera_photo",
        "photo": "camera_photo",
        "screenshot": "screenshot",
        "screen_capture": "screenshot",
        "document": "document",
        "id_document": "document",
        "ai_generated_image": "ai_generated_image",
        "ai_generated_or_reference": "ai_generated_image",
        "generated_image": "ai_generated_image",
        "embedded_media": "embedded_media",
        "scanned_or_embedded_media": "embedded_media",
        "reference_media": "reference_media",
        "saved_web_image": "reference_media",
    }

    def __init__(self, cache_path: str = VLM_CACHE_PATH, task_version: str = ""):
        _v032x_set = {TASK_VERSION_V0323, TASK_VERSION_V0325, TASK_VERSION_V0327_EXP, TASK_VERSION_V0327_DB, TASK_VERSION_V0327_DB_QUERY}
        if task_version in _v032x_set and GEMINI_BASE_URL:
            self.provider = "gemini"
            self.model = GEMINI_MODEL
        elif task_version in _v032x_set:
            self.provider = "openrouter"
            if task_version == TASK_VERSION_V0323:
                self.model = V0323_OPENROUTER_MODEL
            else:
                self.model = V0325_OPENROUTER_VLM_MODEL
        else:
            self.provider = VLM_PROVIDER
            self.model = OPENROUTER_VLM_MODEL if VLM_PROVIDER == "openrouter" else VLM_MODEL
        self.use_proxy = self.provider == "proxy"
        self.use_openrouter = self.provider == "openrouter"
        self.use_bedrock = self.provider == "bedrock"
        self.cache_path = cache_path
        self.task_version = task_version
        self.use_heavy_prompt = self.task_version == TASK_VERSION_V0317_HEAVY
        self.results: List[Dict[str, Any]] = []
        self._result_index: Dict[str, int] = {}
        self.requests = None
        self.http_session = None
        self.genai = None
        self.types = None
        self.bedrock_client = None
        self.bedrock_model_candidates: List[str] = []

        if self.use_proxy:
            if not API_PROXY_URL or not API_PROXY_KEY:
                raise ValueError("使用代理服务需要配置 API_PROXY_URL 和 API_PROXY_KEY")
            try:
                import requests
            except ModuleNotFoundError:
                requests = None
            self.requests = requests
            self.http_session = self._build_http_session(requests)
            self.proxy_url = API_PROXY_URL
            self.proxy_key = API_PROXY_KEY
            self.proxy_model = API_PROXY_MODEL
            print(f"[VLM] 使用代理服务: {self.proxy_url}")
        elif self.use_openrouter:
            try:
                import requests
            except ModuleNotFoundError:
                requests = None
            self.requests = requests
            self.http_session = self._build_http_session(requests)
            self.openrouter_api_key = OPENROUTER_API_KEY or GEMINI_API_KEY
            if not self.openrouter_api_key:
                raise ValueError("使用 OpenRouter 需要配置 OPENROUTER_API_KEY 或 GEMINI_API_KEY")
            self.openrouter_base_url = OPENROUTER_BASE_URL.rstrip("/")
            self.openrouter_site_url = OPENROUTER_SITE_URL
            self.openrouter_app_name = OPENROUTER_APP_NAME
            print(f"[VLM] 使用 OpenRouter: {self.model}")
        elif self.use_bedrock:
            self.bedrock_client = build_bedrock_client(BEDROCK_REGION)
            requested_models = [BEDROCK_VLM_MODEL_PRIMARY]
            if BEDROCK_VLM_MODEL_POLICY != "fallback" and BEDROCK_VLM_MODEL_FALLBACK:
                requested_models.append(BEDROCK_VLM_MODEL_FALLBACK)
            if BEDROCK_VLM_MODEL_POLICY == "fallback":
                requested_models = [BEDROCK_VLM_MODEL_FALLBACK]
            self.bedrock_model_candidates = resolve_bedrock_model_candidates(
                requested_models,
                BEDROCK_REGION,
            )
            if self.bedrock_model_candidates:
                self.model = self.bedrock_model_candidates[0]
            print(f"[VLM] 使用 Bedrock: {self.model} @ {BEDROCK_REGION}")
        else:
            from google import genai
            from google.genai import types

            self.genai = genai
            self.types = types
            client_kwargs: Dict[str, Any] = {"api_key": GEMINI_API_KEY}
            if GEMINI_BASE_URL:
                client_kwargs["http_options"] = {"base_url": GEMINI_BASE_URL}
            self.client = genai.Client(**client_kwargs)
            print(f"[VLM] 使用官方 Gemini API{' (代理: ' + GEMINI_BASE_URL + ')' if GEMINI_BASE_URL else ''}")

    def _build_http_session(self, requests_module):
        if requests_module is None:
            return None
        session = requests_module.Session()
        try:
            from requests.adapters import HTTPAdapter

            adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=0)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
        except Exception:
            pass
        return session

    def _coerce_text_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(part for part in parts if part).strip()
        if isinstance(content, dict):
            text = content.get("text")
            if isinstance(text, str):
                return text.strip()
        raise ValueError(f"无法从内容中提取文本: {type(content).__name__}")

    def _truncate_preview(self, value: Any, *, limit: int = 600) -> str:
        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)] + "..."

    def _extract_json_payload(self, raw_text: str) -> Dict[str, Any]:
        text = str(raw_text or "").strip()
        text = text.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "").strip()

        if text.startswith("```"):
            lines = text.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()

        candidates = [text]
        start = min((idx for idx in (text.find("{"), text.find("[")) if idx != -1), default=-1)
        if start != -1:
            candidates.append(text[start:])

        decoder = json.JSONDecoder()
        last_error = None
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                last_error = exc
                try:
                    payload, _end = decoder.raw_decode(candidate)
                    if isinstance(payload, dict):
                        return payload
                    return {"items": payload}
                except json.JSONDecodeError:
                    continue
        if last_error:
            raise last_error
        raise ValueError("无法提取 JSON payload")

    def _normalize_result(self, result: Any) -> Dict[str, Any]:
        if result is None:
            return {}
        if isinstance(result, list):
            if not result:
                return {}
            return self._normalize_result(result[0])
        if isinstance(result, str):
            result = self._extract_json_payload(result)
        elif isinstance(result, dict) and "text" in result and isinstance(result["text"], str):
            result = self._extract_json_payload(result["text"])
        elif not isinstance(result, dict):
            raise ValueError(f"不支持的 VLM 返回类型: {type(result).__name__}")

        normalized = dict(result)
        normalized.setdefault("summary", "")
        normalized.setdefault("scene", {})
        normalized.setdefault("event", {})
        normalized.setdefault("details", [])
        normalized.setdefault("key_objects", [])
        normalized.setdefault("people", [])
        normalized.setdefault("relations", [])
        normalized.setdefault("ocr_hits", [])
        normalized.setdefault("brands", [])
        normalized.setdefault("place_candidates", [])
        normalized.setdefault("route_plan_clues", [])
        normalized.setdefault("transport_clues", [])
        normalized.setdefault("health_treatment_clues", [])
        normalized.setdefault("object_last_seen_clues", [])
        normalized.setdefault("raw_structured_observations", [])
        normalized.setdefault("uncertainty", [])
        if not isinstance(normalized.get("scene"), dict):
            normalized["scene"] = {}
        if not isinstance(normalized.get("event"), dict):
            normalized["event"] = {}
        normalized["details"] = self._normalize_text_list(normalized.get("details"))
        normalized["people"] = self._normalize_people_block(normalized.get("people"))
        normalized["people_details"] = [dict(item) for item in normalized["people"]]
        normalized["relations"] = self._normalize_relations_block(normalized.get("relations"))
        normalized["scene"] = self._normalize_scene_block(
            normalized.get("scene"),
            normalized["details"],
            normalized["relations"],
        )
        normalized["event"] = self._normalize_event_block(
            normalized.get("event"),
            normalized["people"],
            normalized["relations"],
        )
        normalized["key_objects"] = self._derive_key_objects(normalized)
        normalized["ocr_hits"] = self._derive_ocr_hits(normalized)
        normalized["brands"] = self._normalize_text_list(normalized.get("brands"))
        if not normalized["brands"]:
            normalized["brands"] = self._derive_brands(normalized)
        normalized["place_candidates"] = self._derive_place_candidates(normalized)
        normalized["route_plan_clues"] = self._normalize_text_list(normalized.get("route_plan_clues"))
        normalized["transport_clues"] = self._normalize_text_list(normalized.get("transport_clues"))
        normalized["health_treatment_clues"] = self._normalize_text_list(normalized.get("health_treatment_clues"))
        normalized["object_last_seen_clues"] = self._normalize_text_list(normalized.get("object_last_seen_clues"))
        normalized["raw_structured_observations"] = list(normalized.get("raw_structured_observations") or [])
        normalized["uncertainty"] = list(normalized.get("uncertainty") or [])
        normalized["source_type"] = self._normalize_source_type(normalized.get("source_type"))
        return normalized

    def _normalize_source_type(self, value: Any) -> str:
        normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        return self._SOURCE_TYPE_ALIASES.get(normalized, "")

    def _derive_source_type(self, normalized: Dict[str, Any], *, filename: str = "") -> str:
        explicit = self._normalize_source_type(normalized.get("source_type"))
        if explicit:
            return explicit

        filename_text = str(filename or "").strip().lower()
        scene = dict(normalized.get("scene") or {})
        event = dict(normalized.get("event") or {})
        text_pool = " ".join(
            part
            for part in [
                filename_text,
                str(normalized.get("summary") or ""),
                " ".join(str(item) for item in list(normalized.get("details", []) or [])),
                " ".join(str(item) for item in list(normalized.get("ocr_hits", []) or [])),
                " ".join(str(item) for item in list(normalized.get("uncertainty", []) or [])),
                str(scene.get("location_detected") or ""),
                str(scene.get("environment_description") or ""),
                " ".join(str(item) for item in list(scene.get("environment_details", []) or [])),
                str(event.get("activity") or ""),
                str(event.get("social_context") or ""),
            ]
            if part
        ).lower()

        if any(token in filename_text for token in ("screenshot", "screen shot", "截屏", "截图", "screen_capture")):
            return "screenshot"
        if any(
            token in text_pool
            for token in (
                "screen capture",
                "screen shot",
                "screenshot",
                "chat screenshot",
                "屏幕截图",
                "截屏",
                "截图",
            )
        ):
            return "screenshot"
        if any(token in text_pool for token in ("student id", "id card", "passport", "学生证", "身份证", "证件照")):
            return "document"
        if any(
            token in text_pool
            for token in (
                "midjourney",
                "dall-e",
                "dalle",
                "stable diffusion",
                "sdxl",
                "comfyui",
                "flux",
                "ai generated image",
                "ai-generated image",
                "generated illustration",
                "generated portrait",
                "rendered illustration",
                "rendered portrait",
                "digital illustration",
                "ai生成",
                "生成图",
                "ai图",
                "ai 图",
                "ai绘画",
                "数字插画",
                "合成图",
            )
        ):
            return "ai_generated_image"
        if any(
            token in text_pool
            for token in (
                "polaroid",
                "instant photo",
                "printed photo",
                "photo of a photo",
                "screen within screen",
                "photo on screen",
                "相纸",
                "照片中的照片",
                "屏幕中的照片",
            )
        ):
            return "embedded_media"
        if "reference-only" in text_pool or any(
            token in text_pool
            for token in (
                "reference media",
                "inspiration board",
                "lookbook",
                "poster",
                "wallpaper",
                "meme",
                "网图",
                "海报",
                "参考图",
            )
        ):
            return "reference_media"
        return "camera_photo"

    def _coerce_value_text(
        self,
        value: Any,
        *,
        preferred_keys: tuple[str, ...] = ("name", "label", "value", "text", "title", "person_id", "id"),
    ) -> str:
        if value in (None, "", [], {}):
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (int, float, bool)):
            return str(value).strip()
        if isinstance(value, dict):
            for key in preferred_keys:
                text = self._coerce_value_text(value.get(key))
                if text:
                    return text
            return ""
        if isinstance(value, (list, tuple, set)):
            texts = [
                self._coerce_value_text(item, preferred_keys=preferred_keys)
                for item in value
            ]
            texts = [item for item in texts if item]
            return " / ".join(texts)
        return str(value).strip()

    def _normalize_text_list(self, values: Any) -> List[str]:
        if values in (None, "", [], {}):
            return []
        if not isinstance(values, list):
            values = [values]
        normalized: List[str] = []
        seen = set()
        for value in values:
            text = self._coerce_value_text(value)
            if not text or text in seen:
                continue
            seen.add(text)
            normalized.append(text)
        return normalized

    def _normalize_people_block(self, people: Any) -> List[Dict[str, Any]]:
        if not isinstance(people, list):
            return []
        normalized_people: List[Dict[str, Any]] = []
        for item in people:
            if not isinstance(item, dict):
                continue
            person_id = self._coerce_value_text(item.get("person_id"))
            if not person_id:
                continue
            contact_type = str(item.get("contact_type") or "").strip()
            if contact_type not in self._CONTACT_TYPE_ENUM:
                contact_type = "no_contact"
            try:
                confidence = float(item.get("confidence") or 0.0)
            except Exception:
                confidence = 0.0
            normalized_people.append(
                {
                    "person_id": person_id,
                    "appearance": self._coerce_value_text(item.get("appearance")),
                    "clothing": self._coerce_value_text(item.get("clothing")),
                    "activity": self._coerce_value_text(item.get("activity")),
                    "interaction": self._coerce_value_text(item.get("interaction")),
                    "contact_type": contact_type,
                    "expression": self._coerce_value_text(item.get("expression")),
                    "confidence": max(0.0, min(1.0, confidence)),
                }
            )
        return normalized_people

    def _normalize_relations_block(self, relations: Any) -> List[Dict[str, Any]]:
        if not isinstance(relations, list):
            return []
        normalized_relations: List[Dict[str, Any]] = []
        for item in relations:
            if not isinstance(item, dict):
                continue
            subject = self._coerce_value_text(item.get("subject"))
            relation = self._coerce_value_text(item.get("relation"))
            obj = self._coerce_value_text(item.get("object"))
            if not subject or not relation or not obj:
                continue
            try:
                confidence = float(item.get("confidence") or 0.0)
            except Exception:
                confidence = 0.0
            normalized_relations.append(
                {
                    "subject": subject,
                    "relation": relation,
                    "object": obj,
                    "confidence": max(0.0, min(1.0, confidence)),
                }
            )
        return normalized_relations

    def _normalize_scene_block(
        self,
        scene: Any,
        details: List[str],
        relations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        raw_scene = dict(scene or {}) if isinstance(scene, dict) else {}
        environment_details = self._normalize_text_list(raw_scene.get("environment_details"))
        location_detected = self._coerce_value_text(raw_scene.get("location_detected"))
        location_type = self._coerce_value_text(raw_scene.get("location_type")) or "未知"
        environment_description = self._coerce_value_text(raw_scene.get("environment_description"))
        if not environment_description:
            environment_description = "，".join(environment_details[:3]) or location_detected or location_type
        visual_clues = self._normalize_text_list(raw_scene.get("visual_clues"))
        if not visual_clues:
            visual_clues = environment_details[:4]
        relation_objects = [
            endpoint
            for relation in relations
            for endpoint in (relation.get("subject"), relation.get("object"))
            if endpoint and not str(endpoint).startswith("Person_") and endpoint != "【主角】"
        ]
        if not environment_details:
            environment_details = self._normalize_text_list([*relation_objects, *details[:4]])
        layout = dict(raw_scene.get("layout") or {}) if isinstance(raw_scene.get("layout"), dict) else {}
        return {
            "environment_description": environment_description,
            "environment_details": environment_details,
            "location_detected": location_detected,
            "location_type": location_type,
            "visual_clues": visual_clues,
            "weather": self._coerce_value_text(raw_scene.get("weather")),
            "lighting": self._coerce_value_text(raw_scene.get("lighting")),
            "layout": {
                "foreground": self._coerce_value_text(layout.get("foreground")),
                "midground": self._coerce_value_text(layout.get("midground")),
                "background": self._coerce_value_text(layout.get("background")),
            },
        }

    def _normalize_event_block(
        self,
        event: Any,
        people: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        raw_event = dict(event or {}) if isinstance(event, dict) else {}
        interaction = self._coerce_value_text(raw_event.get("interaction"))
        if not interaction:
            interaction_clauses: List[str] = []
            for person in people[:4]:
                person_id = str(person.get("person_id") or "").strip()
                if person.get("interaction"):
                    interaction_clauses.append(f"{person_id}:{person['interaction']}")
                contact_type = str(person.get("contact_type") or "").strip()
                if contact_type and contact_type != "no_contact":
                    interaction_clauses.append(f"{person_id}:{contact_type}")
            for relation in relations[:3]:
                subject = str(relation.get("subject") or "").strip()
                obj = str(relation.get("object") or "").strip()
                if subject.startswith("Person_") or obj.startswith("Person_") or subject == "【主角】" or obj == "【主角】":
                    interaction_clauses.append(f"{subject} {relation.get('relation')} {obj}")
            interaction = "；".join(interaction_clauses[:4])
        return {
            "activity": self._coerce_value_text(raw_event.get("activity")),
            "social_context": self._coerce_value_text(raw_event.get("social_context")),
            "interaction": interaction,
            "mood": self._coerce_value_text(raw_event.get("mood")),
            "story_hints": self._normalize_text_list(raw_event.get("story_hints")),
        }

    def _derive_key_objects(self, normalized: Dict[str, Any]) -> List[str]:
        scene = dict(normalized.get("scene") or {})
        values: List[str] = []
        values.extend(self._normalize_text_list(normalized.get("key_objects")))
        values.extend(self._normalize_text_list(scene.get("environment_details")))
        for relation in list(normalized.get("relations", []) or []):
            for endpoint in (relation.get("subject"), relation.get("object")):
                text = self._coerce_value_text(endpoint)
                if text and not text.startswith("Person_") and text != "【主角】":
                    values.append(text)
        return self._normalize_text_list(values)[:12]

    def _derive_ocr_hits(self, normalized: Dict[str, Any]) -> List[str]:
        explicit_hits = self._normalize_text_list(normalized.get("ocr_hits"))
        if explicit_hits:
            return explicit_hits[:12]
        derived_hits = [
            detail
            for detail in self._normalize_text_list(normalized.get("details"))
            if re.search(r"[\u4e00-\u9fffA-Za-z0-9]{3,}", detail)
        ]
        return derived_hits[:12]

    def _derive_brands(self, normalized: Dict[str, Any]) -> List[str]:
        details = self._normalize_text_list(normalized.get("details"))
        derived = [
            detail
            for detail in details
            if len(detail) <= 24
            and (
                re.search(r"[A-Z][A-Za-z0-9&+._-]{1,}", detail)
                or any(token in detail for token in ("品牌", "Logo", "logo", "App", "APP", "界面"))
            )
        ]
        return derived[:8]

    def _derive_place_candidates(self, normalized: Dict[str, Any]) -> List[str]:
        candidates = self._normalize_text_list(normalized.get("place_candidates"))
        if candidates:
            return candidates[:5]
        scene = dict(normalized.get("scene") or {})
        location_detected = self._coerce_value_text(scene.get("location_detected"))
        return [location_detected] if location_detected else []

    def _infer_reliable_primary_person_id(self, face_db: Dict) -> str | None:
        if not face_db:
            return None

        stats_by_person = []
        for person_id, person in face_db.items():
            photo_count = int(getattr(person, "photo_count", 0) or 0)
            first_seen = getattr(person, "first_seen", None)
            first_seen_value = first_seen.isoformat() if first_seen else ""
            stats_by_person.append((person_id, photo_count, first_seen_value))

        stats_by_person.sort(key=lambda item: (item[1], item[2]), reverse=True)
        top_person_id, top_photo_count, _ = stats_by_person[0]
        if top_photo_count < 2:
            return None

        tied_top_people = [person_id for person_id, photo_count, _ in stats_by_person if photo_count == top_photo_count]
        if len(tied_top_people) > 1:
            return None

        return top_person_id

    def analyze_photo(self, photo: Photo, face_db: Dict, primary_person_id: str = None) -> Dict[str, Any] | None:
        result, _metadata = self.analyze_photo_with_metadata(photo, face_db, primary_person_id)
        return result

    def analyze_photo_with_metadata(
        self,
        photo: Photo,
        face_db: Dict,
        primary_person_id: str = None,
    ) -> tuple[Dict[str, Any] | None, Dict[str, Any]]:
        image_path = (
            getattr(photo, "boxed_path", None)
            or getattr(photo, "annotated_path", None)
            or photo.compressed_path
        )
        started_at = time.perf_counter()
        metadata: Dict[str, Any] = {
            "photo_id": photo.photo_id,
            "filename": photo.filename,
            "retry_count": 0,
            "runtime_seconds": 0.0,
            "provider": self.provider,
            "model": self.model,
        }
        if not image_path:
            photo.processing_errors["vlm"] = "未找到可供 VLM 分析的图片路径"
            metadata["runtime_seconds"] = round(time.perf_counter() - started_at, 4)
            metadata["error"] = photo.processing_errors["vlm"]
            return None, metadata

        if primary_person_id is None:
            primary_person_id = self._infer_reliable_primary_person_id(face_db)

        prompt = self._create_prompt(photo, face_db, primary_person_id)
        try:
            with open(image_path, "rb") as file_obj:
                image_data = file_obj.read()
            mime_type = self._guess_mime_type(image_path)
            metadata["image_path"] = str(image_path)
            metadata["mime_type"] = mime_type
            metadata["prompt_char_count"] = len(prompt)

            result = None
            for attempt in range(MAX_RETRIES):
                try:
                    if self.use_proxy:
                        result = self._analyze_via_proxy(prompt, image_data, mime_type)
                    elif self.use_openrouter:
                        result = self._analyze_via_openrouter(prompt, image_data, mime_type)
                    elif self.use_bedrock:
                        result = self._analyze_via_bedrock(prompt, image_data, mime_type)
                    else:
                        result = self._analyze_via_official_api(prompt, image_data, mime_type)
                    break
                except Exception as exc:
                    if attempt == MAX_RETRIES - 1 or not self._is_retryable_error(exc):
                        raise
                    metadata["retry_count"] += 1
                    delay_seconds = RETRY_DELAY * (2 ** attempt)
                    print(f"[VLM] 可重试错误，{delay_seconds}s 后重试 ({attempt + 1}/{MAX_RETRIES}): {exc}")
                    time.sleep(delay_seconds)

            normalized = self._normalize_result(result)
            photo.vlm_analysis = normalized
            photo.processing_errors.pop("vlm", None)
            metadata["runtime_seconds"] = round(time.perf_counter() - started_at, 4)
            metadata["model"] = self.model
            return normalized, metadata
        except Exception as exc:
            photo.processing_errors["vlm"] = str(exc)
            print(f"警告：VLM分析失败 ({photo.filename}): {exc}")
            metadata["runtime_seconds"] = round(time.perf_counter() - started_at, 4)
            metadata["model"] = self.model
            metadata["error"] = str(exc)
            metadata["error_type"] = type(exc).__name__
            details = dict(getattr(exc, "details", {}) or {})
            for key in (
                "raw_response_preview",
                "response_status_code",
                "parse_error_type",
                "response_provider",
            ):
                if key in details and details.get(key) not in (None, ""):
                    metadata[key] = details.get(key)
            return None, metadata

    def _guess_mime_type(self, image_path: str) -> str:
        guessed, _ = mimetypes.guess_type(image_path)
        return guessed or "image/jpeg"

    def _is_retryable_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        retry_keywords = ["429", "rate limit", "connection", "timeout", "temporarily unavailable", "reset by peer", "throttl", "too many requests"]
        return any(keyword in message for keyword in retry_keywords)

    def _analyze_via_official_api(self, prompt: str, image_data: bytes, mime_type: str) -> Dict[str, Any]:
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                prompt,
                self.types.Part.from_bytes(data=image_data, mime_type=mime_type),
            ],
            config=self.genai.types.GenerateContentConfig(response_mime_type="application/json"),
        )
        raw_text = response.text
        try:
            return self._extract_json_payload(raw_text)
        except Exception as exc:
            raise VLMCallError(
                f"官方 Gemini JSON 解析失败: {exc}",
                details={
                    "raw_response_preview": self._truncate_preview(raw_text),
                    "parse_error_type": type(exc).__name__,
                    "response_provider": "official",
                },
            ) from exc

    def _analyze_via_bedrock(self, prompt: str, image_data: bytes, mime_type: str) -> Dict[str, Any]:
        candidates = self.bedrock_model_candidates or [self.model]
        last_error: Exception | None = None
        for index, model_id in enumerate(candidates):
            try:
                response = self.bedrock_client.converse(
                    modelId=model_id,
                    messages=build_image_message(prompt, image_data, mime_type),
                    inferenceConfig=build_inference_config(
                        temperature=0.1,
                        max_tokens=BEDROCK_VLM_MAX_OUTPUT_TOKENS or BEDROCK_LLM_MAX_OUTPUT_TOKENS,
                    ),
                )
                self.model = model_id
                raw_text = extract_text_from_converse_response(response)
                try:
                    return self._extract_json_payload(raw_text)
                except Exception as exc:
                    raise VLMCallError(
                        f"Bedrock VLM JSON 解析失败: {exc}",
                        details={
                            "raw_response_preview": self._truncate_preview(raw_text),
                            "parse_error_type": type(exc).__name__,
                            "response_provider": "bedrock",
                        },
                    ) from exc
            except Exception as exc:
                last_error = exc
                if index < len(candidates) - 1 and should_try_next_bedrock_model(exc):
                    continue
                raise
        if last_error:
            raise last_error
        raise RuntimeError("未能调用任何 Bedrock VLM 模型")

    def _analyze_via_proxy(self, prompt: str, image_data: bytes, mime_type: str) -> Dict[str, Any]:
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        headers = {
            "x-api-key": self.proxy_key,
            "Content-Type": "application/json",
        }
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": image_base64,
                            }
                        },
                    ],
                }
            ]
        }
        url = f"{self.proxy_url}/api/gemini/v1beta/models/{self.proxy_model}:generateContent"
        client = self.http_session or self.requests
        response = client.post(url, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            response_data = response.json()
            if "candidates" in response_data and response_data["candidates"]:
                candidate = response_data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    for part in candidate["content"]["parts"]:
                        if "text" in part:
                            raw_text = part["text"]
                            try:
                                return self._extract_json_payload(raw_text)
                            except Exception as exc:
                                raise VLMCallError(
                                    f"代理 VLM JSON 解析失败: {exc}",
                                    details={
                                        "raw_response_preview": self._truncate_preview(raw_text),
                                        "parse_error_type": type(exc).__name__,
                                        "response_status_code": response.status_code,
                                        "response_provider": "proxy",
                                    },
                                ) from exc
            return {}

        error_msg = f"代理 API 返回状态码 {response.status_code}"
        if response.text:
            try:
                error_data = response.json()
                error_msg += f": {error_data.get('error', {}).get('message', response.text)}"
            except Exception:
                error_msg += f": {response.text[:200]}"
        raise Exception(error_msg)

    def _openrouter_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.openrouter_site_url,
            "X-Title": self.openrouter_app_name,
        }

    def _extract_openrouter_content(self, response_data: Dict[str, Any]) -> str:
        choices = response_data.get("choices", [])
        if not choices:
            raise ValueError("OpenRouter 未返回 choices")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not content and message.get("reasoning"):
            content = message["reasoning"]
        return self._coerce_text_content(content)

    def _analyze_via_openrouter(self, prompt: str, image_data: bytes, mime_type: str) -> Dict[str, Any]:
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}",
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.1,
        }

        client = self.http_session or self.requests
        response = client.post(
            f"{self.openrouter_base_url}/chat/completions",
            json=payload,
            headers=self._openrouter_headers(),
            timeout=60,
        )

        if response.status_code == 200:
            response_data = response.json()
            raw_text = self._extract_openrouter_content(response_data)
            try:
                return self._extract_json_payload(raw_text)
            except Exception as exc:
                raise VLMCallError(
                    f"OpenRouter VLM JSON 解析失败: {exc}",
                    details={
                        "raw_response_preview": self._truncate_preview(raw_text),
                        "parse_error_type": type(exc).__name__,
                        "response_status_code": response.status_code,
                        "response_provider": "openrouter",
                    },
                ) from exc

        error_msg = f"OpenRouter 返回状态码 {response.status_code}"
        if response.text:
            try:
                error_data = response.json()
                error_msg += f": {error_data.get('error', {}).get('message', response.text)}"
            except Exception:
                error_msg += f": {response.text[:200]}"
        raise Exception(error_msg)

    def build_result_entry(self, photo: Photo, vlm_result: Dict[str, Any]) -> Dict[str, Any]:
        normalized = self._normalize_result(vlm_result)
        source_type = self._derive_source_type(normalized, filename=photo.filename)
        normalized["source_type"] = source_type
        return {
            "photo_id": photo.photo_id,
            "filename": photo.filename,
            "timestamp": photo.timestamp.isoformat(),
            "location": photo.location,
            "source_type": source_type,
            "face_person_ids": [face["person_id"] for face in photo.faces if face.get("person_id")],
            "vlm_analysis": normalized,
        }

    def replace_results(self, ordered_results: List[Dict[str, Any]]) -> None:
        self.results = list(ordered_results)
        self._rebuild_result_index()

    def _create_prompt(self, photo: Photo, face_db: Dict, primary_person_id: Optional[str]) -> str:
        if getattr(self, "use_heavy_prompt", False):
            return self._create_heavy_prompt(photo, face_db, primary_person_id)
        return self._create_default_prompt(photo, face_db, primary_person_id)

    def _protagonist_photo_count(self, face_db: Dict, primary_person_id: Optional[str]) -> int:
        if not primary_person_id or not isinstance(face_db, dict):
            return 0
        person = face_db.get(primary_person_id)
        if person is None:
            return 0
        if isinstance(person, dict):
            return int(person.get("photo_count") or 0)
        return int(getattr(person, "photo_count", 0) or 0)

    def _build_people_instruction_block(self, photo: Photo, face_db: Dict, primary_person_id: Optional[str]) -> str:
        visible_person_ids = [
            str(face.get("person_id"))
            for face in list(photo.faces or [])
            if face.get("person_id")
        ]
        protagonist_count = self._protagonist_photo_count(face_db, primary_person_id)
        people_list_text = "、".join(visible_person_ids) if visible_person_ids else "无"
        if primary_person_id and primary_person_id in visible_person_ids:
            return (
                "A. 主角在照片中：\n"
                "     **人物说明**（照片中每个人脸用彩色框标注，标签位于人物上方）：\n"
                f"     - {primary_person_id}（红色框）是【主角】，出现在{protagonist_count}张照片中\n"
                "     - 其他人物（蓝色框）用 Person_002、Person_003... 表示\n"
                f"     - 当前照片中所有 person_id：{people_list_text}\n"
                "     **分析原则**：\n"
                '     - summary 中使用"【主角】"指代主角\n'
                "     - people 数组中使用具体的 person_id"
            )
        if primary_person_id and visible_person_ids:
            return (
                "B. 主角已识别，但不在照片中：\n"
                "     **人物说明**：\n"
                f"     - 照片中的人物：{people_list_text}\n"
                "     - 【主角】不在照片中，是拍摄者\n"
                f"     - {primary_person_id} 通常是【主角】，出现在{protagonist_count}张照片中\n"
                "     **分析原则**：\n"
                '     - summary 中使用"【主角】"指代拍摄者（主角）\n'
                '     - 描述从【主角】的拍摄视角观察到的场景\n'
                "     - people 数组中使用照片中具体的 person_id"
            )
        if visible_person_ids:
            return (
                "C. 主角未稳定识别出，但照片里有人脸：\n"
                "     **人物说明**：\n"
                f"     - 照片中的人物：{people_list_text}\n"
                "     - 没有可靠的主用户身份锚点，当前无法从人脸层稳定识别出【主角】\n"
                "     - 统一按拍摄者视角处理，但不要把任何可见人物直接绑定成主用户\n"
                "     - 如出现镜面反射、屏幕、海报、相框、广告牌等媒介场景，要优先识别为疑似载体中的人物\n"
                "     **分析原则**：\n"
                '     - summary 中使用"【拍摄者】"指代用户本人，不要把任何 person_id 写成【主用户】\n'
                '     - 描述从【主角】的拍摄视角观察到的场景\n'
                "     - people 数组中使用照片中具体的 person_id"
            )
        return (
            "D. 无人脸：\n"
            "     **人物说明**：照片中未检测到人脸，这是【主角】的拍摄视角。\n"
            '     **分析原则**：从【主角】的拍摄视角描述场景和事件，summary 中直接使用"【主角】"。'
        )

    def _create_visual_archive_prompt(self, photo: Photo, face_db: Dict, primary_person_id: Optional[str]) -> str:
        time_str = photo.timestamp.strftime("%Y-%m-%d %H:%M")
        if photo.location and photo.location.get("name"):
            location_str = str(photo.location["name"])
        elif photo.location and photo.location.get("lat") is not None and photo.location.get("lng") is not None:
            location_str = f"GPS: {photo.location['lat']:.4f}, {photo.location['lng']:.4f}"
        else:
            location_str = "未知"
        people_instruction = self._build_people_instruction_block(photo, face_db, primary_person_id)
        return f"""## Role
你是一位精通视觉人类学与社会空间重构的专家。你的任务是针对【主角】的相册，建立一套"高度复原、可供画像分析"的客观视觉档案。

### Context & Priorities
- **核心任务**: 以【主角】为圆心，还原每一张照片的物理现场、人物身份与社会学线索。
- **描述准则**: 拒绝模糊词汇（如：肤色白皙），追求物理参数（如：冷白皮、重磅棉、30度斜射光、具体手指动作）。

[动态注入: 人物说明 —— 根据人脸识别结果，分四种情况：
{people_instruction}]

### EXIF信息
- 时间：{time_str}
- 地点：{location_str}

---

### Task: 结构化识别要求 (Output Schema)

请针对每张照片输出以下严格的 JSON 格式（严禁包含主观推测）：

1. **summary** (String, 必须):
   - 完整叙事句，包含：[精确时刻/天气] + [具体地理/室内场景] + [【主角】的行为状态] + [核心事件进度]

2. **people** (List，无人脸时可为空数组):
   - **person_id** (String): 必须与人脸标注一致（Person_001、Person_002...）
   - **appearance** (String): 性别、年龄段、发型细节、脸型特征、体型、修饰痕迹
   - **clothing** (String): 衣物材质（重磅棉/尼龙/真丝等）、版型、品牌Logo、配饰
   - **activity** (String): 人物当前动作/姿态
   - **interaction** (String): 与【主角】的物理距离（亲密/社交/公共）、具体互动动作
   - **contact_type** (String): 身体接触类型，从以下枚举中选择：kiss/hug/holding_hands/arm_in_arm/selfie_together/shoulder_lean/sitting_close/standing_near/no_contact
   - **expression** (String): 面部表情和情绪状态

3. **relations** (List): 画面中可观测到的实体关系三元组：
   - **subject** (String): 主体（person_id 或物品名）
   - **relation** (String): 关系动作（如 sitting_at, holding, interacting_with, placed_on, looking_at, standing_near）
   - **object** (String): 客体（person_id、物品名或场景元素）

4. **scene** (Object):
   - **environment_details** (List): 环境细节列表（木质桌子、绿色植物、暖色调灯光等），如有可见天气信息也写入此处
   - **location_detected** (String): 具体位置识别（如望京漫咖啡、星巴克XX店）
   - **location_type** (String): "室内" 或 "室外"

5. **event** (Object):
   - **activity** (String): 具体活动类型（喝咖啡/吃饭/工作/运动/旅行/购物/学习/其他）
   - **social_context** (String): 社交背景（和朋友/独自/和家人/和同事/和伴侣）
   - **mood** (String): 整体氛围（轻松、愉快、温馨、专注、忙碌...）
   - **story_hints** (List): 1-2条基于视觉证据的社交故事推断（如"可能是生日聚会"、"工作日加班"）

6. **details** (List): 画面中值得关注的硬核线索（品牌Logo、App界面、证件、账单、书籍标题、屏幕内容等）

---

### ⚠️ 输出要求 (Constraints)
- **拒绝模板**: 严禁多人描述雷同，必须捕捉细微差别
- **硬核线索**: 必须扫描并记录所有可见的品牌、文字、屏幕内容
- **JSON Only**: 仅输出结构化 JSON，不要任何开头语或解释

输出JSON格式：
{{
  "summary": "String",
  "people": [
    {{
      "person_id": "String",
      "appearance": "String",
      "clothing": "String",
      "activity": "String",
      "interaction": "String",
      "contact_type": "String",
      "expression": "String"
    }}
  ],
  "relations": [
    {{"subject": "Person_001", "relation": "String", "object": "String"}}
  ],
  "scene": {{
    "environment_details": ["String"],
    "location_detected": "String",
    "location_type": "String"
  }},
  "event": {{
    "activity": "String",
    "social_context": "String",
    "mood": "String",
    "story_hints": ["String"]
  }},
  "details": ["String"]
}}
"""

    def _create_default_prompt(self, photo: Photo, face_db: Dict, primary_person_id: Optional[str]) -> str:
        return self._create_visual_archive_prompt(photo, face_db, primary_person_id)

    def _create_heavy_prompt(self, photo: Photo, face_db: Dict, primary_person_id: Optional[str]) -> str:
        return self._create_visual_archive_prompt(photo, face_db, primary_person_id)

    def save_cache(self) -> None:
        data = {
            "metadata": {
                "total_photos": len(self.results),
                "model": self.model,
                "provider": self.provider,
                "schema_version": 3,
                "face_id_scheme": "Person_###",
            },
            "photos": self.results,
        }
        save_json(data, self.cache_path)

    def _rebuild_result_index(self) -> None:
        self._result_index = {}
        for index, item in enumerate(self.results):
            photo_id = item.get("photo_id")
            if photo_id:
                self._result_index[photo_id] = index

    def load_cache(self) -> bool:
        data = load_json(self.cache_path)
        metadata = data.get("metadata", {}) if data else {}
        schema_version = int(metadata.get("schema_version") or 0)
        if schema_version < 2 or metadata.get("face_id_scheme") != "Person_###":
            if data:
                print("VLM缓存版本与当前人脸协议不兼容，忽略旧缓存")
            return False

        if data and data.get("photos"):
            normalized_results = []
            for item in data["photos"]:
                vlm_analysis = self._normalize_result(item.get("vlm_analysis"))
                normalized_item = dict(item)
                source_type = self._derive_source_type(vlm_analysis, filename=str(item.get("filename") or ""))
                vlm_analysis["source_type"] = source_type
                normalized_item["vlm_analysis"] = vlm_analysis
                normalized_item.setdefault("face_person_ids", [])
                normalized_item["source_type"] = source_type
                normalized_results.append(normalized_item)

            self.results = normalized_results
            self._rebuild_result_index()
            print(f"加载VLM缓存：{len(self.results)} 张照片")
            return True
        return False

    def has_result(self, photo_id: str) -> bool:
        return photo_id in self._result_index

    def get_result(self, photo_id: str) -> Dict[str, Any] | None:
        index = self._result_index.get(photo_id)
        if index is None:
            return None
        return self.results[index]

    def add_result(self, photo: Photo, vlm_result: Dict[str, Any], persist: bool = False) -> None:
        result = self.build_result_entry(photo, vlm_result)
        existing_index = self._result_index.get(photo.photo_id)
        if existing_index is not None:
            self.results[existing_index] = result
        else:
            self.results.append(result)
        self._rebuild_result_index()
        if persist:
            self.save_cache()
