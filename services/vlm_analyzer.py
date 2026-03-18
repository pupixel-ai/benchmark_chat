"""
VLM analysis module.
"""
from __future__ import annotations

import base64
import json
import mimetypes
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
    MAX_RETRIES,
    OPENROUTER_API_KEY,
    OPENROUTER_APP_NAME,
    OPENROUTER_BASE_URL,
    OPENROUTER_SITE_URL,
    OPENROUTER_VLM_MODEL,
    RETRY_DELAY,
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


class VLMAnalyzer:
    """VLM analyzer with Bedrock, Gemini proxy, and OpenRouter support."""

    def __init__(self, cache_path: str = VLM_CACHE_PATH):
        self.provider = VLM_PROVIDER
        self.use_proxy = self.provider == "proxy"
        self.use_openrouter = self.provider == "openrouter"
        self.use_bedrock = self.provider == "bedrock"
        self.model = OPENROUTER_VLM_MODEL if self.use_openrouter else VLM_MODEL
        self.cache_path = cache_path
        self.results: List[Dict[str, Any]] = []
        self._result_index: Dict[str, int] = {}
        self.requests = None
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
            self.client = genai.Client(api_key=GEMINI_API_KEY)
            print("[VLM] 使用官方 Gemini API")

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
        normalized.setdefault("ocr_hits", [])
        normalized.setdefault("brands", [])
        normalized.setdefault("place_candidates", [])
        normalized.setdefault("route_plan_clues", [])
        normalized.setdefault("transport_clues", [])
        normalized.setdefault("health_treatment_clues", [])
        normalized.setdefault("object_last_seen_clues", [])
        normalized.setdefault("raw_structured_observations", [])
        normalized.setdefault("uncertainty", [])
        return normalized

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
        image_path = (
            getattr(photo, "boxed_path", None)
            or getattr(photo, "annotated_path", None)
            or photo.compressed_path
        )
        if not image_path:
            photo.processing_errors["vlm"] = "未找到可供 VLM 分析的图片路径"
            return None

        if primary_person_id is None:
            primary_person_id = self._infer_reliable_primary_person_id(face_db)

        prompt = self._create_prompt(photo, face_db, primary_person_id)
        try:
            with open(image_path, "rb") as file_obj:
                image_data = file_obj.read()
            mime_type = self._guess_mime_type(image_path)

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
                    delay_seconds = RETRY_DELAY * (2 ** attempt)
                    print(f"[VLM] 可重试错误，{delay_seconds}s 后重试 ({attempt + 1}/{MAX_RETRIES}): {exc}")
                    time.sleep(delay_seconds)

            normalized = self._normalize_result(result)
            photo.vlm_analysis = normalized
            photo.processing_errors.pop("vlm", None)
            return normalized
        except Exception as exc:
            photo.processing_errors["vlm"] = str(exc)
            print(f"警告：VLM分析失败 ({photo.filename}): {exc}")
            return None

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
        return self._extract_json_payload(response.text)

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
                return self._extract_json_payload(extract_text_from_converse_response(response))
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
        response = self.requests.post(url, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            response_data = response.json()
            if "candidates" in response_data and response_data["candidates"]:
                candidate = response_data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    for part in candidate["content"]["parts"]:
                        if "text" in part:
                            return self._extract_json_payload(part["text"])
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

        response = self.requests.post(
            f"{self.openrouter_base_url}/chat/completions",
            json=payload,
            headers=self._openrouter_headers(),
            timeout=60,
        )

        if response.status_code == 200:
            response_data = response.json()
            return self._extract_json_payload(self._extract_openrouter_content(response_data))

        error_msg = f"OpenRouter 返回状态码 {response.status_code}"
        if response.text:
            try:
                error_data = response.json()
                error_msg += f": {error_data.get('error', {}).get('message', response.text)}"
            except Exception:
                error_msg += f": {response.text[:200]}"
        raise Exception(error_msg)

    def _create_prompt(self, photo: Photo, face_db: Dict, primary_person_id: Optional[str]) -> str:
        time_str = photo.timestamp.strftime("%Y-%m-%d %H:%M")
        if photo.location and photo.location.get("name"):
            location_str = str(photo.location["name"])
        elif photo.location and photo.location.get("lat") is not None and photo.location.get("lng") is not None:
            location_str = f"GPS: {photo.location['lat']:.4f}, {photo.location['lng']:.4f}"
        else:
            location_str = "未知"

        visible_people = []
        for face in photo.faces:
            person_id = face.get("person_id")
            if not person_id:
                continue
            if person_id == primary_person_id:
                visible_people.append(f"{person_id}（候选主用户脸）")
            else:
                visible_people.append(person_id)
        people_hint = "、".join(visible_people) if visible_people else "未检测到可靠可见人脸"
        primary_hint = primary_person_id or "无稳定主用户脸锚点"

        return f"""你是一个高召回视觉记忆抽取器。你的任务不是写优美描述，而是为后续 LLM 和记忆系统提取尽可能完整、可检索、可回溯的结构化观察。

照片上下文：
- 时间: {time_str}
- EXIF 地点: {location_str}
- 可见人脸 person_id: {people_hint}
- 候选主用户脸锚点: {primary_hint}

要求：
1. 输出必须是严格 JSON。
2. 高召回提取以下信息，宁可保留 uncertainty，也不要胡乱补全。
3. 必须显式记录 OCR、品牌、菜品、价格、服装/材质、地点候选、路线/计划线索、交通线索、健康/治疗线索、物体最后出现线索。
4. 如果信息不存在，返回空数组或空对象，不要编造。
5. 如果人物只出现在海报、屏幕、广告牌、包装或反射中，必须写入 uncertainty 或 raw_structured_observations，不要当成现场实体人物。
6. summary 只做客观一句话概括，不做身份结论。

输出 JSON schema:
{{
  "summary": "一句话客观概括",
  "people": [
    {{
      "person_id": "Person_001",
      "appearance": "可见外貌线索",
      "clothing": "服装/材质/配饰",
      "activity": "动作",
      "interaction": "与他人/拍摄者的互动",
      "expression": "表情",
      "confidence": 0.0
    }}
  ],
  "scene": {{
    "environment_description": "宏观场景",
    "environment_details": ["细节"],
    "location_detected": "地点候选",
    "visual_clues": ["视觉线索"],
    "weather": "天气或空字符串",
    "lighting": "光线描述",
    "layout": {{
      "foreground": "",
      "midground": "",
      "background": ""
    }}
  }},
  "event": {{
    "activity": "活动候选",
    "social_context": "社交上下文",
    "interaction": "互动动作",
    "mood": "氛围"
  }},
  "details": ["值得保留的硬线索"],
  "key_objects": ["关键物体"],
  "ocr_hits": ["OCR 命中"],
  "brands": ["品牌/媒体/IP/商家名"],
  "place_candidates": [
    {{"name": "地点候选", "confidence": 0.0, "reason": "为什么"}}
  ],
  "route_plan_clues": ["路线/计划/景点线索"],
  "transport_clues": ["交通/通勤/车内线索"],
  "health_treatment_clues": ["症状/药品/治疗线索"],
  "object_last_seen_clues": ["物品最后出现线索"],
  "raw_structured_observations": [
    {{
      "observation_type": "ocr_observation",
      "field": "restaurant_name",
      "value": "识别值",
      "confidence": 0.0
    }}
  ],
  "uncertainty": [
    {{
      "field": "exact_city",
      "status": "unknown",
      "reason": "原因"
    }}
  ]
}}
"""

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
                normalized_item["vlm_analysis"] = vlm_analysis
                normalized_item.setdefault("face_person_ids", [])
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
        normalized = self._normalize_result(vlm_result)
        result = {
            "photo_id": photo.photo_id,
            "filename": photo.filename,
            "timestamp": photo.timestamp.isoformat(),
            "location": photo.location,
            "face_person_ids": [face["person_id"] for face in photo.faces if face.get("person_id")],
            "vlm_analysis": normalized,
        }
        existing_index = self._result_index.get(photo.photo_id)
        if existing_index is not None:
            self.results[existing_index] = result
        else:
            self.results.append(result)
        self._rebuild_result_index()
        if persist:
            self.save_cache()
