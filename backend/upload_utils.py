"""
Shared upload helpers for control-plane and worker flows.
"""
from __future__ import annotations

import io
import os
from hashlib import sha256
from pathlib import Path

from fastapi import UploadFile
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from utils import save_json


register_heif_opener()

UPLOAD_FAILURES_FILENAME = "upload_failures.json"
ORIENTATION_TAG = 274


def safe_filename(filename: str, fallback: str) -> str:
    basename = os.path.basename(filename or fallback)
    basename = basename.replace("/", "_").replace("\\", "_")
    return basename or fallback


def stored_upload_filename(filename: str, index: int) -> str:
    safe_name = safe_filename(filename, f"upload_{index:03d}")
    stem = Path(safe_name).stem or f"upload_{index:03d}"
    suffix = Path(safe_name).suffix.lower() or ".bin"
    return f"{index:03d}_{stem}{suffix}"


def preview_filename(filename: str, index: int) -> str:
    safe_name = safe_filename(filename, f"upload_{index:03d}")
    stem = Path(safe_name).stem or f"upload_{index:03d}"
    return f"{index:03d}_{stem}_preview.webp"


def task_asset_path(directory: str, filename: str) -> str:
    return f"{directory}/{filename}"


def is_live_photo_candidate(filename: str, content_type: str | None = None) -> bool:
    suffix = Path(filename or "").suffix.lower()
    normalized_content_type = (content_type or "").strip().lower()
    return suffix in {".heic", ".heif", ".livp"} or normalized_content_type in {
        "image/heic",
        "image/heif",
        "image/heic-sequence",
        "image/heif-sequence",
    }


def normalized_exif_bytes(image: Image.Image) -> bytes | None:
    try:
        exif = image.getexif()
    except Exception:
        return None

    if not exif:
        return None

    if ORIENTATION_TAG in exif:
        exif[ORIENTATION_TAG] = 1
    return exif.tobytes()


def save_upload_original(upload: UploadFile, destination: Path) -> tuple[bytes, dict]:
    upload.file.seek(0)
    payload = upload.file.read()
    if not payload:
        raise ValueError("文件为空")

    destination.write_bytes(payload)
    image_info = {
        "content_type": upload.content_type or "application/octet-stream",
        "width": None,
        "height": None,
    }

    with Image.open(io.BytesIO(payload)) as image:
        image_info["width"], image_info["height"] = image.size
        image_info["content_type"] = upload.content_type or Image.MIME.get(image.format, image_info["content_type"])

    return payload, image_info


def save_upload_original_streamed(upload: UploadFile, destination: Path, chunk_size: int = 1024 * 1024) -> dict:
    upload.file.seek(0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256()

    with destination.open("wb") as handle:
        while True:
            chunk = upload.file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
            handle.write(chunk)

    if destination.stat().st_size == 0:
        raise ValueError("文件为空")

    image_info = {
        "content_type": upload.content_type or "application/octet-stream",
        "width": None,
        "height": None,
        "source_hash": digest.hexdigest(),
    }

    with Image.open(destination) as image:
        image_info["width"], image_info["height"] = image.size
        image_info["content_type"] = upload.content_type or Image.MIME.get(image.format, image_info["content_type"])

    upload.file.seek(0)
    return image_info


def save_preview_as_webp(payload: bytes, destination: Path) -> dict:
    with Image.open(io.BytesIO(payload)) as image:
        normalized = ImageOps.exif_transpose(image)
        exif = normalized_exif_bytes(normalized)
        working = normalized.convert("RGBA") if "A" in normalized.getbands() else normalized.convert("RGB")
        save_kwargs = {
            "format": "WEBP",
            "quality": 90,
            "method": 6,
        }
        if exif:
            save_kwargs["exif"] = exif
        working.save(destination, **save_kwargs)
        width, height = working.size

    return {
        "width": width,
        "height": height,
        "content_type": "image/webp",
    }


def write_upload_failures(task_dir: Path, failures: list[dict]) -> None:
    save_json({"failures": failures}, str(task_dir / UPLOAD_FAILURES_FILENAME))
