"""
任务资产对象存储。
"""
from __future__ import annotations

import mimetypes
from pathlib import Path, PurePosixPath
from typing import Optional, Tuple
from urllib.parse import quote

from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from config import (
    ASSET_URL_PREFIX,
    HIGH_SECURITY_MODE,
    OBJECT_STORAGE_ACCESS_KEY_ID,
    OBJECT_STORAGE_ADDRESSING_STYLE,
    OBJECT_STORAGE_BUCKET,
    OBJECT_STORAGE_ENDPOINT,
    OBJECT_STORAGE_PREFIX,
    OBJECT_STORAGE_REGION,
    OBJECT_STORAGE_SECRET_ACCESS_KEY,
)


class TaskAssetStore:
    """使用 S3 兼容对象存储保存任务资产，并通过后端路由稳定访问。"""

    def __init__(self, asset_url_prefix: str = ASSET_URL_PREFIX):
        self.asset_url_prefix = asset_url_prefix.rstrip("/")
        self.bucket = OBJECT_STORAGE_BUCKET.strip()
        self.endpoint = OBJECT_STORAGE_ENDPOINT.strip()
        self.region = OBJECT_STORAGE_REGION.strip()
        self.access_key_id = OBJECT_STORAGE_ACCESS_KEY_ID.strip()
        self.secret_access_key = OBJECT_STORAGE_SECRET_ACCESS_KEY.strip()
        self.prefix = OBJECT_STORAGE_PREFIX.strip().strip("/")
        self.addressing_style = OBJECT_STORAGE_ADDRESSING_STYLE.strip() or "auto"
        self._client = None

        if self.enabled:
            import boto3

            config_kwargs = {}
            if self.addressing_style in {"auto", "path", "virtual"}:
                config_kwargs["s3"] = {"addressing_style": self.addressing_style}

            client_kwargs = {
                "service_name": "s3",
                "region_name": self.region or None,
                "config": BotoConfig(**config_kwargs) if config_kwargs else None,
            }
            if self.endpoint:
                client_kwargs["endpoint_url"] = self.endpoint
            if self.access_key_id and self.secret_access_key:
                client_kwargs["aws_access_key_id"] = self.access_key_id
                client_kwargs["aws_secret_access_key"] = self.secret_access_key

            self._client = boto3.client(**client_kwargs)

    @property
    def enabled(self) -> bool:
        if HIGH_SECURITY_MODE:
            return False
        if not self.bucket:
            return False
        if self.endpoint:
            return bool(self.access_key_id and self.secret_access_key)
        return True

    def sanitize_relative_path(self, relative_path: str) -> str:
        normalized = PurePosixPath(relative_path.replace("\\", "/"))
        if normalized.is_absolute() or ".." in normalized.parts:
            raise ValueError(f"非法资产路径: {relative_path}")
        return normalized.as_posix()

    def object_key(self, task_id: str, relative_path: str) -> str:
        if not relative_path or relative_path in {"."}:
            if self.prefix:
                return f"{self.prefix}/{task_id}"
            return f"{task_id}"

        safe_relative_path = self.sanitize_relative_path(relative_path)
        if self.prefix:
            return f"{self.prefix}/{task_id}/{safe_relative_path}"
        return f"{task_id}/{safe_relative_path}"

    def asset_url(self, task_id: str, relative_path: str) -> str:
        safe_relative_path = self.sanitize_relative_path(relative_path)
        encoded_path = quote(safe_relative_path, safe="/")
        return f"{self.asset_url_prefix}/{task_id}/{encoded_path}"

    def presigned_get_url(self, task_id: str, relative_path: str, *, expires_in: int = 86400) -> Optional[str]:
        if not self.enabled or self._client is None:
            return None
        safe_relative_path = self.sanitize_relative_path(relative_path)
        key = self.object_key(task_id, safe_relative_path)
        try:
            return self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=max(60, int(expires_in)),
            )
        except Exception:
            return None

    def _content_type_for(self, relative_path: str) -> str:
        content_type, _ = mimetypes.guess_type(relative_path)
        return content_type or "application/octet-stream"

    def upload_file(self, task_id: str, relative_path: str, local_path: str | Path) -> str:
        if not self.enabled or self._client is None:
            raise RuntimeError("对象存储未配置，无法上传文件")

        key = self.object_key(task_id, relative_path)
        content_type = self._content_type_for(relative_path)
        with open(local_path, "rb") as handle:
            self._client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=handle,
                ContentType=content_type,
            )
        return key

    def upload_bytes(self, task_id: str, relative_path: str, payload: bytes) -> str:
        if not self.enabled or self._client is None:
            raise RuntimeError("对象存储未配置，无法上传内容")

        key = self.object_key(task_id, relative_path)
        content_type = self._content_type_for(relative_path)
        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=payload,
            ContentType=content_type,
        )
        return key

    def read_bytes(self, task_id: str, relative_path: str) -> Tuple[bytes, str]:
        if not self.enabled or self._client is None:
            raise RuntimeError("对象存储未配置，无法读取内容")

        key = self.object_key(task_id, relative_path)
        response = self._client.get_object(Bucket=self.bucket, Key=key)
        body = response["Body"].read()
        content_type = response.get("ContentType") or self._content_type_for(relative_path)
        return body, content_type

    def sync_task_directory(self, task_id: str, task_dir: str | Path) -> None:
        if not self.enabled:
            return

        root = Path(task_dir)
        if not root.exists():
            return

        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file():
                continue
            relative_path = file_path.relative_to(root).as_posix()
            self.upload_file(task_id, relative_path, file_path)

    def has_object(self, task_id: str, relative_path: str) -> bool:
        if not self.enabled or self._client is None:
            return False

        key = self.object_key(task_id, relative_path)
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def local_asset_path(self, task_dir: str | Path, relative_path: str) -> Optional[Path]:
        safe_relative_path = self.sanitize_relative_path(relative_path)
        root = Path(task_dir).resolve()
        candidate = (root / safe_relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        return candidate

    def delete_task_assets(self, task_id: str) -> None:
        if not self.enabled or self._client is None:
            return

        prefix = self.object_key(task_id, "")
        continuation_token = None

        while True:
            request_kwargs = {
                "Bucket": self.bucket,
                "Prefix": prefix,
                "MaxKeys": 1000,
            }
            if continuation_token:
                request_kwargs["ContinuationToken"] = continuation_token

            response = self._client.list_objects_v2(**request_kwargs)
            contents = response.get("Contents", [])
            if contents:
                self._client.delete_objects(
                    Bucket=self.bucket,
                    Delete={
                        "Objects": [{"Key": item["Key"]} for item in contents],
                        "Quiet": True,
                    },
                )

            if not response.get("IsTruncated"):
                break
            continuation_token = response.get("NextContinuationToken")
