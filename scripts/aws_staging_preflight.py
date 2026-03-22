from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import (
    APP_ROLE,
    AWS_REGION,
    DATABASE_URL,
    MEMORY_NEO4J_DATABASE,
    MEMORY_NEO4J_PASSWORD,
    MEMORY_NEO4J_URI,
    MEMORY_NEO4J_USERNAME,
    MEMORY_REDIS_URL,
    OBJECT_STORAGE_BUCKET,
    OBJECT_STORAGE_REGION,
    WORKER_INTERNAL_PORT,
    WORKER_ORCHESTRATION_ENABLED,
    WORKER_SHARED_TOKEN,
)


REQUIRED_IMPORTS = [
    "fastapi",
    "httpx",
    "PIL",
    "pillow_heif",
    "sqlalchemy",
    "requests",
    "boto3",
    "redis",
    "neo4j",
]

REQUIRED_FILES = [
    "deploy/aws/README.md",
    "deploy/aws/control-plane.env.example",
    "deploy/aws/worker.env.example",
    "deploy/aws/neo4j/docker-compose.yml",
    "deploy/aws/milvus/docker-compose.yml",
    "deploy/aws/redis/docker-compose.yml",
]


def _bool(value: Any) -> bool:
    return bool(value and str(value).strip())


def main() -> None:
    repo_root = REPO_ROOT
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []
    warnings: list[str] = []

    for module_name in REQUIRED_IMPORTS:
        ok = True
        error = None
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - script only
            ok = False
            error = str(exc)
            blockers.append(f"missing import: {module_name}")
        checks.append({"check": f"import:{module_name}", "ok": ok, "error": error})

    for relative_path in REQUIRED_FILES:
        exists = (repo_root / relative_path).exists()
        checks.append({"check": f"file:{relative_path}", "ok": exists})
        if not exists:
            blockers.append(f"missing file: {relative_path}")

    runtime_checks = {
        "APP_ROLE": APP_ROLE,
        "AWS_REGION": AWS_REGION,
        "WORKER_ORCHESTRATION_ENABLED": WORKER_ORCHESTRATION_ENABLED,
        "WORKER_INTERNAL_PORT": WORKER_INTERNAL_PORT,
        "DATABASE_URL": DATABASE_URL,
        "OBJECT_STORAGE_BUCKET": OBJECT_STORAGE_BUCKET,
        "OBJECT_STORAGE_REGION": OBJECT_STORAGE_REGION,
        "MEMORY_REDIS_URL": MEMORY_REDIS_URL,
        "MEMORY_NEO4J_URI": MEMORY_NEO4J_URI,
        "MEMORY_NEO4J_USERNAME": MEMORY_NEO4J_USERNAME,
        "MEMORY_NEO4J_PASSWORD_SET": _bool(MEMORY_NEO4J_PASSWORD),
        "MEMORY_NEO4J_DATABASE": MEMORY_NEO4J_DATABASE,
        "WORKER_SHARED_TOKEN_SET": _bool(WORKER_SHARED_TOKEN),
    }

    if APP_ROLE != "control-plane":
        blockers.append("APP_ROLE is not control-plane")
    if not _bool(AWS_REGION):
        blockers.append("AWS_REGION is empty")
    if not WORKER_ORCHESTRATION_ENABLED:
        blockers.append("WORKER_ORCHESTRATION_ENABLED is false")
    if not _bool(WORKER_SHARED_TOKEN):
        blockers.append("WORKER_SHARED_TOKEN is empty")
    if not DATABASE_URL.startswith("mysql+pymysql://"):
        warnings.append("DATABASE_URL is not pointing to MySQL/RDS staging target")
    if not _bool(OBJECT_STORAGE_BUCKET):
        blockers.append("OBJECT_STORAGE_BUCKET is empty")
    if not _bool(OBJECT_STORAGE_REGION):
        blockers.append("OBJECT_STORAGE_REGION is empty")
    if not _bool(MEMORY_REDIS_URL):
        blockers.append("MEMORY_REDIS_URL is empty")
    if not _bool(MEMORY_NEO4J_URI):
        blockers.append("MEMORY_NEO4J_URI is empty")
    if not _bool(MEMORY_NEO4J_USERNAME):
        blockers.append("MEMORY_NEO4J_USERNAME is empty")
    if not _bool(MEMORY_NEO4J_PASSWORD):
        blockers.append("MEMORY_NEO4J_PASSWORD is empty")

    report = {
        "ready_for_staging": not blockers,
        "checks": checks,
        "runtime": runtime_checks,
        "blockers": blockers,
        "warnings": warnings,
        "cwd": os.getcwd(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
