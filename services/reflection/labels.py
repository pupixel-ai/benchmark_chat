from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

from config import PROJECT_ROOT


LABEL_TABLE_RELATIVE_PATH = Path("docs") / "reflection_field_bilingual_table.csv"


def load_bilingual_label_rows(*, project_root: str = PROJECT_ROOT) -> List[Dict[str, str]]:
    return list(_load_bilingual_label_rows(project_root))


def lookup_bilingual_label(category: str, key: str, *, project_root: str = PROJECT_ROOT, default: str = "") -> str:
    normalized_category = str(category or "").strip()
    normalized_key = str(key or "").strip()
    if not normalized_key:
        return default
    label = _load_bilingual_label_map(project_root).get((normalized_category, normalized_key), "")
    return label or default or normalized_key


def describe_profile_field(field_key: str, *, project_root: str = PROJECT_ROOT) -> str:
    normalized_key = str(field_key or "").strip()
    if not normalized_key:
        return ""
    resolved_key = resolve_profile_field_hint(normalized_key, project_root=project_root) or normalized_key
    label = lookup_bilingual_label("profile_field", resolved_key, project_root=project_root, default=resolved_key)
    if label == resolved_key and resolved_key != normalized_key:
        return normalized_key
    if label == normalized_key:
        return normalized_key
    if resolved_key != normalized_key:
        return f"{label}（{resolved_key}）"
    return f"{label}（{normalized_key}）"


def resolve_profile_field_hint(field_hint: str, *, project_root: str = PROJECT_ROOT) -> str:
    normalized_hint = str(field_hint or "").strip()
    if not normalized_hint:
        return ""
    profile_keys = [
        str(row.get("key") or "").strip()
        for row in _load_bilingual_label_rows(project_root)
        if str(row.get("category") or "").strip() == "profile_field"
    ]
    if normalized_hint in profile_keys:
        return normalized_hint
    leaf_hint = normalized_hint.split(".")[-1].strip().lower()
    if not leaf_hint:
        return ""
    matches = [key for key in profile_keys if key.lower().endswith(f".{leaf_hint}")]
    if len(matches) == 1:
        return matches[0]
    return ""


@lru_cache(maxsize=4)
def _load_bilingual_label_rows(project_root: str) -> Tuple[Dict[str, str], ...]:
    table_path = Path(project_root) / LABEL_TABLE_RELATIVE_PATH
    if not table_path.exists():
        return tuple()
    with table_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(
                {
                    "category": str(row.get("category") or "").strip(),
                    "key": str(row.get("key") or "").strip(),
                    "zh_label": str(row.get("zh_label") or "").strip(),
                    "source": str(row.get("source") or "").strip(),
                    "notes": str(row.get("notes") or "").strip(),
                }
            )
        return tuple(rows)


@lru_cache(maxsize=4)
def _load_bilingual_label_map(project_root: str) -> Dict[Tuple[str, str], str]:
    mapping: Dict[Tuple[str, str], str] = {}
    for row in _load_bilingual_label_rows(project_root):
        category = str(row.get("category") or "").strip()
        key = str(row.get("key") or "").strip()
        zh_label = str(row.get("zh_label") or "").strip()
        if not category or not key:
            continue
        mapping[(category, key)] = zh_label
    return mapping
