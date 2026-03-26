from __future__ import annotations

import json
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterator

from config import PROJECT_ROOT

from .types import FieldSpec


RULE_ASSET_RELATIVE_DIR = Path("services") / "memory_pipeline" / "rule_assets"

_RUNTIME_RULE_OVERLAYS: Dict[str, Dict[str, Any]] = {
    "field_spec_overrides": {},
    "tool_rules": {},
    "call_policies": {},
}
_ACTIVE_EXPERIMENT_OVERLAYS: list[Dict[str, Any]] = []


def build_rule_asset_paths(*, project_root: str = PROJECT_ROOT) -> Dict[str, str]:
    asset_dir = Path(project_root) / RULE_ASSET_RELATIVE_DIR
    return {
        "asset_dir": str(asset_dir),
        "field_specs_overrides_path": str(asset_dir / "field_specs.overrides.json"),
        "tool_rules_path": str(asset_dir / "tool_rules.json"),
        "call_policies_path": str(asset_dir / "call_policies.json"),
    }


def ensure_rule_asset_files(*, project_root: str = PROJECT_ROOT) -> Dict[str, str]:
    paths = build_rule_asset_paths(project_root=project_root)
    asset_dir = Path(paths["asset_dir"])
    asset_dir.mkdir(parents=True, exist_ok=True)
    for key in ("field_specs_overrides_path", "tool_rules_path", "call_policies_path"):
        file_path = Path(paths[key])
        if not file_path.exists():
            file_path.write_text("{}", encoding="utf-8")
    return paths


def clear_runtime_rule_overlays() -> None:
    for bucket in _RUNTIME_RULE_OVERLAYS.values():
        bucket.clear()
    _ACTIVE_EXPERIMENT_OVERLAYS.clear()


def apply_runtime_field_spec_updates(updates: Dict[str, Dict[str, Any]]) -> None:
    apply_runtime_rule_overlays(field_spec_overrides=updates)


def apply_runtime_rule_overlays(
    *,
    field_spec_overrides: Dict[str, Dict[str, Any]] | None = None,
    tool_rules: Dict[str, Dict[str, Any]] | None = None,
    call_policies: Dict[str, Dict[str, Any]] | None = None,
) -> None:
    if field_spec_overrides:
        _deep_merge_into(_RUNTIME_RULE_OVERLAYS["field_spec_overrides"], field_spec_overrides)
    if tool_rules:
        _deep_merge_into(_RUNTIME_RULE_OVERLAYS["tool_rules"], tool_rules)
    if call_policies:
        _deep_merge_into(_RUNTIME_RULE_OVERLAYS["call_policies"], call_policies)


@contextmanager
def temporary_rule_overlay(overlay_bundle: Dict[str, Any] | None) -> Iterator[None]:
    if not overlay_bundle:
        yield
        return
    _ACTIVE_EXPERIMENT_OVERLAYS.append(deepcopy(dict(overlay_bundle)))
    try:
        yield
    finally:
        if _ACTIVE_EXPERIMENT_OVERLAYS:
            _ACTIVE_EXPERIMENT_OVERLAYS.pop()


def load_repo_rule_assets(*, project_root: str = PROJECT_ROOT) -> Dict[str, Dict[str, Any]]:
    paths = ensure_rule_asset_files(project_root=project_root)
    return {
        "field_spec_overrides": _load_json_object(paths["field_specs_overrides_path"]),
        "tool_rules": _load_json_object(paths["tool_rules_path"]),
        "call_policies": _load_json_object(paths["call_policies_path"]),
    }


def load_active_rule_assets(
    *,
    project_root: str = PROJECT_ROOT,
    overlay_bundle: Dict[str, Any] | None = None,
) -> Dict[str, Dict[str, Any]]:
    effective = load_repo_rule_assets(project_root=project_root)
    _deep_merge_into(effective["field_spec_overrides"], _RUNTIME_RULE_OVERLAYS["field_spec_overrides"])
    _deep_merge_into(effective["tool_rules"], _RUNTIME_RULE_OVERLAYS["tool_rules"])
    _deep_merge_into(effective["call_policies"], _RUNTIME_RULE_OVERLAYS["call_policies"])
    for active_overlay in _ACTIVE_EXPERIMENT_OVERLAYS:
        _apply_overlay_bundle(effective, active_overlay)
    _apply_overlay_bundle(effective, overlay_bundle)
    return effective


def get_effective_field_specs(
    *,
    base_field_specs: Dict[str, FieldSpec],
    project_root: str = PROJECT_ROOT,
    overlay_bundle: Dict[str, Any] | None = None,
) -> Dict[str, FieldSpec]:
    rule_assets = load_active_rule_assets(project_root=project_root, overlay_bundle=overlay_bundle)
    overrides = dict(rule_assets.get("field_spec_overrides") or {})
    effective_specs: Dict[str, FieldSpec] = {}
    for field_key, spec in base_field_specs.items():
        override = dict(overrides.get(field_key) or {})
        if not override:
            effective_specs[field_key] = replace(spec)
            continue
        allowed_keys = set(FieldSpec.__dataclass_fields__.keys())
        filtered_override = {key: value for key, value in override.items() if key in allowed_keys}
        effective_specs[field_key] = replace(spec, **filtered_override)
    return effective_specs


def get_active_tool_rule(
    field_key: str,
    *,
    project_root: str = PROJECT_ROOT,
    overlay_bundle: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    assets = load_active_rule_assets(project_root=project_root, overlay_bundle=overlay_bundle)
    return dict((assets.get("tool_rules") or {}).get(field_key) or {})


def get_active_call_policy(
    field_key: str,
    *,
    project_root: str = PROJECT_ROOT,
    overlay_bundle: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    assets = load_active_rule_assets(project_root=project_root, overlay_bundle=overlay_bundle)
    return dict((assets.get("call_policies") or {}).get(field_key) or {})


def apply_repo_rule_patch(
    *,
    project_root: str = PROJECT_ROOT,
    patch_preview: Dict[str, Any],
) -> Dict[str, str]:
    paths = ensure_rule_asset_files(project_root=project_root)
    field_overrides = _load_json_object(paths["field_specs_overrides_path"])
    tool_rules = _load_json_object(paths["tool_rules_path"])
    call_policies = _load_json_object(paths["call_policies_path"])

    _deep_merge_into(field_overrides, dict(patch_preview.get("field_spec_overrides") or {}))
    _deep_merge_into(tool_rules, dict(patch_preview.get("tool_rules") or {}))
    _deep_merge_into(call_policies, dict(patch_preview.get("call_policies") or {}))

    Path(paths["field_specs_overrides_path"]).write_text(
        json.dumps(field_overrides, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    Path(paths["tool_rules_path"]).write_text(
        json.dumps(tool_rules, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    Path(paths["call_policies_path"]).write_text(
        json.dumps(call_policies, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return paths


def _apply_overlay_bundle(target: Dict[str, Dict[str, Any]], overlay_bundle: Dict[str, Any] | None) -> None:
    if not overlay_bundle:
        return
    _deep_merge_into(target["field_spec_overrides"], dict(overlay_bundle.get("field_spec_overrides") or {}))
    _deep_merge_into(target["tool_rules"], dict(overlay_bundle.get("tool_rules") or {}))
    _deep_merge_into(target["call_policies"], dict(overlay_bundle.get("call_policies") or {}))


def _load_json_object(path: str) -> Dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _deep_merge_into(target: Dict[str, Any], updates: Dict[str, Any]) -> None:
    for key, value in (updates or {}).items():
        if isinstance(value, dict):
            existing = target.get(key)
            if not isinstance(existing, dict):
                target[key] = deepcopy(value)
                continue
            _deep_merge_into(existing, value)
            continue
        target[key] = deepcopy(value)
