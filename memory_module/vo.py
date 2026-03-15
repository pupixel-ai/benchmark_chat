"""Immutable semantic value objects for the memory module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(slots=True, frozen=True)
class UpstreamRefVO:
    source_system: str
    object_type: str
    object_id: str


@dataclass(slots=True, frozen=True)
class IdentityMappingVO:
    user_id: str
    face_person_id: str
    person_uuid: str


@dataclass(slots=True, frozen=True)
class GeoPointVO:
    lat: Optional[float]
    lng: Optional[float]


@dataclass(slots=True, frozen=True)
class LocationHintVO:
    name: str
    geo_point: GeoPointVO
    location_confidence: float
    source: str


@dataclass(slots=True, frozen=True)
class TimePointVO:
    isoformat: str
    timezone_guess: Optional[str] = None
    confidence: float = 0.0


@dataclass(slots=True, frozen=True)
class TimeRangeVO:
    started_at: str
    ended_at: str


@dataclass(slots=True, frozen=True)
class SequenceBoundaryVO:
    previous_id: Optional[str]
    next_id: Optional[str]
    gap_seconds: int


@dataclass(slots=True, frozen=True)
class ConfidenceVO:
    value: float
    source: str


@dataclass(slots=True, frozen=True)
class QualityScoreVO:
    value: float
    flags: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class EvidenceRefVO:
    ref_type: str
    ref_id: str
    summary: str = ""


@dataclass(slots=True, frozen=True)
class ImportanceScoreVO:
    value: float
    reason: str = ""


@dataclass(slots=True, frozen=True)
class VersionStampVO:
    profile_version: int
    generated_at: str


@dataclass(slots=True, frozen=True)
class DeletionScopeVO:
    scope_type: str
    scope_id: str
    affected_object_types: tuple[str, ...] = ()
    metadata: Dict[str, str] = field(default_factory=dict)
