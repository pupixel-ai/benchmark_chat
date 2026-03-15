"""Evaluation and annotation DTOs for the memory module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class GoldProfileLabelDTO:
    label_id: str
    user_id: str
    field_key: str
    field_value: str
    confidence: float = 1.0


@dataclass(slots=True)
class GoldRelationshipLabelDTO:
    label_id: str
    user_id: str
    person_uuid: str
    relationship_type: str
    confidence: float = 1.0


@dataclass(slots=True)
class GoldEventLabelDTO:
    label_id: str
    user_id: str
    event_uuid: str
    title: str
    location: str
    participant_person_uuids: List[str] = field(default_factory=list)


@dataclass(slots=True)
class AnnotationTaskDTO:
    annotation_task_id: str
    task_type: str
    target_id: str
    target_type: str
    status: str


@dataclass(slots=True)
class AnnotationDecisionDTO:
    decision_id: str
    annotation_task_id: str
    decision_type: str
    comment: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReviewIssueDTO:
    issue_id: str
    issue_type: str
    severity: str
    summary: str
    target_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MetricSnapshotDTO:
    metric_name: str
    value: float
    evaluated_at: str
    scope: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ErrorBucketDTO:
    bucket_id: str
    bucket_name: str
    count: int
    examples: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class EvaluationRunDTO:
    evaluation_run_id: str
    user_id: Optional[str]
    evaluated_at: str
    metric_snapshots: List[MetricSnapshotDTO] = field(default_factory=list)
    error_buckets: List[ErrorBucketDTO] = field(default_factory=list)
