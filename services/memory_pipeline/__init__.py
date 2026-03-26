"""
记忆主链路 pipeline。
"""

from .primary_person import PrimaryDecision
from .types import FactFieldDecision, FieldBundle, FieldSpec, GroupArtifact, MemoryState, PersonScreening, RelationshipDossier

__all__ = [
    "FactFieldDecision",
    "FieldBundle",
    "FieldSpec",
    "GroupArtifact",
    "MemoryState",
    "PersonScreening",
    "PrimaryDecision",
    "RelationshipDossier",
]

# 下游审计模块依赖 profile_agent 外部工程配置，默认不阻塞主链包导入。
try:  # pragma: no cover - optional integration
    from .downstream_audit import (
        apply_downstream_profile_backflow,
        apply_downstream_protagonist_backflow,
        apply_downstream_relationship_backflow,
        build_downstream_audit_report,
        run_downstream_profile_agent_audit,
    )
    from .offline_profile_eval import run_two_round_offline_eval
    from .profile_agent_adapter import build_profile_agent_extractor_outputs

    __all__.extend(
        [
            "apply_downstream_profile_backflow",
            "apply_downstream_protagonist_backflow",
            "apply_downstream_relationship_backflow",
            "build_downstream_audit_report",
            "build_profile_agent_extractor_outputs",
            "run_two_round_offline_eval",
            "run_downstream_profile_agent_audit",
        ]
    )
except Exception:  # pragma: no cover - keep core pipeline importable
    pass
