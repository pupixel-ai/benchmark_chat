"""
跨层一致性检查
"""
from __future__ import annotations

from typing import Dict, List

from models import Event, Relationship


def build_consistency_report(events: List[Event], relationships: List[Relationship], structured_profile: Dict) -> Dict:
    """对关系层和画像层的核心字段做轻量一致性检查。"""
    issues = []
    profile_relationships = structured_profile.get("long_term_facts", {}).get("relationships", {})
    profile_partner = profile_relationships.get("intimate_partner", {}).get("value")
    profile_circle_size = profile_relationships.get("close_circle_size", {}).get("value")

    romantic_partner = next(
        (rel.person_id for rel in relationships if rel.relationship_type == "romantic"),
        None,
    )
    close_circle_size = sum(
        1 for rel in relationships if rel.relationship_type in {"romantic", "bestie", "close_friend", "family"}
    )

    if profile_partner and profile_partner != romantic_partner:
        issues.append({
            "code": "INTIMATE_PARTNER_MISMATCH",
            "severity": "high",
            "message": f"profile.intimate_partner={profile_partner} 与 LP2 romantic={romantic_partner} 不一致",
        })

    if profile_circle_size is not None and profile_circle_size != close_circle_size:
        issues.append({
            "code": "CLOSE_CIRCLE_SIZE_MISMATCH",
            "severity": "medium",
            "message": f"profile.close_circle_size={profile_circle_size} 与 LP2 close_circle_size={close_circle_size} 不一致",
        })

    return {
        "summary": {
            "issue_count": len(issues),
            "high_risk_issue_count": sum(1 for issue in issues if issue["severity"] == "high"),
        },
        "issues": issues,
    }
