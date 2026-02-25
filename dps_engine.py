from __future__ import annotations

from typing import Any


def _count_non_empty(values: list[Any] | None) -> int:
    return len([v for v in (values or []) if str(v).strip()])


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_dps(
    *,
    analysis: dict,
    agenda_state_map: dict[str, dict] | None,
    active_agenda_id: str,
) -> dict[str, Any]:
    keywords = analysis.get("keywords") or {}
    k_core = keywords.get("k_core") or {}
    k_facet = keywords.get("k_facet") or {}

    option_count = _count_non_empty(k_facet.get("options"))
    constraint_count = _count_non_empty(k_core.get("constraints"))
    criterion_count = _count_non_empty(k_core.get("criteria"))
    evidence_count = _count_non_empty(k_facet.get("evidence"))
    action_count = _count_non_empty(k_facet.get("actions"))

    option_cov = _clip01(option_count / 3.0)
    constraint_cov = _clip01(constraint_count / 2.0)
    evidence_cov = _clip01(evidence_count / 2.0)

    trade_signal = 0.0
    if option_count >= 2:
        trade_signal += 0.45
    if (constraint_count + criterion_count) >= 2:
        trade_signal += 0.35
    if criterion_count >= 1:
        trade_signal += 0.20
    tradeoff_cov = _clip01(trade_signal)

    decision_lock = bool(((analysis.get("intervention") or {}).get("decision_lock") or {}).get("triggered", False))
    active_state = ""
    if active_agenda_id and (agenda_state_map or {}).get(active_agenda_id):
        active_state = str((agenda_state_map or {}).get(active_agenda_id, {}).get("state") or "")
    if not active_state:
        active_state = str(((analysis.get("agenda") or {}).get("active") or {}).get("status") or "")

    closing_score = 0.0
    if action_count >= 1:
        closing_score += 0.25
    if decision_lock:
        closing_score += 0.35
    if active_state == "CLOSING":
        closing_score += 0.30
    elif active_state == "CLOSED":
        closing_score += 0.40
    elif active_state == "ACTIVE":
        closing_score += 0.10
    closing_readiness = _clip01(closing_score)

    score = (
        0.25 * option_cov
        + 0.20 * constraint_cov
        + 0.20 * evidence_cov
        + 0.20 * tradeoff_cov
        + 0.15 * closing_readiness
    )
    score = _clip01(score)

    return {
        "score": round(score, 4),
        "breakdown": {
            "option_coverage": round(option_cov, 4),
            "constraint_coverage": round(constraint_cov, 4),
            "evidence_coverage": round(evidence_cov, 4),
            "tradeoff_coverage": round(tradeoff_cov, 4),
            "closing_readiness": round(closing_readiness, 4),
            "counts": {
                "options": option_count,
                "constraints": constraint_count,
                "criteria": criterion_count,
                "evidence": evidence_count,
                "actions": action_count,
            },
            "active_state": active_state or "UNKNOWN",
            "decision_lock": decision_lock,
        },
    }

