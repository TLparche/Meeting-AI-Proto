from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any


FSM_STATES = ("PROPOSED", "ACTIVE", "CLOSING", "CLOSED")
RULE_FLAGS = ("topic_shift_sustained", "collective_intent", "decision_slots")
CLOSING_HINTS = (
    "합의",
    "동의",
    "확정",
    "결정",
    "승인",
    "표결",
    "가결",
    "final",
    "confirmed",
    "agreed",
    "approved",
    "vote",
)


def _now_ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _agenda_id(title: str) -> str:
    base = (title or "").strip().lower()
    if not base:
        base = "agenda"
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]
    return f"ag_{digest}"


def _as_bool(x: Any) -> bool:
    return bool(x)


def _candidate_rule_count(candidate: dict[str, Any]) -> int:
    reasons = set(str(r) for r in (candidate.get("reasons") or []))
    count = 0
    for rule in RULE_FLAGS:
        if rule in reasons:
            count += 1
    return count


def _is_promotable(candidate: dict[str, Any]) -> bool:
    return _candidate_rule_count(candidate) >= 2


def _closing_confirmed(transcript: list[dict], lookback_turns: int = 8) -> bool:
    if not transcript:
        return False
    recent = transcript[-lookback_turns:]
    text = " ".join(str(t.get("text") or "").lower() for t in recent)
    return any(h in text for h in CLOSING_HINTS)


def _append_event(
    events: list[dict],
    *,
    event_type: str,
    agenda_id: str,
    title: str,
    prev_state: str | None,
    next_state: str | None,
    reason: str,
    active_before: str | None,
    active_after: str | None,
) -> None:
    events.append(
        {
            "ts": _now_ts(),
            "type": event_type,
            "agenda_id": agenda_id,
            "title": title,
            "prev_state": prev_state,
            "next_state": next_state,
            "reason": reason,
            "active_before": active_before,
            "active_after": active_after,
        }
    )


def run_agenda_fsm(
    *,
    agenda_state_map: dict[str, dict],
    active_agenda_id: str,
    agenda_candidates: list[dict],
    analysis: dict,
    transcript: list[dict],
    existing_events: list[dict] | None = None,
) -> dict[str, Any]:
    state_map = {k: dict(v) for k, v in (agenda_state_map or {}).items()}
    events = list(existing_events or [])
    active_before = active_agenda_id or ""
    now = _now_ts()

    llm_active_title = str((analysis.get("agenda", {}).get("active", {}).get("title") or "")).strip()
    llm_decision_lock = _as_bool(
        ((analysis.get("intervention") or {}).get("decision_lock") or {}).get("triggered", False)
    )

    # Seed active agenda if map is empty.
    if not state_map and llm_active_title:
        seeded_id = _agenda_id(llm_active_title)
        state_map[seeded_id] = {
            "agenda_id": seeded_id,
            "title": llm_active_title,
            "state": "ACTIVE",
            "confidence": float((analysis.get("agenda", {}).get("active", {}).get("confidence") or 0.55)),
            "created_at": now,
            "updated_at": now,
            "source": "llm_active_seed",
        }
        active_agenda_id = seeded_id

    # Upsert PROPOSED candidates.
    candidate_by_id: dict[str, dict] = {}
    for cand in agenda_candidates or []:
        title = str(cand.get("title") or "").strip()
        if not title:
            continue
        aid = _agenda_id(title)
        candidate_by_id[aid] = cand
        entry = state_map.get(aid)
        if entry is None:
            state_map[aid] = {
                "agenda_id": aid,
                "title": title,
                "state": "PROPOSED",
                "confidence": float(cand.get("confidence", 0.6)),
                "created_at": str(cand.get("created_at") or now),
                "updated_at": now,
                "source": "tracker_candidate",
                "signals": dict(cand.get("signals") or {}),
                "reasons": list(cand.get("reasons") or []),
            }
            _append_event(
                events,
                event_type="agenda_created",
                agenda_id=aid,
                title=title,
                prev_state=None,
                next_state="PROPOSED",
                reason="candidate_generated",
                active_before=active_before or None,
                active_after=active_agenda_id or None,
            )
        else:
            if entry.get("state") != "CLOSED":
                entry["confidence"] = max(float(entry.get("confidence", 0.0)), float(cand.get("confidence", 0.0)))
                entry["updated_at"] = now
                entry["signals"] = dict(cand.get("signals") or entry.get("signals") or {})
                entry["reasons"] = list(cand.get("reasons") or entry.get("reasons") or [])

    # ACTIVE -> CLOSING via Decision Lock.
    if active_agenda_id and active_agenda_id in state_map:
        entry = state_map[active_agenda_id]
        if entry.get("state") == "ACTIVE" and llm_decision_lock:
            entry["state"] = "CLOSING"
            entry["updated_at"] = now
            _append_event(
                events,
                event_type="state_transition",
                agenda_id=active_agenda_id,
                title=str(entry.get("title") or ""),
                prev_state="ACTIVE",
                next_state="CLOSING",
                reason="decision_lock_triggered",
                active_before=active_before or None,
                active_after=active_agenda_id,
            )

    # CLOSING -> CLOSED via vote/agreement/final confirmation hints.
    if active_agenda_id and active_agenda_id in state_map:
        entry = state_map[active_agenda_id]
        if entry.get("state") == "CLOSING" and _closing_confirmed(transcript):
            entry["state"] = "CLOSED"
            entry["updated_at"] = now
            _append_event(
                events,
                event_type="state_transition",
                agenda_id=active_agenda_id,
                title=str(entry.get("title") or ""),
                prev_state="CLOSING",
                next_state="CLOSED",
                reason="vote_or_agreement_confirmed",
                active_before=active_before or None,
                active_after=None,
            )
            active_agenda_id = ""

    # PROPOSED -> ACTIVE (2-of-3). If no active, promote best candidate.
    has_active = bool(active_agenda_id and active_agenda_id in state_map and state_map[active_agenda_id].get("state") in ("ACTIVE", "CLOSING"))
    if has_active and active_agenda_id and active_agenda_id in state_map:
        current = state_map[active_agenda_id]
        if current.get("state") == "ACTIVE":
            active_conf = float(current.get("confidence", 0.0))
            switchables: list[tuple[float, str]] = []
            for aid, entry in state_map.items():
                if aid == active_agenda_id or entry.get("state") != "PROPOSED":
                    continue
                cand = candidate_by_id.get(aid, {})
                if not _is_promotable(cand):
                    continue
                reasons = set(str(r) for r in (cand.get("reasons") or []))
                if "topic_shift_sustained" not in reasons:
                    continue
                conf = float(entry.get("confidence", 0.0))
                if conf >= active_conf + 0.05:
                    switchables.append((conf, aid))
            switchables.sort(reverse=True)
            if switchables:
                _, next_id = switchables[0]
                prev_id = active_agenda_id
                prev_entry = state_map[prev_id]
                next_entry = state_map[next_id]
                prev_entry["state"] = "PROPOSED"
                prev_entry["updated_at"] = now
                next_entry["state"] = "ACTIVE"
                next_entry["updated_at"] = now
                active_agenda_id = next_id
                _append_event(
                    events,
                    event_type="state_transition",
                    agenda_id=prev_id,
                    title=str(prev_entry.get("title") or ""),
                    prev_state="ACTIVE",
                    next_state="PROPOSED",
                    reason="focus_shift_to_new_agenda",
                    active_before=prev_id,
                    active_after=next_id,
                )
                _append_event(
                    events,
                    event_type="state_transition",
                    agenda_id=next_id,
                    title=str(next_entry.get("title") or ""),
                    prev_state="PROPOSED",
                    next_state="ACTIVE",
                    reason="2_of_3_rule_satisfied",
                    active_before=prev_id,
                    active_after=next_id,
                )
                has_active = True

    if not has_active:
        promotable = []
        for aid, entry in state_map.items():
            if entry.get("state") != "PROPOSED":
                continue
            cand = candidate_by_id.get(aid, {})
            if _is_promotable(cand):
                promotable.append((float(entry.get("confidence", 0.0)), aid))
        promotable.sort(reverse=True)
        if promotable:
            _, next_id = promotable[0]
            next_entry = state_map[next_id]
            prev_active = active_agenda_id or None
            next_entry["state"] = "ACTIVE"
            next_entry["updated_at"] = now
            active_agenda_id = next_id
            _append_event(
                events,
                event_type="state_transition",
                agenda_id=next_id,
                title=str(next_entry.get("title") or ""),
                prev_state="PROPOSED",
                next_state="ACTIVE",
                reason="2_of_3_rule_satisfied",
                active_before=prev_active,
                active_after=next_id,
            )

    # If active agenda changed, emit explicit active change event.
    if (active_before or "") != (active_agenda_id or ""):
        to_id = active_agenda_id or None
        from_id = active_before or None
        to_title = state_map.get(active_agenda_id, {}).get("title", "") if active_agenda_id else ""
        _append_event(
            events,
            event_type="active_agenda_changed",
            agenda_id=active_agenda_id or "",
            title=to_title,
            prev_state=None,
            next_state=None,
            reason="active_agenda_id_changed",
            active_before=from_id,
            active_after=to_id,
        )

    # Compose agenda_state_map output sorted by status priority.
    priority = {"ACTIVE": 0, "CLOSING": 1, "PROPOSED": 2, "CLOSED": 3}
    ordered = sorted(
        state_map.values(),
        key=lambda x: (priority.get(str(x.get("state") or "PROPOSED"), 9), -float(x.get("confidence", 0.0))),
    )
    out_map = {str(e["agenda_id"]): e for e in ordered}

    return {
        "agenda_state_map": out_map,
        "active_agenda_id": active_agenda_id or "",
        "events": events[-120:],
    }
