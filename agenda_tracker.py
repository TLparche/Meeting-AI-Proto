from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime
from typing import Any

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.+#/\-]*|[가-힣]{2,}")

INTENT_HINTS = (
    "결정",
    "확정",
    "합의",
    "정리",
    "선택",
    "비교",
    "vote",
    "decide",
    "choose",
    "finalize",
    "approve",
)

SUB_ISSUE_HINTS = {
    "pricing": ("가격", "요금", "비용", "원가", "pricing", "cost", "price"),
    "schedule": ("일정", "기한", "마감", "스케줄", "schedule", "deadline", "timeline"),
    "policy": ("정책", "규정", "정합", "가이드", "policy", "rule", "compliance"),
    "owner": ("담당", "오너", "책임", "owner", "assignee"),
}

SEPARATE_CLOSING_HINTS = ("표결", "vote", "승인", "결재", "확정", "finalize")


def _clamp(x: float, low: float, high: float) -> float:
    return max(low, min(high, x))


def _parse_hms(ts: str) -> int | None:
    ts = (ts or "").strip()
    if not ts:
        return None
    try:
        parsed = datetime.strptime(ts, "%H:%M:%S")
        return parsed.hour * 3600 + parsed.minute * 60 + parsed.second
    except ValueError:
        return None


def _window_by_seconds(transcript: list[dict], seconds: int) -> list[dict]:
    if not transcript:
        return []
    last_ts = _parse_hms(str(transcript[-1].get("timestamp") or ""))
    if last_ts is None:
        fallback = max(6, seconds // 8)
        return transcript[-fallback:]

    out: list[dict] = []
    for turn in reversed(transcript):
        ts = _parse_hms(str(turn.get("timestamp") or ""))
        if ts is None:
            break
        diff = last_ts - ts
        if diff < 0:
            diff += 24 * 3600
        if diff <= seconds:
            out.append(turn)
        else:
            break
    out.reverse()
    return out or transcript[-max(6, seconds // 8) :]


def _duration_seconds(turns: list[dict]) -> int:
    if len(turns) <= 1:
        return 0
    first_ts = _parse_hms(str(turns[0].get("timestamp") or ""))
    last_ts = _parse_hms(str(turns[-1].get("timestamp") or ""))
    if first_ts is not None and last_ts is not None:
        diff = last_ts - first_ts
        if diff < 0:
            diff += 24 * 3600
        return diff
    return max(0, (len(turns) - 1) * 8)


def _tokenize(text: str) -> list[str]:
    raw = TOKEN_RE.findall(text or "")
    return [t.lower() if re.fullmatch(r"[A-Za-z0-9_.+#/\-]+", t) else t for t in raw]


def _vectorize(text: str) -> Counter[str]:
    return Counter(_tokenize(text))


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    for k, v in a.items():
        dot += v * b.get(k, 0)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _recent_similarity(active_agenda: str, turns: list[dict]) -> float:
    if not turns:
        return 0.0
    active_vec = _vectorize(active_agenda or "")
    recent_text = " ".join(str(t.get("text") or "") for t in turns)
    recent_vec = _vectorize(recent_text)
    return _cosine(active_vec, recent_vec)


def _collective_intent(turns_60: list[dict]) -> tuple[bool, int, int]:
    intent_count = 0
    speakers: set[str] = set()
    for turn in turns_60:
        text = str(turn.get("text") or "").lower()
        speakers.add(str(turn.get("speaker") or "미상"))
        if any(h in text for h in INTENT_HINTS):
            intent_count += 1
    return intent_count >= 2 and len(speakers) >= 2, intent_count, len(speakers)


def _decision_slots(keywords: dict[str, Any]) -> tuple[bool, int, int]:
    k_core = keywords.get("k_core") or {}
    object_count = len([x for x in (k_core.get("object") or []) if str(x).strip()])
    support_count = len([x for x in (k_core.get("constraints") or []) if str(x).strip()]) + len(
        [x for x in (k_core.get("criteria") or []) if str(x).strip()]
    )
    return object_count >= 1 and support_count >= 1, object_count, support_count


def _detect_sub_issue(
    turns_120: list[dict],
    keywords: dict[str, Any],
    active_agenda: str,
) -> tuple[bool, str, str]:
    joined = " ".join(str(t.get("text") or "").lower() for t in turns_120)
    k_core = keywords.get("k_core") or {}
    k_facet = keywords.get("k_facet") or {}
    constraints = " ".join(str(x).lower() for x in (k_core.get("constraints") or []))
    actions = " ".join(str(x).lower() for x in (k_facet.get("actions") or []))

    best_label = ""
    best_hits = 0
    for label, hints in SUB_ISSUE_HINTS.items():
        hit_count = sum(1 for h in hints if h in joined or h in constraints or h in actions)
        if hit_count > best_hits:
            best_hits = hit_count
            best_label = label

    separate_closing = any(h in joined for h in SEPARATE_CLOSING_HINTS)
    has_owner_or_action = len(k_facet.get("actions") or []) >= 1
    independent_constraints = best_hits >= 2
    sub_issue = bool(best_label) and (separate_closing or has_owner_or_action or independent_constraints)
    if not sub_issue:
        return False, "", ""

    if best_label and best_label in (active_agenda or "").lower():
        return False, "", ""

    reason = []
    if separate_closing:
        reason.append("separate closing/vote")
    if has_owner_or_action:
        reason.append("separate owner/action")
    if independent_constraints:
        reason.append("independent constraints")
    return True, best_label, ", ".join(reason)


def _top_terms(turns: list[dict], limit: int = 10) -> list[dict[str, Any]]:
    text = " ".join(str(t.get("text") or "") for t in turns)
    cnt = Counter(_tokenize(text))
    items = cnt.most_common(limit)
    total = max(1, sum(v for _, v in items))
    return [{"term": k, "weight": round(v / total, 4)} for k, v in items]


def _candidate_title(
    active_agenda: str,
    keywords: dict[str, Any],
    sub_issue: bool,
    sub_issue_label: str,
) -> str:
    if sub_issue and sub_issue_label:
        label_map = {
            "pricing": "가격/비용",
            "schedule": "일정/기한",
            "policy": "정책/규정",
            "owner": "담당/책임",
        }
        return f"{label_map.get(sub_issue_label, sub_issue_label)} 세부 이슈 클로징"

    k_core = keywords.get("k_core") or {}
    objs = [str(x).strip() for x in (k_core.get("object") or []) if str(x).strip()]
    if objs:
        title = objs[0]
    elif active_agenda:
        title = f"{active_agenda} - 세부 결정"
    else:
        title = "신규 의사결정 아젠다"
    return title[:64]


def run_agenda_tracker(
    transcript: list[dict],
    active_agenda: str,
    agenda_stack: list[dict],
    keywords: dict[str, Any],
    existing_vectors: dict[str, Any] | None,
) -> dict[str, Any]:
    turns_60 = _window_by_seconds(transcript, 60)
    turns_120 = _window_by_seconds(transcript, 120)
    sim60 = _recent_similarity(active_agenda, turns_60)
    sim120 = _recent_similarity(active_agenda, turns_120)
    dur120 = _duration_seconds(turns_120)

    topic_shift_sustained = sim60 < 0.55 and dur120 >= 120 and sim120 < 0.62
    collective_intent, intent_count, speaker_count = _collective_intent(turns_60)
    decision_slots_ok, object_count, support_count = _decision_slots(keywords)

    passed = int(topic_shift_sustained) + int(collective_intent) + int(decision_slots_ok)
    sub_issue, sub_issue_label, sub_issue_reason = _detect_sub_issue(turns_120, keywords, active_agenda)

    agenda_candidates: list[dict[str, Any]] = []
    now_ts = str((transcript[-1].get("timestamp") if transcript else "") or "")

    if passed >= 2:
        title = _candidate_title(active_agenda, keywords, sub_issue, sub_issue_label)
        confidence = _clamp(
            0.42
            + 0.15 * passed
            + (0.12 if sub_issue else 0.0)
            + max(0.0, (0.55 - sim60)) * 0.4,
            0.0,
            0.98,
        )
        reasons = []
        if topic_shift_sustained:
            reasons.append("topic_shift_sustained")
        if collective_intent:
            reasons.append("collective_intent")
        if decision_slots_ok:
            reasons.append("decision_slots")
        if sub_issue:
            reasons.append(f"sub_issue:{sub_issue_label}")
        agenda_candidates.append(
            {
                "title": title,
                "status": "PROPOSED",
                "confidence": round(confidence, 2),
                "created_at": now_ts,
                "reasons": reasons,
                "sub_issue_promoted": sub_issue,
                "signals": {
                    "sim60": round(sim60, 3),
                    "sim120": round(sim120, 3),
                    "duration120_sec": dur120,
                    "intent_count_60s": intent_count,
                    "speakers_60s": speaker_count,
                    "object_slots": object_count,
                    "support_slots": support_count,
                },
            }
        )
    elif sub_issue:
        # Sub-issue promotion can independently create a PROPOSED agenda.
        title = _candidate_title(active_agenda, keywords, True, sub_issue_label)
        confidence = _clamp(0.58 + max(0.0, (0.55 - sim60)) * 0.2, 0.0, 0.9)
        agenda_candidates.append(
            {
                "title": title,
                "status": "PROPOSED",
                "confidence": round(confidence, 2),
                "created_at": now_ts,
                "reasons": [f"sub_issue:{sub_issue_label}", "sub_issue_promotion"],
                "sub_issue_promoted": True,
                "signals": {
                    "sim60": round(sim60, 3),
                    "sim120": round(sim120, 3),
                    "duration120_sec": dur120,
                    "intent_count_60s": intent_count,
                    "speakers_60s": speaker_count,
                    "object_slots": object_count,
                    "support_slots": support_count,
                },
            }
        )

    vectors = dict(existing_vectors or {})
    focus_titles = [active_agenda] if active_agenda else []
    focus_titles.extend(
        (str(item.get("title") or "").strip() for item in agenda_stack if str(item.get("title") or "").strip())
    )
    focus_titles.extend(c["title"] for c in agenda_candidates)
    terms = _top_terms(turns_120, limit=12)
    for title in focus_titles:
        if not title:
            continue
        vectors[title] = {
            "updated_at": now_ts,
            "sample_count": len(turns_120),
            "terms": terms,
        }

    return {
        "agenda_candidates": agenda_candidates,
        "agenda_vectors": vectors,
        "tracker_debug": {
            "topic_shift_sustained": topic_shift_sustained,
            "collective_intent": collective_intent,
            "decision_slots": decision_slots_ok,
            "sub_issue_promoted": sub_issue,
            "signals": {
                "sim60": round(sim60, 3),
                "sim120": round(sim120, 3),
                "duration120_sec": dur120,
                "intent_count_60s": intent_count,
                "speakers_60s": speaker_count,
                "object_slots": object_count,
                "support_slots": support_count,
            },
        },
    }
