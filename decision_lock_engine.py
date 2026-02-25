from __future__ import annotations

from datetime import datetime
from typing import Any


AGREE_HINTS = (
    "동의",
    "찬성",
    "승인",
    "확정",
    "진행합시다",
    "좋습니다",
    "go with",
    "agree",
    "approved",
    "final",
    "finalize",
    "yes",
)

DISAGREE_HINTS = (
    "반대",
    "우려",
    "보류",
    "어렵",
    "재검토",
    "disagree",
    "not sure",
    "hold",
    "no",
)


def _parse_hms(ts: str) -> int | None:
    ts = (ts or "").strip()
    if not ts:
        return None
    try:
        dt = datetime.strptime(ts, "%H:%M:%S")
        return dt.hour * 3600 + dt.minute * 60 + dt.second
    except ValueError:
        return None


def _window_by_seconds(transcript: list[dict], seconds: int) -> list[dict]:
    if not transcript:
        return []
    last_ts = _parse_hms(str(transcript[-1].get("timestamp") or ""))
    if last_ts is None:
        return transcript[-max(8, seconds // 8) :]
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
    return out or transcript[-max(8, seconds // 8) :]


def _stance_convergence(transcript_recent: list[dict]) -> tuple[float, int, int, int]:
    agree = 0
    disagree = 0
    stance_total = 0
    speakers: set[str] = set()
    for turn in transcript_recent:
        text = str(turn.get("text") or "").lower()
        speakers.add(str(turn.get("speaker") or "미상"))
        has_agree = any(h in text for h in AGREE_HINTS)
        has_disagree = any(h in text for h in DISAGREE_HINTS)
        if has_agree and not has_disagree:
            agree += 1
            stance_total += 1
        elif has_disagree and not has_agree:
            disagree += 1
            stance_total += 1
        elif has_agree and has_disagree:
            stance_total += 1
    if stance_total <= 0:
        return 0.0, stance_total, agree, disagree
    conv = max(agree, disagree) / stance_total
    if len(speakers) < 2:
        conv *= 0.85
    return conv, stance_total, agree, disagree


def run_decision_lock(
    *,
    transcript: list[dict],
    meeting_elapsed_sec: float,
    stagnation_flag: bool,
    previous_decision_lock: bool,
) -> dict[str, Any]:
    recent = _window_by_seconds(transcript, 120)
    stance_conv, stance_total, agree_count, disagree_count = _stance_convergence(recent)

    trigger_stance = stance_conv >= 0.70
    trigger_stagnation = bool(stagnation_flag)
    trigger_timebox = meeting_elapsed_sec >= 900.0

    triggered = trigger_stance or trigger_stagnation or trigger_timebox or previous_decision_lock
    reasons: list[str] = []
    if trigger_stance:
        reasons.append("stance_convergence>=0.70")
    if trigger_stagnation:
        reasons.append("circular_stagnation")
    if trigger_timebox:
        reasons.append("timebox>=15m")
    if previous_decision_lock and not reasons:
        reasons.append("persisted")

    reason = " | ".join(reasons)
    if not reason:
        reason = "not_triggered"

    return {
        "triggered": triggered,
        "reason": reason,
        "debug": {
            "stance_convergence": round(stance_conv, 4),
            "stance_total": stance_total,
            "agree_count": agree_count,
            "disagree_count": disagree_count,
            "elapsed_sec": round(meeting_elapsed_sec, 1),
            "trigger_stance": trigger_stance,
            "trigger_stagnation": trigger_stagnation,
            "trigger_timebox": trigger_timebox,
        },
    }

