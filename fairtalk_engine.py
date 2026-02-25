from __future__ import annotations

import math
from collections import Counter
from datetime import datetime
from typing import Any

TURN_GRAB_HINTS = (
    "잠깐",
    "제가",
    "저 먼저",
    "한마디",
    "말씀드리면",
    "제 의견",
    "질문",
    "정리하면",
    "말해도",
    "발언",
    "can i",
    "let me",
    "one sec",
    "hold on",
    "i want to say",
)

QUESTION_HINTS = (
    "?",
    "어떻게",
    "왜",
    "뭐",
    "무엇",
    "맞나요",
    "가능한가",
    "how",
    "why",
    "what",
)

FIRST_PERSON_HINTS = (
    "제가",
    "저는",
    "저 ",
    "i ",
    "i'm",
    "let me",
    "my point",
)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _contains_any(text: str, hints: tuple[str, ...]) -> bool:
    lowered = (text or "").lower()
    return any(h in lowered for h in hints)


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


def _turn_intent_score(text: str) -> float:
    s = (text or "").strip()
    if not s:
        return 0.0
    score = 0.10
    has_question_mark = "?" in s
    if has_question_mark:
        score += 0.42
    elif _contains_any(s, QUESTION_HINTS):
        score += 0.12
    if _contains_any(s, TURN_GRAB_HINTS):
        score += 0.30
    if _contains_any(s, FIRST_PERSON_HINTS):
        score += 0.14
    length = len(s)
    if 8 <= length <= 48:
        score += 0.08
    if length <= 5:
        score -= 0.08
    return _clip01(score)


def _estimate_imbalance(turns: list[dict]) -> int:
    if not turns:
        return 0
    counts = Counter(str(t.get("speaker") or "미상") for t in turns)
    if len(counts) <= 1:
        return 0
    mx = max(counts.values())
    mn = min(counts.values())
    if mx <= 0:
        return 0
    return int(round(((mx - mn) / mx) * 100))


def run_fairtalk_engine(
    *,
    transcript: list[dict],
    monitor_state: dict[str, dict[str, float | str | None]] | None,
    now_ts: float,
) -> dict[str, Any]:
    monitor = dict(monitor_state or {})
    turns_90 = _window_by_seconds(transcript, 90)
    turns_120 = _window_by_seconds(transcript, 120)
    if not turns_90:
        return {
            "participants": [],
            "fairtalk_schema": [],
            "imbalance": 0,
            "monitor": monitor,
            "debug": {"active_speakers": 0, "soft_count": 0, "strong_count": 0, "rule": "intent_only"},
        }

    latest_ts_raw = str(turns_90[-1].get("timestamp") or "")
    latest_ts = _parse_hms(latest_ts_raw)
    metrics: dict[str, dict[str, float]] = {}

    for idx, turn in enumerate(turns_90):
        speaker = str(turn.get("speaker") or "미상").strip() or "미상"
        text = str(turn.get("text") or "")
        if latest_ts is not None:
            tts = _parse_hms(str(turn.get("timestamp") or ""))
            if tts is None:
                age_sec = float((len(turns_90) - 1 - idx) * 4)
            else:
                diff = latest_ts - tts
                if diff < 0:
                    diff += 24 * 3600
                age_sec = float(diff)
        else:
            age_sec = float((len(turns_90) - 1 - idx) * 4)

        decay = max(0.25, math.exp(-age_sec / 16.0))
        turn_score = _turn_intent_score(text) * decay
        m = metrics.setdefault(
            speaker,
            {
                "max_score": 0.0,
                "weighted_sum": 0.0,
                "weight_sum": 0.0,
                "turn_count": 0.0,
                "last_age": 999.0,
                "last_sig_idx": -1.0,
            },
        )
        m["max_score"] = max(m["max_score"], turn_score)
        m["weighted_sum"] += turn_score
        m["weight_sum"] += decay
        m["turn_count"] += 1.0
        if age_sec <= m["last_age"]:
            m["last_age"] = age_sec
            m["last_sig_idx"] = float(idx)
            m["last_ts"] = str(turn.get("timestamp") or "")
            m["last_text"] = text[:180]

    participants: list[dict[str, Any]] = []
    union_speakers = set(monitor.keys()) | set(metrics.keys())

    for speaker in union_speakers:
        m = metrics.get(speaker)
        state = dict(monitor.get(speaker) or {})
        strong_since = state.get("strong_since")
        soft_since = state.get("soft_since")
        p_intent = 0.0
        last_seen_sec = 999.0
        glow = "none"

        if m:
            avg = (m["weighted_sum"] / m["weight_sum"]) if m["weight_sum"] > 0 else 0.0
            p_intent = _clip01(max(m["max_score"], avg + min(0.12, 0.02 * (m["turn_count"] - 1.0))))
            latest_sig = f"{m.get('last_ts','')}|{m.get('last_sig_idx',-1)}|{m.get('last_text','')}"
            prev_sig = str(state.get("last_sig") or "")
            prev_seen_wall = state.get("last_seen_wall_ts")
            if prev_sig == latest_sig and isinstance(prev_seen_wall, (int, float)):
                wall_age = max(0.0, now_ts - float(prev_seen_wall))
            else:
                wall_age = 0.0
                state["last_seen_wall_ts"] = now_ts
            state["last_sig"] = latest_sig
            p_intent = _clip01(p_intent * math.exp(-wall_age / 8.0))
            last_seen_sec = wall_age
            state["last_seen_ts"] = now_ts - last_seen_sec
        else:
            prev_seen_wall = state.get("last_seen_wall_ts")
            last_seen_sec = max(999.0, now_ts - float(prev_seen_wall)) if isinstance(prev_seen_wall, (int, float)) else 999.0
            state["last_seen_ts"] = state.get("last_seen_ts")

        # Anti-intervention: 침묵 자체는 트리거하지 않음.
        if m is None or last_seen_sec > 6.0:
            strong_since = None
            soft_since = None
            glow = "none"
        elif p_intent >= 0.70:
            if not isinstance(strong_since, (int, float)):
                strong_since = now_ts
            soft_since = None
            glow = "strong" if (now_ts - float(strong_since)) >= 2.0 else "none"
        elif 0.60 <= p_intent < 0.70:
            strong_since = None
            if not isinstance(soft_since, (int, float)):
                soft_since = now_ts
            glow = "soft"
        else:
            strong_since = None
            soft_since = None
            glow = "none"

        state["strong_since"] = strong_since
        state["soft_since"] = soft_since
        state["p_intent"] = p_intent
        state["glow"] = glow
        monitor[speaker] = state

        if m:
            participants.append(
                {
                    "speaker": speaker,
                    "p_intent": round(p_intent, 4),
                    "glow": glow,
                    "intent_active": glow in {"soft", "strong"},
                    "last_seen_sec": round(last_seen_sec, 2),
                }
            )

    participants.sort(
        key=lambda x: (
            0 if x["glow"] == "strong" else (1 if x["glow"] == "soft" else 2),
            -float(x["p_intent"]),
            float(x["last_seen_sec"]),
        )
    )
    participants = participants[:8]

    fairtalk_schema = [{"speaker": p["speaker"], "p_intent": p["p_intent"]} for p in participants]
    strong_count = len([p for p in participants if p["glow"] == "strong"])
    soft_count = len([p for p in participants if p["glow"] == "soft"])

    return {
        "participants": participants,
        "fairtalk_schema": fairtalk_schema,
        "imbalance": _estimate_imbalance(turns_120),
        "monitor": monitor,
        "debug": {
            "active_speakers": len(participants),
            "soft_count": soft_count,
            "strong_count": strong_count,
            "rule": "intent_only",
        },
    }
