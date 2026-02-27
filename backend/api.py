from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from llm_client import get_client

ROOT = Path(__file__).resolve().parent.parent
WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "large")
SUMMARY_INTERVAL = 4

STOPWORDS = {
    "그냥",
    "이제",
    "저기",
    "그게",
    "그거",
    "이거",
    "저거",
    "그리고",
    "하지만",
    "그러면",
    "그래서",
    "또는",
    "이번",
    "그런",
    "이런",
    "저런",
    "정도",
    "부분",
    "관련",
    "대해서",
    "안건",
    "회의",
    "논의",
    "말씀",
    "의견",
    "지금",
    "오늘",
    "내일",
    "이번주",
    "다음주",
    "정말",
    "진짜",
    "아주",
    "거의",
    "일단",
    "맞아요",
    "맞습니다",
    "있습니다",
    "없습니다",
    "한다",
    "했다",
    "하고",
    "해서",
    "하면",
    "하며",
    "이면",
    "이면은",
    "the",
    "and",
    "that",
    "this",
    "with",
    "from",
    "about",
    "저는",
    "저희",
    "저도",
    "제가",
    "그렇죠",
    "거예요",
    "거죠",
    "이게",
    "그게",
    "어떤",
    "그러니까",
    "근데",
    "같아요",
    "같고",
    "있고",
    "있다",
    "하는",
    "하게",
    "되어",
    "그렇게",
    "이렇게",
    "많이",
    "하나",
    "계속",
    "아니라",
    "보니까",
    "나온",
    "있습니다",
    "합니다",
    "겁니다",
    "수도",
    "때문에",
    "가지고",
    "laughing",
    "감사합니다",
    "포인트",
    "처음",
    "틀에서",
    "party",
    "name",
}

DECISION_PAT = re.compile(r"(결정|확정|합의|채택|의결|하기로|정리하면|정하자)")
ACTION_PAT = re.compile(r"(담당|까지|하겠습니다|진행하겠습니다|준비하겠습니다|검토하겠습니다|공유하겠습니다|작성하겠습니다)")
DUE_PAT = re.compile(r"(\d{4}-\d{2}-\d{2}|\d{1,2}월\s*\d{1,2}일|오늘|내일|이번주|다음주|월요일|화요일|수요일|목요일|금요일|토요일|일요일)")
TRANSITION_PAT = re.compile(r"(다음|한편|반면|이제|정리하면|다시|또 하나|두 번째|세 번째|마지막으로)")


def _now_ts() -> str:
    return time.strftime("%H:%M:%S")


def _safe_text(raw: Any, fallback: str = "") -> str:
    s = str(raw or "").strip()
    return s if s else fallback


def _boolify(raw: Any, default: bool) -> bool:
    if raw is None:
        return default
    s = str(raw).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _sec_to_ts(raw: Any) -> str:
    try:
        sec = max(0, float(raw))
    except Exception:
        return _now_ts()
    total = int(sec)
    hh = (total // 3600) % 24
    mm = (total % 3600) // 60
    ss = total % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9가-힣]{2,}", _safe_text(text).lower()))


def _topic_far_enough(current_title: str, new_title: str) -> bool:
    cur = _tokens(current_title)
    nxt = _tokens(new_title)
    if not cur or not nxt:
        return _safe_text(current_title) != _safe_text(new_title)
    inter = len(cur & nxt)
    union = len(cur | nxt)
    sim = inter / union if union > 0 else 0.0
    return sim < 0.4


def _keyword_tokens(text: str) -> list[str]:
    out: list[str] = []
    for tok in re.findall(r"[A-Za-z0-9가-힣]{2,}", _safe_text(text).lower()):
        if tok.isdigit():
            continue
        if tok in STOPWORDS:
            continue
        if re.fullmatch(r"name\d+", tok):
            continue
        if tok.startswith("name") or tok.startswith("party"):
            continue
        out.append(tok)
    return out


def _text_similarity(a: str, b: str) -> float:
    ta = set(_keyword_tokens(a))
    tb = set(_keyword_tokens(b))
    if not ta or not tb:
        return 0.0
    union = len(ta | tb)
    return (len(ta & tb) / union) if union else 0.0


def _normalize_agenda_state(raw: Any) -> str:
    s = _safe_text(raw, "PROPOSED").upper()
    if s in {"ACTIVE", "CLOSING", "CLOSED", "PROPOSED"}:
        return s
    return "PROPOSED"


def _doc_freq(rows: list[dict[str, Any]]) -> Counter[str]:
    cnt: Counter[str] = Counter()
    for row in rows:
        seen = set(_keyword_tokens(_safe_text(row.get("text"))))
        for tok in seen:
            cnt[tok] += 1
    return cnt


def _top_keywords_from_rows(
    rows: list[dict[str, Any]],
    meeting_goal: str = "",
    limit: int = 6,
    global_doc_freq: Counter[str] | None = None,
    global_turn_count: int = 0,
) -> list[str]:
    banned = _tokens(meeting_goal)
    cnt: Counter[str] = Counter()
    for row in rows:
        text = _safe_text(row.get("text"))
        for tok in _keyword_tokens(text):
            if tok in banned:
                continue
            if global_doc_freq and global_turn_count > 0:
                if global_doc_freq.get(tok, 0) >= max(20, int(global_turn_count * 0.25)):
                    continue
            cnt[tok] += 1
    return [k for k, _ in cnt.most_common(limit)]


def _clean_agenda_title(raw_title: Any, meeting_goal: str = "", keywords: list[str] | None = None) -> str:
    title = _safe_text(raw_title)
    title = re.sub(r"^[0-9]+[.)]\s*", "", title).strip(" -:|")
    title = re.sub(r"\s+", " ", title)

    kws = [k for k in (keywords or []) if _safe_text(k)]
    goal = _safe_text(meeting_goal)
    goal_tokens = _tokens(goal)
    usable = [k for k in kws if k not in goal_tokens]

    if not title:
        if len(usable) >= 2:
            title = f"{usable[0]} · {usable[1]} 논의"
        elif len(usable) == 1:
            title = f"{usable[0]} 중심 논의"
        else:
            title = "세부 쟁점 논의"

    if goal and title == goal:
        if len(usable) >= 2:
            title = f"{usable[0]} · {usable[1]} 논의"
        elif len(usable) == 1:
            title = f"{usable[0]} 중심 논의"
        else:
            title = f"{goal} 세부 쟁점"

    return _safe_text(title[:80], "세부 쟁점 논의")


def _extract_json(raw: str) -> dict[str, Any]:
    txt = _safe_text(raw)
    if txt.startswith("```"):
        txt = txt.strip("`")
        if txt.lower().startswith("json"):
            txt = txt[4:].strip()
    try:
        data = json.loads(txt)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass
    l = txt.find("{")
    r = txt.rfind("}")
    if l >= 0 and r > l:
        try:
            data = json.loads(txt[l : r + 1])
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _speaker_profile_label(age: Any, occupation: Any, role: Any, fallback_id: str) -> str:
    parts = [_safe_text(age), _safe_text(occupation), _safe_text(role)]
    label = " ".join([p for p in parts if p]).strip()
    return label if label else _safe_text(fallback_id, "화자")


def _parse_meeting_json_payload(payload: dict[str, Any]) -> tuple[str | None, list[dict[str, str]]]:
    metadata = payload.get("metadata") or {}
    meeting_goal = _safe_text(metadata.get("topic"))

    speaker_map: dict[str, str] = {}
    for spk in payload.get("speaker") or []:
        if not isinstance(spk, dict):
            continue
        sid = _safe_text(spk.get("id"))
        if not sid:
            continue
        speaker_map[sid] = _speaker_profile_label(spk.get("age"), spk.get("occupation"), spk.get("role"), sid)

    rows = []
    for utt in payload.get("utterance") or []:
        if not isinstance(utt, dict):
            continue
        text = _safe_text(utt.get("original_form")) or _safe_text(utt.get("form"))
        if not text:
            continue
        sid = _safe_text(utt.get("speaker_id"))
        speaker = speaker_map.get(sid) or _safe_text(sid, "화자")
        timestamp = _sec_to_ts(utt.get("start"))
        rows.append(
            {
                "speaker": speaker,
                "text": text,
                "timestamp": timestamp,
            }
        )

    rows.sort(key=lambda x: x.get("timestamp", ""))
    return (meeting_goal if meeting_goal else None), rows


class ConfigInput(BaseModel):
    meeting_goal: str = ""
    window_size: int = Field(default=12, ge=4, le=80)


class UtteranceInput(BaseModel):
    speaker: str = "화자"
    text: str
    timestamp: str | None = None


class ImportDirInput(BaseModel):
    folder: str = "dataset/economy"
    recursive: bool = True
    reset_state: bool = True
    auto_tick: bool = True
    max_files: int = Field(default=500, ge=1, le=2000)


@dataclass
class RuntimeStore:
    lock: threading.Lock = field(default_factory=threading.Lock)
    meeting_goal: str = ""
    window_size: int = 12
    transcript: list[dict[str, str]] = field(default_factory=list)
    agenda_outcomes: list[dict[str, Any]] = field(default_factory=list)
    llm_enabled: bool = False
    last_analyzed_count: int = 0
    agenda_seq: int = 0
    stt_chunk_seq: int = 0
    used_local_fallback: bool = False
    last_analysis_warning: str = ""
    last_tick_mode: str = "windowed"

    def reset(self) -> None:
        self.meeting_goal = ""
        self.window_size = 12
        self.transcript = []
        self.agenda_outcomes = []
        self.last_analyzed_count = 0
        self.agenda_seq = 0
        self.stt_chunk_seq = 0
        self.used_local_fallback = False
        self.last_analysis_warning = ""
        self.last_tick_mode = "windowed"


RT = RuntimeStore()


def _agenda_stack_from_outcomes(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    stack: list[dict[str, str]] = []
    for row in rows:
        st = _safe_text(row.get("agenda_state"), "PROPOSED").upper()
        if st not in {"PROPOSED", "ACTIVE", "CLOSING", "CLOSED"}:
            st = "PROPOSED"
        stack.append({"title": _safe_text(row.get("agenda_title"), "아젠다 미정"), "status": st})
    return stack


def _active_agenda(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        if _safe_text(row.get("agenda_state")).upper() in {"ACTIVE", "CLOSING"}:
            return row
    return None


def _refresh_analysis(rt: RuntimeStore) -> dict[str, Any]:
    outcomes = []
    for row in rt.agenda_outcomes:
        if not isinstance(row, dict):
            continue
        summary_items = list(row.get("_summary_items") or [])
        summary = " • ".join(summary_items[-4:]) if summary_items else _safe_text(row.get("summary"))
        key_utterances = list(row.get("key_utterances") or [])
        outcomes.append(
            {
                "agenda_id": _safe_text(row.get("agenda_id")),
                "agenda_title": _safe_text(row.get("agenda_title"), "아젠다 미정"),
                "agenda_state": _safe_text(row.get("agenda_state"), "PROPOSED"),
                "flow_type": _safe_text(row.get("flow_type")),
                "key_utterances": key_utterances[-8:],
                "agenda_summary_items": summary_items[-8:],
                "summary": summary,
                "summary_references": list(row.get("summary_references") or []),
                "agenda_keywords": list(row.get("agenda_keywords") or []),
                "decision_results": list(row.get("decision_results") or []),
                "action_items": list(row.get("action_items") or []),
                "start_turn_id": int(row.get("start_turn_id") or row.get("_start_turn_id") or 0),
                "end_turn_id": int(row.get("end_turn_id") or row.get("_end_turn_id") or 0),
            }
        )

    active = _active_agenda(outcomes)
    candidates = [
        {"title": _safe_text(row.get("agenda_title")), "confidence": 0.7}
        for row in outcomes
        if _safe_text(row.get("agenda_state")).upper() == "PROPOSED"
    ]
    return {
        "agenda": {
            "active": {
                "title": _safe_text((active or {}).get("agenda_title"), ""),
                "confidence": 0.82 if active else 0.0,
            },
            "candidates": candidates[:6],
        },
        "agenda_outcomes": outcomes,
        "evidence_gate": {"claims": []},
    }


def _state_response(rt: RuntimeStore) -> dict[str, Any]:
    client = get_client()
    _ensure_minimum_agenda(rt)
    analysis = _refresh_analysis(rt)
    return {
        "meeting_goal": rt.meeting_goal,
        "initial_context": "",
        "window_size": rt.window_size,
        "transcript": list(rt.transcript),
        "agenda_stack": _agenda_stack_from_outcomes(analysis["agenda_outcomes"]),
        "llm_enabled": rt.llm_enabled,
        "llm_status": client.status(),
        "analysis_runtime": {
            "tick_mode": _safe_text(rt.last_tick_mode, "windowed"),
            "transcript_count": len(rt.transcript),
            "llm_window_turns": rt.window_size,
            "engine_window_turns": rt.window_size,
            "control_plane_source": "gemini",
            "control_plane_reason": rt.last_analysis_warning or ("full_document_once" if rt.last_tick_mode == "full_document" else "summary_every_4_turns"),
            "used_local_fallback": bool(rt.used_local_fallback),
        },
        "analysis": analysis,
    }


def _create_agenda(rt: RuntimeStore, title: str, state: str = "ACTIVE") -> dict[str, Any]:
    rt.agenda_seq += 1
    row = {
        "agenda_id": f"agenda-{rt.agenda_seq}",
        "agenda_title": _safe_text(title, f"안건 {rt.agenda_seq}"),
        "agenda_state": state,
        "flow_type": "",
        "key_utterances": [],
        "summary": "",
        "_summary_items": [],
        "summary_references": [],
        "agenda_keywords": [],
        "decision_results": [],
        "action_items": [],
        "start_turn_id": 0,
        "end_turn_id": 0,
    }
    rt.agenda_outcomes.append(row)
    return row


def _ensure_active_agenda(rt: RuntimeStore, title: str) -> dict[str, Any]:
    active = _active_agenda(rt.agenda_outcomes)
    if active is None:
        return _create_agenda(rt, title, "ACTIVE")
    return active


def _ensure_minimum_agenda(rt: RuntimeStore) -> None:
    if rt.agenda_outcomes or not rt.transcript:
        return
    title = _clean_agenda_title("", rt.meeting_goal, [])
    row = _create_agenda(rt, title, "ACTIVE")
    row["start_turn_id"] = 1
    row["end_turn_id"] = len(rt.transcript)
    recent = rt.transcript[max(0, len(rt.transcript) - 4) :]
    for t in recent:
        line = f"[{_safe_text(t.get('timestamp'), _now_ts())}] {_safe_text(t.get('text'))}"
        if line:
            row.setdefault("_summary_items", []).append(line)
            row.setdefault("key_utterances", []).append(line)


def _extract_refs(rt: RuntimeStore, evidence_turn_ids: list[int], recent_turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for idx in evidence_turn_ids:
        try:
            pos = int(idx) - 1
        except Exception:
            continue
        if pos < 0 or pos >= len(rt.transcript):
            continue
        t = rt.transcript[pos]
        refs.append(
            {
                "turn_id": pos + 1,
                "speaker": _safe_text(t.get("speaker"), "화자"),
                "timestamp": _safe_text(t.get("timestamp"), _now_ts()),
                "quote": _safe_text(t.get("text")),
                "why": "",
            }
        )
    if refs:
        return refs
    if recent_turns:
        t = recent_turns[-1]
        return [
            {
                "turn_id": int(t.get("turn_id") or 0),
                "speaker": _safe_text(t.get("speaker"), "화자"),
                "timestamp": _safe_text(t.get("timestamp"), _now_ts()),
                "quote": _safe_text(t.get("text")),
                "why": "",
            }
        ]
    return []


def _format_line_from_turn(turn: dict[str, Any], max_chars: int = 180) -> str:
    ts = _safe_text(turn.get("timestamp"), _now_ts())
    text = _safe_text(turn.get("text")).replace("\n", " ").strip()
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "…"
    return f"[{ts}] {text}"


def _ref_from_turn(turn: dict[str, Any], why: str = "요약 근거") -> dict[str, Any]:
    return {
        "turn_id": int(turn.get("turn_id") or 0),
        "speaker": _safe_text(turn.get("speaker"), "화자"),
        "timestamp": _safe_text(turn.get("timestamp"), _now_ts()),
        "quote": _safe_text(turn.get("text")),
        "why": _safe_text(why, "요약 근거"),
    }


def _pick_key_refs(turns: list[dict[str, Any]], keywords: list[str], max_items: int = 6) -> list[dict[str, Any]]:
    scored: list[tuple[float, int, dict[str, Any]]] = []
    kw = [k.lower() for k in keywords[:8]]
    for idx, t in enumerate(turns):
        text = _safe_text(t.get("text"))
        if len(text) < 8:
            continue
        low = text.lower()
        score = min(len(text), 120) / 120.0
        score += sum(2.0 for token in kw if token and token in low)
        if DECISION_PAT.search(text):
            score += 1.4
        if ACTION_PAT.search(text):
            score += 1.0
        scored.append((score, idx, _ref_from_turn(t)))
    if not scored:
        return []
    scored.sort(key=lambda x: (-x[0], x[1]))
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, _, ref in scored:
        key = f"{ref.get('timestamp')}|{ref.get('quote')}"
        if key in seen:
            continue
        out.append(ref)
        seen.add(key)
        if len(out) >= max_items:
            break
    return out


def _segment_turns(turns: list[dict[str, Any]]) -> list[tuple[int, int]]:
    n = len(turns)
    if n == 0:
        return []
    if n <= 40:
        return [(0, n)]

    min_seg = 24
    max_seg = 140
    target = max(2, n // 95)
    target_gap = max(min_seg, n // target)
    win = 16

    bounds = [0]
    last = 0
    i = min_seg
    while i < n - min_seg:
        dist = i - last
        prev_txt = " ".join(_safe_text(t.get("text")) for t in turns[max(last, i - win) : i])
        next_txt = " ".join(_safe_text(t.get("text")) for t in turns[i : min(n, i + win)])
        sim = _text_similarity(prev_txt, next_txt)
        cue = bool(TRANSITION_PAT.search(_safe_text(turns[i].get("text")))) or bool(
            TRANSITION_PAT.search(_safe_text(turns[i - 1].get("text")))
        )
        reached_target = dist >= target_gap
        too_long = dist >= max_seg

        should_split = False
        if too_long:
            should_split = True
        elif sim < 0.22 and dist >= min_seg:
            should_split = True
        elif cue and sim < 0.42 and reached_target:
            should_split = True
        elif reached_target and sim < 0.30:
            should_split = True

        if should_split:
            bounds.append(i)
            last = i
            i += max(4, min_seg // 2)
            continue
        i += 1

    bounds.append(n)
    segments: list[tuple[int, int]] = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        if e <= s:
            continue
        if segments and (e - s) < min_seg:
            ps, _ = segments[-1]
            segments[-1] = (ps, e)
        else:
            segments.append((s, e))

    if len(segments) <= 1 and n >= 120:
        pieces = max(2, min(4, n // 180 + 1))
        step = max(1, n // pieces)
        segments = []
        for p in range(pieces):
            s = p * step
            e = n if p == pieces - 1 else min(n, (p + 1) * step)
            if e > s:
                segments.append((s, e))

    dynamic_cap = max(3, target * 2)
    while len(segments) > dynamic_cap:
        lengths = [(idx, seg[1] - seg[0]) for idx, seg in enumerate(segments)]
        idx = min(lengths, key=lambda x: x[1])[0]
        if idx == 0:
            merged = (segments[0][0], segments[1][1])
            segments = [merged] + segments[2:]
        elif idx == len(segments) - 1:
            merged = (segments[-2][0], segments[-1][1])
            segments = segments[:-2] + [merged]
        else:
            left_len = segments[idx - 1][1] - segments[idx - 1][0]
            right_len = segments[idx + 1][1] - segments[idx + 1][0]
            if left_len <= right_len:
                merged = (segments[idx - 1][0], segments[idx][1])
                segments = segments[: idx - 1] + [merged] + segments[idx + 1 :]
            else:
                merged = (segments[idx][0], segments[idx + 1][1])
                segments = segments[:idx] + [merged] + segments[idx + 2 :]

    return segments


def _pick_key_utterances(turns: list[dict[str, Any]], keywords: list[str], max_items: int = 6) -> list[str]:
    scored: list[tuple[float, int, str]] = []
    kw = [k.lower() for k in keywords[:8]]
    for idx, t in enumerate(turns):
        text = _safe_text(t.get("text"))
        if len(text) < 8:
            continue
        low = text.lower()
        score = min(len(text), 120) / 120.0
        score += sum(2.0 for token in kw if token and token in low)
        if DECISION_PAT.search(text):
            score += 1.4
        if ACTION_PAT.search(text):
            score += 1.0
        scored.append((score, idx, _format_line_from_turn(t)))
    if not scored:
        return []
    scored.sort(key=lambda x: (-x[0], x[1]))
    picked: list[str] = []
    seen: set[str] = set()
    for _, _, line in scored:
        if line in seen:
            continue
        picked.append(line)
        seen.add(line)
        if len(picked) >= max_items:
            break
    return picked


def _extract_decisions_from_turns(turns: list[dict[str, Any]], max_items: int = 6) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for t in turns:
        text = _safe_text(t.get("text"))
        if not text or not DECISION_PAT.search(text):
            continue
        key = text[:120]
        if key in seen:
            continue
        seen.add(key)
        out.append({"opinions": [_format_line_from_turn(t)], "conclusion": key})
        if len(out) >= max_items:
            break
    return out


def _extract_actions_from_turns(turns: list[dict[str, Any]], max_items: int = 10) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for t in turns:
        text = _safe_text(t.get("text"))
        if not text:
            continue
        if not ACTION_PAT.search(text) and not DUE_PAT.search(text):
            continue
        due = ""
        m = DUE_PAT.search(text)
        if m:
            due = _safe_text(m.group(1))
        owner = _safe_text(t.get("speaker"), "-")
        task = text[:160]
        dedup = f"{task}|{owner}|{due}"
        if dedup in seen:
            continue
        seen.add(dedup)
        out.append(
            {
                "item": task,
                "owner": owner,
                "due": due,
                "reasons": [
                    {
                        "speaker": owner,
                        "timestamp": _safe_text(t.get("timestamp"), _now_ts()),
                        "quote": text,
                        "why": "발화 기반 추출",
                    }
                ],
            }
        )
        if len(out) >= max_items:
            break
    return out


def _dedup_preserve(items: list[str], limit: int = 10) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        txt = _safe_text(item)
        if not txt or txt in seen:
            continue
        out.append(txt)
        seen.add(txt)
        if len(out) >= limit:
            break
    return out


def _build_local_outcomes(rt: RuntimeStore, turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments = _segment_turns(turns)
    if not segments and turns:
        segments = [(0, len(turns))]

    global_df = _doc_freq(turns)
    global_turn_count = len(turns)
    outcomes: list[dict[str, Any]] = []
    used_titles: set[str] = set()

    for seg_idx, (s, e) in enumerate(segments):
        seg_turns = turns[s:e]
        if not seg_turns:
            continue
        keywords = _top_keywords_from_rows(
            seg_turns,
            rt.meeting_goal,
            limit=6,
            global_doc_freq=global_df,
            global_turn_count=global_turn_count,
        )
        title = _clean_agenda_title("", rt.meeting_goal, keywords)
        if title in used_titles:
            title = f"{title} #{seg_idx + 1}"
        used_titles.add(title)

        key_refs = _pick_key_refs(seg_turns, keywords, max_items=8)
        key_utterances = [f"[{_safe_text(r.get('timestamp'))}] {_safe_text(r.get('quote'))}" for r in key_refs]
        summary_refs = key_refs[:4] if key_refs else [_ref_from_turn(seg_turns[-1])]
        summary_items = [f"[{_safe_text(r.get('timestamp'))}] {_safe_text(r.get('quote'))}" for r in summary_refs]
        summary = " • ".join(item.split("] ", 1)[-1] for item in summary_items[:3])
        decisions = _extract_decisions_from_turns(seg_turns, max_items=4)
        actions = _extract_actions_from_turns(seg_turns, max_items=6)

        flow_type = "discussion"
        if decisions:
            flow_type = "decision"
        elif actions:
            flow_type = "action-planning"

        outcomes.append(
            {
                "agenda_title": title,
                "agenda_state": "ACTIVE" if seg_idx == len(segments) - 1 else "CLOSED",
                "flow_type": flow_type,
                "key_utterances": _dedup_preserve(key_utterances, limit=8),
                "_summary_items": _dedup_preserve(summary_items, limit=6),
                "summary_references": summary_refs,
                "summary": _safe_text(summary),
                "agenda_keywords": _dedup_preserve(keywords, limit=6),
                "decision_results": decisions,
                "action_items": actions,
                "_start_turn_id": int(seg_turns[0].get("turn_id", 1) or 1),
                "_end_turn_id": int(seg_turns[-1].get("turn_id", 1) or 1),
            }
        )

    return outcomes


def _apply_outcomes(rt: RuntimeStore, outcomes: list[dict[str, Any]]) -> None:
    cleaned = [dict(row) for row in outcomes if isinstance(row, dict)]
    if not cleaned:
        return
    cleaned.sort(key=lambda x: int(x.get("_start_turn_id") or 10**9))

    prev_end = 0
    for idx, row in enumerate(cleaned):
        start_id = int(row.get("_start_turn_id") or row.get("start_turn_id") or 0)
        end_id = int(row.get("_end_turn_id") or row.get("end_turn_id") or 0)
        if start_id <= 0:
            start_id = prev_end + 1 if prev_end > 0 else (idx + 1)
        if end_id < start_id:
            end_id = start_id
        row["_start_turn_id"] = start_id
        row["_end_turn_id"] = end_id
        prev_end = max(prev_end, end_id)

    for idx, row in enumerate(cleaned):
        if idx >= len(cleaned) - 1:
            continue
        next_start = int(cleaned[idx + 1].get("_start_turn_id") or 0)
        end_id = int(row.get("_end_turn_id") or 0)
        start_id = int(row.get("_start_turn_id") or 0)
        if next_start > 0 and (end_id <= start_id or end_id >= next_start):
            row["_end_turn_id"] = max(start_id, next_start - 1)

    active_idx = -1
    for idx, row in enumerate(cleaned):
        if _normalize_agenda_state(row.get("agenda_state")) in {"ACTIVE", "CLOSING"}:
            active_idx = idx
            break
    if active_idx < 0 and cleaned:
        active_idx = len(cleaned) - 1

    for idx, row in enumerate(cleaned):
        if idx == active_idx:
            row["agenda_state"] = "ACTIVE"
        elif _normalize_agenda_state(row.get("agenda_state")) == "ACTIVE":
            row["agenda_state"] = "CLOSED"
        else:
            row["agenda_state"] = _normalize_agenda_state(row.get("agenda_state"))

    rt.agenda_outcomes = []
    rt.agenda_seq = 0
    for row in cleaned:
        created = _create_agenda(rt, _safe_text(row.get("agenda_title"), "세부 쟁점 논의"), _normalize_agenda_state(row.get("agenda_state")))
        created["flow_type"] = _safe_text(row.get("flow_type"))
        created["key_utterances"] = _dedup_preserve(list(row.get("key_utterances") or []), limit=8)
        created["_summary_items"] = _dedup_preserve(list(row.get("_summary_items") or []), limit=6)
        created["summary_references"] = list(row.get("summary_references") or [])
        created["summary"] = _safe_text(row.get("summary"))
        created["agenda_keywords"] = _dedup_preserve(list(row.get("agenda_keywords") or []), limit=6)
        created["decision_results"] = list(row.get("decision_results") or [])
        created["action_items"] = list(row.get("action_items") or [])
        created["start_turn_id"] = int(row.get("_start_turn_id") or 0)
        created["end_turn_id"] = int(row.get("_end_turn_id") or 0)


def _to_ids(raw_ids: Any) -> list[int]:
    out: list[int] = []
    for x in raw_ids or []:
        try:
            out.append(int(str(x)))
        except Exception:
            continue
    return out


def _build_prompt(rt: RuntimeStore, turns: list[dict[str, Any]], current_agenda_title: str, mode: str = "windowed") -> str:
    meeting_goal = _safe_text(rt.meeting_goal, "미정")
    lines = []
    for turn in turns:
        lines.append(
            f"- turn_id={turn['turn_id']} | {turn['timestamp']} | {turn['speaker']} | {turn['text']}"
        )
    transcript_block = "\n".join(lines)

    return f"""
너는 회의록 구조화 분석기다. 출력은 반드시 JSON 하나만 반환한다.

[입력]
- 전체 회의 목표: {meeting_goal}
- 현재 진행 안건: {current_agenda_title or "없음"}
- 분석 모드: {mode}
- 발화 목록(시간순):
{transcript_block}

[중요 규칙]
1) 안건은 "흐름 전환 시점" 기준으로 순서대로 나눈다. 즉, 주제가 전환될 때마다 새 안건을 만든다.
2) 안건 제목은 "회의 목표 문자열 그대로"를 쓰지 말고 세부 쟁점으로 작성한다.
3) 각 안건에 키워드 3~6개를 반드시 넣는다(명사/핵심 용어 중심).
4) 의사결정은 "확정된 내용"만 decision_results에 넣는다.
5) 액션아이템은 누가/무엇/기한(없으면 빈문자열)과 근거 turn_id를 넣는다.
6) evidence_turn_ids, key_utterance_turn_ids는 반드시 입력의 turn_id만 사용한다.
7) 현재 진행 안건이 이미 있으면, 정말로 주제가 크게 바뀌었을 때만 새 ACTIVE 안건으로 둔다.
8) 각 안건은 start_turn_id/end_turn_id를 반드시 포함하고, 안건 간 구간은 시간순/비중첩으로 작성한다.
9) agenda_summary_items는 해당 안건 구간(end_turn_id 이전)에서만 요약하고, 각 요약문마다 evidence_turn_ids를 넣는다.
10) 분석 모드가 full_document이면, 발화 전체를 끝까지 보고 안건을 한 번에 완성한다. 중간 단계 안건 생성은 금지한다.

[출력 JSON 스키마]
{{
  "active_agenda_title": "string",
  "agendas": [
    {{
      "agenda_title": "string",
      "agenda_state": "PROPOSED|ACTIVE|CLOSING|CLOSED",
      "start_turn_id": 1,
      "end_turn_id": 20,
      "flow_type": "discussion|decision|action-planning",
      "agenda_keywords": ["string", "string"],
      "key_utterance_turn_ids": [1,2,3],
      "agenda_summary_items": [
        {{"summary": "string", "evidence_turn_ids": [1,2]}}
      ],
      "decision_results": [
        {{
          "conclusion": "string",
          "opinions": ["string"],
          "evidence_turn_ids": [1,2]
        }}
      ],
      "action_items": [
        {{
          "item": "string",
          "owner": "string",
          "due": "string",
          "reason": "string",
          "evidence_turn_ids": [1,2]
        }}
      ]
    }}
  ]
}}
""".strip()


def _run_local_fallback(rt: RuntimeStore, force: bool = False, reason: str = "", mode: str = "windowed") -> bool:
    if not rt.transcript:
        return False
    if mode != "full_document" and (not force) and (len(rt.transcript) - rt.last_analyzed_count) < SUMMARY_INTERVAL:
        return False

    turns: list[dict[str, Any]] = []
    for i, row in enumerate(rt.transcript, start=1):
        turns.append(
            {
                "turn_id": i,
                "timestamp": _safe_text(row.get("timestamp"), _now_ts()),
                "speaker": _safe_text(row.get("speaker"), "화자"),
                "text": _safe_text(row.get("text")),
            }
        )
    outcomes = _build_local_outcomes(rt, turns)
    if outcomes:
        _apply_outcomes(rt, outcomes)

    rt.last_analyzed_count = len(rt.transcript)
    rt.used_local_fallback = True
    rt.last_analysis_warning = reason or "LLM 비활성/실패로 로컬 폴백 분석 사용"
    rt.last_tick_mode = "full_document" if mode == "full_document" else "windowed"
    return True


def _run_analysis(rt: RuntimeStore, force: bool = False, mode: str = "windowed") -> bool:
    if not rt.transcript:
        rt.used_local_fallback = True
        rt.last_analysis_warning = "전사 데이터가 없어 분석할 수 없습니다."
        rt.last_tick_mode = "full_document" if mode == "full_document" else "windowed"
        return False
    if mode != "full_document" and (not force) and (len(rt.transcript) - rt.last_analyzed_count) < SUMMARY_INTERVAL:
        return False
    if not rt.llm_enabled:
        return _run_local_fallback(rt, force=force, reason="LLM 미연결", mode=mode)

    client = get_client()
    if not client.connected:
        return _run_local_fallback(rt, force=force, reason="LLM 연결 끊김", mode=mode)

    full_document = mode == "full_document"
    base_idx = 0 if (force or full_document) else max(0, len(rt.transcript) - max(220, rt.window_size * 10))
    turns: list[dict[str, Any]] = []
    for i, row in enumerate(rt.transcript[base_idx:], start=base_idx + 1):
        turns.append(
            {
                "turn_id": i,
                "timestamp": _safe_text(row.get("timestamp")),
                "speaker": _safe_text(row.get("speaker")),
                "text": _safe_text(row.get("text")),
            }
        )

    active = _active_agenda(rt.agenda_outcomes)
    current_title = _safe_text((active or {}).get("agenda_title"))
    prompt = _build_prompt(rt, turns, current_title, mode=mode)

    try:
        parsed = client.generate_json(prompt, temperature=0.1, max_tokens=6000)
    except Exception as exc:
        return _run_local_fallback(rt, force=force, reason=f"LLM 오류: {exc}", mode=mode)

    raw_agendas = parsed.get("agendas") or []
    if not isinstance(raw_agendas, list) or not raw_agendas:
        return _run_local_fallback(rt, force=force, reason="LLM 응답에서 agendas가 비어 로컬 폴백 사용", mode=mode)

    active_title = _safe_text(parsed.get("active_agenda_title"))
    normalized_active = _clean_agenda_title(active_title, rt.meeting_goal, []) if active_title else ""
    outcomes: list[dict[str, Any]] = []

    for idx, agenda in enumerate(raw_agendas):
        if not isinstance(agenda, dict):
            continue

        keywords = _dedup_preserve([_safe_text(x) for x in (agenda.get("agenda_keywords") or []) if _safe_text(x)], limit=8)
        title = _clean_agenda_title(agenda.get("agenda_title"), rt.meeting_goal, keywords)

        state = _normalize_agenda_state(agenda.get("agenda_state"))
        if normalized_active and title == normalized_active:
            state = "ACTIVE"

        key_refs = _extract_refs(rt, _to_ids(agenda.get("key_utterance_turn_ids")), turns)
        key_utterances = _dedup_preserve([f"[{r['timestamp']}] {r['quote']}" for r in key_refs], limit=8)

        summary_items: list[str] = []
        summary_references: list[dict[str, Any]] = []
        for it in agenda.get("agenda_summary_items") or []:
            if not isinstance(it, dict):
                continue
            txt = _safe_text(it.get("summary"))
            if not txt:
                continue
            refs = _extract_refs(rt, _to_ids(it.get("evidence_turn_ids")), turns)
            if refs:
                summary_items.append(f"[{refs[0]['timestamp']}] {txt}")
                for ref in refs[:3]:
                    summary_references.append(
                        {
                            "turn_id": int(ref.get("turn_id") or 0),
                            "speaker": ref["speaker"],
                            "timestamp": ref["timestamp"],
                            "quote": ref["quote"],
                            "why": txt,
                        }
                    )
            else:
                summary_items.append(txt)
        if not summary_items:
            summary_items = key_utterances[:4]
        if not summary_references:
            for ref in key_refs[:4]:
                summary_references.append(
                    {
                        "turn_id": 0,
                        "speaker": ref["speaker"],
                        "timestamp": ref["timestamp"],
                        "quote": ref["quote"],
                        "why": "핵심 발언",
                    }
                )

        decisions: list[dict[str, Any]] = []
        for it in agenda.get("decision_results") or []:
            if not isinstance(it, dict):
                continue
            conclusion = _safe_text(it.get("conclusion"))
            if not conclusion:
                continue
            opinions = [_safe_text(x) for x in (it.get("opinions") or []) if _safe_text(x)]
            refs = _extract_refs(rt, _to_ids(it.get("evidence_turn_ids")), turns)
            for r in refs[:3]:
                opinions.append(f"[{r['timestamp']}] {r['quote']}")
            decisions.append({"opinions": _dedup_preserve(opinions, 5), "conclusion": conclusion})

        actions: list[dict[str, Any]] = []
        for it in agenda.get("action_items") or []:
            if not isinstance(it, dict):
                continue
            item = _safe_text(it.get("item"))
            if not item:
                continue
            owner = _safe_text(it.get("owner"), "-")
            due = _safe_text(it.get("due"))
            reason = _safe_text(it.get("reason"))
            refs = _extract_refs(rt, _to_ids(it.get("evidence_turn_ids")), turns)
            reasons = []
            for r in refs:
                reasons.append(
                    {
                        "speaker": r["speaker"],
                        "timestamp": r["timestamp"],
                        "quote": r["quote"],
                        "why": reason,
                    }
                )
            actions.append({"item": item, "owner": owner, "due": due, "reasons": reasons})

        if not keywords:
            synthetic_rows = [{"text": x.split("] ", 1)[-1]} for x in summary_items[:4]]
            keywords = _top_keywords_from_rows(synthetic_rows, rt.meeting_goal, limit=6)
        if not key_utterances and turns:
            pick_idx = min(len(turns) - 1, idx * max(1, len(turns) // max(1, len(raw_agendas))))
            key_utterances = [_format_line_from_turn(turns[pick_idx])]

        summary = " • ".join(x.split("] ", 1)[-1] for x in summary_items[:3])
        all_ids = _to_ids(agenda.get("key_utterance_turn_ids"))
        for s_item in agenda.get("agenda_summary_items") or []:
            if isinstance(s_item, dict):
                all_ids.extend(_to_ids(s_item.get("evidence_turn_ids")))
        start_turn_id = int(agenda.get("start_turn_id") or 0)
        end_turn_id = int(agenda.get("end_turn_id") or 0)
        if start_turn_id <= 0:
            start_turn_id = min(all_ids) if all_ids else (idx + 1) * 1000
        if end_turn_id < start_turn_id:
            end_turn_id = max(all_ids) if all_ids else start_turn_id

        outcomes.append(
            {
                "agenda_title": title,
                "agenda_state": state,
                "flow_type": _safe_text(agenda.get("flow_type"), "discussion"),
                "key_utterances": _dedup_preserve(key_utterances, limit=8),
                "_summary_items": _dedup_preserve(summary_items, limit=6),
                "summary_references": summary_references[:8],
                "summary": _safe_text(summary),
                "agenda_keywords": _dedup_preserve(keywords, limit=6),
                "decision_results": decisions,
                "action_items": actions,
                "_start_turn_id": start_turn_id,
                "_end_turn_id": end_turn_id,
            }
        )

    if not outcomes:
        return _run_local_fallback(rt, force=force, reason="LLM agendas 파싱 실패", mode=mode)

    _apply_outcomes(rt, outcomes)

    rt.last_analyzed_count = len(rt.transcript)
    rt.used_local_fallback = False
    rt.last_analysis_warning = ""
    rt.last_tick_mode = "full_document" if mode == "full_document" else "windowed"
    return True


def _append_turn(rt: RuntimeStore, speaker: str, text: str, timestamp: str | None = None) -> None:
    body = _safe_text(text)
    if not body:
        return
    rt.transcript.append(
        {
            "speaker": _safe_text(speaker, "화자"),
            "text": body,
            "timestamp": _safe_text(timestamp, _now_ts()),
        }
    )


def _append_many_turns(rt: RuntimeStore, rows: list[dict[str, str]]) -> int:
    before = len(rt.transcript)
    for row in rows:
        _append_turn(rt, row.get("speaker", "화자"), row.get("text", ""), row.get("timestamp"))
    return len(rt.transcript) - before


def _load_whisper_model():
    try:
        import whisper
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("whisper 패키지가 없습니다. `pip install openai-whisper` 후 다시 실행하세요.") from exc
    return whisper.load_model(WHISPER_MODEL_NAME)


_WHISPER_MODEL = None
_WHISPER_LOCK = threading.Lock()


def _get_whisper_model():
    global _WHISPER_MODEL
    with _WHISPER_LOCK:
        if _WHISPER_MODEL is None:
            _WHISPER_MODEL = _load_whisper_model()
        return _WHISPER_MODEL


def _transcribe_with_whisper(data: bytes, suffix: str) -> str:
    model = _get_whisper_model()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        kwargs = {"language": "ko", "task": "transcribe", "verbose": False}
        try:
            import torch

            kwargs["fp16"] = bool(torch.cuda.is_available())
        except Exception:
            kwargs["fp16"] = False
        result = model.transcribe(tmp_path, **kwargs)
        return _safe_text((result or {}).get("text"))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


app = FastAPI(title="Meeting STT + Agenda MVP")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def get_health():
    return {"ok": True, "whisper_model": WHISPER_MODEL_NAME}


@app.get("/api/state")
def get_state():
    with RT.lock:
        return _state_response(RT)


@app.get("/api/llm/status")
def get_llm_status():
    return get_client().status()


@app.post("/api/llm/connect")
def post_llm_connect():
    with RT.lock:
        client = get_client()
        result = client.connect()
        RT.llm_enabled = bool(result.get("ok"))
        if RT.llm_enabled:
            try:
                _run_analysis(RT, force=True, mode="full_document")
            except Exception:
                pass
        return {
            "enabled": RT.llm_enabled,
            "result": result,
            "llm_status": client.status(),
            "state": _state_response(RT),
        }


@app.post("/api/llm/disconnect")
def post_llm_disconnect():
    with RT.lock:
        client = get_client()
        result = client.disconnect()
        RT.llm_enabled = False
        return {
            "enabled": False,
            "result": result,
            "llm_status": client.status(),
            "state": _state_response(RT),
        }


@app.post("/api/llm/ping")
def post_llm_ping():
    client = get_client()
    result = client.ping()
    return {"result": result, "llm_status": client.status()}


@app.post("/api/config")
def post_config(payload: ConfigInput):
    with RT.lock:
        RT.meeting_goal = _safe_text(payload.meeting_goal)
        RT.window_size = int(payload.window_size)
        return _state_response(RT)


@app.post("/api/transcript/manual")
def post_transcript_manual(payload: UtteranceInput):
    with RT.lock:
        _append_turn(RT, payload.speaker, payload.text, payload.timestamp)
        try:
            _run_analysis(RT, force=False, mode="windowed")
        except Exception:
            pass
        return _state_response(RT)


@app.post("/api/transcript/import-json-dir")
def post_import_json_dir(payload: ImportDirInput):
    with RT.lock:
        folder = Path(payload.folder)
        target = folder if folder.is_absolute() else (ROOT / folder)
        files = []
        if target.exists() and target.is_dir():
            pattern = "**/*.json" if payload.recursive else "*.json"
            files = list(target.glob(pattern))[: payload.max_files]

        if payload.reset_state:
            RT.reset()

        files_scanned = 0
        files_parsed = 0
        files_skipped = 0
        rows_loaded = 0
        file_stats = []
        applied_goal = None

        for path in files:
            files_scanned += 1
            try:
                raw = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                raw = path.read_text(encoding="utf-8-sig")
            except Exception:
                files_skipped += 1
                continue
            data = _extract_json(raw)
            if not data:
                files_skipped += 1
                continue
            goal, rows = _parse_meeting_json_payload(data)
            if goal and not applied_goal:
                applied_goal = goal
            added = _append_many_turns(RT, rows)
            rows_loaded += added
            files_parsed += 1
            file_stats.append({"file": str(path), "rows": added})

        if applied_goal:
            RT.meeting_goal = applied_goal

        ticked = False
        if payload.auto_tick and RT.transcript:
            try:
                ticked = _run_analysis(RT, force=True, mode="full_document")
            except Exception:
                ticked = False

        return {
            "state": _state_response(RT),
            "import_debug": {
                "folder": str(target),
                "files_scanned": files_scanned,
                "files_parsed": files_parsed,
                "files_skipped": files_skipped,
                "rows_loaded": rows_loaded,
                "meeting_goal": RT.meeting_goal or "",
                "added": rows_loaded,
                "reset_state": bool(payload.reset_state),
                "auto_tick": bool(payload.auto_tick),
                "ticked": bool(ticked),
                "analysis_mode": "none" if not RT.llm_enabled else "full_document_once",
                "meeting_goal_applied": bool(applied_goal),
                "warning": "" if files_parsed > 0 else "파싱된 JSON 파일이 없습니다.",
                "file_stats": file_stats,
            },
        }


@app.post("/api/transcript/import-json-files")
async def post_import_json_files(
    files: list[UploadFile] = File(default=[]),
    reset_state: str = Form(default="true"),
    auto_tick: str = Form(default="true"),
):
    with RT.lock:
        do_reset = _boolify(reset_state, True)
        do_tick = _boolify(auto_tick, True)
        if do_reset:
            RT.reset()

        files_scanned = 0
        files_parsed = 0
        files_skipped = 0
        rows_loaded = 0
        file_stats = []
        applied_goal = None

        for upload in files:
            files_scanned += 1
            try:
                blob = await upload.read()
                raw = blob.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    raw = blob.decode("utf-8-sig")
                except Exception:
                    files_skipped += 1
                    continue
            except Exception:
                files_skipped += 1
                continue

            data = _extract_json(raw)
            if not data:
                files_skipped += 1
                continue
            goal, rows = _parse_meeting_json_payload(data)
            if goal and not applied_goal:
                applied_goal = goal
            added = _append_many_turns(RT, rows)
            rows_loaded += added
            files_parsed += 1
            file_stats.append({"file": upload.filename or "upload.json", "rows": added})

        if applied_goal:
            RT.meeting_goal = applied_goal

        ticked = False
        if do_tick and RT.transcript:
            try:
                ticked = _run_analysis(RT, force=True, mode="full_document")
            except Exception:
                ticked = False

        return {
            "state": _state_response(RT),
            "import_debug": {
                "folder": "<uploaded>",
                "files_scanned": files_scanned,
                "files_parsed": files_parsed,
                "files_skipped": files_skipped,
                "rows_loaded": rows_loaded,
                "meeting_goal": RT.meeting_goal or "",
                "added": rows_loaded,
                "reset_state": do_reset,
                "auto_tick": do_tick,
                "ticked": bool(ticked),
                "analysis_mode": "none" if not RT.llm_enabled else "full_document_once",
                "meeting_goal_applied": bool(applied_goal),
                "warning": "" if files_parsed > 0 else "파싱된 JSON 파일이 없습니다.",
                "file_stats": file_stats,
            },
        }


@app.post("/api/analysis/tick")
def post_analysis_tick():
    with RT.lock:
        _run_analysis(RT, force=True, mode="full_document")
        return _state_response(RT)


@app.post("/api/reset")
def post_reset():
    with RT.lock:
        llm_enabled = RT.llm_enabled
        RT.reset()
        RT.llm_enabled = llm_enabled
        return _state_response(RT)


@app.post("/api/stt/chunk")
async def post_stt_chunk(
    audio: UploadFile = File(...),
    speaker: str = Form(default="시스템오디오"),
    source: str = Form(default="system_audio"),
):
    t0 = time.perf_counter()
    with RT.lock:
        RT.stt_chunk_seq += 1
        chunk_id = RT.stt_chunk_seq

    try:
        blob = await audio.read()
    except Exception as exc:
        blob = b""
        read_err = str(exc)
    else:
        read_err = ""

    steps = [{"step": "read_chunk", "t_ms": int((time.perf_counter() - t0) * 1000)}]
    status = "ok"
    text = ""
    err_msg = ""

    if read_err:
        status = "error"
        err_msg = read_err
    elif not blob:
        status = "empty"
    else:
        suffix = Path(audio.filename or "chunk.webm").suffix or ".webm"
        try:
            text = _transcribe_with_whisper(blob, suffix=suffix)
        except Exception as exc:
            status = "error"
            err_msg = str(exc)
            text = ""
        if status == "ok" and not _safe_text(text):
            status = "empty"

    with RT.lock:
        if status == "ok" and _safe_text(text):
            _append_turn(RT, speaker, text, _now_ts())
            try:
                _run_analysis(RT, force=False, mode="windowed")
            except Exception:
                pass
        state = _state_response(RT)

    steps.append({"step": "done", "t_ms": int((time.perf_counter() - t0) * 1000)})
    duration_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "state": state,
        "stt_debug": {
            "chunk_id": chunk_id,
            "status": status,
            "source": source,
            "speaker": speaker,
            "filename": audio.filename or "chunk.webm",
            "bytes": len(blob),
            "steps": steps,
            "duration_ms": duration_ms,
            "transcript_chars": len(_safe_text(text)),
            "transcript_preview": _safe_text(text)[:240],
            "error": err_msg,
        },
    }
