from __future__ import annotations

import os
import re
import tempfile
import threading
import time
import json
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Optional, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from llm_client import get_client
from schemas import TranscriptUtterance

WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "large")
WHISPER_ENABLE_FALLBACK = os.environ.get("WHISPER_ENABLE_FALLBACK", "0") == "1"
WHISPER_SILENCE_NSP = float(os.environ.get("WHISPER_SILENCE_NSP", "0.65"))
WHISPER_SILENCE_LOGPROB = float(os.environ.get("WHISPER_SILENCE_LOGPROB", "-0.35"))
USE_LLM_CONTROL_PLANE = os.environ.get("USE_LLM_CONTROL_PLANE", "1") == "1"
LLM_CONTEXT_MAX_TURNS = int(os.environ.get("LLM_CONTEXT_MAX_TURNS", "320"))
FULL_CONTEXT_ENGINE_MAX_TURNS = int(os.environ.get("FULL_CONTEXT_ENGINE_MAX_TURNS", "3000"))
LLM_FULL_CONTEXT_MAX_TURNS = int(os.environ.get("LLM_FULL_CONTEXT_MAX_TURNS", "900"))
LLM_TEXT_CLIP_CHARS = int(os.environ.get("LLM_TEXT_CLIP_CHARS", "160"))
HALLUCINATION_PHRASES = {
    "감사합니다",
    "고맙습니다",
}
FLOW_TYPE_ORDER = ["문제정의", "대안비교", "의견충돌", "근거검토", "결론수렴", "실행정의"]
FLOW_TYPE_KEYWORDS = {
    "문제정의": ["문제", "이슈", "현황", "배경", "원인", "정의", "목표", "과제", "왜"],
    "대안비교": ["대안", "옵션", "비교", "장단점", "a안", "b안", "선택", "비용", "효과", "트레이드오프"],
    "의견충돌": ["반대", "이견", "충돌", "논쟁", "우려", "동의하지", "의견이 다르", "갈린다"],
    "근거검토": ["근거", "출처", "데이터", "지표", "통계", "검증", "사실", "레퍼런스", "증거"],
    "결론수렴": ["결론", "정리", "합의", "결정", "확정", "채택", "수렴", "클로징", "마무리"],
    "실행정의": ["액션", "실행", "담당", "기한", "언제", "누가", "까지", "하자", "진행", "todo"],
}


class ConfigInput(BaseModel):
    meeting_goal: str
    initial_context: str = ""
    window_size: int = Field(ge=4, le=80)


class UtteranceInput(BaseModel):
    speaker: str = "화자"
    text: str
    timestamp: Optional[str] = None


class DatasetImportInput(BaseModel):
    folder: str = "dataset/economy"
    recursive: bool = True
    reset_state: bool = True
    auto_tick: bool = True
    max_files: int = Field(ge=1, le=2000, default=500)


def _now_ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _compact_transcript_for_llm(turns: list[dict], max_turns: int, clip_chars: int) -> list[dict]:
    if not turns:
        return []

    def compact_one(t: dict) -> dict:
        txt = str(t.get("text") or "").strip()
        if len(txt) > clip_chars:
            txt = txt[: clip_chars - 1] + "…"
        return {
            "speaker": str(t.get("speaker") or "화자").strip() or "화자",
            "text": txt,
            "timestamp": str(t.get("timestamp") or "").strip(),
        }

    if len(turns) <= max_turns:
        return [compact_one(t) for t in turns]

    # Keep global flow by preserving head/tail and uniformly sampled middle turns.
    head_n = max(50, min(160, max_turns // 4))
    tail_n = max(90, min(260, max_turns // 3))
    if head_n + tail_n >= max_turns:
        head_n = max_turns // 2
        tail_n = max_turns - head_n

    head = turns[:head_n]
    tail = turns[-tail_n:]
    middle = turns[head_n : len(turns) - tail_n]
    middle_budget = max_turns - head_n - tail_n

    sampled_middle: list[dict] = []
    if middle and middle_budget > 0:
        stride = max(1, len(middle) // middle_budget)
        sampled_middle = middle[::stride][:middle_budget]

    merged = head + sampled_middle + tail
    if len(merged) > max_turns:
        merged = merged[-max_turns:]
    return [compact_one(t) for t in merged]


def _normalize_agenda_state(state: str) -> str:
    s = str(state or "").upper().strip()
    if s in {"PROPOSED", "ACTIVE", "CLOSING", "CLOSED"}:
        return s
    return "PROPOSED"


def _normalize_flow_type(flow: str) -> str:
    raw = str(flow or "").strip()
    if raw in FLOW_TYPE_ORDER:
        return raw
    low = raw.lower()
    mapping = {
        "problem": "문제정의",
        "issue": "문제정의",
        "option": "대안비교",
        "compare": "대안비교",
        "conflict": "의견충돌",
        "disagree": "의견충돌",
        "evidence": "근거검토",
        "verify": "근거검토",
        "closing": "결론수렴",
        "decision": "결론수렴",
        "action": "실행정의",
        "execution": "실행정의",
    }
    for k, v in mapping.items():
        if k in low:
            return v
    return ""


def _agenda_title_tokens(title: str) -> list[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z0-9_+\-./#]*|[가-힣]{2,}", str(title or ""))
    return [t.lower() for t in toks if len(t.strip()) >= 2]


def _related_turns_for_agenda(transcript: list[dict], agenda_title: str, limit: int = 6) -> list[dict]:
    if not transcript:
        return []
    tokens = _agenda_title_tokens(agenda_title)
    recent = transcript[-1200:]
    scored: list[tuple[int, int, dict]] = []
    for idx, turn in enumerate(recent):
        text = str(turn.get("text") or "").strip()
        if not text:
            continue
        lowered = text.lower()
        score = 0
        for tok in tokens:
            if tok and tok in lowered:
                score += 1
        if score > 0:
            scored.append((score, idx, turn))
    if not scored:
        return [t for t in recent[-limit:] if str(t.get("text") or "").strip()]
    picked = sorted(scored, key=lambda x: (-x[0], -x[1]))[: max(limit * 3, limit)]
    picked_turns = sorted([p[2] for p in picked], key=lambda t: str(t.get("timestamp") or ""))
    return picked_turns[-limit:]


def _infer_flow_type_hybrid(
    llm_flow_type: str,
    related_turns: list[dict],
    decision_count: int,
    action_count: int,
) -> str:
    scores = {k: 0.0 for k in FLOW_TYPE_ORDER}
    normalized = _normalize_flow_type(llm_flow_type)
    if normalized:
        scores[normalized] += 2.0

    for turn in related_turns:
        txt = str(turn.get("text") or "").lower()
        for label, kws in FLOW_TYPE_KEYWORDS.items():
            for kw in kws:
                if kw in txt:
                    scores[label] += 1.0

    if decision_count > 0:
        scores["대안비교"] += 1.0
        scores["결론수렴"] += 1.0
    if action_count > 0:
        scores["실행정의"] += 2.0

    best = max(FLOW_TYPE_ORDER, key=lambda k: scores[k])
    if scores[best] <= 0:
        if action_count > 0:
            return "실행정의"
        if decision_count > 0:
            return "결론수렴"
        return "문제정의"
    return best


def _compose_key_utterances(llm_keys: Any, related_turns: list[dict]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    if isinstance(llm_keys, list):
        for row in llm_keys:
            s = str(row or "").strip()
            if not s:
                continue
            if len(s) > 140:
                s = s[:139] + "…"
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
            if len(out) >= 4:
                return out

    for turn in related_turns:
        s = str(turn.get("text") or "").strip()
        if not s:
            continue
        if len(s) > 140:
            s = s[:139] + "…"
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= 4:
            break
    return out


def _build_hybrid_agenda_outcomes(
    *,
    existing_outcomes: list[dict] | None,
    agenda_payload: dict,
    agenda_stack: list[dict],
    transcript: list[dict],
) -> list[dict]:
    rows: list[dict] = []
    existing_titles: set[str] = set()
    state_rank = {"PROPOSED": 0, "CLOSED": 1, "CLOSING": 2, "ACTIVE": 3}

    def _norm_title(raw: Any) -> str:
        t = str(raw or "").strip()
        return t if t else "아젠다 미정"

    # Preserve multiple flow-units per same topic title from LLM output.
    for row in (existing_outcomes or []):
        if not isinstance(row, dict):
            continue
        title = _norm_title(row.get("agenda_title"))
        existing_titles.add(title)
        rows.append(
            {
                "agenda_title": title,
                "agenda_state": _normalize_agenda_state(str(row.get("agenda_state") or "PROPOSED")),
                "flow_type": str(row.get("flow_type") or "").strip(),
                "key_utterances": list(row.get("key_utterances") or []),
                "summary": str(row.get("summary") or "").strip(),
                "decision_results": list(row.get("decision_results") or []),
                "action_items": list(row.get("action_items") or []),
            }
        )

    title_state_map: dict[str, str] = {}

    def _upsert_title_state(title: str, state: str) -> None:
        t = _norm_title(title)
        s = _normalize_agenda_state(state)
        prev = title_state_map.get(t)
        if not prev or state_rank.get(s, 0) >= state_rank.get(prev, 0):
            title_state_map[t] = s

    active = (agenda_payload.get("active") or {})
    active_title = str(active.get("title") or "").strip()
    active_state = _normalize_agenda_state(str(active.get("status") or "ACTIVE"))
    if active_title:
        _upsert_title_state(active_title, active_state)
    for cand in (agenda_payload.get("candidates") or []):
        title = str((cand or {}).get("title") or "").strip()
        if title:
            _upsert_title_state(title, "PROPOSED")

    for item in agenda_stack or []:
        title = str((item or {}).get("title") or "").strip()
        if not title:
            continue
        st = _normalize_agenda_state(str((item or {}).get("status") or "PROPOSED"))
        _upsert_title_state(title, st)

    # If title already exists with multiple flow rows, keep all rows and only normalize state.
    for row in rows:
        title = str(row.get("agenda_title") or "").strip()
        mapped = title_state_map.get(title)
        if mapped and state_rank.get(mapped, 0) >= state_rank.get(str(row.get("agenda_state") or "PROPOSED"), 0):
            row["agenda_state"] = mapped

    # Add missing titles from agenda tracker/fsm as placeholder rows.
    for title, st in title_state_map.items():
        if title in existing_titles:
            continue
        rows.append(
            {
                "agenda_title": title,
                "agenda_state": st,
                "flow_type": "",
                "key_utterances": [],
                "summary": "",
                "decision_results": [],
                "action_items": [],
            }
        )

    enriched: list[tuple[int, dict]] = []
    for idx, row in enumerate(rows):
        title = str(row.get("agenda_title") or "").strip() or "아젠다 미정"
        related = _related_turns_for_agenda(transcript, title, limit=6)
        flow_type = _infer_flow_type_hybrid(
            llm_flow_type=str(row.get("flow_type") or ""),
            related_turns=related,
            decision_count=len(row.get("decision_results") or []),
            action_count=len(row.get("action_items") or []),
        )
        key_utts = _compose_key_utterances(row.get("key_utterances"), related)
        summary = str(row.get("summary") or "").strip()
        if not summary:
            summary = key_utts[0] if key_utts else "핵심 발언을 수집 중입니다."
        enriched.append(
            (
                idx,
                {
                    "agenda_title": title,
                    "agenda_state": _normalize_agenda_state(str(row.get("agenda_state") or "PROPOSED")),
                    "flow_type": flow_type,
                    "key_utterances": key_utts,
                    "summary": summary,
                    "decision_results": list(row.get("decision_results") or []),
                    "action_items": list(row.get("action_items") or []),
                },
            )
        )

    state_order = {"ACTIVE": 0, "CLOSING": 1, "PROPOSED": 2, "CLOSED": 3}
    enriched.sort(
        key=lambda item: (
            state_order.get(str(item[1].get("agenda_state") or "PROPOSED"), 9),
            str(item[1].get("agenda_title") or ""),
            item[0],
        )
    )
    return [item[1] for item in enriched]


def _normalize_timestamp(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        sec = max(0, int(float(raw)))
        hh = (sec // 3600) % 24
        mm = (sec % 3600) // 60
        ss = sec % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    s = str(raw).strip()
    if not s:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        sec = max(0, int(float(s)))
        hh = (sec // 3600) % 24
        mm = (sec % 3600) // 60
        ss = sec % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    if re.fullmatch(r"\d{2}:\d{2}:\d{2}", s):
        return s
    if re.fullmatch(r"\d{2}:\d{2}", s):
        return f"{s}:00"
    if re.fullmatch(r"\d{1,2}:\d{2}:\d{2}", s):
        hh, mm, ss = s.split(":")
        return f"{int(hh):02d}:{int(mm):02d}:{int(ss):02d}"
    if re.fullmatch(r"\d{1,2}:\d{2}", s):
        hh, mm = s.split(":")
        return f"{int(hh):02d}:{int(mm):02d}:00"
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.strftime("%H:%M:%S")
    except ValueError:
        return None


def _pick_first_str(row: dict, keys: tuple[str, ...]) -> str:
    for k in keys:
        val = row.get(k)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _format_age_label(raw: Any) -> str:
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    if re.fullmatch(r"\d{2}", s):
        return f"{s}대"
    if re.fullmatch(r"\d{1,2}", s):
        return f"{int(s)}대"
    return s


def _build_profile_label(age: Any, occupation: Any, role: Any, fallback_id: str) -> str:
    parts = [_format_age_label(age), str(occupation or "").strip(), str(role or "").strip()]
    label = " ".join([p for p in parts if p])
    if label:
        return label
    if fallback_id:
        return fallback_id
    return "화자"


def _extract_rows_from_json_payload(payload: Any) -> tuple[list[dict], dict]:
    meta = {"meeting_goal": ""}
    out: list[dict] = []
    seen: set[tuple[str, str, str | None]] = set()

    # Dataset schema first:
    # - meeting goal: metadata.topic
    # - text: utterance.original_form only
    # - speaker label: speaker_id -> (age, occupation, role)
    if isinstance(payload, dict):
        md = payload.get("metadata")
        if isinstance(md, dict):
            topic = str(md.get("topic") or "").strip()
            if topic:
                meta["meeting_goal"] = topic

        raw_speakers = payload.get("speaker", payload.get("speakers"))
        raw_utterances = payload.get("utterance", payload.get("utterances"))
        if isinstance(raw_utterances, list):
            speaker_map: dict[str, dict] = {}
            if isinstance(raw_speakers, list):
                for s in raw_speakers:
                    if not isinstance(s, dict):
                        continue
                    sid = str(s.get("id") or s.get("speaker_id") or "").strip()
                    if sid:
                        speaker_map[sid] = s

            for item in raw_utterances:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("original_form") or "").strip()
                if not text:
                    continue
                sid = str(item.get("speaker_id") or "").strip()
                profile = speaker_map.get(sid, {})
                label = _build_profile_label(
                    profile.get("age"),
                    profile.get("occupation"),
                    profile.get("role", item.get("speaker_role")),
                    sid or str(item.get("speaker") or "").strip(),
                )
                ts = _normalize_timestamp(item.get("timestamp", item.get("time", item.get("ts", item.get("start")))))
                key = (label, text, ts)
                if key in seen:
                    continue
                seen.add(key)
                out.append({"speaker": label, "text": text, "timestamp": ts})

            if out:
                return out, meta

    def push_row(item: Any) -> None:
        if isinstance(item, str):
            text = item.strip()
            if text:
                key = ("화자", text, None)
                if key not in seen:
                    seen.add(key)
                    out.append({"speaker": "화자", "text": text, "timestamp": None})
            return
        if not isinstance(item, dict):
            return
        if "original_form" in item:
            text = str(item.get("original_form") or "").strip()
        else:
            text = _pick_first_str(
                item,
                (
                    "text",
                    "utterance",
                    "sentence",
                    "content",
                    "transcript",
                    "speech",
                    "message",
                    "asr",
                    "normalized_text",
                    "form",
                ),
            )
        if not text:
            return
        speaker = _pick_first_str(
            item, ("speaker", "spk", "speaker_id", "speaker_role", "role", "name", "participant", "author")
        ) or "화자"
        ts_raw = item.get("timestamp", item.get("time", item.get("ts", item.get("start_time", item.get("start")))))
        ts_norm = _normalize_timestamp(ts_raw)
        key = (speaker, text, ts_norm)
        if key in seen:
            return
        seen.add(key)
        out.append({"speaker": speaker, "text": text, "timestamp": ts_norm})

    def walk(node: Any, depth: int = 0) -> None:
        if depth > 12:
            return
        if isinstance(node, dict):
            push_row(node)
            for v in node.values():
                if isinstance(v, (dict, list)):
                    walk(v, depth + 1)
            return
        if isinstance(node, list):
            for v in node:
                if isinstance(v, str):
                    # Only accept explicit list entries as transcript candidates.
                    push_row(v)
                elif isinstance(v, (dict, list)):
                    walk(v, depth + 1)
            return

    if isinstance(payload, list):
        walk(payload)
        return out, meta

    if isinstance(payload, dict):
        direct_text = _pick_first_str(payload, ("text", "utterance", "sentence", "content", "message"))
        if direct_text:
            push_row(payload)
            return out, meta
        for key in (
            "transcript",
            "utterance",
            "utterances",
            "data",
            "items",
            "segments",
            "results",
            "messages",
            "dialogue",
            "conversation",
            "records",
        ):
            if isinstance(payload.get(key), list):
                walk(payload.get(key, []))
        # 루트 키 패턴이 없더라도 재귀 탐색으로 전사 후보를 찾는다.
        if not out:
            walk(payload)
    return out, meta


def _read_json_file(path: Path) -> Any:
    last_exc: Exception | None = None
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except Exception as exc:
            last_exc = exc
    if last_exc:
        raise last_exc
    raise RuntimeError(f"failed to read json: {path}")


def _read_json_bytes(payload: bytes, label: str = "upload.json") -> Any:
    last_exc: Exception | None = None
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try:
            return json.loads(payload.decode(enc))
        except Exception as exc:
            last_exc = exc
    if last_exc:
        raise RuntimeError(f"failed to parse json: {label}") from last_exc
    raise RuntimeError(f"failed to parse json: {label}")


def _load_utterances_from_json_dir(folder: Path, recursive: bool, max_files: int) -> tuple[list[dict], dict]:
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"folder not found: {folder}")

    files = sorted(folder.rglob("*.json") if recursive else folder.glob("*.json"))
    files = files[:max_files]

    utterances: list[dict] = []
    file_stats: list[dict] = []
    base_dt = datetime.now().replace(microsecond=0)
    synthetic_sec = 0
    total_rows = 0
    parsed_files = 0
    skipped_files = 0
    meeting_goal = ""

    for fp in files:
        try:
            payload = _read_json_file(fp)
            rows, meta = _extract_rows_from_json_payload(payload)
            if not meeting_goal:
                meeting_goal = str(meta.get("meeting_goal") or "").strip()
            if not rows:
                skipped_files += 1
                continue
            parsed_files += 1
            local_count = 0
            for row in rows:
                txt = str(row.get("text") or "").strip()
                if not txt:
                    continue
                ts = _normalize_timestamp(row.get("timestamp"))
                if ts is None:
                    ts = (base_dt + timedelta(seconds=synthetic_sec)).strftime("%H:%M:%S")
                    synthetic_sec += 5
                utterances.append(
                    {
                        "speaker": str(row.get("speaker") or "화자").strip() or "화자",
                        "text": txt,
                        "timestamp": ts,
                    }
                )
                local_count += 1
            total_rows += local_count
            file_stats.append({"file": str(fp), "rows": local_count})
        except Exception:
            skipped_files += 1

    return utterances, {
        "folder": str(folder),
        "files_scanned": len(files),
        "files_parsed": parsed_files,
        "files_skipped": skipped_files,
        "rows_loaded": total_rows,
        "meeting_goal": meeting_goal,
        "file_stats": file_stats[:20],
    }


class MeetingRuntime:
    def __init__(self):
        self._lock = threading.Lock()
        self.client = get_client()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.meeting_goal = "롤아웃 방식과 최종 의사결정 책임자를 정한다."
            self.initial_context = ""
            self.window_size = 12
            self.transcript: list[dict] = []
            self.agenda_stack: list[dict] = []
            self.analysis: Optional[dict] = None
            self.artifacts: dict[str, dict] = {}
            self.stt_chunk_seq = 0
            self.agenda_candidates: list[dict] = []
            self.agenda_vectors: dict[str, dict] = {}
            self.agenda_tracker_debug: dict[str, dict] = {}
            self.agenda_state_map: dict[str, dict] = {}
            self.active_agenda_id: str = ""
            self.agenda_events: list[dict] = []
            self.meeting_started_at: float = time.time()
            self.drift_state: str = "Normal"
            self.drift_ui_cues: dict[str, bool] = {
                "glow_k_core": False,
                "fix_k_core_focus": False,
                "reduce_facets": False,
                "show_banner": False,
            }
            self.drift_debug: dict[str, float | str | bool | int] = {
                "s45": 0.0,
                "band": "Green",
                "yellow_seconds": 0.0,
                "red_seconds": 0.0,
                "safe_zone": True,
                "window_turns": 0,
            }
            self.drift_monitor: dict[str, float | str | None] = {
                "yellow_since": None,
                "red_since": None,
                "last_state": "Normal",
                "last_band": "Green",
            }
            self.dps_t: float = 0.0
            self.dps_breakdown: dict[str, dict] = {}
            self.dps_history: list[dict] = []
            self.stagnation_flag: bool = False
            self.loop_state: str = "Normal"
            self.flow_pulse_debug: dict[str, dict | float | int | bool] = {}
            self.decision_lock_debug: dict[str, float | int | bool] = {}
            self.evidence_status: str = "UNVERIFIED"
            self.evidence_snippet: str = ""
            self.evidence_log: list[dict] = []
            self.recommendation_debug: dict[str, object] = {}
            self.fairtalk_glow: list[dict] = []
            self.fairtalk_debug: dict[str, object] = {}
            self.fairtalk_monitor: dict[str, dict[str, float | str | None]] = {}
            self.analysis_runtime: dict[str, object] = {}
            self.llm_enabled: bool = False
            self._refresh_live_artifacts()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "meeting_goal": self.meeting_goal,
                "initial_context": self.initial_context,
                "window_size": self.window_size,
                "transcript": self.transcript,
                "agenda_stack": self.agenda_stack,
                "agenda_candidates": self.agenda_candidates,
                "agenda_vectors": self.agenda_vectors,
                "agenda_tracker_debug": self.agenda_tracker_debug,
                "agenda_state_map": self.agenda_state_map,
                "active_agenda_id": self.active_agenda_id,
                "agenda_events": self.agenda_events,
                "drift_state": self.drift_state,
                "drift_ui_cues": self.drift_ui_cues,
                "drift_debug": self.drift_debug,
                "dps_t": self.dps_t,
                "dps_breakdown": self.dps_breakdown,
                "stagnation_flag": self.stagnation_flag,
                "loop_state": self.loop_state,
                "flow_pulse_debug": self.flow_pulse_debug,
                "decision_lock_debug": self.decision_lock_debug,
                "evidence_status": self.evidence_status,
                "evidence_snippet": self.evidence_snippet,
                "evidence_log": self.evidence_log,
                "recommendation_debug": self.recommendation_debug,
                "fairtalk_glow": self.fairtalk_glow,
                "fairtalk_debug": self.fairtalk_debug,
                "analysis_runtime": self.analysis_runtime,
                "llm_enabled": self.llm_enabled,
                "llm_status": self.client.get_status(),
                "analysis": self.analysis,
                "artifacts": self.artifacts,
            }

    def update_config(self, cfg: ConfigInput) -> None:
        with self._lock:
            self.meeting_goal = cfg.meeting_goal.strip()
            self.initial_context = ""
            self.window_size = cfg.window_size

    def set_meeting_goal(self, goal: str) -> None:
        goal = (goal or "").strip()
        if not goal:
            return
        with self._lock:
            self.meeting_goal = goal

    def set_llm_enabled(self, enabled: bool) -> None:
        with self._lock:
            self.llm_enabled = bool(enabled)

    def reset_keep_config(self) -> None:
        snap = self.snapshot()
        cfg = ConfigInput(
            meeting_goal=str(snap.get("meeting_goal") or "").strip(),
            initial_context="",
            window_size=int(snap.get("window_size") or 12),
        )
        self.reset()
        self.update_config(cfg)

    def append_utterance(self, speaker: str, text: str, timestamp: Optional[str] = None) -> None:
        text = text.strip()
        if not text:
            return
        utterance = TranscriptUtterance(
            speaker=(speaker or "화자").strip(),
            text=text,
            timestamp=(timestamp or _now_ts()).strip(),
        )
        with self._lock:
            self.transcript.append(utterance.model_dump())
            self._refresh_live_artifacts()

    def append_utterances_bulk(self, utterances: list[dict]) -> int:
        with self._lock:
            added = 0
            for row in utterances:
                text = str(row.get("text") or "").strip()
                if not text:
                    continue
                utterance = TranscriptUtterance(
                    speaker=str(row.get("speaker") or "화자").strip() or "화자",
                    text=text,
                    timestamp=(str(row.get("timestamp") or "").strip() or _now_ts()),
                )
                self.transcript.append(utterance.model_dump())
                added += 1
            if added > 0:
                self._refresh_live_artifacts()
            return added

    def _sync_agenda_stack(self, analysis: dict) -> None:
        agenda_map = {item["title"]: item["status"] for item in self.agenda_stack}
        active_title = (analysis.get("agenda", {}).get("active", {}).get("title") or "").strip()
        if active_title:
            agenda_map[active_title] = "ACTIVE"

        for candidate in analysis.get("agenda", {}).get("candidates", []):
            title = (candidate.get("title") or "").strip()
            if title and title not in agenda_map:
                agenda_map[title] = "PROPOSED"

        self.agenda_stack = [{"title": title, "status": status} for title, status in agenda_map.items()]

    def _sync_agenda_stack_from_state_map(self) -> None:
        # Legacy compatibility only: keep normalized stack order.
        order = {"ACTIVE": 0, "CLOSING": 1, "PROPOSED": 2, "CLOSED": 3}
        entries = list(self.agenda_state_map.values())
        entries.sort(key=lambda e: (order.get(str(e.get("state") or "PROPOSED"), 9), -float(e.get("confidence", 0.0))))
        stack: list[dict] = []
        for entry in entries:
            title = (entry.get("title") or "").strip()
            state = str(entry.get("state") or "PROPOSED").upper()
            if state not in {"PROPOSED", "ACTIVE", "CLOSING", "CLOSED"}:
                state = "PROPOSED"
            if title:
                stack.append({"title": title, "status": state})
        self.agenda_stack = stack

    def tick_analysis(self, use_full_context: bool = False) -> bool:
        with self._lock:
            if not self.llm_enabled:
                self.analysis_runtime = {
                    "tick_mode": "full_context" if use_full_context else "windowed",
                    "transcript_count": len(self.transcript),
                    "llm_window_turns": 0,
                    "engine_window_turns": 0,
                    "control_plane_source": "blocked",
                    "control_plane_reason": "llm_not_enabled",
                    "used_local_fallback": False,
                }
                return False
            if use_full_context:
                engine_window = self.transcript[-FULL_CONTEXT_ENGINE_MAX_TURNS:]
                transcript_window = list(engine_window)
            else:
                engine_window = self.transcript
                transcript_window = self.transcript[-self.window_size :]
            llm_turn_limit = LLM_FULL_CONTEXT_MAX_TURNS if use_full_context else LLM_CONTEXT_MAX_TURNS
            llm_window = _compact_transcript_for_llm(
                transcript_window,
                max_turns=llm_turn_limit,
                clip_chars=LLM_TEXT_CLIP_CHARS,
            )
            current_active = ""
            if self.analysis:
                current_active = (
                    self.analysis.get("agenda", {})
                    .get("active", {})
                    .get("title", "")
                )
            analysis = self.client.analyze_meeting(
                meeting_goal=self.meeting_goal,
                initial_context=self.initial_context,
                current_active_agenda=current_active,
                transcript_window=llm_window,
                agenda_stack=self.agenda_stack,
            )
            self.analysis = analysis.model_dump()
            # Scoring-style local engines are intentionally disabled in this prototype.
            self.fairtalk_glow = []
            self.fairtalk_debug = {"rule": "disabled"}
            self.fairtalk_monitor = {}

            # Normalize agenda stack only from simplified analysis.
            self._sync_agenda_stack(self.analysis)
            self.agenda_candidates = [
                {
                    "title": str(c.get("title") or "").strip(),
                    "status": "PROPOSED",
                    "confidence": float(c.get("confidence", 0.5)),
                }
                for c in (self.analysis.get("agenda", {}).get("candidates") or [])
                if str(c.get("title") or "").strip()
            ]
            self.agenda_vectors = {}
            self.agenda_state_map = {}
            self.active_agenda_id = ""
            self.agenda_events = []

            # Keep only requested minimal agenda fields.
            cleaned_outcomes: list[dict] = []
            for row in (self.analysis.get("agenda_outcomes") or []):
                if not isinstance(row, dict):
                    continue
                cleaned_outcomes.append(
                    {
                        "agenda_title": str(row.get("agenda_title") or "").strip() or "아젠다 미정",
                        "key_utterances": [str(x).strip() for x in (row.get("key_utterances") or []) if str(x).strip()],
                        "summary": str(row.get("summary") or "").strip(),
                        "agenda_keywords": [str(x).strip() for x in (row.get("agenda_keywords") or []) if str(x).strip()],
                        "decision_results": [
                            {
                                "opinions": [str(o).strip() for o in (d.get("opinions") or []) if str(o).strip()],
                                "conclusion": str(d.get("conclusion") or "").strip(),
                            }
                            for d in (row.get("decision_results") or [])
                            if isinstance(d, dict)
                        ],
                        "action_items": list(row.get("action_items") or []),
                    }
                )
            self.analysis["agenda_outcomes"] = cleaned_outcomes

            claims = list((self.analysis.get("evidence_gate") or {}).get("claims") or [])
            self.evidence_status = ""
            self.evidence_log = claims[-120:]
            self.evidence_snippet = "\n".join(
                [str(c.get("claim") or "").strip() for c in claims[:2] if isinstance(c, dict) and str(c.get("claim") or "").strip()]
            )
            self.analysis["evidence_gate"] = {"claims": claims}

            self.analysis_runtime = {
                "tick_mode": "full_context" if use_full_context else "windowed",
                "transcript_count": len(self.transcript),
                "llm_window_turns": len(llm_window),
                "engine_window_turns": len(engine_window),
                "control_plane_source": "disabled",
                "control_plane_reason": "analysis_only",
                "used_local_fallback": False,
            }

            self._refresh_live_artifacts()
            return True

    def _refresh_live_artifacts(self) -> None:
        # Disabled: generate_artifact path removed per simplified prototype request.
        self.artifacts = {}

    def next_chunk_id(self) -> int:
        with self._lock:
            self.stt_chunk_seq += 1
            return self.stt_chunk_seq


def _transcribe_with_whisper(audio_bytes: bytes, suffix: str, source: str) -> str:
    model = _get_whisper_model()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        result = model.transcribe(
            tmp_path,
            task="transcribe",
            language="ko",
            verbose=None,
            fp16=_whisper_uses_cuda(model),
            temperature=0.0,
            beam_size=1,
            best_of=1,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        text = _post_filter_whisper_text((result.get("text") or "").strip(), result)
        if not text and WHISPER_ENABLE_FALLBACK:
            fallback = model.transcribe(
                tmp_path,
                task="transcribe",
                language="ko",
                verbose=None,
                fp16=_whisper_uses_cuda(model),
                temperature=0.0,
                beam_size=1,
                best_of=1,
                condition_on_previous_text=False,
                without_timestamps=True,
            )
            text = _post_filter_whisper_text((fallback.get("text") or "").strip(), fallback)
        if not text:
            return ""
        return f"[{source}] {text}"
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def _post_filter_whisper_text(text: str, result: dict) -> str:
    clean = text.strip()
    if not clean:
        return ""

    segments = result.get("segments") or []
    if not segments:
        return clean

    no_speech_probs: list[float] = []
    avg_logprobs: list[float] = []
    for seg in segments:
        try:
            no_speech_probs.append(float(seg.get("no_speech_prob", 1.0)))
        except Exception:
            no_speech_probs.append(1.0)
        try:
            avg_logprobs.append(float(seg.get("avg_logprob", -10.0)))
        except Exception:
            avg_logprobs.append(-10.0)

    all_silence_like = bool(no_speech_probs) and all(p >= WHISPER_SILENCE_NSP for p in no_speech_probs)
    mean_logprob = sum(avg_logprobs) / max(1, len(avg_logprobs))

    normalized = re.sub(r"[\s\.,!?~…\"'`]+", "", clean).lower()
    if normalized in HALLUCINATION_PHRASES and all_silence_like:
        return ""

    if all_silence_like and mean_logprob <= WHISPER_SILENCE_LOGPROB and len(normalized) <= 18:
        return ""

    return clean


@lru_cache(maxsize=1)
def _get_whisper_model():
    import torch
    import whisper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model(WHISPER_MODEL_NAME, device=device)


def _whisper_uses_cuda(model) -> bool:
    try:
        dev = next(model.parameters()).device
        return dev.type == "cuda"
    except Exception:
        return False


runtime = MeetingRuntime()
app = FastAPI(title="Meeting Rhythm API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/state")
def get_state():
    return runtime.snapshot()


@app.get("/api/llm/status")
def get_llm_status():
    return runtime.client.get_status()


@app.post("/api/llm/connect")
def post_llm_connect():
    result = runtime.client.ping()
    enabled = bool(result.get("ok"))
    runtime.set_llm_enabled(enabled)
    return {
        "enabled": enabled,
        "result": result,
        "llm_status": runtime.client.get_status(),
        "state": runtime.snapshot(),
    }


@app.post("/api/llm/disconnect")
def post_llm_disconnect():
    runtime.set_llm_enabled(False)
    return {
        "enabled": False,
        "llm_status": runtime.client.get_status(),
        "state": runtime.snapshot(),
    }


@app.post("/api/llm/ping")
def post_llm_ping():
    result = runtime.client.ping()
    return {
        "result": result,
        "llm_status": runtime.client.get_status(),
    }


@app.post("/api/config")
def post_config(payload: ConfigInput):
    runtime.update_config(payload)
    return runtime.snapshot()


@app.post("/api/transcript/manual")
def post_manual_utterance(payload: UtteranceInput):
    runtime.append_utterance(payload.speaker, payload.text, payload.timestamp)
    return runtime.snapshot()


@app.post("/api/transcript/import-json-dir")
def post_import_json_dir(payload: DatasetImportInput):
    folder = Path(payload.folder).expanduser()
    if not folder.is_absolute():
        folder = (Path.cwd() / folder).resolve()
    if (not folder.exists()) and folder.name.lower() == "economy":
        alt = folder.parent / "ecnomy"
        if alt.exists():
            folder = alt
    utterances, stats = _load_utterances_from_json_dir(folder, recursive=payload.recursive, max_files=payload.max_files)
    goal_from_json = str(stats.get("meeting_goal") or "").strip()

    if payload.reset_state:
        runtime.reset_keep_config()
    if goal_from_json:
        runtime.set_meeting_goal(goal_from_json)
    added = runtime.append_utterances_bulk(utterances)
    ticked = False
    if payload.auto_tick and added > 0:
        ticked = runtime.tick_analysis(use_full_context=True)

    return {
        "state": runtime.snapshot(),
        "import_debug": {
            **stats,
            "added": added,
            "reset_state": payload.reset_state,
            "auto_tick": payload.auto_tick,
            "ticked": ticked,
            "analysis_mode": "full_context_once" if ticked else "none",
            "meeting_goal_applied": bool(goal_from_json),
            "warning": (
                "업로드한 JSON에서 전사 문장을 찾지 못했습니다."
                if added == 0
                else ("LLM 연결 버튼을 누르지 않아 분석은 실행되지 않았습니다." if payload.auto_tick and not ticked else "")
            ),
        },
    }


@app.post("/api/transcript/import-json-files")
async def post_import_json_files(
    files: list[UploadFile] = File(...),
    reset_state: bool = Form(True),
    auto_tick: bool = Form(True),
):
    utterances: list[dict] = []
    file_stats: list[dict] = []
    base_dt = datetime.now().replace(microsecond=0)
    synthetic_sec = 0
    parsed_files = 0
    skipped_files = 0
    meeting_goal = ""

    for idx, upload in enumerate(files):
        name = (upload.filename or f"upload_{idx + 1}.json").strip() or f"upload_{idx + 1}.json"
        try:
            payload = await upload.read()
            if not payload:
                skipped_files += 1
                continue

            rows, meta = _extract_rows_from_json_payload(_read_json_bytes(payload, label=name))
            if not meeting_goal:
                meeting_goal = str(meta.get("meeting_goal") or "").strip()
            if not rows:
                skipped_files += 1
                continue

            parsed_files += 1
            local_count = 0
            for row in rows:
                txt = str(row.get("text") or "").strip()
                if not txt:
                    continue
                ts = _normalize_timestamp(row.get("timestamp"))
                if ts is None:
                    ts = (base_dt + timedelta(seconds=synthetic_sec)).strftime("%H:%M:%S")
                    synthetic_sec += 5
                utterances.append(
                    {
                        "speaker": str(row.get("speaker") or "화자").strip() or "화자",
                        "text": txt,
                        "timestamp": ts,
                    }
                )
                local_count += 1

            file_stats.append({"file": name, "rows": local_count})
        except Exception:
            skipped_files += 1

    if reset_state:
        runtime.reset_keep_config()
    if meeting_goal:
        runtime.set_meeting_goal(meeting_goal)
    added = runtime.append_utterances_bulk(utterances)
    ticked = False
    if auto_tick and added > 0:
        ticked = runtime.tick_analysis(use_full_context=True)

    return {
        "state": runtime.snapshot(),
        "import_debug": {
            "folder": "uploaded_files",
            "files_scanned": len(files),
            "files_parsed": parsed_files,
            "files_skipped": skipped_files,
            "rows_loaded": len(utterances),
            "meeting_goal": meeting_goal,
            "file_stats": file_stats[:20],
            "added": added,
            "reset_state": reset_state,
            "auto_tick": auto_tick,
            "ticked": ticked,
            "analysis_mode": "full_context_once" if ticked else "none",
            "meeting_goal_applied": bool(meeting_goal),
            "warning": (
                "업로드한 JSON에서 전사 문장을 찾지 못했습니다."
                if added == 0
                else ("LLM 연결 버튼을 누르지 않아 분석은 실행되지 않았습니다." if auto_tick and not ticked else "")
            ),
        },
    }


@app.post("/api/analysis/tick")
def post_analysis_tick():
    ok = runtime.tick_analysis()
    if not ok:
        raise HTTPException(status_code=409, detail="LLM 연결 버튼을 먼저 누르세요. 연결 전에는 분석 기능이 비활성화됩니다.")
    return runtime.snapshot()

@app.post("/api/reset")
def post_reset():
    runtime.reset()
    return runtime.snapshot()


@app.post("/api/stt/chunk")
async def post_stt_chunk(
    audio: UploadFile = File(...),
    speaker: str = Form("시스템오디오"),
    source: str = Form("live"),
):
    started = time.perf_counter()
    chunk_id = runtime.next_chunk_id()
    steps: list[dict] = []

    def mark(step: str) -> None:
        elapsed = int((time.perf_counter() - started) * 1000)
        steps.append({"step": step, "t_ms": elapsed})

    mark("request_received")
    payload = await audio.read()
    mark("audio_read")
    if not payload:
        mark("empty_payload")
        return {
            "state": runtime.snapshot(),
            "stt_debug": {
                "chunk_id": chunk_id,
                "status": "empty",
                "source": source,
                "speaker": speaker,
                "filename": audio.filename or "chunk",
                "bytes": 0,
                "steps": steps,
                "duration_ms": int((time.perf_counter() - started) * 1000),
                "transcript_chars": 0,
                "transcript_preview": "",
                "error": "",
            },
        }
    suffix = Path(audio.filename or "").suffix.lower()
    if not suffix:
        ctype = (audio.content_type or "").lower()
        if "mp4" in ctype or "mpeg" in ctype:
            suffix = ".mp4"
        elif "ogg" in ctype:
            suffix = ".ogg"
        elif "wav" in ctype or "wave" in ctype:
            suffix = ".wav"
        else:
            suffix = ".webm"

    mark("transcribe_start")
    try:
        transcript = _transcribe_with_whisper(payload, suffix=suffix, source=source)
        mark("transcribe_end")
    except Exception as exc:
        mark("transcribe_error")
        return {
            "state": runtime.snapshot(),
            "stt_debug": {
                "chunk_id": chunk_id,
                "status": "error",
                "source": source,
                "speaker": speaker,
                "filename": audio.filename or f"chunk{suffix}",
                "bytes": len(payload),
                "steps": steps,
                "duration_ms": int((time.perf_counter() - started) * 1000),
                "transcript_chars": 0,
                "transcript_preview": "",
                "error": str(exc)[:600],
            },
        }

    if transcript:
        runtime.append_utterance(speaker, transcript)
        mark("transcript_appended")
        status = "ok"
    else:
        mark("no_speech")
        status = "empty"

    snapshot = runtime.snapshot()
    return {
        "state": snapshot,
        "stt_debug": {
            "chunk_id": chunk_id,
            "status": status,
            "source": source,
            "speaker": speaker,
            "filename": audio.filename or "chunk",
            "bytes": len(payload),
            "steps": steps,
            "duration_ms": int((time.perf_counter() - started) * 1000),
            "transcript_chars": len(transcript),
            "transcript_preview": transcript[:120],
            "error": "",
        },
    }
