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

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agenda_fsm import run_agenda_fsm
from agenda_tracker import run_agenda_tracker
from decision_lock_engine import run_decision_lock
from deliverables_engine import build_live_artifact
from dps_engine import compute_dps
from drift_dampener import run_drift_dampener
from evidence_gate_engine import run_evidence_gate
from fairtalk_engine import run_fairtalk_engine
from flow_pulse import run_flow_pulse
from llm_client import get_client
from recommendation_engine import run_recommendation_engine
from schemas import AgendaStatus, ArtifactKind, TranscriptUtterance

WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "large")
WHISPER_ENABLE_FALLBACK = os.environ.get("WHISPER_ENABLE_FALLBACK", "0") == "1"
WHISPER_SILENCE_NSP = float(os.environ.get("WHISPER_SILENCE_NSP", "0.65"))
WHISPER_SILENCE_LOGPROB = float(os.environ.get("WHISPER_SILENCE_LOGPROB", "-0.35"))
USE_LLM_CONTROL_PLANE = os.environ.get("USE_LLM_CONTROL_PLANE", "1") == "1"
LLM_CONTEXT_MAX_TURNS = int(os.environ.get("LLM_CONTEXT_MAX_TURNS", "320"))
FULL_CONTEXT_ENGINE_MAX_TURNS = int(os.environ.get("FULL_CONTEXT_ENGINE_MAX_TURNS", "3000"))
HALLUCINATION_PHRASES = {
    "감사합니다",
    "고맙습니다",
}


class ConfigInput(BaseModel):
    meeting_goal: str
    initial_context: str
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


def _extract_rows_from_json_payload(payload: Any) -> list[dict]:
    out: list[dict] = []
    seen: set[tuple[str, str, str | None]] = set()

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
        text = _pick_first_str(
            item,
            (
                "text",
                "utterance",
                "form",
                "original_form",
                "sentence",
                "content",
                "transcript",
                "speech",
                "message",
                "asr",
                "normalized_text",
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
                walk(v, depth + 1)
            return
        if isinstance(node, list):
            for v in node:
                walk(v, depth + 1)
            return
        if isinstance(node, str):
            # 문자열 배열 형태도 전사로 흡수
            push_row(node)
            return

    if isinstance(payload, list):
        walk(payload)
        return out

    if isinstance(payload, dict):
        direct_text = _pick_first_str(payload, ("text", "utterance", "sentence", "content", "message"))
        if direct_text:
            push_row(payload)
            return out
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
    return out


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

    for fp in files:
        try:
            payload = _read_json_file(fp)
            rows = _extract_rows_from_json_payload(payload)
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
            self.initial_context = "엔지니어링/운영이 함께하는 주간 제품 회의."
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
                "llm_status": self.client.get_status(),
                "analysis": self.analysis,
                "artifacts": self.artifacts,
            }

    def update_config(self, cfg: ConfigInput) -> None:
        with self._lock:
            self.meeting_goal = cfg.meeting_goal.strip()
            self.initial_context = cfg.initial_context.strip()
            self.window_size = cfg.window_size

    def reset_keep_config(self) -> None:
        snap = self.snapshot()
        cfg = ConfigInput(
            meeting_goal=str(snap.get("meeting_goal") or "").strip(),
            initial_context=str(snap.get("initial_context") or "").strip(),
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
        active_status = analysis.get("agenda", {}).get("active", {}).get("status", AgendaStatus.ACTIVE.value)
        if active_title:
            agenda_map[active_title] = active_status

        for candidate in analysis.get("agenda", {}).get("candidates", []):
            title = (candidate.get("title") or "").strip()
            if title and title not in agenda_map:
                agenda_map[title] = AgendaStatus.PROPOSED.value

        self.agenda_stack = [{"title": title, "status": status} for title, status in agenda_map.items()]

    def _sync_agenda_stack_from_state_map(self) -> None:
        order = {"ACTIVE": 0, "CLOSING": 1, "PROPOSED": 2, "CLOSED": 3}
        entries = list(self.agenda_state_map.values())
        entries.sort(key=lambda e: (order.get(str(e.get("state") or "PROPOSED"), 9), -float(e.get("confidence", 0.0))))
        stack: list[dict] = []
        for entry in entries:
            title = (entry.get("title") or "").strip()
            state = str(entry.get("state") or "PROPOSED")
            if not title:
                continue
            if state not in {
                AgendaStatus.PROPOSED.value,
                AgendaStatus.ACTIVE.value,
                AgendaStatus.CLOSING.value,
                AgendaStatus.CLOSED.value,
            }:
                state = AgendaStatus.PROPOSED.value
            stack.append({"title": title, "status": state})
        self.agenda_stack = stack

    def tick_analysis(self, use_full_context: bool = False) -> None:
        with self._lock:
            if use_full_context:
                engine_window = self.transcript[-FULL_CONTEXT_ENGINE_MAX_TURNS:]
                transcript_window = list(engine_window)
            else:
                engine_window = self.transcript
                transcript_window = self.transcript[-self.window_size :]
            llm_window = transcript_window[-LLM_CONTEXT_MAX_TURNS:]
            fairtalk_window = engine_window if use_full_context else self.transcript[-240:]
            evidence_window = engine_window if use_full_context else self.transcript[-240:]
            recommendation_window = engine_window if use_full_context else self.transcript[-240:]
            tracker_window = engine_window if use_full_context else self.transcript[-80:]
            flow_window = engine_window if use_full_context else self.transcript[-200:]
            lock_window = engine_window if use_full_context else self.transcript[-180:]
            fsm_window = engine_window if use_full_context else self.transcript[-80:]
            drift_window = engine_window if use_full_context else self.transcript[-120:]
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
            now_tick = time.time()
            fairtalk_out = run_fairtalk_engine(
                transcript=fairtalk_window,
                monitor_state=self.fairtalk_monitor,
                now_ts=now_tick,
            )
            self.fairtalk_glow = list(fairtalk_out.get("participants") or [])
            self.fairtalk_debug = dict(fairtalk_out.get("debug") or {})
            self.fairtalk_monitor = dict(fairtalk_out.get("monitor") or {})
            self.analysis.setdefault("scores", {}).setdefault("participation", {})["fairtalk"] = list(
                fairtalk_out.get("fairtalk_schema") or []
            )
            self.analysis["scores"]["participation"]["imbalance"] = int(fairtalk_out.get("imbalance", 0))

            evidence_out = run_evidence_gate(transcript=evidence_window)
            self.evidence_status = str(evidence_out.get("evidence_status") or "UNVERIFIED")
            self.evidence_snippet = str(evidence_out.get("evidence_snippet") or "")
            self.evidence_log = list(evidence_out.get("evidence_log") or [])[-120:]
            self.analysis.setdefault("evidence_gate", {})["status"] = self.evidence_status
            self.analysis["evidence_gate"]["claims"] = list(evidence_out.get("claims_for_schema") or [])[:8]
            recommendation_out = run_recommendation_engine(
                transcript=recommendation_window,
                analysis=self.analysis,
                evidence_status=self.evidence_status,
            )
            self.analysis.setdefault("recommendations", {})["r1_resources"] = list(
                recommendation_out.get("r1_resources") or []
            )[:3]
            self.analysis["recommendations"]["r2_options"] = list(
                recommendation_out.get("r2_options") or []
            )[:2]
            self.recommendation_debug = dict(recommendation_out.get("debug") or {})

            def _run_local_control_pipeline(reason: str) -> None:
                tracker_output = run_agenda_tracker(
                    transcript=tracker_window,
                    active_agenda=(self.analysis.get("agenda", {}).get("active", {}).get("title") or current_active),
                    agenda_stack=self.agenda_stack,
                    keywords=self.analysis.get("keywords", {}),
                    existing_vectors=self.agenda_vectors,
                )
                self.agenda_candidates = tracker_output.get("agenda_candidates", [])
                self.agenda_vectors = tracker_output.get("agenda_vectors", {})
                self.agenda_tracker_debug = tracker_output.get("tracker_debug", {})
                self.agenda_tracker_debug["fallback_reason"] = reason

                existing_titles = {
                    (c.get("title") or "").strip()
                    for c in self.analysis.get("agenda", {}).get("candidates", [])
                    if (c.get("title") or "").strip()
                }
                for cand in self.agenda_candidates:
                    title = (cand.get("title") or "").strip()
                    if not title or title in existing_titles:
                        continue
                    self.analysis.setdefault("agenda", {}).setdefault("candidates", []).append(
                        {
                            "title": title,
                            "confidence": float(cand.get("confidence", 0.6)),
                        }
                    )
                    existing_titles.add(title)

                dps_pre = compute_dps(
                    analysis=self.analysis,
                    agenda_state_map=self.agenda_state_map,
                    active_agenda_id=self.active_agenda_id,
                )
                local_now_ts = time.time()
                history_for_flow = [h for h in self.dps_history if local_now_ts - float(h.get("ts", local_now_ts)) <= 180.0]
                history_for_flow.append({"ts": local_now_ts, "score": float(dps_pre.get("score", 0.0))})
                k_core = (self.analysis.get("keywords", {}) or {}).get("k_core", {}) or {}
                flow_out = run_flow_pulse(
                    transcript=flow_window,
                    k_core=k_core,
                    dps_history=history_for_flow,
                    now_ts=local_now_ts,
                )
                self.stagnation_flag = bool(flow_out.get("stagnation_flag", False))
                self.loop_state = str(flow_out.get("loop_state", "Normal"))
                self.flow_pulse_debug = dict(flow_out.get("debug", {}))

                if self.stagnation_flag:
                    lock = self.analysis.setdefault("intervention", {}).setdefault("decision_lock", {})
                    existing_reason = str(lock.get("reason") or "").strip()
                    trigger_text = "Flow Pulse trigger #2: circular stagnation detected"
                    lock["triggered"] = True
                    if trigger_text not in existing_reason:
                        lock["reason"] = f"{existing_reason}; {trigger_text}" if existing_reason else trigger_text

                prev_lock = bool(
                    ((self.analysis.get("intervention") or {}).get("decision_lock") or {}).get("triggered", False)
                )
                lock_out = run_decision_lock(
                    transcript=lock_window,
                    meeting_elapsed_sec=time.time() - self.meeting_started_at,
                    stagnation_flag=self.stagnation_flag,
                    previous_decision_lock=prev_lock,
                )
                lock_payload = self.analysis.setdefault("intervention", {}).setdefault("decision_lock", {})
                lock_payload["triggered"] = bool(lock_out.get("triggered", False))
                lock_payload["reason"] = str(lock_out.get("reason", ""))
                self.decision_lock_debug = dict(lock_out.get("debug") or {})

                fsm_out = run_agenda_fsm(
                    agenda_state_map=self.agenda_state_map,
                    active_agenda_id=self.active_agenda_id,
                    agenda_candidates=self.agenda_candidates,
                    analysis=self.analysis,
                    transcript=fsm_window,
                    existing_events=self.agenda_events,
                )
                self.agenda_state_map = fsm_out.get("agenda_state_map", {})
                self.active_agenda_id = str(fsm_out.get("active_agenda_id", "") or "")
                self.agenda_events = list(fsm_out.get("events", []))
                self._sync_agenda_stack_from_state_map()

                dps_out = compute_dps(
                    analysis=self.analysis,
                    agenda_state_map=self.agenda_state_map,
                    active_agenda_id=self.active_agenda_id,
                )
                self.dps_t = float(dps_out.get("score", 0.0))
                self.dps_breakdown = dict(dps_out.get("breakdown", {}))
                self.analysis.setdefault("scores", {}).setdefault("dps", {})["score"] = self.dps_t
                self.analysis["scores"]["dps"]["why"] = (
                    "Option {option_coverage:.2f} | Constraint {constraint_coverage:.2f} | "
                    "Evidence {evidence_coverage:.2f} | Trade-off {tradeoff_coverage:.2f} | "
                    "Closing {closing_readiness:.2f}"
                ).format(**self.dps_breakdown)
                self.dps_history = [h for h in self.dps_history if local_now_ts - float(h.get("ts", local_now_ts)) <= 180.0]
                self.dps_history.append({"ts": local_now_ts, "score": self.dps_t})

                active_title_for_drift = ""
                if self.active_agenda_id and self.active_agenda_id in self.agenda_state_map:
                    active_title_for_drift = str(self.agenda_state_map[self.active_agenda_id].get("title") or "")
                if not active_title_for_drift:
                    active_title_for_drift = str(
                        (self.analysis.get("agenda", {}).get("active", {}).get("title") or "")
                    ).strip()
                active_vec = self.agenda_vectors.get(active_title_for_drift, {})
                drift_out = run_drift_dampener(
                    transcript=drift_window,
                    active_agenda_title=active_title_for_drift,
                    active_agenda_vector=active_vec,
                    monitor_state=self.drift_monitor,
                    meeting_elapsed_sec=time.time() - self.meeting_started_at,
                )
                self.drift_state = str(drift_out.get("drift_state") or "Normal")
                self.drift_ui_cues = dict(drift_out.get("ui_cues") or {})
                self.drift_debug = dict(drift_out.get("debug") or {})
                self.drift_monitor = dict(drift_out.get("monitor") or self.drift_monitor)

            control_plane_source = "local_disabled"
            control_plane_reason = "USE_LLM_CONTROL_PLANE=0"
            used_local_fallback = False
            if USE_LLM_CONTROL_PLANE:
                control_payload = self.client.infer_control_plane(
                    meeting_goal=self.meeting_goal,
                    initial_context=self.initial_context,
                    current_active_agenda=(self.analysis.get("agenda", {}).get("active", {}).get("title") or current_active),
                    transcript_window=llm_window,
                    agenda_stack=self.agenda_stack,
                    analysis=self.analysis,
                    previous_state={
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
                    },
                )
                control_meta = dict(control_payload.get("_meta") or {})
                control_plane_source = str(control_meta.get("source") or "unknown")
                control_plane_reason = str(control_meta.get("reason") or "")
                if "_meta" in control_payload:
                    control_payload.pop("_meta", None)
                control_keywords = control_payload.get("keywords")
                if isinstance(control_keywords, dict) and control_keywords:
                    self.analysis["keywords"] = control_keywords

                tracker_payload = dict(control_payload.get("agenda_tracker") or {})
                self.agenda_candidates = list(tracker_payload.get("agenda_candidates") or [])
                self.agenda_vectors = dict(tracker_payload.get("agenda_vectors") or {})
                self.agenda_tracker_debug = dict(tracker_payload.get("tracker_debug") or {})

                fsm_payload = dict(control_payload.get("agenda_fsm") or {})
                self.agenda_state_map = dict(fsm_payload.get("agenda_state_map") or {})
                self.active_agenda_id = str(fsm_payload.get("active_agenda_id") or "")
                self.agenda_events = list(fsm_payload.get("agenda_events") or [])
                if self.agenda_state_map:
                    self._sync_agenda_stack_from_state_map()
                elif self.analysis:
                    self._sync_agenda_stack(self.analysis)

                dps_payload = dict(control_payload.get("dps") or {})
                self.dps_t = float(dps_payload.get("dps_t", self.dps_t or 0.0))
                self.dps_breakdown = dict(
                    dps_payload.get("dps_breakdown")
                    or self.dps_breakdown
                    or {
                        "option_coverage": 0.0,
                        "constraint_coverage": 0.0,
                        "evidence_coverage": 0.0,
                        "tradeoff_coverage": 0.0,
                        "closing_readiness": 0.0,
                        "counts": {"options": 0, "constraints": 0, "criteria": 0, "evidence": 0, "actions": 0},
                        "active_state": "ACTIVE",
                        "decision_lock": False,
                    }
                )
                self.analysis.setdefault("scores", {}).setdefault("dps", {})["score"] = self.dps_t
                self.analysis["scores"]["dps"]["why"] = (
                    str(dps_payload.get("why") or "").strip()
                    or (
                        "Option {option_coverage:.2f} | Constraint {constraint_coverage:.2f} | "
                        "Evidence {evidence_coverage:.2f} | Trade-off {tradeoff_coverage:.2f} | "
                        "Closing {closing_readiness:.2f}"
                    ).format(**self.dps_breakdown)
                )
                now_ts = time.time()
                self.dps_history = [h for h in self.dps_history if now_ts - float(h.get("ts", now_ts)) <= 180.0]
                self.dps_history.append({"ts": now_ts, "score": self.dps_t})

                flow_payload = dict(control_payload.get("flow_pulse") or {})
                self.stagnation_flag = bool(flow_payload.get("stagnation_flag", False))
                self.loop_state = str(flow_payload.get("loop_state") or "Normal")
                self.flow_pulse_debug = dict(flow_payload.get("flow_pulse_debug") or {})

                lock_payload_llm = dict(control_payload.get("decision_lock") or {})
                lock_payload = self.analysis.setdefault("intervention", {}).setdefault("decision_lock", {})
                lock_payload["triggered"] = bool(lock_payload_llm.get("triggered", False))
                lock_payload["reason"] = str(lock_payload_llm.get("reason", ""))
                self.decision_lock_debug = dict(lock_payload_llm.get("decision_lock_debug") or {})

                drift_payload = dict(control_payload.get("drift_dampener") or {})
                self.drift_state = str(drift_payload.get("drift_state") or "Normal")
                self.drift_ui_cues = dict(
                    drift_payload.get("drift_ui_cues")
                    or {"glow_k_core": False, "fix_k_core_focus": False, "reduce_facets": False, "show_banner": False}
                )
                self.drift_debug = dict(drift_payload.get("drift_debug") or {})
                self.drift_monitor = dict(drift_payload.get("monitor") or self.drift_monitor)
                payload_sparse = (not self.agenda_state_map) or (
                    len(self.agenda_candidates) == 0 and len(self.agenda_vectors) == 0
                )
                if control_plane_source != "llm" or payload_sparse:
                    used_local_fallback = True
                    local_reason = "llm_payload_sparse" if payload_sparse and control_plane_source == "llm" else control_plane_source
                    _run_local_control_pipeline(local_reason)
            else:
                used_local_fallback = True
                _run_local_control_pipeline(control_plane_reason)

            self.analysis_runtime = {
                "tick_mode": "full_context" if use_full_context else "windowed",
                "transcript_count": len(self.transcript),
                "llm_window_turns": len(llm_window),
                "engine_window_turns": len(engine_window),
                "control_plane_source": control_plane_source,
                "control_plane_reason": control_plane_reason,
                "used_local_fallback": used_local_fallback,
            }

            if self.active_agenda_id and self.active_agenda_id in self.agenda_state_map:
                active_entry = self.agenda_state_map[self.active_agenda_id]
                active_title = str(active_entry.get("title") or "").strip()
                active_state = str(active_entry.get("state") or "ACTIVE")
                if active_title:
                    self.analysis.setdefault("agenda", {}).setdefault("active", {})["title"] = active_title
                    if active_state in ("ACTIVE", "CLOSING", "CLOSED"):
                        self.analysis["agenda"]["active"]["status"] = active_state
            elif self.agenda_stack:
                self._sync_agenda_stack(self.analysis)
            self._refresh_live_artifacts()

    def _refresh_live_artifacts(self) -> None:
        analysis_payload = dict(self.analysis or {})
        for kind in ArtifactKind:
            self.artifacts[kind.value] = build_live_artifact(
                kind=kind,
                meeting_goal=self.meeting_goal,
                transcript=self.transcript[-240:],
                analysis=analysis_payload,
                agenda_state_map=self.agenda_state_map,
                active_agenda_id=self.active_agenda_id,
                evidence_status=self.evidence_status,
                evidence_snippet=self.evidence_snippet,
                evidence_log=self.evidence_log[-120:],
                dps_t=self.dps_t,
            )

    def generate_artifact(self, kind: ArtifactKind) -> None:
        with self._lock:
            self.artifacts[kind.value] = build_live_artifact(
                kind=kind,
                meeting_goal=self.meeting_goal,
                transcript=self.transcript[-240:],
                analysis=dict(self.analysis or {}),
                agenda_state_map=self.agenda_state_map,
                active_agenda_id=self.active_agenda_id,
                evidence_status=self.evidence_status,
                evidence_snippet=self.evidence_snippet,
                evidence_log=self.evidence_log[-120:],
                dps_t=self.dps_t,
            )

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

    if payload.reset_state:
        runtime.reset_keep_config()
    added = runtime.append_utterances_bulk(utterances)
    ticked = False
    if payload.auto_tick and added > 0:
        runtime.tick_analysis(use_full_context=True)
        ticked = True

    return {
        "state": runtime.snapshot(),
        "import_debug": {
            **stats,
            "added": added,
            "reset_state": payload.reset_state,
            "auto_tick": payload.auto_tick,
            "ticked": ticked,
            "analysis_mode": "full_context_once" if ticked else "none",
            "warning": "업로드한 JSON에서 전사 문장을 찾지 못했습니다." if added == 0 else "",
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

    for idx, upload in enumerate(files):
        name = (upload.filename or f"upload_{idx + 1}.json").strip() or f"upload_{idx + 1}.json"
        try:
            payload = await upload.read()
            if not payload:
                skipped_files += 1
                continue

            rows = _extract_rows_from_json_payload(_read_json_bytes(payload, label=name))
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
    added = runtime.append_utterances_bulk(utterances)
    ticked = False
    if auto_tick and added > 0:
        runtime.tick_analysis(use_full_context=True)
        ticked = True

    return {
        "state": runtime.snapshot(),
        "import_debug": {
            "folder": "uploaded_files",
            "files_scanned": len(files),
            "files_parsed": parsed_files,
            "files_skipped": skipped_files,
            "rows_loaded": len(utterances),
            "file_stats": file_stats[:20],
            "added": added,
            "reset_state": reset_state,
            "auto_tick": auto_tick,
            "ticked": ticked,
            "analysis_mode": "full_context_once" if ticked else "none",
            "warning": "업로드한 JSON에서 전사 문장을 찾지 못했습니다." if added == 0 else "",
        },
    }


@app.post("/api/analysis/tick")
def post_analysis_tick():
    runtime.tick_analysis()
    return runtime.snapshot()


@app.post("/api/artifacts/{kind}")
def post_artifact(kind: ArtifactKind):
    runtime.generate_artifact(kind)
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
