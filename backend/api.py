from __future__ import annotations

import os
import re
import tempfile
import threading
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

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


def _now_ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


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
                "analysis": self.analysis,
                "artifacts": self.artifacts,
            }

    def update_config(self, cfg: ConfigInput) -> None:
        with self._lock:
            self.meeting_goal = cfg.meeting_goal.strip()
            self.initial_context = cfg.initial_context.strip()
            self.window_size = cfg.window_size

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

    def tick_analysis(self) -> None:
        with self._lock:
            transcript_window = self.transcript[-self.window_size :]
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
                transcript_window=transcript_window,
                agenda_stack=self.agenda_stack,
            )
            self.analysis = analysis.model_dump()
            now_tick = time.time()
            fairtalk_out = run_fairtalk_engine(
                transcript=self.transcript[-240:],
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

            evidence_out = run_evidence_gate(transcript=self.transcript[-240:])
            self.evidence_status = str(evidence_out.get("evidence_status") or "UNVERIFIED")
            self.evidence_snippet = str(evidence_out.get("evidence_snippet") or "")
            self.evidence_log = list(evidence_out.get("evidence_log") or [])[-120:]
            self.analysis.setdefault("evidence_gate", {})["status"] = self.evidence_status
            self.analysis["evidence_gate"]["claims"] = list(evidence_out.get("claims_for_schema") or [])[:8]
            recommendation_out = run_recommendation_engine(
                transcript=self.transcript[-240:],
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

            tracker_output = run_agenda_tracker(
                transcript=self.transcript[-80:],
                active_agenda=(self.analysis.get("agenda", {}).get("active", {}).get("title") or current_active),
                agenda_stack=self.agenda_stack,
                keywords=self.analysis.get("keywords", {}),
                existing_vectors=self.agenda_vectors,
            )
            self.agenda_candidates = tracker_output.get("agenda_candidates", [])
            self.agenda_vectors = tracker_output.get("agenda_vectors", {})
            self.agenda_tracker_debug = tracker_output.get("tracker_debug", {})

            # Merge tracker candidates into agenda candidate pool used by stack sync.
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
            now_ts = time.time()
            history_for_flow = [h for h in self.dps_history if now_ts - float(h.get("ts", now_ts)) <= 180.0]
            history_for_flow.append({"ts": now_ts, "score": float(dps_pre.get("score", 0.0))})
            k_core = (self.analysis.get("keywords", {}) or {}).get("k_core", {}) or {}
            flow_out = run_flow_pulse(
                transcript=self.transcript[-200:],
                k_core=k_core,
                dps_history=history_for_flow,
                now_ts=now_ts,
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
                transcript=self.transcript[-180:],
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
                transcript=self.transcript[-80:],
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
            self.dps_history = [h for h in self.dps_history if now_ts - float(h.get("ts", now_ts)) <= 180.0]
            self.dps_history.append({"ts": now_ts, "score": self.dps_t})

            if self.active_agenda_id and self.active_agenda_id in self.agenda_state_map:
                active_entry = self.agenda_state_map[self.active_agenda_id]
                active_title = str(active_entry.get("title") or "").strip()
                active_state = str(active_entry.get("state") or "ACTIVE")
                if active_title:
                    self.analysis.setdefault("agenda", {}).setdefault("active", {})["title"] = active_title
                    if active_state in ("ACTIVE", "CLOSING", "CLOSED"):
                        self.analysis["agenda"]["active"]["status"] = active_state
            elif self.agenda_stack:
                # fallback compatibility when no active agenda id exists
                self._sync_agenda_stack(self.analysis)

            active_title_for_drift = ""
            if self.active_agenda_id and self.active_agenda_id in self.agenda_state_map:
                active_title_for_drift = str(self.agenda_state_map[self.active_agenda_id].get("title") or "")
            if not active_title_for_drift:
                active_title_for_drift = str((self.analysis.get("agenda", {}).get("active", {}).get("title") or "")).strip()
            active_vec = self.agenda_vectors.get(active_title_for_drift, {})
            drift_out = run_drift_dampener(
                transcript=self.transcript[-120:],
                active_agenda_title=active_title_for_drift,
                active_agenda_vector=active_vec,
                monitor_state=self.drift_monitor,
                meeting_elapsed_sec=time.time() - self.meeting_started_at,
            )
            self.drift_state = str(drift_out.get("drift_state") or "Normal")
            self.drift_ui_cues = dict(drift_out.get("ui_cues") or {})
            self.drift_debug = dict(drift_out.get("debug") or {})
            self.drift_monitor = dict(drift_out.get("monitor") or self.drift_monitor)
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


@app.post("/api/config")
def post_config(payload: ConfigInput):
    runtime.update_config(payload)
    return runtime.snapshot()


@app.post("/api/transcript/manual")
def post_manual_utterance(payload: UtteranceInput):
    runtime.append_utterance(payload.speaker, payload.text, payload.timestamp)
    return runtime.snapshot()


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
