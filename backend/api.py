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

from llm_client import get_client
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

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "meeting_goal": self.meeting_goal,
                "initial_context": self.initial_context,
                "window_size": self.window_size,
                "transcript": self.transcript,
                "agenda_stack": self.agenda_stack,
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
            self._sync_agenda_stack(self.analysis)

    def generate_artifact(self, kind: ArtifactKind) -> None:
        with self._lock:
            transcript_window = self.transcript[-self.window_size :]
            artifact = self.client.generate_artifact(
                kind=kind,
                meeting_goal=self.meeting_goal,
                initial_context=self.initial_context,
                transcript_window=transcript_window,
                analysis=self.analysis or {},
            )
            self.artifacts[kind.value] = artifact.model_dump()

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
