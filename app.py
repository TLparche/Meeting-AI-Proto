from __future__ import annotations

import os
import tempfile
import time
from datetime import datetime
import hashlib
import sys
import ctypes
import warnings
import queue
import threading
import logging
from html import escape
from pathlib import Path
from typing import Dict, List, Optional
import wave

import streamlit as st

from llm_client import get_client
from schemas import (
    AgendaStatus,
    AnalysisOutput,
    ArtifactKind,
    ArtifactOutput,
    TranscriptUtterance,
)

# WhisperLive + faster-whisper(ctranslate2) + torch 조합에서 OpenMP 중복 로드 충돌 방지.
# 참고: 안전한 근본 해결은 단일 OpenMP 런타임 사용이지만, 여기서는 런타임 중단을 피하기 위한 우회 설정.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings(
    "ignore",
    message=".*data discontinuity in recording.*",
    module=r"soundcard\.mediafoundation",
)


st.set_page_config(
    page_title="회의 리듬 파일럿",
    page_icon=":material/groups:",
    layout="wide",
)


LOOPBACK_SOURCE = "컴퓨터 사운드 녹음 (루프백 시뮬레이션)"


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-soft: #f4f7f8;
            --line: #d1dadc;
            --ink: #172126;
            --l1: #ffe6a3;
            --l2: #ffd1d1;
            --good: #2e7d32;
            --warn: #f57f17;
            --bad: #b71c1c;
        }
        .badge {
            display: inline-block;
            border-radius: 999px;
            padding: 0.18rem 0.62rem;
            font-size: 0.78rem;
            font-weight: 700;
            color: white;
        }
        .badge-VERIFIED { background: var(--good); }
        .badge-MIXED { background: var(--warn); color: #1f1f1f; }
        .badge-UNVERIFIED { background: #6c757d; }
        .focus-shell {
            border: 1px solid var(--line);
            border-left: 6px solid #9db4bb;
            border-radius: 12px;
            padding: 0.8rem 0.95rem;
            background: white;
        }
        .focus-l1 { border-left-color: #d19d00; background: #fff8dd; }
        .focus-l2 { border-left-color: #c62828; background: #fff0f0; }
        .small-muted { color: #57646d; font-size: 0.85rem; }
        .agenda-card {
            border: 1px solid var(--line);
            border-radius: 10px;
            background: white;
            padding: 0.5rem 0.7rem;
            margin-bottom: 0.45rem;
        }
        .signal {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 6px;
            transform: translateY(1px);
        }
        .signal-green { background: #2e7d32; }
        .signal-yellow { background: #f9a825; }
        .signal-red { background: #c62828; }
        .signal-gray { background: #8a959b; }
        /* Streamlit rerun/fade 효과를 비활성화해 모니터가 회색으로 흐려지는 현상을 줄인다. */
        [data-stale="true"] {
            opacity: 1 !important;
            filter: none !important;
        }
        .stale-element {
            opacity: 1 !important;
            filter: none !important;
        }
        [data-testid="stAppViewContainer"] * {
            transition: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_state() -> None:
    defaults = {
        "meeting_goal": "롤아웃 방식과 최종 의사결정 책임자를 정한다.",
        "initial_context": "엔지니어링/운영이 함께하는 주간 제품 회의.",
        "window_size": 12,
        "transcript": [],
        "agenda_stack": [],
        "analysis": None,
        "artifacts": {},
        "decision_lock_until": 0.0,
        "decision_lock_reason": "",
        "intervention_banner_until": 0.0,
        "intervention_banner_text": "",
        "stt_running": False,
        "stt_last_audio_sig": "",
        "stt_status_text": "중지됨",
        "stt_pipeline_state": "STOPPED",
        "stt_last_activity_at": 0.0,
        "stt_last_ingest_at": 0.0,
        "stt_last_duration_ms": 0,
        "stt_last_audio_name": "",
        "stt_last_audio_bytes": 0,
        "stt_total_ingested": 0,
        "stt_total_errors": 0,
        "stt_last_error": "",
        "stt_last_preview": "",
        "stt_events": [],
        "stt_stage": "IDLE",
        "stt_auto_loopback": True,
        "stt_loopback_chunk_seconds": 5,
        "stt_loopback_next_at": 0.0,
        "stt_loopback_speaker_id": "",
        "stt_loopback_speaker_name": "",
        "stt_min_rms": 0.002,
        "stt_last_rms": 0.0,
        "stt_rms_history": [],
        "stt_stream_activity": "UNKNOWN",
        "stt_total_captured": 0,
        "stt_backend": "WHISPERLIVE",
        "stt_whisperlive_port": 9090,
        "stt_whisperlive_model": "small",
        "stt_whisperlive_lang": "ko",
        "stt_whisperlive_connected": False,
        "stt_whisperlive_last_error": "",
        "stt_whisperlive_seen": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _append_utterance(speaker: str, text: str, timestamp: str = "") -> None:
    if not text.strip():
        return
    ts = timestamp.strip() or datetime.now().strftime("%H:%M:%S")
    utterance = TranscriptUtterance(speaker=speaker.strip() or "화자", text=text.strip(), timestamp=ts)
    st.session_state.transcript.append(utterance.model_dump())


def _sync_agenda_stack(analysis: AnalysisOutput) -> None:
    agenda_map: Dict[str, str] = {item["title"]: item["status"] for item in st.session_state.agenda_stack}
    active_title = analysis.agenda.active.title.strip()
    if active_title:
        agenda_map[active_title] = analysis.agenda.active.status
    for candidate in analysis.agenda.candidates:
        title = candidate.title.strip()
        if title and title not in agenda_map:
            agenda_map[title] = AgendaStatus.PROPOSED.value
    st.session_state.agenda_stack = [
        {"title": title, "status": status}
        for title, status in agenda_map.items()
    ]


def _run_live_analysis(client) -> None:
    transcript_window = st.session_state.transcript[-st.session_state.window_size :]
    active_agenda = ""
    if st.session_state.analysis:
        active_agenda = st.session_state.analysis.get("agenda", {}).get("active", {}).get("title", "")
    analysis = client.analyze_meeting(
        meeting_goal=st.session_state.meeting_goal,
        initial_context=st.session_state.initial_context,
        current_active_agenda=active_agenda,
        transcript_window=transcript_window,
        agenda_stack=st.session_state.agenda_stack,
    )
    st.session_state.analysis = analysis.model_dump()
    _sync_agenda_stack(analysis)

    if analysis.intervention.level == "L2" and analysis.intervention.banner_text:
        st.session_state.intervention_banner_until = time.time() + 3
        st.session_state.intervention_banner_text = analysis.intervention.banner_text

    if analysis.intervention.decision_lock.triggered:
        st.session_state.decision_lock_until = time.time() + 3
        st.session_state.decision_lock_reason = analysis.intervention.decision_lock.reason


@st.cache_resource(show_spinner=False)
def _get_whisper_model():
    try:
        import whisper
    except Exception as exc:
        raise RuntimeError(
            "openai-whisper가 설치되지 않았습니다. "
            f"현재 인터프리터: {sys.executable} | "
            f"설치 명령: {sys.executable} -m pip install openai-whisper"
        ) from exc

    model_name = "large"
    try:
        return whisper.load_model(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Whisper 모델 로드 실패(model={model_name}). "
            "최초 실행 시 모델 다운로드가 필요합니다."
        ) from exc


def _whisper_transcribe_text(audio_path: str) -> str:
    model = _get_whisper_model()
    try:
        import torch

        fp16 = bool(torch.cuda.is_available())
    except Exception:
        fp16 = False

    result = model.transcribe(
        audio_path,
        fp16=fp16,
        task="transcribe",
        language="ko",
        verbose=None,
    )
    text = (result.get("text") or "").strip()
    if text:
        return text

    # If Korean-forced pass yields empty text, retry with automatic language detection.
    fallback = model.transcribe(
        audio_path,
        fp16=fp16,
        task="transcribe",
        verbose=None,
    )
    return (fallback.get("text") or "").strip()


def _transcribe_with_whisper(uploaded_file, input_source: str) -> str:
    suffix = Path(uploaded_file.name).suffix or ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        merged = _whisper_transcribe_text(tmp_path)
        if not merged:
            return ""
        return f"[{input_source}] {merged}"
    except Exception as exc:
        raise RuntimeError(f"Whisper STT 처리 실패: {exc}") from exc
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def _transcribe_audio_path_with_whisper(audio_path: str, input_source: str) -> str:
    try:
        merged = _whisper_transcribe_text(audio_path)
        if not merged:
            return ""
        return f"[{input_source}] {merged}"
    except Exception as exc:
        raise RuntimeError(f"Whisper STT 처리 실패: {exc}") from exc


@st.cache_resource(show_spinner=False)
def _get_whisperlive_runtime(port: int, model: str, lang: str):
    try:
        from whisper_live.server import TranscriptionServer
        from whisper_live.client import Client
    except Exception as exc:
        raise RuntimeError(
            "whisper-live 모듈을 가져오지 못했습니다. "
            f"현재 인터프리터: {sys.executable} | "
            f"설치 명령: {sys.executable} -m pip install whisper-live"
        ) from exc

    logging.getLogger().setLevel(logging.WARNING)
    event_queue: queue.Queue = queue.Queue()
    server_errors: List[str] = []
    server = TranscriptionServer()

    def _on_transcription(text: str, segments: list) -> None:
        event_queue.put({"text": text, "segments": segments, "at": time.time()})

    def _run_server():
        try:
            server.run(
                host="127.0.0.1",
                port=port,
                backend="faster_whisper",
                single_model=False,
            )
        except Exception as exc:
            server_errors.append(str(exc))

    server_thread = threading.Thread(target=_run_server, daemon=True, name=f"whisperlive-server-{port}")
    server_thread.start()
    time.sleep(0.2)
    if server_errors:
        raise RuntimeError(f"whisper-live 서버 시작 실패: {server_errors[-1]}")

    client = Client(
        host="127.0.0.1",
        port=port,
        lang=lang,
        translate=False,
        model=model,
        use_vad=True,
        log_transcription=False,
        transcription_callback=_on_transcription,
        max_clients=4,
        max_connection_time=86400,
    )

    return {
        "server_thread": server_thread,
        "client": client,
        "queue": event_queue,
        "errors": server_errors,
        "port": port,
        "model": model,
        "lang": lang,
    }


def _wav_to_float32(audio_path: str):
    import numpy as np

    with wave.open(audio_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if sampwidth != 2:
        raise RuntimeError(f"지원하지 않는 PCM 샘플 폭입니다: {sampwidth}")

    pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    if n_channels > 1:
        pcm = pcm.reshape(-1, n_channels).mean(axis=1)
    return pcm / 32768.0


def _ensure_whisperlive_ready(wait_seconds: float = 0.0) -> dict:
    runtime = _get_whisperlive_runtime(
        int(st.session_state.stt_whisperlive_port),
        str(st.session_state.stt_whisperlive_model),
        str(st.session_state.stt_whisperlive_lang),
    )
    client = runtime["client"]
    deadline = time.time() + max(0.0, float(wait_seconds))
    while not bool(getattr(client, "recording", False)) and time.time() < deadline:
        if getattr(client, "server_error", False):
            break
        time.sleep(0.1)

    if getattr(client, "server_error", False):
        msg = str(getattr(client, "error_message", "whisper-live client error"))
        st.session_state.stt_whisperlive_connected = False
        st.session_state.stt_whisperlive_last_error = msg
        raise RuntimeError(msg)
    st.session_state.stt_whisperlive_connected = bool(getattr(client, "recording", False))
    return runtime


def _send_chunk_to_whisperlive(audio_path: str) -> tuple[bool, str]:
    runtime = _ensure_whisperlive_ready(wait_seconds=1.5)
    client = runtime["client"]
    if not getattr(client, "recording", False):
        st.session_state.stt_whisperlive_connected = False
        st.session_state.stt_whisperlive_last_error = ""
        return False, "whisper-live 서버 준비 중입니다. 잠시 후 다시 시도하세요."
    audio_float = _wav_to_float32(audio_path)
    client.send_packet_to_server(audio_float.astype("float32").tobytes())
    st.session_state.stt_whisperlive_connected = True
    st.session_state.stt_whisperlive_last_error = ""
    return True, ""


def _drain_whisperlive_results(stt_speaker: str) -> int:
    try:
        runtime = _ensure_whisperlive_ready()
    except Exception as exc:
        st.session_state.stt_whisperlive_last_error = str(exc)
        return 0
    q = runtime["queue"]
    seen_keys = set(st.session_state.stt_whisperlive_seen)
    newly_seen: List[str] = []
    ingested = 0

    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            break
        segments = item.get("segments") or []
        for seg in segments:
            if not seg.get("completed", False):
                continue
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            start = seg.get("start", "")
            end = seg.get("end", "")
            key = f"{start}|{end}|{text}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            newly_seen.append(key)
            _append_utterance(stt_speaker or "시스템오디오", text)
            ingested += 1

    if newly_seen:
        merged = (st.session_state.stt_whisperlive_seen + newly_seen)[-500:]
        st.session_state.stt_whisperlive_seen = merged
        st.session_state.stt_total_ingested += ingested
        st.session_state.stt_last_ingest_at = time.time()
        st.session_state.stt_pipeline_state = "INGESTED"
        st.session_state.stt_stage = "INGESTED_FROM_WHISPERLIVE"
        st.session_state.stt_last_preview = st.session_state.transcript[-1]["text"][:120]
        st.session_state.stt_last_error = ""
        _append_stt_event(f"whisper-live delivered segments={ingested}")

    return ingested


def _patch_numpy_binary_fromstring(np_module) -> None:
    # soundcard가 numpy 2.x에서 제거된 binary-mode fromstring을 호출하는 경우를 호환 처리
    if getattr(np_module, "_soundcard_fromstring_patched", False):
        return
    original = getattr(np_module, "fromstring", None)
    if original is None:
        return

    def _fromstring_compat(string, dtype=float, count=-1, sep="", **kwargs):
        if sep == "":
            # numpy 2.x: binary-mode fromstring removed. Handle bytes-like/CFFI buffers via frombuffer.
            try:
                return np_module.frombuffer(string, dtype=dtype, count=count)
            except (TypeError, ValueError):
                pass
        return original(string, dtype=dtype, count=count, sep=sep, **kwargs)

    np_module.fromstring = _fromstring_compat
    np_module._soundcard_fromstring_patched = True


def _co_initialize() -> bool:
    # Streamlit script thread may not have COM initialized (0x800401F0).
    try:
        hr = ctypes.windll.ole32.CoInitialize(None)
    except Exception:
        return False
    return hr in (0, 1)


def _co_uninitialize(initialized: bool) -> None:
    if not initialized:
        return
    try:
        ctypes.windll.ole32.CoUninitialize()
    except Exception:
        pass


def _list_loopback_speakers() -> List[dict]:
    com_initialized = _co_initialize()
    try:
        import soundcard as sc

        items = []
        for speaker in sc.all_speakers():
            items.append({"id": speaker.id, "name": speaker.name})
        return items
    except Exception:
        return []
    finally:
        _co_uninitialize(com_initialized)


def _resolve_speaker(sc_module, preferred_speaker_id: str = ""):
    if preferred_speaker_id:
        for speaker in sc_module.all_speakers():
            if speaker.id == preferred_speaker_id:
                return speaker
    return sc_module.default_speaker()


def _capture_system_audio_to_wav(
    seconds: int,
    sample_rate: int = 16000,
    preferred_speaker_id: str = "",
) -> tuple[str, str, float]:
    com_initialized = _co_initialize()
    try:
        try:
            import numpy as np
            _patch_numpy_binary_fromstring(np)
            import soundcard as sc
        except Exception as exc:
            raise RuntimeError(
                "시스템 오디오 캡처 모듈이 없습니다. "
                f"현재 인터프리터: {sys.executable} | "
                f"설치 명령: {sys.executable} -m pip install soundcard numpy"
            ) from exc

        speaker = _resolve_speaker(sc, preferred_speaker_id=preferred_speaker_id)
        if speaker is None:
            raise RuntimeError("기본 스피커를 찾지 못했습니다.")

        loopback_mic = sc.get_microphone(speaker.name, include_loopback=True)
        if loopback_mic is None:
            raise RuntimeError("loopback 입력 장치를 찾지 못했습니다. Windows 오디오 설정을 확인하세요.")

        frames = int(max(1, seconds) * sample_rate)
        with loopback_mic.recorder(samplerate=sample_rate, channels=2) as recorder:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="data discontinuity in recording",
                )
                data = recorder.record(numframes=frames)

        if data is None or len(data) == 0:
            raise RuntimeError("캡처된 오디오 데이터가 없습니다.")

        if len(data.shape) == 2:
            mono = data.mean(axis=1)
        else:
            mono = data

        # loopback 버퍼 불연속 시 NaN/Inf/비정상 대진폭이 섞일 수 있어 안정화 후 RMS를 계산한다.
        mono64 = np.asarray(mono, dtype=np.float64)
        mono64 = np.nan_to_num(mono64, nan=0.0, posinf=0.0, neginf=0.0)
        clip_ratio = float(np.mean(np.abs(mono64) > 1.25))
        if clip_ratio > 0.2:
            mono_norm = np.zeros_like(mono64, dtype=np.float32)
            rms = 0.0
        else:
            mono_norm = np.clip(mono64, -1.0, 1.0).astype(np.float32)
            rms = float(np.sqrt(np.mean(mono_norm.astype(np.float64) * mono_norm.astype(np.float64))))
            if not np.isfinite(rms):
                rms = 0.0
                mono_norm = np.zeros_like(mono_norm, dtype=np.float32)

        pcm16 = (mono_norm * 32767.0).astype(np.int16)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            wav_path = tmp.name

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())

        return wav_path, speaker.name, rms
    finally:
        _co_uninitialize(com_initialized)


def _loopback_input_ready(preferred_speaker_id: str = "") -> bool:
    com_initialized = _co_initialize()
    try:
        import numpy as np
        _patch_numpy_binary_fromstring(np)
        import soundcard as sc

        speaker = _resolve_speaker(sc, preferred_speaker_id=preferred_speaker_id)
        if speaker is None:
            return False
        return sc.get_microphone(speaker.name, include_loopback=True) is not None
    except Exception:
        return False
    finally:
        _co_uninitialize(com_initialized)


def _audio_signature(uploaded_file) -> str:
    data = uploaded_file.getvalue()
    digest = hashlib.sha1(data[:4096]).hexdigest()
    return f"{uploaded_file.name}:{len(data)}:{digest}"


def _append_stt_event(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.stt_events.append(f"{stamp} | {message}")
    st.session_state.stt_events = st.session_state.stt_events[-25:]


def _append_rms_history(rms: float) -> None:
    history = list(st.session_state.stt_rms_history)
    history.append(float(rms))
    st.session_state.stt_rms_history = history[-120:]


def _ingest_audio_if_stt_running(uploaded_audio, source: str, stt_speaker: str) -> bool:
    if not st.session_state.stt_running or not uploaded_audio:
        if not st.session_state.stt_running:
            st.session_state.stt_pipeline_state = "STOPPED"
            st.session_state.stt_stage = "STOPPED"
        else:
            st.session_state.stt_pipeline_state = "WAITING_INPUT"
            st.session_state.stt_stage = "WAITING_FOR_UPLOAD"
        return False
    sig = _audio_signature(uploaded_audio)
    if sig == st.session_state.stt_last_audio_sig:
        st.session_state.stt_pipeline_state = "LISTENING_NO_NEW_AUDIO"
        st.session_state.stt_stage = "WAITING_FOR_NEW_UPLOAD"
        return False
    st.session_state.stt_pipeline_state = "PROCESSING"
    st.session_state.stt_stage = "TRANSCRIBING_WHISPER"
    st.session_state.stt_last_activity_at = time.time()
    st.session_state.stt_last_audio_name = uploaded_audio.name
    st.session_state.stt_last_audio_bytes = len(uploaded_audio.getvalue())
    started = time.time()
    transcript_text = _transcribe_with_whisper(uploaded_audio, source)
    st.session_state.stt_last_duration_ms = int((time.time() - started) * 1000)
    st.session_state.stt_last_activity_at = time.time()
    if transcript_text:
        _append_utterance(stt_speaker, transcript_text)
        st.session_state.stt_last_audio_sig = sig
        st.session_state.stt_status_text = "실행 중 (신규 오디오 자동 전사 완료)"
        st.session_state.stt_pipeline_state = "INGESTED"
        st.session_state.stt_stage = "INGESTED_TO_FEED"
        st.session_state.stt_last_ingest_at = time.time()
        st.session_state.stt_total_ingested += 1
        st.session_state.stt_last_preview = transcript_text[:120]
        st.session_state.stt_last_error = ""
        _append_stt_event(
            f"ingested audio='{uploaded_audio.name}' bytes={st.session_state.stt_last_audio_bytes} "
            f"duration_ms={st.session_state.stt_last_duration_ms}"
        )
        return True
    st.session_state.stt_last_audio_sig = sig
    st.session_state.stt_pipeline_state = "NO_SPEECH"
    st.session_state.stt_stage = "NO_SPEECH_DETECTED"
    st.session_state.stt_last_preview = ""
    _append_stt_event(
        f"processed audio='{uploaded_audio.name}' but no speech detected "
        f"duration_ms={st.session_state.stt_last_duration_ms}"
    )
    return False


def _process_loopback_chunk(
    loop_seconds: int,
    stt_speaker: str,
    trigger: str,
    speaker_id: str = "",
    min_rms: float = 0.002,
) -> tuple[bool, str]:
    if not st.session_state.stt_running:
        return False, "STT is not running"

    audio_path = ""
    started = time.time()
    st.session_state.stt_pipeline_state = "CAPTURING"
    st.session_state.stt_stage = "CAPTURING_LOOPBACK"
    st.session_state.stt_last_activity_at = time.time()
    _append_stt_event(f"{trigger} capture started seconds={loop_seconds}")

    try:
        audio_path, speaker_name, rms = _capture_system_audio_to_wav(
            loop_seconds,
            preferred_speaker_id=speaker_id,
        )
        st.session_state.stt_total_captured += 1
        st.session_state.stt_last_rms = float(rms)
        _append_rms_history(float(rms))
        st.session_state.stt_stream_activity = "ACTIVE" if rms >= float(min_rms) else "SILENT"
        st.session_state.stt_last_audio_name = f"loopback_{loop_seconds}s.wav"
        st.session_state.stt_last_audio_bytes = Path(audio_path).stat().st_size
        st.session_state.stt_last_activity_at = time.time()
        st.session_state.stt_loopback_speaker_name = speaker_name
        _append_stt_event(
            f"{trigger} captured speaker='{speaker_name}' rms={rms:.5f} threshold={float(min_rms):.5f}"
        )

        if rms < float(min_rms):
            st.session_state.stt_last_duration_ms = int((time.time() - started) * 1000)
            st.session_state.stt_pipeline_state = "STREAM_SILENT"
            st.session_state.stt_stage = "AUDIO_BELOW_THRESHOLD"
            st.session_state.stt_last_preview = ""
            _append_stt_event(
                f"{trigger} skipped stt (silent stream) duration_ms={st.session_state.stt_last_duration_ms}"
            )
            return False, ""

        backend = str(st.session_state.stt_backend).upper()
        if backend == "WHISPERLIVE":
            st.session_state.stt_pipeline_state = "STREAMING"
            st.session_state.stt_stage = "SENDING_TO_WHISPERLIVE"
            _append_stt_event(f"{trigger} streaming to whisper-live speaker='{speaker_name}'")
            sent, _ = _send_chunk_to_whisperlive(audio_path)
            st.session_state.stt_last_duration_ms = int((time.time() - started) * 1000)
            if not sent:
                st.session_state.stt_pipeline_state = "WAITING_RESULT"
                st.session_state.stt_stage = "WHISPERLIVE_CONNECTING"
                _append_stt_event(f"{trigger} waiting whisper-live connection")
                return False, ""
            ingested_now = _drain_whisperlive_results(stt_speaker)
            if ingested_now > 0:
                return True, ""
            st.session_state.stt_pipeline_state = "WAITING_RESULT"
            st.session_state.stt_stage = "WHISPERLIVE_PENDING_RESULT"
            return False, ""

        st.session_state.stt_pipeline_state = "TRANSCRIBING"
        st.session_state.stt_stage = "TRANSCRIBING_WHISPER"
        _append_stt_event(f"{trigger} transcribing speaker='{speaker_name}'")
        transcript_text = _transcribe_audio_path_with_whisper(
            audio_path,
            f"시스템사운드:{speaker_name}",
        )

        st.session_state.stt_last_duration_ms = int((time.time() - started) * 1000)

        if transcript_text:
            _append_utterance(stt_speaker or "시스템오디오", transcript_text)
            st.session_state.stt_pipeline_state = "INGESTED"
            st.session_state.stt_stage = "INGESTED_TO_FEED"
            st.session_state.stt_last_ingest_at = time.time()
            st.session_state.stt_total_ingested += 1
            st.session_state.stt_last_preview = transcript_text[:120]
            st.session_state.stt_last_error = ""
            _append_stt_event(
                f"{trigger} ingested seconds={loop_seconds} bytes={st.session_state.stt_last_audio_bytes} "
                f"duration_ms={st.session_state.stt_last_duration_ms}"
            )
            return True, ""

        st.session_state.stt_pipeline_state = "NO_SPEECH"
        st.session_state.stt_stage = "TRANSCRIBE_EMPTY"
        st.session_state.stt_last_preview = ""
        _append_stt_event(
            f"{trigger} processed seconds={loop_seconds} transcribe empty "
            f"duration_ms={st.session_state.stt_last_duration_ms}"
        )
        return False, ""
    except Exception as exc:
        st.session_state.stt_pipeline_state = "ERROR"
        st.session_state.stt_stage = "ERROR"
        st.session_state.stt_last_activity_at = time.time()
        st.session_state.stt_total_errors += 1
        st.session_state.stt_last_error = str(exc)
        _append_stt_event(f"{trigger} error: {exc}")
        return False, str(exc)
    finally:
        if audio_path:
            Path(audio_path).unlink(missing_ok=True)


def _generate_artifact(client, kind: ArtifactKind) -> None:
    analysis_snapshot = st.session_state.analysis or {}
    transcript_window = st.session_state.transcript[-st.session_state.window_size :]
    artifact = client.generate_artifact(
        kind=kind,
        meeting_goal=st.session_state.meeting_goal,
        initial_context=st.session_state.initial_context,
        transcript_window=transcript_window,
        analysis=analysis_snapshot,
    )
    st.session_state.artifacts[kind.value] = artifact.model_dump()


def _render_rms_panel() -> None:
    threshold = float(st.session_state.stt_min_rms)
    last_rms = float(st.session_state.stt_last_rms)
    ratio = 0.0 if threshold <= 0 else min(1.0, last_rms / threshold)
    level_state = "ABOVE THRESHOLD" if last_rms >= threshold else "BELOW THRESHOLD"
    st.markdown("**Audio Level (RMS) vs Threshold**")
    st.progress(ratio, text=f"RMS {last_rms:.5f} / TH {threshold:.5f} ({level_state})")

    history = list(st.session_state.stt_rms_history)
    if history:
        tail = history[-60:]
        st.line_chart(
            {
                "RMS": tail,
                "Threshold": [threshold] * len(tail),
            },
            height=140,
        )


def _render_stt_monitor(uploaded_audio, source: str, include_rms_panel: bool = True) -> None:
    running = bool(st.session_state.stt_running)
    loopback_mode = source == LOOPBACK_SOURCE
    file_connected = uploaded_audio is not None
    selected_speaker_id = st.session_state.stt_loopback_speaker_id
    loopback_ready = _loopback_input_ready(preferred_speaker_id=selected_speaker_id) if loopback_mode else False
    has_input = file_connected or loopback_ready
    pipeline = st.session_state.stt_pipeline_state

    signal_color = "signal-gray"
    signal_text = "STOPPED"
    if st.session_state.stt_last_error:
        signal_color = "signal-red"
        signal_text = "ERROR"
    elif running and has_input and pipeline in {"PROCESSING", "CAPTURING", "TRANSCRIBING", "INGESTED"}:
        signal_color = "signal-green"
        signal_text = "ACTIVE"
    elif running and has_input:
        signal_color = "signal-yellow"
        signal_text = "READY (capture/upload needed)"
    elif running:
        signal_color = "signal-yellow"
        signal_text = "WAITING INPUT"

    st.markdown("#### STT Stream Monitor")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Engine", "RUNNING" if running else "STOPPED")
    with c2:
        if file_connected:
            input_status = "FILE CONNECTED"
        elif loopback_ready:
            input_status = "LOOPBACK READY"
        else:
            input_status = "DISCONNECTED"
        st.metric("Audio Input", input_status)
    with c3:
        st.markdown(
            f"<span class='signal {signal_color}'></span><strong>{signal_text}</strong>",
            unsafe_allow_html=True,
        )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Ingested Chunks", int(st.session_state.stt_total_ingested))
    with m2:
        st.metric("Last STT ms", int(st.session_state.stt_last_duration_ms))
    with m3:
        st.metric("Errors", int(st.session_state.stt_total_errors))

    d1, d2, d3 = st.columns(3)
    with d1:
        st.metric("Captured Chunks", int(st.session_state.stt_total_captured))
    with d2:
        st.metric("Last RMS", f"{float(st.session_state.stt_last_rms):.5f}")
    with d3:
        st.metric("Stream Activity", st.session_state.stt_stream_activity)

    if include_rms_panel:
        _render_rms_panel()

    next_eta = "--"
    if running and loopback_mode and st.session_state.stt_auto_loopback:
        remain = max(0.0, st.session_state.stt_loopback_next_at - time.time())
        next_eta = f"{remain:.1f}s"

    st.caption(
        f"Backend: {st.session_state.stt_backend} | "
        f"Pipeline: {st.session_state.stt_pipeline_state} | "
        f"Stage: {st.session_state.stt_stage} | "
        f"Auto loopback: {'ON' if st.session_state.stt_auto_loopback else 'OFF'} | "
        f"Next capture ETA: {next_eta}"
    )
    if st.session_state.stt_backend == "WHISPERLIVE":
        wl = "CONNECTED" if st.session_state.stt_whisperlive_connected else "CONNECTING"
        st.caption(
            f"WhisperLive: {wl} | model={st.session_state.stt_whisperlive_model} | "
            f"lang={st.session_state.stt_whisperlive_lang} | port={st.session_state.stt_whisperlive_port}"
        )
        if st.session_state.stt_whisperlive_last_error:
            st.caption(f"WhisperLive error: {st.session_state.stt_whisperlive_last_error}")
    if st.session_state.stt_loopback_speaker_name:
        st.caption(f"Selected loopback speaker: {st.session_state.stt_loopback_speaker_name}")

    if st.session_state.stt_last_ingest_at > 0:
        dt = datetime.fromtimestamp(st.session_state.stt_last_ingest_at).strftime("%H:%M:%S")
        st.caption(f"Last ingest: {dt}")
    if st.session_state.stt_last_audio_name:
        st.caption(
            f"Last audio: {st.session_state.stt_last_audio_name} "
            f"({st.session_state.stt_last_audio_bytes} bytes)"
        )
    if st.session_state.stt_last_preview:
        st.caption(f"Last transcript preview: {st.session_state.stt_last_preview}")
    if st.session_state.stt_last_error:
        st.error(f"Last STT error: {st.session_state.stt_last_error}")

    st.markdown("**STT Event Log (최근 12개)**")
    log_box = st.container(border=True)
    with log_box:
        if not st.session_state.stt_events:
            st.caption("No events yet.")
        else:
            for line in reversed(st.session_state.stt_events[-12:]):
                st.text(line)


if hasattr(st, "fragment"):
    @st.fragment(run_every=2.0)
    def _run_stt_runtime_fragment(uploaded_audio, source: str, loop_seconds: int, stt_speaker: str) -> None:
        loopback_mode = source == LOOPBACK_SOURCE
        if st.session_state.stt_running and loopback_mode and st.session_state.stt_auto_loopback:
            now = time.time()
            if st.session_state.stt_backend == "WHISPERLIVE":
                _drain_whisperlive_results(stt_speaker)
            if now >= st.session_state.stt_loopback_next_at:
                _, err = _process_loopback_chunk(
                    loop_seconds=loop_seconds,
                    stt_speaker=stt_speaker,
                    trigger="auto",
                    speaker_id=st.session_state.stt_loopback_speaker_id,
                    min_rms=float(st.session_state.stt_min_rms),
                )
                st.session_state.stt_loopback_next_at = time.time() + 0.2
                if err:
                    st.error(f"자동 루프백 STT 오류: {err}")
        _render_stt_monitor(uploaded_audio, source, include_rms_panel=False)

    @st.fragment(run_every=0.1)
    def _run_rms_panel_fragment() -> None:
        _render_rms_panel()
else:
    def _run_stt_runtime_fragment(uploaded_audio, source: str, loop_seconds: int, stt_speaker: str) -> None:
        loopback_mode = source == LOOPBACK_SOURCE
        if st.session_state.stt_running and loopback_mode and st.session_state.stt_auto_loopback:
            now = time.time()
            if st.session_state.stt_backend == "WHISPERLIVE":
                _drain_whisperlive_results(stt_speaker)
            if now >= st.session_state.stt_loopback_next_at:
                _, err = _process_loopback_chunk(
                    loop_seconds=loop_seconds,
                    stt_speaker=stt_speaker,
                    trigger="auto",
                    speaker_id=st.session_state.stt_loopback_speaker_id,
                    min_rms=float(st.session_state.stt_min_rms),
                )
                st.session_state.stt_loopback_next_at = time.time() + 0.2
                if err:
                    st.error(f"자동 루프백 STT 오류: {err}")
        _render_stt_monitor(uploaded_audio, source, include_rms_panel=False)

    def _run_rms_panel_fragment() -> None:
        _render_rms_panel()


def _render_agenda_stack() -> None:
    st.subheader("아젠다 스택")
    grouped = {
        AgendaStatus.PROPOSED.value: [],
        AgendaStatus.ACTIVE.value: [],
        AgendaStatus.CLOSING.value: [],
        AgendaStatus.CLOSED.value: [],
    }
    for item in st.session_state.agenda_stack:
        grouped.setdefault(item["status"], []).append(item["title"])

    for status, titles in grouped.items():
        st.markdown(f"**{status}**")
        if not titles:
            st.caption("항목 없음")
            continue
        for title in titles:
            st.markdown(f"<div class='agenda-card'>{escape(title)}</div>", unsafe_allow_html=True)


def _render_main_focus() -> None:
    st.subheader("메인 포커스")
    analysis = st.session_state.analysis
    level = "L0"
    active_title = "아직 활성 아젠다가 없습니다"
    k_core = {"object": [], "constraints": [], "criteria": []}

    if analysis:
        level = analysis["intervention"]["level"]
        active_title = analysis["agenda"]["active"]["title"] or active_title
        k_core = analysis["keywords"]["k_core"]

    class_name = "focus-shell"
    if level == "L1":
        class_name += " focus-l1"
    if level == "L2":
        class_name += " focus-l2"

    object_text = ", ".join(k_core.get("object", [])) or "없음"
    constraints_text = ", ".join(k_core.get("constraints", [])) or "없음"
    criteria_text = ", ".join(k_core.get("criteria", [])) or "없음"
    focus_html = f"""
    <div class="{class_name}">
      <div><strong>현재 아젠다:</strong> {escape(active_title)}</div>
      <div style="margin-top:0.45rem;font-weight:700;">K_core 의사결정 변수</div>
      <div class="small-muted"><strong>OBJECT:</strong> {escape(object_text)}</div>
      <div class="small-muted"><strong>CONSTRAINTS:</strong> {escape(constraints_text)}</div>
      <div class="small-muted"><strong>CRITERIA:</strong> {escape(criteria_text)}</div>
    </div>
    """
    st.markdown(focus_html, unsafe_allow_html=True)

    if time.time() < st.session_state.intervention_banner_until:
        st.warning(st.session_state.intervention_banner_text)

    st.markdown("#### 실시간 전사 피드")
    if not st.session_state.transcript:
        st.caption("아직 발화가 없습니다.")
    else:
        transcript_tail = st.session_state.transcript[-60:]
        for item in transcript_tail:
            speaker = item.get("speaker", "화자")
            timestamp = item.get("timestamp", "--:--:--")
            text = item.get("text", "")
            st.markdown(f"**{timestamp} · {speaker}:** {text}")


def _render_decision_cockpit(client) -> None:
    st.subheader("의사결정 콕핏")
    analysis = st.session_state.analysis
    if not analysis:
        st.caption("`틱/업데이트`를 실행하면 점수와 추천이 표시됩니다.")
        return

    status = analysis["evidence_gate"]["status"]
    st.markdown(
        f"근거 상태: <span class='badge badge-{status}'>{status}</span>",
        unsafe_allow_html=True,
    )

    dps = analysis["scores"]["dps"]["score"]
    drift = analysis["scores"]["drift"]["score"]
    stagnation = analysis["scores"]["stagnation"]["score"]

    st.progress(dps / 100, text=f"의사결정 진행도(DPS): {dps}")
    st.progress(drift / 100, text=f"드리프트 점수: {drift}")
    st.progress(stagnation / 100, text=f"정체 점수: {stagnation}")

    with st.expander("R1 리소스 추천", expanded=True):
        resources = analysis["recommendations"]["r1_resources"]
        if not resources:
            st.caption("추천 리소스가 없습니다.")
        for item in resources:
            st.markdown(f"- [{item['title']}]({item['url']}) - {item['reason']}")

    with st.expander("R2 옵션 비교 추천", expanded=True):
        options = analysis["recommendations"]["r2_options"]
        if not options:
            st.caption("추천 옵션이 없습니다.")
        for item in options:
            st.markdown(f"**{item['option']}**")
            st.markdown(f"- 장점: {', '.join(item.get('pros', [])) or '없음'}")
            st.markdown(f"- 리스크: {', '.join(item.get('risks', [])) or '없음'}")
            st.markdown(f"- 근거 메모: {item.get('evidence_note', '없음')}")

    lock_active = time.time() < st.session_state.decision_lock_until
    if lock_active:
        remaining = max(0.0, st.session_state.decision_lock_until - time.time())
        st.info(f"의사결정 잠금 {remaining:.1f}초 유지: {st.session_state.decision_lock_reason}")

    st.markdown("#### 의사결정 잠금 컨트롤")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("투표", use_container_width=True, disabled=lock_active):
            st.success("투표가 기록되었습니다.")
    with col_b:
        if st.button("요약", use_container_width=True, disabled=lock_active):
            _generate_artifact(client, ArtifactKind.MEETING_SUMMARY)
    with col_c:
        if st.button("종료", use_container_width=True, disabled=lock_active):
            active_title = analysis["agenda"]["active"]["title"]
            updated = []
            for item in st.session_state.agenda_stack:
                if item["title"] == active_title:
                    updated.append({"title": item["title"], "status": AgendaStatus.CLOSED.value})
                else:
                    updated.append(item)
            st.session_state.agenda_stack = updated
            st.success("활성 아젠다가 CLOSED로 이동했습니다.")

    st.markdown("#### 산출물")
    if st.button("회의 요약 생성", use_container_width=True):
        _generate_artifact(client, ArtifactKind.MEETING_SUMMARY)
    if st.button("의사결정 결과", use_container_width=True):
        _generate_artifact(client, ArtifactKind.DECISION_RESULTS)
    if st.button("액션 아이템", use_container_width=True):
        _generate_artifact(client, ArtifactKind.ACTION_ITEMS)
    if st.button("근거 로그", use_container_width=True):
        _generate_artifact(client, ArtifactKind.EVIDENCE_LOG)


def _render_artifacts_panel() -> None:
    if not st.session_state.artifacts:
        return
    st.markdown("---")
    st.subheader("생성된 결과")
    for artifact_data in st.session_state.artifacts.values():
        artifact = ArtifactOutput.model_validate(artifact_data)
        with st.expander(artifact.title, expanded=False):
            st.markdown(artifact.markdown)
            for bullet in artifact.bullets:
                st.markdown(f"- {bullet}")


def main() -> None:
    _inject_styles()
    _init_state()
    client = get_client()

    mode_text = "모의 모드 (오프라인 고정 JSON)" if client.mock_mode else "실시간 Gemini 모드"
    st.caption(mode_text)
    st.title("실시간 회의 효율 데모")

    with st.expander("회의 설정", expanded=True):
        st.session_state.meeting_goal = st.text_input("회의 목표", value=st.session_state.meeting_goal)
        st.session_state.initial_context = st.text_area(
            "초기 컨텍스트",
            value=st.session_state.initial_context,
            height=80,
        )
        st.session_state.window_size = st.slider(
            "최근 전사 윈도우 (N개 발화)",
            min_value=4,
            max_value=40,
            value=int(st.session_state.window_size),
        )

    with st.expander("입력 스트림: 수동 + Whisper STT", expanded=True):
        col_left, col_right = st.columns(2)
        with col_left:
            with st.form("manual_utterance_form", clear_on_submit=True):
                speaker = st.text_input("화자")
                text = st.text_area("전사 텍스트", height=80)
                timestamp = st.text_input("타임스탬프 (선택, HH:MM:SS)")
                submitted = st.form_submit_button("발화 추가")
            if submitted:
                _append_utterance(speaker=speaker, text=text, timestamp=timestamp)

        with col_right:
            st.markdown("**Whisper STT (라이브 모드 지원)**")
            backend_label_map = {
                "WHISPERLIVE": "WhisperLive (실시간 스트리밍)",
                "OPENAI_WHISPER": "OpenAI Whisper (청크 배치)",
            }
            backend_options = list(backend_label_map.keys())
            if st.session_state.stt_backend not in backend_options:
                st.session_state.stt_backend = "WHISPERLIVE"
            st.session_state.stt_backend = st.selectbox(
                "STT 엔진",
                options=backend_options,
                index=backend_options.index(st.session_state.stt_backend),
                format_func=lambda key: backend_label_map.get(key, key),
            )
            source = st.selectbox(
                "오디오 소스",
                ["마이크 녹음", LOOPBACK_SOURCE],
            )
            loopback_mode = source == LOOPBACK_SOURCE
            stt_speaker = st.text_input("STT 입력 화자 라벨", value="시스템오디오")
            uploaded_audio = st.file_uploader("오디오 업로드", type=["wav", "mp3", "m4a", "flac", "ogg"])
            st.caption("참고: 브라우저 앱은 PC 시스템 사운드를 직접 스트리밍하지 못할 수 있습니다.")

            if loopback_mode:
                speakers = _list_loopback_speakers()
                speaker_options = [item["id"] for item in speakers]
                speaker_name_map = {item["id"]: item["name"] for item in speakers}
                if speaker_options:
                    current_id = st.session_state.stt_loopback_speaker_id
                    if current_id not in speaker_options:
                        current_id = speaker_options[0]
                    selected_speaker_id = st.selectbox(
                        "루프백 캡처 스피커",
                        options=speaker_options,
                        index=speaker_options.index(current_id),
                        format_func=lambda sid: speaker_name_map.get(sid, sid),
                    )
                    st.session_state.stt_loopback_speaker_id = selected_speaker_id
                    st.session_state.stt_loopback_speaker_name = speaker_name_map.get(selected_speaker_id, "")
                else:
                    st.warning("루프백 가능한 스피커를 찾지 못했습니다. 오디오 장치 상태를 확인하세요.")
                    st.session_state.stt_loopback_speaker_id = ""
                    st.session_state.stt_loopback_speaker_name = ""

                st.session_state.stt_auto_loopback = st.checkbox(
                    "루프백 자동 스트리밍",
                    value=bool(st.session_state.stt_auto_loopback),
                    help="Start STT 후 PC 재생음을 연속 청크로 캡처/전사합니다.",
                )
                st.session_state.stt_min_rms = float(
                    st.slider(
                        "스트림 감지 임계치 (RMS)",
                        min_value=0.0001,
                        max_value=0.0200,
                        value=float(st.session_state.stt_min_rms),
                        step=0.0001,
                        format="%.4f",
                        help="값을 낮추면 작은 소리도 STT로 전달됩니다.",
                    )
                )
            else:
                st.caption("루프백 자동 스트리밍은 컴퓨터 사운드 소스에서만 동작합니다.")

            loop_seconds = st.slider(
                "로컬 시스템 오디오 캡처 길이(초)",
                min_value=3,
                max_value=15,
                value=int(st.session_state.stt_loopback_chunk_seconds),
            )
            st.session_state.stt_loopback_chunk_seconds = int(loop_seconds)

            stt_col1, stt_col2 = st.columns(2)
            with stt_col1:
                if st.button("Start STT", use_container_width=True, disabled=st.session_state.stt_running):
                    st.session_state.stt_running = True
                    st.session_state.stt_status_text = "실행 중"
                    st.session_state.stt_pipeline_state = "WAITING_INPUT"
                    st.session_state.stt_stage = "WAITING_FOR_INPUT"
                    st.session_state.stt_last_activity_at = time.time()
                    st.session_state.stt_loopback_next_at = time.time()
                    _append_stt_event("stt started")
                    if loopback_mode and st.session_state.stt_backend == "WHISPERLIVE":
                        try:
                            _ensure_whisperlive_ready(wait_seconds=20.0)
                            st.session_state.stt_whisperlive_last_error = ""
                            _append_stt_event("whisper-live connected")
                        except Exception as exc:
                            st.session_state.stt_running = False
                            st.session_state.stt_status_text = "중지됨"
                            st.session_state.stt_pipeline_state = "ERROR"
                            st.session_state.stt_stage = "WHISPERLIVE_INIT_FAILED"
                            st.session_state.stt_last_error = str(exc)
                            st.session_state.stt_total_errors += 1
                            _append_stt_event(f"whisper-live init error: {exc}")
                            st.error(f"whisper-live 초기화 오류: {exc}")
            with stt_col2:
                if st.button("Stop STT", use_container_width=True, disabled=not st.session_state.stt_running):
                    st.session_state.stt_running = False
                    st.session_state.stt_status_text = "중지됨"
                    st.session_state.stt_pipeline_state = "STOPPED"
                    st.session_state.stt_stage = "STOPPED"
                    st.session_state.stt_last_activity_at = time.time()
                    _append_stt_event("stt stopped")
            st.caption(f"STT 상태: {st.session_state.stt_status_text}")
            st.caption("분석 LLM 호출은 `틱 / 업데이트` 버튼에서만 수행됩니다.")

            if st.session_state.stt_running and uploaded_audio:
                try:
                    ingested = _ingest_audio_if_stt_running(uploaded_audio, source, stt_speaker)
                    if ingested:
                        st.success("신규 오디오가 자동 전사되어 전사 피드에 추가되었습니다.")
                except Exception as exc:
                    st.session_state.stt_pipeline_state = "ERROR"
                    st.session_state.stt_last_activity_at = time.time()
                    st.session_state.stt_total_errors += 1
                    st.session_state.stt_last_error = str(exc)
                    _append_stt_event(f"error: {exc}")
                    st.error(f"STT 오류: {exc}")

            loop_col1, loop_col2 = st.columns([2, 1])
            with loop_col1:
                st.caption(
                    "루프백 모드에서는 `Start STT`만 누르면 자동 캡처/전사가 반복됩니다. "
                    "아래 버튼은 1회 수동 캡처입니다."
                )
            with loop_col2:
                if st.button("시스템 사운드 캡처 + STT", use_container_width=True):
                    if not st.session_state.stt_running:
                        st.warning("먼저 `Start STT`를 눌러 STT를 시작하세요.")
                    else:
                        ingested, err = _process_loopback_chunk(
                            loop_seconds=loop_seconds,
                            stt_speaker=stt_speaker,
                            trigger="manual",
                            speaker_id=st.session_state.stt_loopback_speaker_id,
                            min_rms=float(st.session_state.stt_min_rms),
                        )
                        st.session_state.stt_loopback_next_at = time.time() + 0.2
                        if err:
                            st.error(f"시스템 오디오 캡처/STT 오류: {err}")
                        elif ingested:
                            st.success("시스템 사운드 캡처/STT가 완료되어 전사 피드에 반영되었습니다.")
                        else:
                            if st.session_state.stt_backend == "WHISPERLIVE":
                                st.info("오디오는 전송되었고, whisper-live 결과를 기다리는 중입니다.")
                            else:
                                st.warning("캡처는 되었지만 음성 구간이 감지되지 않았습니다.")

            sim_text = st.text_area(
                "컴퓨터 사운드 시뮬레이터 텍스트",
                help="실제 오디오 캡처 없이 루프백/시스템 오디오 동작을 데모할 때 사용하세요.",
            )
            if st.button("시뮬레이션된 컴퓨터 사운드 주입", use_container_width=True):
                if not st.session_state.stt_running:
                    st.warning("먼저 `Start STT`를 눌러 STT를 시작하세요.")
                else:
                    _append_utterance(stt_speaker or "시스템오디오", sim_text)
                    st.session_state.stt_pipeline_state = "INGESTED"
                    st.session_state.stt_last_activity_at = time.time()
                    st.session_state.stt_last_ingest_at = time.time()
                    st.session_state.stt_total_ingested += 1
                    st.session_state.stt_last_preview = (sim_text or "")[:120]
                    st.session_state.stt_last_error = ""
                    _append_stt_event("simulated system-audio text ingested")
                    st.success("시뮬레이션 입력이 전사 피드에 추가되었습니다.")

            if st.session_state.stt_running and loopback_mode and st.session_state.stt_auto_loopback:
                _run_stt_runtime_fragment(uploaded_audio, source, loop_seconds, stt_speaker)
                _run_rms_panel_fragment()
            else:
                _render_stt_monitor(uploaded_audio, source, include_rms_panel=True)

    if st.button("틱 / 업데이트", type="primary", use_container_width=True):
        _run_live_analysis(client)

    if st.button("전사 + 상태 초기화", use_container_width=True):
        reset_values = {
            "transcript": [],
            "agenda_stack": [],
            "analysis": None,
            "artifacts": {},
            "decision_lock_until": 0.0,
            "decision_lock_reason": "",
            "intervention_banner_until": 0.0,
            "intervention_banner_text": "",
            "stt_running": False,
            "stt_last_audio_sig": "",
            "stt_status_text": "중지됨",
            "stt_pipeline_state": "STOPPED",
            "stt_last_activity_at": 0.0,
            "stt_last_ingest_at": 0.0,
            "stt_last_duration_ms": 0,
            "stt_last_audio_name": "",
            "stt_last_audio_bytes": 0,
            "stt_total_ingested": 0,
            "stt_total_errors": 0,
            "stt_last_error": "",
            "stt_last_preview": "",
            "stt_events": [],
            "stt_stage": "IDLE",
            "stt_auto_loopback": True,
            "stt_loopback_chunk_seconds": 5,
            "stt_loopback_next_at": 0.0,
            "stt_loopback_speaker_id": "",
            "stt_loopback_speaker_name": "",
            "stt_min_rms": 0.002,
            "stt_last_rms": 0.0,
            "stt_rms_history": [],
            "stt_stream_activity": "UNKNOWN",
            "stt_total_captured": 0,
            "stt_backend": "WHISPERLIVE",
            "stt_whisperlive_port": 9090,
            "stt_whisperlive_model": "small",
            "stt_whisperlive_lang": "ko",
            "stt_whisperlive_connected": False,
            "stt_whisperlive_last_error": "",
            "stt_whisperlive_seen": [],
        }
        for key, value in reset_values.items():
            st.session_state[key] = value
        st.rerun()

    left, center, right = st.columns([1.1, 2.1, 1.4])
    with left:
        _render_agenda_stack()
    with center:
        _render_main_focus()
    with right:
        _render_decision_cockpit(client)

    _render_artifacts_panel()


if __name__ == "__main__":
    main()
