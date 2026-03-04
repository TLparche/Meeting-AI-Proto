from __future__ import annotations

import json
import os
import platform
import queue
import re
import tempfile
import threading
import time
import importlib.util
import copy
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
SUMMARY_POINT_TARGET_LEN = None
REALTIME_MIN_SHIFT_SPAN = 6
LLM_IO_LOG_MAX = 160
LLM_IO_PREVIEW_MAX = 6000

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
    "있는",
    "되는",
    "번째",
    "우리가",
    "굉장히",
    "아마",
    "거",
    "것",
    "수",
    "등",
    "이런식",
    "그런식",
    "해당",
    "관련된",
    "통해",
    "기반",
    "위해",
    "정리",
    "내용",
    "사항",
    "부분은",
    "부분이",
    "부분을",
    "정도는",
    "다음으로",
    "그리고요",
    "그러고",
    "아니면",
    "진행",
    "완료",
    "중인",
    "그니까",
    "보면",
    "어떻게",
    "좋은",
    "바로",
    "그러니",
    "그런데",
    "company",
    "companies",
    "thing",
    "things",
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
    for raw_tok in re.findall(r"[A-Za-z0-9가-힣]{2,}", _safe_text(text).lower()):
        tok = _normalize_keyword_token(raw_tok)
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


def _normalize_flow_type(raw: Any) -> str:
    s = _safe_text(raw, "discussion").lower()
    if s in {"discussion", "decision", "action-planning"}:
        return s
    return "discussion"


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


TITLE_NOISE_TOKENS = {
    "있는",
    "되는",
    "번째",
    "우리가",
    "굉장히",
    "아마",
    "내용",
    "부분",
    "정리",
    "사항",
    "진행",
    "완료",
    "중인",
    "관련",
    "논의",
    "이슈",
    "그니까",
    "보면",
    "어떻게",
    "좋은",
    "바로",
    "company",
    "companies",
    "thing",
    "things",
}

TITLE_TOKEN_MAP = {
    "company": "기업",
    "companies": "기업",
    "investment": "투자",
    "investments": "투자",
    "market": "시장",
    "economy": "경제",
    "policy": "정책",
    "startup": "스타트업",
    "startups": "스타트업",
}


def _normalize_keyword_token(raw_tok: str) -> str:
    tok = _safe_text(raw_tok).lower()
    if not tok:
        return ""
    tok = TITLE_TOKEN_MAP.get(tok, tok)
    # 조사/어미로 인한 파편화를 줄인다.
    for suf in ("으로", "에서", "에게", "처럼", "까지", "부터", "하고", "랑", "와", "과", "을", "를", "은", "는", "이", "가", "도", "로", "에"):
        if len(tok) > 2 and tok.endswith(suf):
            tok = tok[: -len(suf)]
            break
    return tok


def _is_title_keyword_noise(tok: str) -> bool:
    t = _normalize_keyword_token(tok)
    if not t:
        return True
    if t in STOPWORDS or t in TITLE_NOISE_TOKENS:
        return True
    if len(t) < 2:
        return True
    if re.fullmatch(r".*(하는|되는|있는|같은|보는|보면|좋은)$", t):
        return True
    if re.fullmatch(r"\d+", t):
        return True
    if re.fullmatch(r"(name|party)\d*", t):
        return True
    return False


def _usable_title_keywords(keywords: list[str] | None, meeting_goal: str) -> list[str]:
    goal_tokens = _tokens(meeting_goal)
    out: list[str] = []
    seen: set[str] = set()
    for raw in keywords or []:
        tok = _normalize_keyword_token(raw)
        if not tok or tok in seen:
            continue
        if tok in goal_tokens:
            continue
        if _is_title_keyword_noise(tok):
            continue
        seen.add(tok)
        out.append(tok)
    return out


def _is_low_quality_title(title: str, meeting_goal: str) -> bool:
    txt = _safe_text(title)
    if not txt:
        return True
    goal = _safe_text(meeting_goal)
    if goal and txt == goal:
        return True
    toks = [_normalize_keyword_token(t) for t in re.findall(r"[A-Za-z0-9가-힣]{2,}", txt.lower())]
    toks = [t for t in toks if t]
    meaningful = [t for t in toks if not _is_title_keyword_noise(t) and t not in _tokens(goal)]
    if not meaningful:
        return True
    ratio = len(meaningful) / max(1, len(toks))
    if ratio < 0.35:
        return True
    if len(txt) < 6:
        return True
    if "·" in txt or "|" in txt:
        return True
    if re.search(r"^안건\s*\d+", txt):
        return True
    if re.search(r"(관련\s*\S*\s*논의|핵심\s*쟁점|중심\s*논의|세부\s*쟁점)", txt):
        return True
    if re.search(r"\b(논의|이슈|쟁점)\s*$", txt) and len(meaningful) < 2:
        return True
    return False


def _clean_agenda_title(raw_title: Any, meeting_goal: str = "", keywords: list[str] | None = None) -> str:
    title = _safe_text(raw_title)
    title = re.sub(r"^[0-9]+[.)]\s*", "", title).strip(" -:|")
    title = re.sub(r"\s+", " ", title)
    if (not title) or _is_low_quality_title(title, meeting_goal):
        return ""
    return _safe_text(title[:80], "")


def _split_ts_prefix(line: str) -> tuple[str, str]:
    txt = _safe_text(line)
    m = re.match(r"^\[(\d{2}:\d{2}(?::\d{2})?)\]\s*(.*)$", txt)
    if m:
        return _safe_text(m.group(1)), _safe_text(m.group(2))
    return "", txt


def _to_summary_point(text: str, max_len: int | None = SUMMARY_POINT_TARGET_LEN) -> str:
    s = _safe_text(text)
    s = re.sub(r"^\[[0-9:]+\]\s*", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^(음|어|네|예|일단|그리고|근데|그니까|그러니까)\s+", "", s)
    s = re.sub(r"^(저는|제가|저희는|저희가)\s+", "", s)
    s = re.sub(r"(입니다|합니다|했어요|했습니다|같아요|같습니다)\s*$", "", s)
    s = s.strip(" .,!?:;")
    if max_len and max_len > 0 and len(s) > max_len:
        s = s[:max_len].rstrip()
    return _safe_text(s)


def _normalize_summary_item_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    for raw in lines:
        ts, body = _split_ts_prefix(raw)
        summary = _to_summary_point(body)
        if not summary:
            continue
        out.append(f"[{ts}] {summary}" if ts else summary)
    return _dedup_preserve(out, limit=20)


def _extractive_title_from_candidates(candidates: list[str], meeting_goal: str) -> str:
    cleaned = [_to_summary_point(_safe_text(c), max_len=None) for c in candidates if _safe_text(c).strip()]
    cleaned = [_safe_text(c).strip(" ,;:/") for c in cleaned if _safe_text(c).strip(" ,;:/")]
    if not cleaned:
        return ""
    cleaned = _dedup_preserve(cleaned, limit=40)

    goal_tokens = _tokens(meeting_goal)
    doc_freq: Counter[str] = Counter()
    sent_tokens: list[list[str]] = []
    for sent in cleaned:
        toks = [t for t in _keyword_tokens(sent) if t not in goal_tokens and not _is_title_keyword_noise(t)]
        uniq = list(dict.fromkeys(toks))
        sent_tokens.append(uniq)
        for tok in uniq:
            doc_freq[tok] += 1

    if not doc_freq:
        return cleaned[0]

    top_tokens = {tok for tok, _ in doc_freq.most_common(4)}
    ranked: list[tuple[float, str, list[str]]] = []
    for sent, toks in zip(cleaned, sent_tokens):
        if not toks:
            score = min(len(sent), 60) / 120.0
        else:
            coverage = sum(doc_freq[t] for t in toks)
            density = coverage / max(1, len(toks))
            top_hits = sum(1 for t in toks if t in top_tokens)
            score = density + (top_hits * 0.9) + (min(len(sent), 60) / 120.0)
        if _is_low_quality_title(sent, meeting_goal):
            score -= 1.5
        ranked.append((score, sent, toks))

    ranked.sort(key=lambda x: x[0], reverse=True)
    primary = ranked[0][1] if ranked else cleaned[0]
    primary_tokens = set(ranked[0][2]) if ranked else set()

    secondary = ""
    for _, sent, toks in ranked[1:]:
        if not sent:
            continue
        sim = _text_similarity(primary, sent)
        overlap = len(primary_tokens & set(toks))
        # 동일 문장 반복을 피하고, 다른 포인트를 한 줄에 결합하기 위한 보조 문장 선택
        if sim < 0.82 and overlap < max(2, len(primary_tokens)):
            secondary = sent
            break

    def _compact_clause(text: str, max_len: int = 36) -> str:
        s = _safe_text(text)
        s = re.sub(r"^\[[0-9:]+\]\s*", "", s)
        s = re.sub(r"\s+", " ", s).strip(" ,;:/")
        s = re.sub(r"^(그리고|또|또한|다만|하지만|근데|그래서)\s+", "", s)
        s = re.split(r"\s*(?:;|/|·)\s*", s)[0]
        s = re.split(r"\s+(?:그리고|근데|하지만|다만)\s+", s)[0]
        if len(s) > max_len:
            s = s[:max_len].rstrip()
        return _safe_text(s)

    p = _compact_clause(primary)
    s = _compact_clause(secondary) if secondary else ""

    if p and s and p != s:
        merged = f"{p}, {s}"
    else:
        merged = p or s or primary

    merged = _safe_text(merged).strip(" ,;:/")
    if _is_low_quality_title(merged, meeting_goal):
        # 최후 폴백: 빈약한 한 문장 대신 상위 핵심어를 추출해 문장형으로 보정
        top_list = [tok for tok, _ in doc_freq.most_common(3)]
        if top_list:
            merged = f"{' '.join(top_list)}에 대한 논의"
    return _safe_text(merged)


def _finalize_agenda_title(
    raw_title: Any,
    meeting_goal: str,
    keywords: list[str],
    summary_items: list[str],
    key_utterances: list[str] | None = None,
) -> str:
    # 요구사항: 안건 구간 전체를 관통하는 상위 논지를 한 문장으로 요약해 제목으로 사용한다.
    candidates: list[str] = []

    for item in summary_items or []:
        _, body = _split_ts_prefix(item)
        sentence = _to_summary_point(body, max_len=None)
        if sentence:
            candidates.append(sentence)

    for item in key_utterances or []:
        _, body = _split_ts_prefix(item)
        sentence = _to_summary_point(body, max_len=None)
        if sentence:
            candidates.append(sentence)

    raw_clean = _to_summary_point(_safe_text(raw_title), max_len=None)
    if raw_clean and not _is_low_quality_title(raw_clean, meeting_goal):
        return _safe_text(raw_clean[:80], "주요 논의 요약")

    if raw_clean:
        candidates.append(raw_clean)

    best = _extractive_title_from_candidates(candidates, meeting_goal)
    if not best:
        best = raw_clean

    return _safe_text(best.strip(), "")


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


def _looks_like_meeting_payload(payload: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "JSON 최상위가 객체(dict)가 아닙니다."
    if "utterance" not in payload:
        return False, "필수 키 `utterance`가 없습니다."
    utterance = payload.get("utterance")
    if not isinstance(utterance, list):
        return False, "`utterance`는 배열(list)이어야 합니다."
    if len(utterance) == 0:
        return False, "`utterance`가 비어 있습니다."
    return True, ""


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


class ReplayStepInput(BaseModel):
    lines: int = Field(default=1, ge=1, le=100)
    auto_analyze: bool = True


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
    last_title_refine_attempts: int = 0
    last_title_refine_success: int = 0
    last_llm_parsed_json: dict[str, Any] = field(default_factory=dict)
    last_llm_parsed_at: str = ""
    replay_rows: list[dict[str, str]] = field(default_factory=list)
    replay_index: int = 0
    replay_source: str = ""
    replay_loaded_at: str = ""
    analysis_task_seq: int = 0
    analysis_queued: int = 0
    analysis_inflight: bool = False
    analysis_last_enqueued_at: str = ""
    analysis_last_started_at: str = ""
    analysis_last_done_at: str = ""
    analysis_last_error: str = ""
    analysis_last_enqueued_id: int = 0
    analysis_last_started_id: int = 0
    analysis_last_done_id: int = 0
    analysis_generation: int = 0
    transcript_version: int = 0
    analysis_next_windowed_target: int = SUMMARY_INTERVAL
    llm_io_seq: int = 0
    llm_io_logs: list[dict[str, Any]] = field(default_factory=list)

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
        self.last_title_refine_attempts = 0
        self.last_title_refine_success = 0
        self.last_llm_parsed_json = {}
        self.last_llm_parsed_at = ""
        self.replay_rows = []
        self.replay_index = 0
        self.replay_source = ""
        self.replay_loaded_at = ""
        self.analysis_task_seq = 0
        self.analysis_queued = 0
        self.analysis_inflight = False
        self.analysis_last_enqueued_at = ""
        self.analysis_last_started_at = ""
        self.analysis_last_done_at = ""
        self.analysis_last_error = ""
        self.analysis_last_enqueued_id = 0
        self.analysis_last_started_id = 0
        self.analysis_last_done_id = 0
        self.analysis_generation += 1
        self.transcript_version = 0
        self.analysis_next_windowed_target = SUMMARY_INTERVAL
        self.llm_io_seq = 0
        self.llm_io_logs = []


RT = RuntimeStore()
ANALYSIS_QUEUE: "queue.Queue[dict[str, Any]]" = queue.Queue(maxsize=2048)
ANALYSIS_WORKER_STARTED = False


def _analysis_worker_status(rt: RuntimeStore) -> dict[str, Any]:
    observed_waiting = max(0, int(ANALYSIS_QUEUE.qsize()))
    observed_total = observed_waiting + (1 if rt.analysis_inflight else 0)
    logical_total = int(max(0, rt.analysis_queued))
    display_total = max(logical_total, observed_total)
    return {
        "inflight": bool(rt.analysis_inflight),
        "queued": int(display_total),
        "queued_logical": int(logical_total),
        "queued_observed": int(observed_total),
        "last_enqueued_id": int(rt.analysis_last_enqueued_id),
        "last_started_id": int(rt.analysis_last_started_id),
        "last_done_id": int(rt.analysis_last_done_id),
        "last_enqueued_at": _safe_text(rt.analysis_last_enqueued_at),
        "last_started_at": _safe_text(rt.analysis_last_started_at),
        "last_done_at": _safe_text(rt.analysis_last_done_at),
        "last_error": _safe_text(rt.analysis_last_error),
    }


def _truncate_text(raw: Any, limit: int = LLM_IO_PREVIEW_MAX) -> str:
    s = _safe_text(raw)
    if len(s) <= limit:
        return s
    return _safe_text(s[: max(0, limit - 1)] + "…")


def _append_llm_io_log(rt: RuntimeStore, direction: str, stage: str, payload: Any, meta: dict[str, Any] | None = None) -> None:
    rt.llm_io_seq += 1
    preview = _truncate_text(payload)
    entry = {
        "seq": int(rt.llm_io_seq),
        "at": _now_ts(),
        "direction": _safe_text(direction),
        "stage": _safe_text(stage),
        "payload": preview,
        "meta": dict(meta or {}),
    }
    rt.llm_io_logs.append(entry)
    if len(rt.llm_io_logs) > LLM_IO_LOG_MAX:
        rt.llm_io_logs = rt.llm_io_logs[-LLM_IO_LOG_MAX:]


def _call_llm_json(
    rt: RuntimeStore,
    client: Any,
    prompt: str,
    stage: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    _append_llm_io_log(
        rt,
        direction="request",
        stage=stage,
        payload=prompt,
        meta={"temperature": temperature, "max_tokens": max_tokens},
    )
    try:
        parsed = client.generate_json(prompt, temperature=temperature, max_tokens=max_tokens)
    except Exception as exc:
        _append_llm_io_log(rt, direction="error", stage=stage, payload=str(exc), meta={})
        raise
    try:
        payload = json.dumps(parsed, ensure_ascii=False)
    except Exception:
        payload = str(parsed)
    _append_llm_io_log(rt, direction="response", stage=stage, payload=payload, meta={})
    return parsed


def _snapshot_runtime_for_analysis(rt: RuntimeStore) -> RuntimeStore:
    snap = RuntimeStore()
    snap.meeting_goal = _safe_text(rt.meeting_goal)
    snap.window_size = int(rt.window_size)
    snap.transcript = [dict(row) for row in rt.transcript]
    snap.agenda_outcomes = copy.deepcopy(rt.agenda_outcomes)
    snap.llm_enabled = bool(rt.llm_enabled)
    snap.last_analyzed_count = int(rt.last_analyzed_count)
    snap.agenda_seq = int(rt.agenda_seq)
    snap.stt_chunk_seq = int(rt.stt_chunk_seq)
    snap.used_local_fallback = bool(rt.used_local_fallback)
    snap.last_analysis_warning = _safe_text(rt.last_analysis_warning)
    snap.last_tick_mode = _safe_text(rt.last_tick_mode, "windowed")
    snap.last_title_refine_attempts = int(rt.last_title_refine_attempts)
    snap.last_title_refine_success = int(rt.last_title_refine_success)
    snap.last_llm_parsed_json = copy.deepcopy(rt.last_llm_parsed_json) if isinstance(rt.last_llm_parsed_json, dict) else {}
    snap.last_llm_parsed_at = _safe_text(rt.last_llm_parsed_at)
    snap.analysis_generation = int(rt.analysis_generation)
    snap.transcript_version = int(rt.transcript_version)
    snap.llm_io_seq = int(rt.llm_io_seq)
    snap.llm_io_logs = copy.deepcopy(rt.llm_io_logs) if isinstance(rt.llm_io_logs, list) else []
    return snap


def _apply_analysis_result(rt: RuntimeStore, snap: RuntimeStore) -> None:
    rt.agenda_outcomes = copy.deepcopy(snap.agenda_outcomes)
    rt.last_analyzed_count = int(snap.last_analyzed_count)
    rt.agenda_seq = int(snap.agenda_seq)
    rt.used_local_fallback = bool(snap.used_local_fallback)
    rt.last_analysis_warning = _safe_text(snap.last_analysis_warning)
    rt.last_tick_mode = _safe_text(snap.last_tick_mode, "windowed")
    rt.last_title_refine_attempts = int(snap.last_title_refine_attempts)
    rt.last_title_refine_success = int(snap.last_title_refine_success)
    rt.last_llm_parsed_json = copy.deepcopy(snap.last_llm_parsed_json) if isinstance(snap.last_llm_parsed_json, dict) else {}
    rt.last_llm_parsed_at = _safe_text(snap.last_llm_parsed_at)
    rt.llm_io_seq = int(snap.llm_io_seq)
    rt.llm_io_logs = copy.deepcopy(snap.llm_io_logs) if isinstance(snap.llm_io_logs, list) else []


def _enqueue_analysis(
    rt: RuntimeStore,
    force: bool,
    mode: str,
    source: str = "",
    skip_interval: bool = False,
    target_count: int = 0,
) -> tuple[bool, int, str]:
    rt.analysis_task_seq += 1
    task_id = int(rt.analysis_task_seq)
    task = {
        "id": task_id,
        "force": bool(force),
        "mode": "full_document" if _safe_text(mode) == "full_document" else "windowed",
        "source": _safe_text(source),
        "enqueued_at": _now_ts(),
        "generation": int(rt.analysis_generation),
        "transcript_version": int(rt.transcript_version),
        "skip_interval": bool(skip_interval),
        "target_count": int(max(0, target_count)),
    }
    try:
        ANALYSIS_QUEUE.put_nowait(task)
    except queue.Full:
        return False, task_id, "analysis queue is full"
    rt.analysis_queued += 1
    rt.analysis_last_enqueued_id = task_id
    rt.analysis_last_enqueued_at = _safe_text(task.get("enqueued_at"))
    return True, task_id, ""


def _enqueue_windowed_with_backpressure(rt: RuntimeStore, source: str = "") -> tuple[bool, int, str, bool]:
    transcript_count = len(rt.transcript)
    if rt.analysis_next_windowed_target < SUMMARY_INTERVAL:
        rt.analysis_next_windowed_target = SUMMARY_INTERVAL

    enqueued = 0
    last_task_id = 0
    while rt.analysis_next_windowed_target <= transcript_count:
        ok, task_id, err = _enqueue_analysis(
            rt,
            force=False,
            mode="windowed",
            source=source,
            skip_interval=True,
            target_count=rt.analysis_next_windowed_target,
        )
        if not ok:
            return (enqueued > 0), int(last_task_id), _safe_text(err), False
        enqueued += 1
        last_task_id = int(task_id)
        rt.analysis_next_windowed_target += SUMMARY_INTERVAL

    if enqueued <= 0:
        delta = transcript_count - int(rt.last_analyzed_count)
        return False, 0, f"waiting interval: {delta}/{SUMMARY_INTERVAL}", True
    return True, int(last_task_id), "", False


def _analysis_worker_loop() -> None:
    while True:
        task = ANALYSIS_QUEUE.get()
        try:
            task_gen = int(task.get("generation") or 0)
            snap: RuntimeStore | None = None
            with RT.lock:
                current_gen = int(RT.analysis_generation)
                if task_gen != current_gen:
                    RT.analysis_queued = max(0, int(RT.analysis_queued) - 1)
                    continue
                RT.analysis_inflight = True
                RT.analysis_last_started_id = int(task.get("id") or 0)
                RT.analysis_last_started_at = _now_ts()
                RT.analysis_last_error = ""
                snap = _snapshot_runtime_for_analysis(RT)
                target_count = int(task.get("target_count") or 0)
                if snap is not None and target_count > 0:
                    target_count = max(1, min(target_count, len(snap.transcript)))
                    snap.transcript = list(snap.transcript[:target_count])
                    if snap.last_analyzed_count > target_count:
                        snap.last_analyzed_count = target_count
            try:
                if snap is not None:
                    _run_analysis(
                        snap,
                        force=bool(task.get("force")),
                        mode=_safe_text(task.get("mode"), "windowed"),
                        skip_interval=bool(task.get("skip_interval")),
                    )
            except Exception as exc:
                with RT.lock:
                    RT.analysis_last_error = _safe_text(exc)
                    RT.last_analysis_warning = f"analysis worker 오류: {exc}"
            finally:
                with RT.lock:
                    if task_gen == int(RT.analysis_generation) and snap is not None and not _safe_text(RT.analysis_last_error):
                        _apply_analysis_result(RT, snap)
                        rt_count = len(RT.transcript)
                        next_target = ((int(RT.last_analyzed_count) // SUMMARY_INTERVAL) + 1) * SUMMARY_INTERVAL
                        RT.analysis_next_windowed_target = max(SUMMARY_INTERVAL, min(next_target, rt_count + SUMMARY_INTERVAL))
                    RT.analysis_inflight = False
                    RT.analysis_last_done_id = int(task.get("id") or 0)
                    RT.analysis_last_done_at = _now_ts()
                    RT.analysis_queued = max(0, int(RT.analysis_queued) - 1)
        finally:
            ANALYSIS_QUEUE.task_done()


def _ensure_analysis_worker_started() -> None:
    global ANALYSIS_WORKER_STARTED
    if ANALYSIS_WORKER_STARTED:
        return
    t = threading.Thread(target=_analysis_worker_loop, daemon=True, name="analysis-worker")
    t.start()
    ANALYSIS_WORKER_STARTED = True


def _replay_status(rt: RuntimeStore) -> dict[str, Any]:
    total = len(rt.replay_rows)
    cursor = max(0, min(int(rt.replay_index), total))
    remaining = max(0, total - cursor)
    return {
        "queued_total": total,
        "queued_cursor": cursor,
        "queued_remaining": remaining,
        "done": bool(total > 0 and remaining == 0),
        "source": _safe_text(rt.replay_source),
        "loaded_at": _safe_text(rt.replay_loaded_at),
    }


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
                "key_utterances": key_utterances,
                "agenda_summary_items": summary_items,
                "summary": summary,
                "summary_references": list(row.get("summary_references") or []),
                "agenda_keywords": list(row.get("agenda_keywords") or []),
                "opinion_groups": list(row.get("opinion_groups") or []),
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
            "title_refine_attempts": int(rt.last_title_refine_attempts),
            "title_refine_success": int(rt.last_title_refine_success),
            "last_llm_json_available": bool(rt.last_llm_parsed_json),
            "last_llm_json_at": _safe_text(rt.last_llm_parsed_at),
            "analysis_worker": _analysis_worker_status(rt),
            "llm_io_count": len(rt.llm_io_logs),
        },
        "replay": _replay_status(rt),
        "llm_io_logs": list(rt.llm_io_logs[-80:]),
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
        "opinion_groups": [],
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
    title = _clean_agenda_title("", rt.meeting_goal, []) or "안건 제목 미정"
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


def _pick_key_utterances(turns: list[dict[str, Any]], keywords: list[str], max_items: int = 20) -> list[str]:
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


def _slice_turns_by_id_range(turns: list[dict[str, Any]], start_id: int, end_id: int) -> list[dict[str, Any]]:
    if not turns:
        return []
    s = int(start_id or 0)
    e = int(end_id or 0)
    if s <= 0 and e <= 0:
        return list(turns)
    if e > 0 and e < s:
        e = s
    out: list[dict[str, Any]] = []
    for t in turns:
        tid = int(t.get("turn_id") or 0)
        if tid <= 0:
            continue
        if s > 0 and tid < s:
            continue
        if e > 0 and tid > e:
            continue
        out.append(t)
    return out


def _sample_turns_for_title(seg_turns: list[dict[str, Any]], max_items: int = 140) -> list[dict[str, Any]]:
    if len(seg_turns) <= max_items:
        return list(seg_turns)
    if max_items <= 0:
        return []

    head = min(20, max_items // 4)
    tail = min(20, max_items // 4)
    mid = max(0, max_items - head - tail)
    n = len(seg_turns)

    idxs: set[int] = set()
    for i in range(head):
        idxs.add(i)
    for i in range(n - tail, n):
        if i >= 0:
            idxs.add(i)

    if mid > 0:
        span_start = head
        span_end = max(span_start, n - tail)
        span = max(1, span_end - span_start)
        for i in range(mid):
            pos = span_start + int((i / max(1, mid - 1)) * (span - 1))
            idxs.add(pos)

    ordered = sorted(idxs)
    return [seg_turns[i] for i in ordered if 0 <= i < n]


def _request_agenda_title_with_llm(
    client: Any,
    meeting_goal: str,
    turns: list[dict[str, Any]],
    start_turn_id: int,
    end_turn_id: int,
    summary_items: list[str],
    key_utterances: list[str],
    keywords: list[str],
) -> str:
    seg_turns = _slice_turns_by_id_range(turns, start_turn_id, end_turn_id)
    if not seg_turns:
        seg_turns = list(turns)
    if not seg_turns:
        return ""

    sampled = _sample_turns_for_title(seg_turns, max_items=140)
    lines: list[str] = []
    for t in sampled:
        tid = int(t.get("turn_id") or 0)
        ts = _safe_text(t.get("timestamp"), _now_ts())
        speaker = _safe_text(t.get("speaker"), "화자")
        text = _safe_text(t.get("text"))
        if not text:
            continue
        lines.append(f"- turn_id={tid} | {ts} | {speaker} | {text}")
    if not lines:
        return ""

    summary_ctx: list[str] = []
    for item in summary_items[:8]:
        _, body = _split_ts_prefix(item)
        point = _to_summary_point(body, max_len=None)
        if point:
            summary_ctx.append(f"- {point}")

    key_ctx: list[str] = []
    for item in key_utterances[:8]:
        _, body = _split_ts_prefix(item)
        point = _to_summary_point(body, max_len=None)
        if point:
            key_ctx.append(f"- {point}")

    prompt = f"""
너는 회의 안건 제목 생성기다. 출력은 JSON 객체 하나만 반환한다.

[입력]
- 회의 목표: {_safe_text(meeting_goal, "미정")}
- 안건 구간: turn_id {start_turn_id}~{end_turn_id}
- 안건 키워드: {", ".join([_safe_text(k) for k in keywords[:6]]) or "없음"}
- 안건 요약 포인트:
{chr(10).join(summary_ctx) if summary_ctx else "- 없음"}
- 안건 핵심 발언:
{chr(10).join(key_ctx) if key_ctx else "- 없음"}
- 안건 구간 발화(시간순):
{chr(10).join(lines)}

[규칙]
1) 위 안건 구간 전체를 관통하는 상위 논지를 한국어 한 문장으로 요약한다.
2) 발화 한 줄을 그대로 복사하지 않는다.
3) 단어 나열, "A · B 논의", "안건 N" 같은 형식 문구를 금지한다.
4) 자연스러운 한 문장 제목으로 작성한다.

[출력 JSON]
{{
  "agenda_title": "string"
}}
""".strip()

    try:
        parsed = _call_llm_json(
            rt=rt,
            client=client,
            prompt=prompt,
            stage="title_refine.segment",
            temperature=0.05,
            max_tokens=220,
        )
    except Exception:
        return ""

    candidate = _safe_text(parsed.get("agenda_title") or parsed.get("title"))
    candidate = _to_summary_point(candidate, max_len=None)
    candidate = _safe_text(candidate).strip(" .,!?:;/|")
    if not candidate:
        return ""
    if _is_low_quality_title(candidate, meeting_goal):
        return ""
    return _safe_text(candidate[:80], "")


def _refresh_low_quality_titles_with_llm(
    client: Any,
    rt: RuntimeStore,
    turns: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    refreshed: list[dict[str, Any]] = []
    attempts = 0
    success = 0
    for row in outcomes:
        item = dict(row)
        title = _safe_text(item.get("agenda_title"))
        if (not title) or _is_low_quality_title(title, rt.meeting_goal):
            attempts += 1
            regenerated = _request_agenda_title_with_llm(
                client=client,
                meeting_goal=rt.meeting_goal,
                turns=turns,
                start_turn_id=int(item.get("_start_turn_id") or item.get("start_turn_id") or 0),
                end_turn_id=int(item.get("_end_turn_id") or item.get("end_turn_id") or 0),
                summary_items=[_safe_text(x) for x in (item.get("_summary_items") or [])],
                key_utterances=[_safe_text(x) for x in (item.get("key_utterances") or [])],
                keywords=[_safe_text(x) for x in (item.get("agenda_keywords") or [])],
            )
            if regenerated:
                item["agenda_title"] = regenerated
                success += 1
        refreshed.append(item)
    return refreshed, attempts, success


def _compact_summary_line(text: str, max_len: int = 90) -> str:
    s = _safe_text(text)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^(음|어|네|예|일단|그러면|그럼|근데|그러니까)\s+", "", s)
    if len(s) > max_len:
        s = s[: max_len - 1].rstrip() + "…"
    return _safe_text(s)


def _enrich_outcome_summary(
    rt: RuntimeStore,
    row: dict[str, Any],
    turns: list[dict[str, Any]],
) -> dict[str, Any]:
    out = dict(row)
    start_id = int(out.get("_start_turn_id") or out.get("start_turn_id") or 0)
    end_id = int(out.get("_end_turn_id") or out.get("end_turn_id") or 0)
    seg_turns = _slice_turns_by_id_range(turns, start_id, end_id)
    if not seg_turns:
        return out

    keywords = _dedup_preserve([_safe_text(k) for k in out.get("agenda_keywords") or []], limit=6)
    if len(keywords) < 3:
        extra = _top_keywords_from_rows(seg_turns, rt.meeting_goal, limit=6)
        keywords = _dedup_preserve(keywords + extra, limit=6)
    out["agenda_keywords"] = keywords

    refs = _pick_key_refs(seg_turns, keywords, max_items=8)

    key_utterances = _dedup_preserve([_safe_text(x) for x in out.get("key_utterances") or []], limit=20)
    if len(key_utterances) < 3:
        auto_key = [f"[{_safe_text(r.get('timestamp'))}] {_safe_text(r.get('quote'))}" for r in refs[:12]]
        key_utterances = _dedup_preserve(key_utterances + auto_key, limit=20)
    out["key_utterances"] = key_utterances

    summary_items = _normalize_summary_item_lines([_safe_text(x) for x in out.get("_summary_items") or []])
    summary_refs = [dict(x) for x in (out.get("summary_references") or []) if isinstance(x, dict)]

    has_min_summary = len(summary_items) >= 2
    has_min_refs = len(summary_refs) >= 2
    if (not has_min_summary) or (not has_min_refs):
        auto_items: list[str] = []
        auto_refs: list[dict[str, Any]] = []
        for idx, ref in enumerate(refs[:12]):
            quote = _to_summary_point(_safe_text(ref.get("quote")))
            if not quote:
                continue
            ts = _safe_text(ref.get("timestamp"), _now_ts())
            auto_items.append(f"[{ts}] {quote}")
            auto_refs.append(
                {
                    "turn_id": int(ref.get("turn_id") or 0),
                    "speaker": _safe_text(ref.get("speaker"), "화자"),
                    "timestamp": ts,
                    "quote": _safe_text(ref.get("quote")),
                    "why": quote,
                }
            )
            if idx >= 9:
                break

        if not has_min_summary:
            summary_items = _dedup_preserve(summary_items + auto_items, limit=20)
        if not has_min_refs:
            summary_refs = summary_refs + auto_refs

    if not summary_refs:
        summary_refs = [_ref_from_turn(seg_turns[-1], why="요약 근거")]
    out["_summary_items"] = _normalize_summary_item_lines(summary_items)
    out["summary_references"] = summary_refs[:24]
    if not _safe_text(out.get("summary")):
        out["summary"] = " • ".join(x.split("] ", 1)[-1] for x in out["_summary_items"][:10])
    out["agenda_title"] = _finalize_agenda_title(
        out.get("agenda_title"),
        rt.meeting_goal,
        [_safe_text(k) for k in out.get("agenda_keywords") or []],
        out.get("_summary_items") or [],
        out.get("key_utterances") or [],
    )
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

        key_refs = _pick_key_refs(seg_turns, keywords, max_items=8)
        key_utterances = [f"[{_safe_text(r.get('timestamp'))}] {_safe_text(r.get('quote'))}" for r in key_refs]
        summary_refs = key_refs[:10] if key_refs else [_ref_from_turn(seg_turns[-1])]
        summary_items = [f"[{_safe_text(r.get('timestamp'))}] {_to_summary_point(_safe_text(r.get('quote')))}" for r in summary_refs]
        summary_items = _normalize_summary_item_lines(summary_items)

        seed_candidates = [t.get("text") for t in seg_turns[:6]] + [t.get("text") for t in seg_turns[-6:]]
        seed_title = _extractive_title_from_candidates([_safe_text(x) for x in seed_candidates], rt.meeting_goal)
        title = _finalize_agenda_title(seed_title, rt.meeting_goal, keywords, summary_items, key_utterances)
        if not _safe_text(title):
            title = f"안건 {seg_idx + 1}"
        if title in used_titles:
            title = f"{title} #{seg_idx + 1}"
        used_titles.add(title)

        summary = " • ".join(item.split("] ", 1)[-1] for item in summary_items[:10])
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
                "key_utterances": _dedup_preserve(key_utterances, limit=20),
                "_summary_items": _dedup_preserve(summary_items, limit=20),
                "summary_references": summary_refs,
                "summary": _safe_text(summary),
                "agenda_keywords": _dedup_preserve(keywords, limit=6),
                "opinion_groups": [],
                "decision_results": decisions,
                "action_items": actions,
                "_start_turn_id": int(seg_turns[0].get("turn_id", 1) or 1),
                "_end_turn_id": int(seg_turns[-1].get("turn_id", 1) or 1),
            }
        )

    return outcomes


def _normalize_outcome_ranges(
    outcomes: list[dict[str, Any]],
    min_turn_id: int,
    max_turn_id: int,
) -> list[dict[str, Any]]:
    cleaned = [dict(row) for row in outcomes if isinstance(row, dict)]
    if not cleaned:
        return []
    cleaned.sort(key=lambda x: int(x.get("_start_turn_id") or x.get("start_turn_id") or 10**9))

    lo = int(min_turn_id or 1)
    hi = int(max(max_turn_id, lo))
    prev_end = lo - 1

    for idx, row in enumerate(cleaned):
        start_id = int(row.get("_start_turn_id") or row.get("start_turn_id") or 0)
        end_id = int(row.get("_end_turn_id") or row.get("end_turn_id") or 0)

        if start_id <= 0:
            start_id = prev_end + 1 if prev_end >= lo else lo
        start_id = max(start_id, prev_end + 1, lo)
        start_id = min(start_id, hi)

        if end_id < start_id:
            end_id = start_id
        end_id = max(start_id, min(end_id, hi))

        row["_start_turn_id"] = start_id
        row["_end_turn_id"] = end_id
        prev_end = end_id

    for idx, row in enumerate(cleaned[:-1]):
        next_start = int(cleaned[idx + 1].get("_start_turn_id") or 0)
        start_id = int(row.get("_start_turn_id") or 0)
        end_id = int(row.get("_end_turn_id") or 0)
        if next_start > 0 and end_id >= next_start:
            row["_end_turn_id"] = max(start_id, next_start - 1)

    if cleaned:
        cleaned[0]["_start_turn_id"] = lo
        last_start = int(cleaned[-1].get("_start_turn_id") or lo)
        cleaned[-1]["_end_turn_id"] = max(last_start, hi)

    return cleaned


def _refine_outcomes_by_density(
    rt: RuntimeStore,
    outcomes: list[dict[str, Any]],
    turns: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    if not outcomes or not turns:
        return outcomes, ""

    turn_ids = [int(t.get("turn_id") or 0) for t in turns if int(t.get("turn_id") or 0) > 0]
    if not turn_ids:
        return outcomes, ""

    min_turn = min(turn_ids)
    max_turn = max(turn_ids)
    total_turns = max_turn - min_turn + 1
    if total_turns <= 0:
        return outcomes, ""

    normalized = _normalize_outcome_ranges(outcomes, min_turn, max_turn)
    if not normalized:
        return outcomes, ""

    # 대화 길이에 맞춰 최소 안건 수와 최대 안건 폭을 동적으로 조정한다.
    expected_min = 1 if total_turns < 90 else max(2, min(10, round(total_turns / 90)))
    max_span = 120 if total_turns < 220 else max(130, min(240, int(total_turns * 0.33)))
    split_min_span = 75 if total_turns < 220 else max(85, int(total_turns * 0.14))

    turn_map = {int(t.get("turn_id") or 0): t for t in turns}
    adjusted: list[dict[str, Any]] = []
    split_rows = 0

    for row in normalized:
        start_id = int(row.get("_start_turn_id") or 0)
        end_id = int(row.get("_end_turn_id") or 0)
        span = end_id - start_id + 1
        need_more = len(normalized) < expected_min
        should_split = span > max_span or (need_more and span >= split_min_span)

        if not should_split:
            adjusted.append(row)
            continue

        seg_turns = [turn_map[i] for i in range(start_id, end_id + 1) if i in turn_map]
        if len(seg_turns) < 40:
            adjusted.append(row)
            continue

        local_rows = _build_local_outcomes(rt, seg_turns)
        local_rows = _normalize_outcome_ranges(local_rows, start_id, end_id)
        if len(local_rows) <= 1:
            adjusted.append(row)
            continue

        base_state = _normalize_agenda_state(row.get("agenda_state"))
        for idx, local in enumerate(local_rows):
            merged = dict(local)
            if base_state in {"ACTIVE", "CLOSING"}:
                merged["agenda_state"] = "ACTIVE" if idx == len(local_rows) - 1 else "CLOSED"
            elif base_state == "CLOSED":
                merged["agenda_state"] = "CLOSED"
            else:
                merged["agenda_state"] = base_state
            adjusted.append(merged)
        split_rows += len(local_rows) - 1

    adjusted = _normalize_outcome_ranges(adjusted, min_turn, max_turn)
    if not adjusted:
        return normalized, ""

    if len(adjusted) < expected_min and total_turns >= 160:
        local_all = _build_local_outcomes(rt, turns)
        local_all = _normalize_outcome_ranges(local_all, min_turn, max_turn)
        if len(local_all) > len(adjusted):
            return local_all, f"LLM 안건 수가 적어 로컬 경계 보정 적용({len(local_all)}개)"

    if split_rows > 0:
        return adjusted, f"과대 안건 범위 자동 분할 적용(+{split_rows})"

    return adjusted, ""


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
        created = _create_agenda(rt, _safe_text(row.get("agenda_title"), "안건 제목 미정"), _normalize_agenda_state(row.get("agenda_state")))
        created["flow_type"] = _safe_text(row.get("flow_type"))
        created["key_utterances"] = _dedup_preserve(list(row.get("key_utterances") or []), limit=20)
        created["_summary_items"] = _dedup_preserve(list(row.get("_summary_items") or []), limit=20)
        created["summary_references"] = list(row.get("summary_references") or [])
        created["summary"] = _safe_text(row.get("summary"))
        created["agenda_keywords"] = _dedup_preserve(list(row.get("agenda_keywords") or []), limit=6)
        created["opinion_groups"] = list(row.get("opinion_groups") or [])
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


def _build_agenda_outline_prompt(rt: RuntimeStore, turns: list[dict[str, Any]], current_agenda_title: str, mode: str = "windowed") -> str:
    meeting_goal = _safe_text(rt.meeting_goal, "미정")
    turn_count = len(turns)
    agenda_hint_min = 1 if turn_count < 90 else max(2, min(10, round(turn_count / 100)))
    agenda_hint_max = max(agenda_hint_min, min(12, agenda_hint_min + 3))
    lines = []
    for turn in turns:
        lines.append(
            f"- turn_id={turn['turn_id']} | {turn['timestamp']} | {turn['speaker']} | {turn['text']}"
        )
    transcript_block = "\n".join(lines)

    return f"""
너는 회의 아젠다 구간 분할기다. 출력은 반드시 JSON 하나만 반환한다.

[입력]
- 전체 회의 목표: {meeting_goal}
- 현재 진행 안건: {current_agenda_title or "없음"}
- 분석 모드: {mode}
- 발화 목록(시간순):
{transcript_block}

[중요 규칙]
1) 안건은 "흐름 전환 시점" 기준으로 순서대로 나눈다. 즉, 주제가 전환될 때마다 새 안건을 만든다.
2) 안건 제목은 해당 안건 구간의 모든 발언을 관통하는 "상위 논지"를 한국어 한 문장으로 요약해 작성한다. 단어 나열/문장 복사는 금지한다.
3) 현재 진행 안건이 이미 있으면, 정말로 주제가 크게 바뀌었을 때만 새 ACTIVE 안건으로 둔다.
4) 각 안건은 start_turn_id/end_turn_id를 반드시 포함하고, 안건 간 구간은 시간순/비중첩으로 작성한다.
5) 분석 모드가 full_document이면, 발화 전체를 끝까지 보고 안건을 한 번에 완성한다. 중간 단계 안건 생성은 금지한다.
6) full_document에서는 총 발화 수({turn_count})를 고려해 안건 수를 동적으로 잡아라. 권장 안건 수는 {agenda_hint_min}~{agenda_hint_max}개이며, 마지막 안건만 과도하게 길어지지 않게 분할한다.
7) 이 단계에서는 상세 필드(키워드, 핵심발언, 요약, 근거, 의사결정, 액션아이템)를 생성하지 않는다.

[출력 JSON 스키마]
{{
  "active_agenda_title": "string",
  "agendas": [
    {{
      "agenda_title": "string",
      "agenda_state": "PROPOSED|ACTIVE|CLOSING|CLOSED",
      "start_turn_id": 1,
      "end_turn_id": 20,
      "flow_type": "discussion|decision|action-planning"
    }}
  ]
}}
""".strip()


def _build_agenda_detail_prompt(
    rt: RuntimeStore,
    agenda_title: str,
    agenda_state: str,
    flow_type: str,
    start_turn_id: int,
    end_turn_id: int,
    seg_turns: list[dict[str, Any]],
) -> str:
    meeting_goal = _safe_text(rt.meeting_goal, "미정")
    lines = []
    for turn in seg_turns:
        lines.append(
            f"- turn_id={turn['turn_id']} | {turn['timestamp']} | {turn['speaker']} | {turn['text']}"
        )
    transcript_block = "\n".join(lines)

    return f"""
너는 회의 안건 상세 추출기다. 출력은 반드시 JSON 하나만 반환한다.

[입력]
- 전체 회의 목표: {meeting_goal}
- 안건 제목: {_safe_text(agenda_title, "미정")}
- 안건 상태: {_safe_text(agenda_state, "PROPOSED")}
- 안건 흐름 타입: {_safe_text(flow_type, "discussion")}
- 안건 turn 범위: {start_turn_id}~{end_turn_id}
- 안건 구간 발화(시간순):
{transcript_block}

[중요 규칙]
1) 아래 출력 필드만 채운다.
2) evidence_turn_ids, key_utterance_turn_ids는 반드시 입력 turn_id만 사용한다.
3) agenda_keywords는 3~6개 핵심 용어로 작성한다.
4) key_utterance_turn_ids는 핵심 발언 turn_id를 3~10개로 선택한다.
5) agenda_summary_items는 2개 이상 작성하고, 각 항목에 evidence_turn_ids를 포함한다.
6) summary는 위 summary_items를 1~3문장으로 종합한 안건 요약이다.
7) decision_results는 확정된 결론만 포함한다. 없으면 빈 배열.
8) action_items는 누가/무엇/기한/근거를 포함한다. 없으면 빈 배열.
9) 원문 장문 인용은 금지하고, 요약 문장으로 작성한다.
10) opinion_groups를 반드시 작성한다. 안건 내 유사 의견을 묶어 2~8개 그룹으로 정리한다.
11) 각 opinion_groups 항목은 type, summary, evidence_turn_ids를 포함해야 한다.
12) type은 proposal|concern|question|agree|disagree|info 중 하나만 사용한다.

[출력 JSON 스키마]
{{
  "agenda_keywords": ["string", "string"],
  "key_utterance_turn_ids": [1,2,3],
  "agenda_summary_items": [
    {{"summary": "string", "evidence_turn_ids": [1,2]}}
  ],
  "summary": "string",
  "opinion_groups": [
    {{
      "type": "proposal|concern|question|agree|disagree|info",
      "summary": "string",
      "evidence_turn_ids": [1,2]
    }}
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
""".strip()


def _build_windowed_shift_prompt(
    rt: RuntimeStore,
    current_title: str,
    current_flow_type: str,
    current_start_turn_id: int,
    recent_turns: list[dict[str, Any]],
) -> str:
    meeting_goal = _safe_text(rt.meeting_goal, "미정")
    lines = []
    for turn in recent_turns:
        lines.append(
            f"- turn_id={turn['turn_id']} | {turn['timestamp']} | {turn['speaker']} | {turn['text']}"
        )
    transcript_block = "\n".join(lines)
    return f"""
너는 실시간 회의 안건 전환 감지기다. 출력은 JSON 하나만 반환한다.

[입력]
- 회의 목표: {meeting_goal}
- 현재 ACTIVE 안건: {_safe_text(current_title, "없음")}
- 현재 안건 흐름 타입: {_safe_text(current_flow_type, "discussion")}
- 현재 안건 시작 turn_id: {int(current_start_turn_id or 1)}
- 최근 발화:
{transcript_block}

[규칙]
1) 최근 발화가 현재 안건과 동일 흐름이면 shifted=false.
2) 주제 전환이 충분히 명확하면 shifted=true.
3) shifted=true일 때만 new_agenda_title/new_flow_type/shift_turn_id를 채운다.
4) shift_turn_id는 입력 turn_id 중 하나여야 한다.
5) new_agenda_title은 상위 논지 한 문장으로 작성한다.

[출력 JSON]
{{
  "shifted": true,
  "shift_turn_id": 120,
  "new_agenda_title": "string",
  "new_flow_type": "discussion|decision|action-planning",
  "reason": "string"
}}
""".strip()


def _extract_detail_fields_from_parsed(
    rt: RuntimeStore,
    turns: list[dict[str, Any]],
    seg_turns: list[dict[str, Any]],
    detail_parsed: dict[str, Any],
) -> dict[str, Any]:
    keywords = _dedup_preserve([_safe_text(x) for x in (detail_parsed.get("agenda_keywords") or []) if _safe_text(x)], limit=8)
    key_refs = _extract_refs(rt, _to_ids(detail_parsed.get("key_utterance_turn_ids")), turns)
    key_utterances = _dedup_preserve([f"[{r['timestamp']}] {r['quote']}" for r in key_refs], limit=8)

    summary_items: list[str] = []
    summary_references: list[dict[str, Any]] = []
    for it in detail_parsed.get("agenda_summary_items") or []:
        if not isinstance(it, dict):
            continue
        txt = _to_summary_point(_safe_text(it.get("summary")))
        if not txt:
            continue
        refs = _extract_refs(rt, _to_ids(it.get("evidence_turn_ids")), turns)
        if refs:
            summary_items.append(f"[{refs[0]['timestamp']}] {txt}")
            for ref in refs[:6]:
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
        from_keys: list[str] = []
        for line in key_utterances[:10]:
            ts, body = _split_ts_prefix(line)
            point = _to_summary_point(body)
            if not point:
                continue
            from_keys.append(f"[{ts}] {point}" if ts else point)
        summary_items = from_keys
    summary_items = _normalize_summary_item_lines(summary_items)
    if not summary_references:
        for ref in key_refs[:10]:
            summary_references.append(
                {
                    "turn_id": int(ref.get("turn_id") or 0),
                    "speaker": ref["speaker"],
                    "timestamp": ref["timestamp"],
                    "quote": ref["quote"],
                    "why": "핵심 발언",
                }
            )

    if not keywords:
        keywords = _top_keywords_from_rows(seg_turns, rt.meeting_goal, limit=6)
    if not key_utterances and seg_turns:
        key_utterances = [_format_line_from_turn(seg_turns[-1])]

    opinion_groups: list[dict[str, Any]] = []
    for it in detail_parsed.get("opinion_groups") or []:
        if not isinstance(it, dict):
            continue
        typ = _safe_text(it.get("type"), "info").lower()
        if typ not in {"proposal", "concern", "question", "agree", "disagree", "info"}:
            typ = "info"
        summary_txt = _to_summary_point(_safe_text(it.get("summary")), max_len=None)
        if not summary_txt:
            continue
        ids = _to_ids(it.get("evidence_turn_ids"))
        refs = _extract_refs(rt, ids, seg_turns)
        turn_ids = _dedup_preserve([str(int(r.get("turn_id") or 0)) for r in refs if int(r.get("turn_id") or 0) > 0], limit=12)
        evidence_ids = [int(x) for x in turn_ids if str(x).isdigit()]
        if not evidence_ids:
            evidence_ids = [tid for tid in ids if tid > 0][:12]
        opinion_groups.append(
            {
                "type": typ,
                "summary": summary_txt,
                "evidence_turn_ids": evidence_ids,
            }
        )

    decisions: list[dict[str, Any]] = []
    for it in detail_parsed.get("decision_results") or []:
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
    for it in detail_parsed.get("action_items") or []:
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

    summary = _to_summary_point(_safe_text(detail_parsed.get("summary")), max_len=None)
    if not summary:
        summary = " • ".join(x.split("] ", 1)[-1] for x in summary_items[:10])

    return {
        "agenda_keywords": _dedup_preserve(keywords, limit=6),
        "key_utterances": _dedup_preserve(key_utterances, limit=20),
        "_summary_items": _dedup_preserve(summary_items, limit=20),
        "summary_references": summary_references[:24],
        "summary": _safe_text(summary),
        "opinion_groups": opinion_groups[:12],
        "decision_results": decisions,
        "action_items": actions,
    }


def _merge_agenda_fields(target: dict[str, Any], fields: dict[str, Any]) -> None:
    target["agenda_keywords"] = _dedup_preserve(
        list(target.get("agenda_keywords") or []) + list(fields.get("agenda_keywords") or []),
        limit=6,
    )
    target["key_utterances"] = _dedup_preserve(
        list(target.get("key_utterances") or []) + list(fields.get("key_utterances") or []),
        limit=20,
    )
    target["_summary_items"] = _dedup_preserve(
        list(target.get("_summary_items") or []) + list(fields.get("_summary_items") or []),
        limit=20,
    )
    refs = [dict(x) for x in (target.get("summary_references") or []) if isinstance(x, dict)] + [
        dict(x) for x in (fields.get("summary_references") or []) if isinstance(x, dict)
    ]
    dedup_refs: list[dict[str, Any]] = []
    seen_ref: set[str] = set()
    for ref in refs:
        key = f"{int(ref.get('turn_id') or 0)}|{_safe_text(ref.get('quote'))}"
        if key in seen_ref:
            continue
        seen_ref.add(key)
        dedup_refs.append(ref)
        if len(dedup_refs) >= 24:
            break
    target["summary_references"] = dedup_refs
    if _safe_text(fields.get("summary")):
        target["summary"] = _safe_text(fields.get("summary"))

    if fields.get("opinion_groups") is not None:
        target["opinion_groups"] = list(fields.get("opinion_groups") or [])

    dec_src = list(target.get("decision_results") or []) + list(fields.get("decision_results") or [])
    dec_out: list[dict[str, Any]] = []
    seen_dec: set[str] = set()
    for d in dec_src:
        if not isinstance(d, dict):
            continue
        key = _safe_text(d.get("conclusion"))
        if not key or key in seen_dec:
            continue
        seen_dec.add(key)
        dec_out.append(d)
    target["decision_results"] = dec_out

    act_src = list(target.get("action_items") or []) + list(fields.get("action_items") or [])
    act_out: list[dict[str, Any]] = []
    seen_act: set[str] = set()
    for a in act_src:
        if not isinstance(a, dict):
            continue
        key = f"{_safe_text(a.get('item'))}|{_safe_text(a.get('owner'))}|{_safe_text(a.get('due'))}"
        if not _safe_text(a.get("item")) or key in seen_act:
            continue
        seen_act.add(key)
        act_out.append(a)
    target["action_items"] = act_out


def _run_realtime_window_analysis(rt: RuntimeStore, client: Any) -> bool:
    if not rt.transcript:
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
    max_turn = len(turns)
    if max_turn <= 0:
        return False

    active = _active_agenda(rt.agenda_outcomes)
    if active is None:
        seed = _extractive_title_from_candidates([_safe_text(t.get("text")) for t in turns[-8:]], rt.meeting_goal)
        active = _create_agenda(rt, _safe_text(seed, "안건 진행"), "ACTIVE")
        active["start_turn_id"] = max(1, max_turn - min(7, max_turn - 1))
        active["end_turn_id"] = max_turn
        active["flow_type"] = "discussion"

    active_start = int(active.get("start_turn_id") or 1)
    active_end = int(active.get("end_turn_id") or active_start)
    if active_end < active_start:
        active_end = active_start
    active["start_turn_id"] = active_start
    active["end_turn_id"] = max(active_end, max_turn)

    recent_window = max(40, min(160, rt.window_size * 10))
    recent_turns = turns[max(0, len(turns) - recent_window) :]
    shift_prompt = _build_windowed_shift_prompt(
        rt=rt,
        current_title=_safe_text(active.get("agenda_title")),
        current_flow_type=_normalize_flow_type(active.get("flow_type")),
        current_start_turn_id=active_start,
        recent_turns=recent_turns,
    )
    try:
        shift_parsed = _call_llm_json(
            rt=rt,
            client=client,
            prompt=shift_prompt,
            stage="realtime.shift",
            temperature=0.05,
            max_tokens=700,
        )
    except Exception as exc:
        return _run_local_fallback(rt, force=False, reason=f"실시간 안건 전환 감지 실패: {exc}", mode="windowed")

    shifted = _boolify(shift_parsed.get("shifted"), False)
    shift_turn_id = int(shift_parsed.get("shift_turn_id") or 0)
    recent_ids = {int(t.get("turn_id") or 0) for t in recent_turns}
    if shift_turn_id not in recent_ids:
        shift_turn_id = max_turn
    if shift_turn_id <= active_start:
        shifted = False
    shift_guard_reason = ""
    active_title = _safe_text(active.get("agenda_title"))
    candidate_title = _safe_text(shift_parsed.get("new_agenda_title"))
    active_span = max(0, max_turn - active_start + 1)
    if shifted and not candidate_title:
        shifted = False
        shift_guard_reason = "전환 차단: 새 안건 제목 비어 있음"
    if shifted and (not _topic_far_enough(active_title, candidate_title)):
        shifted = False
        shift_guard_reason = "전환 차단: 현재 안건과 제목 유사"
    if shifted and active_span < REALTIME_MIN_SHIFT_SPAN:
        shifted = False
        shift_guard_reason = f"전환 차단: 안건 길이 {active_span}turn < {REALTIME_MIN_SHIFT_SPAN}turn"

    title_refine_attempts = 0
    title_refine_success = 0

    if shifted:
        prev_end = max(active_start, min(max_turn, shift_turn_id - 1))
        active["end_turn_id"] = prev_end
        active["agenda_state"] = "CLOSED"

        prev_turns = _slice_turns_by_id_range(turns, active_start, prev_end)
        prev_detail: dict[str, Any] = {}
        if prev_turns:
            try:
                prev_prompt = _build_agenda_detail_prompt(
                    rt=rt,
                    agenda_title=_safe_text(active.get("agenda_title")),
                    agenda_state="CLOSED",
                    flow_type=_normalize_flow_type(active.get("flow_type")),
                    start_turn_id=active_start,
                    end_turn_id=prev_end,
                    seg_turns=prev_turns,
                )
                prev_detail = _call_llm_json(
                    rt=rt,
                    client=client,
                    prompt=prev_prompt,
                    stage="realtime.prev_detail",
                    temperature=0.1,
                    max_tokens=2200,
                )
            except Exception:
                prev_detail = {}
        prev_fields = _extract_detail_fields_from_parsed(rt, turns, prev_turns or recent_turns, prev_detail or {})
        _merge_agenda_fields(active, prev_fields)

        title = _safe_text(active.get("agenda_title"))
        if (not title) or _is_low_quality_title(title, rt.meeting_goal):
            title_refine_attempts += 1
            regenerated = _request_agenda_title_with_llm(
                client=client,
                meeting_goal=rt.meeting_goal,
                turns=turns,
                start_turn_id=active_start,
                end_turn_id=prev_end,
                summary_items=list(active.get("_summary_items") or []),
                key_utterances=list(active.get("key_utterances") or []),
                keywords=list(active.get("agenda_keywords") or []),
            )
            if regenerated:
                active["agenda_title"] = regenerated
                title_refine_success += 1

        new_title = _safe_text(shift_parsed.get("new_agenda_title"))
        new_flow = _normalize_flow_type(shift_parsed.get("new_flow_type"))
        new_row = _create_agenda(rt, _safe_text(new_title, "새 안건"), "ACTIVE")
        new_row["flow_type"] = new_flow
        new_row["start_turn_id"] = shift_turn_id
        new_row["end_turn_id"] = max_turn

        new_turns = _slice_turns_by_id_range(turns, shift_turn_id, max_turn)
        new_detail: dict[str, Any] = {}
        if new_turns:
            try:
                new_prompt = _build_agenda_detail_prompt(
                    rt=rt,
                    agenda_title=_safe_text(new_row.get("agenda_title")),
                    agenda_state="ACTIVE",
                    flow_type=new_flow,
                    start_turn_id=shift_turn_id,
                    end_turn_id=max_turn,
                    seg_turns=new_turns,
                )
                new_detail = _call_llm_json(
                    rt=rt,
                    client=client,
                    prompt=new_prompt,
                    stage="realtime.new_detail",
                    temperature=0.1,
                    max_tokens=2200,
                )
            except Exception:
                new_detail = {}
        new_fields = _extract_detail_fields_from_parsed(rt, turns, new_turns or recent_turns, new_detail or {})
        _merge_agenda_fields(new_row, new_fields)

        if (not _safe_text(new_row.get("agenda_title"))) or _is_low_quality_title(_safe_text(new_row.get("agenda_title")), rt.meeting_goal):
            title_refine_attempts += 1
            regenerated = _request_agenda_title_with_llm(
                client=client,
                meeting_goal=rt.meeting_goal,
                turns=turns,
                start_turn_id=shift_turn_id,
                end_turn_id=max_turn,
                summary_items=list(new_row.get("_summary_items") or []),
                key_utterances=list(new_row.get("key_utterances") or []),
                keywords=list(new_row.get("agenda_keywords") or []),
            )
            if regenerated:
                new_row["agenda_title"] = regenerated
                title_refine_success += 1
    else:
        active["agenda_state"] = "ACTIVE"
        active["end_turn_id"] = max_turn
        seg_start = max(active_start, max_turn - recent_window + 1)
        seg_turns = _slice_turns_by_id_range(turns, seg_start, max_turn)
        detail_parsed: dict[str, Any] = {}
        if seg_turns:
            try:
                prompt = _build_agenda_detail_prompt(
                    rt=rt,
                    agenda_title=_safe_text(active.get("agenda_title")),
                    agenda_state="ACTIVE",
                    flow_type=_normalize_flow_type(active.get("flow_type")),
                    start_turn_id=seg_start,
                    end_turn_id=max_turn,
                    seg_turns=seg_turns,
                )
                detail_parsed = _call_llm_json(
                    rt=rt,
                    client=client,
                    prompt=prompt,
                    stage="realtime.active_detail",
                    temperature=0.1,
                    max_tokens=2000,
                )
            except Exception:
                detail_parsed = {}
        fields = _extract_detail_fields_from_parsed(rt, turns, seg_turns or recent_turns, detail_parsed or {})
        _merge_agenda_fields(active, fields)

    rt.last_analyzed_count = len(rt.transcript)
    rt.used_local_fallback = False
    rt.last_tick_mode = "windowed"
    rt.last_title_refine_attempts = int(title_refine_attempts)
    rt.last_title_refine_success = int(title_refine_success)
    warn = (
        f"실시간 모드: {'안건 전환 감지' if shifted else '현재 안건 유지'} | "
        f"안건 제목 재요청 {title_refine_success}/{title_refine_attempts} 성공"
    )
    if shift_guard_reason:
        warn = f"{warn} | {shift_guard_reason}"
    rt.last_analysis_warning = warn
    rt.last_llm_parsed_json = {
        "pipeline": "windowed_realtime",
        "shift": shift_parsed,
        "agenda_count": len(rt.agenda_outcomes),
        "active_agenda_title": _safe_text((_active_agenda(rt.agenda_outcomes) or {}).get("agenda_title")),
    }
    rt.last_llm_parsed_at = _now_ts()
    return True


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
    rt.last_title_refine_attempts = 0
    rt.last_title_refine_success = 0
    return True


def _run_analysis(rt: RuntimeStore, force: bool = False, mode: str = "windowed", skip_interval: bool = False) -> bool:
    if not rt.transcript:
        rt.used_local_fallback = True
        rt.last_analysis_warning = "전사 데이터가 없어 분석할 수 없습니다."
        rt.last_tick_mode = "full_document" if mode == "full_document" else "windowed"
        rt.last_title_refine_attempts = 0
        rt.last_title_refine_success = 0
        return False
    if mode != "full_document" and (not force) and (not skip_interval) and (len(rt.transcript) - rt.last_analyzed_count) < SUMMARY_INTERVAL:
        return False
    if not rt.llm_enabled:
        return _run_local_fallback(rt, force=force, reason="LLM 미연결", mode=mode)

    client = get_client()
    if not client.connected:
        return _run_local_fallback(rt, force=force, reason="LLM 연결 끊김", mode=mode)

    if mode == "windowed" and not force:
        return _run_realtime_window_analysis(rt, client)

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

    # 1단계: 전체 전사 기준 안건 구간(제목/상태/흐름)만 먼저 추출
    active = _active_agenda(rt.agenda_outcomes)
    current_title = _safe_text((active or {}).get("agenda_title"))
    outline_prompt = _build_agenda_outline_prompt(rt, turns, current_title, mode=mode)
    try:
        outline_parsed = _call_llm_json(
            rt=rt,
            client=client,
            prompt=outline_prompt,
            stage="full.outline",
            temperature=0.1,
            max_tokens=2800,
        )
    except Exception as exc:
        return _run_local_fallback(rt, force=force, reason=f"LLM 1차(안건 구간) 오류: {exc}", mode=mode)

    raw_agendas = outline_parsed.get("agendas") or []
    if not isinstance(raw_agendas, list) or not raw_agendas:
        return _run_local_fallback(rt, force=force, reason="LLM 1차 응답에서 agendas가 비어 로컬 폴백 사용", mode=mode)

    turn_ids = [int(t.get("turn_id") or 0) for t in turns if int(t.get("turn_id") or 0) > 0]
    if not turn_ids:
        return _run_local_fallback(rt, force=force, reason="안건 구간 계산용 turn_id 없음", mode=mode)
    min_turn = min(turn_ids)
    max_turn = max(turn_ids)

    outline_rows: list[dict[str, Any]] = []
    for idx, agenda in enumerate(raw_agendas):
        if not isinstance(agenda, dict):
            continue
        row = {
            "agenda_title": _safe_text(agenda.get("agenda_title")),
            "agenda_state": _normalize_agenda_state(agenda.get("agenda_state")),
            "flow_type": _safe_text(agenda.get("flow_type"), "discussion"),
            "_start_turn_id": int(agenda.get("start_turn_id") or 0),
            "_end_turn_id": int(agenda.get("end_turn_id") or 0),
        }
        if row["_start_turn_id"] <= 0:
            row["_start_turn_id"] = min_turn + idx
        if row["_end_turn_id"] < row["_start_turn_id"]:
            row["_end_turn_id"] = row["_start_turn_id"]
        outline_rows.append(row)

    if not outline_rows:
        return _run_local_fallback(rt, force=force, reason="LLM 1차 안건 파싱 실패", mode=mode)

    outline_rows = _normalize_outcome_ranges(outline_rows, min_turn, max_turn)
    outline_rows, refine_note = _refine_outcomes_by_density(rt, outline_rows, turns)
    if not outline_rows:
        return _run_local_fallback(rt, force=force, reason="1차 안건 구간 보정 실패", mode=mode)

    # 2단계: 안건 구간별 상세 필드 개별 요청
    active_title = _safe_text(outline_parsed.get("active_agenda_title"))
    active_title_norm = active_title.strip().lower()
    outcomes: list[dict[str, Any]] = []
    title_refine_attempts = 0
    title_refine_success = 0
    detail_attempts = 0
    detail_success = 0
    detail_logs: list[dict[str, Any]] = []

    for idx, agenda in enumerate(outline_rows):
        start_turn_id = int(agenda.get("_start_turn_id") or agenda.get("start_turn_id") or 0)
        end_turn_id = int(agenda.get("_end_turn_id") or agenda.get("end_turn_id") or 0)
        seg_turns = _slice_turns_by_id_range(turns, start_turn_id, end_turn_id)
        if not seg_turns:
            seg_turns = list(turns)

        raw_title = _safe_text(agenda.get("agenda_title"))
        state = _normalize_agenda_state(agenda.get("agenda_state"))
        flow_type = _safe_text(agenda.get("flow_type"), "discussion")

        detail_attempts += 1
        detail_parsed: dict[str, Any] = {}
        detail_error = ""
        try:
            detail_prompt = _build_agenda_detail_prompt(
                rt=rt,
                agenda_title=raw_title,
                agenda_state=state,
                flow_type=flow_type,
                start_turn_id=start_turn_id,
                end_turn_id=end_turn_id,
                seg_turns=seg_turns,
            )
            detail_parsed = _call_llm_json(
                rt=rt,
                client=client,
                prompt=detail_prompt,
                stage=f"full.detail.{idx + 1}",
                temperature=0.1,
                max_tokens=3200,
            )
            detail_success += 1
        except Exception as exc:
            detail_error = str(exc)
            detail_parsed = {}

        detail_logs.append(
            {
                "agenda_index": idx + 1,
                "start_turn_id": start_turn_id,
                "end_turn_id": end_turn_id,
                "title_seed": raw_title,
                "error": detail_error,
                "response": detail_parsed,
            }
        )

        keywords = _dedup_preserve([_safe_text(x) for x in (detail_parsed.get("agenda_keywords") or []) if _safe_text(x)], limit=8)
        key_refs = _extract_refs(rt, _to_ids(detail_parsed.get("key_utterance_turn_ids")), turns)
        key_utterances = _dedup_preserve([f"[{r['timestamp']}] {r['quote']}" for r in key_refs], limit=8)

        summary_items: list[str] = []
        summary_references: list[dict[str, Any]] = []
        for it in detail_parsed.get("agenda_summary_items") or []:
            if not isinstance(it, dict):
                continue
            txt = _to_summary_point(_safe_text(it.get("summary")))
            if not txt:
                continue
            refs = _extract_refs(rt, _to_ids(it.get("evidence_turn_ids")), turns)
            if refs:
                summary_items.append(f"[{refs[0]['timestamp']}] {txt}")
                for ref in refs[:6]:
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
            from_keys: list[str] = []
            for line in key_utterances[:10]:
                ts, body = _split_ts_prefix(line)
                point = _to_summary_point(body)
                if not point:
                    continue
                from_keys.append(f"[{ts}] {point}" if ts else point)
            summary_items = from_keys
        summary_items = _normalize_summary_item_lines(summary_items)
        if not summary_references:
            for ref in key_refs[:10]:
                summary_references.append(
                    {
                        "turn_id": 0,
                        "speaker": ref["speaker"],
                        "timestamp": ref["timestamp"],
                        "quote": ref["quote"],
                        "why": "핵심 발언",
                    }
                )

        opinion_groups: list[dict[str, Any]] = []
        for it in detail_parsed.get("opinion_groups") or []:
            if not isinstance(it, dict):
                continue
            typ = _safe_text(it.get("type"), "info").lower()
            if typ not in {"proposal", "concern", "question", "agree", "disagree", "info"}:
                typ = "info"
            summary_txt = _to_summary_point(_safe_text(it.get("summary")), max_len=None)
            if not summary_txt:
                continue
            ids = _to_ids(it.get("evidence_turn_ids"))
            refs = _extract_refs(rt, ids, seg_turns)
            turn_ids = _dedup_preserve([str(int(r.get("turn_id") or 0)) for r in refs if int(r.get("turn_id") or 0) > 0], limit=12)
            evidence_ids = [int(x) for x in turn_ids if str(x).isdigit()]
            if not evidence_ids:
                evidence_ids = [tid for tid in ids if tid > 0][:12]
            opinion_groups.append(
                {
                    "type": typ,
                    "summary": summary_txt,
                    "evidence_turn_ids": evidence_ids,
                }
            )

        decisions: list[dict[str, Any]] = []
        for it in detail_parsed.get("decision_results") or []:
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
        for it in detail_parsed.get("action_items") or []:
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
            keywords = _top_keywords_from_rows(seg_turns, rt.meeting_goal, limit=6)
        if not key_utterances and turns:
            pick_idx = min(len(turns) - 1, idx * max(1, len(turns) // max(1, len(outline_rows))))
            key_utterances = [_format_line_from_turn(turns[pick_idx])]

        all_ids = _to_ids(detail_parsed.get("key_utterance_turn_ids"))
        for s_item in detail_parsed.get("agenda_summary_items") or []:
            if isinstance(s_item, dict):
                all_ids.extend(_to_ids(s_item.get("evidence_turn_ids")))
        if start_turn_id <= 0:
            start_turn_id = min(all_ids) if all_ids else (idx + 1) * 1000
        if end_turn_id < start_turn_id:
            end_turn_id = max(all_ids) if all_ids else start_turn_id

        need_title_refine = (not _safe_text(raw_title)) or _is_low_quality_title(raw_title, rt.meeting_goal)
        if need_title_refine:
            title_refine_attempts += 1
            regenerated = _request_agenda_title_with_llm(
                client=client,
                meeting_goal=rt.meeting_goal,
                turns=turns,
                start_turn_id=start_turn_id,
                end_turn_id=end_turn_id,
                summary_items=summary_items,
                key_utterances=key_utterances,
                keywords=keywords,
            )
            if regenerated:
                raw_title = regenerated
                title_refine_success += 1

        title = _finalize_agenda_title(raw_title, rt.meeting_goal, keywords, summary_items, key_utterances)
        if (not _safe_text(title)) or _is_low_quality_title(title, rt.meeting_goal):
            title_refine_attempts += 1
            regenerated = _request_agenda_title_with_llm(
                client=client,
                meeting_goal=rt.meeting_goal,
                turns=turns,
                start_turn_id=start_turn_id,
                end_turn_id=end_turn_id,
                summary_items=summary_items,
                key_utterances=key_utterances,
                keywords=keywords,
            )
            if regenerated:
                title = regenerated
                title_refine_success += 1

        direct_match = active_title_norm and title.strip().lower() == active_title_norm
        sim_match = active_title and _text_similarity(active_title, title) >= 0.55
        if direct_match or sim_match:
            state = "ACTIVE"

        summary = _to_summary_point(_safe_text(detail_parsed.get("summary")), max_len=None)
        if not summary:
            summary = " • ".join(x.split("] ", 1)[-1] for x in summary_items[:10])

        outcomes.append(
            {
                "agenda_title": title,
                "agenda_state": state,
                "flow_type": flow_type,
                "key_utterances": _dedup_preserve(key_utterances, limit=20),
                "_summary_items": _dedup_preserve(summary_items, limit=20),
                "summary_references": summary_references[:24],
                "summary": _safe_text(summary),
                "agenda_keywords": _dedup_preserve(keywords, limit=6),
                "opinion_groups": opinion_groups[:12],
                "decision_results": decisions,
                "action_items": actions,
                "_start_turn_id": start_turn_id,
                "_end_turn_id": end_turn_id,
            }
        )

    if not outcomes:
        return _run_local_fallback(rt, force=force, reason="LLM agendas 파싱 실패", mode=mode)

    enriched: list[dict[str, Any]] = []
    for row in outcomes:
        enriched.append(_enrich_outcome_summary(rt, row, turns))
    outcomes = enriched
    outcomes, post_attempts, post_success = _refresh_low_quality_titles_with_llm(client, rt, turns, outcomes)
    title_refine_attempts += post_attempts
    title_refine_success += post_success

    rt.last_llm_parsed_json = {
        "pipeline": "two_stage",
        "outline": outline_parsed,
        "details": detail_logs,
    }
    rt.last_llm_parsed_at = _now_ts()

    _apply_outcomes(rt, outcomes)

    rt.last_analyzed_count = len(rt.transcript)
    rt.used_local_fallback = False
    notes: list[str] = []
    if _safe_text(refine_note):
        notes.append(_safe_text(refine_note))
    notes.append(f"안건 상세 추출 {detail_success}/{detail_attempts} 성공")
    notes.append(f"안건 제목 재요청 {title_refine_success}/{title_refine_attempts} 성공")
    rt.last_analysis_warning = " | ".join(notes)
    rt.last_tick_mode = "full_document" if mode == "full_document" else "windowed"
    rt.last_title_refine_attempts = int(title_refine_attempts)
    rt.last_title_refine_success = int(title_refine_success)
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
    rt.transcript_version += 1


def _append_many_turns(rt: RuntimeStore, rows: list[dict[str, str]]) -> int:
    before = len(rt.transcript)
    for row in rows:
        _append_turn(rt, row.get("speaker", "화자"), row.get("text", ""), row.get("timestamp"))
    return len(rt.transcript) - before


async def _collect_rows_from_uploads(files: list[UploadFile]) -> dict[str, Any]:
    files_scanned = 0
    files_parsed = 0
    files_skipped = 0
    parse_errors: list[dict[str, str]] = []
    file_stats: list[dict[str, Any]] = []
    all_rows: list[dict[str, str]] = []
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
                parse_errors.append({"file": upload.filename or "upload.json", "error": "decode failed"})
                continue
        except Exception:
            files_skipped += 1
            parse_errors.append({"file": upload.filename or "upload.json", "error": "read failed"})
            continue

        data = _extract_json(raw)
        if not data:
            files_skipped += 1
            parse_errors.append({"file": upload.filename or "upload.json", "error": "json parse failed"})
            continue
        ok_payload, payload_reason = _looks_like_meeting_payload(data)
        if not ok_payload:
            files_skipped += 1
            parse_errors.append({"file": upload.filename or "upload.json", "error": payload_reason})
            continue
        goal, rows = _parse_meeting_json_payload(data)
        if not rows:
            files_skipped += 1
            parse_errors.append({"file": upload.filename or "upload.json", "error": "utterance rows extracted = 0"})
            continue

        if goal and not applied_goal:
            applied_goal = goal
        all_rows.extend(rows)
        files_parsed += 1
        file_stats.append({"file": upload.filename or "upload.json", "rows": len(rows)})

    return {
        "rows": all_rows,
        "files_scanned": files_scanned,
        "files_parsed": files_parsed,
        "files_skipped": files_skipped,
        "file_stats": file_stats,
        "parse_errors": parse_errors[:20],
        "applied_goal": applied_goal,
    }


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
_ensure_analysis_worker_started()


@app.get("/api/health")
def get_health():
    return {
        "ok": True,
        "whisper_model": WHISPER_MODEL_NAME,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "deps": {
            "fastapi": importlib.util.find_spec("fastapi") is not None,
            "python_multipart": importlib.util.find_spec("multipart") is not None,
            "whisper": importlib.util.find_spec("whisper") is not None,
            "dotenv": importlib.util.find_spec("dotenv") is not None,
            "numpy": importlib.util.find_spec("numpy") is not None,
        },
    }


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
        queue_ok = False
        queue_err = ""
        queued_task_id = 0
        if RT.llm_enabled:
            queue_ok, queued_task_id, queue_err = _enqueue_analysis(RT, force=True, mode="full_document", source="llm_connect")
            if not queue_ok:
                RT.analysis_last_error = _safe_text(queue_err)
        return {
            "enabled": RT.llm_enabled,
            "result": result,
            "llm_status": client.status(),
            "queued_analysis": {"ok": queue_ok, "task_id": queued_task_id, "error": queue_err},
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
        _enqueue_windowed_with_backpressure(RT, source="manual_turn")
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
        parse_errors: list[dict[str, str]] = []
        applied_goal = None

        for path in files:
            files_scanned += 1
            try:
                raw = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    raw = path.read_text(encoding="utf-8-sig")
                except Exception as exc:
                    files_skipped += 1
                    parse_errors.append({"file": str(path), "error": f"decode failed: {exc}"})
                    continue
            except Exception:
                files_skipped += 1
                parse_errors.append({"file": str(path), "error": "read failed"})
                continue
            data = _extract_json(raw)
            if not data:
                files_skipped += 1
                parse_errors.append({"file": str(path), "error": "json parse failed"})
                continue
            ok_payload, payload_reason = _looks_like_meeting_payload(data)
            if not ok_payload:
                files_skipped += 1
                parse_errors.append({"file": str(path), "error": payload_reason})
                continue
            goal, rows = _parse_meeting_json_payload(data)
            if not rows:
                files_skipped += 1
                parse_errors.append({"file": str(path), "error": "utterance rows extracted = 0"})
                continue
            if goal and not applied_goal:
                applied_goal = goal
            added = _append_many_turns(RT, rows)
            rows_loaded += added
            files_parsed += 1
            file_stats.append({"file": str(path), "rows": added})

        if applied_goal:
            RT.meeting_goal = applied_goal

        ticked = False
        queue_err = ""
        queued_task_id = 0
        if payload.auto_tick and RT.transcript:
            ticked, queued_task_id, queue_err = _enqueue_analysis(RT, force=True, mode="full_document", source="import_json_dir")
            if not ticked:
                RT.analysis_last_error = _safe_text(queue_err)

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
                "queued_task_id": int(queued_task_id),
                "queue_error": _safe_text(queue_err),
                "analysis_mode": "none" if not RT.llm_enabled else "full_document_once",
                "meeting_goal_applied": bool(applied_goal),
                "warning": "" if files_parsed > 0 else ("파싱된 JSON 파일이 없습니다." + (f" 예: {parse_errors[0]['error']}" if parse_errors else "")),
                "file_stats": file_stats,
                "parse_errors": parse_errors[:20],
            },
        }


@app.post("/api/transcript/import-json-files")
async def post_import_json_files(
    files: list[UploadFile] = File(default=[]),
    reset_state: str = Form(default="true"),
    auto_tick: str = Form(default="true"),
):
    parsed = await _collect_rows_from_uploads(files)
    with RT.lock:
        do_reset = _boolify(reset_state, True)
        do_tick = _boolify(auto_tick, True)
        if do_reset:
            RT.reset()

        rows_loaded = _append_many_turns(RT, parsed["rows"])

        if parsed["applied_goal"]:
            RT.meeting_goal = parsed["applied_goal"]

        ticked = False
        queue_err = ""
        queued_task_id = 0
        if do_tick and RT.transcript:
            ticked, queued_task_id, queue_err = _enqueue_analysis(RT, force=True, mode="full_document", source="import_json_files")
            if not ticked:
                RT.analysis_last_error = _safe_text(queue_err)

        return {
            "state": _state_response(RT),
            "import_debug": {
                "folder": "<uploaded>",
                "files_scanned": int(parsed["files_scanned"]),
                "files_parsed": int(parsed["files_parsed"]),
                "files_skipped": int(parsed["files_skipped"]),
                "rows_loaded": rows_loaded,
                "meeting_goal": RT.meeting_goal or "",
                "added": rows_loaded,
                "reset_state": do_reset,
                "auto_tick": do_tick,
                "ticked": bool(ticked),
                "queued_task_id": int(queued_task_id),
                "queue_error": _safe_text(queue_err),
                "analysis_mode": "none" if not RT.llm_enabled else "full_document_once",
                "meeting_goal_applied": bool(parsed["applied_goal"]),
                "warning": ""
                if int(parsed["files_parsed"]) > 0
                else ("파싱된 JSON 파일이 없습니다." + (f" 예: {parsed['parse_errors'][0]['error']}" if parsed["parse_errors"] else "")),
                "file_stats": list(parsed["file_stats"]),
                "parse_errors": list(parsed["parse_errors"]),
            },
        }


@app.post("/api/transcript/replay/import-json-files")
async def post_replay_import_json_files(
    files: list[UploadFile] = File(default=[]),
    reset_state: str = Form(default="true"),
    apply_goal: str = Form(default="true"),
):
    parsed = await _collect_rows_from_uploads(files)
    with RT.lock:
        do_reset = _boolify(reset_state, True)
        do_apply_goal = _boolify(apply_goal, True)
        if do_reset:
            RT.reset()

        RT.replay_rows = list(parsed["rows"])
        RT.replay_index = 0
        RT.replay_source = "upload_json_files"
        RT.replay_loaded_at = _now_ts() if RT.replay_rows else ""

        if do_apply_goal and parsed["applied_goal"]:
            RT.meeting_goal = parsed["applied_goal"]

        return {
            "state": _state_response(RT),
            "replay_debug": {
                "queued_total": len(RT.replay_rows),
                "queued_cursor": int(RT.replay_index),
                "queued_remaining": max(0, len(RT.replay_rows) - int(RT.replay_index)),
                "done": False,
                "source": _safe_text(RT.replay_source),
                "loaded_at": _safe_text(RT.replay_loaded_at),
                "files_scanned": int(parsed["files_scanned"]),
                "files_parsed": int(parsed["files_parsed"]),
                "files_skipped": int(parsed["files_skipped"]),
                "meeting_goal_applied": bool(do_apply_goal and parsed["applied_goal"]),
                "warning": ""
                if int(parsed["files_parsed"]) > 0
                else ("파싱된 JSON 파일이 없습니다." + (f" 예: {parsed['parse_errors'][0]['error']}" if parsed["parse_errors"] else "")),
                "file_stats": list(parsed["file_stats"]),
                "parse_errors": list(parsed["parse_errors"]),
            },
        }


@app.post("/api/transcript/replay/step")
def post_replay_step(payload: ReplayStepInput):
    with RT.lock:
        total = len(RT.replay_rows)
        cursor = max(0, min(int(RT.replay_index), total))
        if total <= 0 or cursor >= total:
            RT.replay_index = total
            return {
                "state": _state_response(RT),
                "replay_debug": {
                    "added": 0,
                    "requested": int(payload.lines),
                    "analyzed": False,
                    "queued_total": total,
                    "queued_cursor": int(RT.replay_index),
                    "queued_remaining": 0,
                    "done": True,
                    "warning": "주입할 replay 큐가 없습니다.",
                },
            }

        take = max(1, min(int(payload.lines), 100))
        end = min(total, cursor + take)
        batch = RT.replay_rows[cursor:end]
        added = _append_many_turns(RT, batch)
        RT.replay_index = end

        analyzed = False
        queued_task_id = 0
        queue_error = ""
        deferred = False
        if payload.auto_analyze and added > 0:
            analyzed, queued_task_id, queue_error, deferred = _enqueue_windowed_with_backpressure(RT, source="replay_step")
            if (not analyzed) and (not deferred) and _safe_text(queue_error):
                RT.analysis_last_error = _safe_text(queue_error)

        remaining = max(0, total - int(RT.replay_index))
        done = remaining == 0
        return {
            "state": _state_response(RT),
            "replay_debug": {
                "added": added,
                "requested": take,
                "analyzed": bool(analyzed or deferred),
                "queued_task_id": int(queued_task_id),
                "queue_error": _safe_text(queue_error),
                "deferred": bool(deferred),
                "queued_total": total,
                "queued_cursor": int(RT.replay_index),
                "queued_remaining": remaining,
                "done": done,
                "warning": "",
            },
        }


@app.post("/api/analysis/tick")
def post_analysis_tick():
    with RT.lock:
        ok, _, err = _enqueue_analysis(RT, force=True, mode="full_document", source="manual_tick")
        if not ok:
            RT.analysis_last_error = _safe_text(err)
            RT.last_analysis_warning = f"분석 요청 큐 적재 실패: {err}"
        return _state_response(RT)


@app.get("/api/analysis/last-llm-json")
def get_last_llm_json():
    with RT.lock:
        return {
            "ok": True,
            "received_at": _safe_text(RT.last_llm_parsed_at),
            "has_json": bool(RT.last_llm_parsed_json),
            "json": RT.last_llm_parsed_json if isinstance(RT.last_llm_parsed_json, dict) else {},
        }


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
            _enqueue_windowed_with_backpressure(RT, source="stt_chunk")
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
