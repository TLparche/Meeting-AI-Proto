from __future__ import annotations

import math
import re
import time
from collections import Counter
from datetime import datetime
from typing import Any

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.+#/\-]*|[가-힣]{2,}")


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
        fallback = max(6, seconds // 6)
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
    return out or transcript[-max(6, seconds // 6) :]


def _tokenize(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(text or "")
    out: list[str] = []
    for tok in tokens:
        if re.fullmatch(r"[A-Za-z0-9_.+#/\-]+", tok):
            out.append(tok.lower())
        else:
            out.append(tok)
    return out


def _vectorize_text(text: str) -> Counter[str]:
    return Counter(_tokenize(text))


def _vectorize_agenda(active_agenda_title: str, agenda_vector: dict[str, Any]) -> Counter[str]:
    terms = agenda_vector.get("terms") if isinstance(agenda_vector, dict) else None
    if isinstance(terms, list) and terms:
        weighted = Counter()
        for t in terms:
            term = str((t or {}).get("term") or "").strip()
            if not term:
                continue
            w = float((t or {}).get("weight") or 0.0)
            if w <= 0.0:
                w = 0.1
            weighted[term.lower()] += w
        if weighted:
            return weighted
    return _vectorize_text(active_agenda_title or "")


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    for k, v in a.items():
        dot += v * b.get(k, 0.0)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def run_drift_dampener(
    *,
    transcript: list[dict],
    active_agenda_title: str,
    active_agenda_vector: dict[str, Any] | None,
    monitor_state: dict[str, Any] | None,
    meeting_elapsed_sec: float,
) -> dict[str, Any]:
    now = time.time()
    monitor = dict(monitor_state or {})
    yellow_since = monitor.get("yellow_since")
    red_since = monitor.get("red_since")
    last_state = str(monitor.get("last_state") or "Normal")

    turns_45 = _window_by_seconds(transcript, 45)
    agenda_vec = _vectorize_agenda(active_agenda_title, active_agenda_vector or {})
    sims: list[float] = []
    for turn in turns_45:
        text = str(turn.get("text") or "")
        utt_vec = _vectorize_text(text)
        sims.append(_cosine(utt_vec, agenda_vec))
    s45 = _mean(sims)

    # band candidate by threshold
    if s45 >= 0.72:
        band = "Green"
    elif s45 >= 0.62:
        band = "Yellow"
    else:
        band = "Red"

    if band == "Green":
        yellow_since = None
        red_since = None
        drift_state = "Normal"
    elif band == "Yellow":
        if yellow_since is None:
            yellow_since = now
        red_since = None
        yellow_dur = now - yellow_since
        drift_state = "Yellow" if yellow_dur > 30 else "Normal"
    else:
        if red_since is None:
            red_since = now
        if yellow_since is None:
            yellow_since = now
        red_dur = now - red_since
        safe_zone = meeting_elapsed_sec < 180
        if safe_zone:
            # suppress Red-level intervention during first 180 seconds.
            drift_state = "Yellow" if red_dur > 30 else "Normal"
        else:
            if red_dur > 120:
                drift_state = "Re-orient"
            elif red_dur > 30:
                drift_state = "Red"
            else:
                drift_state = "Normal"

    yellow_seconds = 0.0 if yellow_since is None else max(0.0, now - float(yellow_since))
    red_seconds = 0.0 if red_since is None else max(0.0, now - float(red_since))

    ui_cues = {
        "glow_k_core": drift_state in ("Yellow", "Red", "Re-orient"),
        "fix_k_core_focus": drift_state in ("Red", "Re-orient"),
        "reduce_facets": drift_state in ("Red", "Re-orient"),
        "show_banner": drift_state == "Re-orient",
    }

    monitor["yellow_since"] = yellow_since
    monitor["red_since"] = red_since
    monitor["last_state"] = drift_state
    monitor["last_band"] = band

    return {
        "drift_state": drift_state,
        "ui_cues": ui_cues,
        "monitor": monitor,
        "debug": {
            "s45": round(s45, 4),
            "band": band,
            "last_state": last_state,
            "yellow_seconds": round(yellow_seconds, 1),
            "red_seconds": round(red_seconds, 1),
            "safe_zone": meeting_elapsed_sec < 180,
            "window_turns": len(turns_45),
        },
    }

