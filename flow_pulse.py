from __future__ import annotations

import re
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
        fallback = max(10, seconds // 6)
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
    return out or transcript[-max(10, seconds // 6) :]


def _tokenize(text: str) -> list[str]:
    raw = TOKEN_RE.findall(text or "")
    out: list[str] = []
    for tok in raw:
        if re.fullmatch(r"[A-Za-z0-9_.+#/\-]+", tok):
            out.append(tok.lower())
        else:
            out.append(tok)
    return out


def _anchor_terms(k_core: dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    for bucket in ("object", "constraints", "criteria"):
        for phrase in (k_core.get(bucket) or []):
            for tok in _tokenize(str(phrase)):
                if tok:
                    terms.add(tok)
    return terms


def _novelty_rate(turn_tokens: list[list[str]]) -> float:
    seen: set[str] = set()
    total = 0
    new_count = 0
    for toks in turn_tokens:
        for t in toks:
            total += 1
            if t not in seen:
                new_count += 1
                seen.add(t)
    if total == 0:
        return 1.0
    return new_count / total


def _arg_novelty(turn_tokens: list[list[str]]) -> float:
    sigs = []
    for toks in turn_tokens:
        c = Counter(toks)
        if not c:
            continue
        # argument signature: top 4 terms
        sig = tuple(sorted([k for k, _ in c.most_common(4)]))
        sigs.append(sig)
    if not sigs:
        return 1.0
    return len(set(sigs)) / len(sigs)


def _delta_dps(dps_history: list[dict], now_ts: float) -> float:
    hist = [h for h in dps_history if now_ts - float(h.get("ts", now_ts)) <= 180.0]
    if len(hist) < 2:
        return 1.0
    first = float(hist[0].get("score", 0.0))
    last = float(hist[-1].get("score", 0.0))
    return last - first


def run_flow_pulse(
    *,
    transcript: list[dict],
    k_core: dict[str, Any],
    dps_history: list[dict],
    now_ts: float,
) -> dict[str, Any]:
    turns_3m = _window_by_seconds(transcript, 180)
    anchors = _anchor_terms(k_core)

    token_rows_non_anchor: list[list[str]] = []
    token_rows_all: list[list[str]] = []
    anchor_hits = 0
    all_hits = 0
    for turn in turns_3m:
        toks = _tokenize(str(turn.get("text") or ""))
        token_rows_all.append(toks)
        all_hits += len(toks)
        anchor_hits += len([t for t in toks if t in anchors])
        token_rows_non_anchor.append([t for t in toks if t not in anchors])

    novelty_rate = _novelty_rate(token_rows_non_anchor)
    arg_novelty = _arg_novelty(token_rows_non_anchor)
    delta_dps = _delta_dps(dps_history, now_ts=now_ts)

    cond_a = novelty_rate < 0.15
    cond_b = arg_novelty < 0.20
    cond_c = delta_dps < 0.05

    anchor_ratio = (anchor_hits / all_hits) if all_hits > 0 else 0.0
    non_anchor_hits = max(0, all_hits - anchor_hits)
    anchoring_exception = anchor_ratio >= 0.40 and non_anchor_hits <= max(8, int(all_hits * 0.35))

    stagnation_flag = cond_a and cond_b and cond_c and not anchoring_exception
    if anchoring_exception:
        loop_state = "Anchoring"
    elif stagnation_flag:
        loop_state = "Looping"
    elif (cond_a and cond_b) or (cond_a and cond_c) or (cond_b and cond_c):
        loop_state = "Watching"
    else:
        loop_state = "Normal"

    return {
        "stagnation_flag": stagnation_flag,
        "loop_state": loop_state,
        "debug": {
            "novelty_rate_3m": round(novelty_rate, 4),
            "arg_novelty": round(arg_novelty, 4),
            "delta_dps": round(delta_dps, 4),
            "anchor_ratio": round(anchor_ratio, 4),
            "conditions": {
                "surface_repetition_a": cond_a,
                "content_repetition_b": cond_b,
                "no_progress_c": cond_c,
                "anchoring_exception": anchoring_exception,
            },
            "turns_3m": len(turns_3m),
        },
    }

