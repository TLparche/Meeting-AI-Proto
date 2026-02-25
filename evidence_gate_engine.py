from __future__ import annotations

import re
from datetime import datetime
from typing import Any

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?(?:%|퍼센트|배|x)?")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.+#/\-]*|[가-힣]{2,}")

COMPARATIVE_HINTS = (
    "비교",
    "더",
    "덜",
    "증가",
    "감소",
    "개선",
    "악화",
    "높",
    "낮",
    "faster",
    "slower",
    "higher",
    "lower",
    "increase",
    "decrease",
)
POLICY_HINTS = (
    "정책",
    "규정",
    "법",
    "컴플라이언스",
    "compliance",
    "must",
    "should",
    "required",
    "필수",
    "금지",
    "허용",
)
EVIDENCE_HINTS = (
    "출처",
    "데이터",
    "지표",
    "통계",
    "레포트",
    "report",
    "paper",
    "benchmark",
    "로그",
    "link",
    "링크",
    "사실",
    "근거",
)
AGREE_HINTS = ("동의", "찬성", "확정", "승인", "agree", "approved", "yes")
DISAGREE_HINTS = ("반대", "이견", "보류", "우려", "disagree", "not", "no")
LOW_QUALITY_HINTS = ("느낌", "추정", "아마", "maybe", "guess")
RECENT_HINTS = ("오늘", "이번", "최근", "latest", "today", "this week", "이번주")


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
        return transcript[-max(12, seconds // 8) :]
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
    return out or transcript[-max(12, seconds // 8) :]


def _tokenize(text: str) -> set[str]:
    toks = TOKEN_RE.findall(text or "")
    out: set[str] = set()
    for t in toks:
        if re.fullmatch(r"[A-Za-z0-9_.+#/\-]+", t):
            out.add(t.lower())
        else:
            out.add(t)
    return out


def _contains_any(text: str, hints: tuple[str, ...]) -> bool:
    txt = (text or "").lower()
    return any(h in txt for h in hints)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def run_evidence_gate(
    *,
    transcript: list[dict],
) -> dict[str, Any]:
    turns = _window_by_seconds(transcript, 300)
    if not turns:
        return {
            "evidence_status": "UNVERIFIED",
            "evidence_snippet": "근거 후보가 아직 감지되지 않았습니다.",
            "evidence_log": [],
            "claims_for_schema": [],
        }

    claims: list[dict[str, Any]] = []
    for idx, turn in enumerate(turns):
        text = str(turn.get("text") or "").strip()
        if not text:
            continue
        has_number = bool(NUMBER_RE.search(text))
        has_comp = _contains_any(text, COMPARATIVE_HINTS)
        has_policy = _contains_any(text, POLICY_HINTS)
        has_evidence_hint = _contains_any(text, EVIDENCE_HINTS)
        score = 0.0
        if has_number:
            score += 0.35
        if has_comp:
            score += 0.25
        if has_policy:
            score += 0.25
        if has_evidence_hint:
            score += 0.15
        detect_score = _clip01(score)
        if detect_score < 0.35:
            continue
        claims.append(
            {
                "idx": idx,
                "speaker": str(turn.get("speaker") or "미상"),
                "timestamp": str(turn.get("timestamp") or ""),
                "claim": text[:160],
                "detect_score": detect_score,
            }
        )

    if not claims:
        return {
            "evidence_status": "UNVERIFIED",
            "evidence_snippet": "주요 주장(수치/비교/정책형)이 부족하여 검증 상태를 산출하지 못했습니다.",
            "evidence_log": [],
            "claims_for_schema": [],
        }

    evidence_log: list[dict[str, Any]] = []
    claims_for_schema: list[dict[str, Any]] = []

    last_ts = _parse_hms(str(turns[-1].get("timestamp") or ""))
    checked = 0
    high_eqs = 0
    low_eqs = 0
    conflict_count = 0

    for c in claims[:8]:
        claim_txt = str(c["claim"])
        detect_score = float(c["detect_score"])
        claim_tokens = _tokenize(claim_txt)
        ctx = turns[max(0, c["idx"] - 2) : c["idx"] + 3]
        ctx_text = " ".join(str(t.get("text") or "") for t in ctx)
        has_src = _contains_any(ctx_text, EVIDENCE_HINTS)

        verifiability = _clip01(
            0.40 * detect_score
            + (0.25 if has_src else 0.0)
            + (0.20 if bool(NUMBER_RE.search(claim_txt)) else 0.0)
            + (0.15 if len(claim_tokens) >= 5 else 0.0)
        )

        # Verifiability gate: run EQS only if >= 0.6
        if verifiability < 0.6:
            note = f"verifiability {verifiability:.2f} < 0.60: EQS 스킵"
            claims_for_schema.append(
                {"claim": claim_txt, "verifiability": round(verifiability, 2), "note": note}
            )
            evidence_log.append(
                {
                    "claim": claim_txt,
                    "verifiability": round(verifiability, 4),
                    "eqs": None,
                    "status": "UNVERIFIED",
                    "note": note,
                }
            )
            continue

        checked += 1

        # T: tier
        ctx_lower = ctx_text.lower()
        if ("http" in ctx_lower or "link" in ctx_lower or "링크" in ctx_lower) and (
            "report" in ctx_lower or "paper" in ctx_lower or "benchmark" in ctx_lower or "레포트" in ctx_lower
        ):
            tier = 0.95
        elif has_src:
            tier = 0.75
        elif _contains_any(ctx_text, LOW_QUALITY_HINTS):
            tier = 0.35
        else:
            tier = 0.55

        # R: recency
        ts = _parse_hms(str(c["timestamp"]))
        if ts is not None and last_ts is not None:
            diff = last_ts - ts
            if diff < 0:
                diff += 24 * 3600
            if diff <= 120:
                recency = 1.0
            elif diff <= 300:
                recency = 0.8
            else:
                recency = 0.55
        else:
            recency = 0.8 if _contains_any(claim_txt, RECENT_HINTS) else 0.65

        # C: claim match (token overlap with evidence-ish context turns)
        match_scores = []
        for t in ctx:
            txt = str(t.get("text") or "")
            if not _contains_any(txt, EVIDENCE_HINTS):
                continue
            tks = _tokenize(txt)
            if not claim_tokens or not tks:
                continue
            inter = len(claim_tokens & tks)
            union = max(1, len(claim_tokens | tks))
            match_scores.append(inter / union)
        claim_match = max(match_scores) if match_scores else 0.45

        # A: agreement
        support = 0
        oppose = 0
        for t in ctx:
            txt = str(t.get("text") or "").lower()
            if _contains_any(txt, AGREE_HINTS):
                support += 1
            if _contains_any(txt, DISAGREE_HINTS):
                oppose += 1
        total_stance = support + oppose
        if total_stance == 0:
            agreement = 0.55
        else:
            agreement = support / total_stance

        eqs = _clip01(0.40 * tier + 0.20 * recency + 0.25 * claim_match + 0.15 * agreement)
        conflict = support > 0 and oppose > 0
        if conflict:
            conflict_count += 1
        if eqs >= 0.75:
            high_eqs += 1
        if eqs < 0.55:
            low_eqs += 1

        status = "VERIFIED" if eqs >= 0.75 and not conflict else ("MIXED" if eqs >= 0.60 else "UNVERIFIED")
        note = (
            f"EQS={eqs:.2f} (T={tier:.2f},R={recency:.2f},C={claim_match:.2f},A={agreement:.2f})"
            + ("; conflict detected" if conflict else "")
        )
        claims_for_schema.append(
            {"claim": claim_txt, "verifiability": round(verifiability, 2), "note": note}
        )
        evidence_log.append(
            {
                "claim": claim_txt,
                "speaker": c["speaker"],
                "timestamp": c["timestamp"],
                "detect_score": round(detect_score, 4),
                "verifiability": round(verifiability, 4),
                "eqs": round(eqs, 4),
                "tier": round(tier, 4),
                "recency": round(recency, 4),
                "claim_match": round(claim_match, 4),
                "agreement": round(agreement, 4),
                "status": status,
                "note": note,
            }
        )

    if checked == 0:
        evidence_status = "UNVERIFIED"
    elif high_eqs >= 1 and conflict_count == 0 and low_eqs == 0:
        evidence_status = "VERIFIED"
    elif high_eqs >= 1 or conflict_count >= 1:
        evidence_status = "MIXED"
    else:
        evidence_status = "UNVERIFIED"

    # 1~2 line snippet
    line1 = f"{evidence_status}: claims={len(claims_for_schema)}, checked={checked}, high_eqs={high_eqs}"
    if conflict_count > 0:
        line2 = f"상충 신호 {conflict_count}건 감지. 결정 강제 없이 상태만 공유합니다."
    else:
        line2 = "검증은 비차단(non-blocking) 방식으로 진행되며 상태만 표시합니다."
    evidence_snippet = f"{line1}\n{line2}"

    return {
        "evidence_status": evidence_status,
        "evidence_snippet": evidence_snippet,
        "evidence_log": evidence_log,
        "claims_for_schema": claims_for_schema[:8],
    }

