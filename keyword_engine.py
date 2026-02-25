from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


KEYWORD_TYPES = (
    "K1_OBJECT",
    "K2_OPTION",
    "K3_CONSTRAINT",
    "K4_CRITERION",
    "K5_EVIDENCE",
    "K6_ACTION",
)

TAXONOMY = {
    "K1_OBJECT": "무엇을 결정하는가(결정 대상)",
    "K2_OPTION": "어떤 대안들이 있는가",
    "K3_CONSTRAINT": "제한/조건(예산/시간/정책 등)",
    "K4_CRITERION": "평가 기준(성능/비용/리스크 등)",
    "K5_EVIDENCE": "근거(출처/데이터/사실 주장)",
    "K6_ACTION": "누가/언제/무엇을 할지(담당/기한)",
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.+#/\-]*|[가-힣]{2,}")
SPACE_RE = re.compile(r"\s+")

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "into",
    "about",
    "have",
    "will",
    "would",
    "could",
    "should",
    "있습니다",
    "합니다",
    "그리고",
    "그런데",
    "저희",
    "우리",
    "이번",
    "다음",
    "관련",
    "부분",
    "정도",
    "내용",
    "회의",
    "진행",
    "검토",
}

CONSTRAINT_HINTS = (
    "예산",
    "비용",
    "budget",
    "deadline",
    "기한",
    "시간",
    "일정",
    "정책",
    "policy",
    "규정",
    "보안",
    "인력",
    "리소스",
    "resource",
    "법무",
    "컴플라이언스",
    "compliance",
)

CRITERION_HINTS = (
    "성능",
    "품질",
    "비용효율",
    "효율",
    "리스크",
    "risk",
    "latency",
    "throughput",
    "정확도",
    "신뢰도",
    "확장성",
    "안정성",
    "만족도",
    "roi",
    "kpi",
    "지표",
)

EVIDENCE_HINTS = (
    "근거",
    "출처",
    "데이터",
    "수치",
    "통계",
    "레포트",
    "report",
    "benchmark",
    "실험",
    "로그",
    "지표",
    "fact",
    "evidence",
    "링크",
    "citation",
    "paper",
)

OPTION_HINTS = (
    "옵션",
    "대안",
    "안",
    "방식",
    "시나리오",
    "전략",
    "option",
    "alternative",
    "plan",
    "rollout",
    "pilot",
)

ACTION_HINTS = (
    "담당",
    "owner",
    "action",
    "todo",
    "task",
    "실행",
    "추진",
    "작성",
    "확인",
    "공유",
    "요청",
    "완료",
    "by",
    "까지",
    "다음주",
    "내일",
    "오늘",
)

TYPE_DECISION_VALUE = {
    "K1_OBJECT": 1.00,
    "K2_OPTION": 0.78,
    "K3_CONSTRAINT": 0.90,
    "K4_CRITERION": 0.92,
    "K5_EVIDENCE": 0.74,
    "K6_ACTION": 0.80,
}


@dataclass
class CandidateState:
    term: str
    frequency: int = 0
    first_seen: str = ""
    first_turn: int = 0
    last_turn: int = 0
    evidence_hits: int = 0


def _normalize_token(token: str) -> str:
    token = token.strip()
    if not token:
        return ""
    if re.fullmatch(r"[A-Za-z0-9_.+#/\-]+", token):
        return token.lower()
    return token


def _timestamp_of(turn: dict, turn_idx: int) -> str:
    ts = (turn.get("timestamp") or "").strip()
    if ts:
        return ts
    return f"T+{turn_idx:02d}"


def _extract_candidates(
    transcript_window: list[dict],
    meeting_goal: str,
    current_active_agenda: str,
    top_n: int = 40,
) -> tuple[list[CandidateState], dict[str, CandidateState]]:
    states: dict[str, CandidateState] = {}
    corpus = [{"text": meeting_goal or "", "timestamp": "goal"}]
    if current_active_agenda:
        corpus.append({"text": current_active_agenda, "timestamp": "agenda"})
    corpus.extend(transcript_window)

    for idx, turn in enumerate(corpus):
        text = str(turn.get("text") or "")
        lowered = text.lower()
        has_evidence_hint = any(h in lowered for h in EVIDENCE_HINTS)
        for raw in TOKEN_RE.findall(text):
            token = _normalize_token(raw)
            if len(token) < 2:
                continue
            if token in STOPWORDS:
                continue
            if token.isdigit():
                continue

            state = states.get(token)
            if state is None:
                state = CandidateState(
                    term=token,
                    frequency=1,
                    first_seen=_timestamp_of(turn, idx),
                    first_turn=idx,
                    last_turn=idx,
                    evidence_hits=1 if has_evidence_hint else 0,
                )
                states[token] = state
            else:
                state.frequency += 1
                state.last_turn = idx
                if has_evidence_hint:
                    state.evidence_hits += 1

    ranked = sorted(
        states.values(),
        key=lambda c: (c.frequency, c.evidence_hits, -c.first_turn, len(c.term)),
        reverse=True,
    )[:top_n]
    return ranked, states


def _classify_term(term: str, focus_terms: set[str]) -> str:
    lower = term.lower()
    if any(h in lower for h in EVIDENCE_HINTS):
        return "K5_EVIDENCE"
    if any(h in lower for h in ACTION_HINTS):
        return "K6_ACTION"
    if any(h in lower for h in OPTION_HINTS):
        return "K2_OPTION"
    if any(h in lower for h in CONSTRAINT_HINTS):
        return "K3_CONSTRAINT"
    if any(h in lower for h in CRITERION_HINTS):
        return "K4_CRITERION"
    if term in focus_terms:
        return "K1_OBJECT"
    if len(term) >= 8:
        return "K1_OBJECT"
    return "K2_OPTION"


def _score_term(c: CandidateState, typ: str, total_turns: int) -> tuple[float, float, float]:
    decision_value = TYPE_DECISION_VALUE.get(typ, 0.7)
    freq_norm = min(1.0, c.frequency / 3.0)
    recency = 0.0
    if total_turns > 1:
        recency = c.last_turn / float(total_turns - 1)

    evidence_boost = min(0.24, c.evidence_hits * 0.06 + (0.08 if typ == "K5_EVIDENCE" else 0.0))
    score = min(1.0, 0.52 * decision_value + 0.30 * freq_norm + 0.18 * recency + evidence_boost)
    return round(score, 4), round(decision_value, 4), round(evidence_boost, 4)


def _enforce_core_slot(
    by_type: dict[str, list[dict[str, Any]]],
    meeting_goal: str,
    preferred_object: str,
) -> tuple[dict[str, list[str]], set[str]]:
    chosen: set[str] = set()

    def pick_terms(type_key: str, limit: int) -> list[str]:
        out: list[str] = []
        for item in by_type.get(type_key, []):
            term = item["keyword"]
            if term in chosen:
                continue
            out.append(term)
            chosen.add(term)
            if len(out) >= limit:
                break
        return out

    preferred = _compact(preferred_object.strip() or meeting_goal.strip() or "의사결정 대상 정의", max_len=48)
    object_terms = [preferred]
    chosen.add(preferred)

    constraints = pick_terms("K3_CONSTRAINT", 2)
    if not constraints:
        constraints = ["제약 조건 명시"]
        chosen.add("제약 조건 명시")
    criteria = pick_terms("K4_CRITERION", 2)
    if not criteria:
        criteria = ["성공 기준 명확화"]
        chosen.add("성공 기준 명확화")

    core = {
        "object": object_terms,
        "constraints": constraints[:2],
        "criteria": criteria[:2],
    }
    return core, chosen


def _build_facet_slot(
    by_type: dict[str, list[dict[str, Any]]],
    already_selected: set[str],
) -> tuple[dict[str, list[str]], bool]:
    pools = {
        "options": [x["keyword"] for x in by_type.get("K2_OPTION", [])],
        "evidence": [x["keyword"] for x in by_type.get("K5_EVIDENCE", [])],
        "actions": [x["keyword"] for x in by_type.get("K6_ACTION", [])],
    }

    facet = {"options": [], "evidence": [], "actions": []}
    diversity_boost = False

    for key in ("options", "evidence", "actions"):
        for term in pools[key]:
            if term in already_selected:
                continue
            facet[key].append(term)
            already_selected.add(term)
            break
    if sum(len(v) for v in facet.values()) >= 2:
        diversity_boost = True

    # round-robin to keep 3~8 facet items with type diversity
    order = ("options", "evidence", "actions")
    while sum(len(v) for v in facet.values()) < 8:
        advanced = False
        for key in order:
            if len(facet[key]) >= 3:
                continue
            candidate = None
            for term in pools[key]:
                if term not in already_selected:
                    candidate = term
                    break
            if candidate is None:
                continue
            facet[key].append(candidate)
            already_selected.add(candidate)
            advanced = True
            if sum(len(v) for v in facet.values()) >= 8:
                break
        if not advanced:
            break

    total_facet = sum(len(v) for v in facet.values())
    if total_facet < 3:
        if not facet["options"]:
            facet["options"].append("옵션 정의")
        if not facet["evidence"]:
            facet["evidence"].append("근거 확인")
        if not facet["actions"]:
            facet["actions"].append("후속 액션 지정")

    return facet, diversity_boost


def _compact(text: str, max_len: int = 80) -> str:
    s = SPACE_RE.sub(" ", text.strip())
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def build_keyword_engine_output(
    meeting_goal: str,
    current_active_agenda: str,
    transcript_window: list[dict],
) -> dict[str, Any]:
    ranked, _ = _extract_candidates(
        transcript_window=transcript_window,
        meeting_goal=meeting_goal,
        current_active_agenda=current_active_agenda,
        top_n=40,
    )
    focus_terms = {
        _normalize_token(x)
        for x in TOKEN_RE.findall(f"{meeting_goal} {current_active_agenda}")
        if _normalize_token(x)
    }

    total_turns = max(1, len(transcript_window) + 2)
    items: list[dict[str, Any]] = []
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for c in ranked:
        typ = _classify_term(c.term, focus_terms=focus_terms)
        score, decision_value, evidence_boost = _score_term(c, typ=typ, total_turns=total_turns)
        item = {
            "keyword": c.term,
            "type": typ,
            "score": score,
            "first_seen": c.first_seen,
            "frequency": c.frequency,
            "decision_value": decision_value,
            "evidence_boost": evidence_boost,
        }
        items.append(item)
        by_type[typ].append(item)

    for typ in KEYWORD_TYPES:
        by_type[typ] = sorted(by_type.get(typ, []), key=lambda x: (x["score"], x["frequency"]), reverse=True)

    preferred_object = current_active_agenda or meeting_goal
    k_core, selected = _enforce_core_slot(
        by_type,
        meeting_goal=meeting_goal,
        preferred_object=preferred_object,
    )
    k_facet, diversity_boost_applied = _build_facet_slot(by_type, already_selected=selected)

    candidates = [
        {"keyword": c.term, "frequency": c.frequency, "first_seen": c.first_seen}
        for c in ranked
    ]
    classifications = [{"keyword": x["keyword"], "type": x["type"]} for x in items]
    scoring = [
        {
            "keyword": x["keyword"],
            "decision_value": x["decision_value"],
            "evidence_boost": x["evidence_boost"],
            "score": x["score"],
        }
        for x in items
    ]

    final_selection = {
        "k_core_required": ["K1_OBJECT x1", "K3/K4 x1~2", "K4 포함 권장"],
        "k_facet_target": "3~8",
        "selected_core": k_core["object"] + k_core["constraints"] + k_core["criteria"],
        "selected_facet": k_facet["options"] + k_facet["evidence"] + k_facet["actions"],
        "diversity_boost_applied": diversity_boost_applied,
    }

    # k_core/k_facet에 포함된 항목은 core 표식 부여
    core_terms = set(k_core["object"] + k_core["constraints"] + k_core["criteria"])
    for item in items:
        item["is_core"] = item["keyword"] in core_terms

    return {
        "taxonomy": TAXONOMY,
        "k_core": k_core,
        "k_facet": k_facet,
        "items": items,
        "pipeline": {
            "candidates": candidates,
            "classification": classifications,
            "scoring": scoring,
            "final_selection": final_selection,
        },
        "summary": {
            "object_focus": _compact(", ".join(k_core["object"])),
            "core_count": len(k_core["object"]) + len(k_core["constraints"]) + len(k_core["criteria"]),
            "facet_count": len(k_facet["options"]) + len(k_facet["evidence"]) + len(k_facet["actions"]),
        },
    }
