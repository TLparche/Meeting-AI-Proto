from __future__ import annotations

import re
from datetime import datetime
from typing import Any

INFO_SEEKING_HINTS = (
    "?",
    "어떻게",
    "뭐",
    "무엇",
    "왜",
    "자료",
    "출처",
    "근거",
    "링크",
    "확인",
    "알려",
    "비교",
    "how",
    "what",
    "why",
    "docs",
    "source",
    "evidence",
    "link",
    "compare",
)

INSTALL_HINTS = (
    "install",
    "설치",
    "cuda",
    "torch",
    "pytorch",
    "whisper",
    "ffmpeg",
    "driver",
    "gpu",
)

POLICY_HINTS = (
    "정책",
    "규정",
    "준수",
    "컴플라이언스",
    "policy",
    "guideline",
    "compliance",
    "audit",
    "보안",
)

COMPARISON_HINTS = (
    "비교",
    "trade-off",
    "트레이드오프",
    "장단점",
    "옵션",
    "대안",
    "option",
    "alternative",
)

PILOT_HINTS = ("파일럿", "단계적", "pilot", "staged", "incremental")
FULL_HINTS = ("전면", "일괄", "big-bang", "full rollout", "full")
HYBRID_HINTS = ("하이브리드", "혼합", "hybrid")


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
        return transcript[-max(8, seconds // 8) :]

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
    return out or transcript[-max(8, seconds // 8) :]


def _contains_any(text: str, hints: tuple[str, ...]) -> bool:
    lowered = (text or "").lower()
    return any(h in lowered for h in hints)


def _count_info_seeking(turns: list[dict]) -> int:
    count = 0
    for turn in turns:
        text = str(turn.get("text") or "")
        if _contains_any(text, INFO_SEEKING_HINTS):
            count += 1
    return count


PLACEHOLDER_TERMS = {
    "의사결정 대상 정의",
    "제약 조건 명시",
    "성공 기준 명확화",
}


def _slot_fulfillment(analysis: dict, turns_context: list[dict]) -> tuple[bool, int, int, int]:
    keywords = analysis.get("keywords") or {}
    k_core = keywords.get("k_core") or {}
    contextual = len(turns_context) >= 4
    obj = len([x for x in (k_core.get("object") or []) if str(x).strip() and str(x).strip() not in PLACEHOLDER_TERMS])
    k3 = len(
        [x for x in (k_core.get("constraints") or []) if str(x).strip() and str(x).strip() not in PLACEHOLDER_TERMS]
    )
    k4 = len([x for x in (k_core.get("criteria") or []) if str(x).strip() and str(x).strip() not in PLACEHOLDER_TERMS])
    return contextual and obj >= 1 and (k3 >= 1 or k4 >= 1), obj, k3, k4


def _build_r1_resources(context: str, info_count: int, evidence_status: str) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []

    def add(title: str, url: str, reason: str) -> None:
        if any(x["url"] == url for x in items):
            return
        items.append({"title": title, "url": url, "reason": reason})

    if _contains_any(context, INSTALL_HINTS):
        add(
            "OpenAI Whisper 설치 가이드",
            "https://github.com/openai/whisper",
            "STT 설치/실행 이슈를 빠르게 재현하고 버전 조건을 확인할 수 있습니다.",
        )
        add(
            "PyTorch 로컬 설치 가이드",
            "https://pytorch.org/get-started/locally/",
            "CUDA/CPU 빌드 매트릭스를 확인해 런타임 충돌을 줄일 수 있습니다.",
        )

    if _contains_any(context, POLICY_HINTS):
        add(
            "NIST AI Risk Management Framework",
            "https://www.nist.gov/itl/ai-risk-management-framework",
            "정책/준수 논의 시 근거 품질과 리스크 통제 기준으로 사용할 수 있습니다.",
        )

    if _contains_any(context, COMPARISON_HINTS) or info_count >= 2:
        add(
            "Decision Matrix Template",
            "https://www.atlassian.com/work-management/decision-matrix",
            "옵션 비교 기준(K4)과 점수표를 빠르게 정렬할 수 있습니다.",
        )

    if evidence_status in {"MIXED", "UNVERIFIED"}:
        add(
            "근거 품질 체크리스트",
            "https://www.gov.uk/service-manual/service-standard/point-4-make-the-service-simple-to-use",
            "검증 미완료 상태에서 주장-근거 매칭 항목을 최소 단위로 점검할 수 있습니다.",
        )

    if not items:
        add(
            "회의 근거 정리 템플릿",
            "https://www.atlassian.com/team-playbook/plays/decision-log",
            "결정 로그 형식으로 주장/근거/결정을 빠르게 축약할 수 있습니다.",
        )
    return items[:3]


def _extract_options(analysis: dict, context: str) -> list[str]:
    keywords = analysis.get("keywords") or {}
    k_facet = keywords.get("k_facet") or {}
    raw_opts = [str(x).strip() for x in (k_facet.get("options") or []) if str(x).strip()]
    out: list[str] = []
    seen: set[str] = set()

    def add(opt: str) -> None:
        o = opt.strip()
        if not o:
            return
        key = o.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(o)

    for opt in raw_opts:
        add(opt)

    lowered = context.lower()
    if _contains_any(lowered, PILOT_HINTS):
        add("단계적 파일럿")
    if _contains_any(lowered, FULL_HINTS):
        add("즉시 전면 적용")
    if _contains_any(lowered, HYBRID_HINTS):
        add("하이브리드 전환")

    # option/안 패턴 추출
    for m in re.findall(r"(?:옵션|option|안)\s*([A-Za-z0-9가-힣_\-]{1,16})", lowered, flags=re.IGNORECASE):
        token = m.strip()
        if token:
            add(f"옵션 {token.upper() if token.isascii() else token}")

    if len(out) < 2:
        add("단계적 파일럿")
        add("즉시 전면 적용")
    return out[:2]


def _option_card(option: str, evidence_status: str) -> dict[str, Any]:
    lower = option.lower()
    if _contains_any(lower, PILOT_HINTS):
        pros = ["리스크를 국소화하고 롤백이 쉽습니다.", "실사용 데이터로 점진 개선이 가능합니다."]
        risks = ["전사 적용까지 시간이 더 필요합니다.", "구간별 운영 복잡도가 늘 수 있습니다."]
    elif _contains_any(lower, FULL_HINTS):
        pros = ["효과를 빠르게 체감할 수 있습니다.", "운영 체계가 단일화되어 관리가 단순합니다."]
        risks = ["장애 영향 반경이 큽니다.", "준비 미흡 시 초기 반발/품질 이슈가 커질 수 있습니다."]
    elif _contains_any(lower, HYBRID_HINTS):
        pros = ["속도와 안정성 사이 균형을 잡기 쉽습니다.", "팀별 준비도 차이를 흡수할 수 있습니다."]
        risks = ["정책/운영 기준이 이원화될 수 있습니다.", "책임 경계가 흐려질 수 있습니다."]
    else:
        pros = ["현재 제약조건과의 정합성을 빠르게 검증할 수 있습니다."]
        risks = ["근거가 부족하면 의사결정 지연이 발생할 수 있습니다."]

    if evidence_status == "VERIFIED":
        note = "근거 상태가 양호합니다. 남은 리스크 중심으로 비교하세요."
    elif evidence_status == "MIXED":
        note = "근거가 혼재되어 있습니다. 핵심 주장 1~2개를 우선 검증하세요."
    else:
        note = "근거가 부족합니다. 출처/수치/비교표를 보강한 뒤 확정하세요."

    return {
        "option": option,
        "pros": pros[:2],
        "risks": risks[:2],
        "evidence_note": note,
    }


def run_recommendation_engine(
    *,
    transcript: list[dict],
    analysis: dict,
    evidence_status: str,
) -> dict[str, Any]:
    turns_60 = _window_by_seconds(transcript, 60)
    turns_180 = _window_by_seconds(transcript, 180)
    context_text = " ".join(str(t.get("text") or "") for t in turns_180)

    info_count = _count_info_seeking(turns_60)
    trigger_a = info_count >= 2
    claim_count = len(((analysis.get("evidence_gate") or {}).get("claims") or []))
    trigger_b = evidence_status in {"MIXED", "UNVERIFIED"} and (claim_count >= 1 or len(turns_180) >= 4)
    trigger_c, obj_count, k3_count, k4_count = _slot_fulfillment(analysis, turns_180)

    triggered = trigger_a or trigger_b or trigger_c
    if not triggered:
        return {
            "r1_resources": [],
            "r2_options": [],
            "debug": {
                "triggered": False,
                "trigger_a_info_seeking": False,
                "trigger_b_evidence_weak": False,
                "trigger_c_slot_fulfillment": False,
                "info_signal_count_60s": info_count,
            "slot_counts": {"k1_object": obj_count, "k3_constraint": k3_count, "k4_criterion": k4_count},
            "claim_count": claim_count,
                "reason": "no_trigger",
            },
        }

    r1_resources: list[dict[str, str]] = []
    r2_options: list[dict[str, Any]] = []

    if trigger_a or trigger_b:
        r1_resources = _build_r1_resources(context_text, info_count, evidence_status)

    if trigger_c or trigger_b:
        options = _extract_options(analysis, context_text)
        r2_options = [_option_card(opt, evidence_status) for opt in options]

    return {
        "r1_resources": r1_resources[:3],
        "r2_options": r2_options[:2],
        "debug": {
            "triggered": True,
            "trigger_a_info_seeking": trigger_a,
            "trigger_b_evidence_weak": trigger_b,
            "trigger_c_slot_fulfillment": trigger_c,
            "info_signal_count_60s": info_count,
            "slot_counts": {"k1_object": obj_count, "k3_constraint": k3_count, "k4_criterion": k4_count},
            "claim_count": claim_count,
            "evidence_status": evidence_status,
            "shown_r1": len(r1_resources[:3]),
            "shown_r2": len(r2_options[:2]),
        },
    }
