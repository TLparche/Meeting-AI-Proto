from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from keyword_engine import build_keyword_engine_output
from schemas import ArtifactKind


def _safe_get_latest_text(transcript_window: List[dict]) -> str:
    if not transcript_window:
        return ""
    return (transcript_window[-1].get("text") or "").lower()


def build_analysis_template(
    meeting_goal: str,
    current_active_agenda: str,
    agenda_stack: List[dict],
    transcript_window: List[dict] | None = None,
) -> Dict[str, Any]:
    active_title = current_active_agenda or "의사결정 대상을 명확히 하기"
    candidate_titles = [item.get("title", "") for item in agenda_stack if item.get("title")]
    if not candidate_titles:
        candidate_titles = [
            "범위와 가정 정리",
            "후보 옵션 비교",
            "의사결정 책임자와 다음 단계 확정",
        ]

    keywords = build_keyword_engine_output(
        meeting_goal=meeting_goal,
        current_active_agenda=current_active_agenda,
        transcript_window=transcript_window or [],
    )

    return {
        "agenda": {
            "active": {"title": active_title, "status": "ACTIVE", "confidence": 0.5},
            "candidates": [
                {"title": title, "confidence": max(0.3, 0.9 - i * 0.2)}
                for i, title in enumerate(candidate_titles[:3])
            ],
        },
        "keywords": keywords,
        "scores": {
            "drift": {"score": 0, "band": "GREEN", "why": ""},
            "stagnation": {"score": 0, "why": ""},
            "participation": {"imbalance": 0, "fairtalk": []},
            "dps": {"score": 0, "why": ""},
        },
        "evidence_gate": {"status": "UNVERIFIED", "claims": []},
        "intervention": {
            "level": "L0",
            "banner_text": "",
            "decision_lock": {"triggered": False, "reason": ""},
        },
        "recommendations": {"r1_resources": [], "r2_options": []},
    }


def build_mock_analysis(
    meeting_goal: str,
    current_active_agenda: str,
    transcript_window: List[dict],
    agenda_stack: List[dict],
) -> Dict[str, Any]:
    latest_text = _safe_get_latest_text(transcript_window)
    utterance_count = len(transcript_window)

    drift_score = 20
    drift_band = "GREEN"
    drift_why = "대화 흐름이 현재 아젠다와 대체로 일치합니다."
    if (
        "later" in latest_text
        or "unrelated" in latest_text
        or "off topic" in latest_text
        or "off-topic" in latest_text
        or "tangent" in latest_text
        or "parking lot" in latest_text
        or "나중에" in latest_text
        or "딴 얘기" in latest_text
        or "주제 벗" in latest_text
    ):
        drift_score = 72
        drift_band = "RED"
        drift_why = "최근 발화가 의사결정과 직접 연결되지 않은 사이드 토픽을 제시합니다."
    elif (
        "maybe" in latest_text
        or "also" in latest_text
        or "아마" in latest_text
        or "또한" in latest_text
        or "한편" in latest_text
    ):
        drift_score = 46
        drift_band = "YELLOW"
        drift_why = "논의 분기가 일부 감지됩니다."

    stagnation_score = 18 if utterance_count < 4 else 38
    stagnation_why = "새로운 논점이 계속 등장하고 있습니다."
    if utterance_count >= 8:
        stagnation_score = 63
        stagnation_why = "반복 표현이 늘어 정체 루프 위험이 있습니다."

    dps_score = min(88, 20 + utterance_count * 7)
    dps_why = "옵션과 실행 항목이 구체화되며 진행도가 상승했습니다."

    imbalance = 25
    speaker_counts: Dict[str, int] = {}
    for item in transcript_window:
        speaker = item.get("speaker", "미상")
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
    if speaker_counts:
        max_turns = max(speaker_counts.values())
        min_turns = min(speaker_counts.values())
        if max_turns > 0:
            imbalance = int(((max_turns - min_turns) / max_turns) * 100)

    active_title = current_active_agenda or "의사결정 대상을 명확히 하기"
    candidate_titles = [item.get("title", "") for item in agenda_stack if item.get("title")]
    if not candidate_titles:
        candidate_titles = [
            "범위와 가정 정리",
            "후보 옵션 비교",
            "의사결정 책임자와 다음 단계 확정",
        ]

    decision_lock_triggered = dps_score >= 70 and drift_band != "RED"
    intervention_level = "L0"
    banner_text = ""
    if drift_band == "RED":
        intervention_level = "L2"
        banner_text = "짧게 정렬: 이 발화가 어떤 아젠다 결정에 기여하나요?"
    elif drift_band == "YELLOW" or stagnation_score >= 50:
        intervention_level = "L1"
        banner_text = "가벼운 유도: 계속하기 전에 새 인사이트 1가지를 요약해 주세요."

    evidence_status = "UNVERIFIED"
    if (
        "source" in latest_text
        or "data" in latest_text
        or "citation" in latest_text
        or "paper" in latest_text
        or "출처" in latest_text
        or "데이터" in latest_text
        or "근거" in latest_text
        or "링크" in latest_text
    ):
        evidence_status = "MIXED"
    if (
        "report link" in latest_text
        or "benchmark result" in latest_text
        or "보고서 링크" in latest_text
        or "벤치마크 결과" in latest_text
    ):
        evidence_status = "VERIFIED"

    keywords = build_keyword_engine_output(
        meeting_goal=meeting_goal,
        current_active_agenda=current_active_agenda,
        transcript_window=transcript_window,
    )

    return {
        "agenda": {
            "active": {
                "title": active_title,
                "status": "CLOSING" if dps_score > 75 else "ACTIVE",
                "confidence": 0.81,
            },
            "candidates": [
                {"title": title, "confidence": max(0.3, 0.9 - i * 0.2)}
                for i, title in enumerate(candidate_titles[:3])
            ],
        },
        "keywords": keywords,
        "scores": {
            "drift": {"score": drift_score, "band": drift_band, "why": drift_why},
            "stagnation": {"score": stagnation_score, "why": stagnation_why},
            "participation": {
                "imbalance": imbalance,
                "fairtalk": [
                    {"speaker": speaker, "p_intent": round(count / max(1, utterance_count), 2)}
                    for speaker, count in list(speaker_counts.items())[:5]
                ],
            },
            "dps": {"score": dps_score, "why": dps_why},
        },
        "evidence_gate": {
            "status": evidence_status,
            "claims": [
                {
                    "claim": "옵션 A가 온보딩 시간을 20% 단축한다",
                    "verifiability": 0.62,
                    "note": "파일럿 지표 출처가 필요합니다.",
                },
                {
                    "claim": "이해관계자들이 단계적 롤아웃을 선호한다",
                    "verifiability": 0.44,
                    "note": "정성 피드백 중심이라 추가 근거가 필요합니다.",
                },
            ],
        },
        "intervention": {
            "level": intervention_level,
            "banner_text": banner_text,
            "decision_lock": {
                "triggered": decision_lock_triggered,
                "reason": "의사결정 변수가 충분히 수렴되어 다음 단계 진행이 가능합니다.",
            },
        },
        "recommendations": {
            "r1_resources": [
                {
                    "title": "의사결정 매트릭스 템플릿",
                    "url": "https://www.atlassian.com/work-management/decision-matrix",
                    "reason": "가중 기준으로 옵션을 빠르게 비교할 수 있습니다.",
                },
                {
                    "title": "NIST 근거 품질 가이드",
                    "url": "https://www.nist.gov/",
                    "reason": "근거 상태가 약하거나 혼합일 때 점검 기준으로 사용하세요.",
                },
            ],
            "r2_options": [
                {
                    "option": "단계적 파일럿으로 진행",
                    "pros": ["롤아웃 리스크를 낮춤", "피드백 루프가 빠름"],
                    "risks": ["전체 효과 발현이 지연될 수 있음"],
                    "evidence_note": "운영팀 일정 확인이 필요합니다.",
                },
                {
                    "option": "즉시 전면 적용",
                    "pros": ["초기 효과를 크게 기대 가능"],
                    "risks": ["변화관리 리스크가 높음"],
                    "evidence_note": "도입 준비도 데이터 보강이 필요합니다.",
                },
            ],
        },
    }


def build_mock_artifact(kind: ArtifactKind, context: Dict[str, Any]) -> Dict[str, Any]:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    active_agenda = context.get("active_agenda") or "현재 의사결정 주제"
    if kind == ArtifactKind.MEETING_SUMMARY:
        return {
            "kind": kind.value,
            "title": "회의 요약",
            "markdown": f"### 요약 ({now})\n**{active_agenda}** 중심으로 논의가 수렴되었고 다음 단계가 명확해졌습니다.",
            "bullets": [
                "팀이 의사결정 대상과 제약을 명확히 했습니다.",
                "두 가지 옵션을 리스크와 함께 비교했습니다.",
                "파일럿 우선 경로로 공감대가 형성되었습니다.",
            ],
        }
    if kind == ArtifactKind.DECISION_RESULTS:
        return {
            "kind": kind.value,
            "title": "의사결정 결과",
            "markdown": "### 결정 스냅샷\n권고안: **단계적 파일럿 진행**",
            "bullets": [
                "담당자: 제품 리드",
                "목표 시점: 이번 스프린트 종료 전",
                "성공 기준: 온보딩 시간 단축 및 도입률",
            ],
        }
    if kind == ArtifactKind.ACTION_ITEMS:
        return {
            "kind": kind.value,
            "title": "액션 아이템",
            "markdown": "### 할당된 작업\n논의에서 바로 수행할 후속 작업입니다.",
            "bullets": [
                "핵심 주장에 대한 근거 링크 정리",
                "파일럿 계획 및 일정 초안 작성",
                "이해관계자 검증 체크인 일정 확정",
            ],
        }
    return {
        "kind": kind.value,
        "title": "근거 로그",
        "markdown": "### 근거 로그\n주장별 검증 가능 상태를 중립적으로 기록합니다.",
        "bullets": [
            "주장: 20% 개선 -> 상태: MIXED",
            "주장: 이해관계자 선호 -> 상태: UNVERIFIED",
            "주장: 롤아웃 리스크 완화 -> 상태: VERIFIED",
        ],
    }
