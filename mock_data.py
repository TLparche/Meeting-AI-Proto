from __future__ import annotations

from typing import Any, Dict, List


def build_analysis_template(
    meeting_goal: str,
    current_active_agenda: str,
    agenda_stack: List[dict],
    transcript_window: List[dict] | None = None,
) -> Dict[str, Any]:
    transcript_window = transcript_window or []
    active_title = current_active_agenda or "핵심 아젠다 정리"
    candidate_titles = [str(item.get("title") or "").strip() for item in agenda_stack if str(item.get("title") or "").strip()]
    if not candidate_titles:
        candidate_titles = [active_title]

    outcomes = []
    for i, title in enumerate(candidate_titles[:8]):
        outcomes.append(
            {
                "agenda_title": title,
                "key_utterances": [],
                "summary": f"{title} 관련 발언을 정리 중입니다.",
                "agenda_keywords": [],
                "decision_results": [],
                "action_items": [],
            }
        )

    return {
        "agenda": {
            "active": {"title": active_title, "confidence": 0.5},
            "candidates": [{"title": t, "confidence": max(0.3, 0.9 - i * 0.1)} for i, t in enumerate(candidate_titles[:10])],
        },
        "agenda_outcomes": outcomes,
        "evidence_gate": {"claims": []},
    }


def build_mock_analysis(
    meeting_goal: str,
    current_active_agenda: str,
    transcript_window: List[dict],
    agenda_stack: List[dict],
) -> Dict[str, Any]:
    base = build_analysis_template(
        meeting_goal=meeting_goal,
        current_active_agenda=current_active_agenda,
        agenda_stack=agenda_stack,
        transcript_window=transcript_window,
    )
    if base["agenda_outcomes"]:
        base["agenda_outcomes"][0]["key_utterances"] = [
            "핵심 안건에 대한 입장 차이를 먼저 정리하자",
            "결론 전에 실행 책임자와 기한을 확정하자",
        ]
        base["agenda_outcomes"][0]["summary"] = "핵심 안건의 의견 차이를 좁히고 결론 조건을 확인 중입니다."
        base["agenda_outcomes"][0]["decision_results"] = [
            {
                "opinions": [
                    "파일럿 후 확장", "즉시 전체 적용"
                ],
                "conclusion": "리스크 완화를 위해 파일럿 우선",
            }
        ]
        base["agenda_outcomes"][0]["action_items"] = [
            {
                "item": "이번 주 금요일까지 파일럿 계획안 작성",
                "owner": "제품 리드",
                "due": "이번 주 금요일",
                "reasons": [
                    {
                        "speaker": "화자1",
                        "timestamp": "",
                        "quote": "작게 시작하고 지표를 보고 확장하자",
                        "why": "전면 적용 리스크를 낮추기 위해",
                    }
                ],
            }
        ]
    base["evidence_gate"] = {
        "claims": [
            {
                "claim": "파일럿이 비용 리스크를 줄인다",
                "verifiability": 0.64,
                "note": "비용 추정 근거 문서 확인 필요",
            }
        ]
    }
    return base
