from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv
from pydantic import ValidationError

from keyword_engine import build_keyword_engine_output
from mock_data import build_analysis_template, build_mock_analysis, build_mock_artifact
from schemas import (
    AnalysisOutput,
    ArtifactKind,
    ArtifactOutput,
    validate_analysis_payload,
    validate_artifact_payload,
)


load_dotenv()


ANALYSIS_JSON_SHAPE = {
    "agenda": {
        "active": {"title": "string", "status": "ACTIVE|CLOSING|CLOSED", "confidence": "0-1"},
        "candidates": [{"title": "string", "confidence": "0-1"}],
    },
    "agenda_outcomes": [
        {
            "agenda_title": "string",
            "agenda_state": "PROPOSED|ACTIVE|CLOSING|CLOSED",
            "flow_type": "문제정의|대안비교|의견충돌|근거검토|결론수렴|실행정의",
            "key_utterances": ["string", "string"],
            "summary": "string",
            "decision_results": [
                {
                    "decision": "string",
                    "opinions": ["string", "string"],
                    "conclusion": "string",
                }
            ],
            "action_items": [
                {
                    "item": "string",
                    "owner": "string",
                    "due": "string",
                    "reasons": [
                        {
                            "speaker": "string",
                            "timestamp": "HH:MM:SS|''",
                            "quote": "string",
                            "why": "string",
                        }
                    ],
                }
            ],
        }
    ],
    "keywords": {
        "taxonomy": {
            "K1_OBJECT": "string",
            "K2_OPTION": "string",
            "K3_CONSTRAINT": "string",
            "K4_CRITERION": "string",
            "K5_EVIDENCE": "string",
            "K6_ACTION": "string",
        },
        "k_core": {"object": ["string"], "constraints": ["string"], "criteria": ["string"]},
        "k_facet": {"options": ["string"], "evidence": ["string"], "actions": ["string"]},
        "items": [
            {
                "keyword": "string",
                "type": "K1_OBJECT|K2_OPTION|K3_CONSTRAINT|K4_CRITERION|K5_EVIDENCE|K6_ACTION",
                "score": "0-1",
                "first_seen": "HH:MM:SS",
                "frequency": ">=1",
                "decision_value": "0-1",
                "evidence_boost": "0-1",
                "is_core": "bool",
            }
        ],
        "pipeline": {
            "candidates": [{"keyword": "string", "frequency": ">=1", "first_seen": "HH:MM:SS"}],
            "classification": [{"keyword": "string", "type": "K1_OBJECT|...|K6_ACTION"}],
            "scoring": [{"keyword": "string", "decision_value": "0-1", "evidence_boost": "0-1", "score": "0-1"}],
            "final_selection": {
                "k_core_required": ["string"],
                "k_facet_target": "3~8",
                "selected_core": ["string"],
                "selected_facet": ["string"],
                "diversity_boost_applied": "bool",
            },
        },
        "summary": {"object_focus": "string", "core_count": ">=0", "facet_count": ">=0"},
    },
    "scores": {
        "drift": {"score": "0-100", "band": "GREEN|YELLOW|RED", "why": "string"},
        "stagnation": {"score": "0-100", "why": "string"},
        "participation": {
            "imbalance": "0-100",
            "fairtalk": [{"speaker": "string", "p_intent": "0-1"}],
        },
        "dps": {"score": "0-1", "why": "string"},
    },
    "evidence_gate": {
        "status": "VERIFIED|MIXED|UNVERIFIED",
        "claims": [{"claim": "string", "verifiability": "0-1", "note": "string"}],
    },
    "intervention": {
        "level": "L0|L1|L2",
        "banner_text": "string",
        "decision_lock": {"triggered": "bool", "reason": "string"},
    },
    "recommendations": {
        "r1_resources": [{"title": "string", "url": "https://...", "reason": "string"}],
        "r2_options": [
            {
                "option": "string",
                "pros": ["string"],
                "risks": ["string"],
                "evidence_note": "string",
            }
        ],
    },
}

CONTROL_PLANE_JSON_SHAPE = {
    "keywords": ANALYSIS_JSON_SHAPE["keywords"],
    "agenda_tracker": {
        "agenda_candidates": [
            {
                "title": "string",
                "status": "PROPOSED",
                "confidence": "0-1",
                "created_at": "HH:MM:SS",
                "reasons": ["string"],
                "sub_issue_promoted": "bool",
                "signals": {
                    "sim60": "0-1",
                    "sim120": "0-1",
                    "duration120_sec": ">=0",
                    "intent_count_60s": ">=0",
                    "speakers_60s": ">=0",
                    "object_slots": ">=0",
                    "support_slots": ">=0",
                },
            }
        ],
        "agenda_vectors": {"agenda_title": {"updated_at": "HH:MM:SS", "sample_count": ">=0", "terms": []}},
        "tracker_debug": {"topic_shift_sustained": "bool", "collective_intent": "bool", "decision_slots": "bool"},
    },
    "agenda_fsm": {
        "agenda_state_map": {
            "agenda_id": {
                "agenda_id": "string",
                "title": "string",
                "state": "PROPOSED|ACTIVE|CLOSING|CLOSED",
                "confidence": "0-1",
                "created_at": "HH:MM:SS",
                "updated_at": "HH:MM:SS",
                "source": "llm_control_plane",
            }
        },
        "active_agenda_id": "string",
        "agenda_events": [
            {
                "ts": "HH:MM:SS",
                "type": "active_agenda_changed",
                "agenda_id": "string",
                "title": "string",
                "reason": "string",
                "active_before": "string|null",
                "active_after": "string|null",
            }
        ],
    },
    "drift_dampener": {
        "drift_state": "Normal|Yellow|Red|Re-orient",
        "drift_ui_cues": {"glow_k_core": "bool", "fix_k_core_focus": "bool", "reduce_facets": "bool", "show_banner": "bool"},
        "drift_debug": {"s45": "0-1", "band": "Green|Yellow|Red", "yellow_seconds": ">=0", "red_seconds": ">=0"},
    },
    "dps": {
        "dps_t": "0-1",
        "dps_breakdown": {
            "option_coverage": "0-1",
            "constraint_coverage": "0-1",
            "evidence_coverage": "0-1",
            "tradeoff_coverage": "0-1",
            "closing_readiness": "0-1",
            "counts": {"options": ">=0", "constraints": ">=0", "criteria": ">=0", "evidence": ">=0", "actions": ">=0"},
            "active_state": "string",
            "decision_lock": "bool",
        },
        "why": "string",
    },
    "flow_pulse": {
        "stagnation_flag": "bool",
        "loop_state": "Normal|Watching|Looping|Anchoring",
        "flow_pulse_debug": {
            "novelty_rate_3m": "0-1",
            "arg_novelty": "0-1",
            "delta_dps": "-1~1",
            "anchor_ratio": "0-1",
            "conditions": {
                "surface_repetition_a": "bool",
                "content_repetition_b": "bool",
                "no_progress_c": "bool",
                "anchoring_exception": "bool",
            },
            "turns_3m": ">=0",
        },
    },
    "decision_lock": {
        "triggered": "bool",
        "reason": "string",
        "decision_lock_debug": {
            "stance_convergence": "0-1",
            "stance_total": ">=0",
            "agree_count": ">=0",
            "disagree_count": ">=0",
            "elapsed_sec": ">=0",
            "trigger_stance": "bool",
            "trigger_stagnation": "bool",
            "trigger_timebox": "bool",
        },
    },
}


@dataclass
class ClientConfig:
    api_key: str = ""
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    model: str = "gemini-2.0-flash"
    timeout_seconds: int = 45


class GeminiClient:
    def __init__(self, config: Optional[ClientConfig] = None):
        cfg = config or ClientConfig(
            api_key=os.getenv("GOOGLE_API_KEY", "").strip(),
            base_url=(os.getenv("GOOGLE_BASE_URL", "").strip() or "https://generativelanguage.googleapis.com/v1beta"),
            model=(os.getenv("MODEL", "").strip() or "gemini-2.0-flash"),
        )
        self.config = cfg
        self.mock_mode = not bool(cfg.api_key)
        self._session = requests.Session()
        self._last_request_at: str = ""
        self._last_success_at: str = ""
        self._last_error: str = ""
        self._last_error_at: str = ""
        self._last_operation: str = ""
        self._request_count: int = 0
        self._success_count: int = 0
        self._error_count: int = 0

    def get_status(self) -> Dict[str, Any]:
        note = ""
        connected = bool(self.config.api_key)
        if self.mock_mode:
            connected = False
            note = "GOOGLE_API_KEY가 없어 mock/fallback 모드로 동작 중입니다."
        elif self._last_error:
            connected = False
            note = "LLM 호출 실패 후 기본값으로 폴백 중입니다."
        elif not self._last_success_at:
            connected = False
            note = "아직 LLM 호출 성공 이력이 없습니다."
        else:
            note = "LLM 연결 정상"
        return {
            "provider": "gemini",
            "model": self.config.model,
            "base_url": self.config.base_url,
            "mode": "mock" if self.mock_mode else "live",
            "api_key_present": bool(self.config.api_key),
            "connected": connected,
            "note": note,
            "request_count": self._request_count,
            "success_count": self._success_count,
            "error_count": self._error_count,
            "last_operation": self._last_operation,
            "last_request_at": self._last_request_at,
            "last_success_at": self._last_success_at,
            "last_error": self._last_error,
            "last_error_at": self._last_error_at,
        }

    def _mark_request(self, operation: str) -> None:
        self._last_operation = operation
        self._last_request_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._request_count += 1

    def _mark_success(self) -> None:
        self._last_success_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._last_error = ""
        self._last_error_at = ""
        self._success_count += 1

    def _mark_error(self, exc: Exception) -> None:
        self._last_error = str(exc)[:500]
        self._last_error_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._error_count += 1

    def ping(self) -> Dict[str, Any]:
        self._mark_request("ping")
        if self.mock_mode:
            self._last_error = "LLM 미연결: GOOGLE_API_KEY 없음 (mock mode)"
            self._last_error_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._error_count += 1
            return {
                "ok": False,
                "message": self._last_error,
                "mode": "mock",
            }
        try:
            raw_text = self._generate_content(
                prompt=(
                    "다음 JSON 객체를 그대로 반환하세요: "
                    '{"ok": true, "pong": "ready"}'
                )
            )
            parsed = self._parse_json_with_repair(raw_text)
            self._mark_success()
            return {
                "ok": True,
                "message": "LLM 응답 성공",
                "mode": "live",
                "response_preview": parsed,
            }
        except (requests.RequestException, json.JSONDecodeError, ValidationError) as exc:
            self._mark_error(exc)
            return {
                "ok": False,
                "message": str(exc)[:500],
                "mode": "live",
            }

    def analyze_meeting(
        self,
        meeting_goal: str,
        initial_context: str,
        current_active_agenda: str,
        transcript_window: list[dict],
        agenda_stack: list[dict],
    ) -> AnalysisOutput:
        self._mark_request("analyze_meeting")
        if self.mock_mode:
            self._last_error = "LLM 미연결: GOOGLE_API_KEY 없음 (mock mode)"
            return validate_analysis_payload(
                build_mock_analysis(
                    meeting_goal=meeting_goal,
                    current_active_agenda=current_active_agenda,
                    transcript_window=transcript_window,
                    agenda_stack=agenda_stack,
                )
            )
        defaults = build_analysis_template(
            meeting_goal=meeting_goal,
            current_active_agenda=current_active_agenda,
            agenda_stack=agenda_stack,
            transcript_window=transcript_window,
        )
        engine_keywords = build_keyword_engine_output(
            meeting_goal=meeting_goal,
            current_active_agenda=current_active_agenda,
            transcript_window=transcript_window,
        )

        prompt = self._analysis_prompt(
            meeting_goal=meeting_goal,
            initial_context=initial_context,
            current_active_agenda=current_active_agenda,
            transcript_window=transcript_window,
            agenda_stack=agenda_stack,
        )

        try:
            raw_text = self._generate_content(prompt=prompt)
            payload = self._parse_json_with_repair(raw_text)
            merged_payload = self._deep_merge(defaults, payload)
            if not merged_payload.get("keywords"):
                merged_payload["keywords"] = engine_keywords
            self._mark_success()
            return validate_analysis_payload(merged_payload)
        except (requests.RequestException, json.JSONDecodeError, ValidationError) as exc:
            self._mark_error(exc)
            return validate_analysis_payload(defaults)

    def generate_artifact(
        self,
        kind: ArtifactKind,
        meeting_goal: str,
        initial_context: str,
        transcript_window: list[dict],
        analysis: Optional[dict] = None,
    ) -> ArtifactOutput:
        self._mark_request("generate_artifact")
        fallback = build_mock_artifact(
            kind=kind,
            context={
                "meeting_goal": meeting_goal,
                "initial_context": initial_context,
                "active_agenda": (analysis or {}).get("agenda", {}).get("active", {}).get("title", ""),
                "transcript_window": transcript_window,
            },
        )
        if self.mock_mode:
            self._last_error = "LLM 미연결: GOOGLE_API_KEY 없음 (mock mode)"
            return validate_artifact_payload(fallback)

        prompt = self._artifact_prompt(
            kind=kind,
            meeting_goal=meeting_goal,
            initial_context=initial_context,
            transcript_window=transcript_window,
            analysis=analysis or {},
        )
        attempts = 2
        for _ in range(attempts):
            try:
                raw_text = self._generate_content(prompt=prompt)
                parsed = self._parse_json_with_repair(raw_text)
                merged_payload = self._deep_merge(fallback, parsed)
                self._mark_success()
                return validate_artifact_payload(merged_payload)
            except (requests.RequestException, json.JSONDecodeError, ValidationError) as exc:
                self._mark_error(exc)
                continue

        return validate_artifact_payload(fallback)

    def infer_control_plane(
        self,
        *,
        meeting_goal: str,
        initial_context: str,
        current_active_agenda: str,
        transcript_window: list[dict],
        agenda_stack: list[dict],
        analysis: dict,
        previous_state: dict[str, Any],
    ) -> Dict[str, Any]:
        self._mark_request("infer_control_plane")
        seed_keywords = build_keyword_engine_output(
            meeting_goal=meeting_goal,
            current_active_agenda=current_active_agenda,
            transcript_window=transcript_window,
        )
        now_hms = datetime.now().strftime("%H:%M:%S")
        active_title = (
            current_active_agenda
            or str(((analysis.get("agenda") or {}).get("active") or {}).get("title") or "").strip()
            or "핵심 의사결정 아젠다"
        )
        prev_map = dict(previous_state.get("agenda_state_map") or {})
        active_id = str(previous_state.get("active_agenda_id") or "")
        if not active_id:
            active_id = "agenda-1"
        if not prev_map:
            prev_map = {
                active_id: {
                    "agenda_id": active_id,
                    "title": active_title,
                    "state": "ACTIVE",
                    "confidence": 0.62,
                    "created_at": now_hms,
                    "updated_at": now_hms,
                    "source": "llm_control_plane_default",
                }
            }

        defaults: Dict[str, Any] = {
            "_meta": {
                "source": "default_seed",
                "reason": "seeded_from_previous_state",
            },
            "keywords": seed_keywords,
            "agenda_tracker": {
                "agenda_candidates": list(previous_state.get("agenda_candidates") or []),
                "agenda_vectors": dict(previous_state.get("agenda_vectors") or {}),
                "tracker_debug": dict(previous_state.get("agenda_tracker_debug") or {}),
            },
            "agenda_fsm": {
                "agenda_state_map": prev_map,
                "active_agenda_id": active_id,
                "agenda_events": list(previous_state.get("agenda_events") or []),
            },
            "drift_dampener": {
                "drift_state": str(previous_state.get("drift_state") or "Normal"),
                "drift_ui_cues": dict(
                    previous_state.get("drift_ui_cues")
                    or {"glow_k_core": False, "fix_k_core_focus": False, "reduce_facets": False, "show_banner": False}
                ),
                "drift_debug": dict(previous_state.get("drift_debug") or {}),
            },
            "dps": {
                "dps_t": float(previous_state.get("dps_t") or 0.0),
                "dps_breakdown": dict(
                    previous_state.get("dps_breakdown")
                    or {
                        "option_coverage": 0.0,
                        "constraint_coverage": 0.0,
                        "evidence_coverage": 0.0,
                        "tradeoff_coverage": 0.0,
                        "closing_readiness": 0.0,
                        "counts": {"options": 0, "constraints": 0, "criteria": 0, "evidence": 0, "actions": 0},
                        "active_state": "ACTIVE",
                        "decision_lock": False,
                    }
                ),
                "why": str((((analysis.get("scores") or {}).get("dps") or {}).get("why") or "")),
            },
            "flow_pulse": {
                "stagnation_flag": bool(previous_state.get("stagnation_flag") or False),
                "loop_state": str(previous_state.get("loop_state") or "Normal"),
                "flow_pulse_debug": dict(previous_state.get("flow_pulse_debug") or {}),
            },
            "decision_lock": {
                "triggered": bool((((analysis.get("intervention") or {}).get("decision_lock") or {}).get("triggered")) or False),
                "reason": str((((analysis.get("intervention") or {}).get("decision_lock") or {}).get("reason")) or ""),
                "decision_lock_debug": dict(previous_state.get("decision_lock_debug") or {}),
            },
        }
        if self.mock_mode:
            self._last_error = "LLM 미연결: GOOGLE_API_KEY 없음 (mock mode)"
            defaults["_meta"] = {
                "source": "fallback_mock",
                "reason": self._last_error,
            }
            return defaults

        prompt = self._control_plane_prompt(
            meeting_goal=meeting_goal,
            initial_context=initial_context,
            current_active_agenda=current_active_agenda,
            transcript_window=transcript_window,
            agenda_stack=agenda_stack,
            current_analysis=analysis,
            previous_state=previous_state,
        )
        try:
            raw_text = self._generate_content(prompt=prompt)
            payload = self._parse_json_with_repair(raw_text)
            self._mark_success()
            merged = self._deep_merge(defaults, payload)
            meta = dict(merged.get("_meta") or {})
            meta["source"] = "llm"
            meta["reason"] = "llm_response_merged"
            merged["_meta"] = meta
            return merged
        except (requests.RequestException, json.JSONDecodeError, ValidationError) as exc:
            self._mark_error(exc)
            defaults["_meta"] = {
                "source": "fallback_error",
                "reason": str(exc)[:300],
            }
            return defaults

    def _analysis_prompt(
        self,
        meeting_goal: str,
        initial_context: str,
        current_active_agenda: str,
        transcript_window: list[dict],
        agenda_stack: list[dict],
    ) -> str:
        return (
            "당신은 실시간 회의 리듬 분석기입니다.\n"
            "반드시 JSON 객체 하나만 반환하고 다른 텍스트는 출력하지 마세요.\n"
            "이 시스템의 목적은 단순 키워드 추출이 아니라, '아젠다 흐름 + 의사결정 + 실행항목'을 구조화하는 것입니다.\n"
            "규칙:\n"
            "- 서술은 중립적으로 작성합니다.\n"
            "- 입력 전사는 한국어 중심이며 일부 영어 단어가 혼용될 수 있습니다.\n"
            "- evidence status는 사실 판정이 아니라 검증 상태를 의미합니다.\n"
            "- banner_text는 짧게 작성합니다.\n"
            "- enum 값은 스키마에 지정된 영문 값 그대로 사용합니다.\n"
            "- 수치 범위는 스키마 제한을 준수합니다.\n"
            "- enum/URL/고유 키를 제외한 자유 텍스트 값은 한국어 중심으로 작성합니다.\n"
            "- 제품명/고유명사/기술 용어는 영어를 유지해도 됩니다.\n\n"
            "핵심 해석 지침(필수):\n"
            "1) agenda 추출:\n"
            "- 회의록에서 시간 순으로 아젠다 흐름을 읽어 agenda.candidates에 반영합니다.\n"
            "- agenda.active는 현재 가장 중심인 아젠다 1개를 선택합니다.\n"
            "- 단순 주제가 아니라 '결론 가능 단위'를 아젠다로 봅니다.\n"
            "- 같은 topic이라도 주요 흐름(문제정의/대안비교/의견충돌/근거검토/결론수렴/실행정의)이 달라지면 agenda_outcomes에 별도 항목으로 분리합니다.\n"
            "- agenda_title은 topic만 쓰지 말고, 해당 흐름의 요지를 포함해 구체적으로 작성합니다.\n"
            "- agenda_outcomes[*].flow_type은 다음 중 하나로 채웁니다: 문제정의/대안비교/의견충돌/근거검토/결론수렴/실행정의\n"
            "- agenda_outcomes[*].key_utterances에는 해당 아젠다를 대표하는 핵심 발언 2~4개를 넣습니다.\n"
            "2) 아젠다별 의사결정 요약:\n"
            "- 한 아젠다 안에서 어떤 의견/대안이 있었는지 K2_OPTION, r2_options.pros/risks에 반영합니다.\n"
            "- 결론/합의가 보이면 intervention.decision_lock.reason, scores.dps.why에 요약합니다.\n"
            "- 반드시 agenda_outcomes[*].decision_results에 다건으로 정리합니다(아젠다당 여러 결정 허용).\n"
            "3) 액션아이템 추출:\n"
            "- '~~까지 ~~하자', '누가 언제 무엇을' 형태 발화를 K6_ACTION으로 우선 추출합니다.\n"
            "- 후속 실행 항목은 keywords.k_facet.actions와 keywords.items(type=K6_ACTION)에 반영합니다.\n"
            "- 반드시 agenda_outcomes[*].action_items에 다건으로 정리합니다(아젠다당 여러 액션 허용).\n"
            "4) 액션아이템 근거 로그:\n"
            "- 왜 그 액션이 필요한지 설명하는 발언을 evidence_gate.claims에 넣습니다.\n"
            "- evidence_gate.claims.note에는 근거 발화 요지를 짧게 작성합니다.\n"
            "- 각 action_item마다 reasons를 여러 개 둘 수 있습니다(한 액션당 근거 다건 허용).\n"
            "5) 요약 우선순위:\n"
            "- 말이 많은 부분보다 '결정/대안/제약/근거/실행' 신호가 강한 발화를 우선 반영합니다.\n\n"
            "필드 매핑 규칙(엄수):\n"
            "- agenda.active/candidates: 아젠다 흐름\n"
            "- agenda_outcomes: 아젠다별 결정/액션/근거 로그의 계층 구조(핵심)\n"
            "- agenda_outcomes.flow_type/key_utterances: 아젠다를 단순 topic이 아닌 '흐름+핵심발언'으로 설명\n"
            "- keywords.items: K1~K6 결정 변수\n"
            "- keywords.k_facet.actions: 후속 실행 항목 요약\n"
            "- recommendations.r2_options: 아젠다 내 대안 비교(의견 구조)\n"
            "- evidence_gate.claims: 액션/결정의 근거 발언 로그\n"
            "- intervention.decision_lock.reason: 결론 상태 한 줄 요약\n\n"
            f"요구 JSON 형태:\n{json.dumps(ANALYSIS_JSON_SHAPE, indent=2)}\n\n"
            f"meeting_goal:\n{meeting_goal}\n\n"
            f"initial_context:\n{initial_context}\n\n"
            f"current_active_agenda:\n{current_active_agenda}\n\n"
            f"recent_transcript_window:\n{json.dumps(transcript_window, ensure_ascii=False, indent=2)}\n\n"
            f"current_agenda_stack:\n{json.dumps(agenda_stack, ensure_ascii=False, indent=2)}\n"
        )

    def _artifact_prompt(
        self,
        kind: ArtifactKind,
        meeting_goal: str,
        initial_context: str,
        transcript_window: list[dict],
        analysis: dict,
    ) -> str:
        output_schema = {
            "kind": kind.value,
            "title": "string",
            "markdown": "string",
            "bullets": ["string"],
        }
        return (
            "당신은 회의 산출물 1개를 생성합니다.\n"
            "반드시 JSON 객체 하나만 반환하고 다른 텍스트는 출력하지 마세요.\n"
            f"kind는 정확히 다음 값이어야 합니다: {kind.value}\n"
            "간결하고 실행 가능한 내용을 작성하세요.\n"
            "입력은 한국어 중심 + 영어 혼용일 수 있습니다.\n"
            "enum/URL/고유 키를 제외한 자유 텍스트 값은 한국어 중심으로 작성하세요.\n"
            "제품명/고유명사/기술 용어는 영어를 유지해도 됩니다.\n\n"
            "산출물 작성 지침:\n"
            "- meeting_summary: 아젠다 흐름(처음→전환→현재)과 아젠다별 결론 상태를 요약\n"
            "- decision_results: 아젠다별 주요 의견(찬반/대안)과 최종 결론을 분리해서 정리\n"
            "- action_items: '누가/언제/무엇' 중심으로 실행 항목만 짧게 나열\n"
            "- evidence_log: 각 액션아이템/결론의 이유가 된 발언 요지를 근거로 기록\n"
            "- 가능하면 대화록 표현(예: '~~까지 ~~하자')을 실행항목 문장으로 정규화\n\n"
            f"요구 JSON 형태:\n{json.dumps(output_schema, indent=2)}\n\n"
            f"meeting_goal:\n{meeting_goal}\n\n"
            f"initial_context:\n{initial_context}\n\n"
            f"analysis_snapshot:\n{json.dumps(analysis, ensure_ascii=False, indent=2)}\n\n"
            f"recent_transcript_window:\n{json.dumps(transcript_window, ensure_ascii=False, indent=2)}\n"
        )

    def _control_plane_prompt(
        self,
        *,
        meeting_goal: str,
        initial_context: str,
        current_active_agenda: str,
        transcript_window: list[dict],
        agenda_stack: list[dict],
        current_analysis: dict,
        previous_state: dict[str, Any],
    ) -> str:
        return (
            "당신은 회의 제어 플레인 추론기입니다.\n"
            "반드시 JSON 객체 하나만 반환하고 설명 텍스트를 출력하지 마세요.\n"
            "목표: 아래 모듈을 '1회 통합 추론'으로 업데이트합니다.\n"
            "- keyword engine\n"
            "- agenda tracker\n"
            "- agenda fsm\n"
            "- drift dampener\n"
            "- dps\n"
            "- flow pulse\n"
            "- decision lock\n"
            "규칙:\n"
            "- 숫자는 안정적으로, 급격한 점프를 피합니다.\n"
            "- 이전 상태(previous_state)와 연속성을 유지합니다.\n"
            "- 근거 없는 확신을 피하고 reason/debug를 간단히 남깁니다.\n"
            "- enum 값은 요구 형태를 정확히 따릅니다.\n"
            "- 자유 텍스트는 한국어 중심으로 작성하세요.\n\n"
            "제어 로직 지침(강화):\n"
            "1) agenda_tracker:\n"
            "- 회의록에서 아젠다 전환 흐름을 추적해 agenda_candidates를 업데이트\n"
            "- 아젠다는 '결론 단위' 기준으로 분리\n"
            "2) agenda_fsm:\n"
            "- ACTIVE 아젠다 1개를 중심으로 상태 전이(PROPOSED→ACTIVE→CLOSING→CLOSED) 정합성 유지\n"
            "3) dps/decision_lock:\n"
            "- 한 아젠다의 의견 분포와 결론 신호를 반영해 진행도와 lock 여부 판단\n"
            "4) action 중심 반영:\n"
            "- 실행 약속(예: '~~까지 ~~하자')이 나오면 K6/ACTION 관련 점수와 상태에 반영\n"
            "5) evidence 연결:\n"
            "- 액션/결정의 이유 발언은 evidence 관련 신호에 반영해 과도한 확신을 방지\n\n"
            f"요구 JSON 형태:\n{json.dumps(CONTROL_PLANE_JSON_SHAPE, indent=2)}\n\n"
            f"meeting_goal:\n{meeting_goal}\n\n"
            f"initial_context:\n{initial_context}\n\n"
            f"current_active_agenda:\n{current_active_agenda}\n\n"
            f"recent_transcript_window:\n{json.dumps(transcript_window, ensure_ascii=False, indent=2)}\n\n"
            f"current_agenda_stack:\n{json.dumps(agenda_stack, ensure_ascii=False, indent=2)}\n\n"
            f"current_analysis_snapshot:\n{json.dumps(current_analysis, ensure_ascii=False, indent=2)}\n\n"
            f"previous_state:\n{json.dumps(previous_state, ensure_ascii=False, indent=2)}\n"
        )

    def _generate_content(self, prompt: str) -> str:
        endpoint = f"{self.config.base_url.rstrip('/')}/models/{self.config.model}:generateContent"
        params = {"key": self.config.api_key}
        payload = {
            "systemInstruction": {
                "parts": [
                    {
                        "text": (
                            "엄격한 JSON만 출력하세요. "
                            "마크다운 래핑이나 설명 문장을 추가하지 마세요."
                        )
                    }
                ]
            },
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
            },
        }
        response = self._session.post(
            endpoint,
            params=params,
            json=payload,
            timeout=self.config.timeout_seconds,
        )
        if response.status_code >= 400:
            body_preview = (response.text or "").strip()
            if len(body_preview) > 500:
                body_preview = body_preview[:500] + "...(truncated)"
            raise requests.HTTPError(
                f"Gemini HTTP {response.status_code}: {body_preview}",
                response=response,
            )
        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise json.JSONDecodeError("No candidates in Gemini response", "", 0)
        parts = ((candidates[0].get("content") or {}).get("parts")) or []
        if not parts:
            raise json.JSONDecodeError("No content parts in Gemini response", "", 0)
        text = parts[0].get("text")
        if not isinstance(text, str) or not text.strip():
            raise json.JSONDecodeError("Empty text in Gemini response", "", 0)
        return text

    def _parse_json_with_repair(self, raw_text: str) -> Dict[str, Any]:
        cleaned = raw_text.strip()
        cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^```\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                raise json.JSONDecodeError("Root JSON must be object", cleaned, 0)
            return parsed
        except json.JSONDecodeError:
            pass

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise json.JSONDecodeError("Could not locate JSON object", cleaned, 0)

        candidate = cleaned[start : end + 1]
        candidate = candidate.replace("\u201c", '"').replace("\u201d", '"')
        candidate = candidate.replace("\u2018", "'").replace("\u2019", "'")
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            raise json.JSONDecodeError("Root JSON must be object", candidate, 0)
        return parsed

    def _deep_merge(self, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        for key, value in updates.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged


def get_client() -> GeminiClient:
    return GeminiClient()
