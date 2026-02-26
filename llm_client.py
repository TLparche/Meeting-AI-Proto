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

    def analyze_meeting(
        self,
        meeting_goal: str,
        initial_context: str,
        current_active_agenda: str,
        transcript_window: list[dict],
        agenda_stack: list[dict],
    ) -> AnalysisOutput:
        if self.mock_mode:
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
            merged_payload["keywords"] = engine_keywords
            return validate_analysis_payload(merged_payload)
        except (requests.RequestException, json.JSONDecodeError, ValidationError):
            return validate_analysis_payload(defaults)

    def generate_artifact(
        self,
        kind: ArtifactKind,
        meeting_goal: str,
        initial_context: str,
        transcript_window: list[dict],
        analysis: Optional[dict] = None,
    ) -> ArtifactOutput:
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
                return validate_artifact_payload(merged_payload)
            except (requests.RequestException, json.JSONDecodeError, ValidationError):
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
            return self._deep_merge(defaults, payload)
        except (requests.RequestException, json.JSONDecodeError, ValidationError):
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
            "규칙:\n"
            "- 서술은 중립적으로 작성합니다.\n"
            "- 입력 전사는 한국어 중심이며 일부 영어 단어가 혼용될 수 있습니다.\n"
            "- evidence status는 사실 판정이 아니라 검증 상태를 의미합니다.\n"
            "- banner_text는 짧게 작성합니다.\n"
            "- enum 값은 스키마에 지정된 영문 값 그대로 사용합니다.\n"
            "- 수치 범위는 스키마 제한을 준수합니다.\n"
            "- enum/URL/고유 키를 제외한 자유 텍스트 값은 한국어 중심으로 작성합니다.\n"
            "- 제품명/고유명사/기술 용어는 영어를 유지해도 됩니다.\n\n"
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
        response.raise_for_status()
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
