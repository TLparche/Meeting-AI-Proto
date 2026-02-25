from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv
from pydantic import ValidationError

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
        "k_core": {"object": ["string"], "constraints": ["string"], "criteria": ["string"]},
        "k_facet": {"options": ["string"], "evidence": ["string"], "actions": ["string"]},
    },
    "scores": {
        "drift": {"score": "0-100", "band": "GREEN|YELLOW|RED", "why": "string"},
        "stagnation": {"score": "0-100", "why": "string"},
        "participation": {
            "imbalance": "0-100",
            "fairtalk": [{"speaker": "string", "p_intent": "0-1"}],
        },
        "dps": {"score": "0-100", "why": "string"},
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
