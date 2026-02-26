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

from mock_data import build_analysis_template, build_mock_analysis
from schemas import AnalysisOutput, validate_analysis_payload


load_dotenv()


ANALYSIS_JSON_SHAPE = {
    "agenda": {
        "active": {"title": "string", "confidence": "0-1"},
        "candidates": [{"title": "string", "confidence": "0-1"}],
    },
    "agenda_outcomes": [
        {
            "agenda_title": "string",
            "key_utterances": ["string", "string"],
            "summary": "string",
            "agenda_keywords": ["string", "string"],
            "decision_results": [
                {
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
    "evidence_gate": {
        "claims": [{"claim": "string", "verifiability": "0-1", "note": "string"}],
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
            self._mark_success()
            return validate_analysis_payload(merged_payload)
        except (requests.RequestException, json.JSONDecodeError, ValidationError) as exc:
            self._mark_error(exc)
            return validate_analysis_payload(defaults)

    def _analysis_prompt(
        self,
        meeting_goal: str,
        initial_context: str,
        current_active_agenda: str,
        transcript_window: list[dict],
        agenda_stack: list[dict],
    ) -> str:
        return (
            "당신은 회의 분석기입니다.\\n"
            "반드시 JSON 객체 하나만 반환하고 다른 텍스트는 출력하지 마세요.\\n"
            "입력 전사는 한국어 중심이며 일부 영어 단어가 혼용될 수 있습니다.\\n"
            "요구사항:\\n"
            "1) agenda에는 현재 중심 아젠다 제목과 후보 아젠다만 반환합니다.\\n"
            "2) agenda_outcomes에는 아젠다별로 다음만 반환합니다:\\n"
            "- agenda_title\\n"
            "- key_utterances(핵심 발언)\\n"
            "- summary(아젠다 요약)\\n"
            "- agenda_keywords(해당 아젠다 키워드)\\n"
            "- decision_results(opinions 요약 + conclusion)\\n"
            "- action_items(유지)\\n"
            "3) keywords/scores/intervention/recommendations는 절대 반환하지 마세요.\\n"
            "4) evidence_gate는 status 없이 claims만 반환하세요.\\n"
            "5) 같은 topic 안에서도 흐름이 다르면 agenda_outcomes를 여러 개 반환하세요.\\n\\n"
            "6) 점수 계산/가중치/룰 기반 추정값은 만들지 말고, 전사 근거를 직접 읽어 요약만 반환하세요.\\n\\n"
            f"요구 JSON 형태:\\n{json.dumps(ANALYSIS_JSON_SHAPE, indent=2)}\\n\\n"
            f"meeting_goal:\\n{meeting_goal}\\n\\n"
            f"initial_context:\\n{initial_context}\\n\\n"
            f"current_active_agenda:\\n{current_active_agenda}\\n\\n"
            f"recent_transcript_window:\\n{json.dumps(transcript_window, ensure_ascii=False, indent=2)}\\n\\n"
            f"current_agenda_stack:\\n{json.dumps(agenda_stack, ensure_ascii=False, indent=2)}\\n"
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
        cleaned = re.sub(r"^```json\\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^```\\s*", "", cleaned)
        cleaned = re.sub(r"\\s*```$", "", cleaned)

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
        candidate = re.sub(r",\\s*([}\\]])", r"\\1", candidate)
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
