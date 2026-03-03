from __future__ import annotations

import json
import os
import re
import ast
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


ROOT = Path(__file__).resolve().parent
if load_dotenv is not None:
    load_dotenv(ROOT / ".env", override=False)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _extract_json(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    l = raw.find("{")
    r = raw.rfind("}")
    if l >= 0 and r > l:
        try:
            parsed = json.loads(raw[l : r + 1])
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _extract_json_loose(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}

    l = raw.find("{")
    r = raw.rfind("}")
    if l >= 0 and r > l:
        raw = raw[l : r + 1]

    raw = raw.replace("\u0000", "")
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    raw = re.sub(r"//.*?$", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.DOTALL)

    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _extract_balanced_json(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    start = raw.find("{")
    if start < 0:
        return {}

    depth = 0
    in_str = False
    escape = False
    quote = ""
    end = -1

    for idx, ch in enumerate(raw[start:], start=start):
        if in_str:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == quote:
                in_str = False
            continue

        if ch in {'"', "'"}:
            in_str = True
            quote = ch
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                end = idx
                break

    if end <= start:
        return {}

    chunk = raw[start : end + 1]
    parsed = _extract_json(chunk)
    if parsed:
        return parsed
    return _extract_json_loose(chunk)


@dataclass
class GeminiClient:
    model: str
    api_key: str
    base_url: str
    connected: bool = False
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    last_operation: str = ""
    last_request_at: str = ""
    last_success_at: str = ""
    last_error: str = ""
    last_error_at: str = ""
    last_raw_preview: str = ""
    last_finish_reason: str = ""
    last_http_status: int = 0

    def status(self) -> dict[str, Any]:
        return {
            "provider": "gemini",
            "model": self.model,
            "base_url": self.base_url,
            "mode": "live",
            "api_key_present": bool(self.api_key),
            "connected": self.connected,
            "note": "Gemini REST API",
            "request_count": self.request_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "last_operation": self.last_operation,
            "last_request_at": self.last_request_at,
            "last_success_at": self.last_success_at,
            "last_error": self.last_error,
            "last_error_at": self.last_error_at,
            "last_raw_preview": self.last_raw_preview,
            "last_finish_reason": self.last_finish_reason,
            "last_http_status": self.last_http_status,
        }

    def _call(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY 또는 GOOGLE_API_KEY가 설정되지 않았습니다.")

        self.request_count += 1
        self.last_request_at = _now_iso()
        self.last_operation = "generate_content"

        timeout_sec = max(20, int(os.environ.get("GEMINI_TIMEOUT_SEC", "60")))
        max_retries = max(1, int(os.environ.get("GEMINI_MAX_RETRIES", "4")))
        retry_base = max(0.2, float(os.environ.get("GEMINI_RETRY_BASE_SEC", "1.0")))
        fallback_models = [
            m.strip()
            for m in os.environ.get("GEMINI_FALLBACK_MODELS", "gemini-2.0-flash-lite,gemini-1.5-flash").split(",")
            if m.strip()
        ]
        model_candidates = [self.model] + [m for m in fallback_models if m != self.model]

        last_error_msg = "알 수 없는 오류"
        last_status = 0
        data: dict[str, Any] | None = None

        for model_name in model_candidates:
            for attempt in range(1, max_retries + 1):
                data = None
                url = f"{self.base_url}/models/{model_name}:generateContent?key={urllib.parse.quote(self.api_key)}"
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": temperature,
                        "maxOutputTokens": max_tokens,
                        "responseMimeType": "application/json",
                    },
                }
                body = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(url, data=body, method="POST")
                req.add_header("Content-Type", "application/json")

                try:
                    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                    # fallback 모델이 성공하면 이후 기본 모델로 승격
                    if model_name != self.model:
                        self.model = model_name
                    self.last_http_status = 200
                    break
                except urllib.error.HTTPError as exc:
                    detail = exc.read().decode("utf-8", errors="ignore")
                    status = int(getattr(exc, "code", 0) or 0)
                    last_status = status
                    retryable = status in {429, 500, 502, 503, 504} or "UNAVAILABLE" in detail or "RESOURCE_EXHAUSTED" in detail
                    last_error_msg = f"HTTP {status}: {detail[:500]}"

                    if retryable and attempt < max_retries:
                        sleep_sec = retry_base * (2 ** (attempt - 1)) + random.uniform(0.0, 0.35)
                        self.last_error = f"{last_error_msg} (재시도 {attempt}/{max_retries}, {sleep_sec:.1f}s 대기)"
                        self.last_error_at = _now_iso()
                        time.sleep(sleep_sec)
                        continue
                    # 재시도 불가 에러는 현재 모델 시도 중단
                    break
                except Exception as exc:
                    last_error_msg = str(exc)
                    status = 0
                    last_status = status
                    if attempt < max_retries:
                        sleep_sec = retry_base * (2 ** (attempt - 1)) + random.uniform(0.0, 0.35)
                        self.last_error = f"{last_error_msg} (재시도 {attempt}/{max_retries}, {sleep_sec:.1f}s 대기)"
                        self.last_error_at = _now_iso()
                        time.sleep(sleep_sec)
                        continue
                    break

            if data is not None:
                break

        if data is None:
            self.error_count += 1
            self.last_error = last_error_msg
            self.last_error_at = _now_iso()
            self.last_http_status = last_status
            raise RuntimeError(self.last_error)

        text = ""
        finish_reason = ""
        try:
            candidates = data.get("candidates") or []
            if candidates:
                finish_reason = str((candidates[0] or {}).get("finishReason") or "")
                parts = (((candidates[0] or {}).get("content") or {}).get("parts") or [])
                if parts:
                    text = str((parts[0] or {}).get("text") or "")
        except Exception:
            text = ""

        if not text:
            self.error_count += 1
            self.last_error = "Gemini 응답 본문이 비어 있습니다."
            self.last_error_at = _now_iso()
            raise RuntimeError(self.last_error)

        self.success_count += 1
        self.last_success_at = _now_iso()
        self.last_error = ""
        self.last_error_at = ""
        self.last_raw_preview = (text or "")[:1000]
        self.last_finish_reason = finish_reason
        return text

    def ping(self) -> dict[str, Any]:
        try:
            raw = self._call(
                "JSON만 반환하세요: {\"ok\": true, \"message\": \"pong\"}",
                temperature=0.0,
                max_tokens=64,
            )
            parsed = _extract_json(raw)
            ok = bool(parsed.get("ok", False))
            msg = str(parsed.get("message", "pong"))
            return {"ok": ok, "message": msg, "mode": "live", "response_preview": parsed}
        except Exception as exc:
            return {"ok": False, "message": str(exc), "mode": "live"}

    def connect(self) -> dict[str, Any]:
        if not self.api_key:
            self.connected = False
            return {"ok": False, "message": "API 키가 없어 연결할 수 없습니다.", "mode": "live"}
        result = self.ping()
        self.connected = bool(result.get("ok"))
        return result

    def disconnect(self) -> dict[str, Any]:
        self.connected = False
        self.last_operation = "disconnect"
        return {"ok": True, "message": "연결 해제됨", "mode": "live"}

    def generate_json(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1400) -> dict[str, Any]:
        if not self.connected:
            raise RuntimeError("LLM이 연결되지 않았습니다. 먼저 연결 버튼을 눌러주세요.")
        raw = self._call(prompt, temperature=temperature, max_tokens=max_tokens)
        parsed = _extract_json(raw)
        if not parsed:
            parsed = _extract_json_loose(raw)
        if not parsed:
            parsed = _extract_balanced_json(raw)
        if parsed:
            return parsed

        repair_input = (raw or "")[:12000]
        repair_prompt = (
            "다음 텍스트를 유효한 JSON 객체 하나로만 정규화해서 반환하세요. "
            "설명/마크다운/코드펜스 없이 JSON만 출력하세요.\n\n"
            f"{repair_input}"
        )
        repair_raw = self._call(repair_prompt, temperature=0.0, max_tokens=max_tokens)
        parsed = _extract_json(repair_raw)
        if not parsed:
            parsed = _extract_json_loose(repair_raw)
        if not parsed:
            parsed = _extract_balanced_json(repair_raw)
        if parsed:
            return parsed

        # 마지막 재시도: 원 프롬프트를 더 강한 JSON 제약으로 재호출
        strict_prompt = (
            "반드시 유효한 JSON 객체 하나만 반환하세요. "
            "설명/주석/코드펜스/추가 텍스트 금지.\n\n"
            + prompt
        )
        strict_raw = self._call(strict_prompt, temperature=0.0, max_tokens=max_tokens)
        parsed = _extract_json(strict_raw)
        if not parsed:
            parsed = _extract_json_loose(strict_raw)
        if not parsed:
            parsed = _extract_balanced_json(strict_raw)
        if not parsed:
            finish_reason = self.last_finish_reason or "-"
            raise RuntimeError(f"LLM JSON 파싱 실패 (finish_reason={finish_reason})")
        return parsed


_LOCK = threading.Lock()
_CLIENT: GeminiClient | None = None


def get_client() -> GeminiClient:
    global _CLIENT
    with _LOCK:
        if _CLIENT is None:
            api_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
            model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
            base_url = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
            _CLIENT = GeminiClient(model=model, api_key=api_key, base_url=base_url)
        return _CLIENT
