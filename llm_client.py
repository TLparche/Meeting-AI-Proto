from __future__ import annotations

import json
import os
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
        }

    def _call(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY 또는 GOOGLE_API_KEY가 설정되지 않았습니다.")

        self.request_count += 1
        self.last_request_at = _now_iso()
        self.last_operation = "generate_content"

        url = f"{self.base_url}/models/{self.model}:generateContent?key={urllib.parse.quote(self.api_key)}"
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
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            self.error_count += 1
            self.last_error = f"HTTP {exc.code}: {detail[:500]}"
            self.last_error_at = _now_iso()
            raise RuntimeError(self.last_error) from exc
        except Exception as exc:
            self.error_count += 1
            self.last_error = str(exc)
            self.last_error_at = _now_iso()
            raise

        text = ""
        try:
            candidates = data.get("candidates") or []
            if candidates:
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
            raise RuntimeError("LLM JSON 파싱 실패")
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
