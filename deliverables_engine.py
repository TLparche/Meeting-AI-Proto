from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from schemas import ArtifactKind

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.+#/\-]*|[가-힣]{2,}")
OWNER_RE = re.compile(r"(?:담당|owner|오너)\s*[:\-]?\s*([A-Za-z0-9가-힣_]{1,20})", re.IGNORECASE)
DUE_RE = re.compile(
    r"(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}|오늘|내일|이번주|다음주|월요일|화요일|수요일|목요일|금요일|토요일|일요일|today|tomorrow|this week|next week)",
    re.IGNORECASE,
)

ACTION_HINTS = (
    "action",
    "todo",
    "진행",
    "작성",
    "공유",
    "검토",
    "확인",
    "배포",
    "테스트",
    "요청",
    "정리",
    "수정",
    "fix",
    "update",
    "follow-up",
    "share",
)

GENERIC_ACTION_TOKENS = {
    "action",
    "actions",
    "todo",
    "task",
    "tasks",
    "owner",
    "update",
    "fix",
}


def _clip(text: str, limit: int = 80) -> str:
    s = " ".join((text or "").split()).strip()
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


def _agenda_order(state: str) -> int:
    return {"ACTIVE": 0, "CLOSING": 1, "PROPOSED": 2, "CLOSED": 3}.get((state or "").upper(), 9)


def _agenda_entries(agenda_state_map: dict[str, dict], active_agenda_id: str) -> list[dict[str, Any]]:
    entries = []
    for aid, entry in (agenda_state_map or {}).items():
        title = str(entry.get("title") or "").strip()
        if not title:
            continue
        state = str(entry.get("state") or "PROPOSED").upper()
        entries.append(
            {
                "agenda_id": aid,
                "title": title,
                "state": state,
                "is_active": aid == active_agenda_id,
            }
        )
    entries.sort(key=lambda x: (_agenda_order(x["state"]), 0 if x["is_active"] else 1, x["title"]))
    return entries


def _related_turns(transcript: list[dict], agenda_title: str, max_turns: int = 3) -> list[str]:
    turns = transcript[-160:] if transcript else []
    title_tokens = {t.lower() for t in TOKEN_RE.findall(agenda_title or "") if len(t) >= 2}
    matched: list[str] = []
    if title_tokens:
        for turn in reversed(turns):
            txt = str(turn.get("text") or "")
            lowered = txt.lower()
            if any(tok in lowered for tok in title_tokens):
                matched.append(_clip(txt, 90))
                if len(matched) >= max_turns:
                    break
    if not matched:
        for turn in turns[-max_turns:]:
            txt = str(turn.get("text") or "").strip()
            if txt:
                matched.append(_clip(txt, 90))
    return list(reversed(matched))


def _agenda_outcomes(analysis: dict) -> list[dict]:
    raw = (analysis.get("agenda_outcomes") or [])
    out: list[dict] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        title = str(item.get("agenda_title") or "").strip()
        if not title:
            continue
        out.append(item)
    return out


def _build_meeting_summary(
    *,
    meeting_goal: str,
    active_agenda_title: str,
    agenda_entries: list[dict[str, Any]],
    transcript: list[dict],
    analysis: dict,
) -> dict[str, Any]:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    outcomes = _agenda_outcomes(analysis)
    lines = [
        "### Meeting Summary (Live)",
        f"- generated_at: {ts}",
        f"- meeting_goal: {meeting_goal or '(none)'}",
        f"- current_agenda: {active_agenda_title or '(none)'}",
        "",
        "#### Agenda Units",
    ]
    bullets: list[str] = []
    if outcomes:
        for ag in outcomes[:20]:
            title = str(ag.get("agenda_title") or "").strip()
            state = str(ag.get("agenda_state") or "PROPOSED").upper()
            flow_type = str(ag.get("flow_type") or "").strip() or "문제정의"
            key_utterances = ag.get("key_utterances") or []
            key_line = " | ".join(_clip(str(x), 48) for x in key_utterances[:3] if str(x).strip()) or "-"
            summary = _clip(str(ag.get("summary") or "요약 없음"), 120)
            decision_count = len(ag.get("decision_results") or [])
            action_count = len(ag.get("action_items") or [])
            lines.append(f"- [{state}] {title}")
            lines.append(f"  - flow_type: {flow_type}")
            lines.append(f"  - key_utterances: {key_line}")
            lines.append(f"  - summary: {summary}")
            lines.append(f"  - decisions: {decision_count}, actions: {action_count}")
            bullets.append(
                f"[{state}] {title} | flow={flow_type} | {summary} | decisions={decision_count}, actions={action_count}"
            )
    elif not agenda_entries:
        lines.append("- 아젠다가 아직 구조화되지 않았습니다.")
        bullets.append("아직 아젠다가 없어 요약을 생성하지 못했습니다.")
    else:
        for entry in agenda_entries[:8]:
            title = entry["title"]
            state = entry["state"]
            sample_turns = _related_turns(transcript, title, max_turns=2)
            snippet = sample_turns[-1] if sample_turns else "관련 발화 없음"
            mark = " *ACTIVE*" if entry["is_active"] else ""
            lines.append(f"- [{state}] {title}{mark}")
            lines.append(f"  - snapshot: {snippet}")
            bullets.append(f"[{state}] {title} | {snippet}")
    return {
        "kind": ArtifactKind.MEETING_SUMMARY.value,
        "title": "회의 요약 (아젠다 단위)",
        "markdown": "\n".join(lines),
        "bullets": bullets[:12],
    }


def _build_decision_results(
    *,
    analysis: dict,
    active_agenda_title: str,
    dps_score: float,
) -> dict[str, Any]:
    outcomes = _agenda_outcomes(analysis)
    if outcomes:
        lines = [
            "### Decision Results (Agenda Grouped)",
            f"- current_agenda: {active_agenda_title or '(none)'}",
            f"- dps: {dps_score:.2f}",
            "",
        ]
        bullets: list[str] = []
        for ag in outcomes[:20]:
            title = str(ag.get("agenda_title") or "").strip()
            state = str(ag.get("agenda_state") or "PROPOSED").upper()
            lines.append(f"#### [{state}] {title}")
            decision_results = ag.get("decision_results") or []
            if not decision_results:
                lines.append("- 결정 결과 없음")
                continue
            for d in decision_results[:12]:
                decision = _clip(str((d or {}).get("decision") or ""), 80)
                conclusion = _clip(str((d or {}).get("conclusion") or ""), 100)
                opinions = (d or {}).get("opinions") or []
                opinions_txt = " | ".join(_clip(str(x), 50) for x in opinions[:5] if str(x).strip()) or "-"
                lines.append(f"- decision: {decision}")
                lines.append(f"  - opinions: {opinions_txt}")
                lines.append(f"  - conclusion: {conclusion}")
                bullets.append(f"[{title}] {decision} -> {conclusion}")
        return {
            "kind": ArtifactKind.DECISION_RESULTS.value,
            "title": "의사결정 결과 (아젠다별)",
            "markdown": "\n".join(lines),
            "bullets": bullets[:20] or ["아젠다별 결정 결과가 없습니다."],
        }

    intervention = (analysis.get("intervention") or {})
    lock = intervention.get("decision_lock") or {}
    k_core = ((analysis.get("keywords") or {}).get("k_core") or {})
    r2 = ((analysis.get("recommendations") or {}).get("r2_options") or [])
    selected_option = ""
    if r2:
        selected_option = str((r2[0] or {}).get("option") or "").strip()

    object_line = ", ".join(str(x) for x in (k_core.get("object") or []) if str(x).strip()) or "미정"
    constraint_line = ", ".join(str(x) for x in (k_core.get("constraints") or []) if str(x).strip()) or "미정"
    criterion_line = ", ".join(str(x) for x in (k_core.get("criteria") or []) if str(x).strip()) or "미정"
    lock_state = "ON" if bool(lock.get("triggered", False)) else "OFF"
    lock_reason = str(lock.get("reason") or "").strip() or "결정 잠금 조건 대기"

    lines = [
        "### Decision Results (Live)",
        f"- current_agenda: {active_agenda_title or '(none)'}",
        f"- decision_lock: {lock_state}",
        f"- dps: {dps_score:.2f}",
        f"- selected_option_hint: {selected_option or '미정'}",
        "",
        "#### Decision Variables",
        f"- OBJECT: {object_line}",
        f"- CONSTRAINT: {constraint_line}",
        f"- CRITERION: {criterion_line}",
        f"- LOCK_REASON: {lock_reason}",
    ]
    bullets = [
        f"의사결정 대상: {object_line}",
        f"제약조건: {constraint_line}",
        f"평가기준: {criterion_line}",
        f"권장 옵션(현재): {selected_option or '미정'}",
        f"결정 잠금 상태: {lock_state} ({_clip(lock_reason, 70)})",
    ]
    return {
        "kind": ArtifactKind.DECISION_RESULTS.value,
        "title": "의사결정 결과",
        "markdown": "\n".join(lines),
        "bullets": bullets,
    }


def _extract_action_candidates(transcript: list[dict]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for turn in transcript[-180:]:
        text = str(turn.get("text") or "").strip()
        if not text:
            continue
        lowered = text.lower()
        owner_match = OWNER_RE.search(text)
        due_match = DUE_RE.search(text)
        if not any(h in lowered for h in ACTION_HINTS) and not owner_match and not due_match:
            continue
        owner = owner_match.group(1).strip() if owner_match else str(turn.get("speaker") or "미지정")
        due = due_match.group(1).strip() if due_match else "미정"
        task = _clip(text, 100)
        key = f"{owner}|{due}|{task.lower()}"
        if key in seen:
            continue
        seen.add(key)
        out.append({"owner": owner, "due": due, "task": task})
    return out


def _build_action_items(*, analysis: dict, transcript: list[dict]) -> dict[str, Any]:
    outcomes = _agenda_outcomes(analysis)
    if outcomes:
        lines = [
            "### Action Items (Agenda Grouped)",
            "- 형식: [agenda] owner / due / item",
        ]
        bullets: list[str] = []
        for ag in outcomes[:20]:
            title = str(ag.get("agenda_title") or "").strip()
            items = ag.get("action_items") or []
            if not items:
                continue
            lines.append(f"#### {title}")
            for item in items[:16]:
                owner = str((item or {}).get("owner") or "미지정").strip() or "미지정"
                due = str((item or {}).get("due") or "미정").strip() or "미정"
                task = _clip(str((item or {}).get("item") or ""), 110)
                if not task:
                    continue
                line = f"[{title}] [owner: {owner} | due: {due}] {task}"
                lines.append(f"- {line}")
                bullets.append(line)
        if not bullets:
            bullets = ["아젠다별 액션 아이템이 아직 없습니다."]
        return {
            "kind": ArtifactKind.ACTION_ITEMS.value,
            "title": "액션 아이템 (아젠다별)",
            "markdown": "\n".join(lines),
            "bullets": bullets[:24],
        }

    candidates = _extract_action_candidates(transcript)
    k_actions = (((analysis.get("keywords") or {}).get("k_facet") or {}).get("actions") or [])
    for act in k_actions:
        if len(candidates) >= 8:
            break
        text = str(act).strip()
        if not text:
            continue
        if text.lower() in GENERIC_ACTION_TOKENS or len(text) < 3:
            continue
        candidates.append({"owner": "미지정", "due": "미정", "task": _clip(text, 100)})

    if not candidates:
        candidates = [{"owner": "미지정", "due": "미정", "task": "후속 액션 항목이 아직 감지되지 않았습니다."}]

    lines = [
        "### Action Items (Live)",
        "- 형식: owner + due date 를 항상 포함합니다.",
    ]
    bullets: list[str] = []
    for item in candidates[:10]:
        line = f"[owner: {item['owner']} | due: {item['due']}] {item['task']}"
        lines.append(f"- {line}")
        bullets.append(line)
    return {
        "kind": ArtifactKind.ACTION_ITEMS.value,
        "title": "액션 아이템",
        "markdown": "\n".join(lines),
        "bullets": bullets,
    }


def _build_evidence_log(
    *,
    analysis: dict,
    evidence_status: str,
    evidence_snippet: str,
    evidence_log: list[dict],
) -> dict[str, Any]:
    outcomes = _agenda_outcomes(analysis)
    if outcomes:
        lines = [
            "### Evidence Log (Agenda Grouped)",
            f"- status: {evidence_status}",
        ]
        bullets: list[str] = []
        for ag in outcomes[:20]:
            title = str(ag.get("agenda_title") or "").strip()
            items = ag.get("action_items") or []
            for item in items[:16]:
                task = _clip(str((item or {}).get("item") or ""), 90)
                reasons = (item or {}).get("reasons") or []
                if not reasons:
                    continue
                lines.append(f"#### {title} / {task}")
                for r in reasons[:12]:
                    speaker = str((r or {}).get("speaker") or "").strip()
                    ts = str((r or {}).get("timestamp") or "").strip()
                    quote = _clip(str((r or {}).get("quote") or ""), 100)
                    why = _clip(str((r or {}).get("why") or ""), 100)
                    lines.append(f"- [{speaker}{' @'+ts if ts else ''}] {quote}")
                    lines.append(f"  - why: {why}")
                    bullets.append(f"[{title}] {task} | {speaker}: {quote} -> {why}")
        if bullets:
            return {
                "kind": ArtifactKind.EVIDENCE_LOG.value,
                "title": "근거 로그 (아젠다/액션별)",
                "markdown": "\n".join(lines),
                "bullets": bullets[:30],
            }

    lines = [
        "### Evidence Log (Live)",
        f"- status: {evidence_status}",
    ]
    for line in (evidence_snippet or "").splitlines():
        if line.strip():
            lines.append(f"- {line.strip()}")
    bullets: list[str] = []
    for item in (evidence_log or [])[-10:]:
        claim = str(item.get("claim") or "").strip()
        if not claim:
            continue
        status = str(item.get("status") or "UNVERIFIED")
        v = float(item.get("verifiability") or 0.0)
        eqs = item.get("eqs")
        eqs_txt = "-" if eqs is None else f"{float(eqs):.2f}"
        bullets.append(f"[{status}] {claim} | v={v:.2f}, eqs={eqs_txt}")
    if not bullets:
        bullets = ["근거 로그가 아직 비어 있습니다."]
    return {
        "kind": ArtifactKind.EVIDENCE_LOG.value,
        "title": "근거 로그",
        "markdown": "\n".join(lines),
        "bullets": bullets,
    }


def build_live_artifact(
    *,
    kind: ArtifactKind,
    meeting_goal: str,
    transcript: list[dict],
    analysis: dict,
    agenda_state_map: dict[str, dict],
    active_agenda_id: str,
    evidence_status: str,
    evidence_snippet: str,
    evidence_log: list[dict],
    dps_t: float,
) -> dict[str, Any]:
    agenda_entries = _agenda_entries(agenda_state_map, active_agenda_id)
    active_agenda_title = ""
    if active_agenda_id and active_agenda_id in (agenda_state_map or {}):
        active_agenda_title = str((agenda_state_map.get(active_agenda_id) or {}).get("title") or "").strip()
    if not active_agenda_title:
        active_agenda_title = str((((analysis.get("agenda") or {}).get("active") or {}).get("title") or "")).strip()

    if kind == ArtifactKind.MEETING_SUMMARY:
        return _build_meeting_summary(
            meeting_goal=meeting_goal,
            active_agenda_title=active_agenda_title,
            agenda_entries=agenda_entries,
            transcript=transcript,
            analysis=analysis,
        )
    if kind == ArtifactKind.DECISION_RESULTS:
        return _build_decision_results(
            analysis=analysis,
            active_agenda_title=active_agenda_title,
            dps_score=float(dps_t or 0.0),
        )
    if kind == ArtifactKind.ACTION_ITEMS:
        return _build_action_items(analysis=analysis, transcript=transcript)
    return _build_evidence_log(
        analysis=analysis,
        evidence_status=evidence_status,
        evidence_snippet=evidence_snippet,
        evidence_log=evidence_log,
    )
