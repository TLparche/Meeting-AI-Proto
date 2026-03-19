"use client";

import "@xyflow/react/dist/style.css";
import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";
import {
  addEdge,
  applyEdgeChanges,
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  Panel,
  Position,
  ReactFlow,
  applyNodeChanges,
  type Connection,
  type Edge,
  type EdgeChange,
  type Node,
  type NodeChange,
} from "@xyflow/react";
import {
  connectLlm,
  disconnectLlm,
  exportAgendaMarkdown,
  exportAgendaSnapshot,
  getLastLlmJson,
  getLlmStatus,
  getState,
  importAgendaSnapshot,
  importJsonDir,
  importJsonFiles,
  importJsonFilesReplay,
  pingLlm,
  replayStep,
  resetState,
  saveConfig,
  tickAnalysis,
  transcribeChunk,
} from "@/lib/api";
import type { MeetingState, SttDebug } from "@/lib/types";
import type {
  Agenda,
  AgendaStatus,
  ActionItem,
  DecisionItem,
  EvidenceItem,
  Participant,
  TranscriptUtterance,
} from "@/lib/meetingData";

type SummaryScope = "current" | "all";
type AgendaState = "PROPOSED" | "ACTIVE" | "CLOSING" | "CLOSED";
type AppSection = "workspace" | "canvas";

type AgendaOutcomeReason = {
  turn_id?: number;
  speaker?: string;
  timestamp?: string;
  quote?: string;
  why?: string;
};

type AgendaOutcomeAction = {
  item?: string;
  owner?: string;
  due?: string;
  reasons?: AgendaOutcomeReason[];
};

type AgendaOutcomeDecision = {
  opinions?: string[];
  conclusion?: string;
};

type AgendaOutcomeOpinionGroup = {
  type?: string;
  summary?: string;
  evidence_turn_ids?: number[];
};

type AgendaOutcome = {
  agenda_id?: string;
  agenda_title?: string;
  agenda_state?: string;
  flow_type?: string;
  key_utterances?: string[];
  agenda_summary_items?: string[];
  summary?: string;
  summary_references?: AgendaOutcomeReason[];
  agenda_keywords?: string[];
  opinion_groups?: AgendaOutcomeOpinionGroup[];
  decision_results?: AgendaOutcomeDecision[];
  action_items?: AgendaOutcomeAction[];
  start_turn_id?: number;
  end_turn_id?: number;
};

type SummaryPointMeta = {
  agendaId: string;
  pointId: string;
  pointText: string;
  rangeLabel: string;
  turnIds: number[];
  references: AgendaOutcomeReason[];
  opinionGroups: OpinionGroup[];
};

type SummaryFocusState = SummaryPointMeta & {
  utterances: TranscriptUtterance[];
};

type CanvasNodeDetail = {
  id: string;
  kind: CanvasGraphNode["kind"];
  agendaId: string;
  pointId?: string;
  title: string;
  subtitle: string;
  badges: string[];
  summaryLines: string[];
  opinionGroups: OpinionGroup[];
  utterances: TranscriptUtterance[];
  noteBody?: string;
};

type OpinionType = "proposal" | "concern" | "question" | "agree" | "disagree" | "info";

type OpinionGroup = {
  id: string;
  type: OpinionType;
  typeLabel: string;
  summary: string;
  detail: string;
  rangeLabel: string;
  utterances: TranscriptUtterance[];
};

type CanvasIdea = {
  id: string;
  agendaId: string;
  title: string;
  body: string;
  createdAt: string;
  linkedPointId?: string;
  linkedPointText?: string;
  colorTone: "blue" | "mint" | "amber" | "rose";
};

type CanvasLane = {
  agendaId: string;
  agendaLabel: string;
  agendaTitle: string;
  status: AgendaStatus;
  flowType: string;
  timeLabel: string;
  keywordLabel: string;
  transcriptCount: number;
  summaryNodes: Array<{
    pointId: string;
    pointText: string;
    rangeLabel: string;
    utteranceCount: number;
    opinionCount: number;
  }>;
  ideaNodes: CanvasIdea[];
};

type CanvasNodePosition = {
  x: number;
  y: number;
};

type CanvasGraphNode = {
  id: string;
  agendaId: string;
  kind: "agenda" | "summary" | "idea";
  title: string;
  body: string;
  subtitle: string;
  meta: string[];
  pointId?: string;
  linkedPointText?: string;
  width: number;
  height: number;
  x: number;
  y: number;
};

type CanvasGraphEdge = {
  id: string;
  fromId: string;
  toId: string;
  kind: "agenda-summary" | "agenda-idea" | "summary-idea";
};

type CanvasFlowNodeData = {
  label: ReactNode;
  nodeId: string;
  agendaId: string;
  kind: CanvasGraphNode["kind"];
  title: string;
  body: string;
  subtitle: string;
  meta: string[];
  pointId?: string;
  linkedPointText?: string;
};

type ResizeTarget = "sidebar" | "canvas-left" | "canvas-right";

const EMPTY_STATE: MeetingState = {
  meeting_goal: "",
  initial_context: "",
  window_size: 12,
  transcript: [],
  agenda_stack: [],
  llm_enabled: false,
  analysis: null,
};

const agendaStatusClass: Record<AgendaStatus, string> = {
  "Not started": "statusChip statusChipNeutral",
  "In progress": "statusChip statusChipProgress",
  Done: "statusChip statusChipDone",
};

const actionStatusClass: Record<ActionItem["status"], string> = {
  Open: "statusChip statusChipNeutral",
  "In progress": "statusChip statusChipProgress",
  Done: "statusChip statusChipDone",
};

const agendaStatusLabel: Record<AgendaStatus, string> = {
  "Not started": "시작 전",
  "In progress": "진행 중",
  Done: "완료",
};

const actionStatusLabel: Record<ActionItem["status"], string> = {
  Open: "대기",
  "In progress": "진행 중",
  Done: "완료",
};

const decisionStatusLabel: Record<DecisionItem["finalStatus"], string> = {
  Approved: "확정",
  Pending: "보류",
  Rejected: "반려",
};

const participantStatusLabel: Record<Participant["status"], string> = {
  Speaking: "발언 중",
  Active: "참여 중",
  Listening: "청취 중",
};

const evidenceSupportLabel: Record<EvidenceItem["supports"], string> = {
  Action: "액션",
  Decision: "의사결정",
  Summary: "요약",
};

function agendaLabel(agenda: Agenda): string {
  return `${agenda.label}: ${agenda.title}`;
}

function renderCanvasFlowLabel(data: {
  kind: CanvasGraphNode["kind"];
  subtitle: string;
  title: string;
  body: string;
  meta: string[];
}): ReactNode {
  return (
    <div className="rfDefaultNodeLabel">
      <div className="rfDefaultNodeHeader">
        <span className={`rfDefaultNodeKind rfDefaultNodeKind${data.kind === "agenda" ? "Agenda" : data.kind === "summary" ? "Summary" : "Idea"}`}>
          {data.kind}
        </span>
        <span>{data.subtitle}</span>
      </div>
      <strong>{data.title}</strong>
      {data.body ? <p>{data.body}</p> : null}
      <div className="rfDefaultNodeMeta">
        {data.meta.map((item) => (
          <span key={`${data.title}-${item}`}>{item}</span>
        ))}
      </div>
    </div>
  );
}

function decisionStatusClass(status: DecisionItem["finalStatus"]): string {
  if (status === "Approved") return "statusChip statusChipDone";
  if (status === "Pending") return "statusChip statusChipProgress";
  return "statusChip statusChipNeutral";
}

function participantStatusClass(status: Participant["status"]): string {
  if (status === "Speaking") return "statusChip statusChipProgress";
  if (status === "Active") return "statusChip statusChipDone";
  return "statusChip statusChipNeutral";
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function normalizeAgendaState(raw: string | undefined): AgendaState {
  const s = String(raw || "").toUpperCase();
  if (s === "ACTIVE" || s === "CLOSING" || s === "CLOSED" || s === "PROPOSED") return s;
  return "PROPOSED";
}

function toAgendaStatus(state: AgendaState): AgendaStatus {
  if (state === "ACTIVE" || state === "CLOSING") return "In progress";
  if (state === "CLOSED") return "Done";
  return "Not started";
}

function statusProgress(status: AgendaStatus): number {
  if (status === "Done") return 100;
  if (status === "In progress") return 65;
  return 15;
}

function formatNowTime(): string {
  return new Date().toLocaleTimeString("ko-KR", { hour12: false });
}

function safeText(raw: unknown, fallback = ""): string {
  const s = String(raw || "").trim();
  return s || fallback;
}

function tokenize(text: string): string[] {
  return (text.match(/[A-Za-z0-9가-힣]{2,}/g) || []).map((t) => t.toLowerCase());
}

function extractTimestampToken(text: string): string {
  const m = safeText(text).match(/(\d{2}:\d{2}(?::\d{2})?)/);
  return m ? m[1] : "";
}

function stripLeadingTimestamp(text: string): string {
  return safeText(text).replace(/^\[\d{2}:\d{2}(?::\d{2})?\]\s*/, "").trim();
}

function buildMeetingStateSignature(state: MeetingState): string {
  const transcript = state.transcript || [];
  const lastTranscript = transcript[transcript.length - 1];
  const outcomes = state.analysis?.agenda_outcomes || [];
  const worker = state.analysis_runtime?.analysis_worker;
  return JSON.stringify({
    meeting_goal: safeText(state.meeting_goal),
    transcript_count: transcript.length,
    last_transcript_ts: safeText(lastTranscript?.timestamp),
    last_transcript_speaker: safeText(lastTranscript?.speaker),
    last_transcript_text: safeText(lastTranscript?.text).slice(-48),
    agenda_stack_count: state.agenda_stack?.length || 0,
    outcome_count: outcomes.length,
    active_agenda: safeText(state.analysis?.agenda?.active?.title),
    active_confidence: Number(state.analysis?.agenda?.active?.confidence || 0),
    llm_enabled: Boolean(state.llm_enabled),
    replay_cursor: Number(state.replay?.queued_cursor || 0),
    replay_remaining: Number(state.replay?.queued_remaining || 0),
    replay_done: Boolean(state.replay?.done),
    worker_inflight: Boolean(worker?.inflight),
    worker_queued: Number(worker?.queued || 0),
    worker_done: Number(worker?.last_done_id || 0),
    llm_json_at: safeText(state.analysis_runtime?.last_llm_json_at),
    control_reason: safeText(state.analysis_runtime?.control_plane_reason),
    fallback: Boolean(state.analysis_runtime?.used_local_fallback),
  });
}

function buildLlmStatusSignature(status: MeetingState["llm_status"]): string {
  return JSON.stringify({
    connected: Boolean(status?.connected),
    request_count: Number(status?.request_count || 0),
    success_count: Number(status?.success_count || 0),
    error_count: Number(status?.error_count || 0),
    last_request_at: safeText(status?.last_request_at),
    last_success_at: safeText(status?.last_success_at),
    last_error_at: safeText(status?.last_error_at),
    last_error: safeText(status?.last_error),
    last_finish_reason: safeText(status?.last_finish_reason),
    last_raw_preview: safeText(status?.last_raw_preview).slice(0, 120),
  });
}

function normalizeSummaryKey(text: string): string {
  return stripLeadingTimestamp(text)
    .toLowerCase()
    .replace(/\s+/g, " ")
    .trim();
}

function timeToSeconds(ts: string): number {
  const token = extractTimestampToken(ts);
  if (!token) return -1;
  const parts = token.split(":").map((v) => Number(v));
  if (parts.some((v) => Number.isNaN(v))) return -1;
  if (parts.length === 2) return parts[0] * 60 + parts[1];
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
  return -1;
}

function buildTimeRangeLabel(timestamps: string[], fallbackTimestamp = ""): string {
  const seen = new Set<string>();
  const normalized = timestamps.map((ts) => extractTimestampToken(ts)).filter(Boolean);
  if (fallbackTimestamp) normalized.push(extractTimestampToken(fallbackTimestamp));
  const unique = normalized.filter((ts) => {
    if (seen.has(ts)) return false;
    seen.add(ts);
    return true;
  });
  if (unique.length === 0) return "-";
  const ordered = unique.slice().sort((a, b) => timeToSeconds(a) - timeToSeconds(b));
  if (ordered.length === 1) return ordered[0];
  return `${ordered[0]} ~ ${ordered[ordered.length - 1]}`;
}

function quoteSimilar(a: string, b: string): boolean {
  const left = safeText(a).toLowerCase();
  const right = safeText(b).toLowerCase();
  if (!left || !right) return false;
  if (left.includes(right) || right.includes(left)) return true;
  const tokens = tokenize(left);
  if (tokens.length === 0) return false;
  const hitCount = tokens.filter((token) => right.includes(token)).length;
  return hitCount >= Math.min(3, Math.ceil(tokens.length * 0.6));
}

function isNearBottom(el: HTMLDivElement, threshold = 16): boolean {
  return el.scrollHeight - el.scrollTop - el.clientHeight <= threshold;
}

function compactLine(text: string, maxLen = 88): string {
  const s = safeText(text).replace(/\s+/g, " ").trim();
  if (!s) return "";
  if (s.length <= maxLen) return s;
  return `${s.slice(0, maxLen - 1).trim()}…`;
}

const OPINION_SUMMARY_STOPWORDS = new Set([
  "그냥",
  "이제",
  "근데",
  "그러면",
  "그니까",
  "정도",
  "부분",
  "관련",
  "대해서",
  "있어요",
  "있습니다",
  "같아요",
  "같습니다",
  "좋아요",
  "좋습니다",
  "우리가",
  "제가",
  "저는",
  "그거",
  "이거",
  "바로",
  "어떻게",
  "아마",
  "씨",
  "씨가",
  "씨는",
  "님",
  "님이",
  "님은",
  "본인",
  "본인의",
  "관련된",
  "의견",
  "정보",
  "공유",
  "같은",
  "보면",
  "그러니까",
  "company",
  "companies",
]);

const OPINION_TOKEN_MAP: Record<string, string> = {
  company: "기업",
  companies: "기업",
  investment: "투자",
  investments: "투자",
  market: "시장",
  policy: "정책",
};

function normalizeOpinionToken(raw: string): string {
  let tok = safeText(raw).toLowerCase();
  if (!tok) return "";
  tok = OPINION_TOKEN_MAP[tok] || tok;
  for (const suf of ["으로", "에서", "에게", "처럼", "까지", "부터", "하고", "랑", "와", "과", "을", "를", "은", "는", "이", "가", "도", "로", "에"]) {
    if (tok.length > 2 && tok.endsWith(suf)) {
      tok = tok.slice(0, -suf.length);
      break;
    }
  }
  return tok;
}

function isOpinionNoiseToken(token: string): boolean {
  const t = normalizeOpinionToken(token);
  if (!t || t.length < 2) return true;
  if (OPINION_SUMMARY_STOPWORDS.has(t)) return true;
  if (/^(씨|님|본인|당사|우리|저희|제가|저는)$/.test(t)) return true;
  if (/(같|보|되|하)$/.test(t) && t.length <= 3) return true;
  if (/^\d+$/.test(t)) return true;
  return false;
}

function opinionKeywords(lines: string[], limit = 2): string[] {
  const freq = new Map<string, number>();
  for (const line of lines) {
    const tokens = tokenize(line);
    for (const t of tokens) {
      const tok = normalizeOpinionToken(t);
      if (isOpinionNoiseToken(tok)) continue;
      freq.set(tok, (freq.get(tok) || 0) + 1);
    }
  }
  return Array.from(freq.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([k]) => k);
}

function normalizeOpinionLineForSummary(line: string): string {
  let s = safeText(line).replace(/\s+/g, " ").trim();
  s = s.replace(/^(저는|제가|저희는|저희가)\s+/g, "");
  s = s.replace(/^(일단|그리고|근데|그니까|그러니까|음|어|네|예)\s+/g, "");
  s = s.replace(/\s+/g, " ").trim();
  return s;
}

function pickOpinionSummaryLines(lines: string[]): string[] {
  const cleaned = lines.map((l) => normalizeOpinionLineForSummary(l)).filter(Boolean);
  if (cleaned.length <= 1) return cleaned.slice(0, 1);

  const freq = new Map<string, number>();
  for (const line of cleaned) {
    for (const tok of opinionKeywords([line], 20)) {
      freq.set(tok, (freq.get(tok) || 0) + 1);
    }
  }

  const scored = cleaned.map((line, idx) => {
    const toks = opinionKeywords([line], 20);
    const score = toks.reduce((acc, t) => acc + (freq.get(t) || 0), 0) + Math.min(line.length, 80) / 40;
    return { idx, line, toks, score };
  });
  scored.sort((a, b) => b.score - a.score);
  const first = scored[0];
  if (!first) return [];

  let second: typeof first | null = null;
  for (const cand of scored.slice(1)) {
    const overlap = cand.toks.filter((t) => first.toks.includes(t)).length;
    const novelty = cand.toks.length - overlap;
    if (novelty >= 1) {
      second = cand;
      break;
    }
  }

  if (!second) return [first.line];
  return [first.line, second.line];
}

function summarizeOpinionGroup(type: OpinionType, lines: string[]): { summary: string; detail: string } {
  if (lines.length === 0) return { summary: "의견 요약 없음", detail: "" };
  const compactLines = lines.map((l) => normalizeOpinionLineForSummary(l)).filter(Boolean);
  const picked = pickOpinionSummaryLines(compactLines);
  let summary = "";
  if (picked.length === 0) {
    summary = "의견 요약 없음";
  } else if (picked.length === 1) {
    summary = picked[0];
  } else {
    summary = `${picked[0]} ${picked[1]}`;
  }

  const sample = compactLines.slice(0, 2).join(" / ");
  const detail = safeText(sample);
  return { summary: safeText(summary), detail };
}

function isOpinionLike(text: string): boolean {
  const line = safeText(text).toLowerCase();
  if (!line || line.length < 8) return false;

  const opinionHint = /(같아요|같습니다|생각|의견|우려|느낌|좋겠|필요|해야|하면|원해|제안|추천|선호|어떨까요|보여요|맞는 것|좋을 것|일 듯|가능할 것|추정)/;
  const firstPersonHint = /(저는|제가|저희|우리는|개인적)/;
  const factHint = /(\d+%|\d+명|\d+건|\d+월|\d+일|\d+시|지표|통계|매출|수치|보고서|근거|확인됨|발표|데이터)/;

  if (opinionHint.test(line) || firstPersonHint.test(line)) return true;
  if (factHint.test(line) && !opinionHint.test(line)) return false;
  return /[?]|(하자|해요|합시다|보죠|볼까요)/.test(line);
}

function classifyOpinionType(text: string): { type: OpinionType; label: string } {
  const line = safeText(text).toLowerCase();

  const disagreePat = /(아니|반대|어렵|곤란|무리|힘들|불가|안 될|안될|하지 말|비추천)/;
  const concernPat = /(우려|리스크|위험|문제|부담|걱정|불안|지연|이슈|한계|부족)/;
  const questionPat = /(\?|어떨까요|가능할까요|맞나요|인가요|일까요|할까요|뭐가)/;
  const proposalPat = /(제안|추천|좋겠|하면 좋|하자|합시다|해보|필요|우선|먼저|방안|대안)/;
  const agreePat = /(동의|찬성|맞아요|맞습니다|좋아요|좋습니다|그대로 가|괜찮|맞는 것)/;

  if (disagreePat.test(line)) return { type: "disagree", label: "반대/보류" };
  if (concernPat.test(line)) return { type: "concern", label: "우려/리스크" };
  if (questionPat.test(line)) return { type: "question", label: "질문/확인" };
  if (proposalPat.test(line)) return { type: "proposal", label: "제안" };
  if (agreePat.test(line)) return { type: "agree", label: "동의" };
  return { type: "info", label: "의견/정보" };
}

function normalizeOpinionType(raw: string): OpinionType {
  const t = safeText(raw).toLowerCase();
  if (t === "proposal" || t === "concern" || t === "question" || t === "agree" || t === "disagree" || t === "info") {
    return t;
  }
  return "info";
}

function opinionTypeLabel(type: OpinionType): string {
  if (type === "proposal") return "제안";
  if (type === "concern") return "우려/리스크";
  if (type === "question") return "질문/확인";
  if (type === "agree") return "동의";
  if (type === "disagree") return "반대/보류";
  return "의견/정보";
}

function buildOpinionGroups(
  agendaId: string,
  pointId: string,
  utterances: TranscriptUtterance[],
  references: AgendaOutcomeReason[],
): OpinionGroup[] {
  const seed: TranscriptUtterance[] = [...utterances];
  if (seed.length === 0) {
    references.forEach((reason, idx) => {
      const quote = safeText(reason.quote);
      if (!quote) return;
      const turnNum = Number(reason.turn_id || 0);
      seed.push({
        id: turnNum > 0 ? `utt-${turnNum}` : `ref-${pointId}-${idx + 1}`,
        timestamp: safeText(reason.timestamp, "--:--"),
        speaker: safeText(reason.speaker, "화자"),
        text: quote,
        agendaId,
      });
    });
  }

  const grouped = new Map<OpinionType, { typeLabel: string; items: TranscriptUtterance[] }>();
  const seen = new Set<string>();
  for (const u of seed) {
    if (!isOpinionLike(u.text)) continue;
    const compact = compactLine(u.text, 120);
    const key = `${u.speaker}|${compact}`;
    if (!compact || seen.has(key)) continue;
    seen.add(key);
    const cls = classifyOpinionType(compact);
    const cur = grouped.get(cls.type);
    if (!cur) {
      grouped.set(cls.type, { typeLabel: cls.label, items: [u] });
    } else {
      cur.items.push(u);
    }
  }

  const priority: Record<OpinionType, number> = {
    proposal: 1,
    concern: 2,
    question: 3,
    agree: 4,
    disagree: 5,
    info: 6,
  };

  const out: OpinionGroup[] = [];
  for (const [type, payload] of grouped.entries()) {
    const ordered = payload.items.slice().sort((a, b) => {
      const ta = Number(a.id.replace("utt-", ""));
      const tb = Number(b.id.replace("utt-", ""));
      if (!Number.isNaN(ta) && !Number.isNaN(tb)) return ta - tb;
      return a.timestamp.localeCompare(b.timestamp);
    });
    const summaryPack = summarizeOpinionGroup(type, ordered.map((u) => u.text));
    if (!summaryPack.summary) continue;
    const summary = summaryPack.summary;
    const rangeLabel = buildTimeRangeLabel(ordered.map((u) => u.timestamp));
    out.push({
      id: `${pointId}-op-${type}`,
      type,
      typeLabel: payload.typeLabel,
      summary,
      detail: summaryPack.detail,
      rangeLabel,
      utterances: ordered,
    });
  }

  out.sort((a, b) => {
    const pa = priority[a.type] ?? 99;
    const pb = priority[b.type] ?? 99;
    if (pa !== pb) return pa - pb;
    return b.utterances.length - a.utterances.length;
  });

  return out.slice(0, 5);
}

export default function Home() {
  const [state, setState] = useState<MeetingState>(EMPTY_STATE);
  const [loading, setLoading] = useState(false);
  const [activeTask, setActiveTask] = useState("");
  const [taskStartedAt, setTaskStartedAt] = useState<number | null>(null);
  const [taskElapsedSec, setTaskElapsedSec] = useState(0);
  const [analysisPending, setAnalysisPending] = useState(false);
  const [focusedTargetDomId, setFocusedTargetDomId] = useState("");
  const [selectedSummaryFocus, setSelectedSummaryFocus] = useState<SummaryFocusState | null>(null);
  const [error, setError] = useState("");

  const [query, setQuery] = useState("");
  const [speakerFilter, setSpeakerFilter] = useState("전체");
  const [highlightRelated, setHighlightRelated] = useState(true);
  const [activeSection, setActiveSection] = useState<AppSection>("workspace");
  const isCanvasMode = activeSection === "canvas";
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [sidebarWidth, setSidebarWidth] = useState(280);
  const [canvasLeftRailOpen, setCanvasLeftRailOpen] = useState(true);
  const [canvasRightRailOpen, setCanvasRightRailOpen] = useState(true);
  const [canvasLeftRailWidth, setCanvasLeftRailWidth] = useState(332);
  const [canvasRightRailWidth, setCanvasRightRailWidth] = useState(360);
  const [canvasComposerOpen, setCanvasComposerOpen] = useState(false);
  const [canvasReturnContext, setCanvasReturnContext] = useState("");
  const [transcriptAutoFollow, setTranscriptAutoFollow] = useState(true);
  const [pendingTranscriptCount, setPendingTranscriptCount] = useState(0);
  const [summaryScope, setSummaryScope] = useState<SummaryScope>("current");
  const [selectedAgendaId, setSelectedAgendaId] = useState("");
  const [canvasIdeaTitle, setCanvasIdeaTitle] = useState("");
  const [canvasIdeaBody, setCanvasIdeaBody] = useState("");
  const [canvasIdeas, setCanvasIdeas] = useState<CanvasIdea[]>([]);
  const [canvasNodeDetail, setCanvasNodeDetail] = useState<CanvasNodeDetail | null>(null);
  const [canvasNodePositions, setCanvasNodePositions] = useState<Record<string, CanvasNodePosition>>({});
  const [flowNodes, setFlowNodes] = useState<Node<CanvasFlowNodeData>[]>([]);
  const [flowEdges, setFlowEdges] = useState<Edge[]>([]);

  const [datasetFolder, setDatasetFolder] = useState("dataset/economy");
  const [datasetFiles, setDatasetFiles] = useState<File[]>([]);
  const [agendaSnapshotFile, setAgendaSnapshotFile] = useState<File | null>(null);
  const [datasetImportInfo, setDatasetImportInfo] = useState("");
  const [lineUploadMode, setLineUploadMode] = useState(false);
  const [replayLinesPerStep, setReplayLinesPerStep] = useState(1);
  const [replayIntervalMs, setReplayIntervalMs] = useState(1200);
  const [replayRunning, setReplayRunning] = useState(false);
  const [meetingGoalDraft, setMeetingGoalDraft] = useState("");
  const [meetingGoalDirty, setMeetingGoalDirty] = useState(false);

  const [llmChecking, setLlmChecking] = useState(false);
  const [llmPingMessage, setLlmPingMessage] = useState("");
  const [llmPingOk, setLlmPingOk] = useState<boolean | null>(null);
  const [llmJsonLoading, setLlmJsonLoading] = useState(false);
  const [lastLlmJson, setLastLlmJson] = useState<Record<string, unknown> | null>(null);
  const [lastLlmJsonAt, setLastLlmJsonAt] = useState("");

  const [sttSpeaker, setSttSpeaker] = useState("시스템오디오");
  const [sttSource, setSttSource] = useState<"system">("system");
  const [sttRunning, setSttRunning] = useState(false);
  const [sttStatusText, setSttStatusText] = useState("STOPPED");
  const [sttStatusDetail, setSttStatusDetail] = useState("대기 중");
  const [sttLogs, setSttLogs] = useState<string[]>([]);
  const [lastDebug, setLastDebug] = useState<SttDebug | null>(null);
  const [debugEvents, setDebugEvents] = useState<string[]>([]);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunkSeqRef = useRef(0);
  const sttSessionRef = useRef(0);
  const sendQueueRef = useRef<Promise<void>>(Promise.resolve());
  const replayTimerRef = useRef<number | null>(null);
  const replayBusyRef = useRef(false);
  const meetingStateSignatureRef = useRef("");
  const llmStatusSignatureRef = useRef("");
  const transcriptListRef = useRef<HTMLDivElement | null>(null);
  const transcriptPrevCountRef = useRef(0);
  const transcriptInitRef = useRef(false);
  const resizeRef = useRef<{
    target: ResizeTarget;
    startX: number;
    startWidth: number;
  } | null>(null);
  const debugSnapshotRef = useRef<{
    transcriptCount: number;
    outcomeCount: number;
    activeAgenda: string;
    decisionCount: number;
    actionCount: number;
  } | null>(null);

  const appendSttLog = useCallback((message: string) => {
    const ts = new Date().toLocaleTimeString();
    setSttLogs((prev) => [...prev, `${ts} | ${message}`].slice(-120));
  }, []);

  const loadState = useCallback(async () => {
    try {
      const next = await getState();
      const nextSignature = buildMeetingStateSignature(next);
      if (meetingStateSignatureRef.current === nextSignature) {
        setError("");
        return;
      }
      meetingStateSignatureRef.current = nextSignature;
      if (next.llm_status) {
        llmStatusSignatureRef.current = buildLlmStatusSignature(next.llm_status);
      }
      setState(next);
      setError("");
    } catch (err) {
      setError((err as Error).message);
    }
  }, []);

  const refreshLlmStatus = useCallback(async () => {
    try {
      const status = await getLlmStatus();
      const nextSignature = buildLlmStatusSignature(status);
      if (llmStatusSignatureRef.current === nextSignature) return;
      llmStatusSignatureRef.current = nextSignature;
      setState((prev) => ({ ...prev, llm_status: status }));
    } catch {
      // noop
    }
  }, []);

  const stopReplayAuto = useCallback((message?: string) => {
    if (replayTimerRef.current !== null) {
      window.clearInterval(replayTimerRef.current);
      replayTimerRef.current = null;
    }
    setReplayRunning(false);
    if (message) {
      setDatasetImportInfo(message);
    }
  }, []);

  useEffect(() => {
    void loadState();
  }, [loadState]);

  useEffect(() => {
    const worker = state.analysis_runtime?.analysis_worker;
    const queued = Number(worker?.queued || 0);
    const inflight = Boolean(worker?.inflight);
    const pollMs = replayRunning || inflight || queued > 0
      ? isCanvasMode
        ? 700
        : 250
      : isCanvasMode
        ? 2200
        : 1200;
    const id = window.setInterval(() => {
      void loadState();
    }, pollMs);
    return () => window.clearInterval(id);
  }, [isCanvasMode, loadState, replayRunning, state.analysis_runtime?.analysis_worker?.inflight, state.analysis_runtime?.analysis_worker?.queued]);

  useEffect(() => {
    const id = window.setInterval(() => {
      void refreshLlmStatus();
    }, 3000);
    return () => window.clearInterval(id);
  }, [refreshLlmStatus]);

  useEffect(() => {
    return () => {
      if (replayTimerRef.current !== null) {
        window.clearInterval(replayTimerRef.current);
        replayTimerRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!loading || !taskStartedAt) {
      setTaskElapsedSec(0);
      return;
    }
    const id = window.setInterval(() => {
      setTaskElapsedSec(Math.max(0, Math.floor((Date.now() - taskStartedAt) / 1000)));
    }, 200);
    return () => window.clearInterval(id);
  }, [loading, taskStartedAt]);

  useEffect(() => {
    if (!lineUploadMode) {
      stopReplayAuto();
    }
  }, [lineUploadMode, stopReplayAuto]);

  useEffect(() => {
    if (!replayRunning) return;
    const remaining = Number(state.replay?.queued_remaining || 0);
    if (remaining <= 0) {
      stopReplayAuto("line-mode 완료: 모든 발화를 주입했습니다.");
    }
  }, [replayRunning, state.replay?.queued_remaining, stopReplayAuto]);

  const beginTask = useCallback((label: string) => {
    setLoading(true);
    setActiveTask(label);
    setTaskStartedAt(Date.now());
  }, []);

  const endTask = useCallback(() => {
    setLoading(false);
    setActiveTask("");
    setTaskStartedAt(null);
    setTaskElapsedSec(0);
  }, []);

  const commitMeetingState = useCallback((next: MeetingState) => {
    meetingStateSignatureRef.current = buildMeetingStateSignature(next);
    if (next.llm_status) {
      llmStatusSignatureRef.current = buildLlmStatusSignature(next.llm_status);
    }
    setState(next);
  }, []);

  const scrollTranscriptToBottom = useCallback((behavior: ScrollBehavior = "smooth") => {
    const el = transcriptListRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior });
    setTranscriptAutoFollow(true);
    setPendingTranscriptCount(0);
  }, []);

  useEffect(() => {
    if (!meetingGoalDirty) {
      setMeetingGoalDraft(state.meeting_goal || "");
    }
  }, [state.meeting_goal, meetingGoalDirty]);

  const apply = async (action: () => Promise<MeetingState>, label = "요청 처리 중", lockAnalysis = false) => {
    beginTask(label);
    if (lockAnalysis) setAnalysisPending(true);
    try {
      const next = await action();
      commitMeetingState(next);
      setError("");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      if (lockAnalysis) setAnalysisPending(false);
      endTask();
    }
  };

  const onSaveConfig = async () => {
    beginTask("설정 저장 중");
    try {
      const next = await saveConfig({
        meeting_goal: meetingGoalDraft,
        window_size: state.window_size,
      });
      commitMeetingState(next);
      setMeetingGoalDraft(next.meeting_goal || "");
      setMeetingGoalDirty(false);
      setError("");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      endTask();
    }
  };

  const onImportDataset = async () => {
    beginTask("JSON 폴더 분석 중");
    setAnalysisPending(true);
    try {
      const res = await importJsonDir({
        folder: datasetFolder || "dataset/economy",
        recursive: true,
        reset_state: true,
        auto_tick: true,
        max_files: 500,
      });
      commitMeetingState(res.state);
      setMeetingGoalDraft(res.state.meeting_goal || "");
      setMeetingGoalDirty(false);
      setError("");
      const d = res.import_debug;
      setDatasetImportInfo(
        `loaded=${d.added}, files=${d.files_parsed}/${d.files_scanned}, skipped=${d.files_skipped}, ticked=${d.ticked ? "yes" : "no"}`,
      );
      const firstParseError = d.parse_errors?.[0];
      if (firstParseError) {
        setDebugEvents((rows) => [
          `${formatNowTime()} | JSON 파싱 오류: ${firstParseError.file} -> ${firstParseError.error}`,
          ...rows,
        ].slice(0, 80));
      }
      if (d.warning) setError(d.warning);
    } catch (err) {
      setError((err as Error).message);
      setDatasetImportInfo("");
    } finally {
      setAnalysisPending(false);
      endTask();
    }
  };

  const onImportDatasetFiles = async () => {
    if (datasetFiles.length === 0) {
      setError("업로드할 JSON 파일을 먼저 선택하세요.");
      return;
    }
    stopReplayAuto();
    beginTask("JSON 업로드 분석 중");
    setAnalysisPending(true);
    try {
      const res = await importJsonFiles({ files: datasetFiles, reset_state: true, auto_tick: true });
      commitMeetingState(res.state);
      setMeetingGoalDraft(res.state.meeting_goal || "");
      setMeetingGoalDirty(false);
      setError("");
      const d = res.import_debug;
      setDatasetImportInfo(
        `uploaded=${datasetFiles.length}, loaded=${d.added}, files=${d.files_parsed}/${d.files_scanned}, skipped=${d.files_skipped}, ticked=${d.ticked ? "yes" : "no"}`,
      );
      const firstParseError = d.parse_errors?.[0];
      if (firstParseError) {
        setDebugEvents((rows) => [
          `${formatNowTime()} | JSON 파싱 오류: ${firstParseError.file} -> ${firstParseError.error}`,
          ...rows,
        ].slice(0, 80));
      }
      if (d.warning) setError(d.warning);
    } catch (err) {
      setError((err as Error).message);
      setDatasetImportInfo("");
    } finally {
      setAnalysisPending(false);
      endTask();
    }
  };

  const onQueueDatasetFilesLineMode = async () => {
    if (datasetFiles.length === 0) {
      setError("업로드할 JSON 파일을 먼저 선택하세요.");
      return;
    }
    stopReplayAuto();
    beginTask("라인모드 큐 적재 중");
    try {
      const res = await importJsonFilesReplay({ files: datasetFiles, reset_state: true, apply_goal: true });
      commitMeetingState(res.state);
      setMeetingGoalDraft(res.state.meeting_goal || "");
      setMeetingGoalDirty(false);
      setError("");
      const d = res.replay_debug;
      setDatasetImportInfo(
        `line-mode queued=${d.queued_total}, parsed=${d.files_parsed}/${d.files_scanned}, skipped=${d.files_skipped}`,
      );
      const firstParseError = d.parse_errors?.[0];
      if (firstParseError) {
        setDebugEvents((rows) => [
          `${formatNowTime()} | JSON 파싱 오류: ${firstParseError.file} -> ${firstParseError.error}`,
          ...rows,
        ].slice(0, 80));
      }
      if (d.warning) setError(d.warning);
    } catch (err) {
      setError((err as Error).message);
      setDatasetImportInfo("");
    } finally {
      endTask();
    }
  };

  const runReplayStepOnce = useCallback(async (lines?: number) => {
    if (replayBusyRef.current) return;
    replayBusyRef.current = true;
    try {
      const take = Math.max(1, Math.min(100, Number(lines || replayLinesPerStep) || 1));
      const res = await replayStep({ lines: take, auto_analyze: true });
      commitMeetingState(res.state);
      setError("");
      const d = res.replay_debug;
      setDatasetImportInfo(
        `line-mode progress ${d.queued_cursor}/${d.queued_total} (remaining=${d.queued_remaining}, last_added=${d.added})`,
      );
      setDebugEvents((rows) => [
        `${formatNowTime()} | line-step added=${d.added}, analysis-queued=${d.analyzed ? "yes" : "no"}, remaining=${d.queued_remaining}`,
        ...rows,
      ].slice(0, 80));
      if (d.warning) {
        setError(d.warning);
      }
      if (d.done) {
        stopReplayAuto("line-mode 완료: 모든 발화를 주입했습니다.");
      }
    } catch (err) {
      stopReplayAuto();
      setError((err as Error).message);
    } finally {
      replayBusyRef.current = false;
    }
  }, [commitMeetingState, replayLinesPerStep, stopReplayAuto]);

  const onStartReplayAuto = () => {
    const remaining = Number(state.replay?.queued_remaining || 0);
    if (remaining <= 0) {
      setError("먼저 라인모드 큐를 적재하세요.");
      return;
    }
    if (replayRunning) return;
    setError("");
    const interval = Math.max(100, Number(replayIntervalMs) || 1200);
    setReplayRunning(true);
    setDatasetImportInfo(`line-mode auto running (${interval}ms, step=${Math.max(1, replayLinesPerStep)})`);
    void runReplayStepOnce();
    replayTimerRef.current = window.setInterval(() => {
      void runReplayStepOnce();
    }, interval);
  };

  const onPingLlm = async () => {
    setLlmChecking(true);
    try {
      const res = await pingLlm();
      llmStatusSignatureRef.current = buildLlmStatusSignature(res.llm_status);
      setState((prev) => ({ ...prev, llm_status: res.llm_status }));
      setLlmPingOk(Boolean(res.result.ok));
      setLlmPingMessage(res.result.message || (res.result.ok ? "LLM 응답 성공" : "LLM 응답 실패"));
      setError("");
    } catch (err) {
      setLlmPingOk(false);
      setLlmPingMessage((err as Error).message);
      setError((err as Error).message);
    } finally {
      setLlmChecking(false);
    }
  };

  const onConnectLlm = async () => {
    setLlmChecking(true);
    setAnalysisPending(true);
    try {
      const res = await connectLlm();
      commitMeetingState(res.state);
      setLlmPingOk(Boolean(res.enabled));
      setLlmPingMessage(res.enabled ? "LLM 연결 완료" : (res.result?.message || "LLM 연결 실패"));
      if (!res.enabled) setError(res.result?.message || "LLM 연결 실패");
      else setError("");
    } catch (err) {
      setLlmPingOk(false);
      setLlmPingMessage((err as Error).message);
      setError((err as Error).message);
    } finally {
      setAnalysisPending(false);
      setLlmChecking(false);
    }
  };

  const onDisconnectLlm = async () => {
    setLlmChecking(true);
    try {
      const res = await disconnectLlm();
      commitMeetingState(res.state);
      setLlmPingOk(null);
      setLlmPingMessage("LLM 연결 해제됨");
      setError("");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLlmChecking(false);
    }
  };

  const stopStt = useCallback(() => {
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
    recorderRef.current = null;
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setSttRunning(false);
    setSttStatusText("STOPPED");
    setSttStatusDetail("사용자가 STT를 중지했습니다.");
    appendSttLog("stt stopped");
  }, [appendSttLog]);

  const sendChunk = useCallback(async (sessionId: number, seq: number, blob: Blob, filename: string, source: string) => {
    if (sessionId !== sttSessionRef.current) return;
    try {
      appendSttLog(`chunk #${seq} upload started (${Math.round(blob.size / 1024)} KB)`);
      const res = await transcribeChunk({ blob, filename, speaker: sttSpeaker || "시스템오디오", source });
      if (sessionId !== sttSessionRef.current) return;
      commitMeetingState(res.state);
      setLastDebug(res.stt_debug);
      if (res.stt_debug.error) appendSttLog(`chunk #${res.stt_debug.chunk_id} error: ${res.stt_debug.error}`);
      if (res.stt_debug.transcript_preview) appendSttLog(`chunk #${res.stt_debug.chunk_id} text: ${res.stt_debug.transcript_preview}`);
      setSttStatusText(res.stt_debug.status === "error" ? "ERROR" : "RUNNING");
      setSttStatusDetail(`최근 청크 #${res.stt_debug.chunk_id} (${res.stt_debug.status})`);
      setError("");
    } catch (err) {
      setError((err as Error).message);
      setSttStatusText("ERROR");
      setSttStatusDetail(`청크 업로드 실패: ${(err as Error).message}`);
      appendSttLog(`chunk #${seq} failed: ${(err as Error).message}`);
    }
  }, [appendSttLog, commitMeetingState, sttSpeaker]);

  const startStt = async () => {
    if (!navigator.mediaDevices || typeof MediaRecorder === "undefined") {
      setError("이 브라우저는 MediaRecorder를 지원하지 않습니다.");
      return;
    }
    if (sttRunning) return;

    try {
      sttSessionRef.current += 1;
      const sessionId = sttSessionRef.current;
      chunkSeqRef.current = 0;
      sendQueueRef.current = Promise.resolve();

      let rawStream: MediaStream;
      if (sttSource === "system") {
        rawStream = await navigator.mediaDevices.getDisplayMedia({ audio: true, video: true });
      } else {
        rawStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      }

      const audioTracks = rawStream.getAudioTracks();
      if (audioTracks.length === 0) {
        rawStream.getTracks().forEach((t) => t.stop());
        setSttStatusText("STOPPED");
        setSttStatusDetail("오디오 트랙이 없습니다. 화면 공유 시 탭 오디오 공유를 켜세요.");
        return;
      }

      const audioOnlyStream = new MediaStream(audioTracks);
      const mimeCandidates = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];
      const mimeType = mimeCandidates.find((m) => MediaRecorder.isTypeSupported(m));
      const sourceLabel = sttSource === "system" ? "system_audio" : "microphone";

      const recorder = mimeType ? new MediaRecorder(audioOnlyStream, { mimeType }) : new MediaRecorder(audioOnlyStream);
      recorderRef.current = recorder;
      streamRef.current = rawStream;

      recorder.ondataavailable = (event) => {
        if (!event.data || event.data.size === 0) return;
        const lowerType = (event.data.type || mimeType || "").toLowerCase();
        let ext = "webm";
        if (lowerType.includes("mp4")) ext = "mp4";
        else if (lowerType.includes("ogg")) ext = "ogg";
        else if (lowerType.includes("wav")) ext = "wav";

        const seq = ++chunkSeqRef.current;
        sendQueueRef.current = sendQueueRef.current.then(() =>
          sendChunk(sessionId, seq, event.data, `chunk.${ext}`, sourceLabel),
        );
      };

      recorder.start(5000);
      setSttRunning(true);
      setSttStatusText("RUNNING");
      setSttStatusDetail(sttSource === "system" ? "시스템 오디오 캡처 중 (5초 청크)" : "마이크 캡처 중 (5초 청크)");
      appendSttLog("stt started");
    } catch (err) {
      setError((err as Error).message);
      setSttStatusText("ERROR");
      setSttStatusDetail(`STT 시작 실패: ${(err as Error).message}`);
      stopStt();
    }
  };

  const outcomeRows = useMemo<AgendaOutcome[]>(() => {
    if (!state.analysis || !Array.isArray(state.analysis.agenda_outcomes)) return [];
    return state.analysis.agenda_outcomes as unknown as AgendaOutcome[];
  }, [state.analysis]);

  const sortedOutcomeRows = useMemo<AgendaOutcome[]>(() => {
    const toTurn = (raw: unknown): number => {
      const n = Number(raw || 0);
      return Number.isFinite(n) && n > 0 ? n : 0;
    };
    return outcomeRows
      .map((row, idx) => ({ row, idx }))
      .sort((a, b) => {
        const aStart = toTurn(a.row.start_turn_id);
        const bStart = toTurn(b.row.start_turn_id);
        const aEnd = toTurn(a.row.end_turn_id);
        const bEnd = toTurn(b.row.end_turn_id);
        const aKey = aStart > 0 ? aStart : aEnd > 0 ? aEnd : Number.MAX_SAFE_INTEGER;
        const bKey = bStart > 0 ? bStart : bEnd > 0 ? bEnd : Number.MAX_SAFE_INTEGER;
        if (aKey !== bKey) return aKey - bKey;
        if (aStart !== bStart) return aStart - bStart;
        if (aEnd !== bEnd) return aEnd - bEnd;
        return a.idx - b.idx;
      })
      .map((item) => item.row);
  }, [outcomeRows]);

  const agendas = useMemo<Agenda[]>(() => {
    if (outcomeRows.length === 0) {
      const stack = state.agenda_stack || [];
      return stack.map((row, idx) => {
        const st = String(row.status || "PROPOSED").toUpperCase();
        const status: AgendaStatus = st === "CLOSED" ? "Done" : st === "ACTIVE" || st === "CLOSING" ? "In progress" : "Not started";
        return {
          id: `agenda-${idx + 1}`,
          label: `안건 ${idx + 1}`,
          title: safeText(row.title, `안건 ${idx + 1}`),
          status,
          confidence: status === "In progress" ? 82 : 72,
          progress: statusProgress(status),
          nextUp: "다음 안건",
          keyPoints: [],
          risks: [],
          decisionSoFar: [],
          nextQuestions: [],
          keywords: [],
          summaryBullets: [],
          recommendation: "",
          lastUpdated: formatNowTime(),
        };
      });
    }
    const activeTitle = safeText(state.analysis?.agenda?.active?.title);
    const rawActiveConfidence = Number(state.analysis?.agenda?.active?.confidence ?? 85);
    const activeConfidence = Number.isFinite(rawActiveConfidence)
      ? Math.max(55, Math.min(98, Math.round(rawActiveConfidence)))
      : 85;
    const rows = sortedOutcomeRows;
    const items: Agenda[] = rows.map((row, idx) => {
      const title = safeText(row.agenda_title, `안건 ${idx + 1}`);
      const agendaState = normalizeAgendaState(row.agenda_state);
      const status = toAgendaStatus(agendaState);
      const summaryPoints = (row.agenda_summary_items || []).map((s) => safeText(s)).filter(Boolean);
      const keyPoints = (summaryPoints.length > 0 ? summaryPoints : (row.key_utterances || [])).filter(Boolean);
      const summaries = (summaryPoints.length > 0 ? summaryPoints : [safeText(row.summary)]).filter(Boolean);
      const keywords = (row.agenda_keywords || []).map((k) => safeText(k)).filter(Boolean);
      const decisionConclusions = (row.decision_results || []).map((d) => safeText(d.conclusion)).filter(Boolean);
      const actionNames = (row.action_items || []).map((a) => safeText(a.item)).filter(Boolean);
      const rid = safeText(row.agenda_id, `agenda-${idx + 1}`);
      return {
        id: rid,
        label: `안건 ${idx + 1}`,
        title,
        status,
        confidence: title === activeTitle ? activeConfidence : 78,
        progress: statusProgress(status),
        nextUp: "다음 안건",
        keyPoints,
        risks: [],
        decisionSoFar: decisionConclusions.slice(0, 6),
        nextQuestions: [],
        keywords: Array.from(new Set(keywords)).slice(0, 8),
        summaryPointIds: keyPoints.map((_, pointIdx) => `summary-${idx}-${pointIdx}`),
        summaryBullets: summaries,
        recommendation: actionNames[0] ? `우선 액션: ${actionNames[0]}` : "핵심 액션 정리가 필요합니다.",
        lastUpdated: formatNowTime(),
      };
    });

    return items.map((agenda, idx) => ({
      ...agenda,
      nextUp: items[idx + 1] ? `${items[idx + 1].label}: ${items[idx + 1].title}` : "마무리",
    }));
  }, [sortedOutcomeRows, state.agenda_stack, state.analysis?.agenda?.active?.confidence, state.analysis?.agenda?.active?.title]);

  useEffect(() => {
    if ((state.transcript?.length || 0) === 0 && agendas.length === 0) {
      setCanvasIdeas([]);
      setCanvasIdeaTitle("");
      setCanvasIdeaBody("");
      setCanvasNodePositions({});
    }
  }, [state.transcript?.length, agendas.length]);

  useEffect(() => {
    if (agendas.length === 0) {
      setSelectedAgendaId("");
      setSelectedSummaryFocus(null);
      return;
    }
    if (!selectedAgendaId || !agendas.some((agenda) => agenda.id === selectedAgendaId)) {
      const active = agendas.find((agenda) => agenda.status === "In progress") || agendas[0];
      setSelectedAgendaId(active.id);
    }
  }, [agendas, selectedAgendaId]);

  const selectedAgenda = agendas.find((agenda) => agenda.id === selectedAgendaId) || agendas[0] || null;

  const transcript = useMemo<TranscriptUtterance[]>(() => {
    const src = state.transcript || [];
    if (src.length === 0) return [];
    const sortedRanges = [...sortedOutcomeRows]
      .map((row, idx) => {
        const id = safeText(row.agenda_id, agendas[idx]?.id || `agenda-${idx + 1}`);
        return {
          id,
          start: Number(row.start_turn_id || 0),
          end: Number(row.end_turn_id || 0),
        };
      })
      .sort((a, b) => a.start - b.start);
    const activeAgendaId = selectedAgenda?.id || agendas[0]?.id || sortedRanges[sortedRanges.length - 1]?.id || "agenda-1";

    return src.map((u, idx) => {
      const text = safeText(u.text);
      const turnId = idx + 1;
      let agendaId = activeAgendaId;
      for (const range of sortedRanges) {
        if (range.start <= 0) continue;
        const end = range.end > 0 ? range.end : Number.MAX_SAFE_INTEGER;
        if (turnId >= range.start && turnId <= end) {
          agendaId = range.id;
          break;
        }
      }
      return {
        id: `utt-${turnId}`,
        timestamp: safeText(u.timestamp, formatNowTime()),
        speaker: safeText(u.speaker, "화자"),
        text,
        agendaId,
      };
    });
  }, [state.transcript, agendas, selectedAgenda?.id, sortedOutcomeRows]);

  const summaryPointMetaMap = useMemo(() => {
    const out = new Map<string, SummaryPointMeta>();
    if (sortedOutcomeRows.length === 0 || transcript.length === 0) return out;

    const transcriptByTurn = new Map<number, TranscriptUtterance>();
    transcript.forEach((utterance) => {
      const turnId = Number(utterance.id.replace("utt-", ""));
      if (!Number.isNaN(turnId) && turnId > 0) transcriptByTurn.set(turnId, utterance);
    });

    sortedOutcomeRows.forEach((row, ridx) => {
      const agendaId = safeText(row.agenda_id, agendas[ridx]?.id || agendas[0]?.id || `agenda-${ridx + 1}`);
      const summaryPoints = (row.agenda_summary_items || []).map((s) => safeText(s)).filter(Boolean);
      const keyPoints = (summaryPoints.length > 0 ? summaryPoints : (row.key_utterances || [])).filter(Boolean);
      const allRefs = (row.summary_references || []).filter(Boolean);
      const llmOpinionGroups = (row.opinion_groups || []).filter(Boolean);

      keyPoints.forEach((pointText, pointIdx) => {
        const pointId = `summary-${ridx}-${pointIdx}`;
        const pointKey = normalizeSummaryKey(pointText);
        let refs = allRefs.filter((reason) => normalizeSummaryKey(safeText(reason.why)) === pointKey);

        if (refs.length === 0 && allRefs.length > 0) {
          refs = allRefs.slice(pointIdx * 3, pointIdx * 3 + 3);
        }
        if (refs.length === 0) {
          const fallbackTs = extractTimestampToken(pointText);
          if (fallbackTs) {
            refs = allRefs.filter((reason) => extractTimestampToken(safeText(reason.timestamp)) === fallbackTs);
          }
        }

        const hitMap = new Map<string, TranscriptUtterance>();
        refs.forEach((reason) => {
          const turnId = Number(reason.turn_id || 0);
          if (turnId > 0) {
            const turnUtterance = transcriptByTurn.get(turnId);
            if (turnUtterance) {
              hitMap.set(turnUtterance.id, turnUtterance);
              return;
            }
          }

          const reasonTs = extractTimestampToken(safeText(reason.timestamp));
          const reasonSpeaker = safeText(reason.speaker);
          const reasonQuote = safeText(reason.quote);
          for (const utterance of transcript) {
            const tsMatch = !reasonTs || extractTimestampToken(utterance.timestamp) === reasonTs;
            const speakerMatch = !reasonSpeaker || utterance.speaker === reasonSpeaker;
            const quoteMatch = !reasonQuote || quoteSimilar(reasonQuote, utterance.text);
            if (tsMatch && speakerMatch && quoteMatch) {
              hitMap.set(utterance.id, utterance);
            }
          }
        });

        const turnIds = Array.from(hitMap.values())
          .map((utterance) => Number(utterance.id.replace("utt-", "")))
          .filter((num) => !Number.isNaN(num) && num > 0)
          .sort((a, b) => a - b);
        const matchedUtterances = Array.from(hitMap.values()).sort(
          (a, b) => Number(a.id.replace("utt-", "")) - Number(b.id.replace("utt-", "")),
        );
        const timeCandidates = refs.map((reason) => safeText(reason.timestamp)).filter(Boolean);
        if (turnIds.length > 0) {
          turnIds.forEach((turnId) => {
            const utt = transcriptByTurn.get(turnId);
            if (utt) timeCandidates.push(utt.timestamp);
          });
        }

        const rangeLabel = buildTimeRangeLabel(timeCandidates, pointText);
        const pointTurnSet = new Set<number>();
        turnIds.forEach((id) => pointTurnSet.add(id));
        refs.forEach((reason) => {
          const id = Number(reason.turn_id || 0);
          if (!Number.isNaN(id) && id > 0) pointTurnSet.add(id);
        });

        const llmGroups: OpinionGroup[] = [];
        if (llmOpinionGroups.length > 0) {
          let selectedGroups = llmOpinionGroups.filter((g) => {
            const ids = Array.isArray(g.evidence_turn_ids)
              ? g.evidence_turn_ids.map((v) => Number(v)).filter((v) => !Number.isNaN(v) && v > 0)
              : [];
            if (ids.length === 0) return false;
            return ids.some((id) => pointTurnSet.has(id));
          });
          if (selectedGroups.length === 0 && keyPoints.length === 1) {
            selectedGroups = llmOpinionGroups;
          }

          selectedGroups.forEach((g, gidx) => {
            const typ = normalizeOpinionType(safeText(g.type, "info"));
            const summary = safeText(g.summary);
            if (!summary) return;
            const ids = Array.isArray(g.evidence_turn_ids)
              ? g.evidence_turn_ids.map((v) => Number(v)).filter((v) => !Number.isNaN(v) && v > 0)
              : [];
            const utterances = ids
              .map((id) => transcriptByTurn.get(id))
              .filter((u): u is TranscriptUtterance => Boolean(u))
              .sort((a, b) => Number(a.id.replace("utt-", "")) - Number(b.id.replace("utt-", "")));
            const llmRange = buildTimeRangeLabel(utterances.map((u) => u.timestamp), pointText);
            llmGroups.push({
              id: `${pointId}-llm-op-${gidx + 1}`,
              type: typ,
              typeLabel: opinionTypeLabel(typ),
              summary,
              detail: "",
              rangeLabel: llmRange,
              utterances,
            });
          });
        }

        const fallbackGroups = buildOpinionGroups(agendaId, pointId, matchedUtterances, refs);
        out.set(`${agendaId}|${pointId}`, {
          agendaId,
          pointId,
          pointText,
          rangeLabel,
          turnIds,
          references: refs,
          opinionGroups: llmGroups.length > 0 ? llmGroups : fallbackGroups,
        });
      });
    });

    return out;
  }, [sortedOutcomeRows, agendas, transcript]);

  useEffect(() => {
    if (!selectedSummaryFocus) return;
    const key = `${selectedSummaryFocus.agendaId}|${selectedSummaryFocus.pointId}`;
    if (!summaryPointMetaMap.has(key)) {
      setSelectedSummaryFocus(null);
      return;
    }
    if (!agendas.some((agenda) => agenda.id === selectedSummaryFocus.agendaId)) {
      setSelectedSummaryFocus(null);
    }
  }, [selectedSummaryFocus, summaryPointMetaMap, agendas]);

  const decisions = useMemo<DecisionItem[]>(() => {
    if (sortedOutcomeRows.length === 0) return [];
    const out: DecisionItem[] = [];
    sortedOutcomeRows.forEach((row, ridx) => {
      const title = safeText(row.agenda_title, agendas[0]?.title || "");
      const agendaId = safeText(row.agenda_id, agendas[ridx]?.id || agendas[0]?.id || "agenda-1");
      (row.decision_results || []).forEach((decision, didx) => {
        const conclusion = safeText(decision.conclusion, "결론 미정");
        const opinions = (decision.opinions || []).filter(Boolean);
        let finalStatus: DecisionItem["finalStatus"] = "Approved";
        if (/보류|pending/i.test(conclusion)) finalStatus = "Pending";
        if (/반려|거절|rejected/i.test(conclusion)) finalStatus = "Rejected";
        out.push({
          id: `decision-${ridx}-${didx}`,
          agendaId,
          issue: title,
          options: opinions.length > 0 ? opinions : ["의견 요약 없음"],
          finalStatus,
          confidence: 80,
          evidence: [],
        });
      });
    });
    return out;
  }, [sortedOutcomeRows, agendas]);

  const actionItems = useMemo<ActionItem[]>(() => {
    if (sortedOutcomeRows.length === 0) return [];
    const out: ActionItem[] = [];
    sortedOutcomeRows.forEach((row, ridx) => {
      const agendaId = safeText(row.agenda_id, agendas[ridx]?.id || agendas[0]?.id || "agenda-1");
      (row.action_items || []).forEach((action, aidx) => {
        const evidence = (action.reasons || []).map((reason) => safeText(reason.timestamp)).filter(Boolean);
        out.push({
          id: `action-${ridx}-${aidx}`,
          agendaId,
          action: safeText(action.item, "액션 항목 미정"),
          owner: safeText(action.owner, "-"),
          due: safeText(action.due, "-"),
          status: "Open",
          evidence,
        });
      });
    });
    return out;
  }, [sortedOutcomeRows, agendas]);

  const evidenceLog = useMemo<EvidenceItem[]>(() => {
    if (sortedOutcomeRows.length === 0) return [];
    const out: EvidenceItem[] = [];
    sortedOutcomeRows.forEach((row, ridx) => {
      const agendaId = safeText(row.agenda_id, agendas[ridx]?.id || agendas[0]?.id || "agenda-1");
      const agendaTitle = safeText(row.agenda_title, agendas[ridx]?.title || "안건");
      const summaryItems = (row.agenda_summary_items || []).map((s) => safeText(s)).filter(Boolean);
      (row.summary_references || []).forEach((reason, qidx) => {
        const summaryTarget = safeText(reason.why);
        const targetLabel =
          summaryTarget && summaryTarget !== "요약 근거"
            ? summaryTarget
            : safeText(summaryItems[qidx] || summaryItems[0], "요약 항목");
        out.push({
          id: `evidence-summary-${ridx}-${qidx}`,
          agendaId,
          agendaTitle,
          supports: "Summary",
          targetId: `summary-${ridx}-${Math.min(qidx, Math.max(0, summaryItems.length - 1))}`,
          targetLabel,
          quote: safeText(reason.quote, "요약 근거 없음"),
          timestamp: safeText(reason.timestamp, "--:--"),
          speaker: safeText(reason.speaker, "화자"),
        });
      });
      (row.decision_results || []).forEach((decision, didx) => {
        const conclusion = safeText(decision.conclusion, `의사결정 ${didx + 1}`);
        (decision.opinions || []).forEach((opinion, oidx) => {
          const line = safeText(opinion);
          if (!line) return;
          const tsMatch = line.match(/\[(\d{2}:\d{2}(?::\d{2})?)\]/);
          const ts = tsMatch ? tsMatch[1] : "--:--";
          const quote = line.replace(/^\[\d{2}:\d{2}(?::\d{2})?\]\s*/, "").trim();
          out.push({
            id: `evidence-decision-${ridx}-${didx}-${oidx}`,
            agendaId,
            agendaTitle,
            supports: "Decision",
            targetId: `decision-${ridx}-${didx}`,
            targetLabel: conclusion,
            quote: quote || line,
            timestamp: ts,
            speaker: "토론자",
          });
        });
      });
      (row.action_items || []).forEach((action, aidx) => {
        (action.reasons || []).forEach((reason, qidx) => {
          out.push({
            id: `evidence-${ridx}-${aidx}-${qidx}`,
            agendaId,
            agendaTitle,
            supports: "Action",
            targetId: `action-${ridx}-${aidx}`,
            targetLabel: safeText(action.item, `액션 ${aidx + 1}`),
            quote: safeText(reason.quote, "근거 발언 없음"),
            timestamp: safeText(reason.timestamp, "--:--"),
            speaker: safeText(reason.speaker, "화자"),
          });
        });
      });
    });
    return out;
  }, [sortedOutcomeRows, agendas]);

  const participantRoster = useMemo<Participant[]>(() => {
    const counts = new Map<string, number>();
    transcript.forEach((u) => {
      counts.set(u.speaker, (counts.get(u.speaker) || 0) + 1);
    });
    const rows = Array.from(counts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8)
      .map(([name], idx) => ({
        name,
        role: "토론자",
        status: idx === 0 ? "Speaking" : idx < 3 ? "Active" : "Listening",
      })) as Participant[];
    return rows;
  }, [transcript]);

  const meeting = useMemo(() => {
    const lastTs = transcript.length > 0 ? safeText(transcript[transcript.length - 1]?.timestamp, "-") : "-";
    return {
      title: safeText(state.meeting_goal, "회의"),
      date: "실시간",
      duration: "-",
      participants: `참여자 ${participantRoster.length}명`,
      elapsed: `전사 ${transcript.length}건`,
      lastUpdated: lastTs,
    };
  }, [state.meeting_goal, participantRoster.length, transcript]);

  const speakerOptions = useMemo(() => ["전체", ...new Set(transcript.map((utterance) => utterance.speaker))], [transcript]);

  const agendaOverview = useMemo(() => {
    const done = agendas.filter((agenda) => agenda.status === "Done").length;
    const inProgress = agendas.filter((agenda) => agenda.status === "In progress").length;
    const notStarted = agendas.filter((agenda) => agenda.status === "Not started").length;
    const averageConfidence =
      agendas.length === 0 ? 0 : Math.round(agendas.reduce((total, agenda) => total + agenda.confidence, 0) / agendas.length);
    return { done, inProgress, notStarted, averageConfidence };
  }, [agendas]);

  const selectedContext = useMemo(() => {
    if (!selectedAgenda) {
      return { transcriptCount: 0, evidenceCount: 0, decisionCount: 0, actionCount: 0, openActionCount: 0 };
    }
    const transcriptCount = transcript.filter((utterance) => utterance.agendaId === selectedAgenda.id).length;
    const evidenceCount = evidenceLog.filter((evidence) => evidence.agendaId === selectedAgenda.id).length;
    const scopedActions = actionItems.filter((action) => action.agendaId === selectedAgenda.id);
    return {
      transcriptCount,
      evidenceCount,
      decisionCount: decisions.filter((decision) => decision.agendaId === selectedAgenda.id).length,
      actionCount: scopedActions.length,
      openActionCount: scopedActions.filter((action) => action.status !== "Done").length,
    };
  }, [selectedAgenda, transcript, evidenceLog, actionItems, decisions]);

  const transcriptCountByAgenda = useMemo(() => {
    const counts = new Map<string, number>();
    transcript.forEach((utterance) => {
      counts.set(utterance.agendaId, (counts.get(utterance.agendaId) || 0) + 1);
    });
    return counts;
  }, [transcript]);

  const agendaUtterancesMap = useMemo(() => {
    const grouped = new Map<string, TranscriptUtterance[]>();
    transcript.forEach((utterance) => {
      const current = grouped.get(utterance.agendaId);
      if (current) current.push(utterance);
      else grouped.set(utterance.agendaId, [utterance]);
    });
    return grouped;
  }, [transcript]);

  const outcomeByAgendaMap = useMemo(() => {
    const out = new Map<string, AgendaOutcome>();
    sortedOutcomeRows.forEach((row, idx) => {
      const fallbackId = agendas[idx]?.id || `agenda-${idx + 1}`;
      out.set(safeText(row.agenda_id, fallbackId), row);
    });
    return out;
  }, [sortedOutcomeRows, agendas]);

  const canvasLanes = useMemo<CanvasLane[]>(() => {
    if (agendas.length === 0) return [];

    const pointsByAgenda = new Map<string, SummaryPointMeta[]>();
    summaryPointMetaMap.forEach((meta) => {
      const current = pointsByAgenda.get(meta.agendaId);
      if (current) current.push(meta);
      else pointsByAgenda.set(meta.agendaId, [meta]);
    });

    return agendas.map((agenda) => {
      const row = outcomeByAgendaMap.get(agenda.id);
      const startTurnId = Number(row?.start_turn_id || 0);
      const endTurnId = Number(row?.end_turn_id || 0);
      const timeCandidates: string[] = [];
      if (startTurnId > 0 && transcript[startTurnId - 1]) timeCandidates.push(transcript[startTurnId - 1].timestamp);
      if (endTurnId > 0 && transcript[endTurnId - 1]) timeCandidates.push(transcript[endTurnId - 1].timestamp);
      const groupedPoints = (pointsByAgenda.get(agenda.id) || [])
        .slice()
        .sort((a, b) => {
          const aTurn = a.turnIds[0] || Number.MAX_SAFE_INTEGER;
          const bTurn = b.turnIds[0] || Number.MAX_SAFE_INTEGER;
          return aTurn - bTurn;
        })
        .map((meta) => ({
          pointId: meta.pointId,
          pointText: stripLeadingTimestamp(meta.pointText),
          rangeLabel: meta.rangeLabel,
          utteranceCount: Math.max(meta.turnIds.length, meta.references.length),
          opinionCount: meta.opinionGroups.length,
        }));
      const ideaNodes = canvasIdeas.filter((idea) => idea.agendaId === agenda.id);
      return {
        agendaId: agenda.id,
        agendaLabel: agenda.label,
        agendaTitle: agenda.title,
        status: agenda.status,
        flowType: safeText(row?.flow_type, "discussion"),
        timeLabel: buildTimeRangeLabel(timeCandidates),
        keywordLabel: (row?.agenda_keywords || []).slice(0, 3).join(" · "),
        transcriptCount: Number(transcriptCountByAgenda.get(agenda.id) || 0),
        summaryNodes: groupedPoints,
        ideaNodes,
      };
    });
  }, [agendas, summaryPointMetaMap, outcomeByAgendaMap, canvasIdeas, transcript, transcriptCountByAgenda]);

  const canvasGraph = useMemo(() => {
    const nodes: CanvasGraphNode[] = [];
    const edges: CanvasGraphEdge[] = [];
    if (canvasLanes.length === 0) {
      return { nodes, edges, width: 1800, height: 960 };
    }

    const nodeMap = new Map<string, CanvasGraphNode>();
    const agendaColumnX = 96;
    const summaryColumnX = 500;
    const ideaColumnX = 904;
    const laneGapY = 144;
    const summaryGapY = 164;
    const ideaGapY = 152;
    const laneMinHeight = 240;
    let currentLaneY = 96;
    let maxX = 1500;
    let maxY = 860;

    canvasLanes.forEach((lane) => {
      const agendaNodeId = `canvas-agenda-${lane.agendaId}`;
      const summaryCount = Math.max(1, lane.summaryNodes.length);
      const summarySlots = lane.summaryNodes.map((node, summaryIdx) => ({
        pointId: node.pointId,
        title: node.pointText,
        rangeLabel: node.rangeLabel,
        utteranceCount: node.utteranceCount,
        opinionCount: node.opinionCount,
        x: summaryColumnX,
        y: currentLaneY + summaryIdx * summaryGapY,
      }));
      const summaryYByPointId = new Map(summarySlots.map((slot) => [slot.pointId, slot.y]));
      const linkedIdeaCounts = new Map<string, number>();
      let unlinkedIdeaCount = 0;
      const ideaSlots = lane.ideaNodes.map((idea) => {
        const linkedPointId = safeText(idea.linkedPointId);
        if (linkedPointId && summaryYByPointId.has(linkedPointId)) {
          const offset = linkedIdeaCounts.get(linkedPointId) || 0;
          linkedIdeaCounts.set(linkedPointId, offset + 1);
          return {
            idea,
            x: ideaColumnX,
            y: Number(summaryYByPointId.get(linkedPointId)) + offset * ideaGapY,
          };
        }

        const slot = {
          idea,
          x: ideaColumnX,
          y: currentLaneY + summaryCount * summaryGapY + unlinkedIdeaCount * ideaGapY,
        };
        unlinkedIdeaCount += 1;
        return slot;
      });

      const laneBottom = Math.max(
        currentLaneY + laneMinHeight,
        ...summarySlots.map((slot) => slot.y + 132),
        ...ideaSlots.map((slot) => slot.y + 148),
      );
      const laneHeight = laneBottom - currentLaneY;

      const agendaNode: CanvasGraphNode = {
        id: agendaNodeId,
        agendaId: lane.agendaId,
        kind: "agenda",
        title: lane.agendaTitle,
        body: lane.keywordLabel,
        subtitle: `${lane.agendaLabel} · ${agendaStatusLabel[lane.status]}`,
        meta: [lane.timeLabel, `${lane.transcriptCount}개 발화`, lane.flowType],
        width: 300,
        height: 156,
        x: agendaColumnX,
        y: currentLaneY + Math.max(0, (laneHeight - 156) / 2),
      };
      nodeMap.set(agendaNodeId, agendaNode);

      summarySlots.forEach((slot) => {
        const summaryNodeId = `canvas-summary-${lane.agendaId}-${slot.pointId}`;
        const summaryNode: CanvasGraphNode = {
          id: summaryNodeId,
          agendaId: lane.agendaId,
          kind: "summary",
          title: slot.title,
          body: "",
          subtitle: "Summary Node",
          meta: [slot.rangeLabel, `원문 ${slot.utteranceCount}`, `의견 ${slot.opinionCount}`],
          pointId: slot.pointId,
          width: 240,
          height: 132,
          x: slot.x,
          y: slot.y,
        };
        nodeMap.set(summaryNodeId, summaryNode);
        edges.push({
          id: `edge-${agendaNodeId}-${summaryNodeId}`,
          fromId: agendaNodeId,
          toId: summaryNodeId,
          kind: "agenda-summary",
        });
      });

      ideaSlots.forEach(({ idea, x, y }) => {
        const ideaNodeId = idea.id;
        const ideaNode: CanvasGraphNode = {
          id: ideaNodeId,
          agendaId: lane.agendaId,
          kind: "idea",
          title: idea.title,
          body: idea.body,
          subtitle: `Idea Note · ${idea.createdAt}`,
          meta: [idea.linkedPointId ? "요약 노드 연결" : "안건 메모"],
          linkedPointText: idea.linkedPointText,
          pointId: idea.linkedPointId,
          width: 248,
          height: 148,
          x,
          y,
        };
        nodeMap.set(ideaNodeId, ideaNode);

        const linkedSummaryId = idea.linkedPointId ? `canvas-summary-${lane.agendaId}-${idea.linkedPointId}` : "";
        edges.push({
          id: `edge-${linkedSummaryId || agendaNodeId}-${ideaNodeId}`,
          fromId: linkedSummaryId || agendaNodeId,
          toId: ideaNodeId,
          kind: linkedSummaryId ? "summary-idea" : "agenda-idea",
        });
      });

      currentLaneY = laneBottom + laneGapY;
      maxX = Math.max(maxX, ideaColumnX + 248 + 180);
      maxY = Math.max(maxY, laneBottom + laneGapY);
    });

    nodeMap.forEach((node, id) => {
      const override = canvasNodePositions[id];
      const nextNode = override ? { ...node, x: override.x, y: override.y } : node;
      nodes.push(nextNode);
      maxX = Math.max(maxX, nextNode.x + nextNode.width + 160);
      maxY = Math.max(maxY, nextNode.y + nextNode.height + 180);
    });

    return {
      nodes,
      edges,
      width: Math.max(1800, maxX),
      height: Math.max(960, maxY),
    };
  }, [canvasLanes, canvasNodePositions]);

  useEffect(() => {
    setFlowNodes((prev) => {
      const prevMap = new Map(prev.map((node) => [node.id, node]));
      return canvasGraph.nodes.map((node) => {
        const prevNode = prevMap.get(node.id);
        const linkedMeta = node.linkedPointText ? [...node.meta, stripLeadingTimestamp(node.linkedPointText)] : node.meta;
        return {
          id: node.id,
          position: prevNode?.position || { x: node.x, y: node.y },
          sourcePosition: Position.Right,
          targetPosition: Position.Left,
          className: `rfNode rfNode${node.kind === "agenda" ? "Agenda" : node.kind === "summary" ? "Summary" : "Idea"} ${selectedAgenda?.id === node.agendaId ? "rfNodeEmphasized" : ""}`,
          data: {
            label: renderCanvasFlowLabel({
              kind: node.kind,
              subtitle: node.subtitle,
              title: node.title,
              body: node.body,
              meta: linkedMeta,
            }),
            nodeId: node.id,
            agendaId: node.agendaId,
            kind: node.kind,
            title: node.title,
            body: node.body,
            subtitle: node.subtitle,
            meta: linkedMeta,
            pointId: node.pointId,
            linkedPointText: node.linkedPointText,
          },
          draggable: true,
          selectable: true,
          selected: prevNode?.selected || false,
        };
      });
    });
    setFlowEdges((prev) => {
      const customEdges = prev.filter((edge) => !String(edge.id).startsWith("base-"));
      const baseEdges = canvasGraph.edges.map((edge) => ({
        id: `base-${edge.id}`,
        source: edge.fromId,
        target: edge.toId,
        type: "step",
        deletable: false,
        selectable: true,
        reconnectable: true,
      }));
      return [...baseEdges, ...customEdges];
    });
  }, [canvasGraph, selectedAgenda?.id]);

  const onFlowNodesChange = useCallback((changes: NodeChange<Node<CanvasFlowNodeData>>[]) => {
    setFlowNodes((nodes) => applyNodeChanges(changes, nodes));
  }, []);

  const onFlowEdgesChange = useCallback((changes: EdgeChange<Edge>[]) => {
    setFlowEdges((edges) => applyEdgeChanges(changes, edges));
  }, []);

  const onFlowConnect = useCallback((connection: Connection) => {
    setFlowEdges((edges) =>
      addEdge(
        {
          ...connection,
          id: `user-edge-${Date.now()}`,
          type: "step",
          reconnectable: true,
        },
        edges,
      ),
    );
  }, []);

  const onFlowNodeDragStop = useCallback((_: unknown, node: Node<CanvasFlowNodeData>) => {
    setCanvasNodePositions((prev) => ({
      ...prev,
      [node.id]: { x: node.position.x, y: node.position.y },
    }));
  }, []);

  const filteredTranscript = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    const baseTranscript = selectedSummaryFocus ? selectedSummaryFocus.utterances : transcript;
    return baseTranscript.filter((utterance) => {
      const speakerMatch = speakerFilter === "전체" || utterance.speaker === speakerFilter;
      const queryMatch =
        normalizedQuery.length === 0 ||
        utterance.text.toLowerCase().includes(normalizedQuery) ||
        utterance.speaker.toLowerCase().includes(normalizedQuery) ||
        utterance.timestamp.includes(normalizedQuery);
      return speakerMatch && queryMatch;
    });
  }, [query, speakerFilter, transcript, selectedSummaryFocus]);

  const onTranscriptScroll = useCallback(() => {
    const el = transcriptListRef.current;
    if (!el) return;
    const atBottom = isNearBottom(el, 18);
    setTranscriptAutoFollow(atBottom);
    if (atBottom) {
      setPendingTranscriptCount(0);
    }
  }, []);

  useEffect(() => {
    if (!transcriptInitRef.current) {
      transcriptInitRef.current = true;
      transcriptPrevCountRef.current = transcript.length;
      requestAnimationFrame(() => {
        const el = transcriptListRef.current;
        if (el) {
          el.scrollTop = el.scrollHeight;
          setTranscriptAutoFollow(true);
          setPendingTranscriptCount(0);
        }
      });
      return;
    }

    const prev = transcriptPrevCountRef.current;
    const next = transcript.length;
    const added = Math.max(0, next - prev);
    transcriptPrevCountRef.current = next;
    if (added <= 0) return;

    if (selectedSummaryFocus) return;

    const el = transcriptListRef.current;
    if (!el) return;
    const atBottomNow = isNearBottom(el, 18);
    if (atBottomNow || transcriptAutoFollow) {
      requestAnimationFrame(() => {
        scrollTranscriptToBottom("auto");
      });
      return;
    }

    setPendingTranscriptCount((count) => count + added);
  }, [transcript.length, transcriptAutoFollow, selectedSummaryFocus, scrollTranscriptToBottom]);

  const summaryAgendas = useMemo(() => {
    if (!selectedAgenda) return [];
    if (summaryScope === "all") return agendas;
    return agendas.filter((agenda) => agenda.id === selectedAgenda.id);
  }, [selectedAgenda, summaryScope, agendas]);

  const summaryEvidence = useMemo(() => {
    const base =
      summaryScope === "all"
        ? [...evidenceLog]
        : !selectedAgenda
          ? []
          : evidenceLog.filter((evidence) => evidence.agendaId === selectedAgenda.id);
    const summaryOnly = base.filter((evidence) => evidence.supports === "Summary");
    if (summaryOnly.length > 0) return summaryOnly.slice(-12).reverse();
    return base.slice(-8).reverse();
  }, [selectedAgenda, summaryScope, evidenceLog]);

  const bottomAgendas = agendas.filter((agenda) => agenda.status === "Done");
  const bottomDecisions = decisions;
  const bottomActions = actionItems;
  const bottomEvidence = evidenceLog;
  const groupedBottomEvidence = useMemo(() => {
    if (bottomEvidence.length === 0) return [];
    const agendaOrder = new Map(agendas.map((agenda, idx) => [agenda.id, idx]));
    const groups = new Map<string, { agendaId: string; agendaTitle: string; items: EvidenceItem[] }>();
    for (const item of bottomEvidence) {
      const agendaTitle = item.agendaTitle || agendas.find((a) => a.id === item.agendaId)?.title || item.agendaId;
      const existing = groups.get(item.agendaId);
      if (!existing) {
        groups.set(item.agendaId, { agendaId: item.agendaId, agendaTitle, items: [item] });
      } else {
        existing.items.push(item);
      }
    }
    const rows = Array.from(groups.values());
    rows.sort((a, b) => (agendaOrder.get(a.agendaId) ?? 9999) - (agendaOrder.get(b.agendaId) ?? 9999));
    return rows;
  }, [bottomEvidence, agendas]);

  const llmEnabled = Boolean(state.llm_enabled);
  const analysisUiDisabled = analysisPending;

  const moveToWorkspaceFromCanvas = useCallback((reason: string) => {
    setCanvasReturnContext(reason);
    setActiveSection("workspace");
  }, []);

  const returnToCanvas = useCallback(() => {
    setActiveSection("canvas");
  }, []);

  const onSelectAgenda = useCallback((agendaId: string) => {
    if (analysisUiDisabled) return;
    setSelectedAgendaId(agendaId);
    setSummaryScope("current");
    setSelectedSummaryFocus(null);
  }, [analysisUiDisabled]);

  const jumpToTranscript = (agendaId: string, timestamp: string) => {
    if (analysisUiDisabled) return;
    moveToWorkspaceFromCanvas("전사 원문 확인 중");
    setSelectedSummaryFocus(null);
    setSelectedAgendaId(agendaId);
    setQuery(timestamp);
  };

  const resolveSummaryFocusUtterances = useCallback((meta: SummaryPointMeta): TranscriptUtterance[] => {
    const hitMap = new Map<string, TranscriptUtterance>();
    for (const turnId of meta.turnIds) {
      if (turnId <= 0 || turnId > transcript.length) continue;
      const utterance = transcript[turnId - 1];
      if (utterance) hitMap.set(utterance.id, utterance);
    }
    if (hitMap.size === 0) {
      for (const reason of meta.references) {
        const reasonTs = extractTimestampToken(safeText(reason.timestamp));
        const reasonSpeaker = safeText(reason.speaker);
        const reasonQuote = safeText(reason.quote);
        for (const utterance of transcript) {
          const tsMatch = !reasonTs || extractTimestampToken(utterance.timestamp) === reasonTs;
          const speakerMatch = !reasonSpeaker || utterance.speaker === reasonSpeaker;
          const quoteMatch = !reasonQuote || quoteSimilar(reasonQuote, utterance.text);
          if (tsMatch && speakerMatch && quoteMatch) {
            hitMap.set(utterance.id, utterance);
          }
        }
      }
    }
    const ordered = Array.from(hitMap.values());
    ordered.sort((a, b) => Number(a.id.replace("utt-", "")) - Number(b.id.replace("utt-", "")));
    return ordered;
  }, [transcript]);

  const jumpBySummary = useCallback((agendaId: string, summaryText: string, pointId: string) => {
    if (analysisUiDisabled) return;
    moveToWorkspaceFromCanvas("요약 근거 확인 중");
    setSelectedAgendaId(agendaId);
    setSummaryScope("current");
    setQuery("");
    setSpeakerFilter("전체");

    const meta = summaryPointMetaMap.get(`${agendaId}|${pointId}`);
    if (meta) {
      const utterances = resolveSummaryFocusUtterances(meta);
      setSelectedSummaryFocus({
        ...meta,
        pointText: stripLeadingTimestamp(summaryText) || stripLeadingTimestamp(meta.pointText),
        utterances,
      });
      return;
    }

    setSelectedSummaryFocus(null);
    const ts = extractTimestampToken(summaryText);
    if (ts) setQuery(ts);
  }, [analysisUiDisabled, moveToWorkspaceFromCanvas, resolveSummaryFocusUtterances, summaryPointMetaMap]);

  const focusByOpinionGroup = (agendaId: string, pointId: string, groupId: string) => {
    if (analysisUiDisabled) return;
    setSelectedAgendaId(agendaId);
    setActiveSection("workspace");
    setSummaryScope("current");
    setQuery("");
    setSpeakerFilter("전체");

    const meta = summaryPointMetaMap.get(`${agendaId}|${pointId}`);
    if (!meta) return;
    const group = meta.opinionGroups.find((item) => item.id === groupId);
    if (!group) return;

    const groupTurnIds = group.utterances
      .map((u) => Number(u.id.replace("utt-", "")))
      .filter((n) => !Number.isNaN(n) && n > 0)
      .sort((a, b) => a - b);

    setSelectedSummaryFocus({
      ...meta,
      pointText: `[${group.typeLabel}] ${group.summary}`,
      rangeLabel: group.rangeLabel,
      turnIds: groupTurnIds,
      utterances: group.utterances,
    });
  };

  const focusCanvasOpinionGroup = useCallback((agendaId: string, pointId: string, groupId: string) => {
    const meta = summaryPointMetaMap.get(`${agendaId}|${pointId}`);
    if (!meta) return;
    const group = meta.opinionGroups.find((item) => item.id === groupId);
    if (!group) return;

    setSelectedAgendaId(agendaId);
    setSummaryScope("current");
    setSelectedSummaryFocus({
      ...meta,
      pointText: `[${group.typeLabel}] ${group.summary}`,
      rangeLabel: group.rangeLabel,
      turnIds: group.utterances
        .map((u) => Number(u.id.replace("utt-", "")))
        .filter((n) => !Number.isNaN(n) && n > 0)
        .sort((a, b) => a - b),
      utterances: group.utterances,
    });
    setCanvasNodeDetail((prev) => {
      if (!prev || prev.agendaId !== agendaId || prev.pointId !== pointId) return prev;
      return {
        ...prev,
        title: `[${group.typeLabel}] ${group.summary}`,
        subtitle: "의견 요약",
        badges: [group.rangeLabel, `원문 ${group.utterances.length}문장`],
        summaryLines: [group.summary, ...(group.detail ? [group.detail] : [])],
        utterances: group.utterances,
      };
    });
  }, [summaryPointMetaMap]);

  const openCanvasAgendaDetail = useCallback((agendaId: string) => {
    const agenda = agendas.find((item) => item.id === agendaId);
    if (!agenda) return;
    const row = outcomeByAgendaMap.get(agendaId);
    const summaryLines = [
      ...((row?.agenda_summary_items || []).map((item) => safeText(item)).filter(Boolean)),
      ...((row?.summary ? [safeText(row.summary)] : []).filter(Boolean)),
    ].filter(Boolean);
    const utterances = (agendaUtterancesMap.get(agendaId) || []).slice();
    setSelectedAgendaId(agendaId);
    setSummaryScope("current");
    setSelectedSummaryFocus(null);
    setCanvasRightRailOpen(true);
    setCanvasNodeDetail({
      id: `detail-${agendaId}`,
      kind: "agenda",
      agendaId,
      title: agenda.title,
      subtitle: `${agenda.label} · ${agendaStatusLabel[agenda.status]}`,
      badges: [safeText(row?.flow_type, "discussion"), `${utterances.length}개 발화`],
      summaryLines: summaryLines.length > 0 ? summaryLines : ["요약이 아직 없습니다."],
      opinionGroups: [],
      utterances,
    });
  }, [agendas, outcomeByAgendaMap, agendaUtterancesMap]);

  const openCanvasSummaryDetail = useCallback((agendaId: string, pointId: string, summaryText: string) => {
    const meta = summaryPointMetaMap.get(`${agendaId}|${pointId}`);
    if (!meta) return;
    const utterances = resolveSummaryFocusUtterances(meta);
    const focusState = {
      ...meta,
      pointText: stripLeadingTimestamp(summaryText) || stripLeadingTimestamp(meta.pointText),
      utterances,
    };
    setSelectedAgendaId(agendaId);
    setSummaryScope("current");
    setSelectedSummaryFocus(focusState);
    setCanvasRightRailOpen(true);
    setCanvasNodeDetail({
      id: `detail-${agendaId}-${pointId}`,
      kind: "summary",
      agendaId,
      pointId,
      title: focusState.pointText,
      subtitle: "요약 노드",
      badges: [focusState.rangeLabel, `원문 ${utterances.length}문장`],
      summaryLines: [focusState.pointText],
      opinionGroups: meta.opinionGroups,
      utterances,
    });
  }, [resolveSummaryFocusUtterances, summaryPointMetaMap]);

  const openCanvasIdeaDetail = useCallback((idea: CanvasIdea) => {
    const utterances =
      idea.linkedPointId && summaryPointMetaMap.has(`${idea.agendaId}|${idea.linkedPointId}`)
        ? resolveSummaryFocusUtterances(summaryPointMetaMap.get(`${idea.agendaId}|${idea.linkedPointId}`) as SummaryPointMeta)
        : (agendaUtterancesMap.get(idea.agendaId) || []).slice(-10);
    const linkedMeta = idea.linkedPointId ? summaryPointMetaMap.get(`${idea.agendaId}|${idea.linkedPointId}`) : undefined;
    if (linkedMeta) {
      setSelectedSummaryFocus({
        ...linkedMeta,
        utterances: resolveSummaryFocusUtterances(linkedMeta),
      });
    } else {
      setSelectedSummaryFocus(null);
    }
    setSelectedAgendaId(idea.agendaId);
    setSummaryScope("current");
    setCanvasRightRailOpen(true);
    setCanvasNodeDetail({
      id: `detail-${idea.id}`,
      kind: "idea",
      agendaId: idea.agendaId,
      pointId: idea.linkedPointId,
      title: idea.title,
      subtitle: `아이디어 메모 · ${idea.createdAt}`,
      badges: [idea.linkedPointId ? "요약 연결" : "안건 메모"],
      summaryLines: [safeText(idea.body, "메모 본문 없음"), ...(idea.linkedPointText ? [stripLeadingTimestamp(idea.linkedPointText)] : [])],
      opinionGroups: [],
      utterances,
      noteBody: safeText(idea.body),
    });
  }, [agendaUtterancesMap, resolveSummaryFocusUtterances, summaryPointMetaMap]);

  const addCanvasIdea = () => {
    if (analysisUiDisabled) return;
    const agendaId = selectedAgenda?.id || agendas[0]?.id || "";
    if (!agendaId) return;

    const body = safeText(canvasIdeaBody);
    const title = safeText(canvasIdeaTitle, body.slice(0, 42));
    if (!title && !body) return;

    const toneCycle: CanvasIdea["colorTone"][] = ["blue", "mint", "amber", "rose"];
    const tone = toneCycle[canvasIdeas.length % toneCycle.length];
    const linkedPointId = selectedSummaryFocus?.agendaId === agendaId ? selectedSummaryFocus.pointId : undefined;
    const linkedPointText = selectedSummaryFocus?.agendaId === agendaId ? selectedSummaryFocus.pointText : undefined;
    const agendaIndex = Math.max(0, agendas.findIndex((agenda) => agenda.id === agendaId));
    const agendaIdeaCount = canvasIdeas.filter((idea) => idea.agendaId === agendaId).length;
    const nextId = `canvas-idea-${Date.now()}`;
    const baseX = 240 + agendaIndex * 430 + (linkedPointId ? 250 : 120) + (agendaIdeaCount % 2) * 24;
    const baseY = 260 + agendaIdeaCount * 152;
    setCanvasIdeas((prev) => [
      {
        id: nextId,
        agendaId,
        title,
        body,
        createdAt: formatNowTime(),
        linkedPointId,
        linkedPointText,
        colorTone: tone,
      },
      ...prev,
    ]);
    setCanvasNodePositions((prev) => ({
      ...prev,
      [nextId]: { x: baseX, y: baseY },
    }));
    setCanvasIdeaTitle("");
    setCanvasIdeaBody("");
    setCanvasComposerOpen(false);
  };

  const openCanvasNode = useCallback((node: Node<CanvasFlowNodeData>) => {
    if (analysisUiDisabled) return;
    if (node.data.kind === "agenda") {
      openCanvasAgendaDetail(node.data.agendaId);
      return;
    }
    if (node.data.kind === "summary" && node.data.pointId) {
      openCanvasSummaryDetail(node.data.agendaId, node.data.pointId, node.data.title);
      return;
    }
    if (node.data.kind === "idea") {
      const sourceIdea = canvasIdeas.find((idea) => idea.id === node.id);
      if (sourceIdea) openCanvasIdeaDetail(sourceIdea);
    }
  }, [analysisUiDisabled, canvasIdeas, openCanvasAgendaDetail, openCanvasSummaryDetail, openCanvasIdeaDetail]);

  const onFlowNodeClick = useCallback((_: unknown, node: Node<CanvasFlowNodeData>) => {
    openCanvasNode(node);
  }, [openCanvasNode]);

  const focusTargetCard = (agendaId: string, targetId: string) => {
    if (analysisUiDisabled) return;
    const cleanTarget = safeText(targetId);
    if (!cleanTarget) return;

    moveToWorkspaceFromCanvas("근거 카드 확인 중");
    setSelectedAgendaId(agendaId);
    setSummaryScope("current");

    const domId = `evi-target-${cleanTarget}`;
    window.setTimeout(() => {
      const el = document.getElementById(domId);
      if (!el) return;
      el.scrollIntoView({ behavior: "smooth", block: "center" });
      setFocusedTargetDomId(domId);
      window.setTimeout(() => {
        setFocusedTargetDomId((current) => (current === domId ? "" : current));
      }, 1400);
    }, 90);
  };

  const copySnippet = async (item: EvidenceItem) => {
    if (analysisUiDisabled) return;
    if (typeof navigator === "undefined" || !navigator.clipboard) return;
    try {
      await navigator.clipboard.writeText(`[${item.timestamp}] ${item.speaker}: ${item.quote}`);
    } catch {
      // noop
    }
  };

  const replayQueuedTotal = Number(state.replay?.queued_total || 0);
  const replayQueuedCursor = Number(state.replay?.queued_cursor || 0);
  const replayQueuedRemaining = Number(state.replay?.queued_remaining || 0);
  const replayDone = Boolean(state.replay?.done);
  const analysisWorker = state.analysis_runtime?.analysis_worker;
  const analysisQueuedCount = Number(analysisWorker?.queued || 0);
  const analysisInflight = Boolean(analysisWorker?.inflight);
  const llmIoLogs = Array.isArray(state.llm_io_logs) ? state.llm_io_logs : [];
  const llmReqCount = llmIoLogs.filter((x) => safeText(x?.direction).toLowerCase() === "request").length;
  const llmResCount = llmIoLogs.filter((x) => safeText(x?.direction).toLowerCase() === "response").length;
  const llmErrCount = llmIoLogs.filter((x) => safeText(x?.direction).toLowerCase() === "error").length;

  useEffect(() => {
    if (!isCanvasMode) {
      setCanvasComposerOpen(false);
      setCanvasNodeDetail(null);
    }
  }, [isCanvasMode]);

  useEffect(() => {
    const onPointerMove = (event: PointerEvent) => {
      const drag = resizeRef.current;
      if (!drag) return;
      const deltaX = event.clientX - drag.startX;
      if (drag.target === "sidebar") {
        setSidebarWidth(Math.max(220, Math.min(420, drag.startWidth + deltaX)));
        return;
      }
      if (drag.target === "canvas-left") {
        setCanvasLeftRailWidth(Math.max(280, Math.min(520, drag.startWidth + deltaX)));
        return;
      }
      setCanvasRightRailWidth(Math.max(300, Math.min(560, drag.startWidth - deltaX)));
    };

    const stopResize = () => {
      resizeRef.current = null;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };

    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", stopResize);
    window.addEventListener("pointercancel", stopResize);
    return () => {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", stopResize);
      window.removeEventListener("pointercancel", stopResize);
    };
  }, []);

  const startResize = useCallback((target: ResizeTarget, startWidth: number, event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    resizeRef.current = {
      target,
      startX: event.clientX,
      startWidth,
    };
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }, []);

  useEffect(() => {
    const current = {
      transcriptCount: state.transcript?.length || 0,
      outcomeCount: outcomeRows.length,
      activeAgenda: safeText(state.analysis?.agenda?.active?.title),
      decisionCount: decisions.length,
      actionCount: actionItems.length,
    };
    const prev = debugSnapshotRef.current;
    if (!prev) {
      debugSnapshotRef.current = current;
      setDebugEvents((rows) => [`${formatNowTime()} | 초기 상태 로드`, ...rows].slice(0, 80));
      return;
    }

    const lines: string[] = [];
    if (prev.transcriptCount !== current.transcriptCount) {
      lines.push(`전사 건수 ${prev.transcriptCount} -> ${current.transcriptCount}`);
    }
    if (prev.outcomeCount !== current.outcomeCount) {
      lines.push(`안건(outcomes) ${prev.outcomeCount} -> ${current.outcomeCount}`);
    }
    if (prev.activeAgenda !== current.activeAgenda) {
      lines.push(`활성 안건 '${prev.activeAgenda || "-"}' -> '${current.activeAgenda || "-"}'`);
    }
    if (prev.decisionCount !== current.decisionCount) {
      lines.push(`의사결정 ${prev.decisionCount} -> ${current.decisionCount}`);
    }
    if (prev.actionCount !== current.actionCount) {
      lines.push(`액션 ${prev.actionCount} -> ${current.actionCount}`);
    }

    if (lines.length > 0) {
      const stamp = formatNowTime();
      setDebugEvents((rows) => [`${stamp} | ${lines.join(" | ")}`, ...rows].slice(0, 80));
    }
    debugSnapshotRef.current = current;
  }, [state.transcript, state.analysis, outcomeRows.length, decisions.length, actionItems.length]);

  const debugSnapshot = useMemo(() => {
    const outcomes = state.analysis?.agenda_outcomes || [];
    return {
      transcript_count: state.transcript?.length || 0,
      agenda_stack_count: state.agenda_stack?.length || 0,
      agenda_outcomes_count: outcomes.length,
      active_agenda: safeText(state.analysis?.agenda?.active?.title),
      used_local_fallback: Boolean(state.analysis_runtime?.used_local_fallback),
      analysis_reason: safeText(state.analysis_runtime?.control_plane_reason),
      llm_json_available: Boolean(state.analysis_runtime?.last_llm_json_available),
      llm_json_at: safeText(state.analysis_runtime?.last_llm_json_at),
      llm_connected: Boolean(state.llm_status?.connected),
      llm_last_error: safeText(state.llm_status?.last_error),
      decision_count: decisions.length,
      action_count: actionItems.length,
      first_outcome: outcomes[0] || null,
      last_transcript: (state.transcript || []).slice(-1)[0] || null,
    };
  }, [state.transcript, state.agenda_stack, state.analysis, state.analysis_runtime, state.llm_status, decisions.length, actionItems.length]);

  const onDebugRefresh = async () => {
    await Promise.all([loadState(), refreshLlmStatus()]);
    setDebugEvents((rows) => [`${formatNowTime()} | 수동 새로고침 실행`, ...rows].slice(0, 80));
  };

  const onLoadLastLlmJson = async () => {
    setLlmJsonLoading(true);
    try {
      const res = await getLastLlmJson();
      setLastLlmJson(res.has_json ? (res.json || {}) : null);
      setLastLlmJsonAt(safeText(res.received_at));
      setDebugEvents((rows) => [
        `${formatNowTime()} | LLM 수신 JSON 조회: ${res.has_json ? "성공" : "데이터 없음"}`,
        ...rows,
      ].slice(0, 80));
      if (!res.has_json) {
        setError("아직 저장된 LLM 수신 JSON이 없습니다. 분석 실행 후 다시 확인하세요.");
      } else {
        setError("");
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLlmJsonLoading(false);
    }
  };

  const onExportAgendaMarkdown = async () => {
    beginTask("Markdown 내보내기 준비 중");
    try {
      const res = await exportAgendaMarkdown();
      const filename = safeText(res.filename, `agenda_export_${Date.now()}.md`);
      const blob = new Blob([safeText(res.markdown)], { type: "text/markdown;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      setError("");
      setDebugEvents((rows) => [
        `${formatNowTime()} | Markdown 내보내기 완료: ${filename} (agenda=${res.agenda_count}, turns=${res.transcript_count})`,
        ...rows,
      ].slice(0, 80));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      endTask();
    }
  };

  const onExportAgendaSnapshot = async () => {
    beginTask("안건 스냅샷 준비 중");
    try {
      const res = await exportAgendaSnapshot();
      const filename = safeText(res.filename, `agenda_snapshot_${Date.now()}.json`);
      const blob = new Blob([JSON.stringify(res.snapshot || {}, null, 2)], { type: "application/json;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      setError("");
      setDebugEvents((rows) => [
        `${formatNowTime()} | 안건 스냅샷 내보내기 완료: ${filename} (agenda=${res.agenda_count}, turns=${res.transcript_count})`,
        ...rows,
      ].slice(0, 80));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      endTask();
    }
  };

  const onImportAgendaSnapshot = async () => {
    if (!agendaSnapshotFile) return;
    beginTask("안건 스냅샷 불러오는 중");
    try {
      const res = await importAgendaSnapshot({ file: agendaSnapshotFile, reset_state: true });
      commitMeetingState(res.state);
      setMeetingGoalDraft(res.state.meeting_goal || "");
      setMeetingGoalDirty(false);
      setDatasetImportInfo(`snapshot imported=${res.import_debug.agenda_count} agendas / ${res.import_debug.transcript_count} turns`);
      setError("");
      setDebugEvents((rows) => [
        `${formatNowTime()} | 안건 스냅샷 불러오기 완료: ${res.import_debug.filename} (agenda=${res.import_debug.agenda_count}, turns=${res.import_debug.transcript_count})`,
        ...rows,
      ].slice(0, 80));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      endTask();
    }
  };

  const renderSummaryCard = () => (
    <details className="card panelCard panelFold" open={false}>
      <summary className="panelHeader panelFoldHeader">
        <h3>회의 요약 (안건별, 완료 안건)</h3>
        <span className="chip chipSoft">{bottomAgendas.length}개 안건 보기</span>
      </summary>
      <div className="panelFoldBody">
      {bottomAgendas.length === 0 ? (
        <p className="emptyState">완료된 안건이 생기면 여기에 정리됩니다.</p>
      ) : (
        <div className="accordionList">
          {bottomAgendas.map((agenda) => (
            <details key={agenda.id} open>
              <summary>
                <span>{agendaLabel(agenda)}</span>
                <span className={agendaStatusClass[agenda.status]}>{agendaStatusLabel[agenda.status]}</span>
              </summary>
              {agenda.summaryBullets.length === 0 ? (
                <p className="emptyState compact">이 안건 논의가 시작되면 요약이 보여요.</p>
              ) : (
                <ul className="bulletList">
                  {agenda.summaryBullets.map((point, pointIdx) => {
                    const pointId = agenda.summaryPointIds?.[pointIdx] || `summary-${agenda.id}-${pointIdx}`;
                    const meta = summaryPointMetaMap.get(`${agenda.id}|${pointId}`);
                    const rangeLabel = meta?.rangeLabel || buildTimeRangeLabel([point]);
                    const clickable = Boolean(meta || extractTimestampToken(point));
                    return (
                      <li key={`${point}-${pointIdx}`}>
                        <button
                          className="ghostButton"
                          type="button"
                          onClick={() => jumpBySummary(agenda.id, point, pointId)}
                          disabled={analysisUiDisabled || !clickable}
                        >
                          <span>{stripLeadingTimestamp(point)}</span>
                          {rangeLabel !== "-" ? <span className="summaryPointRange">{rangeLabel}</span> : null}
                        </button>
                        {meta && meta.opinionGroups.length > 0 ? (
                          <div className="summaryOpinionBlock">
                            <p className="mutedLabel">의견 요약</p>
                            <ul className="summaryOpinionList">
                              {meta.opinionGroups.map((group) => (
                                <li key={group.id}>
                                  <button
                                    className="opinionSummaryButton"
                                    type="button"
                                    onClick={() => focusByOpinionGroup(agenda.id, pointId, group.id)}
                                    disabled={analysisUiDisabled}
                                  >
                                    <span className={`chip chipSoft opinionTypeChip opinionType-${group.type}`}>{group.typeLabel}</span>
                                    <span>{group.summary}</span>
                                    {group.detail ? <span className="opinionDetail">{group.detail}</span> : null}
                                    {group.rangeLabel !== "-" ? <span className="summaryPointRange">{group.rangeLabel}</span> : null}
                                  </button>
                                </li>
                              ))}
                            </ul>
                          </div>
                        ) : null}
                      </li>
                    );
                  })}
                </ul>
              )}
              <div className="callout">
                <p className="calloutLabel">권장 사항</p>
                <p>{agenda.recommendation}</p>
              </div>
            </details>
          ))}
        </div>
      )}
      </div>
    </details>
  );

  const renderDecisionCard = () => (
    <details className="card panelCard panelFold" open={false}>
      <summary className="panelHeader panelFoldHeader">
        <h3>의사결정 결과</h3>
        <span className="chip chipSoft">{bottomDecisions.length}건 기록됨</span>
      </summary>
      <div className="panelFoldBody">
      {bottomDecisions.length === 0 ? (
        <p className="emptyState">이 안건에는 아직 기록된 의사결정이 없어요.</p>
      ) : (
        <div className="decisionGroups">
          {bottomAgendas.map((agenda) => {
            const scopedDecisions = bottomDecisions.filter((decision) => decision.agendaId === agenda.id);
            if (scopedDecisions.length === 0) return null;
            return (
              <section key={agenda.id} className="decisionGroup">
                <h4>{agendaLabel(agenda)}</h4>
                {scopedDecisions.map((decision) => (
                  <article
                    key={decision.id}
                    id={`evi-target-${decision.id}`}
                    className={`decisionItem ${focusedTargetDomId === `evi-target-${decision.id}` ? "focusFlash" : ""}`}
                  >
                    <div className="decisionRow">
                      <p className="decisionIssue">{decision.issue}</p>
                      <span className={decisionStatusClass(decision.finalStatus)}>{decisionStatusLabel[decision.finalStatus]}</span>
                    </div>
                    <p className="mutedLabel">옵션 / 의견</p>
                    <ul className="bulletList">
                      {decision.options.map((option) => (
                        <li key={option}>{option}</li>
                      ))}
                    </ul>
                    <div className="inlineMeta">
                      <span>신뢰도 {decision.confidence}%</span>
                      <div className="chipRow">
                        {decision.evidence.map((timestamp) => (
                          <button
                            key={timestamp}
                            className="chip chipInteractive"
                            type="button"
                            onClick={() => jumpToTranscript(decision.agendaId, timestamp)}
                            disabled={analysisUiDisabled}
                          >
                            근거 {timestamp}
                          </button>
                        ))}
                      </div>
                    </div>
                  </article>
                ))}
              </section>
            );
          })}
        </div>
      )}
      </div>
    </details>
  );

  const renderActionCard = () => (
    <details className="card panelCard panelFold" open={false}>
      <summary className="panelHeader panelFoldHeader">
        <h3>액션 아이템</h3>
        <span className="chip chipSoft">{bottomActions.length}건</span>
      </summary>
      <div className="panelFoldBody">
      {bottomActions.length === 0 ? (
        <p className="emptyState">이 안건에 연결된 액션 아이템이 아직 없어요.</p>
      ) : (
        <div className="tableWrap">
          <table>
            <thead>
              <tr>
                <th>액션</th>
                <th>담당자</th>
                <th>기한</th>
                <th>상태</th>
                <th>근거</th>
              </tr>
            </thead>
            <tbody>
              {bottomActions.map((item) => (
                <tr id={`evi-target-${item.id}`} className={focusedTargetDomId === `evi-target-${item.id}` ? "focusFlash" : ""} key={item.id}>
                  <td>{item.action}</td>
                  <td>{item.owner}</td>
                  <td>{item.due}</td>
                  <td><span className={actionStatusClass[item.status]}>{actionStatusLabel[item.status]}</span></td>
                  <td>
                    <div className="chipRow">
                      {item.evidence.map((timestamp) => (
                        <button
                          key={timestamp}
                          className="chip chipInteractive"
                          type="button"
                          onClick={() => jumpToTranscript(item.agendaId, timestamp)}
                          disabled={analysisUiDisabled}
                        >
                          {timestamp}
                        </button>
                      ))}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      </div>
    </details>
  );

  const renderEvidenceCard = () => (
    <details className="card panelCard panelFold" open={false}>
      <summary className="panelHeader panelFoldHeader">
        <h3>근거 로그</h3>
        <span className="chip chipSoft">{bottomEvidence.length}개 스니펫</span>
      </summary>
      <div className="panelFoldBody">
      {bottomEvidence.length === 0 ? (
        <p className="emptyState">연결된 액션 또는 의사결정이 생기면 근거 스니펫이 표시됩니다.</p>
      ) : (
        <div className="decisionGroups">
          {groupedBottomEvidence.map((group) => (
            <section key={`ev-group-${group.agendaId}`} className="decisionGroup">
              <h4>{group.agendaTitle}</h4>
              <div className="evidenceList">
                {group.items.map((item) => (
                  <article key={item.id} className="evidenceItem">
                    <div className="evidenceMeta">
                      <span className="chip chipSoft">{evidenceSupportLabel[item.supports]}</span>
                      <span className="timestamp">{item.timestamp}</span>
                      <span className="chip chipSpeaker">{item.speaker}</span>
                    </div>
                    <button
                      className="targetLink"
                      type="button"
                      onClick={() => focusTargetCard(item.agendaId, item.targetId)}
                      disabled={analysisUiDisabled}
                    >
                      대상: {item.targetLabel || item.targetId}
                    </button>
                    <p className="quote">&quot;{item.quote}&quot;</p>
                    <div className="evidenceActions">
                      <button className="ghostButton" type="button" onClick={() => jumpToTranscript(item.agendaId, item.timestamp)} disabled={analysisUiDisabled}>
                        전사문으로 이동
                      </button>
                      <button className="ghostButton" type="button" onClick={() => copySnippet(item)} disabled={analysisUiDisabled}>
                        복사
                      </button>
                    </div>
                  </article>
                ))}
              </div>
            </section>
          ))}
        </div>
      )}
      </div>
    </details>
  );

  const layoutStyle = {
    ["--workspace-sidebar-width" as string]: `${sidebarOpen ? sidebarWidth : 0}px`,
    ["--canvas-left-drawer-width" as string]: `${canvasLeftRailWidth}px`,
    ["--canvas-right-drawer-width" as string]: `${canvasRightRailWidth}px`,
  } as CSSProperties;

  return (
    <div className={`workspaceShell ${isCanvasMode ? "workspaceShellCanvasMode" : ""}`} style={layoutStyle}>
      <aside className={`sidebar ${sidebarOpen ? "sidebarOpen" : "sidebarClosed"}`} style={{ width: `${sidebarWidth}px` }}>
        <div className="sidebarInner">
          <div>
            <p className="brand">파르체</p>
            <p className="brandSub">회의 인텔리전스</p>
          </div>
          <nav className="sidebarNav">
            <button className="navItem" type="button">대시보드</button>
            <button
              className={`navItem ${activeSection === "workspace" ? "navItemActive" : ""}`}
              type="button"
              onClick={() => {
                setCanvasReturnContext("");
                setActiveSection("workspace");
              }}
            >
              회의 워크스페이스
            </button>
            <button
              className={`navItem ${activeSection === "canvas" ? "navItemActive" : ""}`}
              type="button"
              onClick={() => setActiveSection("canvas")}
            >
              공용 캔버스
            </button>
            <button className="navItem" type="button">리포트</button>
            <button className="navItem" type="button">팀 노트</button>
          </nav>

        </div>
        <div
          className="sidebarResizeHandle"
          onPointerDown={(event) => startResize("sidebar", sidebarWidth, event)}
          role="separator"
          aria-orientation="vertical"
          aria-label="왼쪽 패널 크기 조절"
        />
      </aside>
      <button
        type="button"
        className={`edgeNudge edgeNudgeSidebar ${sidebarOpen ? "edgeNudgeOpen" : "edgeNudgeClosed"}`}
        onClick={() => setSidebarOpen((open) => !open)}
        aria-label={sidebarOpen ? "왼쪽 메뉴 접기" : "왼쪽 메뉴 펼치기"}
        style={{ left: sidebarOpen ? `${Math.max(10, sidebarWidth - 14)}px` : "8px" }}
      >
        {sidebarOpen ? "‹" : "›"}
      </button>

      <main className={`mainArea ${isCanvasMode ? "mainAreaCanvasMode" : ""}`}>
        <div className={`mainInner ${isCanvasMode ? "mainInnerCanvasMode" : ""}`}>
          <section className={`leftSection ${lineUploadMode ? "leftSectionLineMode" : ""} ${isCanvasMode ? "leftSectionCanvasMode" : ""}`}>
          {!isCanvasMode ? (
          <>
            <header className="pageHeader awsHeader glassStickyHeader">
              <div className="headerMain">
                <div>
                  <h1>{meeting.title}</h1>
                  <div className="metaRow">
                    <span>{meeting.date}</span>
                    <span>{meeting.duration}</span>
                    <span>{meeting.participants}</span>
                  </div>
                </div>
                <div className="headerActions" aria-label="회의 메트릭">
                  <div className="sidebarMetricList">
                    <div className="sidebarMetricRow">
                      <span>커버리지</span>
                      <strong>{agendaOverview.done}/{agendas.length}</strong>
                    </div>
                    <div className="sidebarMetricRow">
                      <span>대상</span>
                      <strong>{selectedContext.transcriptCount}</strong>
                    </div>
                    <div className="sidebarMetricRow">
                      <span>액션</span>
                      <strong>{selectedContext.openActionCount}</strong>
                    </div>
                    <div className="sidebarMetricRow">
                      <span>자신감</span>
                      <strong>{agendaOverview.averageConfidence}%</strong>
                    </div>
                  </div>
                </div>
              </div>
              <div className="contextBar">
                <span className="chip chipInteractive">{selectedAgenda ? agendaLabel(selectedAgenda) : "선택된 안건 없음"}</span>
                <span>{selectedAgenda ? `${selectedAgenda.progress}% 완료` : "0% 완료"} . {meeting.elapsed}</span>
                <span className="mutedLabel">마지막 업데이트 {meeting.lastUpdated}</span>
                <span className="chip chipSoft">LLM: {llmEnabled ? "ON" : "OFF"}</span>
                {canvasReturnContext ? (
                  <button type="button" className="chip chipInteractive" onClick={returnToCanvas}>
                    캔버스로 돌아가기 · {canvasReturnContext}
                  </button>
                ) : null}
              </div>
            </header>

            <nav className="awsTabs awsTabsSeparate" aria-label="회의 워크스페이스 탭">
              <button className="awsTab awsTabActive" type="button">개요</button>
              <button className="awsTab" type="button">전사문 검토</button>
              <button className="awsTab" type="button">안건 인사이트</button>
              <button className="awsTab" type="button">결과</button>
            </nav>
          </>
          ) : (
            <header className="canvasPageHeader">
              <div className="canvasPageHeaderLeft">
                <span className="canvasPageBadge">Canvas</span>
                <h1>공용 캔버스</h1>
              </div>
              <div className="canvasCanvasToolbar">
                <span className="canvasToolbarMeta">{meeting.title}</span>
                <button
                  type="button"
                  className={`canvasToolbarButton ${canvasComposerOpen ? "canvasToolbarButtonActive" : ""}`}
                  onClick={() => setCanvasComposerOpen((open) => !open)}
                >
                  노트 추가
                </button>
              </div>
            </header>
          )}

          <article className={`card panelCard controlPanelCard ${isCanvasMode ? `canvasDrawer canvasDrawerLeft ${canvasLeftRailOpen ? "canvasDrawerOpen" : "canvasDrawerClosed"}` : ""}`}>
            <div className="panelHeader tight">
              <h3>실행 제어</h3>
              <div className="panelHeaderActionsTight">
                <span className="chip chipSoft">LLM {state.llm_status?.connected ? "연결됨" : "미연결"}</span>
              </div>
            </div>
            <div className="transcriptControls">
              <input
                aria-label="회의 목표"
                placeholder="회의 목표"
                value={meetingGoalDraft}
                onChange={(event) => {
                  setMeetingGoalDraft(event.target.value);
                  setMeetingGoalDirty(true);
                }}
              />
              <input
                aria-label="JSON 폴더"
                placeholder="dataset/economy"
                value={datasetFolder}
                onChange={(event) => setDatasetFolder(event.target.value)}
              />
              <input
                aria-label="JSON 파일 업로드"
                type="file"
                accept=".json,application/json"
                multiple
                onChange={(event) => setDatasetFiles(Array.from(event.target.files || []))}
              />
              <input
                aria-label="안건 스냅샷 업로드"
                type="file"
                accept=".json,application/json"
                onChange={(event) => setAgendaSnapshotFile(event.target.files?.[0] || null)}
              />
              <select
                aria-label="전사 윈도우"
                value={state.window_size}
                onChange={(event) => setState((s) => ({ ...s, window_size: Number(event.target.value) || 12 }))}
              >
                {[8, 12, 20, 30, 40, 60].map((n) => (
                  <option key={n} value={n}>{n} turns</option>
                ))}
              </select>
              <label className="toggleLabel">
                <input
                  checked={lineUploadMode}
                  type="checkbox"
                  onChange={(event) => setLineUploadMode(event.target.checked)}
                />
                JSON 라인 주입 모드
              </label>
            </div>
            <div className="panelActions">
              <button type="button" onClick={() => void onSaveConfig()} disabled={loading}>설정 저장</button>
              <button type="button" onClick={() => void onImportDataset()} disabled={loading}>JSON 폴더 로드</button>
              <button type="button" onClick={() => void onImportDatasetFiles()} disabled={loading || datasetFiles.length === 0}>JSON 업로드</button>
              <button type="button" onClick={() => void onImportAgendaSnapshot()} disabled={loading || !agendaSnapshotFile}>안건 가져오기</button>
              {lineUploadMode ? (
                <button type="button" onClick={() => void onQueueDatasetFilesLineMode()} disabled={loading || datasetFiles.length === 0}>
                  라인모드 큐 적재
                </button>
              ) : null}
              <button type="button" onClick={() => void apply(() => tickAnalysis(), "전체 분석 실행 중", true)} disabled={loading || analysisUiDisabled}>분석 실행</button>
              <button type="button" onClick={() => void onExportAgendaMarkdown()} disabled={loading}>안건 MD 내보내기</button>
              <button type="button" onClick={() => void onExportAgendaSnapshot()} disabled={loading}>안건 스냅샷 내보내기</button>
              <button type="button" onClick={() => { stopReplayAuto(); void apply(() => resetState(), "상태 초기화 중"); }} disabled={loading}>초기화</button>
              <button type="button" onClick={() => void onConnectLlm()} disabled={llmChecking}>{llmChecking ? "연결 중" : "LLM 연결"}</button>
              <button type="button" onClick={() => void onDisconnectLlm()} disabled={llmChecking || !llmEnabled}>연결 해제</button>
              <button type="button" onClick={() => void onPingLlm()} disabled={llmChecking}>{llmChecking ? "확인 중" : "연결 테스트"}</button>
            </div>
            {lineUploadMode ? (
              <>
                <div className="transcriptControls transcriptControlsCompact">
                  <label className="toggleLabel">
                    step lines
                    <input
                      aria-label="라인 주입 step 크기"
                      type="number"
                      min={1}
                      max={100}
                      value={replayLinesPerStep}
                      onChange={(event) => setReplayLinesPerStep(Math.max(1, Math.min(100, Number(event.target.value) || 1)))}
                    />
                  </label>
                  <label className="toggleLabel">
                    interval ms
                    <input
                      aria-label="라인 주입 간격(ms)"
                      type="number"
                      min={100}
                      max={10000}
                      step={100}
                      value={replayIntervalMs}
                      onChange={(event) => setReplayIntervalMs(Math.max(100, Math.min(10000, Number(event.target.value) || 1200)))}
                    />
                  </label>
                </div>
                <div className="panelActions">
                  <button
                    type="button"
                    onClick={() => void runReplayStepOnce()}
                    disabled={loading || replayRunning || replayQueuedRemaining <= 0}
                  >
                    {replayLinesPerStep}줄 주입
                  </button>
                  {!replayRunning ? (
                    <button
                      type="button"
                      onClick={() => onStartReplayAuto()}
                      disabled={loading || replayQueuedRemaining <= 0}
                    >
                      자동 재생 시작
                    </button>
                  ) : (
                    <button type="button" onClick={() => stopReplayAuto("line-mode 자동 재생 중지")} disabled={loading}>
                      자동 재생 중지
                    </button>
                  )}
                </div>
              </>
            ) : null}
            {loading ? (
              <p className="runIndicator">
                <span className="runDot" />
                <span>{activeTask || "작업 실행 중"} ({taskElapsedSec}s)</span>
              </p>
            ) : null}
            <details className="panelFold">
              <summary className="panelHeader tight panelFoldHeader">
                <h3>디버그 모드</h3>
                <span className="chip chipSoft">{analysisInflight ? "RUNNING" : "IDLE"}</span>
              </summary>
              <div className="panelFoldBody">
                {lineUploadMode ? (
                  <p className="mutedLabel">
                    line-mode queue: {replayQueuedCursor}/{replayQueuedTotal} (remaining {replayQueuedRemaining}) {replayDone ? "| done" : ""}
                  </p>
                ) : null}
                <p className="mutedLabel">전사 건수: {state.transcript?.length || 0}</p>
                {analysisUiDisabled ? <p className="mutedLabel">분석 비활성화: 결과 수신 대기 중</p> : null}
                {datasetImportInfo ? <p className="mutedLabel">{datasetImportInfo}</p> : null}
                {llmPingMessage ? <p className="mutedLabel">Ping: {llmPingMessage}</p> : null}
                {llmPingOk === false ? <p className="mutedLabel">LLM 연결 오류를 확인하세요.</p> : null}
                {state.llm_status?.last_error ? <p className="mutedLabel">LLM 오류: {state.llm_status.last_error}</p> : null}
                {state.llm_status?.last_finish_reason ? <p className="mutedLabel">LLM finish_reason: {state.llm_status.last_finish_reason}</p> : null}
                {state.llm_status?.last_raw_preview ? (
                  <details>
                    <summary>LLM 원문 미리보기</summary>
                    <pre className="emptyState compact">{String(state.llm_status.last_raw_preview)}</pre>
                  </details>
                ) : null}
                {state.analysis_runtime?.control_plane_reason ? <p className="mutedLabel">분석 상태: {state.analysis_runtime.control_plane_reason}</p> : null}
                <p className="mutedLabel">
                  분석 워커: {analysisInflight ? "처리중" : "대기"} . 큐 {analysisQueuedCount}
                  {analysisWorker?.queued_observed !== undefined ? ` (obs ${Number(analysisWorker.queued_observed)})` : ""}
                  {analysisWorker?.last_done_id ? ` . 마지막 완료 #${analysisWorker.last_done_id}` : ""}
                </p>
                {analysisWorker?.last_error ? <p className="mutedLabel">분석 워커 오류: {analysisWorker.last_error}</p> : null}
                <p className="mutedLabel">
                  안건 제목 재요청: {Number(state.analysis_runtime?.title_refine_success ?? 0)}/{Number(state.analysis_runtime?.title_refine_attempts ?? 0)}
                </p>
                <p className="mutedLabel">
                  수신 JSON: {state.analysis_runtime?.last_llm_json_available ? "있음" : "없음"} {state.analysis_runtime?.last_llm_json_at ? `(${state.analysis_runtime.last_llm_json_at})` : ""}
                </p>
                {state.analysis_runtime?.used_local_fallback ? <p className="mutedLabel">현재 로컬 폴백 분석 모드</p> : null}
                {error ? <p className="emptyState compact">{error}</p> : null}

                <details>
                  <summary>디버그 패널</summary>
                  <div className="transcriptMetaBar">
                    <span className="chip chipSoft">agenda_outcomes: {outcomeRows.length}</span>
                    <span className="chip chipSoft">active: {state.analysis?.agenda?.active?.title || "-"}</span>
                    <span className="chip chipSoft">decisions: {decisions.length}</span>
                    <span className="chip chipSoft">actions: {actionItems.length}</span>
                  </div>
                  <div className="panelActions">
                    <button type="button" onClick={() => void onDebugRefresh()} disabled={loading || llmChecking}>상태 강제 새로고침</button>
                    <button type="button" onClick={() => void onLoadLastLlmJson()} disabled={loading || llmChecking || llmJsonLoading}>
                      {llmJsonLoading ? "조회 중" : "LLM 수신 JSON 보기"}
                    </button>
                    <button type="button" onClick={() => setDebugEvents([])}>디버그 로그 지우기</button>
                  </div>
                  <div className="signalTimeline">
                    {(debugEvents.length ? debugEvents.slice(0, 10) : ["변화 로그 없음"]).map((line, idx) => (
                      <div key={`debug-log-${idx}`}>
                        <p>{line}</p>
                      </div>
                    ))}
                  </div>
                  <details>
                    <summary>Raw State 요약(JSON)</summary>
                    <pre className="emptyState compact">{JSON.stringify(debugSnapshot, null, 2)}</pre>
                  </details>
                  <details open={Boolean(lastLlmJson)}>
                    <summary>LLM 수신 JSON</summary>
                    {lastLlmJson ? (
                      <>
                        <p className="mutedLabel">수신 시각: {lastLlmJsonAt || "-"}</p>
                        <pre className="emptyState compact">{JSON.stringify(lastLlmJson, null, 2)}</pre>
                      </>
                    ) : (
                      <p className="emptyState compact">버튼을 눌러 최근 수신 JSON을 조회하세요.</p>
                    )}
                  </details>
                </details>
              </div>
            </details>
            {isCanvasMode && canvasLeftRailOpen ? (
              <div
                className="canvasDrawerResizeHandle canvasDrawerResizeHandleLeft"
                onPointerDown={(event) => startResize("canvas-left", canvasLeftRailWidth, event)}
                role="separator"
                aria-orientation="vertical"
                aria-label="왼쪽 패널 크기 조절"
              />
            ) : null}
          </article>
          
          <article className={`card panelCard transcriptCard transcriptCardLeft ${isCanvasMode ? "canvasSurfaceCard" : ""}`}>
            {!isCanvasMode ? (
            <div className={`panelHeader ${isCanvasMode ? "panelHeaderCanvas" : ""}`}>
              <div className="panelHeaderMeta">
                <h2>{isCanvasMode ? "공용 캔버스" : "전사문 (전체)"}</h2>
                <p className="mutedLabel">
                  {isCanvasMode
                    ? "React Flow 기반의 노드 캔버스에서 안건, 요약, 메모를 연결해 구조적으로 정리합니다."
                    : "실시간 STT 전사를 필터링하고, 요약 근거와 연결된 원문을 추적합니다."}
                </p>
              </div>
              <div className="panelHeaderActions panelHeaderActionsTight">
                <span className="chip chipSoft">
                  {isCanvasMode ? `안건 ${canvasLanes.length}개 . 아이디어 ${canvasIdeas.length}개` : `${filteredTranscript.length}개 표시`}
                </span>
              </div>
            </div>
            ) : null}
            {!isCanvasMode ? (
              <>
                {selectedSummaryFocus ? (
                  <div className="summaryFocusBar">
                    <span className="chip chipInteractive">요약 포커스</span>
                    <p>{stripLeadingTimestamp(selectedSummaryFocus.pointText)}</p>
                    <span className="mutedLabel">
                      범위 {selectedSummaryFocus.rangeLabel} . 원문 {selectedSummaryFocus.utterances.length}문장
                    </span>
                    <button
                      className="ghostButton"
                      type="button"
                      onClick={() => setSelectedSummaryFocus(null)}
                      disabled={analysisUiDisabled}
                    >
                      포커스 해제
                    </button>
                  </div>
                ) : null}
                <div className="transcriptControls transcriptControlsCompact">
                  <input
                    aria-label="전사문 검색"
                    placeholder="전사문 검색"
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                  />
                  <select aria-label="화자 필터" value={speakerFilter} onChange={(event) => setSpeakerFilter(event.target.value)}>
                    {speakerOptions.map((speaker) => (
                      <option key={speaker} value={speaker}>{speaker}</option>
                    ))}
                  </select>
                  <label className="toggleLabel">
                    <input checked={highlightRelated} type="checkbox" onChange={(event) => setHighlightRelated(event.target.checked)} />
                    관련 발화 강조
                  </label>
                </div>
                <div className="transcriptMetaBar">
                  <span className="chip chipSoft">문맥 발화: {selectedContext.transcriptCount}</span>
                  <span className="chip chipSoft">연결된 의사결정: {selectedContext.decisionCount}</span>
                  <span className="chip chipSoft">연결된 액션: {selectedContext.actionCount}</span>
                </div>
                <div ref={transcriptListRef} className="transcriptList" onScroll={onTranscriptScroll}>
                  {filteredTranscript.length === 0 ? (
                    <p className="emptyState">
                      {selectedSummaryFocus
                        ? "선택한 요약/의견의 원문 발화를 찾지 못했습니다. 포커스를 해제하거나 다른 항목을 선택해 주세요."
                        : "현재 필터와 일치하는 발화가 없습니다. 검색어나 화자 필터를 조정해 주세요."}
                    </p>
                  ) : (
                    filteredTranscript.map((utterance) => {
                      const isRelated = selectedAgenda ? utterance.agendaId === selectedAgenda.id : false;
                      const shouldDim = highlightRelated && !isRelated;
                      const shouldHighlight = highlightRelated && isRelated;
                      return (
                        <article
                          key={utterance.id}
                          className={`utterance ${shouldHighlight ? "utteranceHighlight" : ""} ${shouldDim ? "utteranceDim" : ""}`}
                        >
                          <div className="utteranceMeta">
                            <span className="timestamp">{utterance.timestamp}</span>
                            <span className="chip chipSpeaker">{utterance.speaker}</span>
                          </div>
                          <p>{utterance.text}</p>
                          <div className="utteranceActions">
                            <button type="button">+ 액션</button>
                            <button type="button">+ 의사결정</button>
                            <button type="button">+ 근거</button>
                          </div>
                        </article>
                      );
                    })
                  )}
                </div>
                {!selectedSummaryFocus && pendingTranscriptCount > 0 ? (
                  <button
                    type="button"
                    className="transcriptJumpButton"
                    onClick={() => scrollTranscriptToBottom("smooth")}
                  >
                    새 전사 {pendingTranscriptCount}개 . 아래로
                  </button>
                ) : null}
              </>
            ) : (
              <div className="canvasWorkspace canvasWorkspaceImmersive">
                <div className="canvasBoard canvasBoardImmersive">
                  {flowNodes.length === 0 ? (
                    <p className="emptyState">안건이 아직 없어 공용 캔버스를 구성할 수 없습니다.</p>
                  ) : (
                    <div className="reactFlowCanvas">
                      <ReactFlow
                        className="reactFlowStage"
                        nodes={flowNodes}
                        edges={flowEdges}
                        onNodesChange={onFlowNodesChange}
                        onEdgesChange={onFlowEdgesChange}
                        onConnect={onFlowConnect}
                        onNodeClick={onFlowNodeClick}
                        onPaneClick={() => setCanvasNodeDetail(null)}
                        onNodeDragStop={onFlowNodeDragStop}
                        defaultEdgeOptions={{ reconnectable: true, type: "step" }}
                        fitView
                        fitViewOptions={{ padding: 0.18, duration: 480 }}
                        minZoom={0.35}
                        maxZoom={1.8}
                        zoomOnDoubleClick={false}
                        nodesConnectable
                        elementsSelectable
                        selectNodesOnDrag={false}
                        selectionOnDrag
                        panOnDrag={[1, 2]}
                        panOnScroll
                        zoomOnPinch
                        zoomOnScroll
                        snapToGrid
                        snapGrid={[16, 16]}
                        deleteKeyCode={["Backspace", "Delete"]}
                        elevateEdgesOnSelect
                        proOptions={{ hideAttribution: true }}
                      >
                        <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#e5e7eb" />
                        <MiniMap
                          zoomable
                          pannable
                          nodeColor={(node) => {
                            if (node.type === "agenda") return "#1f2937";
                            if (node.type === "summary") return "#64748b";
                            return "#d97706";
                          }}
                        />
                        <Controls />
                        <Panel position="top-left" className="rfExamplePanel">
                          <div className="rfExamplePanelCard">
                            <strong>Shared Canvas</strong>
                            <p>{selectedAgenda ? agendaLabel(selectedAgenda) : "선택된 안건 없음"}</p>
                            <span>안건 {canvasLanes.length} · 메모 {canvasIdeas.length}</span>
                          </div>
                        </Panel>

                        {canvasComposerOpen ? (
                          <Panel position="top-right" className="rfPanelShell rfComposerPanel">
                            <section className="canvasQuickComposer">
                              <div className="canvasComposerHeader">
                                <div>
                                  <p className="canvasEyebrow">Note</p>
                                  <h3>캔버스 메모 추가</h3>
                                  <p className="mutedLabel">
                                    {selectedAgenda ? `${agendaLabel(selectedAgenda)}에 바로 배치됩니다.` : "먼저 안건을 선택한 뒤 메모를 추가하세요."}
                                    {selectedSummaryFocus ? ` 현재 요약 "${stripLeadingTimestamp(selectedSummaryFocus.pointText)}"와 연결됩니다.` : ""}
                                  </p>
                                </div>
                              </div>
                              <div className="canvasComposerGrid">
                                <input
                                  aria-label="아이디어 제목"
                                  placeholder="짧은 제목"
                                  value={canvasIdeaTitle}
                                  onChange={(event) => setCanvasIdeaTitle(event.target.value)}
                                />
                                <textarea
                                  aria-label="아이디어 메모"
                                  placeholder="노트 내용, 질문, 다음 실험 아이디어를 입력"
                                  value={canvasIdeaBody}
                                  onChange={(event) => setCanvasIdeaBody(event.target.value)}
                                />
                              </div>
                              <div className="panelActions">
                                <button
                                  type="button"
                                  onClick={addCanvasIdea}
                                  disabled={analysisUiDisabled || (!safeText(canvasIdeaTitle) && !safeText(canvasIdeaBody))}
                                >
                                  보드에 놓기
                                </button>
                                <button
                                  type="button"
                                  className="ghostButton"
                                  onClick={() => {
                                    setCanvasIdeaTitle("");
                                    setCanvasIdeaBody("");
                                  }}
                                  disabled={analysisUiDisabled}
                                >
                                  입력 지우기
                                </button>
                              </div>
                            </section>
                          </Panel>
                        ) : null}
                      </ReactFlow>
                      {canvasNodeDetail ? (
                        <aside className="canvasNodeDetailPanel">
                          <div className="canvasNodeDetailCard">
                            <div className="canvasNodeDetailHeader">
                              <div>
                                <p className="canvasEyebrow">Detail</p>
                                <h3>{canvasNodeDetail.title}</h3>
                                <p className="mutedLabel">{canvasNodeDetail.subtitle}</p>
                              </div>
                              <button
                                type="button"
                                className="ghostButton"
                                onClick={() => setCanvasNodeDetail(null)}
                              >
                                닫기
                              </button>
                            </div>
                            <div className="canvasNodeDetailMeta">
                              {canvasNodeDetail.badges.map((badge) => (
                                <span key={`${canvasNodeDetail.id}-${badge}`} className="chip chipSoft">{badge}</span>
                              ))}
                            </div>

                            <section className="canvasNodeDetailSection">
                              <h4>요약본</h4>
                              <div className="canvasNodeDetailSummaryList">
                                {canvasNodeDetail.summaryLines.map((line, idx) => (
                                  <p key={`${canvasNodeDetail.id}-summary-${idx}`}>{line}</p>
                                ))}
                              </div>
                            </section>

                            {canvasNodeDetail.opinionGroups.length > 0 ? (
                              <section className="canvasNodeDetailSection">
                                <h4>의견 요약</h4>
                                <div className="canvasNodeDetailOpinionList">
                                  {canvasNodeDetail.opinionGroups.map((group) => (
                                    <button
                                      key={group.id}
                                      type="button"
                                      className="opinionGroupCard"
                                      onClick={() => {
                                        if (!canvasNodeDetail.pointId) return;
                                        focusCanvasOpinionGroup(canvasNodeDetail.agendaId, canvasNodeDetail.pointId, group.id);
                                      }}
                                    >
                                      <div className="opinionGroupHeader">
                                        <strong>{group.typeLabel}</strong>
                                        <span>{group.rangeLabel}</span>
                                      </div>
                                      <p>{group.summary}</p>
                                    </button>
                                  ))}
                                </div>
                              </section>
                            ) : null}

                            <section className="canvasNodeDetailSection canvasNodeDetailSectionFill">
                              <div className="canvasNodeDetailSectionHeader">
                                <h4>원본 발언</h4>
                                <button
                                  type="button"
                                  className="ghostButton"
                                  onClick={() => moveToWorkspaceFromCanvas("캔버스 상세 원문 확인 중")}
                                >
                                  워크스페이스에서 보기
                                </button>
                              </div>
                              <div className="canvasNodeDetailTranscript">
                                {canvasNodeDetail.utterances.length === 0 ? (
                                  <p className="emptyState compact">연결된 원문 발언이 없습니다.</p>
                                ) : (
                                  canvasNodeDetail.utterances.map((utterance) => (
                                    <article key={`${canvasNodeDetail.id}-${utterance.id}`} className="canvasDetailUtterance">
                                      <div className="utteranceMeta">
                                        <span className="timestamp">{utterance.timestamp}</span>
                                        <span className="chip chipSpeaker">{utterance.speaker}</span>
                                      </div>
                                      <p>{utterance.text}</p>
                                    </article>
                                  ))
                                )}
                              </div>
                            </section>
                          </div>
                        </aside>
                      ) : null}
                    </div>
                  )}
                </div>
              </div>
            )}
          </article>
          </section>

          <section className={`rightSection ${isCanvasMode ? `canvasDrawer canvasDrawerRight ${canvasRightRailOpen ? "canvasDrawerOpen" : "canvasDrawerClosed"}` : ""}`}>
          {isCanvasMode ? (
            <div className="canvasDrawerHeader">
              <strong>인사이트</strong>
            </div>
          ) : null}
          {isCanvasMode && canvasRightRailOpen ? (
            <div
              className="canvasDrawerResizeHandle canvasDrawerResizeHandleRight"
              onPointerDown={(event) => startResize("canvas-right", canvasRightRailWidth, event)}
              role="separator"
              aria-orientation="vertical"
              aria-label="오른쪽 패널 크기 조절"
            />
          ) : null}
          <section className="contentSignalGrid">
            <details className="card panelCard sidebarSection panelFold" open={false}>
              <summary className="panelHeader tight panelFoldHeader">
                <h3>실시간 참여자</h3>
                <span className="chip chipSoft">{participantRoster.length}명 참여 중</span>
              </summary>
              <div className="panelFoldBody">
              <div className="participantList">
                {participantRoster.map((member) => (
                  <div key={member.name} className="participantItem">
                    <div className="participantAvatar">{member.name.slice(0, 2)}</div>
                    <div>
                      <p className="participantName">{member.name}</p>
                      <p className="participantRole">{member.role}</p>
                    </div>
                    <span className={participantStatusClass(member.status)}>{participantStatusLabel[member.status]}</span>
                  </div>
                ))}
              </div>
              </div>
            </details>

            <details className="card panelCard sidebarSection panelFold" open={false}>
              <summary className="panelHeader tight panelFoldHeader">
                <h3>STT 스트림</h3>
                <span className="chip chipSoft">{sttStatusText}</span>
              </summary>
              <div className="panelFoldBody">
              <div className="panelActions">
                <select value={sttSource} disabled>
                  <option value="system">시스템 오디오</option>
                </select>
                <input value={sttSpeaker} onChange={(event) => setSttSpeaker(event.target.value)} placeholder="speaker label" />
                <button type="button" onClick={() => void startStt()} disabled={sttRunning}>Start STT</button>
                <button type="button" onClick={stopStt} disabled={!sttRunning}>Stop STT</button>
              </div>
              <p className="mutedLabel">{sttStatusDetail}</p>
              {lastDebug ? (
                <p className="mutedLabel">마지막 청크 #{lastDebug.chunk_id} / {lastDebug.status} / {formatBytes(lastDebug.bytes)}</p>
              ) : null}
              <div className="signalTimeline">
                {(sttLogs.length ? sttLogs.slice(-4).reverse() : ["로그 없음"]).map((line, idx) => (
                  <div key={`stt-log-${idx}`}>
                    <p>{line}</p>
                  </div>
                ))}
              </div>
              </div>
            </details>

            <details className="card panelCard sidebarSection panelFold" open={false}>
              <summary className="panelHeader tight panelFoldHeader">
                <h3>LLM 디버그 탭</h3>
                <span className="chip chipSoft">{llmIoLogs.length} logs</span>
              </summary>
              <div className="panelFoldBody">
                <div className="transcriptMetaBar">
                  <span className="chip chipSoft">req {llmReqCount}</span>
                  <span className="chip chipSoft">res {llmResCount}</span>
                  <span className="chip chipSoft">err {llmErrCount}</span>
                </div>
                {llmIoLogs.length === 0 ? (
                  <p className="emptyState compact">아직 LLM 요청/응답 로그가 없습니다.</p>
                ) : (
                  <div className="llmIoList">
                    {[...llmIoLogs].slice(-50).reverse().map((log) => {
                      const seq = Number(log?.seq || 0);
                      const at = safeText(log?.at, "-");
                      const direction = safeText(log?.direction, "-").toLowerCase();
                      const stage = safeText(log?.stage, "-");
                      const payload = safeText(log?.payload, "");
                      const dirLabel = direction === "request" ? "REQ" : direction === "response" ? "RES" : direction === "error" ? "ERR" : direction;
                      return (
                        <details key={`llm-io-${seq}-${at}-${stage}`} className="llmIoItem">
                          <summary>
                            <span className="chip chipSoft">{dirLabel}</span>
                            <span>#{seq}</span>
                            <span>{at}</span>
                            <span>{stage}</span>
                          </summary>
                          <pre className="emptyState compact llmIoPayload">{payload || "(empty)"}</pre>
                        </details>
                      );
                    })}
                  </div>
                )}
              </div>
            </details>
          </section>

          <section className="topGrid">
            <details className="card panelCard panelFold" open>
              <summary className="panelHeader panelFoldHeader"><h2>안건</h2></summary>
              <div className="panelFoldBody">
              {selectedAgenda ? (
                <section className="currentAgenda">
                  <p className="mutedLabel">현재 안건</p>
                  <h3>{agendaLabel(selectedAgenda)}</h3>
                  <div className="progressTrack"><span style={{ width: `${selectedAgenda.progress}%` }} /></div>
                  <div className="inlineMeta">
                    <span>{selectedAgenda.progress}% 완료</span>
                    <span>다음: {selectedAgenda.nextUp}</span>
                  </div>
                </section>
              ) : (
                <p className="emptyState compact">진행 중인 안건이 없어요.</p>
              )}

              <div className="agendaHealthGrid">
                <article><p className="mutedLabel">완료</p><strong>{agendaOverview.done}</strong></article>
                <article><p className="mutedLabel">진행 중</p><strong>{agendaOverview.inProgress}</strong></article>
                <article><p className="mutedLabel">시작 전</p><strong>{agendaOverview.notStarted}</strong></article>
              </div>

              <div className="agendaList">
                {agendas.map((agenda) => (
                  <button
                    key={agenda.id}
                    className={`agendaItem ${agenda.id === selectedAgendaId ? "agendaItemSelected" : ""}`}
                    type="button"
                    onClick={() => onSelectAgenda(agenda.id)}
                    disabled={analysisUiDisabled}
                  >
                    <div>
                      <p className="agendaTitle">{agendaLabel(agenda)}</p>
                      <p className="mutedLabel">신뢰도 {agenda.confidence}%</p>
                    </div>
                    <span className={agendaStatusClass[agenda.status]}>{agendaStatusLabel[agenda.status]}</span>
                  </button>
                ))}
              </div>

              <div className="panelActions">
                <button type="button" onClick={() => void apply(() => tickAnalysis(), "안건 추출 재실행 중", true)} disabled={loading || analysisUiDisabled}>추출 다시 실행</button>
                <button type="button" onClick={() => setQuery("")}>전사문으로 이동</button>
              </div>
              </div>
            </details>

            <details className="card panelCard summaryCard panelFold" open={false}>
              <summary className="panelHeader panelFoldHeader">
                <h2>안건 요약</h2>
                <span className="chip chipSoft">{summaryAgendas.length}개</span>
              </summary>
              <div className="panelFoldBody">
              <div className="panelHeader tight">
                <div className="segmented">
                  <button className={summaryScope === "current" ? "active" : ""} type="button" onClick={() => setSummaryScope("current")} disabled={analysisUiDisabled}>현재 안건</button>
                  <button className={summaryScope === "all" ? "active" : ""} type="button" onClick={() => setSummaryScope("all")} disabled={analysisUiDisabled}>전체</button>
                </div>
              </div>

              <div className="summarySignals">
                <article><p className="mutedLabel">신뢰도</p><strong>{selectedAgenda?.confidence ?? 0}%</strong></article>
                <article><p className="mutedLabel">의사결정</p><strong>{selectedContext.decisionCount}</strong></article>
                <article><p className="mutedLabel">근거</p><strong>{selectedContext.evidenceCount}</strong></article>
              </div>

              {summaryAgendas.length === 0 ? (
                <p className="emptyState">안건이 정리되면 요약이 보여요.</p>
              ) : (
                <div className="summarySections">
                  {summaryAgendas.map((agenda) => (
                    <section key={agenda.id} className="summaryBlock">
                      <h3>{agendaLabel(agenda)}</h3>
                      {agenda.keywords && agenda.keywords.length > 0 ? (
                        <div className="chipRow">
                          {agenda.keywords.slice(0, 8).map((kw) => (
                            <span key={`${agenda.id}-kw-${kw}`} className="chip chipSoft">#{kw}</span>
                          ))}
                        </div>
                      ) : null}
                      <div className="summaryGrid">
                        <div>
                          <p className="mutedLabel">핵심 포인트</p>
                          {agenda.keyPoints.length === 0 ? <p className="emptyState compact">아직 핵심 포인트가 없습니다.</p> : <ul className="bulletList">{agenda.keyPoints.map((point, pointIdx) => {
                            const targetId = agenda.summaryPointIds?.[pointIdx] || `summary-${agenda.id}-${pointIdx}`;
                            const domId = `evi-target-${targetId}`;
                            const meta = summaryPointMetaMap.get(`${agenda.id}|${targetId}`);
                            const rangeLabel = meta?.rangeLabel || buildTimeRangeLabel([point]);
                            const clickable = Boolean(meta || extractTimestampToken(point));
                            return (
                              <li key={`${point}-${pointIdx}`}>
                                <button
                                  id={domId}
                                  className={`ghostButton ${focusedTargetDomId === domId ? "focusFlash" : ""}`}
                                  type="button"
                                  onClick={() => jumpBySummary(agenda.id, point, targetId)}
                                  disabled={analysisUiDisabled || !clickable}
                                >
                                  <span>{stripLeadingTimestamp(point)}</span>
                                  {rangeLabel !== "-" ? <span className="summaryPointRange">{rangeLabel}</span> : null}
                                </button>
                                {meta && meta.opinionGroups.length > 0 ? (
                                  <div className="summaryOpinionBlock">
                                    <p className="mutedLabel">의견 요약</p>
                                    <ul className="summaryOpinionList">
                                      {meta.opinionGroups.map((group) => (
                                        <li key={group.id}>
                                          <button
                                            className="opinionSummaryButton"
                                            type="button"
                                            onClick={() => focusByOpinionGroup(agenda.id, targetId, group.id)}
                                            disabled={analysisUiDisabled}
                                          >
                                            <span className={`chip chipSoft opinionTypeChip opinionType-${group.type}`}>{group.typeLabel}</span>
                                            <span>{group.summary}</span>
                                            {group.detail ? <span className="opinionDetail">{group.detail}</span> : null}
                                            {group.rangeLabel !== "-" ? <span className="summaryPointRange">{group.rangeLabel}</span> : null}
                                          </button>
                                        </li>
                                      ))}
                                    </ul>
                                  </div>
                                ) : null}
                              </li>
                            );
                          })}</ul>}
                        </div>
                        <div>
                          <p className="mutedLabel">리스크</p>
                          {agenda.risks.length === 0 ? <p className="emptyState compact">기록된 리스크가 없습니다.</p> : <ul className="bulletList">{agenda.risks.map((risk) => <li key={risk}>{risk}</li>)}</ul>}
                        </div>
                        <div>
                          <p className="mutedLabel">현재까지의 의사결정</p>
                          {agenda.decisionSoFar.length === 0 ? <p className="emptyState compact">아직 의사결정이 없습니다.</p> : <ul className="bulletList">{agenda.decisionSoFar.map((decisionPoint) => <li key={decisionPoint}>{decisionPoint}</li>)}</ul>}
                        </div>
                        <div>
                          <p className="mutedLabel">다음 질문</p>
                          {agenda.nextQuestions.length === 0 ? <p className="emptyState compact">열린 질문이 없습니다.</p> : <ul className="bulletList">{agenda.nextQuestions.map((question) => <li key={question}>{question}</li>)}</ul>}
                        </div>
                      </div>
                      <div className="inlineMeta">
                        <span>신뢰도 {agenda.confidence}%</span>
                        <span>업데이트 {agenda.lastUpdated}</span>
                      </div>
                    </section>
                  ))}
                </div>
              )}

              <section className="summaryEvidence">
                <div className="panelHeader tight">
                  <h3>관련 근거</h3>
                  <span className="chip chipSoft">{summaryEvidence.length}개 링크</span>
                </div>
                {summaryEvidence.length === 0 ? (
                  <p className="emptyState compact">이 안건의 근거 스니펫이 아직 없어요.</p>
                ) : (
                  <div className="miniEvidenceList">
                    {summaryEvidence.slice(0, 5).map((item) => (
                      <button key={item.id} className="miniEvidence" type="button" onClick={() => jumpToTranscript(item.agendaId, item.timestamp)} disabled={analysisUiDisabled}>
                        <span className="timestamp">{item.timestamp}</span>
                        <span className="chip chipSpeaker">{item.speaker}</span>
                        <p>{item.quote}</p>
                      </button>
                    ))}
                  </div>
                )}
              </section>
              </div>
            </details>
          </section>

          <div className="bottomFilter">
            <span className="chip chipInteractive">필터 기준: {selectedAgenda ? agendaLabel(selectedAgenda) : "없음"}</span>
            <span className="mutedLabel">하단 섹션은 선택된 안건과 동기화돼요.</span>
          </div>

          <section className="bottomDesktop">
            <div className="stackColumn">{renderSummaryCard()}{renderDecisionCard()}</div>
            <div className="stackColumn">{renderActionCard()}{renderEvidenceCard()}</div>
          </section>
          </section>
        </div>
      </main>
      {isCanvasMode ? (
        <>
          <button
            type="button"
            className={`edgeNudge edgeNudgeCanvas edgeNudgeCanvasLeft ${canvasLeftRailOpen ? "edgeNudgeOpen" : "edgeNudgeClosed"}`}
            onClick={() => setCanvasLeftRailOpen((open) => !open)}
            aria-label={canvasLeftRailOpen ? "왼쪽 패널 접기" : "왼쪽 패널 펼치기"}
            style={{ left: `${(sidebarOpen ? sidebarWidth : 0) + (canvasLeftRailOpen ? canvasLeftRailWidth + 12 : 8)}px` }}
          >
            {canvasLeftRailOpen ? "‹" : "›"}
          </button>
          <button
            type="button"
            className={`edgeNudge edgeNudgeCanvas edgeNudgeCanvasRight ${canvasRightRailOpen ? "edgeNudgeOpen" : "edgeNudgeClosed"}`}
            onClick={() => setCanvasRightRailOpen((open) => !open)}
            aria-label={canvasRightRailOpen ? "오른쪽 패널 접기" : "오른쪽 패널 펼치기"}
            style={{ right: `${canvasRightRailOpen ? canvasRightRailWidth + 12 : 8}px` }}
          >
            {canvasRightRailOpen ? "›" : "‹"}
          </button>
        </>
      ) : null}
    </div>
  );
}
