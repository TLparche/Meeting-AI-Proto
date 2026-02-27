"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  connectLlm,
  disconnectLlm,
  getLlmStatus,
  getState,
  importJsonDir,
  importJsonFiles,
  pingLlm,
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
  decision_results?: AgendaOutcomeDecision[];
  action_items?: AgendaOutcomeAction[];
  start_turn_id?: number;
  end_turn_id?: number;
};

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

export default function Home() {
  const [state, setState] = useState<MeetingState>(EMPTY_STATE);
  const [loading, setLoading] = useState(false);
  const [activeTask, setActiveTask] = useState("");
  const [taskStartedAt, setTaskStartedAt] = useState<number | null>(null);
  const [taskElapsedSec, setTaskElapsedSec] = useState(0);
  const [analysisPending, setAnalysisPending] = useState(false);
  const [focusedTargetDomId, setFocusedTargetDomId] = useState("");
  const [error, setError] = useState("");

  const [query, setQuery] = useState("");
  const [speakerFilter, setSpeakerFilter] = useState("전체");
  const [highlightRelated, setHighlightRelated] = useState(true);
  const [summaryScope, setSummaryScope] = useState<SummaryScope>("current");
  const [selectedAgendaId, setSelectedAgendaId] = useState("");

  const [datasetFolder, setDatasetFolder] = useState("dataset/economy");
  const [datasetFiles, setDatasetFiles] = useState<File[]>([]);
  const [datasetImportInfo, setDatasetImportInfo] = useState("");
  const [meetingGoalDraft, setMeetingGoalDraft] = useState("");
  const [meetingGoalDirty, setMeetingGoalDirty] = useState(false);

  const [llmChecking, setLlmChecking] = useState(false);
  const [llmPingMessage, setLlmPingMessage] = useState("");
  const [llmPingOk, setLlmPingOk] = useState<boolean | null>(null);

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
      setState(next);
      setError("");
    } catch (err) {
      setError((err as Error).message);
    }
  }, []);

  const refreshLlmStatus = useCallback(async () => {
    try {
      const status = await getLlmStatus();
      setState((prev) => ({ ...prev, llm_status: status }));
    } catch {
      // noop
    }
  }, []);

  useEffect(() => {
    void loadState();
  }, [loadState]);

  useEffect(() => {
    const id = window.setInterval(() => {
      void loadState();
    }, 1400);
    return () => window.clearInterval(id);
  }, [loadState]);

  useEffect(() => {
    const id = window.setInterval(() => {
      void refreshLlmStatus();
    }, 3000);
    return () => window.clearInterval(id);
  }, [refreshLlmStatus]);

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
      setState(next);
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
      setState(next);
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
      setState(res.state);
      setMeetingGoalDraft(res.state.meeting_goal || "");
      setMeetingGoalDirty(false);
      setError("");
      const d = res.import_debug;
      setDatasetImportInfo(
        `loaded=${d.added}, files=${d.files_parsed}/${d.files_scanned}, skipped=${d.files_skipped}, ticked=${d.ticked ? "yes" : "no"}`,
      );
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
    beginTask("JSON 업로드 분석 중");
    setAnalysisPending(true);
    try {
      const res = await importJsonFiles({ files: datasetFiles, reset_state: true, auto_tick: true });
      setState(res.state);
      setMeetingGoalDraft(res.state.meeting_goal || "");
      setMeetingGoalDirty(false);
      setError("");
      const d = res.import_debug;
      setDatasetImportInfo(
        `uploaded=${datasetFiles.length}, loaded=${d.added}, files=${d.files_parsed}/${d.files_scanned}, skipped=${d.files_skipped}, ticked=${d.ticked ? "yes" : "no"}`,
      );
      if (d.warning) setError(d.warning);
    } catch (err) {
      setError((err as Error).message);
      setDatasetImportInfo("");
    } finally {
      setAnalysisPending(false);
      endTask();
    }
  };

  const onPingLlm = async () => {
    setLlmChecking(true);
    try {
      const res = await pingLlm();
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
      setState(res.state);
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
      setState(res.state);
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
      setState(res.state);
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
  }, [appendSttLog, sttSpeaker]);

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

  const sortedOutcomeRows = useMemo<AgendaOutcome[]>(
    () => [...outcomeRows].sort((a, b) => Number(a.start_turn_id || 0) - Number(b.start_turn_id || 0)),
    [outcomeRows],
  );

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
        confidence: title === activeTitle ? Math.round(Number(state.analysis?.agenda?.active?.confidence || 85)) : 78,
        progress: statusProgress(status),
        nextUp: "다음 안건",
        keyPoints: keyPoints.slice(0, 8),
        risks: [],
        decisionSoFar: decisionConclusions.slice(0, 6),
        nextQuestions: [],
        keywords: Array.from(new Set(keywords)).slice(0, 8),
        summaryPointIds: keyPoints.map((_, pointIdx) => `summary-${idx}-${pointIdx}`),
        summaryBullets: summaries.slice(0, 6),
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
    if (agendas.length === 0) {
      setSelectedAgendaId("");
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

  const filteredTranscript = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return transcript.filter((utterance) => {
      const speakerMatch = speakerFilter === "전체" || utterance.speaker === speakerFilter;
      const queryMatch =
        normalizedQuery.length === 0 ||
        utterance.text.toLowerCase().includes(normalizedQuery) ||
        utterance.speaker.toLowerCase().includes(normalizedQuery) ||
        utterance.timestamp.includes(normalizedQuery);
      return speakerMatch && queryMatch;
    });
  }, [query, speakerFilter, transcript]);

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

  const onSelectAgenda = (agendaId: string) => {
    if (analysisUiDisabled) return;
    setSelectedAgendaId(agendaId);
    setSummaryScope("current");
  };

  const jumpToTranscript = (agendaId: string, timestamp: string) => {
    if (analysisUiDisabled) return;
    setSelectedAgendaId(agendaId);
    setQuery(timestamp);
  };

  const extractTimestamp = (text: string): string => {
    const m = text.match(/\[(\d{2}:\d{2}(?::\d{2})?)\]/);
    return m ? m[1] : "";
  };

  const jumpBySummary = (agendaId: string, summaryText: string) => {
    if (analysisUiDisabled) return;
    const ts = extractTimestamp(summaryText);
    if (!ts) return;
    jumpToTranscript(agendaId, ts);
  };

  const focusTargetCard = (agendaId: string, targetId: string) => {
    if (analysisUiDisabled) return;
    const cleanTarget = safeText(targetId);
    if (!cleanTarget) return;

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

  const llmEnabled = Boolean(state.llm_enabled);
  const analysisUiDisabled = analysisPending;

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

  const renderSummaryCard = () => (
    <article className="card panelCard">
      <div className="panelHeader">
        <h3>회의 요약 (안건별, 완료 안건)</h3>
        <span className="chip chipSoft">{bottomAgendas.length}개 안건 보기</span>
      </div>
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
                  {agenda.summaryBullets.map((point) => (
                    <li key={point}>
                      <button
                        className="ghostButton"
                        type="button"
                        onClick={() => jumpBySummary(agenda.id, point)}
                        disabled={analysisUiDisabled || !extractTimestamp(point)}
                      >
                        {point}
                      </button>
                    </li>
                  ))}
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
    </article>
  );

  const renderDecisionCard = () => (
    <article className="card panelCard">
      <div className="panelHeader">
        <h3>의사결정 결과</h3>
        <span className="chip chipSoft">{bottomDecisions.length}건 기록됨</span>
      </div>
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
    </article>
  );

  const renderActionCard = () => (
    <article className="card panelCard">
      <div className="panelHeader">
        <h3>액션 아이템</h3>
        <span className="chip chipSoft">{bottomActions.length}건</span>
      </div>
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
    </article>
  );

  const renderEvidenceCard = () => (
    <article className="card panelCard">
      <div className="panelHeader">
        <h3>근거 로그</h3>
        <span className="chip chipSoft">{bottomEvidence.length}개 스니펫</span>
      </div>
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
    </article>
  );

  return (
    <div className="workspaceShell">
      <aside className="sidebar">
        <div className="sidebarInner">
          <div>
            <p className="brand">파르체</p>
            <p className="brandSub">회의 인텔리전스</p>
          </div>
          <nav className="sidebarNav">
            <button className="navItem" type="button">대시보드</button>
            <button className="navItem navItemActive" type="button">회의 워크스페이스</button>
            <button className="navItem" type="button">리포트</button>
            <button className="navItem" type="button">팀 노트</button>
          </nav>

        </div>
      </aside>

      <main className="mainArea">
        <div className="mainInner">
          <section className="leftSection">
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
            </div>
          </header>

          <nav className="awsTabs awsTabsSeparate" aria-label="회의 워크스페이스 탭">
            <button className="awsTab awsTabActive" type="button">개요</button>
            <button className="awsTab" type="button">전사문 검토</button>
            <button className="awsTab" type="button">안건 인사이트</button>
            <button className="awsTab" type="button">결과</button>
          </nav>

          <article className="card panelCard">
            <div className="panelHeader tight">
              <h3>실행 제어</h3>
              <span className="chip chipSoft">LLM {state.llm_status?.connected ? "연결됨" : "미연결"}</span>
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
              <select
                aria-label="전사 윈도우"
                value={state.window_size}
                onChange={(event) => setState((s) => ({ ...s, window_size: Number(event.target.value) || 12 }))}
              >
                {[8, 12, 20, 30, 40, 60].map((n) => (
                  <option key={n} value={n}>{n} turns</option>
                ))}
              </select>
            </div>
            <div className="panelActions">
              <button type="button" onClick={() => void onSaveConfig()} disabled={loading}>설정 저장</button>
              <button type="button" onClick={() => void onImportDataset()} disabled={loading}>JSON 폴더 로드</button>
              <button type="button" onClick={() => void onImportDatasetFiles()} disabled={loading || datasetFiles.length === 0}>JSON 업로드</button>
              <button type="button" onClick={() => void apply(() => tickAnalysis(), "전체 분석 실행 중", true)} disabled={loading || analysisUiDisabled}>분석 실행</button>
              <button type="button" onClick={() => void apply(() => resetState(), "상태 초기화 중")} disabled={loading}>초기화</button>
              <button type="button" onClick={() => void onConnectLlm()} disabled={llmChecking}>{llmChecking ? "연결 중" : "LLM 연결"}</button>
              <button type="button" onClick={() => void onDisconnectLlm()} disabled={llmChecking || !llmEnabled}>연결 해제</button>
              <button type="button" onClick={() => void onPingLlm()} disabled={llmChecking}>{llmChecking ? "확인 중" : "연결 테스트"}</button>
            </div>
            <p className="mutedLabel">전사 건수: {state.transcript?.length || 0}</p>
            {loading ? (
              <p className="runIndicator">
                <span className="runDot" />
                <span>{activeTask || "작업 실행 중"} ({taskElapsedSec}s)</span>
              </p>
            ) : null}
            {analysisUiDisabled ? <p className="mutedLabel">분석 비활성화: 결과 수신 대기 중</p> : null}
            {datasetImportInfo ? <p className="mutedLabel">{datasetImportInfo}</p> : null}
            {llmPingMessage ? <p className="mutedLabel">Ping: {llmPingMessage}</p> : null}
            {llmPingOk === false ? <p className="mutedLabel">LLM 연결 오류를 확인하세요.</p> : null}
            {state.llm_status?.last_error ? <p className="mutedLabel">LLM 오류: {state.llm_status.last_error}</p> : null}
            {state.analysis_runtime?.control_plane_reason ? <p className="mutedLabel">분석 상태: {state.analysis_runtime.control_plane_reason}</p> : null}
            {state.analysis_runtime?.used_local_fallback ? <p className="mutedLabel">현재 로컬 폴백 분석 모드</p> : null}
            {error ? <p className="emptyState compact">{error}</p> : null}

            <details open>
              <summary>디버그 패널</summary>
              <div className="transcriptMetaBar">
                <span className="chip chipSoft">agenda_outcomes: {outcomeRows.length}</span>
                <span className="chip chipSoft">active: {state.analysis?.agenda?.active?.title || "-"}</span>
                <span className="chip chipSoft">decisions: {decisions.length}</span>
                <span className="chip chipSoft">actions: {actionItems.length}</span>
              </div>
              <div className="panelActions">
                <button type="button" onClick={() => void onDebugRefresh()} disabled={loading || llmChecking}>상태 강제 새로고침</button>
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
            </details>
          </article>
          
          <article className="card panelCard transcriptCard transcriptCardLeft">
            <div className="panelHeader">
              <h2>전사문 (전체)</h2>
              <span className="chip chipSoft">{filteredTranscript.length}개 표시</span>
            </div>
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
            <div className="transcriptList">
              {filteredTranscript.length === 0 ? (
                <p className="emptyState">현재 필터와 일치하는 발화가 없습니다. 검색어나 화자 필터를 조정해 주세요.</p>
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
          </article>
          </section>

          <section className="rightSection">
          <section className="contentSignalGrid">
            <article className="card panelCard sidebarSection">
              <div className="panelHeader tight">
                <h3>실시간 참여자</h3>
                <span className="chip chipSoft">{participantRoster.length}명 참여 중</span>
              </div>
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
            </article>

            <article className="card panelCard sidebarSection">
              <div className="panelHeader tight">
                <h3>STT 스트림</h3>
                <span className="chip chipSoft">{sttStatusText}</span>
              </div>
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
            </article>
          </section>

          <section className="topGrid">
            <article className="card panelCard">
              <div className="panelHeader"><h2>안건</h2></div>
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
            </article>

            <article className="card panelCard summaryCard">
              <div className="panelHeader">
                <h2>안건 요약</h2>
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
                            return (
                              <li key={point}>
                                <button
                                  id={domId}
                                  className={`ghostButton ${focusedTargetDomId === domId ? "focusFlash" : ""}`}
                                  type="button"
                                  onClick={() => jumpBySummary(agenda.id, point)}
                                  disabled={analysisUiDisabled || !extractTimestamp(point)}
                                >
                                  {point}
                                </button>
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
            </article>
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
    </div>
  );
}
