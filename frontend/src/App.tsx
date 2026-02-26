import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  addUtterance,
  createArtifact,
  getState,
  importJsonDir,
  importJsonFiles,
  resetState,
  saveConfig,
  tickAnalysis,
  transcribeChunk,
} from "./api";
import type { ArtifactKind, MeetingState, SttDebug } from "./types";

const EMPTY_STATE: MeetingState = {
  meeting_goal: "",
  initial_context: "",
  window_size: 12,
  transcript: [],
  agenda_stack: [],
  agenda_candidates: [],
  agenda_vectors: {},
  agenda_state_map: {},
  active_agenda_id: "",
  agenda_events: [],
  evidence_status: "UNVERIFIED",
  evidence_snippet: "",
  evidence_log: [],
  fairtalk_glow: [],
  fairtalk_debug: { active_speakers: 0, soft_count: 0, strong_count: 0, rule: "intent_only" },
  analysis: null,
  artifacts: {},
};

const CLIENT_STEPS = [
  "IDLE",
  "REQUEST_PERMISSION",
  "CAPTURING",
  "UPLOAD_CHUNK",
  "WAIT_SERVER",
  "DRAINING",
  "SERVER_TRANSCRIBED",
  "SERVER_NO_SPEECH",
  "EMPTY_CHUNK",
  "NO_AUDIO_TRACK",
  "START_FAILED",
  "UPLOAD_FAILED",
  "ERROR",
];

function formatBytes(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function App() {
  const [state, setState] = useState<MeetingState>(EMPTY_STATE);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [sttSpeaker, setSttSpeaker] = useState("시스템오디오");
  const [sttRunning, setSttRunning] = useState(false);
  const [sttSource, setSttSource] = useState<"mic" | "system">("system");
  const [sttStatusText, setSttStatusText] = useState("STOPPED");
  const [sttStatusDetail, setSttStatusDetail] = useState("Start STT를 누르면 상태가 RUNNING으로 바뀝니다.");
  const [sttStep, setSttStep] = useState("IDLE");
  const [sttLogs, setSttLogs] = useState<string[]>([]);
  const [lastDebug, setLastDebug] = useState<SttDebug | null>(null);
  const [debugHistory, setDebugHistory] = useState<SttDebug[]>([]);
  const [liveBanner, setLiveBanner] = useState<{ text: string; kind: "info" | "lock" | "warn" } | null>(null);
  const [datasetFolder, setDatasetFolder] = useState("dataset/economy");
  const [datasetFiles, setDatasetFiles] = useState<File[]>([]);
  const [datasetImportInfo, setDatasetImportInfo] = useState("");

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const stopRequestedRef = useRef(false);
  const chunkSeqRef = useRef(0);
  const chunkTimerRef = useRef<number | null>(null);
  const sendQueueRef = useRef<Promise<void>>(Promise.resolve());
  const sttSessionRef = useRef(0);
  const pendingChunksRef = useRef(0);
  const stopReasonRef = useRef("사용자가 STT를 중지했습니다.");
  const bannerTimerRef = useRef<number | null>(null);
  const bannerInitializedRef = useRef(false);
  const prevActiveAgendaIdRef = useRef("");
  const prevDecisionLockRef = useRef(false);
  const prevL2Ref = useRef(false);
  const prevDriftStateRef = useRef("Normal");

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

  useEffect(() => {
    void loadState();
  }, [loadState]);

  useEffect(() => {
    const id = window.setInterval(() => {
      void loadState();
    }, 1200);
    return () => window.clearInterval(id);
  }, [loadState]);

  const showLiveBanner = useCallback((text: string, kind: "info" | "lock" | "warn" = "info") => {
    if (bannerTimerRef.current !== null) {
      window.clearTimeout(bannerTimerRef.current);
      bannerTimerRef.current = null;
    }
    setLiveBanner({ text, kind });
    bannerTimerRef.current = window.setTimeout(() => {
      setLiveBanner(null);
      bannerTimerRef.current = null;
    }, 3000);
  }, []);

  useEffect(() => {
    return () => {
      if (bannerTimerRef.current !== null) {
        window.clearTimeout(bannerTimerRef.current);
      }
    };
  }, []);

  const agendaBuckets = useMemo(() => {
    const active: Array<{ title: string; status: string }> = [];
    const proposed: Array<{ title: string; status: string }> = [];
    const closed: Array<{ title: string; status: string }> = [];
    state.agenda_stack.forEach((item) => {
      if (item.status === "ACTIVE" || item.status === "CLOSING") {
        active.push({ title: item.title, status: item.status });
        return;
      }
      if (item.status === "CLOSED") {
        closed.push({ title: item.title, status: item.status });
        return;
      }
      proposed.push({ title: item.title, status: item.status });
    });
    return { active, proposed, closed };
  }, [state.agenda_stack]);

  const currentAgendaText = useMemo(() => {
    const activeId = state.active_agenda_id || "";
    if (activeId && state.agenda_state_map && state.agenda_state_map[activeId]?.title) {
      return state.agenda_state_map[activeId].title;
    }
    return state.analysis?.agenda?.active?.title || "아직 ACTIVE 아젠다가 없습니다.";
  }, [state.active_agenda_id, state.agenda_state_map, state.analysis]);

  const kCoreTags = useMemo(() => {
    const core = state.analysis?.keywords?.k_core;
    if (!core) {
      return [];
    }
    const merged = [...core.object, ...core.constraints, ...core.criteria].filter((x) => String(x).trim());
    return Array.from(new Set(merged)).slice(0, 8);
  }, [state.analysis]);

  const dpsPercent = useMemo(() => {
    const raw = Number(state.analysis?.scores?.dps?.score ?? 0);
    const pct = Math.round(raw * 100);
    return Math.max(0, Math.min(100, pct));
  }, [state.analysis]);
  const evidenceStatus = state.evidence_status || state.analysis?.evidence_gate?.status || "UNVERIFIED";
  const evidenceSnippetLines = useMemo(() => {
    return String(state.evidence_snippet || "")
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean)
      .slice(0, 2);
  }, [state.evidence_snippet]);
  const evidenceLogPreview = useMemo(() => {
    return [...(state.evidence_log || [])].slice(-5).reverse();
  }, [state.evidence_log]);
  const recoDebug = state.recommendation_debug;
  const participantFrames = useMemo(() => {
    const fromState = state.fairtalk_glow || [];
    if (fromState.length > 0) {
      return fromState;
    }
    const fallback = state.analysis?.scores?.participation?.fairtalk || [];
    return fallback.map((p) => ({
      speaker: p.speaker,
      p_intent: Number(p.p_intent || 0),
      glow: "none" as const,
      intent_active: false,
      last_seen_sec: 999,
    }));
  }, [state.fairtalk_glow, state.analysis]);
  const fairtalkDebug = state.fairtalk_debug;
  const transcriptCount = state.transcript.length;
  const activeAgendaCount = agendaBuckets.active.length;
  const proposedAgendaCount = agendaBuckets.proposed.length;
  const closedAgendaCount = agendaBuckets.closed.length;
  const lastTranscript = transcriptCount > 0 ? state.transcript[transcriptCount - 1] : null;
  const engineChecklist = useMemo(
    () => [
      { name: "Keyword Engine", ok: Boolean(state.analysis?.keywords) },
      {
        name: "Agenda Tracker",
        ok: (state.agenda_candidates?.length ?? 0) > 0 || Object.keys(state.agenda_vectors || {}).length > 0,
      },
      { name: "Agenda FSM", ok: Object.keys(state.agenda_state_map || {}).length > 0 || Boolean(state.active_agenda_id) },
      { name: "Drift Dampener", ok: Boolean(state.drift_state) },
      { name: "DPS", ok: typeof state.dps_t === "number" || Boolean(state.analysis?.scores?.dps) },
      { name: "Flow Pulse", ok: Boolean(state.loop_state) },
      { name: "Decision Lock", ok: Boolean(state.decision_lock_debug || state.analysis?.intervention?.decision_lock) },
    ],
    [state],
  );

  const activeAgendaState = useMemo(() => {
    const aid = state.active_agenda_id || "";
    if (!aid || !state.agenda_state_map || !state.agenda_state_map[aid]) {
      return "";
    }
    return state.agenda_state_map[aid].state;
  }, [state.active_agenda_id, state.agenda_state_map]);

  const driftState = state.drift_state || "Normal";
  const driftCues = state.drift_ui_cues || {
    glow_k_core: false,
    fix_k_core_focus: false,
    reduce_facets: false,
    show_banner: false,
  };
  const coreGlow = Boolean(driftCues.glow_k_core);
  const fixedFocus = Boolean(driftCues.fix_k_core_focus);
  const reduceFacets = Boolean(driftCues.reduce_facets);
  const closingModeActive = Boolean(
    state.analysis?.intervention?.decision_lock?.triggered || activeAgendaState === "CLOSING",
  );

  useEffect(() => {
    const activeAgendaId = state.active_agenda_id || "";
    const decisionLock = Boolean(state.analysis?.intervention?.decision_lock?.triggered);
    const isL2 = state.analysis?.intervention?.level === "L2";
    const l2Banner = state.analysis?.intervention?.banner_text || "논의 재정렬이 필요합니다.";

    if (!bannerInitializedRef.current) {
      prevActiveAgendaIdRef.current = activeAgendaId;
      prevDecisionLockRef.current = decisionLock;
      prevL2Ref.current = isL2;
      bannerInitializedRef.current = true;
      return;
    }

    if (activeAgendaId && activeAgendaId !== prevActiveAgendaIdRef.current) {
      showLiveBanner(`Re-orientation: CURRENT AGENDA 변경 -> ${currentAgendaText}`, "warn");
    }
    if (decisionLock && !prevDecisionLockRef.current) {
      showLiveBanner("Decision Lock: 결론 단계(CLOSING)로 전환 신호가 감지되었습니다.", "lock");
    }
    if (isL2 && !prevL2Ref.current) {
      showLiveBanner(`L2 Prompt: ${l2Banner}`, "warn");
    }
    if (driftState === "Re-orient" && prevDriftStateRef.current !== "Re-orient") {
      showLiveBanner("Re-orientation: 논의를 CURRENT AGENDA와 K_core 중심으로 재정렬하세요.", "warn");
    }

    prevActiveAgendaIdRef.current = activeAgendaId;
    prevDecisionLockRef.current = decisionLock;
    prevL2Ref.current = isL2;
    prevDriftStateRef.current = driftState;
  }, [state.active_agenda_id, state.analysis, currentAgendaText, showLiveBanner, driftState]);

  const apply = async (action: () => Promise<MeetingState>) => {
    setLoading(true);
    try {
      const next = await action();
      setState(next);
      setError("");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const onSaveConfig = () =>
    apply(() =>
      saveConfig({
        meeting_goal: state.meeting_goal,
        initial_context: state.initial_context,
        window_size: state.window_size,
      }),
    );

  const onImportDataset = async () => {
    setLoading(true);
    try {
      const res = await importJsonDir({
        folder: datasetFolder || "dataset/economy",
        recursive: true,
        reset_state: true,
        auto_tick: true,
        max_files: 500,
      });
      setState(res.state);
      setError("");
      const d = res.import_debug;
      setDatasetImportInfo(
        `loaded=${d.added}, files=${d.files_parsed}/${d.files_scanned}, skipped=${d.files_skipped}, ticked=${d.ticked ? "yes" : "no"}`,
      );
    } catch (err) {
      setError((err as Error).message);
      setDatasetImportInfo("");
    } finally {
      setLoading(false);
    }
  };

  const onImportDatasetFiles = async () => {
    if (datasetFiles.length === 0) {
      setError("업로드할 JSON 파일을 먼저 선택하세요.");
      return;
    }
    setLoading(true);
    try {
      const res = await importJsonFiles({
        files: datasetFiles,
        reset_state: true,
        auto_tick: true,
      });
      setState(res.state);
      setError("");
      const d = res.import_debug;
      setDatasetImportInfo(
        `uploaded=${datasetFiles.length}, loaded=${d.added}, files=${d.files_parsed}/${d.files_scanned}, skipped=${d.files_skipped}, ticked=${d.ticked ? "yes" : "no"}`,
      );
    } catch (err) {
      setError((err as Error).message);
      setDatasetImportInfo("");
    } finally {
      setLoading(false);
    }
  };

  const onActionVote = () =>
    apply(async () => {
      await addUtterance({
        speaker: "SYSTEM",
        text: "[ACTION] 표결 진행: 현재 안건에 대해 최종 동의/승인을 요청합니다.",
      });
      return tickAnalysis();
    });

  const onActionSummary = () =>
    apply(async () => {
      await addUtterance({
        speaker: "SYSTEM",
        text: "[ACTION] 클로징 요약: 결론 요약을 공유하고 확정 여부를 확인합니다.",
      });
      await tickAnalysis();
      return createArtifact("meeting_summary");
    });

  const onActionDecide = () =>
    apply(async () => {
      await addUtterance({
        speaker: "SYSTEM",
        text: "[ACTION] 최종확정/승인: 결론을 확정하고 CLOSED 전환을 진행합니다.",
      });
      await tickAnalysis();
      return createArtifact("decision_results");
    });

  const sendChunk = async (sessionId: number, localSeq: number, blob: Blob, filename: string, source: string) => {
    if (sessionId !== sttSessionRef.current) {
      return;
    }
    try {
      if (stopRequestedRef.current) {
        setSttStep("DRAINING");
        setSttStatusText("DRAINING");
        setSttStatusDetail(`남은 청크 처리 중... (${pendingChunksRef.current}개 대기)`);
      } else {
        setSttStep("UPLOAD_CHUNK");
        setSttStatusText("RUNNING");
        setSttStatusDetail(`청크 업로드 중... (#${localSeq})`);
      }
      appendSttLog(`chunk #${localSeq} upload started (${Math.round(blob.size / 1024)} KB)`);
      if (stopRequestedRef.current) {
        setSttStep("DRAINING");
        setSttStatusText("DRAINING");
        setSttStatusDetail(`남은 청크 처리 중... (${pendingChunksRef.current}개 대기)`);
      } else {
        setSttStep("WAIT_SERVER");
        setSttStatusDetail(`서버 전사 처리 대기 중... (#${localSeq})`);
      }

      const res = await transcribeChunk({
        blob,
        filename,
        speaker: sttSpeaker || "시스템오디오",
        source,
      });
      if (sessionId !== sttSessionRef.current) {
        return;
      }
      setState(res.state);
      setError("");
      if (stopRequestedRef.current) {
        setSttStatusText("DRAINING");
      } else {
        setSttStatusText(res.stt_debug.status === "error" ? "ERROR" : "RUNNING");
      }
      setLastDebug(res.stt_debug);
      setDebugHistory((prev) => [res.stt_debug, ...prev].slice(0, 20));
      if (stopRequestedRef.current) {
        setSttStep("DRAINING");
      } else if (res.stt_debug.status === "ok") {
        setSttStep("SERVER_TRANSCRIBED");
      } else if (res.stt_debug.status === "empty") {
        setSttStep("SERVER_NO_SPEECH");
      } else {
        setSttStep("ERROR");
      }
      const steps = res.stt_debug.steps
        .map((s, idx) => {
          const prev = idx > 0 ? res.stt_debug.steps[idx - 1].t_ms : 0;
          return `${s.step}(+${s.t_ms - prev}ms)`;
        })
        .join(" > ");
      const errorSuffix = res.stt_debug.error ? ` | ${res.stt_debug.error}` : "";
      if (stopRequestedRef.current) {
        setSttStatusDetail(`남은 청크 처리 중... (${pendingChunksRef.current}개 대기)`);
      } else {
        setSttStatusDetail(
          `chunk #${res.stt_debug.chunk_id} ${res.stt_debug.status} (${res.stt_debug.duration_ms}ms)${errorSuffix}`,
        );
      }
      appendSttLog(
        `chunk #${res.stt_debug.chunk_id} ${res.stt_debug.status} | ${res.stt_debug.duration_ms}ms | ${steps}`,
      );
      if (res.stt_debug.error) {
        appendSttLog(`chunk #${res.stt_debug.chunk_id} error: ${res.stt_debug.error}`);
      }
      if (res.stt_debug.transcript_preview) {
        appendSttLog(`chunk #${res.stt_debug.chunk_id} text: ${res.stt_debug.transcript_preview}`);
      }
    } catch (err) {
      setError((err as Error).message);
      setSttStatusText("ERROR");
      setSttStep("UPLOAD_FAILED");
      setSttStatusDetail(`청크 업로드 실패: ${(err as Error).message}`);
      appendSttLog(`chunk #${localSeq} failed: ${(err as Error).message}`);
    }
  };

  const cleanupStt = (status: string, detail: string) => {
    if (chunkTimerRef.current !== null) {
      window.clearTimeout(chunkTimerRef.current);
      chunkTimerRef.current = null;
    }
    recorderRef.current = null;
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setSttRunning(false);
    setSttStatusText(status);
    setSttStatusDetail(detail);
    setSttStep(status === "STOPPED" ? "IDLE" : status);
  };

  const finalizeStopIfDrained = useCallback(() => {
    const recInactive = !recorderRef.current || recorderRef.current.state === "inactive";
    if (!stopRequestedRef.current || !recInactive || pendingChunksRef.current > 0) {
      return;
    }
    cleanupStt("STOPPED", stopReasonRef.current);
  }, []);

  const requestStopAndDrain = useCallback((reason: string) => {
    stopRequestedRef.current = true;
    stopReasonRef.current = reason;
    if (chunkTimerRef.current !== null) {
      window.clearTimeout(chunkTimerRef.current);
      chunkTimerRef.current = null;
    }

    const rec = recorderRef.current;
    if (rec && rec.state !== "inactive") {
      setSttStatusText("DRAINING");
      setSttStep("DRAINING");
      setSttStatusDetail(`남은 청크 처리 중... (${pendingChunksRef.current}개 대기)`);
      rec.stop();
      return;
    }
    finalizeStopIfDrained();
  }, [finalizeStopIfDrained]);

  const stopStt = () => {
    requestStopAndDrain("사용자가 STT를 중지했습니다.");
  };

  const startStt = async () => {
    if (!navigator.mediaDevices) {
      setError("이 브라우저는 mediaDevices를 지원하지 않습니다.");
      return;
    }
    try {
      sttSessionRef.current += 1;
      const sessionId = sttSessionRef.current;
      chunkSeqRef.current = 0;
      pendingChunksRef.current = 0;
      sendQueueRef.current = Promise.resolve();
      stopRequestedRef.current = false;
      setSttStatusText("CONNECTING");
      setSttStep("REQUEST_PERMISSION");
      setSttStatusDetail("오디오 권한 요청 중...");
      appendSttLog("requesting media permission");

      let rawStream: MediaStream;
      if (sttSource === "system") {
        rawStream = await navigator.mediaDevices.getDisplayMedia({
          audio: true,
          video: true,
        });
      } else {
        rawStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      }

      const audioTracks = rawStream.getAudioTracks();
      if (audioTracks.length === 0) {
        rawStream.getTracks().forEach((t) => t.stop());
        setSttStatusText("STOPPED");
        setSttStep("NO_AUDIO_TRACK");
        setSttStatusDetail("오디오 트랙이 없습니다. 화면 공유 창에서 '탭 오디오 공유'를 켜세요.");
        appendSttLog("no audio track detected from shared source");
        return;
      }

      const audioOnlyStream = new MediaStream(audioTracks);
      const mimeCandidates = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];
      const mimeType = mimeCandidates.find((m) => MediaRecorder.isTypeSupported(m));
      const sourceLabel = sttSource === "system" ? "system_audio" : "microphone";
      const chunkWindowMs = 5000;
      let sessionStarted = false;

      streamRef.current = rawStream;
      audioTracks.forEach((track) => {
        track.onended = () => {
          if (!stopRequestedRef.current) {
            appendSttLog("audio track ended");
            requestStopAndDrain("오디오 공유가 종료되었습니다. 남은 청크를 처리한 뒤 STT를 종료합니다.");
          }
        };
      });

      const startRecorderWindow = () => {
        if (stopRequestedRef.current) {
          return;
        }
        const recorder = mimeType
          ? new MediaRecorder(audioOnlyStream, { mimeType })
          : new MediaRecorder(audioOnlyStream);
        recorderRef.current = recorder;

        recorder.onstart = () => {
          if (!sessionStarted) {
            sessionStarted = true;
            setSttRunning(true);
            setSttStatusText("RUNNING");
            setSttStep("CAPTURING");
            setSttStatusDetail(
              sttSource === "system"
                ? "시스템 오디오 캡처 중 (5초 청크, 탭 오디오 공유 필요)"
                : "마이크 캡처 중 (5초 청크)",
            );
            appendSttLog("media recorder started");
          }
        };

        recorder.ondataavailable = (event) => {
          if (!event.data || event.data.size === 0) {
            setSttStatusText("RUNNING");
            setSttStep("EMPTY_CHUNK");
            setSttStatusDetail("청크가 비어 있습니다. 오디오 신호를 확인하세요.");
            appendSttLog("empty chunk received");
            return;
          }
          const lowerType = (event.data.type || mimeType || "").toLowerCase();
          let ext = "webm";
          if (lowerType.includes("mp4")) {
            ext = "mp4";
          } else if (lowerType.includes("ogg")) {
            ext = "ogg";
          } else if (lowerType.includes("wav")) {
            ext = "wav";
          }
          const localSeq = ++chunkSeqRef.current;
          pendingChunksRef.current += 1;
          if (stopRequestedRef.current) {
            setSttStatusText("DRAINING");
            setSttStep("DRAINING");
            setSttStatusDetail(`남은 청크 처리 중... (${pendingChunksRef.current}개 대기)`);
          }
          sendQueueRef.current = sendQueueRef.current
            .then(() => sendChunk(sessionId, localSeq, event.data, `chunk.${ext}`, sourceLabel))
            .catch((queueErr) => {
              appendSttLog(`chunk #${localSeq} queue error: ${(queueErr as Error).message}`);
            })
            .finally(() => {
              pendingChunksRef.current = Math.max(0, pendingChunksRef.current - 1);
              if (stopRequestedRef.current) {
                setSttStatusText("DRAINING");
                setSttStep("DRAINING");
                setSttStatusDetail(`남은 청크 처리 중... (${pendingChunksRef.current}개 대기)`);
                finalizeStopIfDrained();
              }
            });
        };

        recorder.onerror = () => {
          appendSttLog("media recorder error");
          stopRequestedRef.current = true;
          cleanupStt("ERROR", "MediaRecorder 오류가 발생했습니다.");
        };

        recorder.onstop = () => {
          recorderRef.current = null;
          if (chunkTimerRef.current !== null) {
            window.clearTimeout(chunkTimerRef.current);
            chunkTimerRef.current = null;
          }
          if (stopRequestedRef.current) {
            setSttStatusText("DRAINING");
            setSttStep("DRAINING");
            setSttStatusDetail(`남은 청크 처리 중... (${pendingChunksRef.current}개 대기)`);
            finalizeStopIfDrained();
            return;
          }
          window.setTimeout(startRecorderWindow, 0);
        };

        recorder.start();
        chunkTimerRef.current = window.setTimeout(() => {
          if (recorder.state !== "inactive") {
            recorder.stop();
          }
        }, chunkWindowMs);
      };

      startRecorderWindow();
      appendSttLog(`recording window mode started (${chunkWindowMs}ms)`);
    } catch (err) {
      setError((err as Error).message);
      setSttStatusText("ERROR");
      setSttStep("START_FAILED");
      setSttStatusDetail(`STT 시작 실패: ${(err as Error).message}`);
      appendSttLog(`stt start failed: ${(err as Error).message}`);
      cleanupStt("STOPPED", `STT 시작 실패: ${(err as Error).message}`);
    }
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <h1>Meeting Rhythm AI (React)</h1>
        <div className="top-actions">
          <button onClick={() => void apply(() => tickAnalysis())} disabled={loading}>
            틱 / 업데이트
          </button>
          <button onClick={() => void apply(() => resetState())} disabled={loading}>
            초기화
          </button>
        </div>
      </header>

      {liveBanner ? <div className={`live-banner live-banner-${liveBanner.kind}`}>{liveBanner.text}</div> : null}

      {error ? <div className="error-box">{error}</div> : null}

      <section className="config-panel">
        <label>
          회의 목표
          <input
            value={state.meeting_goal}
            onChange={(e) => setState((s) => ({ ...s, meeting_goal: e.target.value }))}
          />
        </label>
        <label>
          초기 컨텍스트
          <textarea
            value={state.initial_context}
            onChange={(e) => setState((s) => ({ ...s, initial_context: e.target.value }))}
          />
        </label>
        <label>
          최근 전사 윈도우
          <input
            type="number"
            min={4}
            max={80}
            value={state.window_size}
            onChange={(e) => setState((s) => ({ ...s, window_size: Number(e.target.value) || 12 }))}
          />
        </label>
        <label>
          JSON 폴더 경로
          <input value={datasetFolder} onChange={(e) => setDatasetFolder(e.target.value)} placeholder="dataset/economy" />
        </label>
        <label>
          JSON 파일 업로드
          <input
            type="file"
            accept=".json,application/json"
            multiple
            onChange={(e) => setDatasetFiles(Array.from(e.target.files || []))}
          />
        </label>
        <button onClick={() => void onSaveConfig()} disabled={loading}>
          설정 저장
        </button>
        <button onClick={() => void onImportDataset()} disabled={loading}>
          JSON 폴더 로드
        </button>
        <button onClick={() => void onImportDatasetFiles()} disabled={loading || datasetFiles.length === 0}>
          JSON 파일 업로드
        </button>
      </section>
      {datasetFiles.length > 0 ? <div className="hint">선택 파일: {datasetFiles.length}개</div> : null}
      {datasetImportInfo ? <div className="hint">{datasetImportInfo}</div> : null}
      <div className="engine-checklist">
        {engineChecklist.map((e) => (
          <span key={e.name} className={e.ok ? "engine-chip engine-chip-ok" : "engine-chip engine-chip-pending"}>
            {e.name}: {e.ok ? "applied" : "pending"}
          </span>
        ))}
      </div>

      <section className="overview-strip">
        <article className="overview-card">
          <span className="overview-label">STT</span>
          <strong>{sttStatusText}</strong>
          <small>{sttStep}</small>
        </article>
        <article className="overview-card">
          <span className="overview-label">AGENDA</span>
          <strong>{activeAgendaCount}</strong>
          <small>active / {proposedAgendaCount} proposed / {closedAgendaCount} closed</small>
        </article>
        <article className="overview-card">
          <span className="overview-label">DPS</span>
          <strong>{dpsPercent}%</strong>
          <small>{closingModeActive ? "closing mode" : "discussion mode"}</small>
        </article>
        <article className="overview-card">
          <span className="overview-label">EVIDENCE</span>
          <strong>{evidenceStatus}</strong>
          <small>{state.evidence_log?.length ?? 0} logs</small>
        </article>
        <article className="overview-card">
          <span className="overview-label">FAIRTALK</span>
          <strong>{fairtalkDebug?.active_speakers ?? participantFrames.length}</strong>
          <small>strong {fairtalkDebug?.strong_count ?? 0} / soft {fairtalkDebug?.soft_count ?? 0}</small>
        </article>
        <article className="overview-card">
          <span className="overview-label">LAST TURN</span>
          <strong>{lastTranscript ? `${lastTranscript.timestamp} · ${lastTranscript.speaker}` : "none"}</strong>
          <small>{lastTranscript ? lastTranscript.text.slice(0, 42) : "전사가 아직 없습니다."}</small>
        </article>
      </section>

      <main className="workspace-grid">
        <aside className="card nav-column-card">
          <h2>Agenda Navigation</h2>
          <div className="agenda-block">
            <h3>ACTIVE</h3>
            {agendaBuckets.active.length === 0 ? (
              <div className="agenda-item agenda-item-empty">없음</div>
            ) : (
              agendaBuckets.active.map((item) => (
                <div
                  key={`active-${item.status}-${item.title}`}
                  className={item.status === "ACTIVE" ? "agenda-item agenda-item-active" : "agenda-item agenda-item-closing"}
                >
                  {item.title} {item.status === "CLOSING" ? "(CLOSING)" : ""}
                </div>
              ))
            )}
          </div>
          <div className="agenda-block">
            <h3>PROPOSED</h3>
            {agendaBuckets.proposed.length === 0 ? (
              <div className="agenda-item agenda-item-empty">없음</div>
            ) : (
              agendaBuckets.proposed.map((item) => (
                <div key={`proposed-${item.title}`} className="agenda-item">
                  {item.title}
                </div>
              ))
            )}
          </div>
          <div className="agenda-block">
            <h3>CLOSED</h3>
            {agendaBuckets.closed.length === 0 ? (
              <div className="agenda-item agenda-item-empty">없음</div>
            ) : (
              agendaBuckets.closed.map((item) => (
                <div key={`closed-${item.title}`} className="agenda-item agenda-item-closed">
                  {item.title}
                </div>
              ))
            )}
          </div>

          <div className="keyword-block">
            <b>Agenda FSM</b>
            <div className="keyword-line">active_agenda_id: {state.active_agenda_id || "(none)"}</div>
            {Object.values(state.agenda_state_map ?? {}).map((entry) => (
              <div className="keyword-line" key={entry.agenda_id}>
                [{entry.state}] {entry.title}
              </div>
            ))}
          </div>

          <div className="keyword-block">
            <b>Active Change Events</b>
            {(state.agenda_events ?? [])
              .filter((ev) => ev.type === "active_agenda_changed")
              .slice(-8)
              .map((ev, idx) => (
                <div className="keyword-line" key={`${ev.ts}-${idx}`}>
                  {ev.ts} | {ev.active_before || "(none)"} {"->"} {ev.active_after || "(none)"}
                </div>
              ))}
          </div>
        </aside>

        <section className="card live-column-card">
          <h2>Live Stream & Participation</h2>
          <div className={fixedFocus ? "focus-card focus-fixed" : "focus-card"}>
            <div className="focus-title">CURRENT AGENDA: {currentAgendaText}</div>
            <div className={coreGlow ? "core-tags core-tags-glow" : "core-tags"}>
              {kCoreTags.length === 0 ? (
                <span className="hint">K_core 태그 없음</span>
              ) : (
                kCoreTags.map((tag) => (
                  <span key={tag} className="core-tag">
                    {tag}
                  </span>
                ))
              )}
            </div>
          </div>
          <div className="stt-box">
            <div className="row">
              <select value={sttSource} onChange={(e) => setSttSource(e.target.value as "mic" | "system")}>
                <option value="system">시스템 오디오(탭 공유)</option>
                <option value="mic">마이크</option>
              </select>
              <input
                value={sttSpeaker}
                onChange={(e) => setSttSpeaker(e.target.value)}
                placeholder="STT 화자 라벨"
              />
            </div>
            <div className="row">
              <button onClick={() => void startStt()} disabled={sttRunning || loading}>
                Start STT
              </button>
              <button onClick={stopStt} disabled={!sttRunning}>
                Stop STT
              </button>
            </div>
            <p className="hint">
              상태: <strong>{sttStatusText}</strong>
            </p>
            <p className="hint">
              현재 단계: <strong>{sttStep}</strong>
            </p>
            <p className="hint">{sttStatusDetail}</p>
            <p className="hint">
              시스템 오디오 사용 시: 브라우저 공유 창에서 반드시 <strong>탭 오디오 공유</strong>를 켜세요.
            </p>
            {lastDebug ? (
              <div className="hint stt-last-debug">
                마지막 청크: #{lastDebug.chunk_id} | {lastDebug.status} | {lastDebug.duration_ms}ms |{" "}
                {formatBytes(lastDebug.bytes)}
              </div>
            ) : null}
            <div className="stt-stage-panel">
              {CLIENT_STEPS.map((step) => (
                <span
                  key={step}
                  className={step === sttStep ? "stt-stage-chip stt-stage-chip-active" : "stt-stage-chip"}
                >
                  {step}
                </span>
              ))}
            </div>

            <details className="debug-fold" open>
              <summary>STT 처리 파이프라인/로그</summary>
              <h4 className="stt-subtitle">마지막 청크 서버 처리 단계</h4>
              {lastDebug ? (
                <div className="stt-step-table-wrap">
                  <table className="stt-step-table">
                    <thead>
                      <tr>
                        <th>Step</th>
                        <th>누적(ms)</th>
                        <th>구간(ms)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {lastDebug.steps.map((step, idx) => {
                        const prev = idx > 0 ? lastDebug.steps[idx - 1].t_ms : 0;
                        return (
                          <tr key={`${lastDebug.chunk_id}-${step.step}-${idx}`}>
                            <td>{step.step}</td>
                            <td>{step.t_ms}</td>
                            <td>{step.t_ms - prev}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="hint">아직 서버 처리 step 데이터가 없습니다.</div>
              )}

              <h4 className="stt-subtitle">최근 청크 처리 기록</h4>
              <div className="stt-debug-history">
                {debugHistory.length === 0 ? (
                  <div className="hint">기록 없음</div>
                ) : (
                  debugHistory.slice(0, 8).map((d) => (
                    <div key={`h-${d.chunk_id}`} className="stt-debug-row">
                      <span>#{d.chunk_id}</span>
                      <span>{d.status}</span>
                      <span>{d.duration_ms}ms</span>
                      <span>{formatBytes(d.bytes)}</span>
                    </div>
                  ))
                )}
              </div>

              <h4 className="stt-subtitle">STT 로그</h4>
              <div className="stt-log-box">
                {sttLogs.length === 0 ? (
                  <div className="hint">STT 로그가 아직 없습니다.</div>
                ) : (
                  sttLogs.slice(-14).map((line, idx) => (
                    <div className="stt-log-line" key={`${idx}-${line}`}>
                      {line}
                    </div>
                  ))
                )}
              </div>
            </details>
          </div>

          <h3>전사 피드 (전체 {state.transcript.length}줄)</h3>
          <div className="participant-wrap">
            <div className="participant-header">
              <h3>FairTalk Participant Frames</h3>
              <span className="hint">
                strong {fairtalkDebug?.strong_count ?? 0} / soft {fairtalkDebug?.soft_count ?? 0} / active{" "}
                {fairtalkDebug?.active_speakers ?? participantFrames.length}
              </span>
            </div>
            <div className="participant-grid">
              {participantFrames.length === 0 ? (
                <div className="hint">발언 의도(P_intent) 신호가 아직 없습니다.</div>
              ) : (
                participantFrames.slice(0, 8).map((p) => (
                  <article
                    key={`participant-${p.speaker}`}
                    className={
                      p.glow === "strong"
                        ? "participant-frame participant-frame-strong"
                        : p.glow === "soft"
                          ? "participant-frame participant-frame-soft"
                          : "participant-frame"
                    }
                  >
                    <div className="participant-name">{p.speaker}</div>
                    <div className="participant-meta">
                      P_intent {p.p_intent.toFixed(2)} | glow {p.glow} | last {p.last_seen_sec.toFixed(1)}s
                    </div>
                  </article>
                ))
              )}
            </div>
            <div className="hint">침묵 자체는 트리거하지 않고, 발언 의도 신호가 있을 때만 glow를 표시합니다.</div>
          </div>
          <div className="feed">
            {state.transcript.map((item, idx) => (
              <div key={`${item.timestamp}-${idx}`} className="feed-row">
                <b>
                  {item.timestamp} · {item.speaker}
                </b>
                <span>{item.text}</span>
              </div>
            ))}
          </div>
        </section>

        <aside className="card decision-column-card">
          <h2>Decision Intelligence</h2>
          {state.analysis ? (
            <>
              <div className="cockpit-stat-row">
                <span className={`status-badge status-badge-${String(evidenceStatus).toLowerCase()}`}>
                  Evidence: {evidenceStatus}
                </span>
                <span className="cockpit-stat">Drift {state.analysis.scores.drift.score}</span>
                <span className={`status-badge drift-state-badge drift-state-${driftState.toLowerCase().replace("-", "")}`}>
                  Drift Dampener: {driftState}
                </span>
                <span
                  className={`status-badge loop-state-badge loop-state-${String(state.loop_state || "Normal").toLowerCase()}`}
                >
                  Flow Pulse: {state.loop_state || "Normal"}
                </span>
              </div>
              <div className="evidence-snippet-box">
                <b>Evidence Gate Summary</b>
                {evidenceSnippetLines.length === 0 ? (
                  <div className="hint">아직 Evidence Gate 요약이 없습니다.</div>
                ) : (
                  evidenceSnippetLines.map((line, idx) => (
                    <div className="evidence-line" key={`evi-snippet-${idx}`}>
                      {line}
                    </div>
                  ))
                )}
                <div className="evidence-log-mini">
                  {evidenceLogPreview.length === 0 ? (
                    <div className="hint">최근 evidence_log 없음</div>
                  ) : (
                    evidenceLogPreview.map((entry, idx) => (
                      <div className="evidence-log-row" key={`evi-log-${idx}`}>
                        <span className={`status-badge status-badge-${String(entry.status).toLowerCase()}`}>
                          {entry.status}
                        </span>
                        <span className="evidence-log-text">
                          {entry.claim} | v={entry.verifiability.toFixed(2)} | eqs{" "}
                          {entry.eqs === null || entry.eqs === undefined ? "-" : entry.eqs.toFixed(2)}
                        </span>
                      </div>
                    ))
                  )}
                </div>
              </div>

              <div className="cockpit-stat">
                S45 {state.drift_debug?.s45?.toFixed?.(3) ?? "0.000"} | band {state.drift_debug?.band || "Green"} |
                yellow {state.drift_debug?.yellow_seconds ?? 0}s | red {state.drift_debug?.red_seconds ?? 0}s
              </div>
              <div className="cockpit-stat">
                Novelty {state.flow_pulse_debug?.novelty_rate_3m?.toFixed?.(3) ?? "1.000"} | ArgNovelty{" "}
                {state.flow_pulse_debug?.arg_novelty?.toFixed?.(3) ?? "1.000"} | ΔDPS{" "}
                {state.flow_pulse_debug?.delta_dps?.toFixed?.(3) ?? "0.000"} | AnchorRatio{" "}
                {state.flow_pulse_debug?.anchor_ratio?.toFixed?.(3) ?? "0.000"}
              </div>

              <div className="dps-wrap">
                <div className="dps-header">
                  <b>DPS</b>
                  <span>{dpsPercent}%</span>
                </div>
                <div className="dps-bar">
                  <div className="dps-fill" style={{ width: `${dpsPercent}%` }} />
                </div>
                {state.dps_breakdown ? (
                  <div className="cockpit-stat">
                    O {state.dps_breakdown.option_coverage.toFixed(2)} / C {state.dps_breakdown.constraint_coverage.toFixed(2)} /
                    E {state.dps_breakdown.evidence_coverage.toFixed(2)} / T {state.dps_breakdown.tradeoff_coverage.toFixed(2)} /
                    R {state.dps_breakdown.closing_readiness.toFixed(2)}
                  </div>
                ) : null}
              </div>

              <div className={closingModeActive ? "closing-panel closing-panel-active" : "closing-panel"}>
                <div className="closing-title">
                  Closing UI: {closingModeActive ? "ACTIVE" : "WAITING TRIGGER"}
                </div>
                <div className="cockpit-stat">
                  stance {state.decision_lock_debug?.stance_convergence?.toFixed?.(2) ?? "0.00"} | elapsed{" "}
                  {Math.round(state.decision_lock_debug?.elapsed_sec ?? 0)}s | trigger(stance/stag/timebox)=
                  {state.decision_lock_debug?.trigger_stance ? "1" : "0"}/
                  {state.decision_lock_debug?.trigger_stagnation ? "1" : "0"}/
                  {state.decision_lock_debug?.trigger_timebox ? "1" : "0"}
                </div>
                <div className="cockpit-actions">
                  <button onClick={() => void onActionVote()} disabled={loading || !closingModeActive}>
                  vote
                  </button>
                  <button onClick={() => void onActionSummary()} disabled={loading || !closingModeActive}>
                  summary
                  </button>
                  <button onClick={() => void onActionDecide()} disabled={loading || !closingModeActive}>
                  decide
                  </button>
                </div>
              </div>

              {state.analysis.intervention.level === "L2" ? (
                <div className="l2-prompt-box">L2 Shared Prompt: {state.analysis.intervention.banner_text}</div>
              ) : null}
              <div className="cockpit-stat">
                Reco Trigger A/B/C = {recoDebug?.trigger_a_info_seeking ? "1" : "0"}/
                {recoDebug?.trigger_b_evidence_weak ? "1" : "0"}/{recoDebug?.trigger_c_slot_fulfillment ? "1" : "0"} |
                info60s {recoDebug?.info_signal_count_60s ?? 0} | cards R1 {recoDebug?.shown_r1 ?? 0}, R2{" "}
                {recoDebug?.shown_r2 ?? 0}
              </div>

              <div className="recommend-grid">
                {state.analysis.recommendations.r1_resources.slice(0, 2).map((r) => (
                  <article key={`r1-${r.url}`} className="recommend-card">
                    <div className="recommend-badge">R1</div>
                    <a href={r.url} target="_blank" rel="noreferrer">
                      {r.title}
                    </a>
                    <p>{r.reason}</p>
                  </article>
                ))}
                {!reduceFacets
                  ? state.analysis.recommendations.r2_options.slice(0, 2).map((o) => (
                      <article key={`r2-${o.option}`} className="recommend-card">
                        <div className="recommend-badge">R2</div>
                        <b>{o.option}</b>
                        <p>{o.evidence_note}</p>
                    </article>
                  ))
                  : (
                    <article className="recommend-card">
                      <div className="recommend-badge">R2</div>
                      <p>Red/Re-orient 상태로 Facet 노출을 축소하고 K_core 중심으로 고정합니다.</p>
                    </article>
                  )}
                {state.analysis.recommendations.r1_resources.length === 0 &&
                state.analysis.recommendations.r2_options.length === 0 ? (
                  <article className="recommend-card">
                    <div className="recommend-badge">Reco</div>
                    <p>현재 트리거 조건(A/B/C)이 충족되지 않아 추천 카드를 숨깁니다.</p>
                  </article>
                ) : null}
              </div>

              <h3>산출물</h3>
              <div className="artifact-actions">
                {(["meeting_summary", "decision_results", "action_items", "evidence_log"] as ArtifactKind[]).map(
                  (kind) => (
                    <button key={kind} onClick={() => void apply(() => createArtifact(kind))} disabled={loading}>
                      {kind}
                    </button>
                  ),
                )}
              </div>
            </>
          ) : (
            <p>틱/업데이트를 실행하면 표시됩니다.</p>
          )}
        </aside>
      </main>

      <section className="card outputs-stage">
        <h2>Live Deliverables</h2>
        <div className="artifact-grid">
          {Object.values(state.artifacts).map((artifact) => (
            <article key={artifact.kind} className="artifact-card">
              <h3>{artifact.title}</h3>
              <pre>{artifact.markdown}</pre>
              <ul>
                {artifact.bullets.map((b, i) => (
                  <li key={`${artifact.kind}-${i}`}>{b}</li>
                ))}
              </ul>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}

export default App;
