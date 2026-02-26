import { useCallback, useEffect, useRef, useState } from "react";
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
} from "./api";
import type { MeetingState, SttDebug } from "./types";

const EMPTY_STATE: MeetingState = {
  meeting_goal: "",
  initial_context: "",
  window_size: 12,
  transcript: [],
  agenda_stack: [],
  llm_enabled: false,
  analysis: null,
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
] as const;

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
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
  const [sttStep, setSttStep] = useState<(typeof CLIENT_STEPS)[number]>("IDLE");
  const [sttLogs, setSttLogs] = useState<string[]>([]);
  const [lastDebug, setLastDebug] = useState<SttDebug | null>(null);
  const [debugHistory, setDebugHistory] = useState<SttDebug[]>([]);

  const [datasetFolder, setDatasetFolder] = useState("dataset/economy");
  const [datasetFiles, setDatasetFiles] = useState<File[]>([]);
  const [datasetImportInfo, setDatasetImportInfo] = useState("");
  const [meetingGoalDraft, setMeetingGoalDraft] = useState("");
  const [meetingGoalDirty, setMeetingGoalDirty] = useState(false);

  const [llmChecking, setLlmChecking] = useState(false);
  const [llmPingMessage, setLlmPingMessage] = useState("");
  const [llmPingOk, setLlmPingOk] = useState<boolean | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const stopRequestedRef = useRef(false);
  const chunkSeqRef = useRef(0);
  const chunkTimerRef = useRef<number | null>(null);
  const sendQueueRef = useRef<Promise<void>>(Promise.resolve());
  const sttSessionRef = useRef(0);
  const pendingChunksRef = useRef(0);
  const stopReasonRef = useRef("사용자가 STT를 중지했습니다.");

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
    }, 1200);
    return () => window.clearInterval(id);
  }, [loadState]);

  useEffect(() => {
    const id = window.setInterval(() => {
      void refreshLlmStatus();
    }, 3000);
    return () => window.clearInterval(id);
  }, [refreshLlmStatus]);

  useEffect(() => {
    if (!meetingGoalDirty) {
      setMeetingGoalDraft(state.meeting_goal || "");
    }
  }, [state.meeting_goal, meetingGoalDirty]);

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

  const onSaveConfig = async () => {
    setLoading(true);
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
      setLoading(false);
    }
  };

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
      setMeetingGoalDraft(res.state.meeting_goal || "");
      setMeetingGoalDirty(false);
      setError("");
      const d = res.import_debug;
      setDatasetImportInfo(
        `loaded=${d.added}, files=${d.files_parsed}/${d.files_scanned}, skipped=${d.files_skipped}, ticked=${d.ticked ? "yes" : "no"}${
          d.meeting_goal_applied ? `, goal=${d.meeting_goal}` : ""
        }`,
      );
      if (d.warning) setError(d.warning);
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
      const res = await importJsonFiles({ files: datasetFiles, reset_state: true, auto_tick: true });
      setState(res.state);
      setMeetingGoalDraft(res.state.meeting_goal || "");
      setMeetingGoalDirty(false);
      setError("");
      const d = res.import_debug;
      setDatasetImportInfo(
        `uploaded=${datasetFiles.length}, loaded=${d.added}, files=${d.files_parsed}/${d.files_scanned}, skipped=${d.files_skipped}, ticked=${d.ticked ? "yes" : "no"}${
          d.meeting_goal_applied ? `, goal=${d.meeting_goal}` : ""
        }`,
      );
      if (d.warning) setError(d.warning);
    } catch (err) {
      setError((err as Error).message);
      setDatasetImportInfo("");
    } finally {
      setLoading(false);
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
    setSttStep(status === "STOPPED" ? "IDLE" : "ERROR");
  };

  const finalizeStopIfDrained = useCallback(() => {
    const recInactive = !recorderRef.current || recorderRef.current.state === "inactive";
    if (!stopRequestedRef.current || !recInactive || pendingChunksRef.current > 0) return;
    cleanupStt("STOPPED", stopReasonRef.current);
  }, []);

  const sendChunk = async (sessionId: number, localSeq: number, blob: Blob, filename: string, source: string) => {
    if (sessionId !== sttSessionRef.current) return;
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
      const res = await transcribeChunk({ blob, filename, speaker: sttSpeaker || "시스템오디오", source });
      if (sessionId !== sttSessionRef.current) return;
      setState(res.state);
      setLastDebug(res.stt_debug);
      setDebugHistory((prev) => [res.stt_debug, ...prev].slice(0, 20));
      if (!stopRequestedRef.current) {
        setSttStep(res.stt_debug.status === "ok" ? "SERVER_TRANSCRIBED" : res.stt_debug.status === "empty" ? "SERVER_NO_SPEECH" : "ERROR");
        setSttStatusText(res.stt_debug.status === "error" ? "ERROR" : "RUNNING");
      }
      if (res.stt_debug.error) appendSttLog(`chunk #${res.stt_debug.chunk_id} error: ${res.stt_debug.error}`);
      if (res.stt_debug.transcript_preview) appendSttLog(`chunk #${res.stt_debug.chunk_id} text: ${res.stt_debug.transcript_preview}`);
    } catch (err) {
      setError((err as Error).message);
      setSttStatusText("ERROR");
      setSttStep("UPLOAD_FAILED");
      setSttStatusDetail(`청크 업로드 실패: ${(err as Error).message}`);
      appendSttLog(`chunk #${localSeq} failed: ${(err as Error).message}`);
    }
  };

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

  const stopStt = () => requestStopAndDrain("사용자가 STT를 중지했습니다.");

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
        setSttStep("NO_AUDIO_TRACK");
        setSttStatusDetail("오디오 트랙이 없습니다. 화면 공유 시 탭 오디오 공유를 켜세요.");
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
            requestStopAndDrain("오디오 공유가 종료되었습니다. 남은 청크를 처리한 뒤 STT를 종료합니다.");
          }
        };
      });

      const startRecorderWindow = () => {
        if (stopRequestedRef.current) return;
        const recorder = mimeType ? new MediaRecorder(audioOnlyStream, { mimeType }) : new MediaRecorder(audioOnlyStream);
        recorderRef.current = recorder;

        recorder.onstart = () => {
          if (sessionStarted) return;
          sessionStarted = true;
          setSttRunning(true);
          setSttStatusText("RUNNING");
          setSttStep("CAPTURING");
          setSttStatusDetail(sttSource === "system" ? "시스템 오디오 캡처 중 (5초 청크)" : "마이크 캡처 중 (5초 청크)");
        };

        recorder.ondataavailable = (event) => {
          if (!event.data || event.data.size === 0) {
            setSttStep("EMPTY_CHUNK");
            setSttStatusDetail("청크가 비어 있습니다.");
            return;
          }
          const lowerType = (event.data.type || mimeType || "").toLowerCase();
          let ext = "webm";
          if (lowerType.includes("mp4")) ext = "mp4";
          else if (lowerType.includes("ogg")) ext = "ogg";
          else if (lowerType.includes("wav")) ext = "wav";

          const localSeq = ++chunkSeqRef.current;
          pendingChunksRef.current += 1;

          sendQueueRef.current = sendQueueRef.current
            .then(() => sendChunk(sessionId, localSeq, event.data, `chunk.${ext}`, sourceLabel))
            .finally(() => {
              pendingChunksRef.current = Math.max(0, pendingChunksRef.current - 1);
              if (stopRequestedRef.current) finalizeStopIfDrained();
            });
        };

        recorder.onerror = () => {
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
          if (recorder.state !== "inactive") recorder.stop();
        }, chunkWindowMs);
      };

      startRecorderWindow();
    } catch (err) {
      setError((err as Error).message);
      setSttStatusText("ERROR");
      setSttStep("START_FAILED");
      setSttStatusDetail(`STT 시작 실패: ${(err as Error).message}`);
      cleanupStt("STOPPED", `STT 시작 실패: ${(err as Error).message}`);
    }
  };

  const llmEnabled = Boolean(state.llm_enabled);
  const currentAgenda = state.analysis?.agenda?.active?.title || "(none)";
  const agendaOutcomes = state.analysis?.agenda_outcomes || [];
  const evidenceClaims = state.analysis?.evidence_gate?.claims || [];
  const transcript = state.transcript || [];

  return (
    <div className="app-shell">
      <header className="topbar">
        <h1>Meeting Rhythm AI (Lite)</h1>
        <div className="top-actions">
          <button onClick={() => void apply(() => tickAnalysis())} disabled={loading || !llmEnabled}>
            틱 / 업데이트
          </button>
          <button onClick={() => void apply(() => resetState())} disabled={loading}>
            초기화
          </button>
        </div>
      </header>

      {error ? <div className="error-box">{error}</div> : null}

      <section className="card llm-status-card">
        <h2>LLM 연결 상태</h2>
        <div className="llm-status-grid">
          <div>llm_enabled: {String(llmEnabled)}</div>
          <div>connected: {String(Boolean(state.llm_status?.connected))}</div>
          <div>provider: {state.llm_status?.provider || "-"}</div>
          <div>model: {state.llm_status?.model || "-"}</div>
          <div>mode: {state.llm_status?.mode || "-"}</div>
          <div>last_error: {state.llm_status?.last_error || "-"}</div>
        </div>
        {llmPingMessage ? (
          <div className={llmPingOk ? "llm-ping-result llm-ping-ok" : "llm-ping-result llm-ping-fail"}>Ping 결과: {llmPingMessage}</div>
        ) : null}
        <div className="llm-status-actions">
          <button onClick={() => void onConnectLlm()} disabled={llmChecking}>{llmChecking ? "연결 중..." : "LLM 연결"}</button>
          <button onClick={() => void onDisconnectLlm()} disabled={llmChecking || !llmEnabled}>연결 해제</button>
          <button onClick={() => void onPingLlm()} disabled={llmChecking}>{llmChecking ? "확인 중..." : "LLM 연결 테스트"}</button>
        </div>
      </section>

      <section className="config-panel">
        <label>
          회의 목표
          <input
            value={meetingGoalDraft}
            onChange={(e) => {
              setMeetingGoalDraft(e.target.value);
              setMeetingGoalDirty(true);
            }}
          />
        </label>
        <div className="hint">JSON 업로드 시 metadata.topic이 회의 목표로 자동 적용됩니다.</div>
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
          <input type="file" accept=".json,application/json" multiple onChange={(e) => setDatasetFiles(Array.from(e.target.files || []))} />
        </label>
        <button onClick={() => void onSaveConfig()} disabled={loading}>설정 저장</button>
        <button onClick={() => void onImportDataset()} disabled={loading}>JSON 폴더 로드</button>
        <button onClick={() => void onImportDatasetFiles()} disabled={loading || datasetFiles.length === 0}>JSON 파일 업로드</button>
      </section>
      {datasetFiles.length > 0 ? <div className="hint">선택 파일: {datasetFiles.length}개</div> : null}
      {datasetImportInfo ? <div className="hint">{datasetImportInfo}</div> : null}

      <main className="main-grid">
        <section className="card center-column-card">
          <h2>Transcript</h2>
          <div className="hint">current agenda: {currentAgenda}</div>
          <div className="transcript-feed transcript-feed-main">
            {transcript.length === 0 ? (
              <div className="hint">전사 없음</div>
            ) : (
              transcript.slice(-200).map((u, idx) => (
                <article key={`${u.timestamp}-${idx}`} className="utterance">
                  <div className="utterance-meta">[{u.timestamp}] {u.speaker}</div>
                  <div className="utterance-text">{u.text}</div>
                </article>
              ))
            )}
          </div>
        </section>

        <aside className="card decision-column-card">
          <h2>Agenda Analysis</h2>
          <div className="hint">후보 아젠다: {(state.analysis?.agenda?.candidates || []).map((c) => c.title).join(" | ") || "없음"}</div>

          <section className="agenda-outcome-board">
            <div className="agenda-outcome-header">
              <b>Agenda Outcomes</b>
              <span className="hint">{agendaOutcomes.length} agendas</span>
            </div>
            {agendaOutcomes.length === 0 ? (
              <div className="hint">아직 아젠다 분석 결과가 없습니다.</div>
            ) : (
              <div className="agenda-outcome-grid">
                {agendaOutcomes.map((ag, idx) => (
                  <article key={`${ag.agenda_title}-${idx}`} className="agenda-outcome-card">
                    <div className="agenda-outcome-title-row">
                      <div className="agenda-outcome-title">{ag.agenda_title || "(untitled)"}</div>
                    </div>
                    <div className="agenda-outcome-key">핵심 발언: {ag.key_utterances?.join(" | ") || "없음"}</div>
                    <div className="agenda-outcome-summary">요약: {ag.summary || "없음"}</div>
                    <div className="agenda-outcome-summary">키워드: {ag.agenda_keywords?.join(", ") || "없음"}</div>

                    <div className="agenda-outcome-block">
                      <div className="agenda-outcome-block-title">의사결정 결과 ({ag.decision_results.length})</div>
                      {ag.decision_results.length === 0 ? (
                        <div className="hint">없음</div>
                      ) : (
                        ag.decision_results.map((d, didx) => (
                          <div key={`${ag.agenda_title}-d-${didx}`} className="agenda-outcome-item">
                            <div className="agenda-outcome-line">의견 요약: {d.opinions.join(" | ") || "-"}</div>
                            <div className="agenda-outcome-line">결론: {d.conclusion || "-"}</div>
                          </div>
                        ))
                      )}
                    </div>

                    <div className="agenda-outcome-block">
                      <div className="agenda-outcome-block-title">액션 아이템 ({ag.action_items.length})</div>
                      {ag.action_items.length === 0 ? (
                        <div className="hint">없음</div>
                      ) : (
                        ag.action_items.map((a, aidx) => (
                          <div key={`${ag.agenda_title}-a-${aidx}`} className="agenda-outcome-item">
                            <div className="agenda-outcome-item-head">{a.item} | owner: {a.owner || "-"} | due: {a.due || "-"}</div>
                            {a.reasons.map((r, ridx) => (
                              <div key={`${ag.agenda_title}-a-${aidx}-r-${ridx}`} className="agenda-reason-row">
                                {r.timestamp ? `[${r.timestamp}] ` : ""}{r.speaker ? `${r.speaker}: ` : ""}{r.quote || "-"} / 이유: {r.why || "-"}
                              </div>
                            ))}
                          </div>
                        ))
                      )}
                    </div>
                  </article>
                ))}
              </div>
            )}
          </section>

          <section className="evidence-snippet-box">
            <b>Evidence Claims</b>
            {evidenceClaims.length === 0 ? (
              <div className="hint">없음</div>
            ) : (
              evidenceClaims.slice(0, 8).map((c, idx) => (
                <div className="evidence-line" key={`ev-${idx}`}>
                  {c.claim} | v={Number(c.verifiability || 0).toFixed(2)} | {c.note || ""}
                </div>
              ))
            )}
          </section>
        </aside>
      </main>

      <section className="card stt-card">
        <h2>STT Stream Monitor</h2>
        <div className="stt-controls">
          <select value={sttSource} onChange={(e) => setSttSource(e.target.value as "mic" | "system")}> 
            <option value="system">시스템 오디오</option>
            <option value="mic">마이크</option>
          </select>
          <input value={sttSpeaker} onChange={(e) => setSttSpeaker(e.target.value)} placeholder="speaker label" />
          <button onClick={() => void startStt()} disabled={sttRunning}>Start STT</button>
          <button onClick={stopStt} disabled={!sttRunning}>Stop STT</button>
        </div>
        <p className="hint">상태: <strong>{sttStatusText}</strong> | 단계: <strong>{sttStep}</strong></p>
        <p className="hint">{sttStatusDetail}</p>
        {lastDebug ? (
          <div className="hint">마지막 청크: #{lastDebug.chunk_id} | {lastDebug.status} | {lastDebug.duration_ms}ms | {formatBytes(lastDebug.bytes)}</div>
        ) : null}
        <div className="stt-stage-panel">
          {CLIENT_STEPS.map((step) => (
            <span key={step} className={step === sttStep ? "stt-stage-chip stt-stage-chip-active" : "stt-stage-chip"}>{step}</span>
          ))}
        </div>

        <details className="debug-fold">
          <summary>STT 로그</summary>
          <div className="stt-debug-history">
            {debugHistory.length === 0 ? <div className="hint">기록 없음</div> : debugHistory.slice(0, 10).map((d) => (
              <div key={`h-${d.chunk_id}`} className="stt-debug-row">
                <span>#{d.chunk_id}</span>
                <span>{d.status}</span>
                <span>{d.duration_ms}ms</span>
                <span>{formatBytes(d.bytes)}</span>
              </div>
            ))}
          </div>
          <div className="stt-log-box">
            {sttLogs.length === 0 ? <div className="hint">STT 로그가 아직 없습니다.</div> : sttLogs.slice(-20).map((line, idx) => (
              <div className="stt-log-line" key={`${idx}-${line}`}>{line}</div>
            ))}
          </div>
        </details>
      </section>
    </div>
  );
}

export default App;
