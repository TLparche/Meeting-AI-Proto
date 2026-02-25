import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createArtifact, getState, resetState, saveConfig, tickAnalysis, transcribeChunk } from "./api";
import type { AgendaStatus, ArtifactKind, MeetingState, SttDebug } from "./types";

const EMPTY_STATE: MeetingState = {
  meeting_goal: "",
  initial_context: "",
  window_size: 12,
  transcript: [],
  agenda_stack: [],
  analysis: null,
  artifacts: {},
};

const STATUS_ORDER: AgendaStatus[] = ["PROPOSED", "ACTIVE", "CLOSING", "CLOSED"];
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

  useEffect(() => {
    void loadState();
  }, [loadState]);

  useEffect(() => {
    const id = window.setInterval(() => {
      void loadState();
    }, 1200);
    return () => window.clearInterval(id);
  }, [loadState]);

  const groupedAgenda = useMemo(() => {
    const map = new Map<AgendaStatus, string[]>();
    STATUS_ORDER.forEach((k) => map.set(k, []));
    state.agenda_stack.forEach((item) => {
      const existing = map.get(item.status) ?? [];
      existing.push(item.title);
      map.set(item.status, existing);
    });
    return map;
  }, [state.agenda_stack]);

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
        <button onClick={() => void onSaveConfig()} disabled={loading}>
          설정 저장
        </button>
      </section>

      <main className="grid">
        <aside className="card">
          <h2>아젠다 스택</h2>
          {STATUS_ORDER.map((status) => (
            <div key={status} className="agenda-block">
              <h3>{status}</h3>
              {(groupedAgenda.get(status) ?? []).map((title) => (
                <div key={`${status}-${title}`} className="agenda-item">
                  {title}
                </div>
              ))}
            </div>
          ))}
        </aside>

        <section className="card">
          <h2>Live STT 전사</h2>
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
          </div>

          <h3>실시간 전사 피드</h3>
          <div className="feed">
            {state.transcript.slice(-100).map((item, idx) => (
              <div key={`${item.timestamp}-${idx}`} className="feed-row">
                <b>
                  {item.timestamp} · {item.speaker}
                </b>
                <span>{item.text}</span>
              </div>
            ))}
          </div>
        </section>

        <aside className="card">
          <h2>의사결정 콕핏</h2>
          {state.analysis ? (
            <>
              <p>근거 상태: {state.analysis.evidence_gate.status}</p>
              <p>DPS: {state.analysis.scores.dps.score}</p>
              <p>Drift: {state.analysis.scores.drift.score}</p>
              <p>Stagnation: {state.analysis.scores.stagnation.score}</p>

              <h3>Keyword Engine</h3>
              <p>Object Focus: {state.analysis.keywords.summary.object_focus || "없음"}</p>
              <p>
                Core/Facet: {state.analysis.keywords.summary.core_count} / {state.analysis.keywords.summary.facet_count}
              </p>

              <div className="keyword-block">
                <b>K_core</b>
                <div className="keyword-line">
                  OBJECT: {state.analysis.keywords.k_core.object.join(", ") || "없음"}
                </div>
                <div className="keyword-line">
                  CONSTRAINT: {state.analysis.keywords.k_core.constraints.join(", ") || "없음"}
                </div>
                <div className="keyword-line">
                  CRITERION: {state.analysis.keywords.k_core.criteria.join(", ") || "없음"}
                </div>
              </div>

              <div className="keyword-block">
                <b>K_facet</b>
                <div className="keyword-line">
                  OPTION: {state.analysis.keywords.k_facet.options.join(", ") || "없음"}
                </div>
                <div className="keyword-line">
                  EVIDENCE: {state.analysis.keywords.k_facet.evidence.join(", ") || "없음"}
                </div>
                <div className="keyword-line">
                  ACTION: {state.analysis.keywords.k_facet.actions.join(", ") || "없음"}
                </div>
              </div>

              <div className="keyword-block">
                <b>Keyword Items (K1~K6)</b>
                <div className="keyword-table-wrap">
                  <table className="keyword-table">
                    <thead>
                      <tr>
                        <th>Keyword</th>
                        <th>Type</th>
                        <th>Score</th>
                        <th>Seen</th>
                      </tr>
                    </thead>
                    <tbody>
                      {state.analysis.keywords.items.slice(0, 12).map((item) => (
                        <tr key={`${item.keyword}-${item.type}`}>
                          <td>{item.keyword}</td>
                          <td>{item.type}</td>
                          <td>{item.score.toFixed(2)}</td>
                          <td>{item.first_seen || "-"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="keyword-block">
                <b>Pipeline</b>
                <div className="keyword-line">
                  Candidates: {state.analysis.keywords.pipeline.candidates.length} (Top 40)
                </div>
                <div className="keyword-line">
                  Diversity Boost:{" "}
                  {state.analysis.keywords.pipeline.final_selection.diversity_boost_applied ? "ON" : "OFF"}
                </div>
                <div className="keyword-line">
                  Selected Core: {state.analysis.keywords.pipeline.final_selection.selected_core.join(", ") || "없음"}
                </div>
              </div>

              <h3>R1 리소스</h3>
              <ul>
                {state.analysis.recommendations.r1_resources.map((r) => (
                  <li key={r.url}>
                    <a href={r.url} target="_blank" rel="noreferrer">
                      {r.title}
                    </a>
                  </li>
                ))}
              </ul>
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

      <section className="card">
        <h2>생성된 결과</h2>
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
