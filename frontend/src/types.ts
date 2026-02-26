export interface TranscriptUtterance {
  speaker: string;
  text: string;
  timestamp: string;
}

export interface AgendaItem {
  title: string;
  status: "PROPOSED" | "ACTIVE" | "CLOSING" | "CLOSED";
}

export interface LlmStatus {
  provider: string;
  model: string;
  base_url: string;
  mode: "mock" | "live";
  api_key_present: boolean;
  connected: boolean;
  note: string;
  request_count?: number;
  success_count?: number;
  error_count?: number;
  last_operation?: string;
  last_request_at?: string;
  last_success_at?: string;
  last_error?: string;
  last_error_at?: string;
}

export interface LlmPingResponse {
  result: {
    ok: boolean;
    message: string;
    mode: "mock" | "live";
    response_preview?: Record<string, unknown>;
  };
  llm_status: LlmStatus;
}

export interface LlmConnectResponse {
  enabled: boolean;
  result?: {
    ok: boolean;
    message: string;
    mode: "mock" | "live";
    response_preview?: Record<string, unknown>;
  };
  llm_status: LlmStatus;
  state: MeetingState;
}

export interface AgendaActionReason {
  speaker: string;
  timestamp: string;
  quote: string;
  why: string;
}

export interface AgendaActionItemDetail {
  item: string;
  owner: string;
  due: string;
  reasons: AgendaActionReason[];
}

export interface AgendaDecisionDetail {
  opinions: string[];
  conclusion: string;
}

export interface AgendaOutcomeDetail {
  agenda_title: string;
  key_utterances: string[];
  summary: string;
  agenda_keywords: string[];
  decision_results: AgendaDecisionDetail[];
  action_items: AgendaActionItemDetail[];
}

export interface AnalysisOutput {
  agenda: {
    active: { title: string; confidence: number };
    candidates: Array<{ title: string; confidence: number }>;
  };
  agenda_outcomes: AgendaOutcomeDetail[];
  evidence_gate: {
    claims: Array<{ claim: string; verifiability: number; note: string }>;
  };
}

export interface MeetingState {
  meeting_goal: string;
  initial_context: string;
  window_size: number;
  transcript: TranscriptUtterance[];
  agenda_stack: AgendaItem[];
  llm_enabled?: boolean;
  llm_status?: LlmStatus;
  analysis_runtime?: {
    tick_mode?: "full_context" | "windowed";
    transcript_count?: number;
    llm_window_turns?: number;
    engine_window_turns?: number;
    control_plane_source?: string;
    control_plane_reason?: string;
    used_local_fallback?: boolean;
  };
  analysis: AnalysisOutput | null;
}

export interface SttStepMark {
  step: string;
  t_ms: number;
}

export interface SttDebug {
  chunk_id: number;
  status: "ok" | "empty" | "error";
  source: string;
  speaker: string;
  filename: string;
  bytes: number;
  steps: SttStepMark[];
  duration_ms: number;
  transcript_chars: number;
  transcript_preview: string;
  error?: string;
}

export interface SttChunkResponse {
  state: MeetingState;
  stt_debug: SttDebug;
}

export interface ImportJsonDirResponse {
  state: MeetingState;
  import_debug: {
    folder: string;
    files_scanned: number;
    files_parsed: number;
    files_skipped: number;
    rows_loaded: number;
    meeting_goal?: string;
    added: number;
    reset_state: boolean;
    auto_tick: boolean;
    ticked: boolean;
    analysis_mode?: "full_context_once" | "none";
    meeting_goal_applied?: boolean;
    warning?: string;
    file_stats: Array<{ file: string; rows: number }>;
  };
}
