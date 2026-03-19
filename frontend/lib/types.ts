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
  last_raw_preview?: string;
  last_finish_reason?: string;
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
  turn_id?: number;
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
  agenda_id?: string;
  agenda_title: string;
  agenda_state?: string;
  flow_type?: string;
  key_utterances: string[];
  agenda_summary_items?: string[];
  summary: string;
  summary_references?: AgendaActionReason[];
  agenda_keywords: string[];
  opinion_groups?: Array<{
    type?: "proposal" | "concern" | "question" | "agree" | "disagree" | "info" | string;
    summary?: string;
    evidence_turn_ids?: number[];
  }>;
  decision_results: AgendaDecisionDetail[];
  action_items: AgendaActionItemDetail[];
  start_turn_id?: number;
  end_turn_id?: number;
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
  llm_io_logs?: Array<{
    seq?: number;
    at?: string;
    direction?: "request" | "response" | "error" | string;
    stage?: string;
    payload?: string;
    meta?: Record<string, unknown>;
  }>;
  replay?: {
    queued_total?: number;
    queued_cursor?: number;
    queued_remaining?: number;
    done?: boolean;
    source?: string;
    loaded_at?: string;
  };
  analysis_runtime?: {
    tick_mode?: "full_context" | "full_document" | "windowed";
    transcript_count?: number;
    llm_window_turns?: number;
    engine_window_turns?: number;
    control_plane_source?: string;
    control_plane_reason?: string;
    used_local_fallback?: boolean;
    title_refine_attempts?: number;
    title_refine_success?: number;
    last_llm_json_available?: boolean;
    last_llm_json_at?: string;
    llm_io_count?: number;
    analysis_worker?: {
      inflight?: boolean;
      queued?: number;
      queued_logical?: number;
      queued_observed?: number;
      last_enqueued_id?: number;
      last_started_id?: number;
      last_done_id?: number;
      last_enqueued_at?: string;
      last_started_at?: string;
      last_done_at?: string;
      last_error?: string;
    };
  };
  analysis: AnalysisOutput | null;
}

export interface LastLlmJsonResponse {
  ok: boolean;
  received_at?: string;
  has_json: boolean;
  json: Record<string, unknown>;
}

export interface AgendaMarkdownExportResponse {
  ok: boolean;
  filename: string;
  agenda_count: number;
  transcript_count: number;
  markdown: string;
}

export interface AgendaSnapshotExportResponse {
  ok: boolean;
  filename: string;
  agenda_count: number;
  transcript_count: number;
  snapshot: Record<string, unknown>;
}

export interface AgendaSnapshotImportResponse {
  ok: boolean;
  state: MeetingState;
  import_debug: {
    filename: string;
    meeting_goal: string;
    transcript_count: number;
    agenda_count: number;
    reset_state: boolean;
  };
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
    analysis_mode?: "full_context_once" | "full_document_once" | "none";
    meeting_goal_applied?: boolean;
    warning?: string;
    file_stats: Array<{ file: string; rows: number }>;
    parse_errors?: Array<{ file: string; error: string }>;
  };
}

export interface ReplayImportResponse {
  state: MeetingState;
  replay_debug: {
    queued_total: number;
    queued_cursor: number;
    queued_remaining: number;
    done: boolean;
    source?: string;
    loaded_at?: string;
    files_scanned: number;
    files_parsed: number;
    files_skipped: number;
    meeting_goal_applied?: boolean;
    warning?: string;
    file_stats: Array<{ file: string; rows: number }>;
    parse_errors?: Array<{ file: string; error: string }>;
  };
}

export interface ReplayStepResponse {
  state: MeetingState;
  replay_debug: {
    added: number;
    requested: number;
    analyzed: boolean;
    queued_task_id?: number;
    queue_error?: string;
    deferred?: boolean;
    queued_total: number;
    queued_cursor: number;
    queued_remaining: number;
    done: boolean;
    warning?: string;
  };
}
