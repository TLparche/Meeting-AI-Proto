export type AgendaStatus = "PROPOSED" | "ACTIVE" | "CLOSING" | "CLOSED";
export type ArtifactKind =
  | "meeting_summary"
  | "decision_results"
  | "action_items"
  | "evidence_log";

export interface TranscriptUtterance {
  speaker: string;
  text: string;
  timestamp: string;
}

export interface AgendaItem {
  title: string;
  status: AgendaStatus;
}

export interface AgendaTrackerCandidate {
  title: string;
  status: "PROPOSED";
  confidence: number;
  created_at: string;
  reasons: string[];
  sub_issue_promoted: boolean;
  signals: {
    sim60: number;
    sim120: number;
    duration120_sec: number;
    intent_count_60s: number;
    speakers_60s: number;
    object_slots: number;
    support_slots: number;
  };
}

export interface AgendaVectorEntry {
  updated_at: string;
  sample_count: number;
  terms: Array<{ term: string; weight: number }>;
}

export interface AgendaStateEntry {
  agenda_id: string;
  title: string;
  state: "PROPOSED" | "ACTIVE" | "CLOSING" | "CLOSED";
  confidence: number;
  created_at: string;
  updated_at: string;
  source?: string;
}

export interface AgendaEvent {
  ts: string;
  type: string;
  agenda_id: string;
  title: string;
  prev_state?: string | null;
  next_state?: string | null;
  reason: string;
  active_before?: string | null;
  active_after?: string | null;
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

export interface ArtifactOutput {
  kind: ArtifactKind;
  title: string;
  markdown: string;
  bullets: string[];
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
  decision: string;
  opinions: string[];
  conclusion: string;
}

export interface AgendaOutcomeDetail {
  agenda_title: string;
  agenda_state: "PROPOSED" | "ACTIVE" | "CLOSING" | "CLOSED";
  flow_type: string;
  key_utterances: string[];
  summary: string;
  decision_results: AgendaDecisionDetail[];
  action_items: AgendaActionItemDetail[];
}

export interface AnalysisOutput {
  agenda: {
    active: { title: string; status: "ACTIVE" | "CLOSING" | "CLOSED"; confidence: number };
    candidates: Array<{ title: string; confidence: number }>;
  };
  agenda_outcomes: AgendaOutcomeDetail[];
  keywords: {
    taxonomy: {
      K1_OBJECT: string;
      K2_OPTION: string;
      K3_CONSTRAINT: string;
      K4_CRITERION: string;
      K5_EVIDENCE: string;
      K6_ACTION: string;
    };
    k_core: { object: string[]; constraints: string[]; criteria: string[] };
    k_facet: { options: string[]; evidence: string[]; actions: string[] };
    items: Array<{
      keyword: string;
      type: "K1_OBJECT" | "K2_OPTION" | "K3_CONSTRAINT" | "K4_CRITERION" | "K5_EVIDENCE" | "K6_ACTION";
      score: number;
      first_seen: string;
      frequency: number;
      decision_value: number;
      evidence_boost: number;
      is_core: boolean;
    }>;
    pipeline: {
      candidates: Array<{ keyword: string; frequency: number; first_seen: string }>;
      classification: Array<{
        keyword: string;
        type: "K1_OBJECT" | "K2_OPTION" | "K3_CONSTRAINT" | "K4_CRITERION" | "K5_EVIDENCE" | "K6_ACTION";
      }>;
      scoring: Array<{ keyword: string; decision_value: number; evidence_boost: number; score: number }>;
      final_selection: {
        k_core_required: string[];
        k_facet_target: string;
        selected_core: string[];
        selected_facet: string[];
        diversity_boost_applied: boolean;
      };
    };
    summary: {
      object_focus: string;
      core_count: number;
      facet_count: number;
    };
  };
  scores: {
    drift: { score: number; band: "GREEN" | "YELLOW" | "RED"; why: string };
    stagnation: { score: number; why: string };
    participation: { imbalance: number; fairtalk: Array<{ speaker: string; p_intent: number }> };
    dps: { score: number; why: string };
  };
  evidence_gate: {
    status: "VERIFIED" | "MIXED" | "UNVERIFIED";
    claims: Array<{ claim: string; verifiability: number; note: string }>;
  };
  intervention: {
    level: "L0" | "L1" | "L2";
    banner_text: string;
    decision_lock: { triggered: boolean; reason: string };
  };
  recommendations: {
    r1_resources: Array<{ title: string; url: string; reason: string }>;
    r2_options: Array<{ option: string; pros: string[]; risks: string[]; evidence_note: string }>;
  };
}

export interface MeetingState {
  meeting_goal: string;
  initial_context: string;
  window_size: number;
  transcript: TranscriptUtterance[];
  agenda_stack: AgendaItem[];
  agenda_candidates?: AgendaTrackerCandidate[];
  agenda_vectors?: Record<string, AgendaVectorEntry>;
  agenda_state_map?: Record<string, AgendaStateEntry>;
  active_agenda_id?: string;
  agenda_events?: AgendaEvent[];
  drift_state?: "Normal" | "Yellow" | "Red" | "Re-orient";
  drift_ui_cues?: {
    glow_k_core: boolean;
    fix_k_core_focus: boolean;
    reduce_facets: boolean;
    show_banner: boolean;
  };
  drift_debug?: {
    s45: number;
    band: string;
    yellow_seconds: number;
    red_seconds: number;
    safe_zone: boolean;
    window_turns: number;
  };
  dps_t?: number;
  dps_breakdown?: {
    option_coverage: number;
    constraint_coverage: number;
    evidence_coverage: number;
    tradeoff_coverage: number;
    closing_readiness: number;
    counts: {
      options: number;
      constraints: number;
      criteria: number;
      evidence: number;
      actions: number;
    };
    active_state: string;
    decision_lock: boolean;
  };
  stagnation_flag?: boolean;
  loop_state?: "Normal" | "Watching" | "Looping" | "Anchoring";
  flow_pulse_debug?: {
    novelty_rate_3m: number;
    arg_novelty: number;
    delta_dps: number;
    anchor_ratio: number;
    conditions: {
      surface_repetition_a: boolean;
      content_repetition_b: boolean;
      no_progress_c: boolean;
      anchoring_exception: boolean;
    };
    turns_3m: number;
  };
  decision_lock_debug?: {
    stance_convergence: number;
    stance_total: number;
    agree_count: number;
    disagree_count: number;
    elapsed_sec: number;
    trigger_stance: boolean;
    trigger_stagnation: boolean;
    trigger_timebox: boolean;
  };
  evidence_status?: "VERIFIED" | "MIXED" | "UNVERIFIED";
  evidence_snippet?: string;
  evidence_log?: Array<{
    claim: string;
    speaker?: string;
    timestamp?: string;
    detect_score?: number;
    verifiability: number;
    eqs?: number | null;
    tier?: number;
    recency?: number;
    claim_match?: number;
    agreement?: number;
    status: "VERIFIED" | "MIXED" | "UNVERIFIED";
    note: string;
  }>;
  recommendation_debug?: {
    triggered: boolean;
    trigger_a_info_seeking: boolean;
    trigger_b_evidence_weak: boolean;
    trigger_c_slot_fulfillment: boolean;
    info_signal_count_60s: number;
    slot_counts?: {
      k1_object: number;
      k3_constraint: number;
      k4_criterion: number;
    };
    evidence_status?: string;
    shown_r1?: number;
    shown_r2?: number;
    reason?: string;
  };
  fairtalk_glow?: Array<{
    speaker: string;
    p_intent: number;
    glow: "none" | "soft" | "strong";
    intent_active: boolean;
    last_seen_sec: number;
  }>;
  fairtalk_debug?: {
    active_speakers: number;
    soft_count: number;
    strong_count: number;
    rule: string;
  };
  llm_enabled?: boolean;
  analysis_runtime?: {
    tick_mode?: "full_context" | "windowed";
    transcript_count?: number;
    llm_window_turns?: number;
    engine_window_turns?: number;
    control_plane_source?: string;
    control_plane_reason?: string;
    used_local_fallback?: boolean;
  };
  llm_status?: LlmStatus;
  agenda_tracker_debug?: {
    topic_shift_sustained: boolean;
    collective_intent: boolean;
    decision_slots: boolean;
    sub_issue_promoted: boolean;
    signals: {
      sim60: number;
      sim120: number;
      duration120_sec: number;
      intent_count_60s: number;
      speakers_60s: number;
      object_slots: number;
      support_slots: number;
    };
  };
  analysis: AnalysisOutput | null;
  artifacts: Record<string, ArtifactOutput>;
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
