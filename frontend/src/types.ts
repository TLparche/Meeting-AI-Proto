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

export interface ArtifactOutput {
  kind: ArtifactKind;
  title: string;
  markdown: string;
  bullets: string[];
}

export interface AnalysisOutput {
  agenda: {
    active: { title: string; status: "ACTIVE" | "CLOSING" | "CLOSED"; confidence: number };
    candidates: Array<{ title: string; confidence: number }>;
  };
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
