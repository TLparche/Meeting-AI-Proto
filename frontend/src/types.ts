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
    k_core: { object: string[]; constraints: string[]; criteria: string[] };
    k_facet: { options: string[]; evidence: string[]; actions: string[] };
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
