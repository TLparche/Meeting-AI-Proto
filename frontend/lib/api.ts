import type {
  AgendaMarkdownExportResponse,
  AgendaSnapshotExportResponse,
  AgendaSnapshotImportResponse,
  CanvasProblemDefinitionResponse,
  ImportJsonDirResponse,
  LastLlmJsonResponse,
  LlmConnectResponse,
  LlmPingResponse,
  LlmStatus,
  MeetingState,
  ReplayImportResponse,
  ReplayStepResponse,
  SttChunkResponse,
} from "./types";

const JSON_HEADERS = { "Content-Type": "application/json" };
const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL || "").replace(/\/+$/, "");

function apiPath(path: string): string {
  return `${API_BASE_URL}${path}`;
}

async function parse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return (await res.json()) as T;
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(apiPath(path), init);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new Error(`Failed to fetch: ${apiPath(path)} (백엔드 서버 확인 필요) - ${msg}`);
  }
  return parse<T>(res);
}

export async function getState(): Promise<MeetingState> {
  return requestJson<MeetingState>("/api/state", { cache: "no-store" });
}

export async function getLlmStatus(): Promise<LlmStatus> {
  return requestJson<LlmStatus>("/api/llm/status", { cache: "no-store" });
}

export async function pingLlm(): Promise<LlmPingResponse> {
  return requestJson<LlmPingResponse>("/api/llm/ping", { method: "POST" });
}

export async function connectLlm(): Promise<LlmConnectResponse> {
  return requestJson<LlmConnectResponse>("/api/llm/connect", { method: "POST" });
}

export async function disconnectLlm(): Promise<LlmConnectResponse> {
  return requestJson<LlmConnectResponse>("/api/llm/disconnect", { method: "POST" });
}

export async function saveConfig(payload: {
  meeting_goal: string;
  window_size: number;
}): Promise<MeetingState> {
  return requestJson<MeetingState>("/api/config", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(payload),
  });
}

export async function addUtterance(payload: {
  speaker: string;
  text: string;
  timestamp?: string;
}): Promise<MeetingState> {
  return requestJson<MeetingState>("/api/transcript/manual", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(payload),
  });
}

export async function importJsonDir(payload: {
  folder: string;
  recursive?: boolean;
  reset_state?: boolean;
  auto_tick?: boolean;
  max_files?: number;
}): Promise<ImportJsonDirResponse> {
  return requestJson<ImportJsonDirResponse>("/api/transcript/import-json-dir", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(payload),
  });
}

export async function importJsonFiles(payload: {
  files: File[];
  reset_state?: boolean;
  auto_tick?: boolean;
}): Promise<ImportJsonDirResponse> {
  const form = new FormData();
  payload.files.forEach((file) => form.append("files", file, file.name));
  form.append("reset_state", String(payload.reset_state ?? true));
  form.append("auto_tick", String(payload.auto_tick ?? true));
  return requestJson<ImportJsonDirResponse>("/api/transcript/import-json-files", {
    method: "POST",
    body: form,
  });
}

export async function importJsonFilesReplay(payload: {
  files: File[];
  reset_state?: boolean;
  apply_goal?: boolean;
}): Promise<ReplayImportResponse> {
  const form = new FormData();
  payload.files.forEach((file) => form.append("files", file, file.name));
  form.append("reset_state", String(payload.reset_state ?? true));
  form.append("apply_goal", String(payload.apply_goal ?? true));
  return requestJson<ReplayImportResponse>("/api/transcript/replay/import-json-files", {
    method: "POST",
    body: form,
  });
}

export async function replayStep(payload?: {
  lines?: number;
  auto_analyze?: boolean;
}): Promise<ReplayStepResponse> {
  return requestJson<ReplayStepResponse>("/api/transcript/replay/step", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({
      lines: payload?.lines ?? 1,
      auto_analyze: payload?.auto_analyze ?? true,
    }),
  });
}

export async function tickAnalysis(): Promise<MeetingState> {
  return requestJson<MeetingState>("/api/analysis/tick", { method: "POST" });
}

export async function getLastLlmJson(): Promise<LastLlmJsonResponse> {
  return requestJson<LastLlmJsonResponse>("/api/analysis/last-llm-json", { cache: "no-store" });
}

export async function generateCanvasProblemDefinition(payload: {
  topic: string;
  agendas: Array<{
    agenda_id: string;
    title: string;
    keywords: string[];
    summary_bullets: string[];
  }>;
  ideas: Array<{
    id: string;
    agenda_id: string;
    kind: string;
    title: string;
    body: string;
  }>;
}): Promise<CanvasProblemDefinitionResponse> {
  return requestJson<CanvasProblemDefinitionResponse>("/api/canvas/problem-definition", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(payload),
  });
}

export async function exportAgendaMarkdown(): Promise<AgendaMarkdownExportResponse> {
  return requestJson<AgendaMarkdownExportResponse>("/api/export/agenda-markdown", { cache: "no-store" });
}

export async function exportAgendaSnapshot(): Promise<AgendaSnapshotExportResponse> {
  return requestJson<AgendaSnapshotExportResponse>("/api/export/agenda-snapshot", { cache: "no-store" });
}

export async function importAgendaSnapshot(payload: {
  file: File;
  reset_state?: boolean;
}): Promise<AgendaSnapshotImportResponse> {
  const form = new FormData();
  form.append("file", payload.file, payload.file.name);
  form.append("reset_state", String(payload.reset_state ?? true));
  return requestJson<AgendaSnapshotImportResponse>("/api/import/agenda-snapshot", {
    method: "POST",
    body: form,
  });
}

export async function resetState(): Promise<MeetingState> {
  return requestJson<MeetingState>("/api/reset", { method: "POST" });
}

export async function transcribeChunk(payload: {
  blob: Blob;
  filename: string;
  speaker: string;
  source: string;
}): Promise<SttChunkResponse> {
  const form = new FormData();
  form.append("audio", payload.blob, payload.filename);
  form.append("speaker", payload.speaker);
  form.append("source", payload.source);
  return requestJson<SttChunkResponse>("/api/stt/chunk", { method: "POST", body: form });
}
