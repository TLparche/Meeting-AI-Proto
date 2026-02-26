import type { LlmConnectResponse, ImportJsonDirResponse, LlmPingResponse, LlmStatus, MeetingState, SttChunkResponse } from "./types";

const JSON_HEADERS = { "Content-Type": "application/json" };

async function parse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return (await res.json()) as T;
}

export async function getState(): Promise<MeetingState> {
  const res = await fetch("/api/state");
  return parse<MeetingState>(res);
}

export async function getLlmStatus(): Promise<LlmStatus> {
  const res = await fetch("/api/llm/status");
  return parse<LlmStatus>(res);
}

export async function pingLlm(): Promise<LlmPingResponse> {
  const res = await fetch("/api/llm/ping", { method: "POST" });
  return parse<LlmPingResponse>(res);
}

export async function connectLlm(): Promise<LlmConnectResponse> {
  const res = await fetch("/api/llm/connect", { method: "POST" });
  return parse<LlmConnectResponse>(res);
}

export async function disconnectLlm(): Promise<LlmConnectResponse> {
  const res = await fetch("/api/llm/disconnect", { method: "POST" });
  return parse<LlmConnectResponse>(res);
}

export async function saveConfig(payload: {
  meeting_goal: string;
  window_size: number;
}): Promise<MeetingState> {
  const res = await fetch("/api/config", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(payload),
  });
  return parse<MeetingState>(res);
}

export async function addUtterance(payload: {
  speaker: string;
  text: string;
  timestamp?: string;
}): Promise<MeetingState> {
  const res = await fetch("/api/transcript/manual", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(payload),
  });
  return parse<MeetingState>(res);
}

export async function importJsonDir(payload: {
  folder: string;
  recursive?: boolean;
  reset_state?: boolean;
  auto_tick?: boolean;
  max_files?: number;
}): Promise<ImportJsonDirResponse> {
  const res = await fetch("/api/transcript/import-json-dir", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(payload),
  });
  return parse<ImportJsonDirResponse>(res);
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
  const res = await fetch("/api/transcript/import-json-files", {
    method: "POST",
    body: form,
  });
  return parse<ImportJsonDirResponse>(res);
}

export async function tickAnalysis(): Promise<MeetingState> {
  const res = await fetch("/api/analysis/tick", { method: "POST" });
  return parse<MeetingState>(res);
}

export async function resetState(): Promise<MeetingState> {
  const res = await fetch("/api/reset", { method: "POST" });
  return parse<MeetingState>(res);
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
  const res = await fetch("/api/stt/chunk", { method: "POST", body: form });
  return parse<SttChunkResponse>(res);
}
