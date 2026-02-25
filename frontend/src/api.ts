import type { ArtifactKind, MeetingState, SttChunkResponse } from "./types";

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

export async function saveConfig(payload: {
  meeting_goal: string;
  initial_context: string;
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

export async function tickAnalysis(): Promise<MeetingState> {
  const res = await fetch("/api/analysis/tick", { method: "POST" });
  return parse<MeetingState>(res);
}

export async function createArtifact(kind: ArtifactKind): Promise<MeetingState> {
  const res = await fetch(`/api/artifacts/${kind}`, { method: "POST" });
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
