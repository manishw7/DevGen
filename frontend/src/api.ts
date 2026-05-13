const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/+$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  let payload: any = null;

  try {
    payload = await res.json();
  } catch {
    payload = null;
  }

  if (!res.ok) {
    const detail =
      payload?.detail ||
      payload?.message ||
      payload?.error ||
      `${res.status} ${res.statusText}`;
    throw new Error(detail);
  }

  return payload as T;
}

export function getApiBase() {
  return API_BASE;
}

export async function fetchHealth() {
  return await request<{
    status: string;
    service: string;
    version: string;
    model_loaded: boolean;
    model: {
      base_model_name: string;
      adapter_path: string | null;
      adapter_source: string;
      device: string;
      using_adapter: boolean;
    };
  }>("/");
}

export async function fetchModelInfo() {
  const response = await request<{ status: string; data: any }>("/api/v1/model/info");
  return response.data;
}

export async function fetchDatasetInfo() {
  const response = await request<{ status: string; data: any }>("/api/v1/dataset/info");
  return response.data;
}

export async function fetchRandomSamples(split = "train", count = 12) {
  const response = await request<{ status: string; data: any }>(
    `/api/v1/dataset/random?split=${split}&count=${count}`
  );
  return response.data;
}

export async function fetchPaginatedSamples(
  split = "train",
  page = 0,
  pageSize = 20
) {
  const response = await request<{ status: string; data: any }>(
    `/api/v1/dataset/browse?split=${split}&page=${page}&page_size=${pageSize}`
  );
  return response.data;
}

export async function recognizeDocument(file: File) {
  const form = new FormData();
  form.append("file", file);
  return await request<any>("/api/v1/recognize", {
    method: "POST",
    body: form,
  });
}

export async function recognizeFull(file: File) {
  const form = new FormData();
  form.append("file", file);
  return await request<any>("/api/v1/recognize/full", {
    method: "POST",
    body: form,
  });
}

export async function preprocessImage(file: File) {
  const form = new FormData();
  form.append("file", file);
  return await request<any>("/api/v1/preprocess", {
    method: "POST",
    body: form,
  });
}

export async function evaluateCER(prediction: string, reference: string) {
  const response = await request<{ status: string; data: any }>("/api/v1/evaluate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prediction, reference }),
  });
  return response.data;
}

export async function extractNER(text: string) {
  const response = await request<{ status: string; data: any }>("/api/v1/ner", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  return response.data;
}
