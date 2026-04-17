export type AdminKeyMetadata = {
  id: string;
  label: string;
  created_at: string | null;
  last_used_at: string | null;
};

export type AdminSettings = {
  diarization_model_id: string;
  model_cache_path: string;
  huggingface_token: string;
};

export type SettingsResponse = {
  settings: AdminSettings;
  loaded_model_identifier: string[] | null;
};

export type StatsResponse = {
  summary: {
    total_requests: number;
    avg_total_duration_ms: number | null;
    avg_speakers_found: number | null;
    avg_segments_found: number | null;
  };
  history: Array<Record<string, unknown>>;
};

export type TaskResponse = {
  worker_running: boolean;
  pending_requests: number;
  active_request_id: string | null;
  active_started_at: string | null;
  active_audio_seconds: number;
  last_completed_at: string | null;
  last_duration_ms: number | null;
  last_error: string | null;
  total_requests_processed: number;
  current_task: {
    task_name: string;
    progress: number;
    details: string;
  };
  loaded_model_identifier: string[] | null;
};

export type BenchmarkResponse = {
  ok: boolean;
  file_name: string;
  workflow: string;
  model_id: string;
  repeat_count: number;
  audio_seconds: number;
  total_audio_seconds: number;
  total_wall_time_ms: number;
  avg_wall_time_per_run_ms: number;
  avg_single_run_ms: number | null;
  rtf: number | null;
  results_match: boolean;
  speakers_found: number;
  segments_found: number;
  sample_result: Record<string, Array<{ start: number; end: number }>>;
  peak_vram_reserved_mb: number | null;
  peak_vram_allocated_mb: number | null;
};

type RequestOptions = RequestInit & {
  adminKey?: string;
};

async function requestJson<T>(input: string, init?: RequestOptions): Promise<T> {
  const { adminKey, ...requestInit } = init ?? {};
  const nextHeaders = new Headers(init?.headers ?? {});
  if (!(requestInit.body instanceof FormData) && !nextHeaders.has("Content-Type")) {
    nextHeaders.set("Content-Type", "application/json");
  }
  if (adminKey) {
    nextHeaders.set("X-Admin-Key", adminKey);
  }

  const response = await fetch(input, {
    headers: nextHeaders,
    ...requestInit,
  });

  if (response.status === 401) {
    throw new Error("unauthorized");
  }

  if (!response.ok) {
    const payload = await response
      .json()
      .catch(() => ({ detail: `HTTP ${response.status}` })) as { detail?: string };
    throw new Error(payload.detail ?? `HTTP ${response.status}`);
  }

  return response.json() as Promise<T>;
}

async function readErrorDetail(response: Response): Promise<string> {
  const payload = await response
    .json()
    .catch(() => ({ detail: `HTTP ${response.status}` })) as { detail?: string };
  return payload.detail ?? `HTTP ${response.status}`;
}

export async function getKeys(adminKey: string) {
  return requestJson<{ admin_key: AdminKeyMetadata }>("/api/admin/keys", {
    method: "GET",
    adminKey,
  });
}

export async function rotateAdminKey(adminKey: string) {
  return requestJson<{ key: AdminKeyMetadata & { token: string }; keys: { admin_key: AdminKeyMetadata } }>("/api/admin/keys", {
    method: "POST",
    adminKey,
  });
}

export async function getSettings(adminKey: string) {
  return requestJson<SettingsResponse>("/api/admin/settings", {
    method: "GET",
    adminKey,
  });
}

export async function saveSettings(adminKey: string, settings: AdminSettings) {
  return requestJson<SettingsResponse & { ok: boolean; model_reloaded: boolean; model_loaded: boolean | null }>(
    "/api/admin/settings",
    {
      method: "PUT",
      adminKey,
      body: JSON.stringify(settings),
    },
  );
}

export async function getStats(adminKey: string) {
  return requestJson<StatsResponse>("/api/admin/stats", {
    method: "GET",
    adminKey,
  });
}

export async function getTask(adminKey: string) {
  return requestJson<TaskResponse>("/api/admin/task", {
    method: "GET",
    adminKey,
  });
}

export async function runBenchmark(adminKey: string, file: File, repeatCount: number) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("repeat_count", String(repeatCount));

  const response = await fetch("/api/admin/benchmark", {
    method: "POST",
    headers: {
      "X-Admin-Key": adminKey,
    },
    body: formData,
  });

  if (response.status === 401) {
    throw new Error("unauthorized");
  }

  if (!response.ok) {
    throw new Error(await readErrorDetail(response));
  }

  return response.json() as Promise<BenchmarkResponse>;
}
