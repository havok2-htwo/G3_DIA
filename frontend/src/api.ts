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

// --- Auth / API keys ---
export type WhoAmI = {
  username: string;
  must_change_password: boolean;
};

export type ApiKeyUsage = {
  total_seconds_processed: number;
  request_count: number;
  last_used_at: string | null;
};

export type ApiKeyInfo = {
  id: string;
  alias: string;
  created_at: string | null;
  usage: ApiKeyUsage;
};

export type CreatedApiKey = {
  id: string;
  alias: string;
  created_at: string;
  token: string;
};

/**
 * All admin requests carry the httpOnly session cookie (same-origin). 401 -> not logged
 * in; 403 password_change_required -> forced password change.
 */
async function requestJson<T>(input: string, init?: RequestInit): Promise<T> {
  const nextHeaders = new Headers(init?.headers ?? {});
  if (!(init?.body instanceof FormData) && !nextHeaders.has("Content-Type")) {
    nextHeaders.set("Content-Type", "application/json");
  }

  const response = await fetch(input, {
    credentials: "include",
    ...init,
    headers: nextHeaders,
  });

  if (response.status === 401) {
    throw new Error("unauthorized");
  }
  if (response.status === 403) {
    const payload = (await response.json().catch(() => ({}))) as { detail?: string };
    if (payload.detail === "password_change_required") {
      throw new Error("password_change_required");
    }
    throw new Error(payload.detail ?? "Forbidden");
  }
  if (!response.ok) {
    const payload = (await response.json().catch(() => ({ detail: `HTTP ${response.status}` }))) as {
      detail?: string;
    };
    throw new Error(payload.detail ?? `HTTP ${response.status}`);
  }

  if (response.status === 204) {
    return undefined as T;
  }
  return response.json() as Promise<T>;
}

async function readErrorDetail(response: Response): Promise<string> {
  const payload = (await response.json().catch(() => ({ detail: `HTTP ${response.status}` }))) as {
    detail?: string;
  };
  return payload.detail ?? `HTTP ${response.status}`;
}

export async function login(username: string, password: string) {
  return requestJson<WhoAmI>("/api/admin/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}

export async function logout() {
  return requestJson<{ ok: boolean }>("/api/admin/auth/logout", { method: "POST" });
}

export async function whoami() {
  return requestJson<WhoAmI>("/api/admin/auth/whoami", { method: "GET" });
}

export async function changePassword(currentPassword: string, newPassword: string) {
  return requestJson<{ ok: boolean; must_change_password: boolean }>("/api/admin/auth/change-password", {
    method: "POST",
    body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
  });
}

export async function listApiKeys() {
  return requestJson<{ keys: ApiKeyInfo[] }>("/api/admin/api-keys", { method: "GET" });
}

export async function createApiKey(alias: string) {
  return requestJson<CreatedApiKey>("/api/admin/api-keys", {
    method: "POST",
    body: JSON.stringify({ alias }),
  });
}

export async function deleteApiKey(id: string) {
  return requestJson<{ ok: boolean }>(`/api/admin/api-keys/${encodeURIComponent(id)}`, { method: "DELETE" });
}

export async function getSettings() {
  return requestJson<SettingsResponse>("/api/admin/settings", { method: "GET" });
}

export async function saveSettings(settings: AdminSettings) {
  return requestJson<SettingsResponse & { ok: boolean; model_reloaded: boolean; model_loaded: boolean | null }>(
    "/api/admin/settings",
    {
      method: "PUT",
      body: JSON.stringify(settings),
    },
  );
}

export async function getStats() {
  return requestJson<StatsResponse>("/api/admin/stats", { method: "GET" });
}

export async function getTask() {
  return requestJson<TaskResponse>("/api/admin/task", { method: "GET" });
}

export async function runBenchmark(file: File, repeatCount: number) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("repeat_count", String(repeatCount));

  const response = await fetch("/api/admin/benchmark", {
    method: "POST",
    credentials: "include",
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
