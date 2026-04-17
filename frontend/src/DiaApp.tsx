import { FormEvent, startTransition, useEffect, useState } from "react";

import {
  AdminKeyMetadata,
  AdminSettings,
  BenchmarkResponse,
  SettingsResponse,
  StatsResponse,
  TaskResponse,
  getKeys,
  getSettings,
  getStats,
  getTask,
  rotateAdminKey,
  runBenchmark,
  saveSettings,
} from "./api";

type HistoryEntry = {
  timestamp?: string;
  source_ip?: string;
  model_id?: string;
  audio_seconds?: number;
  num_speakers?: number | null;
  min_speakers?: number | null;
  max_speakers?: number | null;
  speakers_found?: number;
  segments_found?: number;
  total_duration_ms?: number;
  summary?: string;
};

const ADMIN_KEY_STORAGE = "genesis_admin_key";
const NUMBER_LOCALE = "de-DE";
const POLL_MS = 1000;
const emptySettings: AdminSettings = { diarization_model_id: "", model_cache_path: "", huggingface_token: "" };

const readStoredAdminKey = () => {
  try {
    return localStorage.getItem(ADMIN_KEY_STORAGE) || sessionStorage.getItem(ADMIN_KEY_STORAGE) || "";
  } catch {
    return "";
  }
};

const writeStoredAdminKey = (value: string) => {
  try {
    localStorage.setItem(ADMIN_KEY_STORAGE, value);
    sessionStorage.setItem(ADMIN_KEY_STORAGE, value);
  } catch {}
};

const clearStoredAdminKey = () => {
  try {
    localStorage.removeItem(ADMIN_KEY_STORAGE);
    sessionStorage.removeItem(ADMIN_KEY_STORAGE);
  } catch {}
};

const formatValue = (value: number | null | undefined, suffix = "") => (
  value === null || value === undefined || Number.isNaN(value)
    ? "n/a"
    : `${new Intl.NumberFormat(NUMBER_LOCALE, { maximumFractionDigits: Number.isInteger(value) ? 0 : 2 }).format(value)}${suffix}`
);

const formatFixed = (value: number | null | undefined, digits = 2, suffix = "") => (
  value === null || value === undefined || Number.isNaN(value)
    ? "n/a"
    : `${new Intl.NumberFormat(NUMBER_LOCALE, { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value)}${suffix}`
);

const formatVram = (value: number | null | undefined) => (
  value === null || value === undefined || Number.isNaN(value)
    ? "n/a"
    : `${new Intl.NumberFormat(NUMBER_LOCALE, { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value)} MB`
);

const formatDateTime = (value?: string | null) => {
  if (!value) return "n/a";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime())
    ? value
    : new Intl.DateTimeFormat(NUMBER_LOCALE, {
        year: "2-digit",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      }).format(parsed);
};

async function copyTextToClipboard(value: string) {
  if (!value) return;
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value);
  }
}

function ProgressBar({ value }: { value: number }) {
  const safeValue = Math.max(0, Math.min(100, Number(value || 0)));
  return (
    <div className="progress-shell">
      <div className="progress-fill" style={{ width: `${safeValue}%` }} />
    </div>
  );
}

export default function DiaApp() {
  const [adminKey, setAdminKey] = useState(() => readStoredAdminKey());
  const [adminKeyInput, setAdminKeyInput] = useState(() => readStoredAdminKey());
  const [authBusy, setAuthBusy] = useState(false);
  const [authError, setAuthError] = useState("");
  const [adminMetadata, setAdminMetadata] = useState<AdminKeyMetadata | null>(null);
  const [newKey, setNewKey] = useState<(AdminKeyMetadata & { token: string }) | null>(null);
  const [settingsForm, setSettingsForm] = useState<AdminSettings>(emptySettings);
  const [loadedModelIdentifier, setLoadedModelIdentifier] = useState<string[] | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [task, setTask] = useState<TaskResponse | null>(null);
  const [message, setMessage] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [saveBusy, setSaveBusy] = useState(false);
  const [benchmarkBusy, setBenchmarkBusy] = useState(false);
  const [benchmarkFile, setBenchmarkFile] = useState<File | null>(null);
  const [benchmarkRepeatCount, setBenchmarkRepeatCount] = useState(1);
  const [benchmarkResult, setBenchmarkResult] = useState<BenchmarkResponse | null>(null);
  const [benchmarkMessage, setBenchmarkMessage] = useState("");

  const applySettings = (payload: SettingsResponse) => {
    setSettingsForm(payload.settings);
    setLoadedModelIdentifier(payload.loaded_model_identifier);
  };

  const clearBrowserKey = (nextMessage = "") => {
    clearStoredAdminKey();
    startTransition(() => {
      setAdminKey("");
      setAdminKeyInput("");
      setAdminMetadata(null);
      setNewKey(null);
      setStats(null);
      setTask(null);
      setSettingsForm(emptySettings);
      setLoadedModelIdentifier(null);
      setMessage(nextMessage);
      setErrorMessage(nextMessage);
      setBenchmarkResult(null);
      setBenchmarkMessage("");
      setAuthError("");
    });
  };

  const loadDashboard = async (currentAdminKey = adminKey) => {
    if (!currentAdminKey) return;
    try {
      const [settingsResponse, statsResponse, taskResponse, keysResponse] = await Promise.all([
        getSettings(currentAdminKey),
        getStats(currentAdminKey),
        getTask(currentAdminKey),
        getKeys(currentAdminKey),
      ]);
      startTransition(() => {
        applySettings(settingsResponse);
        setStats(statsResponse);
        setTask(taskResponse);
        setAdminMetadata(keysResponse.admin_key);
        setErrorMessage("");
      });
    } catch (error) {
      if (error instanceof Error && error.message === "unauthorized") {
        clearBrowserKey("The admin key is invalid or expired.");
        return;
      }
      setErrorMessage(error instanceof Error ? error.message : "The dashboard could not be loaded.");
    }
  };

  useEffect(() => {
    if (!adminKey) return;
    void loadDashboard(adminKey);
  }, [adminKey]);

  useEffect(() => {
    if (!adminKey) return;
    const intervalId = window.setInterval(() => void loadDashboard(adminKey), POLL_MS);
    return () => window.clearInterval(intervalId);
  }, [adminKey]);

  const handleOpenAdmin = async () => {
    const candidate = adminKeyInput.trim();
    if (!candidate) return;
    setAuthBusy(true);
    setAuthError("");
    try {
      await Promise.all([getSettings(candidate), getKeys(candidate)]);
      writeStoredAdminKey(candidate);
      setAdminKey(candidate);
      setAdminKeyInput(candidate);
      setMessage("Admin key accepted. Loading dashboard.");
    } catch (error) {
      setAuthError(error instanceof Error && error.message === "unauthorized" ? "The entered admin key is not valid." : (error instanceof Error ? error.message : "The admin key could not be verified."));
    } finally {
      setAuthBusy(false);
    }
  };

  const handleRotateAdminKey = async () => {
    if (!adminKey) return;
    try {
      const response = await rotateAdminKey(adminKey);
      writeStoredAdminKey(response.key.token);
      setAdminKey(response.key.token);
      setAdminKeyInput(response.key.token);
      setAdminMetadata(response.keys.admin_key);
      setNewKey(response.key);
      setMessage("Admin key rotated successfully.");
      setErrorMessage("");
    } catch (error) {
      if (error instanceof Error && error.message === "unauthorized") {
        clearBrowserKey("The admin key is invalid or expired.");
        return;
      }
      setErrorMessage(error instanceof Error ? error.message : "The admin key could not be rotated.");
    }
  };

  const handleSaveSettings = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!adminKey) return;
    setSaveBusy(true);
    try {
      const response = await saveSettings(adminKey, settingsForm);
      applySettings(response);
      setMessage(response.model_reloaded ? "Settings saved and runtime reloaded." : "Settings saved.");
      setErrorMessage("");
      await loadDashboard(adminKey);
    } catch (error) {
      if (error instanceof Error && error.message === "unauthorized") {
        clearBrowserKey("The admin key is invalid or expired.");
        return;
      }
      setErrorMessage(error instanceof Error ? error.message : "Saving settings failed.");
    } finally {
      setSaveBusy(false);
    }
  };

  const handleRunBenchmark = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!adminKey) return;
    if (!benchmarkFile) {
      setBenchmarkMessage("Please choose an audio or video file first.");
      setBenchmarkResult(null);
      return;
    }
    setBenchmarkBusy(true);
    try {
      const result = await runBenchmark(adminKey, benchmarkFile, benchmarkRepeatCount);
      setBenchmarkResult(result);
      setBenchmarkMessage("Benchmark finished.");
      setErrorMessage("");
      await loadDashboard(adminKey);
    } catch (error) {
      if (error instanceof Error && error.message === "unauthorized") {
        clearBrowserKey("The admin key is invalid or expired.");
        return;
      }
      setBenchmarkMessage(error instanceof Error ? error.message : "Benchmark failed.");
      setBenchmarkResult(null);
    } finally {
      setBenchmarkBusy(false);
    }
  };

  if (!adminKey) {
    return (
      <main className="centered">
        <section className="panel login-panel">
          <div className="hero-copy">
            <span className="eyebrow">GENESIS DIA Server</span>
            <h1>Protected operator access for local speaker diarization.</h1>
            <p>The public diarization API stays open, while the private dashboard for runtime status, benchmarks, and history is protected by the admin key.</p>
          </div>
          <form className="login-form" onSubmit={(event) => { event.preventDefault(); void handleOpenAdmin(); }}>
            <label>
              <span>Enter X-Admin-Key</span>
              <input type="password" value={adminKeyInput} onChange={(event) => setAdminKeyInput(event.target.value)} placeholder="genesis_admin_..." />
            </label>
            <button type="submit" disabled={!adminKeyInput.trim() || authBusy}>{authBusy ? "Verifying..." : "Open Admin Dashboard"}</button>
            <p className="muted">Use the persistent admin key or the temporary startup key shown during server launch.</p>
            {authError && <p className="message error">{authError}</p>}
          </form>
        </section>
      </main>
    );
  }

  const history = (stats?.history ?? []) as HistoryEntry[];
  const loadedModelLabel = loadedModelIdentifier?.[0] || settingsForm.diarization_model_id || "No model loaded";

  return (
    <main className="shell">
      <header className="hero">
        <div className="hero-copy-block">
          <span className="eyebrow">GENESIS DIA Server</span>
          <h1>Speaker diarization with the G3 control surface.</h1>
          <p className="hero-copy">Monitor the current worker, rotate admin access, persist the Hugging Face token, and inspect recent request history from one protected dashboard.</p>
        </div>
        <div className="hero-actions">
          <a className="secondary-button" href="/docs">OpenAPI Docs</a>
          <button type="button" className="ghost-button" onClick={() => clearBrowserKey("Admin key removed from this browser.")}>Forget Browser Key</button>
        </div>
      </header>

      {message && <p className="message">{message}</p>}
      {errorMessage && <p className="message error">{errorMessage}</p>}

      <section className="panel">
        <div className="section-heading">
          <div>
            <span className="eyebrow">Overview</span>
            <h2>Operational Snapshot</h2>
          </div>
          <div className="loaded-model"><span>Loaded Model</span><strong>{loadedModelLabel}</strong></div>
        </div>
        <div className="metric-grid whisper-dashboard-metrics">
          <div className="metric-card"><span>Total Requests</span><strong>{stats?.summary.total_requests ?? 0}</strong></div>
          <div className="metric-card"><span>Average Duration</span><strong>{formatValue(stats?.summary.avg_total_duration_ms, " ms")}</strong></div>
          <div className="metric-card"><span>Average Speakers</span><strong>{formatFixed(stats?.summary.avg_speakers_found, 2)}</strong></div>
          <div className="metric-card"><span>Average Segments</span><strong>{formatFixed(stats?.summary.avg_segments_found, 2)}</strong></div>
          <div className="metric-card"><span>Pending Requests</span><strong>{task?.pending_requests ?? 0}</strong></div>
          <div className="metric-card"><span>Worker</span><strong>{task?.worker_running ? "Busy" : "Ready"}</strong></div>
        </div>
      </section>

      <section className="panel-grid admin-overview-grid">
        <section className="panel stack">
          <span className="eyebrow">Task Manager</span>
          <h2>Current Worker and Progress</h2>
          <div className="progress-meta"><span>{task?.current_task.task_name || "Idle"}</span><strong>{formatFixed(task?.current_task.progress, 1, "%")}</strong></div>
          <ProgressBar value={task?.current_task.progress ?? 0} />
          <p className="dashboard-note">{task?.current_task.details || "Server is ready."}</p>
          <div className="metric-grid compact-metrics">
            <div className="metric-card"><span>Active Request</span><strong>{task?.active_request_id || "No active request"}</strong></div>
            <div className="metric-card"><span>Active Audio</span><strong>{formatFixed(task?.active_audio_seconds, 3, " s")}</strong></div>
            <div className="metric-card"><span>Last Duration</span><strong>{formatValue(task?.last_duration_ms, " ms")}</strong></div>
            <div className="metric-card"><span>Last Completed</span><strong>{formatDateTime(task?.last_completed_at)}</strong></div>
          </div>
          {task?.last_error && <p className="message error">Last worker error: {task.last_error}</p>}
        </section>

        <div className="side-widget-stack stack">
          <section className="panel stack">
            <span className="eyebrow">Admin Key</span>
            <h2>Protected Operator Access</h2>
            {newKey?.token && (
              <div className="key-token-card">
                <div className="key-card-head">
                  <div><strong>Freshly Rotated Key</strong><p>This key is shown once and already stored in this browser.</p></div>
                  <button type="button" className="secondary-button" onClick={() => void copyTextToClipboard(newKey.token)}>Copy</button>
                </div>
                <div className="key-token-value mono">{newKey.token}</div>
              </div>
            )}
            <div className="metric-grid compact-metrics">
              <div className="metric-card"><span>Name</span><strong>{adminMetadata?.label || "Master Admin Key"}</strong></div>
              <div className="metric-card"><span>Created</span><strong>{formatDateTime(adminMetadata?.created_at)}</strong></div>
              <div className="metric-card"><span>Last Used</span><strong>{formatDateTime(adminMetadata?.last_used_at)}</strong></div>
              <div className="metric-card"><span>Browser Token</span><strong>{adminKey ? "Stored" : "Missing"}</strong></div>
            </div>
            <div className="key-token-card">
              <div className="key-card-head">
                <div><strong>Current Browser Key</strong><p>This token is sent as <code>X-Admin-Key</code> with every protected admin request.</p></div>
                <button type="button" className="secondary-button" onClick={() => void copyTextToClipboard(adminKey)}>Copy</button>
              </div>
              <div className="key-token-value mono">{adminKey}</div>
            </div>
            <div className="button-row">
              <button type="button" onClick={() => void handleRotateAdminKey()}>Rotate Admin Key</button>
              <button type="button" className="secondary-button" onClick={() => void copyTextToClipboard(adminKey)}>Copy Current Key</button>
            </div>
          </section>

          <section className="panel stack benchmark-widget">
            <span className="eyebrow">Benchmark</span>
            <h2>Repeat Diarization Through the Active Pipeline</h2>
            <form className="benchmark-form" onSubmit={handleRunBenchmark}>
              <label className="full-width">
                <span>Audio File</span>
                <input type="file" accept="audio/*,video/*" onChange={(event) => setBenchmarkFile(event.target.files?.[0] ?? null)} />
              </label>
              <label>
                <span>Repeats</span>
                <input type="number" min={1} max={32} value={benchmarkRepeatCount} onChange={(event) => setBenchmarkRepeatCount(Math.max(1, Math.min(32, Number(event.target.value) || 1)))} />
              </label>
              <div className="form-actions full-width">
                <button type="submit" disabled={benchmarkBusy}>{benchmarkBusy ? "Benchmark running..." : "Run Benchmark"}</button>
                {benchmarkMessage && <p className={`message ${benchmarkResult ? "" : "error"}`.trim()}>{benchmarkMessage}</p>}
              </div>
            </form>
            {benchmarkResult && (
              <div className="benchmark-results">
                <div className="benchmark-grid">
                  <article className="benchmark-card"><span>Workflow</span><strong>{benchmarkResult.workflow === "serial_diarization" ? "Serial Diarization" : benchmarkResult.workflow}</strong></article>
                  <article className="benchmark-card"><span>RTF</span><strong>{formatFixed(benchmarkResult.rtf, 3)}</strong></article>
                  <article className="benchmark-card"><span>Total Time</span><strong>{formatValue(benchmarkResult.total_wall_time_ms, " ms")}</strong></article>
                  <article className="benchmark-card"><span>Time / Run</span><strong>{formatFixed(benchmarkResult.avg_wall_time_per_run_ms, 2, " ms")}</strong></article>
                  <article className="benchmark-card"><span>Speakers</span><strong>{benchmarkResult.speakers_found}</strong></article>
                  <article className="benchmark-card"><span>Segments</span><strong>{benchmarkResult.segments_found}</strong></article>
                </div>
                <div className="benchmark-meta">
                  <div><span>File</span><strong>{benchmarkResult.file_name}</strong></div>
                  <div><span>Model</span><strong>{benchmarkResult.model_id}</strong></div>
                  <div><span>Repeats</span><strong>{benchmarkResult.repeat_count}</strong></div>
                  <div><span>Peak VRAM</span><strong>{formatVram(benchmarkResult.peak_vram_reserved_mb)}</strong></div>
                </div>
                <div className="benchmark-transcript-wrap">
                  <span className="benchmark-transcript-label">Sample Result</span>
                  <pre className="benchmark-transcript json-preview">{JSON.stringify(benchmarkResult.sample_result, null, 2)}</pre>
                </div>
              </div>
            )}
          </section>
        </div>
      </section>

      <div className="content-grid">
        <section className="panel">
          <div className="section-heading">
            <div><span className="eyebrow">Settings</span><h2>Runtime Access and Cache Path</h2></div>
            <div className="loaded-model"><span>Public API</span><strong className="mono">POST /diarize/</strong></div>
          </div>
          <form className="settings-form" onSubmit={handleSaveSettings}>
            <label className="full-width"><span>Active Model</span><input value={settingsForm.diarization_model_id} readOnly /></label>
            <label className="full-width"><span>Model Cache Path</span><input value={settingsForm.model_cache_path} onChange={(event) => setSettingsForm((current) => ({ ...current, model_cache_path: event.target.value }))} /></label>
            <label className="full-width"><span>Hugging Face Token</span><input type="password" placeholder="hf_... (optional)" value={settingsForm.huggingface_token} onChange={(event) => setSettingsForm((current) => ({ ...current, huggingface_token: event.target.value }))} /></label>
            <p className="field-note full-width">Save the token here if the server should be able to load the gated pyannote model again after restart.</p>
            <div className="form-actions full-width">
              <button type="submit" disabled={saveBusy}>{saveBusy ? "Saving..." : "Save Settings"}</button>
            </div>
          </form>
        </section>

        <section className="panel stack">
          <div className="section-heading">
            <div><span className="eyebrow">Runtime</span><h2>Model and Request Details</h2></div>
          </div>
          <div className="metric-grid compact-metrics">
            <div className="metric-card"><span>Loaded Model</span><strong>{loadedModelLabel}</strong></div>
            <div className="metric-card"><span>Current Task</span><strong>{task?.current_task.task_name || "Idle"}</strong></div>
            <div className="metric-card"><span>Task Progress</span><strong>{formatFixed(task?.current_task.progress, 1, "%")}</strong></div>
            <div className="metric-card"><span>Pending Requests</span><strong>{task?.pending_requests ?? 0}</strong></div>
          </div>
          <div className="key-token-card">
            <div className="key-card-head">
              <div><strong>Current Task Detail</strong><p>This text mirrors the pyannote progress hook and is updated during active diarization.</p></div>
            </div>
            <div className="key-token-value">{task?.current_task.details || "Server is ready."}</div>
          </div>
        </section>
      </div>

      <section className="panel">
        <div className="section-heading">
          <div><span className="eyebrow">History</span><h2>Latest Diarization Requests</h2></div>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Time</th><th>IP</th><th>Model</th><th>Audio</th><th>Speakers</th><th>Segments</th><th>Total</th><th>Constraints</th><th>Summary</th></tr>
            </thead>
            <tbody>
              {history.length === 0 && <tr><td colSpan={9}>No diarization history recorded yet.</td></tr>}
              {history.map((entry, index) => (
                <tr key={`${entry.timestamp ?? "row"}-${index}`}>
                  <td>{entry.timestamp ?? "n/a"}</td>
                  <td>{entry.source_ip ?? "n/a"}</td>
                  <td>{entry.model_id ?? "n/a"}</td>
                  <td>{formatFixed(entry.audio_seconds, 3, " s")}</td>
                  <td>{entry.speakers_found ?? 0}</td>
                  <td>{entry.segments_found ?? 0}</td>
                  <td>{formatValue(entry.total_duration_ms, " ms")}</td>
                  <td>n={entry.num_speakers ?? "-"}, min={entry.min_speakers ?? "-"}, max={entry.max_speakers ?? "-"}</td>
                  <td className="transcript-cell">{entry.summary ?? ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}
