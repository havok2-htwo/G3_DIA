import { FormEvent, startTransition, useEffect, useState } from "react";

import {
  AdminSettings,
  ApiKeyInfo,
  BenchmarkResponse,
  CreatedApiKey,
  SettingsResponse,
  StatsResponse,
  TaskResponse,
  changePassword,
  createApiKey,
  deleteApiKey,
  getSettings,
  getStats,
  getTask,
  listApiKeys,
  login,
  logout,
  runBenchmark,
  saveSettings,
  whoami,
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

type AuthState = "loading" | "login" | "change" | "ready";

const NUMBER_LOCALE = "de-DE";
const POLL_MS = 1000;
const emptySettings: AdminSettings = { diarization_model_id: "", model_cache_path: "", huggingface_token: "" };

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
  const [authState, setAuthState] = useState<AuthState>("loading");
  const [currentUser, setCurrentUser] = useState("");
  const [loginUsername, setLoginUsername] = useState("admin");
  const [loginPassword, setLoginPassword] = useState("");
  const [authBusy, setAuthBusy] = useState(false);
  const [authError, setAuthError] = useState("");

  const [pwCurrent, setPwCurrent] = useState("");
  const [pwNew, setPwNew] = useState("");
  const [pwConfirm, setPwConfirm] = useState("");
  const [pwBusy, setPwBusy] = useState(false);
  const [pwError, setPwError] = useState("");
  const [pwMessage, setPwMessage] = useState("");

  const [apiKeys, setApiKeys] = useState<ApiKeyInfo[]>([]);
  const [newKeyAlias, setNewKeyAlias] = useState("");
  const [createdKey, setCreatedKey] = useState<CreatedApiKey | null>(null);
  const [apiKeyBusy, setApiKeyBusy] = useState(false);

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

  const isReady = authState === "ready";

  const applySettings = (payload: SettingsResponse) => {
    setSettingsForm(payload.settings);
    setLoadedModelIdentifier(payload.loaded_model_identifier);
  };

  const resetDashboardState = () => {
    startTransition(() => {
      setStats(null);
      setTask(null);
      setSettingsForm(emptySettings);
      setLoadedModelIdentifier(null);
      setApiKeys([]);
      setCreatedKey(null);
      setMessage("");
      setErrorMessage("");
      setBenchmarkResult(null);
      setBenchmarkMessage("");
    });
  };

  // Centralized reaction to auth failures raised by any admin call.
  const handleApiError = (error: unknown, fallback: string): boolean => {
    const messageText = error instanceof Error ? error.message : "";
    if (messageText === "unauthorized") {
      resetDashboardState();
      setAuthState("login");
      setAuthError("Your session has expired. Please sign in again.");
      return true;
    }
    if (messageText === "password_change_required") {
      setAuthState("change");
      return true;
    }
    setErrorMessage(error instanceof Error ? error.message : fallback);
    return false;
  };

  const loadDashboard = async () => {
    try {
      const [settingsResponse, statsResponse, taskResponse, keysResponse] = await Promise.all([
        getSettings(),
        getStats(),
        getTask(),
        listApiKeys(),
      ]);
      startTransition(() => {
        applySettings(settingsResponse);
        setStats(statsResponse);
        setTask(taskResponse);
        setApiKeys(keysResponse.keys);
        setErrorMessage("");
      });
    } catch (error) {
      handleApiError(error, "The dashboard could not be loaded.");
    }
  };

  // Bootstrap: check the session on mount.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const me = await whoami();
        if (cancelled) return;
        setCurrentUser(me.username);
        setAuthState(me.must_change_password ? "change" : "ready");
      } catch {
        if (!cancelled) setAuthState("login");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!isReady) return;
    void loadDashboard();
    const intervalId = window.setInterval(() => void loadDashboard(), POLL_MS);
    return () => window.clearInterval(intervalId);
  }, [authState]);

  const handleLogin = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setAuthBusy(true);
    setAuthError("");
    try {
      const me = await login(loginUsername.trim(), loginPassword);
      setCurrentUser(me.username);
      setLoginPassword("");
      setAuthState(me.must_change_password ? "change" : "ready");
    } catch (error) {
      setAuthError(
        error instanceof Error && error.message === "unauthorized"
          ? "Invalid username or password."
          : error instanceof Error
            ? error.message
            : "Login failed.",
      );
    } finally {
      setAuthBusy(false);
    }
  };

  const handleChangePassword = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setPwError("");
    setPwMessage("");
    if (pwNew.length < 4) {
      setPwError("The new password must be at least 4 characters.");
      return;
    }
    if (pwNew !== pwConfirm) {
      setPwError("The new password and its confirmation do not match.");
      return;
    }
    setPwBusy(true);
    try {
      await changePassword(pwCurrent, pwNew);
      setPwCurrent("");
      setPwNew("");
      setPwConfirm("");
      setPwMessage("Password updated.");
      setAuthState("ready");
    } catch (error) {
      setPwError(error instanceof Error ? error.message : "Password change failed.");
    } finally {
      setPwBusy(false);
    }
  };

  const handleLogout = async () => {
    try {
      await logout();
    } catch {
      // ignore
    }
    resetDashboardState();
    setLoginPassword("");
    setAuthState("login");
  };

  const handleCreateApiKey = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setApiKeyBusy(true);
    setMessage("");
    setErrorMessage("");
    try {
      const created = await createApiKey(newKeyAlias.trim());
      setCreatedKey(created);
      setNewKeyAlias("");
      setMessage(`API key "${created.alias}" created. Copy it now — it is shown only once.`);
      const keysResponse = await listApiKeys();
      setApiKeys(keysResponse.keys);
    } catch (error) {
      handleApiError(error, "The API key could not be created.");
    } finally {
      setApiKeyBusy(false);
    }
  };

  const handleDeleteApiKey = async (keyId: string, alias: string) => {
    setMessage("");
    setErrorMessage("");
    try {
      await deleteApiKey(keyId);
      setCreatedKey((current) => (current?.id === keyId ? null : current));
      setMessage(`API key "${alias}" deleted.`);
      const keysResponse = await listApiKeys();
      setApiKeys(keysResponse.keys);
    } catch (error) {
      handleApiError(error, "The API key could not be deleted.");
    }
  };

  const handleSaveSettings = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSaveBusy(true);
    try {
      const response = await saveSettings(settingsForm);
      applySettings(response);
      setMessage(response.model_reloaded ? "Settings saved and runtime reloaded." : "Settings saved.");
      setErrorMessage("");
      await loadDashboard();
    } catch (error) {
      handleApiError(error, "Saving settings failed.");
    } finally {
      setSaveBusy(false);
    }
  };

  const handleRunBenchmark = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!benchmarkFile) {
      setBenchmarkMessage("Please choose an audio or video file first.");
      setBenchmarkResult(null);
      return;
    }
    setBenchmarkBusy(true);
    try {
      const result = await runBenchmark(benchmarkFile, benchmarkRepeatCount);
      setBenchmarkResult(result);
      setBenchmarkMessage("Benchmark finished.");
      setErrorMessage("");
      await loadDashboard();
    } catch (error) {
      if (handleApiError(error, "Benchmark failed.")) {
        return;
      }
      setBenchmarkMessage(error instanceof Error ? error.message : "Benchmark failed.");
      setBenchmarkResult(null);
    } finally {
      setBenchmarkBusy(false);
    }
  };

  if (authState === "loading") {
    return (
      <main className="shell centered">
        <section className="panel login-panel">
          <p className="message">Loading...</p>
        </section>
      </main>
    );
  }

  if (authState === "login") {
    return (
      <main className="shell centered">
        <section className="panel login-panel">
          <div className="hero-copy">
            <span className="eyebrow">Private Access</span>
            <h1>GENESIS DIA Admin</h1>
            <p>
              The public diarization API stays open until you create an API key. The private dashboard for runtime
              status, benchmarks, and history is protected by a username and password login.
            </p>
          </div>

          <form className="login-form" onSubmit={handleLogin}>
            <label>
              <span>Username</span>
              <input
                value={loginUsername}
                autoComplete="username"
                onChange={(event) => setLoginUsername(event.target.value)}
                placeholder="admin"
              />
            </label>
            <label>
              <span>Password</span>
              <input
                type="password"
                value={loginPassword}
                autoComplete="current-password"
                onChange={(event) => setLoginPassword(event.target.value)}
                placeholder="admin"
              />
            </label>

            <button type="submit" disabled={!loginUsername.trim() || !loginPassword || authBusy}>
              {authBusy ? "Signing in..." : "Sign In"}
            </button>

            <p className="message">Default credentials: admin / admin. You must change the password on first login.</p>
            {authError && <p className="message error">{authError}</p>}
          </form>
        </section>
      </main>
    );
  }

  if (authState === "change") {
    return (
      <main className="shell centered">
        <section className="panel login-panel">
          <div className="hero-copy">
            <span className="eyebrow">Security</span>
            <h1>Set a New Password</h1>
            <p>
              You are signed in as <strong>{currentUser || "admin"}</strong>. Choose a new password before continuing
              to the dashboard.
            </p>
          </div>

          <form className="login-form" onSubmit={handleChangePassword}>
            <label>
              <span>Current Password</span>
              <input type="password" value={pwCurrent} autoComplete="current-password" onChange={(event) => setPwCurrent(event.target.value)} />
            </label>
            <label>
              <span>New Password</span>
              <input type="password" value={pwNew} autoComplete="new-password" onChange={(event) => setPwNew(event.target.value)} />
            </label>
            <label>
              <span>Confirm New Password</span>
              <input type="password" value={pwConfirm} autoComplete="new-password" onChange={(event) => setPwConfirm(event.target.value)} />
            </label>

            <button type="submit" disabled={pwBusy || !pwCurrent || !pwNew || !pwConfirm}>
              {pwBusy ? "Saving..." : "Save New Password"}
            </button>
            <button type="button" className="ghost-button" onClick={() => void handleLogout()}>
              Cancel & Sign Out
            </button>

            {pwMessage && <p className="message">{pwMessage}</p>}
            {pwError && <p className="message error">{pwError}</p>}
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
          <p className="hero-copy">Monitor the current worker, manage API keys, persist the Hugging Face token, and inspect recent request history from one protected dashboard.</p>
        </div>
        <div className="hero-actions">
          <a className="secondary-button" href="/docs">OpenAPI Docs</a>
          <button type="button" className="ghost-button" onClick={() => void handleLogout()}>Logout</button>
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
          <div className="loaded-model"><span>Signed in as</span><strong>{currentUser || "admin"}</strong></div>
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
            <span className="eyebrow">Account</span>
            <h2>Admin Access</h2>
            <div className="metric-grid compact-metrics">
              <div className="metric-card"><span>Username</span><strong>{currentUser || "admin"}</strong></div>
              <div className="metric-card"><span>Session</span><strong>Active (cookie)</strong></div>
            </div>
            <form className="login-form" onSubmit={handleChangePassword}>
              <label><span>Current Password</span><input type="password" value={pwCurrent} autoComplete="current-password" onChange={(event) => setPwCurrent(event.target.value)} /></label>
              <label><span>New Password</span><input type="password" value={pwNew} autoComplete="new-password" onChange={(event) => setPwNew(event.target.value)} /></label>
              <label><span>Confirm New Password</span><input type="password" value={pwConfirm} autoComplete="new-password" onChange={(event) => setPwConfirm(event.target.value)} /></label>
              <button type="submit" disabled={pwBusy || !pwCurrent || !pwNew || !pwConfirm}>{pwBusy ? "Saving..." : "Change Password"}</button>
              {pwMessage && <p className="message">{pwMessage}</p>}
              {pwError && <p className="message error">{pwError}</p>}
            </form>
          </section>

          <section className="panel stack">
            <span className="eyebrow">API Keys</span>
            <h2>Public API Access</h2>
            <p className="section-copy">
              While no key exists, <code>POST /diarize/</code> is open to everyone. As soon as one key exists, callers
              must send a valid <code>X-API-Key</code> header. Usage (processed audio seconds) is tracked per key.
            </p>
            {createdKey && (
              <div className="key-token-card">
                <div className="key-card-head">
                  <div><strong>{createdKey.alias}</strong><p>Copy this key now — it is shown only once.</p></div>
                  <button type="button" className="secondary-button" onClick={() => void copyTextToClipboard(createdKey.token)}>Copy</button>
                </div>
                <div className="key-token-value mono">{createdKey.token}</div>
              </div>
            )}
            <form className="benchmark-form" onSubmit={handleCreateApiKey}>
              <label className="full-width"><span>Alias</span><input value={newKeyAlias} onChange={(event) => setNewKeyAlias(event.target.value)} placeholder="e.g. Key fuer Projekt X" /></label>
              <div className="form-actions full-width">
                <button type="submit" disabled={apiKeyBusy || !newKeyAlias.trim()}>{apiKeyBusy ? "Creating..." : "Create API Key"}</button>
              </div>
            </form>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr><th>Alias</th><th>Created</th><th>Audio (s)</th><th>Requests</th><th>Last Used</th><th></th></tr>
                </thead>
                <tbody>
                  {apiKeys.length === 0 && <tr><td colSpan={6}>No API keys — the public API is currently open.</td></tr>}
                  {apiKeys.map((key) => (
                    <tr key={key.id}>
                      <td>{key.alias}</td>
                      <td>{formatDateTime(key.created_at)}</td>
                      <td>{formatFixed(key.usage.total_seconds_processed, 1)}</td>
                      <td>{key.usage.request_count}</td>
                      <td>{formatDateTime(key.usage.last_used_at)}</td>
                      <td><button type="button" className="ghost-button danger-button" onClick={() => void handleDeleteApiKey(key.id, key.alias)}>Delete</button></td>
                    </tr>
                  ))}
                </tbody>
              </table>
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
