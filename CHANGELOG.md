# Changelog

## Unreleased

- **Neu (GPU-Auswahl):** Einstellung `gpu_device` (`auto`, `cpu`, `cuda:N`) mit Dropdown im Admin-Dashboard; die Optionen listet der Server anhand der sichtbaren GPUs (Name und VRAM). Eine Aenderung laedt die Pipeline sofort auf dem neuen Geraet; eine nicht (mehr) vorhandene GPU wird als klarer Fehler gemeldet statt still auf ein anderes Geraet auszuweichen. Alte Settings-Dateien ohne das Feld laufen als `auto` weiter.
- **Neu (Modellstatus im Dashboard):** Ladezustand (`not_loaded`/`loading`/`loaded`/`error`), Geraet, Herkunft des Hugging-Face-Tokens und der konkrete Ladefehler stehen jetzt im Dashboard, in `GET /api/admin/settings`/`task` und in Benchmark-/Speicherfehlern. Bisher landete der Grund nur im Server-Log und das UI meldete nach dem Speichern trotzdem "runtime reloaded". Ladefehler bekommen Hinweise fuer die typischen Ursachen: abgelehnter Gated-Zugriff (Bedingungen mit dem Token-Konto akzeptieren, Fine-grained-Token-Berechtigung), Hugging Face aus dem Container nicht erreichbar, Token ohne `hf_`-Praefix (pyannote verwirft solche Tokens stillschweigend) und fehlende GPU.
- **Neu:** `POST /api/admin/model/load` und Button "Load Model Now" laden das Modell sofort mit den gespeicherten Einstellungen.
- **Fix (Fehlerdiagnose nachgeschaerft):** Ladefehler werden ueber die ganze Ursachenkette eingeordnet. huggingface_hub meldet z. B. einen 403 fuer einen fine-grained Token ohne Gated-Berechtigung als "check your connection"; das erscheint jetzt als Token-Berechtigungsproblem statt als Netzwerkfehler. Eigene Meldungen gibt es fuer ungueltige Tokens (401), 403 ohne Hugging-Face-Kennung (Proxy/Firewall), unbekannte Modell-IDs, fehlende Dateien, Rate-Limit (429) und Hub-Stoerungen (5xx). Tokens ohne `hf_` werden nicht mehr abgewiesen, sondern nur im Fehlerfall erwaehnt.
- **Fix (unvollstaendiger Cache):** Ein lokaler Snapshot wird nur noch genutzt, wenn alle in `config.yaml` referenzierten Teilmodelle (segmentation/embedding/plda) auf der Platte liegen. Ein abgebrochener Erstdownload blockierte sonst dauerhaft jeden weiteren Ladeversuch.
- **Fix:** Bei `gpu_device = cpu` nimmt DIA die geteilte GPU-Lease nicht mehr (CPU-Laeufe haetten Whisper minutenlang blockiert). Lease-/Geraetefehler landen ebenfalls im Modellstatus, ein veralteter Fehlerstatus wird beim naechsten erfolgreichen Zugriff zurueckgesetzt, und ein TypeError beim Laden wird nicht mehr durch den pyannote-3-Fallback verdeckt.
- **Fix (Dashboard):** Schlaegt der erste Settings-Abruf fehl, wird das Formular beim naechsten Poll befuellt und "Save" bleibt bis dahin gesperrt (sonst haette Speichern Token und Cache-Pfad geleert). Die Ladefehler-Meldung bleibt stehen, bis erneut gespeichert/geladen wird.
- **Fix (Settings-Formular):** Das sekuendliche Dashboard-Polling hat das Settings-Formular jedes Mal mit den gespeicherten Werten ueberschrieben; ein eingefuegter Token oder eine Auswahl ging dadurch vor dem Speichern wieder verloren. Das Formular wird jetzt nur beim ersten Laden und nach dem Speichern befuellt.
- **Neu:** `G3_DIA` als neues G3-Projekt aufgebaut, basierend auf dem alten `genesis2_dia_server_project`, aber mit FastAPI + React Admin-Dashboard nach dem visuellen Vorbild von `G3_WHISPER`.
- **Neu:** Geschuetztes Adminpanel mit `X-Admin-Key`, persistentem Key-Store, temporarem Startup-Key und Key-Rotation im Browser.
- **Neu:** Live-Task-Ansicht fuer laufende pyannote-Diarisierung, inklusive Fortschritt, Worker-Status und Fehleranzeige.
- **Neu:** Persistierte DIA-Settings fuer Cache-Pfad und Hugging Face Token.
- **Neu:** Benchmark-Workflow fuer wiederholte Diarisierungslaeufe.
- **Neu:** Request-History mit Sprecher-/Segment-Summaries und Laufzeiten.
- **Fix:** `omegaconf` ist explizit in `requirements.txt` enthalten, damit pyannote auch im lokalen Runtime-Load stabil startet.
- **Fix:** Polling-Intervall im React-Dashboard von 5s auf 1s reduziert, um flüssigere Live-Fortschrittsanzeige zu gewährleisten.
- **Fix:** PyAnnote Progress-Hook implementiert nun korrekt die `__call__` API (statt veraltetem `on_update`), um Fortschritt während Embeddings, Segmentation etc. erfolgreich ans Dashboard zu leiten.
- **Fix:** OpenAPI Docs Button im Dashboard wird nun korrekt als abgerundeter Secondary-Button gerendert.
- **Fix:** Fehlende `formatVram` Formatter-Definition im React-Frontend behoben, um Abstürze bei Benchmark-Results zu vermeiden.
