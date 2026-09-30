# Changelog

## Unreleased

- **Neu (GPU-Auswahl):** Einstellung `gpu_device` (`auto`, `cpu`, `cuda:N`) mit Dropdown im Admin-Dashboard; die Optionen listet der Server anhand der sichtbaren GPUs (Name und VRAM). Eine Aenderung laedt die Pipeline sofort auf dem neuen Geraet; eine nicht (mehr) vorhandene GPU wird als klarer Fehler gemeldet statt still auf ein anderes Geraet auszuweichen. Alte Settings-Dateien ohne das Feld laufen als `auto` weiter.
- **Neu (Modellstatus im Dashboard):** Ladezustand (`not_loaded`/`loading`/`loaded`/`error`), Geraet, Herkunft des Hugging-Face-Tokens und der konkrete Ladefehler stehen jetzt im Dashboard, in `GET /api/admin/settings`/`task` und in Benchmark-/Speicherfehlern. Bisher landete der Grund nur im Server-Log und das UI meldete nach dem Speichern trotzdem "runtime reloaded". Ladefehler bekommen Hinweise fuer die typischen Ursachen: abgelehnter Gated-Zugriff (Bedingungen mit dem Token-Konto akzeptieren, Fine-grained-Token-Berechtigung), Hugging Face aus dem Container nicht erreichbar, Token ohne `hf_`-Praefix (pyannote verwirft solche Tokens stillschweigend) und fehlende GPU.
- **Neu:** `POST /api/admin/model/load` und Button "Load Model Now" laden das Modell sofort mit den gespeicherten Einstellungen.
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
