# Changelog

## Unreleased

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
