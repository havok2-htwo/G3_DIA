# Lokaler venv-Workflow

`G3_DIA` nutzt einen lokalen Python-Ordner `venv` direkt im Projektordner.
Ein separater Anaconda-Workflow ist fuer das neue G3-DIA-Projekt nicht noetig.

## Erstinstallation

1. Unter Windows `start.bat` ausfuehren oder unter Linux/Unix `bash ./start.sh`.
2. Falls `./venv` noch fehlt, richtet das Setup die virtuelle Umgebung automatisch ein.
3. Anschliessend werden Python-Abhaengigkeiten aus `requirements.txt` und Frontend-Abhaengigkeiten installiert oder aktualisiert.
4. Danach startet der DIA-Server direkt.

## Hinweise

- Die virtuelle Umgebung liegt in `./venv`.
- Falls du lokale PyTorch-Wheels verwenden willst, setze vor dem Setup `TORCH_WHEEL_DIR`.
- Wenn `TORCH_WHEEL_DIR` nicht gesetzt ist, installiert das Setup den PyTorch-Stack per pip.
- Wenn `requirements.txt` geaendert wurde, installiert `start.bat` bzw. `install.sh` die Python-Abhaengigkeiten beim naechsten Lauf erneut.
- Fuer gated pyannote-Modelle muss der Hugging Face Token in den Admin Settings oder in `.env` vorhanden sein.
- `omegaconf` ist bewusst Teil von `requirements.txt`, weil pyannote sonst trotz installiertem Hauptpaket beim Runtime-Load scheitern kann.

## Wichtige Dateien

- `start.bat`: Windows-Setup und Start in einem Schritt
- `start.sh`: Linux/Unix-Start, ruft bei Bedarf vorher `install.sh` auf
- `install.sh`: Linux/Unix-Setup fuer venv, PyTorch, Python-Abhaengigkeiten und Frontend-Build
- `requirements.txt`: Python-Abhaengigkeiten fuer den DIA-Server
