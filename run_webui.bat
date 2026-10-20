@echo off
cd /d "%~dp0"
if exist "%~dp0venv\Scripts\python.exe" (
  "%~dp0venv\Scripts\python.exe" scripts\webui_lora.py
) else (
  python scripts\webui_lora.py
)
pause
