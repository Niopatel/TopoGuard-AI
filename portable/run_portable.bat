@echo off
setlocal
cd /d %~dp0

if not exist .venv\Scripts\python.exe (
  echo [setup] Creating portable venv with Python 3.13...
  py -3.13 -m venv .venv
)

if not exist .venv\Scripts\python.exe (
  echo [error] .venv not created. Ensure Python 3.13 is installed.
  exit /b 1
)

if not exist .venv\Lib\site-packages\streamlit (
  echo [setup] Installing dependencies...
  .\.venv\Scripts\python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [error] Dependency install failed.
    exit /b 1
  )
) else (
  echo [setup] Dependencies already present.
)

echo [run] Starting Streamlit (portable)...
.\.venv\Scripts\python -m streamlit run ultra_power_app.py --client.toolbarMode=minimal
