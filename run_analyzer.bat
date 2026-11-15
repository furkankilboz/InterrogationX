@echo off
cd /d "%~dp0"

rem Packages are installed once into a local .venv; the marker file is written only after a successful install.
if not exist ".venv\installed.marker" (
    if not exist ".venv\Scripts\python.exe" (
        echo Creating virtual environment...
        python -m venv .venv || goto :error
    )
    echo Installing required packages, first run only...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error
    echo ok> ".venv\installed.marker"
)

echo Starting the application...
".venv\Scripts\python.exe" interrogation_analyzer_gui.py
pause
exit /b 0

:error
echo Setup failed. See the messages above.
pause
exit /b 1
