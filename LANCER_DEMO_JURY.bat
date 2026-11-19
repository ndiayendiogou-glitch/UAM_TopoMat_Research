@echo off
cd /d "%~dp0"
echo Demarrage du serveur UAM-TOPOMAT sur http://127.0.0.1:8050 ...
start "" cmd /c "timeout /t 3 /nobreak > nul && start http://127.0.0.1:8050"
python -m uvicorn app.server:app --host 127.0.0.1 --port 8050
pause
