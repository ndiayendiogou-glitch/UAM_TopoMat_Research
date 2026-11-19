@echo off
chcp 65001 >nul
title UAM-TOPOMAT - Nuit 11 - Demonstration Jury
cd /d "%~dp0"

echo ============================================================
echo  UAM - UFR STA - CELLULE PHYSIQUE
echo  Realise par : Ndiogou Ndiaye - Annee universitaire 2026-2027
echo  Nuit 11 : Simulateur d'Impact Senegal (100 kWc)
echo  Interface live : http://127.0.0.1:8050
echo ============================================================
echo.

python -c "import fastapi, uvicorn, httpx" >nul 2>nul
if errorlevel 1 (
    echo Installation des dependances web (fastapi, uvicorn, httpx)...
    python -m pip install fastapi uvicorn httpx
)

if not exist "results\figures\night11_senegal_impact.png" (
    echo Generation de la figure 300 dpi...
    python -m src.thermo.senegal_grid_impact >nul
)

echo Demarrage du serveur en arriere-plan (fenetre "UAM-TOPOMAT-Serveur")...
start "UAM-TOPOMAT-Serveur" /MIN python -m uvicorn app.server:app --host 127.0.0.1 --port 8050

timeout /t 3 /nobreak >nul

echo Attente de la disponibilite de http://127.0.0.1:8050 ...
set /a UAM_TRIES=0
:UAM_WAIT_READY
set /a UAM_TRIES+=1
python -c "import sys, urllib.request; o = urllib.request.build_opener(urllib.request.ProxyHandler({})); sys.exit(0 if o.open('http://127.0.0.1:8050/api/health', timeout=1).status == 200 else 1)" >nul 2>nul
if not errorlevel 1 goto UAM_READY
if %UAM_TRIES% GEQ 30 goto UAM_FAILED
timeout /t 1 /nobreak >nul
goto UAM_WAIT_READY

:UAM_READY
start "" "http://127.0.0.1:8050"
echo.
echo Le serveur tourne en arriere-plan (fenetre "UAM-TOPOMAT-Serveur").
echo Le navigateur a ete ouvert sur http://127.0.0.1:8050
echo Fermez la fenetre du serveur pour arreter la demonstration.
echo.
pause
exit /b 0

:UAM_FAILED
echo.
echo ERREUR : le serveur ne repond pas apres 30 secondes.
echo Verifiez la fenetre "UAM-TOPOMAT-Serveur" et le port 8050.
echo.
pause
exit /b 1
