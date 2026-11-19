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

start "UAM-TOPOMAT-Serveur" python -m uvicorn app.server:app --host 127.0.0.1 --port 8050
timeout /t 5 /nobreak >nul
start "" http://127.0.0.1:8050

echo.
echo Le serveur tourne dans la fenetre "UAM-TOPOMAT-Serveur".
echo Le navigateur a ete ouvert sur http://127.0.0.1:8050
echo Fermez la fenetre du serveur pour arreter la demonstration.
echo.
pause
