"""UAM-TOPOMAT : serveur de demonstration jury (FastAPI local).

Application 100 % hors ligne : aucun CDN, aucune ressource externe. Le
serveur ecoute sur http://127.0.0.1:8050 et expose :

* le tableau de bord (``/``) ;
* le catalogue des 2~000 candidats et les fiches materiaux ;
* le pipeline automatique (bandes, Berry/Chern, J(V), Ragone, impact) ;
* la figure 300~dpi et la fiche technique PDF generee par ``export_report``.

Tous les calculs reutilisent les solveurs quantiques, topologiques et
thermodynamiques certifies de la plateforme.
"""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.export_report import PDF_PATH, export_report
from src.optics.solar_optics import KB_SI, Q_E, T_REF, shockley_queisser, sq_efficiency
from src.pde.carrier_transport import site_electrical_result
from src.quantum.materials_generator import FAMILY_MXENE, load_database
from src.quantum.symbolic_proofs import MXENE_CHAMPION_ID, SOLAR_CHAMPION_ID
from src.thermo.sahel_climate import KELVIN_OFFSET, PVModule, energy_balance
from src.thermo.senegal_grid_impact import (
    FIGURE_PATH,
    StorageTechnology,
    champion_modules,
    conventional_modules,
    evaluate_plant,
    evaluate_site,
    site_by_name,
)
from src.thermo.inverse_discovery import haldane_model_for
from src.thermo.supercapacitor_mxene import AsymmetricSupercapacitor
from src.topology.berry_chern import berry_chern

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
ROOT = APP_DIR.parent
MODULE_FACTOR = 0.75
K_GRID_PATH = 10
BERRY_GRID = 12
JV_POINTS = 17
RAGONE_CURRENTS = (50.0, 100.0, 250.0, 500.0)

app = FastAPI(
    title="UAM-TOPOMAT — Simulateur d'impact Sénégal",
    description="Simulateur d'impact multi-régions et interface de demonstration live.",
    version="2026.2",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class PipelineRequest(BaseModel):
    """Parametres du pipeline automatique lance par le jury."""

    material_id: str = Field(default=SOLAR_CHAMPION_ID, min_length=1)
    temperature_c: float = Field(default=47.0, ge=20.0, le=55.0)
    city: str = Field(default="Matam", min_length=1)


def _json(value: Any) -> Any:
    """Conversion recursive NumPy -> types JSON natifs."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return value


@lru_cache(maxsize=1)
def _catalog() -> tuple[dict, ...]:
    records = load_database()
    return tuple(
        {
            "id": record["id"],
            "formula": record["formula"],
            "family": record["family"],
            "subfamily": record["subfamily"],
            "bandgap_ev": float(record["eg_ev"]),
            "cost_usd_per_kg": float(record["cost_usd_per_kg"]),
            "is_champion": record["id"] in (SOLAR_CHAMPION_ID, MXENE_CHAMPION_ID),
        }
        for record in records
    )


@lru_cache(maxsize=None)
def _material(material_id: str) -> dict:
    records = {record["id"]: record for record in load_database()}
    if material_id not in records:
        raise HTTPException(status_code=404, detail=f"Materiau inconnu : {material_id}")
    return records[material_id]


def _module_from_material(material: dict) -> PVModule:
    gap = float(material["eg_ev"])
    return PVModule(
        name=f"{material['id']} {material['formula']}",
        bandgap_ev=gap,
        beta_t=float(material["beta_t"]),
        eta_ref=float(MODULE_FACTOR * sq_efficiency(gap, T_REF)),
    )


def _band_structure(material: dict) -> dict:
    from src.quantum.materials_generator import build_tight_binding_model

    model = build_tight_binding_model(material)
    vectors = np.asarray(model.lattice.reciprocal_vectors, dtype=float)[:2, :2]
    nodes = ((0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.0))
    fractions: list[tuple[float, float]] = []
    labels: list[str] = []
    for index, (start, end) in enumerate(zip(nodes[:-1], nodes[1:])):
        for step in range(K_GRID_PATH):
            if index and step == 0:
                continue
            weight = step / K_GRID_PATH
            fractions.append(
                (start[0] + weight * (end[0] - start[0]),
                 start[1] + weight * (end[1] - start[1]))
            )
            labels.append("Γ" if step == 0 else "")
    fractions.append(nodes[-1])
    labels.append("Γ")
    kpoints = np.asarray(fractions, dtype=float) @ vectors
    energies = np.asarray(model.band_structure(kpoints), dtype=float)
    path = np.linspace(0.0, 1.0, len(fractions))
    bands = [energies[:, band].tolist() for band in range(energies.shape[1])]
    return {
        "path": path.tolist(),
        "bands": bands,
        "labels": labels,
        "fermi_ev": 0.0,
        "n_bands": int(energies.shape[1]),
        "model": model.name,
        "gap_ev": float(material["eg_ev"]),
    }


def _berry(material: dict) -> dict:
    model = haldane_model_for(material)
    analytic = int(model.chern(0))
    try:
        result = berry_chern(model, model.reciprocal_vectors, n1=BERRY_GRID, n2=BERRY_GRID)
        field = np.asarray(result.field, dtype=float)
        numeric = float(result.chern)
        method = "Fukui-Hatsugai-Suzuki"
    except ValueError:
        field = np.zeros((BERRY_GRID, BERRY_GRID), dtype=float)
        numeric = float(analytic)
        method = "analytique (bande non isolee sur la grille)"
    return {
        "field": field.tolist(),
        "chern": analytic,
        "chern_numeric": numeric,
        "grid": [BERRY_GRID, BERRY_GRID],
        "method": method,
    }


def _jv(material: dict, site: Any) -> dict:
    module = _module_from_material(material)
    try:
        result = site_electrical_result(site, module, n_points=JV_POINTS)
        voltage = np.asarray(result.voltage, dtype=float)
        current = np.asarray(result.current_density, dtype=float) / 10.0
        return {
            "voltage": voltage.tolist(),
            "current_ma_cm2": current.tolist(),
            "voc": float(result.voc),
            "jsc_ma_cm2": float(result.jsc / 10.0),
            "ff": float(result.ff),
            "efficiency_percent": float(100.0 * result.efficiency),
            "method": "Équations de dérive-diffusion",
            "junction_c": float(result.junction_temperature_c),
        }
    except (RuntimeError, ValueError, ZeroDivisionError, FloatingPointError, OverflowError):
        junction_k = float(energy_balance(site, module).junction_temperature_k)
        diode = shockley_queisser(
            float(material["eg_ev"]), junction_k, incident_power=float(site.irradiance)
        )
        thermal_voltage = KB_SI * junction_k / Q_E
        voltage = np.linspace(0.0, diode.voc, 41)
        current = (
            diode.jsc_ma_cm2 - diode.j0_ma_cm2 * np.expm1(voltage / thermal_voltage)
        )
        current = np.maximum(current, 0.0)
        power = voltage * current
        peak = int(np.argmax(power))
        return {
            "voltage": voltage.tolist(),
            "current_ma_cm2": current.tolist(),
            "voc": float(diode.voc),
            "jsc_ma_cm2": float(diode.jsc_ma_cm2),
            "ff": float(power[peak] / (diode.voc * diode.jsc_ma_cm2)),
            "efficiency_percent": float(100.0 * diode.efficiency),
            "method": "Diode radiatif Shockley–Queisser",
            "junction_c": junction_k - KELVIN_OFFSET,
        }


def _ragone(material: dict, site: Any) -> dict:
    cell = AsymmetricSupercapacitor(temperature_k=float(site.ambient_k))
    points = cell.ragone_curve(RAGONE_CURRENTS)
    return {
        "current_a_kg": [float(point.current_a_kg) for point in points],
        "energy_wh_kg": [float(point.energy_wh_kg) for point in points],
        "power_w_kg": [float(point.power_w_kg) for point in points],
        "coulombic_efficiency": [float(point.coulombic_efficiency) for point in points],
        "temperature_c": float(site.ambient_c),
    }


def _storage_for_material(material: dict, champion_storage: StorageTechnology) -> StorageTechnology:
    if material["id"] == MXENE_CHAMPION_ID:
        return champion_storage
    if material["family"] != FAMILY_MXENE:
        return champion_storage
    t1 = max(float(material["t1"]), 0.1)
    lattice_a = max(float(material["a"]), 0.1)
    abundance = float(material["abundance_index"])
    energy = (
        9.0 * (1.0 / t1) ** 0.5 * (3.0 / lattice_a) ** 2
        * champion_storage.energy_density_wh_kg / 5.97398
        * (0.95 + 0.05 * abundance)
    )
    return StorageTechnology(
        name=f"{material['id']} {material['formula']}",
        energy_density_wh_kg=energy,
        roundtrip_efficiency=champion_storage.roundtrip_efficiency,
        cycle_life=champion_storage.cycle_life,
        cost_fcfa_per_kwh=champion_storage.cost_fcfa_per_kwh,
        depth_of_discharge=champion_storage.depth_of_discharge,
    )


@lru_cache(maxsize=None)
def _pipeline(material_id: str, temperature_c: float, city: str) -> dict:
    material = _material(material_id)
    site = site_by_name(city, temperature_c)
    solar, mxene = champion_modules()
    silicon, battery = conventional_modules()
    is_mxene = material["family"] == FAMILY_MXENE
    selected_pv = solar if is_mxene else _module_from_material(material)
    selected_storage = _storage_for_material(material, mxene)
    plant = evaluate_plant(site, selected_pv, selected_storage)
    impact = evaluate_site(site, selected_pv, selected_storage, silicon, battery)
    payload = {
        "request": {
            "material_id": material_id,
            "temperature_c": temperature_c,
            "city": site.name,
        },
        "material": {
            "id": material["id"],
            "formula": material["formula"],
            "family": material["family"],
            "subfamily": material["subfamily"],
            "bandgap_ev": float(material["eg_ev"]),
            "role": "stockage" if is_mxene else "photovoltaique",
            "is_champion": material_id in (SOLAR_CHAMPION_ID, MXENE_CHAMPION_ID),
            "cost_usd_per_kg": float(material["cost_usd_per_kg"]),
        },
        "site": {
            "name": site.name,
            "ambient_c": float(site.ambient_c),
            "irradiance_w_m2": float(site.irradiance),
            "wind_m_s": float(site.wind_speed),
        },
        "band_structure": _band_structure(material),
        "berry": _berry(material),
        "jv": _jv(material, site),
        "ragone": _ragone(material, site),
        "impact": _json(impact.summary()),
        "plant": _json(asdict(plant)),
    }
    return _json(payload)


@app.get("/")
def index() -> FileResponse:
    """Tableau de bord principal (HTML local)."""
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/api/health")
def health() -> dict:
    """Etat du service et informations de charte."""
    return {
        "status": "ok",
        "service": "UAM-TOPOMAT — Simulateur d'impact multi-régions",
        "offline": True,
        "host": "127.0.0.1",
        "port": 8050,
        "n_materials": len(_catalog()),
        "champions": [SOLAR_CHAMPION_ID, MXENE_CHAMPION_ID],
        "author": "Ndiogou Ndiaye",
    }


@app.get("/api/catalog")
def catalog() -> dict:
    """Catalogue complet des 2 000 materiaux candidats."""
    materials = _catalog()
    return {
        "count": len(materials),
        "champions": [SOLAR_CHAMPION_ID, MXENE_CHAMPION_ID],
        "materials": list(materials),
    }


@app.get("/api/material/{material_id}")
def material_detail(material_id: str) -> dict:
    """Fiche complete d'un candidat de la base UAM."""
    return _json(_material(material_id))


@app.get("/api/impact")
def impact_summary(
    city: str = Query(default="Matam"),
    temperature_c: float = Query(default=47.0, ge=20.0, le=55.0),
) -> dict:
    """Bilan energetique Sénégal du couple champion face au couple classique."""
    try:
        site = site_by_name(city, temperature_c)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return _json(evaluate_site(site).summary())


@app.post("/api/pipeline/run")
def run_pipeline(request: PipelineRequest) -> dict:
    """Pipeline complet en un clic : toutes les courbes et l'impact."""
    try:
        return _pipeline(request.material_id, request.temperature_c, request.city)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/chart-data/{material_id}")
def chart_data(
    material_id: str,
    temperature_c: float = Query(default=47.0, ge=20.0, le=55.0),
    city: str = Query(default="Matam"),
) -> dict:
    """Donnees des graphiques pour un materiau (equivalent GET du pipeline)."""
    try:
        return _pipeline(material_id, temperature_c, city)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/report/pdf")
def report_pdf() -> FileResponse:
    """Fiche technique PDF officielle generee par src/export_report.py."""
    if not PDF_PATH.exists():
        export_report()
    if not PDF_PATH.exists():
        raise HTTPException(status_code=503, detail="Compilateur LaTeX indisponible.")
    return FileResponse(
        PDF_PATH,
        media_type="application/pdf",
        filename="UAM_TopoMat_Fiche_Champions.pdf",
    )


@app.get("/api/figures/impact")
@app.get("/api/figures/night11")
def impact_figure() -> FileResponse:
    """Figure 300 dpi du bilan energetique multi-regions."""
    from src.thermo.senegal_grid_impact import run as run_impact

    if not FIGURE_PATH.exists():
        run_impact()
    return FileResponse(FIGURE_PATH, media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8050, log_level="info")


__all__ = ["app", "PipelineRequest", "health", "catalog", "run_pipeline", "report_pdf"]
