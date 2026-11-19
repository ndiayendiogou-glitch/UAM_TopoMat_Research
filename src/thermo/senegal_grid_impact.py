"""UAM-TOPOMAT -- Nuit 11 : simulateur d'impact energetique au Senegal.

Mini-centrale solaire de 100~kWc associee a un stockage hybride, deployee a
Dakar (31~degC), Saint-Louis (37~degC) et Matam (47~degC). Le couple champion
UAM-1384 (K2AgBiI6) + UAM-1517 (Ti3C2O2) est compare au couple classique
Silicium + batterie Plomb/Li-ion.

Le modele reutilise explicitement les modules des nuits precedentes :

* rendement Shockley--Queisser et spectre AM1.5G (Nuit 4) ;
* bilan radiatif--convectif et production annuelle (Nuit 5) ;
* donneeselectriques J(V) de la Nuit 6 pour le controle de qualite ;
* densite d'energie du supercondensateur MXene (Nuit 7) ;
* base de 2~000 candidats et champions certifies (Nuits 8--10).

Hypotheses economiques (documentees et deterministes) : tarif SENELEC de
220~FCFA/kWh, facteur de emission du reseau 0,55~t~CO2/MWh, change fixe
600~FCFA/USD, taux d'actualisation 5~% sur 20~ans, capacite de stockage
100~kWh (une heure d'autonomie). Le cout du kWh est un LCOE actualise
incluant CAPEX, O&M et renouvellement des batteries selon leur nombre de
cycles. Python pur : numpy, sympy, matplotlib. Aucune dependance SciPy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import sympy as sp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.export_report import load_champions
from src.optics.solar_optics import T_REF, sq_efficiency
from src.quantum.symbolic_proofs import MXENE_CHAMPION_ID, SOLAR_CHAMPION_ID
from src.thermo.sahel_climate import (
    DAKAR,
    KELVIN_OFFSET,
    MATAM,
    SAINT_LOUIS,
    PVModule,
    SahelSite,
    annual_yield,
    silicon_module,
)

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURE_PATH = FIGURES_DIR / "night11_senegal_impact.png"
JSON_PATH = RESULTS_DIR / "senegal_impact.json"

PLANT_CAPACITY_KWC = 100.0
REFERENCE_IRRADIANCE_W_M2 = 1000.0
STORAGE_CAPACITY_KWH = 100.0
FCFA_PER_USD = 600.0
PV_MASS_KG_PER_KWC = 6.0
BOS_FCFA_PER_KWC = 450_000.0
SILICON_MATERIAL_COST_USD_PER_KG = 3.0
GRID_TARIFF_FCFA_PER_KWH = 220.0
GRID_CO2_KG_PER_KWH = 550.0
GRID_CO2_TONNES_PER_MWH = GRID_CO2_KG_PER_KWH / 1000.0
DISCOUNT_RATE = 0.05
PROJECT_YEARS = 20
PV_OM_FRACTION = 0.02
STORAGE_OM_FRACTION = 0.01
DEFAULT_SITES: tuple[SahelSite, ...] = (DAKAR, SAINT_LOUIS, MATAM)


@dataclass(frozen=True)
class StorageTechnology:
    """Technologie de stockage : rendement, cyclabilite et cout."""

    name: str
    energy_density_wh_kg: float
    roundtrip_efficiency: float
    cycle_life: int
    cost_fcfa_per_kwh: float
    depth_of_discharge: float

    def __post_init__(self) -> None:
        if not 0.0 < self.roundtrip_efficiency <= 1.0:
            raise ValueError("Le rendement aller-retour doit etre dans (0, 1].")
        if self.cycle_life <= 0 or self.cost_fcfa_per_kwh < 0.0:
            raise ValueError("Cyclabilite et cout de stockage invalides.")
        if not 0.0 < self.depth_of_discharge <= 1.0:
            raise ValueError("La profondeur de decharge doit etre dans (0, 1].")


@dataclass(frozen=True)
class PlantResult:
    """Bilan annuel d'une mini-centrale pour un site donne."""

    site_name: str
    ambient_c: float
    pv_name: str
    storage_name: str
    capacity_kwc: float
    area_m2: float
    incident_mwh: float
    pv_energy_mwh: float
    usable_energy_mwh: float
    storage_mass_kg: float
    cycles_per_year: float
    pv_capex_fcfa: float
    storage_capex_fcfa: float
    annualized_capex_fcfa: float
    replacement_fcfa: float
    om_fcfa: float
    total_annual_cost_fcfa: float
    lcoe_fcfa_per_kwh: float
    avoided_grid_cost_fcfa: float
    co2_avoided_tonnes: float
    mean_junction_c: float

    @property
    def net_benefit_fcfa(self) -> float:
        return float(self.avoided_grid_cost_fcfa - self.total_annual_cost_fcfa)


@dataclass(frozen=True)
class ImpactResult:
    """Comparaison champion / classique sur un site sahelien."""

    site_name: str
    ambient_c: float
    champion: PlantResult
    baseline: PlantResult

    @property
    def gain_mwh(self) -> float:
        return float(self.champion.usable_energy_mwh - self.baseline.usable_energy_mwh)

    @property
    def relative_gain_percent(self) -> float:
        return 100.0 * float(self.gain_mwh) / float(self.baseline.usable_energy_mwh)

    @property
    def co2_avoided_tonnes(self) -> float:
        return float(self.gain_mwh * GRID_CO2_TONNES_PER_MWH)

    @property
    def savings_fcfa(self) -> float:
        return float(
            self.gain_mwh * 1000.0 * GRID_TARIFF_FCFA_PER_KWH
            + self.baseline.total_annual_cost_fcfa
            - self.champion.total_annual_cost_fcfa
        )

    @property
    def lcoe_reduction_fcfa_per_kwh(self) -> float:
        return float(self.baseline.lcoe_fcfa_per_kwh - self.champion.lcoe_fcfa_per_kwh)

    @property
    def lcoe_reduction_percent(self) -> float:
        return 100.0 * float(self.lcoe_reduction_fcfa_per_kwh) / float(
            self.baseline.lcoe_fcfa_per_kwh
        )

    def summary(self) -> dict:
        return {
            "site": self.site_name,
            "ambient_c": self.ambient_c,
            "champion_mwh_an": self.champion.usable_energy_mwh,
            "classique_mwh_an": self.baseline.usable_energy_mwh,
            "gain_mwh_an": self.gain_mwh,
            "gain_percent": self.relative_gain_percent,
            "co2_evitees_t_an": self.co2_avoided_tonnes,
            "economies_fcfa_an": self.savings_fcfa,
            "lcoe_champion_fcfa_kwh": self.champion.lcoe_fcfa_per_kwh,
            "lcoe_classique_fcfa_kwh": self.baseline.lcoe_fcfa_per_kwh,
            "lcoe_reduction_fcfa_kwh": self.lcoe_reduction_fcfa_per_kwh,
            "lcoe_reduction_percent": self.lcoe_reduction_percent,
        }


@dataclass(frozen=True)
class SenegalImpact:
    """Bilan national : trois sites et agregats deterministes."""

    sites: dict[str, ImpactResult]
    champions: dict[str, str]
    assumptions: dict[str, float | int | str]
    champion_pv: PVModule
    baseline_pv: PVModule
    champion_storage: StorageTechnology
    baseline_storage: StorageTechnology

    @property
    def total_gain_mwh(self) -> float:
        return float(sum(result.gain_mwh for result in self.sites.values()))

    @property
    def total_co2_tonnes(self) -> float:
        return float(sum(result.co2_avoided_tonnes for result in self.sites.values()))

    @property
    def total_savings_fcfa(self) -> float:
        return float(sum(result.savings_fcfa for result in self.sites.values()))

    def aggregate(self) -> dict:
        return {
            "gain_mwh_an": self.total_gain_mwh,
            "co2_evitees_t_an": self.total_co2_tonnes,
            "economies_fcfa_an": self.total_savings_fcfa,
            "mwh_champion": float(
                sum(result.champion.usable_energy_mwh for result in self.sites.values())
            ),
            "mwh_classique": float(
                sum(result.baseline.usable_energy_mwh for result in self.sites.values())
            ),
        }


def annuity_factor(rate: float = DISCOUNT_RATE, years: int = PROJECT_YEARS) -> float:
    """Facteur d'annuite CRF = r(1+r)^n / ((1+r)^n - 1)."""
    rate, years = float(rate), int(years)
    if rate < 0.0 or years < 1:
        raise ValueError("Taux et duree d'actualisation invalides.")
    if rate == 0.0:
        return 1.0 / years
    growth = (1.0 + rate) ** years
    return float(rate * growth / (growth - 1.0))


def champion_modules() -> tuple[PVModule, StorageTechnology]:
    """Modules champion : cellule UAM-1384 et supercondensateur UAM-1517."""
    payload = load_champions()
    records = {record["id"]: record for record in payload["top10_champions"]}
    solar = records[SOLAR_CHAMPION_ID]
    mxene = records[MXENE_CHAMPION_ID]
    pv = PVModule(
        name=f"{SOLAR_CHAMPION_ID} {solar['formula']}",
        bandgap_ev=float(solar["bandgap_ev"]),
        beta_t=float(solar["thermal_drift_beta_per_k"]),
        eta_ref=float(0.75 * sq_efficiency(float(solar["bandgap_ev"]), T_REF)),
    )
    storage = StorageTechnology(
        name=f"{MXENE_CHAMPION_ID} {mxene['formula']} (hybride MXene)",
        energy_density_wh_kg=float(mxene["energy_density_matam_wh_kg"]),
        roundtrip_efficiency=0.96,
        cycle_life=500_000,
        cost_fcfa_per_kwh=250_000.0,
        depth_of_discharge=0.95,
    )
    return pv, storage


def conventional_modules() -> tuple[PVModule, StorageTechnology]:
    """Couple classique : silicium cristallin et batterie Plomb/Li-ion 50/50."""
    storage = StorageTechnology(
        name="Batterie Plomb/Li-ion (50/50)",
        energy_density_wh_kg=0.5 * (45.0 + 180.0),
        roundtrip_efficiency=0.5 * (0.80 + 0.92),
        cycle_life=int(round(0.5 * (1500 + 4000))),
        cost_fcfa_per_kwh=0.5 * (120_000.0 + 180_000.0),
        depth_of_discharge=0.5 * (0.50 + 0.90),
    )
    return silicon_module(), storage


def site_at_temperature(site: SahelSite, temperature_c: float) -> SahelSite:
    """Site climatique avec temperature ambiante ajustee (25 a 55 degC)."""
    temperature = float(temperature_c)
    if not 25.0 <= temperature <= 55.0:
        raise ValueError("La temperature sahélienne doit etre comprise entre 25 et 55 degC.")
    return SahelSite(
        name=site.name,
        ambient_c=temperature,
        irradiance=float(site.irradiance),
        wind_speed=float(site.wind_speed),
        seasonal_amplitude_k=float(site.seasonal_amplitude_k),
    )


def site_by_name(name: str, temperature_c: float | None = None) -> SahelSite:
    """Retourne Dakar, Saint-Louis ou Matam, avec override de temperature."""
    sites = {site.name.lower(): site for site in DEFAULT_SITES}
    aliases = {"dakar": "dakar", "saint-louis": "saint-louis", "saint_louis": "saint-louis", "matam": "matam"}
    key = aliases.get(str(name).strip().lower())
    if key is None:
        raise ValueError(f"Site inconnu : {name!r}.")
    base = sites[key]
    return site_at_temperature(base, temperature_c) if temperature_c is not None else base


def evaluate_plant(
    site: SahelSite,
    module: PVModule,
    storage: StorageTechnology,
    *,
    capacity_kwc: float = PLANT_CAPACITY_KWC,
    storage_capacity_kwh: float = STORAGE_CAPACITY_KWH,
) -> PlantResult:
    """Production, cout actualise et emissions d'une mini-centrale."""
    capacity = float(capacity_kwc)
    storage_kwh = float(storage_capacity_kwh)
    if capacity <= 0.0 or storage_kwh <= 0.0:
        raise ValueError("Puissance et capacite de stockage doivent etre positives.")
    yield_data = annual_yield(site, module)
    area_m2 = capacity * 1000.0 / (
        REFERENCE_IRRADIANCE_W_M2 * float(module.eta_ref)
    )
    pv_energy_mwh = float(yield_data.kwh_per_m2) * area_m2 / 1000.0
    usable_mwh = pv_energy_mwh * float(storage.roundtrip_efficiency)
    capacity_kwh = usable_mwh * 1000.0
    cycles_per_year = capacity_kwh / storage_kwh

    pv_material_cost = (
        _pv_material_cost_usd_per_kg(module) * FCFA_PER_USD * PV_MASS_KG_PER_KWC
    )
    pv_capex = capacity * (BOS_FCFA_PER_KWC + pv_material_cost)
    storage_capex = storage_kwh * storage.cost_fcfa_per_kwh
    crf = annuity_factor()
    annualized = crf * (pv_capex + storage_capex)
    replacement = storage_capex * cycles_per_year / float(storage.cycle_life)
    om = PV_OM_FRACTION * pv_capex + STORAGE_OM_FRACTION * storage_capex
    total_cost = annualized + replacement + om
    lcoe = total_cost / capacity_kwh
    storage_mass = storage_kwh * 1000.0 / (
        float(storage.energy_density_wh_kg) * float(storage.depth_of_discharge)
    )
    return PlantResult(
        site_name=site.name,
        ambient_c=float(site.ambient_c),
        pv_name=module.name,
        storage_name=storage.name,
        capacity_kwc=capacity,
        area_m2=area_m2,
        incident_mwh=float(yield_data.incident_kwh_per_m2) * area_m2 / 1000.0,
        pv_energy_mwh=pv_energy_mwh,
        usable_energy_mwh=usable_mwh,
        storage_mass_kg=storage_mass,
        cycles_per_year=cycles_per_year,
        pv_capex_fcfa=pv_capex,
        storage_capex_fcfa=storage_capex,
        annualized_capex_fcfa=annualized,
        replacement_fcfa=replacement,
        om_fcfa=om,
        total_annual_cost_fcfa=total_cost,
        lcoe_fcfa_per_kwh=lcoe,
        avoided_grid_cost_fcfa=capacity_kwh * GRID_TARIFF_FCFA_PER_KWH,
        co2_avoided_tonnes=usable_mwh * GRID_CO2_TONNES_PER_MWH,
        mean_junction_c=float(yield_data.mean_junction_c),
    )


def _pv_material_cost_usd_per_kg(module: PVModule) -> float:
    """Cout materiau USD/kg : champion certifie ou silicium de reference."""
    if module.name.startswith(SOLAR_CHAMPION_ID):
        payload = load_champions()
        record = next(
            record for record in payload["top10_champions"]
            if record["id"] == SOLAR_CHAMPION_ID
        )
        return float(record["cost_usd_per_kg"])
    return SILICON_MATERIAL_COST_USD_PER_KG


def evaluate_site(
    site: SahelSite,
    champion_pv: PVModule | None = None,
    champion_storage: StorageTechnology | None = None,
    baseline_pv: PVModule | None = None,
    baseline_storage: StorageTechnology | None = None,
) -> ImpactResult:
    """Compare les deux couples energetiques sur un site."""
    solar, mxene = champion_modules() if champion_pv is None or champion_storage is None else (champion_pv, champion_storage)
    silicon, battery = conventional_modules() if baseline_pv is None or baseline_storage is None else (baseline_pv, baseline_storage)
    champion = evaluate_plant(site, solar, mxene)
    baseline = evaluate_plant(site, silicon, battery)
    return ImpactResult(
        site_name=site.name,
        ambient_c=float(site.ambient_c),
        champion=champion,
        baseline=baseline,
    )


def evaluate_all_sites(sites: Sequence[SahelSite] = DEFAULT_SITES) -> SenegalImpact:
    """Bilan complet sur les trois regions du Senegal."""
    solar, mxene = champion_modules()
    silicon, battery = conventional_modules()
    results = {
        site.name: evaluate_site(site, solar, mxene, silicon, battery)
        for site in sites
    }
    payload = load_champions()
    return SenegalImpact(
        sites=results,
        champions={
            "solar": SOLAR_CHAMPION_ID,
            "storage": MXENE_CHAMPION_ID,
            "solar_formula": solar.name.split(" ", 1)[-1],
            "storage_formula": mxene.name.split(" ", 2)[-1],
            "top10_size": str(len(payload["top10_champions"])),
        },
        assumptions={
            "plant_capacity_kwc": PLANT_CAPACITY_KWC,
            "storage_capacity_kwh": STORAGE_CAPACITY_KWH,
            "grid_tariff_fcfa_per_kwh": GRID_TARIFF_FCFA_PER_KWH,
            "grid_co2_kg_per_kwh": GRID_CO2_KG_PER_KWH,
            "fcfa_per_usd": FCFA_PER_USD,
            "discount_rate": DISCOUNT_RATE,
            "project_years": PROJECT_YEARS,
        },
        champion_pv=solar,
        baseline_pv=silicon,
        champion_storage=mxene,
        baseline_storage=battery,
    )


def export_json(impact: SenegalImpact, path: Path = JSON_PATH) -> dict:
    """Ecrit le bilan structure (JSON) consomme par l'interface web."""
    payload = {
        "metadata": {
            "project": "UAM-TOPOMAT",
            "night": 11,
            "title": "Impact Senegal : mini-centrale 100 kWc + stockage hybride",
            "charter": "Universite Amadou Mahtar Mbow / UFR STA / Cellule Physique",
            "author": "Ndiogou Ndiaye",
            "year": "2026-2027",
            "sites": [site.name for site in DEFAULT_SITES],
            "champions": impact.champions,
            "assumptions": impact.assumptions,
        },
        "sites": {name: result.summary() for name, result in impact.sites.items()},
        "aggregate": impact.aggregate(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def make_figure(impact: SenegalImpact, path: Path = FIGURE_PATH) -> dict:
    """Figure de publication 300 dpi : production, CO2, LCOE et economies."""
    names = list(impact.sites)
    positions = np.arange(len(names))
    width = 0.36
    champion_mwh = np.array([impact.sites[n].champion.usable_energy_mwh for n in names])
    baseline_mwh = np.array([impact.sites[n].baseline.usable_energy_mwh for n in names])
    gains = np.array([impact.sites[n].gain_mwh for n in names])
    co2 = np.array([impact.sites[n].co2_avoided_tonnes for n in names])
    lcoe_c = np.array([impact.sites[n].champion.lcoe_fcfa_per_kwh for n in names])
    lcoe_b = np.array([impact.sites[n].baseline.lcoe_fcfa_per_kwh for n in names])
    savings = np.array([impact.sites[n].savings_fcfa / 1.0e6 for n in names])

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })
    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.6))
    colors = {"champion": "#0b6e4f", "classique": "#9ecae1"}

    ax = axes[0, 0]
    ax.bar(positions - width / 2, champion_mwh, width, color=colors["champion"],
           edgecolor="#053b2a", label="UAM-1384 + UAM-1517")
    ax.bar(positions + width / 2, baseline_mwh, width, color=colors["classique"],
           edgecolor="#08306b", label="Silicium + Plomb/Li-ion")
    for position, value in zip(positions, champion_mwh):
        ax.text(position - width / 2, value, f"{value:.0f}", ha="center",
                va="bottom", fontsize=7.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{n}\n{impact.sites[n].ambient_c:.0f} °C" for n in names])
    ax.set_ylabel("MWh/an utiles")
    ax.set_title("(a) Production electrique annuelle (100 kWc)", fontsize=11)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="upper left", fontsize=7.5)

    ax = axes[0, 1]
    bars = ax.bar(positions, gains, width=0.5, color="#e6550d", edgecolor="#893b00")
    for bar, value in zip(bars, gains):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"+{value:.1f} MWh",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(positions)
    ax.set_xticklabels(names)
    ax.set_ylabel("Gain annuel (MWh/an)")
    ax.set_title("(b) Gain de production du couple champion", fontsize=11)
    ax.grid(alpha=0.25, axis="y")
    twin = ax.twinx()
    twin.plot(positions, co2, "o-", color="#08306b", lw=2.0, ms=6,
              label="CO2 evitees (t/an)")
    twin.set_ylabel("CO2 evitees (t/an)", color="#08306b")
    twin.legend(loc="upper right", fontsize=7.5)

    ax = axes[1, 0]
    ax.bar(positions - width / 2, lcoe_c, width, color=colors["champion"],
           edgecolor="#053b2a", label="Champion")
    ax.bar(positions + width / 2, lcoe_b, width, color=colors["classique"],
           edgecolor="#08306b", label="Classique")
    for position, value in zip(positions, lcoe_c):
        ax.text(position - width / 2, value, f"{value:.1f}", ha="center",
                va="bottom", fontsize=7.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(names)
    ax.set_ylabel("FCFA/kWh (LCOE actualise)")
    ax.set_title("(c) Cout actualise du kWh", fontsize=11)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="upper left", fontsize=7.5)

    ax = axes[1, 1]
    bars = ax.bar(positions, savings, width=0.5, color="#238b45", edgecolor="#145634")
    for bar, value in zip(bars, savings):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f} M FCFA",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(positions)
    ax.set_xticklabels(names)
    ax.set_ylabel("Millions de FCFA/an")
    ax.set_title("(d) Economie financiere annuelle", fontsize=11)
    ax.grid(alpha=0.25, axis="y")
    total = impact.aggregate()
    ax.text(
        0.02, 0.94,
        f"Total : +{total['gain_mwh_an']:.0f} MWh/an\n"
        f"{total['co2_evitees_t_an']:.0f} t CO2/an\n"
        f"{total['economies_fcfa_an'] / 1.0e6:.0f} M FCFA/an",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", fc="#f0f7f4", ec="#0b6e4f"),
    )

    figure.suptitle(
        "UAM-TOPOMAT — Nuit 11 : simulateur d'impact Senegal\n"
        "Universite Amadou Mahtar Mbow / UFR STA / Cellule Physique — Ndiogou Ndiaye",
        fontsize=12.5, y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {
        "path": str(path),
        "dpi": 300,
        "sites": names,
        "gain_mwh_an": float(np.sum(gains)),
        "co2_evitees_t_an": float(np.sum(co2)),
        "economies_fcfa_an": float(total["economies_fcfa_an"]),
    }


def symbolic_impact_laws() -> dict[str, sp.Expr]:
    """Certifications SymPy des lois economiques du simulateur."""
    rate, years, cost, energy, gain, tariff, factor = sp.symbols(
        "r n C E g tarif facteur", positive=True
    )
    crf = rate * (1 + rate) ** years / ((1 + rate) ** years - 1)
    crf_alt = rate / (1 - (1 + rate) ** (-years))
    lcoe = cost / energy
    co2 = gain * factor
    savings = gain * tariff
    return {
        "crf": crf,
        "crf_identity": sp.simplify(crf - crf_alt),
        "lcoe_identity": sp.simplify(lcoe * energy - cost),
        "co2_derivative": sp.simplify(sp.diff(co2, gain) - factor),
        "savings_derivative": sp.simplify(sp.diff(savings, gain) - tariff),
        "savings_zero": sp.simplify(savings.subs(gain, 0)),
    }


@lru_cache(maxsize=1)
def run() -> SenegalImpact:
    """Execute la Nuit 11 : calculs, figure 300 dpi et export JSON."""
    impact = evaluate_all_sites()
    make_figure(impact)
    export_json(impact)
    return impact


if __name__ == "__main__":
    result = run()
    print(json.dumps({
        "sites": {name: item.summary() for name, item in result.sites.items()},
        "aggregate": result.aggregate(),
        "figure": str(FIGURE_PATH),
    }, ensure_ascii=False, indent=2))


__all__ = [
    "ROOT",
    "FIGURE_PATH",
    "JSON_PATH",
    "PLANT_CAPACITY_KWC",
    "GRID_TARIFF_FCFA_PER_KWH",
    "GRID_CO2_TONNES_PER_MWH",
    "StorageTechnology",
    "PlantResult",
    "ImpactResult",
    "SenegalImpact",
    "annuity_factor",
    "champion_modules",
    "conventional_modules",
    "site_at_temperature",
    "site_by_name",
    "evaluate_plant",
    "evaluate_site",
    "evaluate_all_sites",
    "export_json",
    "make_figure",
    "symbolic_impact_laws",
    "run",
]
