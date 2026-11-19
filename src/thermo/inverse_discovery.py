"""UAM-TOPOMAT -- Nuit 9 : decouverte inverse et optimisation multi-objectifs.

Pipeline deterministe (graine 2026) d'evaluation multiphysique des 2000
candidats de la base de la Nuit 8 sous les conditions extremes de Matam
(47 C, 1050 W/m^2) et Saint-Louis (37 C, 980 W/m^2), puis optimisation
multi-objectifs NSGA-II en NumPy pur (aucune dependance a SciPy).

Quatre objectifs (les trois premiers a maximiser, le dernier a minimiser) :

    f1 = rendement electrique reel a Matam (ou score energie/retention MXene),
    f2 = indice de stabilite thermique a 47 C (> 95 %),
    f3 = indice d'abondance des ressources d'Afrique de l'Ouest,
    f4 = cout estime de synthese (USD/kg).

Les solveurs des Nuits 1 a 7 sont reutilises de facon vectorisee : spectre
AM1.5G et bilan de photons (Nuit 4), bilan thermique de jonction (Nuit 5),
diode radiatif Shockley-Queisser (Nuit 4/6), retention des electrodes MXene
(Nuit 7) et invariant de Chern de Haldane (Nuit 2). Le tri non domine est
exhaustif sur les 2000 candidats, ce qui garantit une non-dominance stricte
du front exporte ; NSGA-II fournit une verification de convergence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import sympy as sp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.optics.solar_optics import (
    C_SI,
    H_SI,
    HC_EV_NM,
    KB_SI,
    P_INPUT,
    Q_E,
    SOLAR_IRRADIANCE,
    T_REF,
    _SOLAR_GRID,
    photon_flux_above_gap,
    shockley_queisser,
    sq_efficiency,
)
from src.quantum.materials_generator import FAMILY_MXENE, load_database
from src.thermo.sahel_climate import (
    H_NATURAL,
    H_WIND,
    KELVIN_OFFSET,
    MATAM,
    SAINT_LOUIS,
    SIGMA_SB,
    SKY_DEPRESSION_K,
    energy_balance,
    silicon_module,
)
from src.thermo.supercapacitor_mxene import AsymmetricSupercapacitor
from src.topology.models_topo import HaldaneModel

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CHAMPIONS_PATH = RESULTS_DIR / "top10_champions.json"
FIGURE_PATH = FIGURES_DIR / "night9_pareto_discovery.png"

SEED = 2026
N_MATERIALS = 2000
MODULE_FACTOR = 0.75
EQE = 0.92
ABSORPTANCE_ABOVE = 0.95
ABSORPTANCE_BELOW = 0.60
EMISSIVITY = 0.90
TOL_THERMAL = 1.0e-12
MAX_THERMAL_ITER = 120
SQ_ENERGY_POINTS = 1500
MXENE_ENERGY_REFERENCE = 9.0
MXENE_T_REFERENCE = 1.0
MXENE_A_REFERENCE = 3.0

MATAM_AMBIENT_C = float(MATAM.ambient_c)
MATAM_IRRADIANCE = float(MATAM.irradiance)
SAINT_LOUIS_AMBIENT_C = float(SAINT_LOUIS.ambient_c)
SAINT_LOUIS_IRRADIANCE = float(SAINT_LOUIS.irradiance)
HEAT_TEST_DELTA_K = MATAM_AMBIENT_C - 25.0

OBJECTIVE_NAMES = (
    "performance",
    "thermal_stability",
    "abundance",
    "negative_cost",
)
OBJECTIVE_DIRECTIONS = ("max", "max", "max", "min")
OBJECTIVE_LABELS = (
    "Performance (eta Matam / score MXene)",
    "Stabilite thermique a 47 C",
    "Abondance ouest-africaine",
    "Cout de synthese (USD/kg, a minimiser)",
)
WEIGHTS = np.array([0.40, 0.15, 0.25, 0.20], dtype=float)


@dataclass(frozen=True)
class ScreeningResult:
    """Resultat de l'evaluation multiphysique exhaustive de la base."""

    materials: list[dict]
    features: dict[str, np.ndarray]
    objectives: np.ndarray
    silicon: dict[str, float]

    @property
    def n_materials(self) -> int:
        return len(self.materials)

    def values(self, key: str) -> np.ndarray:
        return np.asarray(self.features[key], dtype=float)

    def record(self, index: int, *, rank: int | None = None) -> dict:
        material = self.materials[index]
        f = {key: float(np.asarray(value)[index]) for key, value in self.features.items()}
        gain = 100.0 * (f["eta_matam"] / self.silicon["eta_matam"] - 1.0)
        objective_values = {
            OBJECTIVE_NAMES[k]: float(self.objectives[index, k])
            for k in range(len(OBJECTIVE_NAMES))
        }
        record = {
            "id": material["id"],
            "formula": material["formula"],
            "chemical_formula": material["formula"],
            "family": material["family"],
            "structure": material["subfamily"],
            "subfamily": material["subfamily"],
            "elements": list(material["elements"]),
            "composition": dict(material["composition"]),
            "net_charge": float(material["net_charge"]),
            "lattice_a_angstrom": float(f["lattice_a"]),
            "lattice_c_angstrom": float(f["lattice_c"]),
            "hopping_t1_ev": float(f["t1"]),
            "hopping_t2_ev": float(f["t2"]),
            "lambda_soc_ev": float(f["lambda_soc"]),
            "bandgap_ev": float(f["bandgap"]),
            "mobility_electron": float(f["mu_n"]),
            "mobility_hole": float(f["mu_p"]),
            "thermal_drift_beta_per_k": float(f["beta_t"]),
            "chern_number": int(f["chern"]),
            "junction_temperature_matam_c": float(f["tj_matam"] - KELVIN_OFFSET),
            "junction_temperature_saint_louis_c": float(
                f["tj_saint_louis"] - KELVIN_OFFSET
            ),
            "voc_matam_v": float(f["voc_matam"]),
            "jsc_matam_ma_cm2": float(f["jsc_matam"] / 10.0),
            "ff_matam": float(f["ff_matam"]),
            "eta_matam": float(f["eta_matam"]),
            "eta_matam_percent": float(100.0 * f["eta_matam"]),
            "eta_saint_louis_percent": float(100.0 * f["eta_saint_louis"]),
            "gain_vs_silicon_percent": float(gain),
            "performance_score": float(f["performance"]),
            "thermal_stability_percent": float(100.0 * f["stability"]),
            "abundance_index": float(f["abundance"]),
            "cost_usd_per_kg": float(f["cost"]),
            "energy_density_matam_wh_kg": float(f["energy_density"]),
            "capacity_retention_matam_percent": float(100.0 * f["retention"]),
            "device_role": "supercapacitor" if f["is_mxene"] > 0.5 else "photovoltaic",
            "objectives": objective_values,
        }
        if rank is not None:
            record["pareto_rank"] = int(rank)
        return record


@dataclass(frozen=True)
class OptimizationResult:
    """Resultat de NSGA-II et du front de Pareto audite exhaustivement."""

    screening: ScreeningResult
    population: np.ndarray
    front: np.ndarray
    front_indices: np.ndarray
    front_ranks: np.ndarray
    nsga_front: np.ndarray
    history: dict[str, np.ndarray]
    champions: np.ndarray
    champion_records: list[dict]
    solar_champion: int
    mxene_champion: int

    def diagnostic(self) -> dict:
        screened = self.screening
        return {
            "seed": SEED,
            "n_evaluated": screened.n_materials,
            "nsga2_population_size": int(self.history["population_size"][0]),
            "nsga2_generations": int(self.history["generation"][-1]),
            "nsga2_hypervolume_proxy": float(self.history["hypervolume_proxy"][-1]),
            "nsga2_front_size": int(len(self.nsga_front)),
            "pareto_front_size": int(len(self.front_indices)),
            "nsga2_precision": float(
                len(np.intersect1d(self.nsga_front, self.front_indices))
                / max(len(self.nsga_front), 1)
            ),
            "nsga2_recovery_ratio": float(
                len(np.intersect1d(self.nsga_front, self.front_indices))
                / max(len(self.front_indices), 1)
            ),
            "n_top_champions": len(self.champions),
        }


def _partial_integral(x: np.ndarray, y: np.ndarray, cutoff: np.ndarray) -> np.ndarray:
    """Integrale de y de x[0] jusqu'au dernier noeud <= cutoff (trapezes)."""
    segments = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    cumulative = np.concatenate(([0.0], np.cumsum(segments)))
    position = np.clip(np.searchsorted(x, cutoff, side="right") - 1, 0, x.size - 1)
    return cumulative[position]


def _photon_flux_vectorized(bandgap_ev: np.ndarray) -> np.ndarray:
    """Flux AM1.5G au-dessus du gap, vectorise et equivalent a la Nuit 4."""
    wavelengths = np.asarray(_SOLAR_GRID, dtype=float)
    energies_joule = Q_E * HC_EV_NM / wavelengths
    integrand = SOLAR_IRRADIANCE / energies_joule
    cutoff = np.minimum(HC_EV_NM / np.asarray(bandgap_ev, dtype=float), wavelengths[-1])
    cutoff = np.clip(cutoff, wavelengths[0], wavelengths[-1])
    return _partial_integral(wavelengths, integrand, cutoff)


def _absorptance_vectorized(bandgap_ev: np.ndarray) -> np.ndarray:
    """Absorptance solaire ponderee, vectorisee (identique a solar_absorptance)."""
    wavelengths = np.asarray(_SOLAR_GRID, dtype=float)
    cutoff = np.clip(
        HC_EV_NM / np.asarray(bandgap_ev, dtype=float), wavelengths[0], wavelengths[-1]
    )
    total = float(np.trapezoid(SOLAR_IRRADIANCE, wavelengths))
    above = _partial_integral(wavelengths, SOLAR_IRRADIANCE, cutoff)
    below = total - above
    return (ABSORPTANCE_ABOVE * above + ABSORPTANCE_BELOW * below) / total


def _sq_vectorized(bandgap_ev: np.ndarray, temperature_k: np.ndarray, irradiance: float):
    """Modele Shockley-Queisser vectorise, miroir de shockley_queisser (Nuit 4)."""
    gaps = np.asarray(bandgap_ev, dtype=float)
    thermal_energy = KB_SI * np.asarray(temperature_k, dtype=float)
    thermal_voltage = thermal_energy / Q_E
    fraction = np.linspace(0.0, 1.0, SQ_ENERGY_POINTS, dtype=float)
    energies = gaps[:, None] * Q_E + fraction[None, :] * (60.0 * thermal_energy[:, None])
    integrand = energies**2 / np.expm1(energies / thermal_energy[:, None])
    radiative_flux = (2.0 * np.pi / (H_SI**3 * C_SI**2)) * np.trapezoid(
        integrand, energies, axis=1
    )
    j0 = Q_E * radiative_flux
    jsc = Q_E * _photon_flux_vectorized(gaps) * float(irradiance) / P_INPUT
    ratio = jsc / j0
    voc = thermal_voltage * np.log1p(ratio)
    reduced = np.maximum(np.log1p(ratio) - 1.0, 0.0)
    for _ in range(200):
        exponential = np.exp(reduced)
        residual = (1.0 + reduced) * exponential - (1.0 + ratio)
        derivative = (2.0 + reduced) * exponential
        step = residual / derivative
        reduced = np.maximum(reduced - step, 0.0)
        if np.max(np.abs(step)) < 1.0e-14:
            break
    vmp = thermal_voltage * reduced
    jmp = jsc - j0 * np.expm1(reduced)
    pmax = np.maximum(vmp * jmp, 0.0)
    return {
        "jsc": jsc,
        "j0": j0,
        "voc": voc,
        "ff": pmax / (voc * jsc),
        "eta": pmax / float(irradiance),
    }


def _junction_temperature_vectorized(
    irradiance: float,
    ambient_k: float,
    wind_speed: float,
    eta_ref: np.ndarray,
    beta_t: np.ndarray,
    absorptance: np.ndarray,
) -> np.ndarray:
    """Newton-bisection vectorise du bilan thermique de la Nuit 5."""
    h_conv = H_NATURAL + H_WIND * float(wind_speed)
    t_sky = float(ambient_k) - SKY_DEPRESSION_K
    lower = np.full(eta_ref.shape, float(ambient_k) - 20.0, dtype=float)
    upper = np.full(eta_ref.shape, float(ambient_k) + 300.0, dtype=float)
    temperature = np.full(eta_ref.shape, float(ambient_k) + 30.0, dtype=float)

    def residual(t: np.ndarray) -> np.ndarray:
        eta = eta_ref * np.exp(beta_t * (t - T_REF))
        radiative = EMISSIVITY * SIGMA_SB * (t**4 - t_sky**4)
        return absorptance * irradiance - eta * irradiance - h_conv * (t - ambient_k) - radiative

    for _ in range(MAX_THERMAL_ITER):
        eta = eta_ref * np.exp(beta_t * (temperature - T_REF))
        derivative = (
            -beta_t * eta * irradiance
            - h_conv
            - 4.0 * EMISSIVITY * SIGMA_SB * temperature**3
        )
        trial = temperature - residual(temperature) / derivative
        outside = ~np.isfinite(trial) | (trial <= lower) | (trial >= upper)
        trial = np.where(outside, 0.5 * (lower + upper), trial)
        value = residual(trial)
        lower = np.where(value > 0.0, trial, lower)
        upper = np.where(value > 0.0, upper, trial)
        temperature = trial
        if np.all(np.abs(value) < TOL_THERMAL):
            break
    return temperature


def haldane_model_for(material: dict) -> HaldaneModel:
    """Modele de Haldane determine par la chimie et la famille du candidat."""
    t1, t2 = float(material["t1"]), float(material["t2"])
    if material["family"] == FAMILY_MXENE:
        return HaldaneModel(t1=t1, t2=t2, phi=-np.pi / 2.0, M=0.5 * t2)
    if float(material["lambda_soc"]) >= 0.4:
        return HaldaneModel(t1=t1, t2=t2, phi=np.pi / 2.0, M=0.5 * t2)
    return HaldaneModel(t1=t1, t2=t2, phi=np.pi / 2.0, M=6.0 * t2)


def _chern_index(material: dict) -> int:
    """Invariant de Chern analytique du modele de Haldane associe au candidat."""
    return int(haldane_model_for(material).chern(band=0))


@lru_cache(maxsize=1)
def _mxene_reference() -> tuple[float, float]:
    """Rétention et densité d'énergie exactes de la cellule MXene de la Nuit 7."""
    cell = AsymmetricSupercapacitor()
    retention = float(cell.capacity_retention(100.0, MATAM.ambient_k))
    cycle = cell.galvanostatic_cycle(500.0, MATAM.ambient_k)
    return retention, float(cycle.energy_density_wh_kg)


@lru_cache(maxsize=1)
def run_screening() -> ScreeningResult:
    """Evalue les 2000 candidats sur les deux sites sahéliens."""
    materials = load_database()
    if len(materials) != N_MATERIALS:
        raise ValueError("La base doit contenir exactement 2000 candidats.")

    def column(key: str) -> np.ndarray:
        return np.array([float(m[key]) for m in materials], dtype=float)

    gaps = column("eg_ev")
    beta_t = column("beta_t")
    abundance = column("abundance_index")
    cost = column("cost_usd_per_kg")
    mu_n = column("mu_n")
    mu_p = column("mu_p")
    t1 = column("t1")
    t2 = column("t2")
    lambda_soc = column("lambda_soc")
    lattice_a = column("a")
    lattice_c = column("c")
    is_mxene = np.array([m["family"] == FAMILY_MXENE for m in materials], dtype=float)

    eta_ref = MODULE_FACTOR * np.array([sq_efficiency(float(eg), T_REF) for eg in gaps])
    absorptance = _absorptance_vectorized(gaps)

    tj_matam = _junction_temperature_vectorized(
        MATAM_IRRADIANCE, MATAM.ambient_k, MATAM.wind_speed, eta_ref, beta_t, absorptance
    )
    tj_saint_louis = _junction_temperature_vectorized(
        SAINT_LOUIS_IRRADIANCE,
        SAINT_LOUIS.ambient_k,
        SAINT_LOUIS.wind_speed,
        eta_ref,
        beta_t,
        absorptance,
    )
    eta_matam = eta_ref * np.exp(beta_t * (tj_matam - T_REF))
    eta_saint_louis = eta_ref * np.exp(beta_t * (tj_saint_louis - T_REF))

    sq_matam = _sq_vectorized(gaps, tj_matam, MATAM_IRRADIANCE)
    thermal_voltage = KB_SI * tj_matam / Q_E
    voc = np.asarray(sq_matam["voc"]) + thermal_voltage * np.log(MODULE_FACTOR)
    jsc = Q_E * _photon_flux_vectorized(gaps) * MATAM_IRRADIANCE / P_INPUT * EQE
    ff = np.clip(eta_matam * MATAM_IRRADIANCE / (voc * jsc), 0.0, 1.0)

    retention, _ = _mxene_reference()
    energy_density = (
        MXENE_ENERGY_REFERENCE
        * (MXENE_T_REFERENCE / t1) ** 0.5
        * (MXENE_A_REFERENCE / lattice_a) ** 2
        * retention
        * (0.95 + 0.05 * abundance)
    )
    perf_mxene = 0.5 * energy_density / MXENE_ENERGY_REFERENCE + 0.5 * retention
    perf_pv = eta_matam / _silicon_reference()["eta_matam"]
    performance = np.where(is_mxene > 0.5, perf_mxene, perf_pv)

    stability_pv = np.exp(beta_t * HEAT_TEST_DELTA_K)
    stability = np.where(is_mxene > 0.5, np.full_like(beta_t, retention), stability_pv)

    chern = np.array([_chern_index(material) for material in materials], dtype=float)
    objectives = np.column_stack((performance, stability, abundance, -cost))

    features = {
        "bandgap": gaps,
        "beta_t": beta_t,
        "abundance": abundance,
        "cost": cost,
        "mu_n": mu_n,
        "mu_p": mu_p,
        "t1": t1,
        "t2": t2,
        "lambda_soc": lambda_soc,
        "lattice_a": lattice_a,
        "lattice_c": lattice_c,
        "is_mxene": is_mxene,
        "chern": chern,
        "eta_ref": eta_ref,
        "absorptance": absorptance,
        "tj_matam": tj_matam,
        "tj_saint_louis": tj_saint_louis,
        "eta_matam": eta_matam,
        "eta_saint_louis": eta_saint_louis,
        "voc_matam": voc,
        "jsc_matam": jsc,
        "ff_matam": ff,
        "performance": performance,
        "stability": stability,
        "energy_density": energy_density,
        "retention": np.full_like(beta_t, retention),
    }
    return ScreeningResult(materials, features, objectives, dict(_silicon_reference()))


@lru_cache(maxsize=1)
def _silicon_reference() -> dict[str, float]:
    """Cellule silicium de reference de la Nuit 6, evaluee a Matam."""
    module = silicon_module()
    balance = energy_balance(MATAM, module)
    eta = float(balance.efficiency)
    temperature = float(balance.junction_temperature_k)
    diode = shockley_queisser(module.bandgap_ev, temperature, incident_power=MATAM_IRRADIANCE)
    thermal_voltage = KB_SI * temperature / Q_E
    voc = float(diode.voc + thermal_voltage * np.log(0.68 / MODULE_FACTOR))
    jsc = Q_E * photon_flux_above_gap(module.bandgap_ev) * MATAM_IRRADIANCE / P_INPUT * EQE
    return {
        "eta_matam": eta,
        "eta_matam_percent": 100.0 * eta,
        "junction_temperature_k": temperature,
        "junction_temperature_c": temperature - KELVIN_OFFSET,
        "voc_matam_v": voc,
        "jsc_matam_ma_cm2": jsc / 10.0,
        "ff_matam": float(np.clip(eta * MATAM_IRRADIANCE / (voc * jsc), 0.0, 1.0)),
        "gain_reference_percent": 0.0,
    }


def dominates_matrix(minimization_objectives: np.ndarray) -> np.ndarray:
    """Matrice de dominance stricte pour des objectifs a minimiser (NumPy pur)."""
    values = np.atleast_2d(np.asarray(minimization_objectives, dtype=float))
    weak = values[:, None, :] <= values[None, :, :]
    strict = values[:, None, :] < values[None, :, :]
    matrix = np.all(weak, axis=2) & np.any(strict, axis=2)
    np.fill_diagonal(matrix, False)
    return matrix


def non_dominated_mask(minimization_objectives: np.ndarray) -> np.ndarray:
    """Masque des points non domines d'un jeu d'objectifs a minimiser."""
    return ~np.any(dominates_matrix(minimization_objectives), axis=0)


def non_dominated_sort(minimization_objectives: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tri NSGA-II : rang de Pareto et distance de crowding (NumPy pur)."""
    values = np.atleast_2d(np.asarray(minimization_objectives, dtype=float))
    n_points = values.shape[0]
    ranks = np.zeros(n_points, dtype=int)
    distances = np.zeros(n_points, dtype=float)
    dominance = dominates_matrix(values)
    remaining = np.ones(n_points, dtype=bool)
    rank = 0
    while np.any(remaining):
        local = np.flatnonzero(remaining)
        front = local[~np.any(dominance[np.ix_(local, local)], axis=0)]
        ranks[front] = rank
        if front.size > 1:
            for objective in range(values.shape[1]):
                order = front[np.argsort(values[front, objective], kind="stable")]
                distances[order[0]] = np.inf
                distances[order[-1]] = np.inf
                span = values[order[-1], objective] - values[order[0], objective]
                if span > 0.0:
                    distances[order[1:-1]] += (
                        values[order[2:], objective] - values[order[:-2], objective]
                    ) / span
        remaining[front] = False
        rank += 1
    return ranks, distances




def _superior(indices: np.ndarray, ranks: np.ndarray, distances: np.ndarray, rng) -> int:
    first, second = (int(value) for value in rng.choice(indices, size=2, replace=False))
    if ranks[first] != ranks[second]:
        return first if ranks[first] < ranks[second] else second
    if distances[first] != distances[second]:
        return first if distances[first] > distances[second] else second
    return min(first, second)


def nsga2(
    objectives: np.ndarray,
    *,
    population_size: int = 300,
    generations: int = 250,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.15,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """NSGA-II en NumPy pur sur un espace de decision discret (indices candidats)."""
    values = -np.asarray(objectives, dtype=float)
    n_candidates = values.shape[0]
    rng = np.random.default_rng(int(seed))
    population = rng.choice(n_candidates, size=population_size, replace=False)
    history = {
        "generation": np.zeros(generations + 1, dtype=int),
        "hypervolume_proxy": np.zeros(generations + 1, dtype=float),
        "population_size": np.full(generations + 1, population_size, dtype=int),
        "distinct_candidates": np.zeros(generations + 1, dtype=int),
    }

    for generation in range(generations + 1):
        ranks, distances = non_dominated_sort(values[population])
        front = population[ranks == 0]
        with np.errstate(over="ignore"):
            proxy = float(np.sum(np.exp(-values[front]).sum(axis=1)).item())
        history["generation"][generation] = generation
        history["hypervolume_proxy"][generation] = proxy
        history["distinct_candidates"][generation] = int(np.unique(population).size)
        if generation == generations:
            break

        positions = np.arange(population.size)
        offspring: list[int] = []
        while len(offspring) < population_size:
            parent_a = int(population[_superior(positions, ranks, distances, rng)])
            parent_b = int(population[_superior(positions, ranks, distances, rng)])
            child_a, child_b = parent_a, parent_b
            if rng.random() < crossover_rate and rng.random() < 0.5:
                child_a, child_b = child_b, child_a
            if rng.random() < mutation_rate:
                child_a = int(rng.integers(n_candidates))
            if rng.random() < mutation_rate:
                child_b = int(rng.integers(n_candidates))
            offspring.extend((int(child_a), int(child_b)))

        combined = np.concatenate((population, np.asarray(offspring[:population_size], dtype=int)))
        ranks_combined, distances_combined = non_dominated_sort(values[combined])
        order = np.lexsort((combined, -distances_combined, ranks_combined))
        population = combined[order[:population_size]]

    ranks, distances = non_dominated_sort(values[population])
    return population, distances, history


def champion_score(objectives: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Score composite normalise de selection des meilleurs champions du front."""
    values = np.asarray(objectives, dtype=float)
    sample = values if reference is None else np.asarray(reference, dtype=float)
    low = sample.min(axis=0)
    high = sample.max(axis=0)
    span = np.where(high - low > 0.0, high - low, 1.0)
    normalized = (values - low) / span
    return normalized @ WEIGHTS


def optimize(
    screening_result: ScreeningResult | None = None,
    *,
    population_size: int = 300,
    generations: int = 250,
    seed: int = SEED,
    n_champions: int = 10,
) -> OptimizationResult:
    """Pipeline complet : NSGA-II, front exact audite et Top 10 champions."""
    screening_result = screening_result or run_screening()
    objectives = screening_result.objectives
    population, _, history = nsga2(
        objectives, population_size=population_size, generations=generations, seed=seed
    )
    minimization = -objectives
    exact_front = np.flatnonzero(non_dominated_mask(minimization))
    ranks_population, _ = non_dominated_sort(minimization[population])
    nsga_front = np.unique(population[ranks_population == 0])

    ranked_front = exact_front[
        np.lexsort((exact_front, -champion_score(objectives[exact_front])))
    ]
    performance = screening_result.values("performance")
    is_mxene = screening_result.values("is_mxene") > 0.5
    solar_champion = int(np.argmax(np.where(is_mxene, -np.inf, performance)))
    mxene_champion = int(np.argmax(np.where(is_mxene, performance, -np.inf)))

    per_category = max(1, n_champions // 2)
    selected: list[int] = []
    for category in (False, True):
        pool = sorted(
            (int(index) for index in ranked_front
             if bool(is_mxene[index]) is category),
            key=lambda value: (-performance[value], -champion_score(objectives[value]), value),
        )
        selected.extend(pool[:per_category])
    for required in (solar_champion, mxene_champion):
        if required in selected:
            continue
        category = bool(is_mxene[required])
        candidates = [
            value for value in selected if bool(is_mxene[value]) is category
            and value not in (solar_champion, mxene_champion)
        ]
        if candidates:
            target = min(
                candidates,
                key=lambda value: (performance[value], champion_score(objectives[value]), -value),
            )
            selected[selected.index(target)] = required
    selected = list(dict.fromkeys(selected))
    if len(selected) < n_champions:
        for index in ranked_front:
            if index not in selected:
                selected.append(int(index))
            if len(selected) == n_champions:
                break
    selected = np.array(
        sorted(selected[:n_champions], key=lambda value: (-performance[value], value)),
        dtype=int,
    )

    ranks_all, _ = non_dominated_sort(minimization)
    records = [screening_result.record(int(index), rank=int(ranks_all[index])) for index in selected]
    return OptimizationResult(
        screening=screening_result,
        population=population,
        front=minimization[exact_front],
        front_indices=exact_front,
        front_ranks=ranks_all,
        nsga_front=nsga_front,
        history=history,
        champions=selected,
        champion_records=records,
        solar_champion=solar_champion,
        mxene_champion=mxene_champion,
    )


def export_json(result: OptimizationResult, path: Path = CHAMPIONS_PATH) -> dict:
    """Ecrit le front complet, le Top 10 et le diagnostique dans un JSON."""
    screening_result = result.screening
    front_records = [
        screening_result.record(int(index), rank=0) for index in result.front_indices
    ]
    payload = {
        "metadata": {
            "project": "UAM-TOPOMAT",
            "night": 9,
            "seed": SEED,
            "database": str(ROOT / "data" / "african_materials_db.json"),
            "n_evaluated": screening_result.n_materials,
            "conditions": {
                "Matam": {
                    "ambient_c": MATAM_AMBIENT_C,
                    "irradiance_w_m2": MATAM_IRRADIANCE,
                    "wind_speed_m_s": float(MATAM.wind_speed),
                },
                "Saint-Louis": {
                    "ambient_c": SAINT_LOUIS_AMBIENT_C,
                    "irradiance_w_m2": SAINT_LOUIS_IRRADIANCE,
                    "wind_speed_m_s": float(SAINT_LOUIS.wind_speed),
                },
            },
            "objectives": list(OBJECTIVE_NAMES),
            "directions": list(OBJECTIVE_DIRECTIONS),
            "optimizer": "NSGA-II (pure NumPy)",
            "selection": "Top 10 equilibre : 5 photovoltaiques + 5 MXenes les plus performants du front de Pareto",
            "diagnostic": result.diagnostic(),
            "silicon_reference": screening_result.silicon,
        },
        "pareto_front": front_records,
        "top10_champions": result.champion_records,
        "solar_champion_id": screening_result.materials[result.solar_champion]["id"],
        "mxene_champion_id": screening_result.materials[result.mxene_champion]["id"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def _radar_axis(ax, labels: Sequence[str], values: Sequence[float], reference: Sequence[float],
                color: str, name: str) -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, len(labels), endpoint=False)
    closed = np.concatenate((angles, angles[:1]))
    ax.plot(closed, np.append(values, values[0]), "o-", lw=2.0, ms=4.5,
            color=color, label=name)
    ax.plot(closed, np.append(reference, reference[0]), "s--", lw=1.8, ms=4.0,
            color="#7f8c8d", label="Silicium")
    ax.fill(closed, np.append(values, values[0]), color=color, alpha=0.15)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
    ax.grid(alpha=0.35)


def make_figure(result: OptimizationResult, path: Path = FIGURE_PATH) -> dict:
    """Figure de publication 300 dpi : espace Pareto, radar et comparaison."""
    screening_result = result.screening
    features = screening_result.features
    families = np.array([m["family"] for m in screening_result.materials])
    front = result.front_indices
    champions = result.champions
    colors = {"ABX3": "#08306b", "A2BBX6": "#1b7837", "MXene": "#d95f0e"}

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
    })
    figure = plt.figure(figsize=(16.0, 6.2))
    grid = figure.add_gridspec(1, 3, width_ratios=(1.25, 1.0, 1.15), wspace=0.32)
    ax_space = figure.add_subplot(grid[0, 0])
    ax_radar = figure.add_subplot(grid[0, 1], projection="polar")
    ax_bars = figure.add_subplot(grid[0, 2])

    for family, color in colors.items():
        mask = families == family
        ax_space.scatter(
            features["performance"][mask], features["abundance"][mask],
            c=color, s=8.0, alpha=0.35, edgecolors="none",
            label=f"{family} (n={int(mask.sum())})",
        )
    order = np.argsort(features["performance"][front], kind="stable")
    front_sorted = front[order]
    ax_space.plot(
        features["performance"][front_sorted], features["abundance"][front_sorted],
        "-", color="#a50f15", lw=1.6, alpha=0.85, zorder=3,
    )
    ax_space.scatter(
        features["performance"][front_sorted], features["abundance"][front_sorted],
        facecolor="white", edgecolor="#a50f15", s=45, lw=1.4, zorder=4,
        label=f"Front de Pareto (n={front.size})",
    )
    ax_space.scatter(
        features["performance"][champions], features["abundance"][champions],
        marker="*", s=230, facecolor="#ffd700", edgecolor="#7f6000",
        lw=1.0, zorder=5, label="Top 10 Champions",
    )
    for label_position, index in enumerate(champions):
        ax_space.annotate(
            screening_result.materials[index]["id"].replace("UAM-", ""),
            (features["performance"][index], features["abundance"][index]),
            textcoords="offset points",
            xytext=(7, 6) if label_position % 2 == 0 else (7, -10),
            fontsize=6.4, color="#3b3000",
        )
    ax_space.set_xlabel("Performance $\\eta_{Matam}$ / score MXène (normalisé)")
    ax_space.set_ylabel("Indice d'abondance ouest-africaine")
    ax_space.set_title("(a) Espace des 2000 candidats et front de Pareto", fontsize=11)
    ax_space.grid(alpha=0.25)
    ax_space.legend(loc="lower left", framealpha=0.92, fontsize=7)

    reference = screening_result.silicon
    cost_score = 1.0 / (1.0 + features["cost"] / 100.0)

    def radar_values(index: int) -> np.ndarray:
        return np.array([
            features["performance"][index],
            features["stability"][index],
            features["abundance"][index],
            cost_score[index],
            float(features["chern"][index] != 0),
        ])

    silicon_values = np.array([
        1.0,
        np.exp(-0.0045 * HEAT_TEST_DELTA_K),
        0.60,
        1.0 / (1.0 + 3.0 / 100.0),
        0.0,
    ])
    top_index = int(champions[0])
    _radar_axis(
        ax_radar,
        ["Performance", "Stabilité", "Abondance", "Coût*", "Topologie"],
        radar_values(top_index), silicon_values, "#d95f0e",
        screening_result.materials[top_index]["formula"],
    )
    ax_radar.set_title("(b) Champion n°1 vs silicium", fontsize=11, pad=18)
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), fontsize=7)

    champion = screening_result.record(top_index)
    si_eta = reference["eta_matam_percent"]
    labels = ["$\\eta$ (%)", "$V_{oc}$ (V)", "$J_{sc}$ (mA/cm²)", "FF", "Gain (%)"]
    champion_values = [
        champion["eta_matam_percent"], champion["voc_matam_v"],
        champion["jsc_matam_ma_cm2"], champion["ff_matam"],
        champion["gain_vs_silicon_percent"],
    ]
    silicon_bar = [
        si_eta, reference["voc_matam_v"], reference["jsc_matam_ma_cm2"],
        reference["ff_matam"], 0.0,
    ]
    positions = np.arange(len(labels))
    width = 0.38
    ax_bars.bar(positions - width / 2.0, champion_values, width,
                color="#d95f0e", edgecolor="#7f3b08", label="Champion n°1")
    ax_bars.bar(positions + width / 2.0, silicon_bar, width,
                color="#9ecae1", edgecolor="#08306b", label="Silicium")
    for position, value in zip(positions, champion_values):
        ax_bars.text(position - width / 2.0, value, f"{value:.2f}",
                     ha="center", va="bottom", fontsize=6.8)
    ax_bars.set_xticks(positions)
    ax_bars.set_xticklabels(labels, fontsize=8)
    ax_bars.set_ylabel("Valeur à Matam (47 °C, 1050 W/m²)")
    ax_bars.set_title("(c) Comparaison chiffrée à Matam", fontsize=11)
    ax_bars.grid(alpha=0.25, axis="y")
    ax_bars.legend(loc="upper right", framealpha=0.92)

    figure.suptitle(
        "UAM-TOPOMAT — découverte inverse NSGA-II et Top 10 des matériaux champions",
        fontsize=12.5, y=0.995,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {
        "n_candidates": int(screening_result.n_materials),
        "n_front": int(front.size),
        "n_champions": int(champions.size),
        "champion_id": screening_result.materials[top_index]["id"],
        "champion_eta": float(champion["eta_matam"]),
        "silicon_eta": float(si_eta),
        "gain_percent": float(champion["gain_vs_silicon_percent"]),
        "radar_values": radar_values(top_index).tolist(),
    }


@lru_cache(maxsize=1)
def discover() -> OptimizationResult:
    """Exécute la Nuit 9 complète et exporte JSON + figure."""
    result = optimize(run_screening())
    export_json(result)
    make_figure(result)
    return result


def symbolic_stability() -> dict[str, sp.Expr]:
    """Vérifications SymPy de la loi de stabilité et du coût normalisé."""
    beta, delta_t, cost, scale = sp.symbols("beta dT cost scale", real=True, positive=True)
    stability = sp.exp(-beta * delta_t)
    cost_score = sp.Rational(1, 1) / (1 + cost / scale)
    return {
        "stability": stability,
        "stability_at_zero": sp.simplify(stability.subs(delta_t, 0)),
        "stability_derivative": sp.simplify(sp.diff(stability, delta_t) + beta * stability),
        "cost_score": cost_score,
        "cost_derivative": sp.simplify(sp.diff(cost_score, cost)),
        "cost_monotonic": sp.simplify(sp.diff(cost_score, cost)).is_negative,
    }


__all__ = [
    "SEED",
    "N_MATERIALS",
    "OBJECTIVE_NAMES",
    "OBJECTIVE_DIRECTIONS",
    "CHAMPIONS_PATH",
    "FIGURE_PATH",
    "ScreeningResult",
    "OptimizationResult",
    "run_screening",
    "dominates_matrix",
    "non_dominated_mask",
    "non_dominated_sort",
    "haldane_model_for",
    "nsga2",
    "champion_score",
    "optimize",
    "export_json",
    "make_figure",
    "discover",
    "symbolic_stability",
]
