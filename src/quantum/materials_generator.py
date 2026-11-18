"""UAM-TOPOMAT -- Nuit 8 : generation deterministe d'une base de 2000 materiaux.

Generateur cristallochimique en Python pur (numpy, sympy, matplotlib,
**aucune dependance a SciPy**) produisant des candidats sans plomb ni cobalt
pour trois familles : perovskites ABX3, perovskites doubles A2B'B''X6 et
MXenes 2D M2X / M3X2. La graine fixe ``SEED = 2026`` garantit une
reproductibilité bit-a-bit de la base.

Chaque candidat porte des parametres physiques correlates a la chimie de ses
elements : constantes de maille (a, c), sauts Tight-Binding (t_1, t_2),
couplage spin-orbite lambda_SOC, gap optique E_g dans [0.5, 2.8] eV,
mobilite electronique/trou, coefficient thermique beta_T, indice
d'abondance en Afrique de l'Ouest et cout estime en USD/kg.

La neutralite de charge est construite exactement : les occupancies des sites
et les etats d'oxydation sont stockes pour verification dans les tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .band_structure import TightBindingModel

SEED = 2026
N_MATERIALS = 2000
FAMILY_ABX3 = "ABX3"
FAMILY_DOUBLE = "A2BBX6"
FAMILY_MXENE = "MXene"

FAMILY_TARGETS = {
    FAMILY_ABX3: 900,
    FAMILY_DOUBLE: 550,
    FAMILY_MXENE: 550,
}

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "african_materials_db.json"
SUMMARY_PATH = DATA_DIR / "db_summary.json"

EG_MIN = 0.5
EG_MAX = 2.8


@dataclass(frozen=True)
class Element:
    """Proprietes elementaires utiles a la modelisation cristallochimique."""

    symbol: str
    z: int
    radius: float
    mass: float
    electronegativity: float
    charge: float
    abundance: float
    cost: float


ELEMENTS: Dict[str, Element] = {
    "Cs": Element("Cs", 55, 1.67, 132.91, 0.79, 1.0, 0.15, 90.0),
    "Rb": Element("Rb", 37, 1.52, 85.47, 0.82, 1.0, 0.10, 110.0),
    "K": Element("K", 19, 1.38, 39.10, 0.82, 1.0, 0.85, 2.0),
    "Na": Element("Na", 11, 1.02, 22.99, 0.93, 1.0, 0.90, 1.0),
    "Sr": Element("Sr", 38, 1.18, 87.62, 0.95, 2.0, 0.55, 6.0),
    "Ba": Element("Ba", 56, 1.35, 137.33, 0.89, 2.0, 0.45, 4.0),
    "Ca": Element("Ca", 20, 1.00, 40.08, 1.00, 2.0, 0.90, 0.5),
    "La": Element("La", 57, 1.03, 138.91, 1.10, 3.0, 0.40, 25.0),
    "Ce": Element("Ce", 58, 1.01, 140.12, 1.12, 3.0, 0.40, 18.0),
    "Nd": Element("Nd", 60, 0.98, 144.24, 1.14, 3.0, 0.35, 65.0),
    "Sm": Element("Sm", 62, 0.96, 150.36, 1.17, 3.0, 0.30, 40.0),
    "Y": Element("Y", 39, 0.90, 88.91, 1.22, 3.0, 0.35, 45.0),
    "Bi": Element("Bi", 83, 1.03, 208.98, 2.02, 3.0, 0.55, 14.0),
    "Sn": Element("Sn", 50, 0.69, 118.71, 1.96, 2.0, 0.60, 26.0),
    "Ge": Element("Ge", 32, 0.53, 72.63, 2.01, 2.0, 0.45, 1400.0),
    "Ti": Element("Ti", 22, 0.605, 47.87, 1.54, 4.0, 0.95, 5.0),
    "Fe": Element("Fe", 26, 0.645, 55.85, 1.83, 3.0, 1.00, 0.6),
    "Mn": Element("Mn", 25, 0.645, 54.94, 1.55, 3.0, 0.95, 1.8),
    "V": Element("V", 23, 0.640, 50.94, 1.63, 3.0, 0.80, 14.0),
    "Ag": Element("Ag", 47, 1.15, 107.87, 1.93, 1.0, 0.20, 720.0),
    "Sb": Element("Sb", 51, 0.76, 121.76, 2.05, 3.0, 0.50, 16.0),
    "In": Element("In", 49, 0.80, 114.82, 1.78, 1.0, 0.25, 320.0),
    "Cl": Element("Cl", 17, 1.81, 35.45, 3.16, -1.0, 0.90, 0.5),
    "Br": Element("Br", 35, 1.96, 79.90, 2.96, -1.0, 0.50, 4.0),
    "I": Element("I", 53, 2.20, 126.90, 2.66, -1.0, 0.25, 70.0),
    "O": Element("O", 8, 1.40, 16.00, 3.44, -2.0, 1.00, 0.2),
    "C": Element("C", 6, 0.77, 12.01, 2.55, -4.0, 1.00, 1.5),
    "N": Element("N", 7, 0.75, 14.01, 3.04, -3.0, 0.70, 0.6),
    "H": Element("H", 1, 0.37, 1.008, 2.20, 1.0, 1.00, 2.0),
    "F": Element("F", 9, 1.33, 19.00, 3.98, -1.0, 0.80, 3.0),
}

FORBIDDEN = frozenset({"Pb", "Co"})

A_SITES_MONO = ("Cs", "Rb", "K", "Na")
A_SITES_DVI = ("Sr", "Ba", "Ca")
A_SITES_TRI = ("La", "Ce", "Nd", "Sm", "Y")
B_SITES_HALIDE = ("Sn", "Ge")
B_SITES_OXIDE_4 = ("Ti", "Sn", "Ge")
B_SITES_OXIDE_3 = ("Bi",)
X_HALIDES = ("Cl", "Br", "I")
M_SITES = ("Ti", "V", "Mn", "Fe")
X_MXENES = ("C", "N")
TERMINATIONS = ("", "O2", "F2", "(OH)2")


def _fmt(value: float) -> str:
    """Formatte un nombre de maniere compacte et deterministe."""
    if abs(value - round(value)) < 1.0e-12:
        return str(int(round(value)))
    return f"{value:g}"


def _site_configs(symbols: Sequence[str], total: float, fractions: Sequence[float]):
    """Enumerate les occupancies ordonnees de 1 ou 2 especes sur un site."""
    configs: List[Tuple[Tuple[str, float], ...]] = []
    for symbol in symbols:
        configs.append(((symbol, float(total)),))
    for first in symbols:
        for second in symbols:
            if first == second:
                continue
            for fraction in fractions:
                amount = float(total) * float(fraction)
                if 1.0e-9 < amount < float(total) - 1.0e-9:
                    configs.append(((first, amount), (second, float(total) - amount)))
    return configs


A1_CONFIGS = _site_configs(A_SITES_MONO, 1.0, (0.25, 0.5, 0.75))
A1_EARTH_CONFIGS = _site_configs(A_SITES_DVI, 1.0, (0.25, 0.5, 0.75))
A1_TRI_CONFIGS = _site_configs(A_SITES_TRI, 1.0, (0.25, 0.5, 0.75))
X3_CONFIGS = _site_configs(X_HALIDES, 3.0, (1.0 / 6.0, 0.5, 2.0 / 3.0))
A2_ALKALI_CONFIGS = _site_configs(A_SITES_MONO, 2.0, (0.25, 0.5, 0.75))
A2_EARTH_CONFIGS = _site_configs(A_SITES_DVI, 2.0, (0.25, 0.5, 0.75))
M2_CONFIGS = _site_configs(M_SITES, 2.0, (0.25, 0.5, 0.75))
M3_CONFIGS = _site_configs(M_SITES, 3.0, (1.0 / 3.0, 2.0 / 3.0))
X1_CONFIGS = _site_configs(X_MXENES, 1.0, (0.5,))
X2_CONFIGS = _site_configs(X_MXENES, 2.0, (0.25, 0.5, 0.75))
X6_CONFIGS = _site_configs(X_HALIDES, 6.0, (0.5 / 3.0, 0.5, 2.0 / 3.0))

TERMINATION_OCCUPANCIES = {
    "": {},
    "O2": {"O": 2.0},
    "F2": {"F": 2.0},
    "(OH)2": {"O": 2.0, "H": 2.0},
}
TERMINATION_CHARGES = {"H": 1.0}


def _site_formula(config) -> str:
    return "".join(f"{symbol}{_fmt(count)}" for symbol, count in config)


def _formula_abx3(a_site, b_symbol, x_site) -> str:
    return f"{_site_formula(a_site)}{b_symbol}{_site_formula(x_site)}"


def _formula_double(a_site, b_prime, b_second, x_site, oxide: bool) -> str:
    return (
        f"{_site_formula(a_site)}{b_prime}{b_second}"
        f"{'O6' if oxide else _site_formula(x_site)}"
    )


def _formula_mxene(m_site, x_site, termination) -> str:
    return f"{_site_formula(m_site)}{_site_formula(x_site)}{termination}"


def _counts_from_sites(*sites) -> Dict[str, float]:
    counts: Dict[str, float] = {}
    for site in sites:
        for symbol, amount in site:
            counts[symbol] = counts.get(symbol, 0.0) + float(amount)
    return counts


def _add_termination(counts: Dict[str, float], termination: str) -> Dict[str, float]:
    result = dict(counts)
    for symbol, amount in TERMINATION_OCCUPANCIES[termination].items():
        result[symbol] = result.get(symbol, 0.0) + float(amount)
    return result


def _mean(attr: str, counts: Dict[str, float]) -> float:
    total = sum(counts.values())
    return float(sum(ELEMENTS[s].__getattribute__(attr) * n for s, n in counts.items()) / total)


def _mass_cost(counts: Dict[str, float]) -> float:
    mass = sum(ELEMENTS[s].mass * n for s, n in counts.items())
    return float(sum(ELEMENTS[s].mass * ELEMENTS[s].cost * n for s, n in counts.items()) / mass)


def _soc_lambda(counts: Dict[str, float]) -> float:
    heaviest = max(ELEMENTS[s].z for s in counts if ELEMENTS[s].z > 0 and s != "H")
    return float(np.clip(0.45 * (heaviest / 83.0) ** 4, 0.001, 0.45))


def _balanced_states(counts: Dict[str, float], groups=(), derive_cations=()):
    """Etats d'oxydation assurant une neutralite exacte.

    ``groups`` contient des paires (symboles du site, charge) : la charge
    depend du contexte cristallin (Sn/Ge sont +2 dans les halogenures et +4
    dans les oxydes). Seuls les cations M des MXenes voient leur etat deduit
    de la charge anionique totale, terminaisons comprises.
    """
    states: Dict[str, float] = {}
    for symbol in counts:
        if symbol in TERMINATION_CHARGES:
            states[symbol] = TERMINATION_CHARGES[symbol]
        else:
            states[symbol] = ELEMENTS[symbol].charge
    for symbols, charge in groups:
        for symbol in symbols:
            states[symbol] = float(charge)
    derived = tuple(derive_cations)
    if derived:
        anionic = sum(
            states[s] * n for s, n in counts.items() if states[s] < 0.0
        )
        total = sum(counts[s] for s in derived)
        per_cation = -anionic / total
        for symbol in derived:
            states[symbol] = per_cation
    return states


def _net_charge(counts: Dict[str, float], states: Dict[str, float]) -> float:
    return float(sum(counts[s] * states[s] for s in counts))


def _physical_parameters(
    counts: Dict[str, float],
    a: float,
    c: float,
    family: str,
    rng: np.random.Generator,
):
    """Parametres quantiques, optiques, thermiques et economiques d'un candidat."""
    cations = [s for s in counts if ELEMENTS[s].charge > 0 and s not in TERMINATION_CHARGES]
    anions = [s for s in counts if ELEMENTS[s].charge < 0 and s not in TERMINATION_CHARGES]
    chi_cations = float(np.mean([ELEMENTS[s].electronegativity for s in cations]))
    chi_anions = float(np.mean([ELEMENTS[s].electronegativity for s in anions]))
    delta_chi = chi_anions - chi_cations
    lam = _soc_lambda(counts)

    hopping_scale = float(np.exp(-(a - 3.9) / 1.2))
    t1 = float(np.clip(1.05 * hopping_scale * (0.82 + 0.18 * delta_chi), 0.25, 2.30))
    t2 = float(t1 * (0.22 + 0.10 * lam / (lam + 0.1)))

    offset = {FAMILY_ABX3: 0.18, FAMILY_DOUBLE: 0.22, FAMILY_MXENE: -0.10}[family]
    eg_raw = 0.85 + 0.70 * delta_chi - 0.35 * lam - 0.10 * (a - 4.0) + offset
    eg = float(np.clip(eg_raw + float(rng.normal(0.0, 0.035)), EG_MIN, EG_MAX))

    mu_n = float(np.clip(900.0 * t1 * (a / 4.0) ** 2 / (1.0 + 8.0 * lam)
                         * float(rng.uniform(0.92, 1.08)), 5.0, 1500.0))
    mu_p = float(np.clip(0.45 * mu_n * (1.0 - 0.10 * lam)
                         * float(rng.uniform(0.92, 1.08)), 2.0, 900.0))

    stability = {FAMILY_ABX3: 1.0, FAMILY_DOUBLE: 0.65, FAMILY_MXENE: 0.90}[family]
    beta_t = float(-stability * (0.0010 + 0.0028 * (1.0 - eg / EG_MAX)) * (1.0 + 0.5 * lam))
    abundance = float(np.clip(_mean("abundance", counts), 0.0, 1.0))
    cost = float(max(0.05, _mass_cost(counts)))

    return {
        "a": round(a, 6),
        "c": round(c, 6),
        "t1": round(t1, 6),
        "t2": round(t2, 6),
        "lambda_soc": round(lam, 6),
        "eg_ev": round(eg, 6),
        "mu_n": round(mu_n, 4),
        "mu_p": round(mu_p, 4),
        "beta_t": round(beta_t, 8),
        "abundance_index": round(abundance, 6),
        "cost_usd_per_kg": round(cost, 4),
    }


def _record(material_id: str, family: str, subfamily: str, formula: str,
            counts: Dict[str, float], states: Dict[str, float],
            parameters: dict, tolerance: float | None = None) -> dict:
    stored_counts = {s: round(counts[s], 12) for s in sorted(counts)}
    stored_states = {s: round(states[s], 12) for s in sorted(states)}
    record = {
        "id": material_id,
        "family": family,
        "subfamily": subfamily,
        "formula": formula,
        "elements": sorted(counts),
        "composition": stored_counts,
        "oxidation_states": stored_states,
        "net_charge": round(_net_charge(stored_counts, stored_states), 12),
        "tolerance_factor": None if tolerance is None else round(tolerance, 6),
    }
    record.update(parameters)
    return record


def _candidate_abx3(rng: np.random.Generator, seen: set[str]):
    """Tire un candidat ABX3 charge-neutral (halogenure ou oxyde)."""
    for _ in range(200):
        kind = int(rng.integers(0, 10))
        if kind < 5:
            a_site = A1_CONFIGS[int(rng.integers(0, len(A1_CONFIGS)))]
            b_symbol = B_SITES_HALIDE[int(rng.integers(0, len(B_SITES_HALIDE)))]
            x_site = X3_CONFIGS[int(rng.integers(0, len(X3_CONFIGS)))]
            counts = _counts_from_sites(a_site, ((b_symbol, 1.0),), x_site)
            a_symbols = tuple(symbol for symbol, _ in a_site)
            states = _balanced_states(counts, groups=((a_symbols, 1.0), ((b_symbol,), 2.0)))
            formula = _formula_abx3(a_site, b_symbol, x_site)
            subfamily = "halide_ABX3"
            r_a = _mean("radius", dict(a_site))
            r_b = ELEMENTS[b_symbol].radius
            r_x = _mean("radius", dict(x_site))
            tolerance = (r_a + r_x) / (np.sqrt(2.0) * (r_b + r_x))
            lattice = 2.0 * (r_b + r_x) * (0.98 + 0.04 * float(tolerance))
        elif kind < 8:
            a_site = A1_EARTH_CONFIGS[int(rng.integers(0, len(A1_EARTH_CONFIGS)))]
            b_symbol = B_SITES_OXIDE_4[int(rng.integers(0, len(B_SITES_OXIDE_4)))]
            counts = _counts_from_sites(a_site, ((b_symbol, 1.0),), (("O", 3.0),))
            a_symbols = tuple(symbol for symbol, _ in a_site)
            states = _balanced_states(counts, groups=((a_symbols, 2.0), ((b_symbol,), 4.0)))
            formula = _formula_abx3(a_site, b_symbol, (("O", 3.0),))
            subfamily = "oxide_ABO3"
            r_a = _mean("radius", dict(a_site))
            r_b = ELEMENTS[b_symbol].radius
            tolerance = (r_a + ELEMENTS["O"].radius) / (np.sqrt(2.0) * (r_b + ELEMENTS["O"].radius))
            lattice = 2.0 * (r_b + ELEMENTS["O"].radius) * (0.98 + 0.04 * float(tolerance))
        else:
            a_site = A1_TRI_CONFIGS[int(rng.integers(0, len(A1_TRI_CONFIGS)))]
            b_symbol = B_SITES_OXIDE_3[int(rng.integers(0, len(B_SITES_OXIDE_3)))]
            counts = _counts_from_sites(a_site, ((b_symbol, 1.0),), (("O", 3.0),))
            a_symbols = tuple(symbol for symbol, _ in a_site)
            states = _balanced_states(counts, groups=((a_symbols, 3.0), ((b_symbol,), 3.0)))
            formula = _formula_abx3(a_site, b_symbol, (("O", 3.0),))
            subfamily = "oxide_ABO3"
            r_a = _mean("radius", dict(a_site))
            tolerance = (r_a + ELEMENTS["O"].radius) / (
                np.sqrt(2.0) * (ELEMENTS[b_symbol].radius + ELEMENTS["O"].radius)
            )
            lattice = 2.0 * (ELEMENTS[b_symbol].radius + ELEMENTS["O"].radius) * (
                0.98 + 0.04 * float(tolerance)
            )
        if formula in seen:
            continue
        if abs(_net_charge(counts, states)) > 1.0e-9:
            raise ValueError("Perovskite ABX3 non neutre : " + formula)
        seen.add(formula)
        return formula, counts, states, subfamily, float(lattice), float(lattice), float(tolerance)
    raise RuntimeError("Impossible de produire un candidat ABX3 unique.")


def _candidate_double(rng: np.random.Generator, seen: set[str]):
    """Tire un candidat A2B'B''X6 charge-neutral (Bi-Ag, Sn-Ti, Sb-In)."""
    for _ in range(200):
        pair = int(rng.integers(0, 3))
        oxide = pair == 1
        if pair == 0:
            b_prime, b_second = "Ag", "Bi"
        elif pair == 1:
            b_prime, b_second = "Sn", "Ti"
        else:
            b_prime, b_second = "In", "Sb"
        if oxide:
            a_site = A2_EARTH_CONFIGS[int(rng.integers(0, len(A2_EARTH_CONFIGS)))]
            x_site = (("O", 6.0),)
        else:
            a_site = A2_ALKALI_CONFIGS[int(rng.integers(0, len(A2_ALKALI_CONFIGS)))]
            x_site = X6_CONFIGS[int(rng.integers(0, len(X6_CONFIGS)))]
        counts = _counts_from_sites(a_site, ((b_prime, 1.0), (b_second, 1.0)), x_site)
        a_symbols = tuple(symbol for symbol, _ in a_site)
        if oxide:
            groups = ((a_symbols, 2.0), ((b_prime,), 4.0), ((b_second,), 4.0))
        elif pair == 0:
            groups = ((a_symbols, 1.0), ((b_prime,), 1.0), ((b_second,), 3.0))
        else:
            groups = ((a_symbols, 1.0), ((b_prime,), 1.0), ((b_second,), 3.0))
        states = _balanced_states(counts, groups=groups)
        formula = _formula_double(a_site, b_prime, b_second, x_site, oxide)
        if formula in seen or abs(_net_charge(counts, states)) > 1.0e-9:
            continue
        seen.add(formula)
        r_x = _mean("radius", dict(x_site))
        r_b = 0.5 * (ELEMENTS[b_prime].radius + ELEMENTS[b_second].radius)
        lattice = 2.0 * (r_b + r_x) * 1.02
        subfamily = ("oxide_A2BB'O6" if oxide
                     else f"halide_A2{b_prime}{b_second}X6")
        return formula, counts, states, subfamily, float(lattice), float(lattice), None
    raise RuntimeError("Impossible de produire un candidat A2BB'X6 unique.")


def _candidate_mxene(rng: np.random.Generator, seen: set[str]):
    """Tire un candidat MXene M2X ou M3X2 charge-neutral."""
    for _ in range(300):
        stacked = bool(rng.integers(0, 2))
        m_configs = M3_CONFIGS if stacked else M2_CONFIGS
        x_configs = X2_CONFIGS if stacked else X1_CONFIGS
        m_site = m_configs[int(rng.integers(0, len(m_configs)))]
        x_site = x_configs[int(rng.integers(0, len(x_configs)))]
        termination = TERMINATIONS[int(rng.integers(0, len(TERMINATIONS)))]
        counts = _add_termination(_counts_from_sites(m_site, x_site), termination)
        m_symbols = tuple(sorted({symbol for symbol, _ in m_site}))
        states = _balanced_states(counts, derive_cations=m_symbols)
        formula = _formula_mxene(m_site, x_site, termination)
        if formula in seen or abs(_net_charge(counts, states)) > 1.0e-9:
            continue
        seen.add(formula)
        r_m = _mean("radius", dict(m_site))
        r_x = _mean("radius", dict(x_site))
        a = 3.0 + (r_m - 0.645) + 0.08 * (len(m_site) - 1)
        c = 4.6 if not stacked else 18.6
        c += 0.25 * len(TERMINATION_OCCUPANCIES[termination])
        subfamily = "M3X2" if stacked else "M2X"
        return formula, counts, states, subfamily, float(a), float(c), None
    raise RuntimeError("Impossible de produire un MXene unique.")


def generate_candidates(n: int = N_MATERIALS, seed: int = SEED) -> List[dict]:
    """Genere ``n`` candidats deterministes, uniques et sans Pb/Co."""
    if int(n) != N_MATERIALS:
        raise ValueError(f"La base UAM-TOPOMAT contient exactement {N_MATERIALS} candidats.")
    rng = np.random.default_rng(int(seed))
    targets = FAMILY_TARGETS
    seen: set[str] = set()
    families: List[dict] = []
    generators = (
        (FAMILY_ABX3, targets[FAMILY_ABX3], _candidate_abx3),
        (FAMILY_DOUBLE, targets[FAMILY_DOUBLE], _candidate_double),
        (FAMILY_MXENE, targets[FAMILY_MXENE], _candidate_mxene),
    )
    for family, count, generator in generators:
        for _ in range(int(count)):
            formula, counts, states, subfamily, a, c, tolerance = generator(rng, seen)
            if any(s in FORBIDDEN for s in counts):
                raise ValueError("Element interdit dans la base : " + formula)
            parameters = _physical_parameters(counts, a, c, family, rng)
            families.append(
                _record("", family, subfamily, formula, counts, states, parameters, tolerance)
            )

    materials: List[dict] = []
    for index, record in enumerate(families, start=1):
        record["id"] = f"UAM-{index:04d}"
        materials.append(record)
    if len(materials) != N_MATERIALS:
        raise RuntimeError("Nombre de candidats incorrect.")
    return materials


def material_summary(materials: Sequence[dict], seed: int = SEED) -> dict:
    """Resumé statistique deterministe de la base de donnees."""
    families: Dict[str, int] = {}
    subfamilies: Dict[str, int] = {}
    histogram: Dict[str, int] = {}
    for material in materials:
        families[material["family"]] = families.get(material["family"], 0) + 1
        subfamilies[material["subfamily"]] = subfamilies.get(material["subfamily"], 0) + 1
        for symbol in material["elements"]:
            histogram[symbol] = histogram.get(symbol, 0) + 1

    def column(key: str) -> np.ndarray:
        return np.array([float(m[key]) for m in materials], dtype=float)

    keys = (
        "a", "c", "t1", "t2", "lambda_soc", "eg_ev", "mu_n", "mu_p",
        "beta_t", "abundance_index", "cost_usd_per_kg",
    )
    means = {key: round(float(np.mean(column(key))), 8) for key in keys}
    ranges = {
        key: [round(float(np.min(column(key))), 8), round(float(np.max(column(key))), 8)]
        for key in keys
    }
    return {
        "seed": int(seed),
        "n_materials": len(materials),
        "families": dict(sorted(families.items())),
        "subfamilies": dict(sorted(subfamilies.items())),
        "element_histogram": dict(sorted(histogram.items())),
        "means": means,
        "ranges": ranges,
        "lead_free": all("Pb" not in m["elements"] for m in materials),
        "cobalt_free": all("Co" not in m["elements"] for m in materials),
        "charge_neutral": all(abs(m["net_charge"]) < 1.0e-9 for m in materials),
        "gap_bounds_ev": [EG_MIN, EG_MAX],
    }


def save_database(
    materials: Sequence[dict] | None = None,
    seed: int = SEED,
    directory: Path | None = None,
) -> Tuple[Path, Path]:
    """Ecrit la base et son resume, puis retourne les deux chemins."""
    target_dir = Path(directory) if directory is not None else DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    dataset = list(materials) if materials is not None else generate_candidates(seed=seed)
    db_path = target_dir / DB_PATH.name
    summary_path = target_dir / SUMMARY_PATH.name
    db_path.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(material_summary(dataset, seed), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return db_path, summary_path


def load_database(path: Path | None = None) -> List[dict]:
    """Charge la base JSON depuis le disque."""
    source = Path(path) if path is not None else DB_PATH
    return list(json.loads(source.read_text(encoding="utf-8")))


def build_tight_binding_model(material: dict):
    """Construit le modele Tight-Binding Nuit 1 d'un candidat de la base."""
    family = material["family"]
    if family == FAMILY_MXENE:
        return TightBindingModel.mxene(
            a=material["a"],
            c=material["c"],
            z=0.13,
            t=material["t1"],
            onsite=(0.0, 0.0, -1.5 * material["t1"]),
        )
    onsite = (-2.0, 0.0, 0.5 * material["eg_ev"], 0.5 * material["eg_ev"],
              0.5 * material["eg_ev"])
    return TightBindingModel.perovskite(
        a=material["a"],
        t_ax=material["t2"],
        t_bx=material["t1"],
        onsite=onsite,
    )


def representative_sample(materials: Sequence[dict]) -> List[dict]:
    """Un candidat par sous-famille pour les tests d'integration."""
    selected: List[dict] = []
    seen_subfamilies: set[str] = set()
    for material in materials:
        if material["subfamily"] not in seen_subfamilies:
            seen_subfamilies.add(material["subfamily"])
            selected.append(material)
    return selected


__all__ = [
    "SEED",
    "N_MATERIALS",
    "FAMILY_ABX3",
    "FAMILY_DOUBLE",
    "FAMILY_MXENE",
    "FAMILY_TARGETS",
    "ELEMENTS",
    "DB_PATH",
    "SUMMARY_PATH",
    "EG_MIN",
    "EG_MAX",
    "generate_candidates",
    "material_summary",
    "save_database",
    "load_database",
    "build_tight_binding_model",
    "representative_sample",
]
