"""UAM-TOPOMAT -- Nuit 7 : supercondensateurs asymetriques a electrodes MXene.

Modele electrochimique et quantique d'un supercondensateur sans plomb a deux
electrodes M2X (Ti2CTx / Nb2CTx). La capacite quantique de chaque electrode est
calculee depuis la DOS du solveur Tight-Binding de la Nuit 1,

    C_Q(V) = e^2 * DOS(E_F + eV) / A_cell,

et s'associe en serie a la capacite de double couche de Helmholtz C_H :

    1/C_tot(V) = 1/C_Q(V) + 1/C_H(T).

Les tensions d'electrode sont determinees par conservation de la charge,
q = f_+ Q_+(V_+) = f_- Q_-(V_-), puis le cycle galvanostatique integre
dq/dt = +/- I avec une chute ohmique I * ESR(T). Le cas climatique est celui
du Sahel (25 a 50 C, site de Matam a 47 C) : la conductivite ionique suit une
loi d'Arrhenius, la resistance electronique des MXenes croit faiblement avec T
et l'accessibilite des pores se degrade au-dela de 40 C.

Python pur : numpy, sympy, matplotlib. Aucune dependance a SciPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import sympy as sp

from src.optics.solar_optics import KB_SI, Q_E, T_REF
from src.quantum.band_structure import TightBindingModel, uniform_k_grid
from src.thermo.sahel_climate import MATAM, SahelSite

KELVIN_OFFSET = 273.15
EV_PER_KELVIN = float(KB_SI) / float(Q_E)
ANGSTROM2_TO_M2 = 1.0e-20
HELMHOLTZ_REFERENCE = 0.50
SPECIFIC_SURFACE = 300_000.0
DOS_BROADENING = 0.35
HELMHOLTZ_T_COEFF = 5.0e-4
AGING_ONSET_K = 313.15
AGING_SLOPE_PER_K = 5.0e-3
ION_ACTIVATION_EV = 0.15
R_ION_REFERENCE = 2.0e-5
R_ELECTRONIC_REFERENCE = 4.0e-6
R_CONTACT = 2.0e-6
R_ELECTRONIC_T_COEFF = 1.0e-3
GRID_POINTS = 601
K_GRID = 32


def _as_temperature(temperature_k) -> np.ndarray:
    temperature = np.asarray(temperature_k, dtype=float)
    if np.any(temperature <= 0.0):
        raise ValueError("La temperature absolue doit etre strictement positive.")
    return temperature


def pore_accessibility(temperature_k):
    """Facteur d'accessibilite des pores, dégradation au-dela de 40 C."""
    temperature = _as_temperature(temperature_k)
    excess = np.maximum(temperature - AGING_ONSET_K, 0.0)
    factor = np.exp(-AGING_SLOPE_PER_K * excess)
    return float(factor) if factor.ndim == 0 else factor


def esr(temperature_k):
    """Resistance serie equivalente (ohm.kg) entre 25 et 50 C.

    R_ion(T) = R_ion,ref exp[(E_a/k_B)(1/T - 1/T_ref)] decroit avec T, alors
    que la resistance electronique R_el(T) = R_el,ref[1 + a(T-T_ref)] croit
    faiblement (comportement metallique des MXenes).
    """
    temperature = _as_temperature(temperature_k)
    ionic = R_ION_REFERENCE * np.exp(
        (ION_ACTIVATION_EV / EV_PER_KELVIN)
        * (1.0 / temperature - 1.0 / T_REF)
    )
    electronic = R_ELECTRONIC_REFERENCE * (
        1.0 + R_ELECTRONIC_T_COEFF * (temperature - T_REF)
    )
    total = R_CONTACT + ionic + electronic
    return float(total) if total.ndim == 0 else total


@dataclass(frozen=True)
class ElectrodeConfig:
    """Parametres d'une electrode MXene M2X sans plomb."""

    name: str
    efermi_ev: float = 0.0
    potential_sign: float = 1.0
    window_v: float = 0.95
    helmholtz_areal_f_m2: float = HELMHOLTZ_REFERENCE
    specific_surface_m2_kg: float = SPECIFIC_SURFACE
    dos_broadening_ev: float = DOS_BROADENING
    lattice_a_angstrom: float = 3.0
    lattice_c_angstrom: float = 4.6
    lattice_z: float = 0.13
    hopping_t_ev: float = 1.0
    onsite_ev: tuple[float, float, float] = (0.0, 0.0, -1.5)

    def __post_init__(self) -> None:
        if self.window_v <= 0.0 or self.helmholtz_areal_f_m2 <= 0.0:
            raise ValueError("La fenetre de potentiel et C_H doivent etre positives.")
        if self.specific_surface_m2_kg <= 0.0 or self.dos_broadening_ev <= 0.0:
            raise ValueError("La surface specifique et l'elargissement doivent etre positifs.")
        if self.potential_sign not in (-1.0, 1.0):
            raise ValueError("potential_sign vaut +1 (cathode) ou -1 (anode).")


class MXeneElectrode:
    """Electrode MXene : capacite quantique, Helmholtz et charge specificque."""

    def __init__(self, config: ElectrodeConfig) -> None:
        self.config = config
        self.name = config.name
        self.model = TightBindingModel.mxene(
            a=config.lattice_a_angstrom,
            c=config.lattice_c_angstrom,
            z=config.lattice_z,
            t=config.hopping_t_ev,
            onsite=config.onsite_ev,
        )
        self.kpoints = uniform_k_grid(
            self.model.lattice.reciprocal_vectors, K_GRID, K_GRID, 1
        )
        self.cell_area_m2 = (
            np.sqrt(3.0) / 2.0 * config.lattice_a_angstrom**2 * ANGSTROM2_TO_M2
        )
        self._grids: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def dos(self, energies_ev, temperature_k: float = T_REF):
        """DOS (etats/eV/cellule) avec elargissement thermique additionnel."""
        temperature = float(_as_temperature(temperature_k))
        broadening = self.config.dos_broadening_ev + EV_PER_KELVIN * temperature
        return self.model.density_of_states(energies_ev, self.kpoints, broadening)

    def quantum_capacitance_areal(self, potentials_v, temperature_k: float = T_REF):
        """C_Q = e^2 DOS(E_F + eV)/A_cell, en F/m^2."""
        potentials = np.asarray(potentials_v, dtype=float)
        energies = self.config.efermi_ev + self.config.potential_sign * potentials
        dos = np.asarray(self.dos(energies, temperature_k), dtype=float)
        capacitance = float(Q_E) * dos / self.cell_area_m2
        return capacitance if capacitance.ndim else float(capacitance)

    def helmholtz_capacitance_areal(self, temperature_k: float = T_REF) -> float:
        """C_H(T) = C_H,ref [1 + a(T-T_ref)] * accessibilite(T), en F/m^2."""
        temperature = float(_as_temperature(temperature_k))
        return float(
            self.config.helmholtz_areal_f_m2
            * (1.0 + HELMHOLTZ_T_COEFF * (temperature - T_REF))
            * pore_accessibility(temperature)
        )

    def capacitance_areal(self, potentials_v, temperature_k: float = T_REF):
        """Association serie C_Q -- C_H par surface (F/m^2)."""
        quantum = np.asarray(
            self.quantum_capacitance_areal(potentials_v, temperature_k), dtype=float
        )
        helmholtz = self.helmholtz_capacitance_areal(temperature_k)
        series = 1.0 / (1.0 / np.maximum(quantum, 1.0e-30) + 1.0 / helmholtz)
        return series if series.ndim else float(series)

    def capacitance_specific(self, potentials_v, temperature_k: float = T_REF):
        """Capacite serie par masse active d'electrode (F/kg)."""
        return self.config.specific_surface_m2_kg * np.asarray(
            self.capacitance_areal(potentials_v, temperature_k), dtype=float
        )

    def _grid(self, temperature_k: float):
        key = round(float(_as_temperature(temperature_k)), 10)
        if key not in self._grids:
            potentials = np.linspace(0.0, self.config.window_v, GRID_POINTS)
            signed = self.config.potential_sign * potentials
            capacitance = self.capacitance_specific(signed, key)
            charge = np.zeros_like(potentials)
            charge[1:] = np.cumsum(
                0.5 * (capacitance[1:] + capacitance[:-1]) * np.diff(potentials)
            )
            self._grids[key] = (potentials, charge, capacitance)
        return self._grids[key]

    def charge_at(self, potential_v, temperature_k: float = T_REF):
        """Charge specificque (C/kg) accumulee de 0 a |V|."""
        potentials, charge, _ = self._grid(temperature_k)
        return np.interp(
            np.asarray(potential_v, dtype=float), potentials, charge
        )

    def potential_from_charge(self, charge_c_kg, temperature_k: float = T_REF):
        """Inversion monotone de la courbe q(V) (V)."""
        potentials, charge, _ = self._grid(temperature_k)
        values = np.asarray(charge_c_kg, dtype=float)
        result = np.interp(values, charge, potentials, left=0.0, right=potentials[-1])
        return result if result.ndim else float(result)

    def max_charge_specific(self, temperature_k: float = T_REF) -> float:
        """Charge specificque maximale de l'electrode (C/kg)."""
        return float(self.charge_at(self.config.window_v, temperature_k))


def positive_mxene_electrode() -> MXeneElectrode:
    """Electrode positive Ti2CTx, Fermi dans la bande de conduction."""
    return MXeneElectrode(
        ElectrodeConfig(
            name="Ti2CTx (MXene M2X, cathode)",
            efermi_ev=0.0,
            potential_sign=1.0,
            window_v=0.95,
        )
    )


def negative_mxene_electrode() -> MXeneElectrode:
    """Electrode negative Nb2CTx, sans plomb, Fermi decale de +0,25 eV."""
    return MXeneElectrode(
        ElectrodeConfig(
            name="Nb2CTx (MXene M2X, anode)",
            efermi_ev=0.25,
            potential_sign=-1.0,
            window_v=0.95,
            onsite_ev=(0.1, 0.1, -1.5),
            hopping_t_ev=1.1,
        )
    )


def _bisect(func, lower: float, upper: float, tol: float = 1.0e-12,
            max_iter: int = 100) -> float:
    """Bissection scalaire en Python pur, sans SciPy."""
    low, high = float(lower), float(upper)
    f_low = float(func(low))
    if abs(f_low) < tol:
        return low
    f_high = float(func(high))
    if abs(f_high) < tol:
        return high
    if f_low * f_high > 0.0:
        raise ValueError("La racine n'est pas encadree par le crochet.")
    for _ in range(int(max_iter)):
        middle = 0.5 * (low + high)
        f_middle = float(func(middle))
        if abs(f_middle) < tol or 0.5 * (high - low) < tol:
            return middle
        if f_low * f_middle <= 0.0:
            high, f_high = middle, f_middle
        else:
            low, f_low = middle, f_middle
    return 0.5 * (low + high)


@dataclass(frozen=True)
class GalvanostaticCycle:
    """Cycle de charge/decharge a courant constant, par kg de materiau actif."""

    temperature_k: float
    current_a_kg: float
    voltage_max_v: float
    esr_ohm_kg: float
    charge_charge_c_kg: np.ndarray
    charge_voltage_v: np.ndarray
    charge_time_s: np.ndarray
    discharge_charge_c_kg: np.ndarray
    discharge_voltage_v: np.ndarray
    discharge_time_s: np.ndarray
    charge_capacity_c_kg: float
    discharge_capacity_c_kg: float
    retained_charge_c_kg: float
    coulombic_efficiency: float
    energy_density_wh_kg: float
    power_density_w_kg: float
    capacity_mah_g: float
    energy_efficiency: float

    @property
    def temperature_c(self) -> float:
        return float(self.temperature_k) - KELVIN_OFFSET


@dataclass(frozen=True)
class RagonePoint:
    """Point energie/puissance specifique d'un regime galvanostatique."""

    temperature_k: float
    current_a_kg: float
    energy_wh_kg: float
    power_w_kg: float
    capacity_mah_g: float
    coulombic_efficiency: float
    esr_ohm_kg: float


class AsymmetricSupercapacitor:
    """Cellule asymetrique a deux electrodes MXene et bilan de charge exact."""

    def __init__(
        self,
        positive: MXeneElectrode | None = None,
        negative: MXeneElectrode | None = None,
        temperature_k: float = T_REF,
        voltage_max_v: float | None = None,
    ) -> None:
        self.positive = positive or positive_mxene_electrode()
        self.negative = negative or negative_mxene_electrode()
        self.temperature_k = float(_as_temperature(temperature_k))
        self.voltage_max_v = float(
            voltage_max_v
            if voltage_max_v is not None
            else self.positive.config.window_v + self.negative.config.window_v
        )
        if self.voltage_max_v <= 0.0:
            raise ValueError("La tension maximale de cellule doit etre positive.")
        q_pos = self.positive.max_charge_specific(T_REF)
        q_neg = self.negative.max_charge_specific(T_REF)
        self.mass_fraction_positive = q_neg / (q_pos + q_neg)
        self.mass_fraction_negative = q_pos / (q_pos + q_neg)

    def max_charge_specific(self, temperature_k: float | None = None) -> float:
        """Charge maximale admissible par kg de materiau actif total."""
        temperature = self.temperature_k if temperature_k is None else float(temperature_k)
        limits = (
            self.mass_fraction_positive * self.positive.max_charge_specific(temperature),
            self.mass_fraction_negative * self.negative.max_charge_specific(temperature),
        )
        return float(min(limits))

    def open_circuit_voltage(self, charge_c_kg, temperature_k: float | None = None):
        """Tension a circuit ouvert U(q) = V_+(q/f_+) + V_-(q/f_-)."""
        temperature = self.temperature_k if temperature_k is None else float(temperature_k)
        charge = np.asarray(charge_c_kg, dtype=float)
        positive_potential = self.positive.potential_from_charge(
            charge / self.mass_fraction_positive, temperature
        )
        negative_potential = self.negative.potential_from_charge(
            charge / self.mass_fraction_negative, temperature
        )
        voltage = np.asarray(positive_potential, dtype=float) + np.asarray(
            negative_potential, dtype=float
        )
        return voltage if voltage.ndim else float(voltage)

    def esr(self, temperature_k: float | None = None) -> float:
        temperature = self.temperature_k if temperature_k is None else temperature_k
        return float(esr(temperature))

    def galvanostatic_cycle(
        self,
        current_a_kg: float,
        temperature_k: float | None = None,
        n_points: int = 401,
    ) -> GalvanostaticCycle:
        """Charge puis decharge a courant constant avec surtension I*ESR(T)."""
        current = float(current_a_kg)
        if current <= 0.0:
            raise ValueError("Le courant galvanostatique doit etre strictement positif.")
        temperature = self.temperature_k if temperature_k is None else float(temperature_k)
        resistance = float(esr(temperature))
        limit = self.max_charge_specific(temperature)

        def charge_residual(charge: float) -> float:
            return float(self.open_circuit_voltage(charge, temperature) + current * resistance
                         - self.voltage_max_v)

        if charge_residual(limit) <= 0.0:
            full_charge = limit
        else:
            full_charge = _bisect(charge_residual, 0.0, limit)

        def discharge_residual(charge: float) -> float:
            return float(self.open_circuit_voltage(charge, temperature) - current * resistance)

        if discharge_residual(full_charge) <= 0.0:
            end_charge = full_charge
        else:
            end_charge = _bisect(discharge_residual, 0.0, full_charge)

        number = max(int(n_points), 8)
        charge_grid = np.linspace(0.0, full_charge, number)
        discharge_grid = np.linspace(full_charge, end_charge, number)
        charge_voltage = self.open_circuit_voltage(charge_grid, temperature) + current * resistance
        discharge_voltage = self.open_circuit_voltage(discharge_grid, temperature) - current * resistance
        charge_time = charge_grid / current
        discharge_time = charge_time[-1] + (full_charge - discharge_grid) / current

        delivered = full_charge - end_charge
        charge_energy = float(np.trapezoid(charge_voltage, charge_grid))
        discharge_energy = float(
            np.trapezoid(discharge_voltage[::-1], discharge_grid[::-1])
        )
        mean_voltage = float(np.mean(discharge_voltage))
        return GalvanostaticCycle(
            temperature_k=temperature,
            current_a_kg=current,
            voltage_max_v=self.voltage_max_v,
            esr_ohm_kg=resistance,
            charge_charge_c_kg=charge_grid,
            charge_voltage_v=np.asarray(charge_voltage, dtype=float),
            charge_time_s=charge_time,
            discharge_charge_c_kg=discharge_grid,
            discharge_voltage_v=np.asarray(discharge_voltage, dtype=float),
            discharge_time_s=discharge_time,
            charge_capacity_c_kg=float(full_charge),
            discharge_capacity_c_kg=float(delivered),
            retained_charge_c_kg=float(end_charge),
            coulombic_efficiency=float(delivered / full_charge),
            energy_density_wh_kg=float(discharge_energy / 3600.0),
            power_density_w_kg=float(current * mean_voltage),
            capacity_mah_g=float(delivered / 3600.0),
            energy_efficiency=float(discharge_energy / charge_energy),
        )

    def ragone_point(self, current_a_kg: float, temperature_k: float | None = None) -> RagonePoint:
        cycle = self.galvanostatic_cycle(current_a_kg, temperature_k)
        return RagonePoint(
            temperature_k=cycle.temperature_k,
            current_a_kg=cycle.current_a_kg,
            energy_wh_kg=cycle.energy_density_wh_kg,
            power_w_kg=cycle.power_density_w_kg,
            capacity_mah_g=cycle.capacity_mah_g,
            coulombic_efficiency=cycle.coulombic_efficiency,
            esr_ohm_kg=cycle.esr_ohm_kg,
        )

    def ragone_curve(
        self,
        currents_a_kg: Sequence[float],
        temperature_k: float | None = None,
    ) -> tuple[RagonePoint, ...]:
        return tuple(
            self.ragone_point(current, temperature_k) for current in currents_a_kg
        )

    def capacity_retention(
        self,
        current_a_kg: float,
        target_temperature_k: float = MATAM.ambient_k,
        reference_temperature_k: float = T_REF,
    ) -> float:
        """Rapport des capacites dechargees a T cible et T de reference."""
        hot = self.galvanostatic_cycle(current_a_kg, target_temperature_k)
        cold = self.galvanostatic_cycle(current_a_kg, reference_temperature_k)
        return float(hot.discharge_capacity_c_kg / cold.discharge_capacity_c_kg)


def site_ragone_point(site: SahelSite, current_a_kg: float,
                      cell: AsymmetricSupercapacitor | None = None) -> RagonePoint:
    """Point Ragone aux conditions climatiques d'un site sahelien."""
    device = cell or AsymmetricSupercapacitor()
    return device.ragone_point(current_a_kg, site.ambient_k)


def symbolic_series_capacitance() -> dict[str, sp.Expr]:
    """Verification SymPy de l'association en serie C_Q -- C_H."""
    quantum, helmholtz, total, charge = sp.symbols(
        "C_Q C_H C_tot q", positive=True
    )
    series = quantum * helmholtz / (quantum + helmholtz)
    voltage_split = charge / quantum + charge / helmholtz
    voltage_series = charge / series
    energy_split = sp.Rational(1, 2) * voltage_split**2 * series
    energy_series = charge**2 / (2 * series)
    return {
        "quantum": quantum,
        "helmholtz": helmholtz,
        "capacitance": series,
        "series_check": sp.simplify(1 / series - 1 / quantum - 1 / helmholtz),
        "voltage_check": sp.simplify(voltage_split - voltage_series),
        "energy_check": sp.simplify(energy_split - energy_series),
        "derivative": sp.diff(series, quantum),
        "quantum_bound": sp.simplify(series - quantum),
        "helmholtz_bound": sp.simplify(series - helmholtz),
    }


def symbolic_quantum_capacitance() -> dict[str, sp.Expr]:
    """Verification SymPy de C_Q = e^2 DOS et de la limite C_H >> C_Q."""
    electron, density, helmholtz = sp.symbols("e g C_H", positive=True)
    quantum = electron**2 * density
    series = quantum * helmholtz / (quantum + helmholtz)
    return {
        "quantum": quantum,
        "series_limit_large_ch": sp.limit(series, helmholtz, sp.oo),
        "series_limit_large_cq": sp.limit(series, density, sp.oo),
        "scaling_check": sp.simplify(quantum / density - electron**2),
    }
