"""UAM-TOPOMAT -- Optique cristalline et rendement photovoltaique.

Deux parties :

  1. absorption optique alpha(lambda) issue de la structure de bandes E_n(k)
     et des elements de matrice dipolaires calcules a partir de
     dH/dk (vitesses de Fermi d'or interbandes) ;
  2. modele complet de Shockley-Queisser : spectre solaire AM1.5G (ASTM G173,
     table interpolee et normalisee a 1000 W/m^2), bilan de photons, courant
     d'obscurite radiatif, facteur de remplissage et rendement eta(E_g, T).

Python pur : numpy, sympy (verifications symboliques dans les tests),
matplotlib (figures). Aucune dependance a SciPy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

Q_E = 1.602176634e-19
H_SI = 6.62607015e-34
HBAR_SI = H_SI / (2.0 * np.pi)
KB_SI = 1.380649e-23
C_SI = 299792458.0
EPS0_SI = 8.8541878128e-12
EV_ANGSTROM_SI = Q_E * 1.0e-10
HC_EV_NM = 1239.8419843320026
T_REF = 298.15
P_INPUT = 1000.0


AM15G_WAVELENGTHS_NM = np.array(
    [
        280.0, 290.0, 300.0, 310.0, 320.0, 330.0, 340.0, 350.0,
        360.0, 370.0, 380.0, 390.0, 400.0, 410.0, 420.0, 430.0,
        440.0, 450.0, 460.0, 470.0, 480.0, 490.0, 500.0, 510.0,
        520.0, 530.0, 540.0, 550.0, 560.0, 570.0, 580.0, 590.0,
        600.0, 610.0, 620.0, 630.0, 640.0, 650.0, 660.0, 670.0,
        680.0, 690.0, 700.0, 710.0, 720.0, 730.0, 740.0, 750.0,
        760.0, 770.0, 780.0, 790.0, 800.0, 820.0, 840.0, 860.0,
        880.0, 900.0, 920.0, 940.0, 960.0, 980.0, 1000.0, 1020.0,
        1040.0, 1060.0, 1080.0, 1100.0, 1130.0, 1160.0, 1190.0, 1220.0,
        1250.0, 1280.0, 1310.0, 1340.0, 1370.0, 1400.0, 1430.0, 1460.0,
        1490.0, 1520.0, 1550.0, 1580.0, 1610.0, 1640.0, 1670.0, 1700.0,
        1750.0, 1800.0, 1850.0, 1900.0, 1950.0, 2000.0, 2050.0, 2100.0,
        2150.0, 2200.0, 2250.0, 2300.0, 2400.0, 2500.0, 2600.0, 2800.0,
        2850.0, 3000.0, 3100.0, 3200.0, 3300.0, 3500.0, 3700.0, 4000.0,
    ]
)

AM15G_IRRADIANCE = np.array(
    [
        0.000, 0.001, 0.005, 0.018, 0.060, 0.115, 0.165, 0.235,
        0.310, 0.420, 0.540, 0.680, 0.850, 1.020, 1.180, 1.330,
        1.450, 1.560, 1.660, 1.740, 1.800, 1.850, 1.895, 1.925,
        1.945, 1.958, 1.962, 1.955, 1.940, 1.920, 1.895, 1.865,
        1.830, 1.790, 1.745, 1.700, 1.650, 1.600, 1.545, 1.490,
        1.435, 1.380, 1.325, 1.270, 1.215, 1.160, 1.105, 1.050,
        0.990, 1.240, 1.315, 1.305, 1.280, 1.220, 1.150, 1.075,
        1.010, 0.955, 0.837, 0.522, 0.794, 0.752, 0.729, 0.682,
        0.639, 0.592, 0.550, 0.508, 0.465, 0.428, 0.395, 0.362,
        0.329, 0.296, 0.268, 0.255, 0.225, 0.195, 0.160, 0.050,
        0.105, 0.205, 0.220, 0.205, 0.185, 0.165, 0.145, 0.125,
        0.105, 0.090, 0.075, 0.055, 0.035, 0.020, 0.010, 0.005,
        0.015, 0.055, 0.085, 0.095, 0.090, 0.075, 0.060, 0.045,
        0.035, 0.025, 0.018, 0.010, 0.006, 0.003, 0.0015, 0.0004,
    ]
)

_SOLAR_GRID = np.arange(280.0, 4000.0 + 1.0, 1.0)
_RAW_IRRADIANCE = np.interp(_SOLAR_GRID, AM15G_WAVELENGTHS_NM, AM15G_IRRADIANCE)
_RAW_IRRADIANCE[_SOLAR_GRID < AM15G_WAVELENGTHS_NM[0]] = 0.0
_RAW_IRRADIANCE[_SOLAR_GRID > AM15G_WAVELENGTHS_NM[-1]] = 0.0
_AM15G_SCALE = P_INPUT / float(np.trapezoid(_RAW_IRRADIANCE, _SOLAR_GRID))
SOLAR_IRRADIANCE = _RAW_IRRADIANCE * _AM15G_SCALE


def am15g_irradiance(wavelength_nm) -> np.ndarray:
    """Irradiance spectral AM1.5G (W/m^2/nm) sur le domaine 280-4000 nm."""
    wavelengths = np.asarray(wavelength_nm, dtype=float)
    values = np.interp(wavelengths, _SOLAR_GRID, SOLAR_IRRADIANCE, left=0.0, right=0.0)
    return values


def solar_grid() -> tuple[np.ndarray, np.ndarray]:
    """Grille de reference (lambda [nm], irradiance [W/m^2/nm])."""
    return _SOLAR_GRID.copy(), SOLAR_IRRADIANCE.copy()


def photon_flux_above_gap(eg_ev: float) -> float:
    """Flux de photons AM1.5G avec E > E_g (photons/m^2/s, EQE = 1)."""
    if eg_ev <= 0.0:
        raise ValueError("E_g doit etre strictement positif.")
    cutoff = min(HC_EV_NM / float(eg_ev), _SOLAR_GRID[-1])
    if cutoff <= _SOLAR_GRID[0]:
        return 0.0
    mask = _SOLAR_GRID <= cutoff
    wavelengths = _SOLAR_GRID[mask]
    irradiance = SOLAR_IRRADIANCE[mask]
    energies_joule = Q_E * HC_EV_NM / wavelengths
    return float(np.trapezoid(irradiance / energies_joule, wavelengths))


def fermi_occupation(energies, chemical_potential: float, temperature: float) -> np.ndarray:
    """Distribution de Fermi-Dirac (energies et potentiel en eV)."""
    kbt = KB_SI * float(temperature) / Q_E
    argument = (np.asarray(energies, dtype=float) - chemical_potential) / kbt
    return 1.0 / (1.0 + np.exp(np.clip(argument, -600.0, 600.0)))


def dipole_matrix_element(
    model,
    kpoint,
    valence_band: int,
    conduction_band: int,
    direction: int = 0,
) -> complex:
    """Element de matrice dipolaire <c|r|v> (metres) pour une polarisation.

    La vitesse interbande v_cv = (1/hbar) <c|dH/dk|v> est convertie en m/s
    depuis les unitees eV.A du modele tight-binding, puis
    r_cv = -i hbar v_cv / (E_c - E_v).
    """
    kpoint = np.asarray(kpoint, dtype=float).reshape(-1)
    energies, vectors = model.eigen_system(kpoint)
    derivatives = np.asarray(model.hamiltonian_derivatives(kpoint))
    if direction >= derivatives.shape[0]:
        raise ValueError("direction hors limites.")
    velocity = np.vdot(vectors[:, conduction_band], derivatives[direction] @ vectors[:, valence_band])
    velocity_si = velocity * (EV_ANGSTROM_SI / HBAR_SI)
    delta_e = float(energies[conduction_band] - energies[valence_band])
    if abs(delta_e) < 1e-12:
        return 0.0j
    return -1j * HBAR_SI * velocity_si / (delta_e * Q_E)


@dataclass
class AbsorptionResult:
    """Spectre d'absorption alpha(lambda) issu d'un modele de bandes."""

    wavelengths_nm: np.ndarray
    photon_energies_ev: np.ndarray
    alpha_cm: np.ndarray
    cell_volume_m3: float
    n_kpoints: int
    broadening_ev: float
    refractive_index: float
    metadata: dict = field(default_factory=dict)

    @property
    def alpha_max_cm(self) -> float:
        return float(np.max(self.alpha_cm))

    def alpha_at(self, wavelength_nm: float) -> float:
        index = int(np.argmin(np.abs(self.wavelengths_nm - wavelength_nm)))
        return float(self.alpha_cm[index])


def absorption_spectrum(
    model,
    kpoints: Sequence | np.ndarray,
    n_valence: int,
    wavelengths_nm: Sequence[float] | np.ndarray,
    broadening_ev: float = 0.05,
    refractive_index: float = 2.5,
    cell_volume_a3: float | None = None,
    temperature: float = T_REF,
    directions: Sequence[int] = (0, 1),
) -> AbsorptionResult:
    """Calcule alpha(lambda) par regle d'or de Fermi avec elargissement Lorentzien.

    Formule utilisee (unites SI, alpha en m^-1) :

        alpha(w) = pi e^2 w / (eps_0 c n_r V)
                   sum_{v,c,k} |e.r_cv|^2 (f_v - f_c) delta(E_c-E_v - hbar w).

    Les occupations f sont those de Fermi-Dirac a dopage intrinsèque, avec
    mu = (max E_v + min E_c)/2. Le facteur 1/N_k normalise la somme sur k.
    """
    wavelengths = np.asarray(wavelengths_nm, dtype=float).reshape(-1)
    if np.any(wavelengths <= 0.0):
        raise ValueError("Les longueurs d'onde doivent etre positives.")
    if broadening_ev <= 0.0:
        raise ValueError("broadening_ev doit etre strictement positif.")
    kpoints = np.atleast_2d(np.asarray(kpoints, dtype=float))
    if kpoints.shape[0] == 0:
        raise ValueError("kpoints ne doit pas etre vide.")

    bands = np.asarray([model.eigenvalues(k) for k in kpoints])
    n_bands = bands.shape[1]
    if not 1 <= n_valence < n_bands:
        raise ValueError("n_valence doit etre compris entre 1 et n_bands-1.")
    chemical_potential = 0.5 * (float(np.max(bands[:, :n_valence])) + float(np.min(bands[:, n_valence:])))
    occupations = fermi_occupation(bands, chemical_potential, temperature)

    if cell_volume_a3 is None:
        lattice = getattr(model, "lattice", None)
        if lattice is None or not hasattr(lattice, "volume"):
            raise ValueError("Fournir cell_volume_a3 ou un modele muni d'un reseau.")
        cell_volume_a3 = float(lattice.volume)
    cell_volume_m3 = float(cell_volume_a3) * 1.0e-30

    energies_ev = HC_EV_NM / wavelengths
    gamma = float(broadening_ev)
    intensities = np.zeros_like(energies_ev)
    n_directions = len(directions)

    for k_index, kpoint in enumerate(kpoints):
        _, vectors = model.eigen_system(kpoint)
        derivatives = np.asarray(model.hamiltonian_derivatives(kpoint))
        for valence in range(n_valence):
            for conduction in range(n_valence, n_bands):
                delta_e = float(bands[k_index, conduction] - bands[k_index, valence])
                if delta_e <= 0.0:
                    continue
                dipole_squared = 0.0
                for direction in directions:
                    velocity = np.vdot(
                        vectors[:, conduction], derivatives[direction] @ vectors[:, valence]
                    )
                    velocity_si = velocity * (EV_ANGSTROM_SI / HBAR_SI)
                    dipole = -1j * HBAR_SI * velocity_si / (delta_e * Q_E)
                    dipole_squared += abs(dipole) ** 2
                dipole_squared /= n_directions
                population = occupations[k_index, valence] - occupations[k_index, conduction]
                if population <= 0.0:
                    continue
                lorentzian = (gamma / np.pi) / ((energies_ev - delta_e) ** 2 + gamma**2)
                intensities += dipole_squared * population * lorentzian / Q_E

    omega = Q_E * energies_ev / HBAR_SI
    prefactor = np.pi * Q_E**2 * omega / (
        EPS0_SI * C_SI * refractive_index * cell_volume_m3 * len(kpoints)
    )
    alpha_m = prefactor * intensities
    return AbsorptionResult(
        wavelengths_nm=wavelengths,
        photon_energies_ev=energies_ev,
        alpha_cm=alpha_m / 100.0,
        cell_volume_m3=cell_volume_m3,
        n_kpoints=int(len(kpoints)),
        broadening_ev=gamma,
        refractive_index=float(refractive_index),
        metadata={
            "chemical_potential_ev": chemical_potential,
            "n_valence": int(n_valence),
            "n_bands": int(n_bands),
        },
    )


@dataclass
class SQResult:
    """Bilan detaille d'une cellule de Shockley-Queisser sous 1 soleil."""

    bandgap_ev: float
    temperature: float
    concentration: float
    photon_flux: float
    jsc_ma_cm2: float
    j0_ma_cm2: float
    voc: float
    ff: float
    vmp: float
    jmp_ma_cm2: float
    efficiency: float
    incident_power: float

    def __repr__(self) -> str:
        return (
            f"SQResult(Eg={self.bandgap_ev:.3f} eV, T={self.temperature:.2f} K, "
            f"eta={100.0 * self.efficiency:.3f} %, Voc={self.voc:.4f} V, "
            f"FF={self.ff:.4f})"
        )


def radiative_saturation_current(eg_ev: float, temperature: float) -> float:
    """Courant de saturations radiatif J_0 (A/m^2) emission hemispherique."""
    eg_joule = eg_ev * Q_E
    kbt = KB_SI * float(temperature)
    upper = eg_joule + 60.0 * kbt
    energies = np.linspace(eg_joule, upper, 4000)
    integrand = energies**2 / np.expm1(energies / kbt)
    flux = (2.0 * np.pi / (H_SI**3 * C_SI**2)) * float(np.trapezoid(integrand, energies))
    return Q_E * flux


def _maximum_power_point(jsc: float, j0: float, thermal_voltage: float) -> tuple[float, float]:
    """Resout (1+v) e^v = 1 + J_sc/J_0 par Newton (sans SciPy)."""
    ratio = jsc / j0
    v = max(0.0, np.log1p(ratio) - 1.0)
    for _ in range(200):
        residual = (1.0 + v) * np.exp(v) - (1.0 + ratio)
        derivative = (2.0 + v) * np.exp(v)
        step = residual / derivative
        v -= step
        if abs(step) < 1e-14:
            break
    if v < 0.0:
        v = 0.0
    vmp = thermal_voltage * v
    jmp = jsc - j0 * np.expm1(v)
    return vmp, jmp


def shockley_queisser(
    eg_ev: float,
    temperature: float = T_REF,
    concentration: float = 1.0,
    incident_power: float = P_INPUT,
) -> SQResult:
    """Modele Shockley-Queisser complet sous illumination AM1.5G.

    J(V) = J_sc - J_0 (exp(qV/kT) - 1), avec J_sc issu du flux de photons
    AM1.5G au-dessus du gap et J_0 de la recombination radiative en 2pi.
    """
    eg_ev = float(eg_ev)
    if eg_ev <= 0.0:
        raise ValueError("E_g doit etre strictement positif.")
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError("La temperature doit etre strictement positive.")
    flux = photon_flux_above_gap(eg_ev)
    jsc = Q_E * flux * float(concentration)
    j0 = radiative_saturation_current(eg_ev, temperature) * float(concentration)
    thermal_voltage = KB_SI * temperature / Q_E
    voc = thermal_voltage * float(np.log1p(jsc / j0))
    vmp, jmp = _maximum_power_point(jsc, j0, thermal_voltage)
    pmax = max(0.0, vmp * jmp)
    ff = pmax / (voc * jsc) if voc * jsc > 0.0 else 0.0
    return SQResult(
        bandgap_ev=eg_ev,
        temperature=temperature,
        concentration=float(concentration),
        photon_flux=flux,
        jsc_ma_cm2=jsc / 10.0,
        j0_ma_cm2=j0 / 10.0,
        voc=voc,
        ff=ff,
        vmp=vmp,
        jmp_ma_cm2=jmp / 10.0,
        efficiency=pmax / float(incident_power),
        incident_power=float(incident_power),
    )


def sq_efficiency(eg_ev: float, temperature: float = T_REF) -> float:
    """Rendement Shockley-Queisser (fraction) pour E_g [eV] et T [K]."""
    return shockley_queisser(eg_ev, temperature).efficiency


def sq_efficiency_curve(
    bandgaps_ev: Sequence[float] | np.ndarray,
    temperature: float = T_REF,
) -> np.ndarray:
    """Rendement pour un tableau de gaps, a temperature fixee."""
    return np.array([sq_efficiency(float(eg), temperature) for eg in bandgaps_ev])


def optimize_bandgap(
    temperature: float = T_REF,
    bounds_ev: tuple[float, float] = (0.5, 2.5),
    n_points: int = 800,
) -> tuple[float, float]:
    """Maximum (E_g, eta) du rendement SQ par balayage plus raffinement dore."""
    low, high = float(bounds_ev[0]), float(bounds_ev[1])
    grid = np.linspace(low, high, int(n_points))
    efficiencies = sq_efficiency_curve(grid, temperature)
    index = int(np.argmax(efficiencies))
    if index == 0 or index == len(grid) - 1:
        return float(grid[index]), float(efficiencies[index])
    left, right = grid[index - 1], grid[index + 1]
    for _ in range(60):
        third = (right - left) / 3.0
        c1, c2 = left + third, right - third
        if sq_efficiency(c1, temperature) < sq_efficiency(c2, temperature):
            left = c1
        else:
            right = c2
    best_gap = 0.5 * (left + right)
    return best_gap, sq_efficiency(best_gap, temperature)


def thermal_degradation(
    eg_ev: float = 1.34,
    temperatures: Sequence[float] | np.ndarray = (298.15, 303.15, 318.15, 330.15),
) -> np.ndarray:
    """Rendements SQ pour une sequence de temperatures (K)."""
    return np.array([sq_efficiency(eg_ev, float(t)) for t in temperatures])
