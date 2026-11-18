"""UAM-TOPOMAT -- Nuit 6 : EDP non lineaires, derive-diffusion et courbes J(V).

Solveur stationnaire 1D couples Poisson -- derive--diffusion pour une junction
pn photovoltaique, discretise par le schema exponentiel conservatif exact de
Scharfetter--Gummel. Python pur : numpy, sympy (verifications symboliques),
matplotlib (figures). Aucune dependance SciPy.

Equations resolues (unites SI, phi en volts, n et p en m^-3) :

    d/dx(eps dphi/dx) = -q(p - n + N_D - N_A)                (Poisson)
    (1/q) dJ_n/dx = R - G,      (1/q) dJ_p/dx = G - R        (continuite)
    J_n = q mu_n n E + q D_n dn/dx,  J_p = q mu_p p E - q D_p dp/dx
    D_n = V_T mu_n,  D_p = V_T mu_p,  V_T = k_B T_j / q      (Einstein)

Variables de Slotboom w = n exp(-phi/V_T), z = p exp(phi/V_T) : le flux de
Scharfetter--Gummel s'ecrit J_n = A_i (w_{i+1}-w_i) et
J_p = C_i (z_i-z_{i+1}), avec A_i et C_i strictement positifs. Le courant
total J_n+J_p est donc conserve a la precision machine.

Le solveur est couple a la temperature de jonction T_j du modele
thermodynamique de la Nuit 5 (src/thermo/sahel_climate.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import sympy as sp

from src.optics.solar_optics import (
    C_SI,
    EPS0_SI,
    H_SI,
    KB_SI,
    P_INPUT,
    Q_E,
    T_REF,
    photon_flux_above_gap,
    radiative_saturation_current,
)
from src.thermo.sahel_climate import PVModule, SahelSite, energy_balance

KELVIN_OFFSET = 273.15
TOL_PHOTO = 1.0e-13
GUMMEL_TOL = 1.0e-9
MAX_GUMMEL = 160
MAX_NEWTON = 40


def thermal_voltage(temperature_k: float) -> float:
    """Tension thermique V_T = k_B T_j / q (V)."""
    return float(KB_SI) * float(temperature_k) / float(Q_E)


def bernoulli(z):
    """B(z) = z/(exp(z)-1), stable pour |z| petit et |z| grand."""
    values = np.asarray(z, dtype=float)
    result = np.empty_like(values)
    small = np.abs(values) < 1.0e-5
    z_small = values[small]
    result[small] = 1.0 - z_small / 2.0 + z_small**2 / 12.0
    outside = ~small
    z_out = values[outside]
    safe = np.clip(z_out, -700.0, 700.0)
    ratio = z_out / np.expm1(safe)
    result[outside] = np.where(
        z_out > 100.0, 0.0, np.where(z_out < -100.0, -z_out, ratio)
    )
    return result if result.ndim else float(result)


def einstein_diffusivity(mobility: float, temperature_k: float) -> float:
    """Relation d'Einstein D = (k_B T_j/q) mu (m^2/s)."""
    return float(mobility) * thermal_voltage(temperature_k)


def intrinsic_density(
    bandgap_ev: float,
    temperature_k: float,
    electron_mass_rel: float,
    hole_mass_rel: float,
) -> float:
    """n_i(T) = sqrt(N_c N_v) exp(-E_g/(2 k_B T)), masses en unites de m_0."""
    m0 = 9.1093837015e-31
    kbt = KB_SI * float(temperature_k)
    prefactor = 2.0 * (2.0 * np.pi * m0 * kbt / H_SI**2) ** 1.5
    nc = prefactor * float(electron_mass_rel) ** 1.5
    nv = prefactor * float(hole_mass_rel) ** 1.5
    return float(np.sqrt(nc * nv) * np.exp(-float(bandgap_ev) * Q_E / (2.0 * kbt)))


def mobility_at_temperature(mobility_300: float, temperature_k: float) -> float:
    """Mobilite phononique mu(T) = mu_300 (T/300)^(-3/2) (m^2/V/s)."""
    return float(mobility_300) * (float(temperature_k) / T_REF) ** (-1.5)


@dataclass(frozen=True)
class DeviceConfig:
    """Parametres physiques et geometriques d'une cellule pn 1D."""

    name: str
    bandgap_ev: float
    thickness_m: float
    n_points: int
    acceptor_density: float
    donor_density: float
    junction_width_m: float
    absorption_coefficient: float
    quantum_efficiency: float
    tau_n: float
    tau_p: float
    mu_n_300: float
    mu_p_300: float
    relative_permittivity: float
    electron_mass_rel: float
    hole_mass_rel: float

    def __post_init__(self) -> None:
        if int(self.n_points) < 21:
            raise ValueError("La grille doit comporter au moins 21 noeuds.")
        if self.thickness_m <= 0.0 or self.bandgap_ev <= 0.0:
            raise ValueError("Epaisseur et gap doivent etre positifs.")

    def __repr__(self) -> str:
        return (
            f"DeviceConfig({self.name}, Eg={self.bandgap_ev:.2f} eV, "
            f"L={1e6 * self.thickness_m:.2f} um, N={self.n_points})"
        )


def perovskite_device(n_points: int = 121) -> DeviceConfig:
    """Pervoskite / topologique : gap 1,45 eV, forte absorption, faible tau."""
    return DeviceConfig(
        name="Perovskite/topologique",
        bandgap_ev=1.45,
        thickness_m=1.5e-6,
        n_points=n_points,
        acceptor_density=1.0e22,
        donor_density=1.0e22,
        junction_width_m=4.0e-8,
        absorption_coefficient=3.0e6,
        quantum_efficiency=0.92,
        tau_n=2.0e-8,
        tau_p=2.0e-8,
        mu_n_300=0.02,
        mu_p_300=0.008,
        relative_permittivity=12.0,
        electron_mass_rel=0.22,
        hole_mass_rel=0.28,
    )


def silicon_device(n_points: int = 121) -> DeviceConfig:
    """Silicium cristallin : gap 1,12 eV, absorption faible, longue diffusion."""
    return DeviceConfig(
        name="Silicium cristallin",
        bandgap_ev=1.12,
        thickness_m=50.0e-6,
        n_points=n_points,
        acceptor_density=1.0e22,
        donor_density=1.0e22,
        junction_width_m=5.0e-7,
        absorption_coefficient=3.0e5,
        quantum_efficiency=0.90,
        tau_n=5.0e-6,
        tau_p=5.0e-6,
        mu_n_300=0.14,
        mu_p_300=0.05,
        relative_permittivity=11.7,
        electron_mass_rel=1.08,
        hole_mass_rel=0.56,
    )


def device_from_module(module: PVModule, n_points: int = 121) -> DeviceConfig:
    """Selectionne la cellule associee a un module de la Nuit 5."""
    device = silicon_device(n_points) if module.bandgap_ev < 1.3 else perovskite_device(n_points)
    if abs(device.bandgap_ev - float(module.bandgap_ev)) > 1.0e-9:
        from dataclasses import replace

        device = replace(device, bandgap_ev=float(module.bandgap_ev))
    return device


@dataclass(frozen=True)
class CarrierSolution:
    """Etat stationnaire converger d'une cellule sous polarisation V."""

    device: DeviceConfig
    temperature_k: float
    voltage: float
    irradiance: float
    x: np.ndarray
    potential: np.ndarray
    electrons: np.ndarray
    holes: np.ndarray
    quasi_fermi_n: np.ndarray
    quasi_fermi_p: np.ndarray
    current_n: np.ndarray
    current_p: np.ndarray
    generation: np.ndarray
    recombination: np.ndarray
    current_density: float
    iterations: int
    converged: bool
    thermal_voltage_v: float = 0.0
    intrinsic_density_m3: float = 0.0
    electron_mobility: float = 0.0
    hole_mobility: float = 0.0

    @property
    def total_current(self) -> np.ndarray:
        return self.current_n + self.current_p

    @property
    def junction_temperature_c(self) -> float:
        return float(self.temperature_k) - KELVIN_OFFSET

    def __repr__(self) -> str:
        return (
            f"CarrierSolution({self.device.name}, V={self.voltage:.4f} V, "
            f"T_j={self.junction_temperature_c:.2f} C, J={self.current_density:.4f} A/m2)"
        )


@dataclass
class _Physics:
    """Coefficients figures d'une resolution a temperature et eclairage donnes."""

    device: DeviceConfig
    temperature_k: float
    irradiance: float
    voltage: float
    x: np.ndarray
    h: float
    n_doping: np.ndarray
    a_doping: np.ndarray
    vt: float
    ni: float
    mu_n: float
    mu_p: float
    eps: float
    generation: np.ndarray
    phi_left: float
    phi_right: float
    w_left: float
    w_right: float
    z_left: float
    z_right: float

    def u(self, phi: np.ndarray) -> np.ndarray:
        return phi / self.vt

    def carriers(self, phi: np.ndarray, w: np.ndarray, z: np.ndarray):
        u = self.u(phi)
        u = np.clip(u, -600.0, 600.0)
        return w * np.exp(u), z * np.exp(-u)

    def recombination_rate(self, n: np.ndarray, p: np.ndarray) -> np.ndarray:
        numerator = n * p - self.ni**2
        denominator = self.device.tau_p * (n + self.ni) + self.device.tau_n * (p + self.ni)
        return numerator / denominator

    def poisson_residual(self, phi: np.ndarray, w: np.ndarray, z: np.ndarray) -> np.ndarray:
        n, p = self.carriers(phi, w, z)
        interior = np.zeros_like(phi)
        interior[1:-1] = (
            self.eps * (phi[2:] - 2.0 * phi[1:-1] + phi[:-2]) / self.h**2
            + Q_E * (p[1:-1] - n[1:-1] + self.n_doping[1:-1] - self.a_doping[1:-1])
        )
        return interior

    def poisson_matrix(self, phi: np.ndarray, w: np.ndarray, z: np.ndarray):
        n, p = self.carriers(phi, w, z)
        size = phi.size
        lower = np.zeros(size)
        diag = np.zeros(size)
        upper = np.zeros(size)
        interior = slice(1, -1)
        coupling = self.eps / self.h**2
        lower[1:-1] = coupling
        upper[1:-1] = coupling
        diag[interior] = -2.0 * coupling - Q_E * (n[interior] + p[interior]) / self.vt
        diag[0] = diag[-1] = 1.0
        return lower, diag, upper

    def scharfetter_coefficients(self, phi: np.ndarray):
        u = np.clip(self.u(phi), -600.0, 600.0)
        du = np.diff(u)
        b_pos = bernoulli(du)
        b_neg = bernoulli(-du)
        scale_n = Q_E * self.mu_n * self.vt / self.h
        scale_p = Q_E * self.mu_p * self.vt / self.h
        a_coeff = scale_n * b_neg * np.exp(u[:-1])
        c_coeff = scale_p * b_pos * np.exp(-u[:-1])
        return a_coeff, c_coeff, b_pos, b_neg


def _build_physics(
    device: DeviceConfig,
    temperature_k: float,
    voltage: float,
    irradiance: float,
) -> _Physics:
    if temperature_k <= 0.0:
        raise ValueError("La temperature de jonction doit etre positive.")
    if irradiance < 0.0:
        raise ValueError("L'irradiance doit etre positive.")

    count = int(device.n_points)
    x = np.linspace(0.0, device.thickness_m, count)
    h = float(x[1] - x[0])
    transition = np.tanh((x - 0.5 * device.thickness_m) / device.junction_width_m)
    donor = 0.5 * device.donor_density * (1.0 + transition)
    acceptor = 0.5 * device.acceptor_density * (1.0 - transition)
    vt = thermal_voltage(temperature_k)
    ni = intrinsic_density(
        device.bandgap_ev, temperature_k, device.electron_mass_rel, device.hole_mass_rel
    )
    flux = photon_flux_above_gap(device.bandgap_ev) * (float(irradiance) / P_INPUT)
    generation = (
        device.quantum_efficiency
        * flux
        * device.absorption_coefficient
        * np.exp(-device.absorption_coefficient * x)
    )
    phi_left = -vt * np.log(acceptor[0] / ni)
    phi_right = -float(voltage) + vt * np.log(donor[-1] / ni)
    w_left = ni * np.exp(float(voltage) / vt * 0.0)
    w_right = ni * np.exp(float(voltage) / vt)
    z_left = ni
    z_right = ni * np.exp(-float(voltage) / vt)
    return _Physics(
        device=device,
        temperature_k=float(temperature_k),
        irradiance=float(irradiance),
        voltage=float(voltage),
        x=x,
        h=h,
        n_doping=donor,
        a_doping=acceptor,
        vt=vt,
        ni=ni,
        mu_n=mobility_at_temperature(device.mu_n_300, temperature_k),
        mu_p=mobility_at_temperature(device.mu_p_300, temperature_k),
        eps=device.relative_permittivity * EPS0_SI,
        generation=generation,
        phi_left=phi_left,
        phi_right=phi_right,
        w_left=w_left,
        w_right=w_right,
        z_left=z_left,
        z_right=z_right,
    )


def _tridiagonal_solve(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Thomas (sans pivotage) pour une matrice tridiagonale de diagonale dominante."""
    size = diag.size
    c_prime = np.zeros(size)
    d_prime = np.zeros(size)
    denominator = diag[0]
    if denominator == 0.0:
        raise ZeroDivisionError("Diagonale singuliere dans le solveur tridiagonal.")
    c_prime[0] = upper[0] / denominator
    d_prime[0] = rhs[0] / denominator
    for index in range(1, size):
        denominator = diag[index] - lower[index] * c_prime[index - 1]
        if denominator == 0.0:
            raise ZeroDivisionError("Pivot nul dans le solveur tridiagonal.")
        c_prime[index] = upper[index] / denominator
        d_prime[index] = (rhs[index] - lower[index] * d_prime[index - 1]) / denominator
    solution = np.zeros(size)
    solution[-1] = d_prime[-1]
    for index in range(size - 2, -1, -1):
        solution[index] = d_prime[index] - c_prime[index] * solution[index + 1]
    return solution


def _newton_poisson(physics: _Physics, phi: np.ndarray, w: np.ndarray, z: np.ndarray, tol: float) -> tuple[np.ndarray, int]:
    """Newton amorti sur Poisson avec w et z figes (variables de Slotboom).

    Le critere d'arret porte sur l'amplitude du pas en potentiel : le residu
    fait intervenir des termes de charge de l'ordre de 10^3 C/m^3 qui s'annulent
    entre eux, si bien qu'un critere absolu trop strict sur le residu serait
    inaccessible en precision flottante.
    """
    current = phi.copy()
    for iteration in range(MAX_NEWTON):
        lower, diag, upper = physics.poisson_matrix(current, w, z)
        residual = physics.poisson_residual(current, w, z)
        rhs = -residual.copy()
        rhs[0] = physics.phi_left - current[0]
        rhs[-1] = physics.phi_right - current[-1]
        step = _tridiagonal_solve(lower, diag, upper, rhs)
        if float(np.max(np.abs(step[1:-1]))) < float(tol):
            break
        norm = float(np.max(np.abs(residual[1:-1])))
        damping = 1.0
        accepted = False
        for _ in range(24):
            trial = current + damping * step
            trial_norm = float(np.max(np.abs(physics.poisson_residual(trial, w, z)[1:-1])))
            if np.isfinite(trial_norm) and trial_norm <= norm * (1.0 + 1.0e-9):
                current = trial
                accepted = True
                break
            damping *= 0.5
        if not accepted:
            break
    return current, iteration + 1


def _initial_potential(physics: _Physics) -> np.ndarray:
    """Potentiel initial : transition de depletion lisse et ecransee.

    Le dopage ionise seul donne un potentiel divergent des que le pas de maillage
    depasse la longueur de Debye. On utilise donc une transition tanh de largeur
    max(L_D, 2h), ou L_D = sqrt(eps V_T/(q N_A)) est la longueur de Debye.
    """
    length = physics.device.thickness_m
    debye = np.sqrt(
        physics.eps * physics.vt / (Q_E * physics.device.acceptor_density)
    )
    width = max(float(debye), 2.0 * physics.h)
    midpoint = 0.5 * (physics.phi_left + physics.phi_right)
    amplitude = 0.5 * (physics.phi_right - physics.phi_left)
    return midpoint + amplitude * np.tanh((physics.x - 0.5 * length) / width)


def _initial_state(physics: _Physics) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Amorcage : potentiel de depletion et quasi-niveaux fendus dans la junction.

    Sous polarisation V, w = n_i exp(phi_n/V_T) passe de n_i (cote p) a
    n_i exp(V/V_T) (cote n), et z = n_i exp(phi_p/V_T) varie en sens inverse.
    La transition est localisee dans la zone de depletion ; un etalement sur
    tout le dispositif donnerait une charge ionisee non compensee enorme.
    """
    phi = _initial_potential(physics)
    fraction = 0.5 * (
        1.0 + np.tanh((physics.x - 0.5 * physics.device.thickness_m)
                      / (0.15 * physics.device.thickness_m))
    )
    ratio = float(physics.voltage) / physics.vt
    w = physics.ni * np.exp(ratio * fraction)
    z = physics.ni * np.exp(-ratio * fraction)
    w[-1] = physics.w_right
    z[0] = physics.z_left
    w[0] = w[1]
    z[-1] = z[-2]
    return phi, w, z


def solve_dark_equilibrium(device: DeviceConfig, temperature_k: float) -> CarrierSolution:
    """Equilibre thermodynamique exact a l'obscurite : J_n = J_p = 0.

    A V=0 et G=0, les quasi-niveaux de Fermi sont plats et egaux a zero :
    w = z = n_i sur tout le dispositif, d'ou np = n_i^2 et R = 0. Le flux de
    Scharfetter--Gummel calcule alors avec ces variables de Slotboom constantes
    s'annule exactement (identite numerique, pas une tolerance).
    """
    physics = _build_physics(device, temperature_k, 0.0, 0.0)
    w = np.full(physics.x.size, physics.ni)
    z = np.full(physics.x.size, physics.ni)
    phi, iterations = _newton_poisson(physics, _initial_potential(physics), w, z, 1.0e-12)
    phi[0] = physics.phi_left
    phi[-1] = physics.phi_right
    n = w * np.exp(np.clip(phi / physics.vt, -600.0, 600.0))
    p = z * np.exp(-np.clip(phi / physics.vt, -600.0, 600.0))
    a_coeff, c_coeff, _, _ = physics.scharfetter_coefficients(phi)
    j_n_edges = a_coeff * np.diff(w)
    j_p_edges = c_coeff * (z[:-1] - z[1:])
    j_n = np.concatenate((j_n_edges[:1], j_n_edges))
    j_p = np.concatenate((j_p_edges[:1], j_p_edges))
    return CarrierSolution(
        device=device,
        temperature_k=physics.temperature_k,
        voltage=0.0,
        irradiance=0.0,
        x=physics.x,
        potential=phi,
        electrons=n,
        holes=p,
        quasi_fermi_n=np.zeros_like(phi),
        quasi_fermi_p=np.zeros_like(phi),
        current_n=j_n,
        current_p=j_p,
        generation=np.zeros_like(phi),
        recombination=np.zeros_like(phi),
        current_density=float(np.mean(j_n_edges + j_p_edges)),
        iterations=iterations,
        converged=True,
        thermal_voltage_v=physics.vt,
        intrinsic_density_m3=physics.ni,
        electron_mobility=physics.mu_n,
        hole_mobility=physics.mu_p,
    )


def _continuity_update(physics: _Physics, phi: np.ndarray, w: np.ndarray, z: np.ndarray):
    """Resout les deux equations de continuite lineaires a potentiel figé."""
    size = phi.size
    n, p = physics.carriers(phi, w, z)
    recombination = physics.recombination_rate(n, p)
    rhs = Q_E * physics.h * (physics.generation - recombination)
    a_coeff, c_coeff, _, _ = physics.scharfetter_coefficients(phi)

    lower_w = np.zeros(size)
    diag_w = np.zeros(size)
    upper_w = np.zeros(size)
    lower_w[1:-1] = -a_coeff[:-1]
    diag_w[1:-1] = a_coeff[:-1] + a_coeff[1:]
    upper_w[1:-1] = -a_coeff[1:]
    diag_w[0] = 1.0
    upper_w[0] = -1.0
    diag_w[-1] = 1.0
    rhs_w = rhs.copy()
    rhs_w[0], rhs_w[-1] = 0.0, physics.w_right
    w_new = _tridiagonal_solve(lower_w, diag_w, upper_w, rhs_w)

    lower_z = np.zeros(size)
    diag_z = np.zeros(size)
    upper_z = np.zeros(size)
    lower_z[1:-1] = -c_coeff[:-1]
    diag_z[1:-1] = c_coeff[:-1] + c_coeff[1:]
    upper_z[1:-1] = -c_coeff[1:]
    diag_z[0] = 1.0
    diag_z[-1] = 1.0
    lower_z[-1] = -1.0
    rhs_z = rhs.copy()
    rhs_z[0], rhs_z[-1] = physics.z_left, 0.0
    z_new = _tridiagonal_solve(lower_z, diag_z, upper_z, rhs_z)

    w_new = np.clip(w_new, physics.ni * 1.0e-12, physics.ni * 1.0e30)
    z_new = np.clip(z_new, physics.ni * 1.0e-12, physics.ni * 1.0e30)
    w_new[-1] = physics.w_right
    z_new[0] = physics.z_left
    w_new[0] = w_new[1]
    z_new[-1] = z_new[-2]
    return w_new, z_new


def solve_carrier_transport(
    device: DeviceConfig,
    temperature_k: float,
    voltage: float = 0.0,
    irradiance: float = P_INPUT,
    initial: CarrierSolution | None = None,
    *,
    tolerance: float = GUMMEL_TOL,
) -> CarrierSolution:
    """Resout Poisson -- derive--diffusion couples par iterations de Gummel.

    A chaque iteration : Newton sur Poisson (w,z figes), puis resolution des
    equations de continuite lineaires en w et z (flux exponentiel de
    Scharfetter--Gummel). Les contacts sont selectifs : p(0)=N_A et n(L)=N_D
    (contacts ohmiques majoritaires), flux de porteurs minoritaires nul aux
    deux extremites (passivation de surface). La polarisation V fixe
    w(L)/w(0) = z(0)/z(L) = exp(qV/k_B T). Le courant renvoye est J=-(J_n+J_p),
    de sorte que J(0)=J_sc>0 sous eclairage et J(V_oc)=0.
    """
    physics = _build_physics(device, temperature_k, voltage, irradiance)
    size = physics.x.size

    if initial is not None:
        phi = initial.potential.copy()
        w = initial.electrons * np.exp(-np.clip(phi / physics.vt, -600.0, 600.0))
        z = initial.holes * np.exp(np.clip(phi / physics.vt, -600.0, 600.0))
        phi[0] = physics.phi_left
        phi[-1] = physics.phi_right
        w[-1] = physics.w_right
        z[0] = physics.z_left
        w[0] = w[1]
        z[-1] = z[-2]
        w = np.clip(w, physics.ni * 1.0e-12, physics.ni * 1.0e30)
        z = np.clip(z, physics.ni * 1.0e-12, physics.ni * 1.0e30)
    else:
        phi, w, z = _initial_state(physics)

    converged = False
    iterations = 0
    for iteration in range(MAX_GUMMEL):
        iterations = iteration + 1
        phi_previous = phi.copy()
        phi, _ = _newton_poisson(physics, phi, w, z, 1.0e-11)
        phi[0] = physics.phi_left
        phi[-1] = physics.phi_right
        w_new, z_new = _continuity_update(physics, phi, w, z)
        change = max(
            float(np.max(np.abs(np.log(w_new[1:-1] / w[1:-1])))),
            float(np.max(np.abs(np.log(z_new[1:-1] / z[1:-1])))),
            float(np.max(np.abs(phi - phi_previous))),
        )
        w, z = w_new, z_new
        if change < tolerance:
            converged = True
            break

    phi[0] = physics.phi_left
    phi[-1] = physics.phi_right
    w, z = _continuity_update(physics, phi, w, z)
    u = np.clip(phi / physics.vt, -600.0, 600.0)
    n, p = w * np.exp(u), z * np.exp(-u)
    recombination = physics.recombination_rate(n, p)
    a_coeff, c_coeff, _, _ = physics.scharfetter_coefficients(phi)
    j_n_edges = a_coeff * np.diff(w)
    j_p_edges = c_coeff * (z[:-1] - z[1:])
    j_n = np.concatenate((j_n_edges[:1], j_n_edges))
    j_p = np.concatenate((j_p_edges[:1], j_p_edges))
    quasi_n = phi - physics.vt * np.log(np.maximum(n, 1.0) / physics.ni)
    quasi_p = phi + physics.vt * np.log(np.maximum(p, 1.0) / physics.ni)
    return CarrierSolution(
        device=device,
        temperature_k=physics.temperature_k,
        voltage=float(voltage),
        irradiance=physics.irradiance,
        x=physics.x,
        potential=phi,
        electrons=n,
        holes=p,
        quasi_fermi_n=quasi_n,
        quasi_fermi_p=quasi_p,
        current_n=j_n,
        current_p=j_p,
        generation=physics.generation,
        recombination=recombination,
        current_density=float(np.mean(j_n_edges + j_p_edges)),
        iterations=iterations,
        converged=converged,
        thermal_voltage_v=physics.vt,
        intrinsic_density_m3=physics.ni,
        electron_mobility=physics.mu_n,
        hole_mobility=physics.mu_p,
    )


@dataclass(frozen=True)
class JVResult:
    """Caracteristique courant-tension complete d'une cellule photovoltaique."""

    device: DeviceConfig
    temperature_k: float
    irradiance: float
    site_name: str
    voltage: np.ndarray
    current_density: np.ndarray
    power_density: np.ndarray
    jsc: float
    voc: float
    vmp: float
    jmp: float
    pmax: float
    ff: float
    efficiency: float
    solutions: dict[float, CarrierSolution] = field(default_factory=dict, repr=False)

    @property
    def junction_temperature_c(self) -> float:
        return float(self.temperature_k) - KELVIN_OFFSET

    def __repr__(self) -> str:
        return (
            f"JVResult({self.site_name}, {self.device.name}, T_j={self.junction_temperature_c:.2f} C, "
            f"Jsc={self.jsc:.2f} A/m2, Voc={self.voc:.4f} V, FF={self.ff:.4f}, "
            f"eta={100.0 * self.efficiency:.2f} %)"
        )


def _bracketed_root(func: Callable[[float], float], lower: float, upper: float, tol: float = 1.0e-10) -> float:
    """Bissection pure Python pour encadrer la racine de func."""
    a, b = float(lower), float(upper)
    fa, fb = func(a), func(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise ValueError("La racine n'est pas encadree.")
    for _ in range(200):
        middle = 0.5 * (a + b)
        fm = func(middle)
        if fm == 0.0 or 0.5 * (b - a) < tol:
            return middle
        if fm * fa > 0.0:
            a, fa = middle, fm
        else:
            b, fb = middle, fm
    return 0.5 * (a + b)


def solve_jv_curve(
    device: DeviceConfig,
    temperature_k: float,
    irradiance: float = P_INPUT,
    *,
    site_name: str = "AM1.5G",
    n_points: int = 61,
    v_max: float | None = None,
    equilibrium: CarrierSolution | None = None,
) -> JVResult:
    """Balaye V, extrait Jsc, Voc, FF, eta et le point de puissance maximale."""
    if v_max is None:
        j0_rad = radiative_saturation_current(device.bandgap_ev, temperature_k)
        flux = photon_flux_above_gap(device.bandgap_ev) * float(irradiance) / P_INPUT
        jsc_guess = Q_E * flux * device.quantum_efficiency
        voc_guess = thermal_voltage(temperature_k) * np.log1p(jsc_guess / j0_rad)
        v_max = float(min(1.25 * voc_guess, device.bandgap_ev + 0.15))
    v_max = max(float(v_max), 0.1)

    equilibrium = equilibrium or solve_dark_equilibrium(device, temperature_k)
    cache: dict[float, CarrierSolution] = {}

    def solution_at(voltage: float) -> CarrierSolution:
        key = round(float(voltage), 10)
        if key in cache:
            return cache[key]
        nearest = min(cache.values(), key=lambda item: abs(item.voltage - voltage)) if cache else equilibrium
        state = solve_carrier_transport(
            device, temperature_k, voltage, irradiance, initial=nearest
        )
        cache[key] = state
        return state

    voltages: list[float] = [0.0]
    currents: list[float] = [-solution_at(0.0).current_density]
    step = min(v_max / max(int(n_points) // 2, 1), 0.04)
    for _ in range(max(2 * int(n_points), 80)):
        if voltages[-1] >= v_max:
            break
        candidate = min(voltages[-1] + step, v_max)
        try:
            value = -solution_at(candidate).current_density
        except (ZeroDivisionError, FloatingPointError, OverflowError):
            step *= 0.5
            if step < 1.0e-5:
                break
            continue
        voltages.append(candidate)
        currents.append(value)
        if value < 0.0:
            break
        step = min(step * 1.05, 0.06)

    voltages_array = np.asarray(voltages, dtype=float)
    currents_array = np.asarray(currents, dtype=float)
    crossing = np.flatnonzero(currents_array[:-1] > 0.0)
    falling = np.flatnonzero(currents_array[1:] < 0.0)
    candidates = np.intersect1d(crossing, falling)
    if candidates.size == 0:
        raise RuntimeError("La courbe J(V) ne presente pas de tension de circuit ouvert.")
    voltages = voltages_array
    currents = currents_array
    index = int(candidates[-1])
    voc = _bracketed_root(
        lambda v: -solution_at(v).current_density,
        float(voltages[index]),
        float(voltages[index + 1]),
    )
    solution_at(voc)

    powers = voltages * currents
    peak = int(np.argmax(powers))
    left = float(voltages[max(peak - 1, 0)])
    right = float(voltages[min(peak + 1, voltages.size - 1)])

    def negative_power(voltage: float) -> float:
        return -voltage * (-solution_at(voltage).current_density)

    golden = (np.sqrt(5.0) - 1.0) / 2.0
    c = right - golden * (right - left)
    d = left + golden * (right - left)
    fc, fd = negative_power(c), negative_power(d)
    for _ in range(30):
        if right - left < 1.0e-9:
            break
        if fc < fd:
            right, d, fd = d, c, fc
            c = right - golden * (right - left)
            fc = negative_power(c)
        else:
            left, c, fc = c, d, fd
            d = left + golden * (right - left)
            fd = negative_power(d)
    vmp = 0.5 * (left + right)
    solution = solution_at(vmp)
    jmp = -solution.current_density
    pmax = max(0.0, vmp * jmp)
    jsc = float(currents[0])
    ff = pmax / (voc * jsc) if voc * jsc > 0.0 else 0.0
    efficiency = pmax / float(irradiance)

    ordered = sorted(cache, key=float)
    return JVResult(
        device=device,
        temperature_k=float(temperature_k),
        irradiance=float(irradiance),
        site_name=site_name,
        voltage=voltages,
        current_density=currents,
        power_density=powers,
        jsc=jsc,
        voc=float(voc),
        vmp=float(vmp),
        jmp=float(jmp),
        pmax=pmax,
        ff=float(ff),
        efficiency=float(efficiency),
        solutions={key: cache[key] for key in ordered},
    )


def junction_temperature(site: SahelSite, module: PVModule) -> float:
    """Temperature de jonction T_j (K) issue du bilan thermodynamique Nuit 5."""
    return float(energy_balance(site, module).junction_temperature_k)


def site_electrical_result(site: SahelSite, module: PVModule, *, n_points: int = 61) -> JVResult:
    """Courbe J(V) d'un site sahelien couplee a T_j et a l'eclairage du site."""
    device = device_from_module(module)
    temperature = junction_temperature(site, module)
    return solve_jv_curve(
        device,
        temperature,
        irradiance=float(site.irradiance),
        site_name=site.name,
        n_points=n_points,
    )


def temperature_sweep(
    device: DeviceConfig,
    irradiance: float = P_INPUT,
    temperatures_c: tuple[float, ...] | np.ndarray = (25.0, 30.0, 35.0, 40.0, 45.0, 50.0),
    *,
    n_points: int = 41,
) -> list[JVResult]:
    """Courbes J(V) pour une plage de temperatures de jonction (en degres C)."""
    return [
        solve_jv_curve(
            device,
            float(temperature_c) + KELVIN_OFFSET,
            irradiance,
            site_name=f"T_j={float(temperature_c):.0f} C",
            n_points=n_points,
        )
        for temperature_c in temperatures_c
    ]


def symbolic_einstein_relation() -> dict[str, sp.Expr]:
    """Verification SymPy de D = (k_B T/q) mu a partir du flux net nul."""
    mu, kb, temperature, charge, d_coeff, phi, step = sp.symbols(
        "mu k_B T_j q D phi h", positive=True
    )
    ni, density = sp.symbols("n_i n", positive=True)
    vt = kb * temperature / charge
    slope = sp.symbols("dphi", real=True)
    dn_dx = density * slope / vt
    net_flux = sp.simplify(-mu * density * slope + d_coeff * dn_dx)
    solution = sp.solve(sp.Eq(net_flux, 0), d_coeff)[0]
    return {
        "thermal_voltage": vt,
        "net_flux": net_flux,
        "diffusivity": solution,
        "einstein_check": sp.simplify(solution - mu * vt),
        "phi": phi,
        "h": step,
        "n_i": ni,
    }


def symbolic_shockley_equation() -> dict[str, sp.Expr]:
    """Verification SymPy de J(V), V_oc et de la condition du point de puissance."""
    jsc, j0, voltage, thermal, reduced = sp.symbols(
        "J_sc J_0 V kBT_q v", positive=True
    )
    current = jsc - j0 * (sp.exp(voltage / thermal) - 1)
    voc = thermal * sp.log(1 + jsc / j0)
    power = voltage * current
    derivative = sp.simplify(sp.diff(power, voltage))
    expected = jsc - j0 * (sp.exp(voltage / thermal) * (1 + voltage / thermal) - 1)
    mpp_condition = sp.simplify((1 + reduced) * sp.exp(reduced) - (1 + jsc / j0))
    return {
        "current": current,
        "open_circuit": voc,
        "current_at_voc": sp.simplify(current.subs(voltage, voc)),
        "power_derivative": derivative,
        "expected_derivative": expected,
        "derivative_check": sp.simplify(derivative - expected),
        "mpp_condition": mpp_condition,
    }


def symbolic_scharfetter_gummel() -> dict[str, sp.Expr]:
    """Verification SymPy du developpement de B(z) et du schema SG.

    Pour un faible saut de potentiel, le flux exponentiel reproduit le schema
    centre derive--diffusion, et le courant s'annule exactement des que les
    variables de Slotboom sont constantes (equilibre).
    """
    z = sp.symbols("z", real=True)
    b_exact = z / (sp.exp(z) - 1)
    expansion = sp.series(b_exact, z, 0, 4).removeO()
    b_neg_expansion = sp.simplify(expansion.subs(z, -z))
    flux = b_exact * sp.Symbol("w_right") - b_neg_expansion * sp.Symbol("w_left")
    return {
        "bernoulli": b_exact,
        "expansion": sp.expand(expansion),
        "expansion_check": sp.simplify(expansion - (1 - z / 2 + z**2 / 12)),
        "b_negative_expansion": sp.expand(b_neg_expansion),
        "flux": flux,
    }
