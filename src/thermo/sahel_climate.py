"""UAM-TOPOMAT -- Nuit 5 : bilan thermodynamique sahelien.

Modele stationnaire et transitoire d'une cellule photovoltaique et d'un module
hybride PV/T sous les conditions reelles du Senegal (Dakar, Saint-Louis, Matam).

Bilan de puissance par unite de surface :

    P_abs = alpha_lambda * G
    P_elec = eta(E_g, T_j) * G
    P_conv = h_conv(v_vent) * (T_j - T_amb)
    P_rad = eps * sigma_SB * (T_j^4 - T_ciel^4)
    P_th = U_ext * (T_j - T_entree)   (module hybride uniquement)

    P_abs = P_elec + P_conv + P_rad + P_th

La resolution de T_j utilise un Newton-Raphson avec der analytique et
sauvegarde par bissection (vectorise NumPy), complete par un solveur de Brent
pour les cas scalaires. Le rendement suit la loi exponentielle a derivee
thermique exacte eta(T) = eta_ref * exp(beta_T * (T - T_ref)), avec
beta_T = (1/eta) * d(eta)/dT. Le spectre AM1.5G de la Nuit 4 fournit
l'absorption ponderee par alpha(lambda) et le rendement Shockley-Queisser
de reference. Python pur : numpy, sympy, matplotlib. Aucune dependance SciPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import sympy as sp

from src.optics.solar_optics import HC_EV_NM, T_REF, solar_grid, sq_efficiency

SIGMA_SB = 5.670374419e-8
KELVIN_OFFSET = 273.15
SKY_DEPRESSION_K = 8.0
H_NATURAL = 5.7
H_WIND = 3.8
DAYLIGHT_HOURS = 12.0
SOLAR_NOON_HOUR = 12.0
PEAK_HEAT_HOUR = 15.0
SEASON_PEAK_DAY = 172.0


@dataclass(frozen=True)
class SahelSite:
    """Conditions climatiques de reference d'un site sahelien."""

    name: str
    ambient_c: float
    irradiance: float
    wind_speed: float
    seasonal_amplitude_k: float

    @property
    def ambient_k(self) -> float:
        return float(self.ambient_c) + KELVIN_OFFSET


DAKAR = SahelSite("Dakar", 31.0, 900.0, 3.5, 6.0)
SAINT_LOUIS = SahelSite("Saint-Louis", 37.0, 980.0, 4.0, 7.5)
MATAM = SahelSite("Matam", 47.0, 1050.0, 1.8, 9.0)
SAHEL_SITES: tuple[SahelSite, ...] = (DAKAR, SAINT_LOUIS, MATAM)


@dataclass(frozen=True)
class PVModule:
    """Materialiau photovoltaique caracterise par son gap et sa derive thermique."""

    name: str
    bandgap_ev: float
    beta_t: float
    eta_ref: float
    absorptance_above: float = 0.95
    absorptance_below: float = 0.60
    emissivity: float = 0.90
    heat_extraction: float = 0.0
    inlet_temperature_k: float = T_REF

    @property
    def is_hybrid(self) -> bool:
        return float(self.heat_extraction) > 0.0

    def __repr__(self) -> str:
        return (
            f"PVModule({self.name}, Eg={self.bandgap_ev:.2f} eV, "
            f"eta_ref={100.0 * self.eta_ref:.2f} %, "
            f"beta_T={100.0 * self.beta_t:.3f} %/K, hybrid={self.is_hybrid})"
        )


def _reference_efficiency(bandgap_ev: float, module_factor: float) -> float:
    return float(module_factor) * float(sq_efficiency(float(bandgap_ev), T_REF))


def optimized_module() -> PVModule:
    """Pervoskite / topologique a gap optimise (Eg ~ 1.45 eV, beta ~ -0.22 %/K)."""
    return PVModule(
        name="Perovskite/topologique (Eg optimise)",
        bandgap_ev=1.45,
        beta_t=-0.0022,
        eta_ref=_reference_efficiency(1.45, 0.78),
    )


def silicon_module() -> PVModule:
    """Silicium cristallin standard (Eg = 1.12 eV, beta ~ -0.45 %/K)."""
    return PVModule(
        name="Silicium cristallin",
        bandgap_ev=1.12,
        beta_t=-0.0045,
        eta_ref=_reference_efficiency(1.12, 0.68),
    )


def hybrid_module() -> PVModule:
    """Module hybride PV/T : meme cellule optimisee avec extraction de chaleur."""
    optimized = optimized_module()
    return PVModule(
        name="Module hybride PV/T (perovskite)",
        bandgap_ev=optimized.bandgap_ev,
        beta_t=optimized.beta_t,
        eta_ref=optimized.eta_ref,
        heat_extraction=8.0,
    )


def sky_temperature(t_amb_k, depression_k: float = SKY_DEPRESSION_K):
    """Temperature effective du ciel (K), degression de 8 K par defaut."""
    return np.asarray(t_amb_k, dtype=float) - float(depression_k)


def convection_coefficient(wind_speed):
    """Coefficient de convection h = 5.7 + 3.8 v (W/m^2/K), v en m/s."""
    wind = np.asarray(wind_speed, dtype=float)
    if np.any(wind < 0.0):
        raise ValueError("La vitesse du vent doit etre positive.")
    return H_NATURAL + H_WIND * wind


def solar_absorptance(
    bandgap_ev: float,
    absorptance_above: float = 0.95,
    absorptance_below: float = 0.60,
) -> float:
    """Absorptance solaire ponderee par alpha(lambda) sur le spectre AM1.5G.

    alpha(lambda) vaut absorptance_above pour les photons au-dessus du gap
    (lambda < lambda_g = hc/E_g) et absorptance_below pour les photons sous
    le gap, qui sont absorbes par l'encapsulation et le contact arriere.
    """
    wavelengths, irradiance = solar_grid()
    cutoff_nm = HC_EV_NM / float(bandgap_ev)
    above = np.where(wavelengths <= cutoff_nm, irradiance, 0.0)
    below = irradiance - above
    total = float(np.trapezoid(irradiance, wavelengths))
    return float(
        (
            float(absorptance_above) * float(np.trapezoid(above, wavelengths))
            + float(absorptance_below) * float(np.trapezoid(below, wavelengths))
        )
        / total
    )


def module_efficiency(module: PVModule, junction_temperature_k):
    """eta(E_g, T_j) = eta_ref * exp(beta_T (T_j - T_ref))."""
    temperature = np.asarray(junction_temperature_k, dtype=float)
    return module.eta_ref * np.exp(module.beta_t * (temperature - T_REF))


def thermal_drift_coefficient(module: PVModule, junction_temperature_k=None):
    """Derivee relative exacte beta_T = (1/eta) d(eta)/dT (1/K).

    Pour la loi exponentielle, beta_T est constant et exactement egal au
    parametre du materiau, quelle que soit la temperature de jonction.
    """
    if junction_temperature_k is None:
        return float(module.beta_t)
    temperature = np.asarray(junction_temperature_k, dtype=float)
    return np.full(temperature.shape, float(module.beta_t), dtype=float)


def balance_terms(
    junction_temperature_k,
    irradiance,
    ambient_temperature_k,
    module: PVModule,
    h_conv,
    t_sky,
    absorptance: float | None = None,
):
    """Renvoie (P_abs, P_elec, P_conv, P_rad, P_th) en W/m^2."""
    temperature = np.asarray(junction_temperature_k, dtype=float)
    irradiance = np.asarray(irradiance, dtype=float)
    ambient = np.asarray(ambient_temperature_k, dtype=float)
    h_conv = np.asarray(h_conv, dtype=float)
    t_sky = np.asarray(t_sky, dtype=float)
    if absorptance is None:
        absorptance = solar_absorptance(
            module.bandgap_ev, module.absorptance_above, module.absorptance_below
        )
    absorbed = absorptance * irradiance
    electrical = module_efficiency(module, temperature) * irradiance
    convective = h_conv * (temperature - ambient)
    radiative = module.emissivity * SIGMA_SB * (temperature**4 - t_sky**4)
    extracted = module.heat_extraction * (temperature - module.inlet_temperature_k)
    return absorbed, electrical, convective, radiative, extracted


def balance_residual(
    junction_temperature_k,
    irradiance,
    ambient_temperature_k,
    module: PVModule,
    h_conv,
    t_sky,
    absorptance: float | None = None,
):
    """Residu f(T_j) = P_abs - P_elec - P_conv - P_rad - P_th (W/m^2)."""
    absorbed, electrical, convective, radiative, extracted = balance_terms(
        junction_temperature_k,
        irradiance,
        ambient_temperature_k,
        module,
        h_conv,
        t_sky,
        absorptance,
    )
    return absorbed - (electrical + convective + radiative + extracted)


def balance_derivative(
    junction_temperature_k,
    irradiance,
    module: PVModule,
    h_conv,
    absorptance: float | None = None,
):
    """Derivee analytique df/dT_j du residu (W/m^2/K)."""
    temperature = np.asarray(junction_temperature_k, dtype=float)
    eta = module_efficiency(module, temperature)
    return (
        -module.beta_t * eta * np.asarray(irradiance, dtype=float)
        - np.asarray(h_conv, dtype=float)
        - 4.0 * module.emissivity * SIGMA_SB * temperature**3
        - module.heat_extraction
    )


def solve_junction_temperature(
    irradiance,
    ambient_temperature_k,
    wind_speed,
    module: PVModule,
    *,
    absorptance: float | None = None,
    tol: float = 1e-12,
    max_iter: int = 120,
):
    """Newton-Raphson vectorise avec der analytique et sauvegarde par bissection.

    Les entrees (G, T_amb, v_vent) peuvent etre des scalaires ou des tableaux
    broadcastables. Le residu est strictement decroissant en T_j, si bien que
    le crochet [T_amb - 20 K, T_amb + 300 K] contient l'unique racine.
    """
    irradiance = np.asarray(irradiance, dtype=float)
    ambient = np.asarray(ambient_temperature_k, dtype=float)
    wind = np.asarray(wind_speed, dtype=float)
    shape = np.broadcast_shapes(irradiance.shape, ambient.shape, wind.shape)
    irradiance, ambient, wind = (
        np.broadcast_to(array, shape) for array in (irradiance, ambient, wind)
    )
    if np.any(irradiance < 0.0):
        raise ValueError("L'irradiance doit etre positive.")
    if absorptance is None:
        absorptance = solar_absorptance(
            module.bandgap_ev, module.absorptance_above, module.absorptance_below
        )

    h_conv = convection_coefficient(wind)
    t_sky = sky_temperature(ambient)
    lo = np.array(ambient - 20.0, dtype=float, copy=True)
    hi = np.array(ambient + 300.0, dtype=float, copy=True)
    temperature = np.array(ambient + 30.0, dtype=float, copy=True)

    for _ in range(int(max_iter)):
        residual = balance_residual(
            temperature, irradiance, ambient, module, h_conv, t_sky, absorptance
        )
        derivative = balance_derivative(
            temperature, irradiance, module, h_conv, absorptance
        )
        step = residual / derivative
        trial = temperature - step
        outside = ~np.isfinite(trial) | (trial <= lo) | (trial >= hi)
        trial = np.where(outside, 0.5 * (lo + hi), trial)
        trial_residual = balance_residual(
            trial, irradiance, ambient, module, h_conv, t_sky, absorptance
        )
        lo = np.where(trial_residual > 0.0, trial, lo)
        hi = np.where(trial_residual > 0.0, hi, trial)
        temperature = trial
        if np.all(np.abs(trial_residual) < float(tol)):
            break
    return temperature


def _brent_scalar(func, lower: float, upper: float, tol: float, max_iter: int) -> float:
    """Brent (racine bracketee) en Python/NumPy pur, sans SciPy."""
    a, b = float(lower), float(upper)
    fa, fb = func(a), func(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise ValueError("Le residu ne change pas de signe sur le crochet.")
    c, fc = a, fa
    d = e = b - a
    for _ in range(int(max_iter)):
        if fb * fc > 0.0:
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol_1 = 2.0 * np.finfo(float).eps * abs(b) + 0.5 * float(tol)
        midpoint = 0.5 * (c - b)
        if abs(midpoint) <= tol_1 or fb == 0.0:
            return b
        if abs(e) >= tol_1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2.0 * midpoint * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * midpoint * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            else:
                p = -p
            min_step_1 = 3.0 * midpoint * q - abs(tol_1 * q)
            min_step_2 = abs(e * q)
            if 2.0 * p < min(min_step_1, min_step_2):
                e, d = d, p / q
            else:
                d = e = midpoint
        else:
            d = e = midpoint
        a, fa = b, fb
        increment = d if abs(d) > tol_1 else (tol_1 if midpoint > 0.0 else -tol_1)
        b += increment
        fb = func(b)
    return b


def brent_junction_temperature(
    irradiance: float,
    ambient_temperature_k: float,
    wind_speed: float,
    module: PVModule,
    *,
    absorptance: float | None = None,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> float:
    """Resolveur de Brent scalaire de verificiation, independant de Newton."""
    g = float(irradiance)
    ambient = float(ambient_temperature_k)
    h_conv = float(convection_coefficient(float(wind_speed)))
    t_sky = float(sky_temperature(ambient))
    if absorptance is None:
        absorptance = solar_absorptance(
            module.bandgap_ev, module.absorptance_above, module.absorptance_below
        )

    def residual(temperature: float) -> float:
        return float(
            balance_residual(
                temperature, g, ambient, module, h_conv, t_sky, absorptance
            )
        )

    return _brent_scalar(residual, ambient - 20.0, ambient + 300.0, tol, max_iter)


@dataclass(frozen=True)
class EnergyBalance:
    """Bilan energetique detaille a l'equilibre stationnaire."""

    site: SahelSite
    module: PVModule
    junction_temperature_k: float
    sky_temperature_k: float
    h_conv: float
    absorptance: float
    absorbed_power: float
    electrical_power: float
    convection_power: float
    radiation_power: float
    extraction_power: float
    efficiency: float
    residual: float

    @property
    def junction_temperature_c(self) -> float:
        return float(self.junction_temperature_k) - KELVIN_OFFSET

    @property
    def operating_drift(self) -> float:
        return float(
            thermal_drift_coefficient(self.module, self.junction_temperature_k)
        )

    def __repr__(self) -> str:
        return (
            f"EnergyBalance({self.site.name}, {self.module.name}, "
            f"T_j={self.junction_temperature_c:.2f} C, eta={100 * self.efficiency:.2f} %, "
            f"P_elec={self.electrical_power:.2f} W/m^2)"
        )


def energy_balance(site: SahelSite, module: PVModule) -> EnergyBalance:
    """Calcule le point de fonctionnement stationnaire d'un site et d'un module."""
    absorptance = solar_absorptance(
        module.bandgap_ev, module.absorptance_above, module.absorptance_below
    )
    temperature = float(
        solve_junction_temperature(
            site.irradiance,
            site.ambient_k,
            site.wind_speed,
            module,
            absorptance=absorptance,
        )
    )
    h_conv = float(convection_coefficient(site.wind_speed))
    t_sky = float(sky_temperature(site.ambient_k))
    absorbed, electrical, convective, radiative, extracted = balance_terms(
        temperature,
        site.irradiance,
        site.ambient_k,
        module,
        h_conv,
        t_sky,
        absorptance,
    )
    return EnergyBalance(
        site=site,
        module=module,
        junction_temperature_k=temperature,
        sky_temperature_k=t_sky,
        h_conv=h_conv,
        absorptance=absorptance,
        absorbed_power=float(absorbed),
        electrical_power=float(electrical),
        convection_power=float(convective),
        radiation_power=float(radiative),
        extraction_power=float(extracted),
        efficiency=float(module_efficiency(module, temperature)),
        residual=float(balance_residual(
            temperature,
            site.irradiance,
            site.ambient_k,
            module,
            h_conv,
            t_sky,
            absorptance,
        )),
    )


@dataclass(frozen=True)
class AnnualYield:
    """Production electrique annuelle integree sur une annee meteorologique type."""

    site: SahelSite
    module: PVModule
    kwh_per_m2: float
    mean_junction_c: float
    mean_efficiency: float
    incident_kwh_per_m2: float

    @property
    def performance_ratio(self) -> float:
        return float(self.kwh_per_m2) / float(self.incident_kwh_per_m2)


def annual_yield(site: SahelSite, module: PVModule, hours: int = 8760) -> AnnualYield:
    """Integre P_elec sur une annee type (cycle solaire et saisonnier)."""
    hours_array = np.arange(int(hours), dtype=float)
    day_index = hours_array // 24.0
    hour_index = hours_array % 24.0
    seasonal = np.cos(2.0 * np.pi * (day_index - SEASON_PEAK_DAY) / 365.0)
    daylight = np.clip(
        np.sin(np.pi * (hour_index + 1.0 - 6.0) / DAYLIGHT_HOURS), 0.0, None
    )
    irradiance = np.maximum(
        site.irradiance * daylight * (1.0 + 0.08 * seasonal), 0.0
    )
    diurnal = 3.0 * np.cos(2.0 * np.pi * (hour_index - PEAK_HEAT_HOUR) / 24.0)
    ambient_c = (
        float(site.ambient_c)
        - float(site.seasonal_amplitude_k)
        + float(site.seasonal_amplitude_k) * seasonal
        + diurnal
    )
    ambient_k = ambient_c + KELVIN_OFFSET
    absorptance = solar_absorptance(
        module.bandgap_ev, module.absorptance_above, module.absorptance_below
    )
    temperature = solve_junction_temperature(
        irradiance,
        ambient_k,
        site.wind_speed,
        module,
        absorptance=absorptance,
    )
    eta = module_efficiency(module, temperature)
    energy_kwh = float(np.sum(eta * irradiance)) / 1000.0
    sunlit = irradiance > 0.0
    return AnnualYield(
        site=site,
        module=module,
        kwh_per_m2=energy_kwh,
        mean_junction_c=float(np.mean(temperature[sunlit]) - KELVIN_OFFSET),
        mean_efficiency=float(np.mean(eta[sunlit])),
        incident_kwh_per_m2=float(np.sum(irradiance)) / 1000.0,
    )


@dataclass(frozen=True)
class AnnualGain:
    """Comparaison annuelle materialiau optimise / silicium sur un site."""

    site: SahelSite
    optimized: AnnualYield
    silicon: AnnualYield

    @property
    def delta_kwh_per_m2(self) -> float:
        return float(self.optimized.kwh_per_m2 - self.silicon.kwh_per_m2)

    @property
    def relative_gain_percent(self) -> float:
        return 100.0 * float(self.delta_kwh_per_m2) / float(self.silicon.kwh_per_m2)


def annual_energy_gain(
    site: SahelSite,
    optimized: PVModule | None = None,
    silicon: PVModule | None = None,
) -> AnnualGain:
    """Gain energetique annuel du materiau optimise face au silicium."""
    return AnnualGain(
        site=site,
        optimized=annual_yield(site, optimized or optimized_module()),
        silicon=annual_yield(site, silicon or silicon_module()),
    )


def symbolic_balance_derivative() -> dict[str, sp.Expr]:
    """Verification SymPy de df/dT_j et de beta_T = d(ln eta)/dT_j."""
    tj, ta, g, eps, sig, h, alpha = sp.symbols(
        "T_j T_amb G eps sigma h alpha", real=True
    )
    eta0, beta, tref, tsky, u, tin = sp.symbols(
        "eta_ref beta T_ref T_sky U_ext T_in", real=True
    )
    eta = eta0 * sp.exp(beta * (tj - tref))
    residual = (
        alpha * g
        - eta * g
        - h * (tj - ta)
        - eps * sig * (tj**4 - tsky**4)
        - u * (tj - tin)
    )
    analytic = -beta * eta * g - h - 4 * eps * sig * tj**3 - u
    check = sp.simplify(sp.diff(residual, tj) - analytic)
    beta_exact = sp.simplify(sp.diff(sp.log(eta), tj))
    return {
        "eta": eta,
        "residual": residual,
        "derivative": analytic,
        "derivative_check": check,
        "beta_exact": beta_exact,
    }


def sweep_ambient_temperature(
    site: SahelSite,
    module: PVModule,
    temperatures_c: Sequence[float] | np.ndarray,
):
    """T_j et eta(T_j) pour une plage de temperatures ambiantes (a G et vent fixes)."""
    ambient_c = np.asarray(temperatures_c, dtype=float)
    temperature = np.asarray(
        solve_junction_temperature(
            site.irradiance, ambient_c + KELVIN_OFFSET, site.wind_speed, module
        ),
        dtype=float,
    )
    return temperature - KELVIN_OFFSET, np.asarray(module_efficiency(module, temperature))
