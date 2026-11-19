"""UAM-TOPOMAT -- Validation de la Nuit 5.

Tests du bilan thermodynamique sahelien : conservation des flux, monotonicite
physique, derive thermique exacte, resolution Newton/Brent, superiorite du
materiau a gap optimise a Matam, et generation de la figure de publication.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.optics.solar_optics import T_REF
from src.thermo.sahel_climate import (
    DAKAR,
    KELVIN_OFFSET,
    MATAM,
    SAHEL_SITES,
    SAINT_LOUIS,
    AnnualGain,
    EnergyBalance,
    annual_energy_gain,
    annual_yield,
    balance_derivative,
    balance_residual,
    balance_terms,
    brent_junction_temperature,
    convection_coefficient,
    energy_balance,
    hybrid_module,
    module_efficiency,
    optimized_module,
    silicon_module,
    sky_temperature,
    solar_absorptance,
    solve_junction_temperature,
    sweep_ambient_temperature,
    symbolic_balance_derivative,
    thermal_drift_coefficient,
)
MODULES = (optimized_module(), silicon_module())
SITE_MODULE_PAIRS = [(site, module) for site in SAHEL_SITES for module in MODULES]


def _png_dpi(path: Path) -> float:
    """Lit la resolution (dpi) stockee dans le chunk pHYs d'un PNG."""
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Le fichier n'est pas un PNG valide.")
    position = 8
    while position + 8 <= len(data):
        length = struct.unpack(">I", data[position : position + 4])[0]
        chunk_type = data[position + 4 : position + 8]
        if chunk_type == b"pHYs":
            pixels_per_metre = struct.unpack(
                ">I", data[position + 8 : position + 12]
            )[0]
            unit = data[position + 16]
            if unit == 1:
                return pixels_per_metre * 0.0254
        position += 12 + length
    raise ValueError("Chunk pHYs introuvable.")


# ---------------------------------------------------------------------------
# 1. Parametres climatiques et optiques
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "site,ambient,irradiance,wind",
    [
        (DAKAR, 31.0, 900.0, 3.5),
        (SAINT_LOUIS, 37.0, 980.0, 4.0),
        (MATAM, 47.0, 1050.0, 1.8),
    ],
)
def test_site_climatic_specifications(site, ambient, irradiance, wind) -> None:
    assert site.ambient_c == pytest.approx(ambient)
    assert site.irradiance == pytest.approx(irradiance)
    assert site.wind_speed == pytest.approx(wind)
    assert site.ambient_k == pytest.approx(ambient + KELVIN_OFFSET)


def test_convection_coefficient_forced_and_natural() -> None:
    assert convection_coefficient(0.0) == pytest.approx(5.7)
    assert convection_coefficient(3.5) == pytest.approx(19.0)
    wind = np.linspace(0.0, 8.0, 17)
    assert np.all(np.diff(convection_coefficient(wind)) > 0.0)
    with pytest.raises(ValueError):
        convection_coefficient(-1.0)


def test_sky_temperature_below_ambient() -> None:
    ambient = np.array([298.15, 310.15, 320.15])
    sky = sky_temperature(ambient)
    assert np.all(sky < ambient)
    assert np.allclose(ambient - sky, 8.0)


def test_solar_absorptance_weighted_by_alpha_lambda() -> None:
    alpha_optimized = solar_absorptance(optimized_module().bandgap_ev)
    alpha_silicon = solar_absorptance(silicon_module().bandgap_ev)
    assert 0.0 < alpha_optimized < alpha_silicon < 1.0
    gaps = np.linspace(0.8, 2.0, 13)
    absorptances = np.array([solar_absorptance(float(eg)) for eg in gaps])
    assert np.all(np.diff(absorptances) < 0.0)


def test_module_gaps_and_reference_efficiencies() -> None:
    optimized, silicon = MODULES
    assert optimized.bandgap_ev == pytest.approx(1.45)
    assert silicon.bandgap_ev == pytest.approx(1.12)
    assert optimized.beta_t == pytest.approx(-0.0022)
    assert silicon.beta_t == pytest.approx(-0.0045)
    assert optimized.eta_ref > silicon.eta_ref
    assert 0.15 < silicon.eta_ref < optimized.eta_ref < 0.30
    assert not optimized.is_hybrid and hybrid_module().is_hybrid


# ---------------------------------------------------------------------------
# 2. Derive thermique exacte et verification symbolique
# ---------------------------------------------------------------------------


def test_efficiency_is_strictly_decreasing_with_temperature() -> None:
    temperatures = np.linspace(T_REF, 360.0, 61)
    for module in MODULES:
        efficiencies = module_efficiency(module, temperatures)
        assert np.all(np.diff(efficiencies) < 0.0)
        assert efficiencies[0] == pytest.approx(module.eta_ref)


@pytest.mark.parametrize("temperature_k", [298.15, 318.15, 348.15])
@pytest.mark.parametrize("module_index", [0, 1])
def test_thermal_drift_coefficient_is_exact(module_index: int, temperature_k: float) -> None:
    module = MODULES[module_index]
    step = 1.0e-4
    numerical = float(
        (
            np.log(module_efficiency(module, temperature_k + step))
            - np.log(module_efficiency(module, temperature_k - step))
        )
        / (2.0 * step)
    )
    exact = thermal_drift_coefficient(module, temperature_k)
    assert numerical == pytest.approx(module.beta_t, abs=1e-10)
    assert float(exact) == pytest.approx(module.beta_t, abs=1e-14)
    assert float(exact) == pytest.approx(numerical, abs=1e-10)


def test_symbolic_balance_derivative_sympy() -> None:
    symbolic = symbolic_balance_derivative()
    beta_symbol = sp.symbols("beta", real=True)
    assert symbolic["derivative_check"] == 0
    assert sp.simplify(symbolic["beta_exact"] - beta_symbol) == 0
    assert "T_j**4" in sp.sstr(symbolic["residual"])
    assert sp.simplify(
        sp.diff(symbolic["residual"], sp.symbols("T_j", real=True))
        - symbolic["derivative"]
    ) == 0


@pytest.mark.parametrize("site,module", SITE_MODULE_PAIRS)
def test_analytic_balance_derivative_matches_finite_difference(site, module) -> None:
    balance = energy_balance(site, module)
    temperature = balance.junction_temperature_k
    step = 1.0e-4
    arguments = dict(
        irradiance=site.irradiance,
        module=module,
        h_conv=balance.h_conv,
        absorptance=balance.absorptance,
    )
    ambient_kw = dict(
        ambient_temperature_k=site.ambient_k,
        module=module,
        h_conv=balance.h_conv,
        t_sky=balance.sky_temperature_k,
        absorptance=balance.absorptance,
    )
    numerical = (
        balance_residual(temperature + step, site.irradiance, **ambient_kw)
        - balance_residual(temperature - step, site.irradiance, **ambient_kw)
    ) / (2.0 * step)
    analytic = balance_derivative(temperature, **arguments)
    assert analytic < 0.0
    assert float(analytic) == pytest.approx(float(numerical), rel=1e-7, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. Solveur et conservation des flux
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("site,module", SITE_MODULE_PAIRS)
def test_newton_and_brent_agree(site, module) -> None:
    newton = float(
        solve_junction_temperature(
            site.irradiance, site.ambient_k, site.wind_speed, module
        )
    )
    brent = brent_junction_temperature(
        site.irradiance, site.ambient_k, site.wind_speed, module
    )
    assert newton == pytest.approx(brent, abs=1e-9)


def test_solver_is_vectorized_and_matches_scalar_calls() -> None:
    module = optimized_module()
    irradiances = np.array([700.0, 900.0, 1050.0])
    ambients = np.array([304.15, 310.15, 320.15])
    winds = np.array([3.5, 4.0, 1.8])
    vector = solve_junction_temperature(irradiances, ambients, winds, module)
    scalar = np.array(
        [
            float(
                solve_junction_temperature(g, t, v, module)
            )
            for g, t, v in zip(irradiances, ambients, winds)
        ]
    )
    assert vector.shape == (3,)
    assert np.allclose(vector, scalar, atol=1e-10)


@pytest.mark.parametrize("site,module", SITE_MODULE_PAIRS)
def test_strict_energy_flux_conservation(site, module) -> None:
    balance: EnergyBalance = energy_balance(site, module)
    total_out = (
        balance.electrical_power
        + balance.convection_power
        + balance.radiation_power
        + balance.extraction_power
    )
    assert abs(balance.residual) < 1e-11
    assert abs(balance.absorbed_power - total_out) < 1e-11
    assert balance.absorbed_power == pytest.approx(
        balance.absorptance * site.irradiance, abs=1e-12
    )
    recomputed = balance_terms(
        balance.junction_temperature_k,
        site.irradiance,
        site.ambient_k,
        module,
        balance.h_conv,
        balance.sky_temperature_k,
        balance.absorptance,
    )
    assert abs(recomputed[0] - sum(recomputed[1:])) < 1e-11


def test_hybrid_module_energy_conservation() -> None:
    module = hybrid_module()
    balance = energy_balance(MATAM, module)
    assert balance.extraction_power > 0.0
    total_out = (
        balance.electrical_power
        + balance.convection_power
        + balance.radiation_power
        + balance.extraction_power
    )
    assert abs(balance.absorbed_power - total_out) < 1e-11
    assert balance.junction_temperature_c < energy_balance(MATAM, optimized_module()).junction_temperature_c


def test_residual_is_monotonically_decreasing_and_bracketed() -> None:
    module = optimized_module()
    temperature = np.linspace(DAKAR.ambient_k - 10.0, DAKAR.ambient_k + 250.0, 501)
    residual = balance_residual(
        temperature,
        DAKAR.irradiance,
        DAKAR.ambient_k,
        module,
        convection_coefficient(DAKAR.wind_speed),
        sky_temperature(DAKAR.ambient_k),
    )
    assert np.all(np.diff(residual) < 0.0)
    assert residual[0] > 0.0 > residual[-1]


# ---------------------------------------------------------------------------
# 4. Monotonie physique
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("site,module", SITE_MODULE_PAIRS)
def test_junction_temperature_above_ambient(site, module) -> None:
    balance = energy_balance(site, module)
    assert balance.junction_temperature_k > site.ambient_k
    assert balance.junction_temperature_c > site.ambient_c
    assert balance.electrical_power > 0.0
    assert balance.radiation_power > 0.0


def test_junction_temperature_rises_with_irradiance() -> None:
    module = optimized_module()
    irradiances = np.linspace(400.0, 1200.0, 17)
    temperatures = np.array(
        [
            float(
                solve_junction_temperature(g, MATAM.ambient_k, MATAM.wind_speed, module)
            )
            - KELVIN_OFFSET
            for g in irradiances
        ]
    )
    assert np.all(np.diff(temperatures) > 0.0)


def test_junction_temperature_falls_with_wind_speed() -> None:
    module = optimized_module()
    winds = np.linspace(0.5, 8.0, 16)
    temperatures = np.array(
        [
            float(
                solve_junction_temperature(MATAM.irradiance, MATAM.ambient_k, v, module)
            )
            - KELVIN_OFFSET
            for v in winds
        ]
    )
    assert np.all(np.diff(temperatures) < 0.0)
    assert temperatures[-1] - temperatures[0] < -10.0


def test_junction_and_efficiency_versus_ambient_temperature() -> None:
    ambients = np.linspace(25.0, 55.0, 31)
    for site in SAHEL_SITES:
        temperatures, efficiencies = sweep_ambient_temperature(
            site, optimized_module(), ambients
        )
        assert np.all(np.diff(temperatures) > 0.0)
        assert np.all(np.diff(efficiencies) < 0.0)
        assert np.all(temperatures > ambients)


# ---------------------------------------------------------------------------
# 5. Superiorite du materiau optimise
# ---------------------------------------------------------------------------


def test_optimized_material_superiority_at_matam_instantaneous() -> None:
    optimized = energy_balance(MATAM, optimized_module())
    silicon = energy_balance(MATAM, silicon_module())
    assert optimized.efficiency > silicon.efficiency
    assert optimized.junction_temperature_c < silicon.junction_temperature_c
    assert optimized.electrical_power > silicon.electrical_power
    assert optimized.electrical_power / silicon.electrical_power > 1.20


@pytest.mark.parametrize("site", SAHEL_SITES)
def test_optimized_material_annual_gain_positive(site) -> None:
    gain: AnnualGain = annual_energy_gain(site)
    assert gain.delta_kwh_per_m2 > 0.0
    assert gain.relative_gain_percent > 0.0
    assert gain.optimized.kwh_per_m2 > gain.silicon.kwh_per_m2


def test_optimized_material_gain_at_matam_is_largest() -> None:
    gains = [annual_energy_gain(site) for site in SAHEL_SITES]
    assert gains[0].relative_gain_percent < gains[1].relative_gain_percent
    assert gains[1].relative_gain_percent < gains[2].relative_gain_percent
    assert gains[2].relative_gain_percent > 15.0
    assert gains[2].delta_kwh_per_m2 > 100.0


def test_annual_yields_are_physically_bounded() -> None:
    for site in SAHEL_SITES:
        for module in MODULES:
            result = annual_yield(site, module)
            assert 2000.0 < result.incident_kwh_per_m2 < 3000.0
            assert 0.0 < result.kwh_per_m2 < result.incident_kwh_per_m2
            assert 0.10 < result.performance_ratio < 0.30
            assert result.mean_junction_c > site.ambient_c


# ---------------------------------------------------------------------------
# 6. Figure de publication
# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)


def _make_night5_figure(output_path: Path) -> dict:
    optimized, silicon = MODULES
    ambients = np.linspace(25.0, 55.0, 121)
    colors = ("#1b7837", "#d95f0e", "#a50f15")
    gains = [annual_energy_gain(site) for site in SAHEL_SITES]

    figure, (ax_profile, ax_production) = plt.subplots(
        1, 2, figsize=(13.2, 5.7), gridspec_kw={"width_ratios": (1.0, 1.0), "wspace": 0.27}
    )

    ax_efficiency = ax_profile.twinx()
    handles = []
    for site, color in zip(SAHEL_SITES, colors):
        temperatures, efficiencies = sweep_ambient_temperature(site, optimized, ambients)
        line_temperature, = ax_profile.plot(
            ambients, temperatures, color=color, lw=2.0, label=site.name
        )
        handles.append(line_temperature)
        ax_profile.plot(ambients, temperatures, "o", color=color, ms=4, markevery=24)
        ax_efficiency.plot(
            ambients, 100.0 * efficiencies, color=color, lw=1.3, ls="--", alpha=0.85
        )
        ax_profile.plot(
            site.ambient_c,
            float(
                energy_balance(site, optimized).junction_temperature_c
            ),
            marker="*",
            ms=12,
            color=color,
            mec="black",
            mew=0.5,
            zorder=6,
        )

    ax_profile.set_xlim(25.0, 55.0)
    ax_profile.set_xlabel("Temperature ambiante $T_{amb}$ (°C)")
    ax_profile.set_ylabel("Temperature de jonction $T_j$ (°C)")
    ax_efficiency.set_ylabel("Rendement réel $\\eta(T_j)$ (%)", color="#08306b")
    ax_efficiency.tick_params(axis="y", labelcolor="#08306b")
    ax_profile.set_title("(a) Réponse thermique — $G$ et vent fixes par site", fontsize=11)
    ax_profile.grid(alpha=0.3)
    ax_profile.legend(handles=handles, loc="upper left", framealpha=0.9, title="Site")
    ax_profile.text(
        0.03,
        0.04,
        "—— $T_j$ (axe gauche)    - - $\\eta(T_j)$ (axe droite)    ★ point de conception",
        transform=ax_profile.transAxes,
        fontsize=7.5,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.7", "alpha": 0.9},
    )

    labels = [site.name for site in SAHEL_SITES]
    positions = np.arange(len(labels))
    width = 0.36
    bars_opt = ax_production.bar(
        positions - width / 2.0,
        [gain.optimized.kwh_per_m2 for gain in gains],
        width,
        color="#1b7837",
        label="Pérovskite / topologique ($E_g$ = 1,45 eV)",
    )
    bars_si = ax_production.bar(
        positions + width / 2.0,
        [gain.silicon.kwh_per_m2 for gain in gains],
        width,
        color="#993404",
        label="Silicium cristallin ($E_g$ = 1,12 eV)",
    )
    for bars in (bars_opt, bars_si):
        for bar in bars:
            ax_production.annotate(
                f"{bar.get_height():.0f}",
                (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=7.5,
            )
    for position, gain in zip(positions, gains):
        ax_production.annotate(
            f"+{gain.relative_gain_percent:.1f} %\n(+{gain.delta_kwh_per_m2:.0f} kWh/m²)",
            (position, gain.optimized.kwh_per_m2),
            textcoords="offset points",
            xytext=(0, -30),
            ha="center",
            fontsize=8,
            color="white",
            fontweight="bold",
        )
    ax_production.set_xticks(positions, labels)
    ax_production.set_ylim(0.0, 1.22 * max(g.optimized.kwh_per_m2 for g in gains))
    ax_production.set_ylabel("Production électrique annuelle (kWh m$^{-2}$ an$^{-1}$)")
    ax_production.set_title("(b) Bilan énergétique annuel comparé", fontsize=11)
    ax_production.grid(alpha=0.3, axis="y")
    ax_production.legend(loc="upper left", framealpha=0.9)

    figure.suptitle(
        "UAM-TOPOMAT — bilan thermodynamique sahélien et rendement haute température",
        fontsize=12.5,
        y=0.995,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    return {
        "gains": [gain.relative_gain_percent for gain in gains],
        "annual_optimized": [gain.optimized.kwh_per_m2 for gain in gains],
        "matam_delta": gains[-1].delta_kwh_per_m2,
        "tj_range": (
            float(
                np.min(
                    sweep_ambient_temperature(SAHEL_SITES[0], optimized, ambients)[0]
                )
            ),
            float(
                np.max(
                    sweep_ambient_temperature(SAHEL_SITES[-1], optimized, ambients)[0]
                )
            ),
        ),
    }


def test_generate_night5_figure() -> None:
    output = ROOT / "results" / "figures" / "night5_sahel_thermal_balance.png"
    metadata = _make_night5_figure(output)
    assert output.exists()
    assert output.stat().st_size > 30_000
    assert abs(_png_dpi(output) - 300.0) < 1.0
    assert all(gain > 0.0 for gain in metadata["gains"])
    assert metadata["gains"] == sorted(metadata["gains"])
    assert metadata["gains"][-1] > 15.0
    assert metadata["matam_delta"] > 100.0
    assert metadata["tj_range"][0] > 25.0
    assert metadata["tj_range"][1] < 120.0
