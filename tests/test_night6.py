"""UAM-TOPOMAT -- Validation de la Nuit 6.

Tests du solveur EDP couple Poisson -- derive--diffusion (Scharfetter--Gummel) :
equilibre thermodynamique a l'obscurite, conservation exacte du courant total,
verifications symboliques (Einstein, Shockley, developpement de Bernoulli),
courbes J(V) et tendances thermiques 25--50 C sous climat sahelien, et figure
de publication 300 dpi.
"""

from __future__ import annotations

import struct
import sys
from functools import lru_cache
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

from src.optics.solar_optics import KB_SI, P_INPUT, Q_E, photon_flux_above_gap
from src.pde.carrier_transport import (
    JVResult,
    bernoulli,
    device_from_module,
    einstein_diffusivity,
    intrinsic_density,
    junction_temperature,
    mobility_at_temperature,
    perovskite_device,
    silicon_device,
    site_electrical_result,
    solve_carrier_transport,
    solve_dark_equilibrium,
    solve_jv_curve,
    symbolic_einstein_relation,
    symbolic_scharfetter_gummel,
    symbolic_shockley_equation,
    temperature_sweep,
    thermal_voltage,
)
from src.thermo.sahel_climate import (
    DAKAR,
    MATAM,
    SAHEL_SITES,
    SAINT_LOUIS,
    optimized_module,
    silicon_module,
)

DEVICES = (perovskite_device(), silicon_device())
TEMPERATURES_K = (298.15, 323.15, 353.85)
TEMPERATURES_C = (25.0, 30.0, 35.0, 40.0, 45.0, 50.0)


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
            pixels_per_metre = struct.unpack(">I", data[position + 8 : position + 12])[0]
            unit = data[position + 16]
            if unit == 1:
                return pixels_per_metre * 0.0254
        position += 12 + length
    raise ValueError("Chunk pHYs introuvable.")


@lru_cache(maxsize=None)
def _equality(site_name: str, material: str) -> JVResult:
    sites = {site.name: site for site in SAHEL_SITES}
    modules = {"perovskite": optimized_module, "silicon": silicon_module}
    return site_electrical_result(sites[site_name], modules[material](), n_points=31)


@lru_cache(maxsize=None)
def _sweep(material: str) -> tuple[JVResult, ...]:
    device = perovskite_device() if material == "perovskite" else silicon_device()
    return tuple(
        temperature_sweep(device, P_INPUT, TEMPERATURES_C, n_points=25)
    )


# ---------------------------------------------------------------------------
# 1. Schema de Scharfetter--Gummel et parametres des cellules
# ---------------------------------------------------------------------------


def test_thermal_voltage_and_einstein_relation_numerically() -> None:
    for temperature in TEMPERATURES_K:
        vt = thermal_voltage(temperature)
        assert vt == pytest.approx(KB_SI * temperature / Q_E, rel=1e-15)
        assert 0.02 < vt < 0.04
        mobility = 0.02
        assert einstein_diffusivity(mobility, temperature) == pytest.approx(mobility * vt)


def test_bernoulli_function_identity_and_limits() -> None:
    values = np.array([-700.0, -20.0, -1.0, -1e-8, 0.0, 1e-8, 1.0, 20.0, 700.0])
    b = bernoulli(values)
    assert np.all(np.isfinite(b))
    assert np.allclose(bernoulli(np.array([1.0, 5.0])) + np.array([1.0, 5.0]),
                       bernoulli(-np.array([1.0, 5.0])), rtol=1e-12)
    assert b[0] == pytest.approx(700.0, rel=1e-12)
    assert b[-1] == 0.0
    assert b[4] == pytest.approx(1.0)
    assert bernoulli(0.0) == pytest.approx(1.0)


def test_bernoulli_small_argument_series() -> None:
    z = np.linspace(-1e-3, 1e-3, 21)
    nonzero = z != 0.0
    exact = np.ones_like(z)
    exact[nonzero] = z[nonzero] / np.expm1(z[nonzero])
    assert np.allclose(bernoulli(z), exact, atol=1e-14)


def test_mobility_and_intrinsic_density_temperature_laws() -> None:
    device = perovskite_device()
    for temperature in TEMPERATURES_K:
        ni = intrinsic_density(
            device.bandgap_ev, temperature, device.electron_mass_rel, device.hole_mass_rel
        )
        assert ni > 0.0 and np.isfinite(ni)
        mu = mobility_at_temperature(device.mu_n_300, temperature)
        assert mu > 0.0
    mobilities = [mobility_at_temperature(device.mu_n_300, t) for t in TEMPERATURES_K]
    assert mobilities == sorted(mobilities, reverse=True)
    nis = [
        intrinsic_density(device.bandgap_ev, t, device.electron_mass_rel, device.hole_mass_rel)
        for t in TEMPERATURES_K
    ]
    assert nis == sorted(nis)


@pytest.mark.parametrize("device", DEVICES, ids=lambda d: d.name)
def test_device_configurations_are_physical(device) -> None:
    assert device.bandgap_ev > 0.0
    assert device.n_points >= 21
    assert 0.0 < device.quantum_efficiency <= 1.0
    assert device.absorption_coefficient > 0.0
    assert device.tau_n > 0.0 and device.tau_p > 0.0
    assert device.mu_n_300 > device.mu_p_300 > 0.0
    assert device_from_module(optimized_module()).bandgap_ev == pytest.approx(1.45)
    assert device_from_module(silicon_module()).bandgap_ev == pytest.approx(1.12)


# ---------------------------------------------------------------------------
# 2. Equilibre thermodynamique et conservation du courant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES, ids=lambda d: d.name)
@pytest.mark.parametrize("temperature_k", TEMPERATURES_K)
def test_dark_equilibrium_has_zero_current(device, temperature_k) -> None:
    solution = solve_dark_equilibrium(device, temperature_k)
    ni = solution.intrinsic_density_m3
    assert abs(solution.current_density) < 1.0e-12
    assert np.all(np.abs(solution.current_n) < 1.0e-12)
    assert np.all(np.abs(solution.current_p) < 1.0e-12)
    assert np.allclose(solution.electrons * solution.holes, ni**2, rtol=1e-12)
    assert np.allclose(solution.quasi_fermi_n, 0.0, atol=1e-14)
    assert np.allclose(solution.quasi_fermi_p, 0.0, atol=1e-14)
    assert np.all(np.diff(solution.potential) >= -1.0e-8)


@pytest.mark.parametrize("device", DEVICES, ids=lambda d: d.name)
def test_dark_forward_bias_follows_diode_law(device) -> None:
    temperature = 298.15
    currents = []
    for voltage in (0.40, 0.50, 0.60):
        solution = solve_carrier_transport(device, temperature, voltage, 0.0)
        assert solution.converged
        currents.append(solution.current_density)
    assert 0.0 < currents[0] < currents[1] < currents[2]
    vt = thermal_voltage(temperature)
    ideality = (0.60 - 0.40) / (vt * np.log(currents[2] / currents[0]))
    assert 1.0 < ideality < 2.2


@pytest.mark.parametrize("device", DEVICES, ids=lambda d: d.name)
def test_total_current_is_conserved_under_illumination(device) -> None:
    solution = solve_carrier_transport(device, 323.15, 0.0, P_INPUT)
    assert solution.converged
    total = solution.total_current
    assert np.ptp(total) < 1.0e-6
    assert np.allclose(total, solution.current_density, atol=1.0e-6)
    assert solution.current_density < 0.0
    assert solution.current_n[-1] < 0.0 and solution.current_p[0] < 0.0
    assert np.all(solution.electrons > 0.0) and np.all(solution.holes > 0.0)


@pytest.mark.parametrize("device", DEVICES, ids=lambda d: d.name)
def test_builtin_potential_bounds(device) -> None:
    temperature = 323.15
    solution = solve_dark_equilibrium(device, temperature)
    ni = solution.intrinsic_density_m3
    v_bi = thermal_voltage(temperature) * np.log(
        device.donor_density * device.acceptor_density / ni**2
    )
    span = solution.potential[-1] - solution.potential[0]
    assert span == pytest.approx(v_bi, rel=1e-8)
    assert 0.5 < span < 1.5


# ---------------------------------------------------------------------------
# 3. Verifications symboliques (SymPy)
# ---------------------------------------------------------------------------


def test_symbolic_einstein_relation_sympy() -> None:
    symbolic = symbolic_einstein_relation()
    assert sp.simplify(symbolic["einstein_check"]) == 0
    expected = sp.Symbol("mu", positive=True) * symbolic["thermal_voltage"]
    assert sp.simplify(symbolic["diffusivity"] - expected) == 0
    assert "k_B" in sp.sstr(symbolic["diffusivity"])


def test_symbolic_ideal_shockley_equation_sympy() -> None:
    symbolic = symbolic_shockley_equation()
    assert sp.simplify(symbolic["current_at_voc"]) == 0
    assert sp.simplify(symbolic["derivative_check"]) == 0
    assert "log" in sp.sstr(symbolic["open_circuit"])
    reduced = sp.Symbol("v", positive=True)
    jsc, j0 = sp.symbols("J_sc J_0", positive=True)
    condition = sp.simplify(symbolic["mpp_condition"].subs(reduced, 0))
    assert condition == -jsc / j0


def test_symbolic_scharfetter_gummel_expansion_sympy() -> None:
    symbolic = symbolic_scharfetter_gummel()
    assert sp.simplify(symbolic["expansion_check"]) == 0
    z = sp.Symbol("z", real=True)
    assert sp.simplify(symbolic["bernoulli"] - z / (sp.exp(z) - 1)) == 0
    assert sp.simplify(symbolic["b_negative_expansion"]
                       - (1 + z / 2 + z**2 / 12)) == 0


# ---------------------------------------------------------------------------
# 4. Caracteristiques J(V) et tendances thermiques
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES, ids=lambda d: d.name)
def test_jv_curve_key_metrics_are_consistent(device) -> None:
    result = solve_jv_curve(device, 323.15, P_INPUT, n_points=21)
    assert isinstance(result, JVResult)
    assert result.jsc > 0.0 and 0.0 < result.voc < device.bandgap_ev
    assert 0.5 < result.ff < 0.95
    assert 0.02 < result.efficiency < 0.40
    assert 0.0 < result.vmp < result.voc
    assert result.pmax == pytest.approx(result.vmp * result.jmp, rel=1e-6)
    assert result.efficiency == pytest.approx(result.pmax / P_INPUT, rel=1e-12)
    assert result.current_density[0] == pytest.approx(result.jsc, rel=1e-6)
    assert np.all(np.diff(result.voltage) > 0.0)
    before_voc = result.voltage <= result.voc
    assert np.all(result.power_density[before_voc] >= -1.0e-6)
    assert all(solution.converged for solution in result.solutions.values())


def test_jsc_matches_am15g_photon_budget() -> None:
    for device in DEVICES:
        result = solve_jv_curve(device, 298.15, P_INPUT, n_points=21)
        maximum = Q_E * photon_flux_above_gap(device.bandgap_ev) * device.quantum_efficiency
        assert 0.5 * maximum < result.jsc <= maximum * 1.01


@pytest.mark.parametrize("material", ["perovskite", "silicon"])
def test_voc_and_ff_decrease_between_25c_and_50c(material) -> None:
    results = _sweep(material)
    vocs = np.array([result.voc for result in results])
    ffs = np.array([result.ff for result in results])
    etas = np.array([result.efficiency for result in results])
    jscs = np.array([result.jsc for result in results])
    assert np.all(np.diff(vocs) < 0.0)
    assert np.all(np.diff(ffs) < 0.0)
    assert np.all(np.diff(etas) < 0.0)
    assert np.max(np.abs(np.diff(jscs))) / jscs[0] < 0.02
    assert 1.0e-4 < vocs[0] - vocs[-1] < 0.2


def test_perovskite_outperforms_silicon_across_temperature_range() -> None:
    perovskite = _sweep("perovskite")
    silicon = _sweep("silicon")
    for hot, cold in zip(perovskite, silicon):
        assert hot.voc > cold.voc
        assert hot.ff > cold.ff
        assert hot.efficiency > cold.efficiency


@pytest.mark.parametrize("site", SAHEL_SITES)
def test_junction_temperature_follows_night5_balance(site) -> None:
    for module in (optimized_module(), silicon_module()):
        temperature = junction_temperature(site, module)
        assert temperature > site.ambient_k
        assert 300.0 < temperature < 370.0


def test_perovskite_superiority_at_matam() -> None:
    perovskite = _equality("Matam", "perovskite")
    silicon = _equality("Matam", "silicon")
    assert perovskite.junction_temperature_c < silicon.junction_temperature_c
    assert perovskite.voc > silicon.voc
    assert perovskite.ff > silicon.ff
    assert perovskite.efficiency > silicon.efficiency
    assert perovskite.efficiency / silicon.efficiency > 1.15
    assert 0.15 < perovskite.efficiency < 0.35


def test_hot_sites_degrade_voc_and_ff() -> None:
    results = [_equality(site.name, "perovskite") for site in SAHEL_SITES]
    vocs = [result.voc for result in results]
    ffs = [result.ff for result in results]
    assert vocs[0] > vocs[1] > vocs[2]
    assert ffs[0] > ffs[1] > ffs[2]
    assert results[2].junction_temperature_c - results[0].junction_temperature_c > 20.0


# ---------------------------------------------------------------------------
# 5. Figure de publication
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


def _make_night6_figure(output_path: Path) -> dict:
    colors = ("#08306b", "#d95f0e", "#a50f15")
    site_results = [_equality(site.name, "perovskite") for site in SAHEL_SITES]
    silicon_matam = _equality("Matam", "silicon")

    figure, axes = plt.subplots(2, 2, figsize=(13.6, 9.6))
    ax_j, ax_p, ax_phi, ax_density = axes.ravel()

    for result, color in zip(site_results, colors):
        visible = result.voltage <= result.voc
        voltage_plot = np.append(result.voltage[visible], result.voc)
        current_plot = np.append(result.current_density[visible], 0.0)
        power_plot = np.append(result.power_density[visible], 0.0)
        ax_j.plot(voltage_plot, current_plot, color=color, lw=2.0,
                  label=f"{result.site_name} — $T_j$ = {result.junction_temperature_c:.1f} °C")
        ax_j.plot(result.voc, 0.0, "o", color=color, ms=5)
        ax_j.plot(result.vmp, result.jmp, "s", color=color, ms=4)
        ax_p.plot(voltage_plot, power_plot, color=color, lw=2.0,
                  label=result.site_name)
        ax_p.plot(result.vmp, result.pmax, "o", color=color, ms=5)
        at_zero = result.solutions[round(0.0, 10)]
        ax_phi.plot(at_zero.x * 1e6, at_zero.potential, color=color, lw=1.8,
                    label=result.site_name)
        ax_density.semilogy(at_zero.x * 1e6, np.maximum(at_zero.electrons, 1.0),
                            color=color, lw=1.6, label=f"{result.site_name} — $n$")
        ax_density.semilogy(at_zero.x * 1e6, np.maximum(at_zero.holes, 1.0),
                            color=color, lw=1.2, ls="--", alpha=0.8,
                            label=f"{result.site_name} — $p$")

    ax_j.axhline(0.0, color="0.35", lw=0.7)
    ax_j.set_xlim(0.0, 1.05)
    ax_j.set_ylim(-1.08 * max(result.jsc for result in site_results),
                  0.45 * max(result.jsc for result in site_results))
    ax_j.set_xlabel("Tension $V$ (V)")
    ax_j.set_ylabel("Densité de courant $J$ (A m$^{-2}$)")
    ax_j.set_title("(a) Caractéristiques $J(V)$ — pérovskite / topologique")
    ax_j.grid(alpha=0.3)
    ax_j.legend(loc="lower right", framealpha=0.9)

    ax_p.axhline(0.0, color="0.35", lw=0.7)
    ax_p.set_xlim(0.0, 1.05)
    ax_p.set_ylim(-10.0, 1.10 * max(result.pmax for result in site_results))
    ax_p.set_xlabel("Tension $V$ (V)")
    ax_p.set_ylabel("Puissance $P = JV$ (W m$^{-2}$)")
    ax_p.set_title("(b) Puissance électrique et point de puissance maximale")
    ax_p.grid(alpha=0.3)
    ax_p.legend(loc="upper right", framealpha=0.9)

    handle_phi = Line2D([0], [0], color="0.4", lw=1.4, label="$\\phi(x)$")
    ax_phi.axvline(0.75, color="0.45", ls=":", lw=1.0)
    ax_phi.set_xlabel("Position $x$ (µm)")
    ax_phi.set_ylabel("Potentiel électrostatique $\\phi$ (V)")
    ax_phi.set_title("(c) Profils internes de potentiel à $V=0$")
    ax_phi.grid(alpha=0.3)

    ax_density.set_xlabel("Position $x$ (µm)")
    ax_density.set_ylabel("Densité de porteurs (m$^{-3}$, échelle log)")
    ax_density.set_title("(d) Profils $n(x)$ (traits) et $p(x)$ (tirets)")
    ax_density.grid(alpha=0.3, which="both")
    ax_density.legend(loc="center left", fontsize=6.5, framealpha=0.9, ncol=2)

    figure.suptitle(
        "UAM-TOPOMAT — Nuit 6 : EDP Poisson–Scharfetter–Gummel et courbes $J(V)$ sous climat sahélien",
        fontsize=12.5,
        y=0.995,
    )
    handles, labels = ax_phi.get_legend_handles_labels()
    ax_phi.legend(handles + [handle_phi], labels + ["$\\phi(x)$"],
                  loc="lower right", framealpha=0.9, fontsize=7.5)
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    return {
        "vocs": [result.voc for result in site_results],
        "ffs": [result.ff for result in site_results],
        "etas": [result.efficiency for result in site_results],
        "jscs": [result.jsc for result in site_results],
        "si_eta_matam": silicon_matam.efficiency,
        "matam_gain": site_results[-1].efficiency / silicon_matam.efficiency - 1.0,
        "profile_points": len(site_results[-1].solutions[round(0.0, 10)].x),
    }


def test_generate_night6_figure() -> None:
    output = ROOT / "results" / "figures" / "night6_jv_curves_sahel.png"
    metadata = _make_night6_figure(output)
    assert output.exists()
    assert output.stat().st_size > 60_000
    assert abs(_png_dpi(output) - 300.0) < 1.0
    assert metadata["vocs"][0] > metadata["vocs"][1] > metadata["vocs"][2]
    assert metadata["ffs"][0] > metadata["ffs"][1] > metadata["ffs"][2]
    assert all(0.5 < value < 0.95 for value in metadata["ffs"])
    assert all(jsc > 0.0 for jsc in metadata["jscs"])
    assert metadata["matam_gain"] > 0.15
    assert metadata["profile_points"] >= 100
