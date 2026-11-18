"""UAM-TOPOMAT -- Validation de la Nuit 7.

Tests du supercondensateur asymetrique MXene sans plomb : capacite quantique
issue de la DOS Tight-Binding, association serie C_Q--C_H (SymPy), conservation
de la charge sur les cycles galvanostatiques, ESR sous climat sahélien 25--50 C,
maintien de la capacite a 47 C (Matam) et figure de Ragone 300 dpi.
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

from src.optics.solar_optics import KB_SI, Q_E, T_REF
from src.thermo.sahel_climate import MATAM, SAHEL_SITES
from src.thermo.supercapacitor_mxene import (
    AGING_ONSET_K,
    AsymmetricSupercapacitor,
    GalvanostaticCycle,
    MXeneElectrode,
    RagonePoint,
    esr,
    negative_mxene_electrode,
    pore_accessibility,
    positive_mxene_electrode,
    site_ragone_point,
    symbolic_quantum_capacitance,
    symbolic_series_capacitance,
)

CURRENTS_A_KG = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0,
                 200.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)
TEMPERATURES_C = np.array([25.0, 30.0, 35.0, 40.0, 45.0, 47.0, 50.0])
LI_ION_POWER = np.array([50.0, 100.0, 200.0, 500.0, 1000.0, 3000.0])
LI_ION_ENERGY = np.array([210.0, 225.0, 215.0, 180.0, 130.0, 65.0])
SUPERCAP_POWER = np.array([30.0, 100.0, 500.0, 2000.0, 8000.0, 20000.0])
SUPERCAP_ENERGY = np.array([9.0, 8.0, 5.5, 3.5, 2.0, 1.0])


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
def _cell() -> AsymmetricSupercapacitor:
    return AsymmetricSupercapacitor()


@lru_cache(maxsize=None)
def _ragone(temperature_c: float) -> tuple[RagonePoint, ...]:
    cell = _cell()
    return cell.ragone_curve(CURRENTS_A_KG, temperature_c + 273.15)


# ---------------------------------------------------------------------------
# 1. Capacite quantique et association serie
# ---------------------------------------------------------------------------


def test_quantum_capacitance_matches_tight_binding_dos() -> None:
    electrode = positive_mxene_electrode()
    potentials = np.array([-0.4, 0.0, 0.35, 0.8])
    energies = electrode.config.efermi_ev + potentials
    dos = electrode.dos(energies, T_REF)
    expected = float(Q_E) * dos / electrode.cell_area_m2
    computed = electrode.quantum_capacitance_areal(potentials, T_REF)
    assert np.all(dos > 0.0)
    assert np.allclose(computed, expected, rtol=1e-14)
    assert np.all(computed > 0.0)
    assert electrode.cell_area_m2 == pytest.approx(
        np.sqrt(3.0) / 2.0 * 9.0e-20, rel=1e-15
    )


def test_series_capacitance_is_below_both_components() -> None:
    for electrode in (positive_mxene_electrode(), negative_mxene_electrode()):
        for temperature in (T_REF, MATAM.ambient_k):
            potentials = np.linspace(-0.6, 0.9, 25)
            quantum = np.asarray(electrode.quantum_capacitance_areal(potentials, temperature))
            helmholtz = electrode.helmholtz_capacitance_areal(temperature)
            series = np.asarray(electrode.capacitance_areal(potentials, temperature))
            assert np.all(series < np.minimum(quantum, helmholtz))
            assert np.allclose(
                1.0 / series, 1.0 / quantum + 1.0 / helmholtz, rtol=1e-13
            )
            assert np.all(series > 0.0) and np.all(np.isfinite(series))


def test_electrode_capacitance_and_charge_curve_are_monotonic() -> None:
    electrode = positive_mxene_electrode()
    potentials = np.linspace(0.0, electrode.config.window_v, 41)
    capacitance = electrode.capacitance_specific(potentials, T_REF)
    charge = electrode.charge_at(potentials, T_REF)
    assert np.all(capacitance > 0.0)
    assert np.all(np.diff(charge) > 0.0)
    assert charge[0] == pytest.approx(0.0, abs=1e-10)
    assert electrode.potential_from_charge(charge) == pytest.approx(potentials, abs=1e-9)
    assert electrode.max_charge_specific(T_REF) > 20_000.0


def test_helmholtz_temperature_and_pore_accessibility() -> None:
    electrode = positive_mxene_electrode()
    helmholtz = [
        electrode.helmholtz_capacitance_areal(t)
        for t in (298.15, 310.15, 313.15, 320.15, 323.15)
    ]
    assert helmholtz[0] < helmholtz[2]
    assert helmholtz[2] > helmholtz[3] > helmholtz[4]
    assert pore_accessibility(308.15) == pytest.approx(1.0)
    assert pore_accessibility(320.15) < 1.0
    assert 0.0 < pore_accessibility(AGING_ONSET_K + 20.0) < 1.0


def test_electrodes_are_lead_free_mxene_m2x() -> None:
    for electrode in (positive_mxene_electrode(), negative_mxene_electrode()):
        assert electrode.model.n_orbitals == 3
        assert electrode.model.orbital_labels == ("M1", "M2", "X")
        assert "MXene" in electrode.name
        assert "Pb" not in electrode.name and "lead" not in electrode.name.lower()
        assert electrode.config.lattice_a_angstrom > 0.0
        assert len(electrode.kpoints) > 500


def test_asymmetric_mass_ratio_balances_electrode_charges() -> None:
    cell = _cell()
    charge = cell.max_charge_specific(T_REF)
    positive_charge = charge / cell.mass_fraction_positive
    negative_charge = charge / cell.mass_fraction_negative
    assert positive_charge == pytest.approx(
        cell.positive.max_charge_specific(T_REF), rel=1e-12
    )
    assert negative_charge == pytest.approx(
        cell.negative.max_charge_specific(T_REF), rel=1e-12
    )
    assert cell.mass_fraction_positive * positive_charge == pytest.approx(
        cell.mass_fraction_negative * negative_charge, rel=1e-12
    )
    assert 0.4 < cell.mass_fraction_positive < 0.6
    assert cell.mass_fraction_positive + cell.mass_fraction_negative == pytest.approx(1.0)
    voltage = cell.open_circuit_voltage(charge, T_REF)
    expected = cell.positive.potential_from_charge(positive_charge, T_REF)
    expected += cell.negative.potential_from_charge(negative_charge, T_REF)
    assert voltage == pytest.approx(expected, rel=1e-12)
    assert voltage > 0.0


# ---------------------------------------------------------------------------
# 2. ESR sous climat sahélien
# ---------------------------------------------------------------------------


def test_esr_decreases_from_25c_to_50c() -> None:
    temperatures = TEMPERATURES_C + 273.15
    resistances = np.asarray(esr(temperatures))
    assert np.all(resistances > 0.0)
    assert np.all(np.diff(resistances) < 0.0)
    assert resistances[0] / resistances[-1] > 1.20
    assert esr(MATAM.ambient_k) == pytest.approx(
        float(esr(np.array([MATAM.ambient_k]))[0]), rel=1e-15
    )


def test_esr_follows_arrhenius_and_metallic_law() -> None:
    temperature = MATAM.ambient_k
    exponent = 0.15 / (float(KB_SI) / float(Q_E)) * (1.0 / temperature - 1.0 / T_REF)
    ionic = 2.0e-5 * np.exp(exponent)
    electronic = 4.0e-6 * (1.0 + 1.0e-3 * (temperature - T_REF))
    expected = 2.0e-6 + ionic + electronic
    assert esr(temperature) == pytest.approx(expected, rel=1e-14)
    assert esr(T_REF) == pytest.approx(2.6e-5, rel=1e-12)
    with pytest.raises(ValueError):
        esr(0.0)


# ---------------------------------------------------------------------------
# 3. Conservation de la charge sur les cycles galvanostatiques
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("current", [10.0, 100.0, 1000.0])
def test_charge_conservation_over_galvanostatic_cycle(current: float) -> None:
    cycle = _cell().galvanostatic_cycle(current, T_REF)
    assert isinstance(cycle, GalvanostaticCycle)
    assert cycle.charge_capacity_c_kg > 0.0
    assert cycle.discharge_capacity_c_kg > 0.0
    assert cycle.charge_capacity_c_kg == pytest.approx(
        float(np.trapezoid(np.full_like(cycle.charge_time_s, current),
                          cycle.charge_time_s)), rel=1e-12
    )
    delivered_from_current = float(
        np.trapezoid(np.full_like(cycle.discharge_time_s, current),
                     cycle.discharge_time_s)
    )
    assert delivered_from_current == pytest.approx(
        cycle.discharge_capacity_c_kg, rel=1e-10
    )
    assert cycle.charge_capacity_c_kg == pytest.approx(
        cycle.discharge_capacity_c_kg + cycle.retained_charge_c_kg, rel=1e-12
    )
    assert cycle.coulombic_efficiency == pytest.approx(
        cycle.discharge_capacity_c_kg / cycle.charge_capacity_c_kg, rel=1e-14
    )
    assert 0.9 < cycle.coulombic_efficiency <= 1.0
    assert cycle.capacity_mah_g == pytest.approx(
        cycle.discharge_capacity_c_kg / 3600.0, rel=1e-14
    )


def test_open_circuit_voltage_is_bounded_by_cell_window() -> None:
    cell = _cell()
    charges = np.linspace(0.0, cell.max_charge_specific(T_REF), 21)
    voltage = cell.open_circuit_voltage(charges, T_REF)
    assert voltage[0] == pytest.approx(0.0, abs=1e-9)
    assert np.all(np.diff(voltage) > 0.0)
    assert voltage[-1] <= cell.voltage_max_v
    assert cell.voltage_max_v == pytest.approx(1.9)


def test_energy_and_power_are_consistent_with_discharge_integral() -> None:
    cycle = _cell().galvanostatic_cycle(500.0, T_REF)
    energy_j_kg = float(
        np.trapezoid(cycle.discharge_voltage_v[::-1],
                     cycle.discharge_charge_c_kg[::-1])
    )
    assert cycle.energy_density_wh_kg == pytest.approx(energy_j_kg / 3600.0, rel=1e-12)
    duration = cycle.discharge_time_s[-1] - cycle.discharge_time_s[0]
    assert cycle.power_density_w_kg == pytest.approx(energy_j_kg / duration, rel=1e-2)
    assert 0.0 < cycle.energy_density_wh_kg < 40.0
    assert 0.0 < cycle.power_density_w_kg < 1.0e4
    assert 0.0 < cycle.energy_efficiency <= 1.0


def test_repeated_cycles_have_identical_state_of_charge() -> None:
    cell = _cell()
    first = cell.galvanostatic_cycle(250.0, T_REF)
    second = cell.galvanostatic_cycle(250.0, T_REF)
    assert first.discharge_capacity_c_kg == pytest.approx(
        second.discharge_capacity_c_kg, rel=1e-14
    )
    assert first.retained_charge_c_kg == pytest.approx(
        second.retained_charge_c_kg, rel=1e-14
    )
    assert first.discharge_voltage_v[-1] == pytest.approx(0.0, abs=1e-9)
    assert first.charge_voltage_v[-1] == pytest.approx(
        first.voltage_max_v, abs=1e-6
    )


def test_higher_current_reduces_capacity_and_coulombic_efficiency() -> None:
    points = _cell().ragone_curve((1.0, 100.0, 1000.0, 5000.0, 10_000.0), T_REF)
    capacities = [point.capacity_mah_g for point in points]
    efficiencies = [point.coulombic_efficiency for point in points]
    powers = [point.power_w_kg for point in points]
    assert capacities == sorted(capacities, reverse=True)
    assert efficiencies == sorted(efficiencies, reverse=True)
    assert powers == sorted(powers)
    assert capacities[0] / capacities[-1] > 1.05


# ---------------------------------------------------------------------------
# 4. Verifications symboliques (SymPy)
# ---------------------------------------------------------------------------


def test_symbolic_series_capacitance_sympy() -> None:
    symbolic = symbolic_series_capacitance()
    quantum, helmholtz = symbolic["quantum"], symbolic["helmholtz"]
    expected = helmholtz**2 / (quantum + helmholtz) ** 2
    assert symbolic["series_check"] == 0
    assert symbolic["voltage_check"] == 0
    assert symbolic["energy_check"] == 0
    assert sp.simplify(symbolic["derivative"] - expected) == 0
    assert symbolic["quantum_bound"].is_negative is True
    assert symbolic["helmholtz_bound"].is_negative is True
    assert sp.simplify(symbolic["capacitance"] - quantum * helmholtz
                       / (quantum + helmholtz)) == 0
    assert sp.simplify(
        symbolic["capacitance"] * (1 / quantum + 1 / helmholtz) - 1
    ) == 0


def test_symbolic_quantum_capacitance_sympy() -> None:
    symbolic = symbolic_quantum_capacitance()
    assert symbolic["scaling_check"] == 0
    assert symbolic["series_limit_large_ch"] == symbolic["quantum"]
    assert symbolic["series_limit_large_cq"] == sp.Symbol("C_H", positive=True)
    assert "e**2" in sp.sstr(symbolic["quantum"])


# ---------------------------------------------------------------------------
# 5. Tenue en temperature (Matam, 47 C)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("current", [10.0, 100.0, 1000.0])
def test_capacity_retention_at_matam_exceeds_94_percent(current: float) -> None:
    cell = _cell()
    retention = cell.capacity_retention(current, MATAM.ambient_k)
    assert retention > 0.94
    assert 0.94 < retention < 1.02
    assert MATAM.ambient_c == pytest.approx(47.0)


def test_retention_peaks_at_40c_then_decreases() -> None:
    cell = _cell()
    retentions = np.array([
        cell.capacity_retention(100.0, temperature_c + 273.15)
        for temperature_c in TEMPERATURES_C
    ])
    assert retentions[0] == pytest.approx(1.0, rel=1e-12)
    assert np.max(retentions) == retentions[3]
    assert retentions[-1] < retentions[-2] < retentions[3]
    assert retentions[-1] > 0.94


def test_site_ragone_point_uses_site_ambient_temperature() -> None:
    for site in SAHEL_SITES:
        point = site_ragone_point(site, 100.0, _cell())
        assert point.temperature_k == pytest.approx(site.ambient_k)
        assert point.energy_wh_kg > 0.0
        assert point.power_w_kg > 0.0
        assert point.esr_ohm_kg == pytest.approx(float(esr(site.ambient_k)))
    hot = site_ragone_point(MATAM, 100.0, _cell())
    cold = site_ragone_point(SAHEL_SITES[0], 100.0, _cell())
    assert hot.energy_wh_kg < cold.energy_wh_kg


# ---------------------------------------------------------------------------
# 6. Figure de publication : diagramme de Ragone
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


def _make_night7_figure(output_path: Path) -> dict:
    cell = _cell()
    cold = _ragone(25.0)
    hot = _ragone(47.0)
    powers_cold = np.array([point.power_w_kg for point in cold])
    energies_cold = np.array([point.energy_wh_kg for point in cold])
    powers_hot = np.array([point.power_w_kg for point in hot])
    energies_hot = np.array([point.energy_wh_kg for point in hot])

    figure = plt.figure(figsize=(13.6, 9.6))
    grid = figure.add_gridspec(
        2, 2, width_ratios=(1.35, 1.0), hspace=0.34, wspace=0.30
    )
    ax_ragone = figure.add_subplot(grid[0, 0])
    ax_esr = figure.add_subplot(grid[0, 1])
    ax_cycle = figure.add_subplot(grid[1, 0])
    ax_retention = figure.add_subplot(grid[1, 1])

    ax_ragone.loglog(LI_ION_POWER, LI_ION_ENERGY, "s-", color="#7b3294",
                     lw=1.8, ms=5, label="Batteries Li-ion (commerciales)")
    ax_ragone.loglog(SUPERCAP_POWER, SUPERCAP_ENERGY, "D--", color="#1b7837",
                     lw=1.8, ms=5, label="Supercondensateurs commerciaux")
    ax_ragone.loglog(powers_cold, energies_cold, "o-", color="#08306b",
                     lw=2.2, ms=5.5, label="MXènes topologiques — 25 °C")
    ax_ragone.loglog(powers_hot, energies_hot, "^--", color="#d95f0e",
                     lw=2.0, ms=5.5, label="MXènes topologiques — Matam 47 °C")
    ax_ragone.fill_between(
        LI_ION_POWER, LI_ION_ENERGY * 0.85, LI_ION_ENERGY * 1.15,
        color="#7b3294", alpha=0.10,
    )
    ax_ragone.fill_between(
        SUPERCAP_POWER, SUPERCAP_ENERGY * 0.80, SUPERCAP_ENERGY * 1.20,
        color="#1b7837", alpha=0.10,
    )
    ax_ragone.set_xlabel("Densité de puissance (W kg$^{-1}$)")
    ax_ragone.set_ylabel("Densité d'énergie (Wh kg$^{-1}$)")
    ax_ragone.set_title("(a) Diagramme de Ragone — comparaison des technologies", fontsize=11)
    ax_ragone.grid(True, which="both", alpha=0.3)
    ax_ragone.set_xlim(20.0, 3.0e4)
    ax_ragone.set_ylim(0.5, 400.0)
    ax_ragone.legend(loc="lower left", framealpha=0.92, fontsize=7.5)
    ax_ragone.text(
        0.98, 0.06,
        "MXènes $M_2X$ sans plomb : $C_Q=e^2$·DOS et $C_H$ en série\n"
        "$U_{max}=1{,}9$ V, $C_{cell}\\approx 30$ F g$^{-1}$",
        transform=ax_ragone.transAxes, ha="right", va="bottom", fontsize=7.5,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.7", "alpha": 0.9},
    )

    temperatures = TEMPERATURES_C + 273.15
    resistances = np.asarray(esr(temperatures)) * 1.0e6
    ax_esr.plot(TEMPERATURES_C, resistances, "o-", color="#08306b", lw=2.0, ms=5)
    ax_esr.plot(MATAM.ambient_c, float(esr(MATAM.ambient_k)) * 1.0e6,
                "*", color="#d95f0e", ms=14, mec="black", mew=0.5, zorder=5)
    ax_esr.set_xlabel("Température $T$ (°C)")
    ax_esr.set_ylabel("ESR (µΩ kg$^{-1}$)")
    ax_esr.set_title("(b) Résistance série sous climat sahélien", fontsize=11)
    ax_esr.grid(alpha=0.3)
    ax_esr.annotate(
        "Matam 47 °C\n" + f"{float(esr(MATAM.ambient_k)) * 1.0e6:.2f} µΩ kg$^{{-1}}$",
        (MATAM.ambient_c, float(esr(MATAM.ambient_k)) * 1.0e6),
        textcoords="offset points", xytext=(-78, 20), fontsize=8,
        arrowprops={"arrowstyle": "->", "color": "0.4"},
    )

    for temperature, color, style in ((298.15, "#08306b", "-"),
                                      (MATAM.ambient_k, "#d95f0e", "--")):
        cycle = cell.galvanostatic_cycle(500.0, temperature)
        ax_cycle.plot(cycle.charge_time_s, cycle.charge_voltage_v,
                      color=color, ls=style, lw=1.7,
                      label=f"Charge {temperature - 273.15:.0f} °C")
        ax_cycle.plot(cycle.discharge_time_s, cycle.discharge_voltage_v,
                      color=color, ls=style, lw=1.7, alpha=0.7,
                      label=f"Décharge {temperature - 273.15:.0f} °C")
    ax_cycle.axhline(0.0, color="0.35", lw=0.7)
    ax_cycle.set_xlabel("Temps (s), courant 500 A kg$^{-1}$")
    ax_cycle.set_ylabel("Tension de cellule $U$ (V)")
    ax_cycle.set_title("(c) Cycles galvanostatiques 25 °C / 47 °C", fontsize=11)
    ax_cycle.set_xlim(0.0, 130.0)
    ax_cycle.set_ylim(0.0, 2.1)
    ax_cycle.grid(alpha=0.3)
    ax_cycle.legend(loc="upper right", framealpha=0.9, fontsize=7)

    retentions = np.array([
        cell.capacity_retention(100.0, temperature) for temperature in temperatures
    ])
    ax_retention.plot(TEMPERATURES_C, 100.0 * retentions, "o-",
                      color="#08306b", lw=2.0, ms=5)
    ax_retention.axhline(94.0, color="#a50f15", ls="--", lw=1.2,
                         label="Seuil 94 %")
    ax_retention.axvline(MATAM.ambient_c, color="0.5", ls=":", lw=1.0)
    ax_retention.plot(MATAM.ambient_c, 100.0 * retentions[-2], "*",
                      color="#d95f0e", ms=14, mec="black", mew=0.5, zorder=5)
    ax_retention.set_xlabel("Température $T$ (°C)")
    ax_retention.set_ylabel("Rétention de capacité (%)")
    ax_retention.set_title("(d) Stabilité thermique à 100 A kg$^{-1}$", fontsize=11)
    ax_retention.set_ylim(92.0, 101.0)
    ax_retention.grid(alpha=0.3)
    ax_retention.legend(loc="lower left", framealpha=0.9)

    figure.suptitle(
        "UAM-TOPOMAT — Nuit 7 : supercondensateurs topologiques MXènes sans plomb et diagramme de Ragone sahélien",
        fontsize=12.0, y=0.995,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    return {
        "energies_cold": energies_cold,
        "powers_cold": powers_cold,
        "energies_hot": energies_hot,
        "powers_hot": powers_hot,
        "retention_matam": float(retentions[-2]),
        "esr_ratio": float(resistances[0] / resistances[-1]),
        "capacity_mah_g": cold[3].capacity_mah_g,
    }


def test_generate_night7_ragone_figure() -> None:
    output = ROOT / "results" / "figures" / "night7_ragone_mxene_sahel.png"
    metadata = _make_night7_figure(output)
    assert output.exists()
    assert output.stat().st_size > 60_000
    assert abs(_png_dpi(output) - 300.0) < 1.0
    assert len(metadata["energies_cold"]) == len(CURRENTS_A_KG)
    assert np.all(metadata["energies_cold"] > 0.0)
    assert np.all(metadata["powers_cold"] > 0.0)
    assert metadata["energies_cold"][0] > metadata["energies_cold"][-1]
    assert metadata["powers_cold"][-1] > 1000.0
    commercial_at_100 = float(np.interp(100.0, SUPERCAP_POWER, SUPERCAP_ENERGY))
    mxene_at_100 = float(np.interp(
        100.0, metadata["powers_cold"], metadata["energies_cold"]
    ))
    assert mxene_at_100 > commercial_at_100 * 1.3
    assert mxene_at_100 < 100.0
    assert metadata["retention_matam"] > 0.94
    assert metadata["esr_ratio"] > 1.20
    assert metadata["capacity_mah_g"] > 5.0
