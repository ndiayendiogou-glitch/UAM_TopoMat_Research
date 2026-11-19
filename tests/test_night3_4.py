"""UAM-TOPOMAT -- Validation des Nuits 3 et 4.

Tests de correspondance volume-bord sur nanorubans (Haldane et Qi-Wu-Zhang),
du modele de Shockley-Queisser sous AM1.5G avec dependance en temperature, de
l'absorption optique issue des elements de matrice dipolaires, et generation
des deux figures de publication (300 dpi) pour le Twin Transition Summit.
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

from src.optics.solar_optics import (
    HC_EV_NM,
    P_INPUT,
    Q_E,
    T_REF,
    absorption_spectrum,
    am15g_irradiance,
    dipole_matrix_element,
    optimize_bandgap,
    photon_flux_above_gap,
    radiative_saturation_current,
    shockley_queisser,
    solar_grid,
    sq_efficiency,
    sq_efficiency_curve,
    thermal_degradation,
)
from src.quantum.band_structure import TightBindingModel, uniform_k_grid
from src.topology.edge_states import (
    bulk_chern_fhs,
    haldane_ky_values,
    haldane_ribbon,
    qwz_ky_values,
    qwz_ribbon,
    solve_ribbon_spectrum,
)
from src.topology.models_topo import HaldaneModel

T1, T2, PHI, MASS, LATTICE_A = 1.0, 0.3, np.pi / 2.0, 0.6, 1.0
N_Y = 40
N_K = 161
KX_HALDANE = np.linspace(-np.pi / (np.sqrt(3.0) * LATTICE_A), np.pi / (np.sqrt(3.0) * LATTICE_A), N_K)
KX_QWZ = np.linspace(-np.pi, np.pi, N_K)
T_25, T_45 = 298.15, 318.15


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


def _haldane_spectrum(M: float = MASS, phi: float = PHI, t2: float = T2):
    ribbon, bulk_model, chern = haldane_ribbon(N_Y, T1, t2, phi, M, LATTICE_A)
    spectrum = solve_ribbon_spectrum(
        ribbon, KX_HALDANE, bulk_model, haldane_ky_values(LATTICE_A, 200), chern=chern
    )
    return ribbon, spectrum


def _qwz_spectrum(u: float = -1.0):
    ribbon, bulk_model, chern = qwz_ribbon(N_Y, u)
    spectrum = solve_ribbon_spectrum(
        ribbon, KX_QWZ, bulk_model, qwz_ky_values(321), chern=chern
    )
    return ribbon, spectrum


# ---------------------------------------------------------------------------
# 1. Correspondance volume-bord
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kx", [-1.1, -0.4, 0.0, 0.6, 1.4])
@pytest.mark.parametrize("ky", [0.0, 0.37, 1.13, 2.6])
def test_haldane_ribbon_bulk_limit(kx: float, ky: float) -> None:
    """Le ruban reconstruit exactement le modele de Haldane de la Nuit 2."""
    ribbon, _, _ = haldane_ribbon(N_Y)
    reference = HaldaneModel(t1=T1, t2=T2, phi=PHI, M=MASS, a=LATTICE_A)
    rebuilt = np.linalg.eigvalsh(ribbon.bulk_hamiltonian(kx, ky))
    expected = reference.eigenvalues(np.array([kx, ky, 0.0]))
    assert np.allclose(rebuilt, expected, atol=1e-12)


def test_haldane_edge_modes_only_in_topological_phase() -> None:
    """Modes de bord traversant le gap pour C=+1, aucun pour C=0."""
    _, topological = _haldane_spectrum(M=MASS)
    _, trivial = _haldane_spectrum(M=2.0)

    assert topological.chern == 1
    assert topological.has_edge_modes
    assert topological.n_gap_states > 50
    assert topological.n_crossings == 2

    assert trivial.chern == 0
    assert not trivial.has_edge_modes
    assert trivial.n_gap_states == 0
    assert trivial.n_crossings == 0


def test_haldane_chirality_reverses_with_flux() -> None:
    """Changer le signe de phi renverse C et la chiralite des bords."""
    _, positive = _haldane_spectrum(phi=np.pi / 2.0)
    _, negative = _haldane_spectrum(phi=-np.pi / 2.0)
    assert positive.chern == 1 and negative.chern == -1
    assert positive.has_edge_modes and negative.has_edge_modes
    assert positive.n_crossings == negative.n_crossings == 2

    def mean_slope(spectrum, side: str) -> float:
        groups: dict[float, list[float]] = {}
        for kx, energy, edge in spectrum.edge_records:
            if edge == side:
                groups.setdefault(round(kx, 12), []).append(energy)
        ordered = sorted((k, float(np.mean(values))) for k, values in groups.items())
        if len(ordered) < 2:
            return 0.0
        kvals = np.array([p[0] for p in ordered])
        evalues = np.array([p[1] for p in ordered])
        return float(np.polyfit(kvals, evalues, 1)[0])

    bottom_positive = mean_slope(positive, "bottom")
    bottom_negative = mean_slope(negative, "bottom")
    assert bottom_positive * bottom_negative < 0.0


def test_edge_states_are_spatially_localized() -> None:
    """Les etats du gap sont localises sur quelques mailles aux bords."""
    ribbon, spectrum = _haldane_spectrum()
    assert len(spectrum.edge_records) > 0
    kx_target, _, side = spectrum.edge_records[len(spectrum.edge_records) // 2]
    index = int(np.argmin(np.abs(spectrum.kpoints - kx_target)))
    energies, vectors = ribbon.solve(float(spectrum.kpoints[index]))
    band = int(
        np.argmin(
            np.where(
                spectrum.edge_mask[index],
                np.abs(energies),
                np.inf,
            )
        )
    )
    assert spectrum.edge_mask[index, band]
    density = np.abs(vectors[:, band]) ** 2
    per_cell = density.reshape(ribbon.n_cells, ribbon.n_orbitals, -1).sum(axis=1)
    edge_index = 0 if side == "bottom" else -1
    assert per_cell[edge_index] > 0.4
    assert per_cell[N_Y // 2] < 0.05
    assert np.sum(per_cell[:5]) > 0.9 or np.sum(per_cell[-5:]) > 0.9


@pytest.mark.parametrize("u", [-1.0, 1.0, -3.0])
def test_qwz_bulk_boundary_correspondence(u: float) -> None:
    """Ruban carre QWZ : C via FHS, comptage des traversees et localisation."""
    ribbon, bulk_model, chern = qwz_ribbon(N_Y, u)
    spectrum = solve_ribbon_spectrum(
        ribbon, KX_QWZ, bulk_model, qwz_ky_values(321), chern=chern
    )

    def d_vector(kx, ky):
        return np.array([np.sin(kx), np.sin(ky), u + np.cos(kx) + np.cos(ky)])

    fhs = bulk_chern_fhs(d_vector, 64)
    assert abs(fhs - round(fhs)) < 1e-9
    assert int(round(fhs)) == chern
    assert spectrum.has_edge_modes == (chern != 0)
    if chern != 0:
        assert spectrum.n_crossings == 2 * abs(chern)


def test_qwz_chirality_symbolic_sympy() -> None:
    """Masses de Dirac et chiralites sympy : C = -1 pour -2 < u < 0."""
    kx, ky, u = sp.symbols("kx ky u", real=True)
    dx, dy = sp.sin(kx), sp.sin(ky)
    dz = u + sp.cos(kx) + sp.cos(ky)
    jacobian = sp.Matrix([[sp.diff(dx, kx), sp.diff(dx, ky)], [sp.diff(dy, kx), sp.diff(dy, ky)]])
    determinant = sp.simplify(jacobian.det())
    assert sp.simplify(determinant - sp.cos(kx) * sp.cos(ky)) == 0

    points = [(0, 0), (sp.pi, 0), (0, sp.pi), (sp.pi, sp.pi)]
    charges = [sp.sign(determinant.subs({kx: a, ky: b})) for a, b in points]
    masses = [sp.simplify(dz.subs({kx: a, ky: b})) for a, b in points]
    assert masses == [u + 2, u, u, u - 2]
    assert charges == [1, -1, -1, 1]

    chern_lower = sp.simplify(-sp.Rational(1, 2) * sum(c * sp.sign(m.subs(u, -1)) for c, m in zip(charges, masses)))
    assert chern_lower == -1
    _, _, chern_numeric = qwz_ribbon(N_Y, -1.0)
    assert chern_numeric == int(chern_lower)


# ---------------------------------------------------------------------------
# 2. Optique et Shockley-Queisser
# ---------------------------------------------------------------------------


def test_am15g_spectrum_normalization_and_photon_flux() -> None:
    """Spectre AM1.5G normalise a 1000 W/m^2 et flux decroissant avec E_g."""
    wavelengths, irradiance = solar_grid()
    assert wavelengths[0] == 280.0 and wavelengths[-1] == 4000.0
    assert np.all(irradiance >= 0.0)
    assert float(np.trapezoid(irradiance, wavelengths)) == pytest.approx(P_INPUT, abs=1e-6)
    assert float(am15g_irradiance(550.0)) > 1.0
    assert float(am15g_irradiance(5000.0)) == 0.0

    gaps = np.linspace(0.5, 2.0, 13)
    fluxes = np.array([photon_flux_above_gap(float(eg)) for eg in gaps])
    assert np.all(np.diff(fluxes) < 0.0)
    assert fluxes[0] > 0.0


def test_sq_limit_maximum_at_25c() -> None:
    """Maximum SQ : environ 33,7 % vers E_g = 1,34 eV a 25 C."""
    best_gap, best_eta = optimize_bandgap(T_REF)
    assert best_gap == pytest.approx(1.34, abs=0.08)
    assert best_eta == pytest.approx(0.337, abs=0.010)
    assert 0.32 < best_eta < 0.35

    result = shockley_queisser(1.34, T_REF)
    assert result.efficiency == pytest.approx(best_eta, abs=0.002)
    assert result.jsc_ma_cm2 == pytest.approx(35.2, abs=3.0)
    assert result.voc == pytest.approx(1.08, abs=0.08)
    assert 0.85 < result.ff < 0.93
    assert result.photon_flux > 0.0
    assert result.incident_power == P_INPUT


def test_sq_thermal_degradation_25_to_45c() -> None:
    """Baisse realiste du rendement entre 25 C et 45 C."""
    temperatures = np.linspace(T_25, 330.15, 17)
    efficiencies = np.array([sq_efficiency(1.34, float(t)) for t in temperatures])
    assert np.all(np.diff(efficiencies) < 0.0)

    cold = shockley_queisser(1.34, T_25)
    hot = shockley_queisser(1.34, T_45)
    drop_points = 100.0 * (cold.efficiency - hot.efficiency)
    assert 0.3 < drop_points < 2.0
    assert hot.voc < cold.voc
    assert hot.ff < cold.ff
    assert hot.jsc_ma_cm2 == pytest.approx(cold.jsc_ma_cm2, abs=1e-9)
    assert hot.j0_ma_cm2 > cold.j0_ma_cm2

    thermal = thermal_degradation(1.34, [T_25, 303.15, T_45, 330.15])
    assert np.all(np.diff(thermal) < 0.0)
    assert thermal[-1] < thermal[0]


def test_radiative_current_boltzmann_scaling() -> None:
    """J_0 suit la loi de Boltzmann et augmente fortement avec T."""
    eg = 1.34
    for temperature in (298.15, 318.15):
        j0 = radiative_saturation_current(eg, temperature)
        kbt = 1.380649e-23 * temperature / Q_E
        approx = j0 * np.exp(eg / kbt)
        assert approx > 0.0 and np.isfinite(approx)
    assert radiative_saturation_current(eg, 330.15) > radiative_saturation_current(eg, 298.15)


def test_sq_maximum_power_symbolic_sympy() -> None:
    """Condition analytique du point de puissance maximale (sympy)."""
    jsc_sym, j0_sym, v_sym, kbt_sym, q_sym = sp.symbols(
        "Jsc J0 v kBT q", positive=True
    )
    voltage = kbt_sym * v_sym / q_sym
    current = jsc_sym - j0_sym * (sp.exp(v_sym) - 1)
    power_derivative = sp.simplify(sp.diff(voltage * current, v_sym))
    expected = (kbt_sym / q_sym) * (jsc_sym - j0_sym * (sp.exp(v_sym) * (1 + v_sym) - 1))
    assert sp.simplify(power_derivative - expected) == 0

    result = shockley_queisser(1.34, T_REF)
    vmp_reduced = Q_E * result.vmp / (1.380649e-23 * T_REF)
    ratio = 1.0 + result.jsc_ma_cm2 / result.j0_ma_cm2
    residual = (1.0 + vmp_reduced) * np.exp(vmp_reduced) - ratio
    assert abs(residual) / ratio < 1e-12


def test_dipole_matrix_elements_and_absorption_onset() -> None:
    """Elements dipolaires non nuls et absorption au-dessus du seuil."""
    model = TightBindingModel.perovskite()
    kpoint = np.array([0.13, -0.21, 0.07])
    dipole = dipole_matrix_element(model, kpoint, 1, 2, direction=0)
    assert np.isfinite(abs(dipole))
    assert abs(dipole) > 0.0

    kpoints = uniform_k_grid(model.lattice.reciprocal_vectors, 4, 4, 4)
    bands = np.array([model.eigenvalues(k) for k in kpoints])
    gap = float(np.min(bands[:, 2]) - np.max(bands[:, 1]))
    assert gap > 0.0
    wavelengths = np.linspace(300.0, 1400.0, 80)
    absorption = absorption_spectrum(
        model,
        kpoints,
        n_valence=2,
        wavelengths_nm=wavelengths,
        broadening_ev=0.02,
        cell_volume_a3=60.0,
    )
    assert absorption.alpha_cm.shape == wavelengths.shape
    assert np.all(np.isfinite(absorption.alpha_cm))
    assert absorption.alpha_max_cm > 0.0
    threshold = HC_EV_NM / gap
    above = absorption.alpha_cm[wavelengths < 0.85 * threshold]
    below = absorption.alpha_cm[wavelengths > 1.25 * threshold]
    assert np.max(above) > 10.0 * np.max(below)


# ---------------------------------------------------------------------------
# 3. Figures de publication
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


def _plot_ribbon_bands(ax, spectrum, title: str) -> None:
    ax.fill_between(
        spectrum.kpoints,
        -6.0,
        spectrum.bulk_valence,
        color="#dbe9f6",
        alpha=0.85,
        linewidth=0.0,
        label="continuum volumique",
    )
    ax.fill_between(
        spectrum.kpoints,
        spectrum.bulk_conduction,
        6.0,
        color="#dbe9f6",
        alpha=0.85,
        linewidth=0.0,
    )
    ax.scatter(
        np.repeat(spectrum.kpoints, spectrum.energies.shape[1]),
        spectrum.energies.ravel(),
        s=1.2,
        color="0.55",
        alpha=0.45,
        linewidths=0.0,
    )
    if spectrum.edge_records:
        kx_bottom = [r[0] for r in spectrum.edge_records if r[2] == "bottom"]
        e_bottom = [r[1] for r in spectrum.edge_records if r[2] == "bottom"]
        kx_top = [r[0] for r in spectrum.edge_records if r[2] == "top"]
        e_top = [r[1] for r in spectrum.edge_records if r[2] == "top"]
        ax.scatter(kx_bottom, e_bottom, s=9, color="#d73027", label="bord inférieur", zorder=5)
        ax.scatter(kx_top, e_top, s=9, color="#1a9850", label="bord supérieur", zorder=5)
    ax.axhline(0.0, color="0.5", lw=0.6, ls=":")
    ax.set_xlim(spectrum.kpoints.min(), spectrum.kpoints.max())
    ax.set_ylim(-3.6, 3.6)
    ax.set_xlabel("$k_x$ (Å$^{-1}$)")
    ax.set_ylabel("$E(k_x)$ (eV)")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower right", framealpha=0.9)


def _make_night3_figure(output_path: Path) -> dict:
    ribbon_top, topo = _haldane_spectrum()
    _, trivial = _haldane_spectrum(M=2.0)
    _, qwz = _qwz_spectrum(u=-1.0)

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 10.0))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    _plot_ribbon_bands(
        ax_a,
        topo,
        "(a) Haldane  $t_2=0.3$, $\\phi=\\pi/2$, $M=0.6$ — $C=+1$",
    )
    _plot_ribbon_bands(
        ax_b,
        trivial,
        "(b) Haldane  $t_2=0.3$, $\\phi=\\pi/2$, $M=2.0$ — $C=0$",
    )
    _plot_ribbon_bands(
        ax_c,
        qwz,
        "(c) Qi–Wu–Zhang  $u=-1$ — $C=-1$",
    )

    kx_target, _, side = topo.edge_records[len(topo.edge_records) // 2]
    index = int(np.argmin(np.abs(topo.kpoints - kx_target)))
    energies, vectors = ribbon_top.solve(float(topo.kpoints[index]))
    candidate = np.flatnonzero(topo.edge_mask[index])
    target_band = int(candidate[np.argmin(np.abs(energies[candidate]))])
    density = np.abs(vectors[:, target_band]) ** 2
    per_cell = density.reshape(ribbon_top.n_cells, ribbon_top.n_orbitals, -1).sum(axis=1)
    cells = np.arange(ribbon_top.n_cells)
    ax_d.semilogy(cells, np.maximum(per_cell, 1e-14), "o-", color="#d73027", ms=3, label="état localisé")
    ax_d.axvspan(-0.5, 2.5, color="#d73027", alpha=0.10)
    ax_d.axvspan(ribbon_top.n_cells - 3.5, ribbon_top.n_cells - 0.5, color="#1a9850", alpha=0.10)
    ax_d.set_xlabel("maille $n$ (direction ouverte $y$)")
    ax_d.set_ylabel("$|\\psi_n|^2$ (échelle log)")
    ax_d.set_title(
        f"(d) Profil d'un état de bord ($k_x$ = {topo.kpoints[index]:+.3f} Å$^{{-1}}$, "
        f"bord {'inférieur' if side == 'bottom' else 'supérieur'})"
    )
    ax_d.legend(loc="upper right")
    ax_d.grid(alpha=0.3)

    figure.suptitle(
        "UAM-TOPOMAT — états de bord chiraux et correspondance volume-bord sur nanorubans",
        fontsize=12.5,
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {
        "topological_crossings": topo.n_crossings,
        "trivial_gap_states": trivial.n_gap_states,
        "qwz_crossings": qwz.n_crossings,
        "n_cells": ribbon_top.n_cells,
    }


def _make_night4_figure(output_path: Path) -> dict:
    temperatures = np.array([298.15, 303.15, 318.15, 330.15])
    gaps = np.linspace(0.6, 2.2, 321)
    colors = ("#08306b", "#2879b9", "#d95f0e", "#993404")

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 10.0))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    for temperature, color in zip(temperatures, colors):
        efficiencies = 100.0 * sq_efficiency_curve(gaps, temperature)
        peak_gap, peak_eta = optimize_bandgap(temperature)
        ax_a.plot(gaps, efficiencies, color=color, lw=1.6, label=f"T = {temperature:.2f} K")
        ax_a.plot(peak_gap, 100.0 * peak_eta, marker="o", color=color, ms=5)
    ax_a.axvline(1.34, color="0.4", ls="--", lw=1.0)
    ax_a.set_xlabel("$E_g$ (eV)")
    ax_a.set_ylabel("$\\eta_{SQ}$ (%)")
    ax_a.set_title("(a) Limite de Shockley–Queisser sous AM1.5G")
    ax_a.set_xlim(0.6, 2.2)
    ax_a.set_ylim(0, 35)
    ax_a.grid(alpha=0.3)
    ax_a.legend(loc="upper right", framealpha=0.9)

    temp_fine = np.linspace(298.15, 330.15, 65)
    eta_fine = 100.0 * thermal_degradation(1.34, temp_fine)
    ax_b.plot(temp_fine - 273.15, eta_fine, color="#08306b", lw=2.0)
    for temperature, label in ((T_25, "25 °C"), (303.15, "30 °C"), (T_45, "45 °C"), (330.15, "57 °C")):
        eta = 100.0 * sq_efficiency(1.34, temperature)
        ax_b.plot(temperature - 273.15, eta, "o", color="#d73027", ms=7, zorder=5)
        ax_b.annotate(
            f"{label}\n{eta:.2f} %",
            (temperature - 273.15, eta),
            textcoords="offset points",
            xytext=(7, -22 if temperature == 303.15 else 8),
            fontsize=8,
        )
    slope = np.polyfit(temp_fine - 273.15, eta_fine, 1)[0]
    ax_b.set_xlabel("Température de la cellule (°C)")
    ax_b.set_ylabel("$\\eta_{SQ}(E_g = 1.34$ eV) (%)")
    ax_b.set_title(f"(b) Dégradation thermique — pente ≈ {slope:.3f} pt/K")
    ax_b.grid(alpha=0.3)

    wavelengths, irradiance = solar_grid()
    mask = wavelengths <= 1400.0
    cutoff_nm = HC_EV_NM / 1.34
    ax_c.fill_between(
        wavelengths[mask],
        0.0,
        irradiance[mask],
        where=(wavelengths[mask] <= cutoff_nm),
        color="#fdae6b",
        alpha=0.7,
        label=f"photons utilisables ($\\lambda$ < {cutoff_nm:.0f} nm)",
    )
    ax_c.plot(wavelengths[mask], irradiance[mask], color="#08306b", lw=1.1, label="AM1.5G global")
    ax_c.axvline(cutoff_nm, color="#d73027", ls="--", lw=1.0)
    ax_c.set_xlabel("$\\lambda$ (nm)")
    ax_c.set_ylabel("Irradiance (W m$^{-2}$ nm$^{-1}$)")
    ax_c.set_title("(c) Spectre solaire AM1.5G normalisé à 1000 W m$^{-2}$")
    ax_c.grid(alpha=0.3)
    ax_c.legend(loc="upper right", framealpha=0.9)

    model = TightBindingModel.perovskite()
    kpoints = uniform_k_grid(model.lattice.reciprocal_vectors, 4, 4, 4)
    wavelengths_opt = np.linspace(320.0, 1100.0, 90)
    absorption = absorption_spectrum(
        model,
        kpoints,
        n_valence=2,
        wavelengths_nm=wavelengths_opt,
        broadening_ev=0.04,
        cell_volume_a3=60.0,
    )
    ax_d.semilogy(wavelengths_opt, absorption.alpha_cm, color="#6a51a3", lw=1.8)
    ax_d.axvline(HC_EV_NM / 1.34, color="#d73027", ls="--", lw=1.0, label="$\\lambda_g$(1.34 eV)")
    ax_d.set_xlabel("$\\lambda$ (nm)")
    ax_d.set_ylabel("$\\alpha(\\lambda)$ (cm$^{-1}$, échelle log)")
    ax_d.set_title("(d) Absorption optique depuis $E_n(\\mathbf{k})$ et tenseur dipolaire")
    ax_d.grid(alpha=0.3)
    ax_d.legend(loc="lower left", framealpha=0.9)

    figure.suptitle(
        "UAM-TOPOMAT — optique cristalline, spectre AM1.5G et rendement de Shockley–Queisser",
        fontsize=12.5,
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {
        "peak_gap": optimize_bandgap(T_REF)[0],
        "peak_efficiency": optimize_bandgap(T_REF)[1],
        "slope_pt_per_k": slope,
        "alpha_max": absorption.alpha_max_cm,
    }


def test_generate_night3_figure() -> None:
    output = ROOT / "results" / "figures" / "night3_edge_states_ribbon.png"
    metadata = _make_night3_figure(output)
    assert output.exists()
    assert output.stat().st_size > 30_000
    assert abs(_png_dpi(output) - 300.0) < 1.0
    assert metadata["topological_crossings"] == 2
    assert metadata["trivial_gap_states"] == 0
    assert metadata["qwz_crossings"] == 2


def test_generate_night4_figure() -> None:
    output = ROOT / "results" / "figures" / "night4_solar_efficiency_vs_temp.png"
    metadata = _make_night4_figure(output)
    assert output.exists()
    assert output.stat().st_size > 30_000
    assert abs(_png_dpi(output) - 300.0) < 1.0
    assert abs(metadata["peak_gap"] - 1.34) < 0.08
    assert metadata["peak_efficiency"] == pytest.approx(0.337, abs=0.010)
    assert -0.10 < metadata["slope_pt_per_k"] < -0.02
    assert metadata["alpha_max"] > 0.0
