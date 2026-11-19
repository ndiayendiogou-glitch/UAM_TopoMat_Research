"""UAM-TOPOMAT -- Validation de la Nuit 2.

Tests numeriques et symboliques du moteur de jauge sur reseau (FHS), de la
quantification du nombre de Chern et de la fermeture du gap du modele de
Haldane, puis generation de la figure de publication (300 dpi).
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
from matplotlib.colors import BoundaryNorm, ListedColormap, SymLogNorm

from src.topology.berry_chern import (
    berry_chern,
    berry_curvature,
    chern_number,
    evaluate_eigenstates,
    extract_band,
    gauge_links,
    k_grid,
    plaquette_field,
    random_gauge_transform,
)
from src.topology.models_topo import (
    HaldaneModel,
    haldane_chern_analytic,
    haldane_critical_mass,
    haldane_dirac_mass,
    haldane_dirac_points,
    haldane_d_vector,
    nearest_neighbour_vectors,
    next_neighbour_vectors,
)

TWO_PI = 2.0 * np.pi
T1, T2, PHI, MASS = 1.0, 0.3, np.pi / 2.0, 0.6


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
# 1. Liens de jauge et champ de plaquette
# ---------------------------------------------------------------------------


def test_gauge_links_modulus_and_periodicity() -> None:
    model = HaldaneModel(t1=T1, t2=T2, phi=PHI, M=MASS)
    _, eigenvectors, _, _ = evaluate_eigenstates(
        model, model.reciprocal_vectors, 12, 14
    )
    u1 = gauge_links(eigenvectors, 0)
    u2 = gauge_links(eigenvectors, 1)

    assert u1.shape == (12, 14) and u2.shape == (12, 14)
    assert np.allclose(np.abs(u1), 1.0, atol=1e-14)
    assert np.allclose(np.abs(u2), 1.0, atol=1e-14)
    assert np.all(np.isfinite(u1)) and np.all(np.isfinite(u2))

    psi = extract_band(eigenvectors, 0)
    shifted = np.roll(psi, -1, axis=0)
    expected = np.einsum("ijk,ijk->ij", psi.conj(), shifted)
    expected /= np.abs(expected)
    assert np.allclose(u1, expected, atol=1e-13)


def test_plaquette_field_is_real_and_bounded() -> None:
    model = HaldaneModel(t1=T1, t2=T2, phi=PHI, M=MASS)
    result = berry_chern(model, model.reciprocal_vectors, 24, 24)
    assert np.isrealobj(result.field)
    assert np.all(np.abs(result.field) <= np.pi + 1e-12)
    assert result.field.shape == (24, 24)
    assert np.isclose(chern_number(result.field), result.chern, atol=1e-14)


# ---------------------------------------------------------------------------
# 2. Invariance de jauge exacte (< 1e-12)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("band", [0, 1])
def test_berry_curvature_gauge_invariance_exact(band: int) -> None:
    """F_12 est invariant sous psi_n(k) -> exp(i theta(k)) psi_n(k)."""
    model = HaldaneModel(t1=T1, t2=T2, phi=PHI, M=MASS)
    _, eigenvectors, _, _ = evaluate_eigenstates(
        model, model.reciprocal_vectors, 28, 28
    )

    original = plaquette_field(
        gauge_links(eigenvectors, 0, band),
        gauge_links(eigenvectors, 1, band),
    )
    rng = np.random.default_rng(2026 + band)
    transformed = random_gauge_transform(eigenvectors, rng)
    gauged = plaquette_field(
        gauge_links(transformed, 0, band),
        gauge_links(transformed, 1, band),
    )

    assert np.max(np.abs(original - gauged)) < 1e-12
    assert np.isclose(chern_number(original), chern_number(gauged), atol=1e-12)


def test_gauge_invariance_repeated_random_phases() -> None:
    """Dix transformations de jauge aleatoires successives, erreur < 1e-12."""
    model = HaldaneModel(t1=T1, t2=T2, phi=PHI, M=MASS)
    _, eigenvectors, _, _ = evaluate_eigenstates(
        model, model.reciprocal_vectors, 20, 20
    )
    reference = plaquette_field(
        gauge_links(eigenvectors, 0), gauge_links(eigenvectors, 1)
    )
    rng = np.random.default_rng(31415)
    worst = 0.0
    for _ in range(10):
        gauged_psi = random_gauge_transform(eigenvectors, rng)
        trial = plaquette_field(
            gauge_links(gauged_psi, 0), gauge_links(gauged_psi, 1)
        )
        worst = max(worst, float(np.max(np.abs(trial - reference))))
    assert worst < 1e-12


def test_curvature_scaling_and_chern_grid_convention() -> None:
    """Omega = F/dA et C = (1/2pi) sum F Independent de la maille."""
    model = HaldaneModel(t1=T1, t2=T2, phi=PHI, M=MASS)
    result = berry_chern(model, model.reciprocal_vectors, 36, 36)
    expected = berry_curvature(result.field, model.reciprocal_vectors, 36, 36)
    assert np.allclose(result.curvature, expected, atol=1e-14)
    assert np.isclose(
        float(result.curvature.sum() * result.metadata["cell_area"]),
        TWO_PI * result.chern,
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# 3. Quantification de Chern dans le modele de Haldane
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phi,m_over_t2,expected",
    [
        (np.pi / 2.0, 0.0, 1),      # M/(3sqrt3 t2 sinphi) = 0 -> C = +1
        (np.pi / 2.0, 2.0, 1),      # ratio ~ 0.385 < 1 -> C = +1
        (np.pi / 2.0, -2.0, 1),
        (-np.pi / 2.0, 0.0, -1),    # sinphi < 0 -> C = -1
        (-np.pi / 2.0, 3.0, -1),
        (np.pi / 4.0, 1.0, 1),      # 3sqrt3 sin(pi/4) ~ 3.674
        (np.pi / 2.0, 7.0, 0),      # ratio > 1 -> isolant trivial
        (np.pi / 2.0, -7.0, 0),
        (-np.pi / 2.0, 7.0, 0),
        (0.0, 1.0, 0),              # sinphi = 0 -> trivial
    ],
)
def test_haldane_chern_quantization(phi: float, m_over_t2: float, expected: int) -> None:
    model = HaldaneModel(t1=T1, t2=T2, phi=phi, M=m_over_t2 * T2)
    result = berry_chern(model, model.reciprocal_vectors, 48, 48, band=0)

    assert abs(result.chern - round(result.chern)) < 1e-9
    assert result.chern_integer == expected
    assert result.chern_integer == haldane_chern_analytic(T2, phi, model.M)
    assert berry_chern(
        model, model.reciprocal_vectors, 48, 48, band=1
    ).chern_integer == -expected


def test_chern_stable_under_grid_refinement() -> None:
    model = HaldaneModel(t1=T1, t2=T2, phi=PHI, M=MASS)
    values = []
    for size in (24, 40, 64):
        result = berry_chern(model, model.reciprocal_vectors, size, size)
        values.append(result.chern_integer)
        assert abs(result.chern - result.chern_integer) < 1e-9
    assert values == [1, 1, 1]


def test_critical_mass_and_ratio_criterion() -> None:
    """La frontiere |M| = 3sqrt3 |t2 sinphi| separe C = 0 du cours C = ±1."""
    critical = haldane_critical_mass(T2, PHI)
    assert np.isclose(critical, 3.0 * np.sqrt(3.0) * T2, atol=1e-14)
    for factor in (0.0, 0.5, 0.99):
        inside = factor * critical
        assert haldane_chern_analytic(T2, PHI, inside) == 1
        assert haldane_chern_analytic(T2, PHI, -inside) == 1
    for factor in (1.01, 1.5, 3.0):
        outside = factor * critical
        assert haldane_chern_analytic(T2, PHI, outside) == 0
        assert haldane_chern_analytic(T2, PHI, -outside) == 0


# ---------------------------------------------------------------------------
# 4. Fermeture du gap aux points de Dirac (numerique + symbolique)
# ---------------------------------------------------------------------------


def test_haldane_dirac_points_and_masses() -> None:
    model = HaldaneModel(t1=T1, t2=T2, phi=PHI, M=MASS)
    points = haldane_dirac_points(model.a)
    assert np.isclose(np.linalg.norm(points["K"]), 4.0 * np.pi / (3.0 * np.sqrt(3.0)))
    assert np.allclose(points["Kprime"], -points["K"])

    for name, k in points.items():
        d0, dx, dy, dz = haldane_d_vector(
            np.array([k[0], k[1], 0.0]), T1, T2, PHI, MASS, model.a
        )
        assert np.isclose(dx, 0.0, atol=1e-14)
        assert np.isclose(dy, 0.0, atol=1e-14)
        assert np.isclose(dz, haldane_dirac_mass(name, T2, PHI, MASS), atol=1e-14)
        assert np.isclose(model.gap([k[0], k[1], 0.0]), 2.0 * abs(dz), atol=1e-12)


@pytest.mark.parametrize("phi", [np.pi / 2.0, -np.pi / 2.0, np.pi / 3.0, 2.3])
def test_gap_closing_exactly_at_topological_boundary_numerical(phi: float) -> None:
    model = HaldaneModel(t1=T1, t2=T2, phi=phi, M=0.0)
    critical = haldane_critical_mass(T2, phi)
    points = haldane_dirac_points(model.a)
    k_k = [points["K"][0], points["K"][1], 0.0]
    k_kp = [points["Kprime"][0], points["Kprime"][1], 0.0]

    model.M = -critical
    assert model.gap(k_k) < 1e-12
    assert model.gap(k_kp) > 1e-3

    model.M = critical
    assert model.gap(k_kp) < 1e-12
    assert model.gap(k_k) > 1e-3

    delta = 0.01
    model.M = critical + delta
    assert np.isclose(model.gap(k_kp), 2.0 * delta, atol=1e-10)
    model.M = critical - delta
    assert np.isclose(model.gap(k_kp), 2.0 * delta, atol=1e-10)


def test_gap_closing_symbolic_sympy() -> None:
    """Verification symbolique exacte de la masse aux points K et K'."""
    t1, t2, phi, M, a = sp.symbols("t1 t2 phi M a", positive=True, real=True)
    root3 = sp.sqrt(3)
    k0 = 4 * sp.pi / (3 * root3 * a)

    deltas = (
        (sp.Integer(0), a),
        (root3 * a / 2, -a / 2),
        (-root3 * a / 2, -a / 2),
    )
    nus = (
        (-root3 * a / 2, 3 * a / 2),
        (root3 * a, sp.Integer(0)),
        (-root3 * a / 2, -3 * a / 2),
    )

    def f_value(kx, ky):
        return sum(sp.exp(-sp.I * (kx * dx + ky * dy)) for dx, dy in deltas)

    def g_value(kx, ky):
        return sum(sp.sin(kx * nx + ky * ny) for nx, ny in nus)

    simplify = lambda expr: sp.simplify(sp.expand_complex(expr))

    # f(K) = 0 : les deux sous-reseaux sont degeneres sans masse.
    assert simplify(f_value(k0, 0)) == 0
    assert simplify(f_value(-k0, 0)) == 0

    g_k = simplify(g_value(k0, 0))
    g_kprime = simplify(g_value(-k0, 0))
    assert g_k == -3 * root3 / 2
    assert g_kprime == 3 * root3 / 2

    dz_k = simplify(M - 2 * t2 * sp.sin(phi) * g_k)
    dz_kprime = simplify(M - 2 * t2 * sp.sin(phi) * g_kprime)
    assert dz_k == M + 3 * root3 * t2 * sp.sin(phi)
    assert dz_kprime == M - 3 * root3 * t2 * sp.sin(phi)

    # E_pm = ±|d_z| en l'absence de masse de Haldane aux points de Dirac.
    eigen_k = sp.solve(sp.Eq(dz_k, 0), M)
    eigen_kprime = sp.solve(sp.Eq(dz_kprime, 0), M)
    assert eigen_k == [-3 * root3 * t2 * sp.sin(phi)]
    assert eigen_kprime == [3 * root3 * t2 * sp.sin(phi)]
    assert sp.simplify(eigen_kprime[0] - 3 * root3 * t2 * sp.sin(phi)) == 0


# ---------------------------------------------------------------------------
# 5. Figure de publication
# ---------------------------------------------------------------------------


def _make_night2_figure(output_path: Path) -> dict:
    model = HaldaneModel(t1=T1, t2=T2, phi=PHI, M=MASS)
    n = 130
    result = berry_chern(model, model.reciprocal_vectors, n, n, band=0)
    curvature = result.curvature

    _, vertices_xy = k_grid(model.reciprocal_vectors, n + 1, n + 1, shift=0.0)
    b1, b2 = model.reciprocal_vectors[:2, :2]
    offset = 0.5 * (b1 + b2)
    vx = vertices_xy[:, :, 0] - offset[0]
    vy = vertices_xy[:, :, 1] - offset[1]

    def fold_into_bz(point: np.ndarray) -> np.ndarray:
        fractions = np.linalg.solve(np.column_stack((b1, b2)), point)
        return (np.mod(fractions, 1.0) @ np.vstack((b1, b2))) - offset

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.linewidth": 0.8,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )
    figure, (ax_berry, ax_phase) = plt.subplots(
        1, 2, figsize=(13.0, 5.6), gridspec_kw={"width_ratios": (1.0, 1.05), "wspace": 0.28}
    )

    # --- Panneau (a) : courbure de Berry dans la ZB -----------------------
    limit = float(np.max(np.abs(curvature)))
    mesh = ax_berry.pcolormesh(
        vx,
        vy,
        curvature,
        cmap="RdBu_r",
        norm=SymLogNorm(linthresh=0.05, vmin=-limit, vmax=limit, base=10.0),
        shading="flat",
    )
    corners = np.array(
        [
            [vx[0, 0], vy[0, 0]],
            [vx[-1, 0], vy[-1, 0]],
            [vx[-1, -1], vy[-1, -1]],
            [vx[0, -1], vy[0, -1]],
            [vx[0, 0], vy[0, 0]],
        ]
    )
    ax_berry.plot(corners[:, 0], corners[:, 1], color="black", lw=1.0)

    dirac = model.dirac_points()
    for label, point, color in (
        ("$K$", dirac["K"], "#08306b"),
        ("$K'$", dirac["Kprime"], "white"),
    ):
        folded = fold_into_bz(point)
        ax_berry.plot(
            folded[0], folded[1], marker="*", ms=12, color=color, mec="black", mew=0.5
        )
        ax_berry.annotate(
            label, folded, textcoords="offset points", xytext=(6, 6), fontsize=11
        )
    ax_berry.plot(0.0, 0.0, marker="o", ms=4, color="black")
    ax_berry.annotate(
        "$\\Gamma$", (0.0, 0.0), textcoords="offset points", xytext=(6, -12), fontsize=10
    )

    ax_berry.set_aspect("equal")
    ax_berry.set_xlabel("$k_x$ (Å$^{-1}$)")
    ax_berry.set_ylabel("$k_y$ (Å$^{-1}$)")
    ax_berry.set_title(
        "(a) Courbure de Berry $\\Omega(\\mathbf{k})$ — phase $C=+1$", fontsize=11
    )
    colorbar = figure.colorbar(mesh, ax=ax_berry, pad=0.02, fraction=0.046)
    colorbar.set_label("$\\Omega(\\mathbf{k})$ (Å$^2$, échelle symlog)", fontsize=10)

    # --- Panneau (b) : diagramme de phase (phi, M/t2) ---------------------
    phi_values = np.linspace(-np.pi, np.pi, 321)
    mass_values = np.linspace(-8.0, 8.0, 321)
    phi_mesh, mass_mesh = np.meshgrid(phi_values, mass_values, indexing="xy")
    chern_mesh = np.vectorize(
        lambda p, m: haldane_chern_analytic(T2, float(p), float(m) * T2)
    )(phi_mesh, mass_mesh)

    colors = ("#f7f7f7", "#4575b4", "#d73027")
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    phase_mesh = ax_phase.pcolormesh(
        phi_mesh, mass_mesh, chern_mesh, cmap=cmap, norm=norm, shading="auto"
    )
    phi_curve = np.linspace(-np.pi, np.pi, 600)
    ax_phase.plot(phi_curve, 3.0 * np.sqrt(3.0) * np.sin(phi_curve), "k--", lw=1.0)
    ax_phase.plot(phi_curve, -3.0 * np.sqrt(3.0) * np.sin(phi_curve), "k--", lw=1.0)
    ax_phase.axhline(0.0, color="0.6", lw=0.6)

    markers = (
        (np.pi / 2.0, MASS / T2, 1, "$C=+1$"),
        (-np.pi / 2.0, MASS / T2, -1, "$C=-1$"),
        (np.pi / 2.0, 7.0, 0, "$C=0$"),
    )
    for marker_phi, marker_m, expected, text in markers:
        ax_phase.plot(
            marker_phi,
            marker_m,
            marker="*",
            ms=13,
            color="black",
            mec="white",
            mew=0.6,
        )
        ax_phase.annotate(
            text,
            (marker_phi, marker_m),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.8},
        )

    ax_phase.set_xlim(-np.pi, np.pi)
    ax_phase.set_ylim(-8.0, 8.0)
    ax_phase.set_xticks(
        [-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi],
        ["$-\\pi$", "$-\\pi/2$", "$0$", "$\\pi/2$", "$\\pi$"],
    )
    ax_phase.set_xlabel("$\\phi$ (rad)")
    ax_phase.set_ylabel("$M/t_2$")
    ax_phase.set_title("(b) Diagramme de phase topologique de Haldane", fontsize=11)
    phase_bar = figure.colorbar(phase_mesh, ax=ax_phase, ticks=[-1, 0, 1], pad=0.02, fraction=0.046)
    phase_bar.ax.set_yticklabels(["$-1$", "$0$", "$+1$"])
    phase_bar.set_label("Nombre de Chern $C$", fontsize=10)
    ax_phase.text(
        0.03,
        0.04,
        "Frontières : $|M/t_2| = 3\\sqrt{3}\\,|\\sin\\phi|$",
        transform=ax_phase.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.7", "alpha": 0.9},
    )

    figure.suptitle(
        "UAM-TOPOMAT — courbure de Berry, invariant de Chern et phase de Haldane",
        fontsize=12.5,
        y=0.995,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    return {
        "chern": result.chern_integer,
        "n_grid": n,
        "curvature_max": limit,
        "field_range": (
            float(result.field.min()),
            float(result.field.max()),
        ),
    }


def test_generate_night2_figure() -> None:
    output = ROOT / "results" / "figures" / "night2_haldane_chern_phase.png"
    metadata = _make_night2_figure(output)

    assert output.exists()
    assert output.stat().st_size > 30_000
    assert abs(_png_dpi(output) - 300.0) < 1.0
    assert metadata["chern"] == 1
    assert metadata["n_grid"] == 130
    assert metadata["curvature_max"] > 0.0
