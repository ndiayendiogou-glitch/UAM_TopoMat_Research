"""UAM-TOPOMAT -- Validation de la Nuit 1.

Tests numeriques et symboliques du moteur cristallographique et du solveur
Tight-Binding, puis generation de la figure de publication (300 dpi).
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

from src.quantum.band_structure import (
    TightBindingModel,
    uniform_k_grid,
)
from src.quantum.lattice import (
    LATTICE_FACTORIES,
    graphene_lattice,
    hexagonal_lattice,
    mxene_lattice,
    perovskite_lattice,
    square_lattice,
)

TWO_PI = 2.0 * np.pi
T_GRAPHENE = 2.7
D_GRAPHENE = 1.42
HBAR_SI = 1.054571817e-34
EV_ANGSTROM_SI = 1.602176634e-19 * 1e-10


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


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("square", {"a": 1.0}),
        ("hexagonal", {"a": 2.46}),
        ("graphene", {"d": 1.42}),
        ("perovskite", {"a": 3.9}),
        ("mxene", {"a": 3.0, "c": 4.6}),
    ],
)
def test_reciprocal_orthogonality(name: str, kwargs: dict) -> None:
    lattice = LATTICE_FACTORIES[name](**kwargs)
    product = lattice.reciprocal_vectors @ lattice.direct_vectors.T
    assert np.allclose(product, TWO_PI * np.eye(3), atol=1e-10)
    assert lattice.volume > 0.0


@pytest.mark.parametrize(
    "name,expected_sites",
    [
        ("square", 1),
        ("hexagonal", 2),
        ("graphene", 2),
        ("perovskite", 5),
        ("mxene", 3),
    ],
)
def test_lattice_families(name: str, expected_sites: int) -> None:
    lattice = LATTICE_FACTORIES[name]()
    assert lattice.n_sites == expected_sites
    assert len(lattice.labels) == expected_sites
    assert "Gamma" in lattice.symmetry_points
    assert len(lattice.path) >= 2


def test_high_symmetry_square_and_hexagonal() -> None:
    square = square_lattice(a=2.0)
    points = square.high_symmetry_cartesian()
    assert np.allclose(points["X"][0], [np.pi / 2.0, 0.0, 0.0])
    assert np.allclose(points["M"][0], [np.pi / 2.0, np.pi / 2.0, 0.0])

    lattice_a = 2.46
    hexagonal = hexagonal_lattice(a=lattice_a)
    gamma = hexagonal.high_symmetry_cartesian()["Gamma"][0]
    k_point = hexagonal.high_symmetry_cartesian()["K"][0]
    assert np.allclose(gamma, 0.0)
    assert np.isclose(
        np.linalg.norm(k_point), 4.0 * np.pi / (3.0 * lattice_a), atol=1e-12
    )

    graphene = graphene_lattice(d=D_GRAPHENE)
    k_graphene = graphene.high_symmetry_cartesian()["K"][0]
    assert np.isclose(
        np.linalg.norm(k_graphene),
        4.0 * np.pi / (3.0 * np.sqrt(3.0) * D_GRAPHENE),
        atol=1e-12,
    )


def test_k_path_geometry_and_continuity() -> None:
    lattice = graphene_lattice(d=D_GRAPHENE)
    n_per_segment = 41
    path = lattice.generate_k_path(n_per_segment=n_per_segment)
    kpoints = path["kpoints"]
    distances = path["distances"]
    ticks = path["ticks"]

    expected_points = 1 + (len(lattice.path) - 1) * (n_per_segment - 1)
    assert kpoints.shape == (expected_points, 3)
    assert np.all(np.diff(distances) >= 0.0)
    assert np.isclose(ticks[0], 0.0)
    assert np.isclose(ticks[-1], distances[-1])
    assert np.allclose(kpoints[0], 0.0)
    assert np.allclose(kpoints[-1], 0.0)

    k_index = int(np.argmin(np.abs(ticks[2] - distances)))
    assert np.allclose(
        kpoints[k_index], lattice.high_symmetry_cartesian()["K"][0], atol=1e-12
    )


def test_graphene_hamiltonian_and_eigenvectors() -> None:
    lattice = graphene_lattice(d=D_GRAPHENE)
    model = TightBindingModel.graphene(t=T_GRAPHENE, a=D_GRAPHENE)
    rng = np.random.default_rng(2026)
    k = rng.uniform(-np.pi, np.pi, size=3)
    hamiltonian = model.hamiltonian(k)

    assert np.allclose(hamiltonian, hamiltonian.conj().T, atol=1e-14)
    values, vectors = model.eigen_system(k)
    assert np.allclose(hamiltonian @ vectors, vectors * values, atol=1e-12)
    assert np.isclose(np.trace(hamiltonian), 0.0, atol=1e-14)
    assert np.all(np.diff(values) >= 0.0)

    path = lattice.generate_k_path(n_per_segment=25)
    bands = model.band_structure(path["kpoints"])
    assert bands.shape == (len(path["kpoints"]), 2)
    assert np.all(np.isfinite(bands))


def test_graphene_dirac_cone_numerical() -> None:
    lattice = graphene_lattice(d=D_GRAPHENE)
    model = TightBindingModel.graphene(t=T_GRAPHENE, a=D_GRAPHENE)
    k_dirac = lattice.high_symmetry_cartesian()["K"][0]
    velocity = 3.0 * T_GRAPHENE * D_GRAPHENE / (2.0 * model.hbar)

    q_norm = 1e-7
    for angle in np.linspace(0.0, TWO_PI, 13)[:-1]:
        q = q_norm * np.array([np.cos(angle), np.sin(angle), 0.0])
        energies = model.eigenvalues(k_dirac + q)
        expected = velocity * q_norm
        assert np.allclose(energies, [-expected, expected], rtol=1e-6, atol=0.0)


def test_graphene_dirac_cone_symbolic_sympy() -> None:
    t, d = sp.symbols("t d", positive=True)
    qx, qy = sp.symbols("qx qy", real=True)

    def simplify(expr):
        return sp.simplify(sp.expand_complex(expr))

    root3 = sp.sqrt(3)
    kx = 2 * sp.pi / (3 * d) + qx
    ky = 2 * sp.pi / (3 * root3 * d) + qy
    deltas = (
        (d, sp.Integer(0)),
        (-d / 2, -root3 * d / 2),
        (-d / 2, root3 * d / 2),
    )
    f = sum(sp.exp(sp.I * (kx * dx + ky * dy)) for dx, dy in deltas)
    f_conjugate = sum(
        sp.exp(-sp.I * (kx * dx + ky * dy)) for dx, dy in deltas
    )
    squared = simplify(sp.expand(f * f_conjugate))
    at_gamma = {qx: 0, qy: 0}

    assert simplify(squared.subs(at_gamma)) == 0
    assert simplify(sp.diff(squared, qx).subs(at_gamma)) == 0
    assert simplify(sp.diff(squared, qy).subs(at_gamma)) == 0

    hessian_xx = simplify(sp.diff(squared, qx, 2).subs(at_gamma))
    hessian_xy = simplify(sp.diff(squared, qx, qy).subs(at_gamma))
    hessian_yy = simplify(sp.diff(squared, qy, 2).subs(at_gamma))
    assert simplify(hessian_xx - sp.Rational(9, 2) * d**2) == 0
    assert simplify(hessian_xy) == 0
    assert simplify(hessian_yy - sp.Rational(9, 2) * d**2) == 0

    series_f = sp.series(squared.subs(qy, 0), qx, 0, 4).removeO()
    assert simplify(series_f - sp.Rational(9, 4) * d**2 * qx**2) == 0

    quadratic = sp.Rational(9, 4) * d**2 * (qx**2 + qy**2)
    dirac_energy = sp.Rational(3, 2) * t * d * sp.sqrt(qx**2 + qy**2)
    assert simplify(t * sp.sqrt(quadratic) - dirac_energy) == 0


def test_graphene_group_velocity_at_dirac_point() -> None:
    lattice = graphene_lattice(d=D_GRAPHENE)
    model = TightBindingModel.graphene(t=T_GRAPHENE, a=D_GRAPHENE)
    k_dirac = lattice.high_symmetry_cartesian()["K"][0]
    expected = 3.0 * T_GRAPHENE * D_GRAPHENE / (2.0 * model.hbar)

    q_norm = 1e-5
    for angle in (0.0, np.pi / 6.0, np.pi / 3.0, np.pi / 2.0):
        q = q_norm * np.array([np.cos(angle), np.sin(angle), 0.0])
        velocities = model.group_velocities([k_dirac + q])[0]
        speed = float(np.linalg.norm(velocities[1]))
        direction = velocities[1, :2] / np.linalg.norm(velocities[1, :2])
        q_hat = q[:2] / np.linalg.norm(q[:2])
        assert np.isclose(speed, expected, rtol=1e-4)
        assert np.isclose(abs(np.dot(direction, q_hat)), 1.0, atol=1e-3)


def test_graphene_dos_normalisation_and_symmetry() -> None:
    lattice = graphene_lattice(d=D_GRAPHENE)
    model = TightBindingModel.graphene(t=T_GRAPHENE, a=D_GRAPHENE)
    kpoints = uniform_k_grid(lattice.reciprocal_vectors, 90, 90, 1)
    energies = np.linspace(-9.0, 9.0, 1801)
    dos = model.density_of_states(energies, kpoints, broadening=0.15)

    integral = float(np.trapezoid(dos, energies))
    assert np.isclose(integral, 2.0, atol=2e-3)
    assert np.all(dos >= 0.0)
    assert np.allclose(dos, dos[::-1], atol=1e-8)
    assert dos[np.argmin(np.abs(energies))] > 0.0
    assert np.all(np.isfinite(dos))


def test_perovskite_and_mxene_tight_binding_models() -> None:
    perovskite = TightBindingModel.perovskite(a=3.9)
    mxene = TightBindingModel.mxene(a=3.0, c=4.6)

    assert perovskite.band_structure(np.zeros(3)).shape == (1, 5)
    assert mxene.band_structure(np.zeros(3)).shape == (1, 3)

    k_points = np.array([[0.0, 0.0, 0.0], [0.2, 0.1, 0.0], [0.4, 0.3, 0.1]])
    for model in (perovskite, mxene):
        bands = model.band_structure(k_points)
        velocities = model.group_velocities(k_points)
        assert bands.shape[0] == len(k_points)
        assert velocities.shape == (len(k_points), model.n_orbitals, 3)
        assert np.all(np.isfinite(bands))
        assert np.all(np.isfinite(velocities))
        for k in k_points:
            hamiltonian = model.hamiltonian(k)
            assert np.allclose(hamiltonian, hamiltonian.conj().T, atol=1e-14)


def _make_night1_figure(output_path: Path) -> dict:
    t, d = T_GRAPHENE, D_GRAPHENE
    lattice = graphene_lattice(d=d)
    model = TightBindingModel.graphene(t=t, a=d)

    path_data = lattice.generate_k_path(n_per_segment=240)
    kpoints = path_data["kpoints"]
    distances = path_data["distances"]
    ticks = path_data["ticks"]
    labels = path_data["tick_labels"]
    bands = model.band_structure(kpoints)

    tick_labels = []
    for label in labels:
        tick_labels.append("$\\Gamma$" if label == "Gamma" else f"${label}$")

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
    figure, (ax_bands, ax_dos) = plt.subplots(
        1,
        2,
        figsize=(10.5, 5.2),
        sharey=True,
        gridspec_kw={"width_ratios": (2.15, 1.0), "wspace": 0.08},
    )

    colors = ("#1f4e9c", "#b2182b")
    for band, color in zip(range(model.n_orbitals), colors):
        ax_bands.plot(distances, bands[:, band], color=color, lw=1.5)

    ax_bands.axhline(0.0, color="0.35", lw=0.8, ls="--")
    ax_bands.set_xlim(distances[0], distances[-1])
    ax_bands.set_ylim(-8.6, 8.6)
    ax_bands.set_xticks(ticks)
    ax_bands.set_xticklabels(tick_labels)
    ax_bands.set_ylabel("$E - E_F$ (eV)")
    ax_bands.set_xlabel("Chemin dans la zone de Brillouin")
    ax_bands.set_title(
        f"Graphène — bandes Tight-Binding ($t={t:g}$ eV, $d_{{CC}}={d:g}$ Å)",
        fontsize=11,
    )
    ax_bands.text(
        0.035,
        0.06,
        "$\\hbar v_F = %.3f$ eV·Å\n$v_F = %.3f\\times10^5$ m/s"
        % (1.5 * t * d, 1.5 * t * d * EV_ANGSTROM_SI / HBAR_SI / 1e5),
        transform=ax_bands.transAxes,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.7", "alpha": 0.95},
    )

    k_tick = ticks[2]
    inset = ax_bands.inset_axes([0.60, 0.60, 0.36, 0.34])
    window = np.abs(distances - k_tick) <= 0.10
    for band, color in zip(range(model.n_orbitals), colors):
        inset.plot(distances[window], bands[window, band], color=color, lw=1.3)
    inset.axhline(0.0, color="0.5", lw=0.6, ls=":")
    inset.set_xlim(k_tick - 0.10, k_tick + 0.10)
    inset.set_ylim(-1.1, 1.1)
    inset.set_xticks([])
    inset.set_yticks([-1.0, 0.0, 1.0])
    inset.tick_params(labelsize=7, length=2)
    inset.set_title("Cône de Dirac au point $K$", fontsize=8, pad=2)
    ax_bands.indicate_inset_zoom(inset, edgecolor="0.4", alpha=0.5)

    grid = uniform_k_grid(lattice.reciprocal_vectors, 150, 150, 1)
    energy_grid = np.linspace(-9.0, 9.0, 1801)
    dos = model.density_of_states(energy_grid, grid, broadening=0.10)

    ax_dos.plot(dos, energy_grid, color="#08306b", lw=1.4)
    ax_dos.fill_betweenx(energy_grid, 0.0, dos, color="#6baed6", alpha=0.30)
    ax_dos.axhline(0.0, color="0.35", lw=0.8, ls="--")
    ax_dos.set_xlim(0.0, 1.05 * float(np.max(dos)))
    ax_dos.set_xlabel("DOS (états eV$^{-1}$ cellule$^{-1}$)")
    ax_dos.set_title("Densité d'états", fontsize=11)
    ax_dos.tick_params(axis="y", labelleft=False)

    figure.suptitle(
        "UAM-TOPOMAT — Nuit 1 : validation du solveur quantique de structures de bandes",
        fontsize=12,
        y=0.995,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    return {
        "n_kpoints": len(kpoints),
        "dos_max": float(np.max(dos)),
        "dos_at_fermi": float(dos[np.argmin(np.abs(energy_grid))]),
        "hbar_vf": float(1.5 * t * d),
    }


def test_generate_night1_figure() -> None:
    output = ROOT / "results" / "figures" / "night1_bands_dos.png"
    metadata = _make_night1_figure(output)

    assert output.exists()
    assert output.stat().st_size > 30_000
    assert abs(_png_dpi(output) - 300.0) < 1.0
    assert metadata["n_kpoints"] == 1 + 3 * 239
    assert metadata["dos_max"] > 0.0
    assert metadata["dos_at_fermi"] > 0.0
    assert np.isclose(metadata["hbar_vf"], 5.751, atol=1e-12)
