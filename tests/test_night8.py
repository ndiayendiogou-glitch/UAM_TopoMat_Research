"""UAM-TOPOMAT -- Validation de la Nuit 8.

Tests de la base deterministe de 2000 materiaux sans plomb ni cobalt :
unicite, neutralite de charge, bornes physiques, determinisme a graine egale,
correlations cristallochimiques, evaluation d'un echantillon representatif par
les solveurs des Nuits 1 a 7 et figure 300 dpi de la carte des materiaux.
"""

from __future__ import annotations

import json
import struct
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.optics.solar_optics import T_REF, shockley_queisser, sq_efficiency
from src.pde.carrier_transport import (
    device_from_module,
    solve_carrier_transport,
    solve_dark_equilibrium,
)
from src.quantum.band_structure import uniform_k_grid
from src.quantum.materials_generator import (
    DB_PATH,
    EG_MAX,
    EG_MIN,
    FAMILY_ABX3,
    FAMILY_DOUBLE,
    FAMILY_MXENE,
    FAMILY_TARGETS,
    N_MATERIALS,
    SEED,
    SUMMARY_PATH,
    build_tight_binding_model,
    generate_candidates,
    load_database,
    material_summary,
    representative_sample,
)
from src.thermo.sahel_climate import DAKAR, PVModule, energy_balance
from src.thermo.supercapacitor_mxene import ElectrodeConfig, MXeneElectrode
from src.topology.berry_chern import berry_chern
from src.topology.edge_states import (
    haldane_ky_values,
    haldane_ribbon,
    solve_ribbon_spectrum,
)
from src.topology.models_topo import HaldaneModel

FORBIDDEN = ("Pb", "Co")
FAMILIES = (FAMILY_ABX3, FAMILY_DOUBLE, FAMILY_MXENE)
PHYSICAL_KEYS = (
    "a", "c", "t1", "t2", "lambda_soc", "eg_ev", "mu_n", "mu_p",
    "beta_t", "abundance_index", "cost_usd_per_kg",
)


def _png_dpi(path: Path) -> float:
    """Lit la resolution (dpi) stockee dans le chunk pHYs d'un PNG."""
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Le fichier n'est pas un PNG valide.")
    position = 8
    while position + 8 <= len(data):
        length = struct.unpack(">I", data[position:position + 4])[0]
        chunk_type = data[position + 4:position + 8]
        if chunk_type == b"pHYs":
            pixels_per_metre = struct.unpack(">I", data[position + 8:position + 12])[0]
            unit = data[position + 16]
            if unit == 1:
                return pixels_per_metre * 0.0254
        position += 12 + length
    raise ValueError("Chunk pHYs introuvable.")


@lru_cache(maxsize=None)
def _database() -> list[dict]:
    return load_database()


@lru_cache(maxsize=None)
def _summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _sample() -> list[dict]:
    return representative_sample(_database())


def _by_family(family: str) -> list[dict]:
    return [m for m in _database() if m["family"] == family]


def _module(material: dict) -> PVModule:
    return PVModule(
        name=material["formula"],
        bandgap_ev=material["eg_ev"],
        beta_t=material["beta_t"],
        eta_ref=0.75 * sq_efficiency(material["eg_ev"], T_REF),
    )


# ---------------------------------------------------------------------------
# 1. Structure de la base
# ---------------------------------------------------------------------------


def test_database_has_exactly_2000_unique_materials() -> None:
    materials = _database()
    assert len(materials) == N_MATERIALS == 2000
    assert len({m["id"] for m in materials}) == 2000
    assert len({m["formula"] for m in materials}) == 2000
    assert [m["id"] for m in materials] == [f"UAM-{i:04d}" for i in range(1, 2001)]


def test_database_is_lead_free_and_cobalt_free() -> None:
    materials = _database()
    for material in materials:
        assert not (set(material["elements"]) & set(FORBIDDEN))
        assert not (set(material["composition"]) & set(FORBIDDEN))
        for symbol in FORBIDDEN:
            assert symbol not in material["formula"]
    assert _summary()["lead_free"] is True
    assert _summary()["cobalt_free"] is True


def test_three_crystal_families_are_represented() -> None:
    materials = _database()
    assert {m["family"] for m in materials} == set(FAMILIES)
    for family, target in FAMILY_TARGETS.items():
        assert sum(1 for m in materials if m["family"] == family) == target
    subfamilies = {m["subfamily"] for m in materials}
    assert {"halide_ABX3", "oxide_ABO3", "halide_A2AgBiX6",
            "halide_A2InSbX6", "oxide_A2BB'O6", "M2X", "M3X2"} <= subfamilies


def test_materials_use_only_allowed_west_african_elements() -> None:
    allowed = {"Cs", "Rb", "K", "Na", "Sr", "Ba", "Ca", "La", "Ce", "Nd",
               "Sm", "Y", "Bi", "Sn", "Ge", "Ti", "Fe", "Mn", "V", "Ag",
               "Sb", "In", "Cl", "Br", "I", "O", "C", "N", "H", "F"}
    for material in _database():
        assert set(material["elements"]) <= allowed


# ---------------------------------------------------------------------------
# 2. Bornes physico-chimiques
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_charge_neutrality_for_every_candidate(family: str) -> None:
    for material in _by_family(family):
        counts = material["composition"]
        states = material["oxidation_states"]
        assert set(counts) == set(states)
        net = sum(counts[s] * states[s] for s in counts)
        assert abs(net) < 1.0e-9
        assert abs(material["net_charge"]) < 1.0e-9


def test_oxidation_states_match_chemistry() -> None:
    for material in _database():
        states = material["oxidation_states"]
        for symbol in ("Cl", "Br", "I", "F"):
            if symbol in states:
                assert states[symbol] == pytest.approx(-1.0)
        if "O" in states and "H" not in states:
            assert states["O"] == pytest.approx(-2.0)
        if "Bi" in states and material["family"] == FAMILY_DOUBLE:
            assert states["Bi"] == pytest.approx(3.0)


def test_lattice_parameters_are_strictly_positive() -> None:
    for material in _database():
        assert material["a"] > 0.0 and np.isfinite(material["a"])
        assert material["c"] > 0.0 and np.isfinite(material["c"])
        assert 2.5 <= material["a"] <= 8.0
        assert material["c"] >= material["a"] - 1.0e-9


def test_optical_gaps_are_within_required_bounds() -> None:
    gaps = np.array([m["eg_ev"] for m in _database()])
    assert np.all(np.isfinite(gaps))
    assert np.all(gaps > 0.0)
    assert gaps.min() >= EG_MIN - 1.0e-9
    assert gaps.max() <= EG_MAX + 1.0e-9
    assert _summary()["gap_bounds_ev"] == [EG_MIN, EG_MAX]


def test_hoppings_and_spin_orbit_coupling_are_physical() -> None:
    for material in _database():
        assert 0.0 < material["t1"] <= 2.5
        assert 0.0 < material["t2"] < material["t1"]
        assert 0.0 <= material["lambda_soc"] <= 0.5


def test_mobilities_are_realistic_and_ordered() -> None:
    for material in _database():
        assert 1.0 <= material["mu_p"] < material["mu_n"] <= 2000.0


def test_thermal_drift_is_negative_and_bounded() -> None:
    betas = np.array([m["beta_t"] for m in _database()])
    assert np.all(betas < 0.0)
    assert betas.min() > -0.006
    assert betas.max() < -0.0005


def test_resource_index_and_cost_are_bounded() -> None:
    for material in _database():
        assert 0.0 <= material["abundance_index"] <= 1.0
        assert material["cost_usd_per_kg"] > 0.0
        assert np.isfinite(material["cost_usd_per_kg"])


def test_perovskite_tolerance_factors_are_reasonable() -> None:
    factors = [m["tolerance_factor"] for m in _database()
               if m["tolerance_factor"] is not None]
    assert len(factors) == 900
    assert all(0.6 < value < 1.2 for value in factors)


# ---------------------------------------------------------------------------
# 3. Determinisme et resume
# ---------------------------------------------------------------------------


def test_determinism_at_equal_seed() -> None:
    first = json.dumps(generate_candidates(seed=SEED), sort_keys=True)
    second = json.dumps(generate_candidates(seed=SEED), sort_keys=True)
    on_disk = json.dumps(_database(), sort_keys=True)
    assert first == second == on_disk


def test_different_seed_produces_a_different_database() -> None:
    other = generate_candidates(seed=SEED + 1)
    assert len(other) == N_MATERIALS
    assert [m["formula"] for m in other] != [m["formula"] for m in _database()]
    assert len({m["formula"] for m in other}) == N_MATERIALS


def test_summary_is_consistent_with_database() -> None:
    materials = _database()
    summary = _summary()
    assert summary["seed"] == SEED
    assert summary["n_materials"] == N_MATERIALS
    assert summary["families"] == {f: FAMILY_TARGETS[f] for f in sorted(FAMILY_TARGETS)}
    assert sum(summary["families"].values()) == N_MATERIALS
    assert sum(summary["subfamilies"].values()) == N_MATERIALS
    assert summary["charge_neutral"] is True
    for key in PHYSICAL_KEYS:
        values = np.array([m[key] for m in materials])
        assert summary["means"][key] == pytest.approx(float(np.mean(values)), abs=1e-6)
        assert summary["ranges"][key][0] == pytest.approx(float(np.min(values)), abs=1e-6)
        assert summary["ranges"][key][1] == pytest.approx(float(np.max(values)), abs=1e-6)


def test_summary_can_be_recomputed_deterministically() -> None:
    assert material_summary(generate_candidates(seed=SEED), seed=SEED) == _summary()
    assert DB_PATH.exists() and SUMMARY_PATH.exists()


# ---------------------------------------------------------------------------
# 4. Correlations cristallochimiques
# ---------------------------------------------------------------------------


def test_gaps_are_correlated_with_family_and_chemistry() -> None:
    halides = [m for m in _by_family(FAMILY_ABX3) if m["subfamily"] == "halide_ABX3"]
    oxides = [m for m in _by_family(FAMILY_ABX3) if m["subfamily"] == "oxide_ABO3"]
    assert np.mean([m["eg_ev"] for m in oxides]) > np.mean([m["eg_ev"] for m in halides])
    iodides = [m for m in halides if "I" in m["elements"]]
    chlorides = [m for m in halides if "I" not in m["elements"]]
    assert np.mean([m["eg_ev"] for m in iodides]) < np.mean([m["eg_ev"] for m in chlorides])


def test_spin_orbit_coupling_tracks_heavy_elements() -> None:
    with_bi = [m for m in _database() if "Bi" in m["elements"]]
    without_bi = [m for m in _database() if "Bi" not in m["elements"]]
    assert np.mean([m["lambda_soc"] for m in with_bi]) > 5.0 * np.mean(
        [m["lambda_soc"] for m in without_bi]
    )
    assert all(m["lambda_soc"] == pytest.approx(0.45) for m in with_bi)


def test_mobility_tracks_hopping_and_gap_tracks_stability() -> None:
    materials = _database()
    t1 = np.array([m["t1"] for m in materials])
    mu = np.array([m["mu_n"] for m in materials])
    assert np.corrcoef(t1, mu)[0, 1] > 0.9
    double = np.array([m["beta_t"] for m in _by_family(FAMILY_DOUBLE)])
    simple = np.array([m["beta_t"] for m in _by_family(FAMILY_ABX3)])
    assert double.mean() > simple.mean()


def test_african_abundance_and_cost_track_elements() -> None:
    mxenes = _by_family(FAMILY_MXENE)
    scarce = [m for m in _database() if "Ag" in m["elements"] or "Ge" in m["elements"]]
    assert np.mean([m["abundance_index"] for m in mxenes]) > 0.8
    assert np.mean([m["abundance_index"] for m in scarce]) < 0.7
    assert np.mean([m["cost_usd_per_kg"] for m in scarce]) > 10.0 * np.mean(
        [m["cost_usd_per_kg"] for m in mxenes]
    )


# ---------------------------------------------------------------------------
# 5. Evaluation des solveurs des Nuits 1 a 7
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index", list(range(7)))
def test_night1_tight_binding_model_from_candidate(index: int) -> None:
    material = _sample()[index]
    model = build_tight_binding_model(material)
    assert model.n_orbitals in (3, 5)
    hamiltonian = model.hamiltonian(np.zeros(3))
    assert np.allclose(hamiltonian, hamiltonian.conj().T, atol=1e-13)
    kpoints = uniform_k_grid(model.lattice.reciprocal_vectors, 8, 8, 1)
    bands = model.band_structure(kpoints)
    velocities = model.group_velocities(kpoints[:4])
    assert bands.shape == (len(kpoints), model.n_orbitals)
    assert np.all(np.isfinite(bands)) and np.all(np.isfinite(velocities))
    assert model.lattice.volume > 0.0


def test_night2_berry_chern_with_candidate_hoppings() -> None:
    material = _sample()[0]
    model = HaldaneModel(
        t1=material["t1"], t2=material["t2"], phi=np.pi / 2.0,
        M=0.5 * material["t2"],
    )
    result = berry_chern(model, model.reciprocal_vectors, n1=12, n2=12, band=0)
    assert result.chern_integer == 1
    assert abs(result.chern - 1.0) < 1.0e-9
    assert np.all(np.isfinite(result.curvature))


def test_night3_edge_states_with_candidate_hoppings() -> None:
    material = _sample()[0]
    ribbon, bulk_model, chern = haldane_ribbon(
        n_y=14, t1=material["t1"], t2=material["t2"],
        phi=np.pi / 2.0, M=0.5 * material["t2"],
    )
    kpoints = np.linspace(-np.pi / np.sqrt(3.0), np.pi / np.sqrt(3.0), 13)
    spectrum = solve_ribbon_spectrum(
        ribbon, kpoints, bulk_model, haldane_ky_values(1.0, 81), chern=chern
    )
    assert chern == 1
    assert spectrum.has_edge_modes
    assert spectrum.n_crossings >= 2
    assert spectrum.metadata["min_projected_gap"] > 0.0


@pytest.mark.parametrize("family", FAMILIES)
def test_night4_shockley_queisser_for_candidate_gaps(family: str) -> None:
    materials = _by_family(family)
    efficiencies = np.array([shockley_queisser(m["eg_ev"], T_REF).efficiency
                             for m in materials[:40]])
    assert np.all(efficiencies > 0.0)
    assert np.all(efficiencies < 0.34)
    assert np.all(np.isfinite(efficiencies))


@pytest.mark.parametrize("index", [0, 1, 2])
def test_night5_thermal_balance_with_candidate_parameters(index: int) -> None:
    material = _sample()[index]
    balance = energy_balance(DAKAR, _module(material))
    assert balance.junction_temperature_k > DAKAR.ambient_k
    assert abs(balance.residual) < 1.0e-9
    assert 0.0 < balance.efficiency < 0.35
    assert balance.operating_drift == pytest.approx(material["beta_t"])


@pytest.mark.parametrize("index", [0, 5])
def test_night6_carrier_transport_with_candidate_parameters(index: int) -> None:
    material = _sample()[index]
    module = _module(material)
    balance = energy_balance(DAKAR, module)
    device = device_from_module(module, n_points=41)
    equilibrium = solve_dark_equilibrium(device, T_REF)
    assert equilibrium.converged
    assert abs(float(np.mean(equilibrium.total_current))) < 1.0e-10
    solution = solve_carrier_transport(
        device, balance.junction_temperature_k, 0.2,
        DAKAR.irradiance, initial=equilibrium,
    )
    assert solution.converged
    assert np.all(np.isfinite(solution.potential))
    assert np.all(np.isfinite(solution.electrons))
    assert -solution.current_density > 0.0


def test_night7_mxene_electrode_from_candidate() -> None:
    material = next(m for m in _sample() if m["family"] == FAMILY_MXENE)
    electrode = MXeneElectrode(ElectrodeConfig(
        name=material["formula"],
        lattice_a_angstrom=material["a"],
        lattice_c_angstrom=material["c"],
        hopping_t_ev=material["t1"],
    ))
    potentials = np.array([0.0, 0.4, 0.8])
    capacitance = np.asarray(electrode.quantum_capacitance_areal(potentials, T_REF))
    assert electrode.model.n_orbitals == 3
    assert np.all(np.isfinite(capacitance)) and np.all(capacitance > 0.0)
    assert electrode.capacitance_specific(0.4, T_REF) > 0.0


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

COLORS = {FAMILY_ABX3: "#08306b", FAMILY_DOUBLE: "#1b7837", FAMILY_MXENE: "#d95f0e"}
LABELS = {
    FAMILY_ABX3: "Pérovskites $ABX_3$",
    FAMILY_DOUBLE: "Pérovskites doubles $A_2B'B''X_6$",
    FAMILY_MXENE: "MXènes 2D $M_2X$ / $M_3X_2$",
}


def _make_night8_figure(output_path: Path) -> dict:
    materials = _database()
    figure = plt.figure(figsize=(14.0, 9.6))
    grid = figure.add_gridspec(
        2, 2, width_ratios=(1.0, 1.0), hspace=0.36, wspace=0.28
    )
    ax_gap = figure.add_subplot(grid[0, 0])
    ax_stability = figure.add_subplot(grid[0, 1])
    ax_resource = figure.add_subplot(grid[1, 0])
    ax_counts = figure.add_subplot(grid[1, 1])

    edges = np.linspace(EG_MIN, EG_MAX, 47)
    for family in FAMILIES:
        gaps = np.array([m["eg_ev"] for m in _by_family(family)])
        ax_gap.hist(gaps, bins=edges, histtype="stepfilled", alpha=0.42,
                    color=COLORS[family], label=f"{LABELS[family]} (n={len(gaps)})")
        ax_gap.hist(gaps, bins=edges, histtype="step", color=COLORS[family], lw=1.4)
    ax_gap.set_xlabel("Gap optique $E_g$ (eV)")
    ax_gap.set_ylabel("Nombre de matériaux")
    ax_gap.set_title("(a) Distribution des gaps optiques", fontsize=11)
    ax_gap.grid(alpha=0.25)
    ax_gap.legend(loc="upper right", framealpha=0.92, fontsize=7.5)

    for family in FAMILIES:
        group = _by_family(family)
        ax_stability.scatter(
            [m["eg_ev"] for m in group],
            [-1000.0 * m["beta_t"] for m in group],
            s=11, alpha=0.55, color=COLORS[family], edgecolors="none",
            label=LABELS[family],
        )
    ax_stability.set_xlabel("Gap optique $E_g$ (eV)")
    ax_stability.set_ylabel("Stabilité thermique $-1000\\,\\beta_T$ (%/kK)")
    ax_stability.set_title("(b) Gap vs. dérive thermique", fontsize=11)
    ax_stability.grid(alpha=0.25)
    ax_stability.legend(loc="upper right", framealpha=0.92, fontsize=7.5)

    scatter = ax_resource.scatter(
        [m["abundance_index"] for m in materials],
        [m["cost_usd_per_kg"] for m in materials],
        c=[m["eg_ev"] for m in materials], cmap="viridis",
        s=11, alpha=0.6, edgecolors="none", norm=plt.Normalize(EG_MIN, EG_MAX),
    )
    colorbar = figure.colorbar(scatter, ax=ax_resource, pad=0.015)
    colorbar.set_label("$E_g$ (eV)")
    ax_resource.set_xlabel("Indice de ressource ouest-africaine")
    ax_resource.set_ylabel("Coût estimé (USD/kg)")
    ax_resource.set_yscale("log")
    ax_resource.set_title("(c) Abondance vs. coût, coloré par $E_g$", fontsize=11)
    ax_resource.grid(alpha=0.25, which="both")

    positions = np.arange(len(FAMILIES))
    abundance_means = [np.mean([m["abundance_index"] for m in _by_family(f)])
                       for f in FAMILIES]
    counts = [len(_by_family(f)) for f in FAMILIES]
    ax_counts.bar(positions - 0.19, counts, width=0.38, color="#9ecae1",
                  edgecolor="#08306b", label="Nombre de candidats")
    ax_counts_bar = ax_counts.twinx()
    ax_counts_bar.bar(positions + 0.19, abundance_means, width=0.38,
                      color="#fdae6b", edgecolor="#d95f0e",
                      label="Indice de ressource moyen")
    ax_counts.set_xticks(positions)
    ax_counts.set_xticklabels(["$ABX_3$", "$A_2B'B''X_6$", "MXène"])
    ax_counts.set_ylabel("Nombre de candidats")
    ax_counts_bar.set_ylabel("Indice de ressource moyen")
    ax_counts_bar.set_ylim(0.0, 1.05)
    ax_counts.set_title("(d) Répartition par famille cristalline", fontsize=11)
    ax_counts.grid(alpha=0.25, axis="y")

    figure.suptitle(
        "UAM-TOPOMAT — cartographie de 2000 matériaux cristallins "
        "sans plomb (ressources d'Afrique de l'Ouest)",
        fontsize=12.0, y=0.995,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    return {
        "counts": counts,
        "abundance_means": abundance_means,
        "gap_min": float(min(m["eg_ev"] for m in materials)),
        "gap_max": float(max(m["eg_ev"] for m in materials)),
        "n_points": len(materials),
    }


def test_generate_night8_materials_landscape_figure() -> None:
    output = ROOT / "results" / "figures" / "night8_materials_landscape.png"
    metadata = _make_night8_figure(output)
    assert output.exists()
    assert output.stat().st_size > 100_000
    assert abs(_png_dpi(output) - 300.0) < 1.0
    assert metadata["n_points"] == 2000
    assert metadata["counts"] == [900, 550, 550]
    assert metadata["gap_min"] >= EG_MIN
    assert metadata["gap_max"] <= EG_MAX
    assert metadata["abundance_means"][2] > metadata["abundance_means"][1]
