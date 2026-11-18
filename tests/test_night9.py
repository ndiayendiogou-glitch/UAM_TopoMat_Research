"""UAM-TOPOMAT -- Validation de la Nuit 9.

Tests du moteur de criblage multiphysique des 2000 materiaux, de la
non-dominance stricte du front de Pareto, de la convergence et de la
reproductibilite NSGA-II, de la superiorite du champion face au silicium
(Matam 47 C, 1050 W/m^2) et de la figure de publication 300 dpi.
"""

from __future__ import annotations

import json
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

from src.optics.solar_optics import T_REF, photon_flux_above_gap, sq_efficiency
from src.pde.carrier_transport import site_electrical_result
from src.quantum.materials_generator import FAMILY_MXENE, N_MATERIALS
from src.thermo.inverse_discovery import (
    CHAMPIONS_PATH,
    FIGURE_PATH,
    MATAM_AMBIENT_C,
    MATAM_IRRADIANCE,
    SAINT_LOUIS_AMBIENT_C,
    SAINT_LOUIS_IRRADIANCE,
    SEED,
    _photon_flux_vectorized,
    discover,
    export_json,
    haldane_model_for,
    make_figure,
    non_dominated_mask,
    nsga2,
    run_screening,
    symbolic_stability,
)
from src.thermo.sahel_climate import MATAM, PVModule, energy_balance, silicon_module, solar_absorptance
from src.thermo.supercapacitor_mxene import AsymmetricSupercapacitor
from src.topology.berry_chern import berry_chern

FEATURE_KEYS = (
    "bandgap", "beta_t", "abundance", "cost", "tj_matam", "tj_saint_louis",
    "eta_matam", "eta_saint_louis", "voc_matam", "jsc_matam", "ff_matam",
    "performance", "stability", "energy_density", "retention", "chern",
)
CHAMPION_KEYS = (
    "id", "formula", "chemical_formula", "family", "structure", "bandgap_ev",
    "chern_number", "voc_matam_v", "jsc_matam_ma_cm2", "ff_matam", "eta_matam",
    "gain_vs_silicon_percent", "thermal_stability_percent", "cost_usd_per_kg",
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
            if data[position + 16] == 1:
                return pixels_per_metre * 0.0254
        position += 12 + length
    raise ValueError("Chunk pHYs introuvable.")


@lru_cache(maxsize=None)
def _screening():
    return run_screening()


@lru_cache(maxsize=None)
def _discovery():
    return discover()


def _payload() -> dict:
    return json.loads(CHAMPIONS_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Evaluation complete des 2000 candidats
# ---------------------------------------------------------------------------


def test_all_2000_candidates_are_evaluated_without_numerical_error() -> None:
    screening = _screening()
    assert screening.n_materials == N_MATERIALS == 2000
    assert screening.objectives.shape == (2000, 4)
    for key in FEATURE_KEYS:
        values = screening.values(key)
        assert values.shape == (2000,), key
        assert np.all(np.isfinite(values)), key
    eta = screening.values("eta_matam")
    assert 0.0 < float(eta.min()) < float(eta.max()) < 0.35
    assert np.all(screening.objectives[:, 1] > 0.95)
    assert np.all(screening.values("cost") > 0.0)
    assert np.all(screening.values("voc_matam") > 0.0)
    assert np.all(screening.values("jsc_matam") > 0.0)
    assert np.all((screening.values("ff_matam") > 0.5)
                  & (screening.values("ff_matam") < 0.95))
    assert set(np.unique(screening.values("chern"))) <= {-1.0, 0.0, 1.0}


def test_vectorized_physics_matches_nights_4_and_5() -> None:
    screening = _screening()
    for index in (0, 17, 500, 1383, 1516):
        material = screening.materials[index]
        bandgap = float(material["eg_ev"])
        module = PVModule(
            material["formula"], bandgap, float(material["beta_t"]),
            0.75 * sq_efficiency(bandgap, T_REF),
        )
        balance = energy_balance(MATAM, module)
        assert screening.values("tj_matam")[index] == pytest.approx(
            balance.junction_temperature_k, abs=0.1
        )
        assert screening.values("absorptance")[index] == pytest.approx(
            solar_absorptance(bandgap), abs=1.0e-3
        )
        assert screening.values("eta_matam")[index] == pytest.approx(
            balance.efficiency, abs=1.0e-3
        )
    for bandgap in (1.0, 1.45, 2.2):
        assert float(_photon_flux_vectorized(np.array([bandgap]))[0]) == pytest.approx(
            photon_flux_above_gap(bandgap), rel=1.0e-3
        )


def test_extreme_site_conditions_and_climate_ordering() -> None:
    screening = _screening()
    assert MATAM_AMBIENT_C == 47.0 and MATAM_IRRADIANCE == 1050.0
    assert SAINT_LOUIS_AMBIENT_C == 37.0 and SAINT_LOUIS_IRRADIANCE == 980.0
    assert float(np.mean(screening.values("eta_matam"))) < float(
        np.mean(screening.values("eta_saint_louis"))
    )
    assert np.all(screening.values("tj_matam") > screening.values("tj_saint_louis"))


def test_mxene_metrics_follow_night7_electrochemistry() -> None:
    screening = _screening()
    cell = AsymmetricSupercapacitor()
    retention = cell.capacity_retention(100.0, MATAM.ambient_k)
    assert retention > 0.95
    assert retention == pytest.approx(float(screening.values("retention")[0]), rel=1.0e-12)
    cycle = cell.galvanostatic_cycle(500.0, MATAM.ambient_k)
    assert 5.0 < cycle.energy_density_wh_kg < 15.0
    mxene = screening.values("is_mxene") > 0.5
    assert int(mxene.sum()) == 550
    energies = screening.values("energy_density")[mxene]
    assert np.all(energies > 0.0) and np.all(np.isfinite(energies))
    assert float(energies.min()) < float(energies.max())


# ---------------------------------------------------------------------------
# 2. NSGA-II et non-dominance stricte
# ---------------------------------------------------------------------------


def test_nsga2_converges_and_is_bitwise_deterministic() -> None:
    result = _discovery()
    diagnostic = result.diagnostic()
    assert diagnostic["seed"] == SEED
    assert diagnostic["nsga2_precision"] > 0.80
    assert diagnostic["pareto_front_size"] > 100
    assert diagnostic["nsga2_front_size"] > 100
    assert result.history["hypervolume_proxy"][-1] > result.history["hypervolume_proxy"][0]
    assert np.unique(result.population).size > 100

    objectives = _screening().objectives
    first, _, _ = nsga2(objectives, population_size=40, generations=15, seed=SEED)
    second, _, _ = nsga2(objectives, population_size=40, generations=15, seed=SEED)
    third, _, _ = nsga2(objectives, population_size=40, generations=15, seed=SEED + 1)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, third)


def test_every_pareto_point_is_strictly_non_dominated_by_the_database() -> None:
    result = _discovery()
    screening = result.screening
    minimization = -screening.objectives
    tolerance = 1.0e-9
    for point in result.front_indices:
        dominated = np.all(
            minimization <= minimization[point] + tolerance, axis=1
        ) & np.any(minimization < minimization[point] - tolerance, axis=1)
        dominated[point] = False
        assert not bool(np.any(dominated)), screening.materials[point]["id"]
    assert np.all(result.front_ranks[result.front_indices] == 0)
    non_front = np.setdiff1d(np.arange(2000), result.front_indices)
    assert not np.any(non_dominated_mask(minimization)[non_front])


def test_exported_front_matches_recomputed_objectives() -> None:
    result = _discovery()
    screening = result.screening
    payload = _payload()
    positions = {material["id"]: index for index, material in enumerate(screening.materials)}
    assert len(payload["pareto_front"]) == len(result.front_indices)
    for record in payload["pareto_front"]:
        index = positions[record["id"]]
        exported = np.array([
            record["performance_score"],
            record["thermal_stability_percent"] / 100.0,
            record["abundance_index"],
            -record["cost_usd_per_kg"],
        ])
        np.testing.assert_allclose(exported, screening.objectives[index], atol=1.0e-9)


# ---------------------------------------------------------------------------
# 3. Top 10, reproductibilite et champion n°1
# ---------------------------------------------------------------------------


def test_top10_export_is_complete_and_reproducible(tmp_path: Path) -> None:
    result = _discovery()
    screening = result.screening
    payload = _payload()
    top = payload["top10_champions"]
    assert len(top) == 10
    assert [record["id"] for record in top] == [
        screening.materials[index]["id"] for index in result.champions
    ]
    assert payload["solar_champion_id"] == screening.materials[result.solar_champion]["id"]
    assert payload["mxene_champion_id"] == screening.materials[result.mxene_champion]["id"]
    ids = {record["id"] for record in top}
    assert payload["solar_champion_id"] in ids
    assert payload["mxene_champion_id"] in ids
    for record in top:
        for key in CHAMPION_KEYS:
            assert key in record
        assert record["pareto_rank"] == 0
        assert record["thermal_stability_percent"] > 95.0
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    export_json(result, first)
    export_json(result, second)
    assert first.read_bytes() == second.read_bytes() == CHAMPIONS_PATH.read_bytes()


def test_champion_number_one_beats_silicon_at_matam() -> None:
    result = _discovery()
    screening = result.screening
    payload = _payload()
    champion = result.champion_records[0]
    silicon = screening.silicon
    assert champion["id"] == payload["solar_champion_id"]
    expected_gain = 100.0 * (champion["eta_matam"] / silicon["eta_matam"] - 1.0)
    assert champion["gain_vs_silicon_percent"] == pytest.approx(expected_gain, rel=1.0e-12)
    assert champion["gain_vs_silicon_percent"] > 20.0
    assert champion["eta_matam"] > silicon["eta_matam"]
    assert champion["voc_matam_v"] > silicon["voc_matam_v"]
    assert champion["ff_matam"] > silicon["ff_matam"]
    assert champion["thermal_stability_percent"] > 95.0

    bandgap = champion["bandgap_ev"]
    module = PVModule(
        champion["formula"], bandgap, champion["thermal_drift_beta_per_k"],
        0.75 * sq_efficiency(bandgap, T_REF),
    )
    exact = site_electrical_result(MATAM, module, n_points=25)
    reference = site_electrical_result(MATAM, silicon_module(), n_points=25)
    assert all(solution.converged for solution in exact.solutions.values())
    assert exact.efficiency > reference.efficiency
    assert exact.efficiency / reference.efficiency > 1.15


def test_chern_numbers_match_numerical_berry_curvature() -> None:
    result = _discovery()
    screening = result.screening
    positions = {material["id"]: index for index, material in enumerate(screening.materials)}
    for record in result.champion_records:
        material = screening.materials[positions[record["id"]]]
        model = haldane_model_for(material)
        numeric = berry_chern(
            model, model.reciprocal_vectors, n1=12, n2=12, band=0
        ).chern_integer
        assert numeric == record["chern_number"]


def test_symbolic_stability_and_cost_laws() -> None:
    symbolic = symbolic_stability()
    assert symbolic["stability_at_zero"] == 1
    assert symbolic["stability_derivative"] == 0
    assert symbolic["cost_derivative"].is_negative is True
    cost_symbol = next(
        symbol for symbol in symbolic["cost_score"].free_symbols if symbol.name == "cost"
    )
    assert sp.simplify(symbolic["cost_score"].subs(cost_symbol, 0)) == 1


# ---------------------------------------------------------------------------
# 4. Figure de publication
# ---------------------------------------------------------------------------


def test_generate_night9_pareto_figure() -> None:
    metadata = make_figure(_discovery())
    assert FIGURE_PATH.exists()
    assert FIGURE_PATH.stat().st_size > 100_000
    assert abs(_png_dpi(FIGURE_PATH) - 300.0) < 1.0
    assert metadata["n_candidates"] == 2000
    assert metadata["n_front"] > 100
    assert metadata["n_champions"] == 10
    assert metadata["gain_percent"] > 20.0
    assert len(metadata["radar_values"]) == 5
