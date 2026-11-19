"""UAM-TOPOMAT -- Validation de la Nuit 11.

Tests du simulateur d'impact Senegal (MWh/an, t CO2/an, FCFA/an, LCOE),
de la figure 300 dpi, de l'export JSON et de toutes les routes de
l'application FastAPI locale (TestClient -> 200 OK).
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.server import app  # noqa: E402
from src.quantum.symbolic_proofs import (  # noqa: E402
    MXENE_CHAMPION_ID,
    SOLAR_CHAMPION_ID,
)
from src.thermo.senegal_grid_impact import (  # noqa: E402
    DEFAULT_SITES,
    FIGURE_PATH,
    JSON_PATH,
    PLANT_CAPACITY_KWC,
    GRID_CO2_TONNES_PER_MWH,
    GRID_TARIFF_FCFA_PER_KWH,
    annuity_factor,
    champion_modules,
    conventional_modules,
    evaluate_all_sites,
    evaluate_plant,
    evaluate_site,
    export_json,
    make_figure,
    run,
    site_at_temperature,
    site_by_name,
    symbolic_impact_laws,
)

SITES = ("Dakar", "Saint-Louis", "Matam")
SENEGAL_ZONES = (
    "Dakar", "Saint-Louis", "Matam", "Podor",
    "Tambacounda", "Kaolack", "Louga", "Ziguinchor",
    "Région personnalisée",
)
ZONE_TEMPERATURES = (31.0, 37.0, 47.0, 48.0, 44.0, 40.0, 39.0, 34.0)
IMPACT_KEYS = (
    "site", "ambient_c", "champion_mwh_an", "classique_mwh_an", "gain_mwh_an",
    "gain_percent", "co2_evitees_t_an", "economies_fcfa_an",
    "lcoe_champion_fcfa_kwh", "lcoe_classique_fcfa_kwh",
    "lcoe_reduction_fcfa_kwh", "lcoe_reduction_percent",
)
PIPELINE_KEYS = (
    "request", "material", "site", "band_structure", "berry", "jv",
    "ragone", "impact", "plant",
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as instance:
        yield instance


@pytest.fixture(scope="module")
def impact():
    return evaluate_all_sites()


def _png_dpi(path: Path) -> float:
    """Lit la resolution (dpi) du chunk pHYs d'un PNG."""
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


# ---------------------------------------------------------------------------
# 1. Physique et climat des trois sites
# ---------------------------------------------------------------------------


def test_site_catalog_and_temperature_override() -> None:
    assert tuple(site.name for site in DEFAULT_SITES) == SITES
    assert [site.ambient_c for site in DEFAULT_SITES] == [31.0, 37.0, 47.0]
    assert site_by_name("Dakar").ambient_c == 31.0
    assert site_by_name("saint-louis", 55.0).ambient_c == 55.0
    assert [site_by_name(name).ambient_c for name in SENEGAL_ZONES[:-1]] == list(
        ZONE_TEMPERATURES
    )
    assert site_by_name("region personnalisee", 20.0).ambient_c == 20.0
    assert site_at_temperature(DEFAULT_SITES[2], 25.0).ambient_c == 25.0
    with pytest.raises(ValueError):
        site_at_temperature(DEFAULT_SITES[0], 19.0)
    with pytest.raises(ValueError):
        site_by_name("Inconnu")


def test_plant_power_area_and_usable_energy(impact) -> None:
    solar, mxene = champion_modules()
    for site in DEFAULT_SITES:
        result = impact.sites[site.name].champion
        expected_area = PLANT_CAPACITY_KWC * 1000.0 / (1000.0 * solar.eta_ref)
        assert result.area_m2 == pytest.approx(expected_area, rel=1.0e-12)
        assert result.pv_energy_mwh > 0.0
        assert result.usable_energy_mwh == pytest.approx(
            result.pv_energy_mwh * mxene.roundtrip_efficiency, rel=1.0e-12
        )
        assert result.storage_mass_kg > 0.0
        assert result.cycles_per_year > 0.0


def test_annual_gain_and_co2_avoided(impact) -> None:
    for name in SITES:
        item = impact.sites[name]
        assert item.gain_mwh > 0.0
        assert item.gain_mwh == pytest.approx(
            item.champion.usable_energy_mwh - item.baseline.usable_energy_mwh,
            rel=1.0e-12,
        )
        assert item.co2_avoided_tonnes == pytest.approx(
            item.gain_mwh * GRID_CO2_TONNES_PER_MWH, rel=1.0e-12
        )
        assert item.relative_gain_percent > 0.0
    assert impact.total_gain_mwh == pytest.approx(
        sum(impact.sites[name].gain_mwh for name in SITES), rel=1.0e-12
    )
    assert impact.total_co2_tonnes == pytest.approx(
        impact.total_gain_mwh * GRID_CO2_TONNES_PER_MWH, rel=1.0e-12
    )
    assert impact.sites["Matam"].gain_mwh > impact.sites["Dakar"].gain_mwh


def test_financial_savings_and_lcoe_reduction(impact) -> None:
    for name in SITES:
        item = impact.sites[name]
        assert item.savings_fcfa > 0.0
        assert item.savings_fcfa == pytest.approx(
            item.gain_mwh * 1000.0 * GRID_TARIFF_FCFA_PER_KWH
            + item.baseline.total_annual_cost_fcfa
            - item.champion.total_annual_cost_fcfa,
            rel=1.0e-10,
        )
        assert item.savings_fcfa == pytest.approx(
            item.champion.net_benefit_fcfa - item.baseline.net_benefit_fcfa,
            rel=1.0e-10,
        )
        assert item.lcoe_reduction_fcfa_per_kwh > 0.0
        assert item.lcoe_reduction_percent > 0.0
        assert item.champion.lcoe_fcfa_per_kwh == pytest.approx(
            item.champion.total_annual_cost_fcfa
            / (item.champion.usable_energy_mwh * 1000.0),
            rel=1.0e-12,
        )
    assert impact.total_savings_fcfa == pytest.approx(
        sum(impact.sites[name].savings_fcfa for name in SITES), rel=1.0e-12
    )


def test_annuity_and_storage_technologies() -> None:
    assert annuity_factor(0.0, 20) == pytest.approx(1.0 / 20.0)
    assert annuity_factor(0.05, 20) > annuity_factor(0.0, 20)
    solar, mxene = champion_modules()
    silicon, battery = conventional_modules()
    assert solar.name.startswith(SOLAR_CHAMPION_ID)
    assert mxene.name.startswith(MXENE_CHAMPION_ID)
    assert mxene.roundtrip_efficiency > battery.roundtrip_efficiency
    assert mxene.cycle_life > 100 * battery.cycle_life
    assert mxene.energy_density_wh_kg > 0.0
    assert silicon.bandgap_ev == pytest.approx(1.12)
    plant = evaluate_site(DEFAULT_SITES[2], solar, mxene, silicon, battery).champion
    assert plant.storage_mass_kg == pytest.approx(
        100.0 * 1000.0 / (mxene.energy_density_wh_kg * mxene.depth_of_discharge),
        rel=1.0e-12,
    )


def test_symbolic_economic_laws_are_exact() -> None:
    checks = symbolic_impact_laws()
    for key in ("crf_identity", "lcoe_identity", "co2_derivative",
                "savings_derivative", "savings_zero"):
        assert checks[key] == 0


def test_no_scipy_dependency_is_introduced() -> None:
    import re

    for filename in (
        ROOT / "src" / "thermo" / "senegal_grid_impact.py",
        ROOT / "app" / "server.py",
        Path(__file__),
    ):
        source = filename.read_text(encoding="utf-8")
        imports = re.findall(r"^\s*(?:from|import)\s+([A-Za-z0-9_.]+)", source, re.MULTILINE)
        assert all(not module.startswith("scipy") for module in imports), filename


# ---------------------------------------------------------------------------
# 2. Figure et export
# ---------------------------------------------------------------------------


def test_figure_and_json_artifacts(impact) -> None:
    assert FIGURE_PATH.exists()
    assert FIGURE_PATH.stat().st_size > 100_000
    assert abs(_png_dpi(FIGURE_PATH) - 300.0) < 1.0
    metadata = make_figure(impact)
    assert metadata["dpi"] == 300
    assert metadata["sites"] == list(SITES)
    assert metadata["gain_mwh_an"] == pytest.approx(impact.total_gain_mwh, rel=1.0e-9)
    payload = export_json(impact)
    assert JSON_PATH.exists()
    assert payload["metadata"]["night"] == 11
    assert payload["metadata"]["author"] == "Ndiogou Ndiaye"
    assert payload["metadata"]["champions"]["solar"] == SOLAR_CHAMPION_ID
    assert set(payload["sites"]) == set(SITES)
    assert payload["aggregate"]["gain_mwh_an"] == pytest.approx(
        impact.total_gain_mwh, rel=1.0e-12
    )
    assert json_is_stable(payload)


def json_is_stable(payload: dict) -> bool:
    import json

    return json.loads(JSON_PATH.read_text(encoding="utf-8")) == payload


def test_cached_pipeline_regenerates_artifacts() -> None:
    result = run()
    assert result.total_gain_mwh > 0.0
    assert result.total_co2_tonnes > 0.0
    assert result.total_savings_fcfa > 0.0


# ---------------------------------------------------------------------------
# 3. Routes FastAPI
# ---------------------------------------------------------------------------


def test_root_and_static_assets_return_200(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    for token in ("Université", "UFR STA", "Cellule Physique", "Ndiogou Ndiaye",
                  "Laboratoire Numérique Souverain", "Sénégal &amp; Sahel",
                  "Région personnalisée", "Bilan énergétique multi-régions (300 dpi)"):
        assert token in html, token
    assert "Nuit" not in html
    assert "cdn." not in html.lower()
    assert "https://" not in html
    assert '<script src="http' not in html.lower()
    for path in ("/static/index.html", "/static/app.js"):
        assert client.get(path).status_code == 200
    assert "fetch(\"/api/pipeline/run\"" in client.get("/static/app.js").text


def test_health_and_catalog_routes_return_200(client: TestClient) -> None:
    health = client.get("/api/health")
    assert health.status_code == 200
    data = health.json()
    assert data["status"] == "ok" and data["offline"] is True
    assert data["n_materials"] == 2000
    assert data["champions"] == [SOLAR_CHAMPION_ID, MXENE_CHAMPION_ID]
    catalog = client.get("/api/catalog")
    assert catalog.status_code == 200
    payload = catalog.json()
    assert payload["count"] == 2000
    assert len(payload["materials"]) == 2000
    champions = [m for m in payload["materials"] if m["is_champion"]]
    assert [m["id"] for m in champions] == [SOLAR_CHAMPION_ID, MXENE_CHAMPION_ID]


def test_material_route_returns_200_for_any_candidate(client: TestClient) -> None:
    for material_id in (SOLAR_CHAMPION_ID, MXENE_CHAMPION_ID, "UAM-0001", "UAM-2000"):
        response = client.get(f"/api/material/{material_id}")
        assert response.status_code == 200
        assert response.json()["id"] == material_id
    assert client.get("/api/material/UAM-9999").status_code == 404


@pytest.mark.parametrize("city", SITES)
@pytest.mark.parametrize("temperature", (25.0, 47.0, 55.0))
def test_impact_route_returns_200_for_all_cities(client: TestClient, city: str, temperature: float) -> None:
    response = client.get(
        "/api/impact",
        params={"city": city, "temperature_c": temperature},
    )
    assert response.status_code == 200
    data = response.json()
    assert tuple(data.keys()) == IMPACT_KEYS
    assert data["site"] == city
    assert data["ambient_c"] == temperature
    assert data["gain_mwh_an"] > 0.0
    assert data["economies_fcfa_an"] > 0.0


@pytest.mark.parametrize("city", SENEGAL_ZONES)
def test_impact_route_returns_200_for_every_senegal_zone(client: TestClient, city: str) -> None:
    response = client.get("/api/impact", params={"city": city, "temperature_c": 20.0})
    assert response.status_code == 200
    data = response.json()
    assert tuple(data.keys()) == IMPACT_KEYS
    assert data["site"] == city
    assert data["ambient_c"] == 20.0
    assert data["gain_mwh_an"] > 0.0


@pytest.mark.parametrize("material_id,chern", ((SOLAR_CHAMPION_ID, 1), (MXENE_CHAMPION_ID, -1)))
def test_pipeline_route_returns_200_for_champions(client: TestClient, material_id: str, chern: int) -> None:
    response = client.post(
        "/api/pipeline/run",
        json={"material_id": material_id, "temperature_c": 47.0, "city": "Matam"},
    )
    assert response.status_code == 200
    data = response.json()
    assert tuple(data.keys()) == PIPELINE_KEYS
    assert data["material"]["id"] == material_id
    assert data["material"]["is_champion"] is True
    assert data["berry"]["chern"] == chern
    assert data["berry"]["grid"] == [12, 12]
    assert len(data["band_structure"]["bands"]) >= 2
    assert len(data["jv"]["voltage"]) > 2
    assert data["jv"]["voc"] > 0.0
    assert len(data["ragone"]["energy_wh_kg"]) == 4
    assert data["impact"]["gain_mwh_an"] > 0.0
    assert data["plant"]["capacity_kwc"] == PLANT_CAPACITY_KWC


def test_pipeline_route_accepts_any_material(client: TestClient) -> None:
    for material_id in ("UAM-0001", "UAM-1000", "UAM-2000"):
        response = client.post(
            "/api/pipeline/run",
            json={"material_id": material_id, "temperature_c": 31.0, "city": "Dakar"},
        )
        assert response.status_code == 200, response.text[:400]
        data = response.json()
        assert data["material"]["id"] == material_id
        assert len(data["band_structure"]["bands"]) >= 2
        assert len(data["jv"]["voltage"]) > 2


def test_chart_data_and_figure_routes_return_200(client: TestClient) -> None:
    charts = client.get(
        "/api/chart-data/UAM-1384",
        params={"temperature_c": 47.0, "city": "Matam"},
    )
    assert charts.status_code == 200
    assert len(charts.json()["band_structure"]["bands"]) >= 2
    figure = client.get("/api/figures/night11")
    assert figure.status_code == 200
    assert figure.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_report_pdf_route_returns_200(client: TestClient) -> None:
    response = client.get("/api/report/pdf")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.content[:4] == b"%PDF"
    assert len(response.content) > 100_000


def test_invalid_requests_are_rejected(client: TestClient) -> None:
    assert client.post(
        "/api/pipeline/run",
        json={"material_id": SOLAR_CHAMPION_ID, "temperature_c": 19.0, "city": "Matam"},
    ).status_code == 422
    assert client.post(
        "/api/pipeline/run",
        json={"material_id": SOLAR_CHAMPION_ID, "temperature_c": 47.0, "city": "Paris"},
    ).status_code == 400
    assert client.post(
        "/api/pipeline/run",
        json={"material_id": "UAM-9999", "temperature_c": 47.0, "city": "Matam"},
    ).status_code == 404
    assert client.get("/api/impact", params={"city": "Paris"}).status_code == 400


def test_every_declared_api_route_returns_200(client: TestClient) -> None:
    routes = (
        ("/", {}),
        ("/api/health", {}),
        ("/api/catalog", {}),
        ("/api/material/UAM-1384", {}),
        ("/api/impact", {"city": "Saint-Louis", "temperature_c": 37.0}),
        ("/api/chart-data/UAM-1517", {"temperature_c": 37.0, "city": "Saint-Louis"}),
        ("/api/figures/night11", {}),
        ("/api/report/pdf", {}),
        ("/static/index.html", {}),
        ("/static/app.js", {}),
    )
    for path, params in routes:
        assert client.get(path, params=params).status_code == 200, path
    assert client.post(
        "/api/pipeline/run",
        json={"material_id": SOLAR_CHAMPION_ID, "temperature_c": 47.0, "city": "Matam"},
    ).status_code == 200
