"""UAM-TOPOMAT -- Nuit 12 Finale : audit d'intégrité de tous les livrables.

Vérifie l'existence, la structure et la cohérence de :
  * logo UAM (interface web, export PDF, documents LaTeX),
  * article scientifique complet (17 pages) et son source LaTeX,
  * poster A0 une page et son source LaTeX,
  * fiche technique des champions, script de démonstration et dossier jury,
  * routes web FastAPI (200 OK) et figures 300 dpi,
  * entrée du journal de recherche.
"""

from __future__ import annotations

import re
import struct
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.server import app  # noqa: E402
from src.export_report import build_tex  # noqa: E402

PAPER = ROOT / "paper_latex"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
STATIC = ROOT / "app" / "static"

ARTICLE_PDF = RESULTS / "Article_UAM_TOPOMAT_2026.pdf"
POSTER_PDF = RESULTS / "Poster_UAM_TOPOMAT_2026.pdf"
FICHE_PDF = RESULTS / "UAM_TopoMat_Fiche_Champions.pdf"
ARTICLE_TEX = PAPER / "main.tex"
POSTER_TEX = PAPER / "poster.tex"
DOSSIER = ROOT / "DOSSIER_SOUTENANCE_JURY.md"
DEMO_BAT = ROOT / "LANCER_DEMO_JURY.bat"
JOURNAL = ROOT / "JOURNAL_RECHERCHE.md"

FIGURE_NAMES = (
    "night1_bands_dos.png",
    "night2_haldane_chern_phase.png",
    "night3_edge_states_ribbon.png",
    "night4_solar_efficiency_vs_temp.png",
    "night5_sahel_thermal_balance.png",
    "night6_jv_curves_sahel.png",
    "night7_ragone_mxene_sahel.png",
    "night8_materials_landscape.png",
    "night9_pareto_discovery.png",
    "night11_senegal_impact.png",
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as instance:
        yield instance


def _png_dpi(path: Path) -> float:
    """Lit la résolution du chunk pHYs d'un PNG (identique aux nuits précédentes)."""
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"PNG invalide : {path}"
    index = data.find(b"pHYs")
    assert index > 0, f"Chunk pHYs absent : {path}"
    pixels_x, pixels_y, unit = struct.unpack(">IIB", data[index + 4 : index + 13])
    assert unit == 1, f"Unité pHYs inattendue : {path}"
    assert pixels_x == pixels_y, f"Résolution anisotrope : {path}"
    return pixels_x * 0.0254


def _pdf_pages(path: Path) -> int:
    """Nombre de pages d'un PDF (pypdf si disponible, repli sur l'objet brut)."""
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(path)).pages)
    except Exception:
        data = path.read_bytes()
        return len(re.findall(rb"/Type\s*/Page\b", data))


def _balanced_braces(text: str) -> bool:
    depth = 0
    escaped = False
    for char in text:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _balanced_environments(text: str) -> bool:
    stack: list[str] = []
    for begin, end in re.findall(r"\\(begin|end)\{([A-Za-z*]+)\}", text):
        if begin == "begin":
            stack.append(end)
        elif not stack or stack.pop() != end:
            return False
    return not stack


# ---------------------------------------------------------------------------
# 1. Logo officiel UAM
# ---------------------------------------------------------------------------

def test_logo_assets_exist_are_valid_and_identical() -> None:
    web_logo = STATIC / "logo_uam.png"
    tex_logo = PAPER / "logo_uam.png"
    for path in (web_logo, tex_logo):
        assert path.exists(), path
        assert path.stat().st_size > 10_000, path
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", path
    assert web_logo.read_bytes() == tex_logo.read_bytes()
    assert _png_dpi(web_logo) > 0


def test_logo_is_displayed_in_web_header() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'src="/static/logo_uam.png"' in html
    assert 'class="brand-logo"' in html
    for token in ("UAM-TOPOMAT", "Cellule Physique UFR STA", "Université Amadou Mahtar Mbow",
                  "Ndiogou Ndiaye", "Laboratoire Numérique Souverain",
                  "Podor", "Tambacounda", "Ziguinchor", "Région personnalisée"):
        assert token in html, token
    assert "Nuit" not in html
    assert "https://" not in html
    assert "cdn." not in html.lower()


def test_logo_is_integrated_in_pdf_export_generator() -> None:
    source = build_tex()
    assert "logo_uam.png" in source
    assert r"\includegraphics" in source
    assert r"\graphicspath{{./}{../results/figures/}{../app/static/}}" in source
    for token in ("UNIVERSITÉ AMADOU MAHTAR MBOW", "UFR SCIENCES ET TECHNOLOGIES AVANCÉES",
                  "CELLULE PHYSIQUE", "Réalisé par : Ndiogou Ndiaye", "2026 -- 2027"):
        assert token in source, token


def test_pdf_sources_have_no_nuit_mentions() -> None:
    sources = (
        ARTICLE_TEX.read_text(encoding="utf-8"),
        POSTER_TEX.read_text(encoding="utf-8"),
        build_tex(),
    )
    for source in sources:
        assert not re.search(r"\b[Nn]uits?\b", source)


# ---------------------------------------------------------------------------
# 2. Article scientifique
# ---------------------------------------------------------------------------

def test_article_pdf_exists_and_has_expected_page_count() -> None:
    assert ARTICLE_PDF.exists(), ARTICLE_PDF
    assert ARTICLE_PDF.read_bytes()[:4] == b"%PDF"
    assert ARTICLE_PDF.stat().st_size > 1_000_000
    pages = _pdf_pages(ARTICLE_PDF)
    assert 15 <= pages <= 20, f"L'article doit faire 15 à 20 pages, il en fait {pages}"


def test_article_latex_source_is_structurally_sound() -> None:
    source = ARTICLE_TEX.read_text(encoding="utf-8")
    assert len(source) > 20_000
    assert r"\begin{document}" in source and r"\end{document}" in source
    assert source.count(r"\section{") + source.count(r"\section*{") >= 8
    assert source.count(r"\begin{figure}[htbp]") >= 10
    assert source.count(r"\begin{equation}") >= 10
    assert "logo_uam.png" in source
    for name in FIGURE_NAMES:
        assert name in source, name
    assert "TODO" not in source and "???" not in source
    assert _balanced_braces(source)
    assert _balanced_environments(source)


def test_article_covers_all_scientific_requirements() -> None:
    source = ARTICLE_TEX.read_text(encoding="utf-8")
    required = (
        "Plateforme Numérique Souveraine et Automatisée",
        "paradoxe énergétique sahélien",
        "Twin Transition",
        "Architecture du pipeline automatisé",
        "Cadre théorique et preuves formelles",
        "Hellmann--Feynman",
        "Shockley--Queisser",
        "Scharfetter--Gummel",
        "Capacité quantique des MXènes",
        "2\\,000 matériaux",
        "NSGA-II",
        "UAM-1384",
        "UAM-1517",
        "126,9",
        "69,8",
        "46,2",
        "44 à 49",
        "Bibliographie",
        "Réalisé par : Ndiogou Ndiaye",
        "2026 – 2027",
    )
    for token in required:
        assert token in source, token


# ---------------------------------------------------------------------------
# 3. Poster A0
# ---------------------------------------------------------------------------

def test_poster_pdf_exists_and_is_single_a0_page() -> None:
    assert POSTER_PDF.exists(), POSTER_PDF
    assert POSTER_PDF.read_bytes()[:4] == b"%PDF"
    assert POSTER_PDF.stat().st_size > 500_000
    assert _pdf_pages(POSTER_PDF) == 1
    try:
        from pypdf import PdfReader

        box = PdfReader(str(POSTER_PDF)).pages[0].mediabox
        assert float(box.width) > 2_000, f"Poster trop petit: {box.width} pt"
        assert float(box.height) > 3_000, f"Poster trop petit: {box.height} pt"
    except ImportError:
        pass


def test_poster_latex_source_contains_key_messages() -> None:
    source = POSTER_TEX.read_text(encoding="utf-8")
    assert len(source) > 5_000
    assert "a0paper" in source
    assert "logo_uam.png" in source
    for token in ("UAM-TOPOMAT", "2\\,000 matériaux", "UAM-1384", "UAM-1517",
                  "126,9", "69,8", "46,2", "44--49", "LANCER\\_DEMO\\_JURY.bat"):
        assert token in source, token
    assert _balanced_braces(source)


# ---------------------------------------------------------------------------
# 4. Fiche, démonstration, dossier jury, journal
# ---------------------------------------------------------------------------

def test_champion_fiche_pdf_is_still_valid() -> None:
    assert FICHE_PDF.exists(), FICHE_PDF
    assert FICHE_PDF.read_bytes()[:4] == b"%PDF"
    assert FICHE_PDF.stat().st_size > 100_000
    assert _pdf_pages(FICHE_PDF) >= 1


def test_demo_batch_file_is_present_and_consistent() -> None:
    assert DEMO_BAT.exists(), DEMO_BAT
    content = DEMO_BAT.read_text(encoding="utf-8", errors="replace")
    lines = [line for line in content.splitlines() if line.strip()]
    assert len(lines) == 6, lines
    for token in ("uvicorn app.server:app", "8050", "http://127.0.0.1:8050",
                  "UAM-TOPOMAT", "timeout /t 3 /nobreak > nul"):
        assert token in content, token
    assert "urllib" not in content and "powershell" not in content.lower()
    assert ":UAM_WAIT_READY" not in content and ":UAM_READY" not in content


def test_jury_dossier_is_complete() -> None:
    assert DOSSIER.exists(), DOSSIER
    content = DOSSIER.read_text(encoding="utf-8")
    assert len(content) > 12_000
    for token in ("Pitch complet (3 minutes)", "Scénario de démonstration en direct",
                  "Douze questions", "Q1.", "Q12.", "126,9", "69,8", "46,2",
                  "Ndiogou Ndiaye", "2026 – 2027"):
        assert token in content, token
    assert all(f"### Q{i}." in content for i in range(1, 13))


def test_journal_records_night12() -> None:
    content = JOURNAL.read_text(encoding="utf-8")
    assert "## Nuit 12" in content
    assert "Article" in content and "Poster" in content
    assert "pytest" in content


# ---------------------------------------------------------------------------
# 5. Figures 300 dpi et routes web
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", FIGURE_NAMES)
def test_publication_figures_are_300_dpi(name: str) -> None:
    path = FIGURES / name
    assert path.exists(), path
    assert path.stat().st_size > 100_000
    assert abs(_png_dpi(path) - 300.0) < 1.0, name


def test_every_required_web_route_returns_200(client: TestClient) -> None:
    for route in ("/", "/api/health", "/api/catalog", "/api/material/UAM-1384",
                  "/api/impact", "/api/chart-data/UAM-1517", "/api/figures/night11",
                  "/api/report/pdf", "/static/index.html", "/static/app.js",
                  "/static/logo_uam.png"):
        response = client.get(route)
        assert response.status_code == 200, f"{route} -> {response.status_code}: {response.text[:200]}"

    response = client.post("/api/pipeline/run", json={
        "material_id": "UAM-1384", "temperature_c": 47.0, "city": "Matam",
    })
    assert response.status_code == 200, response.text[:400]
    assert response.json()["material"]["id"] == "UAM-1384"


def test_logo_and_pdf_served_with_correct_content(client: TestClient) -> None:
    logo = client.get("/static/logo_uam.png")
    assert logo.status_code == 200
    assert logo.content[:8] == b"\x89PNG\r\n\x1a\n"
    report = client.get("/api/report/pdf")
    assert report.status_code == 200
    assert report.content[:4] == b"%PDF"
