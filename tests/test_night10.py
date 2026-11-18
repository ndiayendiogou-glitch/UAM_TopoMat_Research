"""UAM-TOPOMAT -- Validation de la Nuit 10.

Tests de l'exactitude des six preuves formelles SymPy, de la conformité
des certificats LaTeX, des bornes théoriques des deux champions UAM-1384
et UAM-1517, de la génération de la fiche UAM et de sa compilation PDF.
"""

from __future__ import annotations

import json
import re
import struct
import sys
from functools import lru_cache
from pathlib import Path

import pytest
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.export_report import (  # noqa: E402
    FIGURES,
    FIGURES_DIR,
    FICHE_TEX_PATH,
    PDF_PATH,
    build_tex,
    export_report,
    find_compiler,
    load_champions,
    write_tex,
)
from src.quantum.symbolic_proofs import (  # noqa: E402
    CHAMPION_IDS,
    EQUATIONS_PATH,
    PROOF_BUILDERS,
    all_champion_bounds_pass,
    all_proofs,
    all_proofs_exact,
    build_equations_tex,
    proof_residuals,
    verify_champions,
    write_equations_tex,
)

EXPECTED_KEYS = (
    "tight_binding",
    "berry_chern",
    "shockley_queisser",
    "stefan_boltzmann",
    "scharfetter_gummel",
    "series_capacitance",
)


def _balanced_braces(text: str) -> bool:
    """Vrai si les accolades LaTeX (non échappées) sont équilibrées."""
    depth = 0
    index = 0
    while index < len(text):
        character = text[index]
        if character == "\\":
            index += 2
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth < 0:
                return False
        index += 1
    return depth == 0


def _balanced_environments(text: str) -> bool:
    r"""Vrai si chaque \begin{...} possède un \end{...} correspondant."""
    stack: list[str] = []
    for begin, name in re.findall(r"\\begin\{([A-Za-z*]+)\}|\\end\{([A-Za-z*]+)\}", text):
        if begin:
            stack.append(begin)
        elif not stack or stack.pop() != name:
            return False
    return not stack


def _png_dpi(path: Path) -> float:
    """Lit la résolution (dpi) du chunk pHYs d'un PNG."""
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
def _proofs():
    return all_proofs()


@lru_cache(maxsize=None)
def _bounds():
    return verify_champions()


@lru_cache(maxsize=None)
def _summary():
    return export_report()


# ---------------------------------------------------------------------------
# 1. Preuves formelles SymPy
# ---------------------------------------------------------------------------


def test_six_symbolic_theorems_are_formally_exact() -> None:
    proofs = _proofs()
    assert len(proofs) == len(PROOF_BUILDERS) == 6
    assert tuple(proof.key for proof in proofs) == EXPECTED_KEYS
    assert all_proofs_exact() is True
    for proof in proofs:
        assert proof.is_exact() is True
        assert len(proof.checks) >= 2
        for name, residual in proof.checks.items():
            value = complex(sp.N(residual))
            assert abs(value) < 1.0e-12, (proof.key, name, value)
    residuals = proof_residuals()
    assert set(residuals) == set(EXPECTED_KEYS)
    assert sum(len(checks) for checks in residuals.values()) >= 20


def test_symbolic_identities_cover_the_six_foundations() -> None:
    residuals = proof_residuals()
    assert residuals["tight_binding"]["Hellmann_Feynman"] == 0
    assert residuals["berry_chern"]["identite_geometrique"] == 0
    assert residuals["berry_chern"]["c_haldane_plus_un"] == 0
    assert residuals["shockley_queisser"]["point_puissance_max"] == 0
    assert residuals["stefan_boltzmann"]["derivee_bilan"] == 0
    assert residuals["scharfetter_gummel"]["einstein"] == 0
    assert residuals["series_capacitance"]["loi_serie"] == 0


def test_latex_equations_are_clean_and_renderable() -> None:
    for proof in _proofs():
        section = proof.latex_section(1)
        assert section.count(r"\begin{align}") == section.count(r"\end{align}") == 2
        assert r"\bar{\h}" not in section
        assert "None" not in section
        for equation in proof.equations:
            assert equation.strip()
            assert equation.count("$") == 0
            assert _balanced_braces(equation)


def test_no_scipy_dependency_is_introduced() -> None:
    for filename in (
        ROOT / "src" / "quantum" / "symbolic_proofs.py",
        ROOT / "src" / "export_report.py",
        ROOT / "tests" / "test_night10.py",
    ):
        source = filename.read_text(encoding="utf-8")
        imports = re.findall(r"^\s*(?:from|import)\s+([A-Za-z0-9_.]+)", source, re.MULTILINE)
        assert all(not module.startswith("scipy") for module in imports), filename


# ---------------------------------------------------------------------------
# 2. Bornes théoriques des champions
# ---------------------------------------------------------------------------


def test_both_champions_satisfy_all_exact_bounds() -> None:
    bounds = _bounds()
    assert tuple(bounds) == CHAMPION_IDS
    assert all_champion_bounds_pass() is True
    for material_id, checks in bounds.items():
        assert len(checks) == 18
        assert all(check.satisfied for check in checks)
        by_name = {check.name: check for check in checks}
        assert by_name["chern_entier"].satisfied
        assert by_name["rendement_SQ"].value <= by_name["rendement_SQ"].bound
        assert by_name["temperature_jonction"].value > 273.15 + 47.0
        assert by_name["residu_bilan"].satisfied
        assert by_name["einstein"].satisfied
    assert bounds["UAM-1384"][2].value == 1.0
    assert bounds["UAM-1517"][2].value == -1.0


def test_champions_are_the_official_night9_winners() -> None:
    payload = load_champions()
    assert payload["solar_champion_id"] == "UAM-1384"
    assert payload["mxene_champion_id"] == "UAM-1517"
    records = {record["id"]: record for record in payload["top10_champions"]}
    assert records["UAM-1384"]["formula"] == "K2AgBiI6"
    assert records["UAM-1517"]["formula"] == "Ti3C2O2"
    assert len(payload["top10_champions"]) == 10


# ---------------------------------------------------------------------------
# 3. Fichiers LaTeX générés
# ---------------------------------------------------------------------------


def test_equations_certificate_is_generated_and_syntax_valid() -> None:
    path = write_equations_tex()
    assert path == EQUATIONS_PATH
    assert path.exists()
    source = path.read_text(encoding="utf-8")
    assert len(source) > 5_000
    assert source == build_equations_tex()
    assert source.count(r"\section{") == 7
    assert r"\begin{document}" in source and r"\end{document}" in source
    assert "Universite Amadou Mahtar Mbow" in source
    assert "Ndiogou Ndiaye" in source
    assert "SymPy" in source
    assert "TODO" not in source and "???" not in source
    assert _balanced_braces(source)
    assert _balanced_environments(source)


def test_fiche_tex_respects_uam_charter_and_contains_top10() -> None:
    path = write_tex()
    assert path == FICHE_TEX_PATH
    source = path.read_text(encoding="utf-8")
    assert len(source) > 8_000
    assert source == build_tex()
    for required in (
        "UNIVERSITÉ AMADOU MAHTAR MBOW",
        "UFR SCIENCES ET TECHNOLOGIES AVANCÉES",
        "CELLULE PHYSIQUE",
        "Ndiogou Ndiaye",
        "2026 -- 2027",
        r"\usepackage{tcolorbox}",
        r"\begin{tcolorbox}",
        r"\begin{tabular}",
        r"\includegraphics",
    ):
        assert required in source, required
    payload = load_champions()
    for record in payload["top10_champions"]:
        assert record["id"] in source
    for filename, _ in FIGURES:
        assert filename in source
    assert source.count(r"\begin{figure}[htbp]") == len(FIGURES)
    assert _balanced_braces(source)
    assert _balanced_environments(source)


def test_generated_latex_is_deterministic() -> None:
    assert build_equations_tex() == build_equations_tex()
    assert build_tex() == build_tex()


def test_export_without_compilation_produces_both_tex_files(tmp_path: Path) -> None:
    summary = export_report(
        tex_path=tmp_path / "fiche.tex",
        pdf_path=tmp_path / "fiche.pdf",
        compile_pdf_enabled=False,
    )
    assert summary["n_champions"] == 10
    assert summary["compiler"] is None
    assert summary["pdf"] is None
    assert Path(summary["fiche_tex"]).exists()
    assert Path(summary["equations_tex"]).exists()


# ---------------------------------------------------------------------------
# 4. Figures 300 dpi et compilation PDF
# ---------------------------------------------------------------------------


def test_integrated_figures_are_300_dpi() -> None:
    for filename, _ in FIGURES:
        path = FIGURES_DIR / filename
        assert path.exists(), path
        assert path.stat().st_size > 20_000, path
        assert abs(_png_dpi(path) - 300.0) < 1.0, path


def test_pdf_report_is_compiled_when_a_compiler_is_available() -> None:
    compiler = find_compiler()
    if compiler is None:
        pytest.skip("Aucun compilateur LaTeX (tectonic/pdflatex) installé.")
    summary = _summary()
    assert summary["compiler"] == compiler
    assert summary["pdf"] is not None
    path = Path(summary["pdf"])
    assert path == PDF_PATH and path.exists()
    assert path.stat().st_size > 100_000
    assert path.read_bytes()[:4] == b"%PDF"
    assert path.read_bytes().count(b"/Type /Page") >= 1
    assert Path(summary["fiche_tex"]).exists()
    assert Path(summary["equations_tex"]).exists()


def test_json_metadata_certifies_night10_outputs() -> None:
    payload = load_champions()
    assert payload["metadata"]["project"] == "UAM-TOPOMAT"
    assert payload["metadata"]["n_evaluated"] == 2000
    assert payload["solar_champion_id"] in CHAMPION_IDS
    assert payload["mxene_champion_id"] in CHAMPION_IDS
    assert json.dumps(payload, ensure_ascii=False)
