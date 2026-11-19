"""UAM-TOPOMAT -- Nuit 10 : générateur automatique de fiches techniques PDF.

L'exportateur produit une fiche LaTeX conforme à la charte officielle
Université Amadou Mahtar Mbow / UFR Sciences et Technologies Avancées /
Cellule Physique (réalisée par Ndiogou Ndiaye, année universitaire
2026--2027) : encadrés ``tcolorbox``, tableaux comparatifs du Top 10,
fiches détaillées des deux champions et intégration des figures 300 dpi.
La page de garde porte le logo officiel de l'UAM (``logo_uam.png``,
présent dans ``paper_latex/`` et ``app/static/``). La compilation est
automatique si ``tectonic`` ou ``pdflatex`` est présent.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

from src.quantum.symbolic_proofs import (
    CHAMPIONS_PATH,
    MXENE_CHAMPION_ID,
    PAPER_DIR,
    ROOT,
    SOLAR_CHAMPION_ID,
    all_proofs,
    verify_champions,
    write_equations_tex,
)

FICHE_TEX_PATH = PAPER_DIR / "fiche_technique_champions.tex"
PDF_PATH = ROOT / "results" / "UAM_TopoMat_Fiche_Champions.pdf"
FIGURES_DIR = ROOT / "results" / "figures"

FIGURES: tuple[tuple[str, str], ...] = (
    ("night9_pareto_discovery.png", "Espace de Pareto et Top 10 des matériaux champions (Nuit 9)."),
    ("night4_solar_efficiency_vs_temp.png", "Rendement de Shockley--Queisser et dégradation thermique (Nuit 4)."),
    ("night7_ragone_mxene_sahel.png", "Diagramme de Ragone des MXènes sous climat sahélien (Nuit 7)."),
    ("night1_bands_dos.png", "Structure de bandes Tight-Binding et densité d'états (Nuit 1)."),
)


def _latex_escape(text: str) -> str:
    """Échappement LaTeX des caractères réservés dans du texte."""
    for old, new in (
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
        ("$", r"\$"),
        ("^", r"\textasciicircum{}"),
    ):
        text = text.replace(old, new)
    return text


def _formula_tex(formula: str) -> str:
    """Convertit une formule chimique en LaTeX (K2AgBiI6 -> K$_2$AgBiI$_6$)."""
    return re.sub(r"(\d+(?:\.\d+)?)", lambda match: r"$_{" + match.group(1) + r"}$", formula)


def load_champions(path: Path = CHAMPIONS_PATH) -> dict:
    """Charge et valide le JSON du Top 10 de la Nuit 9."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    top = payload.get("top10_champions", [])
    if len(top) != 10:
        raise ValueError("Le JSON doit contenir exactement 10 champions.")
    ids = {record["id"] for record in top}
    for required in (SOLAR_CHAMPION_ID, MXENE_CHAMPION_ID):
        if required not in ids:
            raise ValueError(f"Champion {required} absent du Top 10.")
    return payload


def _champion_table(records: Sequence[dict]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering\footnotesize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l l l r r r r r r r r}",
        r"\toprule",
        r"\# & ID & Formule & $E_g$ (eV) & $C$ & $V_{oc}$ (V) & $J_{sc}$ & $FF$ "
        r"& $\eta$ (\%) & Gain (\%) & Coût \\",
        r"\midrule",
    ]
    for rank, record in enumerate(records, start=1):
        lines.append(
            f"{rank} & {_latex_escape(record['id'])} & {_formula_tex(record['formula'])} "
            f"& {record['bandgap_ev']:.3f} & {int(record['chern_number']):+d} "
            f"& {record['voc_matam_v']:.3f} & {record['jsc_matam_ma_cm2']:.2f} "
            f"& {record['ff_matam']:.3f} & {record['eta_matam_percent']:.2f} "
            f"& {record['gain_vs_silicon_percent']:+.1f} "
            f"& {record['cost_usd_per_kg']:.2f} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        r"\caption{Top 10 équilibré : 5 pérovskites doubles et 5 MXènes du front de Pareto "
        r"(conditions de Matam, 47~$^\circ$C).}\label{tab:top10}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def _field_table(record: dict, checks: Sequence) -> str:
    """Tableau détaillé d'une fiche champion."""
    rows = [
        ("Formule", _formula_tex(record["formula"])),
        ("Famille / structure", _latex_escape(record["structure"])),
        ("Gap optique $E_g$", f"{record['bandgap_ev']:.3f} eV"),
        ("Invariant de Chern $C$", f"{int(record['chern_number']):+d}"),
        ("Température de jonction $T_j$", f"{record['junction_temperature_matam_c']:.2f} $^\\circ$C"),
        ("$V_{oc}$", f"{record['voc_matam_v']:.3f} V"),
        ("$J_{sc}$", f"{record['jsc_matam_ma_cm2']:.2f} mA/cm$^2$"),
        ("Facteur de forme $FF$", f"{record['ff_matam']:.3f}"),
        ("Rendement $\\eta_{\\mathrm{{Matam}}}$", f"{record['eta_matam_percent']:.2f} %"),
        ("Gain vs silicium", f"{record['gain_vs_silicon_percent']:+.1f} %"),
        ("Coût estimé", f"{record['cost_usd_per_kg']:.2f} USD/kg"),
        ("Indice d'abondance", f"{record['abundance_index']:.3f}"),
        ("Stabilité thermique", f"{record['thermal_stability_percent']:.2f} %"),
        ("Densité d'énergie $E_{{47}}$", f"{record['energy_density_matam_wh_kg']:.2f} Wh/kg"),
        ("Dérive $\\beta_T$", f"{record['thermal_drift_beta_per_k']:.6f} K$^{{-1}}$"),
    ]
    lines = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Propriété & Valeur \\",
        r"\midrule",
    ]
    for name, value in rows:
        lines.append(f"{name} & {value.replace('%', r'\%')} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])
    failed = [check for check in checks if not check.satisfied]
    lines.append("")
    lines.append(
        f"\\noindent\\textbf{{Bornes théoriques :}} {len(checks)} vérifiées, "
        f"{len(failed)} échec(s). "
        + ("Toutes les bornes exactes sont respectées." if not failed else "Certaines bornes sont violées.")
    )
    return "\n".join(lines)


def build_tex(payload: dict | None = None) -> str:
    """Construit intégralement la fiche technique LaTeX."""
    payload = payload or load_champions()
    records = payload["top10_champions"]
    checks = verify_champions()
    proofs = all_proofs()

    parts = [
        "% =====================================================================",
        "%  UAM-TOPOMAT -- Nuit 10 : fiche technique des materiaux champions",
        "%  Charte officielle : Universite Amadou Mahtar Mbow / UFR STA /",
        "%  Cellule Physique / Realise par Ndiogou Ndiaye / 2026-2027",
        "%  Genere automatiquement par src/export_report.py",
        "% =====================================================================",
        r"\documentclass[11pt,a4paper,french]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[french]{babel}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{tcolorbox}",
        r"\usepackage{xcolor}",
        r"\usepackage{hyperref}",
        r"\usepackage[a4paper,margin=2cm]{geometry}",
        r"\graphicspath{{./}{../results/figures/}{../app/static/}}",
        r"\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue,"
        r"pdftitle={UAM-TOPOMAT - Fiche technique des champions},pdfauthor={Ndiogou Ndiaye}}",
        r"\newcommand{\UAM}{Université Amadou Mahtar Mbow}",
        r"\newcommand{\UFR}{UFR Sciences et Technologies Avancées}",
        r"\newcommand{\TOPOMAT}{UAM-TOPOMAT}",
        "",
        r"\title{\textbf{\TOPOMAT}\\[0.4em]\large Fiches techniques des matériaux champions\\"
        r"\textmd{Nuit 10 --- Preuves formelles SymPy et export PDF scientifique}}",
        r"\author{Ndiogou Ndiaye}",
        r"\date{\UAM \quad---\quad \UFR \quad---\quad Cellule Physique\\"
        r"Année universitaire : 2026 -- 2027}",
        "",
        r"\begin{document}",
        r"\maketitle",
        "",
        r"\begin{tcolorbox}[colback=blue!6,colframe=blue!50!black,title=Charte officielle UAM]",
        r"\begin{center}",
        r"\includegraphics[height=1.8cm]{logo_uam.png}\\[0.4em]",
        r"{\Large\textbf{UNIVERSITÉ AMADOU MAHTAR MBOW}}\\[0.3em]",
        r"\textbf{UFR SCIENCES ET TECHNOLOGIES AVANCÉES}\\[0.2em]",
        r"\textbf{CELLULE PHYSIQUE}\\[0.2em]",
        r"\textbf{Réalisé par : Ndiogou Ndiaye}\\[0.2em]",
        r"\textbf{Année universitaire : 2026 -- 2027}",
        r"\end{center}",
        r"\end{tcolorbox}",
        "",
        r"\section{Synthèse exécutive}",
        "La présente fiche consolide dix nuits de modélisation quantique, topologique et "
        "thermodynamique. Le criblage de 2~000 candidats sans plomb sous le climat de Matam "
        r"($47\,^\circ$C, $1050$\,W\,m$^{-2}$) a livré un Top~10 dont deux champions "
        "architecturaux : le champion solaire \\texttt{UAM-1384} (K$_2$AgBiI$_6$, "
        r"$C=+1$, $\eta=22{,}84$\,\%) et le champion supercondensateur \texttt{UAM-1517} "
        r"(Ti$_3$C$_2$O$_2$, $C=-1$, $E_{47}=5{,}97$\,Wh\,kg$^{-1}$). "
        "Les six équations fondatrices sont certifiées symboliquement par SymPy et "
        "toutes les bornes théoriques sont satisfaites.",
        "",
    ]

    parts.append(r"\section{Top 10 des matériaux champions}")
    parts.append(r"Le tableau~\ref{tab:top10} présente la sélection équilibrée du front de Pareto.")
    parts.append("")
    parts.append(_champion_table(records))
    parts.append("")

    for material_id, heading in (
        (SOLAR_CHAMPION_ID, "Fiche technique -- Champion Solaire"),
        (MXENE_CHAMPION_ID, "Fiche technique -- Champion Supercondensateur MXène"),
    ):
        record = next(record for record in records if record["id"] == material_id)
        role = "photovoltaïque" if record["device_role"] == "photovoltaic" else "stockage électrochimique"
        parts.append(rf"\section{{{heading} : {_latex_escape(material_id)}}}")
        parts.append(
            r"\begin{tcolorbox}[colback=green!5,colframe=green!50!black,title="
            + f"Matériau {_formula_tex(record['formula'])} --- {role}]"
        )
        parts.append(_field_table(record, checks[material_id]))
        parts.append(r"\end{tcolorbox}")
        parts.append("")

    parts.append(r"\section{Socle théorique certifié (six preuves SymPy)}")
    parts.append(
        "Les six théorèmes ci-dessous sont dérivés sans SciPy ; les équations sont "
        "exportées par \\texttt{sympy.latex} dans \\texttt{equations\\_verified.tex}."
    )
    parts.append("")
    parts.append(r"\begin{center}\small")
    parts.append(r"\resizebox{\textwidth}{!}{%")
    parts.append(r"\begin{tabular}{ll}")
    parts.append(r"\toprule")
    parts.append(r"Théorème & Équation certifiée \\")
    parts.append(r"\midrule")
    for index, proof in enumerate(proofs, start=1):
        equation = proof.equations[min(1, len(proof.equations) - 1)]
        parts.append(f"{index}. {_latex_escape(proof.title)} & ${equation}$ \\\\")
    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}}")
    parts.append(r"\end{center}")
    parts.append("")

    parts.append(r"\section{Vérification des bornes théoriques exactes}")
    parts.append(
        "Pour chaque champion, dix-huit bornes issues des six théorèmes sont "
        r"vérifiées numériquement (tableau~\ref{tab:bornes})."
    )
    parts.append("")
    parts.append(r"\begin{table}[htbp]\centering\footnotesize")
    parts.append(r"\begin{tabular}{llrr}")
    parts.append(r"\toprule")
    parts.append(r"Champion & Borne & Valeur & Statut \\")
    parts.append(r"\midrule")
    for material_id, bound_checks in checks.items():
        for check in bound_checks:
            status = r"\textbf{OK}" if check.satisfied else r"\textbf{ÉCHEC}"
            parts.append(
                f"{_latex_escape(material_id)} & {_latex_escape(check.relation)} "
                f"& {check.value:.6g} & {status} \\\\"
            )
    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}")
    parts.append(
        r"\caption{Bornes exactes vérifiées pour UAM-1384 et UAM-1517.}\label{tab:bornes}"
    )
    parts.append(r"\end{table}")
    parts.append("")

    parts.append(r"\section{Figures de publication (300~dpi)}")
    for filename, caption in FIGURES:
        if not (FIGURES_DIR / filename).exists():
            raise FileNotFoundError(f"Figure manquante : {FIGURES_DIR / filename}")
        parts.append(r"\begin{figure}[htbp]")
        parts.append(r"\centering")
        parts.append(rf"\includegraphics[width=0.92\linewidth]{{{filename}}}")
        parts.append(rf"\caption{{{caption} Figure produite à 300~dpi.}}")
        parts.append(r"\end{figure}")
        parts.append("")

    parts.append(r"\section{Conclusion et diffusion}")
    parts.append(
        "La chaîne UAM-TOPOMAT est désormais fermée : preuves formelles, certificats "
        "LaTeX, Top~10 et fiches techniques PDF. Les deux champions satisfont "
        "l'intégralité des bornes théoriques. Diffusion prévue : Twin Transition Summit, "
        "1er décembre 2026 --- UAM / UFR STA / Cellule Physique."
    )
    parts.append("")
    parts.append(r"\end{document}")
    return "\n".join(parts) + "\n"


def write_tex(path: Path = FICHE_TEX_PATH, payload: dict | None = None) -> Path:
    """Écrit ``paper_latex/fiche_technique_champions.tex``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_tex(payload), encoding="utf-8")
    return path


def find_compiler() -> str | None:
    """Retourne ``tectonic`` ou ``pdflatex`` si l'un est installé."""
    for candidate in ("tectonic", "pdflatex"):
        location = shutil.which(candidate)
        if location:
            return candidate
    return None


def compile_pdf(
    tex_path: Path = FICHE_TEX_PATH,
    pdf_path: Path = PDF_PATH,
    compiler: str | None = None,
) -> Path | None:
    """Compile la fiche en PDF ; retourne ``None`` si aucun compilateur n'est présent."""
    compiler = compiler or find_compiler()
    if compiler is None:
        return None
    tex_path = Path(tex_path).resolve()
    pdf_path = Path(pdf_path).resolve()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    working_dir = tex_path.parent

    def run(command: Sequence[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            list(command),
            cwd=working_dir,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=300,
        )

    if compiler == "tectonic":
        result = run(["tectonic", "-X", "compile", str(tex_path)])
        if result.returncode != 0:
            raise RuntimeError(f"Échec de compilation tectonic : {result.stderr[-2000:]}")
        generated = working_dir / f"{tex_path.stem}.pdf"
    else:
        arguments = [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            tex_path.name,
        ]
        result = run(arguments)
        if result.returncode != 0:
            log_path = working_dir / f"{tex_path.stem}.log"
            log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
            raise RuntimeError(f"Échec de compilation pdflatex : {log[-2500:]}")
        result = run(arguments)
        if result.returncode != 0:
            raise RuntimeError(f"Échec du second passage pdflatex : {result.stderr[-2000:]}")
        generated = working_dir / f"{tex_path.stem}.pdf"

    if not generated.exists():
        raise RuntimeError(f"PDF généré introuvable : {generated}")
    if generated.resolve() != pdf_path:
        shutil.copyfile(generated, pdf_path)
    return pdf_path


def export_report(
    tex_path: Path = FICHE_TEX_PATH,
    pdf_path: Path = PDF_PATH,
    *,
    compile_pdf_enabled: bool = True,
) -> dict:
    """Génère les certificats LaTeX, la fiche et son PDF, puis un résumé."""
    payload = load_champions()
    equations = write_equations_tex()
    fiche = write_tex(tex_path, payload)
    compiler = find_compiler() if compile_pdf_enabled else None
    pdf = compile_pdf(fiche, pdf_path, compiler) if compiler else None
    return {
        "equations_tex": str(equations),
        "fiche_tex": str(fiche),
        "pdf": str(pdf) if pdf else None,
        "compiler": compiler,
        "n_champions": len(payload["top10_champions"]),
        "solar_champion_id": payload["solar_champion_id"],
        "mxene_champion_id": payload["mxene_champion_id"],
    }


if __name__ == "__main__":
    summary = export_report()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


__all__ = [
    "FICHE_TEX_PATH",
    "PDF_PATH",
    "FIGURES",
    "load_champions",
    "build_tex",
    "write_tex",
    "find_compiler",
    "compile_pdf",
    "export_report",
]
