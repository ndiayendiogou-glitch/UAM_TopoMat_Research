"""UAM-TOPOMAT -- Moteur de theorie de jauge sur reseau (Fukui-Hatsugai-Suzuki).

Calcul discret, sur la premiere zone de Brillouin, des liens de jauge, de la
courbure de Berry et de l'invariant de Chern d'une bande isolee.

Method FHS : les vecteurs propres sont evaluator sur une grille periodique
k_i = (i + shift)/N_i. Les liens de jauge sont les recouvrements normalises

    U_mu(k) = <psi_n(k)|psi_n(k + delta_k_mu)> / |...|

et le champ sur une plaquette elementaire est

    F_12(k) = Arg[ U_1(k) U_2(k+1) U_1(k+2)^* U_2(k)^* ]  ∈ (-pi, pi].

La courbure physique vaut Omega(k) = F_12(k) / dA avec dA l'aire d'une maille
de la grille dans l'espace reciproque. Le nombre de Chern

    C_n = (1/2pi) sum_k F_12(k) ∈ Z

est exactement quantifyie, independant de la grille et de la jauge des
vecteurs propres. Dependances : numpy uniquement (aucune dependance a SciPy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

TWO_PI = 2.0 * np.pi
_DEFAULT_TOL = 1e-13


def extract_band(eigenvectors, band: int = 0) -> np.ndarray:
    """Extrait les vecteurs propres d'une bande sur une grille (N1, N2, ...).

    Accepte soit les matrices completes de forme (N1, N2, nb, nb), soit une
    bande deja extraite de forme (N1, N2, nb_comp).
    """
    array = np.asarray(eigenvectors)
    if array.ndim == 4:
        if not 0 <= band < array.shape[3]:
            raise ValueError(f"Indice de bande {band} hors limites.")
        return array[:, :, :, band]
    if array.ndim == 3:
        return array
    raise ValueError(
        "eigenvectors doit etre de forme (N1, N2, nb, nb) ou (N1, N2, nb_comp)."
    )


def gauge_links(eigenvectors, axis: int, band: int = 0) -> np.ndarray:
    """Liens de jauge U_mu(k) le long de l'axe `axis` (0 ou 1).

    Le decalage est periodique : le point k + delta_k_mu est replie sur la
    grille, ce qui realize un tore. Le recouvrement est divise par son module,
    donc |U_mu(k)| = 1 et le resultat est invariant de jauge a un facteur de
    phase pres (la courbure, elle, est strictement invariante).
    """
    if axis not in (0, 1):
        raise ValueError("axis doit valoir 0 ou 1.")
    psi = extract_band(eigenvectors, band)
    if psi.ndim == 2:
        psi = psi[..., None]
    if psi.shape[0] < 2 or psi.shape[1] < 2:
        raise ValueError("La grille doit contenir au moins 2x2 points.")

    shifted = np.roll(psi, -1, axis=axis)
    overlap = np.einsum("ijk,ijk->ij", psi.conj(), shifted)
    magnitude = np.abs(overlap)
    if np.any(magnitude < _DEFAULT_TOL):
        raise ValueError(
            "Recouvrement nul (singularite de jauge ou degenerescence)."
        )
    return overlap / magnitude


def plaquette_field(links_1: np.ndarray, links_2: np.ndarray) -> np.ndarray:
    """Champ de Berry F_12(k) sur chaque plaquette (N1, N2), reel.

    F_12(k) = Arg[U_1(k) U_2(k+1) U_1(k+2)^* U_2(k)^*], avec la branche
    principale de l'argument, donc F_12 ∈ (-pi, pi].
    """
    u1 = np.asarray(links_1)
    u2 = np.asarray(links_2)
    if u1.shape != u2.shape:
        raise ValueError("links_1 et links_2 doivent avoir la meme forme.")
    product = (
        u1
        * np.roll(u2, -1, axis=0)
        * np.conj(np.roll(u1, -1, axis=1))
        * np.conj(u2)
    )
    return np.angle(product)


def chern_number(field: np.ndarray) -> float:
    """Nombre de Chern C = (1/2pi) sum_k F_12(k) (non arrondi)."""
    return float(np.asarray(field, dtype=float).sum() / TWO_PI)


def reciprocal_area(reciprocal_vectors) -> float:
    """Aire de la zone de Brillouin |b_1 × b_2| (2D)."""
    vectors = np.asarray(reciprocal_vectors, dtype=float)
    b1, b2 = vectors[0, :2], vectors[1, :2]
    area = abs(float(b1[0] * b2[1] - b1[1] * b2[0]))
    if area <= 0.0:
        raise ValueError("Vecteurs reciproques 2D lineairement dependants.")
    return area


def berry_curvature(
    field: np.ndarray,
    reciprocal_vectors,
    n1: int,
    n2: int,
) -> np.ndarray:
    """Courbure physique Omega(k) = F_12(k)/dA, en unite d'aire de k."""
    n1, n2 = int(n1), int(n2)
    if n1 < 2 or n2 < 2:
        raise ValueError("La grille doit contenir au moins 2x2 points.")
    cell_area = reciprocal_area(reciprocal_vectors) / (n1 * n2)
    return np.asarray(field, dtype=float) / cell_area


def k_grid(
    reciprocal_vectors,
    n1: int = 48,
    n2: int = 48,
    shift: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Grille de points k cartesiens sur le tore reciproque.

    Retourne (fractions, k_cartesiens) de forme (N1, N2, 2). Les fractions
    sont relatives a (b_1, b_2) ; les coordonnees sont celles de la premiere
    zone de Brillouin.
    """
    n1, n2 = int(n1), int(n2)
    if n1 < 2 or n2 < 2:
        raise ValueError("La grille doit contenir au moins 2x2 points.")
    f1 = (np.arange(n1, dtype=float) + shift) / n1
    f2 = (np.arange(n2, dtype=float) + shift) / n2
    fractions = np.stack(np.meshgrid(f1, f2, indexing="ij"), axis=-1)
    vectors = np.asarray(reciprocal_vectors, dtype=float)[:2, :2]
    cartesian = fractions @ vectors
    return fractions, cartesian


def evaluate_eigenstates(
    model: Any,
    reciprocal_vectors,
    n1: int = 48,
    n2: int = 48,
    shift: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Diagonalise H(k) sur toute la grille.

    Retourne (energies, vecteurs propres, fractions, k_cartesiens) avec
    energies de forme (N1, N2, nb) et vecteurs (N1, N2, nb, nb).
    """
    _, cartesian = k_grid(reciprocal_vectors, n1, n2, shift)
    n_orb = int(np.asarray(model.hamiltonian(np.zeros(3))).shape[0])
    energies = np.empty((n1, n2, n_orb), dtype=float)
    eigenvectors = np.empty((n1, n2, n_orb, n_orb), dtype=complex)
    for i in range(n1):
        for j in range(n2):
            kx, ky = cartesian[i, j]
            values, columns = np.linalg.eigh(
                np.asarray(model.hamiltonian(np.array([kx, ky, 0.0])))
            )
            energies[i, j] = values
            eigenvectors[i, j] = columns
    fractions, _ = k_grid(reciprocal_vectors, n1, n2, shift)
    return energies, eigenvectors, fractions, cartesian


@dataclass
class BerryChernResult:
    """Resultat complet d'un calcul FHS pour une bande."""

    band: int
    n1: int
    n2: int
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    fractions: np.ndarray
    kpoints: np.ndarray
    links_1: np.ndarray
    links_2: np.ndarray
    field: np.ndarray
    curvature: np.ndarray
    chern: float
    chern_integer: int
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"BerryChernResult(band={self.band}, grid=({self.n1}, {self.n2}), "
            f"C={self.chern:.12f} -> {self.chern_integer:+d})"
        )


def berry_chern(
    model: Any,
    reciprocal_vectors,
    n1: int = 48,
    n2: int = 48,
    band: int = 0,
    shift: float = 0.5,
) -> BerryChernResult:
    """Pipeline complet FHS : liens, courbure et nombre de Chern d'une bande."""
    energies, eigenvectors, fractions, cartesian = evaluate_eigenstates(
        model, reciprocal_vectors, n1, n2, shift
    )
    u1 = gauge_links(eigenvectors, 0, band)
    u2 = gauge_links(eigenvectors, 1, band)
    field = plaquette_field(u1, u2)
    chern = chern_number(field)
    rounded = int(np.rint(chern))
    if abs(chern - rounded) > 1e-6:
        raise ValueError(
            f"Chern non quantifyie : C={chern:.9f}. La bande n'est pas isolee "
            "ou la grille est trop grossiere."
        )
    curvature = berry_curvature(field, reciprocal_vectors, n1, n2)
    return BerryChernResult(
        band=int(band),
        n1=int(n1),
        n2=int(n2),
        eigenvalues=energies,
        eigenvectors=eigenvectors,
        fractions=fractions,
        kpoints=cartesian,
        links_1=u1,
        links_2=u2,
        field=field,
        curvature=curvature,
        chern=chern,
        chern_integer=rounded,
        metadata={
            "cell_area": reciprocal_area(reciprocal_vectors) / (int(n1) * int(n2)),
            "field_min": float(field.min()),
            "field_max": float(field.max()),
            "curvature_min": float(curvature.min()),
            "curvature_max": float(curvature.max()),
        },
    )


def random_gauge_transform(
    eigenvectors: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Applique |psi_n(k)> -> e^{i theta(k)} |psi_n(k)> avec theta aleatoire.

    Utile pour verifier l'invariance de jauge exacte de la courbure de Berry.
    La phase est tiree uniformement dans [0, 2pi) et independente a chaque
    point de la grille.
    """
    if rng is None:
        rng = np.random.default_rng(2026)
    array = np.asarray(eigenvectors)
    phases = np.exp(1j * rng.uniform(0.0, TWO_PI, size=array.shape[:2]))
    if array.ndim == 4:
        return array * phases[..., None, None]
    if array.ndim == 3:
        return array * phases[..., None]
    raise ValueError("eigenvectors doit etre de rang 3 ou 4.")
