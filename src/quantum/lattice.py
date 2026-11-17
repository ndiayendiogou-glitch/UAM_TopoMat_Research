"""UAM-TOPOMAT -- Moteur cristallographique et zones de Brillouin.

Generation des reseaux directs et reciproques (2D / 3D), des points de haute
symetrie et des chemins k pour le calcul de structures de bandes.

Dependances : numpy uniquement (aucune dependance a SciPy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence

import numpy as np

TWO_PI = 2.0 * np.pi


def _as_points(vectors, name: str = "points") -> np.ndarray:
    """Convertit une collection de points 2D/3D en tableau (N, 3)."""
    arr = np.asarray(vectors, dtype=float)
    if arr.ndim == 0:
        raise ValueError(f"{name} doit contenir au moins un point.")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError(
            f"{name} doit etre de forme (N, 2) ou (N, 3), recu {arr.shape}."
        )
    if arr.shape[1] == 2:
        arr = np.column_stack((arr, np.zeros(arr.shape[0], dtype=float)))
    return arr


def _as_lattice_vectors(vectors, name: str = "direct_vectors") -> np.ndarray:
    arr = _as_points(vectors, name)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} doit contenir exactement 3 vecteurs.")
    if abs(float(np.linalg.det(arr))) < 1e-14:
        raise ValueError(f"{name} est lineairement dependant (volume nul).")
    return arr


def reciprocal_lattice(direct_vectors) -> np.ndarray:
    """Vecteurs du reseau reciproque.

    b_i = 2*pi * (a_j x a_k) / (a_1 . (a_2 x a_3)), avec (i, j, k) cyclique.
    Les vecteurs directs sont stockes en lignes ; il en va de meme pour b_i.
    """
    a = _as_lattice_vectors(direct_vectors)
    volume = float(a[0] @ np.cross(a[1], a[2]))
    if abs(volume) < 1e-14:
        raise ValueError("Le volume de la cellule primitive est nul.")
    b = np.empty_like(a)
    b[0] = TWO_PI * np.cross(a[1], a[2]) / volume
    b[1] = TWO_PI * np.cross(a[2], a[0]) / volume
    b[2] = TWO_PI * np.cross(a[0], a[1]) / volume
    return b


@dataclass
class CrystalLattice:
    """Reseau cristallin : vecteurs directs, base, points et chemin k.

    Les coordonnees de base et des points de symetrie sont donnees en
    coordonnees fractionnaires (relatives aux vecteurs directs pour la base,
    aux vecteurs reciproques pour les points k).
    """

    name: str
    direct_vectors: np.ndarray
    basis: np.ndarray | None = None
    labels: Sequence[str] = field(default_factory=tuple)
    symmetry_points: Mapping[str, Sequence[float]] = field(default_factory=dict)
    path: Sequence[str] = field(default_factory=tuple)
    description: str = ""
    two_dimensional: bool = False

    def __post_init__(self) -> None:
        self.direct_vectors = _as_lattice_vectors(self.direct_vectors)
        self.reciprocal_vectors = reciprocal_lattice(self.direct_vectors)

        if self.basis is None:
            self.basis = np.zeros((1, 3), dtype=float)
        else:
            self.basis = _as_points(self.basis, "basis")

        if not self.labels:
            self.labels = tuple(f"site{i}" for i in range(len(self.basis)))
        if len(self.labels) != len(self.basis):
            raise ValueError(
                "Le nombre de labels doit egaler le nombre de sites de la base."
            )

        self.symmetry_points = {
            str(label): _as_points([coords], f"point {label}")[0]
            for label, coords in self.symmetry_points.items()
        }
        self.path = tuple(self.path)
        unknown = [label for label in self.path if label not in self.symmetry_points]
        if unknown:
            raise ValueError(f"Points absents du dictionnaire de symetrie : {unknown}")

    @property
    def volume(self) -> float:
        """Volume de la cellule primitive (unite des vecteurs directs)."""
        return abs(float(self.direct_vectors[0] @ np.cross(
            self.direct_vectors[1], self.direct_vectors[2]
        )))

    @property
    def n_sites(self) -> int:
        return len(self.basis)

    def direct_positions(self, fractional=None) -> np.ndarray:
        """Positions cartesiennes des sites (coordonnees fractionnaires)."""
        frac = self.basis if fractional is None else _as_points(fractional)
        return frac @ self.direct_vectors

    def reciprocal_positions(self, fractional) -> np.ndarray:
        """Convertit des coordonnees fractionnaires reciproques en k cartesiens."""
        return _as_points(fractional, "fractional_k") @ self.reciprocal_vectors

    def high_symmetry_cartesian(self) -> Dict[str, np.ndarray]:
        return {
            label: self.reciprocal_positions(coords)
            for label, coords in self.symmetry_points.items()
        }

    def generate_k_path(
        self,
        n_per_segment: int = 80,
        path: Sequence[str] | None = None,
    ) -> Dict[str, np.ndarray | tuple]:
        """Genere un chemin lineaire dans l'espace reciproque.

        Retourne un dictionnaire contenant les points k cartesiens, les
        distances cumulees le long du chemin, les positions des graduations
        et les etiquettes associees.
        """
        labels = tuple(path if path is not None else self.path)
        if len(labels) < 2:
            raise ValueError("Un chemin k doit contenir au moins deux points.")
        missing = [label for label in labels if label not in self.symmetry_points]
        if missing:
            raise ValueError(f"Points de haute symetrie inconnus : {missing}")

        n = max(int(n_per_segment), 2)
        kpoints = []
        distances = []
        ticks = []
        tick_labels = []

        first = self.reciprocal_positions(self.symmetry_points[labels[0]])[0]
        kpoints.append(first)
        distances.append(0.0)
        ticks.append(0.0)
        tick_labels.append(labels[0])

        travelled = 0.0
        for start_label, stop_label in zip(labels[:-1], labels[1:]):
            k_start = self.reciprocal_positions(self.symmetry_points[start_label])[0]
            k_stop = self.reciprocal_positions(self.symmetry_points[stop_label])[0]
            segment = k_stop - k_start
            segment_length = float(np.linalg.norm(segment))
            for index in range(1, n):
                t = index / (n - 1)
                kpoints.append(k_start + t * segment)
                distances.append(travelled + t * segment_length)
            travelled += segment_length
            ticks.append(travelled)
            tick_labels.append(stop_label)

        return {
            "kpoints": np.asarray(kpoints, dtype=float),
            "distances": np.asarray(distances, dtype=float),
            "ticks": np.asarray(ticks, dtype=float),
            "tick_labels": tuple(tick_labels),
        }

    def __repr__(self) -> str:
        return (
            f"CrystalLattice(name={self.name!r}, n_sites={self.n_sites}, "
            f"volume={self.volume:.6g})"
        )


def square_lattice(a: float = 1.0) -> CrystalLattice:
    """Reseau carre 2D, parametre de maille a."""
    if a <= 0:
        raise ValueError("a doit etre strictement positif.")
    direct = np.array([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, 1.0]])
    points = {
        "Gamma": (0.0, 0.0, 0.0),
        "X": (0.5, 0.0, 0.0),
        "M": (0.5, 0.5, 0.0),
    }
    return CrystalLattice(
        name=f"Reseau carre (a={a:g})",
        direct_vectors=direct,
        basis=np.zeros((1, 3)),
        labels=("A",),
        symmetry_points=points,
        path=("Gamma", "X", "M", "Gamma"),
        description="Reseau de Bravais carre primitif, une orbitale par maille.",
        two_dimensional=True,
    )


def hexagonal_lattice(a: float = 1.0) -> CrystalLattice:
    """Reseau hexagonal 2D avec base a deux atomes (nid d'abeille).

    a est le parametre de maille (|a_1| = |a_2| = a).
    """
    if a <= 0:
        raise ValueError("a doit etre strictement positif.")
    direct = np.array(
        [[a, 0.0, 0.0], [a / 2.0, np.sqrt(3.0) * a / 2.0, 0.0], [0.0, 0.0, 1.0]]
    )
    basis = np.array([[0.0, 0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0, 0.0]])
    points = {
        "Gamma": (0.0, 0.0, 0.0),
        "M": (0.5, 0.0, 0.0),
        "K": (1.0 / 3.0, 2.0 / 3.0, 0.0),
    }
    return CrystalLattice(
        name=f"Reseau hexagonal (a={a:g})",
        direct_vectors=direct,
        basis=basis,
        labels=("A", "B"),
        symmetry_points=points,
        path=("Gamma", "M", "K", "Gamma"),
        description="Reseau hexagonal avec base honeycomb (deux sous-reseaux).",
        two_dimensional=True,
    )


def graphene_lattice(d: float = 1.42) -> CrystalLattice:
    """Reseau du graphene oriente selon les trois premiers voisins.

    d est la distance carbone-carbone ; le parametre de maille vaut sqrt(3) d.
    Cette orientation est celle utilisee par TightBindingModel.graphene.
    """
    if d <= 0:
        raise ValueError("d doit etre strictement positif.")
    a = np.sqrt(3.0) * d
    direct = np.array(
        [
            [1.5 * d, np.sqrt(3.0) * d / 2.0, 0.0],
            [1.5 * d, -np.sqrt(3.0) * d / 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    basis = np.array([[0.0, 0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0, 0.0]])
    points = {
        "Gamma": (0.0, 0.0, 0.0),
        "M": (0.5, 0.0, 0.0),
        "K": (2.0 / 3.0, 1.0 / 3.0, 0.0),
    }
    return CrystalLattice(
        name=f"Graphene (d_CC={d:g}, a={a:.6g})",
        direct_vectors=direct,
        basis=basis,
        labels=("A", "B"),
        symmetry_points=points,
        path=("Gamma", "M", "K", "Gamma"),
        description="Nid d'abeille avec voisins a 0, 120 et 240 degres.",
        two_dimensional=True,
    )


def perovskite_lattice(a: float = 3.9) -> CrystalLattice:
    """Perovskite cubique ABX3 (5 atomes par cellule)."""
    if a <= 0:
        raise ValueError("a doit etre strictement positif.")
    direct = np.diag([a, a, a])
    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ]
    )
    points = {
        "Gamma": (0.0, 0.0, 0.0),
        "X": (0.5, 0.0, 0.0),
        "M": (0.5, 0.5, 0.0),
        "R": (0.5, 0.5, 0.5),
    }
    return CrystalLattice(
        name=f"Perovskite ABX3 (a={a:g})",
        direct_vectors=direct,
        basis=basis,
        labels=("A", "B", "X1", "X2", "X3"),
        symmetry_points=points,
        path=("Gamma", "X", "M", "Gamma", "R", "X"),
        description="Perovskite cubique Pm-3m : A aux sommets, B au centre, X aux faces.",
    )


def mxene_lattice(a: float = 3.0, c: float = 4.6, z: float = 0.13) -> CrystalLattice:
    """MXene M2X (P-3m1) : deux metaux M et un X par cellule."""
    if a <= 0 or c <= 0:
        raise ValueError("a et c doivent etre strictement positifs.")
    if not 0.0 < z < 0.5:
        raise ValueError("z doit appartenir a l'intervalle ouvert (0, 0.5).")
    direct = np.array(
        [
            [a, 0.0, 0.0],
            [a / 2.0, np.sqrt(3.0) * a / 2.0, 0.0],
            [0.0, 0.0, c],
        ]
    )
    basis = np.array(
        [
            [1.0 / 3.0, 2.0 / 3.0, z],
            [2.0 / 3.0, 1.0 / 3.0, 1.0 - z],
            [0.0, 0.0, 0.0],
        ]
    )
    points = {
        "Gamma": (0.0, 0.0, 0.0),
        "M": (0.5, 0.0, 0.0),
        "K": (1.0 / 3.0, 2.0 / 3.0, 0.0),
    }
    return CrystalLattice(
        name=f"MXene M2X (a={a:g}, c={c:g})",
        direct_vectors=direct,
        basis=basis,
        labels=("M1", "M2", "X"),
        symmetry_points=points,
        path=("Gamma", "M", "K", "Gamma"),
        description="Structure M2X de type P-3m1, empilement de type MXene.",
        two_dimensional=True,
    )


def build_lattice(kind: str, **kwargs) -> CrystalLattice:
    """Fabrique un reseau par nom : square, hexagonal, graphene, perovskite, mxene."""
    factories = {
        "square": square_lattice,
        "hexagonal": hexagonal_lattice,
        "graphene": graphene_lattice,
        "perovskite": perovskite_lattice,
        "mxene": mxene_lattice,
    }
    key = kind.strip().lower()
    if key not in factories:
        raise ValueError(
            f"Reseau inconnu {kind!r}. Choix possibles : {sorted(factories)}."
        )
    return factories[key](**kwargs)


LATTICE_FACTORIES = {
    "square": square_lattice,
    "hexagonal": hexagonal_lattice,
    "graphene": graphene_lattice,
    "perovskite": perovskite_lattice,
    "mxene": mxene_lattice,
}
