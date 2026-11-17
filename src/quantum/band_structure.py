"""UAM-TOPOMAT -- Solveur quantique Tight-Binding multi-orbitales.

Calcul des valeurs propres E_n(k), des vecteurs propres, des vitesses de
groupe v_g(k) et de la densite d'etats (DOS).

Conventions : les energies sont en eV, les longueurs en Angstrom et l'on
travaille en unites naturelles hbar = 1 sauf mention contraire. Le modele
general accepte neanmoins une valeur explicite de hbar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import numpy as np

from .lattice import (
    CrystalLattice,
    graphene_lattice,
    mxene_lattice,
    perovskite_lattice,
)

HBAR = 1.0


def _k3(k) -> np.ndarray:
    arr = np.asarray(k, dtype=float).reshape(-1)
    if arr.size == 2:
        return np.array([arr[0], arr[1], 0.0])
    if arr.size != 3:
        raise ValueError("Un vecteur k doit avoir 2 ou 3 composantes.")
    return arr


def uniform_k_grid(
    reciprocal_vectors,
    n1: int = 1,
    n2: int = 1,
    n3: int = 1,
    shift: bool = True,
) -> np.ndarray:
    """Grille uniforme dans la zone de Brillouin (coordonnees fractionnaires).

    Si shift=True (defaut), les points sont centres dans les mailles de la
    grille, ce qui evite de doubler les bords de la zone.
    """
    if min(int(n1), int(n2), int(n3)) < 1:
        raise ValueError("Les dimensions de la grille doivent etre >= 1.")

    def axis(n: int) -> np.ndarray:
        values = np.arange(n, dtype=float)
        return (values + 0.5) / n if shift else values / n

    fractions = np.stack(
        np.meshgrid(axis(int(n1)), axis(int(n2)), axis(int(n3)), indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    return fractions @ np.asarray(reciprocal_vectors, dtype=float)


@dataclass
class TightBindingModel:
    """Hamiltonien Tight-Binding multi-orbitales.

    H_{ij}(k) = epsilon_i delta_{ij}
                + sum_R t_{ij}(R) exp(i k.R)  (i != j).

    Chaque saut declare de i vers j est automatiquement accompagne de son
    conjugue hermitien de j vers i. Il ne faut donc pas declarer les deux sens.
    """

    n_orbitals: int
    onsite: Sequence[float]
    hoppings: Iterable[tuple[int, int, Sequence[float], complex]] = field(
        default_factory=list
    )
    hbar: float = HBAR
    name: str = "Tight-binding"
    orbital_labels: Sequence[str] | None = None
    lattice: CrystalLattice | None = None

    def __post_init__(self) -> None:
        self.n_orbitals = int(self.n_orbitals)
        if self.n_orbitals < 1:
            raise ValueError("n_orbitals doit etre >= 1.")
        self.onsite = np.asarray(self.onsite, dtype=float)
        if self.onsite.shape != (self.n_orbitals,):
            raise ValueError("onsite doit avoir une valeur par orbitale.")
        self.hbar = float(self.hbar)
        if self.hbar <= 0:
            raise ValueError("hbar doit etre strictement positif.")

        if self.orbital_labels is None:
            self.orbital_labels = tuple(
                f"orb{i}" for i in range(self.n_orbitals)
            )
        if len(self.orbital_labels) != self.n_orbitals:
            raise ValueError("orbital_labels doit avoir une entree par orbitale.")

        converted = []
        for i, j, vector, amplitude in self.hoppings:
            i, j = int(i), int(j)
            if not (0 <= i < self.n_orbitals and 0 <= j < self.n_orbitals):
                raise ValueError("Indice d'orbitale hors limites.")
            if i == j:
                raise ValueError(
                    "Un saut intra-orbitale doit etre encode dans onsite."
                )
            converted.append((i, j, _k3(vector), complex(amplitude)))
        self.hoppings = converted

    def add_hopping(self, i: int, j: int, vector, amplitude: complex):
        """Ajoute un saut i -> j et renvoie le modele (chaineable)."""
        if int(i) == int(j):
            raise ValueError("Utiliser onsite pour un terme diagonal.")
        self.hoppings.append((int(i), int(j), _k3(vector), complex(amplitude)))
        return self

    def hamiltonian(self, k) -> np.ndarray:
        """Matrice H(k) hermitienne (n x n)."""
        k = _k3(k)
        hamiltonian = np.diag(self.onsite).astype(complex)
        for i, j, vector, amplitude in self.hoppings:
            term = amplitude * np.exp(1j * float(np.dot(k, vector)))
            hamiltonian[i, j] += term
            hamiltonian[j, i] += np.conj(term)
        return hamiltonian

    def hamiltonian_derivatives(self, k) -> np.ndarray:
        """Derivees dH/dk_a, de forme (3, n, n)."""
        k = _k3(k)
        derivatives = np.zeros((3, self.n_orbitals, self.n_orbitals), dtype=complex)
        for i, j, vector, amplitude in self.hoppings:
            term = amplitude * np.exp(1j * float(np.dot(k, vector)))
            derivatives[:, i, j] += 1j * vector * term
            derivatives[:, j, i] += -1j * vector * np.conj(term)
        return derivatives

    def eigenvalues(self, k) -> np.ndarray:
        """Valeurs propres triees par ordre croissant."""
        return np.linalg.eigvalsh(self.hamiltonian(k))

    def eigen_system(self, k) -> tuple[np.ndarray, np.ndarray]:
        """Valeurs propres et vecteurs propres (colonnes) de H(k)."""
        return np.linalg.eigh(self.hamiltonian(k))

    def band_structure(self, kpoints) -> np.ndarray:
        """Energies de forme (N_k, n_orbitals) le long d'un chemin k."""
        kpoints = np.atleast_2d(np.asarray(kpoints, dtype=float))
        energies = np.empty((len(kpoints), self.n_orbitals), dtype=float)
        for index, k in enumerate(kpoints):
            energies[index] = self.eigenvalues(k)
        return energies

    def solve(self, kpoints) -> tuple[np.ndarray, np.ndarray]:
        """Energies (N_k, n) et vecteurs propres (N_k, n, n)."""
        kpoints = np.atleast_2d(np.asarray(kpoints, dtype=float))
        energies = np.empty((len(kpoints), self.n_orbitals), dtype=float)
        vectors = np.empty(
            (len(kpoints), self.n_orbitals, self.n_orbitals), dtype=complex
        )
        for index, k in enumerate(kpoints):
            values, columns = np.linalg.eigh(self.hamiltonian(k))
            energies[index] = values
            vectors[index] = columns
        return energies, vectors

    def group_velocities(self, kpoints) -> np.ndarray:
        """Vitesses de groupe v_g = (1/hbar) <psi|dH/dk|psi> (N_k, n, 3)."""
        kpoints = np.atleast_2d(np.asarray(kpoints, dtype=float))
        velocities = np.empty(
            (len(kpoints), self.n_orbitals, 3), dtype=float
        )
        for index, k in enumerate(kpoints):
            _, vectors = self.eigen_system(k)
            derivatives = self.hamiltonian_derivatives(k)
            for band in range(self.n_orbitals):
                psi = vectors[:, band]
                for direction in range(3):
                    expectation = np.vdot(
                        psi, derivatives[direction] @ psi
                    )
                    velocities[index, band, direction] = (
                        float(np.real(expectation)) / self.hbar
                    )
        return velocities

    def density_of_states(
        self,
        energies,
        kpoints,
        broadening: float = 0.1,
    ) -> np.ndarray:
        """DOS totale par elargissement gaussien.

        La normalisation est telle que l'integrale de la DOS sur toutes les
        energies vaut n_orbitals (une orbitale contribue pour 1 par cellule).
        """
        energies = np.asarray(energies, dtype=float).reshape(-1)
        kpoints = np.atleast_2d(np.asarray(kpoints, dtype=float))
        sigma = float(broadening)
        if sigma <= 0:
            raise ValueError("broadening doit etre strictement positif.")
        if len(kpoints) == 0:
            raise ValueError("kpoints ne doit pas etre vide.")

        bands = self.band_structure(kpoints)
        dos = np.zeros_like(energies)
        normalisation = 1.0 / (
            np.sqrt(2.0 * np.pi) * sigma * len(kpoints)
        )
        for band in range(self.n_orbitals):
            differences = (
                energies[:, None] - bands[:, band][None, :]
            ) / sigma
            dos += normalisation * np.exp(-0.5 * differences**2).sum(axis=1)
        return dos

    @classmethod
    def graphene(cls, t: float = 2.7, a: float = 1.42, hbar: float = HBAR):
        """Graphene : deux orbitales p_z, sauts premiers voisins -t.

        a est la distance carbone-carbone. Pres de K, E = +/- hbar v_F |q| avec
        v_F = 3 t a / (2 hbar).
        """
        if t <= 0 or a <= 0:
            raise ValueError("t et a doivent etre strictement positifs.")
        deltas = (
            (a, 0.0, 0.0),
            (-a / 2.0, -np.sqrt(3.0) * a / 2.0, 0.0),
            (-a / 2.0, np.sqrt(3.0) * a / 2.0, 0.0),
        )
        hoppings = [(0, 1, delta, -t) for delta in deltas]
        return cls(
            n_orbitals=2,
            onsite=(0.0, 0.0),
            hoppings=hoppings,
            hbar=hbar,
            name=f"Graphene (t={t:g} eV, d_CC={a:g} A)",
            orbital_labels=("A", "B"),
            lattice=graphene_lattice(a),
        )

    @classmethod
    def perovskite(
        cls,
        a: float = 3.9,
        t_ax: float = 0.5,
        t_bx: float = 1.0,
        onsite: Sequence[float] = (-2.0, 0.0, 3.0, 3.0, 3.0),
        hbar: float = HBAR,
    ):
        """Modele 5 orbitales pour ABX3 (A, B, X1, X2, X3)."""
        if a <= 0:
            raise ValueError("a doit etre strictement positif.")
        lattice = perovskite_lattice(a)
        hoppings: List[tuple[int, int, tuple[float, float, float], float]] = []

        for x_index, vectors in (
            (2, [(sx * a / 2.0, sy * a / 2.0, 0.0) for sx in (-1, 1) for sy in (-1, 1)]),
            (3, [(sx * a / 2.0, 0.0, sz * a / 2.0) for sx in (-1, 1) for sz in (-1, 1)]),
            (4, [(0.0, sy * a / 2.0, sz * a / 2.0) for sy in (-1, 1) for sz in (-1, 1)]),
        ):
            for vector in vectors:
                hoppings.append((0, x_index, vector, -t_ax))

        for x_index, sign_axis in ((2, 2), (3, 1), (4, 0)):
            for sign in (-1, 1):
                vector = [0.0, 0.0, 0.0]
                vector[sign_axis] = sign * a / 2.0
                hoppings.append((1, x_index, tuple(vector), -t_bx))

        return cls(
            n_orbitals=5,
            onsite=onsite,
            hoppings=hoppings,
            hbar=hbar,
            name=f"Perovskite ABX3 (a={a:g} A)",
            orbital_labels=("A", "B", "X1", "X2", "X3"),
            lattice=lattice,
        )

    @classmethod
    def mxene(
        cls,
        a: float = 3.0,
        c: float = 4.6,
        z: float = 0.13,
        t: float = 1.0,
        onsite: Sequence[float] = (0.0, 0.0, -1.5),
        hbar: float = HBAR,
    ):
        """Modele effectif 3 orbitales pour MXene M2X (M1, M2, X)."""
        if t <= 0:
            raise ValueError("t doit etre strictement positif.")
        lattice = mxene_lattice(a, c, z)
        positions = lattice.direct_positions()
        m1, m2, x_site = positions
        hoppings = [
            (2, 0, m1 - x_site, -t),
            (2, 1, m2 - x_site, -t),
        ]
        return cls(
            n_orbitals=3,
            onsite=onsite,
            hoppings=hoppings,
            hbar=hbar,
            name=f"MXene M2X (a={a:g} A, c={c:g} A)",
            orbital_labels=("M1", "M2", "X"),
            lattice=lattice,
        )
