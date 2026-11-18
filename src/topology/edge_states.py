"""UAM-TOPOMAT -- Etats de bord topologiques sur nanorubans 1D.

Solveur tight-binding en Python pur (numpy uniquement, aucune dependance a
SciPy) pour des rubans periodiques selon k_x et de largeur finie N_y mailles
selon y avec conditions aux limites ouvertes. Deux modeles sont fournis :

  * le modele de Haldane sur reseau nid-d'abeille, dont les sauts sont la
    transformee de Fourier inverse du Hamiltonien de models_topo.py ;
  * le modele de Qi-Wu-Zhang (isolant de Chern carre a deux orbitales).

La correspondance volume-bord est verifiee en comparant les etats localises
dans le gap projete au nombre de Chern du volume : |C| canaux chiraux par bord,
presents uniquement dans la phase topologique.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

TWO_PI = 2.0 * np.pi
ROOT3 = np.sqrt(3.0)


@dataclass(frozen=True)
class Hopping:
    """Saut de l'orbitale i de la maille (0, 0) vers l'orbitale j de (dm, dn).

    Le terme hermitien conjugue est ajoute automatiquement par le solveur ;
    il ne faut donc declarer qu'une seule direction par paire d'orbitale.
    """

    i: int
    j: int
    dm: int
    dn: int
    amplitude: complex


@dataclass
class Nanoribbon:
    """Ruban 1D : mailles (m, n), n = 0 ... N_y-1 ouvertes, m periodique.

    ``a1`` et ``a2`` sont les vecteurs primitifs 2D, ``offsets`` la position
    de chaque orbitale dans la maille. Un saut acquiert la phase e^{i k_x R_x}
    avec R_x = dm a1_x + dn a2_x + tau_j_x - tau_i_x (axe x periodic).
    """

    n_cells: int
    a1: np.ndarray
    a2: np.ndarray
    offsets: np.ndarray
    onsite: np.ndarray
    hoppings: Sequence[Hopping]
    orbital_labels: tuple[str, ...] = ("A", "B")
    name: str = "nanoribbon"

    def __post_init__(self) -> None:
        self.n_cells = int(self.n_cells)
        if self.n_cells < 2:
            raise ValueError("La largeur N_y doit etre au moins 2.")
        self.a1 = np.asarray(self.a1, dtype=float).reshape(2)
        self.a2 = np.asarray(self.a2, dtype=float).reshape(2)
        self.offsets = np.asarray(self.offsets, dtype=float)
        self.onsite = np.asarray(self.onsite, dtype=float)
        n_orb = len(self.orbital_labels)
        if self.offsets.shape != (n_orb, 2):
            raise ValueError("offsets doit avoir une ligne par orbitale.")
        if self.onsite.shape != (n_orb,):
            raise ValueError("onsite doit avoir une valeur par orbitale.")
        if abs(self.a1[1]) > 1e-14:
            raise ValueError("La direction periodique doit etre l'axe x.")

    @property
    def n_orbitals(self) -> int:
        return len(self.orbital_labels)

    @property
    def matrix_size(self) -> int:
        return self.n_cells * self.n_orbitals

    def hamiltonian(self, kx: float) -> np.ndarray:
        """Matrice hermitienne H(k_x) de taille (N_y n_orb, N_y n_orb)."""
        n_orb = self.n_orbitals
        matrix = np.zeros((self.matrix_size, self.matrix_size), dtype=complex)
        matrix[np.diag_indices_from(matrix)] = np.tile(self.onsite, self.n_cells)
        for hop in self.hoppings:
            rx = (
                hop.dm * self.a1[0]
                + hop.dn * self.a2[0]
                + self.offsets[hop.j, 0]
                - self.offsets[hop.i, 0]
            )
            amplitude = complex(hop.amplitude) * np.exp(1j * kx * rx)
            for n in range(self.n_cells):
                target = n + hop.dn
                if not 0 <= target < self.n_cells:
                    continue
                column = n * n_orb + hop.i
                row = target * n_orb + hop.j
                matrix[row, column] += amplitude
                matrix[column, row] += np.conj(amplitude)
        return matrix

    def bulk_hamiltonian(self, kx: float, ky: float) -> np.ndarray:
        """Hamiltonien 2D reconstruit a partir des sauts du ruban.

        La direction x reste Bloch (k_x) et la direction y est restauree avec
        la phase de cellule kappa = k_y a_{2y}. Les valeurs propres sont
        invariantes sous le changement de jauge des sous-reseaux.
        """
        kappa = float(ky) * self.a2[1]
        n_orb = self.n_orbitals
        matrix = np.diag(self.onsite).astype(complex)
        for hop in self.hoppings:
            rx = (
                hop.dm * self.a1[0]
                + hop.dn * self.a2[0]
                + self.offsets[hop.j, 0]
                - self.offsets[hop.i, 0]
            )
            amplitude = complex(hop.amplitude) * np.exp(1j * kx * rx + 1j * kappa * hop.dn)
            matrix[hop.j, hop.i] += amplitude
            matrix[hop.i, hop.j] += np.conj(amplitude)
        return matrix

    def solve(self, kx: float) -> tuple[np.ndarray, np.ndarray]:
        """Energies triees et vecteurs propres (colonnes) de H(k_x)."""
        return np.linalg.eigh(self.hamiltonian(kx))

    def edge_weights(self, eigenvectors: np.ndarray, depth: int = 3) -> tuple[np.ndarray, np.ndarray]:
        """Poids sur les ``depth`` premieres et dernieres mailles, par etat."""
        n_orb = self.n_orbitals
        density = np.abs(eigenvectors) ** 2
        per_cell = density.reshape(self.n_cells, n_orb, -1).sum(axis=1)
        depth = min(max(1, int(depth)), self.n_cells // 2)
        bottom = per_cell[:depth].sum(axis=0)
        top = per_cell[-depth:].sum(axis=0)
        return bottom, top

    def __repr__(self) -> str:
        return (
            f"Nanoribbon(name={self.name!r}, N_y={self.n_cells}, "
            f"n_orb={self.n_orbitals}, n_hoppings={len(self.hoppings)})"
        )


@dataclass
class RibbonSpectrum:
    """Spectre d'un ruban avec detection automatique des etats de bord."""

    kpoints: np.ndarray
    energies: np.ndarray
    bottom_weights: np.ndarray
    top_weights: np.ndarray
    bulk_valence: np.ndarray
    bulk_conduction: np.ndarray
    gap_mask: np.ndarray
    edge_mask: np.ndarray
    edge_records: list[tuple[float, float, str]]
    n_gap_states: int
    n_crossings: int
    chern: int
    metadata: dict = field(default_factory=dict)

    @property
    def has_edge_modes(self) -> bool:
        """Vrai si un canal chiral protege traverse le gap projete."""
        return self.n_gap_states > 0 and self.n_crossings >= max(1, abs(self.chern))

    def __repr__(self) -> str:
        return (
            f"RibbonSpectrum(N_k={len(self.kpoints)}, C={self.chern:+d}, "
            f"gap_states={self.n_gap_states}, crossings={self.n_crossings})"
        )


def _projected_bulk_edges(
    bulk_model: Callable[[float, float], np.ndarray],
    kx: float,
    ky_values: np.ndarray,
) -> tuple[float, float]:
    """Bords haut/bas des bandes volumiques projetees a k_x fixe."""
    energies = np.asarray([bulk_model(kx, float(ky)) for ky in ky_values])
    if energies.ndim != 2 or energies.shape[1] < 2:
        raise ValueError("bulk_model doit renvoyer au moins deux energies triees.")
    return float(np.max(energies[:, 0])), float(np.min(energies[:, 1]))


def _count_crossings(records: list[tuple[float, float, str]], target: float = 0.0) -> int:
    """Nombre total de traversees de ``target`` par les branches de bord."""
    total = 0
    for edge in ("bottom", "top"):
        points = sorted((kx, energy) for kx, energy, side in records if side == edge)
        if len(points) < 2:
            continue
        signs = np.sign([energy - target for _, energy in points])
        nonzero = signs[signs != 0]
        total += int(np.count_nonzero(nonzero[1:] != nonzero[:-1]))
    return total


def solve_ribbon_spectrum(
    ribbon: Nanoribbon,
    kpoints: Sequence[float] | np.ndarray,
    bulk_model: Callable[[float, float], np.ndarray],
    ky_values: Sequence[float] | np.ndarray,
    chern: int = 0,
    edge_depth: int = 3,
    localization_threshold: float = 0.5,
    central_fraction: float = 0.8,
) -> RibbonSpectrum:
    """Diagonalise le ruban et identifie les etats chiraux dans le gap projete.

    ``bulk_model(kx, ky)`` renvoie les deux energies volumiques triees ;
    ``ky_values`` couvre une periode reciproque complete selon y. Un etat est
    dit de bord lorsqu'il est localise sur l'un des deux bords a plus de
    ``localization_threshold`` et se trouve dans la portion centrale du gap
    projete (fraction ``central_fraction`` de la largeur), ce qui exclut les
    etats de Tamm non proteges colles au bord de bande.
    """
    kpoints = np.asarray(kpoints, dtype=float).reshape(-1)
    ky_values = np.asarray(ky_values, dtype=float).reshape(-1)
    if kpoints.size == 0 or ky_values.size == 0:
        raise ValueError("kpoints et ky_values ne doivent pas etre vides.")

    n_k = kpoints.size
    n_bands = ribbon.matrix_size
    energies = np.empty((n_k, n_bands), dtype=float)
    bottom = np.empty((n_k, n_bands), dtype=float)
    top = np.empty((n_k, n_bands), dtype=float)
    valence = np.empty(n_k, dtype=float)
    conduction = np.empty(n_k, dtype=float)

    for index, kx in enumerate(kpoints):
        values, vectors = ribbon.solve(float(kx))
        energies[index] = values
        bottom[index], top[index] = ribbon.edge_weights(vectors, edge_depth)
        valence[index], conduction[index] = _projected_bulk_edges(
            bulk_model, float(kx), ky_values
        )

    margin = 1e-6
    gap_mask = energies > (valence[:, None] + margin)
    gap_mask &= energies < (conduction[:, None] - margin)
    midgap = 0.5 * (valence + conduction)
    half_gap = 0.5 * np.maximum(conduction - valence, 0.0)
    central = np.abs(energies - midgap[:, None]) < (
        central_fraction * half_gap[:, None]
    )
    localized = np.maximum(bottom, top) >= localization_threshold
    edge_mask = gap_mask & central & localized

    records: list[tuple[float, float, str]] = []
    for index, kx in enumerate(kpoints):
        for band in np.flatnonzero(edge_mask[index]):
            side = "bottom" if bottom[index, band] >= top[index, band] else "top"
            records.append((float(kx), float(energies[index, band]), side))

    return RibbonSpectrum(
        kpoints=kpoints,
        energies=energies,
        bottom_weights=bottom,
        top_weights=top,
        bulk_valence=valence,
        bulk_conduction=conduction,
        gap_mask=gap_mask,
        edge_mask=edge_mask,
        edge_records=records,
        n_gap_states=int(edge_mask.sum()),
        n_crossings=_count_crossings(records),
        chern=int(chern),
        metadata={
            "edge_depth": int(edge_depth),
            "localization_threshold": float(localization_threshold),
            "central_fraction": float(central_fraction),
            "min_projected_gap": float(np.min(conduction - valence)),
        },
    )


def haldane_ribbon(
    n_y: int = 40,
    t1: float = 1.0,
    t2: float = 0.3,
    phi: float = np.pi / 2.0,
    M: float = 0.6,
    a: float = 1.0,
) -> tuple[Nanoribbon, Callable[[float, float], np.ndarray], int]:
    """Construit le ruban de Haldane et son modele volumique associe.

    Les conventions sont celles de ``models_topo.py`` : voisins premiers
    delta_j, seconds nu_j, H_AB = -t1 sum exp(-i k.delta_j) et
    H_AA/BB = +/- M + 2 t2 sum cos(k.nu_j +/- phi). La coupure est commutee
    aux vecteurs de maille a_1 = (sqrt3 a, 0), a_2 = (-sqrt3 a/2, 3a/2).

    Retourne (ruban, modele_bulk(kx, ky), nombre_de_Chern).
    """
    from .models_topo import HaldaneModel, haldane_chern_analytic

    model = HaldaneModel(t1=t1, t2=t2, phi=phi, M=M, a=a)
    a1 = np.array([ROOT3 * a, 0.0])
    a2 = np.array([-ROOT3 * a / 2.0, 1.5 * a])
    offsets = np.array([[0.0, 0.0], [0.0, -a]])

    hoppings: list[Hopping] = []
    deltas = np.array([[0.0, a], [ROOT3 * a / 2.0, -a / 2.0], [-ROOT3 * a / 2.0, -a / 2.0]])
    for delta in deltas:
        displacement = -delta
        cell = np.linalg.solve(np.column_stack((a1, a2)), displacement - offsets[1])
        dm, dn = np.rint(cell).astype(int)
        if not np.allclose(dm * a1 + dn * a2 + offsets[1], displacement, atol=1e-10):
            raise ValueError("Decomposition de saut incoherente.")
        hoppings.append(Hopping(0, 1, int(dm), int(dn), -t1))

    nus = np.array(
        [
            [-ROOT3 * a / 2.0, 1.5 * a],
            [ROOT3 * a, 0.0],
            [-ROOT3 * a / 2.0, -1.5 * a],
        ]
    )
    for orbital in (0, 1):
        phase = np.exp(1j * phi) if orbital == 0 else np.exp(-1j * phi)
        for nu in nus:
            cell = np.linalg.solve(np.column_stack((a1, a2)), nu)
            dm, dn = np.rint(cell).astype(int)
            hoppings.append(Hopping(orbital, orbital, int(dm), int(dn), t2 * phase))

    ribbon = Nanoribbon(
        n_cells=n_y,
        a1=a1,
        a2=a2,
        offsets=offsets,
        onsite=np.array([M, -M]),
        hoppings=hoppings,
        name="Haldane zigzag-like nanoribbon",
    )
    bulk_model = lambda kx, ky: model.eigenvalues(np.array([kx, ky, 0.0]))
    chern = haldane_chern_analytic(t2, phi, M)
    return ribbon, bulk_model, chern


def haldane_ky_values(a: float = 1.0, n: int = 240) -> np.ndarray:
    """Une periode reciproque selon y pour le ruban de Haldane."""
    return np.linspace(0.0, 4.0 * np.pi / (3.0 * a), int(n), endpoint=False)


def qwz_ribbon(
    n_y: int = 40,
    u: float = -1.0,
) -> tuple[Nanoribbon, Callable[[float, float], np.ndarray], int]:
    """Construit le ruban du modele de Qi-Wu-Zhang sur reseau carre.

    H(k) = sin(k_x) sigma_x + sin(k_y) sigma_y
           + (u + cos(k_x) + cos(k_y)) sigma_z.
    Le nombre de Chern de la bande inferieure, calcule par la methode FHS,
    vaut -1 pour -2 < u < 0, +1 pour 0 < u < 2 et 0 sinon.
    """
    unit = np.eye(2)
    hoppings = [
        Hopping(0, 1, 1, 0, -0.5j),
        Hopping(0, 1, -1, 0, 0.5j),
        Hopping(0, 1, 0, 1, -0.5),
        Hopping(0, 1, 0, -1, 0.5),
        Hopping(0, 0, 1, 0, 0.5),
        Hopping(1, 1, 1, 0, -0.5),
        Hopping(0, 0, 0, 1, 0.5),
        Hopping(1, 1, 0, 1, -0.5),
    ]
    ribbon = Nanoribbon(
        n_cells=n_y,
        a1=unit[0],
        a2=unit[1],
        offsets=np.zeros((2, 2)),
        onsite=np.array([u, -u]),
        hoppings=hoppings,
        name="Qi-Wu-Zhang Chern ribbon",
    )

    def bulk_model(kx: float, ky: float) -> np.ndarray:
        dx, dy = np.sin(kx), np.sin(ky)
        dz = u + np.cos(kx) + np.cos(ky)
        radius = np.sqrt(dx * dx + dy * dy + dz * dz)
        return np.array([-radius, radius])

    if -2.0 < u < 0.0:
        chern = -1
    elif 0.0 < u < 2.0:
        chern = 1
    else:
        chern = 0
    return ribbon, bulk_model, chern


def qwz_ky_values(n: int = 401) -> np.ndarray:
    """Une periode reciproque selon y pour le ruban QWZ."""
    return np.linspace(0.0, TWO_PI, int(n), endpoint=False)


def bulk_chern_fhs(
    bulk_d_vector: Callable[[float, float], np.ndarray],
    n: int = 48,
) -> float:
    """Nombre de Chern FHS d'un modele a deux bandes, calcul en numpy pur.

    ``bulk_d_vector`` renvoie (d_x, d_y, d_z) pour (kx, ky) cartesiens. Cette
    fonction permet de verifier la correspondance volume-bord sans SCIpy ni
    dependance au mailleur FHS deja present dans le projet.
    """
    grid = TWO_PI * (np.arange(n, dtype=float) + 0.5) / n
    states = np.empty((n, n, 2), dtype=complex)
    for i, kx in enumerate(grid):
        for j, ky in enumerate(grid):
            dx, dy, dz = bulk_d_vector(kx, ky)
            matrix = np.array([[dz, dx - 1j * dy], [dx + 1j * dy, -dz]], dtype=complex)
            _, vectors = np.linalg.eigh(matrix)
            states[i, j] = vectors[:, 0]
    u1 = np.einsum("ijk,ijk->ij", states.conj(), np.roll(states, -1, axis=0))
    u2 = np.einsum("ijk,ijk->ij", states.conj(), np.roll(states, -1, axis=1))
    u1 /= np.abs(u1)
    u2 /= np.abs(u2)
    field = np.angle(u1 * np.roll(u2, -1, axis=0) * np.conj(np.roll(u1, -1, axis=1)) * np.conj(u2))
    return float(field.sum() / TWO_PI)
