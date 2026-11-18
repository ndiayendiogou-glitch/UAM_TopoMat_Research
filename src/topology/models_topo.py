"""UAM-TOPOMAT -- Modeles topologiques : Haldane sur reseau hexagonal.

Hamiltonien a deux sous-reseaux (A, B) avec sauts premiers voisins t_1,
sauts seconds voisins complexes t_2 e^{±i phi} (flux de Haldane brisant la
symetrie par renversement du temps) et masse alternee ±M.

Dependances : numpy uniquement (aucune dependance a SciPy).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..quantum.lattice import reciprocal_lattice

TWO_PI = 2.0 * np.pi
ROOT3 = np.sqrt(3.0)


def nearest_neighbour_vectors(a: float = 1.0) -> np.ndarray:
    """Trois vecteurs premiers voisins (distance a), orientation 'verticale'."""
    return np.array(
        [
            [0.0, a, 0.0],
            [ROOT3 * a / 2.0, -a / 2.0, 0.0],
            [-ROOT3 * a / 2.0, -a / 2.0, 0.0],
        ]
    )


def next_neighbour_vectors(a: float = 1.0) -> np.ndarray:
    """Trois vecteurs seconds voisins independants (distance sqrt(3) a)."""
    return np.array(
        [
            [-ROOT3 * a / 2.0, 3.0 * a / 2.0, 0.0],
            [ROOT3 * a, 0.0, 0.0],
            [-ROOT3 * a / 2.0, -3.0 * a / 2.0, 0.0],
        ]
    )


def haldane_d_vector(k, t1: float, t2: float, phi: float, M: float, a: float = 1.0):
    """Composantes (d_0, d_x, d_y, d_z) de H(k) = d_0 I + d · sigma.

    Conventions :
      H_AB(k) = -t_1 sum_j exp(-i k · delta_j),
      H_AA(k) = +M + 2 t_2 sum_j cos(k · nu_j + phi),
      H_BB(k) = -M + 2 t_2 sum_j cos(k · nu_j - phi).

    Le terme d_0 decale les deux bandes sans changer la topologie ; la masse
    de Haldane est d_z = M - 2 t_2 sin(phi) sum_j sin(k · nu_j).
    """
    k3 = np.asarray(k, dtype=float).reshape(-1)
    if k3.size == 2:
        k3 = np.array([k3[0], k3[1], 0.0])
    if k3.size != 3:
        raise ValueError("k doit avoir 2 ou 3 composantes.")
    deltas = nearest_neighbour_vectors(a)
    nus = next_neighbour_vectors(a)
    phase_nn = np.exp(-1j * (deltas @ k3))
    f = complex(phase_nn.sum())
    g = float(np.sin(nus @ k3).sum())
    h = float(np.cos(nus @ k3).sum())
    d0 = 2.0 * t2 * np.cos(phi) * h
    dx = -t1 * f.real
    dy = t1 * f.imag
    dz = M - 2.0 * t2 * np.sin(phi) * g
    return np.array([d0, dx, dy, dz], dtype=float)


def haldane_critical_mass(t2: float, phi: float) -> float:
    """Masse critique signee M_c = 3 sqrt(3) t_2 sin(phi)."""
    return 3.0 * ROOT3 * float(t2) * np.sin(float(phi))


def haldane_dirac_points(a: float = 1.0) -> dict[str, np.ndarray]:
    """Points de Dirac K et K' en coordonnees cartesiennes 2D."""
    k0 = 4.0 * np.pi / (3.0 * ROOT3 * a)
    return {"K": np.array([k0, 0.0]), "Kprime": np.array([-k0, 0.0])}


def haldane_dirac_mass(point: str, t2: float, phi: float, M: float) -> float:
    """Masse d_z au point de Dirac demande (K ou Kprime)."""
    key = point.lower().replace("'", "prime")
    if key not in ("k", "kprime"):
        raise ValueError("point doit etre 'K' ou 'Kprime'.")
    critical = haldane_critical_mass(t2, phi)
    return M + critical if key == "k" else M - critical


def haldane_chern_analytic(
    t2: float,
    phi: float,
    M: float,
    band: int = 0,
) -> int:
    """Chern exact de la bande `band` (0 = inferieure, 1 = superieure).

    Pour un modele a deux bandes, chaque cone de Dirac massif contribue
    -sgn(m)/2 a la bande inferieure, avec la chiralite chi = sign(det J) du
    doublet (d_x, d_y). Ici chi_K = -1 et chi_K' = +1, d'ou

        C_inf = -1/2 [ -sgn(M + M_c) + sgn(M - M_c) ].

    C_sup = -C_inf car la somme des deux Chern est nulle.
    """
    m_k = haldane_dirac_mass("K", t2, phi, M)
    m_kp = haldane_dirac_mass("Kprime", t2, phi, M)
    lower = -0.5 * (-np.sign(m_k) + np.sign(m_kp))
    if band not in (0, 1):
        raise ValueError("band doit valoir 0 ou 1.")
    return int(round(lower if band == 0 else -lower))


@dataclass
class HaldaneModel:
    """Modele de Haldane sur le reseau hexagonal (nid d'abeille)."""

    t1: float = 1.0
    t2: float = 0.3
    phi: float = np.pi / 2.0
    M: float = 1.0
    a: float = 1.0
    name: str = "Haldane"
    reciprocal_vectors: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.t1 = float(self.t1)
        self.t2 = float(self.t2)
        self.phi = float(self.phi)
        self.M = float(self.M)
        self.a = float(self.a)
        if self.t1 <= 0.0 or self.a <= 0.0:
            raise ValueError("t1 et a doivent etre strictement positifs.")
        if self.t2 < 0.0:
            raise ValueError("t2 doit etre positif ou nul.")

        nu = next_neighbour_vectors(self.a)
        bravais = np.array(
            [[nu[1, 0], nu[1, 1], 0.0], [nu[0, 0], nu[0, 1], 0.0], [0.0, 0.0, 1.0]]
        )
        self.reciprocal_vectors = reciprocal_lattice(bravais)

    @property
    def d_vector(self) -> np.ndarray:
        """Vecteur d a k=0 (raccourci utile pour les tests)."""
        return haldane_d_vector(
            np.zeros(3), self.t1, self.t2, self.phi, self.M, self.a
        )

    @property
    def critical_mass(self) -> float:
        return haldane_critical_mass(self.t2, self.phi)

    def hamiltonian(self, k) -> np.ndarray:
        """Matrice H(k) hermitienne 2x2."""
        d0, dx, dy, dz = haldane_d_vector(
            k, self.t1, self.t2, self.phi, self.M, self.a
        )
        hamiltonian = np.array(
            [[d0 + dz, dx - 1j * dy], [dx + 1j * dy, d0 - dz]], dtype=complex
        )
        return hamiltonian

    def eigenvalues(self, k) -> np.ndarray:
        """Energies triees par ordre croissant."""
        return np.linalg.eigvalsh(self.hamiltonian(k))

    def eigen_system(self, k) -> tuple[np.ndarray, np.ndarray]:
        return np.linalg.eigh(self.hamiltonian(k))

    def gap(self, k) -> float:
        """Ecart direct E_+(k) - E_-(k)."""
        energies = self.eigenvalues(k)
        return float(energies[1] - energies[0])

    def dirac_points(self) -> dict[str, np.ndarray]:
        return haldane_dirac_points(self.a)

    def chern(self, band: int = 0) -> int:
        """Chern analytique de la bande demande."""
        return haldane_chern_analytic(self.t2, self.phi, self.M, band)

    def band_structure(self, kpoints) -> np.ndarray:
        """Energies de forme (N_k, 2) le long d'une liste de points k."""
        kpoints = np.atleast_2d(np.asarray(kpoints, dtype=float))
        return np.asarray([self.eigenvalues(k) for k in kpoints])

    def __repr__(self) -> str:
        return (
            f"HaldaneModel(t1={self.t1:g}, t2={self.t2:g}, "
            f"phi={self.phi:.4g}, M={self.M:g}, a={self.a:g})"
        )
