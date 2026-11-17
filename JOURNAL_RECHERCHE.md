# JOURNAL_RECHERCHE — UAM-TOPOMAT

**Projet :** UAM-TOPOMAT : Moteur Quantique, Topologique et Thermodynamique pour Matériaux Énergétiques Sahéliens  
**Durée :** 5 mois  
**Université Amadou Mahtar Mbow (UAM) — UFR STA — Cellule Physique**  
**Réalisé par : Ndiogou Ndiaye**  
**Nuit 1 — 17 novembre 2026**

---

## Nuit 1 — Architecture scientifique, cristallographie et solveur quantique

### 1. Objectifs

- Initialiser Git et l'architecture modulaire du projet.
- Moteur cristallographique 2D/3D (direct, réciproque, haute symétrie, chemins k) sans SciPy.
- Solveur Tight-Binding multi-orbitales : valeurs propres, vecteurs propres, vitesses de groupe et DOS.
- Valider le cône de Dirac du graphène au point K (SymPy + pytest, précision relative < 1e-6).
- Produire la figure 300 dpi `results/figures/night1_bands_dos.png` et le premier commit.

### 2. Architecture créée

```
UAM_TopoMat_Research/
├── .gitignore, requirements.txt, conftest.py
├── JOURNAL_RECHERCHE.md
├── src/quantum/{__init__.py,lattice.py,band_structure.py}
├── src/{topology,optics,thermo}/__init__.py
├── tests/test_night1.py
├── results/figures/night1_bands_dos.png
└── paper_latex/main.tex  (modèle officiel UAM – UFR STA – Cellule Physique)
```

Dépendances : numpy, sympy, matplotlib, pytest. Aucune dépendance à SciPy.

### 3. Équations implémentées

#### 3.1 Réseau réciproque

Pour des vecteurs primitifs $\mathbf{a}_i$ stockés en lignes :

$$
\mathbf{b}_i = 2\pi\,\frac{\mathbf{a}_j \times \mathbf{a}_k}{\mathbf{a}_1 \cdot (\mathbf{a}_2 \times \mathbf{a}_3)},\qquad
(i,j,k)\ \text{cyclique},\qquad
\mathbf{b}_i\cdot\mathbf{a}_j = 2\pi\delta_{ij}.
$$

Réseaux : carré (2D), hexagonal (2D), graphène (2D, distance C–C $d$), pérovskite cubique $ABX_3$ (3D, 5 atomes), MXène $M_2X$ (2D, 3 atomes). Points de haute symétrie : $\Gamma=(0,0,0)$, $X=(1/2,0,0)$, $M=(1/2,1/2,0)$, $K=(1/3,2/3,0)$ (et $K'=(2/3,1/3,0)$, point de Dirac du graphène dans son orientation propre), $R=(1/2,1/2,1/2)$. Chemins : $\Gamma$–$X$–$M$–$\Gamma$ (carré), $\Gamma$–$M$–$K$–$\Gamma$ (hexagonal/graphène/MXène), $\Gamma$–$X$–$M$–$\Gamma$–$R$–$X$ (pérovskite).

#### 3.2 Hamiltonien Tight-Binding multi-orbitales

$$
H_{ij}(\mathbf{k}) = \varepsilon_i\delta_{ij} + \sum_{\mathbf{R}} t_{ij}(\mathbf{R})\,e^{i\mathbf{k}\cdot\mathbf{R}},\qquad
H_{ji}(\mathbf{k}) = H_{ij}^{*}(\mathbf{k}).
$$

$E_n(\mathbf{k})$ et $|u_{n\mathbf{k}}\rangle$ sont obtenus par diagonalisation hermitienne (`numpy.linalg.eigh`).

#### 3.3 Vitesse de groupe (Hellmann–Feynman)

$$
\mathbf{v}_g^{(n)}(\mathbf{k}) = \frac{1}{\hbar}\left\langle u_{n\mathbf{k}}\middle|\partial_{\mathbf{k}}H\middle|u_{n\mathbf{k}}\right\rangle,\qquad
\partial_{k_\alpha}H_{ij} = \sum_{\mathbf{R}} iR_\alpha\,t_{ij}(\mathbf{R})e^{i\mathbf{k}\cdot\mathbf{R}}.
$$

#### 3.4 Densité d'états (élargissement gaussien)

$$
g(E) = \frac{1}{N_k}\sum_{\mathbf{k},n}\frac{1}{\sigma\sqrt{2\pi}}
\exp\!\left[-\frac{(E-E_n(\mathbf{k}))^2}{2\sigma^2}\right],\qquad
\int g(E)\,dE = N_{\mathrm{orb}}.
$$

#### 3.5 Cône de Dirac du graphène

Avec les trois vecteurs de sauts premiers voisins $\boldsymbol{\delta}_j$ de longueur $d$ :

$$
f(\mathbf{k})=\sum_{j=1}^{3}e^{i\mathbf{k}\cdot\boldsymbol{\delta}_j},\qquad
H(\mathbf{k})=-t\begin{pmatrix}0 & f(\mathbf{k})\\ f^{*}(\mathbf{k}) & 0\end{pmatrix},\qquad
E_\pm(\mathbf{k})=\pm t\,|f(\mathbf{k})|.
$$

Au point $K=(2\pi/3d,\,2\pi/3\sqrt3\,d)$, $f(K)=0$ et

$$
|f(K+\mathbf{q})|^2 = \frac{9}{4}d^2(q_x^2+q_y^2) + O(q^3)
\;\Longrightarrow\;
E_\pm(K+\mathbf{q}) = \pm\hbar v_F|\mathbf{q}|,\qquad
v_F=\frac{3td}{2\hbar}.
$$

Numériquement ($t=2.7$ eV, $d=1.42$ Å) : $\hbar v_F = 5.751$ eV·Å, $v_F \approx 8.74\times10^5$ m/s.

### 4. Résultats de la validation

Commande : `python -m pytest tests/test_night1.py -q` → **19 tests passés**.

| Contrôle | Résultat | Tolérance |
|---|---|---|
| Orthogonalité réciproque (5 réseaux) | < 1e-10 | 1e-10 |
| Cône de Dirac numérique (12 directions, q=1e-7) | erreur rel. max 3.87e-8 | < 1e-6 |
| Cône de Dirac symbolique SymPy | Hessien = diag(9d²/2, 9d²/2) | exact |
| Vitesse de groupe près de K (q=1e-5) | erreur rel. max 7.10e-6 | 1e-4 |
| Normalisation DOS (grille 90x90, sigma=0.15) | 1.999999999995 | 2e-3 |
| Figure 300 dpi | 259730 octets, pHYs 299.9994 dpi | ±1 dpi |

Grandeurs :hbar v_F = 5.751 eV·Å, v_F = 8.7373e5 m/s.

Point technique corrigé durant la nuit : les coordonnées fractionnaires de K dépendent de l'orientation du réseau. Réseau hexagonal standard : K=(1/3,2/3) ; orientation graphène alignée sur les trois premiers voisins : K=(2/3,1/3). Les deux donnent le même vecteur cartésien une fois la bonne base réciproque utilisée.

### 5. Figure de publication

`results/figures/night1_bands_dos.png` — 300 dpi — deux panneaux : bandes E_n(k) le long de Gamma-M-K-Gamma (avec encadré du cône de Dirac) et DOS correspondante. La figure est régénérée à chaque exécution de pytest.

### 6. Prochaines étapes

- Nuit 2 : courbure de Berry, invariant de Chern, phases de Zak.
- Nuit 3 : réponse optique et absorption des pérovskites halogénées.
- Nuit 4 : thermodynamique, chaleur spécifique et transport.
- Nuit 5 : intégration, article de synthèse et diffusion UAM/UFR STA.


