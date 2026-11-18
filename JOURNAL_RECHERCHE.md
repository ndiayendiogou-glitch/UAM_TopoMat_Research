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

- ~~Nuit 2 : courbure de Berry, invariant de Chern, phases de Zak.~~ ✓ réalisée
- Nuit 3 : réponse optique et absorption des pérovskites halogénées.
- Nuit 4 : thermodynamique, chaleur spécifique et transport.
- Nuit 5 : intégration, article de synthèse et diffusion UAM/UFR STA.

---

## Nuit 2 — Théorie de jauge sur réseau, courbure de Berry et phase de Haldane

**Date :** 18 novembre 2026  
**Sauvegarde :** branche `backup/nuit2-pre-travail` créée avant toute modification  
**Branche de travail :** `main`

### 1. Objectifs

- Implémenter le moteur de théorie de jauge sur réseau de Fukui–Hatsugai–Suzuki (FHS) en Python pur (numpy uniquement).
- Ajouter le modèle de Haldane sur réseau hexagonal (sauts premiers voisins, sauts seconds complexes brisant $T$, masse alternée).
- Valider l'invariance de jauge exacte, la quantification entière du nombre de Chern et la fermeture du gap aux points de Dirac (numériquement et symboliquement).
- Produire la figure 300 dpi `results/figures/night2_haldane_chern_phase.png`.

### 2. Fichiers créés

```
src/topology/berry_chern.py    (moteur FHS : liens, plaquettes, courbure, Chern)
src/topology/models_topo.py    (modèle de Haldane + masses de Dirac + Chern analytique)
tests/test_night2.py           (25 tests numériques et symboliques)
results/figures/night2_haldane_chern_phase.png
```

Aucune dépendance à SciPy n'a été introduite (`numpy`, `sympy`, `matplotlib`).

### 3. Équations implémentées

#### 3.1 Liens de jauge et plaquettes (méthode FHS)

Sur une grille périodique $\mathbf{k}_{ij} = \frac{i+\frac12}{N_1}\mathbf{b}_1 + \frac{j+\frac12}{N_2}\mathbf{b}_2$ couvrant la ZB, les liens de jauge entre états propres voisins sont

$$
U_\mu(\mathbf{k}) = \frac{\langle\psi_n(\mathbf{k})|\psi_n(\mathbf{k}+\delta\mathbf{k}_\mu)\rangle}
{\left|\langle\psi_n(\mathbf{k})|\psi_n(\mathbf{k}+\delta\mathbf{k}_\mu)\rangle\right|},
\qquad |U_\mu(\mathbf{k})| = 1.
$$

Le champ de Berry sur la plaquette élémentaire est le produit fermé (branche principale $\mathrm{Arg}\in(-\pi,\pi]$) :

$$
F_{12}(\mathbf{k}) = \mathrm{Arg}\!\left[U_1(\mathbf{k})\,U_2(\mathbf{k}+\hat 1)\,
U_1(\mathbf{k}+\hat 2)^{-1}\,U_2(\mathbf{k})^{-1}\right].
$$

Sous une transformation de jauge locale $\psi_n(\mathbf{k})\to e^{i\theta(\mathbf{k})}\psi_n(\mathbf{k})$, chaque lien devient $U_\mu\to e^{i\theta(\mathbf{k})}U_\mu e^{-i\theta(\mathbf{k}+\hat\mu)}$ ; le produit de plaquette est donc **strictement invariant** (pas seulement à $O(\delta k)^2$ comme pour les dérivées finies).

#### 3.2 Courbure et nombre de Chern

Avec $dA = |\mathbf{b}_1\times\mathbf{b}_2|/(N_1N_2)$ l'aire d'une maille,

$$
\Omega_n(\mathbf{k}) = \frac{F_{12}(\mathbf{k})}{dA},\qquad
C_n = \frac{1}{2\pi}\sum_{\mathbf{k}\in\mathrm{ZB}}F_{12}(\mathbf{k})
= \frac{1}{2\pi}\int_{\mathrm{ZB}}\Omega_n(\mathbf{k})\,d^2k \in \mathbb{Z}.
$$

La somme des $C_n$ sur toutes les bandes est nulle (tests vérifiés pour les deux bandes de Haldane).

#### 3.3 Modèle de Haldane

Réseau nid-d'abeille ($a$ = distance premier voisin), vecteurs $\boldsymbol{\delta}_j$ (1er voisins) et $\boldsymbol{\nu}_j$ (2e voisins) :

$$
H(\mathbf{k}) = \big[M - 2t_2\sin\phi\sum_j\sin(\mathbf{k}\cdot\boldsymbol{\nu}_j)\big]\sigma_z
+ 2t_2\cos\phi\sum_j\cos(\mathbf{k}\cdot\boldsymbol{\nu}_j)\,\mathbb{I}
+ d_x(\mathbf{k})\sigma_x + d_y(\mathbf{k})\sigma_y,
$$

$$
d_x - i d_y = -t_1\sum_j e^{-i\mathbf{k}\cdot\boldsymbol{\delta}_j},
\qquad E_\pm(\mathbf{k}) = d_0(\mathbf{k}) \pm |\mathbf{d}(\mathbf{k})|.
$$

Le terme $t_2 e^{\pm i\phi}$ brise la symétrie par renversement du temps et ouvre un gap topologique ; $M$ est la masse alternée de Stoner–Wannier.

#### 3.4 Fermeture du gap et diagramme de phase

Aux points de Dirac $K=(4\pi/3\sqrt3 a,0)$ et $K'=-K$, $f(K)=f(K')=0$, si bien que $d_x=d_y=0$ et l'écart vaut $2|d_z|$. SymPy donne exactement

$$
\sum_j\sin(K\cdot\boldsymbol{\nu}_j)=-\frac{3\sqrt3}{2},\qquad
\sum_j\sin(K'\cdot\boldsymbol{\nu}_j)=+\frac{3\sqrt3}{2},
$$

$$
d_z(K) = M + 3\sqrt3\,t_2\sin\phi,\qquad
d_z(K') = M - 3\sqrt3\,t_2\sin\phi.
$$

La gap se ferme donc **exactement** sur

$$
|M| = 3\sqrt3\,|t_2\sin\phi| \quad\Longleftrightarrow\quad
\left|\frac{M}{3\sqrt3\,t_2\sin\phi}\right| = 1.
$$

Avec les chiralités $\chi_K=-1$, $\chi_{K'}=+1$, la contribution de chaque cône massif donne

$$
C_{\text{inf}} = -\tfrac12\big[-\mathrm{sgn}(M+M_c)+\mathrm{sgn}(M-M_c)\big],
\qquad M_c = 3\sqrt3\,t_2\sin\phi,
$$

d'où les trois phases : $C=+1$ si $\sin\phi>0$ et $|M|<|M_c|$ ; $C=-1$ si $\sin\phi<0$ et $|M|<|M_c|$ ; $C=0$ si $|M|>|M_c|$.

### 4. Résultats numériques

Commande : `python -m pytest -v` → **44 tests passés** (19 Nuit 1 + 25 Nuit 2).

| Contrôle | Résultat | Tolérance |
|---|---|---|
| Invariance de jauge (grille 28×28, 2 bandes, phase aléatoire) | erreur max $5.90\times10^{-16}$ | $<10^{-12}$ |
| Invariance de jauge (10 transformations successives) | erreur max $<10^{-12}$ | $<10^{-12}$ |
| $|U_\mu(\mathbf{k})|=1$ | précision machine $10^{-14}$ | $10^{-14}$ |
| Chern Haldane $\phi=\pi/2,\ M/t_2=0$ | $C=+1.0000000000$ | $\pm10^{-9}$ |
| Chern Haldane $\phi=-\pi/2,\ M/t_2=0$ | $C=-1.0000000000$ | $\pm10^{-9}$ |
| Chern Haldane $\phi=\pi/2,\ M/t_2=7$ | $C=0.0000000000$ | $\pm10^{-9}$ |
| Stabilité sous raffinement (24, 40, 64) | $[+1,+1,+1]$ | identique |
| Gap à $M=M_c$ (point $K'$) | $<10^{-12}$ | gap nul |
| Gap à $M=-M_c$ (point $K$) | $<10^{-12}$ | gap nul |
| Loi locale du gap $|M-M_c|=0.01$ | $2\,\Delta M = 0.02$ | $10^{-10}$ |
| Frontière symbolique SymPy | $M=\pm3\sqrt3\,t_2\sin\phi$ | exact |
| Figure 300 dpi | 367 696 octets, pHYs $\approx300$ dpi | $\pm1$ dpi |

Paramètres de la figure : $t_1=1$, $t_2=0.3$, $\phi=\pi/2$, $M=0.6$ ($M_c=3\sqrt3\,t_2\simeq1.559$, donc $C=+1$), grille FHS $130\times130$. Courbure $\Omega\in[-12.01,\,12.51]$ Å², champ de plaquette $F_{12}\in[-0.0108,\,0.0113]$ rad, aire de maille $dA=8.9913\times10^{-4}$ Å².

### 5. Figure de publication

`results/figures/night2_haldane_chern_phase.png` — 300 dpi — deux panneaux :

1. **(a)** Distribution $2$D de $\Omega(\mathbf{k})$ dans la première ZB (parallélogramme centré sur $\Gamma$) pour la phase $C=+1$, avec points de Dirac $K,K'$ repliés dans la zone et échelle symlog révélant les pics de courbure.
2. **(b)** Diagramme de phase complet $(\phi,\,M/t_2)$ : régions $C=+1$ (rouge), $C=-1$ (blanc), $C=0$ (bleu), avec les frontières $|M/t_2|=3\sqrt3|\sin\phi|$ en pointillés et les points testés par la méthode FHS (étoiles).

La figure est régénérée à chaque exécution de pytest.

### 6. Prochaines étapes

- Nuit 3 : réponse optique et absorption des pérovskites halogénées.
- Nuit 4 : thermodynamique, chaleur spécifique et transport.
- Nuit 5 : intégration, article de synthèse et diffusion UAM/UFR STA.


