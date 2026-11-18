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

- ~~Nuit 3 : états de bord topologiques sur nanorubans.~~ ✓ réalisée
- ~~Nuit 4 : optique cristalline et rendement photovoltaïque.~~ ✓ réalisée
- Nuit 5 : intégration, article de synthèse et diffusion UAM/UFR STA.

---

## Nuit 3 — États de bord topologiques sur nanorubans

**Date :** 18 novembre 2026 (Sprint Accéléré Nuit 3 + Nuit 4)  
**Branche de travail :** `main`  
**Objectif :** démonstration au Twin Transition Summit du 1er décembre 2026.

### 1. Objectifs

- Construire un solveur de nanoruban 1D en Python pur (numpy uniquement) : périodicité selon $k_x$, largeur finie $N_y$ et conditions aux limites ouvertes selon $y$.
- Implémenter le ruban de Haldane (transformée de Fourier inverse du modèle de la Nuit 2) et le ruban de Qi–Wu–Zhang (isolant de Chern carré).
- Détecter automatiquement les états de bord chiraux traversant le gap projeté et vérifier la correspondance volume-bord avec le nombre de Chern $C$.

### 2. Fichier créé

```
src/topology/edge_states.py   (Nanoribbon, Hopping, solve_ribbon_spectrum,
                               haldane_ribbon, qwz_ribbon, bulk_chern_fhs)
```

### 3. Équations implémentées

#### 3.1 Hamiltonien du ruban

Dans la base $|n,\alpha\rangle$ ($n=0,\dots,N_y-1$, $\alpha\in\{A,B\}$), un saut de l'orbitale $i$ de la maille $(0,0)$ vers l'orbitale $j$ de la maille $(\Delta m,\Delta n)$ porte la phase

$$
t_{ij}(\Delta m,\Delta n;k_x)=t_{ij}\,e^{ik_x R_x},\qquad
R_x=\Delta m\,a_{1x}+\Delta n\,a_{2x}+\tau_{jx}-\tau_{ix},
$$

et son conjugué hermitien est ajouté automatiquement. Les données du ruban de Haldane sont $a_1=(\sqrt3 a,0)$, $a_2=(-\sqrt3 a/2,3a/2)$, $\tau_A=(0,0)$, $\tau_B=(0,-a)$, les sauts premiers voisins $-t_1$ vers $-\boldsymbol\delta_j$ et seconds voisins $t_2e^{\pm i\phi}$ vers $+\boldsymbol\nu_j$.

#### 3.2 Limite volumique et gap projeté

À $N_y$ infini, la reconstruction 2D donne $H(k_x,k_y)$ dont les valeurs propres reproduisent `HaldaneModel` à $8.9\times10^{-16}$ près (à jauge de sous-réseau près). À $k_x$ fixé, les bords du continuum volumique sont

$$
E_v(k_x)=\max_{k_y}E_-(k_x,k_y),\qquad E_c(k_x)=\min_{k_y}E_+(k_x,k_y).
$$

Un état est déclaré « de bord protégé » s'il est dans la partie centrale (80 %) du gap projeté et si son poids sur les 3 mailles de bord dépasse 0,5. Le comptage des traversées de $E=0$ par les branches de bord donne le nombre de canaux chiraux : $2|C|$ (un par bord).

### 4. Résultats de la validation

| Contrôle | Résultat | Attendu |
|---|---|---|
| Limite bulk du ruban vs Haldane Nuit 2 | $8.9\times10^{-16}$ eV | $<10^{-12}$ |
| Haldane $M=0{,}6$ ($\phi=\pi/2$, $C=+1$) | 2 traversées, 220 états de gap | $2|C|=2$ |
| Haldane $M=2{,}0$ ($C=0$) | 0 état de gap, 0 traversée | aucun |
| Haldane $\phi\to-\phi$ ($C=-1$) | chiralité inversée, 2 traversées | $|C|=1$ |
| QWZ $u=-1$ / $u=+1$ ($C=\mp1$) | FHS $=\mp1$, 2 traversées | $|C|=1$ |
| QWZ $u=-3$ ($C=0$) | aucun état de bord | aucun |
| Localisation d'un état de gap | poids $>0{,}99$ sur 5 mailles | décroissance exponentielle |

La figure `results/figures/night3_edge_states_ribbon.png` (300 dpi) montre les spectres $E(k_x)$ topologicalique/trivial de Haldane et le ruban QWZ avec les états de bord en couleur, ainsi que le profil spatial exponentiel d'un état localisé.

---

## Nuit 4 — Optique cristalline et rendement photovoltaïque

**Date :** 18 novembre 2026 (Sprint Accéléré Nuit 3 + Nuit 4)

### 1. Objectifs

- Calculer le spectre d'absorption $\alpha(\lambda)$ à partir de $E_n(\mathbf{k})$ et des éléments de matrice dipolaires déduits de $\partial H/\partial\mathbf{k}$.
- Implémenter le modèle complet de Shockley–Queisser sous spectre AM1.5G avec dépendance explicite en température (298–330 K).
- Valider le maximum théorique ($\approx 33{,}7\%$ à $E_g\approx1{,}34$ eV, 25 °C) et la décroissance thermique jusqu'à 45 °C.

### 2. Fichier créé

```
src/optics/solar_optics.py   (absorption_spectrum, dipole_matrix_element,
                              am15g_irradiance, photon_flux_above_gap,
                              radiative_saturation_current, shockley_queisser,
                              optimize_bandgap, thermal_degradation)
```

### 3. Équations implémentées

#### 3.1 Absorption interbandes

La vitesse interbande $\mathbf v_{cv}=\hbar^{-1}\langle c|\partial_{\mathbf k}H|v\rangle$ convertit les dérivées eV·Å du solveur Tight-Binding en m/s et donne le dipôle $\mathbf r_{cv}=-i\hbar\mathbf v_{cv}/(E_c-E_v)$. La règle d'or de Fermi avec élargissement lorentzien $\gamma$ fournit

$$
\alpha(\omega)=\frac{\pi e^2\omega}{\varepsilon_0 c\,n_r V}
\sum_{v,c,\mathbf k}|\hat{\mathbf e}\!\cdot\!\mathbf r_{cv}|^2(f_v-f_c)\,
\delta_\gamma(E_c-E_v-\hbar\omega).
$$

#### 3.2 Spectre solaire et bilan détaillé

Le spectre AM1.5G (table ASTM G173 interpolée, 280–4000 nm) est normalisé à $1000$ W/m². Le flux de photons au-dessus du gap, le courant radiatif de saturation émis dans l'hémisphère avant et la caractéristique $J(V)$ sont

$$
\Phi_{sun}(E_g)=\int_{\lambda<\lambda_g}\frac{I(\lambda)\lambda}{hc}\,d\lambda,\quad
J_0=q\frac{2\pi}{h^3c^2}\int_{E_g}^{\infty}\frac{E^2\,dE}{e^{E/k_BT}-1},
$$

$$
J(V)=q\Phi_{sun}-J_0\left(e^{qV/k_BT}-1\right),\qquad
(1+v)e^v=1+\frac{J_{sc}}{J_0},\quad v=\frac{qV_{mp}}{k_BT}.
$$

Le point de puissance maximale est résolu par itération de Newton (sans SciPy) ; la condition analytique est vérifiée symboliquement avec SymPy.

### 4. Résultats de la validation

| Contrôle | Résultat | Référence |
|---|---|---|
| Intégrale AM1.5G | $1000{,}000$ W/m² | $1000$ W/m² |
| Maximum SQ à 25 °C | $E_g=1{,}339$ eV, $\eta=33{,}94\%$ | $\approx1{,}34$ eV, $\approx33{,}7\%$ |
| $E_g=1{,}34$ eV, 25 °C | $J_{sc}=35{,}2$ mA/cm², $V_{oc}=1{,}084$ V, $FF=0{,}890$ | valeurs SQ standards |
| $\eta$ à 45 °C | $33{,}06\%$ (chute de $0{,}87$ pt) | décroissance réaliste |
| Pente thermique 25–57 °C | $-0{,}043$ pt/K | typique $-0{,}04$/$-0{,}05$ pt/K |
| Absorption hors gap / au-dessus du seuil | rapport $<0{,}1$ | seuil optique |

La figure `results/figures/night4_solar_efficiency_vs_temp.png` (300 dpi) présente $\eta(E_g,T)$, la dégradation thermique à $E_g=1{,}34$ eV, le spectre AM1.5G avec la longueur d'onde de coupure et le spectre $\alpha(\lambda)$ calculé depuis le modèle de pérovskite de la Nuit 1.

### 5. Bilan du sprint et prochaines étapes

Sprint validé : `python -m pytest -q` → **79 tests passés** (19 Nuit 1 + 25 Nuit 2 + 35 Nuit 3/4), figures de publication régénérées à chaque exécution. Aucune dépendance SciPy n'a été introduite.

- Nuit 5 : intégration des modules, article de synthèse et diffusion UAM/UFR STA.
- Diffusion prévue : Twin Transition Summit, 1er décembre 2026.

---

## Nuit 5 — Bilan thermodynamique sahélien et résistance thermique à 45 °C

**Date :** 18 novembre 2026 (suite du sprint accéléré)  
**Branche de travail :** `main`  
**Objectif :** coupler les moteurs des Nuits 1 à 4 au climat réel du Sénégal et quantifier
l'avantage d'un matériau à gap optimisé sous les températures extrêmes de Matam.

### 1. Objectifs

- Résoudre le bilan énergétique stationnaire couplant absorption solaire pondérée par
  $\alpha(\lambda)$, extraction électrique $\eta(E_g,T_j)$, convection (naturelle + forcée)
  et rayonnement infrarouge de Stefan–Boltzmann.
- Calculer la dérive thermique exacte $\beta_T=(1/\eta)\,d\eta/dT$ et le gain énergétique
  annuel (kWh/m²/an) du matériau optimisé face au silicium cristallin.
- Vérifier symboliquement (SymPy) la dérivée analytique utilisée par le solveur de Newton.
- Valider 100 % des tests (Nuits 1 à 5) et produire la figure 300 dpi
  `results/figures/night5_sahel_thermal_balance.png`.

### 2. Fichiers créés

```
src/thermo/sahel_climate.py   (SahelSite, PVModule, solve_junction_temperature,
                               brent_junction_temperature, energy_balance,
                               thermal_drift_coefficient, annual_yield,
                               annual_energy_gain, symbolic_balance_derivative)
tests/test_night5.py          (52 tests numériques, symboliques et de figure)
results/figures/night5_sahel_thermal_balance.png
```

Aucune dépendance à SciPy n'a été introduite (numpy, sympy, matplotlib, pytest).

### 3. Équations implémentées

#### 3.1 Conditions climatiques de référence

| Site | $T_{amb}$ (°C) | $G$ (W/m²) | $v_{\text{vent}}$ (m/s) |
|---|---:|---:|---:|
| Dakar | 31 | 900 | 3,5 |
| Saint-Louis | 37 | 980 | 4,0 |
| Matam | 47 | 1050 | 1,8 |

#### 3.2 Bilan de puissance stationnaire

Pour une surface de cellule, à l'équilibre :

$$
P_{\text{abs}} = \alpha_{\text{sol}}(E_g)\,G,
\qquad
P_{\text{élec}} = \eta(E_g,T_j)\,G,
$$

$$
P_{\text{conv}} = h_{\text{conv}}(v)\,(T_j-T_{\text{amb}}),
\qquad
P_{\text{rad}} = \varepsilon\sigma_{\text{SB}}\left(T_j^4-T_{\text{ciel}}^4\right),
\qquad
P_{\text{th}} = U_{\text{ext}}(T_j-T_{\text{entrée}}),
$$

$$
\boxed{P_{\text{abs}} = P_{\text{élec}} + P_{\text{conv}} + P_{\text{rad}} + P_{\text{th}}}.
$$

$P_{\text{th}}$ n'intervient que pour le module hybride PV/T ($U_{\text{ext}}=8$ W/m²/K) ;
$P_{\text{th}}=0$ pour la cellule photovoltaïque nue. Le ciel est modélisé par une
dépression de $8$ K : $T_{\text{ciel}}=T_{\text{amb}}-8$ K. La convection suit la
corrélation standard $h_{\text{conv}}=5{,}7+3{,}8\,v$ (W/m²/K).

L'absorptance spectrale est pondérée par le spectre AM1.5G de la Nuit 4 :

$$
\alpha_{\text{sol}}(E_g)=
\frac{\alpha_+ \int_{\lambda<\lambda_g} I(\lambda)\,d\lambda
     + \alpha_- \int_{\lambda\ge\lambda_g} I(\lambda)\,d\lambda}
     {\int I(\lambda)\,d\lambda},
\qquad \lambda_g = hc/E_g,
$$

avec $\alpha_+=0{,}95$, $\alpha_-=0{,}60$ : $\alpha_{\text{sol}}=0{,}8223$ à $E_g=1{,}45$ eV
et $0{,}8800$ à $E_g=1{,}12$ eV (le silicium absorbe davantage de photons sous-gap et
chauffe donc plus).

#### 3.3 Rendement et dérive thermique exacte

$$
\eta(E_g,T_j)=\eta_{\text{réf}}(E_g)\,
\exp\!\left[\beta_T(T_j-T_{\text{réf}})\right],
\qquad T_{\text{réf}}=298{,}15\ \text{K},
$$

avec $\eta_{\text{réf}}$ déduit de la limite de Shockley–Queisser de la Nuit 4
($\eta_{\text{réf}}=25{,}64\%$ pour $E_g=1{,}45$ eV et $22{,}74\%$ pour $E_g=1{,}12$ eV).
La loi exponentielle rend la dérive **exacte et constante** :

$$
\beta_T=\frac{1}{\eta}\frac{d\eta}{dT}
=\frac{d\ln\eta}{dT}=-0{,}22\%/\text{K}\ \text{(optimisé)},\quad
-0{,}45\%/\text{K}\ \text{(silicium)}.
$$

#### 3.4 Résolution et dérivée analytique

Le résidu $f(T_j)=P_{\text{abs}}-P_{\text{élec}}-P_{\text{conv}}-P_{\text{rad}}-P_{\text{th}}$
est strictement décroissant, donc admet une racine unique dans
$[T_{\text{amb}}-20, T_{\text{amb}}+300]$ K. Le solveur est un Newton–Raphson vectorisé
avec dérivée analytique et sauvegarde par bissection ; un Brent scalaire indépendant
sert de vérification croisée (accord à $10^{-9}$ K près). La dérivée est

$$
\frac{df}{dT_j}
= -\beta_T\,\eta\,G - h_{\text{conv}}
  - 4\varepsilon\sigma_{\text{SB}}T_j^3 - U_{\text{ext}} < 0.
$$

SymPy confirme exactement $\partial f/\partial T_j$ et $\partial(\ln\eta)/\partial T_j=\beta_T$
(résidu symbolique nul), et un test compare la dérivée analytique à une dérivée centrée.

#### 3.5 Production annuelle

Intégration horaire sur une année type (8760 pas) avec cycle solaire diurne
($G=G_{\text{max}}\sin[\pi(t-5)/12]$), cycle saisonnier d'amplitude 6–9 K et pic de
température à 15 h. Le gain annuel est $\Delta E=E_{\text{opt}}-E_{\text{Si}}$.

### 4. Résultats de la validation

Commande : `python -m pytest -v` → **131 tests passés** (19 + 25 + 35 + 52), soit 100 %.

| Contrôle | Résultat | Tolérance / référence |
|---|---|---|
| Conservation des flux (6 cas + hybride) | $\|f(T_j)\|\le 8{,}0\times10^{-13}$ W/m² | $<10^{-11}$ W/m² |
| Newton vs Brent (6 cas) | écart $0$ K | $<10^{-9}$ K |
| $T_j > T_{\text{amb}}$ (6 cas) | oui | physique |
| $T_j$ croît avec $G$ / décroît avec $v$ | oui sur 17/16 points | monotone |
| $\beta_T$ par différence finie | erreur $<10^{-10}$ K$^{-1}$ | exact |
| Dérivée symbolique SymPy | résidu $=0$ | exact |
| Figure 300 dpi | 456 861 octets, pHYs $\approx300$ dpi | $\pm1$ dpi |

**Points de fonctionnement (cellule nue) :**

| Site | Matériau | $T_j$ (°C) | $\eta(T_j)$ (%) | $P_{\text{abs}}$ | $P_{\text{élec}}$ | $P_{\text{conv}}$ | $P_{\text{rad}}$ (W/m²) | $f$ (W/m²) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Dakar | optimisé | 49,87 | 24,27 | 740,09 | 218,46 | 358,56 | 163,07 | $4{,}6\times10^{-13}$ |
| Dakar | silicium | 53,34 | 20,02 | 791,96 | 180,15 | 424,48 | 187,33 | $2{,}3\times10^{-13}$ |
| Saint-Louis | optimisé | 56,02 | 23,95 | 805,87 | 234,68 | 397,42 | 173,77 | $-1{,}1\times10^{-13}$ |
| Saint-Louis | silicium | 59,56 | 19,46 | 862,36 | 190,75 | 471,58 | 200,02 | $-8{,}0\times10^{-13}$ |
| Matam | optimisé | 75,29 | 22,95 | 863,44 | 241,01 | 354,72 | 267,71 | $-2{,}3\times10^{-13}$ |
| Matam | silicium | 80,70 | 17,70 | 923,95 | 185,83 | 422,58 | 315,54 | $1{,}1\times10^{-13}$ |

À Matam, le matériau optimisé fonctionne **5,4 K plus froid** et délivre **29,7 % de
puissance électrique instantanée en plus** (241,0 contre 185,8 W/m²). Le module hybride
PV/T abaisse $T_j$ de 75,29 °C à 60,98 °C et porte $P_{\text{élec}}$ à 248,7 W/m²,
l'extraction thermique évacuant le surplus de chaleur.

**Bilan annuel :**

| Site | Optimisé (kWh/m²/an) | Silicium (kWh/m²/an) | Gain absolu | Gain relatif |
|---|---:|---:|---:|---:|
| Dakar | 617,4 | 521,5 | +95,9 | +18,4 % |
| Saint-Louis | 665,4 | 555,8 | +109,6 | +19,7 % |
| Matam | 688,1 | 550,4 | +137,8 | **+25,0 %** |

Le gain croît avec la sévérité thermique : c'est précisément là que la faible dérive
$\beta_T$ du matériau à gap optimisé est la plus décisive.

### 5. Figure de publication

`results/figures/night5_sahel_thermal_balance.png` — 300 dpi — deux panneaux :

1. **(a)** $T_j$ (axe gauche, traits pleins) et $\eta(T_j)$ (axe droite, tirets) pour
   $T_{\text{amb}}\in[25,55]$ °C aux conditions $G$ et vent de chaque site ; les étoiles
   marquent les trois points de conception.
2. **(b)** Production électrique annuelle comparée à Dakar, Saint-Louis et Matam, avec
   les gains absolus et relatifs annotés sur les barres.

La figure est régénérée à chaque exécution de pytest.

### 6. Bilan du projet

Les cinq nuits forment une chaîne complète et validée : cristallographie et
Tight-Binding (Nuit 1) → courbure de Berry et invariant de Chern (Nuit 2) → états de bord
chiraux (Nuit 3) → optique et limite de Shockley–Queisser (Nuit 4) → thermodynamique
appliquée au climat sahélien (Nuit 5). Aucune dépendance SciPy n'a été introduite et
les figures de publication sont reproduites à chaque passage de la suite pytest.

**Diffusion prévue :** Twin Transition Summit, 1er décembre 2026 — UAM / UFR STA / Cellule Physique.

---

## Nuit 6 — Équations aux dérivées partielles non linéaires : dérive–diffusion et courbes $J(V)$

**Date :** 18 novembre 2026 (suite du sprint accéléré)  
**Branche de travail :** `main`  
**Objectif :** fermer la chaîne physique en résolvant le système EDP couplé de la
cellule photovoltaïque, et en reliant directement les paramètres électriques au
climat réel du Sénégal via la température de jonction $T_j$ de la Nuit 5.

### 1. Objectifs

- Implémenter en Python pur (numpy, sympy, matplotlib, **aucune dépendance SciPy**)
  un solveur stationnaire 1D Poisson–dérive–diffusion discrétisé par le schéma
  exponentiel conservatif exact de Scharfetter–Gummel.
- Coupler le solveur à $T_j$ issue du bilan thermodynamique de la Nuit 5 et au
  flux de photons AM1.5G de la Nuit 4.
- Extraire automatiquement $J(V)$, $V_{oc}$, $J_{sc}$, $FF$ et $\eta_{\text{élec}}$
  pour deux matériaux (pérovskite/topologique, silicium) et trois climats.
- Valider l'équilibre thermodynamique, la conservation exacte du courant total,
  les relations d'Einstein et de Shockley (SymPy), les tendances thermiques
  25–50 °C, et produire la figure 300 dpi `night6_jv_curves_sahel.png`.

### 2. Fichiers créés

```
src/pde/__init__.py
src/pde/carrier_transport.py   (DeviceConfig, CarrierSolution, JVResult,
                                bernoulli, solve_dark_equilibrium,
                                solve_carrier_transport, solve_jv_curve,
                                junction_temperature, site_electrical_result,
                                temperature_sweep, verifications symboliques)
tests/test_night6.py           (33 tests numériques, symboliques et de figure)
results/figures/night6_jv_curves_sahel.png
```

### 3. Équations implémentées

#### 3.1 Système couplé et relation d'Einstein

Dans les unités SI, avec $\phi$ le potentiel électrostatique (V), $n,p$ les
densités de porteurs (m$^{-3}$) et $T_j$ la température de jonction :

$$
\frac{d}{dx}\!\left(\varepsilon\frac{d\phi}{dx}\right)
= -q\,(p-n+N_D-N_A),
\qquad
\frac{1}{q}\frac{dJ_n}{dx}=R-G,
\qquad
\frac{1}{q}\frac{dJ_p}{dx}=G-R,
$$

$$
J_n=-q\mu_n n\frac{d\phi}{dx}+qD_n\frac{dn}{dx},
\qquad
J_p=-q\mu_p p\frac{d\phi}{dx}-qD_p\frac{dp}{dx},
\qquad
D_{n,p}=V_T\mu_{n,p},\quad V_T=\frac{k_BT_j}{q}.
$$

La relation d'Einstein est vérifiée symboliquement en annulant le flux net de
porteurs à l'équilibre : $-mu\,n\,\partial_x\phi+D\,\partial_x n=0$ avec
$n\propto e^{\phi/V_T}$ donne exactement $D=mu\,k_BT_j/q$ (résidu SymPy nul).
Les mobilités suivent $\mu(T)=\mu_{300}(T/300)^{-3/2}$ et la densité intrinsèque
$n_i(T)=\sqrt{N_cN_v}\,e^{-E_g/2k_BT}$, avec $N_{c,v}\propto T^{3/2}$.

#### 3.2 Schéma exponentiel de Scharfetter–Gummel

Sur un intervalle $[x_i,x_{i+1}]$ de pas $h$ où $\phi$ est supposé linéaire et le
courant constant, l'intégration exacte des équations de dérive–diffusion donne
les flux aux interfaces :

$$
J_{n,i+1/2}=\frac{q\mu_n V_T}{h}\Big[B(\Delta u_i)\,n_{i+1}-B(-\Delta u_i)\,n_i\Big],
$$

$$
J_{p,i+1/2}=\frac{q\mu_p V_T}{h}\Big[B(\Delta u_i)\,p_i-B(-\Delta u_i)\,p_{i+1}\Big],
\qquad
\Delta u_i=\frac{\phi_{i+1}-\phi_i}{V_T},
$$

où $B(z)=z/(e^z-1)$ est la fonction de Bernoulli (évaluée par sa série pour
$|z|<10^{-5}$ et par ses asymptotes $B\to0$, $B(-z)\to-z$). Le développement
$B(z)=1-z/2+z^2/12+O(z^3)$ reproduce le schéma centré à faible champ et est
vérifié avec SymPy.

En variables de Slotboom $w=ne^{-\phi/V_T}=n_ie^{-\phi_n/V_T}$ et
$z=pe^{\phi/V_T}=n_ie^{\phi_p/V_T}$, ces flux deviennent
$J_{n,i+1/2}=A_i(w_{i+1}-w_i)$ et $J_{p,i+1/2}=C_i(z_i-z_{i+1})$ avec
$A_i,C_i>0$. La somme des deux équations de continuite donne alors
$J_{n,i+1/2}+J_{p,i+1/2}=\text{constante}$ **identiquement**, indépendamment du
pas de maillage. À l'équilibre ($\phi_n=\phi_p=0$), $w=z=n_i$ et les deux courants
s'annulent exactement (pas une tolérance numérique).

#### 3.3 Recombinaison, éclairement et conditions aux limites

Recombinaison Shockley–Read–Hall à pièges mi-gap ($n_1=p_1=n_i$) :

$$
R=\frac{np-n_i^2}{\tau_p(n+n_i)+\tau_n(p+n_i)},\qquad R=0\ \text{si}\ np=n_i^2.
$$

Génération optique issue du spectre AM1.5G de la Nuit 4 :
$G(x)=\eta_Q\,\Phi_{\text{sun}}(E_g)\,(G_{\text{site}}/1000)\,\alpha e^{-\alpha x}$,
où $\Phi_{\text{sun}}$ est le flux de photons au-dessus du gap. Les contacts sont
sélectifs (passivés) : $p(0)=N_A$, $n(L)=N_D$, et flux minoritaire nul aux deux
extrémités. La polarisation impose $w(L)/w(0)=z(0)/z(L)=e^{V/V_T}$, soit une
barrière $V_{bi}-V$ avec $V_{bi}=V_T\ln(N_AN_D/n_i^2)$.

#### 3.4 Résolution et courbe $J(V)$

Itérations de Gummel : (i) Newton amorti sur Poisson à $w,z$ figés, (ii) résolution
des deux systèmes tridiagonaux linéaires de continuité (algorithme de Thomas),
avec initialisation par transition tanh de largeur $\max(L_D,2h)$ ($L_D$ = longueur
de Debye). Le potentiel $T_j$ est fourni par `energy_balance(site, module)` de la
Nuit 5. Le balayage adaptatif en tension (pas $\leq0{,}04$ V, arrêt après $V_{oc}$)
fournit $J_{sc}=J(0)$, $V_{oc}$ par bissection, et le point de puissance maximale
par recherche dorée : $FF=P_{max}/(V_{oc}J_{sc})$, $\eta=P_{max}/G_{\text{site}}$.
Le courant renvoyé est $J=-(J_n+J_p)$ : $J(0)=J_{sc}>0$ et $J(V_{oc})=0$.

SymPy vérifie en outre $J(V)=J_{sc}-J_0(e^{qV/k_BT}-1)$,
$V_{oc}=V_T\ln(1+J_{sc}/J_0)$, $J(V_{oc})=0$ et la condition du point de puissance
maximale $\partial_V(VJ)=0 \Leftrightarrow J_{sc}-J_0[e^{v}(1+v)-1]=0$.

### 4. Résultats de la validation

Commande : `python -m pytest -v` → **164 tests passés** (19 + 25 + 35 + 52 + 33), soit 100 %.

| Contrôle | Résultat | Tolérance / référence |
|---|---|---|
| $J(V=0)$ à l'obscurité (6 cas) $|J|<10^{-12}$ | $0$ (identité sur $w,z$) | $<10^{-12}$ A/m² |
| $np=n_i^2$ à l'équilibre | erreur relative $<10^{-12}$ | exact |
| Conservation $J_n+J_p$ sous éclairement | $\max|J_{tot}-\overline{J_{tot}}|\approx1{,}5\times10^{-7}$ A/m² | $<10^{-6}$ A/m² |
| $V_{bi}$ théorique vs solution | accord relatif $<10^{-8}$ | $V_T\ln(N_AN_D/n_i^2)$ |
| Facteur d'idéalité (dark, 0,4–0,6 V) | $1<n<2{,}2$ | diode réelle |
| SymPy Einstein / Shockley / SG | résidus $=0$ | exact |
| Figure 300 dpi | > 60 ko, pHYs $\approx300$ dpi | ±1 dpi |

**Points de fonctionnement électriques (cellule couplée à $T_j$ du site) :**

| Site | Matériau | $T_j$ (°C) | $J_{sc}$ (A/m²) | $V_{oc}$ (V) | $FF$ | $\eta_{\text{élec}}$ (%) | $P_{max}$ (W/m²) |
|---|---|---:|---:|---:|---:|---:|---:|
| Dakar | pérovskite / topologique | 49,87 | 246,07 | 0,9367 | 0,8270 | 21,18 | 190,6 |
| Dakar | silicium cristallin | 53,34 | 327,22 | 0,5853 | 0,8069 | 17,17 | 154,5 |
| Saint-Louis | pérovskite / topologique | 56,02 | 267,90 | 0,9282 | 0,8243 | 20,91 | 205,0 |
| Saint-Louis | silicium cristallin | 59,56 | 356,25 | 0,5761 | 0,8021 | 16,80 | 164,6 |
| Matam | pérovskite / topologique | 75,29 | 286,91 | 0,8947 | 0,8131 | 19,88 | 208,7 |
| Matam | silicium cristallin | 80,70 | 381,50 | 0,5381 | 0,7833 | 15,31 | 160,8 |

À Matam, malgré un $J_{sc}$ plus faible (gap plus large), le matériau pérovskite
/topologique délivre **$+0{,}357$ V de $V_{oc}$**, un facteur de forme supérieur
de $3{,}0$ points et **$+30\%$ de rendement électrique** (19,88 % contre 15,31 %).
$J_{sc}$ croît avec l'irradiance du site tandis que $V_{oc}$ et $FF$ chutent avec
$T_j$ : le dilemme courant/tension est ici tranché en faveur de la pérovskite.

**Dérive thermique 25 → 50 °C (sous 1000 W/m²) :**

| Matériau | $V_{oc}$(25 °C) | $V_{oc}$(50 °C) | $FF$(25 °C) | $FF$(50 °C) | $\eta$(25 °C) | $\eta$(50 °C) |
|---|---:|---:|---:|---:|---:|---:|
| Pérovskite / topologique | 0,9862 V | 0,9401 V | 0,8431 | 0,8282 | 22,74 % | 21,29 % |
| Silicium cristallin | 0,6416 V | 0,5947 V | 0,8305 | 0,8104 | 19,39 % | 17,53 % |

Les six points de la plage 25–50 °C sont strictement décroissants en $V_{oc}$,
$FF$ et $\eta$, avec $J_{sc}$ constant à moins de 2 % près — comportement attendu
d'une cellule dont la recombinaison radiative et SRH croît avec $T_j$.

### 5. Figure de publication

`results/figures/night6_jv_curves_sahel.png` — 300 dpi — quatre panneaux :

1. **(a)** $J(V)$ des trois climats pour la pérovskite, avec $V_{oc}$ (cercles) et
   point de puissance maximale (carrés) ;
2. **(b)** $P(V)=J(V)\,V$ correspondantes et maximum de puissance ;
3. **(c)** profils internes du potentiel électrostatique $\phi(x)$ à $V=0$ ;
4. **(d)** profils $n(x)$ (traits pleins) et $p(x)$ (tirets) en échelle
   logarithmique, montrant la zone de déplétion centrale.

La figure est régénérée à chaque passage de la suite pytest.

### 6. Bilan définitif du projet

Les six nuits forment une chaîne complète et validée : cristallographie et
Tight-Binding (1) → courbure de Berry et invariant de Chern (2) → états de bord
chiraux (3) → optique AM1.5G et limite de Shockley–Queisser (4) → bilan
thermodynamique sahélien (5) → EDP de transport et caractéristiques $J(V)$ (6).
Aucune dépendance SciPy n'a été introduite ; 164 tests et six figures de
publication sont reproduits à chaque exécution de `python -m pytest`.

**Diffusion :** Twin Transition Summit, 1er décembre 2026 — UAM / UFR STA / Cellule Physique.
