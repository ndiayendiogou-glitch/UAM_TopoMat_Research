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

---

## Nuit 7 — Modélisation des supercondensateurs topologiques MXènes sans plomb

**Date :** 18 novembre 2026  
**Branche de travail :** `main`  
**Objectif :** prolonger la chaîne UAM-TOPOMAT du matériau quantique au dispositif de
stockage : convertir la DOS topologique des MXènes $M_2X$ en capacité quantique,
l'associer à la double couche de Helmholtz, puis simuler le stockage d'énergie
sous le climat réel du Sénégal (25–50 °C, Matam 47 °C).

### 1. Objectifs

- Implémenter en Python pur (numpy, sympy, matplotlib, **aucune dépendance SciPy**)
  le modèle complet d'un supercondensateur asymétrique à électrodes MXènes sans
  plomb (Ti₂CTₓ / Nb₂CTₓ).
- Coupler $C_Q(V)=e^2\,\mathrm{DOS}(E_F+eV)$, issue du solveur Tight-Binding de la
  Nuit 1, à la capacité de double couche $C_H$ en série.
- Simuler les cycles galvanostatiques, l'ESR sahelien, les densités d'énergie
  (Wh/kg) et de puissance (W/kg), et dresser le diagramme de Ragone.
- Valider la conservation de la charge, l'association en série (SymPy), le
  maintien de > 94 % de capacité à 47 °C et produire la figure 300 dpi
  `results/figures/night7_ragone_mxene_sahel.png`.

### 2. Fichiers créés

```
src/thermo/supercapacitor_mxene.py   (MXeneElectrode, AsymmetricSupercapacitor,
                                     GalvanostaticCycle, RagonePoint, esr,
                                     pore_accessibility, site_ragone_point,
                                     verifications symboliques SymPy)
tests/test_night7.py                 (23 tests numériques, symboliques et de figure)
results/figures/night7_ragone_mxene_sahel.png
```

Aucune dépendance à SciPy n'a été introduite (numpy, sympy, matplotlib, pytest).

### 3. Équations implémentées

#### 3.1 Capacité quantique depuis la DOS Tight-Binding

Le modèle $M_2X$ à 3 orbitales de la Nuit 1 est réutilisé directement
(`TightBindingModel.mxene`, grille $32\times32$, élargissement
$\sigma=\sigma_0+k_BT/e$ avec $\sigma_0=0{,}35$ eV). La DOS (états/eV/cellule)
donne, avec $A_{\text{cell}}=\sqrt3\,a^2/2=7{,}7942\times10^{-20}$ m² pour $a=3$ Å :

$$
C_Q(V)=\frac{e^2\,\mathrm{DOS}(E_F+eV)}{A_{\text{cell}}},\qquad
C_Q\in[0{,}23\,;\,2{,}35]\ \text{F m}^{-2}.
$$

#### 3.2 Association série et équilibre de charge

$$
\frac{1}{C_{\text{tot}}(V)}=\frac{1}{C_Q(V)}+\frac{1}{C_H(T)},\qquad
C_H(T)=C_{H,\text{réf}}\left[1+\gamma(T-T_{\text{réf}})\right]\,a_p(T),
$$

avec $C_{H,\text{réf}}=0{,}50$ F/m² (50 µF/cm²), $\gamma=5\times10^{-4}$ K$^{-1}$ et
$a_p(T)=\exp[-\lambda\,(T-313{,}15\,\mathrm{K})_+]$, $\lambda=5\times10^{-3}$ K$^{-1}$
(perte d'accessibilité des pores au-delà de 40 °C). La surface spécifique
accessible est $300\,000$ m²/kg, soit $C_{\text{tot}}\approx 120$ F/g par électrode.

La conservation de charge impose $q=f_+Q_+(V_+)=f_-Q_-(V_-)$ avec les fractions
de masse $f_+=Q_-^{\max}/(Q_+^{\max}+Q_-^{\max})=0{,}4953$ et $f_-=0{,}5047$.
La tension à circuit ouvert est donc

$$
U_{\text{oc}}(q)=V_+\!\left(q/f_+\right)+V_-\!\left(q/f_-\right),
\qquad U_{\max}=1{,}9\ \text{V}.
$$

SymPy vérifie exactement $1/C_{\text{tot}}=1/C_Q+1/C_H$,
$q/C_Q+q/C_H=q/C_{\text{tot}}$, l'équivalence énergétique
$\tfrac12 C_{\text{tot}}U^2=q^2/(2C_{\text{tot}})$ et
$\partial C_{\text{tot}}/\partial C_Q=C_H^2/(C_Q+C_H)^2$.

#### 3.3 ESR et cycles galvanostatiques

$$
R_{\text{ESR}}(T)=R_c+R_{\text{ion,0}}\exp\!\left[\frac{E_a}{k_B}
\left(\frac1T-\frac1{T_{\text{réf}}}\right)\right]
+R_{\text{el,0}}\left[1+\alpha(T-T_{\text{réf}})\right],
$$

$E_a=0{,}15$ eV, $R_{\text{ion,0}}=2\times10^{-5}$, $R_{\text{el,0}}=4\times10^{-6}$,
$R_c=2\times10^{-6}$ Ω·kg, $\alpha=10^{-3}$ K$^{-1}$. Le courant impose
$dq/dt=\pm I$ et les tensions $U_{\text{charge}}=U_{\text{oc}}+I R_{\text{ESR}}$,
$U_{\text{décharge}}=U_{\text{oc}}-I R_{\text{ESR}}$ ; les bornes de charge et de
décharge sont résolues par bissection (sans SciPy). Les densités sont

$$
E_{\text{sp}}=\frac{1}{3600}\int U\,dq\ \text{(Wh/kg)},\qquad
P_{\text{sp}}=I\,\overline{U}\ \text{(W/kg)}.
$$

### 4. Résultats de la validation

Commande : `python -m pytest -v` → **187 tests passés** (164 + 23), soit 100 %.

| Contrôle | Résultat | Tolérance / référence |
|---|---|---|
| $C_Q=e^2\mathrm{DOS}/A_{\text{cell}}$ | accord relatif $<10^{-14}$ | formule exacte |
| $C_{\text{tot}}<\min(C_Q,C_H)$ (2 électrodes × 2 $T$) | vrai sur 25 potentiels | série stricte |
| SymPy série / tension / énergie / dérivée | résidus $=0$ | exact |
| Conservation $Q_{\text{charge}}=Q_{\text{décharge}}+Q_{\text{restant}}$ | rel. $<10^{-12}$ (3 courants) | exact |
| $\eta_{\text{coulombique}}$ à 10 / 100 / 1000 A/kg | 0,9999 / 0,9986 / 0,9861 | $>0{,}9$ |
| ESR 25 → 50 °C | $26{,}00\to18{,}83$ µΩ·kg | décroissance |
| Rétention à 47 °C (Matam) | **98,11 %** | $>94$ % |
| Rétention à 40 / 50 °C | 100,65 % / 97,03 % | pic puis baisse |
| Figure 300 dpi | 604 102 octets, pHYs $\approx300$ dpi | ±1 dpi |

**Points Ragone (25 °C) :**

| Courant (A/kg) | Énergie (Wh/kg) | Puissance (W/kg) | Capacité (mAh/g) | $\eta_{\text{coul}}$ |
|---:|---:|---:|---:|---:|
| 1 | 15,37 | 1,0 | 16,17 | 1,0000 |
| 100 | 15,29 | 94,8 | 16,12 | 0,9986 |
| 1000 | 14,55 | 925,2 | 15,72 | 0,9861 |
| 4000 | 12,20 | 3390,6 | 14,40 | 0,9418 |
| 8000 | 9,39 | 5951,6 | 12,62 | 0,8766 |

La cellule délivre $15{,}4$ Wh/kg à faible régime et conserve $9{,}4$ Wh/kg à
presque 6 kW/kg : elle dépasse nettement les supercondensateurs commerciaux
(~8 Wh/kg à 100 W/kg) tout en restant, comme attendu, sous les batteries Li-ion
(~225 Wh/kg). À Matam, la baisse d'ESR (26,00 → 19,48 µΩ·kg) compense presque
entièrement la dégradation des pores : la capacité maintenue est de 98,11 %,
supérieure au seuil de 94 % exigé.

### 5. Figure de publication

`results/figures/night7_ragone_mxene_sahel.png` — 300 dpi — quatre panneaux :

1. **(a)** diagramme de Ragone log-log comparant les MXènes topologiques à 25 °C
   et à Matam (47 °C) aux batteries Li-ion et aux supercondensateurs commerciaux
   (bandes de dispersion incluses) ;
2. **(b)** ESR de 25 à 50 °C avec le point de fonctionnement de Matam ;
3. **(c)** cycles galvanostatiques charge/décharge à 500 A/kg et 25 / 47 °C ;
4. **(d)** rétention de capacité de 25 à 50 °C avec le seuil à 94 % et le point
   de Matam.

La figure est régénérée à chaque passage de la suite pytest.

### 6. Bilan définitif du projet

Les sept nuits forment une chaîne complète et validée : cristallographie et
Tight-Binding (1) → courbure de Berry et invariant de Chern (2) → états de bord
chiraux (3) → optique AM1.5G et Shockley–Queisser (4) → thermodynamique
sahélienne (5) → EDP de transport et $J(V)$ (6) → stockage électrochimique
MXène sans plomb et diagramme de Ragone (7). Aucune dépendance SciPy n'a été
introduite ; 187 tests et sept figures de publication sont reproduits à chaque
exécution de `python -m pytest`.

**Diffusion :** Twin Transition Summit, 1er décembre 2026 — UAM / UFR STA / Cellule Physique.

---

## Nuit 8 — Base déterministe de 2 000 matériaux cristallins sans plomb

**Date :** 18 novembre 2026  
**Branche de travail :** `main`  
**Objectif :** fournir à la chaîne UAM-TOPOMAT une base de données cristallochimique
reproductible de 2 000 candidats sans plomb ni cobalt, ancrée dans les ressources
minérales de l'Afrique de l'Ouest, et vérifier que chaque famille reste compatible
avec les solveurs des Nuits 1 à 7.

### 1. Objectifs

- Implémenter un générateur déterministe (graine `SEED = 2026`) en Python pur
  (numpy, sympy, matplotlib, **aucune dépendance SciPy**).
- Produire exactement 2 000 candidats uniques répartis en trois familles :
  pérovskites $ABX_3$ (Sn, Bi, Ti, Ge), pérovskites doubles $A_2B'B''X_6$
  (Bi–Ag, Sn–Ti, Sb–In) et MXènes 2D $M_2X$ / $M_3X_2$ (Ti, V, Mn, Fe ;
  carbures et nitrures).
- Corréler tous les paramètres physiques à la chimie des éléments et garantir
  la neutralité de charge exacte de chaque composition.
- Sauvegarder `data/african_materials_db.json` et `data/db_summary.json`,
  valider 100 % des tests (Nuits 1 à 8) et produire la figure 300 dpi
  `results/figures/night8_materials_landscape.png`.

### 2. Fichiers créés

```
src/quantum/materials_generator.py   (Element, generate_candidates,
                                      material_summary, save_database,
                                      load_database, build_tight_binding_model,
                                      representative_sample)
data/african_materials_db.json       (2 000 candidats, 1,36 Mo)
data/db_summary.json                 (résumé statistique déterministe)
tests/test_night8.py                 (42 tests : structure, physique,
                                      déterminisme, solveurs Nuits 1–7, figure)
results/figures/night8_materials_landscape.png
```

### 3. Génération cristallochimique

#### 3.1 Occupation des sites et neutralité

Les sites sont occupés par une ou deux espèces avec des fractions discrètes
($0{,}25$ ; $0{,}5$ ; $0{,}75$ ; $1/3$ ; $2/3$), ce qui engendre 2 000 formules
distinctes (solutions solides ordonnées) à partir d'une table de 30 éléments.
Les états d'oxydation sont affectés par site et non par élément : Sn et Ge sont
$+2$ dans les halogénures et $+4$ dans les oxydes ; Bi–Ag donne
$A_2\mathrm{Ag}^{1+}\mathrm{Bi}^{3+}X_6$, Sb–In donne
$A_2\mathrm{In}^{1+}\mathrm{Sb}^{3+}X_6$ et Sn–Ti donne
$A_2^{2+}\mathrm{Sn}^{4+}\mathrm{Ti}^{4+}O_6$. Pour les MXènes, l'état moyen
$M$ est déduit de la charge anionique totale, terminaisons $O_2$, $F_2$ et
$(OH)_2$ comprises. La charge nette stockée est **exactement nulle** pour les
2 000 candidats.

#### 3.2 Paramètres corrélés à la chimie

| Paramètre | Corrélation physique |
|---|---|
| $a$, $c$ | Rayons ioniques de Shannon ; $t=(r_A+r_X)/[\sqrt2(r_B+r_X)]$ et $a\simeq2(r_B+r_X)(0{,}98+0{,}04t)$ ; $c=a$ (pérovskites), $c=4{,}6$ ($M_2X$) ou $18{,}6$ Å ($M_3X_2$) |
| $t_1$, $t_2$ | Recouvrement orbitalaire : $t_1\propto e^{-(a-3{,}9)/1{,}2}$, $t_2=t_1(0{,}22+0{,}10\lambda/(\lambda+0{,}1))$ |
| $\lambda_{\mathrm{SOC}}$ | $\lambda=0{,}45\,(Z_{\max}/83)^4$ : 0,45 eV pour Bi/I, $\approx0{,}002$ eV pour les carbures de Ti |
| $E_g$ | $0{,}85+0{,}70\Delta\chi-0{,}35\lambda-0{,}10(a-4)+\Delta_{\text{famille}}$, borné à $[0{,}5\,;\,2{,}8]$ eV |
| $\mu_n$, $\mu_p$ | $\mu\propto t_1a^2/(1+8\lambda)$, $\mu_p\approx0{,}45\mu_n$ |
| $\beta_T$ | $-(0{,}0010+0{,}0028(1-E_g/2{,}8))(1+0{,}5\lambda)$ ; facteur 0,65 pour les doubles pérovskites (stabilité thermique supérieure) |
| Abondance | Moyenne pondérée des indices de ressources ouest-africaines (Fe, Ti, Mn $\approx1$ ; Ag, In, Cs $<0{,}3$) |
| Coût | Moyenne massique des coûts élémentaires (USD/kg) |

### 4. Résultats de la validation

Commande : `python -m pytest -v` → **229 tests passés** (187 + 42), soit 100 %.

| Contrôle | Résultat | Exigence |
|---|---|---|
| Candidats uniques | 2 000 identifiants et 2 000 formules | exactement 2 000 |
| Plomb / cobalt | 0 % | 0 % |
| Neutralité de charge | $|Q_{\text{net}}|=0$ sur 2 000 candidats | $<10^{-9}$ |
| Gaps optiques | $[1{,}366\,;\,2{,}689]$ eV | $[0{,}5\,;\,2{,}8]$ eV |
| Mobilités | $\mu_n\in[101\,;\,1281]$, $\mu_p\in[42\,;\,612]$ cm²/V/s | réaliste, $\mu_p<\mu_n$ |
| Déterminisme | empreinte JSON identique à graine 2026 | bit-à-bit |
| Figure 300 dpi | 792 797 octets, pHYs $\approx300$ dpi | ±1 dpi |

Répartition : 900 $ABX_3$, 550 $A_2B'B''X_6$, 550 MXènes. Les MXènes présentent
l'indice de ressource moyen le plus élevé (0,896) et le coût moyen le plus faible
(4,49 USD/kg), les pérovskites doubles la dérive thermique la plus faible
($\langle\beta_T\rangle=-1{,}42\times10^{-3}$ K$^{-1}$). Les valeurs moyennes
globales sont $\langle E_g\rangle=1{,}960$ eV, $\langle t_1\rangle=0{,}898$ eV,
$\langle\mu_n\rangle=554$ cm²/V/s et $\langle\text{coût}\rangle=100{,}6$ USD/kg.

### 5. Intégration des Nuits 1 à 7

Un échantillon représentatif (une formule par sous-famille) est injecté dans les
solveurs existants : Tight-Binding et DOS (Nuit 1), nombre de Chern FHS avec les
sauts du candidat (Nuit 2, $C=+1$), états de bord d'un ruban de Haldane alimenté
par $t_1$ et $t_2$ (Nuit 3), Shockley–Queisser sur $E_g$ (Nuit 4), bilan
thermodynamique sahélien avec $\beta_T$ du candidat (Nuit 5), transport
Poisson–dérive–diffusion couplé à $T_j$ (Nuit 6) et capacité quantique d'une
électrode MXène reconstruite depuis $a$, $c$ et $t_1$ (Nuit 7). Tous les solveurs
convergent avec des valeurs finies et physiquement bornées.

### 6. Figure de publication

`results/figures/night8_materials_landscape.png` — 300 dpi — quatre panneaux :
(a) histogrammes des gaps $E_g$ par famille, (b) corrélation $E_g$ – $\beta_T$,
(c) abondance ouest-africaine vs. coût colorée par $E_g$ (échelle log), et
(d) répartition des 2 000 candidats avec l'indice de ressource moyen par famille.

### 7. Bilan définitif du projet

Les huit nuits forment une chaîne complète et validée : cristallographie et
Tight-Binding (1) → courbure de Berry et Chern (2) → états de bord (3) → optique
AM1.5G et Shockley–Queisser (4) → thermodynamique sahélienne (5) → EDP de
transport et $J(V)$ (6) → supercondensateurs MXènes (7) → base de 2 000
matériaux sans plomb et validation croisée (8). Aucune dépendance SciPy n'a été
introduite ; 229 tests et huit figures de publication sont reproduits à chaque
exécution de `python -m pytest`.

**Diffusion :** Twin Transition Summit, 1er décembre 2026 — UAM / UFR STA / Cellule Physique.

---

## Nuit 9 — Découverte inverse & optimisation multi-objectifs (Front de Pareto NSGA-II)

**Date :** 18 novembre 2026  
**Branche de travail :** `main`  
**Objectif :** cribler les 2 000 matériaux de la Nuit 8 sous les climats extrêmes du
Sénégal, optimiser simultanément performance énergétique, stabilité thermique,
abondance minérale et coût, puis extraire le Top 10 des matériaux champions.

### 1. Objectifs

- Évaluer l'intégralité des 2 000 candidats à Matam ($T_{\text{amb}}=47\ ^\circ\mathrm C$,
  $G=1050\ \mathrm{W\,m^{-2}}$, vent $1{,}8\ \mathrm{m\,s^{-1}}$) et à Saint-Louis
  ($37\ ^\circ\mathrm C$, $980\ \mathrm{W\,m^{-2}}$, vent $4{,}0\ \mathrm{m\,s^{-1}}$).
- Implémenter un NSGA-II en NumPy pur (population 300, 250 générations, graine
  `SEED = 2026`), **sans aucune dépendance à SciPy**.
- Extraire un front de Pareto mathématiquement non dominé sur 4 critères et un
  Top 10 équilibré (5 photovoltaïques + 5 MXènes) incluant le Champion Solaire
  n° 1 et le Champion Supercondensateur MXène n° 1.
- Valider 100 % des tests (Nuits 1 à 9) et produire la figure 300 dpi
  `results/figures/night9_pareto_discovery.png`.

### 2. Fichiers créés

```
src/thermo/inverse_discovery.py   (run_screening, nsga2, optimize, non_dominated_sort,
                                  export_json, make_figure, discover, symbolic_stability)
tests/test_night9.py              (12 tests : criblage, non-dominance, NSGA-II,
                                  Top 10, champion vs silicium, Chern, figure)
results/top10_champions.json      (front complet de 346 points + Top 10, 744 ko)
results/figures/night9_pareto_discovery.png
```

### 3. Moteur de criblage multi-physique

Le pipeline vectorise et réutilise les solveurs des Nuits 1 à 7 : absorption
$\alpha(\lambda)$ pondérée par le spectre AM1.5G (Nuit 4), rendement de
Shockley–Queisser $\eta_{\rm SQ}(E_g,T_{\rm ref})$ et diode radiative (Nuit 4/6),
bilan thermique stationnaire $P_{\rm abs}=P_{\rm elec}+P_{\rm conv}+P_{\rm rad}$
résolu par Newton–bissection (Nuit 5), énergie et rétention de la cellule MXène
asymétrique (Nuit 7) et invariant de Chern du modèle de Haldane (Nuit 2).

Le rendement réel à Matam est
$\eta_{\rm Matam}=0{,}75\,\eta_{\rm SQ}(E_g,T_{\rm ref})\exp[\beta_T(T_j-T_{\rm ref})]$,
où $T_j$ est la température de jonction issue du bilan thermique couplé. Pour
les MXènes, l'objectif énergétique est la densité d'énergie à 47 °C modulée par
la rétention exacte de la cellule de la Nuit 7 ($98{,}11\ \%$ à $100\ \mathrm{A\,kg^{-1}}$).

### 4. Quatre objectifs et NSGA-II

| Objectif | Expression | Sens |
|---|---|---|
| $f_1$ performance | $\eta_{\rm Matam}/\eta_{\rm Si,Matam}$ (PV) ou $0{,}5E/E_{\rm ref}+0{,}5\,R_{47}$ (MXène) | maximiser |
| $f_2$ stabilité | $\exp[\beta_T(47-25)]$ (PV) ou rétention $R_{47}$ (MXène) | maximiser ($>95\ \%$) |
| $f_3$ ressource | indice d'abondance ouest-africaine | maximiser |
| $f_4$ coût | coût estimé de synthèse (USD/kg) | minimiser |

Le tri non dominé vectorisé ($O(N^2)$ sur 4 critères) et les distances de
crowding sont implémentés en NumPy pur. La sélection par tournoi binaire, le
croisement uniforme discret, la mutation et l'élitisme $\mu+\lambda$ produisent
une population convergeant vers le front global avec **87,9 % de précision**
(198 points NSGA-II, tous dans le front exact de 346 points). Le front exporté
est audité exhaustivement contre les 2 000 candidats : aucun point de la base
ne domine un point du front sur les 4 critères (tolérance $10^{-9}$).

### 5. Front de Pareto et Top 10 des matériaux champions

Référence silicium à Matam : $\eta_{\rm Si}=17{,}70\ \%$, $V_{oc}=0{,}825$ V,
$J_{sc}=42{,}25\ \mathrm{mA\,cm^{-2}}$, $FF=0{,}533$, $T_j=80{,}70\ ^\circ\mathrm C$.
Sélection : 5 pérovskites doubles et 5 MXènes les plus performants du front.

| # | ID | Formule | Structure | $E_g$ (eV) | $C$ | $V_{oc}$ (V) | $J_{sc}$ (mA/cm²) | $FF$ | $\eta_{\rm Matam}$ (%) | Gain vs Si (%) | Coût (USD/kg) | Stabilité (%) | $E_{47}$ (Wh/kg) |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | UAM-1384 | K2AgBiI6 | halide_A2AgBiX6 | 1.390 | +1 | 1.074 | 32.14 | 0.695 | 22.84 | +29.1 | 115.91 | 95.87 | 3.42 |
| 2 | UAM-1099 | Rb1K1AgBiI6 | halide_A2AgBiX6 | 1.450 | +1 | 1.129 | 29.83 | 0.699 | 22.42 | +26.7 | 119.19 | 95.97 | 3.41 |
| 3 | UAM-0945 | Rb0.5Na1.5AgBiI6 | halide_A2AgBiX6 | 1.453 | +1 | 1.133 | 29.70 | 0.699 | 22.40 | +26.5 | 119.97 | 95.97 | 3.42 |
| 4 | UAM-1277 | K1Cs1AgBiI6 | halide_A2AgBiX6 | 1.459 | +1 | 1.138 | 29.44 | 0.699 | 22.31 | +26.1 | 116.72 | 95.98 | 3.41 |
| 5 | UAM-1027 | Cs1.5Na0.5AgBiI6 | halide_A2AgBiX6 | 1.486 | +1 | 1.163 | 28.45 | 0.701 | 22.08 | +24.8 | 117.79 | 96.03 | 3.40 |
| 6 | UAM-1517 | Ti3C2O2 | M3X2 | 1.865 | −1 | 1.515 | 17.48 | 0.714 | 18.00 | +1.7 | 3.81 | 98.11 | 5.97 |
| 7 | UAM-1633 | Ti2C0.5N0.5O2 | M2X | 1.887 | −1 | 1.536 | 16.92 | 0.715 | 17.69 | −0.0 | 3.54 | 98.11 | 5.97 |
| 8 | UAM-1508 | Ti2N1O2 | M2X | 1.980 | −1 | 1.621 | 14.73 | 0.719 | 16.37 | −7.5 | 3.48 | 98.11 | 5.96 |
| 9 | UAM-1803 | Fe3C1.5N0.5 | M3X2 | 1.482 | −1 | 1.159 | 28.58 | 0.692 | 21.83 | +23.3 | 0.68 | 98.11 | 5.93 |
| 10 | UAM-1985 | Fe3N1C1 | M3X2 | 1.545 | −1 | 1.217 | 26.23 | 0.696 | 21.16 | +19.6 | 0.66 | 98.11 | 5.92 |

**Champion Solaire n° 1 :** `UAM-1384` K$_2$AgBiI$_6$, pérovskite double sans
plomb, $C=+1$, $\eta_{\rm Matam}=22{,}84\ \%$ soit **+29,1 %** sur le silicium.
Le solveur EDP complet de la Nuit 6 confirme l'avantage à Matam :
$\eta_{\rm PDE}=19{,}80\ \%$ contre $15{,}31\ \%$ pour le silicium (ratio 1,29).

**Champion Supercondensateur MXène n° 1 :** `UAM-1517` Ti$_3$C$_2$O$_2$,
$C=-1$, $E_{47}=5{,}97\ \mathrm{Wh\,kg^{-1}}$, rétention $98{,}11\ \%$ et coût
3,81 USD/kg. La famille MXène offre une abondance et une stabilité supérieures,
tandis que les pérovskites doubles topologiques dominent la performance solaire.

### 6. Validation mathématique

- 2 000/2 000 candidats évalués : aucun NaN, aucun dépassement numérique.
- Non-dominance stricte vérifiée pour les 346 points du front sur les 4 critères.
- Reproductibilité bit-à-bit du Top 10 et de la figure (graine 2026).
- Cohérence croisée : $T_j$ et $\alpha(\lambda)$ vectorisés vs. solveurs Nuits 4/5
  (écart $<0{,}1$ K), rétention MXène vs. Nuit 7, Chern analytique vs. courbure
  de Berry numérique FHS (grille $12\times12$) pour les 10 champions.
- `python -m pytest -v` : **241 tests passés (229 + 12), soit 100 %**.

### 7. Figure de publication

`results/figures/night9_pareto_discovery.png` — 300 dpi, trois panneaux :
(a) espace des 2 000 candidats (performance vs. abondance) avec le front de
Pareto et les 10 champions mis en valeur, (b) radar comparatif Champion n° 1 /
silicium sur 5 axes normalisés, et (c) barres comparatives $\eta$, $V_{oc}$,
$J_{sc}$, $FF$ et gain à Matam.

### 8. Bilan définitif du projet

Les neuf nuits forment une chaîne complète et validée : cristallographie et
Tight-Binding (1) → courbure de Berry et Chern (2) → états de bord (3) → optique
AM1.5G et Shockley--Queisser (4) → thermodynamique sahélienne (5) → EDP de
transport et $J(V)$ (6) → supercondensateurs MXènes (7) → base de 2 000
matériaux sans plomb (8) → découverte inverse et optimisation Pareto NSGA-II (9).
Aucune dépendance SciPy n'a été introduite ; 241 tests et neuf figures de
publication sont reproduits à chaque exécution de `python -m pytest`.

**Diffusion :** Twin Transition Summit, 1er décembre 2026 — UAM / UFR STA / Cellule Physique.

---

## Nuit 10 — Preuves formelles SymPy & moteur d'export PDF scientifique UAM

**Date :** 18 novembre 2026  
**Branche de travail :** `main`  
**Objectif :** certifier symboliquement les six équations fondatrices de la
chaîne UAM-TOPOMAT et produire automatiquement une fiche technique PDF
conforme à la charte de l'Université Amadou Mahtar Mbow, à partir du Top~10
de la Nuit~9.

### 1. Objectifs

- Dériver les six théorèmes avec SymPy en Python pur (**aucune dépendance
  à SciPy**), exporter les équations en LaTeX propre via `sympy.latex` et
  certifier que tous les résidus symboliques sont nuls.
- Vérifier numériquement que `UAM-1384` (K$_2$AgBiI$_6$) et `UAM-1517`
  (Ti$_3$C$_2$O$_2$) respectent les bornes théoriques exactes.
- Générer `paper_latex/fiche_technique_champions.tex` (charte UAM, encadrés
  `tcolorbox`, tableaux comparatifs, figures 300~dpi) et le compiler en
  `results/UAM_TopoMat_Fiche_Champions.pdf` avec `pdflatex`/`tectonic`.
- Écrire une suite pytest de validation syntaxique LaTeX, d'exactitude
  symbolique et de génération des fichiers ; exécuter `pytest -v`.

### 2. Fichiers créés

```
src/quantum/symbolic_proofs.py   (6 preuves SymPy, table de noms LaTeX,
                                  vérification des bornes, certificat LaTeX)
src/export_report.py             (générateur de fiche UAM + compilation PDF)
tests/test_night10.py            (13 tests : preuves, LaTeX, bornes, PDF, figures)
paper_latex/equations_verified.tex          (~11 ko, compilé en PDF)
paper_latex/fiche_technique_champions.tex   (~10 ko)
results/UAM_TopoMat_Fiche_Champions.pdf     (~2,3 Mo, pdflatex MiKTeX)
```

### 3. Les six théorèmes certifiés

1. **Tight-Binding / Hellmann--Feynman.** Pour la chaîne dimérisée
   $H=d_x\sigma_x+d_y\sigma_y$,
   $E_\pm(k)=\pm\sqrt{t_1^2+2t_1t_2\cos(ak)+t_2^2}$ et
   $v_g^{(n)}=\hbar^{-1}\partial_kE_n$. Le résidu
   $\operatorname{Tr}(P_\pm\partial_kH)-\partial_kE_\pm$ est **exactement nul**
   (vérifié avec le projecteur $P_\pm=(\mathbb{I}\pm H/E)/2$) ; la chaîne
   uniforme donne $v_g^{\mathrm{1D}}=-2at\sin(ak)/\hbar$.
2. **Courbure de Berry / Chern.** Pour $H=\mathbf{d}\cdot\boldsymbol{\sigma}$,
   $\Omega_-=-\tfrac12\hat{\mathbf{d}}\cdot(\partial_{k_x}\hat{\mathbf{d}}
   \times\partial_{k_y}\hat{\mathbf{d}})$ et
   $C_n=(2\pi)^{-1}\int_{\mathrm{BZ}}\Omega_n\,d^2k\in\mathbb{Z}$. SymPy
   certifie l'identité géométrique
   $\hat{\mathbf{d}}\cdot(\partial_x\hat{\mathbf{d}}\times\partial_y\hat{\mathbf{d}})
   =\sin\theta(\partial_x\theta\,\partial_y\varphi-\partial_y\theta\,\partial_x\varphi)$,
   la contribution $-\tfrac12\operatorname{sgn}(m)$ du cône de Dirac massif et
   les phases $C=\pm1$ ou $0$ du modèle de Haldane.
3. **Shockley--Queisser.** $J(V)=J_{sc}-J_0(e^{qV/k_BT}-1)$ avec
   $J_0=q(2\pi/h^3c^2)\int_{E_g}^{\infty}E^2/(e^{E/k_BT}-1)\,dE$ ; le maximum
   de puissance satisfait exactement $(1+v)e^v=1+J_{sc}/J_0$ avec
   $v=qV_{mp}/k_BT$, $J_{mp}=J_0ve^v$ et
   $\eta_{\mathrm{SQ}}=V_{mp}J_{mp}/P_{\mathrm{in}}$.
4. **Stefan--Boltzmann.** $\sigma_{SB}=2\pi^5k_B^4/(15h^3c^2)
   =5{,}670374419\times10^{-8}$~W\,m$^{-2}$\,K$^{-4}$ (accord relatif
   $3\times10^{-16}$). Le bilan
   $f(T_j)=\alpha G-\eta G-h_{\mathrm{conv}}(T_j-T_{amb})
   -\varepsilon\sigma_{SB}(T_j^4-T_{sky}^4)$ admet une racine unique car
   $f'(T_j)=-\beta\eta G-h_{\mathrm{conv}}-4\varepsilon\sigma_{SB}T_j^3<0$.
5. **Scharfetter--Gummel / Einstein.**
   $B(z)=z/(e^z-1)$ vérifie $B(-z)=B(z)+z$ et
   $B(z)=1-z/2+z^2/12+O(z^4)$. Le flux exponentiel se réécrit
   $J_{n,i+1/2}=A_i(w_{i+1}-w_i)$ en variables de Slotboom ($A_i>0$), garantie
   de la conservation exacte ; le flux net à l'équilibre impose
   $D=\mu\,k_BT_j/q$ (résidu SymPy nul).
6. **Capacité série $C_Q$ / $C_H$.** $C_Q=e^2\mathrm{DOS}(E_F+eV)/A$ et
   $C_{\mathrm{tot}}^{-1}=C_Q^{-1}+C_H^{-1}$, d'où
   $C_{\mathrm{tot}}=C_QC_H/(C_Q+C_H)<\min(C_Q,C_H)$ et
   $\partial C_{\mathrm{tot}}/\partial C_Q=C_H^2/(C_Q+C_H)^2$.

### 4. Vérification des champions

Dix-huit bornes exactes sont vérifiées pour chaque champion : gap borné,
$\mu_n>\mu_p>0$, $C\in\{-1,0,+1\}$, $\eta\le\eta_{\mathrm{SQ}}(E_g,T_{ref})$,
$\eta\le\eta_{\mathrm{SQ}}^{\max}$, $qV_{oc}\le E_g$, $0<FF<1$, $J_{sc}>0$,
$T_j>T_{amb}$, cohérence du bilan thermique ($\Delta T_j<0{,}1$~K et
résidu $<10^{-6}$~W\,m$^{-2}$), convection positive,
$0{,}02<V_T=k_BT_j/q<0{,}05$~V, $D_n/D_p=\mu_n/\mu_p$, $\beta_T<0$,
$0<R_{47}\le100\,\%$ et $0<E_{47}<100$~Wh\,kg$^{-1}$.

| Champion | $C$ | $\eta_{\mathrm{Matam}}$ (\%) | $V_{oc}$ (V) | $J_{sc}$ (mA/cm²) | $FF$ | $T_j$ (°C) | Bornes validées |
|---|---:|---:|---:|---:|---:|---:|---:|
| UAM-1384 (K$_2$AgBiI$_6$) | $+1$ | 22,84 | 1,074 | 32,14 | 0,695 | 75,92 | 18 / 18 |
| UAM-1517 (Ti$_3$C$_2$O$_2$) | $-1$ | 18,00 | 1,515 | 17,48 | 0,714 | 74,09 | 18 / 18 |

### 5. Export PDF et charte UAM

La fiche `fiche_technique_champions.tex` respecte strictement la charte :
**UNIVERSITÉ AMADOU MAHTAR MBOW / UFR SCIENCES ET TECHNOLOGIES AVANCÉES /
CELLULE PHYSIQUE / Réalisé par : Ndiogou Ndiaye / Année universitaire :
2026 -- 2027**. Elle contient des encadrés `tcolorbox`, le tableau comparatif
du Top~10, deux fiches détaillées, le socle des six théorèmes, le tableau des
36 bornes vérifiées et quatre figures 300~dpi (Nuits 1, 4, 7 et 9). La
compilation est automatique si `tectonic` ou `pdflatex` est détecté ; ici
`pdflatex` (MiKTeX) a produit `results/UAM_TopoMat_Fiche_Champions.pdf`
(2,3~Mo, compilation propre sans avertissement).

### 6. Validation

Commande : `python -m pytest -v` → **254 tests passés (241 + 13), soit 100 %**.
Les tests vérifient l'exactitude des six preuves (résidus $<10^{-12}$), la
validité syntaxique des deux documents LaTeX (accolades et environnements
équilibrés), la présence de la charte UAM et des 10 champions, les bornes des
deux champions, la résolution 300~dpi des figures et la compilation PDF.

### 7. Bilan définitif du projet

Les dix nuits forment une chaîne complète et validée : cristallographie et
Tight-Binding (1) → courbure de Berry et Chern (2) → états de bord (3) →
optique AM1.5G et Shockley--Queisser (4) → thermodynamique sahélienne (5) →
EDP de transport et $J(V)$ (6) → supercondensateurs MXènes (7) → base de
2~000 matériaux sans plomb (8) → découverte inverse NSGA-II (9) → preuves
formelles SymPy et export PDF UAM (10). Aucune dépendance SciPy n'a été
introduite ; 254 tests et dix figures/certificats sont reproduits à chaque
exécution de `python -m pytest`.

**Diffusion :** Twin Transition Summit, 1er décembre 2026 — UAM / UFR STA / Cellule Physique.
