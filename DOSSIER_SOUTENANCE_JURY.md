# DOSSIER DE SOUTENANCE JURY — UAM-TOPOMAT (Nuit 12 Finale)

**Université Amadou Mahtar Mbow (UAM) — UFR Sciences et Technologies Avancées — Cellule Physique**  
**Réalisé par : Ndiogou Ndiaye — Année universitaire : 2026 – 2027**  
**Diffusion : Twin Transition Summit, 1er décembre 2026 — Dakar, Sénégal**

---

## 0. Fiche express (30 secondes)

| Indicateur | Valeur |
|---|---|
| Chaîne informative | 12 nuits, 11 figures 300 dpi, 6 théorèmes certifiés, 282 tests pytest |
| Matériaux criblés | 2 000 candidats sans plomb ni cobalt |
| Champion solaire | UAM-1384 — K$_2$AgBiI$_6$ — $C=+1$ — $\eta_{\text{Matam}} = 22{,}84\,\%$ (+29,1 % vs Si) |
| Champion stockage | UAM-1517 — Ti$_3$C$_2$O$_2$ — $C=-1$ — $E_{47} = 5{,}97$ Wh/kg — 98,11 % de rétention |
| Impact Sénégal | **+126,9 MWh/an · 69,8 t CO$_2$/an · 46,2 M FCFA/an · LCOE −44 à −49 %** |
| Souveraineté | 100 % hors ligne, Python pur, zéro dépendance SciPy, zéro CDN |
| Livrables Nuit 12 | Article 17 pages, poster A0, logo UAM, dossier jury, test final |

---

## 1. Pitch complet (3 minutes)

> Monsieur le Président, Mesdames et Messieurs les membres du jury,

> Le Sahel est un paradoxe énergétique. Notre région reçoit entre 1 500 et 2 200 kWh par mètre carré et par an d'ensoleillement — parmi les plus forts du monde — et pourtant, des millions de nos concitoyens attendent encore une électricité fiable et abordable. À Matam, la température ambiante atteint 47 °C. À cette température, un module silicium perd plusieurs points de rendement, et une batterie classique vieillit deux à trois fois plus vite qu'à Dakar. Nous importons simultanément la technologie, le logiciel de conception et la décision d'investissement. Ce n'est pas seulement un déficit énergétique : c'est un déficit de souveraineté.

> Notre réponse s'appelle **UAM-TOPOMAT**. C'est une plateforme numérique souveraine, développée ici, à l'Université Amadou Mahtar Mbow, par la Cellule Physique de l'UFR Sciences et Technologies Avancées, entièrement en Python, exécutable hors ligne sur un simple ordinateur portable, sans service cloud étranger et sans dépendance à SciPy. En douze nuits de travail, nous avons construit une chaîne complète qui va du cristal hypothétique jusqu'au bilan en francs CFA.

> Concrètement, cette chaîne comporte cinq étapes. Nous calculons d'abord les bandes électroniques par Tight-Binding et nous validons le cône de Dirac du graphène à 8,74 × 10⁵ mètres par seconde. Ensuite, nous calculons la courbure de Berry et l'invariant de Chern par la méthode de Fukui–Hatsugai–Suzuki, et nous vérifions les états de bord : un matériau topologique de Chern +1 possède exactement deux canaux conducteurs sur son bord. Troisièmement, nous couplons l'optique AM1.5G et la limite de Shockley–Queisser — 33,94 % à 1,34 eV — à un bilan radiatif–convectif qui détermine la vraie température de jonction sous climat sahélien. Quatrièmement, nous résolvons le transport de porteurs par le schéma conservatif de Scharfetter–Gummel et nous modélisons le stockage MXène avec sa capacité quantique. Cinquièmement, nous criblons 2 000 composés sans plomb avec un algorithme génétique NSGA-II à quatre objectifs.

> Les résultats sont nets. Sur ces 2 000 candidats, deux champions émergent. Le premier, **UAM-1384**, est une pérovskite double sans plomb, K$_2$AgBiI$_6$ : gap de 1,39 eV, invariant de Chern +1, rendement calculé de **22,84 % à Matam**, soit 29 % de plus que le silicium de référence, et surtout une dérive thermique deux fois plus faible. Le second, **UAM-1517**, est un MXène, Ti$_3$C$_2$O$_2$ : il conduit remarquablement bien, stocke 5,97 Wh/kg à 47 °C avec 98,11 % de capacité retenue, coûte 3,81 dollars le kilo et supporte 500 000 cycles.

> Mais un article scientifique ne s'arrête pas à une courbe. Nous avons donc placé ce couple champion dans une mini-centrale de 100 kWc avec 100 kWh de stockage, à Dakar, Saint-Louis et Matam. Résultat : **126,9 MWh supplémentaires par an, 69,8 tonnes de CO$_2$ évitées, 46,2 millions de FCFA économisés chaque année, et une réduction du coût actualisé de l'électricité de 44 à 49 %**. Le gain est le plus fort à Matam, précisément là où le climat est le plus sévère — c'est le cœur de notre message.

> Je dois être transparent sur les limites. Ce sont des modèles Tight-Binding paramétrés, pas des calculs DFT ab initio. Aucun échantillon n'a encore été synthétisé. Le facteur de module de 0,75 est une hypothèse de conception. Notre front NSGA-II couvre 87,9 % du front exact. Nous publions ces chiffres avec les bornes, les résidus symboliques nuls et les 282 tests qui les rendent auditables — pas comme des vérités de laboratoire, mais comme une feuille de route vérifiable.

> La prochaine étape est claire : calcul DFT avec couplage spin–orbite, synthèse de K$_2$AgBiI$_6$ et de Ti$_3$C$_2$O$_2$ au laboratoire, puis un prototype instrumenté de 1 kWc à Matam. Ce que nous démontrons aujourd'hui, ce n'est pas seulement un matériau performant — c'est qu'une chaîne scientifique complète, souveraine et reproductible peut être conçue et validée ici, au Sénégal, pour le Sahel et pour l'Afrique. Je vous remercie.

---

## 2. Scénario de démonstration en direct (8 minutes)

### 2.1 Pré-vol (avant l'arrivée du jury)

1. Vérifier que `results/UAM_TopoMat_Fiche_Champions.pdf`, `results/Article_UAM_TOPOMAT_2026.pdf`, `results/Poster_UAM_TOPOMAT_2026.pdf`, `results/figures/night11_senegal_impact.png` et `app/static/logo_uam.png` existent.
2. Lancer `python -m pytest -q` une fois : **282 tests + 18 tests Nuit 12 doivent passer**.
3. Garder le poster A0 ouvert en PDF et l'article à la page de garde (logo UAM visible).

### 2.2 Déroulé

| Temps | Action | Résultat attendu / point à commenter |
|---|---|---|
| 0:00 | Double-clic sur **`LANCER_DEMO_JURY.bat`** | Installation silencieuse de FastAPI/uvicorn si besoin ; ouverture du serveur `UAM-TOPOMAT-Serveur` puis du navigateur sur `http://127.0.0.1:8050`. Aucun accès Internet requis. |
| 0:30 | Montrer l'en-tête : **logo UAM en haut à gauche**, titre UAM-TOPOMAT — Cellule Physique UFR STA | Charte officielle ; souligner « 100 % hors ligne » en bas de page. |
| 1:00 | Dans « Matériau », choisir **UAM-1384 (K$_2$AgBiI$_6$)** ; laisser ville = Matam, T = 47 °C | Le couple champion est présélectionné ; expliquer pourquoi Matam est le scénario le plus sévère. |
| 1:30 | Cliquer sur **« Lancer le Pipeline Automatique Complet »** (POST `/api/pipeline/run`) | Les cinq panneaux se remplissent : bandes + Fermi, courbure de Berry (Ω), J(V) avec MPP, Ragone, bilan d'impact. |
| 2:15 | Zoomer sur le panneau (b) : Chern $C=+1$ | Rappeler la méthode FHS et la correspondance $2\|C\|$ canaux de bord. |
| 2:45 | Zoomer sur (c) J(V) : $V_{oc}\approx1{,}074$ V, $J_{sc}\approx32{,}1$ mA/cm², $\eta\approx22{,}8\%$ | Montrer que le modèle respecte $\eta\le\eta_{\mathrm{SQ}}$ et $qV_{oc}\le E_g$. |
| 3:20 | Remplacer le matériau par **UAM-1517 (Ti$_3$C$_2$O$_2$)** ; relancer | Passage au $C=-1$, $E_{47}=5{,}97$ Wh/kg ; expliquer le rôle de stockage. |
| 4:00 | Choisir successivement **Dakar (31 °C)** puis **Saint-Louis (37 °C)**, puis revenir à Matam | Les KPI se mettent à jour : le gain passe de +34,8 à +40,2 à +51,9 MWh/an. Insister sur la démonstration du gradient climatique. |
| 5:00 | Déplacer le curseur de température à **55 °C** | Montrer que l'outil répond sur toute la plage validée [25 ; 55] °C ; dire explicitement que c'est la limite du domaine de validité. |
| 5:30 | Cliquer sur **« Télécharger la Fiche Technique PDF »** (`/api/report/pdf`) | Fiche UAM officielle avec logo, Top 10, 4 figures 300 dpi et 36 bornes vérifiées. |
| 6:15 | Ouvrir `results/Article_UAM_TOPOMAT_2026.pdf` (17 pages) puis le poster A0 | Montrer la page de garde avec logo, le sommaire, la section impact et le poster prêt à imprimer. |
| 7:00 | Ouvrir un terminal et lancer `python -m pytest -q tests/test_night12_final.py` | Tous les livrables, le logo, les PDF et les routes web sont vérifiés en direct. |
| 7:40 | Questions du jury | Se référer à la section 3 ; toujours distinguer **prédiction de modèle** et **mesure expérimentale**. |

### 2.3 Plan B (si le réseau local ou le navigateur échoue)

- Le serveur est sur `127.0.0.1:8050` : vérifier dans la fenêtre « UAM-TOPOMAT-Serveur » qu'uvicorn affiche `Application startup complete`.
- Si le port est occupé : `python -m uvicorn app.server:app --host 127.0.0.1 --port 8051` puis ouvrir `http://127.0.0.1:8051`.
- Si le PDF ne se télécharge pas : ouvrir directement `results/UAM_TopoMat_Fiche_Champions.pdf` depuis l'explorateur ; le bouton ne fait que servir ce fichier.
- Toujours possible : `python -m src.thermo.senegal_grid_impact` régénère la figure et le JSON, et `python -m src.export_report` régénère la fiche.

---

## 3. Douze questions difficiles du jury — réponses précises et honnêtes

### Q1. « Ces rendements de 22,84 % sont-ils mesurés en laboratoire ou calculés ? »

**Calculés, pas mesurés.** Le chiffre provient du couplage Shockley–Queisser + bilan radiatif–convectif + paramètres matériau de la base. Deux garde-fous : (i) il reste sous la limite de Shockley–Queisser du même gap (22,8413 %) ; (ii) un calcul indépendant par le solveur d'équations aux dérivées partielles (Nuit 6) donne 19,80 % pour UAM-1384 contre 15,31 % pour le silicium, donc la tendance reste stable même si le niveau absolu change. Aucun échantillon n'a été fabriqué : c'est une prédiction de modèle, pas une donnée expérimentale.

### Q2. « Pourquoi n'avez-vous pas fait de DFT ? Un Tight-Binding paramétré est-il scientifiquement recevable ? »

C'est un choix de compromis assumé, pas une négligence. Le criblage porte sur 2 000 composés : une étude DFT avec couplage spin–orbite sur chacun est hors de portée d'une chaîne souveraine exécutable sur portable et contredit l'objectif d'autonomie. Le Tight-Binding paramétré est recevable **à condition d'être explicite et borné** : corrélations chimiques documentées, graine fixe 2026, six théorèmes certifiés par SymPy (résidus < 10⁻¹²) et 36 bornes physiques vérifiées. Je ne présente pas ces gaps comme des vérités ab initio. La prochaine étape est précisément un calcul DFT G₀W₀ + spin–orbite sur les deux champions pour valider ou corriger les paramètres.

### Q3. « L'invariant de Chern $C=+1$ de K$_2$AgBiI$_6$ n'est-il pas un artefact de votre modèle jouet ? »

La question est légitime. Le nombre de Chern est calculé par la méthode de Fukui–Hatsugai–Suzuki sur le Hamiltonien Tight-Binding construit à partir du gap et du couplage spin–orbite $\lambda_{\mathrm{SOC}}=0{,}45$ eV du candidat. La méthode numérique elle-même est robuste (quantification à 10⁻⁶, tests sur Haldane et QWZ où la réponse analytique est connue). En revanche, **la valeur physique dépend entièrement du modèle de bandes** : une modification des sauts ou de la terminologie de surface peut fermer le gap et changer $C$. Je considère donc ce $C$ comme une prédiction de phase topologique à confirmer par DFT et, idéalement, par une mesure de Hall quantique ou d'ARPES. L'article le dit explicitement.

### Q4. « D'où vient le facteur module 0,75 ? Est-ce arbitré ? »

C'est une **hypothèse de conception**, pas une mesure. Elle traduit les pertes typiques d'un module réel par rapport à la limite d'équilibre détaillé (EQE 0,92, absorption sélective α₊ = 0,95 / α₋ = 0,60, émissivité 0,90). Elle est cohérente avec les ordres de grandeur industriels, mais elle mérite une analyse de sensibilité : c'est une limite que nous signalons nous-mêmes dans la section 6. Si le facteur était 0,65, le gain brut diminuerait, mais l'écart relatif entre le couple champion et le couple silicium/plomb resterait, lui, porté par les coefficients thermiques $\beta_T$ et la cyclabilité.

### Q5. « Sur quoi repose exactement l'économie de 46,2 M FCFA/an ? »

Sur un jeu d'hypothèses que nous affichons : tarif SENELEC 220 FCFA/kWh, facteur d'émission 0,55 t CO₂/MWh, taux d'actualisation 5 % sur 20 ans, change 600 FCFA/USD, O&M 2 % du CAPEX PV et 1 % du stockage, cyclabilité 500 000 cycles pour le MXène contre 2 750 pour le plomb/Li-ion. Le poste dominant est le **renouvellement évité des batteries**. Nous ignorons volontairement taxes, logistique, purification, main-d'œuvre et risque de change : le chiffre est donc une indication d'ordre de grandeur, pas un business plan bancable. La formule du CRF et l'identité LCOE × énergie = coût sont certifiées nulles par SymPy.

### Q6. « Le champion solaire contient de l'argent à 720 USD/kg. Est-il vraiment "abordable" à 115 USD/kg ? »

Excellente objection. Notre indice d'abondance est un indicateur **géologique ouest-africain**, pas un prix de marché : il valorise la disponibilité locale, pas le coût de raffinage. L'argent est effectivement cher et K$_2$AgBiI$_6$ reste 30 fois plus coûteux que Ti$_3$C$_2$O$_2$ à la masse. Deux réponses : (i) la masse active d'une cellule est de l'ordre du gramme par watt, donc l'impact sur le CAPEX d'un module est bien moindre que le prix matière brute ne le suggère ; (ii)NSGA-II a précisément conservé des alternatives moins chères (UAM-1803 Fe$_3$C$_{1,5}$N$_{0,5}$ à 0,68 USD/kg, $\eta=21{,}83\,\%$) dans le Top 10. Nous ne prétendons pas que K$_2$AgBiI$_6$ est le moins cher : c'est le meilleur compromis performance/stabilité thermique sous notre pondération [0,40 ; 0,15 ; 0,25 ; 0,20], qui doit être discutée avec les partenaires industriels.

### Q7. « Votre NSGA-II ne trouve que 87,9 % du front exact. Le Top 10 est-il fiable ? »

Le chiffre est honnête : 198 solutions approchées sur 346 solutions exactes, taux de récupération 50,3 %, hypervolume 2 151,99. Cela signifie que des compromis intermédiaires peuvent manquer — je ne le cache pas. En revanche, les **deux champions retenus sont de rang 0 et strictement non dominés sur les 2 000 candidats** (tolérance 10⁻⁹) : leur appartenance au front de Pareto est vérifiée indépendamment de l'algorithme. Une population plus large, plus de générations ou NSGA-III amélioreraient la couverture ; le Top 10 reste donc robuste, mais la carte complète des compromis est perfectible.

### Q8. « Avez-vous comparé à de vraies technologies commerciales, pas à un silicium "de référence" ? »

Nous comparons à un module silicium dont le rendement de référence est $\eta_{\mathrm{Si}}=17{,}70\,\%$ à Matam avec $\beta_T=-4{,}5\times10^{-3}$ K⁻¹, et à un stockage plomb/Li-ion 50/50 (112,5 Wh/kg, 0,86 de rendement aller-retour, 2 750 cycles). C'est une référence physique explicite, mais ce n'est pas une fiche constructeur : pas de garantie, pas de dégradation mesurée, pas de coût d'installation réel. Pour un jury industriel, il faudrait remplacer cette référence par des données de bancs de test (PERC/TOPCon et LFP/NMC) — c'est dans nos perspectives.

### Q9. « Comment un jury peut-il auditer vos résultats ? »

Trois niveaux. (i) **Déterminisme** : graines fixes (2026), base de 2 000 composés et JSON de Pareto régénérés à l'identique, comparaison bit-à-bit dans les tests. (ii) **Traçabilité** : six certificats LaTeX avec résidus symboliques nuls, trente-six bornes numériques, onze figures 300 dpi toutes produites par le code. (iii) **Exécution** : `pytest -v` → 282 tests (plus 18 tests de la Nuit 12) vérifient physique, symbolique, LaTeX, routes HTTP 200 et intégrité des PDF. Tout tourne hors ligne ; aucun service externe ne peut masquer un résultat.

### Q10. « Que se passe-t-il au-delà de 47 °C, par exemple à 55 °C ? »

L'interface accepte 25 à 55 °C et les tests couvrent explicitement les trois bornes 25, 47 et 55 °C. À 55 °C, les lois de dégradation thermique et de perte de pores sont encore évaluées, mais le rendement baisse et l'écart avec le silicium se réduit. **Au-delà de 55 °C, nous ne revendiquons aucune validité** : le modèle de perte de pores $a_p(T)=e^{-\lambda(T-313{,}15)_+}$ est une extrapolation linéaire en exponentielle, et la résistance ESR suit une loi d'Arrhenius dont les paramètres sont calibrés vers 25–50 °C. Il faudrait des données de vieillissement accéléré pour aller plus loin.

### Q11. « Pourquoi s'obstiner sur Python pur et refuser SciPy ou le cloud ? »

Pour la souveraineté, qui est le cœur du projet. Un élu, un ingénieur ou un étudiant au Sénégal doit pouvoir télécharger le dépôt, lancer un `.bat`, et obtenir les mêmes chiffres **sans carte bancaire, sans compte cloud et sans dépendance indisponible**. NumPy, SymPy et Matplotlib suffisent : le pipeline complet s'exécute en quelques secondes ; Ngspice/COMSOL n'auraient rien apporté à la traçabilité. Le rejet de SciPy est un choix de minimalisme vérifiable (un test interdit tout import SciPy), pas un rejet des bibliothèques scientifiques en général. Le passage à des solveurs HPC ou à DFT se fera en aval, sur des machines dédiées, tout en gardant cette chaîne légère comme référence.

### Q12. « Quel est le chemin exact du modèle au prototype, et qu'est-ce qui pourrait vous faire mentir ? »

Feuille de route en quatre étapes : (1) DFT + spin–orbite et G₀W₀ sur K$_2$AgBiI$_6$ et Ti$_3$C$_2$O$_2$ — trois mois ; (2) synthèse en laboratoire et caractérisation XRD/UV–Vis/Hall/électrochimie — six mois ; (3) cellule de quelques cm² puis module, avec mesures $J(V)$ à 25, 37 et 47 °C ; (4) prototype instrumenté de 1 kWc à Matam avec partenaires locaux. Ce qui pourrait nous faire mentir : un gap DFT très différent de 1,39 eV, un invariant de Chern qui s'effondre avec une meilleure description de la structure, une dégradation humide rapide de la pérovskite double, ou un coût réel de gravure du MXène supérieur à notre indice. Nous avons intentionally écrit ces risques dans la section « Limites » : un résultat scientifique se juge aussi à sa capacité à être réfuté.

---

## 4. Checklist des livrables Nuit 12

- [x] `app/static/logo_uam.png` et `paper_latex/logo_uam.png` — sceau officiel UAM (remplaçable par écrasement)
- [x] `app/static/index.html` — logo en haut à gauche, à côté du titre UAM-TOPOMAT — Cellule Physique UFR STA
- [x] `src/export_report.py` — logo sur la page de garde de la fiche PDF officielle
- [x] `paper_latex/main.tex` → `results/Article_UAM_TOPOMAT_2026.pdf` — **17 pages**, double passe pdflatex
- [x] `paper_latex/poster.tex` → `results/Poster_UAM_TOPOMAT_2026.pdf` — **1 page A0 portrait**
- [x] `DOSSIER_SOUTENANCE_JURY.md` — pitch, démo, 12 questions
- [x] `tests/test_night12_final.py` — audit d'intégrité des livrables
- [x] `JOURNAL_RECHERCHE.md` — entrée Nuit 12

**Diffusion :** Twin Transition Summit, 1er décembre 2026 — UAM / UFR STA / Cellule Physique.
