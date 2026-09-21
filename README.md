# Diagnostic du paludisme par deep learning sur frottis sanguin

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CNN%20%2B%20transfer%20learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![torchvision](https://img.shields.io/badge/torchvision-ResNet18-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/vision/stable/index.html)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-métriques%20%2B%20split-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![CI](https://github.com/juliettebm/malaria-microscopy-deep-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/juliettebm/malaria-microscopy-deep-learning/actions/workflows/ci.yml)
[![Reproductibilité](https://img.shields.io/badge/reproductibilit%C3%A9-environnement%20fig%C3%A9-success)](environment.yml)

Un projet de bout en bout d'IA appliquée à la biologie médicale : structurer, qualifier et modéliser 27 558 images de frottis sanguins pour détecter le paludisme, puis évaluer le modèle avec des indicateurs cliniques plutôt qu'une seule accuracy.

---

## Objectif

L'examen microscopique du frottis sanguin reste l'examen de référence pour le diagnostic du paludisme en biologie médicale, mais c'est une lecture humaine longue, répétitive et sujette à variabilité inter-observateur. Ce projet explore ce qu'une chaîne IA peut apporter à cette étape : recenser et qualifier les images disponibles, construire un jeu de données annoté exploitable, comparer deux approches de modélisation, puis évaluer le résultat avec les indicateurs qu'un laboratoire regarderait réellement (sensibilité, spécificité, valeurs prédictives), pas uniquement une accuracy globale.

Le projet reproduit les étapes d'une mission de chargé(e) de projet IA/data science appliquée à la biologie médicale : audit et structuration de données image, état de l'art et faisabilité de modèles ML/DL, définition d'indicateurs de performance, puis formalisation de résultats et de recommandations sur la valeur médicale et opérationnelle.

---

## Jeu de données

- **Source** : [Malaria Cell Images Dataset](https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip), National Library of Medicine (NIH, Lister Hill National Center for Biomedical Communications), public.
- **Taille** : 27 558 images de cellules segmentées issues de frottis sanguins colorés, équilibrées : 13 779 parasitées, 13 779 non infectées.
- **Label** : chaque image est annotée par des experts en parasitologie du Chittagong Medical College Hospital (Bangladesh).
- **Qualité** : audit complet (hash MD5 + `PIL.verify()`) sur les 27 558 fichiers, 0 corrompu, 0 doublon après exclusion des fichiers `Thumbs.db` non-image présents dans les dossiers bruts.
- **Accès** : téléchargement direct depuis le serveur NIH (voir Reproduire). Données brutes et traitées non versionnées (voir `.gitignore`).

---

## Structure du projet

```
malaria-microscopy-deep-learning/
│
├── 01_data_ingestion_eda.ipynb                     # inventaire, audit qualité, EDA
├── 02_preprocessing_annotated_dataset.ipynb        # split par patient, pipeline, jeu de données annoté
├── 03_baseline_cnn_vs_transfer_learning.ipynb      # état de l'art, CNN from scratch vs transfer learning
├── 04_clinical_evaluation_interpretability.ipynb   # indicateurs cliniques, interprétabilité, discussion
├── data/                                            # non versionné (voir .gitignore)
│   ├── raw/                                         # dataset NIH brut (cell_images/)
│   └── processed/                                   # manifestes annotés (inventory, train, val, test)
├── models/                                          # poids entraînés, non versionné
├── malaria_evaluation.py                            # métriques et bootstrap par patient
├── scripts/evaluate_predictions.py                  # évaluation interne/externe reproductible
├── docs/external_validation.md                      # protocole de validation externe
├── results/key_results.json                         # valeurs de référence contrôlées par les tests
├── environment.yml                                  # environnement Conda verrouillé
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Reproduire

### 1. Cloner

```bash
git clone https://github.com/juliettebm/malaria-microscopy-deep-learning.git
cd malaria-microscopy-deep-learning
```

### 2. Installer

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS : source .venv/bin/activate
pip install -r requirements.txt
```

Pour reproduire strictement l'environnement utilisé :

```bash
conda env create -f environment.yml
conda activate malaria-microscopy
```

### 3. Données

Télécharger et extraire le dataset NIH dans `data/raw/` :

```
data/raw/cell_images/Parasitized/*.png
data/raw/cell_images/Uninfected/*.png
```

### 4. Exécuter

Ouvrir et exécuter les notebooks dans l'ordre (01 à 04). Le notebook 02 écrit les manifestes que 03 et 04 consomment ; le notebook 03 sauvegarde le modèle retenu que 04 évalue.

### 5. Tests rapides

Les smoke tests vérifient la validité et la compilation des quatre notebooks, les contrats de fichiers entre les étapes, les garde-fous du split par patient, un passage avant du CNN sur des tenseurs synthétiques, les métriques cliniques et la calibration. Ils contrôlent aussi que les résultats clés versionnés dans `results/key_results.json` apparaissent toujours dans le README. Ils ne téléchargent pas les données et ne lancent pas l'entraînement.

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
```

Ces tests sont exécutés automatiquement par GitHub Actions à chaque push et pull request.

---

## Méthodologie

1. **Ingestion et audit qualité** (01) : inventaire des 27 558 images, audit des dimensions/mode, détection de fichiers corrompus et de doublons, balance des classes, échantillon visuel par classe.
2. **Jeu de données annoté** (02) : split 70/15/15 (seed 42) **groupé par patient** (`GroupShuffleSplit`, 200 patients : 140 / 30 / 30, l'identifiant étant lu dans le nom de fichier), pipeline de prétraitement, manifestes `train.csv` / `val.csv` / `test.csv` incluant `patient_id`, manifeste complet `split_manifest.csv` et matrice `patient_class_matrix.csv`.
3. **État de l'art et modélisation** (03) : comparaison d'un CNN entraîné from scratch et d'un ResNet18 pré-entraîné (transfer learning), sur toutes les images du split d'entraînement.
4. **Évaluation clinique et interprétabilité** (04) : indicateurs cliniques avec IC à 95 % par bootstrap de 2 000 rééchantillonnages de patients (seed 42), cartes de saillance, simulation de l'effet de la prévalence sur la VPP et export des prédictions auditables.

### Matrice patients/classes des splits

| Split | Classe | Patients uniques | Images |
| --- | --- | ---: | ---: |
| Train | Parasitized | 106 | 10 234 |
| Train | Uninfected | 140 | 9 681 |
| Validation | Parasitized | 22 | 1 573 |
| Validation | Uninfected | 30 | 2 050 |
| Test | Parasitized | 22 | 1 972 |
| Test | Uninfected | 30 | 2 048 |

Un même patient peut contribuer aux deux classes ; le total de patients uniques par split reste 140 / 30 / 30. Les affectations exactes sont enregistrées dans `data/processed/split_manifest.csv` lors de l'exécution.

---

## Résultats clés

### Comparaison des modèles (validation, 8 epochs, CPU)

| Modèle | Paramètres entraînés | Val Accuracy | Val F1 | Temps d'entraînement |
| --- | ---: | :---: | :---: | ---: |
| **CNN from scratch** | 548 258 | **96,85 %** | **96,34 %** | 324,9 s |
| ResNet18 (transfer learning) | 1 026 | 80,02 % | 80,77 % | 418,8 s |

Le CNN from scratch bat le transfer learning ImageNet : la texture de coloration microscopique n'est pas bien représentée dans les features pré-entraînées sur des photos naturelles, un rappel utile face au réflexe "transfer learning toujours gagnant".

### Évaluation clinique (test set : 4 020 images de 30 patients jamais vus)

| Métrique | Valeur |
| --- | :---: |
| Accuracy | 96,49 % [IC 95 % patient : 95,12–97,62] |
| **Sensibilité** (rappel Parasitized) | 95,08 % [91,65–97,11] |
| Spécificité | 97,85 % [97,11–98,49] |
| VPP | 97,71 % [95,60–98,62] |
| VPN | 95,38 % [92,88–97,58] |
| ROC-AUC | 99,20 % [98,68–99,54] |
| Score de Brier | 0,0285 [0,0200–0,0382] |

La calibration interne donne une **ECE à 10 intervalles de 0,0075** (plus proche de 0 est meilleur). Ces deux mesures décrivent la qualité des probabilités prédites, mais elles restent estimées sur la même petite cohorte interne de 30 patients et ne remplacent pas une validation externe.

**Matrice de confusion :**

| | Prédit Parasitized | Prédit Uninfected |
| --- | :---: | :---: |
| **Réel Parasitized** | 1 875 (VP) | 97 (FN) |
| **Réel Uninfected** | 44 (FP) | 2 004 (VN) |

**Effet de la prévalence sur la VPP** : ce dataset est équilibré 50/50, ce qui n'est pas la prévalence réelle terrain. Une simulation bayésienne à partir de la sensibilité/spécificité mesurées montre qu'à une prévalence de 2 % (dépistage en zone peu endémique), la VPP chute à 47,46 % malgré une ROC-AUC de 99,20 %, un rappel que la performance d'un test dépend du contexte de déploiement, pas seulement du modèle.

---

## Limites

- **Petit nombre de patients dans le test** : le split est groupé par patient et les IC à 95 % sont calculés par bootstrap au niveau patient, mais 30 patients restent une petite cohorte. Les proportions de classes ne sont plus exactement 50/50 dans chaque split (validation 43/57, test 49/51).
- Un seul type de microscope et de coloration (dataset Chittagong Medical College Hospital). La validation externe n'est pas encore exécutée faute de second corpus avec identifiants patient ; le protocole gelé et la commande d'analyse sont fournis dans [`docs/external_validation.md`](docs/external_validation.md).
- Prévalence artificiellement équilibrée à 50/50 dans le dataset, contrairement à la prévalence réelle sur le terrain (voir simulation ci-dessus).
- Un déploiement clinique réel nécessiterait un marquage réglementaire CE-IVD, une validation clinique multicentrique, et positionnerait l'outil en aide au tri / second lecteur pour le technicien de laboratoire, jamais en diagnostic autonome.

---

## Avertissement

Projet pédagogique sur un dataset public (NIH). Utilisé à des fins d'analyse et de démonstration uniquement, pas destiné à un usage diagnostique réel.

---

## Stack

Python 3.11 · pandas · scikit-learn · PyTorch · torchvision · Pillow · Matplotlib / Seaborn

---

## License

Released under the [MIT License](LICENSE).

---

## Autrice

**Juliette Bouli-Mengue**
De la recherche clinique à la data science
