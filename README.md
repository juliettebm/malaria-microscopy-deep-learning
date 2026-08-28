# Diagnostic du paludisme par deep learning sur frottis sanguin

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CNN%20%2B%20transfer%20learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![torchvision](https://img.shields.io/badge/torchvision-ResNet18-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/vision/stable/index.html)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-métriques%20%2B%20split-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

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
├── 02_preprocessing_annotated_dataset.ipynb        # split stratifié, pipeline, jeu de données annoté
├── 03_baseline_cnn_vs_transfer_learning.ipynb      # état de l'art, CNN from scratch vs transfer learning
├── 04_clinical_evaluation_interpretability.ipynb   # indicateurs cliniques, interprétabilité, discussion
├── data/                                            # non versionné (voir .gitignore)
│   ├── raw/                                         # dataset NIH brut (cell_images/)
│   └── processed/                                   # manifestes annotés (inventory, train, val, test)
├── models/                                          # poids entraînés, non versionné
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

### 3. Données

Télécharger et extraire le dataset NIH dans `data/raw/` :

```
data/raw/cell_images/Parasitized/*.png
data/raw/cell_images/Uninfected/*.png
```

### 4. Exécuter

Ouvrir et exécuter les notebooks dans l'ordre (01 à 04). Le notebook 02 écrit les manifestes que 03 et 04 consomment ; le notebook 03 sauvegarde le modèle retenu que 04 évalue.

---

## Méthodologie

1. **Ingestion et audit qualité** (01) : inventaire des 27 558 images, audit des dimensions/mode, détection de fichiers corrompus et de doublons, balance des classes, échantillon visuel par classe.
2. **Jeu de données annoté** (02) : split stratifié 70/15/15 (seed fixe), pipeline de prétraitement (resize, normalisation, augmentation sur le train uniquement), manifestes annotés `train.csv` / `val.csv` / `test.csv`.
3. **État de l'art et modélisation** (03) : comparaison d'un CNN entraîné from scratch et d'un ResNet18 pré-entraîné (transfer learning), sur un sous-échantillon stratifié de 6 000 images d'entraînement pour un temps de calcul raisonnable en CPU (pas de GPU disponible).
4. **Évaluation clinique et interprétabilité** (04) : indicateurs cliniques (sensibilité, spécificité, VPP, VPN, ROC-AUC, matrice de confusion) sur le test set jamais vu, cartes de saillance pour l'interprétabilité, simulation de l'effet de la prévalence sur la VPP, discussion de la valeur médicale et opérationnelle.

---

## Résultats clés

### Comparaison des modèles (validation, 8 epochs, CPU)

| Modèle | Paramètres entraînés | Val Accuracy | Val F1 | Temps d'entraînement |
| --- | ---: | :---: | :---: | ---: |
| **CNN from scratch** | 548 258 | **95,36 %** | **95,31 %** | 141,8 s |
| ResNet18 (transfer learning) | 1 026 | 87,45 % | 87,45 % | 181,0 s |

Le CNN from scratch bat le transfer learning ImageNet : la texture de coloration microscopique n'est pas bien représentée dans les features pré-entraînées sur des photos naturelles, un rappel utile face au réflexe "transfer learning toujours gagnant".

### Évaluation clinique (test set, 4 134 images jamais vues)

| Métrique | Valeur |
| --- | :---: |
| Accuracy | 96,13 % |
| **Sensibilité** (rappel Parasitized) | 94,68 % |
| Spécificité | 97,58 % |
| VPP | 97,51 % |
| VPN | 94,83 % |
| ROC-AUC | 99,13 % |

**Matrice de confusion :**

| | Prédit Parasitized | Prédit Uninfected |
| --- | :---: | :---: |
| **Réel Parasitized** | 1 957 (VP) | 110 (FN) |
| **Réel Uninfected** | 50 (FP) | 2 017 (VN) |

**Effet de la prévalence sur la VPP** : ce dataset est équilibré 50/50, ce qui n'est pas la prévalence réelle terrain. Une simulation bayésienne à partir de la sensibilité/spécificité mesurées montre qu'à une prévalence de 2 % (dépistage en zone peu endémique), la VPP chute à 44,4 % malgré une ROC-AUC de 99,13 %, un rappel que la performance d'un test dépend du contexte de déploiement, pas seulement du modèle.

---

## Limites

- Sous-échantillon d'entraînement (6 000 / 19 290 images disponibles) pour tenir en temps raisonnable sur CPU : un entraînement sur le train set complet et davantage d'epochs améliorerait probablement encore la marge.
- Un seul type de microscope et de coloration (dataset Chittagong Medical College Hospital) : aucune validation externe sur d'autres centres, protocoles de coloration ou populations n'a été faite.
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
