# Protocole de validation externe

Une validation externe doit utiliser un corpus provenant d'un autre centre ou
d'un autre microscope que le corpus NIH de Chittagong. Elle ne doit servir ni à
l'entraînement, ni au réglage du seuil, ni à la sélection du modèle.

## Données attendues

Produire un fichier `external_predictions.csv` avec une ligne par image et les
colonnes suivantes :

| colonne | définition |
| --- | --- |
| `patient_id` | identifiant pseudonymisé stable du patient |
| `y_true` | référence experte, 0 = Uninfected, 1 = Parasitized |
| `y_pred` | prédiction binaire au seuil fixé sur la validation interne |
| `y_prob` | probabilité de la classe Parasitized |

Les métadonnées du centre, du microscope, de la coloration, de la population,
des critères d'inclusion et du nombre d'images par patient doivent être décrites
avec le rapport. Les patients ne doivent pas chevaucher le corpus interne.

## Analyse gelée

```bash
python -m scripts.evaluate_predictions external_predictions.csv \
  --output results/external_metrics_patient_bootstrap.csv \
  --calibration-output results/external_calibration.csv \
  --n-bootstrap 2000 --seed 42
```

Le script calcule accuracy, sensibilité, spécificité, VPP, VPN, ROC-AUC et score de Brier avec
des IC à 95 % obtenus par bootstrap des patients. Toutes les images d'un patient
sont conservées ensemble à chaque tirage. Il rapporte également l'ECE à 10
intervalles et peut exporter la table de calibration avec `--calibration-output`.

## Statut

La validation externe n'a pas encore été exécutée : aucun second corpus avec
identifiants patient n'est fourni dans ce dépôt. Aucun résultat externe ne doit
être revendiqué avant l'ajout du fichier de prédictions et du rapport de cohorte.
