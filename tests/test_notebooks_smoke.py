"""Fast checks for the notebook-based malaria pipeline.

The tests use synthetic tensors and need neither the NIH dataset nor training.
"""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat
import pandas as pd
import pytest
import torch
import torch.nn as nn

from malaria_evaluation import (
    aggregate_patient_predictions,
    select_patient_count_rule,
    calibration_table,
    expected_calibration_error,
    patient_bootstrap_ci,
    patient_class_matrix,
)
from src.malaria_pipeline import SimpleCNN, ppv_npv_at_prevalence
from src.malaria_pipeline import keep_frozen_batchnorm_in_eval, select_threshold_for_sensitivity


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "01_data_ingestion_eda.ipynb",
    ROOT / "02_preprocessing_annotated_dataset.ipynb",
    ROOT / "03_baseline_cnn_vs_transfer_learning.ipynb",
    ROOT / "04_clinical_evaluation_interpretability.ipynb",
]


def _read_notebook(path: Path):
    return nbformat.read(path, as_version=4)


def _code(path: Path) -> str:
    return "\n\n".join(
        cell.source for cell in _read_notebook(path).cells if cell.cell_type == "code"
    )


def _definition(path: Path, name: str):
    """Compile one top-level definition without running notebook setup."""
    tree = ast.parse(_code(path), filename=str(path))
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.ClassDef)) and item.name == name
    )
    namespace = {"torch": torch, "nn": nn}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_is_valid_and_code_compiles(path: Path):
    notebook = _read_notebook(path)
    nbformat.validate(notebook)
    compile(_code(path), str(path), "exec")


def test_pipeline_artifact_contract_is_complete():
    ingestion, preprocessing, training, evaluation = map(_code, NOTEBOOKS)
    assert '"inventory.csv"' in ingestion
    assert '[("train", train_df), ("val", val_df), ("test", test_df)]' in preprocessing
    assert 'f"{name}.csv"' in preprocessing
    assert '"train.csv"' in training and '"val.csv"' in training
    assert '"best_model.pt"' in training
    assert '"test.csv"' in evaluation and '"best_model.pt"' in evaluation


def test_patient_grouping_guardrails_are_present():
    preprocessing = _code(NOTEBOOKS[1])
    assert "GroupShuffleSplit" in preprocessing
    assert 'groups=clean["patient_id"]' in preprocessing
    assert 'groups=temp_df["patient_id"]' in preprocessing
    assert preprocessing.count("assert not (set(") >= 3
    assert '"split_manifest.csv"' in preprocessing
    assert '"patient_class_matrix.csv"' in preprocessing
    assert '.assign(seed=SEED)' in preprocessing


def test_simple_cnn_accepts_a_synthetic_batch():
    model = SimpleCNN()
    output = model(torch.zeros(2, 3, 64, 64))
    assert output.shape == (2, 2)
    assert torch.isfinite(output).all()


def test_training_uses_the_complete_manifest():
    training = _code(NOTEBOOKS[2])
    assert "train_df = train_full.reset_index(drop=True)" in training
    assert "train_df.groupby" not in training


def test_frozen_batchnorm_stays_in_evaluation_mode():
    model = nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(3, 2, 1))
    for parameter in model[0].parameters():
        parameter.requires_grad = False
    model.train()
    keep_frozen_batchnorm_in_eval(model)
    assert model[0].training is False
    assert model[1].training is True


def test_threshold_and_patient_aggregation_are_explicit():
    selected = select_threshold_for_sensitivity(
        [0, 0, 1, 1], [0.1, 0.4, 0.7, 0.9], target_sensitivity=1.0
    )
    assert selected["threshold"] == pytest.approx(0.7)
    cells = pd.DataFrame(
        {"patient_id": ["p1", "p1", "p2"], "y_true": [1, 1, 0], "y_prob": [0.8, 0.2, 0.1]}
    )
    patients = aggregate_patient_predictions(cells, threshold=0.7, min_suspicious_cells=1)
    assert patients.set_index("patient_id").loc["p1", "y_pred"] == 1


def test_patient_rule_prefers_least_strict_among_ties():
    cells = pd.DataFrame(
        {
            "patient_id": ["p1"] * 6 + ["n1"],
            "y_true": [1] * 6 + [0],
            "y_prob": [0.9] * 6 + [0.1],
        }
    )
    rule = select_patient_count_rule(cells, threshold=0.5, target_sensitivity=1.0)
    assert rule["min_suspicious_cells"] == 1

def test_prevalence_metrics_smoke_case():
    ppv, npv = ppv_npv_at_prevalence(sensitivity=0.95, specificity=0.97, prevalence=0.02)
    assert ppv == pytest.approx(0.3926, abs=1e-4)
    assert npv == pytest.approx(0.9990, abs=1e-4)


def test_patient_bootstrap_is_reproducible_and_reports_patient_unit():
    predictions = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"],
            "y_true": [1, 1, 0, 0, 1, 1, 0, 0],
            "y_pred": [1, 0, 0, 0, 1, 1, 1, 0],
            "y_prob": [0.9, 0.4, 0.1, 0.2, 0.8, 0.7, 0.6, 0.3],
        }
    )
    first = patient_bootstrap_ci(predictions, n_bootstrap=100, seed=7)
    second = patient_bootstrap_ci(predictions, n_bootstrap=100, seed=7)
    assert first.equals(second)
    assert set(first.bootstrap_unit) == {"patient"}
    assert set(first.n_patients) == {4}
    assert (first.ci_low <= first.ci_high).all()


def test_patient_class_matrix_counts_patients_and_images():
    manifest = pd.DataFrame(
        {
            "split": ["train", "train", "train", "test"],
            "patient_id": ["p1", "p1", "p2", "p3"],
            "label": ["positive", "positive", "negative", "positive"],
        }
    )
    matrix = patient_class_matrix(manifest)
    positive_train = matrix.query("split == 'train' and label == 'positive'").iloc[0]
    assert positive_train.n_patients == 1
    assert positive_train.n_images == 2


def test_calibration_metrics_are_bounded_and_perfect_case_is_zero():
    y_true = [0, 0, 1, 1]
    perfect_probability = [0.0, 0.0, 1.0, 1.0]
    table = calibration_table(y_true, perfect_probability, n_bins=5)
    assert table["count"].sum() == 4
    assert expected_calibration_error(y_true, perfect_probability, n_bins=5) == 0.0


def test_readme_key_metrics_match_versioned_results():
    import json

    results = json.loads((ROOT / "results" / "key_results.json").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for displayed_value in results["readme_values"]:
        assert displayed_value in readme
