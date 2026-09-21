"""Patient-level reporting utilities for internal and external evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score


METRICS = ("accuracy", "sensitivity", "specificity", "ppv", "npv", "roc_auc", "brier_score")


def classification_metrics(y_true, y_pred, y_prob) -> dict[str, float]:
    """Return binary clinical metrics (positive label: 1)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator else np.nan

    return {
        "accuracy": ratio(tp + tn, tp + tn + fp + fn),
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "ppv": ratio(tp, tp + fp),
        "npv": ratio(tn, tn + fn),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if np.unique(y_true).size == 2 else np.nan,
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


def calibration_table(y_true, y_prob, *, n_bins: int = 10) -> pd.DataFrame:
    """Return an equal-width reliability table and bin contributions to ECE."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape or y_true.ndim != 1:
        raise ValueError("y_true and y_prob must be one-dimensional arrays of equal length")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    if np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("probabilities must be between 0 and 1")

    bin_id = np.minimum((y_prob * n_bins).astype(int), n_bins - 1)
    rows = []
    for current in range(n_bins):
        mask = bin_id == current
        count = int(mask.sum())
        mean_probability = float(y_prob[mask].mean()) if count else np.nan
        observed_rate = float(y_true[mask].mean()) if count else np.nan
        rows.append({
            "bin": current + 1,
            "lower": current / n_bins,
            "upper": (current + 1) / n_bins,
            "count": count,
            "mean_probability": mean_probability,
            "observed_rate": observed_rate,
            "ece_contribution": count / len(y_true) * abs(mean_probability - observed_rate) if count else 0.0,
        })
    return pd.DataFrame(rows)


def expected_calibration_error(y_true, y_prob, *, n_bins: int = 10) -> float:
    """Compute ECE from an equal-width reliability table (lower is better)."""
    return float(calibration_table(y_true, y_prob, n_bins=n_bins).ece_contribution.sum())


def patient_bootstrap_ci(
    predictions: pd.DataFrame,
    *,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap patients, retaining every image belonging to sampled patients.

    Each sampled occurrence contributes all images from that patient. This
    preserves the patient as the independent sampling unit.
    """
    required = {"patient_id", "y_true", "y_pred", "y_prob"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    patients = predictions["patient_id"].drop_duplicates().to_numpy()
    if len(patients) < 2:
        raise ValueError("At least two patients are required for a patient bootstrap")

    rng = np.random.default_rng(seed)
    draws: list[dict[str, float]] = []
    grouped = {patient: frame for patient, frame in predictions.groupby("patient_id", sort=False)}
    for _ in range(n_bootstrap):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        frame = pd.concat([grouped[patient] for patient in sampled], ignore_index=True)
        draws.append(classification_metrics(frame.y_true, frame.y_pred, frame.y_prob))

    point = classification_metrics(predictions.y_true, predictions.y_pred, predictions.y_prob)
    samples = pd.DataFrame(draws)
    alpha = (1 - confidence) / 2
    return pd.DataFrame(
        {
            "metric": METRICS,
            "estimate": [point[name] for name in METRICS],
            "ci_low": [samples[name].quantile(alpha) for name in METRICS],
            "ci_high": [samples[name].quantile(1 - alpha) for name in METRICS],
            "confidence": confidence,
            "bootstrap_unit": "patient",
            "n_patients": len(patients),
            "n_bootstrap": n_bootstrap,
            "seed": seed,
        }
    )


def patient_class_matrix(split_manifest: pd.DataFrame) -> pd.DataFrame:
    """Count unique patients and images for each split/class combination."""
    required = {"split", "patient_id", "label"}
    missing = required.difference(split_manifest.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return (
        split_manifest.groupby(["split", "label"], observed=True)
        .agg(n_patients=("patient_id", "nunique"), n_images=("patient_id", "size"))
        .reset_index()
        .sort_values(["split", "label"])
        .reset_index(drop=True)
    )


def aggregate_patient_predictions(
    predictions: pd.DataFrame, *, threshold: float, min_suspicious_cells: int
) -> pd.DataFrame:
    """Aggregate cell probabilities into one decision per patient."""
    frame = predictions.copy()
    frame["suspicious"] = frame["y_prob"] >= threshold
    patients = (
        frame.groupby("patient_id", sort=False)
        .agg(y_true=("y_true", "max"), suspicious_cells=("suspicious", "sum"), y_prob=("y_prob", "max"))
        .reset_index()
    )
    patients["y_pred"] = (patients["suspicious_cells"] >= min_suspicious_cells).astype(int)
    return patients


def select_patient_count_rule(
    predictions: pd.DataFrame, *, threshold: float, target_sensitivity: float = 0.98
) -> dict[str, float | int]:
    """Select the least strict cell-count rule with the best validation specificity."""
    max_count = int(predictions.groupby("patient_id").size().max())
    feasible = []
    for minimum in range(1, max_count + 1):
        patients = aggregate_patient_predictions(
            predictions, threshold=threshold, min_suspicious_cells=minimum
        )
        metrics = classification_metrics(patients.y_true, patients.y_pred, patients.y_prob)
        if metrics["sensitivity"] >= target_sensitivity:
            feasible.append((metrics["specificity"], minimum, metrics["sensitivity"]))
    if not feasible:
        raise ValueError("No patient aggregation rule satisfies the requested sensitivity")
    # Parmi les regles a specificite egale, garder la moins stricte : la plus
    # stricte se situe au bord du plateau et generalise moins bien.
    specificity, minimum, sensitivity = max(feasible, key=lambda item: (item[0], -item[1]))
    return {
        "min_suspicious_cells": int(minimum),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }
