"""Evaluate predictions from an internal or external cohort."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from malaria_evaluation import calibration_table, expected_calibration_error, patient_bootstrap_ci


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path, help="CSV with patient_id,y_true,y_pred,y_prob")
    parser.add_argument("--output", type=Path, default=Path("metrics_patient_bootstrap.csv"))
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-output", type=Path)
    args = parser.parse_args()
    predictions = pd.read_csv(args.predictions)
    result = patient_bootstrap_ci(predictions, n_bootstrap=args.n_bootstrap, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"ECE (10 bins): {expected_calibration_error(predictions.y_true, predictions.y_prob):.6f}")
    if args.calibration_output:
        args.calibration_output.parent.mkdir(parents=True, exist_ok=True)
        calibration_table(predictions.y_true, predictions.y_prob).to_csv(args.calibration_output, index=False)


if __name__ == "__main__":
    main()
