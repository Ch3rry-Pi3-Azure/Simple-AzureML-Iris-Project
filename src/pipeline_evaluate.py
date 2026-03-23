"""
Evaluation entry point for the Azure ML pipeline.

This script loads an MLflow model produced by the training component,
recreates the deterministic Iris test split, and writes evaluation
artifacts to a pipeline output folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow.pyfunc

try:
    from .data import load_data
    from .evaluate import evaluate_model
except ImportError:
    from data import load_data
    from evaluate import evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-input", required=True)
    parser.add_argument("--evaluation-output", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--data-random-state", type=int, default=5901)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    evaluation_output = Path(args.evaluation_output)
    evaluation_output.mkdir(parents=True, exist_ok=True)

    _, X_test, _, y_test = load_data(
        test_size=args.test_size,
        random_state=args.data_random_state,
    )

    model = mlflow.pyfunc.load_model(args.model_input)
    results = evaluate_model(model, X_test, y_test)

    metrics = {
        "accuracy": results["accuracy"],
        "test_size": args.test_size,
        "data_random_state": args.data_random_state,
    }

    (evaluation_output / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    (evaluation_output / "classification_report.txt").write_text(
        results["classification_report_text"],
        encoding="utf-8",
    )
    (evaluation_output / "confusion_matrix.json").write_text(
        json.dumps(results["confusion_matrix"].tolist(), indent=2),
        encoding="utf-8",
    )

    print("Pipeline evaluation step completed.")
    print(f"Evaluation output: {evaluation_output}")


if __name__ == "__main__":
    main()
