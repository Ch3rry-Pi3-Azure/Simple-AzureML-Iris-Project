"""
Training entry point for the Azure ML pipeline.

This script trains the Iris model and writes two pipeline outputs:

1. an MLflow model folder
2. a metrics/report folder
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

try:
    from .data import load_data
    from .evaluate import evaluate_model
except ImportError:
    from data import load_data
    from evaluate import evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--model-random-state", type=int, default=5901)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--data-random-state", type=int, default=5901)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_output = Path(args.model_output)
    metrics_output = Path(args.metrics_output)

    model_output.mkdir(parents=True, exist_ok=True)
    metrics_output.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data(
        test_size=args.test_size,
        random_state=args.data_random_state,
    )

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.model_random_state,
    )
    model.fit(X_train, y_train)

    results = evaluate_model(model, X_test, y_test)
    predictions = results["predictions"]

    metrics = {
        "accuracy": results["accuracy"],
        "precision_weighted": precision_score(
            y_test,
            predictions,
            average="weighted",
        ),
        "recall_weighted": recall_score(
            y_test,
            predictions,
            average="weighted",
        ),
        "f1_weighted": f1_score(
            y_test,
            predictions,
            average="weighted",
        ),
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "model_random_state": args.model_random_state,
        "test_size": args.test_size,
        "data_random_state": args.data_random_state,
    }

    mlflow.log_params(
        {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "model_random_state": args.model_random_state,
            "test_size": args.test_size,
            "data_random_state": args.data_random_state,
        }
    )
    mlflow.log_metrics(
        {
            "train_accuracy": metrics["accuracy"],
            "train_precision_weighted": metrics["precision_weighted"],
            "train_recall_weighted": metrics["recall_weighted"],
            "train_f1_weighted": metrics["f1_weighted"],
        }
    )

    signature = infer_signature(X_test, predictions)
    input_example = X_train.head(1)

    mlflow.sklearn.save_model(
        sk_model=model,
        path=str(model_output),
        signature=signature,
        input_example=input_example,
    )

    (metrics_output / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    (metrics_output / "classification_report.txt").write_text(
        results["classification_report_text"],
        encoding="utf-8",
    )
    (metrics_output / "confusion_matrix.json").write_text(
        json.dumps(results["confusion_matrix"].tolist(), indent=2),
        encoding="utf-8",
    )
    print("Pipeline training step completed.")
    print(f"MLflow model output: {model_output}")
    print(f"Metrics output: {metrics_output}")


if __name__ == "__main__":
    main()
