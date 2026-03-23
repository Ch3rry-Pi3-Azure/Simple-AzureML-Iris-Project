"""
Evaluation entry point for the Azure ML pipeline.

This module loads the MLflow model produced by the training component,
recreates the deterministic test split used in the project, computes
evaluation metrics, and writes both structured reports and plots to
the declared Azure ML pipeline output folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score, precision_score, recall_score

try:
    from ..core.artifact_names import (
        CLASSIFICATION_REPORT_JSON,
        CLASSIFICATION_REPORT_PNG,
        CLASSIFICATION_REPORT_TXT,
        CONFUSION_MATRIX_JSON,
        CONFUSION_MATRIX_PNG,
        METRICS_JSON,
        ROC_CURVE_PNG,
    )
    from ..core.data import load_data
    from ..core.evaluate import evaluate_model
    from ..core.visualize import (
        save_classification_report_heatmap,
        save_confusion_matrix_plot,
        save_multiclass_roc_curve,
    )
except ImportError:
    from core.artifact_names import (
        CLASSIFICATION_REPORT_JSON,
        CLASSIFICATION_REPORT_PNG,
        CLASSIFICATION_REPORT_TXT,
        CONFUSION_MATRIX_JSON,
        CONFUSION_MATRIX_PNG,
        METRICS_JSON,
        ROC_CURVE_PNG,
    )
    from core.data import load_data
    from core.evaluate import evaluate_model
    from core.visualize import (
        save_classification_report_heatmap,
        save_confusion_matrix_plot,
        save_multiclass_roc_curve,
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the pipeline evaluation step.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments containing the input model path,
        output folder, and evaluation split controls.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-input", required=True)
    parser.add_argument("--evaluation-output", required=True)
    parser.add_argument("--data-input", required=False, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--data-random-state", type=int, default=5901)
    return parser.parse_args()


def main() -> None:
    """
    Run the Azure ML pipeline evaluation step.

    The function performs the following steps:

        1. parses command-line arguments
        2. recreates the deterministic evaluation split
        3. loads the trained MLflow model
        4. computes evaluation metrics and plots
        5. logs key metrics to MLflow for the Azure ML UI
        6. writes reports and images to the declared output folder

    Returns
    -------
    None
        The function writes evaluation artefacts to disk and prints
        a compact completion summary.
    """

    args = parse_args()

    evaluation_output = Path(args.evaluation_output)
    evaluation_output.mkdir(parents=True, exist_ok=True)

    # Recreate the deterministic test partition used across the project.
    _, X_test, _, y_test = load_data(
        data_path=args.data_input,
        test_size=args.test_size,
        random_state=args.data_random_state,
    )

    # Load the pipeline-produced MLflow model and generate predictions.
    model = mlflow.sklearn.load_model(args.model_input)
    results = evaluate_model(model, X_test, y_test)
    predictions = results["predictions"]
    probabilities = model.predict_proba(X_test)

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
        "test_size": args.test_size,
        "data_random_state": args.data_random_state,
    }

    # Log parameters and summary metrics so they appear in Azure ML.
    mlflow.log_params(
        {
            "evaluation_test_size": args.test_size,
            "evaluation_data_random_state": args.data_random_state,
            "data_input": args.data_input or "builtin_or_default",
        }
    )
    mlflow.log_metrics(
        {
            "eval_accuracy": metrics["accuracy"],
            "eval_precision_weighted": metrics["precision_weighted"],
            "eval_recall_weighted": metrics["recall_weighted"],
            "eval_f1_weighted": metrics["f1_weighted"],
        }
    )

    # Produce a one-vs-rest ROC chart and per-class AUC metrics.
    auc_scores = save_multiclass_roc_curve(
        y_true=y_test,
        y_score=probabilities,
        output_path=evaluation_output / ROC_CURVE_PNG,
    )
    mlflow.log_metrics(
        {
            "eval_auc_setosa": auc_scores["auc_setosa"],
            "eval_auc_versicolor": auc_scores["auc_versicolor"],
            "eval_auc_virginica": auc_scores["auc_virginica"],
        }
    )
    metrics.update(
        {
            "eval_auc_setosa": auc_scores["auc_setosa"],
            "eval_auc_versicolor": auc_scores["auc_versicolor"],
            "eval_auc_virginica": auc_scores["auc_virginica"],
        }
    )

    # Generate plot artefacts alongside the structured reports.
    save_confusion_matrix_plot(
        confusion_matrix=results["confusion_matrix"],
        output_path=evaluation_output / CONFUSION_MATRIX_PNG,
    )
    save_classification_report_heatmap(
        classification_report_dict=results["classification_report_dict"],
        output_path=evaluation_output / CLASSIFICATION_REPORT_PNG,
    )

    # Persist structured outputs for downstream inspection or download.
    (evaluation_output / METRICS_JSON).write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    (evaluation_output / CLASSIFICATION_REPORT_TXT).write_text(
        results["classification_report_text"],
        encoding="utf-8",
    )
    (evaluation_output / CLASSIFICATION_REPORT_JSON).write_text(
        json.dumps(results["classification_report_dict"], indent=2),
        encoding="utf-8",
    )
    (evaluation_output / CONFUSION_MATRIX_JSON).write_text(
        json.dumps(results["confusion_matrix"].tolist(), indent=2),
        encoding="utf-8",
    )

    print("Pipeline evaluation step completed.")
    print(f"Evaluation output: {evaluation_output}")


if __name__ == "__main__":
    main()
