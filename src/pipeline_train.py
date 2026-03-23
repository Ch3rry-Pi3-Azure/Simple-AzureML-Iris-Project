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
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import f1_score, precision_score, recall_score

try:
    from .artifact_names import (
        BEST_PARAMS_JSON,
        CLASSIFICATION_REPORT_JSON,
        CLASSIFICATION_REPORT_PNG,
        CLASSIFICATION_REPORT_TXT,
        CONFUSION_MATRIX_JSON,
        CONFUSION_MATRIX_PNG,
        CV_RESULTS_CSV,
        GRID_SEARCH_SUMMARY_JSON,
        LEARNING_CURVE_PNG,
        METRICS_JSON,
        OOB_ERROR_CURVE_PNG,
        ROC_CURVE_PNG,
    )
    from .data import load_data
    from .evaluate import evaluate_model
    from .modeling import run_grid_search
    from .visualize import (
        save_classification_report_heatmap,
        save_confusion_matrix_plot,
        save_learning_curve_plot,
        save_oob_error_curve,
        save_multiclass_roc_curve,
    )
except ImportError:
    from artifact_names import (
        BEST_PARAMS_JSON,
        CLASSIFICATION_REPORT_JSON,
        CLASSIFICATION_REPORT_PNG,
        CLASSIFICATION_REPORT_TXT,
        CONFUSION_MATRIX_JSON,
        CONFUSION_MATRIX_PNG,
        CV_RESULTS_CSV,
        GRID_SEARCH_SUMMARY_JSON,
        LEARNING_CURVE_PNG,
        METRICS_JSON,
        OOB_ERROR_CURVE_PNG,
        ROC_CURVE_PNG,
    )
    from data import load_data
    from evaluate import evaluate_model
    from modeling import run_grid_search
    from visualize import (
        save_classification_report_heatmap,
        save_confusion_matrix_plot,
        save_learning_curve_plot,
        save_oob_error_curve,
        save_multiclass_roc_curve,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--metrics-output", required=True)
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

    search = run_grid_search(
        X_train=X_train,
        y_train=y_train,
        random_state=args.model_random_state,
    )
    model = search.best_estimator_

    results = evaluate_model(model, X_test, y_test)
    predictions = results["predictions"]
    probabilities = model.predict_proba(X_test)
    best_params = search.best_params_
    best_cv_score = float(search.best_score_)

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
        "best_n_estimators": best_params["n_estimators"],
        "best_max_depth": best_params["max_depth"],
        "best_min_samples_split": best_params["min_samples_split"],
        "best_min_samples_leaf": best_params["min_samples_leaf"],
        "best_cv_score": best_cv_score,
        "model_random_state": args.model_random_state,
        "test_size": args.test_size,
        "data_random_state": args.data_random_state,
    }

    mlflow.log_params(
        {
            "search_strategy": "GridSearchCV",
            "best_n_estimators": best_params["n_estimators"],
            "best_max_depth": best_params["max_depth"],
            "best_min_samples_split": best_params["min_samples_split"],
            "best_min_samples_leaf": best_params["min_samples_leaf"],
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
            "train_best_cv_score": best_cv_score,
        }
    )

    auc_scores = save_multiclass_roc_curve(
        y_true=y_test,
        y_score=probabilities,
        output_path=metrics_output / ROC_CURVE_PNG,
    )
    train_auc_metrics = {
        "train_auc_setosa": auc_scores["auc_setosa"],
        "train_auc_versicolor": auc_scores["auc_versicolor"],
        "train_auc_virginica": auc_scores["auc_virginica"],
    }
    mlflow.log_metrics(train_auc_metrics)
    metrics.update(train_auc_metrics)

    save_confusion_matrix_plot(
        confusion_matrix=results["confusion_matrix"],
        output_path=metrics_output / CONFUSION_MATRIX_PNG,
    )
    save_classification_report_heatmap(
        classification_report_dict=results["classification_report_dict"],
        output_path=metrics_output / CLASSIFICATION_REPORT_PNG,
    )
    save_learning_curve_plot(
        estimator=model,
        X=X_train,
        y=y_train,
        output_path=metrics_output / LEARNING_CURVE_PNG,
    )
    save_oob_error_curve(
        X=X_train,
        y=y_train,
        output_path=metrics_output / OOB_ERROR_CURVE_PNG,
        best_params=best_params,
        random_state=args.model_random_state,
    )

    signature = infer_signature(X_test, predictions)
    input_example = X_train.head(1)

    mlflow.sklearn.save_model(
        sk_model=model,
        path=str(model_output),
        signature=signature,
        input_example=input_example,
    )

    (metrics_output / METRICS_JSON).write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    (metrics_output / BEST_PARAMS_JSON).write_text(
        json.dumps(best_params, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(search.cv_results_).to_csv(
        metrics_output / CV_RESULTS_CSV,
        index=False,
    )
    (metrics_output / GRID_SEARCH_SUMMARY_JSON).write_text(
        json.dumps(
            {
                "best_params": best_params,
                "best_cv_score": best_cv_score,
                "scoring": "accuracy",
                "cv_folds": 5,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (metrics_output / CLASSIFICATION_REPORT_TXT).write_text(
        results["classification_report_text"],
        encoding="utf-8",
    )
    (metrics_output / CLASSIFICATION_REPORT_JSON).write_text(
        json.dumps(results["classification_report_dict"], indent=2),
        encoding="utf-8",
    )
    (metrics_output / CONFUSION_MATRIX_JSON).write_text(
        json.dumps(results["confusion_matrix"].tolist(), indent=2),
        encoding="utf-8",
    )
    print("Pipeline training step completed.")
    print(f"MLflow model output: {model_output}")
    print(f"Metrics output: {metrics_output}")


if __name__ == "__main__":
    main()
