"""
Training utilities for a simple Iris classification demo.

This module trains a Random Forest classifier on the Iris dataset,
logs parameters and evaluation metrics to MLflow, and saves the
trained model locally in MLflow format for later loading,
registration, or deployment.

The workflow is intentionally kept simple and reliable by saving the
model to a local file path rather than attempting registration
directly from an MLflow run URI.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import f1_score, precision_score, recall_score

try:
    from ..core.artifact_names import (
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
    from ..core.data import load_data
    from ..core.evaluate import evaluate_model
    from ..core.modeling import run_grid_search
    from ..core.visualize import (
        save_classification_report_heatmap,
        save_confusion_matrix_plot,
        save_learning_curve_plot,
        save_oob_error_curve,
        save_multiclass_roc_curve,
    )
except ImportError:
    from core.artifact_names import (
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
    from core.data import load_data
    from core.evaluate import evaluate_model
    from core.modeling import run_grid_search
    from core.visualize import (
        save_classification_report_heatmap,
        save_confusion_matrix_plot,
        save_learning_curve_plot,
        save_oob_error_curve,
        save_multiclass_roc_curve,
    )


# Name of the MLflow experiment used to track runs
EXPERIMENT_NAME = "simple_iris_demo"

# Local directory where the trained MLflow model will be saved
LOCAL_MODEL_DIR = Path("outputs") / "iris_mlflow_model"
LOCAL_RUNS_DIR = Path("outputs") / "local_runs"


def train_model() -> None:
    """
    Train a Random Forest classifier and save it in MLflow format.

    The function performs the following steps:

        1. sets the MLflow experiment
        2. loads the Iris dataset
        3. trains a Random Forest classifier
        4. evaluates the trained model on the test set
        5. logs parameters and metrics to MLflow
        6. saves the trained model locally in MLflow format

    Returns
    -------
    None
        The function trains the model, logs metrics, saves the
        trained model locally, and prints a compact summary of
        the training run.

    Notes
    -----
    - The model is saved to a local folder rather than being registered
      directly from an MLflow run artifact URI.

    - This approach is often simpler for small demos and reduces the
      number of moving parts when validating the training workflow.

    - The saved model includes:

        - the trained scikit-learn estimator
        - an inferred model signature
        - an input example for downstream serving or inspection

    - In addition to accuracy, the function logs weighted precision,
      weighted recall, and weighted F1 score to MLflow for a richer
      view of model performance.

    Example
    -------
    Train the model from a script.

    >>> train_model()

    After execution, the saved MLflow model can be found in:

    >>> LOCAL_MODEL_DIR
    PosixPath('outputs/iris_mlflow_model')
    """

    # Set the MLflow experiment
    #   - If the experiment does not already exist, MLflow will create it
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load training and test data
    X_train, X_test, y_train, y_test = load_data()

    random_state = 5901

    # Start an MLflow run to track this training session
    with mlflow.start_run() as run:
        search = run_grid_search(
            X_train=X_train,
            y_train=y_train,
            random_state=random_state,
        )
        model = search.best_estimator_
        best_params = search.best_params_
        best_cv_score = float(search.best_score_)

        # Evaluate model performance on the held-out test data
        results = evaluate_model(model, X_test, y_test)

        # Reuse predictions returned by the evaluation function
        #   - This avoids computing model.predict() twice
        predictions = results["predictions"]
        probabilities = model.predict_proba(X_test)

        # Infer model input/output schema for MLflow
        #   - This is useful for deployment and validation
        signature = infer_signature(X_test, predictions)

        # Provide a small example input for saved model metadata
        input_example = X_train.head(1)

        # Log model configuration parameters
        mlflow.log_param("model_type", "RandomForestClassifier + GridSearchCV")
        mlflow.log_param("best_n_estimators", best_params["n_estimators"])
        mlflow.log_param("best_max_depth", best_params["max_depth"])
        mlflow.log_param("best_min_samples_split", best_params["min_samples_split"])
        mlflow.log_param("best_min_samples_leaf", best_params["min_samples_leaf"])
        mlflow.log_param("random_state", random_state)

        # Log core evaluation metrics
        mlflow.log_metric("accuracy", results["accuracy"])
        mlflow.log_metric("best_cv_score", best_cv_score)

        # Log additional weighted metrics
        #   - Weighted averaging accounts for class support
        mlflow.log_metric(
            "precision_weighted",
            precision_score(y_test, predictions, average="weighted"),
        )
        mlflow.log_metric(
            "recall_weighted",
            recall_score(y_test, predictions, average="weighted"),
        )
        mlflow.log_metric(
            "f1_weighted",
            f1_score(y_test, predictions, average="weighted"),
        )

        timestamp = datetime.now().strftime("%H%M%S")
        date_folder = datetime.now().strftime("%Y-%m-%d")
        run_output_dir = LOCAL_RUNS_DIR / date_folder / f"{timestamp}_{run.info.run_id}"
        run_output_dir.mkdir(parents=True, exist_ok=True)

        auc_scores = save_multiclass_roc_curve(
            y_true=y_test,
            y_score=probabilities,
            output_path=run_output_dir / ROC_CURVE_PNG,
        )
        save_confusion_matrix_plot(
            confusion_matrix=results["confusion_matrix"],
            output_path=run_output_dir / CONFUSION_MATRIX_PNG,
        )
        save_classification_report_heatmap(
            classification_report_dict=results["classification_report_dict"],
            output_path=run_output_dir / CLASSIFICATION_REPORT_PNG,
        )
        save_learning_curve_plot(
            estimator=model,
            X=X_train,
            y=y_train,
            output_path=run_output_dir / LEARNING_CURVE_PNG,
        )
        save_oob_error_curve(
            X=X_train,
            y=y_train,
            output_path=run_output_dir / OOB_ERROR_CURVE_PNG,
            best_params=best_params,
            random_state=random_state,
        )

        local_metrics = {
            "accuracy": results["accuracy"],
            "precision_weighted": precision_score(y_test, predictions, average="weighted"),
            "recall_weighted": recall_score(y_test, predictions, average="weighted"),
            "f1_weighted": f1_score(y_test, predictions, average="weighted"),
            "train_auc_setosa": auc_scores["auc_setosa"],
            "train_auc_versicolor": auc_scores["auc_versicolor"],
            "train_auc_virginica": auc_scores["auc_virginica"],
            "best_n_estimators": best_params["n_estimators"],
            "best_max_depth": best_params["max_depth"],
            "best_min_samples_split": best_params["min_samples_split"],
            "best_min_samples_leaf": best_params["min_samples_leaf"],
            "best_cv_score": best_cv_score,
            "random_state": random_state,
            "run_id": run.info.run_id,
        }

        (run_output_dir / METRICS_JSON).write_text(
            json.dumps(local_metrics, indent=2),
            encoding="utf-8",
        )
        (run_output_dir / BEST_PARAMS_JSON).write_text(
            json.dumps(best_params, indent=2),
            encoding="utf-8",
        )
        pd.DataFrame(search.cv_results_).to_csv(
            run_output_dir / CV_RESULTS_CSV,
            index=False,
        )
        (run_output_dir / GRID_SEARCH_SUMMARY_JSON).write_text(
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
        (run_output_dir / CLASSIFICATION_REPORT_TXT).write_text(
            results["classification_report_text"],
            encoding="utf-8",
        )
        (run_output_dir / CLASSIFICATION_REPORT_JSON).write_text(
            json.dumps(results["classification_report_dict"], indent=2),
            encoding="utf-8",
        )
        (run_output_dir / CONFUSION_MATRIX_JSON).write_text(
            json.dumps(results["confusion_matrix"].tolist(), indent=2),
            encoding="utf-8",
        )

        # Ensure the output directory exists before saving the model
        LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Save the trained model locally in MLflow format
        mlflow.sklearn.save_model(
            sk_model=model,
            path=str(LOCAL_MODEL_DIR),
            signature=signature,
            input_example=input_example,
        )

        # Print a compact training summary
        print("Run completed.")
        print(f"Run ID: {run.info.run_id}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Best CV Score: {best_cv_score:.4f}")
        print(f"Best Params: {best_params}")
        print(f"Local MLflow model saved to: {LOCAL_MODEL_DIR.resolve()}")
        print(f"Local run artifacts saved to: {run_output_dir.resolve()}")

        # Print detailed evaluation outputs
        print("\nClassification report:")
        print(results["classification_report_text"])

        print("\nConfusion matrix:")
        print(results["confusion_matrix"])


if __name__ == "__main__":
    train_model()
