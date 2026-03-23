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


# Name of the MLflow experiment used to track runs
EXPERIMENT_NAME = "simple_iris_demo"

# Local directory where the trained MLflow model will be saved
LOCAL_MODEL_DIR = Path("outputs") / "iris_mlflow_model"


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

    # Define model hyperparameters
    #   - These are kept explicit to make logging and later adjustment easier
    n_estimators = 100
    max_depth = 4
    random_state = 42

    # Start an MLflow run to track this training session
    with mlflow.start_run() as run:

        # Initialise the Random Forest classifier
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )

        # Fit the model on the training data
        model.fit(X_train, y_train)

        # Evaluate model performance on the held-out test data
        results = evaluate_model(model, X_test, y_test)

        # Reuse predictions returned by the evaluation function
        #   - This avoids computing model.predict() twice
        predictions = results["predictions"]

        # Infer model input/output schema for MLflow
        #   - This is useful for deployment and validation
        signature = infer_signature(X_test, predictions)

        # Provide a small example input for saved model metadata
        input_example = X_train.head(1)

        # Log model configuration parameters
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)

        # Log core evaluation metrics
        mlflow.log_metric("accuracy", results["accuracy"])

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
        print(f"Local MLflow model saved to: {LOCAL_MODEL_DIR.resolve()}")

        # Print detailed evaluation outputs
        print("\nClassification report:")
        print(results["classification_report_text"])

        print("\nConfusion matrix:")
        print(results["confusion_matrix"])


if __name__ == "__main__":
    train_model()
