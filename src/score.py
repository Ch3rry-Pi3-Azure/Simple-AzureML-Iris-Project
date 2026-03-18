"""
Scoring script for the Iris Random Forest model deployed in Azure ML.

This module defines the inference interface required by Azure
Machine Learning managed online endpoints. The Azure ML runtime
calls:

    1. init() once when the container starts
    2. run() for each inference request

The script loads the deployed MLflow model from the Azure ML
model directory and performs predictions on incoming JSON data.
"""

from __future__ import annotations

import json
import logging
import os

import mlflow.pyfunc
import pandas as pd


# Configure module logger
#   - Azure ML surfaces log output in container logs
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Global model object
#   - Loaded once during container initialisation
#   - Reused for each inference request
model = None

# Expected feature columns used during training
#   - These names must match the schema seen by the model
FEATURE_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

# Mapping from numeric class indices to human-readable labels
CLASS_LABELS = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}


def init() -> None:
    """
    Initialise the scoring environment and load the MLflow model.

    Azure ML calls this function once when the inference container
    starts. The function locates the deployed MLflow model using
    the ``AZUREML_MODEL_DIR`` environment variable and loads it
    into memory for reuse during prediction requests.

    Returns
    -------
    None
        The loaded model is stored in the global ``model`` variable.

    Notes
    -----
    - ``AZUREML_MODEL_DIR`` is provided automatically by Azure ML
      and points to the directory where deployed model assets are
      mounted inside the inference container.

    - Loading the model during initialisation avoids repeated model
      loading on each request, which improves inference performance.

    - The deployed MLflow model is expected to be stored in a
      subdirectory called ``iris_mlflow_model``.
    """

    global model

    # Retrieve the model directory mounted by Azure ML
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    if not model_dir:
        raise ValueError("AZUREML_MODEL_DIR environment variable is not set.")

    # Construct full path to the deployed MLflow model
    #   - The subdirectory name matches the deployed model folder
    model_path = os.path.join(model_dir, "iris_mlflow_model")

    logger.info("AZUREML_MODEL_DIR: %s", model_dir)
    logger.info("Resolved MLflow model path: %s", model_path)

    # Load the MLflow model using the generic pyfunc interface
    model = mlflow.pyfunc.load_model(model_path)

    logger.info("Model loaded successfully")


def run(raw_data: str) -> dict:
    """
    Run model inference on incoming JSON request data.

    Azure ML calls this function for each prediction request.
    The request payload is expected to contain tabular feature
    data under a top-level ``data`` field.

    Parameters
    ----------
    raw_data : str
        JSON string containing the request payload.

        Expected format:

        {
            "data": [
                [5.1, 3.5, 1.4, 0.2],
                [6.2, 3.4, 5.4, 2.3]
            ]
        }

    Returns
    -------
    dict
        Dictionary containing the prediction results.

        predictions : list
            Predicted class indices as a JSON-serialisable list.

        predicted_labels : list
            Human-readable Iris class labels corresponding to the
            predicted class indices.

        If an error occurs, the dictionary contains:

        error : str
            Error message describing the failure.

    Notes
    -----
    - The function validates the request structure before
      attempting inference.

    - Input rows must contain exactly four feature values,
      matching the schema used during training.

    - The input data is converted into a pandas DataFrame with
      explicit column names so that the inference schema matches
      the training schema.

    - The MLflow ``pyfunc`` interface allows predictions to be
      generated without requiring direct access to the original
      training code.
    """

    try:
        # Ensure the model was loaded successfully during init()
        if model is None:
            raise RuntimeError("Model is not loaded.")

        # Parse the incoming JSON request
        data = json.loads(raw_data)

        # Validate top-level request structure
        if not isinstance(data, dict):
            raise ValueError("Request payload must be a JSON object.")

        if "data" not in data:
            raise ValueError("Request payload must contain a 'data' field.")

        rows = data["data"]

        # Validate data payload
        if not isinstance(rows, list):
            raise ValueError("The 'data' field must contain a list of rows.")

        if len(rows) == 0:
            raise ValueError("The 'data' field must not be empty.")

        if any(not isinstance(row, list) for row in rows):
            raise ValueError("Each entry in 'data' must be a list of feature values.")

        if any(len(row) != len(FEATURE_COLUMNS) for row in rows):
            raise ValueError(
                f"Each input row must contain exactly {len(FEATURE_COLUMNS)} values."
            )

        # Convert request data into a DataFrame with explicit column names
        #   - This keeps inference schema consistent with training
        df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)

        # Generate predictions
        preds = model.predict(df)

        # Convert predictions into JSON-serialisable outputs
        prediction_list = preds.tolist()
        predicted_labels = [
            CLASS_LABELS.get(int(pred), str(pred))
            for pred in prediction_list
        ]

        # Return prediction results
        return {
            "predictions": prediction_list,
            "predicted_labels": predicted_labels,
        }

    except Exception as e:
        # Log and return error information if prediction fails
        logger.exception("Inference failed")
        return {"error": str(e)}