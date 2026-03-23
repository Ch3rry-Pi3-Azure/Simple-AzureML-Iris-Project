"""
Scoring script for the Iris Random Forest model deployed in Azure ML.

This module defines the inference interface required by Azure
Machine Learning managed online endpoints. The Azure ML runtime
calls:

    1. init() once when the container starts
    2. run() for each inference request

The script loads the deployed MLflow model from the Azure ML
model directory and performs predictions on incoming JSON data.

The request payload accepts either:

    1. a list of feature lists
    2. a list of objects keyed by feature name
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import mlflow.pyfunc
import pandas as pd
from azureml.ai.monitoring import Collector


# Configure module logger
#   - Azure ML surfaces log output in container logs
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Global model object
#   - Loaded once during container initialisation
#   - Reused for each inference request
model = None

# Global Azure ML collectors used for production data monitoring
inputs_collector = None
outputs_collector = None

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


def _log_collection_error(error: Exception) -> None:
    """
    Log data collection failures without breaking endpoint inference.

    Parameters
    ----------
    error : Exception
        Exception raised by the Azure ML data collector.

    Returns
    -------
    None
        The function logs the error and allows inference to continue.
    """

    logger.warning("Azure ML monitoring collection failed: %s", error)


def _resolve_model_path(model_dir: str) -> str:
    """
    Resolve the deployed MLflow model path inside the Azure ML mount.

    The repository supports two registration paths:

    1. local/manual registration from ``outputs/iris_mlflow_model``
    2. pipeline registration from the ``trained_model`` output

    Parameters
    ----------
    model_dir : str
        Root directory mounted by Azure ML via ``AZUREML_MODEL_DIR``.

    Returns
    -------
    str
        Path to the MLflow model directory containing ``MLmodel``.

    Raises
    ------
    FileNotFoundError
        If no valid MLflow model directory can be found.
    """

    candidate_paths = [
        os.path.join(model_dir, "iris_mlflow_model"),
        os.path.join(model_dir, "model_output"),
        model_dir,
    ]

    for candidate_path in candidate_paths:
        mlmodel_path = os.path.join(candidate_path, "MLmodel")
        if os.path.exists(mlmodel_path):
            return candidate_path

    raise FileNotFoundError(
        "Could not locate an MLflow model directory under AZUREML_MODEL_DIR."
    )


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

    - The deployed MLflow model may be mounted either under the
      original ``iris_mlflow_model`` folder or under the pipeline
      registration layout used by Azure ML model assets.
    """

    global model, inputs_collector, outputs_collector

    # Retrieve the model directory mounted by Azure ML
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    if not model_dir:
        raise ValueError("AZUREML_MODEL_DIR environment variable is not set.")

    # Resolve the deployed MLflow model path.
    model_path = _resolve_model_path(model_dir)

    logger.info("AZUREML_MODEL_DIR: %s", model_dir)
    logger.info("Resolved MLflow model path: %s", model_path)

    # Load the MLflow model using the generic pyfunc interface
    model = mlflow.pyfunc.load_model(model_path)

    # Instantiate Azure ML data collectors for production monitoring.
    # The names `model_inputs` and `model_outputs` let Azure ML model
    # monitoring auto-detect the collected datasets more easily.
    inputs_collector = Collector(
        name="model_inputs",
        on_error=_log_collection_error,
    )
    outputs_collector = Collector(
        name="model_outputs",
        on_error=_log_collection_error,
    )

    logger.info("Model loaded successfully")


def run(raw_data: str) -> dict[str, Any]:
    """
    Run model inference on incoming JSON request data.

    Azure ML calls this function for each prediction request.
    The request payload is expected to contain tabular feature
    data under a top-level ``data`` field.

    Parameters
    ----------
    raw_data : str
        JSON string containing the request payload.

        Accepted formats:

        {
            "data": [
                [5.1, 3.5, 1.4, 0.2],
                [6.2, 3.4, 5.4, 2.3]
            ]
        }

        or

        {
            "data": [
                {
                    "sepal length (cm)": 5.1,
                    "sepal width (cm)": 3.5,
                    "petal length (cm)": 1.4,
                    "petal width (cm)": 0.2
                }
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

        - Input rows must either be all lists or all objects.

        - List-based rows must contain exactly four feature values,
          matching the schema used during training.

        - Object-based rows must contain exactly the expected
          feature names used during training.

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

        # Accept two request shapes:
        #   1. rows as ordered feature lists
        #   2. rows as objects keyed by feature name
        if all(isinstance(row, list) for row in rows):
            if any(len(row) != len(FEATURE_COLUMNS) for row in rows):
                raise ValueError(
                    f"Each input row must contain exactly {len(FEATURE_COLUMNS)} values."
                )

            df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)

        elif all(isinstance(row, dict) for row in rows):
            missing_columns = [
                [
                    column
                    for column in FEATURE_COLUMNS
                    if column not in row
                ]
                for row in rows
            ]
            extra_columns = [
                [
                    column
                    for column in row.keys()
                    if column not in FEATURE_COLUMNS
                ]
                for row in rows
            ]

            if any(missing for missing in missing_columns):
                raise ValueError(
                    "Each object row must contain all required feature names."
                )

            if any(extra for extra in extra_columns):
                raise ValueError(
                    "Object rows may only contain the expected feature names."
                )

            # Reorder columns explicitly to keep inference schema stable.
            df = pd.DataFrame(rows)[FEATURE_COLUMNS]

        else:
            raise ValueError(
                "Each entry in 'data' must use the same format: all lists or all objects."
            )

        # Collect the model input rows for Azure ML monitoring.
        context = inputs_collector.collect(df)

        # Generate predictions
        preds = model.predict(df)

        # Convert predictions into JSON-serialisable outputs
        prediction_list = preds.tolist()
        predicted_labels = [
            CLASS_LABELS.get(int(pred), str(pred))
            for pred in prediction_list
        ]

        output_df = pd.DataFrame(
            {
                "prediction": prediction_list,
                "predicted_label": predicted_labels,
            }
        )

        # Collect the correlated model outputs for Azure ML monitoring.
        outputs_collector.collect(output_df, context)

        # Return prediction results
        return {
            "predictions": prediction_list,
            "predicted_labels": predicted_labels,
        }

    except Exception as e:
        # Log and return error information if prediction fails
        logger.exception("Inference failed")
        return {"error": str(e)}
