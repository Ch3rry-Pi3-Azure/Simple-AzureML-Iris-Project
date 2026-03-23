"""
Model registration utilities for the Iris classification demo.

This module registers a locally saved MLflow model in the Azure ML
MLflow model registry. The model must already exist in MLflow format
within the project outputs directory.

The script is intended as a simple example showing how a trained
model artefact can be promoted into the Azure ML registry for
versioning and deployment.
"""

from __future__ import annotations

from pathlib import Path
import mlflow


# Name used when registering the model in Azure ML
REGISTERED_MODEL_NAME = "simple_iris_rf_model"

# Local directory containing the saved MLflow model
LOCAL_MODEL_DIR = Path("outputs") / "iris_mlflow_model"


def register_local_model() -> None:
    """
    Register a locally saved MLflow model in Azure ML.

    The function performs the following steps:

        1. resolves the local filesystem path of the saved MLflow model
        2. verifies that the model directory exists
        3. constructs a file-based MLflow model URI
        4. registers the model in the Azure ML MLflow registry
        5. prints the resulting registered model metadata

    Returns
    -------
    None
        The function does not return a value. Its effects are to
        submit a model registration request and print the resulting
        registration information.

    Notes
    -----
    - The model must already exist in MLflow format at the specified
      local directory.

    - Registration creates a new **model version** within the Azure ML
      MLflow registry under the specified registered model name.

    - Subsequent registrations with the same name will create
      additional model versions.

    - This workflow is commonly used in simple ML pipelines where
      models are first trained and saved locally before being promoted
      to a registry for deployment.

    Example
    -------
    Register the locally saved model.

    >>> register_local_model()

    The script will print:

    - the model URI
    - the registered model name
    - the registered model version
    """

    # Resolve the absolute path of the saved MLflow model directory
    model_local_path = LOCAL_MODEL_DIR.resolve()

    # Verify that the model directory exists before attempting registration
    if not model_local_path.exists():
        raise ValueError(
            f"Local model path does not exist: {model_local_path}"
        )

    # Construct an MLflow file-based model URI
    #   - MLflow accepts URIs to locate models for registration
    model_uri = f"file://{model_local_path}"

    # Register the model in the Azure ML MLflow registry
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    # Print registration information
    print("Model registration requested.")
    print(f"Model URI: {model_uri}")
    print(f"Registered model name: {registered_model.name}")
    print(f"Registered model version: {registered_model.version}")


if __name__ == "__main__":
    register_local_model()