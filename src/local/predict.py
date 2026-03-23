"""
Prediction utilities for the Iris classification demo.

This module demonstrates how to load a trained MLflow model
from disk and perform a simple prediction using a manually
constructed sample input.

It provides a minimal example of how a saved MLflow model
can be loaded and used for inference.
"""

from pathlib import Path
import pandas as pd
import mlflow.pyfunc


# Determine the root directory of the project
#   - This allows the script to locate the saved model
#     regardless of where it is executed from
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Path to the locally saved MLflow model
MODEL_DIR = PROJECT_ROOT / "outputs" / "iris_mlflow_model"


def predict_example() -> None:
    """
    Load the saved MLflow model and perform a prediction example.

    The function performs the following steps:

        1. verifies that the saved MLflow model directory exists
        2. loads the model using MLflow's pyfunc interface
        3. constructs a sample Iris observation
        4. runs inference using the loaded model
        5. prints the input sample and prediction result

    Returns
    -------
    None
        The function prints prediction results to the console.

    Notes
    -----
    - The model is loaded using ``mlflow.pyfunc.load_model`` which
      provides a generic inference interface for MLflow models.

    - The example input corresponds to the four features in the
      Iris dataset:

        - sepal length (cm)
        - sepal width (cm)
        - petal length (cm)
        - petal width (cm)

    - The prediction returned corresponds to the predicted class
      label of the Iris species.

    Example
    -------
    Run the prediction example.

    >>> predict_example()

    The script will print:

    - the model directory
    - the input sample
    - the predicted class label
    """

    # Verify that the saved MLflow model directory exists
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    # Load the MLflow model using the generic pyfunc interface
    #   - This allows inference without needing the original
    #     training code or model class
    model = mlflow.pyfunc.load_model(str(MODEL_DIR))

    # Construct a sample observation using Iris feature names
    #   - The column names must match the schema expected by the model
    sample = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ],
    )

    # Run inference using the loaded model
    prediction = model.predict(sample)

    # Display model location
    print("Model directory:")
    print(MODEL_DIR)

    # Display input data
    print("\nInput sample:")
    print(sample)

    # Display prediction result
    print("\nPrediction:")
    print(prediction)


if __name__ == "__main__":
    predict_example()
