"""
Tests for model evaluation utilities used in the Iris classification demo.

This module verifies that the evaluation helper returns the expected
output structure after scoring a trained estimator. The test uses the
shared modelling pipeline rather than a bare Random Forest so it stays
aligned with the project's real training path.
"""

from src.core.data import load_data
from src.core.evaluate import evaluate_model
from src.core.modeling import get_base_model


def test_evaluate_model_returns_expected_keys() -> None:
    """
    Verify that the evaluation function returns the expected keys.

    The test trains a small Random Forest classifier on the Iris
    dataset, evaluates it on the test split, and confirms that
    the returned result dictionary contains the expected entries.

    Returns
    -------
    None
        The test passes if the result dictionary contains all
        expected evaluation fields.

    Notes
    -----
    - This is a lightweight functional test of the evaluation
      helper rather than a strict test of model performance.

    - The classifier uses a small number of trees to keep the
      test fast and simple.
    """

    # Load train and test data
    X_train, X_test, y_train, y_test = load_data()

    # Create and train a small test model using the same pipeline shape
    # used by the main training codepath.
    model = get_base_model(random_state=42, use_scaling=False)
    model.set_params(model__n_estimators=10)
    model.fit(X_train, y_train)

    # Evaluate the trained model
    results = evaluate_model(model, X_test, y_test)

    # Confirm that the evaluation helper returns the
    # expected metrics and prediction outputs
    assert "accuracy" in results
    assert "classification_report_dict" in results
    assert "classification_report_text" in results
    assert "confusion_matrix" in results
    assert "predictions" in results
