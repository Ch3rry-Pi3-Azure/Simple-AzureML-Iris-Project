"""
Tests for model evaluation utilities used in the Iris classification demo.

This module verifies that the evaluation helper returns the
expected output structure after scoring a trained classifier.
"""

from sklearn.ensemble import RandomForestClassifier

from src.data import load_data
from src.evaluate import evaluate_model


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

    # Create and train a small test model
    #   - A smaller model keeps the test fast
    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
    )
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