"""
Utilities for evaluating classification models.

This module provides helper functions used to evaluate trained
classification models on held-out test data. The returned metrics
can be used for console reporting, MLflow logging, or downstream
analysis.
"""

from __future__ import annotations

from typing import Any

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(model: Any, X_test, y_test) -> dict[str, Any]:
    """
    Evaluate a trained classification model on test data.

    The function generates predictions using the provided model
    and computes several standard evaluation metrics commonly
    used for classification tasks.

    Parameters
    ----------
    model : Any
        Trained scikit-learn compatible model implementing
        a ``predict()`` method.

    X_test :
        Feature matrix used for evaluation.

    y_test :
        True target labels corresponding to ``X_test``.

    Returns
    -------
    dict[str, Any]
        Dictionary containing evaluation results:

        accuracy : float
            Overall classification accuracy.

        classification_report_dict : dict
            Structured classification report containing
            precision, recall, F1-score, and support for
            each class.

        classification_report_text : str
            Human-readable classification report formatted
            as plain text.

        confusion_matrix : ndarray
            Matrix describing prediction counts for each
            true/predicted class combination.

        predictions : array-like
            Model predictions for the test set.

    Notes
    -----
    - The following metrics are computed:

        1. **Accuracy**
            Proportion of correctly classified samples.

        2. **Classification report**
            Detailed per-class metrics including:

            - precision
            - recall
            - F1 score
            - support

        3. **Confusion matrix**
            A tabular representation showing how many samples
            from each true class were predicted as each class.

    - Both a structured and a text-based classification report
      are returned so that the results can be used for logging,
      further analysis, and console display.

    - Returning predictions avoids recomputing them elsewhere
      in the training or evaluation pipeline.

    Example
    -------
    Evaluate a trained model.

    >>> results = evaluate_model(model, X_test, y_test)

    Inspect accuracy.

    >>> results["accuracy"]
    0.93

    View the confusion matrix.

    >>> results["confusion_matrix"]

    Access the text classification report.

    >>> print(results["classification_report_text"])
    """

    # Generate predictions from the trained model
    predictions = model.predict(X_test)

    # Compute evaluation metrics
    #   - Accuracy provides overall model performance
    accuracy = accuracy_score(y_test, predictions)

    # Create structured and text-based classification reports
    #   - The structured version is useful for MLflow logging
    #     or downstream programmatic analysis
    classification_report_dict = classification_report(
        y_test,
        predictions,
        output_dict=True,
    )

    #   - The text version is useful for console output
    #     and quick human-readable inspection
    classification_report_text = classification_report(
        y_test,
        predictions,
    )

    # Compute confusion matrix
    #   - This shows how predictions are distributed across
    #     true and predicted classes
    matrix = confusion_matrix(y_test, predictions)

    # Return metrics and predictions in a structured dictionary
    return {
        "accuracy": accuracy,
        "classification_report_dict": classification_report_dict,
        "classification_report_text": classification_report_text,
        "confusion_matrix": matrix,
        "predictions": predictions,
    }