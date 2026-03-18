"""
Tests for data-loading utilities used in the Iris classification demo.

This module verifies that the dataset loading helper returns
non-empty train and test splits in the expected structure.
"""

from src.data import load_data


def test_load_data_returns_non_empty_splits() -> None:
    """
    Verify that the data-loading function returns non-empty splits.

    The test checks that the train and test feature matrices and
    target vectors all contain at least one observation.

    Returns
    -------
    None
        The test passes if all returned dataset partitions are non-empty.

    Notes
    -----
    - This is a lightweight smoke test intended to confirm that the
      data-loading pipeline is functioning correctly.

    - The test does not validate exact dataset sizes, only that the
      returned outputs are populated.
    """

    # Load training and test data
    X_train, X_test, y_train, y_test = load_data()

    # Check that all returned partitions contain data
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0