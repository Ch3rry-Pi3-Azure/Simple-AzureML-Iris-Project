"""
Tests for data-loading utilities used in the Iris classification demo.

This module verifies that the dataset loading helper returns
non-empty train and test splits in the expected structure.
"""

from pathlib import Path

from src.core.data import load_data, load_dataset_frame


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


def test_load_data_supports_csv_input_with_canonical_schema() -> None:
    """
    Verify that CSV-backed loading preserves the serving schema.

    The test points `load_data()` at the checked-in Iris CSV and
    confirms that the feature columns are normalised to the same
    names expected by the trained model and scoring code.
    """

    data_path = Path(__file__).resolve().parents[1] / "data" / "iris.csv"

    X_train, X_test, y_train, y_test = load_data(data_path=data_path)

    assert list(X_train.columns) == [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]
    assert list(X_test.columns) == list(X_train.columns)
    assert set(y_train.unique()).issubset({0, 1, 2})
    assert set(y_test.unique()).issubset({0, 1, 2})


def test_load_dataset_frame_prefers_csv_schema_when_available() -> None:
    """
    Verify that loading the full dataset frame preserves species labels.
    """

    data_path = Path(__file__).resolve().parents[1] / "data" / "iris.csv"
    dataset = load_dataset_frame(data_path=data_path)

    assert list(dataset.columns) == [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
        "species",
    ]
    assert set(dataset["species"].unique()) == {"setosa", "versicolor", "virginica"}
