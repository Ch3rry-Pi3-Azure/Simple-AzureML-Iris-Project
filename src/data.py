"""
Utilities for loading and splitting the Iris dataset.

This module provides a small helper function used in simple
classification demonstrations. It can load the dataset either
from a CSV file or from scikit-learn's built-in Iris dataset and
then prepares a reproducible train-test split.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_DATA_PATH = PROJECT_ROOT / "data" / "iris.csv"

FEATURE_RENAME_MAP = {
    "sepal_length": "sepal length (cm)",
    "sepal_width": "sepal width (cm)",
    "petal_length": "petal length (cm)",
    "petal_width": "petal width (cm)",
}

FEATURE_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

TARGET_RENAME_MAP = {
    "setosa": 0,
    "versicolor": 1,
    "virginica": 2,
}


def _load_csv_frame(data_path: Path) -> pd.DataFrame:
    """
    Load the full Iris dataset from a CSV file.

    Parameters
    ----------
    data_path : Path
        Filesystem path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame normalised to match the canonical project schema.
    """

    dataset = pd.read_csv(data_path)
    dataset = dataset.rename(columns=FEATURE_RENAME_MAP)

    missing_feature_columns = [
        column for column in FEATURE_COLUMNS if column not in dataset.columns
    ]
    if missing_feature_columns:
        raise ValueError(
            f"CSV data is missing required feature columns: {missing_feature_columns}"
        )

    if "species" not in dataset.columns:
        raise ValueError("CSV data must contain a 'species' column.")

    return dataset[FEATURE_COLUMNS + ["species"]].copy()


def _load_builtin_frame() -> pd.DataFrame:
    """
    Load the full Iris dataset from scikit-learn.

    Returns
    -------
    pd.DataFrame
        DataFrame with canonical feature columns and string species
        labels.
    """

    iris = load_iris(as_frame=True)
    dataset = iris.data.copy()
    dataset["species"] = iris.target.map(lambda value: iris.target_names[int(value)])
    return dataset


def load_dataset_frame(data_path: str | Path | None = None) -> pd.DataFrame:
    """
    Load the full Iris dataset from CSV when available, otherwise use
    the built-in scikit-learn dataset.

    Parameters
    ----------
    data_path : str | Path | None
        Optional path to an Iris CSV file.

    Returns
    -------
    pd.DataFrame
        Canonical project dataset containing the feature columns plus
        the `species` label column.
    """

    resolved_data_path = Path(data_path) if data_path is not None else DEFAULT_LOCAL_DATA_PATH

    if resolved_data_path.exists():
        return _load_csv_frame(resolved_data_path)

    return _load_builtin_frame()


def load_data(
    data_path: str | Path | None = None,
    test_size: float = 0.2,
    random_state: int = 5901,
) -> tuple:
    """
    Load the Iris dataset and produce a stratified train-test split.

    The function first tries to load a CSV-backed dataset when a
    path is provided or when the repository's default local CSV is
    available. If no usable CSV path is found, it falls back to the
    scikit-learn built-in Iris dataset. The selected dataset is then
    split into training and testing subsets using a stratified
    sampling strategy so that class proportions are preserved in
    both partitions.

    Parameters
    ----------
    data_path : str | Path | None
        Optional path to an Iris CSV file.

        If supplied and the file exists, the CSV is used. If omitted,
        the function tries the repository-local default path
        ``data/iris.csv`` before falling back to scikit-learn.

    test_size : float
        Fraction of the dataset reserved for the test set.

        For example, a value of ``0.2`` means that 20% of the
        observations will be placed in the test partition while
        the remaining 80% are used for training.

    random_state : int
        Random seed used by the train-test split.

        Providing a fixed seed ensures reproducibility so that
        repeated runs produce identical dataset partitions.

    Returns
    -------
    tuple
        Four objects returned in the following order:

        X_train : pandas.DataFrame
            Feature matrix used to train the model.

        X_test : pandas.DataFrame
            Feature matrix used for evaluation.

        y_train : pandas.Series
            Target labels corresponding to ``X_train``.

        y_test : pandas.Series
            Target labels corresponding to ``X_test``.

    Notes
    -----
    - The Iris dataset contains **150 observations** of iris flowers
      across three species:

        1. *Setosa*
        2. *Versicolor*
        3. *Virginica*

    - Each observation includes four numerical features:

        - sepal length
        - sepal width
        - petal length
        - petal width

    - Stratified splitting is used to ensure that the class
      distribution remains consistent between training and test sets.

    Example
    -------
    Load the dataset and inspect the resulting shapes.

    >>> X_train, X_test, y_train, y_test = load_data()

    >>> X_train.shape
    (120, 4)

    >>> X_test.shape
    (30, 4)
    """

    dataset = load_dataset_frame(data_path=data_path)
    X = dataset[FEATURE_COLUMNS].copy()
    y = dataset["species"].map(TARGET_RENAME_MAP)

    if y.isna().any():
        unknown_labels = sorted(dataset.loc[y.isna(), "species"].astype(str).unique())
        raise ValueError(
            f"Dataset contains unsupported species labels: {unknown_labels}"
        )
    y = y.astype(int)

    # Perform a train-test split.
    #   - Stratification preserves class balance.
    #   - Random state ensures reproducibility.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Return partitioned datasets.
    return X_train, X_test, y_train, y_test
