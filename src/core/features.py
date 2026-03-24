"""
Shared feature-engineering helpers for model training and feature-store prep.

This module centralises the small set of reusable derived Iris
features used across the project. Keeping the feature-engineering logic
in one place serves two practical goals:

1. the training pipeline can build a scikit-learn preprocessing
   workflow around stable helper functions
2. the feature-store preparation workflow can materialise the same
   derived columns into Azure-managed data assets

The current derived features are intentionally simple. They are not
intended to improve the Iris benchmark materially. Instead, they give
the repository a clearer template for future projects that need:

- reusable derived features
- model-side preprocessing pipelines
- feature-store-ready transformations that remain consistent with
  training code
"""

from __future__ import annotations

import pandas as pd

try:
    from .data import FEATURE_COLUMNS
except ImportError:
    from data import FEATURE_COLUMNS


# Reusable derived columns that can either be generated on the fly
# inside a scikit-learn pipeline or materialised into a feature store.
DERIVED_FEATURE_COLUMNS = [
    "sepal length squared",
    "petal length squared",
    "sepal area (cm^2)",
    "petal area (cm^2)",
    "sepal length x petal length",
]

MODEL_FEATURE_COLUMNS = [*FEATURE_COLUMNS, *DERIVED_FEATURE_COLUMNS]


def add_derived_feature_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Add the project's reusable derived features to a dataset frame.

    Parameters
    ----------
    dataset : pd.DataFrame
        DataFrame containing the canonical raw Iris measurement
        columns.

    Returns
    -------
    pd.DataFrame
        Copy of the input frame with the reusable derived feature
        columns added or refreshed.

    Raises
    ------
    ValueError
        Raised when one or more canonical raw feature columns are
        missing from the supplied frame.

    Notes
    -----
    - The function preserves all existing columns and only appends or
      refreshes the derived columns.

    - Derived columns are recalculated from the raw measurements each
      time so that stale values cannot silently persist when upstream
      data changes.
    """

    missing_columns = [column for column in FEATURE_COLUMNS if column not in dataset.columns]
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required raw feature columns: {missing_columns}"
        )

    feature_frame = dataset.copy()
    sepal_length = feature_frame["sepal length (cm)"]
    sepal_width = feature_frame["sepal width (cm)"]
    petal_length = feature_frame["petal length (cm)"]
    petal_width = feature_frame["petal width (cm)"]

    feature_frame["sepal length squared"] = sepal_length.pow(2)
    feature_frame["petal length squared"] = petal_length.pow(2)
    feature_frame["sepal area (cm^2)"] = sepal_length * sepal_width
    feature_frame["petal area (cm^2)"] = petal_length * petal_width
    feature_frame["sepal length x petal length"] = sepal_length * petal_length

    return feature_frame


def build_model_feature_frame(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature matrix expected by the training pipeline.

    Parameters
    ----------
    dataset : pd.DataFrame
        DataFrame containing at least the canonical raw Iris
        measurement columns.

    Returns
    -------
    pd.DataFrame
        Feature matrix containing the raw measurements followed by the
        derived reusable feature columns in a stable order.
    """

    feature_frame = add_derived_feature_columns(dataset)
    return feature_frame[MODEL_FEATURE_COLUMNS].copy()
