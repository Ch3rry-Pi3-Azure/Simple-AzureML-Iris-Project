"""
Tests for reusable feature-engineering and preprocessing helpers.
"""

import pandas as pd

from src.core.features import (
    MODEL_FEATURE_COLUMNS,
    add_derived_feature_columns,
    build_model_feature_frame,
)
from src.core.preprocessing import build_preprocessing_pipeline


def test_add_derived_feature_columns_creates_expected_values() -> None:
    """
    Verify that the shared derived-feature helper materialises the expected columns.
    """

    dataset = pd.DataFrame(
        {
            "sepal length (cm)": [5.1],
            "sepal width (cm)": [3.5],
            "petal length (cm)": [1.4],
            "petal width (cm)": [0.2],
        }
    )

    feature_frame = add_derived_feature_columns(dataset)

    assert feature_frame.loc[0, "sepal length squared"] == 5.1**2
    assert feature_frame.loc[0, "petal length squared"] == 1.4**2
    assert feature_frame.loc[0, "sepal area (cm^2)"] == 5.1 * 3.5
    assert feature_frame.loc[0, "petal area (cm^2)"] == 1.4 * 0.2
    assert feature_frame.loc[0, "sepal length x petal length"] == 5.1 * 1.4


def test_build_model_feature_frame_returns_stable_column_order() -> None:
    """
    Verify that the model feature frame keeps a stable raw-plus-derived schema.
    """

    dataset = pd.DataFrame(
        {
            "sepal length (cm)": [5.1],
            "sepal width (cm)": [3.5],
            "petal length (cm)": [1.4],
            "petal width (cm)": [0.2],
        }
    )

    feature_frame = build_model_feature_frame(dataset)

    assert list(feature_frame.columns) == MODEL_FEATURE_COLUMNS


def test_build_preprocessing_pipeline_generates_full_feature_matrix() -> None:
    """
    Verify that the preprocessing pipeline expands the raw feature frame.
    """

    dataset = pd.DataFrame(
        {
            "sepal length (cm)": [5.1, 4.9],
            "sepal width (cm)": [3.5, 3.0],
            "petal length (cm)": [1.4, 1.4],
            "petal width (cm)": [0.2, 0.2],
        }
    )

    pipeline = build_preprocessing_pipeline(use_scaling=False)
    transformed = pipeline.fit_transform(dataset)

    assert transformed.shape == (2, len(MODEL_FEATURE_COLUMNS))
