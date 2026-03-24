"""
Tests for feature-store preparation helpers.
"""

import pandas as pd

from src.feature_store import (
    build_abfss_uri,
    build_feature_source_dataframe,
    render_feature_set_spec_yaml,
)


def test_build_feature_source_dataframe_adds_entity_and_timestamp_columns() -> None:
    """
    Verify that the derived feature-source schema includes the required columns.
    """

    dataset = {
        "sepal length (cm)": [5.1, 4.9],
        "sepal width (cm)": [3.5, 3.0],
        "petal length (cm)": [1.4, 1.4],
        "petal width (cm)": [0.2, 0.2],
        "species": ["setosa", "setosa"],
    }

    feature_source = build_feature_source_dataframe(pd.DataFrame(dataset))

    assert list(feature_source.columns) == [
        "flower_id",
        "event_timestamp",
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
        "sepal length squared",
        "petal length squared",
        "sepal area (cm^2)",
        "petal area (cm^2)",
        "sepal length x petal length",
        "species",
    ]
    assert feature_source.loc[0, "flower_id"] == "flower_00001"
    assert feature_source.loc[1, "flower_id"] == "flower_00002"
    assert feature_source.loc[0, "event_timestamp"].endswith("Z")
    assert feature_source.loc[0, "sepal length squared"] == 5.1**2
    assert feature_source.loc[0, "sepal area (cm^2)"] == 5.1 * 3.5


def test_build_abfss_uri_uses_expected_adls_format() -> None:
    """
    Verify that ADLS paths are rendered in ABFSS format.
    """

    uri = build_abfss_uri(
        account_name="azuremlirishp01",
        filesystem="irisfs",
        relative_path="feature-store/source/iris_feature_source.csv",
    )

    assert (
        uri
        == "abfss://irisfs@azuremlirishp01.dfs.core.windows.net/"
        "feature-store/source/iris_feature_source.csv"
    )


def test_render_feature_set_spec_yaml_points_to_timestamped_source() -> None:
    """
    Verify that the scaffolded feature set spec names the timestamp and features.
    """

    yaml_text = render_feature_set_spec_yaml(
        "abfss://irisfs@azuremlirishp01.dfs.core.windows.net/feature-store/source/iris_feature_source.csv"
    )

    assert "event_timestamp" in yaml_text
    assert "flower_id" in yaml_text
    assert "sepal length (cm)" in yaml_text
    assert "sepal length squared" in yaml_text
    assert "sepal area (cm^2)" in yaml_text
