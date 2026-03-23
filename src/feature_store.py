"""
Helpers for preparing Iris data for Azure ML feature store workflows.

This module keeps the feature-source transformation logic and YAML
template rendering separate from Azure CLI orchestration code so the
core behaviour can be reused and tested locally.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from .data import FEATURE_COLUMNS
except ImportError:
    from data import FEATURE_COLUMNS


FEATURE_SOURCE_COLUMNS = [
    "flower_id",
    "event_timestamp",
    *FEATURE_COLUMNS,
    "species",
]


def build_feature_source_dataframe(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Build a feature-store-friendly source dataset from canonical Iris data.

    Parameters
    ----------
    dataset : pd.DataFrame
        DataFrame containing the canonical project feature columns plus
        a `species` label column.

    Returns
    -------
    pd.DataFrame
        DataFrame containing entity and timestamp columns alongside the
        original features and species label.
    """

    required_columns = [*FEATURE_COLUMNS, "species"]
    missing_columns = [column for column in required_columns if column not in dataset.columns]
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns for feature-source generation: {missing_columns}"
        )

    feature_source = dataset[required_columns].copy().reset_index(drop=True)
    feature_source.insert(
        0,
        "event_timestamp",
        pd.date_range(
            start="2024-01-01T00:00:00Z",
            periods=len(feature_source),
            freq="h",
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    feature_source.insert(
        0,
        "flower_id",
        [f"flower_{index:05d}" for index in range(1, len(feature_source) + 1)],
    )
    return feature_source[FEATURE_SOURCE_COLUMNS]


def build_abfss_uri(
    account_name: str,
    filesystem: str,
    relative_path: str,
) -> str:
    """
    Build an ABFSS URI for ADLS Gen2-backed feature-store source data.
    """

    normalized_relative_path = relative_path.lstrip("/")
    return (
        f"abfss://{filesystem}@{account_name}.dfs.core.windows.net/"
        f"{normalized_relative_path}"
    )


def render_feature_store_entity_yaml(
    entity_name: str = "flower",
    entity_version: str = "1",
) -> str:
    """
    Render a minimal feature-store entity YAML definition.
    """

    return f"""$schema: http://azureml/sdk-2-0/FeatureStoreEntity.json
name: {entity_name}
version: "{entity_version}"
description: Iris flower entity keyed by flower_id.
stage: Development
index_columns:
  - name: flower_id
    type: string
tags:
  project: azure-aml-iris
  source: derived-feature-source
"""


def render_feature_set_yaml(
    feature_set_name: str = "iris_measurements",
    feature_set_version: str = "1",
    entity_name: str = "flower",
    entity_version: str = "1",
    specification_path: str = "./spec",
) -> str:
    """
    Render a minimal feature set YAML definition.
    """

    return f"""$schema: http://azureml/sdk-2-0/Featureset.json
name: {feature_set_name}
version: "{feature_set_version}"
description: Canonical Iris measurement features sourced from ADLS Gen2.
specification:
  path: {specification_path}
entities:
  - azureml:{entity_name}:{entity_version}
stage: Development
tags:
  project: azure-aml-iris
  source: derived-feature-source
"""


def render_feature_set_spec_yaml(source_abfss_uri: str) -> str:
    """
    Render a simple feature set specification without transformation code.
    """

    return f"""$schema: http://azureml/sdk-2-0/FeatureSetSpec.json
source:
  type: csv
  path: {source_abfss_uri}
  timestamp_column:
    name: event_timestamp
    format: yyyy-MM-dd'T'HH:mm:ss'Z'
features:
  - name: sepal length (cm)
    type: double
  - name: sepal width (cm)
    type: double
  - name: petal length (cm)
    type: double
  - name: petal width (cm)
    type: double
index_columns:
  - name: flower_id
    type: string
"""


def write_feature_store_scaffold(
    root_dir: Path,
    source_abfss_uri: str,
    entity_name: str = "flower",
    entity_version: str = "1",
    feature_set_name: str = "iris_measurements",
    feature_set_version: str = "1",
) -> list[Path]:
    """
    Write feature-store scaffold files to disk.
    """

    spec_dir = root_dir / "spec"
    spec_dir.mkdir(parents=True, exist_ok=True)

    entity_path = root_dir / "flower_entity.yml"
    feature_set_path = root_dir / "iris_measurements.yml"
    spec_path = spec_dir / "FeatureSetSpec.yaml"

    entity_path.write_text(
        render_feature_store_entity_yaml(
            entity_name=entity_name,
            entity_version=entity_version,
        ),
        encoding="utf-8",
    )
    feature_set_path.write_text(
        render_feature_set_yaml(
            feature_set_name=feature_set_name,
            feature_set_version=feature_set_version,
            entity_name=entity_name,
            entity_version=entity_version,
        ),
        encoding="utf-8",
    )
    spec_path.write_text(
        render_feature_set_spec_yaml(source_abfss_uri=source_abfss_uri),
        encoding="utf-8",
    )

    return [entity_path, feature_set_path, spec_path]
