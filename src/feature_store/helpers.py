"""
Helpers for preparing Iris data for Azure ML feature store workflows.

This module contains the small transformation and YAML-rendering
helpers used by the repository's feature-store preparation script.
The goal is to keep the pure data-shaping logic separate from Azure
CLI orchestration so that:

1. the derived feature-source schema stays easy to inspect
2. the transformation logic can be tested locally
3. the YAML scaffold generation remains deterministic

The helpers in this file do not call Azure services directly. They
only:

- reshape the canonical Iris dataset into a feature-store-friendly form
- build ADLS Gen2 ABFSS URIs
- render minimal entity, feature-set, and feature-set-spec YAML content
- write those scaffold files to disk
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from ..core.data import FEATURE_COLUMNS
except ImportError:
    from core.data import FEATURE_COLUMNS


# Canonical column order used by the derived feature-source dataset.
FEATURE_SOURCE_COLUMNS = [
    "flower_id",
    "event_timestamp",
    *FEATURE_COLUMNS,
    "species",
]


def build_feature_source_dataframe(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Build a feature-store-friendly source dataset from canonical Iris data.

    The feature store discussion for this repository needs two columns
    that are not present in the original toy dataset:

    1. a stable entity key
    2. a timestamp column for time-aware feature retrieval

    This helper derives those fields while preserving the original
    canonical feature columns and the human-readable species label.

    Parameters
    ----------
    dataset : pd.DataFrame
        DataFrame containing the canonical project feature columns plus
        a ``species`` label column.

    Returns
    -------
    pd.DataFrame
        DataFrame containing ``flower_id`` and ``event_timestamp``
        alongside the original features and species label.

    Raises
    ------
    ValueError
        Raised when the supplied dataset does not contain the canonical
        feature columns or the ``species`` column required to build the
        derived feature-source dataset.

    Notes
    -----
    - ``flower_id`` is synthetic because the Iris dataset does not have
      a natural business entity key.

    - ``event_timestamp`` is also synthetic and increases hourly from a
      fixed anchor timestamp so the resulting dataset is compatible with
      time-series-oriented feature-store concepts.
    """

    required_columns = [*FEATURE_COLUMNS, "species"]
    missing_columns = [column for column in required_columns if column not in dataset.columns]
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns for feature-source generation: {missing_columns}"
        )

    # Keep only the canonical dataset columns before deriving the
    # feature-store-specific entity and timestamp fields.
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

    Parameters
    ----------
    account_name : str
        Name of the Azure Storage account hosting the ADLS Gen2 data.

    filesystem : str
        ADLS Gen2 filesystem name.

    relative_path : str
        Relative path inside the filesystem.

    Returns
    -------
    str
        Fully-qualified ``abfss://`` URI suitable for feature-set
        source specifications.
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

    Parameters
    ----------
    entity_name : str, default="flower"
        Name of the feature-store entity.

    entity_version : str, default="1"
        Version string written into the entity YAML.

    Returns
    -------
    str
        YAML text defining a single-string-key entity based on
        ``flower_id``.
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

    Parameters
    ----------
    feature_set_name : str, default="iris_measurements"
        Name of the feature set asset.

    feature_set_version : str, default="1"
        Version string written into the feature set YAML.

    entity_name : str, default="flower"
        Entity name referenced by the feature set.

    entity_version : str, default="1"
        Entity version referenced by the feature set.

    specification_path : str, default="./spec"
        Relative path to the feature set specification folder.

    Returns
    -------
    str
        YAML text defining the feature set asset metadata.
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

    Parameters
    ----------
    source_abfss_uri : str
        ABFSS URI pointing at the derived feature-source CSV stored in
        ADLS Gen2.

    Returns
    -------
    str
        Feature set specification YAML text that points directly at the
        prepared source data and declares the timestamp, entity, and
        feature schema.

    Notes
    -----
    - This repository uses a simple direct-source specification first.
      It avoids custom feature transformation code in the initial
      scaffold because the source data has already been pre-shaped by
      the preparation script.
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

    Parameters
    ----------
    root_dir : Path
        Root directory under which the scaffold files should be
        written.

    source_abfss_uri : str
        ABFSS URI for the derived feature-source dataset.

    entity_name : str, default="flower"
        Entity name used for the generated entity YAML.

    entity_version : str, default="1"
        Entity version used for the generated entity YAML.

    feature_set_name : str, default="iris_measurements"
        Feature set name used for the generated feature set YAML.

    feature_set_version : str, default="1"
        Feature set version used for the generated feature set YAML.

    Returns
    -------
    list[Path]
        Paths of the generated scaffold files.

    Notes
    -----
    - The scaffold contains three files:

        1. entity YAML
        2. feature set YAML
        3. feature set specification YAML
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
