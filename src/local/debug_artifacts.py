"""
Utilities for inspecting MLflow run artefacts.

This module provides a small command-line helper that connects to the
configured MLflow tracking server, identifies the most recent run in
the project's experiment, and recursively lists the artefacts logged
for that run.
"""

from __future__ import annotations

from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient


# Name of the MLflow experiment inspected by this debugging helper.
EXPERIMENT_NAME = "simple_iris_demo"


def get_latest_run_id(experiment_name: str = EXPERIMENT_NAME) -> str:
    """
    Retrieve the latest run ID for the configured MLflow experiment.

    Parameters
    ----------
    experiment_name : str, default=EXPERIMENT_NAME
        Name of the MLflow experiment whose latest run should be
        inspected.

    Returns
    -------
    str
        Run ID of the most recently started run in the experiment.

    Raises
    ------
    ValueError
        Raised when the named experiment cannot be found or contains
        no runs.
    """

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        output_format="pandas",
    )
    if runs_df.empty:
        raise ValueError(f"No runs found for MLflow experiment: {experiment_name}")

    latest_run_id: str = runs_df.sort_values(
        by="start_time",
        ascending=False,
    ).iloc[0]["run_id"]

    return latest_run_id


def walk_artifacts(
    client: MlflowClient,
    run_id: str,
    path: str = "",
) -> None:
    """
    Recursively list MLflow artefacts for a given run.

    Parameters
    ----------
    client : MlflowClient
        MLflow client used to query the tracking server.

    run_id : str
        Unique identifier of the MLflow run whose artefacts should be
        listed.

    path : str, default=""
        Relative artefact path from which to begin traversal.

    Returns
    -------
    None
        The function prints artefact paths to standard output.
    """

    items = client.list_artifacts(run_id, path)

    for item in items:
        print(item.path)

        if item.is_dir:
            walk_artifacts(client, run_id, item.path)


def main(run_id: Optional[str] = None) -> None:
    """
    Print the artefact tree for a specific or inferred MLflow run.

    Parameters
    ----------
    run_id : str | None, default=None
        Run ID to inspect. When omitted, the most recent run in the
        configured experiment is used.

    Returns
    -------
    None
        The function prints the selected run ID and artefact paths.
    """

    client = MlflowClient()
    selected_run_id = run_id or get_latest_run_id()

    print(f"Latest run ID: {selected_run_id}")
    walk_artifacts(client, selected_run_id)


if __name__ == "__main__":
    main()
