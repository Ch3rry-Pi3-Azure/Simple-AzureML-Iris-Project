"""
Utilities for inspecting MLflow run artefacts.

This module connects to the MLflow tracking server, retrieves the
most recent run from a specified experiment, and recursively lists
all artefacts stored for that run.

It is useful for quickly exploring the structure of logged outputs,
such as model files, plots, metrics, and other saved artefacts.
"""

from mlflow.tracking import MlflowClient
import mlflow


# Name of the MLflow experiment to inspect
EXPERIMENT_NAME = "simple_iris_demo"


# Create an MLflow client for artefact access
client = MlflowClient()

# Retrieve the experiment object by name
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

# Search all runs associated with the experiment
runs_df = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    output_format="pandas",
)

# Select the most recent run based on start time
latest_run_id = runs_df.sort_values(
    by="start_time",
    ascending=False,
).iloc[0]["run_id"]

print(f"Latest run ID: {latest_run_id}")


def walk(run_id: str, path: str = "") -> None:
    """
    Recursively list MLflow artefacts for a given run.

    The function traverses the artefact directory tree associated
    with an MLflow run and prints the full path of each item found.

    Parameters
    ----------
    run_id : str
        Unique identifier of the MLflow run whose artefacts
        should be inspected.

    path : str, default=""
        Relative artefact path from which to begin traversal.

        The default value of an empty string starts traversal
        at the top level of the run's artefact directory.

    Returns
    -------
    None
        The function prints artefact paths to the console.

    Notes
    -----
    - MLflow stores artefacts in a hierarchical structure, where
      directories may contain nested files or subdirectories.

    - This function uses recursion to descend through that
      directory structure until all artefacts have been listed.

    - The printed output can help identify where models, logs,
      plots, or other saved files are located within a run.

    Example
    -------
    List all artefacts for a specific run.

    >>> walk("abc123def456")

    Start traversal from a subdirectory.

    >>> walk("abc123def456", "model")
    """

    # List artefacts at the current path level
    items = client.list_artifacts(run_id, path)

    # Visit each artefact or subdirectory
    for item in items:
        print(item.path)

        # Recurse into subdirectories
        if item.is_dir:
            walk(run_id, item.path)


# Walk the artefact tree for the latest run
walk(latest_run_id)