"""
Prepare a derived Iris feature-source dataset for Azure ML feature store work.

This script provides a pragmatic bridge between the repository's
existing Azure ML data-asset setup and a future managed feature store
workflow. It does not register feature store assets directly against a
feature store workspace. Instead, it handles the preparatory work that
is easy to automate inside the current project:

1. discover the existing workspace data asset when available
2. fall back to local or built-in Iris data when it is not
3. derive synthetic entity and timestamp columns required for a useful
   feature-store demonstration
4. upload the derived dataset back to ADLS Gen2
5. register the uploaded file as a new Azure ML data asset
6. write local scaffold YAML files for later feature store registration

Keeping this logic in a dedicated script makes the workflow repeatable
without forcing the main training and deployment codepaths to take on
feature-store-specific responsibilities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any

try:
    from ..core.data import DEFAULT_LOCAL_DATA_PATH, load_dataset_frame
    from .helpers import (
        build_abfss_uri,
        build_feature_source_dataframe,
        write_feature_store_scaffold,
    )
except ImportError:
    from core.data import DEFAULT_LOCAL_DATA_PATH, load_dataset_frame
    from feature_store.helpers import (
        build_abfss_uri,
        build_feature_source_dataframe,
        write_feature_store_scaffold,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "feature_store"
DEFAULT_SCAFFOLD_DIR = PROJECT_ROOT / "featurestore" / "iris_demo"


def _run_command(
    command: list[str],
    allow_failure: bool = False,
) -> subprocess.CompletedProcess:
    """
    Run a subprocess command and optionally allow failure.

    Parameters
    ----------
    command : list[str]
        Command and arguments passed directly to ``subprocess.run``.

    allow_failure : bool, default=False
        Whether non-zero exit codes should be tolerated and returned to
        the caller instead of raising an exception.

    Returns
    -------
    subprocess.CompletedProcess
        Completed subprocess result containing the return code plus the
        captured standard output and standard error streams.

    Raises
    ------
    RuntimeError
        Raised when the command fails and ``allow_failure`` is
        ``False``.
    """

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 and not allow_failure:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{stderr}")
    return result


def _run_az_json(
    command_suffix: list[str],
    allow_failure: bool = False,
) -> Any:
    """
    Run an Azure CLI command that returns JSON output.

    Parameters
    ----------
    command_suffix : list[str]
        Azure CLI arguments appended after the leading ``az`` command.

    allow_failure : bool, default=False
        Whether Azure CLI failures should return ``None`` instead of
        raising an exception.

    Returns
    -------
    Any
        Parsed JSON output from the Azure CLI command. When
        ``allow_failure`` is ``True`` and the command fails, the
        function returns ``None``.
    """

    result = _run_command(
        ["az", *command_suffix, "-o", "json"],
        allow_failure=allow_failure,
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)


def _parse_azureml_datastore_path(path: str) -> tuple[str, str] | None:
    """
    Parse an Azure ML datastore URI into datastore name and relative path.

    Parameters
    ----------
    path : str
        Azure ML datastore URI of the form
        ``azureml://datastores/<name>/paths/<relative-path>``.

    Returns
    -------
    tuple[str, str] | None
        Datastore name and relative path when the input matches the
        expected Azure ML datastore URI shape, otherwise ``None``.
    """

    prefix = "azureml://datastores/"
    path_segment = "/paths/"

    if not path.startswith(prefix) or path_segment not in path:
        return None

    remainder = path[len(prefix) :]
    datastore_name, relative_path = remainder.split(path_segment, maxsplit=1)
    return datastore_name, relative_path


def _get_next_data_asset_version(asset_name: str) -> str:
    """
    Determine the next numeric version for a workspace data asset.

    Parameters
    ----------
    asset_name : str
        Name of the Azure ML data asset container.

    Returns
    -------
    str
        Next numeric version represented as a string.

    Notes
    -----
    - Non-numeric versions are ignored when calculating the next
      automatic version number.
    """

    existing_assets = _run_az_json(
        ["ml", "data", "list", "--name", asset_name],
        allow_failure=True,
    )
    if not existing_assets:
        return "1"

    numeric_versions = []
    for asset in existing_assets:
        version = asset.get("version")
        try:
            numeric_versions.append(int(version))
        except (TypeError, ValueError):
            continue

    if not numeric_versions:
        return "1"

    return str(max(numeric_versions) + 1)


def _download_source_data_asset(
    source_asset_name: str,
    source_asset_version: str,
    temp_dir: Path,
) -> tuple[Path, str, dict[str, Any]] | None:
    """
    Download a workspace data asset that points at an ADLS datastore file.

    Parameters
    ----------
    source_asset_name : str
        Name of the Azure ML data asset to inspect.

    source_asset_version : str
        Version of the Azure ML data asset to inspect.

    temp_dir : Path
        Temporary directory used to hold the downloaded source file.

    Returns
    -------
    tuple[Path, str, dict[str, Any]] | None
        Downloaded local file path, datastore name, and datastore
        metadata when the source asset exists. Returns ``None`` when the
        asset is not found.

    Raises
    ------
    RuntimeError
        Raised when the asset exists but does not point at an Azure ML
        datastore-backed path of the expected form.
    """

    data_asset = _run_az_json(
        [
            "ml",
            "data",
            "show",
            "--name",
            source_asset_name,
            "--version",
            source_asset_version,
        ],
        allow_failure=True,
    )
    if not data_asset:
        return None

    asset_path = data_asset.get("path", "")
    parsed_path = _parse_azureml_datastore_path(asset_path)
    if not parsed_path:
        raise RuntimeError(
            "Only azureml://datastores/.../paths/... data assets are supported by "
            "src.feature_store.prepare_source"
        )

    datastore_name, relative_path = parsed_path
    datastore = _run_az_json(
        ["ml", "datastore", "show", "--name", datastore_name],
    )

    destination = temp_dir / Path(relative_path).name

    # Download the underlying ADLS file directly so the rest of the
    # script can operate on a normal local CSV path.
    _run_command(
        [
            "az",
            "storage",
            "fs",
            "file",
            "download",
            "--account-name",
            datastore["account_name"],
            "--file-system",
            datastore["filesystem"],
            "--path",
            relative_path,
            "--destination",
            str(destination),
            "--auth-mode",
            "login",
        ]
    )

    return destination, datastore_name, datastore


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for feature-source preparation.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments describing the source asset,
        fallback local path, output locations, datastore target, and
        generated asset names.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data-asset-name", default="iris_csv")
    parser.add_argument("--source-data-asset-version", default="1")
    parser.add_argument("--local-data-path", default=str(DEFAULT_LOCAL_DATA_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--datastore-name", default="iris_adls_ds")
    parser.add_argument(
        "--upload-path",
        default="feature-store/source/iris_feature_source.csv",
    )
    parser.add_argument("--derived-data-asset-name", default="iris_feature_source")
    parser.add_argument("--derived-data-asset-version", default="auto")
    parser.add_argument("--entity-name", default="flower")
    parser.add_argument("--entity-version", default="1")
    parser.add_argument("--feature-set-name", default="iris_measurements")
    parser.add_argument("--feature-set-version", default="1")
    parser.add_argument("--scaffold-dir", default=str(DEFAULT_SCAFFOLD_DIR))
    return parser.parse_args()


def main() -> None:
    """
    Prepare and register a derived feature-source dataset.

    The function performs the following high-level steps:

        1. resolves the best available source dataset
        2. derives feature-store-friendly columns
        3. writes the derived CSV locally
        4. uploads the derived CSV to ADLS Gen2
        5. registers the uploaded file as a new Azure ML data asset
        6. writes local entity / feature-set scaffold files

    Returns
    -------
    None
        The function writes local files, uploads the derived dataset,
        registers a data asset, and prints a compact summary.
    """

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        source_info = _download_source_data_asset(
            source_asset_name=args.source_data_asset_name,
            source_asset_version=args.source_data_asset_version,
            temp_dir=temp_dir,
        )

        # Resolve the source dataset in descending preference order:
        # Azure ML data asset, local CSV, then built-in scikit-learn data.
        if source_info is not None:
            source_path, source_datastore_name, source_datastore = source_info
            dataset = load_dataset_frame(data_path=source_path)
            source_description = (
                f"azureml:{args.source_data_asset_name}:{args.source_data_asset_version}"
            )
        elif Path(args.local_data_path).exists():
            source_path = Path(args.local_data_path)
            dataset = load_dataset_frame(data_path=source_path)
            source_description = str(source_path)
            source_datastore_name = args.datastore_name
            source_datastore = _run_az_json(
                ["ml", "datastore", "show", "--name", source_datastore_name],
            )
        else:
            dataset = load_dataset_frame(data_path=None)
            source_description = "sklearn.datasets.load_iris()"
            source_datastore_name = args.datastore_name
            source_datastore = _run_az_json(
                ["ml", "datastore", "show", "--name", source_datastore_name],
            )

        feature_source = build_feature_source_dataframe(dataset)
        local_output_path = output_dir / "iris_feature_source.csv"
        feature_source.to_csv(local_output_path, index=False)

        target_datastore_name = args.datastore_name or source_datastore_name
        if target_datastore_name != source_datastore_name:
            target_datastore = _run_az_json(
                ["ml", "datastore", "show", "--name", target_datastore_name],
            )
        else:
            target_datastore = source_datastore

        # Upload the derived file back to ADLS so it can be registered
        # as a normal Azure ML data asset and referenced later by the
        # feature-store scaffold.
        _run_command(
            [
                "az",
                "storage",
                "fs",
                "file",
                "upload",
                "--source",
                str(local_output_path),
                "--path",
                args.upload_path,
                "--file-system",
                target_datastore["filesystem"],
                "--account-name",
                target_datastore["account_name"],
                "--auth-mode",
                "login",
                "--overwrite",
                "true",
            ]
        )

    derived_version = (
        _get_next_data_asset_version(args.derived_data_asset_name)
        if args.derived_data_asset_version == "auto"
        else args.derived_data_asset_version
    )

    data_asset_path = (
        f"azureml://datastores/{target_datastore_name}/paths/{args.upload_path}"
    )
    _run_command(
        [
            "az",
            "ml",
            "data",
            "create",
            "--name",
            args.derived_data_asset_name,
            "--version",
            derived_version,
            "--type",
            "uri_file",
            "--path",
            data_asset_path,
            "--description",
            f"Derived Iris feature-source dataset generated from {source_description}.",
        ]
    )

    scaffold_root = Path(args.scaffold_dir)
    source_abfss_uri = build_abfss_uri(
        account_name=target_datastore["account_name"],
        filesystem=target_datastore["filesystem"],
        relative_path=args.upload_path,
    )
    scaffold_paths = write_feature_store_scaffold(
        root_dir=scaffold_root,
        source_abfss_uri=source_abfss_uri,
        entity_name=args.entity_name,
        entity_version=args.entity_version,
        feature_set_name=args.feature_set_name,
        feature_set_version=args.feature_set_version,
    )

    print("Feature-source preparation completed.")
    print(f"Source dataset: {source_description}")
    print(f"Local derived file: {local_output_path.resolve()}")
    print(f"Uploaded ADLS path: {data_asset_path}")
    print(
        "Registered derived data asset: "
        f"azureml:{args.derived_data_asset_name}:{derived_version}"
    )
    print("Feature store scaffold files:")
    for scaffold_path in scaffold_paths:
        print(f"- {scaffold_path.resolve()}")


if __name__ == "__main__":
    main()
