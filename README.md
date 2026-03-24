# Azure ML Iris Endpoint Deployment

End-to-end Azure Machine Learning demo built around the Iris dataset.

This repository covers:

1. data loading and splitting
2. model training and evaluation
3. GridSearchCV-based model selection
4. MLflow model packaging
5. Azure ML pipeline execution
6. model registration
7. managed online endpoint deployment
8. endpoint monitoring for drift and data quality

## Table Of Contents

- [Overview](#overview)
- [Workflow Diagrams](#workflow-diagrams)
- [Repository Structure](#repository-structure)
- [Azure Prerequisites](#azure-prerequisites)
- [Windows Azure CLI Setup](#windows-azure-cli-setup)
- [Git And GitHub Setup](#git-and-github-setup)
- [Local Workflow](#local-workflow)
- [GitHub Actions CI/CD](#github-actions-cicd)
- [ADLS Datastore And Data Asset Setup](#adls-datastore-and-data-asset-setup)
- [Feature Store Prep](#feature-store-prep)
- [Azure ML Pipeline Workflow](#azure-ml-pipeline-workflow)
- [Endpoint Deployment And Teardown](#endpoint-deployment-and-teardown)
- [Azure ML Monitoring](#azure-ml-monitoring)
- [Endpoint Request Format](#endpoint-request-format)
- [Verification Commands](#verification-commands)
- [Troubleshooting Notes](#troubleshooting-notes)
- [Windows Note](#windows-note)
- [References](#references)

## Overview

<details open>
<summary>Show or hide section</summary>

The project trains a `RandomForestClassifier` on the Iris dataset, packages it in MLflow format, registers it in Azure ML, and deploys it to a managed online endpoint.

The training path now uses a modular scikit-learn pipeline so the repo
can act as a better template for future projects:

- shared derived feature generation lives in `src/core/features.py`
- shared preprocessing pipeline assembly lives in `src/core/preprocessing.py`
- the model still receives the original four Iris measurements at
  inference time, while the preprocessing pipeline expands them
  internally before prediction

There are two main ways to use the repo:

- local workflow for training and artifact generation under `outputs/`
- Azure ML workflow for registered components, pipeline runs, model registration, deployment, and monitoring

At inference time, Azure ML:

1. provisions the managed online endpoint and deployment
2. builds the inference environment from `deployment/conda.yml`
3. mounts the registered MLflow model inside the container
4. calls `src/serving/score.py` to load the model and serve predictions
5. collects production input/output data for monitoring

</details>

## Workflow Diagrams

<details open>
<summary>Show or hide section</summary>

Main project workflow:

```mermaid
flowchart LR
    A[Raw Iris Source<br/>Azure ML data asset or local CSV or load_iris()] --> B[Shared Data Loading<br/>src/core/data.py]
    B --> C[Shared Feature Engineering<br/>src/core/features.py]
    C --> D[Training Pipeline<br/>src/core/preprocessing.py + src/core/modeling.py]
    D --> E[MLflow Model Artifact]
    E --> F[Azure ML Model Registration]
    F --> G[Managed Online Deployment]
    G --> H[Endpoint Requests<br/>src/serving/score.py]
    H --> I[Collected Inputs / Outputs]
    I --> J[Azure ML Monitoring]
```

Feature-store preparation workflow:

```mermaid
flowchart LR
    A[iris_csv data asset<br/>or local fallback] --> B[src/feature_store/prepare_source.py]
    B --> C[Derived reusable features<br/>flower_id, event_timestamp,<br/>squared terms, areas, interaction]
    C --> D[Derived CSV in ADLS]
    D --> E[iris_feature_source data asset]
    E --> F[Feature store scaffold YAML<br/>entity + feature set + spec]
```

</details>

## Repository Structure

<details>
<summary>Show or hide section</summary>

```text
Azure-AML-Iris/
|-- deployment/
|   |-- conda.yml
|   |-- deployment.yml
|   |-- endpoint.yml
|   `-- sample-request.json
|-- models/
|   `-- register_from_job.yml
|-- monitoring/
|   `-- endpoint_monitor.yml
|-- pipelines/
|   |-- conda.yml
|   |-- train_evaluate.yml
|   `-- components/
|       |-- evaluate.yml
|       `-- train.yml
|-- scripts/
|   |-- deployment/
|   |   |-- autoscale.sh
|   |   |-- check-azureml.sh
|   |   |-- cleanup.sh
|   |   |-- deploy.sh
|   |   |-- describe.sh
|   |   |-- destroy-deployment.sh
|   |   |-- get-key.sh
|   |   |-- invoke-curl.sh
|   |   |-- logs.sh
|   |   |-- reset.sh
|   |   |-- status.sh
|   |   `-- test-endpoint.sh
|   |-- model/
|   |   `-- register-from-job.sh
|   |-- monitoring/
|   |   |-- create-monitor.sh
|   |   |-- delete-monitor.sh
|   |   `-- status.sh
|   |-- feature-store/
|   |   `-- prepare-source.sh
|   `-- pipeline/
|       |-- download-outputs.sh
|       |-- register-components.sh
|       `-- submit.sh
|-- data/
|   `-- iris.csv
|-- infra/
|   |-- adls_datastore.yml
|   `-- iris_data.yml
|-- src/
|   |-- __init__.py
|   |-- core/
|   |   |-- __init__.py
|   |   |-- artifact_names.py
|   |   |-- data.py
|   |   |-- evaluate.py
|   |   |-- features.py
|   |   |-- modeling.py
|   |   |-- preprocessing.py
|   |   `-- visualize.py
|   |-- feature_store/
|   |   |-- __init__.py
|   |   |-- helpers.py
|   |   `-- prepare_source.py
|   |-- local/
|   |   |-- __init__.py
|   |   |-- debug_artifacts.py
|   |   |-- predict.py
|   |   `-- train.py
|   |-- pipeline/
|   |   |-- __init__.py
|   |   |-- evaluate.py
|   |   `-- train.py
|   |-- registry/
|   |   |-- __init__.py
|   |   `-- register.py
|   `-- serving/
|       |-- __init__.py
|       `-- score.py
|-- tests/
|   |-- test_data.py
|   |-- test_evaluate.py
|   |-- test_features.py
|   `-- test_feature_store.py
|-- .amlignore
|-- .gitignore
|-- README.md
`-- requirements.txt
```

Core files:

- `src/local/train.py`
  local training entry point; saves the deployable MLflow model and local run artifacts
- `src/core/modeling.py`
  shared GridSearchCV and estimator assembly logic used by both local and pipeline training
- `src/core/preprocessing.py`
  shared scikit-learn preprocessing pipeline that generates reusable derived features and can optionally apply scaling
- `src/core/features.py`
  shared raw-to-derived feature engineering used by training and feature-store preparation
- `src/core/evaluate.py`
  shared evaluation helpers for metrics, reports, and plots
- `src/serving/score.py`
  Azure ML inference entry point with production data collection for monitoring
- `src/feature_store/prepare_source.py`
  CLI helper that derives a feature-store-ready source dataset from the Azure ML data asset or local fallback data
- `pipelines/train_evaluate.yml`
  Azure ML pipeline job that chains training and evaluation
- `models/register_from_job.yml`
  YAML template for registering a pipeline output as a model asset
- `deployment/deployment.yml`
  managed online deployment spec, including production data collection
- `monitoring/endpoint_monitor.yml`
  recurring Azure ML monitor schedule

</details>

## Azure Prerequisites

<details>
<summary>Show or hide section</summary>

You need:

- an Azure subscription
- an Azure Machine Learning workspace
- Azure CLI
- Azure ML CLI v2 extension
- permission to create or update models, pipelines, endpoints, deployments, and schedules

Basic setup:

```bash
az version
az extension add -n ml
az extension update -n ml
az login
az account set --subscription "<YOUR_SUBSCRIPTION_ID_OR_NAME>"
az configure --defaults group="<YOUR_RESOURCE_GROUP>" workspace="<YOUR_WORKSPACE_NAME>" location="<YOUR_REGION>"
az configure --list-defaults
az ml workspace show
```

Provider registration checks:

```bash
az provider list --query "[?namespace=='Microsoft.MachineLearningServices' || namespace=='Microsoft.Storage' || namespace=='Microsoft.ContainerRegistry' || namespace=='Microsoft.KeyVault'].{Provider:namespace,State:registrationState}" -o table
az provider register --namespace Microsoft.MachineLearningServices
az provider register --namespace Microsoft.Storage
az provider register --namespace Microsoft.ContainerRegistry
az provider register --namespace Microsoft.KeyVault
```

</details>

## Windows Azure CLI Setup

<details>
<summary>Show or hide section</summary>

On local Windows machines, two practical setup points matter:

1. use the 64-bit Azure CLI
2. if the default `C:\Users\<YOU>\.azure` folder causes permission issues, point Azure CLI at a repo-local config folder for the current shell session

If `az` is not on `PATH` after reinstalling Azure CLI, add it for the current PowerShell session:

```powershell
$env:PATH = "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin;$env:PATH"
az version
```

Use a repo-local Azure CLI config directory in the current shell:

```powershell
$env:AZURE_CONFIG_DIR = "$PWD\.azure"
New-Item -ItemType Directory -Force -Path $env:AZURE_CONFIG_DIR | Out-Null
az version
```

Check your current subscription:

```powershell
az account show --query "{subscriptionName:name,subscriptionId:id,tenantId:tenantId}" -o table
```

List Azure ML workspaces:

```powershell
az ml workspace list --query "[].{workspace:name,resourceGroup:resourceGroup,location:location}" -o table
```

Set Azure CLI defaults for this repo:

```powershell
az configure --defaults group="<YOUR_RESOURCE_GROUP>" workspace="<YOUR_WORKSPACE_NAME>" location="<YOUR_REGION>"
az configure --list-defaults
```

Example from this repo setup:

```powershell
az configure --defaults group="azureml_rg" workspace="azureml_test" location="westeurope"
```

</details>

## Git And GitHub Setup

<details>
<summary>Show or hide section</summary>

This repo was connected from an existing folder rather than cloned into a fresh directory.

Local Windows folder:

```text
C:\Users\HP\OneDrive\Documents\Projects\Azure\Azure-AML-Iris
```

Typical Azure ML notebook / compute instance folder:

```text
~/cloudfiles/code/Users/<YOUR_USERNAME>/simple_aml_project
```

Connect an existing folder:

```bash
git init
git remote add origin https://github.com/Ch3rry-Pi3-Azure/Simple-AzureML-Iris-Project.git
git pull origin main
git branch --set-upstream-to=origin/main main
```

This project used a GitHub classic PAT over HTTPS.

When prompted:

- username: your GitHub username
- password: your GitHub classic PAT

Store credentials in Azure ML Linux terminals:

```bash
git config --global credential.helper store
git pull origin main
```

That typically stores credentials in:

```text
~/.git-credentials
```

If Git reports `dubious ownership` on local Windows:

```bash
git config --global --add safe.directory C:/Users/HP/OneDrive/Documents/Projects/Azure/Azure-AML-Iris
git status
```

</details>

## Local Workflow

<details>
<summary>Show or hide section</summary>

Optional virtual environment:

```bash
python -m venv .venv
pip install -r requirements.txt
```

Train locally:

```bash
python -m src.local.train
```

This:

- loads and splits the Iris dataset from `data/iris.csv` when present, otherwise falls back to `sklearn.datasets.load_iris()`
- builds reusable derived features before fitting the estimator
- runs `GridSearchCV` for the Random Forest pipeline
- refits the best estimator
- logs MLflow metrics
- saves the deployable MLflow model to `outputs/iris_mlflow_model`
- writes dated local artifacts under `outputs/local_runs/<YYYY-MM-DD>/<HHMMSS>_<RUN_ID>/`

The local training entry point keeps scaling disabled by default
because the current model is a Random Forest. The preprocessing logic
is still isolated in `src/core/preprocessing.py` so future models can
turn scaling on without rewriting the project structure.

Local artifact examples:

- `best_params.json`
- `metrics.json`
- `cv_results.csv`
- `grid_search_summary.json`
- `classification_report.txt`
- `classification_report.json`
- `classification_report.png`
- `confusion_matrix.json`
- `confusion_matrix.png`
- `roc_curve.png`
- `learning_curve.png`
- `oob_error_curve.png`

Local prediction check:

```bash
python -m src.local.predict
```

Register the locally saved model:

```bash
python -m src.registry.register
```

The registered model name is:

```text
simple_iris_rf_model
```

Notes on plots:

- `roc_curve.png` is multiclass one-vs-rest, so it contains three curves
- a true loss-vs-iteration plot is not included because the deployed model is a `RandomForestClassifier`
- the more appropriate training progression plots here are `learning_curve.png` and `oob_error_curve.png`

</details>

## GitHub Actions CI/CD

<details>
<summary>Show or hide section</summary>

The repository now includes two GitHub Actions workflows under `.github/workflows/`:

- `ci.yml`
  runs on pushes to `main` and pull requests targeting `main`; installs dependencies, compiles `src/` and `tests/`, and runs `pytest`
- `cd-register-model.yml`
  runs after a successful `CI` workflow on `main` or can be started manually; registers Azure ML components, submits the training pipeline, and registers the trained model from the resulting pipeline job

The CD workflow intentionally stops after model registration. It does not deploy the online endpoint or run endpoint smoke tests.

Required GitHub repository secrets for Azure OIDC login:

- `AZURE_CLIENT_ID`
- `AZURE_TENANT_ID`
- `AZURE_SUBSCRIPTION_ID`

These are identifiers used by the OIDC login flow. This repository's
GitHub Actions workflow does **not** use an Azure client secret.
Do not create or store `AZURE_CLIENT_SECRET` for this setup unless you
intentionally change the workflow away from OIDC.

Required GitHub repository or environment variables:

- `AZURE_RESOURCE_GROUP`
- `AZURE_ML_WORKSPACE`
- `AZURE_LOCATION`

Optional GitHub repository or environment variables:

- `AZURE_ML_DATA_ASSET_NAME`
  defaults to `iris_csv`
- `AZURE_ML_DATA_ASSET_VERSION`
  defaults to `1`
- `AZURE_ML_REGISTERED_MODEL_NAME`
  defaults to `simple_iris_rf_model`

Recommended setup:

1. create a GitHub environment such as `azureml-dev`
2. add the Azure OIDC secrets to that environment
3. add the Azure ML workspace variables to that environment
4. create a federated credential in Azure AD for the repository/environment

The federated credential must use this subject format:

```text
repo:<OWNER>/<REPO>:environment:azureml-dev
```

For this repository, the value should be:

```text
repo:Ch3rry-Pi3-Azure/Simple-AzureML-Iris-Project:environment:azureml-dev
```

Use your own environment-specific values for:

```text
AZURE_CLIENT_ID=<your-app-registration-client-id>
AZURE_TENANT_ID=<your-tenant-id>
AZURE_SUBSCRIPTION_ID=<your-subscription-id>
AZURE_RESOURCE_GROUP=<your-resource-group>
AZURE_ML_WORKSPACE=<your-aml-workspace>
AZURE_LOCATION=<your-region>
```

The federated credential should be configured like this for this
repository/environment:

```text
name=github-azureml-dev
issuer=https://token.actions.githubusercontent.com
subject=repo:Ch3rry-Pi3-Azure/Simple-AzureML-Iris-Project:environment:azureml-dev
audience=api://AzureADTokenExchange
```

Step 1. Create the Azure application identity:

```powershell
$APP_NAME = "github-azureml-simple-iris"
$APP_ID = az ad app create --display-name $APP_NAME --query appId -o tsv
az ad sp create --id $APP_ID
az role assignment create --assignee $APP_ID --role Contributor --scope /subscriptions/<SUBSCRIPTION_ID>/resourceGroups/<RESOURCE_GROUP>
```

Step 2. Create the GitHub OIDC federated credential:

```powershell
@'
{
  "name": "github-azureml-dev",
  "issuer": "https://token.actions.githubusercontent.com",
  "subject": "repo:<OWNER>/<REPO>:environment:azureml-dev",
  "description": "GitHub Actions OIDC for azureml-dev",
  "audiences": [
    "api://AzureADTokenExchange"
  ]
}
'@ | Set-Content -Path .\github-federated-credential.json

az ad app federated-credential create --id $APP_ID --parameters .\github-federated-credential.json
```

Step 3. Put the outputs into the GitHub environment:

Environment secrets:

- `AZURE_CLIENT_ID`
  set this to the value stored in `$APP_ID`
- `AZURE_TENANT_ID`
  set this to your Azure tenant ID
- `AZURE_SUBSCRIPTION_ID`
  set this to your Azure subscription ID

Environment variables:

- `AZURE_RESOURCE_GROUP`
- `AZURE_ML_WORKSPACE`
- `AZURE_LOCATION`

Optional environment variables:

- `AZURE_ML_DATA_ASSET_NAME`
- `AZURE_ML_DATA_ASSET_VERSION`
- `AZURE_ML_REGISTERED_MODEL_NAME`

For this project, the non-secret GitHub environment variables are:

```text
AZURE_RESOURCE_GROUP=azureml_rg
AZURE_ML_WORKSPACE=azureml_test
AZURE_LOCATION=westeurope
```

The CD workflow uses the same repo scripts you run manually:

- `./scripts/pipeline/register-components.sh`
- `./scripts/pipeline/submit.sh`
- `./scripts/model/register-from-job.sh`

</details>

## ADLS Datastore And Data Asset Setup

<details>
<summary>Show or hide section</summary>

If you want the Iris data to live in Azure storage instead of being loaded directly from `sklearn.datasets.load_iris()`, a clean first step is:

1. create an ADLS Gen2-capable storage account
2. create a filesystem
3. upload `iris.csv`
4. register an Azure ML datastore that points at that filesystem
5. create an Azure ML data asset for the CSV

This setup is useful even before introducing a feature store.

Set helper variables in PowerShell:

```powershell
$RG="azureml_rg"
$LOCATION="westeurope"
$WORKSPACE="azureml_test"
$SA="azuremlirishp01"
$FS="irisfs"
$DS="iris_adls_ds"
$DATA_ASSET="iris_csv"
```

Create the storage account with hierarchical namespace enabled:

```powershell
az storage account create `
  --name $SA `
  --resource-group $RG `
  --location $LOCATION `
  --sku Standard_LRS `
  --kind StorageV2 `
  --allow-blob-public-access false `
  --enable-hierarchical-namespace true `
  --min-tls-version TLS1_2
```

Verify the storage account:

```powershell
az storage account show `
  --name $SA `
  --resource-group $RG `
  --query "{name:name,kind:kind,location:primaryLocation,isHnsEnabled:isHnsEnabled,minimumTlsVersion:minimumTlsVersion}" `
  -o table
```

If the storage account was created before `TLS1_2` was applied, correct it explicitly:

```powershell
az storage account update --name $SA --resource-group $RG --min-tls-version TLS1_2
```

Create the ADLS Gen2 filesystem:

```powershell
az storage fs create --name $FS --account-name $SA --auth-mode login
az storage fs list --account-name $SA --auth-mode login -o table
```

Upload `iris.csv` into the filesystem:

```powershell
az storage fs file upload `
  --source ".\data\iris.csv" `
  --path "raw/iris.csv" `
  --file-system $FS `
  --account-name $SA `
  --auth-mode login
```

Create the `infra/` folder and write the datastore YAML:

```powershell
New-Item -ItemType Directory -Force -Path .\infra | Out-Null

@'
$schema: https://azuremlschemas.azureedge.net/latest/azureDataLakeGen2.schema.json
name: iris_adls_ds
type: azure_data_lake_gen2
description: Iris data in ADLS Gen2
account_name: azuremlirishp01
filesystem: irisfs
'@ | Set-Content -Path .\infra\adls_datastore.yml
```

Register the datastore:

```powershell
az ml datastore create --file .\infra\adls_datastore.yml
az ml datastore show --name $DS
```

Write the data asset YAML:

```powershell
@'
$schema: https://azuremlschemas.azureedge.net/latest/data.schema.json
name: iris_csv
version: 1
description: Iris CSV stored in ADLS Gen2
type: uri_file
path: azureml://datastores/iris_adls_ds/paths/raw/iris.csv
'@ | Set-Content -Path .\infra\iris_data.yml
```

Register the data asset:

```powershell
az ml data create --file .\infra\iris_data.yml
az ml data show --name $DATA_ASSET --version 1
```

Notes:

- `src/core/data.py` now supports a pragmatic fallback chain: use an explicit CSV path when provided, otherwise use `data/iris.csv` when present, otherwise fall back to `sklearn.datasets.load_iris()`.
- The datastore and data asset are the storage and cataloguing layer. They do not automatically change online inference in `src/serving/score.py`.
- The Azure ML pipeline definitions now accept an optional `data_input` so training and evaluation can consume the registered data asset without breaking the fallback path.
- The training component also exposes a `use_scaling` flag. It defaults to `false` for the current Random Forest example, but the pipeline shape is already ready for future estimators that need scaling.

</details>

## Feature Store Prep

<details>
<summary>Show or hide section</summary>

This repository now includes a pragmatic first step toward Azure ML feature store integration.

The helper script:

1. checks whether the Azure ML data asset `iris_csv` exists
2. downloads the asset from the registered datastore when present
3. falls back to `data/iris.csv` and then to `sklearn.datasets.load_iris()` if needed
4. derives `flower_id` and `event_timestamp`
5. materialises reusable derived features such as squared lengths, areas, and a cross-term interaction
6. writes a feature-store-friendly source CSV locally under `outputs/feature_store/`
7. uploads that derived file back to ADLS Gen2
8. registers a derived Azure ML data asset such as `iris_feature_source`
9. writes feature store scaffold YAML files locally under `featurestore/iris_demo/`

Run it with defaults:

```bash
python -m src.feature_store.prepare_source
./scripts/feature-store/prepare-source.sh
```

Useful overrides:

```bash
python -m src.feature_store.prepare_source \
  --source-data-asset-name iris_csv \
  --source-data-asset-version 1 \
  --datastore-name iris_adls_ds \
  --derived-data-asset-name iris_feature_source \
  --derived-data-asset-version auto \
  --upload-path feature-store/source/iris_feature_source.csv
```

Generated outputs:

- local derived file: `outputs/feature_store/iris_feature_source.csv`
- derived data asset: `azureml:iris_feature_source:<VERSION>`
- scaffold entity YAML: `featurestore/iris_demo/flower_entity.yml`
- scaffold feature set YAML: `featurestore/iris_demo/iris_measurements.yml`
- scaffold feature set spec: `featurestore/iris_demo/spec/FeatureSetSpec.yaml`

The derived feature-source dataset contains:

- the synthetic entity key `flower_id`
- the synthetic retrieval timestamp `event_timestamp`
- the four canonical raw Iris measurements
- reusable derived features:
  - `sepal length squared`
  - `petal length squared`
  - `sepal area (cm^2)`
  - `petal area (cm^2)`
  - `sepal length x petal length`

Important scope note:

- The script prepares the source data and scaffold files needed for feature store registration.
- It does not create a managed feature store workspace for you.
- It does not register the entity or feature set automatically because those steps target a feature store workspace rather than the regular Azure ML workspace used by this repo.
- If the source data asset exists but Azure ML exposes it through a workspace-asset URI shape that the helper cannot download directly, the helper falls back to `data/iris.csv` instead of failing the whole workflow.

</details>

## Azure ML Pipeline Workflow

<details>
<summary>Show or hide section</summary>

The repo includes a component-based Azure ML pipeline in `pipelines/train_evaluate.yml`.

Pipeline stages:

1. `train_job`
   trains the best model with the shared preprocessing-and-model pipeline and writes `trained_model` plus training artifacts
2. `evaluate_job`
   reloads the model, recreates the deterministic test split, and writes evaluation artifacts

Register the pipeline components first:

```bash
./scripts/pipeline/register-components.sh
```

Submit the pipeline:

```bash
./scripts/pipeline/submit.sh
```

By default, `submit.sh`:

- submits the pipeline
- checks whether `azureml:iris_csv:1` exists and, if so, passes it as the optional `data_input`
- falls back to the built-in/local Iris loading path if the data asset is not found
- waits for completion
- downloads outputs locally into `outputs/azure_runs/<YYYY-MM-DD>/<PIPELINE_JOB_NAME>/`

Disable waiting and auto-download:

```bash
WAIT_FOR_COMPLETION=false AUTO_DOWNLOAD_OUTPUTS=false ./scripts/pipeline/submit.sh
```

Control the automatic data-asset behaviour:

```bash
USE_DATA_ASSET=false ./scripts/pipeline/submit.sh
USE_DATA_ASSET=true DATA_ASSET_NAME=iris_csv DATA_ASSET_VERSION=1 ./scripts/pipeline/submit.sh
```

If you want to turn scaling on for the training pipeline template, set
the pipeline input in `pipelines/train_evaluate.yml`:

```yaml
inputs:
  use_scaling: true
```

Use a specific compute target instead of serverless:

```bash
az ml job create \
  --file pipelines/train_evaluate.yml \
  --set settings.default_compute=azureml:cpu-cluster
```

Download outputs later:

```bash
./scripts/pipeline/download-outputs.sh
./scripts/pipeline/download-outputs.sh <PIPELINE_JOB_NAME>
INCLUDE_MODEL_OUTPUT=true ./scripts/pipeline/download-outputs.sh <PIPELINE_JOB_NAME>
```

Register the pipeline-produced model:

```bash
./scripts/model/register-from-job.sh
./scripts/model/register-from-job.sh <PIPELINE_JOB_NAME>
```

Recommended Azure ML flow:

```bash
./scripts/pipeline/register-components.sh
./scripts/pipeline/submit.sh
./scripts/model/register-from-job.sh
./scripts/deployment/deploy.sh
```

Pipeline output folders contain both metric files and richer artifacts such as:

- `best_params.json`
- `cv_results.csv`
- `grid_search_summary.json`
- `classification_report.txt`
- `classification_report.json`
- `classification_report.png`
- `confusion_matrix.json`
- `confusion_matrix.png`
- `roc_curve.png`
- `learning_curve.png`
- `oob_error_curve.png`

Key MLflow metrics shown in Azure ML:

- training: `train_accuracy`, `train_precision_weighted`, `train_recall_weighted`, `train_f1_weighted`
- evaluation: `eval_accuracy`, `eval_precision_weighted`, `eval_recall_weighted`, `eval_f1_weighted`

</details>

## Endpoint Deployment And Teardown

<details>
<summary>Show or hide section</summary>

Default endpoint settings in this repo:

- endpoint name: `roger-iris-endpoint-01`
- deployment name: `blue`

Deploy the latest registered model:

```bash
./scripts/deployment/deploy.sh
```

Deploy a specific model version:

```bash
MODEL_VERSION=3 ./scripts/deployment/deploy.sh
```

Deployment helpers:

```bash
./scripts/deployment/status.sh
./scripts/deployment/describe.sh
./scripts/deployment/logs.sh
./scripts/deployment/test-endpoint.sh
./scripts/deployment/invoke-curl.sh
./scripts/deployment/get-key.sh
./scripts/deployment/check-azureml.sh
```

Destroy only the deployment:

```bash
./scripts/deployment/destroy-deployment.sh
```

Destroy the deployment and then the endpoint:

```bash
./scripts/deployment/cleanup.sh
```

`cleanup.sh` now:

1. deletes the named deployment first
2. waits for Azure to fully remove the deployment
3. deletes the endpoint
4. waits for Azure to fully remove the endpoint

Reset from scratch:

```bash
./scripts/deployment/reset.sh
```

Optional autoscale setup:

```bash
RESOURCE_GROUP="<YOUR_RESOURCE_GROUP>" ./scripts/deployment/autoscale.sh
```

</details>

## Azure ML Monitoring

<details>
<summary>Show or hide section</summary>

This repo now includes a first monitoring pass for the managed online endpoint.

Monitoring is wired through:

1. `deployment/deployment.yml`
   enables `data_collector` for `model_inputs` and `model_outputs`
2. `deployment/conda.yml`
   installs `azureml-ai-monitoring`
3. `src/serving/score.py`
   logs pandas DataFrames for model inputs and model outputs
4. `monitoring/endpoint_monitor.yml`
   defines a recurring Azure ML monitor schedule

The checked-in schedule uses the minimal out-of-box Azure ML monitor shape.

That means Azure ML creates the default basic monitor against the collected endpoint data first, which is a safer starting point than a fully custom signal definition when CLI support varies by environment.

Deploy the endpoint with monitoring support:

```bash
./scripts/deployment/deploy.sh
```

Create the monitor schedule:

```bash
./scripts/monitoring/create-monitor.sh
```

By default the monitoring target is:

```text
azureml:roger-iris-endpoint-01:blue
```

Override endpoint or deployment:

```bash
ENDPOINT_NAME=my-endpoint DEPLOYMENT_NAME=green ./scripts/monitoring/create-monitor.sh
```

Check monitor status:

```bash
./scripts/monitoring/status.sh
az ml schedule list -o table
```

Delete the monitor schedule:

```bash
./scripts/monitoring/delete-monitor.sh
```

Notes:

- drift and quality signals are only useful after the endpoint has seen production traffic
- this does not replace endpoint health checks or container logs
- this is not yet ground-truth performance monitoring
- if a monitor run stays queued for a long time, the usual issue is Azure serverless Spark capacity or quota rather than your endpoint code
- the checked-in monitor YAML now uses `standard_e8s_v3` as the Spark instance type because it can be a more practical retry target than `standard_e4s_v3` for stuck serverless runs
- some Azure ML compute images still require `runtime_version: "3.4"` explicitly in the monitor YAML even though newer docs describe it as optional
- once monitor creation is stable in your workspace, the schedule can be expanded later with explicit custom signals and thresholds

</details>

## Endpoint Request Format

<details>
<summary>Show or hide section</summary>

The scoring script accepts either named feature objects or ordered feature lists under the top-level `data` field.

Named feature objects:

```json
{
  "data": [
    {
      "sepal length (cm)": 5.1,
      "sepal width (cm)": 3.5,
      "petal length (cm)": 1.4,
      "petal width (cm)": 0.2
    }
  ]
}
```

Ordered feature lists:

```json
{
  "data": [
    [5.1, 3.5, 1.4, 0.2]
  ]
}
```

The checked-in sample request is:

```text
deployment/sample-request.json
```

</details>

## Verification Commands

<details>
<summary>Show or hide section</summary>

Azure account and defaults:

```bash
az account show --query "{name:name,id:id}" -o table
az configure --list-defaults
az ml workspace show
```

Pipeline and model checks:

```bash
az ml component list -o table
az ml job list -o table
az ml model list --name simple_iris_rf_model -o table
```

Endpoint and deployment checks:

```bash
az ml online-endpoint list -o table
az ml online-endpoint show --name roger-iris-endpoint-01
az ml online-deployment show --endpoint-name roger-iris-endpoint-01 --name blue
az ml online-endpoint show --name roger-iris-endpoint-01 --query traffic
```

Logs and invocation:

```bash
az ml online-deployment get-logs --endpoint-name roger-iris-endpoint-01 --name blue --lines 200
az ml online-deployment get-logs --endpoint-name roger-iris-endpoint-01 --name blue --container storage-initializer --lines 200
az ml online-endpoint invoke --name roger-iris-endpoint-01 --deployment-name blue --request-file deployment/sample-request.json
az ml online-endpoint get-credentials --name roger-iris-endpoint-01
az ml online-endpoint show --name roger-iris-endpoint-01 --query scoring_uri -o tsv
```

</details>

## Troubleshooting Notes

<details>
<summary>Show or hide section</summary>

- If `az ml` commands fail because the workspace or resource group is missing, fix Azure CLI defaults first.
- If component code changes, rerun `./scripts/pipeline/register-components.sh` before submitting the pipeline again.
- If the pipeline run completed in Azure but you do not see artifacts locally, run `./scripts/pipeline/download-outputs.sh`.
- If deployment provisioning fails, inspect deployment logs first, then `storage-initializer` logs.
- If the endpoint already exists during redeploy, use `./scripts/deployment/destroy-deployment.sh` or `./scripts/deployment/cleanup.sh` depending on whether you want to keep the endpoint.
- If deletion races happen, the current teardown scripts now poll until Azure actually removes the resource.
- If the endpoint receives no traffic, inspect `az ml online-endpoint show --query traffic`.
- If you need more CLI detail, rerun the failing command with `--debug`.

</details>

## Windows Note

<details>
<summary>Show or hide section</summary>

The operational scripts in `scripts/` are Bash scripts.

On Windows, run them from:

- Git Bash
- WSL
- another Bash-compatible shell

If needed, the underlying Azure CLI commands can still be run directly in PowerShell with minor syntax adjustments.

</details>

## References

<details>
<summary>Show or hide section</summary>

- Azure ML CLI v2 setup  
  https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?view=azureml-api-2
- Managed online endpoints  
  https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints?view=azureml-api-2
- Online endpoint authentication  
  https://learn.microsoft.com/en-us/azure/machine-learning/how-to-authenticate-online-endpoint?view=azureml-api-2
- Online endpoint troubleshooting  
  https://learn.microsoft.com/en-us/azure/machine-learning/how-to-troubleshoot-online-endpoints?view=azureml-api-2
- Collect production data from online endpoints  
  https://learn.microsoft.com/en-us/azure/machine-learning/how-to-collect-production-data?view=azureml-api-2
- Model monitor schedule YAML reference  
  https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-monitor?view=azureml-api-2
- Azure resource provider registration  
  https://learn.microsoft.com/en-us/azure/azure-resource-manager/troubleshooting/error-register-resource-provider

</details>
