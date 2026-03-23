# Azure ML Iris Endpoint Deployment

This repository is a small end-to-end Azure Machine Learning demo built around the Iris dataset.

It covers the full path from:

1. loading and splitting the data
2. training and evaluating a scikit-learn model
3. saving the trained model in MLflow format
4. registering the model in Azure ML
5. deploying the registered model to a managed online endpoint
6. invoking the endpoint as a REST API

The deployed endpoint is intended to run in Azure Machine Learning Managed Online Endpoints, with Azure handling the serving infrastructure for the model container.

## What This Project Does

The project trains a `RandomForestClassifier` on the Iris dataset and packages it for Azure ML inference.

At deployment time:

1. Azure ML creates a managed online endpoint.
2. Azure ML provisions a deployment under that endpoint.
3. Azure ML builds the inference environment defined in `deployment/conda.yml`.
4. Azure ML mounts the registered MLflow model in the container.
5. Azure ML calls `src/score.py` to load the model and score requests.

## Repository Structure

```text
Azure-AML-Iris/
├── deployment/
│   ├── conda.yml
│   ├── deployment.yml
│   ├── endpoint.yml
│   └── sample-request.json
├── pipelines/
│   ├── conda.yml
│   ├── train_evaluate.yml
│   └── components/
│       ├── evaluate.yml
│       └── train.yml
├── models/
│   └── register_from_job.yml
├── scripts/
│   ├── deployment/
│   │   ├── autoscale.sh
│   │   ├── check-azureml.sh
│   │   ├── cleanup.sh
│   │   ├── deploy.sh
│   │   ├── describe.sh
│   │   ├── get-key.sh
│   │   ├── invoke-curl.sh
│   │   ├── logs.sh
│   │   ├── reset.sh
│   │   ├── status.sh
│   │   └── test-endpoint.sh
│   ├── model/
│   │   └── register-from-job.sh
│   └── pipeline/
│       ├── register-components.sh
│       └── submit.sh
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── debug_artifacts.py
│   ├── evaluate.py
│   ├── pipeline_evaluate.py
│   ├── pipeline_train.py
│   ├── predict.py
│   ├── register.py
│   ├── score.py
│   └── train.py
├── tests/
│   ├── test_data.py
│   └── test_evaluate.py
├── .amlignore
├── .amlignore.amltmp
├── .gitignore
├── README.md
└── requirements.txt
```

## Core Files

- `src/data.py`
  Loads the Iris dataset and produces a reproducible stratified train/test split.

- `src/train.py`
  Trains the Random Forest model, logs metrics to MLflow, and saves the model locally in MLflow format under `outputs/iris_mlflow_model`.

- `src/evaluate.py`
  Computes accuracy, classification report output, confusion matrix, and predictions.

- `src/predict.py`
  Loads the locally saved MLflow model and runs a simple local prediction example.

- `src/register.py`
  Registers the saved MLflow model into the Azure ML model registry as `simple_iris_rf_model`.

- `src/score.py`
  Azure ML inference entry point. Azure calls `init()` once when the container starts and `run()` for each request.

- `deployment/endpoint.yml`
  Defines the managed online endpoint.

- `deployment/deployment.yml`
  Defines the deployment attached to the endpoint, including the registered model, scoring script, environment, and compute.

- `deployment/conda.yml`
  Defines the inference container environment used by Azure ML.

- `pipelines/components/train.yml`
  Defines the reusable Azure ML command component that trains the Iris model and emits an MLflow model output.

- `pipelines/components/evaluate.yml`
  Defines the reusable Azure ML command component that evaluates the trained model.

- `pipelines/train_evaluate.yml`
  Defines the Azure ML pipeline job that chains the training and evaluation components together.

- `models/register_from_job.yml`
  Template for registering a pipeline-produced MLflow model output as an Azure ML model asset.

## Azure Prerequisites

You need:

- an Azure subscription
- an Azure Machine Learning workspace
- Azure CLI installed locally
- the Azure ML CLI v2 extension
- permission to create or update managed online endpoints in your Azure ML workspace

Check your Azure CLI version:

```bash
az version
```

Install the Azure ML extension if needed:

```bash
az extension add -n ml
```

If the extension is already installed, update it:

```bash
az extension update -n ml
```

Sign in:

```bash
az login
```

Set the subscription you want to use:

```bash
az account set --subscription "<YOUR_SUBSCRIPTION_ID_OR_NAME>"
```

Set Azure CLI defaults so you do not need to keep repeating workspace and resource group names:

```bash
az configure --defaults group="<YOUR_RESOURCE_GROUP>" workspace="<YOUR_WORKSPACE_NAME>" location="<YOUR_REGION>"
```

Check the defaults currently set:

```bash
az configure --list-defaults
```

Check the active Azure ML workspace:

```bash
az ml workspace show
```

## Git And GitHub Setup

This project was connected to GitHub as an existing local folder rather than by cloning directly into a new directory.

### Local Windows Folder

The local project folder used for Git setup was:

```text
C:\Users\HP\OneDrive\Documents\Projects\Azure\Azure-AML-Iris
```

### Azure ML / Cloud Notebook Folder

In Azure ML notebooks or compute instances, the working folder may instead look like:

```text
~/cloudfiles/code/Users/<YOUR_USERNAME>/simple_aml_project
```

Use the folder where the project files already exist before running the Git commands below.

### Connect An Existing Folder To GitHub

From the project root:

```bash
git init
git remote add origin https://github.com/Ch3rry-Pi3-Azure/Simple-AzureML-Iris-Project.git
git pull origin main
git branch --set-upstream-to=origin/main main
```

This connects the current folder to the GitHub repository and pulls the tracked files into that folder without using `git clone`.

### GitHub Authentication For This Project

This project used a GitHub **classic personal access token** with HTTPS authentication.

When Git prompts during `pull` or `push`:

- username: your GitHub username
- password: paste the GitHub classic personal access token

### Store Credentials So Git Does Not Keep Prompting

In Azure ML Linux-based terminals, a common setup is:

```bash
git config --global credential.helper store
```

That causes Git to save credentials after the first successful authentication so later `git pull` and `git push` commands do not keep asking for the token.

After setting the helper, run a Git operation such as:

```bash
git pull origin main
```

Then enter:

- GitHub username
- GitHub classic PAT

Git will usually store the credentials in:

```text
~/.git-credentials
```

Note that `credential.helper store` keeps credentials in plain text, so it is simple but not the most secure option.

### Safe Directory Fix On Local Windows

Because this repository was initialized once from a different user context, Git on Windows may report a `dubious ownership` warning.

If that happens, run:

```bash
git config --global --add safe.directory C:/Users/HP/OneDrive/Documents/Projects/Azure/Azure-AML-Iris
```

Then verify Git works normally:

```bash
git status
```

## Azure Resource Provider Checks

If this subscription has not been used for Azure ML before, provider registration is often part of the initial setup.

Check registration state:

```bash
az provider list --query "[?namespace=='Microsoft.MachineLearningServices' || namespace=='Microsoft.Storage' || namespace=='Microsoft.ContainerRegistry' || namespace=='Microsoft.KeyVault'].{Provider:namespace,State:registrationState}" -o table
```

If needed, register the providers:

```bash
az provider register --namespace Microsoft.MachineLearningServices
az provider register --namespace Microsoft.Storage
az provider register --namespace Microsoft.ContainerRegistry
az provider register --namespace Microsoft.KeyVault
```

These are commonly relevant because Azure ML workspaces and managed online deployments rely on them.

## Python Environment

Create and activate a virtual environment if you want to run the training and registration steps locally:

```bash
python -m venv .venv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Model Lifecycle

### 1. Train the Model

Run:

```bash
python -m src.train
```

This will:

- load the Iris dataset
- split the data
- train a Random Forest classifier
- log metrics to MLflow
- save the model locally to `outputs/iris_mlflow_model`

### 2. Run a Local Prediction Check

Run:

```bash
python -m src.predict
```

This loads the saved MLflow model and runs a sample prediction locally.

### 3. Register the Model in Azure ML

Run:

```bash
python -m src.register
```

This registers the saved MLflow model as:

```text
simple_iris_rf_model
```

The deployment YAML currently references:

```text
azureml:simple_iris_rf_model:1
```

The checked-in YAML keeps that as a fallback placeholder, but the deployment script now resolves the latest registered version automatically unless you explicitly set `MODEL_VERSION`.

## Azure ML Pipelines

This repository now includes a component-based Azure ML pipeline so you can start using the `Pipelines` asset in Azure ML Studio.

The pipeline is designed to keep training and evaluation inside Azure ML while leaving endpoint deployment as a separate, controlled step.

### What the Pipeline Does

The pipeline in `pipelines/train_evaluate.yml` runs two jobs:

1. `train_job`
   Trains the Iris classifier and writes:
   - an MLflow model output
   - training metrics and reports

2. `evaluate_job`
   Loads the model output from `train_job`, recreates the deterministic test split, and writes evaluation reports.

### What Appears in Azure ML Studio

When you submit `pipelines/train_evaluate.yml`:

- the run appears under the `Pipelines` section in Azure ML Studio
- the child jobs appear under `Jobs`

If you also register the components first, they will appear under the `Components` section:

```bash
./scripts/pipeline/register-components.sh
```

### Submit the Pipeline

The pipeline defaults to:

```text
settings.default_compute = azureml:serverless
```

If your workspace supports serverless jobs, you can submit immediately:

```bash
./scripts/pipeline/submit.sh
```

You can also submit the YAML directly:

```bash
az ml job create --file pipelines/train_evaluate.yml
```

If you want to use a named compute cluster instead, override the pipeline default compute at submission time. For example:

```bash
az ml job create \
  --file pipelines/train_evaluate.yml \
  --set settings.default_compute=azureml:cpu-cluster
```

### Pipeline Outputs

The pipeline exposes three top-level outputs:

- `trained_model`
  MLflow model output from the training step

- `train_metrics`
  Training metrics and reports

- `evaluation_report`
  Evaluation metrics and reports

The pipeline steps also log first-class MLflow metrics into Azure ML so you can see them under the `Metrics` tab for both `train_job` and `evaluate_job`.

The key logged metrics are:

- training: `train_accuracy`, `train_precision_weighted`, `train_recall_weighted`, `train_f1_weighted`
- evaluation: `eval_accuracy`, `eval_precision_weighted`, `eval_recall_weighted`, `eval_f1_weighted`

This means the pipeline is now a clean upstream workflow for:

- model training
- model evaluation
- later model registration
- later endpoint deployment

### Register The Pipeline Output As A Model

The train/evaluate pipeline does not automatically register the model asset. Registration is handled separately using the model YAML in `models/register_from_job.yml`.

After the pipeline run finishes, get the job name:

```bash
az ml job list -o table
```

Then register the `trained_model` pipeline output.

If you want the script to automatically use the latest pipeline run:

```bash
./scripts/model/register-from-job.sh
```

If you want to be explicit about the run:

```bash
./scripts/model/register-from-job.sh <PIPELINE_JOB_NAME>
```

For example:

```bash
./scripts/model/register-from-job.sh upbeat_tree_q9x2k1lm4n
```

This creates or versions the Azure ML model asset:

```text
simple_iris_rf_model
```

You can then verify it under `Models` in Azure ML Studio or by CLI:

```bash
az ml model list --name simple_iris_rf_model -o table
```

### Suggested Next Step

The clean next extension is:

1. run the Azure ML pipeline
2. inspect the pipeline outputs in Studio
3. register the pipeline-produced model as a model asset
4. deploy a chosen registered model version to the online endpoint

That gives you a clearer separation between:

- ML workflow orchestration in `Pipelines`
- model serving in `Endpoints`

## Managed Online Endpoint Deployment

This repository is configured for:

- endpoint name: `roger-iris-endpoint-01`
- deployment name: `blue`

Create the endpoint and deployment:

```bash
./scripts/deployment/deploy.sh
```

The deploy script:

- creates the managed online endpoint from `deployment/endpoint.yml`
- creates the deployment from `deployment/deployment.yml`
- resolves the latest registered version of `simple_iris_rf_model` by default
- routes traffic to the deployment

Deploy a specific registered model version:

```bash
MODEL_VERSION=3 ./scripts/deployment/deploy.sh
```

Deploy a different model name explicitly:

```bash
MODEL_NAME=my_other_model MODEL_VERSION=7 ./scripts/deployment/deploy.sh
```

Check deployment state:

```bash
./scripts/deployment/status.sh
```

Describe the endpoint:

```bash
./scripts/deployment/describe.sh
```

Get logs:

```bash
./scripts/deployment/logs.sh
```

Delete the endpoint when finished:

```bash
./scripts/deployment/cleanup.sh
```

Redeploy from scratch:

```bash
./scripts/deployment/reset.sh
```

`reset.sh` uses the same deployment logic, so it also deploys the latest registered model version by default unless you set `MODEL_VERSION`.

## Endpoint Request Format

The scoring script now accepts either of these request formats under the top-level `data` key.

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

The checked-in sample request uses the named-feature object format in `deployment/sample-request.json`.

## Testing the Endpoint

Invoke the deployment through Azure ML:

```bash
./scripts/deployment/test-endpoint.sh
```

Invoke the scoring URI directly with `curl`:

```bash
./scripts/deployment/invoke-curl.sh
```

Retrieve endpoint keys:

```bash
./scripts/deployment/get-key.sh
```

## Azure ML Verification Commands

If you had deployment trouble before, these are the main commands worth remembering.

Verify Azure account and defaults:

```bash
az account show --query "{name:name,id:id}" -o table
az configure --list-defaults
az ml workspace show
```

Verify endpoint and deployment status:

```bash
az ml online-endpoint list -o table
az ml online-endpoint show --name roger-iris-endpoint-01
az ml online-deployment show --endpoint-name roger-iris-endpoint-01 --name blue
az ml online-endpoint show --name roger-iris-endpoint-01 --query traffic
```

Check deployment logs:

```bash
az ml online-deployment get-logs --endpoint-name roger-iris-endpoint-01 --name blue --lines 200
az ml online-deployment get-logs --endpoint-name roger-iris-endpoint-01 --name blue --container storage-initializer --lines 200
```

Invoke the endpoint manually:

```bash
az ml online-endpoint invoke \
  --name roger-iris-endpoint-01 \
  --deployment-name blue \
  --request-file deployment/sample-request.json
```

Show credentials and scoring URI:

```bash
az ml online-endpoint get-credentials --name roger-iris-endpoint-01
az ml online-endpoint show --name roger-iris-endpoint-01 --query scoring_uri -o tsv
```

For a bundled verification pass, run:

```bash
./scripts/deployment/check-azureml.sh
```

## Common Azure ML Troubleshooting Notes

- If `az ml` commands fail because the workspace or resource group is missing, set Azure CLI defaults with `az configure --defaults ...`.
- If deployment provisioning fails, inspect `get-logs` output first.
- If the inference container never comes up, also inspect `storage-initializer` logs.
- If a deployment succeeds but does not receive traffic, check `az ml online-endpoint show --query traffic`.
- If you re-register the model and want to deploy a newer version, update the model version in `deployment/deployment.yml`.
- If Azure reports provider or subscription registration errors, register the required resource providers for the subscription.
- If you need more detail from the CLI, rerun the failing command with `--debug`.

## Windows Note

The operational scripts in `scripts/` are Bash scripts.

On Windows, run them from:

- Git Bash
- WSL
- another Bash-compatible shell

If you prefer, the equivalent Azure CLI commands listed in this README can be run directly in PowerShell with minor syntax adjustments.

## References

- Azure ML CLI v2 setup:
  https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?view=azureml-api-2
- Managed online endpoints:
  https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints?view=azureml-api-2
- Online endpoint authentication:
  https://learn.microsoft.com/en-us/azure/machine-learning/how-to-authenticate-online-endpoint?view=azureml-api-2
- Online endpoint troubleshooting:
  https://learn.microsoft.com/en-us/azure/machine-learning/how-to-troubleshoot-online-endpoints?view=azureml-api-2
- Azure resource provider registration:
  https://learn.microsoft.com/en-us/azure/azure-resource-manager/troubleshooting/error-register-resource-provider
