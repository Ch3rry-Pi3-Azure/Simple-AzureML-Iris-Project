# Azure ML Iris Endpoint Deployment

This project demonstrates how to deploy a trained scikit-learn model to an Azure Machine Learning Managed Online Endpoint.

The deployed endpoint exposes a REST API that can receive inference requests and return predictions.

## Technologies Used

* Azure Machine Learning CLI (v2)
* MLflow model registry
* Managed online endpoints
* Containerised inference environments
* Kubernetes infrastructure managed by Azure ML

## Project Structure

```text
simple_aml_project/
├── deployment/
│   ├── endpoint.yml
│   ├── deployment.yml
│   ├── conda.yml
│   └── sample-request.json
├── src/
│   └── score.py
├── scripts/
│   ├── deploy.sh
│   ├── test-endpoint.sh
│   ├── logs.sh
│   ├── status.sh
│   ├── cleanup.sh
│   └── invoke-curl.sh
└── README.md
```

## Prerequisites

Install the Azure CLI.

Azure CLI installation guide:

[https://learn.microsoft.com/en-us/cli/azure/install-azure-cli](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli)

Install the Azure ML CLI extension:

```bash
az extension add -n ml
```

Log in to Azure:

```bash
az login
```

Set the active subscription if necessary:

```bash
az account set --subscription "<YOUR_SUBSCRIPTION_ID_OR_NAME>"
```

## Making Scripts Executable

Run this once after cloning the repository:

```bash
chmod +x scripts/*.sh
```

## Deployment Workflow

Create the endpoint and deployment:

```bash
./scripts/deploy.sh
```

This script performs the following steps:

* Creates the Azure ML endpoint
* Builds the inference container environment
* Deploys the model
* Routes traffic to the deployment

Deployment usually takes several minutes while Azure builds the container environment.

## Check Deployment Status

```bash
./scripts/status.sh
```

Expected output once deployment is ready:

```text
Succeeded
```

## Test the Endpoint

Send a test inference request using the sample input:

```bash
./scripts/test-endpoint.sh
```

The request payload is stored in:

```text
deployment/sample-request.json
```

## View Container Logs

If deployment fails or debugging is required:

```bash
./scripts/logs.sh
```

This retrieves logs from the inference container running in Azure ML.

## Invoke the Endpoint Using curl

To test the endpoint like a normal REST API:

```bash
./scripts/invoke-curl.sh
```

This script retrieves the endpoint key and sends a request using `curl`.

## Delete the Endpoint

To remove the deployment and avoid ongoing compute charges:

```bash
./scripts/cleanup.sh
```

Deleting the endpoint automatically removes:

* deployments
* containers
* serving compute

## Typical Development Workflow

Deploy the endpoint:

```bash
./scripts/deploy.sh
```

Check deployment status:

```bash
./scripts/status.sh
```

Send a test request:

```bash
./scripts/test-endpoint.sh
```

Inspect logs if something fails:

```bash
./scripts/logs.sh
```

Delete the endpoint when finished:

```bash
./scripts/cleanup.sh
```

## Inference Architecture

Azure Machine Learning Managed Endpoints run on Kubernetes infrastructure managed internally by Azure.

Each deployment runs inside a container that hosts the Azure ML inference server.

The scoring script defines two required functions.

### `init()`

Executed once when the container starts. This function loads the model into memory.

### `run()`

Executed for every inference request sent to the endpoint.

## Useful Azure CLI Commands

Show endpoint details:

```bash
az ml online-endpoint show --name roger-iris-endpoint-01
```

Retrieve endpoint credentials:

```bash
az ml online-endpoint get-credentials --name roger-iris-endpoint-01
```

Invoke the endpoint directly:

```bash
az ml online-endpoint invoke \
  --name roger-iris-endpoint-01 \
  --deployment-name blue \
  --request-file deployment/sample-request.json
```

## Resources

Azure ML Managed Endpoints:

[https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints](https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints)

Azure ML CLI v2:

[https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli)

Azure ML Deployment Guide:

[https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints)
