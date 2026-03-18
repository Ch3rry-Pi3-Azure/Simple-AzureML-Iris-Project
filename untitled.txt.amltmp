# Azure ML Iris Endpoint Deployment

This project demonstrates how to deploy a trained scikit-learn model to an Azure Machine Learning Managed Online Endpoint.

The deployed endpoint exposes a REST API that can receive inference requests and return predictions.

The project uses:

- Azure Machine Learning CLI (v2)
- MLflow model registry
- Managed online endpoints
- Containerised inference environments
- Kubernetes infrastructure managed by Azure ML


Project Structure

simple_aml_project
│
├── deployment
│   ├── endpoint.yml
│   ├── deployment.yml
│   ├── conda.yml
│   └── sample-request.json
│
├── src
│   └── score.py
│
├── scripts
│   ├── deploy.sh
│   ├── test-endpoint.sh
│   ├── logs.sh
│   ├── status.sh
│   ├── cleanup.sh
│   └── invoke-curl.sh
│
└── README.md


Prerequisites

Install the Azure CLI

https://learn.microsoft.com/en-us/cli/azure/install-azure-cli

Install the Azure ML CLI extension

az extension add -n ml

Login to Azure

az login

Set the active subscription if necessary

az account set --subscription "<subscription-id>"


Making Scripts Executable

Run this once after cloning the repository.

chmod +x scripts/*.sh


Deployment Workflow

Create the endpoint and deployment

./scripts/deploy.sh

This script performs the following steps:

1. Creates the Azure ML endpoint
2. Builds the inference container environment
3. Deploys the model
4. Routes traffic to the deployment

Deployment usually takes several minutes while Azure builds the container environment.


Check Deployment Status

./scripts/status.sh

Expected output once deployment is ready:

Succeeded


Test the Endpoint

Send a test inference request using the sample input.

./scripts/test-endpoint.sh

The request payload is stored in

deployment/sample-request.json


View Container Logs

If deployment fails or debugging is required:

./scripts/logs.sh

This retrieves logs from the inference container running in Azure ML.


Invoke the Endpoint Using curl

To test the endpoint like a normal REST API:

./scripts/invoke-curl.sh

This script retrieves the endpoint key and sends a request using curl.


Scale the Deployment

To manually increase the number of serving instances:

./scripts/scale.sh


Delete the Endpoint

To remove the deployment and avoid ongoing compute charges:

./scripts/cleanup.sh

Deleting the endpoint automatically removes:

- deployments
- containers
- serving compute


Typical Development Workflow

Deploy the endpoint

./scripts/deploy.sh

Check deployment status

./scripts/status.sh

Send a test request

./scripts/test-endpoint.sh

Inspect logs if something fails

./scripts/logs.sh

Delete the endpoint when finished

./scripts/cleanup.sh


Inference Architecture

Azure Machine Learning Managed Endpoints run on Kubernetes infrastructure managed internally by Azure.

Each deployment runs inside a container that hosts the Azure ML inference server.

The scoring script defines two required functions.

init()

Executed once when the container starts.  
This function loads the model into memory.

run()

Executed for every inference request sent to the endpoint.


Useful Azure CLI Commands

Show endpoint details

az ml online-endpoint show --name roger-iris-endpoint-01

Retrieve endpoint credentials

az ml online-endpoint get-credentials --name roger-iris-endpoint-01

Invoke endpoint directly

az ml online-endpoint invoke \
  --name roger-iris-endpoint-01 \
  --deployment-name blue \
  --request-file deployment/sample-request.json


Resources

Azure ML Managed Endpoints  
https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints

Azure ML CLI v2  
https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli

Azure ML Deployment Guide  
https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints