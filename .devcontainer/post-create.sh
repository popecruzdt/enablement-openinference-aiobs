#!/bin/bash
#loading functions to script
export SECONDS=0
source .devcontainer/util/source_framework.sh

setUpTerminal

startK3dCluster

installK9s

#TODO: BeforeGoLive: uncomment this. This is only needed for professors to have the Mkdocs live in the container

#installMkdocs


# Dynatrace Operator can be deployed automatically
#dynatraceDeployOperator

# You can deploy CNFS or AppOnly
#deployCloudNative
#deployApplicationMonitoring

# Declares the variables this training reads (also used to generate .devcontainer/.env.example for local runs).
# All optional: without Dynatrace or Arize credentials the stack still deploys, but the collector does not export.
variablesNeeded DT_ENVIRONMENT:false DT_INGEST_TOKEN:false ARIZE_SPACE_ID:false ARIZE_API_KEY:false

# AI Travel Advisor stack: nginx, travel-advisor (FastAPI), ollama, weaviate, otel-collector.
# Defined in .devcontainer/util/my_functions.sh, manifests in app/k8s/.
deployTravelAdvisor

# The Astroshop keeping changes of demo.live needs certmanagerdocker
#certmanagerInstall
#certmanagerEnable
#deployAstroshop

# If you want to deploy your own App, just create a function in the functions.sh file and call it here.
# deployMyCustomApp

# If the Codespace was created via Workflow end2end test will be done, otherwise
# it'll verify if there are error in the logs and will show them in the greeting as well a monitoring 
# notification will be sent on the instantiation details
finalizePostCreation

printInfoSection "Your dev container finished creating"
