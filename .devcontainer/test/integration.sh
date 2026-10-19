#!/bin/bash
# Load framework
source .devcontainer/util/source_framework.sh

printInfoSection "Running integration Tests for $RepositoryName"

#TODO: In here you add your assertions
#assertRunningPod dynatrace operator

#assertRunningPod dynatrace activegate

#assertRunningPod dynatrace oneagent

assertRunningPod ai-travel-advisor ollama

assertRunningPod ai-travel-advisor weaviate

assertRunningPod ai-travel-advisor otel-collector

assertRunningPod ai-travel-advisor travel-advisor

assertRunningPod ai-travel-advisor travel-advisor-nginx

assertRunningApp ai-travel-advisor

LAB_WAIT=1 checkTravelAdvisorReady || exit 1

# TODO(starter): replace with your own steps' fail-first proof.
# Step 2's check must FAIL before its solution and PASS after it: a check that cannot fail proves nothing.
if checkNginxScaled; then
  printError "checkNginxScaled passed before the step's solution ran: the check cannot fail"
  exit 1
fi
kubectl scale deployment travel-advisor-nginx -n ai-travel-advisor --replicas=2
LAB_WAIT=1 checkNginxScaled || exit 1
