#!/bin/bash
# ======================================================================
#          ------- Custom Functions -------                            #
#  Your training-specific functions. This file is sourced into every   #
#  shell (Codespaces, Dev Containers and Orbital), so a function here  #
#  can be called from post-create.sh, from the terminal, from a        #
#  shell-verification `command:` and from a LAB_SOLUTION.              #
#                                                                      #
#  Write only what the framework lacks: it already ships checkNodeReady,#
#  checkTodoAppRunning, checkOperatorReady, checkDynakube,             #
#  checkOneAgentInjected, restartTodoApp, generateTodoTraffic and the  #
#  waitFor* helpers.                                                   #
#                                                                      #
#  Rules:                                                              #
#  - Use `return`, never `exit` — this file is sourced, not run.       #
#  - Use printInfoSection / printInfo / printError from the framework. #
#  - Keep functions idempotent — they may be called many times.        #
#  - A check returns 0 = pass. Probe once for a learner's click; wait  #
#    first only when LAB_WAIT is set (training test, Run solution).    #
# ======================================================================

# ---------------------------------------------------------------------
# AI Travel Advisor stack (nginx, travel-advisor, ollama, weaviate, otel-collector)
# Manifests live in app/k8s/. Namespace: ai-travel-advisor.
# ---------------------------------------------------------------------
TRAVEL_ADVISOR_NS="ai-travel-advisor"

deployTravelAdvisor() {
  local k8s_dir="${REPO_PATH:-$(pwd)}/app/k8s"
  local prompts_dir="${REPO_PATH:-$(pwd)}/app/prompts"

  printInfoSection "Deploying AI Travel Advisor stack"

  kubectl apply -f "$k8s_dir/00-namespace.yaml"

  # Collector credentials (OTLP export to Dynatrace). Never written to the repo.
  if dynatraceEvalReadSaveCredentials && [ -n "${DT_INGEST_TOKEN:-}" ]; then
    kubectl -n "$TRAVEL_ADVISOR_NS" create secret generic dynatrace \
      --from-literal="token=$DT_INGEST_TOKEN" \
      --from-literal="endpoint=$DT_OTEL_ENDPOINT" \
      --dry-run=client -o yaml | kubectl apply -f -
  else
    printWarn "Dynatrace credentials missing: creating the dynatrace secret with placeholders; the collector will not export"
    kubectl -n "$TRAVEL_ADVISOR_NS" create secret generic dynatrace \
      --from-literal="token=unset" \
      --from-literal="endpoint=http://localhost:4318" \
      --dry-run=client -o yaml | kubectl apply -f -
  fi

  # Prompts are a ConfigMap so lab steps can edit them (then restart travel-advisor).
  kubectl -n "$TRAVEL_ADVISOR_NS" create configmap travel-advisor-prompts \
    --from-file="$prompts_dir" --dry-run=client -o yaml | kubectl apply -f -

  printInfo "Deploying Ollama and pulling models"
  kubectl apply -f "$k8s_dir/10-ollama.yaml"
  kubectl -n "$TRAVEL_ADVISOR_NS" rollout status deploy/ollama --timeout=10m
  kubectl -n "$TRAVEL_ADVISOR_NS" delete job ollama-pull --ignore-not-found
  kubectl apply -f "$k8s_dir/11-ollama-pull-job.yaml"

  printInfo "Deploying Weaviate"
  kubectl apply -f "$k8s_dir/20-weaviate.yaml"
  kubectl -n "$TRAVEL_ADVISOR_NS" rollout status deploy/weaviate --timeout=10m

  printInfo "Deploying OpenTelemetry Collector"
  kubectl apply -f "$k8s_dir/30-otel-collector.yaml"
  kubectl -n "$TRAVEL_ADVISOR_NS" rollout status deploy/otel-collector --timeout=5m

  # The app init container waits for the models, so wait for the pull job first.
  kubectl -n "$TRAVEL_ADVISOR_NS" wait --for=condition=complete job/ollama-pull --timeout=15m \
    || { printError "ollama-pull did not complete; check: kubectl -n $TRAVEL_ADVISOR_NS logs job/ollama-pull"; return 1; }

  printInfo "Deploying AI Travel Advisor and nginx"
  kubectl apply -f "$k8s_dir/40-travel-advisor.yaml"
  kubectl apply -f "$k8s_dir/50-nginx.yaml"
  # Knowledge base indexing runs on app startup (embeddings through Ollama), so allow time.
  kubectl -n "$TRAVEL_ADVISOR_NS" rollout status deploy/travel-advisor --timeout=15m
  kubectl -n "$TRAVEL_ADVISOR_NS" rollout status deploy/travel-advisor-nginx --timeout=5m

  # registerApp already sets 3600s proxy timeouts on the ingress, which covers slow LLM responses.
  registerApp "ai-travel-advisor" "$TRAVEL_ADVISOR_NS" "travel-advisor-nginx" 80
}

undeployTravelAdvisor() {
  printInfoSection "Removing AI Travel Advisor stack"
  unregisterApp "ai-travel-advisor" "$TRAVEL_ADVISOR_NS" 2>/dev/null || true
  kubectl delete ns "$TRAVEL_ADVISOR_NS" --ignore-not-found
}

# All five stack components have a Running pod.
checkTravelAdvisorRunning() {
  local components="ollama weaviate otel-collector travel-advisor travel-advisor-nginx"
  local attempt=0 max=60 missing c
  while :; do
    missing=""
    for c in $components; do
      kubectl -n "$TRAVEL_ADVISOR_NS" get pods -l "app=$c" --no-headers 2>/dev/null | grep -q " Running " || missing="$missing $c"
    done
    [ -z "$missing" ] && { printInfo "All AI Travel Advisor components are Running"; return 0; }
    [ -z "${LAB_WAIT:-}" ] || [ "$attempt" -ge "$max" ] && break
    attempt=$((attempt + 1)); printInfo "Waiting for:$missing ($attempt/$max), sleeping 5s"; sleep 5
  done
  printError "Not Running:$missing"; return 1
}
waitForTravelAdvisorRunning() { LAB_WAIT=1 checkTravelAdvisorRunning; }

# Weaviate holds the knowledge base (the app indexes it on startup). Reports through /api/v1/status.
checkTravelAdvisorReady() {
  local attempt=0 max=36
  while :; do
    if kubectl -n "$TRAVEL_ADVISOR_NS" exec deploy/travel-advisor -- \
         python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/api/v1/status',timeout=10).status==200 else 1)" >/dev/null 2>&1; then
      printInfo "AI Travel Advisor API is answering"; return 0
    fi
    [ -z "${LAB_WAIT:-}" ] || [ "$attempt" -ge "$max" ] && break
    attempt=$((attempt + 1)); printInfo "AI Travel Advisor API not ready ($attempt/$max), sleeping 5s"; sleep 5
  done
  printError "AI Travel Advisor API is not answering"; return 1
}
waitForTravelAdvisorReady() { LAB_WAIT=1 checkTravelAdvisorReady; }

# Placeholder step check (replace with real lab checks): at least two nginx pods are Running.
# nginx is the stateless tier, so it is safe to scale; travel-advisor must stay at one replica.
checkNginxScaled() {
  local n
  if [ -n "${LAB_WAIT:-}" ]; then
    local i=0
    while [ "$i" -lt 24 ]; do
      n=$(kubectl -n "$TRAVEL_ADVISOR_NS" get pods -l app=travel-advisor-nginx --no-headers 2>/dev/null | grep -c " Running ")
      [ "$n" -ge 2 ] && break
      i=$((i + 1)); printInfo "fewer than 2 nginx pods Running ($i/24), waiting 5s"; sleep 5
    done
  fi
  n=$(kubectl -n "$TRAVEL_ADVISOR_NS" get pods -l app=travel-advisor-nginx --no-headers 2>/dev/null | grep -c " Running ")
  if [ "$n" -ge 2 ]; then printInfo "Two or more nginx pods are Running"; return 0; fi
  printError "Fewer than 2 nginx pods are Running: scale the deployment to 2 replicas, wait for the new pod, then check again"; return 1
}
