# 1. Check the Environment

<!-- TODO(starter): check ONLY what .devcontainer/post-create.sh builds. This step never asks the
     learner to change anything, so it carries LAB_NO_SOLUTION, never a LAB_SOLUTION. Keep one
     check per thing post-create.sh deploys; remove the ones your training does not deploy. -->

Your environment was built when you pressed **Start**: a Kubernetes cluster with a small demo
application, the AI Travel Advisor (nginx, the Python app, Ollama, Weaviate and an OpenTelemetry Collector). Before changing anything, confirm both are up.

## Validation

<!-- LAB_QUESTION
type: shell-verification
question: "Verify the cluster node is Ready"
buttonText: "Check Cluster"
command: "checkNodeReady"
expect:
  operator: exit-zero
hint: "The cluster is still starting. Wait a minute, then check again."
explanation: "The cluster node is Ready."
-->

<!-- LAB_QUESTION
type: shell-verification
question: "Verify the AI Travel Advisor is Running"
buttonText: "Check the app"
command: "checkTravelAdvisorRunning"
expect:
  operator: exit-zero
hint: "The app deploys after the cluster. Wait a minute, then check again."
explanation: "All five components are Running in the ai-travel-advisor namespace."
-->

<!-- LAB_NO_SOLUTION: sanity check of what post-create.sh built — nothing to solve -->

<!-- Review checklist for this step (contract C7; the format is explained in docs/index.md). -->
<!-- checklist [ ] author-env-check: Step 1 checks only what `post-create.sh` builds, with `LAB_NO_SOLUTION` -->
