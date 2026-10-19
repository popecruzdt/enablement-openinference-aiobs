# 2. Scale the Web Tier

<!-- TODO(starter): this page is the example step. Replace it with your first real task and copy
     it once per further step (3-<slug>.md, 4-<slug>.md, …), adding each to the nav in mkdocs.yaml.
     Every step has the same rhythm: TELL why, DO one action, CHECK it, SOLVE it. -->

One replica of an application is one failure away from an outage. Kubernetes keeps as many
copies of a pod running as the deployment asks for, so scaling out is a single command.

Scale the nginx web tier of the AI Travel Advisor to two replicas. Only nginx is stateless: the Python app must stay at one replica because its embedding-model override is per process.

```bash
kubectl scale deployment travel-advisor-nginx -n ai-travel-advisor --replicas=2
```

Then watch the second pod start:

```bash
kubectl get pods -n ai-travel-advisor -l app=travel-advisor-nginx
```

## Validation

<!-- A shell check calls a function by name: the platform loads the framework and
     .devcontainer/util/my_functions.sh first. checkNginxScaled is defined there.
     Run it once BEFORE the step's action and confirm it fails — a check that cannot fail
     proves nothing. -->
<!-- LAB_QUESTION
type: shell-verification
question: "Verify nginx runs two replicas"
buttonText: "Check replicas"
command: "checkNginxScaled"
expect:
  operator: exit-zero
hint: "Run the kubectl scale command above, wait until both pods show Running, then check again."
explanation: "Two nginx pods are Running: one can fail and the web tier stays up."
-->

<!-- A multiple-choice question checks understanding. `correct` is the 0-based index of the right
     option: double-check it, a wrong index strands the learner. -->
<!-- LAB_QUESTION
type: multiple-choice
question: "What does `kubectl scale deployment travel-advisor-nginx --replicas=2` change?"
options:
  - "The number of nginx pods the deployment keeps running"
  - "The CPU and memory each nginx pod may use"
  - "The number of nodes in the cluster"
correct: 0
explanation: "A deployment's replica count is how many identical pods Kubernetes keeps running. Resources and nodes are set elsewhere."
-->

<!-- TODO(starter): a step that changes what Dynatrace sees gets a dql-verification check too.
     Scope it to the learner with endsWith(k8s.cluster.name, "{{DT_SESSION_ID}}") and bound it in
     time with from:now()-15m. Copy the example from the handbook's Interactive Blocks page:
     https://dynatrace-wwse.github.io/enablement-app-training-template/04-interactive-blocks/ -->

<!-- The solution: `commands` run by Run solution, the nightly training test and resume;
     `verify` must exit 0 afterwards and is the same check the learner clicks. -->
<!-- LAB_SOLUTION
reveal: |
  Scale the deployment to two replicas: `kubectl scale deployment travel-advisor-nginx -n ai-travel-advisor --replicas=2`.
commands:
  - kubectl scale deployment travel-advisor-nginx -n ai-travel-advisor --replicas=2
verify:
  - LAB_WAIT=1 checkNginxScaled
-->

<!-- The scored assessment goes on the LAST hands-on step, once. Its id must equal the file name
     .assessment/<id>.json and the `id` inside it. -->
<!-- LAB_QUESTIONAIRE: starter-quiz retake=true -->

<!-- Review checklist for this step (contract C7; the format is explained in docs/index.md).
     When you copy this page for a new step, delete these lines: an id may appear once per repo. -->
<!-- checklist [ ] gen-assessment: `.assessment/<id>.json` parses and its id matches the `LAB_QUESTIONAIRE` line -->
<!-- checklist [ ] author-assessment: The assessment is on the last step, its questions reviewed -->
