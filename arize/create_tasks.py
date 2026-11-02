#!/usr/bin/env python3
"""Create the workshop's six live evaluation tasks on an Arize tracing project with the `ax` CLI, then verify each one.

  python3 create_tasks.py --space <space id> --project <project name or id> [--suffix " (test)"] [--no-continuous]
                          [--skip task_completion,rag_relevancy] [--dry-run]

The evaluators must exist first (create_evaluators.py). Every task is ONE filter string plus `column_mappings` (each template
variable -> an attribute path of the span it scores); the SDK is not needed. See config/tasks.json.
Continuous tasks only score data that arrives after they exist; to score earlier traces use
`ax tasks trigger-run <task id> --data-start-time <ISO UTC> --data-end-time <ISO UTC>`.
"""
import argparse, json, os, pathlib, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent


def ax_json(args):
    r = subprocess.run(["ax"] + args, capture_output=True, text=True)
    try:
        return json.loads(r.stdout[r.stdout.find("{"):])
    except ValueError:
        sys.exit("ax %s failed: %s" % (" ".join(args[:3]), (r.stdout + r.stderr).strip()[:300]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--space", default=os.getenv("ARIZE_SPACE_ID"), help="space ID (or env ARIZE_SPACE_ID)")
    p.add_argument("--project", default=os.getenv("ARIZE_PROJECT"), help="tracing project name or ID (or env ARIZE_PROJECT): the project your app sends its traces to")
    p.add_argument("--suffix", default="", help="appended to task and evaluator names (must match create_evaluators.py)")
    p.add_argument("--no-continuous", dest="continuous", action="store_false", help="create the tasks switched off (for tests)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip", default="", help="comma-separated EVALUATOR names whose task to leave out")
    a = p.parse_args()
    if not (a.space and a.project):
        sys.exit("--space and --project are required (or ARIZE_SPACE_ID and ARIZE_PROJECT)")
    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    ids = {}
    if not a.dry_run:
        ids = {e["name"]: e["id"] for e in ax_json(["evaluators", "list", "--space", a.space, "--limit", "100", "-o", "json"])["evaluators"]}
    created, bad = {}, 0
    for t in json.load(open(HERE / "config" / "tasks.json"))["tasks"]:
        if t["evaluator"] in skip:
            continue
        name, ev_name = t["name"] + a.suffix, t["evaluator"] + a.suffix
        print("\n" + name)
        print("  filter:", t["query_filter"])
        for k, v in t["column_mappings"].items():
            print("   map", k, "<-", v)
        if a.dry_run:
            continue
        if ev_name not in ids:
            sys.exit("evaluator %r not found in the space: run create_evaluators.py first" % ev_name)
        cmd = ["ax", "tasks", "create-evaluation", "--name", name, "--task-type", "TEMPLATE_EVALUATION",
               "--project", a.project, "--space", a.space, "--sampling-rate", str(t["sampling_rate"]),
               "--is-continuous" if a.continuous else "--no-continuous", "--query-filter", t["query_filter"],
               "--evaluators", json.dumps([{"evaluator_id": ids[ev_name], "column_mappings": t["column_mappings"]}])]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit("create failed: " + (r.stdout + r.stderr).strip()[:400])
        tid = json.loads(r.stdout[r.stdout.find("{"):])["id"]
        got = ax_json(["tasks", "get", tid, "-o", "json"])
        problems = [m for m, ok in [("sampling", got.get("sampling_rate") == t["sampling_rate"]),
                                    ("continuous", bool(got.get("is_continuous")) == a.continuous),
                                    ("query_filter", got.get("query_filter") == t["query_filter"]),
                                    ("column_mappings", got["evaluators"][0].get("column_mappings") == t["column_mappings"])] if not ok]
        print("  created", tid, "| read back:", "OK" if not problems else "MISMATCH " + ", ".join(problems))
        created[name] = tid
        bad += bool(problems)
    if created:
        json.dump(created, open(HERE / "last-created-tasks.json", "w"), indent=1)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
