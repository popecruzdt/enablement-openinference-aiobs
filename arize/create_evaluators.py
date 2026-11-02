#!/usr/bin/env python3
"""Create the workshop's Arize AX evaluators with the `ax` CLI.

Six LLM-judge evaluators (the pre-built templates' text, in config/templates/) and the code evaluator `key-facts-coverage`.

  python3 create_evaluators.py [--space <space id>] --ai-integration-id <id> --model-name <deployment/model> \
      [--provider-params '<json>'] [--suffix " (test)"] [--skip a,b] [--dry-run]

The judge is an AI integration that already exists in your Arize account (Settings > AI Integrations; `ax ai-integrations list`).
The settings can also come from the environment: ARIZE_SPACE_ID, ARIZE_AI_INTEGRATION_ID, ARIZE_JUDGE_MODEL, ARIZE_JUDGE_PROVIDER_PARAMS.
Azure OpenAI example for --provider-params:
  '{"azure_params": {"azure_deployment_name": "<deployment>", "azure_openai_endpoint": "https://<resource>.openai.azure.com", "azure_openai_version": "2025-04-01-preview"}}'
Evaluator names are unique within a space: run once per space, or use --suffix for a test copy.
"""
import argparse, json, os, pathlib, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
CFG = HERE / "config"


def run(cmd, dry):
    shown = " ".join(c if len(c) < 70 else c[:60] + "...<%d chars>" % len(c) for c in cmd)
    print("$", shown)
    if dry:
        return True
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("  FAILED:", (r.stdout + r.stderr).strip()[:400])
        return False
    try:
        d = json.loads(r.stdout[r.stdout.find("{"):])
        print("  created:", d.get("name"), d.get("id"))
    except ValueError:
        print("  ok")
    return True


def template_cmd(e, a):
    cmd = ["ax", "evaluators", "create-evaluator", "template",
           "--name", e["name"] + a.suffix, "--space", a.space,
           "--commit-message", e.get("commit_message") or "Pre-built %s template" % e["name"],
           "--template-name", e["template_name"] + a.suffix,
           "--template", (CFG / e["template_file"]).read_text(),  # --template takes the text inline (it does not read @file)
           "--ai-integration-id", a.ai_integration_id, "--model-name", a.model_name,
           "--classification-choices", json.dumps(e["classification_choices"]),
           "--direction", e["direction"], "--data-granularity", e["data_granularity"]]
    if a.provider_params:
        cmd += ["--provider-params", a.provider_params]
    if e["include_explanations"]:
        cmd.append("--include-explanations")
    if e["use_function_calling"]:
        cmd.append("--use-function-calling")
    if e.get("description"):
        cmd += ["--description", e["description"]]
    return cmd


def code_cmd(a):
    return ["ax", "evaluators", "create-evaluator", "code",
            "--name", "key-facts-coverage" + a.suffix, "--space", a.space,
            "--commit-message", "Share of key facts found in the answer; rows without key facts are not applicable",
            "--code-type", "custom", "--code-name", "key_facts_coverage" + "".join(c if c.isalnum() else "_" for c in a.suffix),
            "--variables", '["output","key_facts"]',
            "--code", "@" + str(HERE / "evaluators" / "key-facts-coverage.py"),
            "--imports", "@" + str(HERE / "evaluators" / "key-facts-imports.py"),
            "--data-granularity", "span",
            "--description", "Share of the row's key_facts (semicolon-separated, A OR B alternatives) found in the answer. Rows without key facts are not applicable."]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--space", default=os.getenv("ARIZE_SPACE_ID"), help="space ID (or env ARIZE_SPACE_ID); use the ID: two spaces can look alike")
    p.add_argument("--ai-integration-id", default=os.getenv("ARIZE_AI_INTEGRATION_ID"))
    p.add_argument("--model-name", default=os.getenv("ARIZE_JUDGE_MODEL"))
    p.add_argument("--provider-params", default=os.getenv("ARIZE_JUDGE_PROVIDER_PARAMS"), help="JSON for the provider (see above)")
    p.add_argument("--suffix", default="")
    p.add_argument("--skip", default="", help="comma-separated evaluator names to leave out (key-facts-coverage is the code evaluator)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if not a.space:
        sys.exit("--space (or ARIZE_SPACE_ID) is required")
    if not a.dry_run and not (a.ai_integration_id and a.model_name):
        sys.exit("--ai-integration-id and --model-name are required (or ARIZE_AI_INTEGRATION_ID / ARIZE_JUDGE_MODEL)")
    a.ai_integration_id = a.ai_integration_id or "<ai integration id>"
    a.model_name = a.model_name or "<judge model>"
    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    ok = True
    for e in json.load(open(CFG / "evaluators.json"))["evaluators"]:
        if e["name"] not in skip:
            ok &= run(template_cmd(e, a), a.dry_run)
    if "key-facts-coverage" not in skip:
        ok &= run(code_cmd(a), a.dry_run)
    print("done" if ok else "finished with failures")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
