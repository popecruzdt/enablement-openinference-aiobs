#!/usr/bin/env python3
"""Create the workshop's annotation configs and labeling queues with the `ax` CLI (config/annotations.json).

  python3 create_annotations.py --space <space id> --annotator-email <your Arize login> [--record-sources FILE] [--dry-run]

The categorical configs are created with --optimization-direction NONE: with a direction the API demands scores, which the CLI
cannot set. Comparing human and judge labels works by label, so scores are not needed.
The queues are created empty unless --record-sources is given (a JSON file of span record sources, see `ax annotation-queues create
--help`); otherwise add records in the UI (Traces table > select > Send to Labeling Queue). `--annotator-email` must be the email
of an Arize user in the space. The `notes` (freeform) config exists in every space already.
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
    p.add_argument("--annotator-email", default=os.environ.get("ARIZE_ANNOTATOR_EMAIL"), help="your Arize login email (or env ARIZE_ANNOTATOR_EMAIL)")
    p.add_argument("--record-sources", help="JSON file with initial record sources (applied to every queue)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if not a.space:
        sys.exit("--space (or ARIZE_SPACE_ID) is required")
    if not a.annotator_email:
        sys.exit("--annotator-email (or ARIZE_ANNOTATOR_EMAIL) is required")
    cfg = json.load(open(HERE / "config" / "annotations.json"))
    for c in cfg["annotation_configs"]:
        cmd = ["ax", "annotation-configs", "create", "categorical", "--name", c["name"], "--space", a.space,
               "--optimization-direction", "NONE", "-o", "json"]
        for v in c["values"]:
            cmd += ["--value", v]
        print("$", " ".join(cmd))
        if not a.dry_run:
            r = subprocess.run(cmd, capture_output=True, text=True)
            print("  ", "created" if r.returncode == 0 else "FAILED: " + (r.stdout + r.stderr).strip()[:300])
    ids = {} if a.dry_run else {c["name"]: c["id"] for c in ax_json(["annotation-configs", "list", "--space", a.space, "--limit", "100", "-o", "json"])["annotation_configs"]}
    for q in cfg["annotation_queues"]:
        cmd = ["ax", "annotation-queues", "create", "--name", q["name"], "--space", a.space, "--annotator-email", a.annotator_email,
               "--assignment-method", "ALL", "--instructions", q["instructions"] or "", "-o", "json"]
        for n in q["annotation_configs"]:
            cmd += ["--annotation-config-id", ids.get(n, "<%s id>" % n)]
        if a.record_sources:
            cmd += ["--record-sources", a.record_sources]
        print("$", " ".join(c if len(c) < 70 else c[:50] + "...<%d chars>" % len(c) for c in cmd))
        if not a.dry_run:
            r = subprocess.run(cmd, capture_output=True, text=True)
            print("  ", "created" if r.returncode == 0 else "FAILED: " + (r.stdout + r.stderr).strip()[:300])


if __name__ == "__main__":
    main()
