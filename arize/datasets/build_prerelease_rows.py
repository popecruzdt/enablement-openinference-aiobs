#!/usr/bin/env python3
"""Build the rows for the pre-release check (Experiment 4.2): what the answer model will be given once a page exists.

For each destination it builds the `tool_results` text in the app's own format (validate + search + season), from a
page file, 3 samples each (temperature 0.7 is noisy). Used with arize/experiments/prerelease_task.py, which calls the answer model
directly with the deployed respond prompt: no traffic to the app, no change to the cluster.

  python3 build_prerelease_rows.py [--sf-page PATH] > prerelease-rows.csv

The San Francisco page is not part of the repo until the workshop's Improve step adds it: pass the draft with --sf-page
(default: app/destinations/san-francisco.html).
"""
import csv, html, re, sys, datetime, os

REPO_PAGES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "app", "destinations") + os.sep
SF_PAGE = REPO_PAGES + "san-francisco.html"
# (input, slug, page file, key facts: the office address; the restaurant)
DESTS = [
    ("San Francisco", "san-francisco", SF_PAGE, "260 California St; Schroeder's"),
    ("Boston", "boston", REPO_PAGES + "boston.html", "280 Congress; Row 34"),
]


def page_text(path):
    s = open(path, encoding="utf-8").read()
    h1 = re.search(r"<h1>(.*?)</h1>", s, re.S).group(1).strip()
    ps = [html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", p)).strip()) for p in re.findall(r"<p>(.*?)</p>", s, re.S)]
    return h1 + "\n" + "\n\n".join(ps)


def season(today):  # northern hemisphere, as the app's get_current_season
    return {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}[today.month]


def main():
    global DESTS
    if "--sf-page" in sys.argv:
        DESTS = [(n, sl, sys.argv[sys.argv.index("--sf-page") + 1] if sl == "san-francisco" else p, f) for n, sl, p, f in DESTS]
    today = datetime.date.today()
    w = csv.writer(sys.stdout)
    w.writerow(["case_id", "input", "tool_results", "key_facts", "category"])
    for name, slug, path, facts in DESTS:
        tr = (f"[validate_destination]\nVALID: '{name}' is in the knowledge base as '{slug}'.\n\n"
              f"[search_destination_kb]\n{page_text(path)}\n\n"
              f"[get_current_season]\nToday is {today.isoformat()}. It is {season(today)} in {name} (northern hemisphere).")
        for i in (1, 2, 3):
            w.writerow([f"{slug}-s{i}", name, tr, facts, "draft-page" if slug == "san-francisco" else "regression"])


if __name__ == "__main__":
    main()
