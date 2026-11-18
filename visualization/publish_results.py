"""
TARA-FL Results Publisher (Mode B: static GitHub Pages hosting).

Copies the real experiment JSON produced by experiments/results_export.py
into demo/results/, so the browser dashboard (demo/index.html) can display
the LATEST PUBLISHED ACTUAL RESULTS on static hosting such as GitHub Pages.

What it does
------------
dashboard_data/live.json            -> demo/results/latest.json
dashboard_data/history/<id>.json    -> demo/results/history/<id>.json
dashboard_data/index.json           -> demo/results/index.json

GitHub Pages is static hosting: it cannot execute the Python backend or
connect to a visitor's locally running experiment. Pages therefore shows the
latest *published* snapshot; the dashboard labels it accordingly. For live
round-by-round updates, run the backend locally (Mode A):
    python visualization/live_server.py

Usage (Windows CMD):
    cd C:\\path\\to\\TARA-FL
    python visualization\\publish_results.py
    (optionally: --dashboard_dir dashboard_data --demo_dir demo --experiment_id exp-...)

Then commit and push demo/results/ - GitHub Pages serves the new snapshot.
"""

import argparse
import json
import os
import shutil
import sys

DEFAULT_DASHBOARD_DIR = "dashboard_data"
DEFAULT_DEMO_DIR = "demo"


def publish(dashboard_dir: str, demo_dir: str, experiment_id: str = None) -> dict:
    results_dir = os.path.join(demo_dir, "results")
    history_out_dir = os.path.join(results_dir, "history")
    os.makedirs(history_out_dir, exist_ok=True)

    copied = []

    # 1. Latest run -> latest.json (prefer the requested experiment if given)
    live_path = os.path.join(dashboard_dir, "live.json")
    if experiment_id:
        source_path = os.path.join(dashboard_dir, "history", f"{experiment_id}.json")
    else:
        source_path = live_path
    if os.path.isfile(source_path):
        with open(source_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        latest_out = os.path.join(results_dir, "latest.json")
        with open(latest_out, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        copied.append(f"{source_path} -> {latest_out}")
        latest_id = doc.get("experiment_id")
    else:
        latest_id = None
        print(f"[publish] WARNING: {source_path} not found - nothing to publish as latest.json")

    # 2. All history documents
    history_in_dir = os.path.join(dashboard_dir, "history")
    published_ids = []
    if os.path.isdir(history_in_dir):
        for name in sorted(os.listdir(history_in_dir)):
            if not name.endswith(".json"):
                continue
            src = os.path.join(history_in_dir, name)
            dst = os.path.join(history_out_dir, name)
            shutil.copyfile(src, dst)
            published_ids.append(name[:-5])
            copied.append(f"{src} -> {dst}")

    # 3. Index: keep only experiments that were actually published
    index_in = os.path.join(dashboard_dir, "index.json")
    index_out = os.path.join(results_dir, "index.json")
    published_index = {"updated_at": None, "count": 0, "experiments": []}
    if os.path.isfile(index_in):
        with open(index_in, "r", encoding="utf-8") as f:
            index_doc = json.load(f)
        keep = set(published_ids)
        published_index = {
            "updated_at": index_doc.get("updated_at"),
            "count": 0,
            "experiments": [
                e for e in index_doc.get("experiments", [])
                if e.get("experiment_id") in keep
            ],
        }
        published_index["count"] = len(published_index["experiments"])
        if latest_id:
            published_index["latest_experiment_id"] = latest_id
    with open(index_out, "w", encoding="utf-8") as f:
        json.dump(published_index, f, indent=2, ensure_ascii=False)
    copied.append(f"{index_in} -> {index_out}")

    return {
        "copied": copied,
        "latest_experiment_id": latest_id,
        "history_count": len(published_ids),
        "index_count": published_index["count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish real experiment results to demo/results (static hosting)")
    parser.add_argument("--dashboard_dir", type=str, default=DEFAULT_DASHBOARD_DIR)
    parser.add_argument("--demo_dir", type=str, default=DEFAULT_DEMO_DIR)
    parser.add_argument("--experiment_id", type=str, default=None,
                        help="Publish a specific experiment as latest.json (default: the current live.json)")
    args = parser.parse_args()

    print("=" * 65)
    print("   TARA-FL RESULTS PUBLISHER (dashboard_data -> demo/results)")
    print("=" * 65)
    result = publish(args.dashboard_dir, args.demo_dir, args.experiment_id)

    for line in result["copied"]:
        print(f"  copied: {line}")
    print("-" * 65)
    print(f"  Latest published experiment : {result['latest_experiment_id']}")
    print(f"  History files published     : {result['history_count']}")
    print(f"  Index entries               : {result['index_count']}")
    print("-" * 65)
    print("  Next steps (GitHub Pages):")
    print("    git add demo/results")
    print('    git commit -m "Publish latest real TARA-FL experiment results"')
    print("    git push")
    print("  The GitHub Pages dashboard will now show the latest published results.")


if __name__ == "__main__":
    main()
