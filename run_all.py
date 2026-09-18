"""
Master runner script for all comparison analyses.

Runs each analysis script for each group definition (concordant and median_concordant).

Usage:
    python run_all.py                       # Run all analyses for all groups (baseline world)
    python run_all.py --group concordant    # Run only the concordant (primary) analysis
    python run_all.py --skip-sentiment      # Skip sentiment/clustering (requires OpenAI API)
    python run_all.py --world span=D__case=C   # Use another world's concordant sets and output folder
    python run_all.py --changed-worlds         # Scope guard: only worlds whose primary set differs from the baseline
"""

import argparse
import subprocess
import sys
from config import GROUPS, add_world_argument, set_active_world

COMPARISON_SCRIPTS = [
    "demographic_comparison.py",
    "temporal_correlation_analysis_v2.py",
    "bland_altman_comparison.py",
]

SENTIMENT_SCRIPTS = [
    "sentiment_analysis.py",
    "inductive_clustering_analysis.py",
]


def run_script(script: str, group: str, world: str, summary: str | None) -> bool:
    """Run a single analysis script for a given group, world and summary spec. Returns True on success."""
    cmd = [sys.executable, script, "--group", group, "--world", world]
    if summary:
        cmd += ["--summary", summary]
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd[1:])}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"WARNING: {script} exited with code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Run all comparison analyses.")
    parser.add_argument("--group", choices=list(GROUPS.keys()),
                        help="Run only this group (default: run all groups)")
    parser.add_argument("--skip-sentiment", action="store_true",
                        help="Skip sentiment and clustering analyses (require OpenAI API)")
    add_world_argument(parser)
    parser.add_argument("--changed-worlds", action="store_true",
                        help="Scope guard (v2 item 18): instead of --world, run for every world listed in "
                             "results/v2/sensitivity/worlds_to_rerun.json, i.e. those whose primary concordant set "
                             "differs from the baseline's. Worlds with an empty group are reported and skipped.")
    args = parser.parse_args()

    if args.changed_worlds:
        import json
        from aamos_concordance.worlds import V2_DIR
        listing = json.loads((V2_DIR / "sensitivity" / "worlds_to_rerun.json").read_text())
        worlds = listing["rerun"]
        print(f"Scope guard: {len(worlds)} world(s) differ from {listing['baseline']} at {listing['spec']}, t={listing['threshold']} "
              f"with a non-empty set: {worlds}")
        if listing.get("empty"):
            print(f"  differ but empty (nothing to compare): {listing['empty']}")
        if listing.get("case_b_flagged"):
            print(f"  case B spans flagged for a decision: {listing['case_b_flagged']}")
    else:
        worlds = [args.world]

    scripts = COMPARISON_SCRIPTS + ([] if args.skip_sentiment else SENTIMENT_SCRIPTS)
    results = {}
    skipped = []
    for world in worlds:
        set_active_world(world, args.summary)
        groups = [args.group] if args.group else list(GROUPS.keys())
        for group in groups:
            if not GROUPS[group]["patients"]:
                skipped.append(f"{world} / {group} (empty group)")
                continue
            group_label = GROUPS[group]["label"]
            print(f"\n{'#'*60}")
            print(f"# WORLD: {world}   GROUP: {group_label} ({group})")
            print(f"# Patients: {GROUPS[group]['patients']}")
            print(f"{'#'*60}")
            for script in scripts:
                key = f"{script} --group {group} --world {world}"
                results[key] = run_script(script, group, world, args.summary)

    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for key, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  [{status}] {key}")
    for item in skipped:
        print(f"  [SKIPPED] {item}")

    n_failed = sum(1 for v in results.values() if not v)
    if n_failed > 0:
        print(f"\n{n_failed} script(s) failed. Check output above for details.")
        sys.exit(1)
    else:
        print(f"\nAll {len(results)} analyses completed successfully.")


if __name__ == "__main__":
    main()
