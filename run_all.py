"""
Master runner script for all comparison analyses.

Runs each analysis script for each group definition (concordant and median_concordant).

Usage:
    python run_all.py                       # Run all analyses for all groups (baseline world)
    python run_all.py --group concordant    # Run only the concordant (primary) analysis
    python run_all.py --skip-sentiment      # Skip sentiment/clustering (requires OpenAI API)
    python run_all.py --world span=D__case=C   # Use another world's concordant sets and output folder
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


def run_script(script: str, group: str, world: str) -> bool:
    """Run a single analysis script for a given group and world. Returns True on success."""
    print(f"\n{'='*60}")
    print(f"Running: {script} --group {group} --world {world}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, script, "--group", group, "--world", world],
        capture_output=False
    )
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
    args = parser.parse_args()
    set_active_world(args.world)

    groups = [args.group] if args.group else list(GROUPS.keys())
    scripts = COMPARISON_SCRIPTS + ([] if args.skip_sentiment else SENTIMENT_SCRIPTS)

    results = {}
    for group in groups:
        group_label = GROUPS[group]["label"]
        print(f"\n{'#'*60}")
        print(f"# GROUP: {group_label} ({group})")
        print(f"# Patients: {GROUPS[group]['patients']}")
        print(f"{'#'*60}")

        for script in scripts:
            key = f"{script} --group {group} --world {args.world}"
            results[key] = run_script(script, group, args.world)

    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for key, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  [{status}] {key}")

    n_failed = sum(1 for v in results.values() if not v)
    if n_failed > 0:
        print(f"\n{n_failed} script(s) failed. Check output above for details.")
        sys.exit(1)
    else:
        print(f"\nAll {len(results)} analyses completed successfully.")


if __name__ == "__main__":
    main()
