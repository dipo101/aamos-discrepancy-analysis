#!/usr/bin/env python3
"""
Sensitivity Analysis Runner

Runs the full concordance analysis pipeline for multiple statistical measures:
- median (primary analysis)
- mean
- q25 (25th percentile)
- q75 (75th percentile)
- min
- max

Results are saved to: results/permutation_by_measure/<measure>/

Usage:
    python run_sensitivity_analysis.py                    # Run all recommended measures
    python run_sensitivity_analysis.py --measure median   # Run specific measure
    python run_sensitivity_analysis.py --measures median mean q25  # Run multiple specific measures
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import sys
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from sensitivity_core import (
    MEASURE_CONFIG, 
    RECOMMENDED_MEASURES,
    load_observed_statistics,
    load_null_distributions,
    compute_p_values_for_measure,
    get_significant_patients,
    get_concordant_patients,
)
from sensitivity_plots import (
    load_detailed_results,
    generate_all_plots_for_measure,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# File paths (relative to aggregator directory)
SUMMARY_STATS_FILE = Path("../../results/per_patient_analysis/per_patient_summary_statistics.csv")
DETAILED_RESULTS_FILE = Path("../../results/per_patient_analysis/per_patient_correlation_results.csv")
NULL_PARQUET_FILE = Path("../../results/permutation_aggregate_extended/all_permutations.parquet")

# Output directory
OUTPUT_BASE_DIR = Path("../../results/permutation_by_measure")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def run_analysis_for_measure(measure: str, detailed_df: pd.DataFrame, null_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full analysis pipeline for a single measure.
    
    Args:
        measure: Statistical measure ('median', 'mean', 'q25', 'q75', 'min', 'max')
        detailed_df: Detailed per-configuration correlation results
        null_df: Full null distribution data
    
    Returns:
        DataFrame with p-values and significance results
    """
    config = MEASURE_CONFIG[measure]
    
    logger.info(f"\n{'='*80}")
    logger.info(f"RUNNING ANALYSIS FOR: {config['display_name']}")
    logger.info(f"{'='*80}")
    
    # Create output directory
    output_dir = OUTPUT_BASE_DIR / measure
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load observed statistics
    logger.info(f"Loading observed {config['display_name']} values...")
    observed_df = load_observed_statistics(
        SUMMARY_STATS_FILE, 
        DETAILED_RESULTS_FILE, 
        measure
    )
    logger.info(f"  Loaded observed values for {len(observed_df)} patients")
    
    # 2. Load null distribution for this measure
    logger.info(f"Loading null distribution...")
    null_measure_df = load_null_distributions(NULL_PARQUET_FILE, measure)
    logger.info(f"  Loaded {len(null_measure_df)} null permutation results")
    
    # 3. Compute p-values
    logger.info(f"Computing p-values...")
    results_df = compute_p_values_for_measure(observed_df, null_measure_df, measure)
    
    # 4. Save results CSV
    results_file = output_dir / 'permutation_test_results.csv'
    results_df.to_csv(results_file, index=False)
    logger.info(f"  Saved results to: {results_file}")
    
    # 5. Generate all plots
    generate_all_plots_for_measure(
        detailed_df=detailed_df,
        sig_df=results_df,
        null_df=null_df,
        output_dir=output_dir,
        measure=measure
    )
    
    # 6. Print summary
    n_sig_bonf = results_df['significant_bonferroni'].sum()
    n_sig_nom = results_df['significant_uncorrected'].sum()
    n_optimal = len(get_concordant_patients(results_df, measure))
    
    logger.info(f"\n SUMMARY for {config['display_name']}:")
    logger.info(f"   Total patients: {len(results_df)}")
    logger.info(f"   Bonferroni significant: {n_sig_bonf}")
    logger.info(f"   Nominally significant: {n_sig_nom}")
    logger.info(f"   Optimal patients (Z≥0.5 & significant): {n_optimal}")
    
    sig_patients = get_significant_patients(results_df, bonferroni=True)
    if sig_patients:
        logger.info(f"   Significant patient IDs: {sig_patients}")
    
    return results_df


def generate_comparison_summary(all_results: dict, output_dir: Path):
    """
    Generate a comparison summary across all measures.
    """
    logger.info(f"\n{'='*80}")
    logger.info("GENERATING CROSS-MEASURE COMPARISON")
    logger.info(f"{'='*80}")
    
    # Build comparison table
    rows = []
    for measure, results_df in all_results.items():
        config = MEASURE_CONFIG[measure]
        
        n_sig_bonf = results_df['significant_bonferroni'].sum()
        n_sig_nom = results_df['significant_uncorrected'].sum()
        sig_patients = get_significant_patients(results_df, bonferroni=True)
        optimal_patients = get_concordant_patients(results_df, measure)
        
        rows.append({
            'Measure': config['display_name'],
            'N Bonferroni Sig.': n_sig_bonf,
            'N Nominal Sig.': n_sig_nom,
            'Significant Patient IDs': ', '.join(sig_patients) if sig_patients else 'None',
            'N Optimal': len(optimal_patients),
            'Optimal Patient IDs': ', '.join(optimal_patients) if optimal_patients else 'None'
        })
    
    comparison_df = pd.DataFrame(rows)
    
    # Save CSV
    comparison_file = output_dir / 'measure_comparison.csv'
    comparison_df.to_csv(comparison_file, index=False)
    logger.info(f"Saved comparison to: {comparison_file}")
    
    # Print summary
    print("\n" + "="*100)
    print("SENSITIVITY ANALYSIS SUMMARY: Comparison Across Measures")
    print("="*100)
    print(comparison_df.to_string(index=False))
    print("="*100 + "\n")
    
    # Check consistency
    if len(all_results) > 1:
        primary = list(all_results.values())[0]
        primary_sig = set(get_significant_patients(primary, bonferroni=True))
        
        print("CONSISTENCY CHECK:")
        for measure, results_df in all_results.items():
            if measure == list(all_results.keys())[0]:
                continue
            other_sig = set(get_significant_patients(results_df, bonferroni=True))
            
            both = primary_sig & other_sig
            only_primary = primary_sig - other_sig
            only_other = other_sig - primary_sig
            
            config = MEASURE_CONFIG[measure]
            print(f"\n  {MEASURE_CONFIG[list(all_results.keys())[0]]['display_name']} vs {config['display_name']}:")
            print(f"    Both significant: {sorted(both) if both else 'None'}")
            print(f"    Only in primary: {sorted(only_primary) if only_primary else 'None'}")
            print(f"    Only in {config['display_name']}: {sorted(only_other) if only_other else 'None'}")
    
    return comparison_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run sensitivity analysis across different summary statistics'
    )
    parser.add_argument(
        '--measure', 
        type=str, 
        help='Single measure to analyze'
    )
    parser.add_argument(
        '--measures', 
        nargs='+', 
        type=str, 
        help='Multiple measures to analyze'
    )
    parser.add_argument(
        '--all', 
        action='store_true',
        help='Run all available measures (not just recommended)'
    )
    
    args = parser.parse_args()
    
    # Determine which measures to run
    if args.measure:
        measures = [args.measure]
    elif args.measures:
        measures = args.measures
    elif args.all:
        measures = list(MEASURE_CONFIG.keys())
    else:
        measures = RECOMMENDED_MEASURES
    
    # Validate measures
    for m in measures:
        if m not in MEASURE_CONFIG:
            logger.error(f"Unknown measure: {m}. Choose from: {list(MEASURE_CONFIG.keys())}")
            sys.exit(1)
    
    logger.info("="*80)
    logger.info("SENSITIVITY ANALYSIS - MULTIVERSE CONCORDANCE STUDY")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    logger.info(f"Measures to analyze: {measures}")
    
    # Load shared data (only once)
    logger.info("\n Loading data files...")
    
    if not DETAILED_RESULTS_FILE.exists():
        logger.error(f"Detailed results not found: {DETAILED_RESULTS_FILE}")
        sys.exit(1)
    
    if not NULL_PARQUET_FILE.exists():
        logger.error(f"Null distribution file not found: {NULL_PARQUET_FILE}")
        logger.info("Please run the aggregator first: python aggregate.py")
        sys.exit(1)
    
    detailed_df = load_detailed_results(DETAILED_RESULTS_FILE)
    detailed_df['user_key'] = detailed_df['user_key'].astype(str)
    logger.info(f"  Loaded detailed results: {len(detailed_df)} rows, {detailed_df['user_key'].nunique()} patients")
    
    null_df = pd.read_parquet(NULL_PARQUET_FILE)
    null_df['patient_id'] = null_df['patient_id'].astype(str)
    logger.info(f"  Loaded null distributions: {len(null_df)} rows, {null_df['patient_id'].nunique()} patients")
    
    # Create output base directory
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run analysis for each measure
    all_results = {}
    
    for measure in measures:
        try:
            results_df = run_analysis_for_measure(measure, detailed_df, null_df)
            all_results[measure] = results_df
        except Exception as e:
            logger.error(f"Failed to analyze {measure}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate comparison summary
    if len(all_results) > 0:
        generate_comparison_summary(all_results, OUTPUT_BASE_DIR)
    
    logger.info("\n" + "="*80)
    logger.info("SENSITIVITY ANALYSIS COMPLETE")
    logger.info(f"Results saved to: {OUTPUT_BASE_DIR.absolute()}")
    logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)


if __name__ == '__main__':
    main()

