"""
Demographic Comparison Analysis
================================
Generates demographic comparison tables for different patient groups.

This script provides reusable functions to compute demographic statistics
for any given set of patient IDs, allowing validation of old results and
generation of new comparison tables.

Usage:
    python demographic_comparison.py [--group concordant|median_concordant]

Groups are defined in config.py and selected via the --group argument.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from config import (
    GROUPS, ALL_PATIENTS, ASSESSED_PATIENTS,
    get_group_config, get_remaining_patients, get_output_dir
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data file paths
DATA_DIR = Path(__file__).parent
PATIENT_INFO_FILE = DATA_DIR / "anonym_aamos00_patient_info.csv"
DAILY_QUESTIONNAIRE_FILE = DATA_DIR / "anonym_aamos00_dailyquestionnaire_dt.csv"
SMART_INHALER_FILE = DATA_DIR / "anonym_aamos00_smartinhaler_dt.csv"

# Age range to midpoint mapping (for numeric calculations)
AGE_MIDPOINTS = {
    "18-29yo": 23.5,
    "30-39yo": 34.5,
    "40-49yo": 44.5,
    "50+yo": 55.0  # Conservative estimate for 50+
}

# BMI range to midpoint mapping (based on WHO categories)
# Normal: 18.5-24.9 -> midpoint ~22
# Pre-obesity (Overweight): 25-29.9 -> midpoint ~27.5
# Obesity: 30+ -> using 32 as conservative estimate
BMI_MIDPOINTS = {
    "Normal": 22.0,
    "Pre-obesity": 27.5,
    "Obesity": 32.0
}

# Old patient group definition (kept for validation only)
OLD_OPTIMAL_PATIENTS = [294, 343, 473]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all required data files."""
    patient_info = pd.read_csv(PATIENT_INFO_FILE)
    daily_questionnaire = pd.read_csv(DAILY_QUESTIONNAIRE_FILE)
    smart_inhaler = pd.read_csv(SMART_INHALER_FILE)
    
    # Prepare date columns
    if 'date' not in daily_questionnaire.columns and 'time' in daily_questionnaire.columns:
        daily_questionnaire['date'] = pd.to_datetime(daily_questionnaire['time']).dt.date
    
    if 'date' not in smart_inhaler.columns and 'timestamp' in smart_inhaler.columns:
        smart_inhaler['date'] = pd.to_datetime(smart_inhaler['timestamp']).dt.date
    
    return patient_info, daily_questionnaire, smart_inhaler


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def convert_age_to_numeric(age_range: str) -> float:
    """Convert age range string to numeric midpoint."""
    return AGE_MIDPOINTS.get(age_range, np.nan)


def convert_bmi_to_numeric(bmi_range: str) -> float:
    """Convert BMI range string to numeric midpoint."""
    return BMI_MIDPOINTS.get(bmi_range, np.nan)


def sex_summary(df: pd.DataFrame) -> str:
    """Generate sex distribution summary."""
    if df.empty or 'sex' not in df.columns:
        return "N/A"
    
    counts = df['sex'].value_counts()
    total = counts.sum()
    
    parts = []
    for sex in ['female', 'male']:  # Consistent ordering
        if sex in counts.index:
            count = counts[sex]
            pct = count / total * 100
            parts.append(f"{sex.capitalize()}: {count} ({pct:.0f}%)")
    
    return ', '.join(parts)


def age_summary(df: pd.DataFrame, use_midpoints: bool = False) -> str:
    """Generate age summary - categorical distribution."""
    if df.empty or 'age_range' not in df.columns:
        return "N/A"
    
    if use_midpoints:
        ages = df['age_range'].map(convert_age_to_numeric).dropna()
        if len(ages) == 0:
            return "N/A"
        median = ages.median()
        iqr = ages.quantile(0.75) - ages.quantile(0.25)
        return f"{median:.0f} years ({iqr:.0f} years)"
    else:
        # Return categorical distribution (ordered)
        counts = df['age_range'].value_counts()
        total = counts.sum()
        # Order by age range
        order = ['18-29yo', '30-39yo', '40-49yo', '50+yo']
        parts = []
        for cat in order:
            if cat in counts.index:
                cnt = counts[cat]
                parts.append(f"{cat}: {cnt} ({cnt/total*100:.0f}%)")
        return ', '.join(parts)


def bmi_summary(df: pd.DataFrame, use_midpoints: bool = False) -> str:
    """Generate BMI summary - categorical distribution."""
    if df.empty or 'bmi_range' not in df.columns:
        return "N/A"
    
    if use_midpoints:
        bmis = df['bmi_range'].map(convert_bmi_to_numeric).dropna()
        if len(bmis) == 0:
            return "N/A"
        mean = bmis.mean()
        std = bmis.std()
        return f"{mean:.1f} kg/m² ({std:.1f} kg/m²)"
    else:
        # Return categorical distribution (ordered by BMI)
        counts = df['bmi_range'].value_counts()
        total = counts.sum()
        # Order by BMI category
        order = ['Normal', 'Pre-obesity', 'Obesity']
        parts = []
        for cat in order:
            if cat in counts.index:
                cnt = counts[cat]
                parts.append(f"{cat}: {cnt} ({cnt/total*100:.0f}%)")
        return ', '.join(parts)


def daily_questionnaire_puffs(df: pd.DataFrame, exclude_zeros: bool = True) -> str:
    """Calculate mean (SD) of daily questionnaire puffs."""
    if df.empty or 'daily_relief_inhaler' not in df.columns:
        return "N/A"
    
    vals = df['daily_relief_inhaler'].dropna()
    if exclude_zeros:
        vals = vals[vals != 0]
    
    if len(vals) == 0:
        return "N/A"
    
    return f"{vals.mean():.2f} ({vals.std():.2f})"


def smart_inhaler_puffs(df: pd.DataFrame, exclude_zeros: bool = True) -> str:
    """Calculate mean (SD) of smart inhaler puffs per day."""
    if df.empty:
        return "N/A"
    
    # Group by user_key and date to get daily counts
    if 'date' not in df.columns:
        return "N/A"
    
    daily = df.groupby(['user_key', 'date']).size().reset_index(name='inhaler_usage')
    vals = daily['inhaler_usage']
    
    if exclude_zeros:
        vals = vals[vals != 0]
    
    if len(vals) == 0:
        return "N/A"
    
    return f"{vals.mean():.2f} ({vals.std():.2f})"


def rcp3_summary(df: pd.DataFrame, drop_na: bool = False) -> str:
    """Calculate RCP3 score summary (mean, SD) - per-user average, then overall.
    
    Args:
        df: Daily questionnaire DataFrame
        drop_na: If True, drop rows with missing symptom data (no imputation).
                 If False, impute missing values as 0.
    """
    symptom_cols = ['daily_night_symp', 'daily_day_symp', 'daily_limit_activity']
    required_cols = symptom_cols + ['user_key']
    if df.empty or not all(c in df.columns for c in required_cols):
        return "N/A"
    
    df_copy = df.copy()
    
    if drop_na:
        # Drop rows with missing symptom data (don't impute)
        df_copy = df_copy.dropna(subset=symptom_cols)
        if df_copy.empty:
            return "N/A"
        # Calculate RCP score for each daily entry (sum of 3 binary symptoms)
        # Note: Must convert to int first - bool + bool = bool (OR), not numeric sum!
        df_copy['rcp_score'] = (
            df_copy['daily_night_symp'].astype(np.int32) + 
            df_copy['daily_day_symp'].astype(np.int32) + 
            df_copy['daily_limit_activity'].astype(np.int32)
        )
    else:
        # Impute missing values as 0
        # Note: fillna(0) + astype(int32) ensures proper numeric addition
        df_copy['rcp_score'] = (
            df_copy['daily_night_symp'].fillna(0).astype(np.int32) + 
            df_copy['daily_day_symp'].fillna(0).astype(np.int32) + 
            df_copy['daily_limit_activity'].fillna(0).astype(np.int32)
        )
    
    # Calculate average RCP3 score per user
    avg_rcp_by_user = df_copy.groupby('user_key')['rcp_score'].mean()
    
    if len(avg_rcp_by_user) == 0:
        return "N/A"
    
    # Calculate mean and SD of per-user averages
    mean_rcp = avg_rcp_by_user.mean()
    std_rcp = avg_rcp_by_user.std()
    
    return f"{mean_rcp:.1f} ({std_rcp:.1f})"


# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def compute_group_statistics(
    patient_ids: List[int],
    patient_info: pd.DataFrame,
    daily_questionnaire: pd.DataFrame,
    smart_inhaler: pd.DataFrame,
    use_age_bmi_midpoints: bool = False
) -> Dict[str, str]:
    """
    Compute all demographic statistics for a given group of patients.
    
    Args:
        patient_ids: List of patient IDs to include
        patient_info: Patient demographics DataFrame
        daily_questionnaire: Daily questionnaire DataFrame
        smart_inhaler: Smart inhaler DataFrame
        use_age_bmi_midpoints: If True, convert ranges to numeric midpoints
    
    Returns:
        Dictionary with all demographic statistics
    """
    # Filter data for this group
    pi_group = patient_info[patient_info['user_key'].isin(patient_ids)]
    dq_group = daily_questionnaire[daily_questionnaire['user_key'].isin(patient_ids)]
    si_group = smart_inhaler[smart_inhaler['user_key'].isin(patient_ids)]
    
    return {
        'n': len(pi_group),
        'sex': sex_summary(pi_group),
        'age': age_summary(pi_group, use_midpoints=use_age_bmi_midpoints),
        'bmi': bmi_summary(pi_group, use_midpoints=use_age_bmi_midpoints),
        'questionnaire_puffs': daily_questionnaire_puffs(dq_group),
        'inhaler_puffs': smart_inhaler_puffs(si_group),
        'rcp3_imputed': rcp3_summary(dq_group, drop_na=False),
        'rcp3_dropna': rcp3_summary(dq_group, drop_na=True)
    }


def generate_comparison_table(
    groups: Dict[str, List[int]],
    patient_info: pd.DataFrame,
    daily_questionnaire: pd.DataFrame,
    smart_inhaler: pd.DataFrame,
    use_age_bmi_midpoints: bool = False
) -> pd.DataFrame:
    """
    Generate a comparison table for multiple patient groups.
    
    Args:
        groups: Dictionary mapping group names to lists of patient IDs
        patient_info: Patient demographics DataFrame
        daily_questionnaire: Daily questionnaire DataFrame
        smart_inhaler: Smart inhaler DataFrame
        use_age_bmi_midpoints: If True, convert ranges to numeric midpoints
    
    Returns:
        DataFrame with comparison statistics
    """
    results = {}
    
    for group_name, patient_ids in groups.items():
        stats = compute_group_statistics(
            patient_ids, patient_info, daily_questionnaire, smart_inhaler,
            use_age_bmi_midpoints
        )
        results[group_name] = stats
    
    # Convert to DataFrame
    df = pd.DataFrame(results).T
    df.index.name = 'Group'
    
    return df


def print_comparison_table(df: pd.DataFrame, title: str = "Demographic Comparison"):
    """Pretty print a comparison table."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    
    # Rename columns for display
    display_names = {
        'n': 'N',
        'sex': 'Sex, n (%)',
        'age': 'Age range, n (%)',
        'bmi': 'BMI category, n (%)',
        'questionnaire_puffs': 'Questionnaire puffs/day, mean (SD)',
        'inhaler_puffs': 'Inhaler puffs/day, mean (SD)',
        'rcp3_imputed': 'RCP3 (missing=0), mean (SD)',
        'rcp3_dropna': 'RCP3 (drop missing), mean (SD)'
    }
    
    for col in df.columns:
        display_name = display_names.get(col, col)
        print(f"\n{display_name}:")
        for group in df.index:
            print(f"  {group}: {df.loc[group, col]}")
    
    print("\n" + "=" * 80)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Demographic comparison analysis.")
    parser.add_argument("--group", choices=list(GROUPS.keys()), default="concordant",
                        help="Patient group definition to use (default: concordant)")
    return parser.parse_args()


def main():
    args = parse_args()
    group_cfg = get_group_config(args.group)
    concordant_patients = group_cfg["patients"]
    group_label = group_cfg["label"]
    remaining_patients = get_remaining_patients(args.group)

    print(f"Group: {group_label} (n={len(concordant_patients)})")
    print(f"Remaining: n={len(remaining_patients)}")

    print("Loading data...")
    patient_info, daily_questionnaire, smart_inhaler = load_data()

    print(f"Loaded {len(patient_info)} patients")
    print(f"Loaded {len(daily_questionnaire)} daily questionnaire entries")
    print(f"Loaded {len(smart_inhaler)} smart inhaler entries")

    # =========================================================================
    # Comparison table with config-driven groups
    # =========================================================================
    n_conc = len(concordant_patients)
    n_rem = len(remaining_patients)

    groups = {
        'Overall (n=22)': ALL_PATIENTS,
        'Assessed (n=15)': ASSESSED_PATIENTS,
        f'{group_label} (n={n_conc})': concordant_patients,
        f'Remaining (n={n_rem})': remaining_patients
    }

    table = generate_comparison_table(
        groups, patient_info, daily_questionnaire, smart_inhaler
    )
    print_comparison_table(table, f"Table ({group_label} vs Remaining)")

    # =========================================================================
    # Save results
    # =========================================================================
    output_dir = get_output_dir(args.group, "demographic_comparison")

    table.to_csv(output_dir / "comparison_table.csv")

    print(f"\nResults saved to: {output_dir}")
    print("  - comparison_table.csv")

    # =========================================================================
    # Print raw values for easy copy-paste
    # =========================================================================
    print("\n" + "#" * 80)
    print("# RAW VALUES FOR COPY-PASTE INTO DISSERTATION")
    print("#" * 80)

    characteristics = [
        ('Sex, n (%)', 'sex'),
        ('Age range, n (%)', 'age'),
        ('BMI category, n (%)', 'bmi'),
        ('Questionnaire puffs/day (excl. zeros), mean (SD)', 'questionnaire_puffs'),
        ('Smart inhaler puffs/day (excl. zeros), mean (SD)', 'inhaler_puffs'),
        ('RCP3 (missing=0), mean (SD)', 'rcp3_imputed'),
        ('RCP3 (drop missing), mean (SD)', 'rcp3_dropna')
    ]

    group_keys = list(groups.keys())
    header = f"{'Characteristic':<55}"
    for gk in group_keys:
        header += f" | {gk:^15}"
    print(header)
    print("-" * (55 + 18 * len(group_keys)))

    for display_name, col in characteristics:
        row = f"{display_name:<55}"
        for gk in group_keys:
            row += f" | {table.loc[gk, col]:^15}"
        print(row)


if __name__ == "__main__":
    main()

