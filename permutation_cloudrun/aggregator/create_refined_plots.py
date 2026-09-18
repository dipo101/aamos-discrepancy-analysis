import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.patches as mpatches
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import CONCORDANCE_Z_THRESHOLD, GROUPS

# Measure-specific column mappings
# Column names match permutation_by_measure/{measure}/permutation_test_results.csv
MEASURE_COLUMNS = {
    "median": {
        "observed_z_col": "observed_median_z",
        "null_mean_col": "null_median_mean",
        "null_std_col": "null_median_std",
        "agg_func": "median",
        "label": "Median",
        "group_key": "median_concordant",
    },
    "mean": {
        "observed_z_col": "observed_mean_z",
        "null_mean_col": "null_mean_mean",
        "null_std_col": "null_mean_std",
        "agg_func": "mean",
        "label": "Mean",
        "group_key": "mean_concordant",
    },
}


def load_data(measure="median"):
    """Load necessary data files for the specified measure.

    Reads from:
    - results/per_patient_analysis/per_patient_correlation_results.csv (detailed per-config results)
    - results/permutation_by_measure/{measure}/permutation_test_results.csv (significance results)
    """
    col_cfg = MEASURE_COLUMNS[measure]

    # 1. Detailed per-configuration results (shared across measures)
    potential_paths = [
        Path("../../results/v1/per_patient_analysis/per_patient_correlation_results.csv"),
        Path("../results/v1/per_patient_analysis/per_patient_correlation_results.csv"),
        Path("results/v1/per_patient_analysis/per_patient_correlation_results.csv"),
    ]

    detailed_results_path = None
    for p in potential_paths:
        if p.exists():
            detailed_results_path = p
            print(f"Found detailed results at: {p}")
            break

    if not detailed_results_path:
        raise FileNotFoundError("Could not find per_patient_correlation_results.csv")

    detailed_df = pd.read_csv(detailed_results_path)

    # 2. Significance results for this measure
    sig_path = Path(f"../../results/v1/permutation_by_measure/{measure}/permutation_test_results.csv")
    if not sig_path.exists():
        raise FileNotFoundError(f"Could not find {sig_path}")
    sig_df = pd.read_csv(sig_path)
    print(f"Loaded significance results from: {sig_path}")

    # Normalize column names to standard names used by plotting functions
    sig_df["observed_z"] = sig_df[col_cfg["observed_z_col"]]
    sig_df["null_mean"] = sig_df[col_cfg["null_mean_col"]]
    sig_df["null_std"] = sig_df[col_cfg["null_std_col"]]

    # Use precomputed quantiles for null band (more precise than mean +/- 1.96*std)
    q025_col = col_cfg["null_mean_col"].replace("_mean", "_q025")
    q975_col = col_cfg["null_mean_col"].replace("_mean", "_q975")
    if q025_col in sig_df.columns and q975_col in sig_df.columns:
        sig_df["null_q025"] = sig_df[q025_col]
        sig_df["null_q975"] = sig_df[q975_col]

    return detailed_df, sig_df, col_cfg


def plot_point_with_null_band(detailed_df, sig_df, col_cfg, output_dir,
                              filter_type='all', title_suffix=''):
    """
    Plot 1: Simplified point plot with null bands.
    Shows per-patient summary statistic (median or mean) as a point.
    """
    measure_label = col_cfg["label"]
    agg_func = col_cfg["agg_func"]
    group_label = GROUPS.get(col_cfg["group_key"], {}).get("label", measure_label + "-Concordant")

    print(f"Generating {measure_label} Point Plot ({filter_type})...")

    detailed_df = detailed_df.copy()
    sig_df = sig_df.copy()
    detailed_df['user_key'] = detailed_df['user_key'].astype(str)
    sig_df['patient_id'] = sig_df['patient_id'].astype(str)

    merge_cols = ['patient_id', 'significant_bonferroni', 'significant_uncorrected', 'observed_z']
    for c in ['null_mean', 'null_std']:
        if c in sig_df.columns:
            merge_cols.append(c)

    plot_df = detailed_df.merge(
        sig_df[merge_cols],
        left_on='user_key',
        right_on='patient_id',
        how='inner'
    )

    if len(plot_df) == 0:
        print("Error: No common patients found!")
        return

    def finite_agg(x):
        finite_vals = x[np.isfinite(x)]
        if len(finite_vals) == 0:
            return np.nan
        return finite_vals.median() if agg_func == "median" else finite_vals.mean()

    patient_scores = plot_df.groupby('user_key')['spearman_z'].apply(finite_agg).sort_values()
    sorted_patients = patient_scores.index.tolist()

    if filter_type == 'significant':
        sig_patients = sig_df[sig_df['significant_bonferroni']]['patient_id'].astype(str).tolist()
        sorted_patients = [p for p in sorted_patients if p in sig_patients]
    elif filter_type == 'concordant':
        sig_patients = sig_df[sig_df['significant_bonferroni']]['patient_id'].astype(str).tolist()
        sorted_patients = [p for p in sorted_patients
                          if patient_scores[p] >= CONCORDANCE_Z_THRESHOLD and p in sig_patients]

    if len(sorted_patients) == 0:
        print(f"No patients match filter '{filter_type}'")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    colors = []
    scores = []
    null_lows = []
    null_highs = []

    for i, pid in enumerate(sorted_patients):
        patient_sig = sig_df[sig_df['patient_id'] == pid].iloc[0]
        score = patient_scores[pid]
        scores.append(score)

        if 'null_q025' in patient_sig.index and pd.notna(patient_sig['null_q025']):
            low = patient_sig['null_q025']
            high = patient_sig['null_q975']
        else:
            low = patient_sig['null_mean'] - 1.96 * patient_sig['null_std']
            high = patient_sig['null_mean'] + 1.96 * patient_sig['null_std']

        null_lows.append(low)
        null_highs.append(high)

        is_sig_strict = patient_sig['significant_bonferroni']
        is_sig_nominal = patient_sig['significant_uncorrected']

        if is_sig_strict:
            colors.append('#2E8B57' if score >= 0 else '#FF6B6B')
        elif is_sig_nominal:
            colors.append('#DAA520')
        else:
            colors.append('#808080')

    x_positions = np.arange(len(sorted_patients))

    for i, (low, high) in enumerate(zip(null_lows, null_highs)):
        ax.fill_between([i - 0.3, i + 0.3], [low, low], [high, high],
                       color='#4682B4', alpha=0.25, edgecolor=None, zorder=0)

    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.4, zorder=1)
    ax.axhline(0.5, color='#FF6B00', linestyle='--', linewidth=1.5,
               alpha=0.7, zorder=2, label='Concordance Threshold (Z = 0.5)')

    ax.scatter(x_positions, scores, c=colors, s=120, zorder=10, edgecolors='black', linewidth=0.5)

    for i, pid in enumerate(sorted_patients):
        patient_data = plot_df[plot_df['user_key'] == pid]['spearman_z']
        q25, q75 = patient_data.quantile([0.25, 0.75])
        ax.plot([i, i], [q25, q75], color=colors[i], linewidth=2, alpha=0.6, zorder=5)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(sorted_patients, rotation=45)
    ax.set_xlabel('Patient ID', fontsize=12)
    ax.set_ylabel(f"Fisher's Z-Score ({measure_label})", fontsize=12)

    title = f'Per-Patient {measure_label} Correlation with Null Band{title_suffix}'
    ax.set_title(title, fontsize=14, pad=15)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    legend_elements = [
        mpatches.Patch(color='#2E8B57', label='Significant Concordance'),
        mpatches.Patch(color='#FF6B6B', label='Significant Discordance'),
        mpatches.Patch(color='#DAA520', label='Nominally Significant'),
        mpatches.Patch(color='#808080', label='Non-Significant'),
        mpatches.Patch(facecolor='#4682B4', alpha=0.25, label='Null 95% CI'),
        plt.Line2D([0], [0], color='#FF6B00', linestyle='--', linewidth=1.5, label='Concordance Threshold (Z >= 0.5)')
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    filename = f'{agg_func}_null_band_{filter_type}.png'
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_concordance_only(detailed_df, sig_df, col_cfg, output_dir,
                         filter_type='all', title_suffix=''):
    """
    Plot 4a: Pure concordance plot - just shows Z-scores without significance colors.
    filter_type: 'all', 'significant', 'concordant', 'nominal_and_significant'
    """
    agg_func = col_cfg["agg_func"]
    measure_label = col_cfg["label"]
    print(f"Generating Concordance-Only Plot ({filter_type})...")

    detailed_df = detailed_df.copy()
    sig_df = sig_df.copy()
    detailed_df['user_key'] = detailed_df['user_key'].astype(str)
    sig_df['patient_id'] = sig_df['patient_id'].astype(str)

    def finite_agg(x):
        finite_vals = x[np.isfinite(x)]
        if len(finite_vals) == 0:
            return np.nan
        return finite_vals.median() if agg_func == "median" else finite_vals.mean()

    patient_scores = detailed_df.groupby('user_key')['spearman_z'].apply(finite_agg).sort_values()
    sorted_patients = patient_scores.index.tolist()
    sorted_patients = [p for p in sorted_patients if p in sig_df['patient_id'].values]

    if filter_type == 'significant':
        sig_patients = sig_df[sig_df['significant_bonferroni']]['patient_id'].tolist()
        sorted_patients = [p for p in sorted_patients if p in sig_patients]
    elif filter_type == 'concordant':
        sig_patients = sig_df[sig_df['significant_bonferroni']]['patient_id'].tolist()
        sorted_patients = [p for p in sorted_patients
                         if patient_scores[p] >= CONCORDANCE_Z_THRESHOLD and p in sig_patients]
    elif filter_type == 'nominal_and_significant':
        nom_or_sig = sig_df[sig_df['significant_uncorrected'] | sig_df['significant_bonferroni']]['patient_id'].tolist()
        sorted_patients = [p for p in sorted_patients if p in nom_or_sig]

    if len(sorted_patients) == 0:
        print(f"No patients match filter '{filter_type}'")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(sorted_patients) * 0.8), 6))

    x_positions = np.arange(len(sorted_patients))
    scores = [patient_scores[p] for p in sorted_patients]
    colors = ['#2E8B57' if m >= 0 else '#FF6B6B' for m in scores]

    ax.axhline(0, color='black', linestyle='-', linewidth=1.2, alpha=0.6)
    ax.bar(x_positions, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(sorted_patients, rotation=45)
    ax.set_xlabel('Patient ID', fontsize=12)
    ax.set_ylabel(f"{measure_label} Fisher's Z-Score", fontsize=12)
    ax.set_title(f'Concordance Direction by Patient{title_suffix}\n(Positive = Concordance, Negative = Discordance)', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    legend_elements = [
        mpatches.Patch(color='#2E8B57', label='Concordance (Z >= 0)'),
        mpatches.Patch(color='#FF6B6B', label='Discordance (Z < 0)')
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    filename = f'concordance_direction_{filter_type}.png'
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_significance_only(sig_df, output_dir):
    """
    Plot 4b: Pure significance plot - just shows p-values / significance tiers.
    """
    print("Generating Significance-Only Plot...")
    
    sig_df = sig_df.copy()
    sig_df['patient_id'] = sig_df['patient_id'].astype(str)
    
    # Sort by p-value
    sig_df = sig_df.sort_values('p_value')
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x_positions = np.arange(len(sig_df))
    
    # -log10(p) for visualization
    neg_log_p = -np.log10(sig_df['p_value'].clip(lower=1e-10))
    
    # Colors based on significance
    colors = []
    for _, row in sig_df.iterrows():
        if row['significant_bonferroni']:
            colors.append('#2E8B57')  # Strong
        elif row['significant_uncorrected']:
            colors.append('#DAA520')  # Nominal
        else:
            colors.append('#808080')  # Not significant
    
    ax.bar(x_positions, neg_log_p, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add significance thresholds
    bonferroni_threshold = -np.log10(0.05 / len(sig_df))
    nominal_threshold = -np.log10(0.05)
    
    ax.axhline(bonferroni_threshold, color='#2E8B57', linestyle='--', linewidth=1.5, 
               label=f'Bonferroni (p < {0.05/len(sig_df):.4f})')
    ax.axhline(nominal_threshold, color='#DAA520', linestyle='--', linewidth=1.5,
               label='Nominal (p < 0.05)')
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(sig_df['patient_id'], rotation=45)
    ax.set_xlabel('Patient ID (sorted by p-value)', fontsize=12)
    ax.set_ylabel('-log₁₀(p-value)', fontsize=12)
    ax.set_title('Statistical Significance by Patient', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(loc='upper right')
    
    output_path = output_dir / 'significance_only.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_significance_table(sig_df, col_cfg, output_dir):
    """
    Plot 4c: Create a nice table image showing p-values and significance.
    """
    measure_label = col_cfg["label"]
    print(f"Generating Significance Table ({measure_label})...")

    sig_df = sig_df.copy()
    sig_df['patient_id'] = sig_df['patient_id'].astype(str)

    sig_df = sig_df.sort_values('p_value').reset_index(drop=True)

    table_data = []
    cell_colors = []

    for _, row in sig_df.iterrows():
        pid = row['patient_id']
        p = row['p_value']
        median_z = row['observed_z']
        
        # Format p-value
        if p < 0.0001:
            p_str = '< 0.0001'
        elif p < 0.001:
            p_str = f'{p:.4f}'
        elif p < 0.01:
            p_str = f'{p:.4f}'
        else:
            p_str = f'{p:.3f}'
        
        # Determine significance level
        if row['significant_bonferroni']:
            sig_level = '✓✓ Bonferroni'
            row_color = ['#C8E6C9', '#C8E6C9', '#C8E6C9', '#C8E6C9']  # Light green
        elif row['significant_uncorrected']:
            sig_level = '✓ Nominal'
            row_color = ['#FFF9C4', '#FFF9C4', '#FFF9C4', '#FFF9C4']  # Light yellow
        else:
            sig_level = '✗ Not Sig.'
            row_color = ['#F5F5F5', '#F5F5F5', '#F5F5F5', '#F5F5F5']  # Light grey
        
        # Format Z-score
        z_str = f'{median_z:.3f}'
        
        table_data.append([pid, p_str, z_str, sig_level])
        cell_colors.append(row_color)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    columns = ['Patient ID', 'P-Value', f'{measure_label} Z', 'Significance']
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        cellColours=cell_colors,
        colColours=['#E3F2FD'] * 4,  # Light blue header
        loc='center',
        cellLoc='center'
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Make header bold
    for j in range(len(columns)):
        table[(0, j)].set_text_props(fontweight='bold')
    
    # Title
    ax.set_title(f'Permutation Test Results ({measure_label} Z): Statistical Significance\n(Bonferroni threshold: p < 0.0033)',
                fontsize=14, fontweight='bold', pad=20)
    
    # Add legend at bottom
    legend_text = '✓✓ Bonferroni Significant (p < 0.0033)  |  ✓ Nominally Significant (p < 0.05)  |  ✗ Not Significant'
    fig.text(0.5, 0.02, legend_text, ha='center', fontsize=10, style='italic')
    
    output_path = output_dir / 'significance_table.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def plot_specification_curve_plain(detailed_df, patient_id, output_dir):
    """
    Plot 3: Specification curve without color coding - plain/single color.
    """
    print(f"Generating Plain Specification Curve for Patient {patient_id}...")
    
    df = detailed_df[detailed_df['user_key'] == patient_id].copy()
    
    if len(df) == 0:
        print(f"No data for patient {patient_id}")
        return
    
    df = df.sort_values('spearman_z').reset_index(drop=True)
    df['rank'] = df.index
    
    specs = {
        'Time Window': ['timestamp_window', [12, 24, 36, 48]],
        'Window Type': ['use_daily_max_windows', [False, True], ['Rolling', 'Daily Max']],
        'Day Type': ['use_calendar_days', [False, True], ['24h Period', 'Calendar Day']],
        'Zero Handling': ['filter_out_zero_usage_entries', [False, True], ['Keep Zeros', 'Drop Zeros']],
        'Categorization': ['categorization_method', [
            'one_hot', 'lower_bound', 'midpoint', 'upper_bound', 
            'midpoint_with_inhaler', 'upper_bound_with_inhaler'
        ]]
    }
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5], hspace=0.05)
    
    # Top Panel: Curve (no color)
    ax_curve = fig.add_subplot(gs[0])
    
    # Single color for all points
    ax_curve.scatter(df.index, df['spearman_z'], c='#4169E1', s=20, zorder=2, alpha=0.7)
    ax_curve.plot(df.index, df['spearman_z'], color='#4169E1', linewidth=0.8, zorder=1, alpha=0.5)
    
    # Add median line (excluding non-finite values)
    finite_z = df['spearman_z'][np.isfinite(df['spearman_z'])]
    median_z = finite_z.median() if len(finite_z) > 0 else 0
    ax_curve.axhline(median_z, color='#FF6B6B', linestyle='--', linewidth=1.5, 
                     label=f'Median: {median_z:.3f}')
    ax_curve.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
    
    ax_curve.set_ylabel("Fisher's Z-Score", fontsize=12)
    ax_curve.set_title(f'Specification Curve: Patient {patient_id}', fontsize=14, pad=15)
    ax_curve.grid(True, linestyle='--', alpha=0.3)
    ax_curve.set_xticklabels([])
    ax_curve.set_xlim(-2, len(df)+2)
    ax_curve.legend(loc='upper left')
    
    # Bottom Panel: Barcode
    ax_bar = fig.add_subplot(gs[1], sharex=ax_curve)
    
    y_pos = 0
    y_labels = []
    
    for group_name, spec_info in specs.items():
        col = spec_info[0]
        values = spec_info[1]
        labels = spec_info[2] if len(spec_info) > 2 else [str(v) for v in values]
        
        ax_bar.text(-5, y_pos - (len(values)/2) + 0.5, group_name, 
                   fontsize=10, fontweight='bold', va='center', ha='right')
        
        for i, val in enumerate(values):
            matches = df[col] == val
            x_matches = df.index[matches]
            ax_bar.scatter(x_matches, [y_pos] * len(x_matches), 
                          marker='|', color='#333333', s=60, linewidth=0.8)
            
            label_text = labels[i] if isinstance(labels[i], str) else str(labels[i])
            label_text = label_text.replace('_', ' ').title()
            y_labels.append((y_pos, label_text))
            
            y_pos -= 1
        
        y_pos -= 0.5
    
    ax_bar.set_yticks([y[0] for y in y_labels])
    ax_bar.set_yticklabels([y[1] for y in y_labels], fontsize=9)
    ax_bar.set_ylim(y_pos + 0.5, 0.5)
    ax_bar.set_xlabel('Specifications (sorted by correlation)', fontsize=12)
    ax_bar.grid(axis='y', linestyle='-', alpha=0.2)
    
    save_path = output_dir / f'specification_curve_plain_{patient_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate refined permutation plots.")
    parser.add_argument("--measure", choices=["median", "mean"], default="median",
                        help="Summary statistic to use (default: median)")
    return parser.parse_args()


def main():
    args = parse_args()
    measure = args.measure
    col_cfg = MEASURE_COLUMNS[measure]
    group_label = GROUPS.get(col_cfg["group_key"], {}).get("label", col_cfg["label"])

    current_dir = Path.cwd()
    print(f"Working directory: {current_dir}")
    print(f"Measure: {measure} (group: {group_label})")

    output_dir = Path(f"../../results/v1/permutation_aggregate_extended/{col_cfg['group_key']}/refined_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        detailed_df, sig_df, col_cfg = load_data(measure)
        print("Data loaded successfully.\n")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # =========================================================================
    # PLOT 1: Point plot with null band (3 versions)
    # =========================================================================
    plot_point_with_null_band(detailed_df, sig_df, col_cfg, output_dir,
                              filter_type='all', title_suffix=' (All Patients)')
    plot_point_with_null_band(detailed_df, sig_df, col_cfg, output_dir,
                              filter_type='significant', title_suffix=' (Significant Only)')
    plot_point_with_null_band(detailed_df, sig_df, col_cfg, output_dir,
                              filter_type='concordant',
                              title_suffix=f' ({group_label} Patients)')

    # =========================================================================
    # PLOT 4: Separate concordance and significance
    # =========================================================================
    plot_concordance_only(detailed_df, sig_df, col_cfg, output_dir,
                         filter_type='all', title_suffix=' (All Patients)')
    plot_concordance_only(detailed_df, sig_df, col_cfg, output_dir,
                         filter_type='significant', title_suffix=' (Bonferroni Significant)')
    plot_concordance_only(detailed_df, sig_df, col_cfg, output_dir,
                         filter_type='concordant',
                         title_suffix=f' ({group_label}: Z>=0.5 & Significant)')
    plot_concordance_only(detailed_df, sig_df, col_cfg, output_dir,
                         filter_type='nominal_and_significant',
                         title_suffix=' (Nominally + Bonferroni Significant)')

    plot_significance_only(sig_df, output_dir)
    plot_significance_table(sig_df, col_cfg, output_dir)

    # =========================================================================
    # PLOT 3: Plain specification curves (no color coding)
    # =========================================================================
    sig_patients = sig_df[sig_df['significant_bonferroni']]['patient_id'].tolist()
    print(f"\nGenerating plain specification curves for {len(sig_patients)} significant patients...")

    for patient_id in sig_patients:
        plot_specification_curve_plain(detailed_df, patient_id, output_dir)

    print("\n" + "="*60)
    print(f"All refined plots generated successfully! (measure={measure})")
    print(f"Output directory: {output_dir.absolute()}")
    print("="*60)


if __name__ == "__main__":
    main()

