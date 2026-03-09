import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.patches as mpatches

def load_data():
    """Load necessary data files."""
    # 1. Detailed results (192 configs per patient)
    # Try multiple potential locations
    potential_paths = [
        Path("../../results/per_patient_analysis/per_patient_correlation_results.csv"),
        Path("../results/per_patient_analysis/per_patient_correlation_results.csv"),
        Path("results/per_patient_analysis/per_patient_correlation_results.csv"),
    ]
    
    detailed_results_path = None
    for p in potential_paths:
        if p.exists():
            detailed_results_path = p
            print(f"Found detailed results at: {p}")
            break
            
    if not detailed_results_path:
        raise FileNotFoundError("Could not find per_patient_correlation_results.csv in standard locations")
    
    detailed_df = pd.read_csv(detailed_results_path)
    
    # 2. Significance results (from permutation test)
    sig_results_path = Path("../../results/permutation_aggregate_extended/permutation_test_results.csv")
    sig_df = pd.read_csv(sig_results_path)
    
    # 3. Permutation null data (for the null band)
    # We'll use the all_permutations.parquet if available, or approximate from summary stats
    perm_path = Path("../../results/permutation_aggregate_extended/all_permutations.parquet")
    if perm_path.exists():
        null_df = pd.read_parquet(perm_path)
    else:
        print("Warning: Null distribution file not found. Null band will be approximate.")
        null_df = None
        
    return detailed_df, sig_df, null_df

def plot_distribution_boxplot(detailed_df, sig_df, null_df, output_dir):
    """Create box plot of correlations per patient sorted by median."""
    print("Generating Box Plot...")
    
    # Prepare data
    # Merge significance info into detailed_df
    # Ensure types match
    detailed_df['user_key'] = detailed_df['user_key'].astype(str)
    sig_df['patient_id'] = sig_df['patient_id'].astype(str)
    
    print(f"Detailed DF patients: {len(detailed_df['user_key'].unique())}")
    print(f"Sig DF patients: {len(sig_df['patient_id'].unique())}")
    
    plot_df = detailed_df.merge(
        sig_df[['patient_id', 'significant_bonferroni', 'significant_uncorrected']], 
        left_on='user_key', 
        right_on='patient_id', 
        how='inner' # Only keep patients present in both
    )
    
    if len(plot_df) == 0:
        print("Error: No common patients found between detailed results and significance results!")
        print(f"Detailed sample IDs: {detailed_df['user_key'].unique()[:5]}")
        print(f"Sig sample IDs: {sig_df['patient_id'].unique()[:5]}")
        return

    # Calculate median Z per patient for sorting
    patient_medians = plot_df.groupby('user_key')['spearman_z'].median().sort_values()
    sorted_patients = patient_medians.index.tolist()
    
    # Define colors
    # Teal/Red: Significant (Bonferroni)
    # Gold: Nominally Significant (p < 0.05 uncorrected)
    # Grey: Non-significant
    palette = {}
    
    for pid in sorted_patients:
        # Get significance for this patient
        patient_rows = plot_df[plot_df['user_key'] == pid]
        if len(patient_rows) == 0:
            continue
            
        # Get values from first row (repeated for patient)
        is_sig_strict = patient_rows['significant_bonferroni'].iloc[0]
        is_sig_nominal = patient_rows['significant_uncorrected'].iloc[0]
        median_z = patient_medians[pid]
        
        if is_sig_strict:
            if median_z >= 0:
                palette[pid] = '#2E8B57' # Teal (Concordance - Strict)
            else:
                palette[pid] = '#FF6B6B' # Red (Discordance - Strict)
        elif is_sig_nominal:
            palette[pid] = '#DAA520'     # Gold (Marginal/Nominal Significance)
        else:
            palette[pid] = '#808080'     # Grey (Not Significant)
    
    # Setup plot
    plt.figure(figsize=(16, 8))
    
    # Draw Null Intervals (Patient-Specific) instead of Pooled Band
    # This handles heteroscedasticity (different sample sizes = different noise widths)
    print("Drawing patient-specific null intervals...")
    
    # Calculate positions (0, 1, 2...) corresponding to sorted_patients
    for i, pid in enumerate(sorted_patients):
        # Get null stats for this patient
        patient_sig = sig_df[sig_df['patient_id'] == pid].iloc[0]
        
        # Calculate 95% CI from null distribution
        if null_df is not None and not null_df.empty:
            # Exact percentiles if we have the full data
            patient_nulls = null_df[null_df['patient_id'] == pid]['null_median_z'].dropna()
            if not patient_nulls.empty:
                low = np.percentile(patient_nulls, 2.5)
                high = np.percentile(patient_nulls, 97.5)
            else:
                # Fallback to stats (handle both old and new column names)
                mean_col = 'dist_null_median_z_mean' if 'dist_null_median_z_mean' in patient_sig.index else 'null_mean'
                std_col = 'dist_null_median_z_std' if 'dist_null_median_z_std' in patient_sig.index else 'null_std'
                mean = patient_sig[mean_col]
                std = patient_sig[std_col]
                low = mean - 1.96 * std
                high = mean + 1.96 * std
        else:
            # Fallback to stats (handle both old and new column names)
            mean_col = 'dist_null_median_z_mean' if 'dist_null_median_z_mean' in patient_sig.index else 'null_mean'
            std_col = 'dist_null_median_z_std' if 'dist_null_median_z_std' in patient_sig.index else 'null_std'
            mean = patient_sig[mean_col]
            std = patient_sig[std_col]
            low = mean - 1.96 * std
            high = mean + 1.96 * std
            
        # Draw the null interval as a grey bar behind the box
        # x-coordinates: i - 0.4 to i + 0.4 (width of the column)
        # Changed color to 'steelblue' to contrast with grey/teal/red boxes
        plt.fill_between(
            [i - 0.4, i + 0.4], 
            [low, low], 
            [high, high], 
            color='#4682B4',  # SteelBlue
            alpha=0.2, 
            edgecolor=None,
            zorder=0 # Behind the box plots
        )
        
        # Optional: Add a small horizontal line for the null mean
        # plt.hlines(patient_sig['null_mean'], i - 0.3, i + 0.3, color='gray', linestyle=':', alpha=0.5)

    # Draw Zero Line
    plt.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.4, zorder=1)
    
    # Draw Box Plot
    sns.boxplot(
        data=plot_df,
        x='user_key',
        y='spearman_z',
        order=sorted_patients,
        palette=palette,
        fliersize=2,
        linewidth=1.2,
        zorder=10,
        # Make boxes transparent to see the band behind them if they overlap
        boxprops=dict(alpha=0.6)
    )
    
    # Customizing legend
    legend_elements = [
        mpatches.Patch(color='#2E8B57', label='Significant (Bonferroni p < 0.0023)'),
        mpatches.Patch(color='#FF6B6B', label='Significant Discordance (Bonferroni)'),
        mpatches.Patch(color='#DAA520', label='Nominally Significant (p < 0.05)'),
        mpatches.Patch(color='#808080', label='Non-Significant'),
        mpatches.Patch(facecolor='#4682B4', alpha=0.2, label='Patient-Specific Null (95% CI)')
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    
    plt.title('Distribution of Correlations Across Multiverse Configurations', fontsize=14, pad=20)
    plt.xlabel('Patient ID', fontsize=12)
    plt.ylabel("Fisher's Z-Score (Spearman)", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Save
    output_path = output_dir / 'concordance_distribution_boxplot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved box plot to {output_path}")
    plt.close()

def plot_specification_curve(detailed_df, patient_id, output_dir):
    """Generate specification curve for a single patient."""
    print(f"Generating Specification Curve for Patient {patient_id}...")
    
    # Filter data for patient
    df = detailed_df[detailed_df['user_key'] == patient_id].copy()
    
    # Sort by correlation coefficient
    df = df.sort_values('spearman_z').reset_index(drop=True)
    df['rank'] = df.index
    
    # Define specifications (parameters) to plot
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
    
    # Setup Grid
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5], hspace=0.05)
    
    # Top Panel: Curve
    ax_curve = fig.add_subplot(gs[0])
    
    # Color points by categorization method to show sensitivity
    # Or just simple black points with error bars if we had them. 
    # Let's use color to represent the most impactful parameter (often Categorization)
    cat_codes = pd.Categorical(df['categorization_method']).codes
    cmap = plt.get_cmap('viridis', len(df['categorization_method'].unique()))
    
    scatter = ax_curve.scatter(df.index, df['spearman_z'], c=cat_codes, cmap=cmap, s=20, zorder=2)
    
    # Add line connecting points
    ax_curve.plot(df.index, df['spearman_z'], color='gray', linewidth=0.5, zorder=1, alpha=0.5)
    
    ax_curve.set_ylabel("Fisher's Z-Score", fontsize=12)
    ax_curve.set_title(f'Specification Curve Analysis: Patient {patient_id}', fontsize=14, pad=20)
    ax_curve.grid(True, linestyle='--', alpha=0.3)
    ax_curve.set_xticklabels([])
    ax_curve.set_xlim(-2, len(df)+2)
    
    # Bottom Panel: Barcode
    ax_bar = fig.add_subplot(gs[1], sharex=ax_curve)
    
    # Iterate through parameters and plot ticks
    y_pos = 0
    y_labels = []
    
    # Group specifications
    for group_name, spec_info in specs.items():
        col = spec_info[0]
        values = spec_info[1]
        labels = spec_info[2] if len(spec_info) > 2 else [str(v) for v in values]
        
        # Add group header
        ax_bar.text(-5, y_pos - (len(values)/2) + 0.5, group_name, 
                   fontsize=10, fontweight='bold', va='center', ha='right')
        
        for i, val in enumerate(values):
            # Find indices where this parameter value matches
            matches = df[col] == val
            
            # Plot ticks
            # We use scatter instead of vlines for better performance/look with dots
            x_matches = df.index[matches]
            ax_bar.scatter(x_matches, [y_pos] * len(x_matches), 
                          marker='|', color='black', s=60, linewidth=0.8)
            
            # Add label
            label_text = labels[i]
            # Clean up labels
            label_text = label_text.replace('_', ' ').title()
            y_labels.append((y_pos, label_text))
            
            y_pos -= 1
        
        # Add spacer between groups
        y_pos -= 0.5
        
    # Formatting Bottom Panel
    ax_bar.set_yticks([y[0] for y in y_labels])
    ax_bar.set_yticklabels([y[1] for y in y_labels], fontsize=9)
    ax_bar.set_ylim(y_pos + 0.5, 0.5)
    ax_bar.set_xlabel('Specifications (sorted by correlation magnitude)', fontsize=12)
    
    # Add alternating background shading for readability
    # (simplified version: just grid lines)
    ax_bar.grid(axis='y', linestyle='-', alpha=0.2)
    
    # Color bar for the curve (optional, helps identify cat method)
    # cbar = plt.colorbar(scatter, ax=ax_curve, orientation='vertical', pad=0.01)
    # cbar.set_label('Categorization Method')
    
    save_path = output_dir / f'specification_curve_patient_{patient_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved specification curve to {save_path}")
    plt.close()

def main():
    # Setup paths
    current_dir = Path.cwd()
    print(f"Working directory: {current_dir}")
    
    output_dir = Path("../../results/permutation_aggregate_extended/advanced_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        detailed_df, sig_df, null_df = load_data()
        print("Data loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure you are in the 'permutation_cloudrun/aggregator' directory.")
        return

    # 1. Generate Distribution Box Plot
    plot_distribution_boxplot(detailed_df, sig_df, null_df, output_dir)
    
    # 2. Generate Specification Curves for Significant Patients
    # Filter significant patients (uncorrected or Bonferroni - Bonferroni for strictness)
    sig_patients = sig_df[sig_df['significant_bonferroni']]['patient_id'].tolist()
    print(f"Generating specification curves for {len(sig_patients)} significant patients: {sig_patients}")
    
    for patient_id in sig_patients:
        plot_specification_curve(detailed_df, patient_id, output_dir)
        
    print("\nAll plots generated successfully!")

if __name__ == "__main__":
    main()

