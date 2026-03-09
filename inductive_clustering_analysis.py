"""
Inductive Clustering Analysis for User Feedback

This script performs proper inductive clustering to:
1. Identify natural thematic groupings in feedback (without predefined labels)
2. Compare emergent clusters with our deductive codebook
3. Validate or refine our thematic categories

Usage:
    python inductive_clustering_analysis.py [--group concordant|median_concordant]

Groups are defined in config.py and selected via the --group argument.
"""

import os
import argparse
import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage
import openai
from config import GROUPS, FEEDBACK_PATIENTS, get_group_config, get_output_dir

# These module-level variables are set in main() based on --group arg
OUTPUT_DIR = None
CONCORDANT_USERS = None
REMAINING_ASSESSED = None
ALL_INCLUDED = None
GROUP_LABEL = None

# API key - loaded lazily; only needed if embedding cache misses
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


def load_feedback_data():
    """Load and prepare feedback data, de-duplicating user 514."""
    print("Loading feedback data...")
    
    feedback_df = pd.read_csv("aamos00-end-final-freetext.csv")
    
    # Combine feedback columns
    feedback_df['combined_feedback'] = (
        feedback_df['What features / information would you most want to see in an asthma management system?'].fillna('') + 
        ' ' + 
        feedback_df['Any other comments'].fillna('')
    )
    
    # Clean the text
    feedback_df['combined_feedback'] = feedback_df['combined_feedback'].apply(
        lambda x: re.sub(r'\s+', ' ', x).strip()
    )
    
    # Filter to only include assessed patients
    feedback_df = feedback_df[feedback_df['user_key'].isin(ALL_INCLUDED)]
    feedback_df = feedback_df[feedback_df['combined_feedback'] != '']
    
    # De-duplicate user 514
    if feedback_df['user_key'].duplicated().any():
        print(f"De-duplicating users with multiple entries...")
        feedback_dedup = feedback_df.groupby('user_key').agg({
            'combined_feedback': lambda x: ' '.join(x.dropna().astype(str)),
            **{col: 'first' for col in feedback_df.columns if col not in ['user_key', 'combined_feedback']}
        }).reset_index()
        
        feedback_dedup['combined_feedback'] = feedback_dedup['combined_feedback'].apply(
            lambda x: re.sub(r'\s+', ' ', x).strip()
        )
        feedback_df = feedback_dedup
    
    # Assign groups
    feedback_df['group'] = feedback_df['user_key'].apply(
        lambda x: GROUP_LABEL if x in CONCORDANT_USERS else 'Remaining'
    )
    
    print(f"Loaded {len(feedback_df)} unique patients with feedback.")
    return feedback_df


EMBEDDING_CACHE_PATH = None  # Set in main() after OUTPUT_DIR is initialized

def get_openai_embeddings(texts, client, use_cache=True):
    """Get embeddings from OpenAI, with caching to avoid repeated API calls."""
    
    # Try to load from cache
    if use_cache and EMBEDDING_CACHE_PATH is not None and EMBEDDING_CACHE_PATH.exists():
        print("  Loading embeddings from cache...")
        cached = np.load(EMBEDDING_CACHE_PATH, allow_pickle=True)
        cached_texts = cached['texts'].tolist()
        cached_embeddings = cached['embeddings']
        
        # Check if texts match
        if cached_texts == texts:
            print(f"  Cache hit! Loaded {len(cached_embeddings)} embeddings.")
            return cached_embeddings
        else:
            print("  Cache miss (texts changed). Fetching fresh embeddings...")
    
    # Fetch from API
    print("  Calling OpenAI API for embeddings...")
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )
    embeddings = np.array([item.embedding for item in response.data])
    
    # Save to cache
    np.savez(EMBEDDING_CACHE_PATH, texts=np.array(texts, dtype=object), embeddings=embeddings)
    print(f"  Cached {len(embeddings)} embeddings to {EMBEDDING_CACHE_PATH}")
    
    return embeddings


def find_optimal_clusters(embeddings, max_k=8):
    """Find optimal number of clusters using silhouette analysis."""
    print("\n" + "="*60)
    print("SILHOUETTE ANALYSIS FOR OPTIMAL K")
    print("="*60)
    
    n_samples = len(embeddings)
    max_k = min(max_k, n_samples - 1)  # k must be less than n_samples
    
    k_range = range(2, max_k + 1)
    silhouette_scores = []
    inertias = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        silhouette_scores.append(score)
        inertias.append(kmeans.inertia_)
        print(f"  k={k}: silhouette={score:.3f}, inertia={kmeans.inertia_:.1f}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Silhouette scores
    axes[0].plot(list(k_range), silhouette_scores, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Silhouette Score')
    axes[0].set_title('Silhouette Analysis')
    axes[0].set_xticks(list(k_range))
    
    best_k = list(k_range)[np.argmax(silhouette_scores)]
    axes[0].axvline(x=best_k, color='r', linestyle='--', label=f'Best k={best_k}')
    axes[0].legend()
    
    # Elbow plot (inertia)
    axes[1].plot(list(k_range), inertias, 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Inertia (Within-cluster sum of squares)')
    axes[1].set_title('Elbow Method')
    axes[1].set_xticks(list(k_range))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cluster_selection.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nOptimal k by silhouette: {best_k} (score={max(silhouette_scores):.3f})")
    return best_k, silhouette_scores


def perform_clustering(embeddings, feedback_df, k):
    """Perform k-means clustering and analyze results."""
    print(f"\n" + "="*60)
    print(f"K-MEANS CLUSTERING (k={k})")
    print("="*60)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    feedback_df['cluster'] = labels
    
    # Reduce to 2D for visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)
    feedback_df['tsne_x'] = embeddings_2d[:, 0]
    feedback_df['tsne_y'] = embeddings_2d[:, 1]
    
    return feedback_df, kmeans


def visualize_clusters(feedback_df, k):
    """Create single merged visualization: shape = group, fill = cluster."""
    from matplotlib.lines import Line2D

    dark_color = '#1A5276'
    light_color = '#A0A0A0'
    cluster_colors = [dark_color, light_color] + ['#D35400', '#8E44AD', '#2980B9', '#27AE60', '#C0392B']
    group_markers = {GROUP_LABEL: 's', 'Remaining': 'o'}

    fig, ax = plt.subplots(figsize=(8, 6))

    for cluster_id in range(k):
        for group in [GROUP_LABEL, 'Remaining']:
            mask = (feedback_df['cluster'] == cluster_id) & (feedback_df['group'] == group)
            if mask.sum() == 0:
                continue
            ax.scatter(
                feedback_df.loc[mask, 'tsne_x'],
                feedback_df.loc[mask, 'tsne_y'],
                c=cluster_colors[cluster_id % len(cluster_colors)],
                marker=group_markers[group],
                s=150, alpha=0.85, edgecolors='black', linewidth=1.5,
            )
            for idx, row in feedback_df[mask].iterrows():
                ax.annotate(
                    str(int(row['user_key'])),
                    (row['tsne_x'], row['tsne_y'] + 8),
                    fontsize=9, ha='center', va='bottom', fontweight='bold',
                )

    legend_elements = [
        Line2D([0], [0], marker='s', markerfacecolor=dark_color, markeredgecolor='black',
               markeredgewidth=1.5, label=f'{GROUP_LABEL}, Cluster 1', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='s', markerfacecolor=light_color, markeredgecolor='black',
               markeredgewidth=1.5, label=f'{GROUP_LABEL}, Cluster 2', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='o', markerfacecolor=dark_color, markeredgecolor='black',
               markeredgewidth=1.5, label=f'Remaining, Cluster 1', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='o', markerfacecolor=light_color, markeredgecolor='black',
               markeredgewidth=1.5, label=f'Remaining, Cluster 2', markersize=10, linestyle='None'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)

    ax.set_xlabel('t-SNE 1', fontsize=11)
    ax.set_ylabel('t-SNE 2', fontsize=11)
    ax.set_title('Inductive Clustering\n(Shape = Group, Shade = Cluster)', fontsize=12, fontweight='bold')

    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + 15)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'inductive_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cluster visualization to {OUTPUT_DIR / 'inductive_clusters.png'}")


def analyze_cluster_content(feedback_df, k):
    """Analyze what feedback is in each cluster to identify emergent themes."""
    print("\n" + "="*60)
    print("CLUSTER CONTENT ANALYSIS")
    print("="*60)
    
    results = []
    
    for cluster_id in range(k):
        mask = feedback_df['cluster'] == cluster_id
        cluster_df = feedback_df[mask]
        
        print(f"\n{'='*50}")
        print(f"CLUSTER {cluster_id + 1} ({len(cluster_df)} patients)")
        print(f"{'='*50}")
        
        # Group composition
        optimal_count = (cluster_df['group'] == GROUP_LABEL).sum()
        remaining_count = (cluster_df['group'] == 'Remaining').sum()
        print(f"Composition: {optimal_count} Optimal, {remaining_count} Remaining")
        
        # Patient IDs
        patient_ids = cluster_df['user_key'].tolist()
        print(f"Patients: {patient_ids}")
        
        # Print each feedback
        print(f"\nFeedback in this cluster:")
        for idx, row in cluster_df.iterrows():
            group_tag = "[CONC]" if row['group'] == GROUP_LABEL else "[REM]"
            print(f"\n  Patient {int(row['user_key'])} {group_tag}:")
            # Wrap text for readability
            feedback = row['combined_feedback']
            print(f"    \"{feedback[:300]}{'...' if len(feedback) > 300 else ''}\"")
        
        results.append({
            'cluster': cluster_id + 1,
            'n_patients': len(cluster_df),
            'n_optimal': optimal_count,
            'n_remaining': remaining_count,
            'patient_ids': patient_ids
        })
    
    # Summary table
    print("\n" + "="*60)
    print("CLUSTER SUMMARY")
    print("="*60)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    return results_df


def hierarchical_clustering_dendrogram(embeddings, feedback_df):
    """Create a dendrogram to visualize hierarchical structure."""
    print("\n" + "="*60)
    print("HIERARCHICAL CLUSTERING DENDROGRAM")
    print("="*60)
    
    # Compute linkage
    Z = linkage(embeddings, method='ward')
    
    # Create labels
    labels = [f"{int(row['user_key'])} ({'C' if row['group'] == GROUP_LABEL else 'R'})" 
              for _, row in feedback_df.iterrows()]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z, labels=labels, leaf_rotation=45, leaf_font_size=10, ax=ax)
    ax.set_title(f'Hierarchical Clustering of User Feedback\n(C={GROUP_LABEL}, R=Remaining)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distance')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dendrogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved dendrogram to {OUTPUT_DIR / 'dendrogram.png'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Inductive clustering analysis.")
    parser.add_argument("--group", choices=list(GROUPS.keys()), default="concordant",
                        help="Patient group definition to use (default: concordant)")
    return parser.parse_args()


def main():
    """Run complete inductive clustering analysis."""
    global OUTPUT_DIR, CONCORDANT_USERS, REMAINING_ASSESSED, ALL_INCLUDED, GROUP_LABEL, EMBEDDING_CACHE_PATH

    args = parse_args()
    group_cfg = get_group_config(args.group)
    CONCORDANT_USERS = group_cfg["patients"]
    GROUP_LABEL = group_cfg["label"]

    concordant_with_feedback = [p for p in CONCORDANT_USERS if p in FEEDBACK_PATIENTS]
    REMAINING_ASSESSED = [p for p in FEEDBACK_PATIENTS if p not in CONCORDANT_USERS]
    ALL_INCLUDED = concordant_with_feedback + REMAINING_ASSESSED

    OUTPUT_DIR = get_output_dir(args.group, "sentiment_analysis") / "clustering"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_CACHE_PATH = OUTPUT_DIR / 'embedding_cache.npz'

    print("="*60)
    print(f"INDUCTIVE CLUSTERING ANALYSIS - {GROUP_LABEL} vs Remaining")
    print("="*60)

    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY not set. Will rely on cached embeddings.")
    client = openai.OpenAI(api_key=OPENAI_API_KEY or "dummy")
    
    # Load data
    feedback_df = load_feedback_data()
    
    # Get embeddings
    print("\nGetting embeddings...")
    texts = feedback_df['combined_feedback'].tolist()
    embeddings = get_openai_embeddings(texts, client)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Find optimal k
    best_k, scores = find_optimal_clusters(embeddings, max_k=min(7, len(texts)-1))
    
    # Also try k=3 and k=4 explicitly (since we have 7 themes, mid-range clusters might be interpretable)
    print("\n" + "="*60)
    print("ANALYZING MULTIPLE K VALUES")
    print("="*60)
    
    for k in [2, 3, 4, best_k]:
        if k > len(texts) - 1:
            continue
            
        print(f"\n{'#'*60}")
        print(f"# K = {k}")
        print(f"{'#'*60}")
        
        feedback_df_copy = feedback_df.copy()
        feedback_df_copy, kmeans = perform_clustering(embeddings, feedback_df_copy, k)
        
        if k == best_k or k == 3:  # Save visualizations for best k and k=3
            visualize_clusters(feedback_df_copy, k)
        
        analyze_cluster_content(feedback_df_copy, k)
    
    # Hierarchical clustering for structure view
    hierarchical_clustering_dendrogram(embeddings, feedback_df)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nOutput saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

