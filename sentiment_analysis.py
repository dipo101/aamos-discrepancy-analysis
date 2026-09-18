"""
Sentiment Analysis - Aligned with Refined Methodology

This script implements the methodology as described:
A. Preliminary Semantic Clustering (done separately in inductive_clustering_analysis.py)
B. Multi-Agent LLM Ensemble for Thematic Coding (with internal reliability)
C. Embedding-Based Sentiment Classifier
D. Inter-Method Agreement (Cohen's Kappa between B & C)

Usage:
    python sentiment_analysis.py [--group concordant|median_concordant]

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

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
from scipy import stats
import openai
from aamos_concordance import find_raw_file
from config import GROUPS, FEEDBACK_PATIENTS, get_group_config, get_output_dir, add_world_argument, set_active_world, record_run_provenance

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

# Initialize VADER
sia = SentimentIntensityAnalyzer()

# These module-level variables are set in main() based on --group arg
OUTPUT_DIR = None
MAIN_DIR = None
SUPP_DIR = None
CONCORDANT_USERS = None
REMAINING_ASSESSED = None
ALL_INCLUDED = None
GROUP_LABEL = None

# API key - load from environment variable
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")


def load_feedback_data():
    """Load and prepare feedback data with de-duplication."""
    print("Loading feedback data...")
    
    feedback_df = pd.read_csv(find_raw_file("aamos00-end-final-freetext.csv"))
    
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
        print("De-duplicating users with multiple entries...")
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
    print(f"  {GROUP_LABEL}: {len(feedback_df[feedback_df['group'] == GROUP_LABEL])}")
    print(f"  Remaining: {len(feedback_df[feedback_df['group'] == 'Remaining'])}")
    
    return feedback_df


def fleiss_kappa(ratings_matrix):
    """
    Calculate Fleiss' Kappa for multiple raters.
    
    Parameters:
    - ratings_matrix: numpy array of shape (n_subjects, n_categories)
                      where each cell contains the count of raters who assigned that category
    
    Returns:
    - kappa: Fleiss' Kappa score
    """
    n_subjects, n_categories = ratings_matrix.shape
    n_raters = ratings_matrix.sum(axis=1)[0]  # Assume same number of raters per subject
    
    # Proportion of all assignments to each category
    p_j = ratings_matrix.sum(axis=0) / (n_subjects * n_raters)
    
    # Extent of agreement for each subject
    P_i = (ratings_matrix.sum(axis=1)**2 - ratings_matrix.sum(axis=1)) / (n_raters * (n_raters - 1))
    # Correction: P_i should be sum of n_ij^2 minus n, divided by n*(n-1)
    P_i = ((ratings_matrix ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    
    # Mean extent of agreement
    P_bar = P_i.mean()
    
    # Expected agreement by chance
    P_e = (p_j ** 2).sum()
    
    # Fleiss' Kappa
    if P_e == 1:
        return 1.0
    kappa = (P_bar - P_e) / (1 - P_e)
    
    return kappa


def llm_ensemble_with_internal_reliability(feedback_df, client):
    """
    LLM Ensemble classification capturing individual persona ratings for internal reliability.
    """
    print("\n" + "="*60)
    print("LLM ENSEMBLE WITH INTERNAL RELIABILITY")
    print("="*60)
    
    sentiment_categories = ["Positive", "Neutral", "Negative"]
    theme_list = [
        "Reliability & Accuracy",
        "Integration & Interoperability",
        "Reminders & Motivation",
        "Insights & Self-Discovery",
        "Trends & Visualization",
        "Clinical Care Value",
        "Positive Experience & Gratitude"
    ]
    
    base_prompt = """
You are an expert analyst coding patient feedback from a mobile health study about asthma monitoring.

**Study Context:** The "AAMOS-00 Study" monitored asthma patients using smart inhalers and peak flow meters. 
Patients provided end-of-study feedback about their experience.

**Your task:** 
1. Classify overall sentiment toward the monitoring technology
2. Identify which themes are present (multi-label: check ALL that apply)
3. For each theme detected, provide a brief quote (max 12 words) as evidence

**Theme Codebook:**
- "Reliability & Accuracy": Device reliability issues, inaccurate/inconsistent readings, things not working
- "Integration & Interoperability": Multiple apps, wanting one unified system, FindAir vs study app issues
- "Reminders & Motivation": Wanting nudges, targets, goals, weekly improvement summaries, exercise prompts
- "Insights & Self-Discovery": Personal health awareness, learning about triggers, "opened my eyes", surprise at patterns
- "Trends & Visualization": Charts over time, historical comparisons, seeing patterns over weeks/months
- "Clinical Care Value": Value for GP/nurse appointments, sharing data with healthcare providers
- "Positive Experience & Gratitude": Thanks, success stories, overall satisfaction, "complete success"

**Rules:**
- Don't infer meaning from redacted/unclear text (marked with ...)
- Only tag themes with clear textual evidence
- A feedback can have 0 themes if nothing applies, or many themes if all are present

**Output JSON:**
{
  "sentiment": "Positive|Neutral|Negative",
  "themes": [
    {"code": "Theme Name", "evidence": "brief quote from text"},
    ...
  ]
}
"""
    
    personas = {
        "neutral": {"prompt": base_prompt + "\n\nBe objective and balanced in your assessment.", "temp": 0.2},
        "skeptic": {"prompt": base_prompt + "\n\nBe slightly critical. Look for potential issues or negative undertones.", "temp": 0.4},
        "optimist": {"prompt": base_prompt + "\n\nBe slightly generous. Look for positive aspects or constructive intent.", "temp": 0.4}
    }
    
    def get_single_classification(text, persona_prompt, temperature):
        if not text or not isinstance(text, str):
            return None
        try:
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": persona_prompt},
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
                seed=42
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    # Store individual persona ratings for Fleiss' Kappa
    all_persona_sentiments = []  # List of dicts: {user_key, neutral, skeptic, optimist}
    
    results_list = []
    
    for idx, row in feedback_df.iterrows():
        text = row['combined_feedback']
        user_key = row['user_key']
        
        print(f"  Processing Patient {user_key}...")
        
        persona_results = {}
        for persona_name, config in personas.items():
            result = get_single_classification(text, config["prompt"], config["temp"])
            persona_results[persona_name] = result
        
        # Store individual sentiment ratings
        sentiment_record = {'user_key': user_key}
        for persona_name, result in persona_results.items():
            if result and result.get('sentiment') in sentiment_categories:
                sentiment_record[persona_name] = result.get('sentiment')
            else:
                sentiment_record[persona_name] = 'Neutral'  # Default
        all_persona_sentiments.append(sentiment_record)
        
        # Aggregate for final classification (majority vote)
        valid_results = [r for r in persona_results.values() if r]
        
        if not valid_results:
            results_list.append({
                'user_key': user_key,
                'final_sentiment': 'Neutral',
                'themes': [],
                'evidence': {}
            })
            continue
        
        # Sentiment majority vote
        sentiments = [r.get('sentiment') for r in valid_results if r.get('sentiment') in sentiment_categories]
        final_sentiment = Counter(sentiments).most_common(1)[0][0] if sentiments else "Neutral"
        
        # Theme aggregation (present if >=2/3 personas)
        theme_counts = {t: 0 for t in theme_list}
        theme_evidence = {t: [] for t in theme_list}
        
        for r in valid_results:
            themes = r.get('themes', [])
            if isinstance(themes, list):
                for t in themes:
                    code = t.get('code', '') if isinstance(t, dict) else t
                    evidence = t.get('evidence', '') if isinstance(t, dict) else ''
                    if code in theme_list:
                        theme_counts[code] += 1
                        if evidence:
                            theme_evidence[code].append(evidence)
        
        threshold = len(valid_results) / 2
        final_themes = [t for t in theme_list if theme_counts[t] >= threshold]
        final_evidence = {t: theme_evidence[t][0] for t in final_themes if theme_evidence[t]}
        
        results_list.append({
            'user_key': user_key,
            'final_sentiment': final_sentiment,
            'themes': final_themes,
            'evidence': final_evidence
        })
    
    # Calculate Fleiss' Kappa for internal reliability
    print("\nCalculating internal reliability (Fleiss' Kappa across 3 personas)...")
    
    # Build ratings matrix for Fleiss' Kappa
    # Shape: (n_subjects, n_categories) where each cell = count of raters assigning that category
    sentiment_to_idx = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    n_subjects = len(all_persona_sentiments)
    n_categories = 3
    ratings_matrix = np.zeros((n_subjects, n_categories))
    
    for i, record in enumerate(all_persona_sentiments):
        for persona in ['neutral', 'skeptic', 'optimist']:
            sent = record.get(persona, 'Neutral')
            ratings_matrix[i, sentiment_to_idx[sent]] += 1
    
    internal_kappa = fleiss_kappa(ratings_matrix)
    print(f"  Fleiss' Kappa (3 personas): {internal_kappa:.3f}")
    
    # Add results to dataframe
    for result in results_list:
        mask = feedback_df['user_key'] == result['user_key']
        feedback_df.loc[mask, 'llm_sentiment'] = result['final_sentiment']
        feedback_df.loc[mask, 'llm_themes'] = json.dumps(result['themes'])
        feedback_df.loc[mask, 'llm_evidence'] = json.dumps(result['evidence'])
        
        # Also add individual theme columns
        for t in theme_list:
            col_name = 'theme_' + t.lower().replace(' & ', '_').replace(' ', '_')
            feedback_df.loc[mask, col_name] = t in result['themes']
    
    # Save persona-level ratings for supplementary
    persona_df = pd.DataFrame(all_persona_sentiments)
    persona_df.to_csv(SUPP_DIR / 'persona_level_ratings.csv', index=False)
    
    return feedback_df, internal_kappa


def embedding_sentiment_classifier(feedback_df, client):
    """Embedding-based sentiment classifier."""
    print("\n" + "="*60)
    print("EMBEDDING-BASED SENTIMENT CLASSIFIER")
    print("="*60)
    
    # Get embeddings for feedback
    texts = feedback_df['combined_feedback'].tolist()
    print(f"Getting embeddings for {len(texts)} feedback entries...")
    
    embeddings = []
    for text in texts:
        response = client.embeddings.create(model="text-embedding-3-large", input=text)
        embeddings.append(np.array(response.data[0].embedding))
    embeddings = np.array(embeddings)
    
    # Get reference embeddings
    reference_texts = {
        'Positive': "This is excellent, I love it, very helpful and useful, great experience, amazing.",
        'Neutral': "It was okay, average, nothing special, acceptable, standard experience.",
        'Negative': "This is terrible, frustrating, doesn't work, very disappointing, awful experience."
    }
    
    ref_embeddings = {}
    for sentiment, text in reference_texts.items():
        response = client.embeddings.create(model="text-embedding-3-large", input=text)
        ref_embeddings[sentiment] = np.array(response.data[0].embedding)
    
    # Classify based on cosine similarity
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    classifications = []
    for emb in embeddings:
        similarities = {sent: cosine_similarity(emb, ref_emb) for sent, ref_emb in ref_embeddings.items()}
        classifications.append(max(similarities, key=similarities.get))
    
    feedback_df['embedding_sentiment'] = classifications
    
    print("\nEmbedding classification results:")
    print(feedback_df[['user_key', 'group', 'embedding_sentiment']])
    
    return feedback_df, embeddings


def calculate_inter_method_agreement(feedback_df):
    """Calculate Cohen's Kappa between LLM ensemble and embedding classifier."""
    print("\n" + "="*60)
    print("INTER-METHOD AGREEMENT (Cohen's Kappa)")
    print("="*60)
    
    # Get the two classification columns
    llm_labels = feedback_df['llm_sentiment'].tolist()
    emb_labels = feedback_df['embedding_sentiment'].tolist()
    
    # Calculate Cohen's Kappa
    kappa = cohen_kappa_score(llm_labels, emb_labels)
    
    # Calculate percentage agreement
    agreement = sum(1 for l, e in zip(llm_labels, emb_labels) if l == e) / len(llm_labels) * 100
    
    print(f"  Cohen's Kappa (LLM vs Embedding): {kappa:.3f}")
    print(f"  Percentage Agreement: {agreement:.1f}%")
    
    # Create confusion matrix
    categories = ['Negative', 'Neutral', 'Positive']
    confusion = pd.crosstab(
        pd.Categorical(llm_labels, categories=categories),
        pd.Categorical(emb_labels, categories=categories),
        rownames=['LLM Ensemble'],
        colnames=['Embedding Classifier']
    )
    print("\nConfusion Matrix:")
    print(confusion)
    
    # Save
    confusion.to_csv(MAIN_DIR / 'llm_vs_embedding_confusion.csv')
    
    return kappa, agreement


def lexicon_sentiment_analysis(feedback_df):
    """Lexicon-based analysis for supplementary material."""
    print("\n" + "="*60)
    print("LEXICON-BASED ANALYSIS (for Supplementary Material)")
    print("="*60)
    
    # VADER
    def vader_sentiment(text):
        scores = sia.polarity_scores(text)
        return pd.Series([scores['compound'], scores['pos'], scores['neu'], scores['neg']])
    
    feedback_df[['vader_compound', 'vader_pos', 'vader_neu', 'vader_neg']] = \
        feedback_df['combined_feedback'].apply(vader_sentiment)
    
    # TextBlob
    def textblob_sentiment(text):
        blob = TextBlob(text)
        return pd.Series([blob.sentiment.polarity, blob.sentiment.subjectivity])
    
    feedback_df[['tb_polarity', 'tb_subjectivity']] = \
        feedback_df['combined_feedback'].apply(textblob_sentiment)
    
    # Classify
    def classify_vader(score):
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    
    def classify_textblob(score):
        if score > 0:
            return 'Positive'
        elif score < 0:
            return 'Negative'
        else:
            return 'Neutral'
    
    feedback_df['vader_class'] = feedback_df['vader_compound'].apply(classify_vader)
    feedback_df['tb_class'] = feedback_df['tb_polarity'].apply(classify_textblob)
    
    # Plot for supplementary
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = {GROUP_LABEL: '#1f77b4', 'Remaining': '#d62728'}

    for i, (col, title) in enumerate([
        ('vader_compound', 'VADER Compound'),
        ('tb_polarity', 'TextBlob Polarity'),
        ('tb_subjectivity', 'TextBlob Subjectivity')
    ]):
        for j, group in enumerate([GROUP_LABEL, 'Remaining']):
            data = feedback_df[feedback_df['group'] == group][col]
            bp = axes[i].boxplot([data], positions=[j], widths=0.6, patch_artist=True)
            bp['boxes'][0].set_facecolor(colors[group])
            bp['boxes'][0].set_alpha(0.7)
            bp['medians'][0].set_color('black')
            bp['medians'][0].set_linewidth(2)
        axes[i].set_title(title, fontsize=12, fontweight='bold')
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels([GROUP_LABEL, 'Remaining'])
        axes[i].axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    plt.suptitle('Lexicon-Based Sentiment Analysis (Supplementary)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(SUPP_DIR / 'lexicon_sentiment.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved lexicon plot to {SUPP_DIR / 'lexicon_sentiment.png'}")
    
    return feedback_df


def plot_main_results(feedback_df):
    """Create main text figures."""
    print("\n" + "="*60)
    print("CREATING MAIN TEXT FIGURES")
    print("="*60)
    
    # Figure: LLM Sentiment + Theme Distribution
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = {GROUP_LABEL: '#1f77b4', 'Remaining': '#d62728'}

    # Sentiment Distribution
    sentiment_counts = feedback_df.groupby(['group', 'llm_sentiment']).size().unstack(fill_value=0)
    sentiment_pcts = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0) * 100

    x = np.arange(3)
    width = 0.35

    for i, group in enumerate([GROUP_LABEL, 'Remaining']):
        if group in sentiment_pcts.index:
            values = [sentiment_pcts.loc[group, col] if col in sentiment_pcts.columns else 0 
                      for col in ['Negative', 'Neutral', 'Positive']]
            bars = axes[0].bar(x + (i - 0.5) * width, values, width, label=group, color=colors[group], alpha=0.8)
            for bar, val in zip(bars, values):
                if val > 0:
                    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                                f'{val:.0f}%', ha='center', va='bottom', fontsize=10)
    
    axes[0].set_xlabel('Sentiment', fontsize=11)
    axes[0].set_ylabel('Percentage of Feedback', fontsize=11)
    axes[0].set_title('(a) Sentiment Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['Negative', 'Neutral', 'Positive'])
    axes[0].legend()
    axes[0].set_ylim(0, 100)
    
    # Theme Presence
    theme_cols = [
        'theme_reliability_accuracy',
        'theme_integration_interoperability',
        'theme_reminders_motivation',
        'theme_insights_self-discovery',
        'theme_trends_visualization',
        'theme_clinical_care_value',
        'theme_positive_experience_gratitude'
    ]
    theme_labels = [
        'Reliability\n& Accuracy',
        'Integration\n& Interop',
        'Reminders\n& Motivation',
        'Insights &\nDiscovery',
        'Trends &\nVisualization',
        'Clinical\nCare Value',
        'Positive &\nGratitude'
    ]
    
    x = np.arange(len(theme_cols))
    width = 0.35
    
    for i, group in enumerate([GROUP_LABEL, 'Remaining']):
        group_data = feedback_df[feedback_df['group'] == group]
        n_group = len(group_data)
        values = []
        for col in theme_cols:
            if col in group_data.columns:
                values.append(group_data[col].sum() / n_group * 100)
            else:
                values.append(0)
        bars = axes[1].bar(x + (i - 0.5) * width, values, width, label=group, color=colors[group], alpha=0.8)
        for bar, val in zip(bars, values):
            if val > 0:
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                            f'{val:.0f}%', ha='center', va='bottom', fontsize=8)
    
    axes[1].set_xlabel('Theme', fontsize=11)
    axes[1].set_ylabel('% of Group with Theme Present', fontsize=11)
    axes[1].set_title('(b) Theme Presence by Group', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(theme_labels, fontsize=8)
    axes[1].legend()
    axes[1].set_ylim(0, 130)
    
    plt.suptitle('LLM-Assisted Thematic Analysis of User Feedback', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(MAIN_DIR / 'llm_thematic_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved main figure to {MAIN_DIR / 'llm_thematic_analysis.png'}")


def generate_summary(feedback_df, internal_kappa, inter_method_kappa, inter_method_agreement):
    """Generate summary for main text and supplementary."""
    print("\n" + "="*60)
    print("GENERATING SUMMARY")
    print("="*60)
    
    optimal = feedback_df[feedback_df['group'] == GROUP_LABEL]
    remaining = feedback_df[feedback_df['group'] == 'Remaining']
    
    theme_list = [
        'Reliability & Accuracy',
        'Integration & Interoperability',
        'Reminders & Motivation',
        'Insights & Self-Discovery',
        'Trends & Visualization',
        'Clinical Care Value',
        'Positive Experience & Gratitude'
    ]
    
    # Build theme table
    theme_table = f"| Theme | {GROUP_LABEL} | Remaining | Difference |\n|-------|---------|-----------|------------|\n"
    for theme in theme_list:
        col_name = 'theme_' + theme.lower().replace(' & ', '_').replace(' ', '_').replace('-', '')
        # Try to find column
        matching_cols = [c for c in feedback_df.columns if theme.split()[0].lower() in c.lower()]
        if matching_cols:
            col = matching_cols[0]
            opt_pct = optimal[col].mean() * 100 if col in optimal.columns else 0
            rem_pct = remaining[col].mean() * 100 if col in remaining.columns else 0
            diff = opt_pct - rem_pct
            diff_str = f"+{diff:.0f}%" if diff > 0 else f"{diff:.0f}%"
            theme_table += f"| {theme} | {opt_pct:.0f}% | {rem_pct:.0f}% | {diff_str} |\n"
    
    summary = f"""
SENTIMENT ANALYSIS SUMMARY
===================================================================

METHODOLOGY ALIGNMENT:
A. Preliminary Semantic Clustering: See inductive_clustering_analysis.py results
B. Multi-Agent LLM Ensemble: 3 personas (Neutral, Skeptic, Optimist), 7 themes
C. Embedding-Based Sentiment Classifier: Cosine similarity to reference sentiments
D. Inter-Method Agreement: Cohen's Kappa between B and C

PATIENT GROUPS:
- {GROUP_LABEL} (n={len(optimal)}): {sorted(optimal['user_key'].tolist())}
- Remaining (n=6): {sorted(remaining['user_key'].tolist())}
- Total: {len(feedback_df)} unique patients with feedback

INTERNAL RELIABILITY (Section B):
- Fleiss' Kappa across 3 LLM personas: {internal_kappa:.3f}
  (Interpretation: {"Slight" if internal_kappa < 0.2 else "Fair" if internal_kappa < 0.4 else "Moderate" if internal_kappa < 0.6 else "Substantial"} agreement)

INTER-METHOD AGREEMENT (Section D):
- Cohen's Kappa (LLM vs Embedding): {inter_method_kappa:.3f}
  (Interpretation: {"Slight" if inter_method_kappa < 0.2 else "Fair" if inter_method_kappa < 0.4 else "Moderate" if inter_method_kappa < 0.6 else "Substantial"} agreement)
- Percentage Agreement: {inter_method_agreement:.1f}%

LLM ENSEMBLE SENTIMENT RESULTS:
{GROUP_LABEL} Group: {dict(optimal['llm_sentiment'].value_counts())}
Remaining Group: {dict(remaining['llm_sentiment'].value_counts())}

EMBEDDING CLASSIFIER SENTIMENT RESULTS:
{GROUP_LABEL} Group: {dict(optimal['embedding_sentiment'].value_counts())}
Remaining Group: {dict(remaining['embedding_sentiment'].value_counts())}

THEME PRESENCE (LLM Ensemble):
{theme_table}

EVIDENCE QUOTES BY PATIENT:
"""
    
    for idx, row in feedback_df.iterrows():
        evidence = json.loads(row['llm_evidence']) if row.get('llm_evidence') else {}
        group_label = f"★ {GROUP_LABEL.upper()}" if row['group'] == GROUP_LABEL else "  Remaining"
        summary += f"\n{group_label} Patient {row['user_key']}:\n"
        if evidence:
            for theme, quote in evidence.items():
                summary += f"  - {theme}: \"{quote}\"\n"
        else:
            summary += f"  - (no theme evidence recorded)\n"
    
    print(summary)
    
    # Save
    with open(MAIN_DIR / 'analysis_summary.txt', 'w') as f:
        f.write(summary)
    
    # Save full results
    feedback_df.to_csv(OUTPUT_DIR / 'full_results.csv', index=False)
    
    print(f"\nResults saved to {OUTPUT_DIR}/")


def compute_fisher_exact_tests(feedback_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Fisher's Exact Tests for theme presence between groups.
    
    Tests whether each theme is significantly more/less present in the
    optimal group compared to the remaining group.
    
    Outputs results to supplementary folder.
    """
    print("\n" + "="*60)
    print("FISHER'S EXACT TESTS FOR THEME PRESENCE")
    print("="*60)
    
    # Get group sizes
    optimal_df = feedback_df[feedback_df['group'] == GROUP_LABEL]
    remaining_df = feedback_df[feedback_df['group'] == 'Remaining']
    n_opt = len(optimal_df)
    n_rem = len(remaining_df)
    
    print(f"{GROUP_LABEL} group: n={n_opt}")
    print(f"Remaining group: n={n_rem}")
    
    # Theme columns
    theme_cols = [col for col in feedback_df.columns if col.startswith('theme_')]
    theme_names = {
        'theme_reliability_accuracy': 'Reliability & Accuracy',
        'theme_integration_interoperability': 'Integration & Interoperability',
        'theme_reminders_motivation': 'Reminders & Motivation',
        'theme_insights_self-discovery': 'Insights & Self-Discovery',
        'theme_trends_visualization': 'Trends & Visualization',
        'theme_clinical_care_value': 'Clinical Care Value',
        'theme_positive_experience_gratitude': 'Positive Experience & Gratitude'
    }
    
    results = []
    for col in theme_cols:
        theme_display = theme_names.get(col, col)
        
        # Count presence in each group
        opt_yes = optimal_df[col].sum()
        opt_no = n_opt - opt_yes
        rem_yes = remaining_df[col].sum()
        rem_no = n_rem - rem_yes
        
        # 2x2 contingency table
        table = [[opt_yes, opt_no], [rem_yes, rem_no]]
        
        # Fisher's exact test
        odds_ratio, p_value = stats.fisher_exact(table)
        
        results.append({
            'Theme': theme_display,
            GROUP_LABEL: f"{int(opt_yes)}/{n_opt} ({100*opt_yes/n_opt:.0f}%)",
            'Remaining': f"{int(rem_yes)}/{n_rem} ({100*rem_yes/n_rem:.0f}%)",
            'Odds Ratio': round(odds_ratio, 4) if odds_ratio != float('inf') else 'inf',
            'p-value': round(p_value, 4),
            'Significant (p<0.05)': 'Yes' if p_value < 0.05 else 'No'
        })
        
        sig_marker = '*' if p_value < 0.05 else ''
        print(f"  {theme_display}: p={p_value:.4f}{sig_marker}")
    
    # Create DataFrame and save
    results_df = pd.DataFrame(results)
    output_path = SUPP_DIR / 'fisher_exact_tests.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    
    # Summary
    all_p = [r['p-value'] for r in results]
    print(f"\nSummary: All p > 0.05? {all(p > 0.05 for p in all_p)}")
    min_p = min(all_p)
    min_theme = results[all_p.index(min_p)]['Theme']
    print(f"Minimum p-value: {min_p:.4f} ({min_theme})")
    
    return results_df


def parse_args():
    parser = argparse.ArgumentParser(description="Sentiment analysis.")
    parser.add_argument("--group", choices=list(GROUPS.keys()), default="concordant",
                        help="Patient group definition to use (default: concordant)")
    add_world_argument(parser)
    args = parser.parse_args()
    set_active_world(args.world)
    return args


def main():
    """Main execution."""
    global OUTPUT_DIR, MAIN_DIR, SUPP_DIR, CONCORDANT_USERS, REMAINING_ASSESSED, ALL_INCLUDED, GROUP_LABEL

    args = parse_args()
    group_cfg = get_group_config(args.group)
    CONCORDANT_USERS = group_cfg["patients"]
    GROUP_LABEL = group_cfg["label"]

    # Determine which concordant patients have feedback
    concordant_with_feedback = [p for p in CONCORDANT_USERS if p in FEEDBACK_PATIENTS]
    # Remaining = feedback patients who are NOT in the concordant group
    REMAINING_ASSESSED = [p for p in FEEDBACK_PATIENTS if p not in CONCORDANT_USERS]
    ALL_INCLUDED = concordant_with_feedback + REMAINING_ASSESSED

    OUTPUT_DIR = get_output_dir(args.group, "sentiment_analysis")
    record_run_provenance(OUTPUT_DIR, group_name=args.group, script="sentiment_analysis.py")
    MAIN_DIR = OUTPUT_DIR / 'main_text'
    SUPP_DIR = OUTPUT_DIR / 'supplementary'
    MAIN_DIR.mkdir(exist_ok=True)
    SUPP_DIR.mkdir(exist_ok=True)

    print("="*60)
    print(f"SENTIMENT ANALYSIS - {GROUP_LABEL} vs Remaining")
    print("="*60)
    print(f"{GROUP_LABEL} patients: {CONCORDANT_USERS}")
    print(f"{GROUP_LABEL} with feedback: {concordant_with_feedback}")
    print(f"Remaining with feedback: {REMAINING_ASSESSED}")

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    feedback_df = load_feedback_data()

    if len(feedback_df) == 0:
        print("No feedback data found!")
        return

    feedback_df, internal_kappa = llm_ensemble_with_internal_reliability(feedback_df, client)
    feedback_df, embeddings = embedding_sentiment_classifier(feedback_df, client)
    inter_method_kappa, inter_method_agreement = calculate_inter_method_agreement(feedback_df)
    plot_main_results(feedback_df)
    feedback_df = lexicon_sentiment_analysis(feedback_df)
    compute_fisher_exact_tests(feedback_df)
    generate_summary(feedback_df, internal_kappa, inter_method_kappa, inter_method_agreement)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"Main text outputs: {MAIN_DIR}/")
    print(f"Supplementary outputs: {SUPP_DIR}/")


if __name__ == "__main__":
    main()

