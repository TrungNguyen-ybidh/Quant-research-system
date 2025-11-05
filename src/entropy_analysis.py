"""
Entropy Analysis for Indicator Quality Assessment

This module calculates Shannon entropy for trading indicator signals to assess
the consistency and predictability of return distributions. Lower entropy
indicates more consistent, predictable returns, which is desirable for trading signals.
"""

import numpy as np
import pandas as pd
from math import log2
import os
import sys

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


# ============================================================================
# STEP 2: Function Blueprints
# ============================================================================

def categorize_returns(returns, bins):
    """
    Assign each return into one of 5 categories.
    
    Categories:
    1. Large Down (< -2%)
    2. Small Down (-2% to -0.5%)
    3. Flat (-0.5% to +0.5%)
    4. Small Up (+0.5% to +2%)
    5. Large Up (> +2%)
    
    Args:
        returns: Array of return percentages
        bins: List of bin edges [min, -2, -0.5, 0.5, 2, max]
    
    Returns:
        Array of category labels (1-5)
    """
    categories = np.digitize(returns, bins, right=False)
    # Adjust for edge cases
    categories = np.clip(categories, 1, 5)
    return categories


def calculate_entropy(counts):
    """
    Apply Shannon entropy formula: H = -Σ (p_i * log2(p_i))
    
    Args:
        counts: Array of category counts [n1, n2, n3, n4, n5]
    
    Returns:
        Shannon entropy value
    """
    total = np.sum(counts)
    if total == 0:
        return 0.0
    
    # Calculate probabilities
    probs = counts / total
    
    # Calculate entropy (avoid log(0))
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * log2(p)
    
    return entropy


def normalize_entropy(H, n_categories):
    """
    Scale entropy to 0-1 range by dividing by log2(n_categories).
    
    Args:
        H: Raw entropy value
        n_categories: Number of categories (5)
    
    Returns:
        Normalized entropy (0-1)
    """
    max_entropy = log2(n_categories)
    return H / max_entropy if max_entropy > 0 else 0.0


def assign_quality_rating(H_norm):
    """
    Map normalized entropy to quality rating with stars.
    
    Rating scale:
    - 0.0 - 0.4: Excellent (⭐⭐⭐⭐⭐)
    - 0.4 - 0.6: Good (⭐⭐⭐⭐)
    - 0.6 - 0.8: Moderate (⭐⭐⭐)
    - 0.8 - 0.9: Moderate-Poor (⭐⭐)
    - > 0.9: Poor (⭐)
    
    Args:
        H_norm: Normalized entropy (0-1)
    
    Returns:
        Tuple of (stars string, quality rating string)
    """
    if H_norm <= 0.4:
        return ("⭐⭐⭐⭐⭐", "Excellent")
    elif H_norm <= 0.6:
        return ("⭐⭐⭐⭐", "Good")
    elif H_norm <= 0.8:
        return ("⭐⭐⭐", "Moderate")
    elif H_norm <= 0.9:
        return ("⭐⭐", "Moderate-Poor")
    else:
        return ("⭐", "Poor")


def entropy_for_indicator(signals_df):
    """
    Orchestrator function: categorize → count → entropy → normalize → rating.
    
    Args:
        signals_df: DataFrame with 'return_6h' column
    
    Returns:
        Dictionary with entropy metrics and details
    """
    returns = signals_df['return_6h'].values
    
    # Define bin edges for 5 categories
    # Category 1: < -2%
    # Category 2: -2% to -0.5%
    # Category 3: -0.5% to +0.5%
    # Category 4: +0.5% to +2%
    # Category 5: > +2%
    bins = [-np.inf, -2.0, -0.5, 0.5, 2.0, np.inf]
    
    # Categorize returns
    categories = categorize_returns(returns, bins)
    
    # Count frequencies for each category
    counts = np.array([
        np.sum(categories == 1),  # Large Down
        np.sum(categories == 2),  # Small Down
        np.sum(categories == 3),  # Flat
        np.sum(categories == 4),  # Small Up
        np.sum(categories == 5)   # Large Up
    ])
    
    # Calculate probabilities
    total = np.sum(counts)
    probs = counts / total if total > 0 else np.zeros(5)
    
    # Calculate raw entropy
    raw_entropy = calculate_entropy(counts)
    
    # Normalize entropy
    n_categories = 5
    norm_entropy = normalize_entropy(raw_entropy, n_categories)
    
    # Assign quality rating
    stars, rating = assign_quality_rating(norm_entropy)
    
    return {
        'counts': counts,
        'probabilities': probs,
        'raw_entropy': raw_entropy,
        'norm_entropy': norm_entropy,
        'quality_stars': stars,
        'quality_rating': rating,
        'sample_size': total
    }


# ============================================================================
# STEP 3: Main Analysis Function
# ============================================================================

def run_entropy_analysis():
    """
    Main function to calculate entropy for all indicators.
    
    Creates:
    - indicator_entropy_scores.csv
    - entropy_calculations.txt
    - indicator_quality_ranking.csv
    - indicator_ranking_report.txt
    """
    print("=" * 80)
    print("ENTROPY ANALYSIS FOR INDICATOR QUALITY ASSESSMENT")
    print("=" * 80)
    
    # Load signal details
    signals_path = os.path.join(config.PROCESSED_DATA_PATH, "indicator_signal_details.csv")
    print(f"\nLoading signals from: {signals_path}")
    signals = pd.read_csv(signals_path)
    print(f"Loaded {len(signals)} signals")
    
    # Group by indicator + signal_type
    grouped = signals.groupby(['indicator', 'signal_type'])
    
    # Storage for results
    entropy_results = []
    detailed_calculations = []
    
    print("\nCalculating entropy for each indicator...")
    print("-" * 80)
    
    # Process each indicator group
    for (indicator, signal_type), group_df in grouped:
        print(f"Processing: {indicator} - {signal_type} ({len(group_df)} signals)")
        
        # Calculate entropy
        result = entropy_for_indicator(group_df)
        
        # Store results
        entropy_results.append({
            'indicator': indicator,
            'signal_type': signal_type,
            'sample_size': result['sample_size'],
            'raw_entropy': result['raw_entropy'],
            'norm_entropy': result['norm_entropy'],
            'quality_stars': result['quality_stars'],
            'quality_rating': result['quality_rating']
        })
        
        # Store detailed calculations
        detailed_calculations.append({
            'indicator': indicator,
            'signal_type': signal_type,
            'result': result
        })
    
    # Create entropy scores DataFrame
    entropy_df = pd.DataFrame(entropy_results)
    
    # Merge with performance metrics from indicator_test_results.csv
    test_results_path = os.path.join(config.PROCESSED_DATA_PATH, "indicator_test_results.csv")
    print(f"\nLoading test results from: {test_results_path}")
    test_results = pd.read_csv(test_results_path)
    
    # Merge on indicator and signal_type
    # Note: signal_type names might need mapping
    # Map signal_type names to match
    signal_type_mapping = {
        'MACD_Bearish_Cross': 'MACD_Bearish_Cross',
        'MACD_Bullish_Cross': 'MACD_Bullish_Cross',
        'RSI_Overbought': 'RSI_Overbought',
        'RSI_Oversold': 'RSI_Oversold',
        'SMA_50_Bounce': 'SMA_50_Bounce'
    }
    
    # Merge entropy with test results
    merged_df = entropy_df.merge(
        test_results,
        on=['indicator', 'signal_type'],
        how='left'
    )
    
    # Create combined entropy scores CSV
    entropy_scores_df = merged_df[[
        'indicator', 'signal_type', 'sample_size',
        'win_rate_pct', 'avg_return_pct',
        'raw_entropy', 'norm_entropy', 'quality_stars', 'quality_rating'
    ]].copy()
    
    entropy_scores_df.columns = [
        'indicator', 'signal_type', 'sample_size',
        'win_rate', 'avg_return',
        'raw_entropy', 'norm_entropy', 'quality_stars', 'quality_rating'
    ]
    
    # Save entropy scores
    entropy_scores_path = os.path.join(config.PROCESSED_DATA_PATH, "indicator_entropy_scores.csv")
    entropy_scores_df.to_csv(entropy_scores_path, index=False)
    print(f"\n✓ Saved: {entropy_scores_path}")
    
    # Write detailed calculations
    write_detailed_calculations(detailed_calculations)
    
    # Create quality ranking
    final_ranking = create_quality_ranking(merged_df)
    
    # Write report
    write_ranking_report(entropy_scores_df, merged_df)
    
    print("\n" + "=" * 80)
    print("ENTROPY ANALYSIS COMPLETE")
    print("=" * 80)


def write_detailed_calculations(detailed_calculations):
    """Write detailed entropy calculations to text file."""
    output_path = os.path.join(config.PROCESSED_DATA_PATH, "entropy_calculations.txt")
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DETAILED ENTROPY CALCULATIONS\n")
        f.write("=" * 80 + "\n\n")
        
        category_labels = [
            "Large Down (< -2%)",
            "Small Down (-2% to -0.5%)",
            "Flat (-0.5% to +0.5%)",
            "Small Up (+0.5% to +2%)",
            "Large Up (> +2%)"
        ]
        
        for calc in detailed_calculations:
            indicator = calc['indicator']
            signal_type = calc['signal_type']
            result = calc['result']
            
            f.write(f"\n{indicator} - {signal_type}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Sample Size: {result['sample_size']}\n\n")
            
            f.write("Category Counts:\n")
            for i, (label, count) in enumerate(zip(category_labels, result['counts'])):
                f.write(f"  {i+1}. {label}: {int(count)}\n")
            
            f.write("\nProbabilities:\n")
            for i, (label, prob) in enumerate(zip(category_labels, result['probabilities'])):
                f.write(f"  {i+1}. {label}: {prob:.4f} ({prob*100:.2f}%)\n")
            
            f.write(f"\nRaw Entropy = {result['raw_entropy']:.4f}\n")
            f.write(f"Normalized Entropy = {result['norm_entropy']:.4f}\n")
            f.write(f"Quality = {result['quality_stars']} ({result['quality_rating']})\n")
            f.write("\n")
    
    print(f"✓ Saved: {output_path}")


def create_quality_ranking(merged_df):
    """Create final quality ranking combining entropy, win rate, and p-value."""
    # Ranking rules:
    # Primary: Lowest normalized entropy (best consistency)
    # Secondary: Highest win rate
    # Tertiary: Lowest p-value
    
    # Create ranking
    ranking_df = merged_df.copy()
    
    # Sort by: entropy (ascending), win_rate (descending), p_value (ascending)
    ranking_df = ranking_df.sort_values(
        by=['norm_entropy', 'win_rate_pct', 'p_value'],
        ascending=[True, False, True]
    ).reset_index(drop=True)
    
    # Add rank column
    ranking_df.insert(0, 'rank', range(1, len(ranking_df) + 1))
    
    # Create final ranking table
    final_ranking = ranking_df[[
        'rank', 'indicator', 'signal_type',
        'win_rate_pct', 'avg_return_pct',
        'norm_entropy', 'quality_stars', 'quality_rating',
        'p_value', 'statistically_significant'
    ]].copy()
    
    # Rename columns for readability
    final_ranking.columns = [
        'Rank', 'Indicator', 'Signal',
        'Win%', 'Avg Return',
        'Entropy', 'Quality Stars', 'Quality Rating',
        'p-value', 'Significant'
 ]
    
    # Format columns
    final_ranking['Win%'] = final_ranking['Win%'].round(1)
    final_ranking['Avg Return'] = final_ranking['Avg Return'].round(2)
    final_ranking['Entropy'] = final_ranking['Entropy'].round(3)
    final_ranking['p-value'] = final_ranking['p-value'].apply(lambda x: f"{x:.6f}" if x > 0 else "0.0")
    final_ranking['Significant'] = final_ranking['Significant'].map({True: 'Yes', False: 'No'})
    
    # Format avg return as percentage
    final_ranking['Avg Return'] = final_ranking['Avg Return'].apply(
        lambda x: f"+{x:.2f}%" if x >= 0 else f"{x:.2f}%"
    )
    
    # Save ranking
    output_path = os.path.join(config.PROCESSED_DATA_PATH, "indicator_quality_ranking.csv")
    final_ranking.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path}")
    
    return final_ranking


def write_ranking_report(entropy_scores_df, merged_df):
    """Write narrative ranking report."""
    output_path = os.path.join(config.PROCESSED_DATA_PATH, "indicator_ranking_report.txt")
    
    # Get ranking for report
    ranking_df = merged_df.sort_values(
        by=['norm_entropy', 'win_rate_pct', 'p_value'],
        ascending=[True, False, True]
    ).reset_index(drop=True)
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("INDICATOR QUALITY RANKING REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Introduction
        f.write("ENTROPY ANALYSIS OVERVIEW\n")
        f.write("-" * 80 + "\n\n")
        f.write(
            "Shannon entropy quantifies the consistency and predictability of return "
            "distributions for trading signals. Lower entropy indicates more consistent, "
            "predictable returns, which is desirable for trading signals.\n\n"
        )
        f.write(
            "Entropy measures how evenly distributed the returns are across 5 categories:\n"
            "1. Large Down (< -2%)\n"
            "2. Small Down (-2% to -0.5%)\n"
            "3. Flat (-0.5% to +0.5%)\n"
            "4. Small Up (+0.5% to +2%)\n"
            "5. Large Up (> +2%)\n\n"
        )
        f.write(
            "A perfectly uniform distribution (all categories equally likely) would have "
            "maximum entropy (1.0). A distribution that concentrates returns in fewer "
            "categories has lower entropy, indicating more predictable outcomes.\n\n"
        )
        f.write(
            "Note: Entropy quantifies consistency, not accuracy. A signal with low entropy "
            "may consistently lose money, but it's still predictable. We combine entropy "
            "with win rate and statistical significance for a complete quality assessment.\n\n"
        )
        
        # Individual indicator summaries
        f.write("INDICATOR RELIABILITY SUMMARIES\n")
        f.write("-" * 80 + "\n\n")
        
        for _, row in entropy_scores_df.iterrows():
            indicator = row['indicator']
            signal = row['signal_type']
            win_rate = row['win_rate']
            avg_return = row['avg_return']
            entropy = row['norm_entropy']
            stars = row['quality_stars']
            rating = row['quality_rating']
            
            f.write(f"{indicator} - {signal}\n")
            f.write(f"  Win Rate: {win_rate:.1f}% | Avg Return: {avg_return:.2f}%\n")
            f.write(f"  Entropy: {entropy:.3f} | Quality: {stars} ({rating})\n")
            
            # Interpretation (matching quality rating thresholds)
            if entropy <= 0.4:
                f.write(f"  → Excellent consistency; returns are highly predictable.\n")
            elif entropy <= 0.6:
                f.write(f"  → Good consistency; returns show moderate predictability.\n")
            elif entropy <= 0.8:
                f.write(f"  → Moderate consistency; returns are somewhat unpredictable.\n")
            elif entropy <= 0.9:
                f.write(f"  → Moderate-Poor consistency; returns are quite unpredictable.\n")
            else:
                f.write(f"  → Poor consistency; returns are highly unpredictable.\n")
            f.write("\n")
        
        # Top performers
        f.write("\n" + "=" * 80 + "\n")
        f.write("TOP PERFORMERS\n")
        f.write("=" * 80 + "\n\n")
        
        top_performers = ranking_df.head(3)
        for idx, (_, row) in enumerate(top_performers.iterrows(), 1):
            indicator = row['indicator']
            signal = row['signal_type']
            win_rate = row['win_rate_pct']
            avg_return = row['avg_return_pct']
            entropy = row['norm_entropy']
            stars = row['quality_stars']
            significant = "Yes" if row['statistically_significant'] else "No"
            
            f.write(f"{idx}. {indicator} - {signal}\n")
            f.write(f"   Entropy: {entropy:.3f} ({stars}) | Win Rate: {win_rate:.1f}% | ")
            f.write(f"Avg Return: {avg_return:+.2f}% | Significant: {significant}\n")
            f.write(f"   → Strong candidate with consistent, predictable returns.\n\n")
        
        # Signals to avoid
        f.write("\n" + "=" * 80 + "\n")
        f.write("SIGNALS TO AVOID\n")
        f.write("=" * 80 + "\n\n")
        
        bottom_performers = ranking_df.tail(2)
        for idx, (_, row) in enumerate(bottom_performers.iterrows(), 1):
            indicator = row['indicator']
            signal = row['signal_type']
            win_rate = row['win_rate_pct']
            avg_return = row['avg_return_pct']
            entropy = row['norm_entropy']
            stars = row['quality_stars']
            significant = "Yes" if row['statistically_significant'] else "No"
            
            f.write(f"{idx}. {indicator} - {signal}\n")
            f.write(f"   Entropy: {entropy:.3f} ({stars}) | Win Rate: {win_rate:.1f}% | ")
            f.write(f"Avg Return: {avg_return:+.2f}% | Significant: {significant}\n")
            f.write(f"   → High entropy indicates unreliable, unpredictable returns.\n\n")
        
        # Key takeaways
        f.write("\n" + "=" * 80 + "\n")
        f.write("KEY TAKEAWAYS FOR GOLD 1-HOUR TIMEFRAME\n")
        f.write("=" * 80 + "\n\n")
        
        best_signal = ranking_df.iloc[0]
        worst_signal = ranking_df.iloc[-1]
        
        f.write(f"1. Best Signal: {best_signal['indicator']} - {best_signal['signal_type']}\n")
        f.write(f"   This signal shows the lowest entropy ({best_signal['norm_entropy']:.3f}), ")
        f.write(f"indicating highly consistent return patterns. Combined with a ")
        f.write(f"{best_signal['win_rate_pct']:.1f}% win rate, this is the most reliable signal.\n\n")
        
        f.write(f"2. Entropy vs Win Rate Trade-off:\n")
        f.write(f"   Some signals may have moderate entropy but high win rates (or vice versa). ")
        f.write(f"Consider the full ranking table to find the best balance for your risk tolerance.\n\n")
        
        f.write(f"3. Statistical Significance Matters:\n")
        significant_count = ranking_df['statistically_significant'].sum()
        f.write(f"   {significant_count} out of {len(ranking_df)} signals are statistically significant. ")
        f.write(f"Prioritize signals that are both consistent (low entropy) and statistically significant.\n\n")
        
        f.write(f"4. Worst Signal: {worst_signal['indicator']} - {worst_signal['signal_type']}\n")
        f.write(f"   This signal has the highest entropy ({worst_signal['norm_entropy']:.3f}), ")
        f.write(f"indicating unpredictable returns. Avoid this signal or use with extreme caution.\n\n")
    
    print(f"✓ Saved: {output_path}")


if __name__ == "__main__":
    run_entropy_analysis()

