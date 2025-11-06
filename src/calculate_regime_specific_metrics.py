"""
Calculate Regime-Specific Performance Metrics for Indicators

This script calculates performance metrics (win rate, avg return, sample size)
for each indicator signal separately in each market regime (up, down, range).

It matches indicator signals to ML regime predictions by timestamp and
calculates metrics for each indicator-regime combination.
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def load_indicator_signals() -> pd.DataFrame:
    """Load indicator signal details."""
    signals_path = os.path.join(config.PROCESSED_DATA_PATH, 'indicator_signal_details.csv')
    
    if not os.path.exists(signals_path):
        raise FileNotFoundError(f"Indicator signals file not found: {signals_path}")
    
    df = pd.read_csv(signals_path, parse_dates=['signal_timestamp'])
    return df


def load_regime_predictions() -> pd.DataFrame:
    """Load ML regime predictions."""
    predictions_path = os.path.join(config.PROCESSED_DATA_PATH, 'regime_predictions.csv')
    
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Regime predictions file not found: {predictions_path}")
    
    df = pd.read_csv(predictions_path, parse_dates=['timestamp'], index_col='timestamp')
    return df


def match_signals_to_regimes(signals_df: pd.DataFrame, regimes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match indicator signals to ML regime predictions by timestamp.
    
    Args:
        signals_df: DataFrame with indicator signals
        regimes_df: DataFrame with ML regime predictions (indexed by timestamp)
        
    Returns:
        DataFrame with signals matched to regimes
    """
    # Create a copy of signals
    matched_df = signals_df.copy()
    
    # Set timestamp as index for easier matching
    matched_df = matched_df.set_index('signal_timestamp')
    
    # Match to nearest regime prediction (using merge_asof for time-based matching)
    # First, ensure both are sorted by timestamp
    matched_df = matched_df.sort_index()
    regimes_df = regimes_df.sort_index()
    
    # Merge on timestamp (nearest match)
    matched_df = pd.merge_asof(
        matched_df.reset_index(),
        regimes_df[['ml_prediction_label']].reset_index(),
        left_on='signal_timestamp',
        right_on='timestamp',
        direction='nearest',
        suffixes=('', '_regime')
    )
    
    # Rename ml_prediction_label to regime
    matched_df['regime'] = matched_df['ml_prediction_label']
    
    # Drop temporary columns
    matched_df = matched_df.drop(columns=['timestamp', 'ml_prediction_label'], errors='ignore')
    
    return matched_df


def calculate_regime_metrics(signals_df: pd.DataFrame, indicator: str, 
                            signal_type: str, regime: str) -> Dict:
    """
    Calculate performance metrics for a specific indicator-signal-regime combination.
    
    Args:
        signals_df: DataFrame with matched signals and regimes
        indicator: Indicator name (e.g., 'RSI')
        signal_type: Signal type (e.g., 'RSI_Oversold')
        regime: Regime name ('up', 'down', 'range')
        
    Returns:
        Dictionary with metrics: win_rate, avg_return, sample_size, etc.
    """
    # Filter for this indicator-signal-regime combination
    mask = (
        (signals_df['indicator'] == indicator) &
        (signals_df['signal_type'] == signal_type) &
        (signals_df['regime'] == regime)
    )
    
    group = signals_df[mask].copy()
    
    if len(group) == 0:
        return {
            'indicator': indicator,
            'signal_type': signal_type,
            'regime': regime,
            'sample_size': 0,
            'win_rate_pct': 0.0,
            'avg_return_pct': 0.0,
            'median_return_pct': 0.0,
            'std_return_pct': 0.0,
            'min_return_pct': 0.0,
            'max_return_pct': 0.0
        }
    
    # Calculate metrics
    total_signals = len(group)
    wins = group['was_profitable'].sum()
    win_rate_pct = (wins / total_signals) * 100 if total_signals > 0 else 0.0
    
    returns = group['return_6h'].dropna()
    avg_return_pct = returns.mean() if len(returns) > 0 else 0.0
    median_return_pct = returns.median() if len(returns) > 0 else 0.0
    std_return_pct = returns.std() if len(returns) > 0 else 0.0
    min_return_pct = returns.min() if len(returns) > 0 else 0.0
    max_return_pct = returns.max() if len(returns) > 0 else 0.0
    
    return {
        'indicator': indicator,
        'signal_type': signal_type,
        'regime': regime,
        'sample_size': int(total_signals),
        'win_rate_pct': round(win_rate_pct, 2),
        'avg_return_pct': round(avg_return_pct, 3),
        'median_return_pct': round(median_return_pct, 3),
        'std_return_pct': round(std_return_pct, 3),
        'min_return_pct': round(min_return_pct, 3),
        'max_return_pct': round(max_return_pct, 3)
    }


def calculate_all_regime_metrics(signals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate regime-specific metrics for all indicator-signal combinations.
    
    Args:
        signals_df: DataFrame with matched signals and regimes
        
    Returns:
        DataFrame with regime-specific metrics
    """
    results = []
    
    # Get all unique indicator-signal combinations
    signal_combos = signals_df.groupby(['indicator', 'signal_type']).size().reset_index()
    
    # Regimes to analyze
    regimes = ['up', 'down', 'range']
    
    print("Calculating regime-specific metrics...")
    print(f"  Total indicator-signal combinations: {len(signal_combos)}")
    print(f"  Regimes: {', '.join(regimes)}")
    print()
    
    for _, row in signal_combos.iterrows():
        indicator = row['indicator']
        signal_type = row['signal_type']
        
        print(f"  Processing: {indicator} - {signal_type}")
        
        for regime in regimes:
            metrics = calculate_regime_metrics(signals_df, indicator, signal_type, regime)
            results.append(metrics)
            
            if metrics['sample_size'] > 0:
                print(f"    {regime.upper()}: {metrics['sample_size']} signals, "
                      f"{metrics['win_rate_pct']:.1f}% win rate, "
                      f"{metrics['avg_return_pct']:+.3f}% avg return")
    
    return pd.DataFrame(results)


def generate_regime_specific_report(metrics_df: pd.DataFrame, output_path: str):
    """
    Generate a detailed report of regime-specific performance.
    
    Args:
        metrics_df: DataFrame with regime-specific metrics
        output_path: Path to save the report
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("REGIME-SPECIFIC INDICATOR PERFORMANCE REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("This report shows how each indicator signal performs")
    report_lines.append("in different market regimes (uptrend, downtrend, ranging).")
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Group by indicator-signal
    signal_combos = metrics_df.groupby(['indicator', 'signal_type'])
    
    for (indicator, signal_type), group in signal_combos:
        report_lines.append(f"--- {indicator}: {signal_type} ---")
        report_lines.append("")
        
        # Overall metrics (across all regimes)
        total_signals = group['sample_size'].sum()
        if total_signals > 0:
            # Weighted average win rate
            weighted_win_rate = (group['win_rate_pct'] * group['sample_size']).sum() / total_signals
            weighted_avg_return = (group['avg_return_pct'] * group['sample_size']).sum() / total_signals
            
            report_lines.append(f"Overall Performance (All Regimes):")
            report_lines.append(f"  Total Signals: {total_signals}")
            report_lines.append(f"  Weighted Win Rate: {weighted_win_rate:.2f}%")
            report_lines.append(f"  Weighted Avg Return: {weighted_avg_return:+.3f}%")
            report_lines.append("")
        
        # Regime-specific metrics
        report_lines.append("Regime-Specific Performance:")
        report_lines.append("")
        
        for regime in ['up', 'down', 'range']:
            regime_data = group[group['regime'] == regime].iloc[0]
            
            if regime_data['sample_size'] > 0:
                report_lines.append(f"  {regime.upper()}:")
                report_lines.append(f"    Sample Size: {regime_data['sample_size']}")
                report_lines.append(f"    Win Rate: {regime_data['win_rate_pct']:.2f}%")
                report_lines.append(f"    Avg Return: {regime_data['avg_return_pct']:+.3f}%")
                report_lines.append(f"    Median Return: {regime_data['median_return_pct']:+.3f}%")
                report_lines.append(f"    Std Dev: {regime_data['std_return_pct']:.3f}%")
                report_lines.append(f"    Min Return: {regime_data['min_return_pct']:+.3f}%")
                report_lines.append(f"    Max Return: {regime_data['max_return_pct']:+.3f}%")
                report_lines.append("")
            else:
                report_lines.append(f"  {regime.upper()}: No signals in this regime")
                report_lines.append("")
        
        # Key findings
        report_lines.append("Key Findings:")
        
        # Find best and worst regimes
        regimes_with_data = group[group['sample_size'] > 0]
        if len(regimes_with_data) > 0:
            best_regime = regimes_with_data.loc[regimes_with_data['win_rate_pct'].idxmax()]
            worst_regime = regimes_with_data.loc[regimes_with_data['win_rate_pct'].idxmin()]
            
            report_lines.append(f"  Best Regime: {best_regime['regime'].upper()} "
                              f"({best_regime['win_rate_pct']:.1f}% win rate, "
                              f"n={best_regime['sample_size']})")
            report_lines.append(f"  Worst Regime: {worst_regime['regime'].upper()} "
                              f"({worst_regime['win_rate_pct']:.1f}% win rate, "
                              f"n={worst_regime['sample_size']})")
            
            # Check if regime-dependent
            win_rate_range = regimes_with_data['win_rate_pct'].max() - regimes_with_data['win_rate_pct'].min()
            if win_rate_range > 15:
                report_lines.append(f"  ⚠️  HIGHLY REGIME-DEPENDENT: Win rate varies by {win_rate_range:.1f}% across regimes")
                report_lines.append(f"     → Signal MUST be filtered by regime for best results")
            elif win_rate_range > 10:
                report_lines.append(f"  ⚠️  MODERATELY REGIME-DEPENDENT: Win rate varies by {win_rate_range:.1f}%")
                report_lines.append(f"     → Consider regime filtering for improved performance")
            else:
                report_lines.append(f"  ✓ RELATIVELY REGIME-INDEPENDENT: Win rate varies by {win_rate_range:.1f}%")
                report_lines.append(f"     → Signal works consistently across regimes")
        
        report_lines.append("")
        report_lines.append("")
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n✓ Saved regime-specific report to: {output_path}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("REGIME-SPECIFIC INDICATOR PERFORMANCE CALCULATION")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading indicator signals...")
    signals_df = load_indicator_signals()
    print(f"  Loaded {len(signals_df):,} indicator signals")
    print()
    
    print("Loading ML regime predictions...")
    regimes_df = load_regime_predictions()
    print(f"  Loaded {len(regimes_df):,} regime predictions")
    print()
    
    # Match signals to regimes
    print("Matching signals to regime predictions...")
    matched_df = match_signals_to_regimes(signals_df, regimes_df)
    print(f"  Matched {len(matched_df):,} signals to regimes")
    print()
    
    # Calculate regime-specific metrics
    metrics_df = calculate_all_regime_metrics(matched_df)
    print()
    
    # Save results
    output_dir = config.PROCESSED_DATA_PATH
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'indicator_regime_specific_metrics.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"✓ Saved regime-specific metrics to: {csv_path}")
    print(f"  Total combinations: {len(metrics_df)}")
    print()
    
    # Save JSON (for easy programmatic access)
    json_path = os.path.join(output_dir, 'indicator_regime_specific_metrics.json')
    metrics_dict = {}
    for _, row in metrics_df.iterrows():
        key = f"{row['indicator']}_{row['signal_type']}_{row['regime']}"
        metrics_dict[key] = {
            'sample_size': int(row['sample_size']),
            'win_rate_pct': float(row['win_rate_pct']),
            'avg_return_pct': float(row['avg_return_pct']),
            'median_return_pct': float(row['median_return_pct']),
            'std_return_pct': float(row['std_return_pct'])
        }
    
    with open(json_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(f"✓ Saved regime-specific metrics (JSON) to: {json_path}")
    print()
    
    # Generate report
    report_path = os.path.join(output_dir, 'indicator_regime_specific_report.txt')
    generate_regime_specific_report(metrics_df, report_path)
    print()
    
    print("=" * 80)
    print("CALCULATION COMPLETE")
    print("=" * 80)
    print()
    print("Output files:")
    print(f"  1. {csv_path}")
    print(f"  2. {json_path}")
    print(f"  3. {report_path}")
    print()


if __name__ == "__main__":
    main()

