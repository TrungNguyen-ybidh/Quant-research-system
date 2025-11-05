"""
Trend Analysis Module for AI-Driven Quantitative Trading Research System.

This module identifies trends, measures their behavior, and performs statistical
comparisons between up-trends and down-trends.

Functions:
- identify_trends(df): Labels trends and creates segments
- calculate_trend_statistics(trend_segments): Computes statistics and confidence intervals
- measure_pullbacks(df, trend_segments): Quantifies pullbacks and rallies
- perform_trend_significance_test(uptrends, downtrends): Performs t-tests
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from scipy import stats
from typing import Tuple, Dict

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def identify_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify trends using SMA-based rules and create trend segments.
    
    Algorithm:
    1. Label each bar:
       - if close > SMA_50 and SMA_50 > SMA_200 → trend_type = "up"
       - elif close < SMA_50 and SMA_50 < SMA_200 → trend_type = "down"
       - else → "range"
    
    2. Group consecutive labels into segments
    3. Aggregate by segment to get duration, return, etc.
    
    Args:
        df: DataFrame with columns: timestamp, close, SMA_50, SMA_200
            (or calculate SMAs if not present)
        
    Returns:
        DataFrame with columns: segment_id, start_date, end_date, trend_type,
        duration_hours, start_price, end_price, total_return_pct
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Ensure timestamp is index or column
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have timestamp index or 'timestamp' column")
    
    # Sort by timestamp
    df = df.sort_index()
    
    # Calculate SMA_50 and SMA_200 if they don't exist
    if 'SMA_50' not in df.columns:
        df['SMA_50'] = df['close'].rolling(window=50).mean()
    if 'SMA_200' not in df.columns:
        df['SMA_200'] = df['close'].rolling(window=200).mean()
    
    # Check required columns
    required_cols = ['close', 'SMA_50', 'SMA_200']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Drop rows where SMAs are not available (need at least 200 bars)
    df = df.dropna(subset=['SMA_50', 'SMA_200'])
    
    if df.empty:
        raise ValueError("No data available after calculating SMAs")
    
    # Label each bar
    conditions = [
        (df['close'] > df['SMA_50']) & (df['SMA_50'] > df['SMA_200']),  # Uptrend
        (df['close'] < df['SMA_50']) & (df['SMA_50'] < df['SMA_200'])   # Downtrend
    ]
    choices = ['up', 'down']
    df['trend_type'] = np.select(conditions, choices, default='range')
    
    # Create segment_id: increment whenever trend_type changes
    df['segment_id'] = (df['trend_type'] != df['trend_type'].shift()).cumsum()
    
    # Aggregate by segment
    segments = []
    for seg_id, group in df.groupby('segment_id'):
        if group.empty:
            continue
            
        start_date = group.index[0]
        end_date = group.index[-1]
        trend_type = group['trend_type'].iloc[0]
        
        # Calculate duration in hours
        if isinstance(start_date, pd.Timestamp) and isinstance(end_date, pd.Timestamp):
            duration_hours = (end_date - start_date).total_seconds() / 3600.0
            # Add one bar's worth of time (assume 1 hour for hourly data)
            duration_hours += 1.0
        else:
            # Fallback: count number of bars
            duration_hours = len(group)
        
        start_price = group['close'].iloc[0]
        end_price = group['close'].iloc[-1]
        total_return_pct = ((end_price - start_price) / start_price) * 100
        
        segments.append({
            'segment_id': seg_id,
            'start_date': start_date,
            'end_date': end_date,
            'trend_type': trend_type,
            'duration_hours': duration_hours,
            'start_price': start_price,
            'end_price': end_price,
            'total_return_pct': total_return_pct
        })
    
    trend_segments = pd.DataFrame(segments)
    
    # Validation checks
    if len(trend_segments) == 0:
        raise ValueError("No trend segments identified")
    
    # Check durations > 0
    if (trend_segments['duration_hours'] <= 0).any():
        print("Warning: Some segments have duration <= 0")
    
    # Check reasonable number of segments (hundreds, not thousands)
    if len(trend_segments) > 5000:
        print(f"Warning: Very high number of segments ({len(trend_segments)}). "
              "This might indicate data issues.")
    elif len(trend_segments) < 10:
        print(f"Warning: Very low number of segments ({len(trend_segments)}). "
              "This might indicate data issues.")
    
    # Check distribution
    type_counts = trend_segments['trend_type'].value_counts()
    print(f"\nTrend type distribution:")
    for trend_type, count in type_counts.items():
        pct = (count / len(trend_segments)) * 100
        print(f"  {trend_type}: {count} ({pct:.1f}%)")
    
    return trend_segments


def calculate_trend_statistics(trend_segments: pd.DataFrame) -> Tuple[str, Dict]:
    """
    Calculate comprehensive statistics for each trend type.
    
    Computes:
    - Count, mean, median, std, min, max of duration_hours
    - Mean, median, std of total_return_pct
    - 95% confidence intervals
    - Time distribution percentages
    
    Args:
        trend_segments: DataFrame from identify_trends()
        
    Returns:
        Tuple of (summary_text, results_dict)
    """
    results = {}
    summary_lines = []
    
    summary_lines.append("=" * 80)
    summary_lines.append("TREND ANALYSIS SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    
    # Split by trend type
    up = trend_segments[trend_segments['trend_type'] == 'up'].copy()
    down = trend_segments[trend_segments['trend_type'] == 'down'].copy()
    ranging = trend_segments[trend_segments['trend_type'] == 'range'].copy()
    
    # Helper function to compute stats with CI
    def compute_stats_with_ci(data: pd.Series, name: str) -> Dict:
        """Compute statistics with 95% confidence interval."""
        if len(data) == 0:
            return {
                'count': 0,
                'mean': None,
                'median': None,
                'std': None,
                'min': None,
                'max': None,
                'ci_lower': None,
                'ci_upper': None
            }
        
        mean = data.mean()
        median = data.median()
        std = data.std()
        n = len(data)
        
        # 95% CI: t-distribution (for n < 30 use t, else approximate with z)
        if n < 30:
            t_critical = stats.t.ppf(0.975, df=n-1)
            se = std / np.sqrt(n)
            margin = t_critical * se
        else:
            # For large samples, use z ≈ 1.96
            z_critical = 1.96
            se = std / np.sqrt(n)
            margin = z_critical * se
        
        ci_lower = mean - margin
        ci_upper = mean + margin
        
        return {
            'count': n,
            'mean': float(mean),
            'median': float(median),
            'std': float(std),
            'min': float(data.min()),
            'max': float(data.max()),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper)
        }
    
    # UPTRENDS
    summary_lines.append("UPTRENDS")
    summary_lines.append("-" * 80)
    if len(up) > 0:
        duration_stats = compute_stats_with_ci(up['duration_hours'], 'duration')
        return_stats = compute_stats_with_ci(up['total_return_pct'], 'return')
        
        summary_lines.append(f"Count: {duration_stats['count']}")
        summary_lines.append("")
        summary_lines.append("Duration (hours):")
        summary_lines.append(f"  Mean: {duration_stats['mean']:.2f} hours ({duration_stats['mean']/24:.2f} days)")
        summary_lines.append(f"  Median: {duration_stats['median']:.2f} hours ({duration_stats['median']/24:.2f} days)")
        summary_lines.append(f"  Std Dev: {duration_stats['std']:.2f} hours")
        summary_lines.append(f"  Min: {duration_stats['min']:.2f} hours")
        summary_lines.append(f"  Max: {duration_stats['max']:.2f} hours")
        summary_lines.append(f"  95% CI: [{duration_stats['ci_lower']:.2f}, {duration_stats['ci_upper']:.2f}] hours")
        summary_lines.append("")
        summary_lines.append("Total Return (%):")
        summary_lines.append(f"  Mean: {return_stats['mean']:.2f}%")
        summary_lines.append(f"  Median: {return_stats['median']:.2f}%")
        summary_lines.append(f"  Std Dev: {return_stats['std']:.2f}%")
        summary_lines.append(f"  95% CI: [{return_stats['ci_lower']:.2f}%, {return_stats['ci_upper']:.2f}%]")
        
        results['uptrends'] = {
            'duration': duration_stats,
            'return': return_stats
        }
    else:
        summary_lines.append("No uptrends found.")
        results['uptrends'] = None
    
    summary_lines.append("")
    summary_lines.append("")
    
    # DOWNTRENDS
    summary_lines.append("DOWNTRENDS")
    summary_lines.append("-" * 80)
    if len(down) > 0:
        duration_stats = compute_stats_with_ci(down['duration_hours'], 'duration')
        return_stats = compute_stats_with_ci(down['total_return_pct'], 'return')
        
        summary_lines.append(f"Count: {duration_stats['count']}")
        summary_lines.append("")
        summary_lines.append("Duration (hours):")
        summary_lines.append(f"  Mean: {duration_stats['mean']:.2f} hours ({duration_stats['mean']/24:.2f} days)")
        summary_lines.append(f"  Median: {duration_stats['median']:.2f} hours ({duration_stats['median']/24:.2f} days)")
        summary_lines.append(f"  Std Dev: {duration_stats['std']:.2f} hours")
        summary_lines.append(f"  Min: {duration_stats['min']:.2f} hours")
        summary_lines.append(f"  Max: {duration_stats['max']:.2f} hours")
        summary_lines.append(f"  95% CI: [{duration_stats['ci_lower']:.2f}, {duration_stats['ci_upper']:.2f}] hours")
        summary_lines.append("")
        summary_lines.append("Total Return (%):")
        summary_lines.append(f"  Mean: {return_stats['mean']:.2f}%")
        summary_lines.append(f"  Median: {return_stats['median']:.2f}%")
        summary_lines.append(f"  Std Dev: {return_stats['std']:.2f}%")
        summary_lines.append(f"  95% CI: [{return_stats['ci_lower']:.2f}%, {return_stats['ci_upper']:.2f}%]")
        
        results['downtrends'] = {
            'duration': duration_stats,
            'return': return_stats
        }
    else:
        summary_lines.append("No downtrends found.")
        results['downtrends'] = None
    
    summary_lines.append("")
    summary_lines.append("")
    
    # RANGING PERIODS
    summary_lines.append("RANGING PERIODS")
    summary_lines.append("-" * 80)
    if len(ranging) > 0:
        duration_stats = compute_stats_with_ci(ranging['duration_hours'], 'duration')
        return_stats = compute_stats_with_ci(ranging['total_return_pct'], 'return')
        
        summary_lines.append(f"Count: {duration_stats['count']}")
        summary_lines.append("")
        summary_lines.append("Duration (hours):")
        summary_lines.append(f"  Mean: {duration_stats['mean']:.2f} hours ({duration_stats['mean']/24:.2f} days)")
        summary_lines.append(f"  Median: {duration_stats['median']:.2f} hours ({duration_stats['median']/24:.2f} days)")
        summary_lines.append(f"  Std Dev: {duration_stats['std']:.2f} hours")
        summary_lines.append(f"  95% CI: [{duration_stats['ci_lower']:.2f}, {duration_stats['ci_upper']:.2f}] hours")
        
        results['ranging'] = {
            'duration': duration_stats,
            'return': return_stats
        }
    else:
        summary_lines.append("No ranging periods found.")
        results['ranging'] = None
    
    summary_lines.append("")
    summary_lines.append("")
    
    # TIME DISTRIBUTION
    summary_lines.append("TIME DISTRIBUTION")
    summary_lines.append("-" * 80)
    total_duration = trend_segments['duration_hours'].sum()
    
    if total_duration > 0:
        up_pct = (up['duration_hours'].sum() / total_duration) * 100
        down_pct = (down['duration_hours'].sum() / total_duration) * 100
        range_pct = (ranging['duration_hours'].sum() / total_duration) * 100
        
        summary_lines.append(f"Uptrends: {up_pct:.1f}% of time")
        summary_lines.append(f"Downtrends: {down_pct:.1f}% of time")
        summary_lines.append(f"Ranging: {range_pct:.1f}% of time")
        summary_lines.append(f"Total time analyzed: {total_duration:.0f} hours ({total_duration/24:.1f} days)")
        
        results['time_distribution'] = {
            'uptrend_pct': float(up_pct),
            'downtrend_pct': float(down_pct),
            'ranging_pct': float(range_pct),
            'total_hours': float(total_duration)
        }
    else:
        summary_lines.append("No time distribution available.")
        results['time_distribution'] = None
    
    summary_text = "\n".join(summary_lines)
    
    return summary_text, results


def measure_pullbacks(df: pd.DataFrame, trend_segments: pd.DataFrame) -> Tuple[str, Dict]:
    """
    Measure pullbacks (in uptrends) and rallies (in downtrends).
    
    For uptrends: find local peaks → next trough = pullback%
    For downtrends: find local trough → next peak = rally%
    
    Args:
        df: Original DataFrame with price data
        trend_segments: DataFrame from identify_trends()
        
    Returns:
        Tuple of (summary_text, stats_dict)
    """
    # Ensure timestamp is index
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    df = df.sort_index()
    
    pullbacks = []
    rallies = []
    
    # Helper function to find local peaks/troughs
    def find_local_extrema(prices: pd.Series, window: int = 3, is_peak: bool = True) -> pd.Index:
        """Find local peaks (is_peak=True) or troughs (is_peak=False) using rolling window."""
        if is_peak:
            # Peak: higher than neighbors
            return prices[(prices == prices.rolling(window=window, center=True).max()) & 
                         (prices.shift(1) < prices) & (prices.shift(-1) < prices)].index
        else:
            # Trough: lower than neighbors
            return prices[(prices == prices.rolling(window=window, center=True).min()) & 
                         (prices.shift(1) > prices) & (prices.shift(-1) > prices)].index
    
    # Process uptrends
    uptrends = trend_segments[trend_segments['trend_type'] == 'up']
    for _, segment in uptrends.iterrows():
        start = segment['start_date']
        end = segment['end_date']
        
        # Get price data for this segment
        segment_data = df.loc[start:end].copy()
        if len(segment_data) < 6:  # Need at least 6 bars for window of 3
            continue
        
        # Find peaks and troughs
        peaks = find_local_extrema(segment_data['close'], window=3, is_peak=True)
        troughs = find_local_extrema(segment_data['close'], window=3, is_peak=False)
        
        # For each peak, find next trough and calculate pullback
        for peak_idx in peaks:
            peak_price = segment_data.loc[peak_idx, 'close']
            # Find next trough after this peak
            next_troughs = troughs[troughs > peak_idx]
            if len(next_troughs) > 0:
                trough_idx = next_troughs[0]
                trough_price = segment_data.loc[trough_idx, 'close']
                pullback_pct = ((peak_price - trough_price) / peak_price) * 100
                pullbacks.append(pullback_pct)
    
    # Process downtrends
    downtrends = trend_segments[trend_segments['trend_type'] == 'down']
    for _, segment in downtrends.iterrows():
        start = segment['start_date']
        end = segment['end_date']
        
        # Get price data for this segment
        segment_data = df.loc[start:end].copy()
        if len(segment_data) < 6:
            continue
        
        # Find peaks and troughs
        peaks = find_local_extrema(segment_data['close'], window=3, is_peak=True)
        troughs = find_local_extrema(segment_data['close'], window=3, is_peak=False)
        
        # For each trough, find next peak and calculate rally
        for trough_idx in troughs:
            trough_price = segment_data.loc[trough_idx, 'close']
            # Find next peak after this trough
            next_peaks = peaks[peaks > trough_idx]
            if len(next_peaks) > 0:
                peak_idx = next_peaks[0]
                peak_price = segment_data.loc[peak_idx, 'close']
                rally_pct = ((peak_price - trough_price) / trough_price) * 100
                rallies.append(rally_pct)
    
    # Compute statistics
    summary_lines = []
    summary_lines.append("")
    summary_lines.append("PULLBACK ANALYSIS")
    summary_lines.append("-" * 80)
    
    pullback_stats = {}
    rally_stats = {}
    
    if len(pullbacks) > 0:
        pullbacks_series = pd.Series(pullbacks)
        summary_lines.append("Uptrend Pullbacks (%):")
        summary_lines.append(f"  Count: {len(pullbacks)}")
        summary_lines.append(f"  Mean: {pullbacks_series.mean():.2f}%")
        summary_lines.append(f"  Median: {pullbacks_series.median():.2f}%")
        summary_lines.append(f"  Std Dev: {pullbacks_series.std():.2f}%")
        summary_lines.append(f"  25th percentile: {pullbacks_series.quantile(0.25):.2f}%")
        summary_lines.append(f"  75th percentile: {pullbacks_series.quantile(0.75):.2f}%")
        
        pullback_stats = {
            'count': len(pullbacks),
            'mean': float(pullbacks_series.mean()),
            'median': float(pullbacks_series.median()),
            'std': float(pullbacks_series.std()),
            'p25': float(pullbacks_series.quantile(0.25)),
            'p75': float(pullbacks_series.quantile(0.75))
        }
    else:
        summary_lines.append("Uptrend Pullbacks: No pullbacks found")
        pullback_stats = None
    
    summary_lines.append("")
    
    if len(rallies) > 0:
        rallies_series = pd.Series(rallies)
        summary_lines.append("Downtrend Rallies (%):")
        summary_lines.append(f"  Count: {len(rallies)}")
        summary_lines.append(f"  Mean: {rallies_series.mean():.2f}%")
        summary_lines.append(f"  Median: {rallies_series.median():.2f}%")
        summary_lines.append(f"  Std Dev: {rallies_series.std():.2f}%")
        summary_lines.append(f"  25th percentile: {rallies_series.quantile(0.25):.2f}%")
        summary_lines.append(f"  75th percentile: {rallies_series.quantile(0.75):.2f}%")
        
        rally_stats = {
            'count': len(rallies),
            'mean': float(rallies_series.mean()),
            'median': float(rallies_series.median()),
            'std': float(rallies_series.std()),
            'p25': float(rallies_series.quantile(0.25)),
            'p75': float(rallies_series.quantile(0.75))
        }
    else:
        summary_lines.append("Downtrend Rallies: No rallies found")
        rally_stats = None
    
    stats_dict = {
        'pullbacks': pullback_stats,
        'rallies': rally_stats
    }
    
    return "\n".join(summary_lines), stats_dict


def perform_trend_significance_test(uptrends: pd.DataFrame, downtrends: pd.DataFrame) -> str:
    """
    Perform statistical significance tests comparing uptrends vs downtrends.
    
    Tests:
    1. Duration comparison (t-test)
    2. Return magnitude comparison (t-test on absolute returns)
    3. Pullback vs Rally size comparison (if available)
    
    Args:
        uptrends: DataFrame with uptrend segments
        downtrends: DataFrame with downtrend segments
        
    Returns:
        String with test results to append to summary
    """
    summary_lines = []
    summary_lines.append("")
    summary_lines.append("STATISTICAL SIGNIFICANCE TESTS")
    summary_lines.append("-" * 80)
    
    # Test 1: Duration comparison
    if len(uptrends) > 0 and len(downtrends) > 0:
        up_durations = uptrends['duration_hours'].values
        down_durations = downtrends['duration_hours'].values
        
        # Independent two-sample t-test
        # H0: mean(up) = mean(down)
        # H1: mean(up) > mean(down) (one-tailed)
        t_stat_duration, p_value_duration = stats.ttest_ind(up_durations, down_durations, 
                                                             alternative='greater')
        
        summary_lines.append("1. Duration Comparison (Uptrends vs Downtrends)")
        summary_lines.append(f"   Uptrend mean duration: {up_durations.mean():.2f} hours")
        summary_lines.append(f"   Downtrend mean duration: {down_durations.mean():.2f} hours")
        summary_lines.append(f"   t-statistic: {t_stat_duration:.4f}")
        summary_lines.append(f"   p-value: {p_value_duration:.6f}")
        if p_value_duration < 0.05:
            summary_lines.append(f"   Result: SIGNIFICANT (p < 0.05) - Uptrends are significantly longer")
        else:
            summary_lines.append(f"   Result: NOT SIGNIFICANT (p >= 0.05)")
        summary_lines.append("")
        
        # Test 2: Return magnitude comparison (absolute values)
        up_returns = np.abs(uptrends['total_return_pct'].values)
        down_returns = np.abs(downtrends['total_return_pct'].values)
        
        t_stat_return, p_value_return = stats.ttest_ind(up_returns, down_returns)
        
        summary_lines.append("2. Return Magnitude Comparison (|Return|, Uptrends vs Downtrends)")
        summary_lines.append(f"   Uptrend mean |return|: {up_returns.mean():.2f}%")
        summary_lines.append(f"   Downtrend mean |return|: {down_returns.mean():.2f}%")
        summary_lines.append(f"   t-statistic: {t_stat_return:.4f}")
        summary_lines.append(f"   p-value: {p_value_return:.6f}")
        if p_value_return < 0.05:
            if up_returns.mean() > down_returns.mean():
                summary_lines.append(f"   Result: SIGNIFICANT (p < 0.05) - Uptrend returns are larger")
            else:
                summary_lines.append(f"   Result: SIGNIFICANT (p < 0.05) - Downtrend returns are larger")
        else:
            summary_lines.append(f"   Result: NOT SIGNIFICANT (p >= 0.05)")
    else:
        summary_lines.append("Cannot perform tests: insufficient data")
        summary_lines.append("")
    
    return "\n".join(summary_lines)


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    """
    Main execution: Load data, identify trends, calculate statistics,
    measure pullbacks, perform significance tests, and save outputs.
    """
    print("=" * 80)
    print("TREND ANALYSIS")
    print("=" * 80)
    print()
    
    # Load data
    data_file = os.path.join(config.PROCESSED_DATA_PATH, 
                             "XAU_USD_1Hour_with_indicators.csv")
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    print(f"Loading data from: {data_file}")
    df = pd.read_csv(data_file, parse_dates=['timestamp'])
    
    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print()
    
    # Step 1: Identify trends
    print("Step 1: Identifying trends...")
    trend_segments = identify_trends(df)
    print(f"Identified {len(trend_segments)} trend segments")
    print()
    
    # Save trend segments
    segments_file = os.path.join(config.PROCESSED_DATA_PATH, "trend_segments.csv")
    trend_segments.to_csv(segments_file, index=False)
    print(f"✓ Saved trend segments to: {segments_file}")
    print()
    
    # Step 2: Calculate statistics
    print("Step 2: Calculating trend statistics...")
    summary_text, results_dict = calculate_trend_statistics(trend_segments)
    print("✓ Statistics calculated")
    print()
    
    # Step 3: Measure pullbacks
    print("Step 3: Measuring pullbacks and rallies...")
    pullback_text, pullback_stats = measure_pullbacks(df, trend_segments)
    print("✓ Pullbacks and rallies measured")
    print()
    
    # Step 4: Statistical significance tests
    print("Step 4: Performing statistical significance tests...")
    uptrends_df = trend_segments[trend_segments['trend_type'] == 'up']
    downtrends_df = trend_segments[trend_segments['trend_type'] == 'down']
    test_text = perform_trend_significance_test(uptrends_df, downtrends_df)
    print("✓ Significance tests completed")
    print()
    
    # Combine all summary text
    full_summary = summary_text + pullback_text + test_text
    
    # Save summary text file
    summary_file = os.path.join(config.PROCESSED_DATA_PATH, "trend_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(full_summary)
    print(f"✓ Saved summary to: {summary_file}")
    
    # Save JSON results
    json_file = os.path.join(config.PROCESSED_DATA_PATH, "trend_analysis_results.json")
    
    # Add pullback/rally stats to results
    results_dict['pullbacks_and_rallies'] = pullback_stats
    
    with open(json_file, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    print(f"✓ Saved JSON results to: {json_file}")
    
    print()
    print("=" * 80)
    print("TREND ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print("Output files created:")
    print(f"  1. {segments_file}")
    print(f"  2. {summary_file}")
    print(f"  3. {json_file}")
    print()

