"""
Analysis Module for AI-Driven Quantitative Trading Research System.

This module provides functions to analyze volume patterns, trading activity,
relationships between volume and price movements, and volatility analysis.

Functions:
- analyze_volume_distribution: Compare volume across timeframes
- analyze_intraday_volume: Identify peak trading hours
- analyze_volume_price_relationship: Correlate volume with price movements
- analyze_volatility_distribution: Compare ATR across timeframes
- analyze_intraday_volatility: Discover peak volatility hours
- test_volatility_clustering: Detect volatility clustering patterns
- create_volume_heatmap: Visualize hourly volume across weekdays
- create_volume_by_timeframe_chart: Compare volume across timeframes
- create_volatility_clustering_plot: Visualize volatility regimes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def analyze_volume_distribution(df: pd.DataFrame, timeframe: str = None) -> pd.DataFrame:
    """
    Analyze volume distribution for a given timeframe.
    
    Computes mean, median, std, min, max volume for the DataFrame.
    
    Args:
        df: DataFrame with 'volume' column
        timeframe: Optional timeframe label for identification
        
    Returns:
        DataFrame with volume statistics
    """
    if 'volume' not in df.columns:
        raise ValueError("DataFrame must contain 'volume' column")
    
    # Calculate volume statistics
    volume_stats = {
        'timeframe': timeframe or 'Unknown',
        'mean': df['volume'].mean(),
        'median': df['volume'].median(),
        'std': df['volume'].std(),
        'min': df['volume'].min(),
        'max': df['volume'].max(),
        'count': len(df),
        'total_volume': df['volume'].sum()
    }
    
    return pd.DataFrame([volume_stats])


def analyze_volume_distribution_all_timeframes(symbol: str = None) -> pd.DataFrame:
    """
    Compare average volume across all timeframes.
    
    Loads processed data for each timeframe and computes volume statistics.
    Prints ranking of timeframes by mean volume.
    
    Args:
        symbol: Trading symbol (defaults to config.DEFAULT_SYMBOL)
        
    Returns:
        DataFrame with volume statistics for all timeframes
    """
    if symbol is None:
        symbol = config.DEFAULT_SYMBOL
    
    # Sanitize symbol for filename
    symbol_sanitized = symbol.replace('/', '_').replace('=', '_').replace('-', '_')
    
    all_stats = []
    
    print("="*70)
    print("Volume Distribution Analysis Across Timeframes")
    print("="*70)
    print()
    
    for timeframe in config.DEFAULT_TIMEFRAMES:
        try:
            # Load processed data
            file_path = config.get_processed_data_path(symbol_sanitized, timeframe)
            
            if not os.path.exists(file_path):
                print(f"⚠️  File not found: {file_path}")
                continue
            
            df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
            
            # Analyze volume distribution
            stats = analyze_volume_distribution(df, timeframe)
            all_stats.append(stats)
            
            print(f"✓ {timeframe:8s}: Mean={stats['mean'].iloc[0]:>12,.0f}, "
                  f"Median={stats['median'].iloc[0]:>12,.0f}, "
                  f"Count={stats['count'].iloc[0]:>8,}")
            
        except Exception as e:
            print(f"⚠️  Error processing {timeframe}: {str(e)}")
            continue
    
    if not all_stats:
        print("No data found for any timeframe")
        return pd.DataFrame()
    
    # Combine all statistics
    combined_stats = pd.concat(all_stats, ignore_index=True)
    
    # Sort by mean volume (descending)
    combined_stats = combined_stats.sort_values('mean', ascending=False)
    
    print()
    print("="*70)
    print("Ranking by Mean Volume (Highest to Lowest)")
    print("="*70)
    for idx, row in combined_stats.iterrows():
        print(f"{idx+1}. {row['timeframe']:8s}: {row['mean']:>12,.0f} (mean volume)")
    
    return combined_stats


def analyze_intraday_volume(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Analyze intraday volume patterns to identify peak trading hours.
    
    Extracts hour from timestamps and groups by hour to compute mean volume.
    Identifies top 3 peak hours and bottom 3 quiet hours.
    
    Args:
        df: DataFrame with timestamp index and 'volume' column
        
    Returns:
        Tuple of (hourly_stats DataFrame, insights dictionary)
    """
    if 'volume' not in df.columns:
        raise ValueError("DataFrame must contain 'volume' column")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    
    # Extract hour from timestamps
    df_copy = df.copy()
    df_copy['hour'] = df_copy.index.hour
    
    # Group by hour and compute mean volume
    hourly_stats = df_copy.groupby('hour')['volume'].agg([
        'mean', 'median', 'std', 'count', 'sum'
    ]).reset_index()
    
    hourly_stats.columns = ['hour', 'mean_volume', 'median_volume', 'std_volume', 'count', 'total_volume']
    hourly_stats = hourly_stats.sort_values('mean_volume', ascending=False)
    
    # Find top 3 peak hours
    top_3_hours = hourly_stats.head(3)
    
    # Find bottom 3 quiet hours
    bottom_3_hours = hourly_stats.tail(3)
    
    # Create insights dictionary
    insights = {
        'peak_hours': top_3_hours[['hour', 'mean_volume']].to_dict('records'),
        'quiet_hours': bottom_3_hours[['hour', 'mean_volume']].to_dict('records'),
        'highest_activity_hour': int(top_3_hours.iloc[0]['hour']),
        'lowest_activity_hour': int(bottom_3_hours.iloc[0]['hour']),
        'peak_mean_volume': float(top_3_hours.iloc[0]['mean_volume']),
        'quiet_mean_volume': float(bottom_3_hours.iloc[0]['mean_volume'])
    }
    
    return hourly_stats, insights


def analyze_volume_price_relationship(df: pd.DataFrame) -> Dict:
    """
    Analyze the relationship between volume and price movements.
    
    Computes correlations between volume and returns, and volume and absolute returns.
    Compares average absolute returns for high vs low volume periods.
    
    Args:
        df: DataFrame with 'volume' and 'close' columns
        
    Returns:
        Dictionary with correlation results and insights
    """
    if 'volume' not in df.columns or 'close' not in df.columns:
        raise ValueError("DataFrame must contain 'volume' and 'close' columns")
    
    # Calculate returns
    df_copy = df.copy()
    df_copy['returns'] = df_copy['close'].pct_change()
    df_copy['abs_returns'] = df_copy['returns'].abs()
    
    # Drop NaN values
    df_copy = df_copy.dropna(subset=['volume', 'returns'])
    
    # Calculate correlations
    corr_volume_returns = df_copy['volume'].corr(df_copy['returns'])
    corr_volume_abs_returns = df_copy['volume'].corr(df_copy['abs_returns'])
    
    # Calculate average absolute return for top 25% vs bottom 25% volume periods
    volume_25th = df_copy['volume'].quantile(0.25)
    volume_75th = df_copy['volume'].quantile(0.75)
    
    low_volume_returns = df_copy[df_copy['volume'] <= volume_25th]['abs_returns'].mean()
    high_volume_returns = df_copy[df_copy['volume'] >= volume_75th]['abs_returns'].mean()
    
    results = {
        'corr_volume_returns': corr_volume_returns,
        'corr_volume_abs_returns': corr_volume_abs_returns,
        'low_volume_avg_abs_return': low_volume_returns,
        'high_volume_avg_abs_return': high_volume_returns,
        'volume_25th_percentile': volume_25th,
        'volume_75th_percentile': volume_75th,
        'low_volume_periods': len(df_copy[df_copy['volume'] <= volume_25th]),
        'high_volume_periods': len(df_copy[df_copy['volume'] >= volume_75th])
    }
    
    return results


def create_volume_heatmap(symbol: str = None, save_path: str = None) -> None:
    """
    Create a heatmap showing hourly volume activity across weekdays.
    
    Loads 1-hour processed data and creates a pivot table of average volume
    per hour × day, then plots as a heatmap.
    
    Args:
        symbol: Trading symbol (defaults to config.DEFAULT_SYMBOL)
        save_path: Path to save the heatmap (defaults to data/processed/volume_heatmap.png)
    """
    if symbol is None:
        symbol = config.DEFAULT_SYMBOL
    
    if save_path is None:
        save_path = os.path.join(config.PROCESSED_DATA_PATH, 'volume_heatmap.png')
    
    # Sanitize symbol for filename
    symbol_sanitized = symbol.replace('/', '_').replace('=', '_').replace('-', '_')
    
    # Load 1-hour processed data
    file_path = config.get_processed_data_path(symbol_sanitized, '1Hour')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"1-hour processed data not found: {file_path}")
    
    df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
    
    # Extract hour and day name
    df['hour'] = df.index.hour
    df['day'] = df.index.day_name()
    
    # Create pivot table: average volume per hour × day
    pivot_table = df.pivot_table(
        values='volume',
        index='hour',
        columns='day',
        aggfunc='mean'
    )
    
    # Reorder days to start with Monday
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_table = pivot_table.reindex(columns=day_order)
    
    # Create heatmap
    plt.figure(figsize=(14, 8))
    if sns is not None:
        sns.heatmap(
            pivot_table,
            cmap='YlOrRd',
            annot=True,
            fmt='.0f',
            cbar_kws={'label': 'Average Volume'},
            linewidths=0.5,
            linecolor='gray'
        )
    else:
        # Fallback: use imshow with matplotlib if seaborn is unavailable
        print("⚠️  seaborn not available; using Matplotlib fallback for volume heatmap.")
        plt.imshow(pivot_table.values, aspect='auto', cmap='YlOrRd', origin='lower')
        plt.colorbar(label='Average Volume')
        plt.xticks(range(len(pivot_table.columns)), pivot_table.columns, rotation=45, ha='right')
        plt.yticks(range(len(pivot_table.index)), pivot_table.index)
    
    plt.title(f'Volume Heatmap: Hourly Trading Activity by Day of Week\n{symbol}', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Day of Week', fontsize=12)
    plt.ylabel('Hour (UTC)', fontsize=12)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=config.CHART_DPI, bbox_inches='tight')
    print(f"✓ Volume heatmap saved to: {save_path}")
    
    plt.close()


def create_volume_by_timeframe_chart(symbol: str = None, save_path: str = None) -> None:
    """
    Create a bar chart comparing mean volume across all timeframes.
    
    Args:
        symbol: Trading symbol (defaults to config.DEFAULT_SYMBOL)
        save_path: Path to save the chart (defaults to data/processed/volume_by_timeframe.png)
    """
    if symbol is None:
        symbol = config.DEFAULT_SYMBOL
    
    if save_path is None:
        save_path = os.path.join(config.PROCESSED_DATA_PATH, 'volume_by_timeframe.png')
    
    # Get volume distribution statistics
    stats_df = analyze_volume_distribution_all_timeframes(symbol)
    
    if stats_df.empty:
        raise ValueError("No data found for volume comparison")
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(stats_df['timeframe'], stats_df['mean'], color='steelblue', edgecolor='navy', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,.0f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.title(f'Mean Volume by Timeframe\n{symbol}', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Timeframe', fontsize=12)
    plt.ylabel('Mean Volume', fontsize=12)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=config.CHART_DPI, bbox_inches='tight')
    print(f"✓ Volume by timeframe chart saved to: {save_path}")
    
    plt.close()


def run_all_volume_analysis(symbol: str = None) -> None:
    """
    Run all volume analysis functions and generate visualizations.
    
    This is a convenience function that runs all analysis functions
    and creates all visualizations in one go.
    
    Args:
        symbol: Trading symbol (defaults to config.DEFAULT_SYMBOL)
    """
    if symbol is None:
        symbol = config.DEFAULT_SYMBOL
    
    print("="*70)
    print("Volume Analysis - Complete Report")
    print("="*70)
    print()
    
    # 1. Volume distribution across timeframes
    print("\n1. Volume Distribution Across Timeframes")
    print("-" * 70)
    stats_df = analyze_volume_distribution_all_timeframes(symbol)
    
    # 2. Intraday volume analysis (using 1-hour data)
    print("\n2. Intraday Volume Analysis (1-Hour Data)")
    print("-" * 70)
    symbol_sanitized = symbol.replace('/', '_').replace('=', '_').replace('-', '_')
    file_path = config.get_processed_data_path(symbol_sanitized, '1Hour')
    
    if os.path.exists(file_path):
        df_1h = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
        hourly_stats, insights = analyze_intraday_volume(df_1h)
        
        print(f"\nTop 3 Peak Trading Hours (UTC):")
        for i, hour_data in enumerate(insights['peak_hours'], 1):
            print(f"  {i}. Hour {hour_data['hour']:02d}:00 - Mean Volume: {hour_data['mean_volume']:,.0f}")
        
        print(f"\nBottom 3 Quiet Trading Hours (UTC):")
        for i, hour_data in enumerate(reversed(insights['quiet_hours']), 1):
            print(f"  {i}. Hour {hour_data['hour']:02d}:00 - Mean Volume: {hour_data['mean_volume']:,.0f}")
        
        # 3. Volume-price relationship
        print("\n3. Volume-Price Relationship Analysis")
        print("-" * 70)
        vp_results = analyze_volume_price_relationship(df_1h)
        
        print(f"\nCorrelations:")
        print(f"  Volume vs Returns: {vp_results['corr_volume_returns']:.4f}")
        print(f"  Volume vs Absolute Returns: {vp_results['corr_volume_abs_returns']:.4f}")
        
        print(f"\nAverage Absolute Returns by Volume:")
        print(f"  Low Volume (bottom 25%): {vp_results['low_volume_avg_abs_return']:.6f}")
        print(f"  High Volume (top 25%): {vp_results['high_volume_avg_abs_return']:.6f}")
        print(f"  Ratio (High/Low): {vp_results['high_volume_avg_abs_return'] / vp_results['low_volume_avg_abs_return']:.2f}x")
        
        if vp_results['corr_volume_abs_returns'] > 0.3:
            print("\n  ✓ Strong positive correlation: High volume tends to predict larger price movements")
        elif vp_results['corr_volume_abs_returns'] > 0.1:
            print("\n  ⚠️  Moderate correlation: Volume has some predictive power")
        else:
            print("\n  ⚠️  Weak correlation: Volume may not strongly predict price movements")
    else:
        print(f"⚠️  1-hour processed data not found: {file_path}")
    
    # 4. Create visualizations
    print("\n4. Generating Visualizations")
    print("-" * 70)
    try:
        create_volume_heatmap(symbol)
        create_volume_by_timeframe_chart(symbol)
        print("\n✓ All visualizations generated successfully!")
    except Exception as e:
        print(f"⚠️  Error generating visualizations: {str(e)}")
    
    print("\n" + "="*70)
    print("Analysis Complete")
    print("="*70)


# ============================================================================
# Volatility Analysis Functions
# ============================================================================

def analyze_volatility_distribution(df: pd.DataFrame, timeframe: str = None) -> pd.DataFrame:
    """
    Analyze volatility distribution for a given timeframe using ATR.
    
    Computes mean ATR, median ATR, and ATR normalized by price (ATR/Close * 100).
    
    Args:
        df: DataFrame with 'atr' and 'close' columns
        timeframe: Optional timeframe label for identification
        
    Returns:
        DataFrame with volatility statistics
    """
    if 'atr' not in df.columns or 'close' not in df.columns:
        raise ValueError("DataFrame must contain 'atr' and 'close' columns")
    
    # Calculate ATR statistics
    atr_stats = {
        'timeframe': timeframe or 'Unknown',
        'mean_atr': df['atr'].mean(),
        'median_atr': df['atr'].median(),
        'std_atr': df['atr'].std(),
        'min_atr': df['atr'].min(),
        'max_atr': df['atr'].max(),
        'count': len(df[df['atr'].notna()])
    }
    
    # Calculate ATR normalized by price (as percentage)
    df_valid = df[df['atr'].notna() & df['close'].notna()].copy()
    if len(df_valid) > 0:
        atr_normalized = (df_valid['atr'] / df_valid['close']) * 100
        atr_stats['mean_atr_pct'] = atr_normalized.mean()
        atr_stats['median_atr_pct'] = atr_normalized.median()
    else:
        atr_stats['mean_atr_pct'] = np.nan
        atr_stats['median_atr_pct'] = np.nan
    
    return pd.DataFrame([atr_stats])


def analyze_volatility_distribution_all_timeframes(symbol: str = None) -> pd.DataFrame:
    """
    Compare volatility distribution across all timeframes.
    
    Loads processed data for each timeframe and computes ATR statistics.
    Prints ranking of timeframes by volatility.
    
    Args:
        symbol: Trading symbol (defaults to config.DEFAULT_SYMBOL)
        
    Returns:
        DataFrame with volatility statistics for all timeframes
    """
    if symbol is None:
        symbol = config.DEFAULT_SYMBOL
    
    # Sanitize symbol for filename
    symbol_sanitized = symbol.replace('/', '_').replace('=', '_').replace('-', '_')
    
    all_stats = []
    
    print("="*70)
    print("Volatility Distribution Analysis Across Timeframes")
    print("="*70)
    print()
    
    for timeframe in config.DEFAULT_TIMEFRAMES:
        try:
            # Load processed data
            file_path = config.get_processed_data_path(symbol_sanitized, timeframe)
            
            if not os.path.exists(file_path):
                print(f"⚠️  File not found: {file_path}")
                continue
            
            df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
            
            # Analyze volatility distribution
            stats = analyze_volatility_distribution(df, timeframe)
            all_stats.append(stats)
            
            print(f"✓ {timeframe:8s}: Mean ATR={stats['mean_atr'].iloc[0]:>8.2f}, "
                  f"Median ATR={stats['median_atr'].iloc[0]:>8.2f}, "
                  f"ATR %={stats['mean_atr_pct'].iloc[0]:>6.3f}%")
            
        except Exception as e:
            print(f"⚠️  Error processing {timeframe}: {str(e)}")
            continue
    
    if not all_stats:
        print("No data found for any timeframe")
        return pd.DataFrame()
    
    # Combine all statistics
    combined_stats = pd.concat(all_stats, ignore_index=True)
    
    # Sort by mean ATR (descending)
    combined_stats = combined_stats.sort_values('mean_atr', ascending=False)
    
    print()
    print("="*70)
    print("Ranking by Volatility (Highest to Lowest)")
    print("="*70)
    for idx, row in combined_stats.iterrows():
        print(f"{idx+1}. {row['timeframe']:8s}: Mean ATR={row['mean_atr']:>8.2f}, "
              f"ATR %={row['mean_atr_pct']:>6.3f}%")
    
    # Print summary
    most_volatile = combined_stats.iloc[0]['timeframe']
    least_volatile = combined_stats.iloc[-1]['timeframe']
    print()
    print(f"Summary: {most_volatile} most volatile, {least_volatile} least volatile")
    
    return combined_stats


def analyze_intraday_volatility(df: pd.DataFrame, save_path: str = None) -> Tuple[pd.DataFrame, None]:
    """
    Analyze intraday volatility patterns to identify peak volatility hours.
    
    Phase 3 - Volatility Analysis Step 5A: Shows how volatility changes throughout 
    the trading day using 1-Hour timeframe data with ATR.
    
    Groups by hour and computes average ATR for each hour (0-23 UTC).
    Labels peaks (13-17 UTC) as the most volatile periods (US/London overlap).
    
    Args:
        df: DataFrame with timestamp index and 'atr' column (1-Hour timeframe)
        save_path: Path to save the chart (defaults to data/processed/intraday_volatility.png)
        
    Returns:
        Tuple of (hourly_stats DataFrame, None)
    """
    if 'atr' not in df.columns:
        raise ValueError("DataFrame must contain 'atr' column")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    
    if save_path is None:
        save_path = os.path.join(config.PROCESSED_DATA_PATH, 'intraday_volatility.png')
    
    # Extract hour from timestamps (UTC)
    df_copy = df.copy()
    df_copy['hour'] = df_copy.index.hour
    
    # Group by hour and compute average ATR
    hourly_stats = df_copy.groupby('hour')['atr'].mean().reset_index()
    hourly_stats.columns = ['hour', 'mean_atr']
    hourly_stats = hourly_stats.sort_values('hour')
    
    # Create line chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot ATR by hour
    ax.plot(hourly_stats['hour'], hourly_stats['mean_atr'], 
            marker='o', linewidth=2, markersize=8, color='steelblue', label='Average ATR')
    
    # Highlight and label peak volatility hours (13-17 UTC: US/London overlap)
    peak_hours = hourly_stats[hourly_stats['hour'].between(13, 17)]
    if len(peak_hours) > 0:
        ax.scatter(peak_hours['hour'], peak_hours['mean_atr'], 
                  color='red', s=150, zorder=5, marker='*', 
                  label='Peak Volatility (13-17 UTC)')
        
        # Add text labels for peak hours
        for _, row in peak_hours.iterrows():
            ax.annotate(f"Peak\n{int(row['hour']):02d}:00", 
                       xy=(row['hour'], row['mean_atr']),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9, fontweight='bold', color='red',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    ax.set_xlabel('Hour (UTC)', fontsize=12)
    ax.set_ylabel('Average ATR', fontsize=12)
    ax.set_title('Intraday Volatility: Average ATR by Hour of Day\n(Peak Volatility During US/London Overlap: 13-17 UTC)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(range(24))
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=config.CHART_DPI, bbox_inches='tight')
    print(f"✓ Intraday volatility chart saved to: {save_path}")
    
    plt.close()
    
    return hourly_stats, None


def test_volatility_clustering(df: pd.DataFrame) -> Dict:
    """
    Test for volatility clustering using autocorrelation and persistence analysis.
    
    Calculates autocorrelation of ATR and squared returns, computes persistence
    probability, and average run length of high vs low volatility regimes.
    
    Args:
        df: DataFrame with timestamp index and 'atr', 'close' columns
        
    Returns:
        Dictionary with clustering metrics and results
    """
    if 'atr' not in df.columns or 'close' not in df.columns:
        raise ValueError("DataFrame must contain 'atr' and 'close' columns")
    
    # Prepare data
    df_copy = df.copy()
    df_copy = df_copy[df_copy['atr'].notna()].copy()
    
    # Calculate returns and squared returns
    df_copy['returns'] = df_copy['close'].pct_change()
    df_copy['squared_returns'] = df_copy['returns'] ** 2
    
    # Drop NaN values
    df_copy = df_copy.dropna(subset=['atr', 'returns'])
    
    if len(df_copy) < 2:
        raise ValueError("Insufficient data for volatility clustering analysis")
    
    # 1. Calculate autocorrelation of ATR (lag=1)
    atr_autocorr_lag1 = df_copy['atr'].autocorr(lag=1)
    
    # 2. Calculate autocorrelation of squared returns (lag=1)
    squared_returns_autocorr_lag1 = df_copy['squared_returns'].autocorr(lag=1)
    
    # 3. Determine probability of persistence
    # P(ATR_t+1 > median | ATR_t > median)
    atr_median = df_copy['atr'].median()
    
    high_vol_this_period = df_copy['atr'] > atr_median
    high_vol_next_period = df_copy['atr'].shift(-1) > atr_median
    
    # Count transitions
    high_to_high = ((high_vol_this_period) & (high_vol_next_period)).sum()
    high_to_low = ((high_vol_this_period) & (~high_vol_next_period)).sum()
    low_to_high = ((~high_vol_this_period) & (high_vol_next_period)).sum()
    low_to_low = ((~high_vol_this_period) & (~high_vol_next_period)).sum()
    
    total_high = high_vol_this_period.sum()
    total_low = (~high_vol_this_period).sum()
    
    if total_high > 0:
        prob_high_to_high = high_to_high / total_high
        prob_high_to_low = high_to_low / total_high
    else:
        prob_high_to_high = np.nan
        prob_high_to_low = np.nan
    
    if total_low > 0:
        prob_low_to_low = low_to_low / total_low
        prob_low_to_high = low_to_high / total_low
    else:
        prob_low_to_low = np.nan
        prob_low_to_high = np.nan
    
    # 4. Compute average run length of high vs low volatility regimes
    # Create regime indicator
    regime = (df_copy['atr'] > atr_median).astype(int)
    
    # Calculate run lengths
    regime_changes = (regime != regime.shift()).astype(int)
    run_ids = regime_changes.cumsum()
    
    run_lengths = regime.groupby(run_ids).size()
    high_vol_runs = regime.groupby(run_ids).first() == 1
    
    avg_run_length_high = run_lengths[high_vol_runs].mean() if high_vol_runs.sum() > 0 else 0
    avg_run_length_low = run_lengths[~high_vol_runs].mean() if (~high_vol_runs).sum() > 0 else 0
    
    results = {
        'atr_autocorr_lag1': atr_autocorr_lag1,
        'squared_returns_autocorr_lag1': squared_returns_autocorr_lag1,
        'prob_high_to_high': prob_high_to_high,
        'prob_high_to_low': prob_high_to_low,
        'prob_low_to_low': prob_low_to_low,
        'prob_low_to_high': prob_low_to_high,
        'avg_run_length_high': avg_run_length_high,
        'avg_run_length_low': avg_run_length_low,
        'atr_median': atr_median,
        'total_observations': len(df_copy)
    }
    
    return results


def create_volatility_clustering_plot(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Create a plot showing volatility clustering over time.
    
    Phase 3 - Volatility Analysis Step 5B: Shows how high-volatility periods 
    cluster together over time using 1-Hour timeframe data with ATR.
    
    Computes median ATR as threshold, then color-codes each point:
    - High volatility (ATR > median): Red
    - Low volatility (ATR ≤ median): Blue
    
    Plots ATR over time with color coding and shades background zones for 
    high-volatility regimes to visualize clustering.
    
    Args:
        df: DataFrame with timestamp index and 'atr' column (1-Hour timeframe)
        save_path: Path to save the chart (defaults to data/processed/volatility_clustering.png)
    """
    if 'atr' not in df.columns:
        raise ValueError("DataFrame must contain 'atr' column")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    
    if save_path is None:
        save_path = os.path.join(config.PROCESSED_DATA_PATH, 'volatility_clustering.png')
    
    # Prepare data
    df_copy = df.copy()
    df_copy = df_copy[df_copy['atr'].notna()].copy()
    
    if len(df_copy) == 0:
        raise ValueError("No valid ATR data found")
    
    # Calculate median ATR as threshold
    atr_median = df_copy['atr'].median()
    
    # Create color mapping: High volatility (ATR > median) = red, Low (ATR ≤ median) = blue
    high_vol = df_copy['atr'] > atr_median
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Identify high-volatility regime zones for background shading
    # Find start and end of high-volatility regimes
    high_vol_regimes = []
    in_high_regime = False
    regime_start = None
    prev_idx = None
    
    for idx, is_high in high_vol.items():
        if is_high and not in_high_regime:
            # Start of high-volatility regime
            regime_start = idx
            in_high_regime = True
        elif not is_high and in_high_regime:
            # End of high-volatility regime
            if regime_start is not None:
                # Use previous index as the end of the regime
                if prev_idx is not None:
                    high_vol_regimes.append((regime_start, prev_idx))
            in_high_regime = False
            regime_start = None
        prev_idx = idx
    
    # Handle case where regime extends to the end
    if in_high_regime and regime_start is not None:
        high_vol_regimes.append((regime_start, df_copy.index[-1]))
    
    # Shade background zones for high-volatility regimes
    for regime_start, regime_end in high_vol_regimes:
        ax.axvspan(regime_start, regime_end, alpha=0.2, color='red', zorder=0)
    
    # Plot ATR over time with color coding
    # Create segments for color-coded line plot
    colors = ['red' if hv else 'blue' for hv in high_vol]
    
    # Plot line segments with colors
    for i in range(len(df_copy) - 1):
        ax.plot([df_copy.index[i], df_copy.index[i+1]], 
               [df_copy['atr'].iloc[i], df_copy['atr'].iloc[i+1]],
               color=colors[i], linewidth=1.5, alpha=0.7)
    
    # Plot points with color coding for better visibility
    high_vol_data = df_copy[high_vol]
    low_vol_data = df_copy[~high_vol]
    
    if len(high_vol_data) > 0:
        ax.scatter(high_vol_data.index, high_vol_data['atr'], 
                  color='red', alpha=0.6, s=15, zorder=3, label='High Volatility (ATR > median)')
    
    if len(low_vol_data) > 0:
        ax.scatter(low_vol_data.index, low_vol_data['atr'], 
                  color='blue', alpha=0.6, s=15, zorder=3, label='Low Volatility (ATR ≤ median)')
    
    # Add median threshold line
    ax.axhline(y=atr_median, color='black', linestyle='--', linewidth=2, 
              label=f'Median ATR ({atr_median:.2f})', zorder=2)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('ATR', fontsize=12)
    ax.set_title('Volatility Clustering: ATR Over Time\n(Red Bursts Show High-Volatility Clustering)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=config.CHART_DPI, bbox_inches='tight')
    print(f"✓ Volatility clustering plot saved to: {save_path}")
    
    plt.close()


def generate_analysis_summary(symbol: str = None, save_path: str = None) -> str:
    """
    Generate a comprehensive analysis summary report in Markdown format.
    
    Runs all analysis functions and compiles results into a structured report.
    
    Args:
        symbol: Trading symbol (defaults to config.DEFAULT_SYMBOL)
        save_path: Path to save the summary (defaults to data/processed/week2_analysis_summary.md)
        
    Returns:
        Markdown formatted summary string
    """
    if symbol is None:
        symbol = config.DEFAULT_SYMBOL
    
    if save_path is None:
        save_path = os.path.join(config.PROCESSED_DATA_PATH, 'week2_analysis_summary.md')
    
    symbol_sanitized = symbol.replace('/', '_').replace('=', '_').replace('-', '_')
    
    # Initialize markdown content
    md_lines = []
    md_lines.append("# Week 2 Analysis Summary")
    md_lines.append(f"**Symbol:** {symbol}")
    md_lines.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    
    # 1. Volume Distribution Across Timeframes
    md_lines.append("## 1. Volume Distribution Across Timeframes")
    md_lines.append("")
    try:
        volume_stats = analyze_volume_distribution_all_timeframes(symbol)
        if not volume_stats.empty:
            md_lines.append("| Timeframe | Mean Volume | Median Volume | Count |")
            md_lines.append("|-----------|-------------|---------------|-------|")
            for _, row in volume_stats.iterrows():
                md_lines.append(f"| {row['timeframe']} | {row['mean']:,.0f} | {row['median']:,.0f} | {row['count']:,} |")
            md_lines.append("")
            highest_volume = volume_stats.iloc[0]['timeframe']
            md_lines.append(f"**Highest Volume Timeframe:** {highest_volume}")
            md_lines.append("")
        else:
            md_lines.append("*No data available*")
            md_lines.append("")
    except Exception as e:
        md_lines.append(f"*Error: {str(e)}*")
        md_lines.append("")
    
    # 2. Intraday Volume Patterns
    md_lines.append("## 2. Intraday Volume Patterns")
    md_lines.append("")
    try:
        file_path = config.get_processed_data_path(symbol_sanitized, '1Hour')
        if os.path.exists(file_path):
            df_1h = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
            hourly_stats, insights = analyze_intraday_volume(df_1h)
            
            md_lines.append("### Top 3 Busiest Hours (UTC)")
            md_lines.append("")
            for i, hour_data in enumerate(insights['peak_hours'], 1):
                md_lines.append(f"{i}. **Hour {hour_data['hour']:02d}:00** - Mean Volume: {hour_data['mean_volume']:,.0f}")
            md_lines.append("")
            
            md_lines.append("### Quietest 3 Hours (UTC)")
            md_lines.append("")
            for i, hour_data in enumerate(reversed(insights['quiet_hours']), 1):
                md_lines.append(f"{i}. **Hour {hour_data['hour']:02d}:00** - Mean Volume: {hour_data['mean_volume']:,.0f}")
            md_lines.append("")
        else:
            md_lines.append("*1-hour processed data not found*")
            md_lines.append("")
    except Exception as e:
        md_lines.append(f"*Error: {str(e)}*")
        md_lines.append("")
    
    # 3. Volume-Price Relationship
    md_lines.append("## 3. Volume-Price Relationship")
    md_lines.append("")
    try:
        file_path = config.get_processed_data_path(symbol_sanitized, '1Hour')
        if os.path.exists(file_path):
            df_1h = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
            vp_results = analyze_volume_price_relationship(df_1h)
            
            md_lines.append("### Correlations")
            md_lines.append("")
            md_lines.append(f"- **Volume vs Returns:** {vp_results['corr_volume_returns']:.4f}")
            md_lines.append(f"- **Volume vs Absolute Returns:** {vp_results['corr_volume_abs_returns']:.4f}")
            md_lines.append("")
            
            md_lines.append("### Average Absolute Returns by Volume")
            md_lines.append("")
            md_lines.append(f"- **Low Volume (bottom 25%):** {vp_results['low_volume_avg_abs_return']:.6f}")
            md_lines.append(f"- **High Volume (top 25%):** {vp_results['high_volume_avg_abs_return']:.6f}")
            md_lines.append(f"- **Ratio (High/Low):** {vp_results['high_volume_avg_abs_return'] / vp_results['low_volume_avg_abs_return']:.2f}x")
            md_lines.append("")
        else:
            md_lines.append("*1-hour processed data not found*")
            md_lines.append("")
    except Exception as e:
        md_lines.append(f"*Error: {str(e)}*")
        md_lines.append("")
    
    # 4. Volatility Across Timeframes
    md_lines.append("## 4. Volatility Across Timeframes")
    md_lines.append("")
    try:
        volatility_stats = analyze_volatility_distribution_all_timeframes(symbol)
        if not volatility_stats.empty:
            md_lines.append("| Timeframe | Mean ATR | Median ATR | ATR % |")
            md_lines.append("|-----------|----------|------------|-------|")
            for _, row in volatility_stats.iterrows():
                md_lines.append(f"| {row['timeframe']} | {row['mean_atr']:.2f} | {row['median_atr']:.2f} | {row['mean_atr_pct']:.3f}% |")
            md_lines.append("")
            most_volatile = volatility_stats.iloc[0]['timeframe']
            least_volatile = volatility_stats.iloc[-1]['timeframe']
            md_lines.append(f"**Most Volatile:** {most_volatile}")
            md_lines.append(f"**Least Volatile:** {least_volatile}")
            md_lines.append("")
        else:
            md_lines.append("*No data available*")
            md_lines.append("")
    except Exception as e:
        md_lines.append(f"*Error: {str(e)}*")
        md_lines.append("")
    
    # 5. Intraday Volatility Patterns
    md_lines.append("## 5. Intraday Volatility Patterns")
    md_lines.append("")
    try:
        file_path = config.get_processed_data_path(symbol_sanitized, '1Hour')
        if os.path.exists(file_path):
            df_1h = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
            hourly_stats, _ = analyze_intraday_volatility(df_1h, save_path=None)  # Don't save during summary generation
            
            # Find peak volatility hours
            peak_hours = hourly_stats.nlargest(3, 'mean_atr')
            quiet_hours = hourly_stats.nsmallest(3, 'mean_atr')
            
            md_lines.append("### Peak Volatility Hours (UTC)")
            md_lines.append("")
            for i, (_, row) in enumerate(peak_hours.iterrows(), 1):
                md_lines.append(f"{i}. **Hour {int(row['hour']):02d}:00** - Mean ATR: {row['mean_atr']:.2f}")
            md_lines.append("")
            
            md_lines.append("### Quietest Volatility Hours (UTC)")
            md_lines.append("")
            for i, (_, row) in enumerate(quiet_hours.iterrows(), 1):
                md_lines.append(f"{i}. **Hour {int(row['hour']):02d}:00** - Mean ATR: {row['mean_atr']:.2f}")
            md_lines.append("")
        else:
            md_lines.append("*1-hour processed data not found*")
            md_lines.append("")
    except Exception as e:
        md_lines.append(f"*Error: {str(e)}*")
        md_lines.append("")
    
    # 6. Volatility Clustering
    md_lines.append("## 6. Volatility Clustering")
    md_lines.append("")
    try:
        file_path = config.get_processed_data_path(symbol_sanitized, '1Hour')
        if os.path.exists(file_path):
            df_1h = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
            clustering_results = test_volatility_clustering(df_1h)
            
            md_lines.append("### Autocorrelation")
            md_lines.append("")
            md_lines.append(f"- **ATR Autocorrelation (lag=1):** {clustering_results['atr_autocorr_lag1']:.4f}")
            md_lines.append(f"- **Squared Returns Autocorrelation (lag=1):** {clustering_results['squared_returns_autocorr_lag1']:.4f}")
            md_lines.append("")
            
            md_lines.append("### Persistence Probabilities")
            md_lines.append("")
            md_lines.append(f"- **P(High → High):** {clustering_results['prob_high_to_high']:.2%}")
            md_lines.append(f"- **P(Low → Low):** {clustering_results['prob_low_to_low']:.2%}")
            md_lines.append(f"- **P(High → Low):** {clustering_results['prob_high_to_low']:.2%}")
            md_lines.append(f"- **P(Low → High):** {clustering_results['prob_low_to_high']:.2%}")
            md_lines.append("")
            
            md_lines.append("### Regime Durations")
            md_lines.append("")
            md_lines.append(f"- **Average High Volatility Run Length:** {clustering_results['avg_run_length_high']:.1f} periods")
            md_lines.append(f"- **Average Low Volatility Run Length:** {clustering_results['avg_run_length_low']:.1f} periods")
            md_lines.append("")
        else:
            md_lines.append("*1-hour processed data not found*")
            md_lines.append("")
    except Exception as e:
        md_lines.append(f"*Error: {str(e)}*")
        md_lines.append("")
    
    # Key Findings
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## Key Findings")
    md_lines.append("")
    
    # Generate key findings based on analysis results
    findings = []
    
    try:
        # Re-run analyses to get fresh data for findings
        volume_stats = analyze_volume_distribution_all_timeframes(symbol)
        if not volume_stats.empty:
            highest_vol = volume_stats.iloc[0]['timeframe']
            findings.append(f"**{highest_vol}** timeframe shows the highest mean trading volume across all timeframes.")
        
        # Intraday volume findings
        file_path = config.get_processed_data_path(symbol_sanitized, '1Hour')
        if os.path.exists(file_path):
            df_1h = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
            hourly_stats, insights = analyze_intraday_volume(df_1h)
            peak_hour = insights['highest_activity_hour']
            findings.append(f"Peak trading activity occurs at **{peak_hour:02d}:00 UTC**, with significantly higher volume compared to quiet hours.")
            
            # Volume-price relationship
            vp_results = analyze_volume_price_relationship(df_1h)
            if vp_results['corr_volume_abs_returns'] > 0.3:
                findings.append(f"Strong positive correlation ({vp_results['corr_volume_abs_returns']:.3f}) between volume and absolute returns suggests high volume periods predict larger price movements.")
            
            # Volatility findings
            volatility_stats = analyze_volatility_distribution_all_timeframes(symbol)
            if not volatility_stats.empty:
                most_vol = volatility_stats.iloc[0]['timeframe']
                least_vol = volatility_stats.iloc[-1]['timeframe']
                findings.append(f"**{most_vol}** timeframe exhibits the highest volatility (normalized ATR), while **{least_vol}** shows the lowest volatility.")
            
            # Volatility clustering
            clustering_results = test_volatility_clustering(df_1h)
            if clustering_results['atr_autocorr_lag1'] > 0.5:
                findings.append(f"Strong volatility clustering detected (ATR autocorrelation: {clustering_results['atr_autocorr_lag1']:.3f}), indicating volatility tends to persist over time with average high-volatility regimes lasting {clustering_results['avg_run_length_high']:.1f} periods.")
    except Exception as e:
        findings.append(f"*Error generating automated findings: {str(e)}*")
    
    # Add findings as bullets
    for finding in findings[:5]:  # Limit to 5 findings
        md_lines.append(f"- {finding}")
    
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("*Report generated automatically by the Quantitative Trading Research System*")
    
    # Combine all lines
    summary = "\n".join(md_lines)
    
    # Save to file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(summary)
    
    print(f"✓ Analysis summary saved to: {save_path}")
    
    return summary


# ============================================================================
# Volatility Visualization Helper Functions
# ============================================================================

def generate_volatility_visualizations(symbol: str = None) -> None:
    """
    Generate both volatility visualizations (Phase 3 - Steps 5A and 5B).
    
    Creates:
    1. Intraday Volatility chart (data/processed/intraday_volatility.png)
    2. Volatility Clustering chart (data/processed/volatility_clustering.png)
    
    Both use the 1-Hour timeframe processed data with ATR.
    
    Args:
        symbol: Trading symbol (defaults to config.DEFAULT_SYMBOL)
    """
    if symbol is None:
        symbol = config.DEFAULT_SYMBOL
    
    # Sanitize symbol for filename
    symbol_sanitized = symbol.replace('/', '_').replace('=', '_').replace('-', '_')
    
    # Load 1-Hour processed data
    file_path = config.get_processed_data_path(symbol_sanitized, '1Hour')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"1-Hour processed data not found: {file_path}\n"
                               f"Please run data processing first.")
    
    print("="*70)
    print("Generating Volatility Visualizations (Phase 3 - Steps 5A & 5B)")
    print("="*70)
    print()
    
    # Load data
    df_1h = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
    
    # Check if ATR column exists
    if 'atr' not in df_1h.columns:
        raise ValueError("ATR column not found in processed data. Please run indicator calculation first.")
    
    # Generate Step 5A: Intraday Volatility
    print("Step 5A: Generating Intraday Volatility Chart...")
    print("-" * 70)
    try:
        hourly_stats, _ = analyze_intraday_volatility(df_1h)
        print(f"✓ Successfully generated intraday volatility chart")
        print(f"  Peak volatility hours (13-17 UTC) highlighted and labeled")
        print()
    except Exception as e:
        print(f"⚠️  Error generating intraday volatility chart: {str(e)}")
        print()
    
    # Generate Step 5B: Volatility Clustering
    print("Step 5B: Generating Volatility Clustering Chart...")
    print("-" * 70)
    try:
        create_volatility_clustering_plot(df_1h)
        print(f"✓ Successfully generated volatility clustering chart")
        print(f"  High-volatility regimes (red) and low-volatility periods (blue) shown")
        print(f"  Background shading highlights high-volatility clustering zones")
        print()
    except Exception as e:
        print(f"⚠️  Error generating volatility clustering chart: {str(e)}")
        print()
    
    print("="*70)
    print("Volatility Visualizations Complete")
    print("="*70)
    print(f"\nFiles saved to:")
    print(f"  - {os.path.join(config.PROCESSED_DATA_PATH, 'intraday_volatility.png')}")
    print(f"  - {os.path.join(config.PROCESSED_DATA_PATH, 'volatility_clustering.png')}")


# ============================================================================
# Main execution for testing
# ============================================================================

if __name__ == "__main__":
    """
    Run all volume analysis functions.
    """
    run_all_volume_analysis()

