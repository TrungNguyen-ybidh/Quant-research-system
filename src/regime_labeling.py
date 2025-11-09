"""
Regime Labeling Module for Automatic Market Regime Classification

This module automatically classifies hourly data points into "up", "down", or "range"
regimes based on technical indicators (ADX, SMA-50, SMA-200).

Labeling Rules:
- "up": ADX > 25, close > SMA-50, SMA-50 > SMA-200
- "down": ADX > 25, close < SMA-50, SMA-50 < SMA-200
- "range": Everything else (mixed signals or low ADX periods)

"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict, Any

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.config_manager import (
    get_setting,
    get_processed_data_file_path,
    get_regime_labels_path,
)


def label_regimes(df: pd.DataFrame, adx_threshold: float = 25.0) -> pd.DataFrame:
    """
    Automatically label data points into "up", "down", or "range" regimes.
    
    Args:
        df: DataFrame with columns: close, adx, sma_long (SMA-50), sma_200
        adx_threshold: ADX threshold for strong trend (default: 25.0)
        
    Returns:
        DataFrame with added columns: regime_label, regime_numeric
    """
    result_df = df.copy()
    
    # Check required columns
    required_cols = ['close', 'adx', 'sma_long', 'sma_200']
    missing_cols = [col for col in required_cols if col not in result_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Initialize regime labels
    result_df['regime_label'] = 'range'  # Default to range
    result_df['regime_numeric'] = 0  # 0 = range, 1 = up, 2 = down
    
    # Create masks for each regime
    # Up regime: ADX > threshold, close > SMA-50, SMA-50 > SMA-200
    up_mask = (
        (result_df['adx'] > adx_threshold) &
        (result_df['close'] > result_df['sma_long']) &
        (result_df['sma_long'] > result_df['sma_200'])
    )
    
    # Down regime: ADX > threshold, close < SMA-50, SMA-50 < SMA-200
    down_mask = (
        (result_df['adx'] > adx_threshold) &
        (result_df['close'] < result_df['sma_long']) &
        (result_df['sma_long'] < result_df['sma_200'])
    )
    
    # Apply labels
    result_df.loc[up_mask, 'regime_label'] = 'up'
    result_df.loc[up_mask, 'regime_numeric'] = 1
    
    result_df.loc[down_mask, 'regime_label'] = 'down'
    result_df.loc[down_mask, 'regime_numeric'] = 2
    
    return result_df


def validate_labeling(df: pd.DataFrame) -> dict:
    """
    Validate labeling results by checking distribution and statistics.
    
    Args:
        df: DataFrame with regime_label column
        
    Returns:
        Dictionary with validation statistics
    """
    if 'regime_label' not in df.columns:
        raise ValueError("DataFrame must contain 'regime_label' column")
    
    # Calculate label distribution
    label_counts = df['regime_label'].value_counts()
    total = len(df)
    
    distribution = {
        'range': label_counts.get('range', 0) / total * 100,
        'up': label_counts.get('up', 0) / total * 100,
        'down': label_counts.get('down', 0) / total * 100
    }
    
    # Calculate average hourly return for each regime
    if 'close' in df.columns:
        df = df.copy()
        df['hourly_return'] = df['close'].pct_change() * 100
        
        returns_by_regime = {}
        for regime in ['range', 'up', 'down']:
            regime_mask = df['regime_label'] == regime
            regime_returns = df.loc[regime_mask, 'hourly_return'].dropna()
            returns_by_regime[regime] = {
                'mean': regime_returns.mean(),
                'std': regime_returns.std(),
                'count': len(regime_returns)
            }
    else:
        returns_by_regime = None
    
    validation_stats = {
        'total_samples': total,
        'label_counts': label_counts.to_dict(),
        'distribution_pct': distribution,
        'returns_by_regime': returns_by_regime
    }
    
    return validation_stats


def visualize_labeled_data(df: pd.DataFrame, output_path: str = None, 
                          sample_size: int = 5000):
    """
    Visualize labeled data by plotting price with regime colors.
    
    Args:
        df: DataFrame with close and regime_label columns
        output_path: Path to save the plot (optional)
        sample_size: Number of samples to plot (for performance)
    """
    if 'close' not in df.columns or 'regime_label' not in df.columns:
        raise ValueError("DataFrame must contain 'close' and 'regime_label' columns")
    
    # Sample data if too large
    if len(df) > sample_size:
        df_plot = df.tail(sample_size).copy()
        print(f"Plotting last {sample_size} samples for performance")
    else:
        df_plot = df.copy()
    
    # Ensure timestamp is index
    if 'timestamp' in df_plot.columns:
        df_plot = df_plot.set_index('timestamp')
    
    if not isinstance(df_plot.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex for plotting")
    
    # Create color map
    color_map = {
        'up': 'green',
        'down': 'red',
        'range': 'gray'
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot price with regime colors
    for regime in ['range', 'up', 'down']:
        regime_mask = df_plot['regime_label'] == regime
        if regime_mask.any():
            regime_data = df_plot.loc[regime_mask]
            ax.plot(regime_data.index, regime_data['close'], 
                   color=color_map[regime], alpha=0.6, linewidth=1,
                   label=f'{regime.capitalize()} regime')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.set_title('Gold Price with Regime Labels', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()


def run_regime_labeling(data_path: str = None, output_path: str = None,
                        adx_threshold: float = 25.0,
                        asset_config: Dict[str, Any] = None):
    """
    Main function to run regime labeling on 1-hour Gold data.
    
    Args:
        data_path: Path to processed 1-hour data (defaults to config)
        output_path: Path to save labeled data (defaults to config)
        adx_threshold: ADX threshold for strong trend (default: 25.0)
        
    Returns:
        DataFrame with regime labels
    """
    print("=" * 80)
    print("REGIME LABELING: AUTOMATIC MARKET REGIME CLASSIFICATION")
    print("=" * 80)
    
    if asset_config:
        if data_path is None:
            timeframe = get_setting(asset_config, 'data.primary_timeframe')
            data_path = get_processed_data_file_path(asset_config, timeframe)
        if output_path is None:
            output_path = get_regime_labels_path(asset_config)
        if adx_threshold is None:
            adx_threshold = get_setting(asset_config, 'analysis.adx_threshold')
        
        asset_name = get_setting(asset_config, 'asset.name')
        symbol = get_setting(asset_config, 'asset.symbol')
        print(f"Asset: {asset_name} ({symbol})")
    else:
        # Set default paths for legacy usage
        if data_path is None:
            data_path = os.path.join(
                config.PROCESSED_DATA_PATH,
                'XAU_USD_1Hour_with_indicators.csv'
            )
        if output_path is None:
            output_path = os.path.join(config.PROCESSED_DATA_PATH, 'regime_labels.csv')
    
    # Load data
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
    print(f"✓ Loaded {len(df)} rows")
    
    # Apply labeling
    print(f"\nApplying regime labels (ADX threshold: {adx_threshold})...")
    df_labeled = label_regimes(df, adx_threshold=adx_threshold)
    print(f"✓ Labeled {len(df_labeled)} rows")
    
    # Validate results
    print("\nValidating labeling results...")
    validation_stats = validate_labeling(df_labeled)
    
    print("\n" + "-" * 80)
    print("LABEL DISTRIBUTION")
    print("-" * 80)
    print(f"Total samples: {validation_stats['total_samples']:,}")
    print(f"\nLabel counts:")
    for regime, count in validation_stats['label_counts'].items():
        pct = validation_stats['distribution_pct'][regime]
        print(f"  {regime.capitalize()}: {count:,} ({pct:.1f}%)")
    
    print("\n" + "-" * 80)
    print("AVERAGE HOURLY RETURNS BY REGIME")
    print("-" * 80)
    if validation_stats['returns_by_regime']:
        for regime in ['range', 'up', 'down']:
            stats = validation_stats['returns_by_regime'][regime]
            print(f"{regime.capitalize()}:")
            print(f"  Mean return: {stats['mean']:.4f}%")
            print(f"  Std dev: {stats['std']:.4f}%")
            print(f"  Sample count: {stats['count']:,}")
    
    # Check if distribution is balanced
    distribution = validation_stats['distribution_pct']
    print("\n" + "-" * 80)
    print("DISTRIBUTION VALIDATION")
    print("-" * 80)
    
    range_pct = distribution['range']
    up_pct = distribution['up']
    down_pct = distribution['down']
    
    # Expected ranges: 30-40% range, 35-38% up, 22-25% down
    range_ok = 30 <= range_pct <= 40
    up_ok = 35 <= up_pct <= 38
    down_ok = 22 <= down_pct <= 25
    
    print(f"Range: {range_pct:.1f}% (expected: 30-40%) {'✓' if range_ok else '⚠'}")
    print(f"Up: {up_pct:.1f}% (expected: 35-38%) {'✓' if up_ok else '⚠'}")
    print(f"Down: {down_pct:.1f}% (expected: 22-25%) {'✓' if down_ok else '⚠'}")
    
    if not (range_ok and up_ok and down_ok):
        print("\n⚠ Warning: Distribution outside expected ranges.")
        print("   Consider adjusting ADX threshold (try 20 or 30 instead of 25)")
    
    # Check statistical validity
    returns = validation_stats['returns_by_regime']
    if returns:
        up_positive = returns['up']['mean'] > 0
        down_negative = returns['down']['mean'] < 0
        range_near_zero = abs(returns['range']['mean']) < 0.01
        
        print("\n" + "-" * 80)
        print("STATISTICAL VALIDATION")
        print("-" * 80)
        print(f"Up periods show positive returns: {up_positive} {'✓' if up_positive else '⚠'}")
        print(f"Down periods show negative returns: {down_negative} {'✓' if down_negative else '⚠'}")
        print(f"Range periods show near-zero returns: {range_near_zero} {'✓' if range_near_zero else '⚠'}")
    
    # Save labeled data
    print(f"\nSaving labeled data to: {output_path}")
    df_labeled.to_csv(output_path)
    print(f"✓ Saved {len(df_labeled)} labeled rows")
    
    # Generate visualization
    viz_path = os.path.join(config.PROCESSED_DATA_PATH, 'regime_labels_visualization.png')
    print(f"\nGenerating visualization...")
    visualize_labeled_data(df_labeled, output_path=viz_path)
    
    print("\n" + "=" * 80)
    print("REGIME LABELING COMPLETE")
    print("=" * 80)
    
    return df_labeled


if __name__ == "__main__":
    # Run regime labeling with default settings
    df_labeled = run_regime_labeling()
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total labeled samples: {len(df_labeled):,}")
    print(f"Date range: {df_labeled.index.min()} to {df_labeled.index.max()}")
    print(f"\nRegime labels saved to: data/processed/regime_labels.csv")
    print(f"Visualization saved to: data/processed/regime_labels_visualization.png")
