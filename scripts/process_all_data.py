"""
Data Processing Script for AI-Driven Quantitative Trading Research System.

This script processes all raw CSV files by:
1. Loading each raw CSV file
2. Cleaning the data (sort, drop duplicates, convert types)
3. Adding all technical indicators
4. Saving to processed directory with indicators

Usage:
    python scripts/process_all_data.py
"""

import argparse
import pandas as pd
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path to import modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config
from src.indicators import add_all_indicators
from src.analysis import generate_volatility_visualizations, create_volume_heatmap, create_volume_by_timeframe_chart
from src.config_manager import (
    load_config,
    validate_config,
    get_setting,
    get_timeframes,
    get_sanitized_symbol,
)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data: sort, drop duplicates, convert types.
    
    Args:
        df: Raw DataFrame from CSV
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Ensure timestamp is datetime and set as index
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have timestamp column or DatetimeIndex")
    
    # Sort chronologically
    df = df.sort_index()
    
    # Drop duplicates (keep first occurrence)
    initial_count = len(df)
    df = df[~df.index.duplicated(keep='first')]
    duplicates_removed = initial_count - len(df)
    
    if duplicates_removed > 0:
        print(f"    Removed {duplicates_removed} duplicate rows")
    
    # Convert OHLCV columns to numeric
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows where essential price data is missing
    essential_cols = ['open', 'high', 'low', 'close']
    missing_essential = df[essential_cols].isna().any(axis=1)
    if missing_essential.any():
        removed = missing_essential.sum()
        df = df[~missing_essential].reset_index(drop=False)
        print(f"    Removed {removed} rows with missing essential price data")
    
    # Ensure timestamp index is timezone-aware (UTC)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
    
    return df


def verify_indicators(df: pd.DataFrame) -> dict:
    """
    Verify indicator calculations meet expected criteria.
    
    Args:
        df: DataFrame with indicators
        
    Returns:
        Dictionary with verification results
    """
    verification = {
        'passed': True,
        'warnings': []
    }
    
    # Check RSI is between 0-100
    if 'rsi' in df.columns:
        rsi_valid = df['rsi'].dropna()
        if len(rsi_valid) > 0:
            rsi_min = rsi_valid.min()
            rsi_max = rsi_valid.max()
            if rsi_min < 0 or rsi_max > 100:
                verification['warnings'].append(
                    f"RSI out of range: min={rsi_min:.2f}, max={rsi_max:.2f}"
                )
                verification['passed'] = False
    
    # Check ATR is positive
    if 'atr' in df.columns:
        atr_valid = df['atr'].dropna()
        if len(atr_valid) > 0:
            atr_min = atr_valid.min()
            if atr_min < 0:
                verification['warnings'].append(
                    f"ATR has negative values: min={atr_min:.2f}"
                )
                verification['passed'] = False
    
    # Check SMA_200 has NaN for first 200 rows (normal behavior)
    if 'sma_long' in df.columns:
        sma_long_valid = df['sma_long'].dropna()
        first_valid_idx = df['sma_long'].first_valid_index()
        if first_valid_idx is not None:
            first_valid_pos = df.index.get_loc(first_valid_idx)
            if first_valid_pos > config.SMA_LONG_LENGTH:
                verification['warnings'].append(
                    f"SMA_long first valid value at position {first_valid_pos} "
                    f"(expected ~{config.SMA_LONG_LENGTH})"
                )
    
    # Check VWAP increases smoothly (for intraday data)
    if 'vwap' in df.columns:
        vwap_valid = df['vwap'].dropna()
        if len(vwap_valid) > 1:
            # Check if VWAP resets daily (for intraday data)
            # This is a basic check - VWAP should be relatively smooth
            vwap_diff = vwap_valid.diff()
            large_negative_jumps = (vwap_diff < -0.1 * vwap_valid.shift(1)).sum()
            if large_negative_jumps > len(vwap_valid) * 0.1:  # More than 10% large negative jumps
                verification['warnings'].append(
                    f"VWAP has {large_negative_jumps} large negative jumps (may indicate daily resets)"
                )
    
    return verification


def process_timeframe(symbol: str, timeframe: str) -> bool:
    """
    Process a single timeframe: load, clean, add indicators, save.
    
    Args:
        symbol: Trading symbol (e.g., 'XAU_USD')
        timeframe: Timeframe string (e.g., '1Hour')
        
    Returns:
        True if successful, False otherwise
    """
    # Generate file paths
    raw_file = config.get_raw_data_path(symbol, timeframe)
    processed_file = config.get_processed_data_path(symbol, timeframe)
    
    # Check if raw file exists
    if not os.path.exists(raw_file):
        print(f"  ⚠️  Raw file not found: {raw_file}")
        return False
    
    try:
        # Load raw CSV
        print(f"  Loading {timeframe} data...", end=' ')
        df = pd.read_csv(raw_file, parse_dates=['timestamp'])
        print(f"✓ ({len(df)} rows)")
        
        # Clean data
        print(f"    Cleaning data...", end=' ')
        df = clean_data(df)
        print(f"✓ ({len(df)} rows)")
        
        # Add all indicators
        print(f"    Calculating indicators...", end=' ')
        df = add_all_indicators(df)
        print(f"✓")
        
        # Verify indicators
        print(f"    Verifying indicators...", end=' ')
        verification = verify_indicators(df)
        if verification['passed']:
            print("✓")
        else:
            print("⚠️  Warnings:")
            for warning in verification['warnings']:
                print(f"      - {warning}")
        
        # Ensure processed directory exists
        os.makedirs(config.PROCESSED_DATA_PATH, exist_ok=True)
        
        # Reset index to save timestamp as column
        df_to_save = df.reset_index()
        
        # Save processed CSV
        print(f"    Saving to {processed_file}...", end=' ')
        df_to_save.to_csv(processed_file, index=False)
        print(f"✓")
        
        # Print summary
        print(f"  ✓ Processed {timeframe} data — {len(df):,} rows")
        print(f"    Columns: {', '.join(df.columns.tolist())}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error processing {timeframe}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def process_asset_data(asset_config: Dict[str, Any]) -> Dict[str, bool]:
    """
    Process all timeframes for the asset defined in configuration.
    
    Args:
        asset_config: Configuration dictionary for the asset
        
    Returns:
        Dictionary mapping timeframe to processing success (True/False)
    """
    symbol = get_setting(asset_config, 'asset.symbol')
    sanitized_symbol = get_sanitized_symbol(asset_config)
    timeframes = get_timeframes(asset_config)
    
    print("="*70)
    print("AI-Driven Quantitative Trading Research System")
    print("Data Processing Module")
    print("="*70)
    print()
    
    print(f"Processing data for: {symbol}")
    print(f"Timeframes: {', '.join(timeframes)}")
    print(f"Symbol (sanitized): {sanitized_symbol}")
    print()
    
    results = {}
    for timeframe in timeframes:
        print(f"Processing {timeframe}...")
        success = process_timeframe(sanitized_symbol, timeframe)
        results[timeframe] = success
        print()
    
    successful = sum(1 for success in results.values() if success)
    failed = len(timeframes) - successful
    
    print("="*70)
    print(f"Processing Complete: {successful}/{len(timeframes)} timeframes successful")
    if failed > 0:
        print(f"Failed: {failed} timeframes")
    print("="*70)
    
    # List processed files
    if successful > 0:
        print("\nProcessed files:")
        for timeframe in timeframes:
            processed_file = config.get_processed_data_path(sanitized_symbol, timeframe)
            if os.path.exists(processed_file):
                file_size = os.path.getsize(processed_file) / 1024  # KB
                print(f"  ✓ {processed_file} ({file_size:.1f} KB)")
    
    # Generate visualizations if data was processed successfully
    if successful > 0:
        print()
        print("="*70)
        print("Generating Visualizations")
        print("="*70)
        
        # Check if 1Hour processed data exists for volume and volatility visualizations
        hour_file = config.get_processed_data_path(sanitized_symbol, '1Hour')
        
        # Generate Volume Visualizations
        print("\nVolume Visualizations:")
        print("-" * 70)
        try:
            # Volume heatmap (requires 1Hour data)
            if os.path.exists(hour_file):
                print("  Generating volume heatmap...", end=' ')
                create_volume_heatmap(symbol)
                print("✓")
            else:
                print(f"  ⚠️  1Hour data not found: {hour_file}")
                print("     Skipping volume heatmap")
            
            # Volume by timeframe chart (works with any processed data)
            print("  Generating volume by timeframe chart...", end=' ')
            create_volume_by_timeframe_chart(symbol)
            print("✓")
        except Exception as e:
            print(f"  ⚠️  Error generating volume visualizations: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Generate Volatility Visualizations
        print("\nVolatility Visualizations:")
        print("-" * 70)
        if os.path.exists(hour_file):
            try:
                generate_volatility_visualizations(symbol)
            except Exception as e:
                print(f"  ⚠️  Error generating volatility visualizations: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  ⚠️  1Hour processed data not found: {hour_file}")
            print("     Skipping volatility visualizations")
        
        # List all generated visualization files
        print()
        print("="*70)
        print("Generated Visualization Files:")
        print("="*70)
        
        visualization_files = [
            ('volume_heatmap.png', 'Volume Heatmap'),
            ('volume_by_timeframe.png', 'Volume by Timeframe'),
            ('intraday_volatility.png', 'Intraday Volatility'),
            ('volatility_clustering.png', 'Volatility Clustering')
        ]
        
        for filename, description in visualization_files:
            filepath = os.path.join(config.PROCESSED_DATA_PATH, filename)
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath) / 1024  # KB
                print(f"  ✓ {description:30s} - {filename:35s} ({file_size:.1f} KB)")
            else:
                print(f"  ⚠️  {description:30s} - {filename:35s} (not generated)")
        
        print("="*70)
    
    return results


def main():
    """
    Main function: process all raw CSV files for all timeframes.
    """
    parser = argparse.ArgumentParser(description="Process raw data and add indicators for an asset.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to asset configuration YAML file (default: use config.DEFAULT settings)",
    )
    args = parser.parse_args()
    
    if args.config:
        asset_config = load_config(args.config)
        validate_config(asset_config)
        process_asset_data(asset_config)
    else:
        # Fallback to default config settings
        default_config = {
            'asset': {
                'symbol': config.DEFAULT_SYMBOL,
            },
            'data': {
                'timeframes': config.DEFAULT_TIMEFRAMES,
                'primary_timeframe': '1Hour',
            }
        }
        process_asset_data(default_config)


if __name__ == "__main__":
    main()

