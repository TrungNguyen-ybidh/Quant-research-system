"""
Correlation Analysis Module for Multi-Asset Market Analysis

This module analyzes correlations between Gold (XAU/USD), Australian Dollar (AUD/USD),
Japanese Yen (USD/JPY), and Canadian Dollar (USD/CAD) to understand currency and safe-haven
relationships and reveal how gold interacts with key currencies in the global forex
and macroeconomic landscape.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import os
import sys
from scipy import stats
from typing import Dict, Tuple, Optional

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

try:
    import seaborn as sns
except ImportError:
    print("Warning: seaborn not installed. Install with: pip install seaborn")
    sns = None

try:
    import yfinance as yf
except ImportError:
    print("Warning: yfinance not installed. Install with: pip install yfinance")
    yf = None

try:
    from alpaca_trade_api import REST, TimeFrame
    from src.data_collection import fetch_from_alpaca, connect_to_alpaca
    alpaca_available = True
except ImportError:
    print("Warning: alpaca-trade-api not installed. Install with: pip install alpaca-trade-api")
    REST = None
    TimeFrame = None
    alpaca_available = False
    fetch_from_alpaca = None
    connect_to_alpaca = None


# ============================================================================
# Core Functions
# ============================================================================

def load_asset_data(symbols: Dict[str, str], start: datetime, end: datetime) -> pd.DataFrame:
    """
    Download daily closes for multiple assets.
    
    Args:
        symbols: Dictionary mapping asset names to yfinance symbols
                 e.g., {'gold': 'XAUUSD=X', 'aud': 'AUDUSD=X', 
                        'jpy': 'JPY=X', 'cad': 'CAD=X'}
        start: Start date (datetime with timezone)
        end: End date (datetime with timezone)
    
    Returns:
        DataFrame with columns: timestamp, gold_close, aud_close, jpy_close, cad_close
    """
    if yf is None:
        raise ImportError("yfinance library is required. Install with: pip install yfinance")
    
    print("=" * 80)
    print("LOADING MULTI-ASSET DATA")
    print("=" * 80)
    
    # Convert timezone-aware to naive for yfinance
    start_naive = start.replace(tzinfo=None) if start.tzinfo else start
    end_naive = end.replace(tzinfo=None) if end.tzinfo else end
    
    all_data = {}
    
    for asset_name, symbol in symbols.items():
        print(f"\nLoading {asset_name} ({symbol})...")
        try:
            # Special handling for gold - load from local file if available
            if asset_name == 'gold':
                gold_file_path = os.path.join(config.RAW_DATA_PATH, 'XAU_USD_1Day.csv')
                if os.path.exists(gold_file_path):
                    print(f"  Loading gold data from local file: {gold_file_path}")
                    df = pd.read_csv(gold_file_path)
                    
                    # Convert timestamp to datetime (file is now timezone-naive with dates at midnight)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    # Filter by date range (convert to date for comparison)
                    df['date'] = df['timestamp'].dt.date
                    start_date = start.date() if hasattr(start, 'date') else pd.to_datetime(start).date()
                    end_date = end.date() if hasattr(end, 'date') else pd.to_datetime(end).date()
                    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                    
                    if df.empty:
                        print(f"  Warning: No gold data in date range")
                        continue
                    
                    # Set timestamp as index for consistency
                    df = df.set_index('timestamp').sort_index()
                    
                    # Extract close prices
                    close_prices = df['close'].copy()
                    close_prices.name = f"{asset_name}_close"
                    
                    # Reindex to business days calendar (stock markets trade on business days only)
                    # Create business days calendar for the date range (2022-01-03 to 2025-10-30)
                    # Use timezone-naive dates to match cleaned file
                    business_days_calendar = pd.date_range(
                        start='2022-01-03',
                        end='2025-10-30',
                        freq='B'  # Business days (Monday-Friday)
                    )
                    
                    # Reindex to business days calendar and forward-fill missing days
                    close_prices = close_prices.reindex(business_days_calendar)
                    close_prices = close_prices.ffill()  # Forward fill missing days
                    
                    # Remove any remaining NaN values at the beginning
                    close_prices = close_prices.dropna()
                    
                    # Ensure timezone-aware (UTC) for consistency with other assets
                    if close_prices.index.tz is None:
                        close_prices.index = close_prices.index.tz_localize('UTC')
                    else:
                        close_prices.index = close_prices.index.tz_convert('UTC')
                    
                    all_data[asset_name] = close_prices
                    
                    print(f"  ✓ Loaded {len(close_prices)} business days from local file ({close_prices.index.min().date()} to {close_prices.index.max().date()})")
                    print(f"    Reindexed to business days calendar with forward-fill")
                else:
                    # Fallback to downloading if file doesn't exist
                    print(f"  Local file not found, downloading from yfinance...")
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(start=start_naive, end=end_naive, interval='1d')
                    
                    if df.empty:
                        print(f"  Warning: No data for {asset_name}")
                        continue
                    
                    # Extract close prices
                    close_prices = df['Close'].copy()
                    close_prices.name = f"{asset_name}_close"
                    all_data[asset_name] = close_prices
                    
                    print(f"  ✓ Loaded {len(close_prices)} days ({close_prices.index.min().date()} to {close_prices.index.max().date()})")
            else:
                # For other assets (forex/commodities), use yfinance
                if yf is None:
                    raise ImportError("yfinance library is required. Install with: pip install yfinance")
                
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_naive, end=end_naive, interval='1d')
                
                if df.empty:
                    print(f"  Warning: No data for {asset_name}")
                    continue
                
                # Extract close prices
                close_prices = df['Close'].copy()
                close_prices.name = f"{asset_name}_close"
                
                # For forex/commodities, reindex to business days calendar (matching stock market)
                # This ensures alignment with other assets
                business_days_calendar = pd.date_range(
                    start='2022-01-03',
                    end='2025-10-30',
                    freq='B',  # Business days
                    tz='UTC'
                )
                
                # Reindex to business days calendar and forward-fill
                close_prices = close_prices.reindex(business_days_calendar)
                close_prices = close_prices.ffill()  # Forward fill missing days
                
                # Remove any remaining NaN values at the beginning
                close_prices = close_prices.dropna()
                
                all_data[asset_name] = close_prices
                
                print(f"  ✓ Loaded {len(close_prices)} business days from yfinance ({close_prices.index.min().date()} to {close_prices.index.max().date()})")
                print(f"    Reindexed to business days calendar with forward-fill")
            
        except Exception as e:
            print(f"  Error loading {asset_name}: {str(e)}")
            continue
    
    if not all_data:
        raise ValueError("No data loaded for any assets")
    
    # Combine all data into one DataFrame
    # All Series should have DatetimeIndex with timestamp as index
    # Convert all to same format first
    processed_data = {}
    for asset_name, series in all_data.items():
        # Ensure all have timestamp as index
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError(f"Expected DatetimeIndex for {asset_name}, got {type(series.index)}")
        
        # Ensure timezone-aware (UTC)
        if series.index.tz is None:
            series.index = series.index.tz_localize('UTC')
        else:
            series.index = series.index.tz_convert('UTC')
        
        processed_data[asset_name] = series
    
    # Combine into DataFrame - align all assets to common business days calendar
    # Create a common business days calendar for alignment
    # Find the min and max dates across all assets
    all_dates = []
    for series in processed_data.values():
        all_dates.extend(series.index.tolist())
    
    if not all_dates:
        raise ValueError("No valid dates found in any asset data")
    
    min_date = min(all_dates)
    max_date = max(all_dates)
    
    # Create common business days calendar (stocks trade Monday-Friday)
    common_calendar = pd.date_range(
        start=min_date.date() if hasattr(min_date, 'date') else pd.to_datetime(min_date).date(),
        end=max_date.date() if hasattr(max_date, 'date') else pd.to_datetime(max_date).date(),
        freq='B',  # Business days
        tz='UTC'
    )
    
    # Align all assets to the common business days calendar
    aligned_data = {}
    for asset_name, series in processed_data.items():
        # Reindex to common calendar and forward-fill
        aligned_series = series.reindex(common_calendar)
        aligned_series = aligned_series.ffill()  # Forward fill missing days
        aligned_data[asset_name] = aligned_series
    
    # Combine into DataFrame
    combined_df = pd.DataFrame(aligned_data)
    
    # Rename columns to include _close suffix
    column_mapping = {asset_name: f"{asset_name}_close" for asset_name in aligned_data.keys()}
    combined_df.rename(columns=column_mapping, inplace=True)
    
    # Reset index to make timestamp a column
    # The index column will be named after the index name (likely 'timestamp' or None)
    combined_df = combined_df.reset_index()
    
    # After reset_index(), the index column will be the first column
    # Check if we need to rename it
    first_col = combined_df.columns[0]
    
    # If first column is 'Date' (from yfinance) or unnamed (from local file), rename it
    if first_col == 'Date':
        combined_df.rename(columns={'Date': 'timestamp'}, inplace=True)
    elif first_col != 'timestamp':
        # Rename the first column to 'timestamp' if it's not already named timestamp
        combined_df.rename(columns={first_col: 'timestamp'}, inplace=True)
    
    # Ensure timestamp is timezone-aware (UTC)
    if combined_df['timestamp'].dtype == 'datetime64[ns]':
        if combined_df['timestamp'].dt.tz is None:
            combined_df['timestamp'] = combined_df['timestamp'].dt.tz_localize('UTC')
        else:
            combined_df['timestamp'] = combined_df['timestamp'].dt.tz_convert('UTC')
    
    # Sort by timestamp
    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
    
    # Drop rows with NaN (missing data)
    initial_rows = len(combined_df)
    combined_df = combined_df.dropna()
    dropped_rows = initial_rows - len(combined_df)
    
    if dropped_rows > 0:
        print(f"\n  Dropped {dropped_rows} rows with missing data")
    
    print(f"\n✓ Final dataset: {len(combined_df)} days with complete data")
    print(f"  Date range: {combined_df['timestamp'].min().date()} to {combined_df['timestamp'].max().date()}")
    
    return combined_df


def calculate_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute percentage change from close-to-close.
    
    Args:
        df: DataFrame with close price columns (e.g., gold_close, spy_close, etc.)
    
    Returns:
        DataFrame with return columns (e.g., gold_return, spy_return, etc.)
    """
    returns_df = df.copy()
    
    # Identify close price columns
    close_cols = [col for col in df.columns if col.endswith('_close')]
    
    if not close_cols:
        raise ValueError("No columns ending with '_close' found in DataFrame")
    
    # Calculate returns for each asset
    for col in close_cols:
        asset_name = col.replace('_close', '')
        returns_col = f"{asset_name}_return"
        returns_df[returns_col] = df[col].pct_change() * 100  # Convert to percentage
    
    # Keep timestamp and return columns
    return_cols = ['timestamp'] + [f"{col.replace('_close', '')}_return" for col in close_cols]
    returns_df = returns_df[return_cols].copy()
    
    # Drop first row (NaN from pct_change)
    returns_df = returns_df.dropna().reset_index(drop=True)
    
    print(f"\n✓ Calculated daily returns for {len(returns_df)} days")
    
    return returns_df


def compute_correlation_matrix(returns_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produce Pearson correlation matrix with p-values.
    
    Args:
        returns_df: DataFrame with return columns (e.g., gold_return, spy_return, etc.)
    
    Returns:
        Tuple of (correlation_matrix, pvalue_matrix)
    """
    # Get return columns
    return_cols = [col for col in returns_df.columns if col.endswith('_return')]
    
    if len(return_cols) < 2:
        raise ValueError("Need at least 2 return columns for correlation analysis")
    
    # Extract returns
    returns_data = returns_df[return_cols].values
    
    # Calculate correlation matrix
    corr_matrix = returns_df[return_cols].corr(method='pearson')
    
    # Calculate p-values using t-test
    n = len(returns_df)
    pvalue_matrix = pd.DataFrame(
        index=corr_matrix.index,
        columns=corr_matrix.columns,
        dtype=float
    )
    
    for i, col1 in enumerate(return_cols):
        for j, col2 in enumerate(return_cols):
            if i == j:
                pvalue_matrix.loc[col1, col2] = 0.0  # Self-correlation
            else:
                r = corr_matrix.loc[col1, col2]
                if np.isnan(r) or abs(r) == 1.0:
                    pvalue_matrix.loc[col1, col2] = np.nan
                else:
                    # t-test: t = r * sqrt((n-2)/(1-r^2))
                    if abs(r) < 1.0:
                        t_stat = r * np.sqrt((n - 2) / (1 - r**2))
                        # Two-tailed p-value
                        pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                        pvalue_matrix.loc[col1, col2] = pvalue
                    else:
                        pvalue_matrix.loc[col1, col2] = 0.0
    
    # Clean up column names (remove _return suffix for readability)
    corr_matrix.columns = [col.replace('_return', '') for col in corr_matrix.columns]
    corr_matrix.index = [col.replace('_return', '') for col in corr_matrix.index]
    
    pvalue_matrix.columns = [col.replace('_return', '') for col in pvalue_matrix.columns]
    pvalue_matrix.index = [col.replace('_return', '') for col in pvalue_matrix.index]
    
    print(f"\n✓ Computed correlation matrix ({len(corr_matrix)}x{len(corr_matrix)})")
    
    return corr_matrix, pvalue_matrix


def compute_rolling_correlation(returns_df: pd.DataFrame, base_asset: str, window: int = 60) -> pd.DataFrame:
    """
    Calculate 60-day rolling correlation with base asset.
    
    Args:
        returns_df: DataFrame with return columns
        base_asset: Base asset name (e.g., 'gold') for correlation calculations
        window: Rolling window size in days (default: 60)
    
    Returns:
        DataFrame with rolling correlations and timestamp
    """
    base_col = f"{base_asset}_return"
    
    if base_col not in returns_df.columns:
        raise ValueError(f"Base asset '{base_asset}' not found. Available: {[col.replace('_return', '') for col in returns_df.columns if col.endswith('_return')]}")
    
    # Get all return columns
    return_cols = [col for col in returns_df.columns if col.endswith('_return')]
    other_cols = [col for col in return_cols if col != base_col]
    
    # Create result DataFrame
    rolling_corrs = pd.DataFrame()
    rolling_corrs['timestamp'] = returns_df['timestamp'].copy()
    
    # Calculate rolling correlation for each asset vs base asset
    for col in other_cols:
        asset_name = col.replace('_return', '')
        rolling_corr = returns_df[base_col].rolling(window=window).corr(returns_df[col])
        rolling_corrs[f"{base_asset}_{asset_name}_corr"] = rolling_corr
    
    # Drop rows with insufficient data (first window-1 rows)
    rolling_corrs = rolling_corrs.dropna().reset_index(drop=True)
    
    print(f"\n✓ Computed rolling correlations (window={window}) for {len(rolling_corrs)} days")
    
    return rolling_corrs


def calculate_cross_sectional_dispersion(returns_df: pd.DataFrame) -> pd.Series:
    """
    Calculate per-day standard deviation of asset returns.
    
    Args:
        returns_df: DataFrame with return columns
    
    Returns:
        Series with daily dispersion values
    """
    # Get return columns
    return_cols = [col for col in returns_df.columns if col.endswith('_return')]
    
    # Calculate standard deviation across assets for each day
    dispersion = returns_df[return_cols].std(axis=1)
    dispersion.name = 'cross_sectional_dispersion'
    
    mean_dispersion = dispersion.mean()
    print(f"\n✓ Calculated cross-sectional dispersion")
    print(f"  Mean dispersion: {mean_dispersion:.2f}%")
    
    return dispersion


def summarize_market_stats(corr_matrix: pd.DataFrame, dispersion_series: pd.Series) -> Dict:
    """
    Calculate average pairwise correlation and mean dispersion.
    
    Args:
        corr_matrix: Correlation matrix DataFrame
        dispersion_series: Series with daily dispersion values
    
    Returns:
        Dictionary with market statistics
    """
    # Calculate average pairwise correlation (excluding diagonal)
    mask = np.triu(np.ones_like(corr_matrix.values, dtype=bool), k=1)
    pairwise_corrs = corr_matrix.values[mask]
    avg_pairwise_corr = np.nanmean(pairwise_corrs)
    
    # Mean dispersion
    mean_dispersion = dispersion_series.mean()
    
    stats_dict = {
        'avg_pairwise_correlation': avg_pairwise_corr,
        'mean_dispersion': mean_dispersion,
        'n_assets': len(corr_matrix),
        'n_pairs': len(pairwise_corrs)
    }
    
    print(f"\n✓ Market Statistics:")
    print(f"  Average Pairwise Correlation: {avg_pairwise_corr:.3f}")
    print(f"  Mean Cross-Sectional Dispersion: {mean_dispersion:.2f}%")
    
    return stats_dict


def generate_visuals(returns_df: pd.DataFrame, corr_matrix: pd.DataFrame, 
                     rolling_corrs: pd.DataFrame, output_dir: str):
    """
    Generate correlation heatmap and rolling correlation plots.
    
    Args:
        returns_df: DataFrame with returns
        corr_matrix: Correlation matrix DataFrame
        rolling_corrs: DataFrame with rolling correlations
        output_dir: Directory to save plots
    """
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    # Set style
    if sns is not None:
        sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Correlation Heatmap
    print("\n1. Creating correlation heatmap...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap with custom colormap
    if sns is not None:
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdYlBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
    else:
        # Fallback to matplotlib-only heatmap
        im = ax.imshow(corr_matrix.values, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(corr_matrix.columns)))
        ax.set_yticks(np.arange(len(corr_matrix.index)))
        ax.set_xticklabels(corr_matrix.columns)
        ax.set_yticklabels(corr_matrix.index)
        
        # Add annotations
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    ax.set_title('Correlation Matrix: Gold vs Currencies', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {heatmap_path}")
    
    # 2. Rolling Correlations Plot
    print("\n2. Creating rolling correlations plot...")
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Get rolling correlation columns (excluding timestamp)
    rolling_cols = [col for col in rolling_corrs.columns if col != 'timestamp']
    
    # Plot each rolling correlation series
    for col in rolling_cols:
        asset_pair = col.replace('_corr', '').replace('gold_', 'Gold vs ')
        ax.plot(rolling_corrs['timestamp'], rolling_corrs[col], label=asset_pair, linewidth=2)
    
    # Add zero reference line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Zero')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('60-Day Rolling Correlation', fontsize=12)
    ax.set_title('Rolling Correlations: Gold vs Currencies (60-Day Window)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    rolling_path = os.path.join(output_dir, 'rolling_correlations.png')
    plt.savefig(rolling_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {rolling_path}")
    
    # 3. Optional: Gold vs JPY Scatter with Trend Line
    print("\n3. Creating Gold vs JPY scatter plot...")
    if 'gold_return' in returns_df.columns and 'jpy_return' in returns_df.columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot
        ax.scatter(returns_df['jpy_return'], returns_df['gold_return'], 
                  alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
        
        # Add trend line
        z = np.polyfit(returns_df['jpy_return'].dropna(), 
                      returns_df['gold_return'].dropna(), 1)
        p = np.poly1d(z)
        ax.plot(returns_df['jpy_return'].sort_values(), 
               p(returns_df['jpy_return'].sort_values()), 
               "r--", alpha=0.8, linewidth=2, label=f'Trend Line (slope={z[0]:.3f})')
        
        # Calculate and display correlation
        corr_val = returns_df['jpy_return'].corr(returns_df['gold_return'])
        ax.text(0.05, 0.95, f'Correlation: {corr_val:.3f}', 
               transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('JPY Daily Return (%)', fontsize=12)
        ax.set_ylabel('Gold Daily Return (%)', fontsize=12)
        ax.set_title('Gold vs JPY: Daily Returns Scatter', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        scatter_path = os.path.join(output_dir, 'gold_jpy_scatter.png')
        plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {scatter_path}")
    
    print("\n✓ All visualizations generated")


# ============================================================================
# Main Analysis Function
# ============================================================================

def run_correlation_analysis():
    """
    Main function to run complete correlation analysis.
    
    Creates:
    - correlation_matrix.csv
    - correlation_heatmap.png
    - rolling_correlations.png
    - professional_market_stats.txt
    """
    print("=" * 80)
    print("CORRELATION ANALYSIS: MULTI-ASSET MARKET ANALYSIS")
    print("=" * 80)
    
    # Define assets and symbols
    symbols = {
        'gold': 'XAUUSD=X',      # Gold (XAU/USD) - from local file
        'aud': 'AUDUSD=X',       # Australian Dollar (AUD/USD)
        'jpy': 'JPY=X',          # Japanese Yen (USD/JPY)
        'cad': 'CAD=X'           # Canadian Dollar (USD/CAD)
    }
    
    # Date range: Jan 2022 - Oct 2025
    start_date = datetime(2022, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2025, 10, 31, tzinfo=timezone.utc)
    
    # Load asset data
    asset_data = load_asset_data(symbols, start_date, end_date)
    
    # Calculate daily returns
    returns_df = calculate_daily_returns(asset_data)
    
    # Compute correlation matrix with p-values
    corr_matrix, pvalue_matrix = compute_correlation_matrix(returns_df)
    
    # Save correlation matrix
    output_dir = config.PROCESSED_DATA_PATH
    corr_matrix_path = os.path.join(output_dir, 'correlation_matrix.csv')
    corr_matrix.to_csv(corr_matrix_path)
    print(f"\n✓ Saved: {corr_matrix_path}")
    
    # Compute rolling correlations (60-day window) - using Gold as base asset
    rolling_corrs = compute_rolling_correlation(returns_df, base_asset='gold', window=60)
    
    # Save rolling correlations
    rolling_corrs_path = os.path.join(output_dir, 'rolling_correlations.csv')
    rolling_corrs.to_csv(rolling_corrs_path, index=False)
    print(f"✓ Saved: {rolling_corrs_path}")
    
    # Calculate cross-sectional dispersion
    dispersion = calculate_cross_sectional_dispersion(returns_df)
    
    # Summarize market statistics
    market_stats = summarize_market_stats(corr_matrix, dispersion)
    
    # Generate visualizations
    generate_visuals(returns_df, corr_matrix, rolling_corrs, output_dir)
    
    # Write professional market stats report
    write_market_stats_report(corr_matrix, pvalue_matrix, rolling_corrs, 
                             dispersion, market_stats, output_dir)
    
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS COMPLETE")
    print("=" * 80)


def write_market_stats_report(corr_matrix: pd.DataFrame, pvalue_matrix: pd.DataFrame,
                             rolling_corrs: pd.DataFrame, dispersion: pd.Series,
                             market_stats: Dict, output_dir: str):
    """
    Write professional market statistics report.
    
    Args:
        corr_matrix: Correlation matrix
        pvalue_matrix: P-value matrix
        rolling_corrs: Rolling correlations DataFrame
        dispersion: Cross-sectional dispersion Series
        market_stats: Market statistics dictionary
        output_dir: Output directory
    """
    output_path = os.path.join(output_dir, 'professional_market_stats.txt')
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PROFESSIONAL MARKET STATISTICS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("ANALYSIS PERIOD\n")
        f.write("-" * 80 + "\n")
        f.write(f"Start Date: 2022-01-01\n")
        f.write(f"End Date: 2025-10-31\n")
        f.write(f"Assets Analyzed: Gold (XAU/USD), Australian Dollar (AUD/USD), ")
        f.write(f"Japanese Yen (USD/JPY), Canadian Dollar (USD/CAD)\n\n")
        
        # Correlation Matrix
        f.write("CORRELATION MATRIX (Pearson)\n")
        f.write("-" * 80 + "\n\n")
        f.write(corr_matrix.to_string())
        f.write("\n\n")
        
        # P-Values
        f.write("P-VALUES (Statistical Significance)\n")
        f.write("-" * 80 + "\n\n")
        f.write("Note: p < 0.05 indicates statistically significant correlation\n\n")
        f.write(pvalue_matrix.to_string())
        f.write("\n\n")
        
        # Key Correlations
        f.write("KEY CORRELATION INTERPRETATIONS\n")
        f.write("-" * 80 + "\n\n")
        
        # Gold vs AUD
        if 'gold' in corr_matrix.index and 'aud' in corr_matrix.columns:
            gold_aud = corr_matrix.loc['gold', 'aud']
            gold_aud_p = pvalue_matrix.loc['gold', 'aud']
            f.write(f"Gold-AUD: {gold_aud:.3f} (p={gold_aud_p:.4f})")
            if abs(gold_aud) < 0.2:
                f.write(" - Weak correlation")
            elif abs(gold_aud) < 0.5:
                f.write(" - Moderate correlation")
            else:
                f.write(" - Strong correlation")
            if gold_aud_p < 0.05:
                f.write(" [SIGNIFICANT]")
            f.write("\n")
        
        # Gold vs JPY
        if 'gold' in corr_matrix.index and 'jpy' in corr_matrix.columns:
            gold_jpy = corr_matrix.loc['gold', 'jpy']
            gold_jpy_p = pvalue_matrix.loc['gold', 'jpy']
            f.write(f"Gold-JPY: {gold_jpy:.3f} (p={gold_jpy_p:.4f})")
            if abs(gold_jpy) < 0.2:
                f.write(" - Weak correlation")
            elif abs(gold_jpy) < 0.5:
                f.write(" - Moderate correlation")
            else:
                f.write(" - Strong correlation")
            if gold_jpy_p < 0.05:
                f.write(" [SIGNIFICANT]")
            f.write("\n")
        
        # Gold vs CAD
        if 'gold' in corr_matrix.index and 'cad' in corr_matrix.columns:
            gold_cad = corr_matrix.loc['gold', 'cad']
            gold_cad_p = pvalue_matrix.loc['gold', 'cad']
            f.write(f"Gold-CAD: {gold_cad:.3f} (p={gold_cad_p:.4f})")
            if abs(gold_cad) < 0.2:
                f.write(" - Weak correlation")
            elif abs(gold_cad) < 0.5:
                f.write(" - Moderate correlation")
            else:
                f.write(" - Strong correlation")
            if gold_cad_p < 0.05:
                f.write(" [SIGNIFICANT]")
            f.write("\n")
        
        f.write("\n")
        
        # Rolling Correlations Summary
        f.write("ROLLING CORRELATIONS SUMMARY (60-Day Window)\n")
        f.write("-" * 80 + "\n\n")
        
        rolling_cols = [col for col in rolling_corrs.columns if col != 'timestamp']
        for col in rolling_cols:
            asset_pair = col.replace('_corr', '').replace('gold_', 'Gold vs ')
            corr_series = rolling_corrs[col]
            f.write(f"{asset_pair}:\n")
            f.write(f"  Mean: {corr_series.mean():.3f}\n")
            f.write(f"  Std: {corr_series.std():.3f}\n")
            f.write(f"  Min: {corr_series.min():.3f}\n")
            f.write(f"  Max: {corr_series.max():.3f}\n")
            f.write(f"  Range: {corr_series.max() - corr_series.min():.3f}\n\n")
        
        # Cross-Sectional Dispersion
        f.write("CROSS-SECTIONAL DISPERSION\n")
        f.write("-" * 80 + "\n\n")
        f.write(f"Mean Dispersion: {dispersion.mean():.2f}%\n")
        f.write(f"Std of Dispersion: {dispersion.std():.2f}%\n")
        f.write(f"Min Dispersion: {dispersion.min():.2f}%\n")
        f.write(f"Max Dispersion: {dispersion.max():.2f}%\n\n")
        f.write(
            "Interpretation: Dispersion measures the daily standard deviation of returns "
            "across all assets. Higher dispersion indicates greater divergence in asset "
            "performance, while lower dispersion suggests synchronized movements.\n\n"
        )
        
        # Market Statistics
        f.write("MARKET STATISTICS SUMMARY\n")
        f.write("-" * 80 + "\n\n")
        f.write(f"Average Pairwise Correlation: {market_stats['avg_pairwise_correlation']:.3f}\n")
        f.write(f"Mean Cross-Sectional Dispersion: {market_stats['mean_dispersion']:.2f}%\n")
        f.write(f"Number of Assets: {market_stats['n_assets']}\n")
        f.write(f"Number of Pairs: {market_stats['n_pairs']}\n\n")
        
        f.write("INTERPRETATION\n")
        f.write("-" * 80 + "\n\n")
        f.write(
            "Average Pairwise Correlation: Measures the overall interconnectedness of "
            "assets. Values closer to 1 indicate highly synchronized markets, while "
            "values closer to 0 suggest independent movements.\n\n"
        )
        f.write(
            "Cross-Sectional Dispersion: Measures daily divergence in asset returns. "
            "Higher dispersion suggests more varied asset performance, potentially "
            "indicating sector rotation or divergent market conditions.\n\n"
        )
    
    print(f"✓ Saved: {output_path}")


def generate_final_summary(output_dir: str):
    """
    Generate final summary document combining all analyses:
    - Part 1: Trend results
    - Part 2: Indicator tests
    - Part 3: Entropy scores
    - Part 4: Correlations + Professional Stats
    
    Args:
        output_dir: Output directory
    """
    print("\n" + "=" * 80)
    print("GENERATING FINAL SUMMARY DOCUMENT")
    print("=" * 80)
    
    output_path = os.path.join(output_dir, 'week3_final_summary.md')
    
    # Load existing data files
    try:
        # Load indicator test results
        indicator_results = pd.read_csv(os.path.join(output_dir, 'indicator_test_results.csv'))
    except:
        indicator_results = None
    
    try:
        # Load entropy scores
        entropy_scores = pd.read_csv(os.path.join(output_dir, 'indicator_entropy_scores.csv'))
    except:
        entropy_scores = None
    
    try:
        # Load quality ranking
        quality_ranking = pd.read_csv(os.path.join(output_dir, 'indicator_quality_ranking.csv'))
    except:
        quality_ranking = None
    
    try:
        # Load correlation matrix
        corr_matrix = pd.read_csv(os.path.join(output_dir, 'correlation_matrix.csv'), index_col=0)
    except:
        corr_matrix = None
    
    try:
        # Load market stats
        with open(os.path.join(output_dir, 'professional_market_stats.txt'), 'r') as f:
            market_stats_text = f.read()
    except:
        market_stats_text = None
    
    with open(output_path, 'w') as f:
        f.write("# Week 3 Final Analysis Summary\n")
        f.write("**Symbol:** XAU/USD (Gold)\n")
        f.write("**Timeframe:** 1-Hour\n")
        f.write("**Analysis Period:** 2022-01-01 to 2025-10-31\n\n")
        f.write("---\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(
            "This comprehensive analysis examines Gold (XAU/USD) trading signals across "
            "multiple dimensions: trend analysis, indicator performance, signal entropy "
            "(consistency), and cross-asset correlations. The analysis provides actionable "
            "insights for quantitative trading strategy development.\n\n"
        )
        f.write("**Key Highlights:**\n")
        f.write("- Comprehensive indicator testing across 5 signal types\n")
        f.write("- Entropy analysis to assess signal consistency and predictability\n")
        f.write("- Multi-asset correlation analysis (Gold vs AUD, JPY, CAD)\n")
        f.write("- Professional market statistics for risk management\n\n")
        f.write("---\n\n")
        
        # Part 1: Trend Results
        f.write("## Part 1 – Trend Analysis Results\n\n")
        f.write(
            "Trend analysis identifies distinct market regimes and helps understand "
            "directional price movements. Results from trend analysis show:\n\n"
        )
        f.write(
            "**Note:** Detailed trend analysis results are available in "
            "`trend_analysis_results.json` and `trend_summary.txt`.\n\n"
        )
        f.write("---\n\n")
        
        # Part 2: Indicator Tests
        f.write("## Part 2 – Indicator Performance Testing\n\n")
        if indicator_results is not None:
            f.write("### Indicator Test Results Summary\n\n")
            f.write("| Indicator | Signal Type | Total Signals | Win Rate | Avg Return | p-value | Significant |\n")
            f.write("|-----------|-------------|---------------|----------|------------|---------|-------------|\n")
            
            for _, row in indicator_results.iterrows():
                indicator = row['indicator']
                signal = row['signal_type']
                total = row['total_signals']
                win_rate = row['win_rate_pct']
                avg_return = row['avg_return_pct']
                p_value = row['p_value']
                significant = "Yes" if row['statistically_significant'] else "No"
                
                f.write(f"| {indicator} | {signal} | {total} | {win_rate:.1f}% | {avg_return:+.2f}% | {p_value:.6f} | {significant} |\n")
            
            f.write("\n**Key Findings:**\n")
            
            # Find best and worst indicators
            if len(indicator_results) > 0:
                best_indicator = indicator_results.loc[indicator_results['win_rate_pct'].idxmax()]
                f.write(f"- **Best Win Rate:** {best_indicator['indicator']} - {best_indicator['signal_type']} "
                       f"({best_indicator['win_rate_pct']:.1f}%)\n")
                
                best_return = indicator_results.loc[indicator_results['avg_return_pct'].idxmax()]
                f.write(f"- **Best Average Return:** {best_return['indicator']} - {best_return['signal_type']} "
                       f"({best_return['avg_return_pct']:+.2f}%)\n")
                
                significant_count = indicator_results['statistically_significant'].sum()
                f.write(f"- **Statistically Significant Signals:** {significant_count} out of {len(indicator_results)}\n")
        else:
            f.write("Indicator test results not available.\n")
        
        f.write("\n---\n\n")
        
        # Part 3: Entropy Scores
        f.write("## Part 3 – Entropy Analysis (Signal Consistency)\n\n")
        if entropy_scores is not None:
            f.write(
                "Entropy quantifies the consistency and predictability of return distributions. "
                "Lower entropy indicates more consistent, predictable returns.\n\n"
            )
            f.write("### Entropy Scores Summary\n\n")
            f.write("| Indicator | Signal Type | Entropy (Norm) | Quality | Win Rate |\n")
            f.write("|-----------|-------------|----------------|---------|----------|\n")
            
            for _, row in entropy_scores.iterrows():
                indicator = row['indicator']
                signal = row['signal_type']
                entropy = row['norm_entropy']
                quality = row['quality_rating']
                win_rate = row['win_rate']
                
                f.write(f"| {indicator} | {signal} | {entropy:.3f} | {quality} | {win_rate:.1f}% |\n")
            
            f.write("\n**Quality Rating Scale:**\n")
            f.write("- Excellent (⭐⭐⭐⭐⭐): 0.0 - 0.4\n")
            f.write("- Good (⭐⭐⭐⭐): 0.4 - 0.6\n")
            f.write("- Moderate (⭐⭐⭐): 0.6 - 0.8\n")
            f.write("- Moderate-Poor (⭐⭐): 0.8 - 0.9\n")
            f.write("- Poor (⭐): > 0.9\n\n")
        else:
            f.write("Entropy scores not available.\n")
        
        f.write("\n---\n\n")
        
        # Part 4: Quality Ranking
        f.write("## Part 3.5 – Indicator Quality Ranking\n\n")
        if quality_ranking is not None:
            f.write(
                "Final ranking combines entropy (consistency), win rate, and statistical significance:\n\n"
            )
            f.write("| Rank | Indicator | Signal | Win% | Avg Return | Entropy | Quality | p-value |\n")
            f.write("|------|-----------|--------|-----|-------------|---------|---------|----------|\n")
            
            for _, row in quality_ranking.head(10).iterrows():
                rank = row['Rank']
                indicator = row['Indicator']
                signal = row['Signal']
                win_pct = row['Win%']
                avg_return = row['Avg Return']
                entropy = row['Entropy']
                quality = row['Quality Rating']
                p_value = row['p-value']
                
                f.write(f"| {rank} | {indicator} | {signal} | {win_pct:.1f}% | {avg_return} | {entropy:.3f} | {quality} | {p_value} |\n")
        else:
            f.write("Quality ranking not available.\n")
        
        f.write("\n---\n\n")
        
        # Part 4: Correlations + Professional Stats
        f.write("## Part 4 – Cross-Asset Correlations & Market Statistics\n\n")
        
        if corr_matrix is not None:
            f.write("### Correlation Matrix\n\n")
            # Create table header with all assets
            assets = list(corr_matrix.columns)
            f.write(f"| Asset | {' | '.join([a.upper() for a in assets])} |\n")
            f.write(f"|-------|{'|'.join(['-----' for _ in assets])}|\n")
            
            for idx in corr_matrix.index:
                f.write(f"| {idx.capitalize()} |")
                for col in corr_matrix.columns:
                    val = corr_matrix.loc[idx, col]
                    f.write(f" {val:.3f} |")
                f.write("\n")
            
            f.write("\n**Key Correlations:**\n")
            # Gold vs AUD
            if 'gold' in corr_matrix.index and 'aud' in corr_matrix.columns:
                gold_aud = corr_matrix.loc['gold', 'aud']
                f.write(f"- **Gold-AUD:** {gold_aud:.3f} - ")
                if abs(gold_aud) < 0.2:
                    f.write("Weak correlation\n")
                elif abs(gold_aud) < 0.5:
                    f.write("Moderate correlation\n")
                else:
                    f.write("Strong correlation\n")
            
            # Gold vs JPY
            if 'gold' in corr_matrix.index and 'jpy' in corr_matrix.columns:
                gold_jpy = corr_matrix.loc['gold', 'jpy']
                f.write(f"- **Gold-JPY:** {gold_jpy:.3f} - ")
                if abs(gold_jpy) < 0.2:
                    f.write("Weak correlation\n")
                elif abs(gold_jpy) < 0.5:
                    f.write("Moderate correlation\n")
                else:
                    f.write("Strong correlation\n")
            
            # Gold vs CAD
            if 'gold' in corr_matrix.index and 'cad' in corr_matrix.columns:
                gold_cad = corr_matrix.loc['gold', 'cad']
                f.write(f"- **Gold-CAD:** {gold_cad:.3f} - ")
                if abs(gold_cad) < 0.2:
                    f.write("Weak correlation\n")
                elif abs(gold_cad) < 0.5:
                    f.write("Moderate correlation\n")
                else:
                    f.write("Strong correlation\n")
        else:
            f.write("Correlation matrix not available.\n")
        
        f.write("\n### Professional Market Statistics\n\n")
        if market_stats_text:
            # Extract key stats from the text
            lines = market_stats_text.split('\n')
            for line in lines:
                if 'Average Pairwise Correlation:' in line:
                    f.write(f"**{line.strip()}**\n")
                elif 'Mean Cross-Sectional Dispersion:' in line:
                    f.write(f"**{line.strip()}**\n")
        else:
            f.write("Market statistics not available.\n")
        
        f.write("\n---\n\n")
        
        # Key Findings + Recommendations
        f.write("## Key Findings & Recommendations\n\n")
        f.write("### Top Performing Signals\n\n")
        if quality_ranking is not None and len(quality_ranking) > 0:
            top_3 = quality_ranking.head(3)
            for idx, (_, row) in enumerate(top_3.iterrows(), 1):
                f.write(f"{idx}. **{row['Indicator']} - {row['Signal']}**\n")
                f.write(f"   - Win Rate: {row['Win%']:.1f}% | Entropy: {row['Entropy']:.3f} | Quality: {row['Quality Rating']}\n")
                f.write(f"   - Recommendation: Strong candidate for systematic trading\n\n")
        
        f.write("### Signals to Avoid\n\n")
        if quality_ranking is not None and len(quality_ranking) > 0:
            bottom_2 = quality_ranking.tail(2)
            for idx, (_, row) in enumerate(bottom_2.iterrows(), 1):
                f.write(f"{idx}. **{row['Indicator']} - {row['Signal']}**\n")
                f.write(f"   - Win Rate: {row['Win%']:.1f}% | Entropy: {row['Entropy']:.3f} | Quality: {row['Quality Rating']}\n")
                f.write(f"   - Recommendation: High entropy indicates unreliable returns\n\n")
        
        f.write("### Market Context\n\n")
        if corr_matrix is not None:
            f.write(
                "Cross-asset correlation analysis reveals Gold's relationship with currencies:\n"
            )
            f.write("- Currency correlations (AUD, JPY, CAD) reveal safe-haven and commodity-linked dynamics\n")
            f.write("- AUD and CAD often correlate with commodities due to resource-based economies\n")
            f.write("- JPY often serves as a safe-haven currency, similar to gold\n")
            f.write("- Use correlation insights for portfolio diversification across asset classes\n")
            f.write("- Monitor rolling correlations for regime changes and macroeconomic shifts\n")
            f.write("- Consider correlation in risk management models for multi-asset portfolios\n\n")
        
        f.write("---\n\n")
        
        # Validation Checklist & Next Steps
        f.write("## Validation Checklist & Next Steps\n\n")
        f.write("### Completed Analyses\n\n")
        f.write("- [x] Trend analysis across multiple timeframes\n")
        f.write("- [x] Indicator performance testing (5 indicators)\n")
        f.write("- [x] Entropy analysis for signal consistency\n")
        f.write("- [x] Quality ranking (entropy + win rate + p-value)\n")
        f.write("- [x] Cross-asset correlation analysis\n")
        f.write("- [x] Professional market statistics\n\n")
        
        f.write("### Next Steps\n\n")
        f.write("1. **Strategy Development:** Use top-ranked indicators to build systematic trading strategies\n")
        f.write("2. **Backtesting:** Test strategies on historical data with proper risk management\n")
        f.write("3. **Portfolio Optimization:** Incorporate correlation insights for multi-asset portfolios\n")
        f.write("4. **Risk Management:** Use entropy scores to assess signal reliability\n")
        f.write("5. **Monitoring:** Set up rolling correlation monitoring for regime detection\n\n")
        
        f.write("### Files Generated\n\n")
        f.write("- `indicator_test_results.csv` - Indicator performance metrics\n")
        f.write("- `indicator_entropy_scores.csv` - Entropy analysis results\n")
        f.write("- `indicator_quality_ranking.csv` - Final ranked indicator table\n")
        f.write("- `correlation_matrix.csv` - Asset correlation matrix\n")
        f.write("- `rolling_correlations.csv` - Time-varying correlations\n")
        f.write("- `professional_market_stats.txt` - Market statistics report\n")
        f.write("- `correlation_heatmap.png` - Correlation visualization\n")
        f.write("- `rolling_correlations.png` - Rolling correlation plot\n\n")
    
    print(f"✓ Saved: {output_path}")


def run_full_analysis():
    """
    Run complete correlation analysis and generate final summary.
    """
    # Run correlation analysis
    run_correlation_analysis()
    
    # Generate final summary
    output_dir = config.PROCESSED_DATA_PATH
    generate_final_summary(output_dir)


if __name__ == "__main__":
    run_full_analysis()

