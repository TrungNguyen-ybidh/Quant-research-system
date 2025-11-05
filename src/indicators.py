"""
Technical Indicators Module for AI-Driven Quantitative Trading Research System.

This module provides functions to calculate various technical indicators
and enrich OHLCV DataFrames with indicator values.

Indicators implemented:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- ATR (Average True Range)
- SMA (Simple Moving Averages)
- VWAP (Volume Weighted Average Price)
- ADX (Average Directional Index)
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def calculate_rsi(df: pd.DataFrame, period: int = None) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    RSI measures momentum by comparing the magnitude of recent gains to recent losses.
    Formula: RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss over the period
    
    Args:
        df: DataFrame with 'close' column
        period: RSI period (defaults to config.RSI_LENGTH)
        
    Returns:
        Series with RSI values
    """
    if period is None:
        period = config.RSI_LENGTH
    
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain 'close' column for RSI calculation")
    
    # Calculate price changes
    delta = df['close'].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss using exponential moving average
    # Using Wilder's smoothing method (same as TradingView/MetaTrader)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    # Calculate RS (Relative Strength)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(df: pd.DataFrame, fast: int = None, slow: int = None, signal: int = None) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    MACD = 12-period EMA - 26-period EMA
    Signal = 9-period EMA of MACD
    Histogram = MACD - Signal
    
    Args:
        df: DataFrame with 'close' column
        fast: Fast EMA period (defaults to config.MACD_FAST_SPAN)
        slow: Slow EMA period (defaults to config.MACD_SLOW_SPAN)
        signal: Signal EMA period (defaults to config.MACD_SIGNAL_SPAN)
        
    Returns:
        DataFrame with 'macd', 'macd_signal', and 'macd_histogram' columns
    """
    if fast is None:
        fast = config.MACD_FAST_SPAN
    if slow is None:
        slow = config.MACD_SLOW_SPAN
    if signal is None:
        signal = config.MACD_SIGNAL_SPAN
    
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain 'close' column for MACD calculation")
    
    # Calculate EMAs
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    
    # Calculate MACD line
    macd = ema_fast - ema_slow
    
    # Calculate signal line (EMA of MACD)
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    
    # Calculate histogram
    macd_histogram = macd - macd_signal
    
    # Create result DataFrame
    result = pd.DataFrame({
        'macd': macd,
        'macd_signal': macd_signal,
        'macd_histogram': macd_histogram
    })
    
    return result


def calculate_atr(df: pd.DataFrame, period: int = None) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    True Range = max(high - low, |high - prev_close|, |low - prev_close|)
    ATR = rolling mean of True Range over the period
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR period (defaults to config.ATR_PERIOD)
        
    Returns:
        Series with ATR values
    """
    if period is None:
        period = config.ATR_PERIOD
    
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain {required_cols} columns for ATR calculation")
    
    # Calculate True Range components
    high_low = df['high'] - df['low']
    high_close_prev = abs(df['high'] - df['close'].shift(1))
    low_close_prev = abs(df['low'] - df['close'].shift(1))
    
    # True Range is the maximum of the three
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # Calculate ATR as rolling mean of True Range
    # Using Wilder's smoothing method (same as TradingView/MetaTrader)
    atr = true_range.ewm(alpha=1/period, adjust=False).mean()
    
    return atr


def calculate_sma(df: pd.DataFrame, short: int = None, long: int = None, sma_200: bool = True) -> pd.DataFrame:
    """
    Calculate Simple Moving Averages (SMA).
    
    Calculates short and long period SMAs of the close price.
    Optionally calculates SMA-200 for trend analysis.
    
    Args:
        df: DataFrame with 'close' column
        short: Short SMA period (defaults to config.SMA_SHORT_LENGTH)
        long: Long SMA period (defaults to config.SMA_LONG_LENGTH)
        sma_200: Whether to calculate SMA-200 (default: True)
        
    Returns:
        DataFrame with 'sma_short', 'sma_long', and optionally 'sma_200' columns
    """
    if short is None:
        short = config.SMA_SHORT_LENGTH
    if long is None:
        long = config.SMA_LONG_LENGTH
    
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain 'close' column for SMA calculation")
    
    # Calculate SMAs
    sma_short = df['close'].rolling(window=short).mean()
    sma_long = df['close'].rolling(window=long).mean()
    
    # Create result DataFrame
    result = pd.DataFrame({
        'sma_short': sma_short,
        'sma_long': sma_long
    })
    
    # Add SMA-200 if requested
    if sma_200:
        result['sma_200'] = df['close'].rolling(window=200).mean()
    
    return result


def calculate_adx(df: pd.DataFrame, period: int = None) -> pd.Series:
    """
    Calculate Average Directional Index (ADX).
    
    ADX measures trend strength. Values above 25 indicate strong trends,
    while values below 25 suggest weak or sideways markets.
    
    ADX is calculated from:
    1. +DI (Plus Directional Indicator) and -DI (Minus Directional Indicator)
    2. DX = 100 * |+DI - -DI| / (+DI + -DI)
    3. ADX = smoothed average of DX
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ADX period (defaults to config.ADX_PERIOD)
        
    Returns:
        Series with ADX values
    """
    if period is None:
        period = config.ADX_PERIOD
    
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain {required_cols} columns for ADX calculation")
    
    # Calculate True Range (TR) - same as in ATR
    high_low = df['high'] - df['low']
    high_close_prev = abs(df['high'] - df['close'].shift(1))
    low_close_prev = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    # +DM = high - prev_high (if positive and greater than prev_low - low, else 0)
    # -DM = prev_low - low (if positive and greater than high - prev_high, else 0)
    high_diff = df['high'] - df['high'].shift(1)
    low_diff = df['low'].shift(1) - df['low']
    
    # +DM and -DM cannot both be positive - take the larger one
    plus_dm = pd.Series(index=df.index, dtype=float)
    minus_dm = pd.Series(index=df.index, dtype=float)
    
    for i in range(1, len(df)):
        if high_diff.iloc[i] > low_diff.iloc[i] and high_diff.iloc[i] > 0:
            plus_dm.iloc[i] = high_diff.iloc[i]
            minus_dm.iloc[i] = 0
        elif low_diff.iloc[i] > high_diff.iloc[i] and low_diff.iloc[i] > 0:
            plus_dm.iloc[i] = 0
            minus_dm.iloc[i] = low_diff.iloc[i]
        else:
            plus_dm.iloc[i] = 0
            minus_dm.iloc[i] = 0
    
    # Smooth TR, +DM, and -DM using Wilder's smoothing (same as RSI/ATR)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di_smooth = plus_dm.ewm(alpha=1/period, adjust=False).mean()
    minus_di_smooth = minus_dm.ewm(alpha=1/period, adjust=False).mean()
    
    # Calculate +DI and -DI as percentages
    plus_di = 100 * (plus_di_smooth / atr)
    minus_di = 100 * (minus_di_smooth / atr)
    
    # Calculate DX (Directional Index)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    
    # Calculate ADX as smoothed average of DX
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return adx


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP).
    
    For intraday data (minutes/hours), VWAP resets daily:
    VWAP = cumulative(price * volume) / cumulative(volume) per day
    
    Typical price = (high + low + close) / 3
    
    For daily timeframe, VWAP is just the typical price.
    
    Args:
        df: DataFrame with 'high', 'low', 'close', 'volume' columns and timestamp index
        
    Returns:
        Series with VWAP values
    """
    required_cols = ['high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain {required_cols} columns for VWAP calculation")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex for VWAP calculation")
    
    # Calculate typical price
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    # Check if this is daily data (only one observation per day)
    # Or intraday data (multiple observations per day)
    time_diff = df.index.to_series().diff()
    is_daily = (time_diff.min() >= pd.Timedelta(days=0.9))
    
    if is_daily:
        # For daily data, VWAP is just the typical price
        vwap = typical_price
    else:
        # For intraday data, calculate VWAP that resets daily
        # Group by date and calculate cumulative VWAP per day
        df_copy = df.copy()
        df_copy['typical_price'] = typical_price
        df_copy['price_volume'] = typical_price * df['volume']
        
        # Get date component for grouping (normalize to date)
        df_copy['date'] = df_copy.index.normalize().date
        
        # Calculate cumulative sums per day
        df_copy['cum_price_volume'] = df_copy.groupby('date')['price_volume'].cumsum()
        df_copy['cum_volume'] = df_copy.groupby('date')['volume'].cumsum()
        
        # Calculate VWAP (avoid division by zero)
        vwap = df_copy['cum_price_volume'] / df_copy['cum_volume'].replace(0, np.nan)
        
        # Ensure it's a Series with correct index
        vwap = pd.Series(vwap.values, index=df.index, name='vwap')
    
    return vwap


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to a DataFrame.
    
    This function calculates RSI, MACD, ATR, SMA, and VWAP and adds them
    as new columns to the input DataFrame.
    
    Args:
        df: DataFrame with OHLCV data (must have timestamp index)
        
    Returns:
        DataFrame with all indicators added as new columns
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Ensure timestamp is the index if it's a column
    if 'timestamp' in result_df.columns and not isinstance(result_df.index, pd.DatetimeIndex):
        result_df = result_df.set_index('timestamp')
    
    if not isinstance(result_df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex for indicator calculations")
    
    # Sort by timestamp to ensure correct calculations
    result_df = result_df.sort_index()
    
    # Calculate RSI
    try:
        result_df['rsi'] = calculate_rsi(result_df)
    except Exception as e:
        print(f"Warning: Could not calculate RSI: {str(e)}")
    
    # Calculate MACD
    try:
        macd_result = calculate_macd(result_df)
        result_df['macd'] = macd_result['macd']
        result_df['macd_signal'] = macd_result['macd_signal']
        result_df['macd_histogram'] = macd_result['macd_histogram']
    except Exception as e:
        print(f"Warning: Could not calculate MACD: {str(e)}")
    
    # Calculate ATR
    try:
        result_df['atr'] = calculate_atr(result_df)
    except Exception as e:
        print(f"Warning: Could not calculate ATR: {str(e)}")
    
    # Calculate SMA (including SMA-200)
    try:
        sma_result = calculate_sma(result_df, sma_200=True)
        result_df['sma_short'] = sma_result['sma_short']
        result_df['sma_long'] = sma_result['sma_long']
        result_df['sma_200'] = sma_result['sma_200']
    except Exception as e:
        print(f"Warning: Could not calculate SMA: {str(e)}")
    
    # Calculate ADX
    try:
        result_df['adx'] = calculate_adx(result_df)
    except Exception as e:
        print(f"Warning: Could not calculate ADX: {str(e)}")
    
    # Calculate VWAP
    try:
        result_df['vwap'] = calculate_vwap(result_df)
    except Exception as e:
        print(f"Warning: Could not calculate VWAP: {str(e)}")
    
    return result_df


# ============================================================================
# Main execution for testing
# ============================================================================

if __name__ == "__main__":
    """
    Test the indicators module by loading a sample file and calculating indicators.
    """
    import sys
    import os
    
    # Example usage
    sample_file = "data/raw/XAU_USD_1Day.csv"
    
    if os.path.exists(sample_file):
        print(f"Loading {sample_file}...")
        df = pd.read_csv(sample_file, parse_dates=['timestamp'], index_col='timestamp')
        
        print(f"Original columns: {df.columns.tolist()}")
        print(f"Data shape: {df.shape}")
        
        # Add all indicators
        enriched_df = add_all_indicators(df)
        
        print(f"\nEnriched columns: {enriched_df.columns.tolist()}")
        print(f"\nSample data with indicators:")
        print(enriched_df.tail(10))
        
        print("\n✓ Indicator calculations completed successfully!")
    else:
        print(f"Sample file {sample_file} not found. Please run data collection first.")

