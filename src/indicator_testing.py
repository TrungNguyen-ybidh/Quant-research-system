"""
Indicator Testing Module for AI-Driven Quantitative Trading Research System.

This module tests the predictive power of various technical indicators by:
1. Identifying signals based on indicator conditions
2. Calculating forward returns after signals
3. Computing statistical metrics (win rate, mean returns, p-values)
4. Generating detailed reports

Functions:
- test_rsi_signal: Tests RSI oversold (<30) and overbought (>70) signals
- test_macd_crossover: Tests MACD bullish and bearish crossover signals
- test_sma_bounce: Tests SMA-50 bounce signals in uptrends
- test_vwap_reversion: Tests VWAP mean reversion signals
- calculate_signal_statistics: Aggregates results and generates reports
"""

import pandas as pd
import numpy as np
import os
import sys
from scipy import stats
from typing import Tuple, Dict, Optional

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def calculate_trend_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate trend_type for each row based on SMA_50 and SMA_200.
    
    Rules:
    - up: close > SMA_50 and SMA_50 > SMA_200
    - down: close < SMA_50 and SMA_50 < SMA_200
    - range: otherwise
    
    Args:
        df: DataFrame with close, SMA_50, SMA_200 columns
        
    Returns:
        DataFrame with trend_type column added
    """
    df = df.copy()
    
    # Calculate SMA_50 and SMA_200 if not present
    if 'SMA_50' not in df.columns:
        df['SMA_50'] = df['close'].rolling(window=50).mean()
    if 'SMA_200' not in df.columns:
        df['SMA_200'] = df['close'].rolling(window=200).mean()
    
    # Calculate trend_type
    conditions = [
        (df['close'] > df['SMA_50']) & (df['SMA_50'] > df['SMA_200']),  # Uptrend
        (df['close'] < df['SMA_50']) & (df['SMA_50'] < df['SMA_200'])   # Downtrend
    ]
    choices = ['up', 'down']
    df['trend_type'] = np.select(conditions, choices, default='range')
    
    return df


def test_rsi_signal(df: pd.DataFrame, threshold: float, horizon: int = 6) -> pd.DataFrame:
    """
    Test RSI signals (oversold <30 or overbought >70).
    
    For oversold (RSI < 30): Expect positive returns (price rise)
    For overbought (RSI > 70): Expect negative returns (price fall)
    
    Args:
        df: DataFrame with timestamp, close, RSI, trend_type columns
        threshold: RSI threshold (30 for oversold, 70 for overbought)
        horizon: Number of hours forward to calculate return (default: 6)
        
    Returns:
        DataFrame with signal details:
        - indicator, signal_type, signal_timestamp, current_price,
          future_price, return_6h, was_profitable, trend_type
    """
    df = df.copy()
    
    # Ensure timestamp is index
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    
    # Sort by timestamp
    df = df.sort_index()
    
    # Ensure trend_type exists
    if 'trend_type' not in df.columns:
        df = calculate_trend_type(df)
    
    # Determine signal type
    if threshold < 50:
        signal_type = 'RSI_Oversold'
        condition = df['rsi'] < threshold
        expect_positive = True
    else:
        signal_type = 'RSI_Overbought'
        condition = df['rsi'] > threshold
        expect_positive = False
    
    # Find signals
    signal_rows = df[condition].copy()
    
    if signal_rows.empty:
        return pd.DataFrame(columns=[
            'indicator', 'signal_type', 'signal_timestamp', 'current_price',
            'future_price', 'return_6h', 'was_profitable', 'trend_type'
        ])
    
    # Calculate future price (horizon hours ahead)
    signal_rows['future_price'] = signal_rows['close'].shift(-horizon)
    
    # Drop signals without future price (near end of data)
    signal_rows = signal_rows.dropna(subset=['future_price'])
    
    # Calculate return
    signal_rows['return_6h'] = ((signal_rows['future_price'] - signal_rows['close']) / signal_rows['close']) * 100
    
    # Determine if profitable
    if expect_positive:
        signal_rows['was_profitable'] = signal_rows['return_6h'] > 0
    else:
        signal_rows['was_profitable'] = signal_rows['return_6h'] < 0
    
    # Create result DataFrame
    results = pd.DataFrame({
        'indicator': 'RSI',
        'signal_type': signal_type,
        'signal_timestamp': signal_rows.index,
        'current_price': signal_rows['close'],
        'future_price': signal_rows['future_price'],
        'return_6h': signal_rows['return_6h'],
        'was_profitable': signal_rows['was_profitable'],
        'trend_type': signal_rows['trend_type']
    })
    
    return results.reset_index(drop=True)


def test_macd_crossover(df: pd.DataFrame, signal_type: str, horizon: int = 6) -> pd.DataFrame:
    """
    Test MACD crossover signals (bullish or bearish).
    
    Bullish crossover: MACD crosses above MACD_signal
    Bearish crossover: MACD crosses below MACD_signal
    
    Args:
        df: DataFrame with timestamp, close, macd, macd_signal columns
        signal_type: 'bullish' or 'bearish'
        horizon: Number of hours forward to calculate return (default: 6)
        
    Returns:
        DataFrame with signal details
    """
    df = df.copy()
    
    # Ensure timestamp is index
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    
    # Sort by timestamp
    df = df.sort_index()
    
    # Ensure trend_type exists
    if 'trend_type' not in df.columns:
        df = calculate_trend_type(df)
    
    # Calculate previous values
    df['macd_prev'] = df['macd'].shift(1)
    df['macd_signal_prev'] = df['macd_signal'].shift(1)
    
    # Find crossovers
    if signal_type.lower() == 'bullish':
        # MACD > MACD_signal and previous MACD <= MACD_signal
        condition = (df['macd'] > df['macd_signal']) & (df['macd_prev'] <= df['macd_signal_prev'])
        signal_name = 'MACD_Bullish_Cross'
        expect_positive = True
    elif signal_type.lower() == 'bearish':
        # MACD < MACD_signal and previous MACD >= MACD_signal
        condition = (df['macd'] < df['macd_signal']) & (df['macd_prev'] >= df['macd_signal_prev'])
        signal_name = 'MACD_Bearish_Cross'
        expect_positive = False
    else:
        raise ValueError("signal_type must be 'bullish' or 'bearish'")
    
    # Find signals
    signal_rows = df[condition].copy()
    
    if signal_rows.empty:
        return pd.DataFrame(columns=[
            'indicator', 'signal_type', 'signal_timestamp', 'current_price',
            'future_price', 'return_6h', 'was_profitable', 'trend_type'
        ])
    
    # Calculate future price
    signal_rows['future_price'] = signal_rows['close'].shift(-horizon)
    
    # Drop signals without future price
    signal_rows = signal_rows.dropna(subset=['future_price'])
    
    # Calculate return
    signal_rows['return_6h'] = ((signal_rows['future_price'] - signal_rows['close']) / signal_rows['close']) * 100
    
    # Determine if profitable
    signal_rows['was_profitable'] = signal_rows['return_6h'] > 0 if expect_positive else signal_rows['return_6h'] < 0
    
    # Create result DataFrame
    results = pd.DataFrame({
        'indicator': 'MACD',
        'signal_type': signal_name,
        'signal_timestamp': signal_rows.index,
        'current_price': signal_rows['close'],
        'future_price': signal_rows['future_price'],
        'return_6h': signal_rows['return_6h'],
        'was_profitable': signal_rows['was_profitable'],
        'trend_type': signal_rows['trend_type']
    })
    
    return results.reset_index(drop=True)


def test_sma_bounce(df: pd.DataFrame, ma_period: int = 50, horizon: int = 6) -> pd.DataFrame:
    """
    Test SMA bounce signals (price touches SMA-50 in uptrend).
    
    Signal: abs(close - SMA_50) / close * 100 < 0.5 (price "touches" SMA-50)
    Filter: Only rows where trend_type == "up"
    
    Args:
        df: DataFrame with timestamp, close, SMA_50, trend_type columns
        ma_period: Moving average period (default: 50)
        horizon: Number of hours forward to calculate return (default: 6)
        
    Returns:
        DataFrame with signal details
    """
    df = df.copy()
    
    # Ensure timestamp is index
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    
    # Sort by timestamp
    df = df.sort_index()
    
    # Calculate SMA_50 if not present
    if 'SMA_50' not in df.columns:
        df['SMA_50'] = df['close'].rolling(window=ma_period).mean()
    
    # Ensure trend_type exists
    if 'trend_type' not in df.columns:
        df = calculate_trend_type(df)
    
    # Calculate distance from SMA
    df['distance_pct'] = abs(df['close'] - df['SMA_50']) / df['close'] * 100
    
    # Signal: price touches SMA-50 (within 0.5%) AND in uptrend
    condition = (df['distance_pct'] < 0.5) & (df['trend_type'] == 'up')
    
    # Find signals
    signal_rows = df[condition].copy()
    
    if signal_rows.empty:
        return pd.DataFrame(columns=[
            'indicator', 'signal_type', 'signal_timestamp', 'current_price',
            'future_price', 'return_6h', 'was_profitable', 'trend_type'
        ])
    
    # Calculate future price
    signal_rows['future_price'] = signal_rows['close'].shift(-horizon)
    
    # Drop signals without future price
    signal_rows = signal_rows.dropna(subset=['future_price'])
    
    # Calculate return (expect positive in uptrend)
    signal_rows['return_6h'] = ((signal_rows['future_price'] - signal_rows['close']) / signal_rows['close']) * 100
    
    # Determine if profitable (expect positive return)
    signal_rows['was_profitable'] = signal_rows['return_6h'] > 0
    
    # Create result DataFrame
    results = pd.DataFrame({
        'indicator': 'SMA',
        'signal_type': f'SMA_{ma_period}_Bounce',
        'signal_timestamp': signal_rows.index,
        'current_price': signal_rows['close'],
        'future_price': signal_rows['future_price'],
        'return_6h': signal_rows['return_6h'],
        'was_profitable': signal_rows['was_profitable'],
        'trend_type': signal_rows['trend_type']
    })
    
    return results.reset_index(drop=True)


def test_vwap_reversion(df: pd.DataFrame, threshold: float = 2.0, horizon: int = 6) -> pd.DataFrame:
    """
    Test VWAP mean reversion signals.
    
    Signal: (close - VWAP) / VWAP * 100 > threshold (price too high)
    Success: distance_6h < distance_now (price moved closer to VWAP)
    
    Args:
        df: DataFrame with timestamp, close, vwap columns
        threshold: Distance threshold percentage (default: 2.0)
        horizon: Number of hours forward to calculate return (default: 6)
        
    Returns:
        DataFrame with signal details
    """
    df = df.copy()
    
    # Ensure timestamp is index
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    
    # Sort by timestamp
    df = df.sort_index()
    
    # Ensure trend_type exists
    if 'trend_type' not in df.columns:
        df = calculate_trend_type(df)
    
    # Calculate distance from VWAP
    df['distance_pct'] = ((df['close'] - df['vwap']) / df['vwap']) * 100
    
    # Signal: price is more than threshold% above VWAP
    condition = df['distance_pct'] > threshold
    
    # Find signals
    signal_rows = df[condition].copy()
    
    if signal_rows.empty:
        return pd.DataFrame(columns=[
            'indicator', 'signal_type', 'signal_timestamp', 'current_price',
            'future_price', 'return_6h', 'was_profitable', 'trend_type'
        ])
    
    # Calculate future price and future VWAP
    signal_rows['future_price'] = signal_rows['close'].shift(-horizon)
    signal_rows['future_vwap'] = signal_rows['vwap'].shift(-horizon)
    
    # Drop signals without future data
    signal_rows = signal_rows.dropna(subset=['future_price', 'future_vwap'])
    
    # Calculate future distance
    signal_rows['distance_6h'] = ((signal_rows['future_price'] - signal_rows['future_vwap']) / signal_rows['future_vwap']) * 100
    
    # Success: price moved closer to VWAP (mean reversion)
    signal_rows['was_profitable'] = abs(signal_rows['distance_6h']) < abs(signal_rows['distance_pct'])
    
    # Calculate return (expect negative if price was above VWAP)
    signal_rows['return_6h'] = ((signal_rows['future_price'] - signal_rows['close']) / signal_rows['close']) * 100
    
    # Create result DataFrame
    results = pd.DataFrame({
        'indicator': 'VWAP',
        'signal_type': f'VWAP_Reversion_{threshold}',
        'signal_timestamp': signal_rows.index,
        'current_price': signal_rows['close'],
        'future_price': signal_rows['future_price'],
        'return_6h': signal_rows['return_6h'],
        'was_profitable': signal_rows['was_profitable'],
        'trend_type': signal_rows['trend_type']
    })
    
    return results.reset_index(drop=True)


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio for signal returns.
    
    Sharpe Ratio = (Mean Return - Risk Free Rate) / Standard Deviation of Returns
    
    Args:
        returns: Series or array of percentage returns
        risk_free_rate: Risk-free rate percentage (default: 0.0 for simplicity)
        
    Returns:
        float: Sharpe ratio (0.0 if std dev is 0 or no returns)
    """
    if len(returns) == 0:
        return 0.0
    
    returns_array = np.array(returns.dropna())
    
    if len(returns_array) == 0:
        return 0.0
    
    mean_return = np.mean(returns_array)
    std_return = np.std(returns_array, ddof=1)  # Sample standard deviation
    
    if std_return == 0:
        return 0.0
    
    sharpe = (mean_return - risk_free_rate) / std_return
    return sharpe


def calculate_profit_factor(returns: pd.Series) -> float:
    """
    Calculate profit factor (sum of wins / sum of losses).
    
    Profit Factor = Sum of all winning trades / Sum of all losing trades
    
    Args:
        returns: Series or array of percentage returns
        
    Returns:
        float: Profit factor (0.0 if no losses, inf if no wins but losses exist)
    """
    if len(returns) == 0:
        return 0.0
    
    returns_array = np.array(returns.dropna())
    
    if len(returns_array) == 0:
        return 0.0
    
    wins = returns_array[returns_array > 0]
    losses = returns_array[returns_array < 0]
    
    sum_wins = np.sum(wins) if len(wins) > 0 else 0.0
    sum_losses = abs(np.sum(losses)) if len(losses) > 0 else 0.0
    
    if sum_losses == 0:
        if sum_wins > 0:
            return float('inf')  # Perfect strategy (no losses)
        else:
            return 0.0  # No wins, no losses
    
    profit_factor = sum_wins / sum_losses
    return profit_factor


def calculate_risk_reward_ratio(returns: pd.Series) -> float:
    """
    Calculate risk-reward ratio (average win / average loss).
    
    Risk-Reward Ratio = Average Winning Trade / Average Losing Trade
    
    Args:
        returns: Series or array of percentage returns
        
    Returns:
        float: Risk-reward ratio (0.0 if no losses, inf if no wins but losses exist)
    """
    if len(returns) == 0:
        return 0.0
    
    returns_array = np.array(returns.dropna())
    
    if len(returns_array) == 0:
        return 0.0
    
    wins = returns_array[returns_array > 0]
    losses = returns_array[returns_array < 0]
    
    avg_win = np.mean(wins) if len(wins) > 0 else 0.0
    avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0.0
    
    if avg_loss == 0:
        if avg_win > 0:
            return float('inf')  # Perfect strategy (no losses)
        else:
            return 0.0  # No wins, no losses
    
    risk_reward = avg_win / avg_loss
    return risk_reward


def calculate_maximum_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown from equity curve.
    
    For chronological sequence of signal returns, builds equity curve and finds
    peak-to-trough drawdown.
    
    Args:
        returns: Series or array of percentage returns (in chronological order!)
        
    Returns:
        float: Maximum drawdown percentage (positive value, e.g., 5.2 means 5.2% drawdown)
    """
    if len(returns) == 0:
        return 0.0
    
    returns_array = np.array(returns.dropna())
    
    if len(returns_array) == 0:
        return 0.0
    
    # Build equity curve starting from 100
    equity_curve = [100.0]
    for ret in returns_array:
        # Convert percentage return to multiplier (e.g., +1.2% -> 1.012)
        multiplier = 1.0 + (ret / 100.0)
        equity_curve.append(equity_curve[-1] * multiplier)
    
    equity_curve = np.array(equity_curve)
    
    # Find maximum drawdown
    # For each point, find the peak before it
    peak = equity_curve[0]
    max_drawdown = 0.0
    
    for value in equity_curve:
        if value > peak:
            peak = value
        else:
            # Calculate drawdown from peak
            drawdown = ((peak - value) / peak) * 100.0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
    
    return max_drawdown


def calculate_signal_statistics(signals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate aggregated statistics for each signal type.
    
    Computes:
    - total_signals: Number of signals
    - win_rate_pct: Percentage of profitable signals
    - avg_return_pct: Mean return percentage
    - median_return_pct: Median return percentage
    - std_return_pct: Standard deviation of returns
    - min_return_pct: Minimum return
    - max_return_pct: Maximum return
    - sharpe_ratio: Risk-adjusted return metric
    - profit_factor: Sum of wins / sum of losses
    - risk_reward_ratio: Average win / average loss
    - max_drawdown_pct: Maximum drawdown percentage
    - p_value: Binomial test p-value vs 50% random chance
    - statistically_significant: True if p < 0.05
    
    Args:
        signals_df: DataFrame with signal details from test functions
        
    Returns:
        DataFrame with aggregated statistics per signal type
    """
    if signals_df.empty:
        return pd.DataFrame(columns=[
            'indicator', 'signal_type', 'total_signals', 'win_rate_pct',
            'avg_return_pct', 'median_return_pct', 'std_return_pct',
            'min_return_pct', 'max_return_pct', 'sharpe_ratio', 'profit_factor',
            'risk_reward_ratio', 'max_drawdown_pct', 'p_value', 'statistically_significant'
        ])
    
    # Group by indicator and signal_type
    results = []
    
    for (indicator, signal_type), group in signals_df.groupby(['indicator', 'signal_type']):
        total_signals = len(group)
        wins = group['was_profitable'].sum()
        win_rate_pct = (wins / total_signals) * 100 if total_signals > 0 else 0
        
        # Return statistics
        returns = group['return_6h'].dropna()
        
        # Ensure returns are sorted chronologically for maximum drawdown calculation
        # Sort by timestamp if available, otherwise use index order
        if 'signal_timestamp' in group.columns:
            # Sort by timestamp to ensure chronological order
            sorted_group = group.sort_values('signal_timestamp')
            returns_sorted = sorted_group['return_6h'].dropna()
        else:
            # Use original order (should already be chronological)
            returns_sorted = returns
        
        avg_return_pct = returns.mean() if len(returns) > 0 else 0
        median_return_pct = returns.median() if len(returns) > 0 else 0
        std_return_pct = returns.std() if len(returns) > 0 else 0
        min_return_pct = returns.min() if len(returns) > 0 else 0
        max_return_pct = returns.max() if len(returns) > 0 else 0
        
        # Risk-adjusted metrics
        sharpe_ratio = calculate_sharpe_ratio(returns)
        profit_factor = calculate_profit_factor(returns)
        risk_reward_ratio = calculate_risk_reward_ratio(returns)
        max_drawdown_pct = calculate_maximum_drawdown(returns_sorted)  # Use sorted returns for drawdown
        
        # Binomial test: test if win rate is significantly different from 50%
        # H0: win rate = 50% (random chance)
        # H1: win rate != 50%
        if total_signals > 0:
            # Use binomtest (newer API) or binom_test (older API)
            try:
                # Try newer API first
                result = stats.binomtest(wins, total_signals, p=0.5, alternative='two-sided')
                p_value = result.pvalue
            except AttributeError:
                # Fall back to older API
                p_value = stats.binom_test(wins, total_signals, p=0.5, alternative='two-sided')
        else:
            p_value = 1.0
        
        statistically_significant = p_value < 0.05
        
        # Handle infinite values for profit_factor and risk_reward_ratio
        profit_factor_val = profit_factor if not np.isinf(profit_factor) else 999.99
        risk_reward_val = risk_reward_ratio if not np.isinf(risk_reward_ratio) else 999.99
        
        results.append({
            'indicator': indicator,
            'signal_type': signal_type,
            'total_signals': total_signals,
            'win_rate_pct': round(win_rate_pct, 2),
            'avg_return_pct': round(avg_return_pct, 3),
            'median_return_pct': round(median_return_pct, 3),
            'std_return_pct': round(std_return_pct, 3),
            'min_return_pct': round(min_return_pct, 3),
            'max_return_pct': round(max_return_pct, 3),
            'sharpe_ratio': round(sharpe_ratio, 3),
            'profit_factor': round(profit_factor_val, 2),
            'risk_reward_ratio': round(risk_reward_val, 2),
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'p_value': round(p_value, 6),
            'statistically_significant': statistically_significant
        })
    
    return pd.DataFrame(results)


def generate_report(signals_df: pd.DataFrame, stats_df: pd.DataFrame, output_path: str):
    """
    Generate a detailed text report of indicator test results.
    
    Args:
        signals_df: DataFrame with all signal details
        stats_df: DataFrame with aggregated statistics
        output_path: Path to save the report file
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("INDICATOR TESTING REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total Signals Analyzed: {len(signals_df)}")
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("SUMMARY BY SIGNAL TYPE")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Summary table
    for _, row in stats_df.iterrows():
        indicator = row['indicator']
        signal_type = row['signal_type']
        win_rate = row['win_rate_pct']
        p_value = row['p_value']
        significant = "Significant" if row['statistically_significant'] else "Not Significant"
        avg_return = row['avg_return_pct']
        sharpe = row.get('sharpe_ratio', 0.0)
        profit_factor = row.get('profit_factor', 0.0)
        risk_reward = row.get('risk_reward_ratio', 0.0)
        max_dd = row.get('max_drawdown_pct', 0.0)
        total = row['total_signals']
        
        report_lines.append(f"{indicator} {signal_type}")
        report_lines.append(f"  Total Signals: {total}")
        report_lines.append(f"  Win Rate: {win_rate:.2f}%")
        report_lines.append(f"  Avg Return: {avg_return:.3f}%")
        report_lines.append(f"  Sharpe Ratio: {sharpe:.3f}")
        report_lines.append(f"  Profit Factor: {profit_factor:.2f}")
        report_lines.append(f"  Risk-Reward Ratio: {risk_reward:.2f}")
        report_lines.append(f"  Max Drawdown: {max_dd:.2f}%")
        report_lines.append(f"  p-value: {p_value:.6f}")
        report_lines.append(f"  Status: {significant}")
        report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("DETAILED ANALYSIS BY INDICATOR")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Detailed analysis for each signal type
    for (indicator, signal_type), group in signals_df.groupby(['indicator', 'signal_type']):
        stats_row = stats_df[(stats_df['indicator'] == indicator) & (stats_df['signal_type'] == signal_type)].iloc[0]
        
        report_lines.append(f"--- {indicator}: {signal_type} ---")
        report_lines.append("")
        
        # Overall statistics
        report_lines.append(f"Overall Performance:")
        report_lines.append(f"  Total Signals: {stats_row['total_signals']}")
        report_lines.append(f"  Win Rate: {stats_row['win_rate_pct']:.2f}%")
        report_lines.append(f"  Mean Return: {stats_row['avg_return_pct']:.3f}%")
        report_lines.append(f"  Median Return: {stats_row['median_return_pct']:.3f}%")
        report_lines.append(f"  Std Dev: {stats_row['std_return_pct']:.3f}%")
        report_lines.append(f"  Min Return: {stats_row['min_return_pct']:.3f}%")
        report_lines.append(f"  Max Return: {stats_row['max_return_pct']:.3f}%")
        report_lines.append(f"  p-value (vs 50%): {stats_row['p_value']:.6f}")
        report_lines.append(f"  Statistically Significant: {stats_row['statistically_significant']}")
        report_lines.append("")
        
        # Risk-adjusted metrics
        sharpe = stats_row.get('sharpe_ratio', 0.0)
        profit_factor = stats_row.get('profit_factor', 0.0)
        risk_reward = stats_row.get('risk_reward_ratio', 0.0)
        max_dd = stats_row.get('max_drawdown_pct', 0.0)
        
        report_lines.append(f"Risk-Adjusted Metrics:")
        report_lines.append(f"  Sharpe Ratio: {sharpe:.3f}")
        if sharpe > 1.0:
            report_lines.append(f"    → Excellent risk-adjusted returns (>1.0)")
        elif sharpe > 0.5:
            report_lines.append(f"    → Good risk-adjusted returns (0.5-1.0)")
        else:
            report_lines.append(f"    → Poor risk-adjusted returns (<0.5)")
        
        report_lines.append(f"  Profit Factor: {profit_factor:.2f}")
        if profit_factor > 2.0:
            report_lines.append(f"    → Excellent (wins are 2x larger than losses)")
        elif profit_factor > 1.5:
            report_lines.append(f"    → Good (1.5-2.0)")
        elif profit_factor > 1.0:
            report_lines.append(f"    → Marginal (1.0-1.5)")
        else:
            report_lines.append(f"    → Losing strategy (<1.0)")
        
        report_lines.append(f"  Risk-Reward Ratio: {risk_reward:.2f}")
        if risk_reward > 2.0:
            report_lines.append(f"    → Excellent (wins are 2x bigger than losses)")
        elif risk_reward > 1.5:
            report_lines.append(f"    → Good (1.5-2.0)")
        elif risk_reward > 1.0:
            report_lines.append(f"    → Moderate (1.0-1.5)")
        else:
            report_lines.append(f"    → Poor (losses bigger than wins)")
        
        report_lines.append(f"  Maximum Drawdown: {max_dd:.2f}%")
        if max_dd < 5.0:
            report_lines.append(f"    → Excellent (small losing streaks)")
        elif max_dd < 10.0:
            report_lines.append(f"    → Good (5-10%)")
        elif max_dd < 20.0:
            report_lines.append(f"    → Moderate (10-20%)")
        else:
            report_lines.append(f"    → High (large losing streaks, >20%)")
        report_lines.append("")
        
        # Performance by trend type
        if 'trend_type' in group.columns:
            report_lines.append("Performance by Market Regime:")
            for trend_type in ['up', 'down', 'range']:
                trend_group = group[group['trend_type'] == trend_type]
                if len(trend_group) > 0:
                    trend_wins = trend_group['was_profitable'].sum()
                    trend_win_rate = (trend_wins / len(trend_group)) * 100
                    trend_avg_return = trend_group['return_6h'].mean()
                    report_lines.append(f"  {trend_type.upper()}: {len(trend_group)} signals, "
                                      f"{trend_win_rate:.2f}% win rate, "
                                      f"{trend_avg_return:.3f}% avg return")
            report_lines.append("")
        
        # Interpretation
        report_lines.append("Interpretation:")
        if stats_row['statistically_significant']:
            if stats_row['win_rate_pct'] > 50:
                report_lines.append(f"  This signal shows a statistically significant edge ({stats_row['win_rate_pct']:.2f}% win rate).")
                
                # Economic significance check
                sharpe = stats_row.get('sharpe_ratio', 0.0)
                profit_factor = stats_row.get('profit_factor', 0.0)
                if sharpe > 0.5 and profit_factor > 1.5:
                    report_lines.append(f"  ✓ Economically significant: Sharpe {sharpe:.3f} > 0.5 AND Profit Factor {profit_factor:.2f} > 1.5")
                    report_lines.append(f"  → Signal is both statistically AND economically significant - suitable for trading.")
                elif sharpe > 0.5 or profit_factor > 1.5:
                    report_lines.append(f"  ⚠️  Partially economically significant: Sharpe {sharpe:.3f}, Profit Factor {profit_factor:.2f}")
                    report_lines.append(f"  → Signal is statistically significant but may need regime filter or risk management.")
                else:
                    report_lines.append(f"  ⚠️  Statistically significant but economically marginal: Sharpe {sharpe:.3f}, Profit Factor {profit_factor:.2f}")
                    report_lines.append(f"  → Consider using with regime filter or avoiding without proper risk management.")
            else:
                report_lines.append(f"  This signal shows a statistically significant negative edge ({stats_row['win_rate_pct']:.2f}% win rate).")
                report_lines.append(f"  Consider inverting the signal or avoiding it.")
        else:
            report_lines.append(f"  This signal does not show statistical significance (p={stats_row['p_value']:.6f}).")
            report_lines.append(f"  The win rate of {stats_row['win_rate_pct']:.2f}% is not significantly different from random chance (50%).")
        
        report_lines.append("")
        report_lines.append("")
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Report saved to: {output_path}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    """
    Run all indicator tests and generate output files.
    """
    print("=" * 80)
    print("INDICATOR TESTING SYSTEM")
    print("=" * 80)
    print("")
    
    # Load data
    data_path = os.path.join(config.PROCESSED_DATA_PATH, "XAU_USD_1Hour_with_indicators.csv")
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    print("")
    
    # Ensure required columns exist
    required_cols = ['close', 'rsi', 'macd', 'macd_signal', 'vwap']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        print("Some tests may not run correctly.")
    
    # Calculate SMA_50 and SMA_200 if needed (for trend_type)
    if 'SMA_50' not in df.columns:
        df['SMA_50'] = df['close'].rolling(window=50).mean()
    if 'SMA_200' not in df.columns:
        df['SMA_200'] = df['close'].rolling(window=200).mean()
    
    # Calculate trend_type
    print("Calculating trend_type...")
    df = calculate_trend_type(df)
    print("")
    
    # Run all tests
    print("Running indicator tests...")
    print("")
    
    all_signals = []
    
    # Test 1: RSI Oversold (< 30)
    print("Test 1: RSI Oversold (< 30)...")
    rsi_oversold = test_rsi_signal(df, threshold=30, horizon=6)
    if not rsi_oversold.empty:
        all_signals.append(rsi_oversold)
        print(f"  Found {len(rsi_oversold)} signals")
    else:
        print("  No signals found")
    print("")
    
    # Test 2: RSI Overbought (> 70)
    print("Test 2: RSI Overbought (> 70)...")
    rsi_overbought = test_rsi_signal(df, threshold=70, horizon=6)
    if not rsi_overbought.empty:
        all_signals.append(rsi_overbought)
        print(f"  Found {len(rsi_overbought)} signals")
    else:
        print("  No signals found")
    print("")
    
    # Test 3: MACD Bullish Crossover
    print("Test 3: MACD Bullish Crossover...")
    macd_bullish = test_macd_crossover(df, signal_type='bullish', horizon=6)
    if not macd_bullish.empty:
        all_signals.append(macd_bullish)
        print(f"  Found {len(macd_bullish)} signals")
    else:
        print("  No signals found")
    print("")
    
    # Test 4: MACD Bearish Crossover
    print("Test 4: MACD Bearish Crossover...")
    macd_bearish = test_macd_crossover(df, signal_type='bearish', horizon=6)
    if not macd_bearish.empty:
        all_signals.append(macd_bearish)
        print(f"  Found {len(macd_bearish)} signals")
    else:
        print("  No signals found")
    print("")
    
    # Test 5: SMA-50 Bounce
    print("Test 5: SMA-50 Bounce (in uptrends)...")
    sma_bounce = test_sma_bounce(df, ma_period=50, horizon=6)
    if not sma_bounce.empty:
        all_signals.append(sma_bounce)
        print(f"  Found {len(sma_bounce)} signals")
    else:
        print("  No signals found")
    print("")
    
    # Test 6: VWAP Mean Reversion
    print("Test 6: VWAP Mean Reversion (> 2% above VWAP)...")
    vwap_reversion = test_vwap_reversion(df, threshold=2.0, horizon=6)
    if not vwap_reversion.empty:
        all_signals.append(vwap_reversion)
        print(f"  Found {len(vwap_reversion)} signals")
    else:
        print("  No signals found")
    print("")
    
    # Combine all signals
    if all_signals:
        signals_df = pd.concat(all_signals, ignore_index=True)
    else:
        print("Warning: No signals found across all tests!")
        signals_df = pd.DataFrame(columns=[
            'indicator', 'signal_type', 'signal_timestamp', 'current_price',
            'future_price', 'return_6h', 'was_profitable', 'trend_type'
        ])
    
    # Calculate statistics
    print("Calculating statistics...")
    stats_df = calculate_signal_statistics(signals_df)
    print("")
    
    # Save output files
    output_dir = config.PROCESSED_DATA_PATH
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. indicator_signal_details.csv
    details_path = os.path.join(output_dir, "indicator_signal_details.csv")
    signals_df.to_csv(details_path, index=False)
    print(f"Saved signal details to: {details_path}")
    print(f"  Total signals: {len(signals_df)}")
    print("")
    
    # 2. indicator_test_results.csv
    results_path = os.path.join(output_dir, "indicator_test_results.csv")
    stats_df.to_csv(results_path, index=False)
    print(f"Saved test results to: {results_path}")
    print(f"  Total test types: {len(stats_df)}")
    print("")
    
    # 3. indicator_test_report.txt
    report_path = os.path.join(output_dir, "indicator_test_report.txt")
    generate_report(signals_df, stats_df, report_path)
    print("")
    
    print("=" * 80)
    print("INDICATOR TESTING COMPLETE")
    print("=" * 80)
    print("")
    print("Output files:")
    print(f"  1. {details_path}")
    print(f"  2. {results_path}")
    print(f"  3. {report_path}")
    print("")

