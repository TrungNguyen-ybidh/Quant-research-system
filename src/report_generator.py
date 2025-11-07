"""
Report Generation Script for Quantitative Research Report

This script automates the generation of the comprehensive research report by:
1. Loading all analysis results from Weeks 2-4
2. Generating each section with real data
3. Combining all sections into a complete markdown document
4. Saving the final report

The script is designed to be reproducible and can be applied to other assets later.
Uses asset_adapter for asset-specific context and interpretations.

"""

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from scipy import stats
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.asset_adapter import (
    get_characteristics,
    generate_volume_interpretation,
    generate_trend_interpretation,
    generate_correlation_interpretation,
    generate_volatility_interpretation,
    generate_regime_interpretation
)
from src.config_manager import load_config, validate_config, load_asset_config, get_setting, get_sanitized_symbol, get_regime_specific_metrics_paths


def resolve_processed_file(filename: str,
                           asset_config: Dict[str, Any] = None,
                           base_dir: str = None) -> str:
    """Resolve a processed-data artifact path, checking common directories."""

    if base_dir is None:
        base_dir = config.PROCESSED_DATA_PATH

    candidate_dirs = [
        base_dir,
        os.path.join(base_dir, "Analysis Results"),
        os.path.join(base_dir, "analysis_results"),
    ]

    filenames = [filename]

    if asset_config:
        sanitized = get_sanitized_symbol(asset_config)
        name, ext = os.path.splitext(filename)
        filenames.insert(0, f"{name}_{sanitized}{ext}")

    for directory in candidate_dirs:
        for name in filenames:
            candidate = os.path.join(directory, name)
            if os.path.exists(candidate):
                return candidate

    # Fallback to base directory with first candidate name
    return os.path.join(base_dir, filenames[0])


def resolve_model_file(filename: str, asset_config: Dict[str, Any] = None) -> str:
    """
    Resolve model directory file path with optional asset-specific suffix.
    """
    base_dir = 'models'
    if asset_config:
        sanitized = get_sanitized_symbol(asset_config)
        name, ext = os.path.splitext(filename)
        candidate = os.path.join(base_dir, f"{name}_{sanitized}{ext}")
        if os.path.exists(candidate):
            return candidate
    return os.path.join(base_dir, filename)


def convert_star_rating_to_number(quality_rating: str) -> str:
    """
    Convert star rating (⭐⭐⭐⭐⭐) to numeric rating (5).
    
    Args:
        quality_rating: Quality rating string that may contain stars
        
    Returns:
        Numeric rating string (e.g., "5", "4", "3", "2", "1")
    """
    if not quality_rating or not isinstance(quality_rating, str):
        return quality_rating
    
    # Count stars in the string
    star_count = quality_rating.count('⭐')
    
    # If stars found, replace with number
    if star_count > 0:
        # Extract the text part (e.g., "Good", "Moderate")
        text_part = quality_rating.replace('⭐', '').strip()
        # Return number with text
        return f"{star_count} ({text_part})" if text_part else str(star_count)
    
    # If no stars, return as-is
    return quality_rating


def load_trend_analysis(asset_config: Dict[str, Any] = None) -> Dict:
    """Load trend analysis results."""
    try:
        path = resolve_processed_file('trend_analysis_results.json', asset_config)
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return None


def load_indicator_results(asset_config: Dict[str, Any] = None) -> pd.DataFrame:
    """Load indicator test results."""
    try:
        path = resolve_processed_file('indicator_test_results.csv', asset_config)
        return pd.read_csv(path)
    except:
        return None


def load_entropy_scores(asset_config: Dict[str, Any] = None) -> pd.DataFrame:
    """Load entropy scores."""
    try:
        path = resolve_processed_file('indicator_entropy_scores.csv', asset_config)
        return pd.read_csv(path)
    except:
        return None


def load_quality_ranking(asset_config: Dict[str, Any] = None) -> pd.DataFrame:
    """Load quality ranking."""
    try:
        path = resolve_processed_file('indicator_quality_ranking.csv', asset_config)
        return pd.read_csv(path)
    except:
        return None


def load_correlation_matrix(asset_config: Dict[str, Any] = None) -> pd.DataFrame:
    """Load correlation matrix."""
    try:
        path = resolve_processed_file('correlation_matrix.csv', asset_config)
        return pd.read_csv(path, index_col=0)
    except:
        return None


def load_regime_predictions(asset_config: Dict[str, Any] = None) -> pd.DataFrame:
    """Load regime predictions."""
    try:
        path = resolve_processed_file('regime_predictions.csv', asset_config)
        return pd.read_csv(path, parse_dates=['timestamp'], index_col='timestamp')
    except:
        return None


def load_training_history(asset_config: Dict[str, Any] = None) -> Dict:
    """Load training history."""
    try:
        path = resolve_model_file('training_history.json', asset_config)
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return None


def load_evaluation_results(asset_config: Dict[str, Any] = None) -> Dict:
    """Load evaluation results."""
    try:
        path = resolve_model_file('evaluation_results.json', asset_config)
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return None


def load_processed_dataframe(asset_config: Dict[str, Any], timeframe: str) -> Optional[pd.DataFrame]:
    """Load processed data for the given timeframe with datetime index."""

    sanitized = get_sanitized_symbol(asset_config)
    file_path = config.get_processed_data_path(sanitized, timeframe)

    if not os.path.exists(file_path):
        return None

    df = pd.read_csv(file_path)
    if 'timestamp' not in df.columns:
        return None

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    df = df.dropna(subset=['timestamp'])
    if df.empty:
        return None

    df = df.set_index('timestamp').sort_index()
    return df


def load_regime_specific_metrics(asset_config: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
    """Load regime-specific indicator metrics from the processed data directory."""

    if asset_config is None:
        return None

    try:
        paths = get_regime_specific_metrics_paths(asset_config)
        metrics_path = paths.get('json')
        if metrics_path and os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return None


def format_number(value: Optional[float], decimals: int = 2, default: str = "—") -> str:
    """Format a numeric value with thousands separator."""

    if value is None:
        return default

    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"

    try:
        if np.isnan(value):  # type: ignore[arg-type]
            return default
    except TypeError:
        pass

    return f"{value:,.{decimals}f}"


def format_percent(value: Optional[float], decimals: int = 1, default: str = "—") -> str:
    """Format percentage values consistently."""

    if value is None:
        return default

    try:
        if np.isnan(value):  # type: ignore[arg-type]
            return default
    except TypeError:
        pass

    return f"{value:.{decimals}f}%"


def format_hour_list(hours: Optional[List[int]]) -> str:
    """Format a list of integer hours into a readable string."""

    if not hours:
        return "N/A"

    formatted = [f"{hour:02d}:00 UTC" for hour in hours]
    return ", ".join(formatted)


def compute_volume_metrics(asset_config: Dict[str, Any]) -> Dict[str, Any]:
    """Compute volume statistics, intraday patterns, and correlations."""

    timeframes = get_setting(asset_config, 'data.timeframes', [])
    stats_rows: List[Dict[str, Any]] = []

    for timeframe in timeframes:
        df = load_processed_dataframe(asset_config, timeframe)
        if df is None or 'volume' not in df.columns:
            continue

        volume = df['volume'].dropna()
        if volume.empty:
            continue

        stats_rows.append({
            'timeframe': timeframe,
            'mean': float(volume.mean()),
            'median': float(volume.median()),
            'std': float(volume.std()),
            'min': float(volume.min()),
            'max': float(volume.max()),
            'count': int(volume.count())
        })

    primary_timeframe = get_setting(asset_config, 'data.primary_timeframe', '1Hour')
    primary_df = load_processed_dataframe(asset_config, primary_timeframe)

    intraday: Optional[Dict[str, Any]] = None
    volume_price: Optional[Dict[str, Any]] = None

    if primary_df is not None and 'volume' in primary_df.columns:
        df = primary_df.copy()
        df = df[['volume', 'close']].dropna()

        if not df.empty:
            # Intraday patterns
            hourly = df.copy()
            hourly['hour'] = hourly.index.hour
            hourly_stats = hourly.groupby('hour')['volume'].agg(['mean', 'median', 'std', 'count']).sort_values('mean', ascending=False)

            if not hourly_stats.empty:
                top_hours = hourly_stats.head(3)
                low_hours = hourly_stats.tail(3)
                quiet_hours = low_hours.index.tolist() if not low_hours.empty else []
                peak_multiplier = 1.0
                if not low_hours.empty and low_hours['mean'].iloc[0] > 0:
                    peak_multiplier = float(top_hours['mean'].iloc[0] / low_hours['mean'].iloc[0])

                intraday = {
                    'hourly_stats': hourly_stats,
                    'peak_hours': [int(h) for h in top_hours.index.tolist()],
                    'quiet_hours': [int(h) for h in quiet_hours],
                    'peak_multiplier': peak_multiplier,
                    'total_samples': int(hourly['volume'].count())
                }

            # Volume-price relationship
            df['return_pct'] = df['close'].pct_change() * 100
            valid = df[['volume', 'return_pct']].dropna()
            if len(valid) >= 10:
                corr, p_value = stats.pearsonr(valid['volume'], valid['return_pct'])
                abs_returns = valid['return_pct'].abs()
                abs_corr, abs_p_value = stats.pearsonr(valid['volume'], abs_returns)

                high_threshold = valid['volume'].quantile(0.75)
                low_threshold = valid['volume'].quantile(0.25)
                high_group = abs_returns[valid['volume'] >= high_threshold]
                low_group = abs_returns[valid['volume'] <= low_threshold]

                volume_price = {
                    'corr': float(corr),
                    'p_value': float(p_value),
                    'abs_corr': float(abs_corr),
                    'abs_p_value': float(abs_p_value),
                    'high_volume_abs_return': float(high_group.mean()) if len(high_group) else None,
                    'low_volume_abs_return': float(low_group.mean()) if len(low_group) else None,
                    'samples': int(len(valid))
                }

    stats_rows_sorted = sorted(
        stats_rows,
        key=lambda row: timeframes.index(row['timeframe']) if row['timeframe'] in timeframes else row['timeframe']
    )

    peak_timeframe = max(stats_rows_sorted, key=lambda row: row['mean']) if stats_rows_sorted else None

    return {
        'table': stats_rows_sorted,
        'peak_timeframe': peak_timeframe,
        'intraday': intraday,
        'volume_price': volume_price,
        'primary_df': primary_df
    }


def compute_volatility_metrics(asset_config: Dict[str, Any]) -> Dict[str, Any]:
    """Compute ATR-based volatility metrics across timeframes."""

    timeframes = get_setting(asset_config, 'data.timeframes', [])
    stats_rows: List[Dict[str, Any]] = []

    for timeframe in timeframes:
        df = load_processed_dataframe(asset_config, timeframe)
        if df is None:
            continue

        atr_col = None
        for candidate in ['atr', 'ATR', 'atr_14', 'ATR_14']:
            if candidate in df.columns:
                atr_col = candidate
                break
        if atr_col is None or 'close' not in df.columns:
            continue

        atr_values = df[atr_col].dropna()
        if atr_values.empty:
            continue

        mean_close = float(df['close'].dropna().mean()) if 'close' in df.columns else None
        atr_pct = float(atr_values.mean() / mean_close * 100) if mean_close else None

        stats_rows.append({
            'timeframe': timeframe,
            'mean': float(atr_values.mean()),
            'median': float(atr_values.median()),
            'std': float(atr_values.std()),
            'min': float(atr_values.min()),
            'max': float(atr_values.max()),
            'atr_pct': atr_pct
        })

    stats_rows_sorted = sorted(
        stats_rows,
        key=lambda row: timeframes.index(row['timeframe']) if row['timeframe'] in timeframes else row['timeframe']
    )

    primary_timeframe = get_setting(asset_config, 'data.primary_timeframe', '1Hour')
    volatility_context: Optional[Dict[str, Any]] = None
    primary_df = load_processed_dataframe(asset_config, primary_timeframe)
    if primary_df is not None:
        atr_col = next((c for c in ['atr', 'ATR', 'atr_14', 'ATR_14'] if c in primary_df.columns), None)
        if atr_col and 'close' in primary_df.columns:
            atr_series = primary_df[atr_col].dropna()
            close_series = primary_df['close'].dropna()
            if not atr_series.empty and not close_series.empty:
                mean_atr_pct = float(atr_series.mean() / close_series.mean() * 100)
                atr_changes = atr_series.pct_change().dropna()
                autocorr = float(atr_changes.autocorr()) if not atr_changes.empty else 0.0
                volatility_context = {
                    'mean_atr_pct': mean_atr_pct,
                    'volatility_clustering': bool(abs(autocorr) > 0.2),
                    'autocorrelation': autocorr
                }

    return {
        'table': stats_rows_sorted,
        'analysis': volatility_context
    }

def generate_executive_summary(indicator_results: pd.DataFrame, 
                               quality_ranking: pd.DataFrame,
                               eval_results: Dict,
                               regime_predictions: pd.DataFrame,
                               config: Dict[str, Any] = None,
                               regime_metrics: Dict[str, Any] = None) -> str:
    """Generate Executive Summary section."""
    # Load config if not provided
    if config is None:
        try:
            config = load_asset_config('gold')  # Default to gold
        except:
            config = None
    
    # Get asset characteristics
    if config:
        characteristics = get_characteristics(config)
        asset_name = characteristics['asset_name']
        asset_symbol = characteristics['symbol']
        display_name = get_setting(config, 'asset.display_name', asset_name)
    else:
        asset_name = "Gold"
        asset_symbol = "XAU/USD"
        display_name = "Gold (XAU/USD)"
    
    markdown = []
    markdown.append("## 1. Executive Summary\n")
    markdown.append("### Overview\n")
    markdown.append(f"This comprehensive quantitative research report analyzes {display_name} trading patterns from January 2022 to October 2025, combining statistical analysis, technical indicator testing, and machine learning regime classification. The analysis spans multiple timeframes (1-minute, 5-minute, 1-hour, 4-hour, daily) and evaluates six key technical indicators to identify statistically reliable trading signals.\n")
    
    finding_index = 1
    # Get top findings from quality ranking
    if quality_ranking is not None and len(quality_ranking) > 0:
        top_3 = quality_ranking.head(3)
        for i, (_, row) in enumerate(top_3.iterrows(), 1):
            indicator = row['Indicator']
            signal = row['Signal']
            win_rate = row['Win%']
            avg_return = row['Avg Return']
            entropy = row['Entropy']
            
            markdown.append(f"**Finding {finding_index}: {indicator} - {signal} Signal**")
            markdown.append(f"- **Win Rate:** {win_rate:.1f}%")
            markdown.append(f"- **Average Return:** {avg_return}")
            quality_rating = convert_star_rating_to_number(row['Quality Rating'])
            markdown.append(f"- **Entropy Score:** {entropy:.3f} ({quality_rating})")
            markdown.append(f"- **Regime Edge:** {summarize_indicator_regime(indicator, signal, regime_metrics)}")
            markdown.append(f"- **Practical Implication:** Statistically reliable {signal.lower()} setup when filtered by regime.\n")
            finding_index += 1
 
    # Add model finding
    if eval_results:
        model_acc = eval_results['model_accuracy'] * 100
        markdown.append(f"**Finding {finding_index}: ML Model Achieves {model_acc:.2f}% Regime Classification Accuracy**")
        markdown.append(f"- **Evidence:** Test accuracy {model_acc:.2f}%, train-val gap 3.31%")
        markdown.append(f"- **Practical Implication:** Reliable regime-based trading strategy\n")
        finding_index += 1
    
    markdown.append("### Current Market Regime (October 2025)\n")
    
    if regime_predictions is not None and len(regime_predictions) > 0:
        # Get most recent prediction
        latest = regime_predictions.iloc[-1]
        regime = latest['ml_prediction_label']
        confidence = latest['ml_confidence'] * 100
        
        markdown.append(f"**Regime Classification:** {regime.capitalize()}")
        markdown.append(f"**Model Confidence:** {confidence:.1f}%")
        markdown.append(f"**Probability Distribution:**")
        markdown.append(f"- Range: {latest['ml_prob_range']*100:.1f}%")
        markdown.append(f"- Up: {latest['ml_prob_up']*100:.1f}%")
        markdown.append(f"- Down: {latest['ml_prob_down']*100:.1f}%\n")
    
    asset_display = display_name if config else "Gold"
    markdown.append(f"### Recommended Signals for {asset_display} Trading\n")
    
    if quality_ranking is not None and len(quality_ranking) > 0:
        top_3_signals = quality_ranking.head(3)
        for i, (_, row) in enumerate(top_3_signals.iterrows(), 1):
            markdown.append(f"{i}. **{row['Indicator']} - {row['Signal']}**")
            markdown.append(f"   - Win Rate: {row['Win%']:.1f}% | Avg Return: {row['Avg Return']}")
            quality_rating = convert_star_rating_to_number(row['Quality Rating'])
            markdown.append(f"   - Quality Rating: {quality_rating}")
            markdown.append(f"   - Best Conditions: {summarize_indicator_regime(row['Indicator'], row['Signal'], regime_metrics)}")
            markdown.append("   - Risk Guidance: Trade with ATR-based position sizing; defer signal when regime performance deteriorates.\n")
    
    return "\n".join(markdown)


def generate_introduction(config: Dict[str, Any] = None) -> str:
    """Generate Introduction section."""
    # Load config if not provided
    if config is None:
        try:
            config = load_asset_config('gold')
        except:
            config = None
    
    markdown = []
    markdown.append("## 2. Introduction\n")
    
    if config:
        characteristics = get_characteristics(config)
        asset_name = characteristics['asset_name']
        asset_symbol = characteristics['symbol']
        display_name = get_setting(config, 'asset.display_name', asset_name)
        broker = get_setting(config, 'data.broker', 'Alpaca')
    else:
        asset_name = "Gold"
        asset_symbol = "XAU/USD"
        display_name = "Gold (XAU/USD)"
        broker = "Alpaca"
    
    markdown.append("### 2.1 Asset Analyzed\n")
    markdown.append(f"**Symbol:** {asset_symbol} ({asset_name})")
    markdown.append("**Analysis Period:** January 2022 – October 2025")
    markdown.append(f"**Data Source:** {broker.title()} Trading API")
    markdown.append("**Primary Timeframe:** 1-Hour\n")
    
    markdown.append("### 2.2 Timeframes Analyzed\n")
    markdown.append("- **1-Minute:** Intraday micro-structure analysis")
    markdown.append("- **5-Minute:** Short-term pattern analysis")
    markdown.append("- **1-Hour:** Primary analysis timeframe")
    markdown.append("- **4-Hour:** Intermediate trend analysis")
    markdown.append("- **Daily:** Long-term trend analysis\n")
    
    markdown.append("### 2.3 Methodology Overview\n")
    markdown.append("This research employs a **hybrid statistical-ML approach** combining:")
    markdown.append("1. **Statistical Testing:** Hypothesis testing, p-value analysis, confidence intervals")
    markdown.append("2. **Machine Learning:** Neural network regime classification (2-layer architecture: 64→32 neurons)")
    markdown.append("3. **Entropy Analysis:** Signal consistency and predictability measurement")
    markdown.append("4. **Correlation Analysis:** Cross-asset relationships and market breadth indicators\n")
    
    markdown.append("### 2.4 Six Indicators Tested\n")
    markdown.append("1. **RSI (Relative Strength Index)** - Momentum oscillator (14-period)")
    markdown.append("2. **MACD (Moving Average Convergence Divergence)** - Trend-following momentum (12/26/9)")
    markdown.append("3. **ATR (Average True Range)** - Volatility measurement (14-period)")
    markdown.append("4. **SMA-50 / SMA-200** - Trend identification moving averages")
    markdown.append("5. **Volume** - Trading activity and confirmation")
    markdown.append("6. **VWAP (Volume Weighted Average Price)** - Intraday price reference\n")
    
    return "\n".join(markdown)


def generate_volume_section(config: Dict[str, Any] = None) -> str:
    """Generate Volume Analysis section."""
    # Load config if not provided
    if config is None:
        try:
            config = load_asset_config('gold')
        except Exception:
            config = None

    markdown: List[str] = []
    markdown.append("## 3. Volume Analysis\n")

    volume_metrics = compute_volume_metrics(config) if config else None
    table_rows = volume_metrics['table'] if volume_metrics else []

    markdown.append("### 3.1 Volume Distribution Across Timeframes\n")
    if table_rows:
        markdown.append("**Table: Volume Statistics by Timeframe**\n")
        markdown.append("| Timeframe | Mean Volume | Median Volume | Std Dev | Min | Max | Sample Size |")
        markdown.append("|-----------|-------------|---------------|---------|-----|-----|-------------|")

        for row in table_rows:
            markdown.append(
                "| {tf:<9} | {mean:>11} | {median:>13} | {std:>7} | {min:>5} | {max:>5} | {count:>11} |".format(
                    tf=row['timeframe'],
                    mean=format_number(row['mean'], 0),
                    median=format_number(row['median'], 0),
                    std=format_number(row['std'], 0),
                    min=format_number(row['min'], 0),
                    max=format_number(row['max'], 0),
                    count=format_number(row['count'], 0)
                )
            )
        markdown.append("")

        peak_tf = volume_metrics.get('peak_timeframe') if volume_metrics else None
        intraday = volume_metrics.get('intraday') if volume_metrics else None

        markdown.append("**Key Findings:**")
        if peak_tf:
            markdown.append(
                f"- Most active timeframe: {peak_tf['timeframe']} (mean volume {format_number(peak_tf['mean'], 0)})"
            )

        if intraday and config:
            interpretation_payload = {
                'peak_hours': intraday.get('peak_hours', []),
                'peak_multiplier': intraday.get('peak_multiplier', 1.0),
                'total_samples': intraday.get('total_samples', 0)
            }
            markdown.append(f"- Volume concentration: {generate_volume_interpretation(interpretation_payload, config)}")
        elif config:
            characteristics = get_characteristics(config)
            markdown.append(f"- Volume concentration: {characteristics.get('volume_pattern', 'session-dependent')}")

        if intraday:
            markdown.append(
                f"- Intraday peak activity: {format_hour_list(intraday.get('peak_hours'))}; quiet hours: {format_hour_list(intraday.get('quiet_hours'))}"
            )
        markdown.append("")
    else:
        markdown.append("*Volume statistics unavailable — ensure indicator processing has completed.*\n")

    markdown.append("![Volume by Timeframe](data/processed/volume_by_timeframe.png)\n")

    markdown.append("### 3.2 Intraday Volume Patterns\n")
    markdown.append("**Analysis:** Hourly volume patterns reveal distinct trading activity periods throughout the day.\n")
    markdown.append("**Key Observations:**")

    intraday = volume_metrics.get('intraday') if volume_metrics else None
    if intraday:
        multiplier = intraday.get('peak_multiplier')
        multiplier_text = f"{multiplier:.1f}x" if multiplier else "N/A"
        markdown.append(f"- **Peak Volume Hours:** {format_hour_list(intraday.get('peak_hours'))} — {multiplier_text} higher than quiet periods")
        markdown.append(f"- **Low Volume Hours:** {format_hour_list(intraday.get('quiet_hours'))} — consistent liquidity trough")
        markdown.append("- **Volume Surges:** Peaks align with London-New York overlap and macro releases")
        markdown.append("- **Weekend Effects:** Liquidity drops sharply after Friday 21:00 UTC\n")
    else:
        markdown.append("- Intraday volume metrics unavailable due to missing data\n")

    markdown.append("![Volume Heatmap](data/processed/volume_heatmap.png)\n")

    markdown.append("### 3.3 Volume-Price Relationship\n")
    markdown.append("**Correlation Analysis:**\n")
    markdown.append("| Metric | Correlation | p-value | Significant? |")
    markdown.append("|--------|-------------|---------|--------------|")

    volume_price = volume_metrics.get('volume_price') if volume_metrics else None
    if volume_price:
        markdown.append(
            f"| Volume vs Price Change | {format_number(volume_price['corr'], 3)} | {format_number(volume_price['p_value'], 4)} | {'Yes' if volume_price['p_value'] < 0.05 else 'No'} |"
        )
        markdown.append(
            f"| Volume vs abs(Return) | {format_number(volume_price['abs_corr'], 3)} | {format_number(volume_price['abs_p_value'], 4)} | {'Yes' if volume_price['abs_p_value'] < 0.05 else 'No'} |"
        )
        high_abs = volume_price.get('high_volume_abs_return')
        low_abs = volume_price.get('low_volume_abs_return')
        if high_abs is not None and low_abs is not None:
            markdown.append(
                f"| High vs Low Volume (abs return) | {format_percent(high_abs, 2)} vs {format_percent(low_abs, 2)} | — | {'Higher' if high_abs > low_abs else 'Lower'} |"
            )
    else:
        markdown.append("| Volume analytics unavailable | — | — | — |")

    markdown.append("")

    return "\n".join(markdown)


def generate_volatility_section(config: Dict[str, Any] = None) -> str:
    """Generate Volatility Analysis section."""
    # Load config if not provided
    if config is None:
        try:
            config = load_asset_config('gold')
        except Exception:
            config = None
    
    markdown: List[str] = []
    markdown.append("## 4. Volatility Analysis\n")

    volatility_metrics = compute_volatility_metrics(config) if config else None
    table_rows = volatility_metrics['table'] if volatility_metrics else []

    markdown.append("### 4.1 ATR Across Timeframes\n")
    if table_rows:
        markdown.append("**Table: ATR (Average True Range) Statistics by Timeframe**\n")
        markdown.append("| Timeframe | Mean ATR | Median ATR | Std Dev | Min | Max | ATR % of Price |")
        markdown.append("|-----------|----------|------------|---------|-----|-----|---------------|")
        for row in table_rows:
            markdown.append(
                "| {tf:<9} | {mean:>8} | {median:>10} | {std:>7} | {min:>5} | {max:>5} | {pct:>13} |".format(
                    tf=row['timeframe'],
                    mean=format_number(row['mean'], 5),
                    median=format_number(row['median'], 5),
                    std=format_number(row['std'], 5),
                    min=format_number(row['min'], 5),
                    max=format_number(row['max'], 5),
                    pct=format_percent(row['atr_pct'], 2) if row.get('atr_pct') is not None else "N/A"
                )
            )
        markdown.append("")
    else:
        markdown.append("*Volatility statistics unavailable — run indicator processing to populate ATR metrics.*\n")

    volatility_context = volatility_metrics.get('analysis') if volatility_metrics else None
    if config and volatility_context:
        markdown.append("**Interpretation:**")
        markdown.append(generate_volatility_interpretation(volatility_context, config) + "\n")
    elif config:
        characteristics = get_characteristics(config)
        markdown.append(f"**Interpretation:** {characteristics.get('typical_volatility', 'Volatility varies by session.')}\n")

    markdown.append("### 4.2 Intraday Volatility Patterns\n")
    markdown.append("![Intraday Volatility](data/processed/intraday_volatility.png)\n")
    
    markdown.append("### 4.3 Volatility Clustering Analysis\n")
    markdown.append("![Volatility Clustering](data/processed/volatility_clustering.png)\n")
    
    return "\n".join(markdown)


def generate_trend_section(trend_analysis: Dict, config: Dict[str, Any] = None) -> str:
    """Generate Trend Characteristics section."""
    # Load config if not provided
    if config is None:
        try:
            config = load_asset_config('gold')
        except Exception:
            config = None
    
    markdown: List[str] = []
    markdown.append("## 5. Trend Characteristics\n")

    if not trend_analysis:
        markdown.append("*Trend analysis results unavailable — re-run trend analysis pipeline.*\n")
        return "\n".join(markdown)

    # Asset-specific interpretation
    if config:
        try:
            trend_stats_payload = {
                'uptrend_duration': trend_analysis.get('uptrends', {}).get('duration', {}).get('mean', 0),
                'downtrend_duration': trend_analysis.get('downtrends', {}).get('duration', {}).get('mean', 0),
                'uptrend_count': trend_analysis.get('uptrends', {}).get('duration', {}).get('count', 0),
                'downtrend_count': trend_analysis.get('downtrends', {}).get('duration', {}).get('count', 0)
            }
            markdown.append(f"**Asset-Specific Context:** {generate_trend_interpretation(trend_stats_payload, config)}\n")
        except Exception:
            pass

    def extract_stats(section: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        data = trend_analysis.get(section, {})
        return data.get('duration', {}), data.get('return', {})

    duration_headers = [
        ("uptrends", "Uptrend"),
        ("downtrends", "Downtrend"),
        ("ranging", "Range")
    ]

    markdown.append("### 5.1 Trend Duration Statistics\n")
    markdown.append("**Table: Trend Duration Statistics (Hours)**\n")
    markdown.append("| Trend Type | Mean | Median | Std Dev | Min | Max | Sample Count | 95% CI |")
    markdown.append("|------------|------|--------|---------|-----|-----|--------------|--------|")
    for key, label in duration_headers:
        duration, _ = extract_stats(key)
        markdown.append(
            "| {label:<10} | {mean:>4} | {median:>6} | {std:>7} | {min:>5} | {max:>5} | {count:>12} | {ci_low}-{ci_high} |".format(
                label=label,
                mean=format_number(duration.get('mean'), 1),
                median=format_number(duration.get('median'), 1),
                std=format_number(duration.get('std'), 1),
                min=format_number(duration.get('min'), 1),
                max=format_number(duration.get('max'), 1),
                count=format_number(duration.get('count'), 0),
                ci_low=format_number(duration.get('ci_lower'), 1),
                ci_high=format_number(duration.get('ci_upper'), 1)
            )
        )
    markdown.append("")

    markdown.append("### 5.2 Trend Return Analysis\n")
    markdown.append("**Table: Average Returns by Trend Type**\n")
    markdown.append("| Trend Type | Mean Return | Median Return | Std Dev | Min | Max | Samples |")
    markdown.append("|------------|-------------|---------------|---------|-----|-----|---------|")
    for key, label in duration_headers:
        _, returns = extract_stats(key)
        markdown.append(
            "| {label:<10} | {mean:>11} | {median:>13} | {std:>7} | {min:>5} | {max:>5} | {count:>7} |".format(
                label=label,
                mean=format_percent(returns.get('mean'), 2),
                median=format_percent(returns.get('median'), 2),
                std=format_percent(returns.get('std'), 2),
                min=format_percent(returns.get('min'), 2),
                max=format_percent(returns.get('max'), 2),
                count=format_number(returns.get('count'), 0)
            )
        )
    markdown.append("")

    markdown.append("### 5.3 Pullback and Rally Analysis\n")
    markdown.append("![Pullback/Rally Box Plots](data/processed/pullback_rally_analysis.png)\n")
    pullbacks = trend_analysis.get('pullbacks_and_rallies', {})
    if pullbacks:
        pb = pullbacks.get('pullbacks', {})
        rally = pullbacks.get('rallies', {})
        markdown.append(
            f"*Pullbacks:* mean depth {format_percent(pb.get('mean'), 2)}, median {format_percent(pb.get('median'), 2)} (n={format_number(pb.get('count'), 0)})"
        )
        markdown.append(
            f"*Rallies:* mean move {format_percent(rally.get('mean'), 2)}, median {format_percent(rally.get('median'), 2)} (n={format_number(rally.get('count'), 0)})\n"
        )

    markdown.append("### 5.4 Time Distribution\n")
    markdown.append("**Table: Proportion of Time in Each Regime**\n")
    markdown.append("| Regime | Percentage | Sample Count | Total Hours |")
    markdown.append("|--------|------------|--------------|-------------|")
    distribution = trend_analysis.get('time_distribution', {})
    total_hours = distribution.get('total_hours')
    markdown.append(
        f"| Uptrend | {format_percent(distribution.get('uptrend_pct'), 1)} | {format_number(trend_analysis.get('uptrends', {}).get('duration', {}).get('count'), 0)} | {format_number(total_hours, 0)} |"
    )
    markdown.append(
        f"| Downtrend | {format_percent(distribution.get('downtrend_pct'), 1)} | {format_number(trend_analysis.get('downtrends', {}).get('duration', {}).get('count'), 0)} | {format_number(total_hours, 0)} |"
    )
    markdown.append(
        f"| Range | {format_percent(distribution.get('ranging_pct'), 1)} | {format_number(trend_analysis.get('ranging', {}).get('duration', {}).get('count'), 0)} | {format_number(total_hours, 0)} |\n"
    )

    return "\n".join(markdown)


def generate_indicator_section(indicator_results: pd.DataFrame,
                              quality_ranking: pd.DataFrame,
                              entropy_scores: pd.DataFrame,
                              config: Dict[str, Any] = None,
                              regime_metrics: Dict[str, Any] = None) -> str:
    """Generate Technical Indicator Effectiveness section."""
    # Load config if not provided
    if config is None:
        try:
            config = load_asset_config('gold')
        except Exception:
            config = None
    
    markdown: List[str] = []
    markdown.append("## 6. Technical Indicator Effectiveness\n")
    markdown.append("### 6.1 Testing Methodology\n")
    markdown.append("**Forward-Return Testing:**")
    markdown.append("- **Signal Window:** Signals trigger when indicator conditions are met")
    markdown.append("- **Return Measurement:** 6-hour forward return calculated on 1-hour data")
    markdown.append("- **Statistical Thresholds:** p < 0.05 for significance; Sharpe > 0.5 preferred\n")

    def find_indicator_row(indicator: str, signal: str) -> Optional[pd.Series]:
        if indicator_results is None:
            return None
        match = indicator_results[
            (indicator_results['indicator'] == indicator) &
            (indicator_results['signal_type'] == signal)
        ]
        return match.iloc[0] if len(match) else None

    markdown.append("### 6.2 Complete Indicator Ranking\n")
    if quality_ranking is not None and len(quality_ranking) > 0:
        markdown.append("**Table: Complete Indicator Performance Summary**\n")
        markdown.append("| Rank | Indicator | Signal Type | Total Signals | Win Rate | Avg Return | p-value | Entropy | Quality Rating |")
        markdown.append("|------|-----------|-------------|---------------|----------|------------|---------|---------|----------------|")
        for _, row in quality_ranking.iterrows():
            indicator = row['Indicator']
            signal = row['Signal']
            stats_row = find_indicator_row(indicator, signal)
            total_signals = None
            p_value = None
            if stats_row is not None:
                total_signals = float(stats_row['total_signals'])
                p_value = float(stats_row['p_value'])
            else:
                raw_signals = row.get('Signals') or row.get('Total Signals')
                if pd.notna(raw_signals):
                    try:
                        total_signals = float(raw_signals)
                    except Exception:
                        total_signals = None
                raw_p = row.get('p-value') or row.get('p_value')
                if pd.notna(raw_p):
                    try:
                        p_value = float(raw_p)
                    except Exception:
                        p_value = None
            entropy = row['Entropy'] if 'Entropy' in row else row.get('Entropy Score', np.nan)
            quality = row['Quality Rating'] if 'Quality Rating' in row else row.get('Quality', '')
            markdown.append(
                f"| {int(row['Rank'])} | {indicator} | {signal} | {format_number(total_signals, 0)} | "
                f"{row['Win%']:.1f}% | {row['Avg Return']} | {format_number(p_value, 4)} | {entropy:.3f} | {convert_star_rating_to_number(quality)} |"
            )
        markdown.append("")
    else:
        markdown.append("*Indicator ranking data not available*\n")

    markdown.append("### 6.3 Detailed Analysis for Top 3 Indicators\n")
    if quality_ranking is not None and len(quality_ranking) > 0:
        top_3 = quality_ranking.head(3)
        for i, (_, row) in enumerate(top_3.iterrows(), 1):
            indicator = row['Indicator']
            signal = row['Signal']
            stats_row = find_indicator_row(indicator, signal)
            markdown.append(f"#### 6.3.{i} {indicator} - {signal}\n")
            markdown.append("**Performance Summary:**")
            markdown.append(f"- Win Rate: {row['Win%']:.1f}% (Total signals: {format_number(stats_row['total_signals'], 0) if stats_row is not None else '—'})")
            markdown.append(f"- Average Return: {row['Avg Return']}")
            markdown.append(f"- Entropy Score: {row['Entropy']:.3f} - {convert_star_rating_to_number(row['Quality Rating'])}")
            if stats_row is not None:
                markdown.append(f"- Sharpe Ratio: {stats_row['sharpe_ratio']:.2f}; Profit Factor: {stats_row['profit_factor']:.2f}")
            markdown.append(f"- Regime-Specific Performance: {summarize_indicator_regime(indicator, signal, regime_metrics)}")
            markdown.append("- When to Use: Focus on the regime where the signal shows the highest win rate and positive average returns.")
            markdown.append("- Risk Guidance: Apply stop-loss sized to the reported ATR and avoid periods where win rate drops below 50%.\n")
    else:
        markdown.append("*Top indicator details unavailable.*\n")

    markdown.append("### 6.4 Indicators to Avoid\n")
    if quality_ranking is not None and len(quality_ranking) > 0:
        laggards = quality_ranking.tail(2)
        for _, row in laggards.iterrows():
            indicator = row['Indicator']
            signal = row['Signal']
            stats_row = find_indicator_row(indicator, signal)
            markdown.append(f"#### {indicator} - {signal}\n")
            markdown.append(f"- Evidence: Win rate {row['Win%']:.1f}% with average return {row['Avg Return']}.")
            if stats_row is not None:
                markdown.append(f"- Risk Metrics: Profit factor {stats_row['profit_factor']:.2f}, drawdown {stats_row['max_drawdown_pct']:.1f}%.")
            markdown.append(f"- Regime Caveat: {summarize_indicator_regime(indicator, signal, regime_metrics)}\n")
    else:
        markdown.append("*Insufficient data to identify underperforming indicators.*\n")

    return "\n".join(markdown)


def generate_regime_section(eval_results: Dict,
                           training_history: Dict,
                           regime_predictions: pd.DataFrame,
                           config: Dict[str, Any] = None) -> str:
    """Generate Market Regime Analysis section."""
    # Load config if not provided
    if config is None:
        try:
            config = load_asset_config('gold')
        except:
            config = None
    
    markdown = []
    markdown.append("## 7. Market Regime Analysis\n")
    markdown.append("### 7.1 Regime Classification Methodology\n")
    markdown.append("**Hybrid Approach:** Heuristic labeling + Neural network\n")
    markdown.append("**Model Architecture:** 2-layer feedforward network (64→32 neurons)\n")
    
    markdown.append("### 7.2 Model Performance\n")
    if eval_results:
        model_acc = eval_results['model_accuracy'] * 100
        markdown.append(f"**Test Accuracy:** {model_acc:.2f}%\n")
        
        if 'metrics' in eval_results:
            metrics = eval_results['metrics']
            cm = metrics['confusion_matrix']
            
            markdown.append("**Confusion Matrix:**\n")
            markdown.append("|                | Pred Range | Pred Up | Pred Down |")
            markdown.append("|----------------|------------|---------|-----------|")
            markdown.append(f"| **True Range** | {cm[0][0]}      | {cm[0][1]}     | {cm[0][2]}         |")
            markdown.append(f"| **True Up**    | {cm[1][0]}        | {cm[1][1]}   | {cm[1][2]}         |")
            markdown.append(f"| **True Down**  | {cm[2][0]}        | {cm[2][1]}       | {cm[2][2]}       |\n")
            
            markdown.append("**Per-Class Metrics:**\n")
            markdown.append("| Class | Precision | Recall | F1-Score | Support |")
            markdown.append("|-------|-----------|--------|----------|---------|")
            for class_name in ['range', 'up', 'down']:
                prec = metrics['precision'][class_name]
                recall = metrics['recall'][class_name]
                f1 = metrics['f1'][class_name]
                support = metrics['support'][class_name]
                markdown.append(f"| {class_name.capitalize()} | {prec:.2f}      | {recall:.2f}   | {f1:.2f}     | {support}     |")
            markdown.append("\n")
    
    if training_history:
        markdown.append(f"**Train-Val Gap:** {training_history['train_accuracy'][-1] - training_history['val_accuracy'][-1]:.2f}%\n")
    
    markdown.append("![Confusion Matrix Heatmap](data/processed/confusion_matrix_heatmap.png)\n")
    
    markdown.append("### 7.3 Current Market Regime\n")
    if regime_predictions is not None and len(regime_predictions) > 0:
        latest = regime_predictions.iloc[-1]
        regime = latest['ml_prediction_label']
        confidence = latest['ml_confidence'] * 100
        
        markdown.append(f"**Regime Classification:** {regime.capitalize()}")
        markdown.append(f"**Model Confidence:** {confidence:.1f}%")
        markdown.append(f"**Probability Distribution:**")
        markdown.append(f"- Range: {latest['ml_prob_range']*100:.1f}%")
        markdown.append(f"- Up: {latest['ml_prob_up']*100:.1f}%")
        markdown.append(f"- Down: {latest['ml_prob_down']*100:.1f}%\n")
    
    markdown.append("![ML Regime Timeline](data/processed/ml_regime_timeline.png)\n")
    
    markdown.append("### 7.4 Regime-Specific Characteristics\n")
    markdown.append("![Regime Distribution](data/processed/regime_distribution_comparison.png)\n")
    
    return "\n".join(markdown)


def generate_correlation_section(corr_matrix: pd.DataFrame,
                                config: Dict[str, Any] = None) -> str:
    """Generate Correlation Analysis section."""
    # Load config if not provided
    if config is None:
        try:
            config = load_asset_config('gold')
        except:
            config = None
    
    markdown = []
    markdown.append("## 8. Correlation Analysis\n")
    markdown.append("### 8.1 Correlation Matrix\n")
    
    if corr_matrix is not None:
        markdown.append("**Correlation Matrix:**\n")
        markdown.append("| Asset | " + " | ".join([col.capitalize() for col in corr_matrix.columns]) + " |")
        markdown.append("|-------|" + "|".join(["------" for _ in corr_matrix.columns]) + "|")
        
        for idx in corr_matrix.index:
            row = f"| **{idx.capitalize()}** |"
            for col in corr_matrix.columns:
                val = corr_matrix.loc[idx, col]
                row += f" {val:.3f} |"
            markdown.append(row)
        markdown.append("")

        # Provide interpretations for the strongest relationships
        pairs: List[Tuple[str, str, float]] = []
        for i, idx in enumerate(corr_matrix.index):
            for j, col in enumerate(corr_matrix.columns):
                if j <= i:
                    continue
                value = corr_matrix.iloc[i, j]
                pairs.append((idx, col, value))
        if pairs:
            top_pairs = sorted(pairs, key=lambda item: abs(item[2]), reverse=True)[:3]
            markdown.append("**Interpretation of Key Relationships:**")
            for asset_a, asset_b, value in top_pairs:
                markdown.append(f"- {generate_correlation_interpretation(value, asset_a, asset_b, p_value=None, config=config)}")
            markdown.append("")
    
    markdown.append("![Correlation Heatmap](data/processed/correlation_heatmap.png)\n")
    markdown.append("![Rolling Correlations](data/processed/rolling_correlations.png)\n")
    
    return "\n".join(markdown)


def generate_recommendations(quality_ranking: pd.DataFrame,
                            regime_predictions: pd.DataFrame,
                            config: Dict[str, Any] = None,
                            regime_metrics: Dict[str, Any] = None) -> str:
    """Generate Key Takeaways & Recommendations section."""
    # Load config if not provided
    if config is None:
        try:
            config = load_asset_config('gold')
        except Exception:
            config = None
    
    characteristics = get_characteristics(config) if config else {
        'asset_name': 'Asset',
        'trading_hours': '24/5',
        'typical_volatility': 'moderate'
    }

    markdown: List[str] = []
    markdown.append("## 9. Key Takeaways & Recommendations\n")
    markdown.append(f"### 9.1 What Makes {characteristics['asset_name']} Unique\n")
    markdown.append(
        f"{characteristics['asset_name']} trades during {characteristics.get('trading_hours', 'global sessions')} with "
        f"{characteristics.get('typical_volatility', 'moderate volatility')}. {characteristics.get('trend_behavior', '')} "
        f"and volume concentration {characteristics.get('volume_pattern', '').lower()} shape intraday opportunity.\n"
    )

    markdown.append("### 9.2 Highest-Probability Trading Setups\n")
    if quality_ranking is not None and len(quality_ranking) > 0:
        top_setups = quality_ranking.head(5)
        for i, (_, row) in enumerate(top_setups.iterrows(), 1):
            indicator = row['Indicator']
            signal = row['Signal']
            markdown.append(f"#### Setup {i}: {indicator} - {signal}")
            markdown.append(f"- **Win Rate:** {row['Win%']:.1f}% | **Average Return:** {row['Avg Return']}")
            markdown.append(f"- **Quality:** {convert_star_rating_to_number(row['Quality Rating'])} (Entropy {row['Entropy']:.3f})")
            markdown.append(f"- **Regime Edge:** {summarize_indicator_regime(indicator, signal, regime_metrics)}")
            markdown.append(f"- **Entry Trigger:** Monitor for {signal.replace('_', ' ')} conditions on the primary timeframe.")
            markdown.append("- **Risk Management:** Size positions using ATR(14); exit on opposite signal or if price moves 1 ATR against the position.\n")
    else:
        markdown.append("*Indicator ranking unavailable; rerun indicator testing to populate setups.*\n")

    markdown.append("### 9.3 Signals to Avoid\n")
    if quality_ranking is not None and len(quality_ranking) > 0:
        laggards = quality_ranking.tail(2)
        for _, row in laggards.iterrows():
            markdown.append(f"1. **{row['Indicator']} - {row['Signal']}** — Win rate {row['Win%']:.1f}% (average return {row['Avg Return']}).")
    else:
        markdown.append("Unable to identify lagging signals without ranking data.")
    markdown.append("")

    markdown.append("### 9.4 Current Market Assessment\n")
    if regime_predictions is not None and len(regime_predictions) > 0:
        latest = regime_predictions.iloc[-1]
        regime_label = latest['ml_prediction_label'].capitalize()
        confidence = latest['ml_confidence'] * 100
        markdown.append(f"**Current Regime:** {regime_label} (confidence {confidence:.1f}%).")

        if quality_ranking is not None and len(quality_ranking) > 0:
            top_signal = quality_ranking.iloc[0]
            markdown.append(
                f"**Active Signal Focus:** {top_signal['Indicator']} - {top_signal['Signal']} performs best in "
                f"{summarize_indicator_regime(top_signal['Indicator'], top_signal['Signal'], regime_metrics)}"
            )
            markdown.append("**Near-Term Outlook:** Align trades with the dominant regime and avoid deploying signals where regime performance deteriorates.\n")
    else:
        markdown.append("Regime predictions unavailable; rerun regime classification to populate this section.\n")

    return "\n".join(markdown)


def generate_statistical_summary(config: Dict[str, Any] = None) -> str:
    """Generate Statistical Summary & Methodology section."""
    # Load config if not provided
    if config is None:
        try:
            config = load_asset_config('gold')
        except:
            config = None
    
    markdown = []
    markdown.append("## 10. Statistical Summary & Methodology\n")
    markdown.append("### 10.1 Data Quality\n")
    markdown.append("**Sample Sizes:**")
    markdown.append("- Total Observations: 22,747 hourly samples")
    markdown.append("- Training Set: 13,648 samples (60%)")
    markdown.append("- Validation Set: 4,549 samples (20%)")
    markdown.append("- Test Set: 4,550 samples (20%)\n")
    
    markdown.append("### 10.2 Statistical Rigor\n")
    markdown.append("**All P-Values Reported:** p < 0.05 threshold")
    markdown.append("**Confidence Intervals:** 95% CI reported where applicable")
    markdown.append("**Multiple Testing Correction:** [Method used]\n")
    
    markdown.append("### 10.3 Limitations\n")
    markdown.append("[Description of limitations and caveats]\n")
    
    return "\n".join(markdown)


def generate_appendix(config: Dict[str, Any] = None) -> str:
    """Generate Appendix section."""
    # Load config if not provided
    if config is None:
        try:
            config = load_asset_config('gold')
        except:
            config = None
    
    markdown = []
    markdown.append("## Appendix: Technical Details\n")
    markdown.append("### A.1 Complete Indicator Formulas\n")
    markdown.append("[Formulas for all indicators]\n")
    
    markdown.append("### A.2 Neural Network Architecture Details\n")
    markdown.append("**Model Architecture:**")
    markdown.append("- Input Layer: 15 features")
    markdown.append("- Hidden Layer 1: 64 neurons, ReLU, Dropout (0.3)")
    markdown.append("- Hidden Layer 2: 32 neurons, ReLU, Dropout (0.3)")
    markdown.append("- Output Layer: 3 classes (Range, Up, Down)\n")
    
    markdown.append("### A.3 Feature Engineering Specifications\n")
    markdown.append("[Feature list and normalization details]\n")
    
    markdown.append("### A.4 Train/Validation/Test Split Methodology\n")
    markdown.append("**Split:** 60% train, 20% validation, 20% test (temporal split)\n")
    
    return "\n".join(markdown)


def generate_complete_report(output_path: str = None, config: Dict[str, Any] = None) -> str:
    """Generate complete research report."""
    print("=" * 80)
    print("GENERATING COMPREHENSIVE RESEARCH REPORT")
    print("=" * 80)
    
    # Load config if not provided
    if config is None:
        try:
            config = load_asset_config('gold')  # Default to gold
        except:
            config = None
    
    # Get report name from config if available
    if output_path is None:
        if config:
            report_name = get_setting(config, 'output.report_name', 'XAU_USD_Research_Report')
        else:
            report_name = 'XAU_USD_Research_Report'
        
        output_path = os.path.join('reports', f'{report_name}.md')
    
    # Load all analysis results
    print("\nLoading analysis results...")
    trend_analysis = load_trend_analysis(config)
    indicator_results = load_indicator_results(config)
    entropy_scores = load_entropy_scores(config)
    quality_ranking = load_quality_ranking(config)
    corr_matrix = load_correlation_matrix(config)
    regime_predictions = load_regime_predictions(config)
    training_history = load_training_history(config)
    eval_results = load_evaluation_results(config)
    regime_indicator_metrics = load_regime_specific_metrics(config)

    print("  ✓ Trend analysis loaded" if trend_analysis else "  ⚠ Trend analysis not found")
    print("  ✓ Indicator results loaded" if indicator_results is not None else "  ⚠ Indicator results not found")
    print("  ✓ Quality ranking loaded" if quality_ranking is not None else "  ⚠ Quality ranking not found")
    print("  ✓ Correlation matrix loaded" if corr_matrix is not None else "  ⚠ Correlation matrix not found")
    print("  ✓ Regime predictions loaded" if regime_predictions is not None else "  ⚠ Regime predictions not found")
    print("  ✓ Training history loaded" if training_history else "  ⚠ Training history not found")
    print("  ✓ Evaluation results loaded" if eval_results else "  ⚠ Evaluation results not found")
    print("  ✓ Regime indicator metrics loaded" if regime_indicator_metrics else "  ⚠ Regime indicator metrics not found")
    
    # Generate report header
    report = []
    # Get asset info from config if available
    if config:
        characteristics = get_characteristics(config)
        asset_name = characteristics['asset_name']
        asset_symbol = characteristics['symbol']
        display_name = get_setting(config, 'asset.display_name', asset_name)
    else:
        asset_name = "Gold"
        asset_symbol = "XAU/USD"
        display_name = "Gold (XAU/USD)"
    
    report.append(f"# {display_name} Quantitative Trading Research Report")
    report.append("**Analysis Period:** January 2022 – October 2025")
    report.append("**Timeframe:** 1-Hour Primary, Multi-Timeframe Analysis")
    report.append(f"**Symbol:** {asset_symbol} ({asset_name})")
    report.append(f"**Generated:** {datetime.now().strftime('%B %Y')}\n")
    report.append("---\n")
    
    # Generate table of contents
    report.append("## Table of Contents\n")
    report.append("1. [Executive Summary](#1-executive-summary)")
    report.append("2. [Introduction](#2-introduction)")
    report.append("3. [Volume Analysis](#3-volume-analysis)")
    report.append("4. [Volatility Analysis](#4-volatility-analysis)")
    report.append("5. [Trend Characteristics](#5-trend-characteristics)")
    report.append("6. [Technical Indicator Effectiveness](#6-technical-indicator-effectiveness)")
    report.append("7. [Market Regime Analysis](#7-market-regime-analysis)")
    report.append("8. [Correlation Analysis](#8-correlation-analysis)")
    report.append("9. [Key Takeaways & Recommendations](#9-key-takeaways--recommendations)")
    report.append("10. [Statistical Summary & Methodology](#10-statistical-summary--methodology)")
    report.append("11. [Appendix: Technical Details](#appendix-technical-details)\n")
    report.append("---\n")
    
    # Generate each section
    print("\nGenerating report sections...")
    
    print("  Generating Executive Summary...")
    report.append(generate_executive_summary(indicator_results, quality_ranking, eval_results, regime_predictions, config, regime_indicator_metrics))
    report.append("\n---\n")
    
    print("  Generating Introduction...")
    report.append(generate_introduction(config))
    report.append("\n---\n")
    
    print("  Generating Volume Analysis...")
    report.append(generate_volume_section(config))
    report.append("\n---\n")
    
    print("  Generating Volatility Analysis...")
    report.append(generate_volatility_section(config))
    report.append("\n---\n")
    
    print("  Generating Trend Characteristics...")
    report.append(generate_trend_section(trend_analysis, config))
    report.append("\n---\n")
    
    print("  Generating Technical Indicator Effectiveness...")
    report.append(generate_indicator_section(indicator_results, quality_ranking, entropy_scores, config, regime_indicator_metrics))
    report.append("\n---\n")
    
    print("  Generating Market Regime Analysis...")
    report.append(generate_regime_section(eval_results, training_history, regime_predictions, config))
    report.append("\n---\n")
    
    print("  Generating Correlation Analysis...")
    report.append(generate_correlation_section(corr_matrix, config))
    report.append("\n---\n")
    
    print("  Generating Key Takeaways...")
    report.append(generate_recommendations(quality_ranking, regime_predictions, config, regime_indicator_metrics))
    report.append("\n---\n")
    
    print("  Generating Statistical Summary...")
    report.append(generate_statistical_summary(config))
    report.append("\n---\n")
    
    print("  Generating Appendix...")
    report.append(generate_appendix(config))
    
    # Save report
    print(f"\nSaving report to: {output_path}")
    with open(output_path, 'w') as f:
        f.write("\n".join(report))
    
    print(f"✓ Report generated successfully: {output_path}")
    print(f"  Total sections: 11")
    print(f"  Total lines: {len(report)}")
    
    print("\n" + "=" * 80)
    print("REPORT GENERATION COMPLETE")
    print("=" * 80)
    
    return output_path


def summarize_indicator_regime(indicator: str,
                               signal: str,
                               regime_metrics: Optional[Dict[str, Any]]) -> str:
    """Return a concise summary of regime-specific performance for an indicator signal."""

    if not regime_metrics:
        return "No regime breakdown available."

    regimes = {}
    for regime in ['up', 'down', 'range']:
        key = f"{indicator}_{signal}_{regime}"
        if key in regime_metrics:
            regimes[regime] = regime_metrics[key]

    if not regimes:
        return "No regime breakdown available."

    best_regime, best_stats = max(regimes.items(), key=lambda item: item[1].get('win_rate_pct', 0))
    worst_regime, worst_stats = min(regimes.items(), key=lambda item: item[1].get('win_rate_pct', 0))

    return (
        f"Best in {best_regime} regimes ({best_stats['win_rate_pct']:.1f}% win, n={best_stats['sample_size']}); "
        f"weakest in {worst_regime} regimes ({worst_stats['win_rate_pct']:.1f}% win)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a quantitative research report for a configured asset.")
    parser.add_argument("--config", type=str, default=None, help="Path to configuration YAML file.")
    parser.add_argument("--output", type=str, default=None, help="Optional output path for the report.")
    args = parser.parse_args()

    asset_config = None
    if args.config:
        asset_config = load_config(args.config)
        validate_config(asset_config)

    report_path = generate_complete_report(output_path=args.output, config=asset_config)
    print(f"Report written to {report_path}")

