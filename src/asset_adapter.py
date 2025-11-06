"""
Asset Adapter Module for Context-Aware Report Generation

This module provides asset-specific characteristics and interpretations
that adapt report content based on the asset being analyzed (Gold, Bitcoin, Stocks, etc.).

Functions:
- get_characteristics: Get asset-specific characteristics
- generate_volume_interpretation: Contextualize volume patterns
- generate_trend_interpretation: Contextualize trend statistics
- generate_correlation_interpretation: Contextualize correlation values
- generate_volatility_interpretation: Contextualize volatility patterns
"""

import sys
import os
from typing import Dict, Any, List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config_manager import get_setting


def get_characteristics(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Get asset-specific characteristics based on config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with asset characteristics:
        - trading_hours: When asset trades
        - typical_volatility: Expected volatility range
        - trend_behavior: How asset trends
        - volume_pattern: Volume concentration patterns
        - correlation_context: Correlation behavior context
    """
    asset_name = get_setting(config, 'asset.name', 'Asset')
    market_type = get_setting(config, 'asset.market', 'unknown')
    symbol = get_setting(config, 'asset.symbol', '')
    
    # Base characteristics by market type
    characteristics = {
        'asset_name': asset_name,
        'market_type': market_type,
        'symbol': symbol
    }
    
    # Commodity (Gold, Silver, etc.)
    if market_type == 'commodity':
        characteristics.update({
            'trading_hours': '24/7 (London-New York overlap is most active)',
            'typical_volatility': 'moderate (0.8-1.2% daily)',
            'trend_behavior': 'subtle trends, mean-reverting characteristics',
            'volume_pattern': 'peaks during US/London session overlap (13:00-21:00 UTC)',
            'correlation_context': 'safe-haven asset, inversely correlated with risk during market stress',
            'regime_characteristics': 'tends to range more than trend, with gradual regime transitions'
        })
    
    # Cryptocurrency (Bitcoin, Ethereum, etc.)
    elif market_type == 'cryptocurrency':
        characteristics.update({
            'trading_hours': '24/7 (decentralized market)',
            'typical_volatility': 'high (2-4% daily, can exceed 5% during volatility spikes)',
            'trend_behavior': 'strong momentum, trending asset with extended moves',
            'volume_pattern': 'peaks during US hours (13:00-21:00 UTC) despite 24/7 market, reflects institutional participation',
            'correlation_context': 'risk-on asset, correlates with tech stocks and risk sentiment',
            'regime_characteristics': 'exhibits strong trending behavior with longer regime persistence'
        })
    
    # Stock (AAPL, MSFT, etc.)
    elif market_type == 'stock':
        characteristics.update({
            'trading_hours': 'Market hours (9:30-16:00 EST, 14:30-21:00 UTC)',
            'typical_volatility': 'moderate (0.5-1.0% daily, higher during earnings/events)',
            'trend_behavior': 'follows sector and market trends, influenced by earnings and news',
            'volume_pattern': 'peaks at market open (9:30-10:00 EST) and close (15:30-16:00 EST)',
            'correlation_context': 'sector exposure, follows sector ETFs (QQQ for tech, SPY for broad market)',
            'regime_characteristics': 'trends align with market cycles and sector rotation'
        })
    
    # Forex (EUR/USD, GBP/USD, etc.)
    elif market_type == 'forex':
        characteristics.update({
            'trading_hours': '24/5 (Monday-Friday, closes Friday evening)',
            'typical_volatility': 'moderate (0.5-1.5% daily, higher during news events)',
            'trend_behavior': 'currency pair trends driven by interest rate differentials and economic data',
            'volume_pattern': 'peaks during London (8:00-16:00 UTC) and New York (13:00-21:00 UTC) sessions',
            'correlation_context': 'correlates with related currency pairs and economic indicators',
            'regime_characteristics': 'trends persist based on central bank policy and economic cycles'
        })
    
    # Default/Unknown
    else:
        characteristics.update({
            'trading_hours': '24/7 or market hours (varies by asset)',
            'typical_volatility': 'moderate (varies by asset type)',
            'trend_behavior': 'varies by asset type',
            'volume_pattern': 'varies by asset type',
            'correlation_context': 'varies by asset type',
            'regime_characteristics': 'varies by asset type'
        })
    
    return characteristics


def generate_volume_interpretation(volume_stats: Dict[str, Any], 
                                   config: Dict[str, Any]) -> str:
    """
    Generate asset-specific volume pattern interpretation.
    
    Args:
        volume_stats: Dictionary with volume statistics:
            - peak_hours: List of peak hours (UTC)
            - peak_multiplier: Volume multiplier at peak vs low
            - total_samples: Number of samples
        config: Configuration dictionary
        
    Returns:
        Formatted text interpretation
    """
    characteristics = get_characteristics(config)
    asset_name = characteristics['asset_name']
    market_type = characteristics['market_type']
    
    peak_hours = volume_stats.get('peak_hours', [])
    peak_multiplier = volume_stats.get('peak_multiplier', 1.0)
    total_samples = volume_stats.get('total_samples', 0)
    
    # Format peak hours
    if peak_hours:
        if len(peak_hours) == 1:
            peak_str = f"{peak_hours[0]:02d}:00"
        elif len(peak_hours) == 2:
            peak_str = f"{peak_hours[0]:02d}:00-{peak_hours[1]:02d}:00"
        else:
            peak_str = f"{peak_hours[0]:02d}:00-{peak_hours[-1]:02d}:00"
    else:
        peak_str = "13:00-21:00"  # Default
    
    # Generate interpretation based on market type
    if market_type == 'cryptocurrency':
        interpretation = (
            f"{asset_name}, despite being a decentralized 24/7 market, exhibits "
            f"clear volume concentration during US trading hours ({peak_str} UTC) "
            f"with {peak_multiplier:.1f}x higher activity than Asian overnight hours. "
            f"This pattern reflects institutional participation and US retail trader "
            f"dominance in crypto markets. Analysis of {total_samples:,} hourly bars "
            f"reveals consistent intraday patterns despite the market's continuous operation."
        )
    
    elif market_type == 'commodity':
        interpretation = (
            f"{asset_name} trading peaks during US market hours ({peak_str} UTC) "
            f"with {peak_multiplier:.1f}x higher volume than quiet periods. "
            f"This concentration reflects London-New York session overlap where "
            f"the majority of {asset_name.lower()} trading occurs. Analysis of "
            f"{total_samples:,} hourly bars shows consistent intraday patterns "
            f"driven by institutional and central bank activity."
        )
    
    elif market_type == 'stock':
        interpretation = (
            f"{asset_name} exhibits typical equity volume patterns with peaks "
            f"at market open (9:30-10:00 EST) and close (15:30-16:00 EST). "
            f"Volume during active hours ({peak_str} UTC) is {peak_multiplier:.1f}x "
            f"higher than overnight periods. Analysis of {total_samples:,} hourly bars "
            f"shows predictable patterns driven by institutional trading and retail participation."
        )
    
    elif market_type == 'forex':
        interpretation = (
            f"{asset_name} volume peaks during London ({peak_str} UTC) and New York "
            f"trading sessions with {peak_multiplier:.1f}x higher activity than "
            f"Asian session. This pattern reflects the concentration of major market "
            f"participants in these regions. Analysis of {total_samples:,} hourly bars "
            f"reveals consistent session-based volume patterns."
        )
    
    else:
        interpretation = (
            f"{asset_name} exhibits volume concentration during {peak_str} UTC "
            f"with {peak_multiplier:.1f}x higher activity than low-volume periods. "
            f"Analysis of {total_samples:,} hourly bars reveals consistent intraday patterns."
        )
    
    return interpretation


def generate_trend_interpretation(trend_stats: Dict[str, Any],
                                 config: Dict[str, Any]) -> str:
    """
    Generate asset-specific trend interpretation.
    
    Args:
        trend_stats: Dictionary with trend statistics:
            - uptrend_duration: Mean uptrend duration (hours)
            - downtrend_duration: Mean downtrend duration (hours)
            - uptrend_count: Number of uptrends
            - downtrend_count: Number of downtrends
        config: Configuration dictionary
        
    Returns:
        Formatted text interpretation
    """
    characteristics = get_characteristics(config)
    asset_name = characteristics['asset_name']
    market_type = characteristics['market_type']
    
    uptrend_duration = trend_stats.get('uptrend_duration', 0)
    downtrend_duration = trend_stats.get('downtrend_duration', 0)
    uptrend_count = trend_stats.get('uptrend_count', 0)
    downtrend_count = trend_stats.get('downtrend_count', 0)
    
    # Convert hours to days
    uptrend_days = uptrend_duration / 24 if uptrend_duration > 0 else 0
    downtrend_days = downtrend_duration / 24 if downtrend_duration > 0 else 0
    
    # Generate interpretation based on market type
    if market_type == 'cryptocurrency':
        interpretation = (
            f"{asset_name} uptrends persist {uptrend_days:.1f} days on average "
            f"(n={uptrend_count}), significantly longer than traditional assets "
            f"due to strong momentum effects and retail FOMO behavior characteristic "
            f"of cryptocurrency markets. Downtrends average {downtrend_days:.1f} days "
            f"(n={downtrend_count}), reflecting the asset's tendency for extended moves "
            f"in both directions. This extended trend persistence makes momentum-based "
            f"strategies particularly effective for {asset_name.lower()}."
        )
    
    elif market_type == 'commodity':
        interpretation = (
            f"{asset_name} uptrends persist {uptrend_days:.1f} days on average "
            f"(n={uptrend_count}), moderate duration reflecting the precious metal's "
            f"balanced safe-haven and speculative characteristics. Downtrends average "
            f"{downtrend_days:.1f} days (n={downtrend_count}). The relatively balanced "
            f"trend durations suggest {asset_name.lower()} exhibits both trending and "
            f"mean-reverting behavior, making regime-aware strategies important for "
            f"optimal signal selection."
        )
    
    elif market_type == 'stock':
        interpretation = (
            f"{asset_name} uptrends persist {uptrend_days:.1f} days on average "
            f"(n={uptrend_count}), aligned with typical equity trend cycles driven "
            f"by earnings momentum and sector rotation. Downtrends average "
            f"{downtrend_days:.1f} days (n={downtrend_count}). These durations reflect "
            f"the influence of quarterly earnings cycles, analyst upgrades/downgrades, "
            f"and broader market sentiment on {asset_name.lower()} price action."
        )
    
    elif market_type == 'forex':
        interpretation = (
            f"{asset_name} uptrends persist {uptrend_days:.1f} days on average "
            f"(n={uptrend_count}), reflecting currency pair trends driven by interest "
            f"rate differentials and economic data releases. Downtrends average "
            f"{downtrend_days:.1f} days (n={downtrend_count}). Trend persistence in "
            f"forex markets is influenced by central bank policy cycles and major "
            f"economic indicators, making fundamental context important for trend interpretation."
        )
    
    else:
        interpretation = (
            f"{asset_name} uptrends persist {uptrend_days:.1f} days on average "
            f"(n={uptrend_count}), while downtrends average {downtrend_days:.1f} days "
            f"(n={downtrend_count}). These durations reflect the asset's trend "
            f"characteristics and market structure."
        )
    
    return interpretation


def generate_correlation_interpretation(correlation_value: float,
                                       asset1: str,
                                       asset2: str,
                                       p_value: float = None,
                                       config: Dict[str, Any] = None) -> str:
    """
    Generate asset-specific correlation interpretation.
    
    Args:
        correlation_value: Correlation coefficient (-1 to 1)
        asset1: First asset name/symbol
        asset2: Second asset name/symbol
        p_value: Statistical significance p-value (optional)
        config: Configuration dictionary (optional)
        
    Returns:
        Formatted text interpretation
    """
    abs_corr = abs(correlation_value)
    
    # Determine correlation strength
    if abs_corr > 0.7:
        strength = "strong"
    elif abs_corr > 0.4:
        strength = "moderate"
    elif abs_corr > 0.2:
        strength = "weak"
    else:
        strength = "negligible"
    
    # Determine direction
    direction = "positive" if correlation_value > 0 else "negative"
    
    # Significance text
    sig_text = ""
    if p_value is not None:
        if p_value < 0.001:
            sig_text = " (highly significant, p<0.001)"
        elif p_value < 0.01:
            sig_text = " (significant, p<0.01)"
        elif p_value < 0.05:
            sig_text = " (marginally significant, p<0.05)"
        else:
            sig_text = " (not significant, p>0.05)"
    
    # Generate interpretation based on asset types
    if config:
        characteristics = get_characteristics(config)
        market_type = characteristics.get('market_type', 'unknown')
    else:
        market_type = 'unknown'
    
    # Crypto-crypto correlation
    if 'bitcoin' in asset1.lower() or 'btc' in asset1.lower() or \
       'ethereum' in asset1.lower() or 'eth' in asset1.lower() or \
       'crypto' in asset1.lower():
        if 'bitcoin' in asset2.lower() or 'btc' in asset2.lower() or \
           'ethereum' in asset2.lower() or 'eth' in asset2.lower() or \
           'crypto' in asset2.lower():
            interpretation = (
                f"{asset1} shows {strength} {direction} correlation with {asset2} "
                f"({correlation_value:.2f}{sig_text}), expected for cryptocurrencies "
                f"sharing market infrastructure, regulatory environment, and investor base. "
                f"Both assets react similarly to crypto-specific news and DeFi developments."
            )
            return interpretation
    
    # Commodity-stock correlation
    if ('gold' in asset1.lower() or 'xau' in asset1.lower()) and \
       ('spy' in asset2.lower() or 'stock' in asset2.lower() or 'qqq' in asset2.lower()):
        if abs_corr < 0.3:
            interpretation = (
                f"{asset1} shows {strength} correlation with {asset2} "
                f"({correlation_value:.2f}{sig_text}), suggesting {asset1.lower()} "
                f"moves independently of equity markets on daily timeframe. "
                f"This independence supports {asset1.lower()}'s role as a portfolio diversifier."
            )
        else:
            interpretation = (
                f"{asset1} shows {strength} {direction} correlation with {asset2} "
                f"({correlation_value:.2f}{sig_text}). This relationship varies with "
                f"market conditions, with correlation typically increasing during "
                f"market stress periods when both assets react to risk sentiment."
            )
        return interpretation
    
    # Commodity-currency correlation
    if ('gold' in asset1.lower() or 'xau' in asset1.lower()) and \
       ('eur' in asset2.lower() or 'usd' in asset2.lower() or 'currency' in asset2.lower()):
        if abs_corr < 0.2:
            interpretation = (
                f"{asset1} shows no meaningful correlation with {asset2} "
                f"({correlation_value:.2f}{sig_text}), suggesting {asset1.lower()} "
                f"moves independently of currency markets on daily timeframe. "
                f"This independence supports {asset1.lower()}'s role as a currency hedge."
            )
        else:
            interpretation = (
                f"{asset1} shows {strength} {direction} correlation with {asset2} "
                f"({correlation_value:.2f}{sig_text}), reflecting the relationship "
                f"between precious metals and currency strength, particularly the US Dollar."
            )
        return interpretation
    
    # Default interpretation
    if abs_corr < 0.2:
        interpretation = (
            f"{asset1} shows {strength} correlation with {asset2} "
            f"({correlation_value:.2f}{sig_text}), suggesting the assets move "
            f"independently on daily timeframe."
        )
    else:
        interpretation = (
            f"{asset1} shows {strength} {direction} correlation with {asset2} "
            f"({correlation_value:.2f}{sig_text}), indicating the assets share "
            f"common drivers or market structure."
        )
    
    return interpretation


def generate_volatility_interpretation(volatility_stats: Dict[str, Any],
                                      config: Dict[str, Any]) -> str:
    """
    Generate asset-specific volatility interpretation.
    
    Args:
        volatility_stats: Dictionary with volatility statistics:
            - mean_atr: Mean ATR value
            - mean_atr_pct: Mean ATR as percentage of price
            - volatility_clustering: Whether volatility clusters (bool)
            - autocorrelation: Volatility autocorrelation value
        config: Configuration dictionary
        
    Returns:
        Formatted text interpretation
    """
    characteristics = get_characteristics(config)
    asset_name = characteristics['asset_name']
    market_type = characteristics['market_type']
    
    mean_atr_pct = volatility_stats.get('mean_atr_pct', 0)
    volatility_clustering = volatility_stats.get('volatility_clustering', False)
    autocorrelation = volatility_stats.get('autocorrelation', 0)
    
    # Generate interpretation based on market type
    if market_type == 'cryptocurrency':
        interpretation = (
            f"{asset_name} exhibits high volatility with average ATR of "
            f"{mean_atr_pct:.2f}% of price, characteristic of cryptocurrency markets. "
        )
        if volatility_clustering:
            interpretation += (
                f"Volatility demonstrates strong clustering (autocorrelation={autocorrelation:.2f}), "
                f"with high volatility periods followed by high volatility, and low volatility "
                f"periods followed by low volatility. This pattern reflects the momentum-driven "
                f"nature of crypto markets where news and sentiment create extended volatility regimes."
            )
        else:
            interpretation += (
                f"Volatility patterns show moderate persistence, reflecting the 24/7 nature "
                f"of crypto markets where volatility can spike at any time."
            )
    
    elif market_type == 'commodity':
        interpretation = (
            f"{asset_name} exhibits moderate volatility with average ATR of "
            f"{mean_atr_pct:.2f}% of price, typical for precious metals. "
        )
        if volatility_clustering:
            interpretation += (
                f"Volatility demonstrates clustering (autocorrelation={autocorrelation:.2f}), "
                f"with volatility regimes persisting for several days. This pattern reflects "
                f"the influence of macroeconomic events, central bank announcements, and "
                f"geopolitical developments on {asset_name.lower()} price action."
            )
        else:
            interpretation += (
                f"Volatility shows moderate persistence, reflecting the influence of "
                f"scheduled economic releases and central bank policy announcements."
            )
    
    elif market_type == 'stock':
        interpretation = (
            f"{asset_name} exhibits moderate volatility with average ATR of "
            f"{mean_atr_pct:.2f}% of price, typical for equity markets. "
        )
        if volatility_clustering:
            interpretation += (
                f"Volatility demonstrates clustering (autocorrelation={autocorrelation:.2f}), "
                f"with volatility spikes around earnings announcements, analyst updates, and "
                f"market-wide events. This pattern reflects the scheduled nature of corporate "
                f"information releases and their impact on {asset_name.lower()} price action."
            )
        else:
            interpretation += (
                f"Volatility shows moderate persistence, with spikes around earnings "
                f"announcements and sector-specific news events."
            )
    
    else:
        interpretation = (
            f"{asset_name} exhibits volatility with average ATR of "
            f"{mean_atr_pct:.2f}% of price. "
        )
        if volatility_clustering:
            interpretation += (
                f"Volatility demonstrates clustering (autocorrelation={autocorrelation:.2f}), "
                f"indicating volatility regimes persist over time."
            )
    
    return interpretation


def generate_regime_interpretation(regime_stats: Dict[str, Any],
                                  config: Dict[str, Any]) -> str:
    """
    Generate asset-specific regime interpretation.
    
    Args:
        regime_stats: Dictionary with regime statistics:
            - current_regime: Current regime label
            - confidence: Model confidence
            - regime_distribution: Distribution of regimes
        config: Configuration dictionary
        
    Returns:
        Formatted text interpretation
    """
    characteristics = get_characteristics(config)
    asset_name = characteristics['asset_name']
    market_type = characteristics['market_type']
    
    current_regime = regime_stats.get('current_regime', 'unknown')
    confidence = regime_stats.get('confidence', 0)
    regime_dist = regime_stats.get('regime_distribution', {})
    
    # Generate interpretation
    interpretation = (
        f"Current {asset_name} regime: {current_regime.upper()} "
        f"(model confidence: {confidence:.1f}%). "
    )
    
    if market_type == 'cryptocurrency':
        interpretation += (
            f"Given {asset_name.lower()}'s strong momentum characteristics, "
            f"trending regimes tend to persist longer than in traditional assets. "
        )
    elif market_type == 'commodity':
        interpretation += (
            f"Given {asset_name.lower()}'s balanced trending and mean-reverting behavior, "
            f"regime transitions are typically gradual. "
        )
    elif market_type == 'stock':
        interpretation += (
            f"Given {asset_name.lower()}'s alignment with market cycles, "
            f"regime changes often coincide with earnings cycles and sector rotation. "
        )
    
    # Add distribution context
    if regime_dist:
        range_pct = regime_dist.get('range', 0)
        up_pct = regime_dist.get('up', 0)
        down_pct = regime_dist.get('down', 0)
        
        interpretation += (
            f"Historical distribution: {range_pct:.1f}% ranging, "
            f"{up_pct:.1f}% uptrending, {down_pct:.1f}% downtrending."
        )
    
    return interpretation


def get_indicator_recommendations(config: Dict[str, Any],
                                  indicator_results: Dict[str, Any]) -> List[str]:
    """
    Generate asset-specific indicator recommendations.
    
    Args:
        config: Configuration dictionary
        indicator_results: Dictionary with indicator test results
        config: Configuration dictionary
        
    Returns:
        List of recommendation strings
    """
    characteristics = get_characteristics(config)
    asset_name = characteristics['asset_name']
    market_type = characteristics['market_type']
    
    recommendations = []
    
    # Get top indicators from results
    # This would need to be implemented based on actual indicator_results structure
    # For now, return generic recommendations
    
    if market_type == 'cryptocurrency':
        recommendations.append(
            f"For {asset_name}, momentum-based indicators (MACD, SMA crossovers) "
            f"tend to perform well due to the asset's strong trending characteristics."
        )
        recommendations.append(
            f"RSI signals should be used with caution as {asset_name.lower()} can "
            f"remain overbought/oversold for extended periods during strong trends."
        )
    
    elif market_type == 'commodity':
        recommendations.append(
            f"For {asset_name}, mean-reversion indicators (VWAP, SMA bounces) "
            f"work well during ranging regimes, while trend-following indicators "
            f"perform better during trending periods."
        )
        recommendations.append(
            f"Regime filtering is critical for {asset_name.lower()} as indicator "
            f"effectiveness varies significantly by market regime."
        )
    
    elif market_type == 'stock':
        recommendations.append(
            f"For {asset_name}, sector and market context is important. "
            f"Indicators should be combined with earnings calendar awareness."
        )
        recommendations.append(
            f"Volume confirmation is particularly important for {asset_name.lower()} "
            f"signals given the influence of institutional trading."
        )
    
    return recommendations

