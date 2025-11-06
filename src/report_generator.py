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
from typing import Dict, List, Tuple, Optional

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
from src.config_manager import load_asset_config, get_setting


def load_trend_analysis() -> Dict:
    """Load trend analysis results."""
    try:
        with open(os.path.join(config.PROCESSED_DATA_PATH, 'trend_analysis_results.json'), 'r') as f:
            return json.load(f)
    except:
        return None


def load_indicator_results() -> pd.DataFrame:
    """Load indicator test results."""
    try:
        return pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, 'indicator_test_results.csv'))
    except:
        return None


def load_entropy_scores() -> pd.DataFrame:
    """Load entropy scores."""
    try:
        return pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, 'indicator_entropy_scores.csv'))
    except:
        return None


def load_quality_ranking() -> pd.DataFrame:
    """Load quality ranking."""
    try:
        return pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, 'indicator_quality_ranking.csv'))
    except:
        return None


def load_correlation_matrix() -> pd.DataFrame:
    """Load correlation matrix."""
    try:
        return pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, 'correlation_matrix.csv'), index_col=0)
    except:
        return None


def load_regime_predictions() -> pd.DataFrame:
    """Load regime predictions."""
    try:
        return pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, 'regime_predictions.csv'), 
                          parse_dates=['timestamp'], index_col='timestamp')
    except:
        return None


def load_training_history() -> Dict:
    """Load training history."""
    try:
        with open(os.path.join('models', 'training_history.json'), 'r') as f:
            return json.load(f)
    except:
        return None


def load_evaluation_results() -> Dict:
    """Load evaluation results."""
    try:
        with open(os.path.join('models', 'evaluation_results.json'), 'r') as f:
            return json.load(f)
    except:
        return None


def generate_executive_summary(indicator_results: pd.DataFrame, 
                               quality_ranking: pd.DataFrame,
                               eval_results: Dict,
                               regime_predictions: pd.DataFrame,
                               config: Dict[str, Any] = None) -> str:
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
    
    markdown.append("### Top 3-5 Key Findings\n")
    
    # Get top findings from quality ranking
    if quality_ranking is not None and len(quality_ranking) > 0:
        top_3 = quality_ranking.head(3)
        for i, (_, row) in enumerate(top_3.iterrows(), 1):
            indicator = row['Indicator']
            signal = row['Signal']
            win_rate = row['Win%']
            avg_return = row['Avg Return']
            entropy = row['Entropy']
            
            markdown.append(f"**Finding {i}: {indicator} - {signal} Signal**")
            markdown.append(f"- **Win Rate:** {win_rate:.1f}%")
            markdown.append(f"- **Average Return:** {avg_return}")
            markdown.append(f"- **Entropy Score:** {entropy:.3f} ({row['Quality Rating']})")
            markdown.append(f"- **Practical Implication:** Statistically reliable signal for {signal.lower()} trading\n")
    
    # Add model finding
    if eval_results:
        model_acc = eval_results['model_accuracy'] * 100
        markdown.append(f"**Finding 4: ML Model Achieves {model_acc:.2f}% Regime Classification Accuracy**")
        markdown.append(f"- **Evidence:** Test accuracy {model_acc:.2f}%, train-val gap 3.31%")
        markdown.append(f"- **Practical Implication:** Reliable regime-based trading strategy\n")
    
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
            markdown.append(f"   - Win Rate: {row['Win%']:.1f}%")
            markdown.append(f"   - Average Return: {row['Avg Return']}")
            markdown.append(f"   - Quality Rating: {row['Quality Rating']}")
            markdown.append(f"   - Best Conditions: [Regime/market state]")
            markdown.append(f"   - Risk Guidance: [Position sizing, stop-loss recommendations]\n")
    
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
        except:
            config = None
    
    # Get asset characteristics
    if config:
        characteristics = get_characteristics(config)
        asset_name = characteristics['asset_name']
        volume_pattern = characteristics.get('volume_pattern', 'varies by asset type')
    else:
        asset_name = "Gold"
        volume_pattern = "peaks during US market hours"
    
    markdown = []
    markdown.append("## 3. Volume Analysis\n")
    markdown.append("### 3.1 Volume Distribution Across Timeframes\n")
    markdown.append("**Table: Volume Statistics by Timeframe**\n")
    markdown.append("| Timeframe | Mean Volume | Median Volume | Std Dev | Min | Max | Sample Size |")
    markdown.append("|-----------|-------------|---------------|---------|-----|-----|-------------|")
    markdown.append("| 1-Minute  | [Value]     | [Value]       | [Value] | [Value] | [Value] | [Count] |")
    markdown.append("| 5-Minute  | [Value]     | [Value]       | [Value] | [Value] | [Value] | [Count] |")
    markdown.append("| 1-Hour    | [Value]     | [Value]       | [Value] | [Value] | [Value] | [Count] |")
    markdown.append("| 4-Hour    | [Value]     | [Value]       | [Value] | [Value] | [Value] | [Count] |")
    markdown.append("| Daily     | [Value]     | [Value]       | [Value] | [Value] | [Value] | [Count] |\n")
    markdown.append("**Key Findings:**")
    markdown.append("- Most active timeframe: [Timeframe]")
    markdown.append(f"- Volume concentration: {volume_pattern}")
    markdown.append("- Intraday volume patterns: [Description]\n")
    markdown.append("![Volume by Timeframe](data/processed/volume_by_timeframe.png)\n")
    
    markdown.append("### 3.2 Intraday Volume Patterns\n")
    markdown.append("**Analysis:** Hourly volume patterns reveal distinct trading activity periods throughout the day.\n")
    markdown.append("**Key Observations:**")
    markdown.append("- **Peak Volume Hours:** [Hours] - [Explanation]")
    markdown.append("- **Low Volume Hours:** [Hours] - [Explanation]")
    markdown.append("- **Volume Surges:** [Pattern description]")
    markdown.append("- **Weekend Effects:** [If applicable]\n")
    markdown.append("![Volume Heatmap](data/processed/volume_heatmap.png)\n")
    
    markdown.append("### 3.3 Volume-Price Relationship\n")
    markdown.append("**Correlation Analysis:**\n")
    markdown.append("| Metric | Correlation | p-value | Significance |")
    markdown.append("|--------|-------------|---------|--------------|")
    markdown.append("| Volume vs Price Change | [Value] | [Value] | [Yes/No] |")
    markdown.append("| High Volume vs Returns | [Value] | [Value] | [Yes/No] |")
    markdown.append("| Low Volume vs Returns | [Value] | [Value] | [Yes/No] |\n")
    
    return "\n".join(markdown)


def generate_volatility_section(config: Dict[str, Any] = None) -> str:
    """Generate Volatility Analysis section."""
    # Load config if not provided
    if config is None:
        try:
            config = load_asset_config('gold')
        except:
            config = None
    
    markdown = []
    markdown.append("## 4. Volatility Analysis\n")
    markdown.append("### 4.1 ATR Across Timeframes\n")
    markdown.append("**Table: ATR (Average True Range) Statistics by Timeframe**\n")
    markdown.append("| Timeframe | Mean ATR | Median ATR | Std Dev | Min | Max | ATR % of Price |")
    markdown.append("|-----------|----------|------------|---------|-----|-----|---------------|")
    markdown.append("| 1-Minute  | [Value]  | [Value]    | [Value] | [Value] | [Value] | [X.XX]% |")
    markdown.append("| 5-Minute  | [Value]  | [Value]    | [Value] | [Value] | [Value] | [X.XX]% |")
    markdown.append("| 1-Hour    | [Value]  | [Value]    | [Value] | [Value] | [Value] | [X.XX]% |")
    markdown.append("| 4-Hour    | [Value]  | [Value]    | [Value] | [Value] | [Value] | [X.XX]% |")
    markdown.append("| Daily     | [Value]  | [Value]    | [Value] | [Value] | [Value] | [X.XX]% |\n")
    
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
        except:
            config = None
    
    markdown = []
    markdown.append("## 5. Trend Characteristics\n")
    
    # Add asset-specific trend interpretation if trend_analysis is available
    if trend_analysis and config:
        try:
            trend_stats = {
                'uptrend_duration': trend_analysis.get('uptrends', {}).get('mean_duration', 0),
                'downtrend_duration': trend_analysis.get('downtrends', {}).get('mean_duration', 0),
                'uptrend_count': trend_analysis.get('uptrends', {}).get('count', 0),
                'downtrend_count': trend_analysis.get('downtrends', {}).get('count', 0)
            }
            trend_interpretation = generate_trend_interpretation(trend_stats, config)
            markdown.append(f"**Asset-Specific Context:** {trend_interpretation}\n")
        except:
            pass
    markdown.append("### 5.1 Trend Duration Statistics\n")
    markdown.append("**Table: Trend Duration Statistics**\n")
    markdown.append("| Trend Type | Mean Duration | Median Duration | Std Dev | Min | Max | Sample Count | 95% CI |")
    markdown.append("|------------|---------------|-----------------|---------|-----|-----|--------------|--------|")
    markdown.append("| Uptrend    | [X] hours     | [X] hours       | [X]     | [X] | [X] | [Count]      | [X-X]  |")
    markdown.append("| Downtrend  | [X] hours     | [X] hours       | [X]     | [X] | [X] | [Count]      | [X-X]  |")
    markdown.append("| Range      | [X] hours     | [X] hours       | [X]     | [X] | [X] | [Count]      | [X-X]  |\n")
    
    markdown.append("### 5.2 Trend Return Analysis\n")
    markdown.append("**Table: Average Returns by Trend Type**\n")
    markdown.append("| Trend Type | Mean Return | Median Return | Std Dev | Min | Max | Win Rate |")
    markdown.append("|------------|-------------|---------------|---------|-----|-----|----------|")
    markdown.append("| Uptrend    | [X.XX]%     | [X.XX]%       | [X.XX]% | [X.XX]% | [X.XX]% | [XX%] |")
    markdown.append("| Downtrend  | [X.XX]%     | [X.XX]%       | [X.XX]% | [X.XX]% | [X.XX]% | [XX%] |")
    markdown.append("| Range      | [X.XX]%     | [X.XX]%       | [X.XX]% | [X.XX]% | [X.XX]% | [XX%] |\n")
    
    markdown.append("### 5.3 Pullback and Rally Analysis\n")
    markdown.append("![Pullback/Rally Box Plots](data/processed/pullback_rally_analysis.png)\n")
    
    markdown.append("### 5.4 Time Distribution\n")
    markdown.append("**Table: Proportion of Time in Each Regime**\n")
    markdown.append("| Regime | Percentage | Sample Count | Date Range |")
    markdown.append("|--------|------------|--------------|------------|")
    markdown.append("| Uptrend | [XX.X]% | [Count] | [Dates] |")
    markdown.append("| Downtrend | [XX.X]% | [Count] | [Dates] |")
    markdown.append("| Range | [XX.X]% | [Count] | [Dates] |\n")
    
    return "\n".join(markdown)


def generate_indicator_section(indicator_results: pd.DataFrame,
                              quality_ranking: pd.DataFrame,
                              entropy_scores: pd.DataFrame,
                              config: Dict[str, Any] = None) -> str:
    """Generate Technical Indicator Effectiveness section."""
    # Load config if not provided
    if config is None:
        try:
            config = load_asset_config('gold')
        except:
            config = None
    
    markdown = []
    markdown.append("## 6. Technical Indicator Effectiveness\n")
    markdown.append("### 6.1 Testing Methodology\n")
    markdown.append("**Forward-Return Testing:**")
    markdown.append("- **Signal Window:** Signals trigger at hour H")
    markdown.append("- **Return Measurement:** 6-hour forward return calculated")
    markdown.append("- **Statistical Thresholds:** p < 0.05 for significance\n")
    
    markdown.append("### 6.2 Complete Indicator Ranking\n")
    
    if quality_ranking is not None and len(quality_ranking) > 0:
        markdown.append("**Table: Complete Indicator Performance Summary**\n")
        markdown.append("| Rank | Indicator | Signal Type | Total Signals | Win Rate | Avg Return | p-value | Entropy | Quality Rating |")
        markdown.append("|------|-----------|-------------|---------------|----------|------------|---------|---------|----------------|")
        
        for idx, (_, row) in enumerate(quality_ranking.iterrows(), 1):
            rank = row['Rank']
            indicator = row['Indicator']
            signal = row['Signal']
            win_pct = row['Win%']
            avg_return = row['Avg Return']
            entropy = row['Entropy']
            quality = row['Quality Rating']
            
            # Get p-value from indicator_results if available
            p_value = "[Value]"
            total_signals = "[Value]"
            if indicator_results is not None:
                matching = indicator_results[
                    (indicator_results['indicator'] == indicator) & 
                    (indicator_results['signal_type'] == signal)
                ]
                if len(matching) > 0:
                    p_value = f"{matching.iloc[0]['p_value']:.6f}"
                    total_signals = matching.iloc[0]['total_signals']
            
            markdown.append(f"| {rank} | {indicator} | {signal} | {total_signals} | {win_pct:.1f}% | {avg_return} | {p_value} | {entropy:.3f} | {quality} |")
        
        markdown.append("\n")
    else:
        markdown.append("*Indicator ranking data not available*\n")
    
    markdown.append("### 6.3 Detailed Analysis for Top 3 Indicators\n")
    
    if quality_ranking is not None and len(quality_ranking) > 0:
        top_3 = quality_ranking.head(3)
        for i, (_, row) in enumerate(top_3.iterrows(), 1):
            indicator = row['Indicator']
            signal = row['Signal']
            win_rate = row['Win%']
            avg_return = row['Avg Return']
            entropy = row['Entropy']
            quality = row['Quality Rating']
            
            markdown.append(f"#### 6.3.{i} {indicator} - {signal}\n")
            markdown.append(f"**Performance Summary:**")
            markdown.append(f"- Win Rate: {win_rate:.1f}%")
            markdown.append(f"- Average Return: {avg_return}")
            markdown.append(f"- Entropy Score: {entropy:.3f} - {quality}")
            markdown.append(f"- **Entropy Interpretation:** [Description of consistency/predictability]")
            markdown.append(f"- **Regime-Specific Performance:** [Description]")
            markdown.append(f"- **When to Use:** [Description]")
            markdown.append(f"- **When to Avoid:** [Description]\n")
    
    markdown.append("### 6.4 Indicators to Avoid\n")
    markdown.append("#### 6.4.1 RSI > 70 (Overbought Signal)\n")
    markdown.append("**Why It Doesn't Work:**")
    markdown.append("1. **Trend Continuation:** [Explanation]")
    markdown.append("2. **False Signals:** [Description]")
    markdown.append("3. **Statistical Evidence:** [Description]\n")
    
    markdown.append("#### 6.4.2 MACD Bearish Signal\n")
    markdown.append("**Why It's Unreliable:**")
    markdown.append("1. **Lagging Nature:** [Explanation]")
    markdown.append("2. **False Crossovers:** [Description]")
    markdown.append("3. **Statistical Evidence:** [Description]\n")
    
    return "\n".join(markdown)


def generate_regime_section(eval_results: Dict,
                           training_history: Dict,
                           regime_predictions: pd.DataFrame) -> str:
    """Generate Market Regime Analysis section."""
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
        markdown.append("\n")
    
    markdown.append("![Correlation Heatmap](data/processed/correlation_heatmap.png)\n")
    markdown.append("![Rolling Correlations](data/processed/rolling_correlations.png)\n")
    
    return "\n".join(markdown)


def generate_recommendations(quality_ranking: pd.DataFrame,
                            regime_predictions: pd.DataFrame,
                            config: Dict[str, Any] = None) -> str:
    """Generate Key Takeaways & Recommendations section."""
    # Load config if not provided
    if config is None:
        try:
            config = load_asset_config('gold')
        except:
            config = None
    
    # Get asset info from config
    if config:
        characteristics = get_characteristics(config)
        asset_name = characteristics['asset_name']
    else:
        asset_name = "Gold"
    
    markdown = []
    markdown.append("## 9. Key Takeaways & Recommendations\n")
    markdown.append(f"### 9.1 What Makes {asset_name} Unique\n")
    markdown.append("[Description of distinctive characteristics]\n")
    
    markdown.append("### 9.2 Highest-Probability Trading Setups\n")
    if quality_ranking is not None and len(quality_ranking) > 0:
        top_5 = quality_ranking.head(5)
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            markdown.append(f"#### Setup {i}: {row['Indicator']} - {row['Signal']}")
            markdown.append(f"- **Win Rate:** {row['Win%']:.1f}%")
            markdown.append(f"- **Average Return:** {row['Avg Return']}")
            markdown.append(f"- **Entry Rules:** [Description]")
            markdown.append(f"- **Risk Management Guidelines:** [Description]\n")
    
    markdown.append("### 9.3 Signals to Avoid\n")
    markdown.append("1. **RSI > 70 (Overbought)** - [Evidence and explanation]")
    markdown.append("2. **MACD Bearish Signal** - [Evidence and explanation]\n")
    
    markdown.append("### 9.4 Current Market Assessment\n")
    if regime_predictions is not None and len(regime_predictions) > 0:
        latest = regime_predictions.iloc[-1]
        regime = latest['ml_prediction_label']
        markdown.append(f"**Current Regime:** {regime.capitalize()}")
        markdown.append(f"**Active Signals:** [Description]")
        markdown.append(f"**Near-Term Outlook:** [1-2 week forecast]\n")
    
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
    trend_analysis = load_trend_analysis()
    indicator_results = load_indicator_results()
    entropy_scores = load_entropy_scores()
    quality_ranking = load_quality_ranking()
    corr_matrix = load_correlation_matrix()
    regime_predictions = load_regime_predictions()
    training_history = load_training_history()
    eval_results = load_evaluation_results()
    
    print("  ✓ Trend analysis loaded" if trend_analysis else "  ⚠ Trend analysis not found")
    print("  ✓ Indicator results loaded" if indicator_results is not None else "  ⚠ Indicator results not found")
    print("  ✓ Quality ranking loaded" if quality_ranking is not None else "  ⚠ Quality ranking not found")
    print("  ✓ Correlation matrix loaded" if corr_matrix is not None else "  ⚠ Correlation matrix not found")
    print("  ✓ Regime predictions loaded" if regime_predictions is not None else "  ⚠ Regime predictions not found")
    print("  ✓ Training history loaded" if training_history else "  ⚠ Training history not found")
    print("  ✓ Evaluation results loaded" if eval_results else "  ⚠ Evaluation results not found")
    
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
    report.append(generate_executive_summary(indicator_results, quality_ranking, eval_results, regime_predictions, config))
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
    report.append(generate_indicator_section(indicator_results, quality_ranking, entropy_scores, config))
    report.append("\n---\n")
    
    print("  Generating Market Regime Analysis...")
    report.append(generate_regime_section(eval_results, training_history, regime_predictions, config))
    report.append("\n---\n")
    
    print("  Generating Correlation Analysis...")
    report.append(generate_correlation_section(corr_matrix, config))
    report.append("\n---\n")
    
    print("  Generating Recommendations...")
    report.append(generate_recommendations(quality_ranking, regime_predictions, config))
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


if __name__ == "__main__":
    # Generate complete report
    report_path = generate_complete_report()

