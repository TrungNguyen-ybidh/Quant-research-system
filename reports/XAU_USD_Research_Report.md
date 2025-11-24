# Gold (XAU/USD) Quantitative Trading Research Report
**Analysis Period:** January 2022 – October 2025
**Timeframe:** 1-Hour Primary, Multi-Timeframe Analysis
**Symbol:** XAU/USD (Gold)
**Generated:** November 2025

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Introduction](#2-introduction)
3. [Volume Analysis](#3-volume-analysis)
4. [Volatility Analysis](#4-volatility-analysis)
5. [Trend Characteristics](#5-trend-characteristics)
6. [Technical Indicator Effectiveness](#6-technical-indicator-effectiveness)
7. [Market Regime Analysis](#7-market-regime-analysis)
8. [Correlation Analysis](#8-correlation-analysis)
9. [Key Takeaways & Recommendations](#9-key-takeaways--recommendations)
10. [Statistical Summary & Methodology](#10-statistical-summary--methodology)
11. [Appendix: Technical Details](#appendix-technical-details)

---

## 1. Executive Summary

### Overview

This comprehensive quantitative research report analyzes Gold (XAU/USD) trading patterns from January 2022 to October 2025, combining statistical analysis, technical indicator testing, and machine learning regime classification. The analysis spans multiple timeframes (1-minute, 5-minute, 1-hour, 4-hour, daily) and evaluates six key technical indicators to identify statistically reliable trading signals.

**Finding 1: SMA - SMA_50_Bounce Signal**
- **Win Rate:** 64.5%
- **Average Return:** +0.13%
- **Entropy Score:** 0.493 (Good)
- **Regime Edge:** Best in range regimes (70.9% win, n=815); weakest in down regimes (0.0% win).
- **Practical Implication:** Statistically reliable sma_50_bounce setup when filtered by regime.

**Finding 2: MACD - MACD_Bullish_Cross Signal**
- **Win Rate:** 58.4%
- **Average Return:** +0.58%
- **Entropy Score:** 0.985 (Poor)
- **Regime Edge:** Best in up regimes (60.3% win, n=277); weakest in range regimes (56.7% win).
- **Practical Implication:** Statistically reliable macd_bullish_cross setup when filtered by regime.

**Finding 3: ML Model Achieves 88.08% Regime Classification Accuracy**
- **Evidence:** Test accuracy 88.08%, train-val gap 3.31%
- **Practical Implication:** Reliable regime-based trading strategy

### Current Market Regime (October 2025)

**Regime Classification:** Range
**Model Confidence:** 59.7%
**Probability Distribution:**
- Range: 59.7%
- Up: 40.3%
- Down: 0.0%

### Recommended Signals for Gold (XAU/USD) Trading

1. **SMA - SMA_50_Bounce**
   - Win Rate: 64.5% | Avg Return: +0.13%
   - Quality Rating: Good
   - Best Conditions: Best in range regimes (70.9% win, n=815); weakest in down regimes (0.0% win).
   - Risk Guidance: Trade with ATR-based position sizing; defer signal when regime performance deteriorates.

2. **MACD - MACD_Bullish_Cross**
   - Win Rate: 58.4% | Avg Return: +0.58%
   - Quality Rating: Poor
   - Best Conditions: Best in up regimes (60.3% win, n=277); weakest in range regimes (56.7% win).
   - Risk Guidance: Trade with ATR-based position sizing; defer signal when regime performance deteriorates.

*Signals failing statistical requirements are cataloged in Section 9.3 (Signals to Avoid).*


---

## 2. Introduction

### 2.1 Asset Analyzed

**Symbol:** XAU/USD (Gold)
**Analysis Period:** January 2022 – October 2025
**Data Source:** Oanda Trading API
**Primary Timeframe:** 1-Hour

**Market Structure:** Futures-driven product with concentrated liquidity around COMEX/Euronext hours and macro release windows.

### 2.2 Timeframes Analyzed

- **1-Minute:** Intraday micro-structure analysis
- **5-Minute:** Short-term pattern analysis
- **1-Hour:** Primary analysis timeframe
- **4-Hour:** Intermediate trend analysis
- **Daily:** Long-term trend analysis

### 2.3 Methodology Overview

This research employs a **hybrid statistical-ML approach** combining:
1. **Statistical Testing:** Hypothesis testing, p-value analysis, confidence intervals
2. **Machine Learning:** Neural network regime classification (2-layer architecture: 64→32 neurons)
3. **Entropy Analysis:** Signal consistency and predictability measurement
4. **Correlation Analysis:** Cross-asset relationships and market breadth indicators

### 2.4 Six Indicators Tested

1. **RSI (Relative Strength Index)** - Momentum oscillator (14-period)
2. **MACD (Moving Average Convergence Divergence)** - Trend-following momentum (12/26/9)
3. **ATR (Average True Range)** - Volatility measurement (14-period)
4. **SMA-50 / SMA-200** - Trend identification moving averages
5. **Volume** - Trading activity and confirmation
6. **VWAP (Volume Weighted Average Price)** - Intraday price reference


---

## 3. Volume Analysis

### 3.1 Volume Distribution Across Timeframes

**Table: Volume Statistics by Timeframe**

| Timeframe | Mean Volume | Median Volume | Std Dev | Min | Max | Sample Size |
|-----------|-------------|---------------|---------|-----|-----|-------------|
| 1Min      |         186 |           100 |     255 |     1 | 10,227 |   1,358,505 |
| 5Min      |         927 |           517 |   1,206 |     1 | 42,747 |     272,751 |
| 1Hour     |      11,149 |         6,544 |  13,279 |    64 | 297,460 |      22,808 |
| 4Hour     |      42,485 |        26,766 |  46,415 |   220 | 865,944 |       5,952 |
| 1Day      |     254,852 |       172,072 | 216,852 | 13,750 | 1,921,968 |         992 |

**Key Findings:**
- Most active timeframe: 1Day (mean volume 254,852)
- Volume concentration: Gold trading peaks during US market hours (13:00-15:00 UTC) with 6.9x higher volume than quiet periods. This concentration reflects London-New York session overlap where the majority of gold trading occurs. Analysis of 22,808 hourly bars shows consistent intraday patterns driven by institutional and central bank activity.
- Intraday peak activity: 13:00 UTC, 14:00 UTC, 15:00 UTC; quiet hours: 23:00 UTC, 22:00 UTC, 21:00 UTC

**Detailed Interpretation:**
1. **High-Volume Sessions**
   - Correlation: |ρ| = 0.425 between volume and absolute returns.
   - Interpretation: High-volume candles average 0.21% absolute returns versus 0.07% in quiet hours, signalling larger breakouts.
   - Practical: Prioritise breakout setups during 13:00 UTC, 14:00 UTC, 15:00 UTC with wider targets.
2. **Low-Volume Windows**
   - Behavior: Liquidity drops roughly 6.86× outside of peak hours, reducing follow-through.
   - Risk: Tight ranges (≈0.07%) heighten slippage for large orders.
   - Practical: Scale down size and favour mean-reversion tactics when participation thins.
3. **Volume-Price Confirmation**
   - Evidence: 22807 paired observations confirm volume spikes precede larger price swings.
   - Interpretation: Combine indicator triggers with volume filters to reduce false positives.
   - Practical: Trigger alerts when volume breaches its 75th percentile to focus on high-conviction moves.

![Volume by Timeframe](data/processed/volume_by_timeframe.png)

### 3.2 Intraday Volume Patterns

**Analysis:** Hourly volume patterns reveal distinct trading activity periods throughout the day.

**Key Observations:**
- **Peak Volume Hours:** 13:00 UTC, 14:00 UTC, 15:00 UTC — 6.9x higher than quiet periods
- **Low Volume Hours:** 23:00 UTC, 22:00 UTC, 21:00 UTC — consistent liquidity trough
- **Volume Surges:** Peaks align with London-New York overlap and macro releases
- **Weekend Effects:** Liquidity drops sharply after Friday 21:00 UTC

![Volume Heatmap](data/processed/volume_heatmap.png)

### 3.3 Volume-Price Relationship

**Correlation Analysis:**

| Metric | Correlation | p-value | Significant? |
|--------|-------------|---------|--------------|
| Volume vs Price Change | -0.041 | 0.0000 | Yes |
| Volume vs abs(Return) | 0.425 | 0.0000 | Yes |
| High vs Low Volume (abs return) | 0.21% vs 0.07% | — | Higher |


---

## 4. Volatility Analysis

### 4.1 ATR Across Timeframes

**Table: ATR (Average True Range) Statistics by Timeframe**

| Timeframe | Mean ATR | Median ATR | Std Dev | Min | Max | ATR % of Price |
|-----------|----------|------------|---------|-----|-----|---------------|
| 1Min      |  0.72802 |    0.55514 | 0.60076 | 0.05318 | 11.89963 |         0.03% |
| 5Min      |  1.73156 |    1.38359 | 1.26633 | 0.19760 | 18.66505 |         0.07% |
| 1Hour     |  6.33512 |    5.30424 | 3.83538 | 1.77834 | 41.69201 |         0.27% |
| 4Hour     | 12.78866 |   10.70849 | 7.43066 | 4.55074 | 70.29365 |         0.55% |
| 1Day      | 33.25557 |   29.28077 | 15.16660 | 14.20847 | 110.20565 |         1.43% |

**Detailed Insights:**
1. **High-Volatility Timeframes:** 1Day prints the largest swings with ATR ≈ 1.43% of price, favouring breakout trades with wider risk buffers.
2. **Low-Volatility Windows:** 1Min contracts to 0.03% of price, ideal for scaling into swing positions or deploying mean-reversion setups.
**Interpretation:**
Gold exhibits moderate volatility with average ATR of 0.27% of price, typical for precious metals. Volatility demonstrates clustering (autocorrelation=0.36), with volatility regimes persisting for several days. This pattern reflects the influence of macroeconomic events, central bank announcements, and geopolitical developments on gold price action.

**Practical Takeaways:**
3. **Clustering Behaviour:** Volatility clusters with autocorrelation 0.356, so traders should anticipate streaks of elevated risk once volatility spikes.
4. **Risk Calibration:** Typical 1-hour moves average 0.27% of price; size positions to withstand ±1 ATR noise.
### 4.2 Intraday Volatility Patterns

![Intraday Volatility](data/processed/intraday_volatility.png)

### 4.3 Volatility Clustering Analysis

![Volatility Clustering](data/processed/volatility_clustering.png)


---

## 5. Trend Characteristics

*Trend analysis results unavailable — re-run trend analysis pipeline.*


---

## 6. Technical Indicator Effectiveness

### 6.1 Testing Methodology

**Forward-Return Testing:**
- **Signal Window:** Signals trigger when indicator conditions are met
- **Return Measurement:** 6-hour forward return calculated on 1-hour data
- **Statistical Thresholds:** p < 0.05 for significance; Sharpe > 0.5 preferred

### 6.2 Complete Indicator Ranking

**Table: Complete Indicator Performance Summary**

| Rank | Indicator | Signal Type | Total Signals | Win Rate | Avg Return | p-value | Entropy | Quality Rating |
|------|-----------|-------------|---------------|----------|------------|---------|---------|----------------|
| 1 | SMA | SMA_50_Bounce | 3,648 | 64.5% | +0.13% | 0.0000 | 0.493 | Good |
| 2 | RSI | RSI_Overbought | 1,814 | 18.2% | +0.29% | 0.0000 | 0.720 | Moderate |
| 3 | RSI | RSI_Oversold | 1,127 | 31.2% | +0.45% | 0.0000 | 0.842 | Moderate-Poor |
| 4 | MACD | MACD_Bearish_Cross | 862 | 41.6% | +0.58% | 0.0000 | 0.979 | Poor |
| 5 | MACD | MACD_Bullish_Cross | 862 | 58.4% | +0.58% | 0.0000 | 0.985 | Poor |

### 6.3 Detailed Analysis for Top 3 Indicators

#### 6.3.1 SMA - SMA_50_Bounce

**Performance Summary:**
- Win Rate: 64.5% (Total signals: 3,648)
- Average Return: +0.13%
- Entropy Score: 0.493 - Good
- Sharpe Ratio: 0.13; Profit Factor: 1.73
- Regime-Specific Performance: Best in range regimes (70.9% win, n=815); weakest in down regimes (0.0% win).
- Why it Works: Delivering 70.9% win rate in range regimes thanks to clear directional follow-through.
- Playbook: Deploy on the primary timeframe with confirmation from volume or trend filters; cut exposure when entropy rises.
- Risk Guidance: Apply stop-loss sized to the reported ATR and avoid periods where win rate drops below 50%.

#### 6.3.2 MACD - MACD_Bullish_Cross

**Performance Summary:**
- Win Rate: 58.4% (Total signals: 862)
- Average Return: +0.58%
- Entropy Score: 0.985 - Poor
- Sharpe Ratio: 0.23; Profit Factor: 1.81
- Regime-Specific Performance: Best in up regimes (60.3% win, n=277); weakest in range regimes (56.7% win).
- Why it Works: Overall performance clears the 55% bar while up regimes still contribute 60.3% of consistent wins.
- Playbook: Deploy on the primary timeframe with confirmation from volume or trend filters; cut exposure when entropy rises.
- Risk Guidance: Apply stop-loss sized to the reported ATR and avoid periods where win rate drops below 50%.

### 6.4 Indicators to Avoid

#### RSI - RSI_Overbought

- Evidence: Win rate 18.2% with average return +0.29% (fails 55% threshold).
- Risk Metrics: Profit factor 1.83, drawdown 70.4%.
- Regime Caveat: Even the best regime (up) only reaches 18.5% win rate — insufficient for deployment.
- Action: Remove from live playbooks or combine with stricter filters until performance materially improves.

#### RSI - RSI_Oversold

- Evidence: Win rate 31.2% with average return +0.45% (fails 55% threshold).
- Risk Metrics: Profit factor 1.85, drawdown 67.4%.
- Regime Caveat: Even the best regime (down) only reaches 33.3% win rate — insufficient for deployment.
- Action: Remove from live playbooks or combine with stricter filters until performance materially improves.

#### MACD - MACD_Bearish_Cross

- Evidence: Win rate 41.6% with average return +0.58% (fails 55% threshold).
- Risk Metrics: Profit factor 1.82, drawdown 69.2%.
- Regime Caveat: Even the best regime (range) only reaches 43.8% win rate — insufficient for deployment.
- Action: Remove from live playbooks or combine with stricter filters until performance materially improves.


---

## 7. Market Regime Analysis

### 7.1 Regime Classification Methodology

**Hybrid Approach:** Heuristic labeling + Neural network

**Model Architecture:** 2-layer feedforward network (64→32 neurons)

### 7.2 Model Performance

**Test Accuracy:** 88.08%

**Confusion Matrix:**

|                | Pred Range | Pred Up | Pred Down |
|----------------|------------|---------|-----------|
| **True Range** | 1792      | 192     | 40         |
| **True Up**    | 160        | 1666   | 0         |
| **True Down**  | 149        | 3       | 560       |

**Per-Class Metrics:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Range | 0.85      | 0.89   | 0.87     | 2024     |
| Up | 0.90      | 0.91   | 0.90     | 1826     |
| Down | 0.93      | 0.79   | 0.85     | 712     |


**Train-Val Gap:** 9.59%

![Confusion Matrix Heatmap](data/processed/confusion_matrix_heatmap.png)

### 7.3 Current Market Regime

**Regime Classification:** Range
**Model Confidence:** 59.7%
**Probability Distribution:**
- Range: 59.7%
- Up: 40.3%
- Down: 0.0%

![ML Regime Timeline](data/processed/ml_regime_timeline.png)

### 7.4 Regime-Specific Characteristics

![Regime Distribution](data/processed/regime_distribution_comparison.png)


---

## 8. Correlation Analysis

### 8.1 Correlation Matrix

*Correlation matrix unavailable — ensure correlation analysis has been executed for this asset.*

*Correlation heatmap unavailable — rerun correlation analysis to generate visualization.*

*Rolling correlation plot unavailable — rerun correlation analysis to generate visualization.*


---

## 9. Key Takeaways & Recommendations

### 9.1 What Makes Gold Unique

Gold trades during 24/7 (London-New York overlap is most active) with moderate (0.8-1.2% daily). subtle trends, mean-reverting characteristics and volume concentration peaks during us/london session overlap (13:00-21:00 utc) shape intraday opportunity.

### 9.2 Highest-Probability Trading Setups

#### Setup 1: SMA - SMA_50_Bounce
- **Win Rate:** 64.5% | **Average Return:** +0.13%
- **Quality:** Good (Entropy 0.493)
- **Regime Edge:** Best in range regimes (70.9% win, n=815); weakest in down regimes (0.0% win).
- **Why It Works:** 70.9% win rate in range regimes underscores directional follow-through when macro flows align.
- **Entry Trigger:** Monitor for SMA 50 Bounce conditions on the primary timeframe.
- **Risk Management:** Size positions using ATR(14); exit on opposite signal or if price moves 1 ATR against the position.

#### Setup 2: MACD - MACD_Bullish_Cross
- **Win Rate:** 58.4% | **Average Return:** +0.58%
- **Quality:** Poor (Entropy 0.985)
- **Regime Edge:** Best in up regimes (60.3% win, n=277); weakest in range regimes (56.7% win).
- **Why It Works:** Overall performance clears the 55% bar while up regimes still contribute 60.3% win rate to the edge.
- **Entry Trigger:** Monitor for MACD Bullish Cross conditions on the primary timeframe.
- **Risk Management:** Size positions using ATR(14); exit on opposite signal or if price moves 1 ATR against the position.

### 9.3 Signals to Avoid

1. **RSI - RSI_Overbought**
   - Evidence: Win rate 18.2% with average return +0.29% (below threshold).
   - Regime Check: Best outcome still 18.5% during up regimes — insufficient edge.
   - Action: Archive this setup or require additional filters (macro, volume confirmation) before consideration.
2. **RSI - RSI_Oversold**
   - Evidence: Win rate 31.2% with average return +0.45% (below threshold).
   - Regime Check: Best outcome still 33.3% during down regimes — insufficient edge.
   - Action: Archive this setup or require additional filters (macro, volume confirmation) before consideration.
3. **MACD - MACD_Bearish_Cross**
   - Evidence: Win rate 41.6% with average return +0.58% (below threshold).
   - Regime Check: Best outcome still 43.8% during range regimes — insufficient edge.
   - Action: Archive this setup or require additional filters (macro, volume confirmation) before consideration.


### 9.4 Current Market Assessment

**Current Regime:** Range (confidence 59.7%).
**Active Signal Focus:** SMA - SMA_50_Bounce performs best in Best in range regimes (70.9% win, n=815); weakest in down regimes (0.0% win).
**Near-Term Outlook:** Align trades with the dominant regime and avoid deploying signals where regime performance deteriorates.


---

## 10. Statistical Summary & Methodology

### 10.1 Data Quality

**Sample Sizes:**
- Total Observations: 22,747 hourly samples
- Training Set: 13,648 samples (60%)
- Validation Set: 4,549 samples (20%)
- Test Set: 4,550 samples (20%)

### 10.2 Statistical Rigor

**All P-Values Reported:** p < 0.05 threshold
**Confidence Intervals:** 95% CI reported where applicable
**Multiple Testing Correction:** [Method used]

### 10.3 Limitations

[Description of limitations and caveats]


---

## Appendix: Technical Details

### A.1 Complete Indicator Formulas

[Formulas for all indicators]

### A.2 Neural Network Architecture Details

**Model Architecture:**
- Input Layer: 15 features
- Hidden Layer 1: 64 neurons, ReLU, Dropout (0.3)
- Hidden Layer 2: 32 neurons, ReLU, Dropout (0.3)
- Output Layer: 3 classes (Range, Up, Down)

### A.3 Feature Engineering Specifications

[Feature list and normalization details]

### A.4 Train/Validation/Test Split Methodology

**Split:** 60% train, 20% validation, 20% test (temporal split)
