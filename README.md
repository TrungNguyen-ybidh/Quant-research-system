# AI-Driven Quantitative Trading Research System

## Project Overview

This quantitative trading research system analyzes Gold (XAU/USD) trading patterns using hybrid statistical and machine learning methodologies. The system processes historical OHLCV data, calculates technical indicators, performs statistical testing, and generates comprehensive research reports with actionable trading recommendations.

**Key Features:**
- Multi-timeframe analysis (1-minute, 5-minute, 1-hour, 4-hour, daily)
- Technical indicator testing with entropy-based quality scoring
- Neural network regime classification (86.66% accuracy)
- Statistical validation with confidence intervals and p-values
- Automated report generation with professional visualizations

**Analysis Period:** January 2022 – October 2025  
**Primary Timeframe:** 1-Hour  
**Total Samples:** 22,747 hourly observations

---

## Installation

### Requirements

- **Python:** 3.8 or higher
- **Operating System:** macOS, Linux, or Windows
- **Dependencies:** See `requirements.txt`

### Installation Steps

1. **Clone Repository:**
   ```bash
   git clone <repository-url>
   cd quant-research-system
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Key packages:
   - `pandas` - Data manipulation
   - `numpy` - Numerical computations
   - `matplotlib` - Visualization
   - `seaborn` - Statistical visualizations
   - `scikit-learn` - Machine learning utilities
   - `torch` - PyTorch for neural networks
   - `yfinance` - Yahoo Finance data
   - `alpaca-trade-api` - Alpaca API (optional)

3. **Configure API Keys:**
   
   Create a `.env` file in the project root:
   ```bash
   ALPACA_API_KEY=your_api_key_here
   ALPACA_SECRET_KEY=your_secret_key_here
   ALPACA_BASE_URL=https://paper-api.alpaca.markets
   ```

   **Note:** For Gold (XAU/USD) analysis, Yahoo Finance is used as the primary data source (no API key required).

---

## Usage

### Quick Start: Generate Complete Research Report

The fastest way to generate a complete research report:

```bash
python src/report_generator.py
```

This will generate `reports/XAU_USD_Research_Report.md` with all findings from Weeks 2-4.

### Step-by-Step Workflow

#### Step 1: Collect Data

**For Gold (XAU/USD) from Yahoo Finance:**
```bash
python src/data_collection.py --symbol XAU-USD --start 2022-01-01 --end 2025-10-31
```

**For other assets:**
```bash
python src/data_collection.py --symbol SPY --start 2022-01-01 --end 2025-10-31
```

**Output:** Raw CSV files saved to `data/raw/`

#### Step 2: Calculate Indicators

Process raw data and calculate all technical indicators:

```bash
python scripts/process_all_data.py
```

**Output:** Processed CSV files with indicators saved to `data/processed/`

**Indicators Calculated:**
- RSI (14-period)
- MACD (12/26/9)
- ATR (14-period)
- SMA-20, SMA-50, SMA-200
- ADX (14-period)
- VWAP

#### Step 3: Run Analysis Modules

**Volume & Volatility Analysis (Week 2):**
```bash
python src/trend_analysis.py
```

**Indicator Testing (Week 3):**
```bash
python src/indicator_testing.py
python src/entropy_analysis.py
```

**Correlation Analysis (Week 3):**
```bash
python src/correlation_analysis.py
```

**Regime Classification (Week 4):**
```bash
# Label data
python src/regime_labeling.py

# Split data
python src/data_preparation.py

# Train model
python src/train_regime_model.py

# Evaluate model
python src/evaluate_regime_model.py

# Robustness testing
python src/robustness_testing.py

# Full dataset prediction
python src/full_dataset_prediction.py

# Generate visualizations
python src/create_visualizations.py
```

#### Step 4: Generate Report

**Automated Report Generation:**
```bash
python src/report_generator.py
```

**Output:** `reports/XAU_USD_Research_Report.md`

**Manual Report Editing:**
- Edit `reports/XAU_USD_Research_Report.md` directly
- Re-run `report_generator.py` to regenerate automated sections

---

## Project Structure

```
quant-research-system/
├── data/
│   ├── raw/                    # Raw OHLCV data from APIs
│   │   ├── XAU_USD_1Hour.csv
│   │   └── ...
│   └── processed/              # Processed data with indicators
│       ├── XAU_USD_1Hour_with_indicators.csv
│       ├── regime_labels.csv
│       ├── regime_predictions.csv
│       └── ... (analysis results)
│
├── models/                     # Machine learning models
│   ├── regime_classifier.pth
│   ├── model_config.json
│   ├── training_history.json
│   └── ...
│
├── reports/                    # Generated research reports
│   └── XAU_USD_Research_Report.md
│
├── src/                        # Source code modules
│   ├── data_collection.py      # Data fetching from APIs
│   ├── indicators.py           # Technical indicator calculations
│   ├── trend_analysis.py       # Trend identification and analysis
│   ├── indicator_testing.py    # Forward-return testing
│   ├── entropy_analysis.py     # Signal consistency analysis
│   ├── correlation_analysis.py # Cross-asset correlation
│   ├── regime_labeling.py     # Automatic regime classification
│   ├── regime_model.py        # Neural network architecture
│   ├── train_regime_model.py  # Model training pipeline
│   ├── evaluate_regime_model.py # Model evaluation
│   ├── robustness_testing.py  # SAFE framework testing
│   ├── full_dataset_prediction.py # Full dataset predictions
│   ├── create_visualizations.py # Visualization generation
│   ├── report_generator.py    # Automated report generation
│   └── validate_report.py     # Report validation script
│
├── scripts/                    # Utility scripts
│   └── process_all_data.py     # Batch data processing
│
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Methodology

### Hybrid Statistical-ML Approach

This research employs a **hybrid methodology** combining:

1. **Statistical Testing:**
   - Forward-return testing (6-hour window)
   - Hypothesis testing with p-values (threshold: p < 0.05)
   - Confidence intervals (95% CI)
   - Multiple testing correction (Bonferroni)

2. **Entropy Analysis:**
   - Signal consistency measurement
   - Quality rating based on entropy scores
   - Predictability assessment

3. **Machine Learning:**
   - Neural network regime classification
   - 2-layer architecture (64→32 neurons)
   - Hybrid heuristic + ML labeling
   - Robustness testing (SAFE framework)

4. **Correlation Analysis:**
   - Pearson correlation with p-values
   - Rolling correlation (60-day window)
   - Cross-sectional dispersion
   - Market breadth indicators

### Technical Indicators Evaluated

Six key indicators tested across multiple signal types:

1. **RSI (Relative Strength Index)** - Momentum oscillator (14-period)
2. **MACD (Moving Average Convergence Divergence)** - Trend-following (12/26/9)
3. **ATR (Average True Range)** - Volatility measurement (14-period)
4. **SMA-50 / SMA-200** - Trend identification moving averages
5. **Volume** - Trading activity and confirmation
6. **VWAP (Volume Weighted Average Price)** - Intraday price reference

### Model Architecture

**Neural Network:**
- **Input Layer:** 15 normalized features
- **Hidden Layer 1:** 64 neurons (ReLU, Dropout 0.3)
- **Hidden Layer 2:** 32 neurons (ReLU, Dropout 0.3)
- **Output Layer:** 3 classes (Range, Up, Down)

**Training:**
- Optimizer: Adam (learning rate: 0.001)
- Loss: Cross-entropy
- Regularization: L2 weight decay (0.0001), Dropout (0.3)
- Early stopping: Patience of 10 epochs
- Split: 60% train, 20% validation, 20% test (temporal)

---

## Results

### Key Findings

**Volume & Timing:**
- Gold trading concentrates 75% of volume during US hours (13:00-21:00 UTC)
- Peak activity: 13:00 UTC (US market open)
- 9.5x volume difference between peak and quiet hours

**Trend Persistence:**
- Uptrends persist 47% longer than downtrends (27.86h vs 18.95h, p=0.000532)
- Normal pullbacks in uptrends: 0.19% ± 0.20%
- Gold spends 40.7% of time ranging, 36.3% uptrending, 23.0% downtrending

**Indicator Effectiveness:**
- **SMA-50 Bounce:** 64.6% win rate, 0.492 entropy (Good quality) - Most reliable
- **RSI < 30:** 31.2% win rate, regime-dependent (72% in uptrends, 48% in downtrends)
- **VWAP Mean Reversion:** 64.7% success rate

**Regime Classification:**
- Neural network achieves **86.66% accuracy** (2.59x better than random)
- Excellent generalization (3.32% train-val gap)
- Zero confusion between opposite trends

**Volatility Clustering:**
- Extremely strong persistence (autocorrelation = 0.9965, p < 0.001)
- High volatility regimes persist ~30 hours on average (96.64% probability)

### Sample Report

See `reports/XAU_USD_Research_Report.md` for complete findings.

**Report Statistics:**
- **Length:** ~1,040 lines (15-20 pages)
- **Visualizations:** 14 professional charts
- **Sections:** 11 complete sections
- **Statistical Rigor:** All claims backed by p-values, confidence intervals, sample sizes

---

## Future Work

### Planned Enhancements

1. **Additional Indicators:**
   - Bollinger Bands
   - Fibonacci retracements
   - Ichimoku Cloud
   - Stochastic Oscillator

2. **Model Improvements:**
   - Ensemble methods for robustness
   - LSTM/GRU for sequence modeling
   - Attention mechanisms for regime transitions
   - Data augmentation for training

3. **Extended Analysis:**
   - Multi-asset portfolio optimization
   - Risk-adjusted return metrics (Sharpe ratio, Sortino ratio)
   - Backtesting framework
   - Live trading integration

4. **Alternative Assets:**
   - Apply methodology to other commodities (Silver, Oil)
   - Crypto assets (BTC, ETH)
   - Forex pairs (EUR/USD, GBP/USD)
   - Stock indices (SPY, QQQ)

5. **Advanced Features:**
   - Real-time data streaming
   - Automated signal generation
   - Portfolio rebalancing
   - Risk management automation

---

## Validation

### Report Validation

Run comprehensive validation checks:

```bash
python src/validate_report.py
```

**Checks Performed:**
1. Completeness - No placeholders, all sections filled
2. Statistical Rigor - All claims have supporting statistics
3. Actionability - Recommendations are specific and usable
4. Professional Quality - Formatting, consistency
5. Limitations - Honest reporting of weaknesses
6. Visualizations - All referenced images exist

---

## Contributing

### Development Guidelines

1. **Code Style:**
   - Follow PEP 8 Python style guide
   - Use type hints where appropriate
   - Document all functions with docstrings

2. **Testing:**
   - Add unit tests for new functions
   - Validate outputs match expected formats
   - Test with sample data before full runs

3. **Documentation:**
   - Update README for new features
   - Document methodology changes
   - Include examples in docstrings

---

## License

This project is for academic/research purposes. See LICENSE file for details.

---

## Contact & Support

For questions or issues:
- Review the comprehensive report: `reports/XAU_USD_Research_Report.md`
- Check validation results: `python src/validate_report.py`
- Review code documentation in `src/` modules

---

**Last Updated:** November 2025  
**System Version:** 1.0  
**Status:** Production Ready

