"""
Configuration file for AI-Driven Quantitative Trading Research System.

This module centralizes all configuration settings including API credentials,
data paths, indicator parameters, file naming conventions, visualization options,
logging controls, and validation thresholds.
"""

from dotenv import load_dotenv
import os
from datetime import datetime, timedelta, timezone

try:
    load_dotenv()
except PermissionError:
    # In restricted environments (e.g., sandboxed execution), .env may not be accessible.
    # Continue without env file rather than failing.
    pass

# ============================================================================
# API Configuration
# ============================================================================

# Alpaca API Configuration
APCA_API_KEY_ID = os.getenv('APCA_API_KEY_ID')
APCA_API_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
APCA_API_BASE_URL = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')  # Default to paper trading

# Switch between paper and live endpoints easily
# Paper: https://paper-api.alpaca.markets
# Live: https://api.alpaca.markets

# OANDA API Configuration
OANDA_API_TOKEN = os.getenv('OANDA_API_TOKEN')
OANDA_ENVIRONMENT = os.getenv('OANDA_ENVIRONMENT', 'practice')  # 'practice' or 'live'
OANDA_ACCOUNT_ID = os.getenv('OANDA_ACCOUNT_ID')  # Optional, needed for some operations


# ============================================================================
# Data Settings
# ============================================================================

# Main directories for raw and processed data
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"  # Processed data with indicators

# Default trading symbol
# For gold: Use OANDA format "XAU/USD" (gold/commodities)
# For Alpaca crypto: Use slash format "BTC/USD", "ETH/USD"
# For OANDA forex: Use underscore or slash format "EUR_USD", "GBP/USD", "XAU/USD", etc.
# For stocks: Use standard ticker "AAPL", "MSFT", etc.
DEFAULT_SYMBOL = "XAU/USD"  # Gold via OANDA

# Alternative symbols to collect
# Examples:
# - OANDA: "XAU/USD" (gold), "EUR_USD", "GBP_USD", "EUR/USD", "GBP/USD" (forex pairs)
# - Yahoo Finance: "XAUUSD=X" (gold), "XAGUSD=X" (silver), "BTC-USD" (crypto)
# - Alpaca: "BTC/USD", "ETH/USD" (crypto), "AAPL" (stocks)
DEFAULT_ASSETS = ['XAU/USD']

# All timeframes to fetch (Alpaca format)
DEFAULT_TIMEFRAMES = ["1Min", "5Min", "1Hour", "4Hour", "1Day"]

# Historical data range
# Start date: How far back to collect data (2022-2024 timeline)
# Using UTC timezone for consistency with Alpaca API
HISTORICAL_START_DATE = datetime(2022, 1, 1, tzinfo=timezone.utc)  # 2-3 years of data
# End date: Latest date to collect (defaults to today in UTC)
HISTORICAL_END_DATE = datetime.now(timezone.utc)


# ============================================================================
# Indicator Parameters
# ============================================================================

# RSI (Relative Strength Index)
RSI_LENGTH = 14

# ATR (Average True Range)
ATR_PERIOD = 14

# MACD (Moving Average Convergence Divergence)
MACD_FAST_SPAN = 12
MACD_SLOW_SPAN = 26
MACD_SIGNAL_SPAN = 9

# Moving Averages
SMA_SHORT_LENGTH = 20
SMA_LONG_LENGTH = 50
EMA_SHORT_LENGTH = 12
EMA_LONG_LENGTH = 26

# Bollinger Bands
BB_PERIOD = 20
BB_STD_DEV = 2

# Stochastic Oscillator
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3

# Additional indicators
ADX_PERIOD = 14
CCI_PERIOD = 20


# ============================================================================
# File Naming and Path Rules
# ============================================================================

# Raw data file naming: "symbol_timeframe.csv"
# Example: "XAUUSD_1Min.csv"
RAW_DATA_FILENAME_PATTERN = "{symbol}_{timeframe}.csv"

# Processed data file naming: "symbol_timeframe_with_indicators.csv"
# Example: "XAU_USD_1Min_with_indicators.csv"
PROCESSED_DATA_FILENAME_PATTERN = "{symbol}_{timeframe}_with_indicators.csv"

# Helper function to generate file paths
def get_raw_data_path(symbol: str, timeframe: str) -> str:
    """Generate path for raw data file."""
    filename = RAW_DATA_FILENAME_PATTERN.format(symbol=symbol, timeframe=timeframe)
    return os.path.join(RAW_DATA_PATH, filename)

def get_processed_data_path(symbol: str, timeframe: str) -> str:
    """Generate path for processed data file."""
    filename = PROCESSED_DATA_FILENAME_PATTERN.format(symbol=symbol, timeframe=timeframe)
    return os.path.join(PROCESSED_DATA_PATH, filename)


# ============================================================================
# Visualization and Reporting Options
# ============================================================================

# Chart style and theme
CHART_STYLE = "seaborn-v0_8-darkgrid"  # matplotlib style
CHART_COLOR_THEME = "dark"  # 'light' or 'dark'
CHART_FIGURE_SIZE = (12, 6)
CHART_DPI = 100

# Color palette for indicators
COLOR_PALETTE = {
    'price': '#1f77b4',
    'volume': '#ff7f0e',
    'sma_short': '#2ca02c',
    'sma_long': '#d62728',
    'ema_short': '#9467bd',
    'ema_long': '#8c564b',
    'rsi': '#e377c2',
    'macd': '#7f7f7f',
    'signal': '#bcbd22',
    'atr': '#17becf'
}

# Date format for charts and reports
DATE_FORMAT = "%Y-%m-%d"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


# ============================================================================
# Logging and Debugging Controls
# ============================================================================

# Verbose output control
VERBOSE = True

# Log file name
LOG_FILE = "quant_research_system.log"

# Log level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
LOG_LEVEL = "INFO"

# Whether to log to console in addition to file
LOG_TO_CONSOLE = True


# ============================================================================
# Validation and Quality Thresholds
# ============================================================================

# Maximum allowed percentage of missing data (0.0 to 1.0)
# Data with more missing values will trigger warnings or be rejected
MAX_MISSING_DATA_PERCENTAGE = 0.05  # 5% missing data allowed

# Outlier filtering strictness (Z-score threshold)
# Values beyond this many standard deviations will be considered outliers
OUTLIER_Z_SCORE_THRESHOLD = 3.0

# Minimum number of data points required for analysis
MIN_DATA_POINTS = 100

# Price change validation (percentage)
# Price changes beyond this threshold will be flagged as potential data errors
MAX_PRICE_CHANGE_PERCENTAGE = 0.10  # 10% change threshold

# Volume validation
# Volume values below this threshold will be considered suspicious
MIN_VOLUME_THRESHOLD = 0
