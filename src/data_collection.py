"""
Data Collection Module for AI-Driven Quantitative Trading Research System.

This module handles the complete data ingestion pipeline:
- API connection and authentication
- Historical OHLCV data fetching from multiple sources (Alpaca, OANDA, Yahoo Finance)
- Data cleaning and validation
- Saving to disk with consistent naming

Supports:
- Alpaca API: Crypto pairs (BTC/USD, ETH/USD) and US equities
- OANDA API: Forex pairs (EUR_USD, GBP_USD, EUR/USD, etc.)
- Yahoo Finance: Gold (XAUUSD=X), commodities, forex, and other assets
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import os
import sys
from typing import Optional, Tuple
import warnings
import time

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    from alpaca_trade_api import REST, TimeFrame
except ImportError:
    print("Warning: alpaca-trade-api not installed. Install with: pip install alpaca-trade-api")
    REST = None
    TimeFrame = None

try:
    import yfinance as yf
except ImportError:
    print("Warning: yfinance not installed. Install with: pip install yfinance")
    yf = None

try:
    from oandapyV20 import API as OandaAPI
    from oandapyV20.endpoints import instruments
    oandapyV20_available = True
except ImportError:
    print("Warning: oandapyV20 not installed. Install with: pip install oandapyV20")
    OandaAPI = None
    instruments = None
    oandapyV20_available = False


# ============================================================================
# API Connection
# ============================================================================

def connect_to_alpaca() -> Optional[REST]:
    """
    Connect to Alpaca API using credentials from config.
    
    Returns:
        REST API client instance if successful, None otherwise.
    """
    if REST is None:
        raise ImportError("alpaca-trade-api library is required. Install with: pip install alpaca-trade-api")
    
    if not config.APCA_API_KEY_ID or not config.APCA_API_SECRET_KEY:
        raise ValueError("Alpaca API credentials not found in environment variables. "
                        "Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in your .env file.")
    
    try:
        api = REST(
            key_id=config.APCA_API_KEY_ID,
            secret_key=config.APCA_API_SECRET_KEY,
            base_url=config.APCA_API_BASE_URL,
            api_version='v2'
        )
        
        # Test connection by getting account info
        account = api.get_account()
        if config.VERBOSE:
            print(f"✓ Connected to Alpaca API ({config.APCA_API_BASE_URL})")
            print(f"  Account Status: {account.status}")
        
        return api
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Alpaca API: {str(e)}")


def connect_to_oanda():
    """
    Connect to OANDA API using credentials from config.
    
    Returns:
        OANDA API client instance if successful
    """
    if not oandapyV20_available or OandaAPI is None:
        raise ImportError("oandapyV20 library is required. Install with: pip install oandapyV20")
    
    if not config.OANDA_API_TOKEN:
        raise ValueError("OANDA API token not found in environment variables. "
                        "Set OANDA_API_TOKEN in your .env file.")
    
    try:
        # OANDA API uses access token and environment (practice or live)
        environment = config.OANDA_ENVIRONMENT or 'practice'
        api = OandaAPI(access_token=config.OANDA_API_TOKEN, environment=environment)
        
        if config.VERBOSE:
            print(f"✓ Connected to OANDA API ({environment} environment)")
        
        return api
    except Exception as e:
        raise ConnectionError(f"Failed to connect to OANDA API: {str(e)}")


def determine_data_source(symbol: str) -> str:
    """
    Determine which data source to use for a given symbol.
    
    OANDA supports:
    - Forex pairs: EUR_USD, GBP_USD, EUR/USD, GBP/USD (with underscore or slash)
    - Gold/commodities: XAU/USD, XAG/USD (gold, silver)
    - Major currency pairs and CFDs
    
    Alpaca supports:
    - Crypto pairs with slash format: BTC/USD, ETH/USD
    - US equities: AAPL, MSFT, etc.
    
    Yahoo Finance supports:
    - Gold/commodities: XAUUSD=X (gold), XAGUSD=X (silver)
    - Forex pairs: EURUSD=X, GBPUSD=X
    - Crypto: BTC-USD, ETH-USD
    - Stocks: AAPL, MSFT, etc.
    
    Args:
        symbol: Trading symbol to check
        
    Returns:
        'oanda', 'alpaca', or 'yahoo_finance'
    """
    symbol_upper = symbol.upper()
    
    # Check for OANDA gold/commodities (XAU/USD, XAG/USD)
    if '/' in symbol and (symbol_upper.startswith('XAU/') or symbol_upper.startswith('XAG/')):
        return 'oanda'
    
    # Common forex pairs (major currencies)
    forex_pairs = ['EUR', 'GBP', 'USD', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD', 'SGD', 'HKD', 'NOK', 'SEK', 'ZAR']
    
    # Check if symbol is a forex pair (contains underscore or slash with forex currencies)
    has_forex_indicator = any(currency in symbol_upper for currency in forex_pairs)
    has_pair_separator = '_' in symbol or '/' in symbol
    
    # OANDA forex pairs typically use underscore (EUR_USD) or can use slash (EUR/USD)
    if has_forex_indicator and has_pair_separator and not symbol_upper.startswith('BTC') and not symbol_upper.startswith('ETH'):
        # Check if it's a valid forex pair format
        if '_' in symbol or ('/' in symbol and len(symbol.split('/')) == 2):
            return 'oanda'
    
    # Alpaca crypto symbols use slash format (BTC/USD, ETH/USD)
    if '/' in symbol and (symbol_upper.startswith('BTC') or symbol_upper.startswith('ETH')):
        return 'alpaca'
    
    # Yahoo Finance gold/commodity symbols end with =X
    if symbol.endswith('=X'):
        return 'yahoo_finance'
    
    # For symbols without special formatting, try Alpaca first (for crypto/equities)
    # but fall back to Yahoo Finance if needed
    # Default to Yahoo Finance as it has broader coverage
    return 'yahoo_finance'


def is_crypto_symbol(symbol: str) -> bool:
    """
    Determine if a symbol is a cryptocurrency (uses crypto endpoint on Alpaca).
    
    Note: Alpaca requires crypto symbols in format BTC/USD (with slash).
    This function checks if symbol appears to be crypto.
    
    Args:
        symbol: Trading symbol to check
        
    Returns:
        True if symbol appears to be crypto, False otherwise
    """
    # Alpaca crypto symbols use slash format
    if '/' in symbol:
        return True
    
    # Check for crypto indicators
    crypto_indicators = ['BTC', 'ETH', 'USD']
    symbol_upper = symbol.upper()
    
    return any(indicator in symbol_upper for indicator in crypto_indicators)


# ============================================================================
# Data Fetching
# ============================================================================

def format_rfc3339(dt: datetime) -> str:
    """
    Format datetime to RFC 3339 format with UTC timezone (Z suffix).
    
    Alpaca API requires timestamps in RFC 3339 format with timezone.
    Example: "2022-01-01T00:00:00Z"
    
    Args:
        dt: Datetime object (will be converted to UTC if timezone-naive)
        
    Returns:
        RFC 3339 formatted string with Z suffix
    """
    # Ensure timezone-aware (convert to UTC if naive)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    
    # Format as RFC 3339 with Z suffix
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_from_alpaca(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    api: Optional[REST] = None
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Alpaca API.
    
    Args:
        symbol: Trading symbol (e.g., 'BTC/USD' for crypto, 'AAPL' for stocks)
        timeframe: Timeframe string (e.g., '1Min', '5Min', '1Hour', '1Day')
        start: Start date for data collection
        end: End date for data collection
        api: Optional REST API client (creates new if not provided)
        
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    if REST is None:
        raise ImportError("alpaca-trade-api library is required for Alpaca data source")
    
    if api is None:
        api = connect_to_alpaca()
    
    # Convert timeframe string to Alpaca TimeFrame enum
    timeframe_map = {
        '1Min': TimeFrame.Minute,
        '5Min': TimeFrame(5, TimeFrame.Minute),
        '15Min': TimeFrame(15, TimeFrame.Minute),
        '30Min': TimeFrame(30, TimeFrame.Minute),
        '1Hour': TimeFrame.Hour,
        '4Hour': TimeFrame(4, TimeFrame.Hour),
        '1Day': TimeFrame.Day
    }
    
    if timeframe not in timeframe_map:
        raise ValueError(f"Unsupported timeframe for Alpaca: {timeframe}. "
                        f"Supported: {list(timeframe_map.keys())}")
    
    alpaca_timeframe = timeframe_map[timeframe]
    
    # Determine if we should use crypto or stock endpoint
    use_crypto = is_crypto_symbol(symbol)
    
    try:
        if config.VERBOSE:
            print(f"Fetching {symbol} {timeframe} data from Alpaca ({start.date()} to {end.date()})...")
        
        # Format dates as RFC 3339 with UTC timezone (Z suffix)
        start_str = format_rfc3339(start)
        end_str = format_rfc3339(end)
        
        # Fetch bars data
        if use_crypto:
            bars = api.get_crypto_bars(
                symbol,
                alpaca_timeframe,
                start=start_str,
                end=end_str
            ).df
        else:
            bars = api.get_bars(
                symbol,
                alpaca_timeframe,
                start=start_str,
                end=end_str
            ).df
        
        if bars.empty:
            warnings.warn(f"No data returned for {symbol} {timeframe} in the specified date range.")
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Rename columns to lowercase for consistency
        bars.columns = bars.columns.str.lower()
        
        # Ensure timestamp is in the index or as a column
        if bars.index.name == 'timestamp' or isinstance(bars.index, pd.DatetimeIndex):
            bars = bars.reset_index()
            if 'timestamp' not in bars.columns and 'time' in bars.columns:
                bars.rename(columns={'time': 'timestamp'}, inplace=True)
        
        # Select and rename columns to standard format
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        available_columns = [col for col in required_columns if col in bars.columns]
        
        if 'timestamp' not in bars.columns:
            raise ValueError("Timestamp column not found in API response")
        
        df = bars[available_columns].copy()
        
        return df
        
    except Exception as e:
        raise RuntimeError(f"Error fetching data from Alpaca for {symbol} {timeframe}: {str(e)}")


def fetch_from_yahoo_finance(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Yahoo Finance.
    
    Args:
        symbol: Trading symbol (e.g., 'XAUUSD=X' for gold, 'BTC-USD' for crypto, 'AAPL' for stocks)
        timeframe: Timeframe string (e.g., '1Min', '5Min', '1Hour', '1Day')
        start: Start date for data collection
        end: End date for data collection
        
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    if yf is None:
        raise ImportError("yfinance library is required for Yahoo Finance data source. "
                         "Install with: pip install yfinance")
    
    # Convert timeframe to yfinance interval
    # yfinance supports: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    timeframe_map = {
        '1Min': '1m',
        '5Min': '5m',
        '15Min': '15m',
        '30Min': '30m',
        '1Hour': '1h',
        '4Hour': '1h',  # yfinance doesn't support 4h, use 1h and aggregate later if needed
        '1Day': '1d'
    }
    
    if timeframe not in timeframe_map:
        raise ValueError(f"Unsupported timeframe for Yahoo Finance: {timeframe}. "
                        f"Supported: {list(timeframe_map.keys())}")
    
    yf_interval = timeframe_map[timeframe]
    
    # For 4Hour timeframe, we'll use 1h and note it in the warning
    if timeframe == '4Hour':
        warnings.warn("Yahoo Finance doesn't support 4Hour directly. Using 1Hour data instead.")
        yf_interval = '1h'
    
    try:
        if config.VERBOSE:
            print(f"Fetching {symbol} {timeframe} data from Yahoo Finance ({start.date()} to {end.date()})...")
        
        # Convert timezone-aware datetimes to timezone-naive for yfinance
        # yfinance expects naive datetimes
        start_naive = start.replace(tzinfo=None) if start.tzinfo else start
        end_naive = end.replace(tzinfo=None) if end.tzinfo else end
        
        # Download data
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_naive, end=end_naive, interval=yf_interval)
        
        if df.empty:
            warnings.warn(f"No data returned for {symbol} {timeframe} in the specified date range.")
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Rename Date to timestamp and convert to UTC
        if 'Date' in df.columns:
            df.rename(columns={'Date': 'timestamp'}, inplace=True)
        elif df.index.name == 'Date':
            df = df.reset_index()
            df.rename(columns={'Date': 'timestamp'}, inplace=True)
        
        # Ensure timestamp is timezone-aware (UTC)
        if df['timestamp'].dtype == 'datetime64[ns]':
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            else:
                df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
        
        # Rename columns to lowercase
        df.columns = df.columns.str.lower()
        
        # Select and rename columns to standard format
        column_mapping = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }
        
        # Select available columns
        available_columns = ['timestamp']
        for std_col, yf_col in column_mapping.items():
            if yf_col in df.columns:
                available_columns.append(yf_col)
        
        df = df[available_columns].copy()
        
        # Ensure all required columns exist (fill with NaN if missing)
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                df[col] = np.nan
        
        return df
        
    except Exception as e:
        raise RuntimeError(f"Error fetching data from Yahoo Finance for {symbol} {timeframe}: {str(e)}")


def calculate_chunk_size(timeframe: str) -> int:
    """
    Calculate the number of days per chunk based on timeframe to stay within OANDA's 5000-candle limit.
    
    OANDA has a limit of 5000 candles per request. We calculate chunk sizes conservatively:
    - 1Min: ~3 days (1440 minutes/day * 3 = 4320 candles)
    - 5Min: ~15 days (288 candles/day * 15 = 4320 candles)
    - 15Min: ~50 days (96 candles/day * 50 = 4800 candles)
    - 30Min: ~100 days (48 candles/day * 100 = 4800 candles)
    - 1Hour: ~200 days (24 candles/day * 200 = 4800 candles)
    - 4Hour: ~800 days (6 candles/day * 800 = 4800 candles)
    - 1Day: ~5000 days (1 candle/day * 5000 = 5000 candles)
    
    Args:
        timeframe: Timeframe string (e.g., '1Min', '5Min', '1Hour', '1Day')
        
    Returns:
        Number of days per chunk
    """
    chunk_sizes = {
        '1Min': 3,      # ~4320 candles (3 days)
        '5Min': 15,     # ~4320 candles (15 days)
        '15Min': 50,    # ~4800 candles (50 days)
        '30Min': 100,   # ~4800 candles (100 days)
        '1Hour': 200,   # ~4800 candles (200 days)
        '4Hour': 800,  # ~4800 candles (800 days)
        '1Day': 5000    # ~5000 candles (5000 days)
    }
    
    return chunk_sizes.get(timeframe, 3)  # Default to 3 days for safety


def fetch_from_oanda(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    api=None
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from OANDA API using chunked fetching.
    
    OANDA has a limit of 5000 candles per request, so large date ranges are split
    into smaller chunks and fetched sequentially, then combined into one DataFrame.
    
    Args:
        symbol: Forex pair symbol (e.g., 'EUR_USD', 'GBP_USD', 'EUR/USD', 'XAU/USD')
        timeframe: Timeframe string (e.g., '1Min', '5Min', '1Hour', '1Day')
        start: Start date for data collection
        end: End date for data collection
        api: Optional OANDA API client (creates new if not provided)
        
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    if not oandapyV20_available or OandaAPI is None or instruments is None:
        raise ImportError("oandapyV20 library is required for OANDA data source. "
                         "Install with: pip install oandapyV20")
    
    if api is None:
        api = connect_to_oanda()
    
    # Convert symbol to OANDA format (EUR/USD -> EUR_USD)
    oanda_symbol = symbol.replace('/', '_').upper()
    
    # Convert timeframe to OANDA granularity
    # OANDA supports: S5, S10, S15, S30, M1, M2, M4, M5, M10, M15, M30, H1, H2, H3, H4, H6, H8, H12, D, W, M
    timeframe_map = {
        '1Min': 'M1',
        '5Min': 'M5',
        '15Min': 'M15',
        '30Min': 'M30',
        '1Hour': 'H1',
        '4Hour': 'H4',
        '1Day': 'D'
    }
    
    if timeframe not in timeframe_map:
        raise ValueError(f"Unsupported timeframe for OANDA: {timeframe}. "
                        f"Supported: {list(timeframe_map.keys())}")
    
    granularity = timeframe_map[timeframe]
    
    # Calculate chunk size in days
    chunk_size_days = calculate_chunk_size(timeframe)
    
    try:
        if config.VERBOSE:
            total_days = (end - start).days
            estimated_chunks = max(1, int(total_days / chunk_size_days) + 1)
            print(f"Fetching {symbol} {timeframe} data from OANDA ({start.date()} to {end.date()})...")
            print(f"  Total range: {total_days} days, estimated {estimated_chunks} chunks ({chunk_size_days} days/chunk)")
        
        # Initialize list to store all chunk DataFrames
        all_chunks = []
        
        # Start chunked fetching
        chunk_start = start
        chunk_number = 0
        
        while chunk_start < end:
            chunk_number += 1
            
            # Calculate chunk end (don't exceed final end date)
            chunk_end = min(chunk_start + timedelta(days=chunk_size_days), end)
            
            if config.VERBOSE:
                print(f"  Fetching chunk {chunk_number}: {chunk_start.date()} to {chunk_end.date()}...", end=' ')
            
            try:
                # Format dates as RFC 3339 (OANDA expects RFC 3339 format)
                start_str = format_rfc3339(chunk_start)
                end_str = format_rfc3339(chunk_end)
                
                # Create parameters for OANDA API
                params = {
                    "from": start_str,
                    "to": end_str,
                    "granularity": granularity,
                    "price": "M"  # Mid prices (can also use "B" for bid or "A" for ask)
                }
                
                # Fetch candles data for this chunk
                r = instruments.InstrumentsCandles(instrument=oanda_symbol, params=params)
                api.request(r)
                
                # Parse response
                candles = r.response.get('candles', [])
                
                if not candles:
                    if config.VERBOSE:
                        print("No data")
                    chunk_start = chunk_end
                    time.sleep(0.5)  # Rate limiting
                    continue
                
                # Convert to DataFrame
                chunk_data = []
                for candle in candles:
                    if candle.get('complete', False):  # Only include complete candles
                        mid = candle.get('mid', {})
                        chunk_data.append({
                            'timestamp': pd.to_datetime(candle['time']),
                            'open': float(mid.get('o', 0)),
                            'high': float(mid.get('h', 0)),
                            'low': float(mid.get('l', 0)),
                            'close': float(mid.get('c', 0)),
                            'volume': int(candle.get('volume', 0))
                        })
                
                if chunk_data:
                    chunk_df = pd.DataFrame(chunk_data)
                    
                    # Ensure timestamp is timezone-aware (UTC)
                    if chunk_df['timestamp'].dtype == 'datetime64[ns]':
                        if chunk_df['timestamp'].dt.tz is None:
                            chunk_df['timestamp'] = chunk_df['timestamp'].dt.tz_localize('UTC')
                        else:
                            chunk_df['timestamp'] = chunk_df['timestamp'].dt.tz_convert('UTC')
                    
                    # Set timestamp as index
                    chunk_df = chunk_df.set_index('timestamp').sort_index()
                    
                    all_chunks.append(chunk_df)
                    
                    if config.VERBOSE:
                        print(f"✓ {len(chunk_df)} candles")
                else:
                    if config.VERBOSE:
                        print("No complete candles")
                
            except Exception as e:
                warnings.warn(f"Error fetching chunk {chunk_number} ({chunk_start.date()} to {chunk_end.date()}): {str(e)}")
                # Continue with next chunk
                pass
            
            # Move to next chunk
            chunk_start = chunk_end
            
            # Rate limiting: pause between requests to respect API limits
            # OANDA typically allows 120 requests per second, but we'll be conservative
            time.sleep(0.5)  # 500ms delay between chunks
        
        # Combine all chunks into one DataFrame
        if not all_chunks:
            warnings.warn(f"No data returned for {symbol} {timeframe} in the specified date range.")
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Concatenate all chunks
        df = pd.concat(all_chunks, axis=0)
        
        # Sort by timestamp
        df = df.sort_index()
        
        # Remove duplicates (in case of overlaps between chunks)
        initial_count = len(df)
        df = df[~df.index.duplicated(keep='first')]
        duplicates_removed = initial_count - len(df)
        
        if duplicates_removed > 0 and config.VERBOSE:
            print(f"  Removed {duplicates_removed} duplicate timestamps")
        
        # Final summary
        if config.VERBOSE:
            print(f"  ✓ Total: {len(df)} candles collected from {df.index.min().date()} to {df.index.max().date()}")
        
        return df
        
    except Exception as e:
        raise RuntimeError(f"Error fetching data from OANDA for {symbol} {timeframe}: {str(e)}")


def fetch_data(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    api: Optional[REST] = None,
    data_source: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from appropriate data source (Alpaca, OANDA, or Yahoo Finance).
    
    Automatically determines data source based on symbol format, or uses specified source.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe string (e.g., '1Min', '5Min', '1Hour', '1Day')
        start: Start date for data collection
        end: End date for data collection
        api: Optional REST API client for Alpaca (creates new if not provided)
        data_source: Optional data source override ('alpaca', 'oanda', or 'yahoo_finance')
        
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    # Determine data source if not specified
    if data_source is None:
        data_source = determine_data_source(symbol)
    
    # Route to appropriate data source
    if data_source == 'alpaca':
        return fetch_from_alpaca(symbol, timeframe, start, end, api)
    elif data_source == 'oanda':
        return fetch_from_oanda(symbol, timeframe, start, end, api)
    elif data_source == 'yahoo_finance':
        return fetch_from_yahoo_finance(symbol, timeframe, start, end)
    else:
        raise ValueError(f"Unknown data source: {data_source}. Use 'alpaca', 'oanda', or 'yahoo_finance'")


# ============================================================================
# Data Cleaning
# ============================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data: sort, remove duplicates, convert types, handle missing values.
    
    Args:
        df: Raw DataFrame from API
        
    Returns:
        Cleaned DataFrame
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Convert timestamp to datetime and ensure UTC
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    elif df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    # Sort chronologically by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Drop duplicates (keep first occurrence)
    initial_count = len(df)
    df = df.drop_duplicates(subset=['timestamp'], keep='first')
    duplicates_removed = initial_count - len(df)
    
    if duplicates_removed > 0 and config.VERBOSE:
        print(f"  Removed {duplicates_removed} duplicate rows")
    
    # Convert OHLCV columns to numeric
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows where essential price data is missing
    essential_cols = ['open', 'high', 'low', 'close']
    missing_essential = df[essential_cols].isna().any(axis=1)
    if missing_essential.any():
        df = df[~missing_essential].reset_index(drop=True)
        if config.VERBOSE:
            print(f"  Removed {missing_essential.sum()} rows with missing essential price data")
    
    # Forward-fill small gaps in volume (optional, for minor gaps)
    if 'volume' in df.columns:
        df['volume'] = df['volume'].fillna(0)
    
    # Ensure timestamp is the index for easier time-based operations
    df = df.set_index('timestamp').sort_index()
    
    return df


# ============================================================================
# Data Validation
# ============================================================================

def validate_data(df: pd.DataFrame, timeframe: str) -> Tuple[bool, dict]:
    """
    Validate data quality: check frequency, coverage, and basic stats.
    
    Args:
        df: DataFrame to validate
        timeframe: Expected timeframe string
        
    Returns:
        Tuple of (is_valid, validation_info_dict)
    """
    validation_info = {
        'valid': True,
        'rows': len(df),
        'start_date': None,
        'end_date': None,
        'date_range_days': None,
        'missing_percentage': 0.0,
        'warnings': []
    }
    
    if df.empty:
        validation_info['valid'] = False
        validation_info['warnings'].append("DataFrame is empty")
        return False, validation_info
    
    # Check minimum data points
    if len(df) < config.MIN_DATA_POINTS:
        validation_info['warnings'].append(
            f"Only {len(df)} rows, below minimum of {config.MIN_DATA_POINTS}"
        )
        validation_info['valid'] = False
    
    # Date range
    validation_info['start_date'] = df.index.min()
    validation_info['end_date'] = df.index.max()
    validation_info['date_range_days'] = (validation_info['end_date'] - validation_info['start_date']).days
    
    # Check for missing timestamps (gaps in time series)
    timeframe_map = {
        '1Min': '1T',
        '5Min': '5T',
        '15Min': '15T',
        '30Min': '30T',
        '1Hour': '1H',
        '4Hour': '4H',
        '1Day': '1D'
    }
    
    expected_freq = timeframe_map.get(timeframe, '1D')
    expected_range = pd.date_range(
        start=validation_info['start_date'],
        end=validation_info['end_date'],
        freq=expected_freq,
        tz='UTC'
    )
    
    missing_timestamps = set(expected_range) - set(df.index)
    missing_count = len(missing_timestamps)
    total_expected = len(expected_range)
    
    if missing_count > 0:
        missing_pct = (missing_count / total_expected) * 100
        validation_info['missing_percentage'] = missing_pct
        
        if missing_pct > config.MAX_MISSING_DATA_PERCENTAGE * 100:
            validation_info['warnings'].append(
                f"Missing {missing_pct:.2f}% of expected timestamps ({missing_count} gaps)"
            )
            validation_info['valid'] = False
    
    # Check for missing values in price columns
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                missing_pct = (missing / len(df)) * 100
                validation_info['warnings'].append(
                    f"Column '{col}' has {missing_pct:.2f}% missing values"
                )
    
    return validation_info['valid'], validation_info


def verify_data_integrity(df: pd.DataFrame) -> dict:
    """
    Verify data integrity by detecting missing timestamps and anomalies.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Dictionary with integrity check results
    """
    integrity = {
        'has_gaps': False,
        'gap_count': 0,
        'duplicate_timestamps': 0,
        'negative_prices': 0,
        'invalid_ohlc': 0
    }
    
    if df.empty:
        return integrity
    
    # Check for duplicate timestamps
    integrity['duplicate_timestamps'] = df.index.duplicated().sum()
    
    # Check for negative or zero prices (count rows with any invalid price)
    price_cols = ['open', 'high', 'low', 'close']
    available_price_cols = [col for col in price_cols if col in df.columns]
    if available_price_cols:
        invalid_prices = (df[available_price_cols] <= 0).any(axis=1).sum()
        integrity['negative_prices'] = invalid_prices
    
    # Check OHLC logic (high >= low, high >= open, high >= close, etc.)
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        ).sum()
        integrity['invalid_ohlc'] = invalid_ohlc
    
    return integrity


def print_summary(df: pd.DataFrame, symbol: str, timeframe: str):
    """
    Print summary information about collected data.
    
    Args:
        df: DataFrame with collected data
        symbol: Trading symbol
        timeframe: Timeframe string
    """
    if df.empty:
        print(f"{symbol} {timeframe}: No data collected")
        return
    
    start_date = df.index.min().strftime(config.DATE_FORMAT)
    end_date = df.index.max().strftime(config.DATE_FORMAT)
    row_count = len(df)
    
    print(f"\n{'='*60}")
    print(f"{symbol} {timeframe} Data Summary")
    print(f"{'='*60}")
    print(f"Records: {row_count:,}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Timeframe: {timeframe}")
    
    if 'close' in df.columns:
        print(f"Price Range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    if 'volume' in df.columns:
        print(f"Total Volume: {df['volume'].sum():,.0f}")
    
    # Check data coverage
    is_valid, validation_info = validate_data(df, timeframe)
    if validation_info['missing_percentage'] > 0:
        print(f"Data Coverage: {100 - validation_info['missing_percentage']:.1f}%")
    
    if validation_info['warnings']:
        print("\n⚠️  Warnings:")
        for warning in validation_info['warnings']:
            print(f"   - {warning}")
    else:
        print("✓ Data validation passed")
    
    print(f"{'='*60}\n")


# ============================================================================
# Data Saving
# ============================================================================

def save_data(df: pd.DataFrame, symbol: str, timeframe: str) -> str:
    """
    Save cleaned data to disk using naming convention from config.
    
    Args:
        df: DataFrame to save
        symbol: Trading symbol (may contain special chars like =, /, -)
        timeframe: Timeframe string
        
    Returns:
        Path to saved file
    """
    # Ensure directory exists
    os.makedirs(config.RAW_DATA_PATH, exist_ok=True)
    
    # Sanitize symbol for filename (replace special chars with underscores)
    # e.g., "XAUUSD=X" -> "XAUUSD_X", "BTC/USD" -> "BTC_USD", "EUR/USD" -> "EUR_USD"
    sanitized_symbol = symbol.replace('=', '_').replace('/', '_').replace('-', '_')
    
    # Generate file path using config helper with sanitized symbol
    file_path = config.get_raw_data_path(sanitized_symbol, timeframe)
    
    # Ensure we have a timestamp column
    if df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
        df_to_save = df.reset_index()
    else:
        df_to_save = df.copy()
    
    # Ensure timestamp column exists
    if 'timestamp' not in df_to_save.columns:
        if 'time' in df_to_save.columns:
            df_to_save.rename(columns={'time': 'timestamp'}, inplace=True)
        else:
            raise ValueError(f"No timestamp column found in DataFrame for {symbol} {timeframe}")
    
    # Save to CSV
    try:
        df_to_save.to_csv(file_path, index=False)
        
        # Verify file was created
        if not os.path.exists(file_path):
            raise IOError(f"File was not created at {file_path}")
        
        # Verify file has content
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise IOError(f"File was created but is empty: {file_path}")
        
        if config.VERBOSE:
            print(f"✓ Saved {len(df_to_save)} rows to: {file_path} ({file_size:,} bytes)")
        
        return file_path
    
    except Exception as e:
        print(f"❌ Error saving data to {file_path}: {str(e)}")
        raise


# ============================================================================
# Main Workflow
# ============================================================================

def collect_data_for_symbol(
    symbol: str,
    timeframes: list = None,
    start: datetime = None,
    end: datetime = None,
    save: bool = True
) -> dict:
    """
    Main workflow: collect data for a symbol across multiple timeframes.
    
    Args:
        symbol: Trading symbol to collect
        timeframes: List of timeframes (defaults to config.DEFAULT_TIMEFRAMES)
        start: Start date (defaults to config.HISTORICAL_START_DATE)
        end: End date (defaults to config.HISTORICAL_END_DATE)
        save: Whether to save data to disk
        
    Returns:
        Dictionary mapping timeframe to DataFrame
    """
    if timeframes is None:
        timeframes = config.DEFAULT_TIMEFRAMES
    
    if start is None:
        start = config.HISTORICAL_START_DATE
    
    if end is None:
        end = config.HISTORICAL_END_DATE
    
    # Determine data source
    data_source = determine_data_source(symbol)
    
    # Connect to API only if using Alpaca or OANDA
    api = None
    oanda_api = None
    if data_source == 'alpaca':
        try:
            api = connect_to_alpaca()
        except Exception as e:
            print(f"Warning: Could not connect to Alpaca: {str(e)}")
            print(f"Attempting to use Yahoo Finance instead...")
            data_source = 'yahoo_finance'
    elif data_source == 'oanda':
        try:
            oanda_api = connect_to_oanda()
        except Exception as e:
            print(f"Warning: Could not connect to OANDA: {str(e)}")
            print(f"Attempting to use Yahoo Finance instead...")
            data_source = 'yahoo_finance'
    
    results = {}
    
    for timeframe in timeframes:
        try:
            # Fetch data from appropriate source
            # Use oanda_api for OANDA, api for Alpaca
            if data_source == 'oanda':
                df = fetch_data(symbol, timeframe, start, end, oanda_api, data_source)
            else:
                df = fetch_data(symbol, timeframe, start, end, api, data_source)
            
            # Check if data was fetched
            if df.empty:
                print(f"⚠️  No data returned for {symbol} {timeframe}")
                results[timeframe] = pd.DataFrame()
                continue
            
            # Clean data
            df = clean_data(df)
            
            # Check if data is still valid after cleaning
            if df.empty:
                print(f"⚠️  Data became empty after cleaning for {symbol} {timeframe}")
                results[timeframe] = pd.DataFrame()
                continue
            
            # Validate data
            is_valid, validation_info = validate_data(df, timeframe)
            
            if not is_valid and config.VERBOSE:
                print(f"⚠️  Validation warnings for {symbol} {timeframe}:")
                for warning in validation_info['warnings']:
                    print(f"   - {warning}")
            
            # Save data to data/raw directory
            if save and not df.empty:
                try:
                    saved_path = save_data(df, symbol, timeframe)
                    if os.path.exists(saved_path):
                        print(f"✓ Successfully saved {symbol} {timeframe} to: {saved_path}")
                    else:
                        print(f"⚠️  Warning: File was not created at {saved_path}")
                except Exception as save_error:
                    print(f"❌ Error saving {symbol} {timeframe}: {str(save_error)}")
                    import traceback
                    traceback.print_exc()
            
            # Store result
            results[timeframe] = df
            
            # Print summary
            if config.VERBOSE:
                print_summary(df, symbol, timeframe)
        
        except Exception as e:
            print(f"❌ Error collecting {symbol} {timeframe}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[timeframe] = pd.DataFrame()
            continue
    
    return results


# ============================================================================
# Script Entry Point
# ============================================================================

if __name__ == "__main__":
    """
    Main entry point when running script directly.
    Collects data for default symbol and timeframes from config.
    """
    print("="*60)
    print("AI-Driven Quantitative Trading Research System")
    print("Data Collection Module")
    print("="*60)
    
    # Collect data for default symbol
    symbol = config.DEFAULT_SYMBOL
    timeframes = config.DEFAULT_TIMEFRAMES
    start = config.HISTORICAL_START_DATE
    end = config.HISTORICAL_END_DATE
    
    print(f"\nCollecting data for: {symbol}")
    print(f"Timeframes: {', '.join(timeframes)}")
    print(f"Date Range: {start.date()} to {end.date()}")
    print()
    
    results = collect_data_for_symbol(
        symbol=symbol,
        timeframes=timeframes,
        start=start,
        end=end,
        save=True
    )
    
    # Summary
    successful = sum(1 for df in results.values() if not df.empty)
    total = len(results)
    
    print(f"\n{'='*60}")
    print(f"Collection Complete: {successful}/{total} timeframes successful")
    print(f"{'='*60}")

