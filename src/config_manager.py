"""
Configuration Manager for AI-Driven Quantitative Trading Research System

This module provides functions to load, validate, and access configuration
settings from YAML files. This enables easy switching between different assets
(e.g., Gold, Bitcoin) without modifying code.

Functions:
- load_config: Load configuration from YAML file
- validate_config: Validate configuration structure and values
- get_setting: Get specific setting using dot notation (e.g., 'asset.symbol')
"""

import yaml
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file (e.g., 'configs/gold_config.yaml')
                    Can be absolute or relative to project root
        
    Returns:
        Dictionary with configuration settings
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML file is malformed
    """
    # Convert to absolute path if relative
    if not os.path.isabs(config_path):
        project_root = Path(__file__).parent.parent
        config_path = project_root / config_path
    
    config_path = str(config_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError(f"Configuration file is empty: {config_path}")
    
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure and values.
    
    Checks:
    - Required sections exist (asset, data, analysis, etc.)
    - Required fields are present and not empty
    - Date formats are valid
    - Values are within reasonable ranges
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails with descriptive error message
    """
    errors = []
    
    # Check required sections
    required_sections = ['asset', 'data', 'analysis', 'model', 'output']
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
    
    # Validate asset section
    if 'asset' in config:
        asset = config['asset']
        required_fields = ['symbol', 'name', 'market', 'display_name']
        for field in required_fields:
            if field not in asset or not asset[field]:
                errors.append(f"Missing or empty asset.{field}")
        
        # Validate market type
        valid_markets = ['commodity', 'cryptocurrency', 'forex', 'stock']
        if asset.get('market') not in valid_markets:
            errors.append(f"Invalid market type: {asset.get('market')}. Must be one of: {valid_markets}")
    
    # Validate data section
    if 'data' in config:
        data = config['data']
        required_fields = ['broker', 'start_date', 'timeframes', 'primary_timeframe']
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing data.{field}")
        
        # Validate broker
        valid_brokers = ['alpaca', 'oanda', 'yahoo_finance']
        if data.get('broker') not in valid_brokers:
            errors.append(f"Invalid broker: {data.get('broker')}. Must be one of: {valid_brokers}")
        
        # Validate dates
        if 'start_date' in data and data['start_date']:
            try:
                datetime.strptime(data['start_date'], '%Y-%m-%d')
            except ValueError:
                errors.append(f"Invalid start_date format: {data['start_date']}. Use YYYY-MM-DD")
        
        if 'end_date' in data and data['end_date']:
            try:
                datetime.strptime(data['end_date'], '%Y-%m-%d')
            except ValueError:
                errors.append(f"Invalid end_date format: {data['end_date']}. Use YYYY-MM-DD")
        
        # Validate timeframes
        valid_timeframes = ['1Min', '5Min', '15Min', '30Min', '1Hour', '4Hour', '1Day']
        if 'timeframes' in data:
            for tf in data['timeframes']:
                if tf not in valid_timeframes:
                    errors.append(f"Invalid timeframe: {tf}. Must be one of: {valid_timeframes}")
    
    # Validate analysis section
    if 'analysis' in config:
        analysis = config['analysis']
        if 'adx_threshold' in analysis:
            threshold = analysis['adx_threshold']
            if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 50:
                errors.append(f"Invalid adx_threshold: {threshold}. Must be between 0 and 50")
    
    # Validate model section
    if 'model' in config:
        model = config['model']
        if 'learning_rate' in model:
            lr = model['learning_rate']
            if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1:
                errors.append(f"Invalid learning_rate: {lr}. Must be between 0 and 1")
        
        if 'batch_size' in model:
            bs = model['batch_size']
            if not isinstance(bs, int) or bs <= 0:
                errors.append(f"Invalid batch_size: {bs}. Must be positive integer")
    
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)
    
    return True


def get_setting(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Get specific setting using dot notation.
    
    Examples:
        get_setting(config, 'asset.symbol') → "XAU/USD"
        get_setting(config, 'data.broker') → "oanda"
        get_setting(config, 'analysis.adx_threshold') → 14
        get_setting(config, 'model.hidden_sizes') → [64, 32]
    
    Args:
        config: Configuration dictionary
        path: Dot-separated path to setting (e.g., 'asset.symbol')
        default: Default value if setting not found (optional)
        
    Returns:
        Setting value or default if not found
    """
    keys = path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        if default is not None:
            return default
        raise KeyError(f"Setting not found: {path}")


def get_config_path(asset_name: str = None) -> str:
    """
    Get path to config file for given asset.
    
    Args:
        asset_name: Asset name (e.g., 'gold', 'bitcoin'). If None, returns default.
        
    Returns:
        Path to config file
    """
    if asset_name is None:
        asset_name = 'gold'  # Default to gold
    
    asset_name = asset_name.lower()
    config_file = f"{asset_name}_config.yaml"
    config_path = os.path.join('configs', config_file)
    
    return config_path


def load_asset_config(asset_name: str = None) -> Dict[str, Any]:
    """
    Convenience function to load config for a specific asset.
    
    Args:
        asset_name: Asset name (e.g., 'gold', 'bitcoin'). Defaults to 'gold'.
        
    Returns:
        Validated configuration dictionary
    """
    config_path = get_config_path(asset_name)
    config = load_config(config_path)
    validate_config(config)
    return config


# ============================================================================
# Helper functions for asset-specific paths
# ============================================================================

def sanitize_symbol(symbol: str) -> str:
    """
    Sanitize trading symbol for use in filenames.
    
    Replaces special characters with underscores.
    Examples:
        "EUR/USD" -> "EUR_USD"
        "BTC-USD" -> "BTC_USD"
        "XAUUSD=X" -> "XAUUSD_X"
    """
    if symbol is None:
        return ""
    return symbol.replace('/', '_').replace('-', '_').replace('=', '_').replace(':', '_')


def get_asset_symbol(config_dict: Dict[str, Any]) -> str:
    """Get raw asset symbol from configuration."""
    return get_setting(config_dict, 'asset.symbol')


def get_sanitized_symbol(config_dict: Dict[str, Any]) -> str:
    """Get sanitized asset symbol based on configuration."""
    return sanitize_symbol(get_asset_symbol(config_dict))


def get_primary_timeframe(config_dict: Dict[str, Any]) -> str:
    """Get primary timeframe from configuration."""
    return get_setting(config_dict, 'data.primary_timeframe')


def get_timeframes(config_dict: Dict[str, Any]) -> list:
    """Get list of timeframes from configuration."""
    return get_setting(config_dict, 'data.timeframes')


def get_raw_data_file_path(config_dict: Dict[str, Any], timeframe: str) -> str:
    """Get path to raw data file for asset and timeframe."""
    sanitized = get_sanitized_symbol(config_dict)
    return config.get_raw_data_path(sanitized, timeframe)


def get_processed_data_file_path(config_dict: Dict[str, Any], timeframe: str) -> str:
    """Get path to processed data file (with indicators) for asset and timeframe."""
    sanitized = get_sanitized_symbol(config_dict)
    return config.get_processed_data_path(sanitized, timeframe)


def get_regime_labels_path(config_dict: Dict[str, Any]) -> str:
    """Get path to regime labels CSV for asset."""
    sanitized = get_sanitized_symbol(config_dict)
    return os.path.join(config.PROCESSED_DATA_PATH, f"regime_labels_{sanitized}.csv")


def get_regime_split_paths(config_dict: Dict[str, Any]) -> Dict[str, str]:
    """Get paths to train/validation/test CSVs for asset."""
    sanitized = get_sanitized_symbol(config_dict)
    base = config.PROCESSED_DATA_PATH
    return {
        'train': os.path.join(base, f"regime_train_{sanitized}.csv"),
        'validation': os.path.join(base, f"regime_validation_{sanitized}.csv"),
        'test': os.path.join(base, f"regime_test_{sanitized}.csv"),
    }


def get_model_paths(config_dict: Dict[str, Any]) -> Dict[str, str]:
    """Get paths for model, history, normalization, and summary files."""
    sanitized = get_sanitized_symbol(config_dict)
    return {
        'model': os.path.join('models', f"regime_classifier_{sanitized}.pth"),
        'history': os.path.join('models', f"training_history_{sanitized}.json"),
        'summary': os.path.join('models', f"training_summary_{sanitized}.txt"),
        'normalization': os.path.join('models', f"normalization_params_{sanitized}.json"),
        'evaluation': os.path.join('models', f"evaluation_results_{sanitized}.json"),
        'robustness': os.path.join('models', f"robustness_results_{sanitized}.json"),
    }


def get_predictions_path(config_dict: Dict[str, Any]) -> str:
    """Get path to regime predictions CSV for asset."""
    sanitized = get_sanitized_symbol(config_dict)
    return os.path.join(config.PROCESSED_DATA_PATH, f"regime_predictions_{sanitized}.csv")


def get_indicator_output_paths(config_dict: Dict[str, Any]) -> Dict[str, str]:
    """Get paths for indicator testing outputs."""
    sanitized = get_sanitized_symbol(config_dict)
    base = config.PROCESSED_DATA_PATH
    return {
        'details': os.path.join(base, f"indicator_signal_details_{sanitized}.csv"),
        'results': os.path.join(base, f"indicator_test_results_{sanitized}.csv"),
        'report': os.path.join(base, f"indicator_test_report_{sanitized}.txt"),
    }


def get_regime_specific_metrics_paths(config_dict: Dict[str, Any]) -> Dict[str, str]:
    """Get paths for regime-specific metrics outputs."""
    sanitized = get_sanitized_symbol(config_dict)
    base = config.PROCESSED_DATA_PATH
    return {
        'csv': os.path.join(base, f"indicator_regime_specific_metrics_{sanitized}.csv"),
        'json': os.path.join(base, f"indicator_regime_specific_metrics_{sanitized}.json"),
        'report': os.path.join(base, f"indicator_regime_specific_report_{sanitized}.txt"),
    }


# Example usage
if __name__ == "__main__":
    """
    Example: Load and use Gold configuration
    """
    print("=" * 80)
    print("CONFIGURATION MANAGER EXAMPLE")
    print("=" * 80)
    print()
    
    # Load Gold config
    print("Loading Gold configuration...")
    try:
        gold_config = load_asset_config('gold')
        print("✓ Configuration loaded successfully")
        print()
        
        # Access settings using get_setting
        symbol = get_setting(gold_config, 'asset.symbol')
        name = get_setting(gold_config, 'asset.name')
        broker = get_setting(gold_config, 'data.broker')
        adx_threshold = get_setting(gold_config, 'analysis.adx_threshold')
        
        print("Configuration Settings:")
        print(f"  Symbol: {symbol}")
        print(f"  Name: {name}")
        print(f"  Broker: {broker}")
        print(f"  ADX Threshold: {adx_threshold}")
        print()
        
        # Validate config
        print("Validating configuration...")
        validate_config(gold_config)
        print("✓ Configuration is valid")
        print()
        
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
    
    print("=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)

