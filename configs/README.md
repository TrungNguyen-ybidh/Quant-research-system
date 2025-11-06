# Configuration Files Guide

This folder contains YAML configuration files for different assets. Each config file defines all settings needed to analyze a specific asset.

## Available Configs

- **gold_config.yaml** - Configuration for Gold (XAU/USD) analysis
- **bitcoin_config.yaml** - Example configuration for Bitcoin (BTC-USD) analysis
- **template_config.yaml** - Template for creating new asset configurations

## Using Configs

### Load a Config in Python

```python
from src.config_manager import load_asset_config, get_setting

# Load Gold config
config = load_asset_config('gold')

# Access settings using dot notation
symbol = get_setting(config, 'asset.symbol')  # "XAU/USD"
broker = get_setting(config, 'data.broker')   # "oanda"
adx_threshold = get_setting(config, 'analysis.adx_threshold')  # 14
```

### Create a New Asset Config

1. Copy the template:
   ```bash
   cp configs/template_config.yaml configs/my_asset_config.yaml
   ```

2. Edit `my_asset_config.yaml` and fill in:
   - Asset symbol and name
   - Data source (broker)
   - Date ranges
   - Indicator parameters
   - Model settings

3. Use it in your code:
   ```python
   config = load_asset_config('my_asset')
   ```

## Config Structure

Each config file has these sections:

- **asset**: Asset information (symbol, name, market type)
- **data**: Data collection settings (broker, dates, timeframes)
- **correlation**: Assets to compare against
- **analysis**: Analysis parameters (ADX threshold, indicators)
- **model**: ML model settings (architecture, training params)
- **output**: Output settings (report name, visualization paths)

## Validation

Configs are automatically validated when loaded. The validator checks:
- Required sections and fields exist
- Date formats are valid
- Broker names are valid
- Timeframes are valid
- Parameter ranges are reasonable

If validation fails, you'll get a clear error message indicating what needs to be fixed.

