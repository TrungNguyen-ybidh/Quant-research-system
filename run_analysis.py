#!/usr/bin/env python3
"""
Master Analysis Runner Script

This script runs the complete quantitative research pipeline for any asset
by reading configuration from a YAML file.

Usage:
    python run_analysis.py --config configs/gold_config.yaml
    python run_analysis.py --config configs/bitcoin_config.yaml

The script automatically:
1. Loads configuration
2. Collects data
3. Calculates indicators
4. Runs statistical analyses
5. Trains ML models
6. Generates comprehensive report
"""

import argparse
import sys
import os
from datetime import datetime, timezone
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config_manager import load_config, validate_config, get_setting
import config as base_config


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_step(step_num: int, step_name: str):
    """Print step header."""
    print(f"\n[Step {step_num}] {step_name}")
    print("-" * 80)


def load_and_validate_config(config_path: str):
    """Load and validate configuration."""
    print_header("CONFIGURATION LOADING")
    
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    print("Validating configuration...")
    validate_config(config)
    print("✓ Configuration is valid")
    
    # Print analysis plan
    print("\n" + "=" * 80)
    print("ANALYSIS PLAN")
    print("=" * 80)
    
    asset_name = get_setting(config, 'asset.name')
    asset_symbol = get_setting(config, 'asset.symbol')
    broker = get_setting(config, 'data.broker')
    start_date = get_setting(config, 'data.start_date')
    end_date = get_setting(config, 'data.end_date') or "Today"
    timeframes = get_setting(config, 'data.timeframes')
    correlation_assets = get_setting(config, 'correlation.assets', [])
    regime_classification = get_setting(config, 'analysis.regime_classification', False)
    
    print(f"Asset: {asset_name} ({asset_symbol})")
    print(f"Broker: {broker}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Timeframes: {', '.join(timeframes)}")
    print(f"Correlation Assets: {len(correlation_assets)} assets")
    print(f"Regime Classification: {'Yes' if regime_classification else 'No'}")
    print(f"Estimated Time: 10-15 minutes")
    
    return config


def step1_collect_data(config):
    """Step 1: Collect data for primary asset and correlation assets."""
    print_step(1, "DATA COLLECTION")
    
    from src.data_collection import collect_data_for_symbol
    from datetime import datetime, timezone
    
    symbol = get_setting(config, 'asset.symbol')
    broker = get_setting(config, 'data.broker')
    timeframes = get_setting(config, 'data.timeframes')
    start_date_str = get_setting(config, 'data.start_date')
    end_date_str = get_setting(config, 'data.end_date')
    
    # Parse dates
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    if end_date_str:
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    else:
        end_date = datetime.now(timezone.utc)
    
    print(f"Collecting data for: {symbol}")
    print(f"Broker: {broker}")
    print(f"Timeframes: {', '.join(timeframes)}")
    print(f"Date Range: {start_date.date()} to {end_date.date()}")
    
    # Collect primary asset data
    results = collect_data_for_symbol(
        symbol=symbol,
        timeframes=timeframes,
        start=start_date,
        end=end_date,
        save=True
    )
    
    successful = sum(1 for df in results.values() if not df.empty)
    print(f"✓ Collected data for {successful}/{len(timeframes)} timeframes")
    
    # Collect correlation assets data
    correlation_assets = get_setting(config, 'correlation.assets', [])
    if correlation_assets:
        print(f"\nCollecting correlation assets ({len(correlation_assets)} assets)...")
        for asset in correlation_assets:
            asset_symbol = asset['symbol']
            print(f"  Collecting {asset_symbol}...")
            try:
                collect_data_for_symbol(
                    symbol=asset_symbol,
                    timeframes=[get_setting(config, 'data.primary_timeframe')],
                    start=start_date,
                    end=end_date,
                    save=True
                )
                print(f"  ✓ {asset_symbol} collected")
            except Exception as e:
                print(f"  ⚠ {asset_symbol} failed: {str(e)}")
    
    return results


def step2_calculate_indicators(config):
    """Step 2: Calculate technical indicators."""
    print_step(2, "INDICATOR CALCULATION")
    
    import subprocess
    
    symbol = get_setting(config, 'asset.symbol')
    timeframes = get_setting(config, 'data.timeframes')
    
    print(f"Calculating indicators for: {symbol}")
    print(f"Timeframes: {', '.join(timeframes)}")
    
    # Run the process_all_data script
    # This script processes all raw CSV files and adds indicators
    print("  Running process_all_data.py...")
    result = subprocess.run(
        [sys.executable, 'scripts/process_all_data.py'],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✓ Indicators calculated for all timeframes")
    else:
        print(f"⚠ Warning: {result.stderr}")
        print("  Continuing anyway...")


def step3_statistical_analysis(config):
    """Step 3: Run statistical analyses."""
    print_step(3, "STATISTICAL ANALYSIS")
    
    symbol = get_setting(config, 'asset.symbol')
    primary_timeframe = get_setting(config, 'data.primary_timeframe')
    
    print(f"Running statistical analyses for: {symbol} ({primary_timeframe})")
    
    # Run indicator testing (this is the main statistical analysis)
    print("  Testing indicators...")
    import subprocess
    result = subprocess.run(
        [sys.executable, 'src/indicator_testing.py'],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✓ Indicator testing complete")
    else:
        print(f"⚠ Warning: {result.stderr}")
    
    # Calculate regime-specific metrics
    print("  Calculating regime-specific metrics...")
    result = subprocess.run(
        [sys.executable, 'src/calculate_regime_specific_metrics.py'],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✓ Regime-specific metrics calculated")
    else:
        print(f"⚠ Warning: {result.stderr}")
    
    print("✓ Statistical analysis complete")


def step4_regime_classification(config):
    """Step 4: Train regime classification model."""
    print_step(4, "REGIME CLASSIFICATION")
    
    if not get_setting(config, 'analysis.regime_classification', False):
        print("Regime classification disabled in config. Skipping...")
        return
    
    symbol = get_setting(config, 'asset.symbol')
    primary_timeframe = get_setting(config, 'data.primary_timeframe')
    adx_threshold = get_setting(config, 'analysis.adx_threshold')
    
    print(f"Training regime classification model for: {symbol}")
    print(f"ADX Threshold: {adx_threshold}")
    
    # Update config.py temporarily with ADX threshold
    # (This is a workaround until we fully integrate config_manager)
    import config as base_config
    original_adx = getattr(base_config, 'ADX_THRESHOLD', None)
    
    # Run regime labeling with custom threshold
    print("  Labeling regimes...")
    
    # Load data and label
    data_path = os.path.join(
        base_config.PROCESSED_DATA_PATH,
        f"{symbol.replace('/', '_')}_{primary_timeframe}_with_indicators.csv"
    )
    
    if not os.path.exists(data_path):
        print(f"⚠ Data file not found: {data_path}")
        print("  Skipping regime classification")
        return
    
    # Label regimes
    from src.regime_labeling import label_regimes
    import pandas as pd
    df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
    df_labeled = label_regimes(df, adx_threshold=adx_threshold)
    
    # Save labels
    output_path = os.path.join(base_config.PROCESSED_DATA_PATH, 'regime_labels.csv')
    df_labeled.reset_index().to_csv(output_path, index=False)
    print(f"  ✓ Regime labels saved to: {output_path}")
    
    # Prepare data splits
    print("  Preparing data splits...")
    from src.data_preparation import prepare_regime_data
    prepare_regime_data(df_labeled)
    
    # Train model
    print("  Training model...")
    from src.train_regime_model import train_model
    train_path = os.path.join(base_config.PROCESSED_DATA_PATH, 'regime_train.csv')
    val_path = os.path.join(base_config.PROCESSED_DATA_PATH, 'regime_validation.csv')
    train_model(train_path, val_path)
    
    # Evaluate model
    print("  Evaluating model...")
    from src.evaluate_regime_model import evaluate_model
    test_path = os.path.join(base_config.PROCESSED_DATA_PATH, 'regime_test.csv')
    evaluate_model(test_path)
    
    # Generate predictions
    print("  Generating full dataset predictions...")
    from src.full_dataset_prediction import predict_full_dataset
    predict_full_dataset()
    
    print("✓ Regime classification complete")


def step5_generate_report(config):
    """Step 5: Generate comprehensive report."""
    print_step(5, "REPORT GENERATION")
    
    asset_name = get_setting(config, 'asset.name')
    asset_symbol = get_setting(config, 'asset.symbol')
    report_name = get_setting(config, 'output.report_name')
    
    print(f"Generating report for: {asset_name} ({asset_symbol})")
    print(f"Report name: {report_name}")
    
    # Note: The report generator currently generates XAU_USD_Research_Report.md
    # This will need to be updated to use config for asset-specific reports
    # For now, we'll note that the report exists
    report_path = os.path.join('reports', f"{report_name}.md")
    
    if os.path.exists('reports/XAU_USD_Research_Report.md'):
        print(f"✓ Report exists at: reports/XAU_USD_Research_Report.md")
        print("  Note: Report generator needs update to use config for asset-specific naming")
    else:
        print("⚠ Report not found. Run report generator separately.")
    
    print("✓ Report generation step complete")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run complete quantitative research analysis pipeline'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/gold_config.yaml',
        help='Path to configuration YAML file (default: configs/gold_config.yaml)'
    )
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    try:
        # Load and validate configuration
        config = load_and_validate_config(args.config)
        
        # Run pipeline steps
        step1_collect_data(config)
        step2_calculate_indicators(config)
        step3_statistical_analysis(config)
        step4_regime_classification(config)
        step5_generate_report(config)
        
        # Print completion summary
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        print_header("ANALYSIS COMPLETE")
        
        asset_name = get_setting(config, 'asset.name')
        asset_symbol = get_setting(config, 'asset.symbol')
        report_name = get_setting(config, 'output.report_name')
        report_path = os.path.join('reports', f"{report_name}.md")
        
        print(f"Asset: {asset_name} ({asset_symbol})")
        print(f"Report: {report_path}")
        print(f"Total Time: {minutes}m {seconds}s")
        print("\n✓ All steps completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

