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
import time
import pandas as pd
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config_manager import (
    load_config,
    validate_config,
    get_setting,
    get_processed_data_file_path,
    get_regime_labels_path,
    get_regime_split_paths,
    get_model_paths,
    get_predictions_path,
    get_sanitized_symbol,
)
from src.data_collection import collect_data_for_asset
from scripts.process_all_data import process_asset_data
from src.indicator_testing import run_indicator_tests
from src.calculate_regime_specific_metrics import calculate_metrics_for_asset
from src.regime_labeling import run_regime_labeling
from src.data_preparation import prepare_data
from src.train_regime_model import train_model, generate_training_summary
from src.evaluate_regime_model import evaluate_model
from src.full_dataset_prediction import predict_full_dataset
from src.robustness_testing import test_robustness
from src.unsupervised_validation import kmeans_validation
from src.report_generator import generate_complete_report
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
    
    return collect_data_for_asset(config)


def step2_calculate_indicators(config):
    """Step 2: Calculate technical indicators."""
    print_step(2, "INDICATOR CALCULATION")
    
    results = process_asset_data(config)
    successful = sum(1 for success in results.values() if success)
    print(f"✓ Indicators calculated for {successful}/{len(results)} timeframes")
    return results


def step3_statistical_analysis(config):
    """Step 3: Run indicator testing and other statistical analyses."""
    print_step(3, "STATISTICAL ANALYSIS")
    
    symbol = get_setting(config, 'asset.symbol')
    primary_timeframe = get_setting(config, 'data.primary_timeframe')
    
    print(f"Running statistical analyses for: {symbol} ({primary_timeframe})")
    
    print("  Testing indicators...")
    indicator_outputs = run_indicator_tests(config, timeframe=primary_timeframe)
    print("✓ Indicator testing complete")
    
    print("✓ Statistical analysis complete")
    return {
        'indicator_outputs': indicator_outputs,
    }


def step4_regime_classification(config):
    """Step 4: Train regime classification model."""
    print_step(4, "REGIME CLASSIFICATION")
    
    if not get_setting(config, 'analysis.regime_classification', False):
        print("Regime classification disabled in config. Skipping...")
        return
    
    symbol = get_setting(config, 'asset.symbol')
    primary_timeframe = get_setting(config, 'data.primary_timeframe')
    adx_threshold = get_setting(config, 'analysis.adx_threshold')
    sanitized_symbol = get_sanitized_symbol(config)
    
    print(f"Training regime classification model for: {symbol}")
    print(f"ADX Threshold: {adx_threshold}")
    
    data_path = get_processed_data_file_path(config, primary_timeframe)
    regime_labels_path = get_regime_labels_path(config)
    model_paths = get_model_paths(config)
    split_paths = get_regime_split_paths(config)
    predictions_path = get_predictions_path(config)
    unsupervised_output_path = os.path.join('models', f"unsupervised_validation_{sanitized_symbol}.json")
    
    # Label regimes
    print("  Labeling regimes...")
    df_labeled = run_regime_labeling(
        data_path=data_path,
        output_path=regime_labels_path,
        adx_threshold=adx_threshold,
        asset_config=config
    )
    
    # Prepare data splits
    print("  Preparing data splits...")
    prep_result = prepare_data(data_path=regime_labels_path, asset_config=config)
    split_paths = prep_result['paths']
    
    # Train model
    print("  Training model...")
    training_results = train_model(
        train_path=split_paths['train'],
        val_path=split_paths['validation'],
        model_save_path=model_paths['model'],
        history_save_path=model_paths['history'],
        asset_config=config
    )
    
    generate_training_summary(
        training_results['history_save_path'],
        summary_path=model_paths['summary'],
        asset_config=config
    )
    
    # Evaluate model
    print("  Evaluating model...")
    evaluation_results = evaluate_model(
        split_paths['test'],
        model_path=model_paths['model'],
        output_dir='models',
        asset_config=config
    )
    
    # Generate predictions
    print("  Generating full dataset predictions...")
    predict_full_dataset(
        data_path=regime_labels_path,
        model_path=model_paths['model'],
        norm_params_path=model_paths['normalization'],
        output_path=predictions_path,
        asset_config=config
    )
    
    # Robustness testing
    print("  Running robustness testing...")
    robustness_results = test_robustness(
        test_path=split_paths['test'],
        model_path=model_paths['model'],
        norm_params_path=model_paths['normalization'],
        output_path=model_paths['robustness'],
        asset_config=config
    )
    
    # Unsupervised validation
    print("  Running unsupervised validation (K-Means)...")
    test_df = pd.read_csv(split_paths['test'])
    kmeans_validation(
        test_df,
        n_clusters=3,
        output_path=unsupervised_output_path,
        asset_config=config
    )
    
    print("✓ Regime classification complete")
    return {
        'training': training_results,
        'evaluation': evaluation_results,
        'robustness': robustness_results,
        'predictions_path': predictions_path,
        'unsupervised_output': unsupervised_output_path,
    }


def step5_regime_metrics(config):
    """Step 5: Calculate regime-specific indicator metrics once predictions exist."""
    print_step(5, "REGIME-SPECIFIC METRICS")

    predictions_path = get_predictions_path(config)

    if not os.path.exists(predictions_path):
        print("Regime predictions file not found. Skipping regime-specific metrics.")
        print("→ Run regime classification or set analysis.regime_classification to false to avoid this step.")
        return None

    metrics_outputs = calculate_metrics_for_asset(config)
    print("✓ Regime-specific metrics calculated")
    return metrics_outputs


def step6_generate_report(config):
    """Step 6: Generate comprehensive report."""
    print_step(6, "REPORT GENERATION")
    
    asset_name = get_setting(config, 'asset.name')
    asset_symbol = get_setting(config, 'asset.symbol')
    report_name = get_setting(config, 'output.report_name')
    
    print(f"Generating report for: {asset_name} ({asset_symbol})")
    report_path = os.path.join('reports', f"{report_name}.md")
    
    report_output = generate_complete_report(output_path=report_path, config=config)
    print(f"✓ Report generated at: {report_output}")
    return report_output


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
        data_collection_results = step1_collect_data(config)
        indicator_calculation_results = step2_calculate_indicators(config)
        statistical_results = step3_statistical_analysis(config)
        regime_results = step4_regime_classification(config)
        regime_enabled = get_setting(config, 'analysis.regime_classification', False)
        metrics_results = step5_regime_metrics(config)

        report_path = step6_generate_report(config)
        
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
        regime_enabled = get_setting(config, 'analysis.regime_classification', False)

        print("\nKey Outputs:")
        if statistical_results:
            details = statistical_results.get('indicator_outputs', {})
            for key, path in details.items():
                print(f"  Indicator {key}: {path}")
        if metrics_results:
            for key, path in metrics_results.items():
                print(f"  Regime metrics {key}: {path}")
        if regime_enabled:
            print(f"  Regime labels: {get_regime_labels_path(config)}")
            print(f"  Train set: {get_regime_split_paths(config)['train']}")
            print(f"  Model: {get_model_paths(config)['model']}")
            print(f"  Predictions: {get_predictions_path(config)}")
            if regime_results:
                predictions_output = regime_results.get('predictions_path')
                if predictions_output:
                    print(f"  Regime predictions: {predictions_output}")
                training_info = regime_results.get('training')
                if training_info:
                    print(f"  Training history: {training_info.get('history_save_path')}")
                print(f"  Robustness results: {get_model_paths(config)['robustness']}")
                unsupervised_path = regime_results.get('unsupervised_output')
                if unsupervised_path:
                    print(f"  Unsupervised validation: {unsupervised_path}")
        else:
            print("  Regime classification: disabled (set analysis.regime_classification = true to enable)")
        print("\n✓ All steps completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

