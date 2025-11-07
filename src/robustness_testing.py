"""
Robustness Testing Module for Regime Classification Model

This module implements robustness testing using the SAFE framework:
- Perturb prices (±5%) and volumes (±10%)
- Recalculate technical indicators
- Test model performance on perturbed data
- Compute robustness metrics

"""
import argparse
import torch
import pandas as pd
import numpy as np
import json
import os
import sys
from typing import Dict, Tuple, Any
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.regime_model import RegimeClassifier
from src.data_preparation import get_feature_columns
from src.indicators import add_all_indicators
from src.config_manager import (
    load_config,
    validate_config,
    get_model_paths,
    get_regime_split_paths,
    get_processed_data_file_path,
    get_predictions_path,
    get_setting,
    get_sanitized_symbol,
)


def perturb_data(df: pd.DataFrame, price_noise: float = 0.05, volume_noise: float = 0.10) -> pd.DataFrame:
    """
    Perturb data by adding random noise to prices and volumes.
    
    Args:
        df: Original DataFrame with OHLCV data
        price_noise: Price noise level (default: 0.05 = ±5%)
        volume_noise: Volume noise level (default: 0.10 = ±10%)
        
    Returns:
        Perturbed DataFrame
    """
    perturbed_df = df.copy()
    
    # Perturb prices (open, high, low, close)
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in perturbed_df.columns:
            noise = np.random.uniform(-price_noise, price_noise, size=len(perturbed_df))
            perturbed_df[col] = perturbed_df[col] * (1 + noise)
    
    # Perturb volume
    if 'volume' in perturbed_df.columns:
        noise = np.random.uniform(-volume_noise, volume_noise, size=len(perturbed_df))
        perturbed_df['volume'] = perturbed_df['volume'] * (1 + noise)
        perturbed_df['volume'] = perturbed_df['volume'].clip(lower=0)  # Ensure non-negative
    
    return perturbed_df


def recalculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalculate all technical indicators from perturbed OHLCV data.
    
    Args:
        df: DataFrame with perturbed OHLCV data
        
    Returns:
        DataFrame with recalculated indicators
    """
    # Ensure timestamp is index
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    
    # Recalculate all indicators
    df_with_indicators = add_all_indicators(df)
    
    return df_with_indicators


def normalize_features(df: pd.DataFrame, norm_params_path: str) -> pd.DataFrame:
    """
    Normalize features using saved normalization parameters.
    
    Args:
        df: DataFrame with features
        norm_params_path: Path to normalization parameters JSON
        
    Returns:
        Normalized DataFrame
    """
    # Load normalization parameters
    with open(norm_params_path, 'r') as f:
        norm_params = json.load(f)
    
    feature_cols = norm_params['feature_columns']
    mean = np.array(norm_params['mean'])
    scale = np.array(norm_params['scale'])
    
    # Normalize features
    df_norm = df.copy()
    features = df_norm[feature_cols].fillna(0).values
    features_norm = (features - mean) / scale
    df_norm[feature_cols] = features_norm
    
    return df_norm


def calculate_robustness_metrics(y_true: np.ndarray, y_pred_clean: np.ndarray, 
                                y_pred_perturbed: np.ndarray) -> Dict:
    """
    Calculate robustness metrics.
    
    Args:
        y_true: True labels
        y_pred_clean: Predictions on clean data
        y_pred_perturbed: Predictions on perturbed data
        
    Returns:
        Dictionary with robustness metrics
    """
    from sklearn.metrics import accuracy_score
    
    # 1. Perturbed accuracy
    clean_accuracy = accuracy_score(y_true, y_pred_clean)
    perturbed_accuracy = accuracy_score(y_true, y_pred_perturbed)
    accuracy_degradation = (clean_accuracy - perturbed_accuracy) / clean_accuracy * 100
    relative_loss = accuracy_degradation
    
    # 2. Prediction correlation
    correlation, p_value = pearsonr(y_pred_clean, y_pred_perturbed)
    
    # 3. Per-class robustness
    class_names = ['range', 'up', 'down']
    per_class_robustness = {}
    
    for i, class_name in enumerate(class_names):
        class_mask = y_true == i
        if class_mask.sum() > 0:
            clean_class_acc = accuracy_score(y_true[class_mask], y_pred_clean[class_mask])
            perturbed_class_acc = accuracy_score(y_true[class_mask], y_pred_perturbed[class_mask])
            class_degradation = (clean_class_acc - perturbed_class_acc) / clean_class_acc * 100 if clean_class_acc > 0 else 0
            
            per_class_robustness[class_name] = {
                'clean_accuracy': float(clean_class_acc),
                'perturbed_accuracy': float(perturbed_class_acc),
                'degradation': float(class_degradation),
                'samples': int(class_mask.sum())
            }
    
    metrics = {
        'clean_accuracy': float(clean_accuracy),
        'perturbed_accuracy': float(perturbed_accuracy),
        'accuracy_degradation': float(accuracy_degradation),
        'relative_loss': float(relative_loss),
        'prediction_correlation': float(correlation),
        'correlation_p_value': float(p_value),
        'per_class_robustness': per_class_robustness,
        'acceptable_degradation': relative_loss < 30,
        'acceptable_correlation': correlation > 0.85
    }
    
    return metrics


def test_robustness(test_path: str, model_path: str, norm_params_path: str,
                   raw_data_path: str = None, output_path: str = None,
                   asset_config: Dict[str, Any] = None) -> Dict:
    """
    Main robustness testing function.
    
    Args:
        test_path: Path to test CSV
        model_path: Path to trained model
        norm_params_path: Path to normalization parameters
        raw_data_path: Path to raw OHLCV data (for recalculating indicators)
        output_path: Path to save results (optional)
        
    Returns:
        Dictionary with robustness results
    """
    print("=" * 80)
    print("ROBUSTNESS TESTING (SAFE Framework)")
    print("=" * 80)
    
    if asset_config:
        asset_name = get_setting(asset_config, 'asset.name')
        symbol = get_setting(asset_config, 'asset.symbol')
        print(f"Asset: {asset_name} ({symbol})")
    
    # Load test data
    print(f"\nLoading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"  Test samples: {len(test_df):,}")
    
    # Get true labels
    labels = test_df['regime_numeric'].values.astype(np.int64)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    model_config = checkpoint['model_config']
    feature_cols = checkpoint['feature_cols']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RegimeClassifier(
        input_size=len(feature_cols),
        hidden_sizes=model_config['hidden_sizes'],
        dropout_rate=model_config['dropout_rate'],
        num_classes=model_config['num_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Get features for clean predictions
    print("\nMaking predictions on clean data...")
    clean_features = test_df[feature_cols].fillna(0).values.astype(np.float32)
    clean_features_tensor = torch.from_numpy(clean_features).to(device)
    
    with torch.no_grad():
        clean_outputs = model(clean_features_tensor)
        _, clean_predictions = torch.max(clean_outputs, 1)
        clean_predictions = clean_predictions.cpu().numpy()
    
    # Perturb data and recalculate indicators
    print("\nPerturbing data (prices ±5%, volumes ±10%)...")
    
    # Use test data directly for perturbation
    # Extract OHLCV columns from test data
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    raw_df = test_df[ohlcv_cols].copy()
    
    # Perturb
    perturbed_raw = perturb_data(raw_df, price_noise=0.05, volume_noise=0.10)
    
    # Add timestamp if available for indicator calculation
    if 'timestamp' in test_df.columns:
        perturbed_raw['timestamp'] = pd.to_datetime(test_df['timestamp'].values)
        perturbed_raw = perturbed_raw.set_index('timestamp')
    else:
        # Create index from test_df index
        if isinstance(test_df.index, pd.DatetimeIndex):
            perturbed_raw.index = test_df.index
        else:
            # Try to parse index as datetime
            perturbed_raw.index = pd.to_datetime(test_df.index)
    
    # Recalculate indicators
    print("Recalculating technical indicators from perturbed data...")
    perturbed_with_indicators = recalculate_indicators(perturbed_raw)
    
    # Normalize features
    print("Normalizing perturbed features...")
    perturbed_normalized = perturbed_with_indicators.reset_index()
    perturbed_normalized = normalize_features(perturbed_normalized, norm_params_path)
    
    # Make predictions on perturbed data
    print("\nMaking predictions on perturbed data...")
    perturbed_features = perturbed_normalized[feature_cols].fillna(0).values.astype(np.float32)
    perturbed_features_tensor = torch.from_numpy(perturbed_features).to(device)
    
    with torch.no_grad():
        perturbed_outputs = model(perturbed_features_tensor)
        _, perturbed_predictions = torch.max(perturbed_outputs, 1)
        perturbed_predictions = perturbed_predictions.cpu().numpy()
    
    # Calculate robustness metrics
    print("\nCalculating robustness metrics...")
    metrics = calculate_robustness_metrics(labels, clean_predictions, perturbed_predictions)
    
    # Print results
    print("\n" + "=" * 80)
    print("ROBUSTNESS RESULTS")
    print("=" * 80)
    
    print(f"\nAccuracy Metrics:")
    print(f"  Clean accuracy: {metrics['clean_accuracy']*100:.2f}%")
    print(f"  Perturbed accuracy: {metrics['perturbed_accuracy']*100:.2f}%")
    print(f"  Accuracy degradation: {metrics['accuracy_degradation']:.2f}%")
    print(f"  Relative loss: {metrics['relative_loss']:.2f}%")
    
    if metrics['acceptable_degradation']:
        print(f"  ✓ Acceptable degradation (<30%)")
    else:
        print(f"  ✗ Degradation exceeds threshold (>30%)")
    
    print(f"\nPrediction Stability:")
    print(f"  Prediction correlation: {metrics['prediction_correlation']:.4f}")
    print(f"  Correlation p-value: {metrics['correlation_p_value']:.4f}")
    
    if metrics['acceptable_correlation']:
        print(f"  ✓ Acceptable correlation (>0.85)")
    else:
        print(f"  ✗ Correlation below threshold (<0.85)")
    
    print(f"\nPer-Class Robustness:")
    class_names = ['range', 'up', 'down']
    for class_name in class_names:
        if class_name in metrics['per_class_robustness']:
            class_metrics = metrics['per_class_robustness'][class_name]
            print(f"\n  {class_name.capitalize()}:")
            print(f"    Clean accuracy: {class_metrics['clean_accuracy']*100:.2f}%")
            print(f"    Perturbed accuracy: {class_metrics['perturbed_accuracy']*100:.2f}%")
            print(f"    Degradation: {class_metrics['degradation']:.2f}%")
            print(f"    Samples: {class_metrics['samples']:,}")
    
    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Saved robustness results to: {output_path}")
    
    print("\n" + "=" * 80)
    print("ROBUSTNESS TESTING COMPLETE")
    print("=" * 80)
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run robustness testing for a configured asset.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/gold_config.yaml",
        help="Path to configuration YAML file (default: configs/gold_config.yaml)",
    )
    args = parser.parse_args()
    
    try:
        asset_config = load_config(args.config)
        validate_config(asset_config)
        
        split_paths = get_regime_split_paths(asset_config)
        model_paths = get_model_paths(asset_config)
        timeframe = get_setting(asset_config, 'data.primary_timeframe')
        sanitized_symbol = get_sanitized_symbol(asset_config)
        
        test_path = split_paths['test']
        model_path = model_paths['model']
        norm_params_path = model_paths['normalization']
        raw_data_path = config.get_raw_data_path(sanitized_symbol, timeframe)
        output_path = model_paths['robustness']
        
        test_robustness(
            test_path=test_path,
            model_path=model_path,
            norm_params_path=norm_params_path,
            raw_data_path=raw_data_path,
            output_path=output_path,
            asset_config=asset_config,
        )
    except Exception as error:
        print(f"Error: {error}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

