"""
Full Dataset Prediction Module

This module applies the trained model to the entire Gold dataset (2022-2025)
to generate hourly regime predictions across the full historical period.

"""

import argparse
import torch
import pandas as pd
import numpy as np
import json
import os
import sys
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.regime_model import RegimeClassifier
from src.data_preparation import get_feature_columns
from src.config_manager import (
    load_config,
    validate_config,
    get_model_paths,
    get_predictions_path,
    get_processed_data_file_path,
    get_regime_labels_path,
    get_setting,
    get_sanitized_symbol,
)


def predict_full_dataset(data_path: str, model_path: str, norm_params_path: str,
                         output_path: str = None,
                         asset_config: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Apply model to full dataset and generate predictions.
    
    Args:
        data_path: Path to full labeled dataset CSV
        model_path: Path to trained model
        norm_params_path: Path to normalization parameters
        output_path: Path to save predictions (optional)
        
    Returns:
        DataFrame with predictions
    """
    print("=" * 80)
    print("FULL DATASET PREDICTION (2022-2025)")
    print("=" * 80)
    
    if asset_config:
        asset_name = get_setting(asset_config, 'asset.name')
        symbol = get_setting(asset_config, 'asset.symbol')
        print(f"Asset: {asset_name} ({symbol})")
    
    model_paths = get_model_paths(asset_config) if asset_config else None
    sanitized_symbol = get_sanitized_symbol(asset_config) if asset_config else None

    if asset_config:
        if data_path is None:
            timeframe = get_setting(asset_config, 'data.primary_timeframe')
            data_path = get_processed_data_file_path(asset_config, timeframe)
        if model_path is None:
            model_path = model_paths['model']
        if norm_params_path is None:
            norm_params_path = model_paths['normalization']
        if output_path is None:
            output_path = get_predictions_path(asset_config)

    # Load full dataset
    print(f"\nLoading full dataset from: {data_path}")
    df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
    print(f"  Total samples: {len(df):,}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    
    # Get features and labels
    feature_cols = get_feature_columns(df)
    print(f"  Features: {len(feature_cols)}")
    
    # Remove rows with missing values
    df = df.dropna(subset=feature_cols + ['regime_label', 'regime_numeric'])
    print(f"  Valid samples: {len(df):,}")
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    model_config = checkpoint['model_config']
    model_feature_cols = checkpoint['feature_cols']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RegimeClassifier(
        input_size=len(model_feature_cols),
        hidden_sizes=model_config['hidden_sizes'],
        dropout_rate=model_config['dropout_rate'],
        num_classes=model_config['num_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"  Model loaded successfully")
    
    # Normalize features
    print("\nNormalizing features...")
    # Load normalization parameters
    with open(norm_params_path, 'r') as f:
        norm_params = json.load(f)
    
    norm_feature_cols = norm_params['feature_columns']
    mean = np.array(norm_params['mean'])
    scale = np.array(norm_params['scale'])
    
    # Normalize features
    df_norm = df.reset_index()
    features = df_norm[norm_feature_cols].fillna(0).values
    features_norm = (features - mean) / scale
    df_norm[norm_feature_cols] = features_norm
    
    # Prepare features
    features = df_norm[model_feature_cols].fillna(0).values.astype(np.float32)
    features_tensor = torch.from_numpy(features).to(device)
    
    # Make predictions
    print("\nMaking predictions...")
    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs, 1)
        predictions = predictions.cpu().numpy()
        probabilities = probabilities.cpu().numpy()
    
    # Create results DataFrame
    results_df = df.copy()
    results_df['ml_prediction'] = predictions
    results_df['ml_prediction_label'] = results_df['ml_prediction'].map({0: 'range', 1: 'up', 2: 'down'})
    results_df['ml_prob_range'] = probabilities[:, 0]
    results_df['ml_prob_up'] = probabilities[:, 1]
    results_df['ml_prob_down'] = probabilities[:, 2]
    results_df['ml_confidence'] = probabilities.max(axis=1)
    
    # Calculate agreement with heuristic labels
    heuristic_labels = results_df['regime_numeric'].values
    agreement = (predictions == heuristic_labels).sum() / len(predictions) * 100
    
    print(f"\nPrediction Statistics:")
    print(f"  Agreement with heuristics: {agreement:.2f}%")
    
    # Per-class agreement
    class_names = ['range', 'up', 'down']
    print(f"\nPer-Class Agreement:")
    for i, class_name in enumerate(class_names):
        class_mask = heuristic_labels == i
        if class_mask.sum() > 0:
            class_predictions = predictions[class_mask]
            class_agreement = (class_predictions == i).sum() / class_mask.sum() * 100
            print(f"  {class_name.capitalize()}: {class_agreement:.2f}% ({class_mask.sum():,} samples)")
    
    # Save results
    if output_path:
        results_df.to_csv(output_path)
        print(f"\n✓ Saved predictions to: {output_path}")
    
    print("\n" + "=" * 80)
    print("FULL DATASET PREDICTION COMPLETE")
    print("=" * 80)
    
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate full dataset predictions for a configured asset.")
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
        
        data_path = get_regime_labels_path(asset_config)
        model_paths = get_model_paths(asset_config)
        output_path = get_predictions_path(asset_config)
        
        predict_full_dataset(
            data_path=data_path,
            model_path=model_paths['model'],
            norm_params_path=model_paths['normalization'],
            output_path=output_path,
            asset_config=asset_config
        )
    except Exception as error:
        print(f"Error: {error}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

