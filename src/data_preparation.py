"""
Data Preparation Module for Regime Classification

This module handles:
1. Splitting labeled data into train/validation/test sets (60/20/20)
2. Feature normalization (standardization)
3. Saving normalization parameters for inference

"""

import pandas as pd
import json
import os
import sys
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.config_manager import (
    get_setting,
    get_regime_labels_path,
    get_regime_split_paths,
    get_model_paths,
)


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Get list of feature columns (excluding non-feature columns).
    
    Args:
        df: DataFrame with all columns
        
    Returns:
        List of feature column names
    """
    # Exclude non-feature columns
    exclude_cols = ['timestamp', 'regime_label', 'regime_numeric', 'Unnamed: 0', 'index']
    
    # Feature columns are technical indicators
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    return feature_cols


def split_data(df: pd.DataFrame, train_ratio: float = 0.6, 
               val_ratio: float = 0.2, test_ratio: float = 0.2,
               shuffle: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/validation/test sets.
    
    Args:
        df: Full labeled dataset
        train_ratio: Training set ratio (default: 0.6)
        val_ratio: Validation set ratio (default: 0.2)
        test_ratio: Test set ratio (default: 0.2)
        shuffle: Whether to shuffle before splitting (default: False for time series)
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")
    
    # For time series data, we typically don't shuffle (preserve temporal order)
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    return train_df, val_df, test_df


def normalize_features(train_df: pd.DataFrame, val_df: pd.DataFrame, 
                      test_df: pd.DataFrame, feature_cols: list) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Normalize features using StandardScaler (fit on training data only).
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        feature_cols: List of feature column names
        
    Returns:
        Tuple of (normalized_train_df, normalized_val_df, normalized_test_df, scaler)
    """
    # Initialize scaler
    scaler = StandardScaler()
    
    # Fit scaler on training data only
    train_features = train_df[feature_cols].fillna(0).values
    scaler.fit(train_features)
    
    # Transform all datasets
    train_normalized = scaler.transform(train_features)
    val_normalized = scaler.transform(val_df[feature_cols].fillna(0).values)
    test_normalized = scaler.transform(test_df[feature_cols].fillna(0).values)
    
    # Create normalized DataFrames
    train_df_norm = train_df.copy()
    val_df_norm = val_df.copy()
    test_df_norm = test_df.copy()
    
    # Replace feature columns with normalized values
    train_df_norm[feature_cols] = train_normalized
    val_df_norm[feature_cols] = val_normalized
    test_df_norm[feature_cols] = test_normalized
    
    return train_df_norm, val_df_norm, test_df_norm, scaler


def save_normalization_params(scaler: StandardScaler, feature_cols: list, 
                              output_path: str):
    """
    Save normalization parameters to JSON file.
    
    Args:
        scaler: Fitted StandardScaler
        feature_cols: List of feature column names
        output_path: Path to save normalization parameters
    """
    params = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'feature_columns': feature_cols,
        'n_features': len(feature_cols)
    }
    
    with open(output_path, 'w') as f:
        json.dump(params, f, indent=2)


def prepare_data(data_path: str = None, output_dir: str = None,
                asset_config: Dict[str, Any] = None) -> Dict:
    """
    Main function to prepare data for training.
    
    Args:
        data_path: Path to labeled data CSV (defaults to config)
        output_dir: Output directory (defaults to config)
        
    Returns:
        Dictionary with prepared data and metadata
    """
    print("=" * 80)
    print("DATA PREPARATION: TRAIN/VALIDATION/TEST SPLIT & NORMALIZATION")
    print("=" * 80)
    
    # Set default paths
    if asset_config:
        if data_path is None:
            data_path = get_regime_labels_path(asset_config)
        if output_dir is None:
            output_dir = config.PROCESSED_DATA_PATH
        asset_name = get_setting(asset_config, 'asset.name')
        symbol = get_setting(asset_config, 'asset.symbol')
        print(f"Asset: {asset_name} ({symbol})")
    else:
        if data_path is None:
            data_path = os.path.join(config.PROCESSED_DATA_PATH, 'regime_labels.csv')
        if output_dir is None:
            output_dir = config.PROCESSED_DATA_PATH
    
    # Load labeled data
    print(f"\nLoading labeled data from: {data_path}")
    df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
    print(f"✓ Loaded {len(df)} rows")
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")
    
    # Split data (60/20/20)
    print("\nSplitting data into train/validation/test (60/20/20)...")
    train_df, val_df, test_df = split_data(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    
    print(f"  Training set: {len(train_df):,} rows ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation set: {len(val_df):,} rows ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test set: {len(test_df):,} rows ({len(test_df)/len(df)*100:.1f}%)")
    
    # Check label distribution in each split
    print("\nLabel distribution in splits:")
    for name, split_df in [("Training", train_df), ("Validation", val_df), ("Test", test_df)]:
        label_counts = split_df['regime_label'].value_counts()
        print(f"\n  {name}:")
        for regime in ['range', 'up', 'down']:
            count = label_counts.get(regime, 0)
            pct = count / len(split_df) * 100
            print(f"    {regime.capitalize()}: {count:,} ({pct:.1f}%)")
    
    # Normalize features
    print("\nNormalizing features (StandardScaler fit on training data)...")
    train_df_norm, val_df_norm, test_df_norm, scaler = normalize_features(
        train_df, val_df, test_df, feature_cols
    )
    print("✓ Features normalized")
    
    # Save normalization parameters
    if asset_config:
        model_paths = get_model_paths(asset_config)
        norm_params_path = model_paths['normalization']
    else:
        model_paths = {
            'model': os.path.join('models', 'regime_classifier.pth'),
            'history': os.path.join('models', 'training_history.json'),
            'summary': os.path.join('models', 'training_summary.txt'),
            'normalization': os.path.join('models', 'normalization_params.json'),
            'evaluation': os.path.join('models', 'evaluation_results.json'),
            'robustness': os.path.join('models', 'robustness_results.json'),
        }
        norm_params_path = model_paths['normalization']
    print(f"\nSaving normalization parameters to: {norm_params_path}")
    save_normalization_params(scaler, feature_cols, norm_params_path)
    print("✓ Normalization parameters saved")
    
    # Save split datasets
    if asset_config:
        split_paths = get_regime_split_paths(asset_config)
    else:
        split_paths = {
            'train': os.path.join(output_dir, 'regime_train.csv'),
            'validation': os.path.join(output_dir, 'regime_validation.csv'),
            'test': os.path.join(output_dir, 'regime_test.csv'),
        }
    
    print(f"\nSaving split datasets:")
    print(f"  Training: {split_paths['train']}")
    train_df_norm.to_csv(split_paths['train'])
    print(f"  Validation: {split_paths['validation']}")
    val_df_norm.to_csv(split_paths['validation'])
    print(f"  Test: {split_paths['test']}")
    test_df_norm.to_csv(split_paths['test'])
    print("✓ All datasets saved")
    
    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE")
    print("=" * 80)
    
    return {
        'train_df': train_df_norm,
        'val_df': val_df_norm,
        'test_df': test_df_norm,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'n_features': len(feature_cols),
        'paths': {
            'train': split_paths['train'],
            'validation': split_paths['validation'],
            'test': split_paths['test'],
            'normalization': norm_params_path,
        }
    }


if __name__ == "__main__":
    # Run data preparation
    result = prepare_data()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Number of features: {result['n_features']}")
    print(f"Feature columns: {result['feature_cols']}")
    print(f"\nTraining samples: {len(result['train_df']):,}")
    print(f"Validation samples: {len(result['val_df']):,}")
    print(f"Test samples: {len(result['test_df']):,}")


