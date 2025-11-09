"""
Evaluation Module for Regime Classification Model

This module evaluates the trained regime classification model on test data,
calculates performance metrics, and compares against baseline methods.

"""

import torch
import pandas as pd
import numpy as np
import json
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, List, Tuple, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.regime_model import RegimeClassifier
from src.data_preparation import get_feature_columns
from src.config_manager import get_model_paths, get_setting, get_sanitized_symbol


def load_model(model_path: str, config_path: str = None) -> Tuple[RegimeClassifier, List[str], Dict]:
    """
    Load trained model and configuration.
    
    Args:
        model_path: Path to saved model (.pth file)
        config_path: Path to model config JSON (optional)
        
    Returns:
        Tuple of (model, feature_cols, model_config)
    """
    print(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Get model config
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
    else:
        # Load from file if not in checkpoint
        if config_path is None:
            config_path = os.path.join('models', 'model_config.json')
        with open(config_path, 'r') as f:
            full_config = json.load(f)
            model_config = full_config['model']
    
    # Get feature columns
    if 'feature_cols' in checkpoint:
        feature_cols = checkpoint['feature_cols']
    else:
        # Need to infer from data
        feature_cols = None
    
    # Create model
    input_size = len(feature_cols) if feature_cols else model_config.get('n_features', 15)
    model = RegimeClassifier(
        input_size=input_size,
        hidden_sizes=model_config['hidden_sizes'],
        dropout_rate=model_config['dropout_rate'],
        num_classes=model_config['num_classes']
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  Model loaded successfully")
    print(f"  Validation accuracy: {checkpoint.get('val_accuracy', 'N/A'):.2f}%")
    print(f"  Trained epoch: {checkpoint.get('epoch', 'N/A')}")
    
    return model, feature_cols, model_config


def predict(model: RegimeClassifier, features: torch.Tensor, device: torch.device) -> np.ndarray:
    """
    Make predictions using the model.
    
    Args:
        model: Trained model
        features: Feature tensor
        device: Device (CPU or CUDA)
        
    Returns:
        Predictions array
    """
    model.eval()
    features = features.to(device)
    
    with torch.no_grad():
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.cpu().numpy()


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     class_names: List[str] = None) -> Dict:
    """
    Calculate performance metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names (default: ['range', 'up', 'down'])
        
    Returns:
        Dictionary with metrics
    """
    if class_names is None:
        class_names = ['range', 'up', 'down']
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    metrics = {
        'accuracy': accuracy,
        'precision': {class_names[i]: precision[i] for i in range(len(class_names))},
        'recall': {class_names[i]: recall[i] for i in range(len(class_names))},
        'f1': {class_names[i]: f1[i] for i in range(len(class_names))},
        'support': {class_names[i]: support[i] for i in range(len(class_names))},
        'confusion_matrix': cm.tolist()
    }
    
    return metrics


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                         output_path: str = None):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        output_path: Path to save plot (optional)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def calculate_baselines(y_true: np.ndarray, df: pd.DataFrame = None) -> Dict:
    """
    Calculate baseline accuracies for comparison.
    
    Args:
        y_true: True labels
        df: DataFrame with features (for rule-based baseline)
        
    Returns:
        Dictionary with baseline accuracies
    """
    baselines = {}
    
    # 1. Random guessing (33.33% for 3 classes)
    n_samples = len(y_true)
    random_pred = np.random.randint(0, 3, size=n_samples)
    baselines['random'] = accuracy_score(y_true, random_pred)
    
    # 2. Majority class (predict most common class)
    majority_class = np.bincount(y_true).argmax()
    majority_pred = np.full(n_samples, majority_class)
    baselines['majority_class'] = accuracy_score(y_true, majority_pred)
    
    # 3. Rule-based (using ADX, SMA-50, SMA-200)
    if df is not None:
        try:
            rule_pred = rule_based_predict(df)
            baselines['rule_based'] = accuracy_score(y_true, rule_pred)
        except Exception as e:
            print(f"  Warning: Could not calculate rule-based baseline: {e}")
            baselines['rule_based'] = None
    else:
        baselines['rule_based'] = None
    
    return baselines


def rule_based_predict(df: pd.DataFrame, adx_threshold: float = 14.0) -> np.ndarray:
    """
    Rule-based prediction using the same logic as labeling.
    
    Args:
        df: DataFrame with features
        adx_threshold: ADX threshold (default: 14.0)
        
    Returns:
        Predicted labels (0=range, 1=up, 2=down)
    """
    predictions = np.zeros(len(df), dtype=int)
    
    # Check if required columns exist
    required_cols = ['adx', 'close', 'sma_long', 'sma_200']
    if not all(col in df.columns for col in required_cols):
        return predictions
    
    # Fill NaN values
    df = df.fillna(0)
    
    # Up regime: ADX > threshold, close > SMA-50, SMA-50 > SMA-200
    up_mask = (
        (df['adx'] > adx_threshold) &
        (df['close'] > df['sma_long']) &
        (df['sma_long'] > df['sma_200'])
    )
    predictions[up_mask] = 1
    
    # Down regime: ADX > threshold, close < SMA-50, SMA-50 < SMA-200
    down_mask = (
        (df['adx'] > adx_threshold) &
        (df['close'] < df['sma_long']) &
        (df['sma_long'] < df['sma_200'])
    )
    predictions[down_mask] = 2
    
    # Everything else is range (already 0)
    
    return predictions


def evaluate_model(test_path: str, model_path: str = None, 
                  output_dir: str = None,
                  asset_config: Dict[str, Any] = None) -> Dict:
    """
    Main evaluation function.
    
    Args:
        test_path: Path to test CSV
        model_path: Path to saved model (defaults to models/regime_classifier.pth)
        output_dir: Output directory (defaults to models/)
        
    Returns:
        Dictionary with evaluation results
    """
    print("=" * 80)
    print("REGIME CLASSIFICATION MODEL EVALUATION")
    print("=" * 80)
    
    if asset_config:
        asset_name = get_setting(asset_config, 'asset.name')
        symbol = get_setting(asset_config, 'asset.symbol')
        print(f"Asset: {asset_name} ({symbol})")
    
    # Set default paths
    model_paths = get_model_paths(asset_config) if asset_config else None
    sanitized_symbol = get_sanitized_symbol(asset_config) if asset_config else None
    
    if model_path is None:
        model_path = model_paths['model'] if model_paths else os.path.join('models', 'regime_classifier.pth')
    
    if output_dir is None:
        output_dir = 'models'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data
    print(f"\nLoading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"  Test samples: {len(test_df):,}")
    
    # Get feature columns
    feature_cols = get_feature_columns(test_df)
    print(f"  Features: {len(feature_cols)}")
    
    # Remove rows with missing values
    test_df = test_df.dropna(subset=feature_cols + ['regime_numeric'])
    print(f"  Valid samples after removing NaN: {len(test_df):,}")
    
    # Prepare features and labels
    features = test_df[feature_cols].fillna(0).values.astype(np.float32)
    labels = test_df['regime_numeric'].values.astype(np.int64)
    
    features_tensor = torch.from_numpy(features)
    
    # Load model
    model, feature_cols, model_config = load_model(model_path)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = predict(model, features_tensor, device)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    class_names = ['range', 'up', 'down']
    metrics = calculate_metrics(labels, predictions, class_names)
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    print(f"\nOverall Accuracy: {metrics['accuracy']*100:.2f}%")
    
    print("\nPer-Class Metrics:")
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)
    for class_name in class_names:
        print(f"{class_name:<10} {metrics['precision'][class_name]:<12.4f} "
              f"{metrics['recall'][class_name]:<12.4f} {metrics['f1'][class_name]:<12.4f} "
              f"{metrics['support'][class_name]:<10}")
    
    # Confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    print("\nConfusion Matrix:")
    print(f"{'':<10} {'Pred Range':<12} {'Pred Up':<12} {'Pred Down':<12}")
    print("-" * 50)
    for i, class_name in enumerate(class_names):
        print(f"{'True ' + class_name:<10} {cm[i, 0]:<12} {cm[i, 1]:<12} {cm[i, 2]:<12}")
    
    # Plot confusion matrix
    cm_filename = f"confusion_matrix_{sanitized_symbol}.png" if sanitized_symbol else 'confusion_matrix.png'
    cm_path = os.path.join(output_dir, cm_filename)
    plot_confusion_matrix(cm, class_names, cm_path)
    
    # Calculate baselines
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON")
    print("=" * 80)
    
    baselines = calculate_baselines(labels, test_df)
    
    print(f"\nBaseline Accuracies:")
    print(f"  Random Guessing (33%): {baselines['random']*100:.2f}%")
    print(f"  Majority Class: {baselines['majority_class']*100:.2f}%")
    if baselines['rule_based'] is not None:
        print(f"  Rule-Based: {baselines['rule_based']*100:.2f}%")
    
    print(f"\nModel Accuracy: {metrics['accuracy']*100:.2f}%")
    
    # Compare
    print("\nModel Performance vs Baselines:")
    model_acc = metrics['accuracy']
    
    if model_acc > baselines['random']:
        improvement = (model_acc - baselines['random']) / baselines['random'] * 100
        print(f"  ✓ Model outperforms random guessing by {improvement:.1f}%")
    else:
        print(f"  ✗ Model does not outperform random guessing")
    
    if model_acc > baselines['majority_class']:
        improvement = (model_acc - baselines['majority_class']) / baselines['majority_class'] * 100
        print(f"  ✓ Model outperforms majority class by {improvement:.1f}%")
    else:
        print(f"  ✗ Model does not outperform majority class")
    
    if baselines['rule_based'] is not None and model_acc > baselines['rule_based']:
        improvement = (model_acc - baselines['rule_based']) / baselines['rule_based'] * 100
        print(f"  ✓ Model outperforms rule-based by {improvement:.1f}%")
    else:
        if baselines['rule_based'] is not None:
            print(f"  ✗ Model does not outperform rule-based")
    
    # Save results (convert numpy types to native Python types for JSON)
    results = {
        'metrics': {
            'accuracy': float(metrics['accuracy']),
            'precision': {k: float(v) for k, v in metrics['precision'].items()},
            'recall': {k: float(v) for k, v in metrics['recall'].items()},
            'f1': {k: float(v) for k, v in metrics['f1'].items()},
            'support': {k: int(v) for k, v in metrics['support'].items()},
            'confusion_matrix': metrics['confusion_matrix']
        },
        'baselines': {k: float(v) if v is not None else None for k, v in baselines.items()},
        'model_accuracy': float(metrics['accuracy']),
        'test_samples': int(len(labels))
    }
    
    results_filename = f"evaluation_results_{sanitized_symbol}.json" if sanitized_symbol else 'evaluation_results.json'
    results_path = os.path.join(output_dir, results_filename)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved evaluation results to: {results_path}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    # Set paths
    test_path = os.path.join(config.PROCESSED_DATA_PATH, 'regime_test.csv')
    
    # Evaluate model
    results = evaluate_model(test_path)
