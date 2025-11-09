"""
Unsupervised Validation Module for Regime Classification

This module uses K-Means clustering to independently validate that
the supervised labels represent natural groupings in the data.

"""

import argparse
import pandas as pd
import numpy as np
import json
import os
import sys
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preparation import get_feature_columns
from src.config_manager import (
    load_config,
    validate_config,
    get_regime_split_paths,
    get_model_paths,
    get_setting,
    get_sanitized_symbol,
)


def kmeans_validation(df: pd.DataFrame, n_clusters: int = 3, output_path: str = None,
                      asset_config: Dict[str, Any] = None) -> Dict:
    """
    Validate regime labels using K-Means clustering.
    
    Args:
        df: DataFrame with features and labels
        n_clusters: Number of clusters (default: 3 for range/up/down)
        output_path: Path to save results (optional)
        
    Returns:
        Dictionary with validation results
    """
    print("=" * 80)
    print("UNSUPERVISED VALIDATION (K-Means Clustering)")
    print("=" * 80)
    
    if asset_config:
        asset_name = get_setting(asset_config, 'asset.name')
        symbol = get_setting(asset_config, 'asset.symbol')
        print(f"Asset: {asset_name} ({symbol})")
    
    # Get features and labels
    feature_cols = get_feature_columns(df)
    features = df[feature_cols].fillna(0).values.astype(np.float32)
    labels = df['regime_numeric'].values.astype(np.int64)
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Samples: {len(df):,}")
    print(f"Clusters: {n_clusters}")
    
    # Run K-Means
    print("\nRunning K-Means clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    
    # Calculate alignment
    print("\nCalculating cluster alignment...")
    
    # Map clusters to regimes (find best mapping)
    # Create confusion matrix
    cm = confusion_matrix(labels, cluster_labels)
    
    # Find best mapping between clusters and regimes
    from scipy.optimize import linear_sum_assignment
    cost_matrix = -cm  # Negative because we want to maximize
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Calculate alignment percentage
    aligned = 0
    for i, j in zip(row_ind, col_ind):
        aligned += cm[i, j]
    
    alignment_percentage = aligned / len(labels) * 100
    
    # Calculate Adjusted Rand Index
    ari = adjusted_rand_score(labels, cluster_labels)
    
    # Per-class alignment
    class_names = ['range', 'up', 'down']
    per_class_alignment = {}
    
    for i, class_name in enumerate(class_names):
        class_mask = labels == i
        if class_mask.sum() > 0:
            class_clusters = cluster_labels[class_mask]
            # Find most common cluster for this class
            most_common_cluster = np.bincount(class_clusters).argmax()
            class_aligned = (class_clusters == most_common_cluster).sum()
            class_alignment = class_aligned / class_mask.sum() * 100
            
            per_class_alignment[class_name] = {
                'alignment': float(class_alignment),
                'samples': int(class_mask.sum()),
                'assigned_cluster': int(most_common_cluster)
            }
    
    results = {
        'n_clusters': n_clusters,
        'alignment_percentage': float(alignment_percentage),
        'adjusted_rand_index': float(ari),
        'per_class_alignment': per_class_alignment,
        'confusion_matrix': cm.tolist(),
        'high_alignment': bool(alignment_percentage >= 80)
    }
    
    # Print results
    print("\n" + "=" * 80)
    print("K-MEANS VALIDATION RESULTS")
    print("=" * 80)
    
    print(f"\nOverall Alignment: {alignment_percentage:.2f}%")
    print(f"Adjusted Rand Index: {ari:.4f}")
    
    if results['high_alignment']:
        print(f"✓ High alignment (≥80%) - Regime definitions are validated")
    else:
        print(f"⚠ Moderate alignment (<80%) - Regime definitions may need refinement")
    
    print(f"\nPer-Class Alignment:")
    for class_name in class_names:
        if class_name in per_class_alignment:
            class_metrics = per_class_alignment[class_name]
            print(f"  {class_name.capitalize()}: {class_metrics['alignment']:.2f}% "
                  f"(Cluster {class_metrics['assigned_cluster']}, "
                  f"{class_metrics['samples']:,} samples)")
    
    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved validation results to: {output_path}")
    
    print("\n" + "=" * 80)
    print("UNSUPERVISED VALIDATION COMPLETE")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run K-Means validation for a configured asset.")
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
        sanitized_symbol = get_sanitized_symbol(asset_config)
        
        test_path = split_paths['test']
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test dataset not found: {test_path}")
        
        test_df = pd.read_csv(test_path)
        output_path = os.path.join('models', f"unsupervised_validation_{sanitized_symbol}.json")
        
        kmeans_validation(test_df, n_clusters=3, output_path=output_path, asset_config=asset_config)
    except Exception as error:
        print(f"Error: {error}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

