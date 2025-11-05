"""
Visualization Module for Regime Classification Model

This module creates 5 visualizations to interpret and communicate model behavior:
1. Heuristic Regime Timeline
2. ML Regime Timeline
3. Distribution Comparison
4. Confusion Matrix Heatmap
5. Prediction Confidence Histogram

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict

try:
    import seaborn as sns
except ImportError:
    sns = None

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def create_heuristic_timeline(df: pd.DataFrame, output_path: str):
    """Create heuristic regime timeline visualization."""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot price
    ax.plot(df.index, df['close'], color='black', linewidth=1, alpha=0.7, label='Price')
    
    # Color backgrounds by regime
    color_map = {'range': 'gray', 'up': 'green', 'down': 'red'}
    
    for regime in ['range', 'up', 'down']:
        regime_mask = df['regime_label'] == regime
        regime_data = df.loc[regime_mask]
        if len(regime_data) > 0:
            ax.fill_between(regime_data.index, 
                           ax.get_ylim()[0], ax.get_ylim()[1],
                           alpha=0.2, color=color_map[regime], label=f'{regime.capitalize()} regime')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.set_title('Heuristic Regime Timeline (ADX + SMA Rules)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def create_ml_timeline(df: pd.DataFrame, output_path: str):
    """Create ML regime timeline visualization."""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot price
    ax.plot(df.index, df['close'], color='black', linewidth=1, alpha=0.7, label='Price')
    
    # Color backgrounds by ML predictions
    color_map = {'range': 'gray', 'up': 'green', 'down': 'red'}
    
    for regime in ['range', 'up', 'down']:
        regime_mask = df['ml_prediction_label'] == regime
        regime_data = df.loc[regime_mask]
        if len(regime_data) > 0:
            ax.fill_between(regime_data.index, 
                           ax.get_ylim()[0], ax.get_ylim()[1],
                           alpha=0.2, color=color_map[regime], label=f'{regime.capitalize()} regime (ML)')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.set_title('ML Regime Timeline (Neural Network Predictions)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def create_distribution_comparison(df: pd.DataFrame, output_path: str):
    """Create distribution comparison pie charts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Heuristic distribution
    heuristic_counts = df['regime_label'].value_counts()
    colors = {'range': 'gray', 'up': 'green', 'down': 'red'}
    heuristic_colors = [colors.get(regime, 'blue') for regime in heuristic_counts.index]
    
    ax1.pie(heuristic_counts.values, labels=heuristic_counts.index, autopct='%1.1f%%',
            colors=heuristic_colors, startangle=90)
    ax1.set_title('Heuristic Regime Distribution', fontsize=12, fontweight='bold')
    
    # ML distribution
    ml_counts = df['ml_prediction_label'].value_counts()
    ml_colors = [colors.get(regime, 'blue') for regime in ml_counts.index]
    
    ax2.pie(ml_counts.values, labels=ml_counts.index, autopct='%1.1f%%',
            colors=ml_colors, startangle=90)
    ax2.set_title('ML Regime Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def create_confusion_heatmap(df: pd.DataFrame, output_path: str):
    """Create confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix
    
    # Create confusion matrix
    y_true = df['regime_numeric'].values
    y_pred = df['ml_prediction'].values
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if sns is not None:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Range', 'Up', 'Down'],
                    yticklabels=['Range', 'Up', 'Down'],
                    ax=ax, cbar_kws={'label': 'Count'})
    else:
        # Fallback to matplotlib
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        ax.set_xticks(np.arange(len(['Range', 'Up', 'Down'])))
        ax.set_yticks(np.arange(len(['Range', 'Up', 'Down'])))
        ax.set_xticklabels(['Range', 'Up', 'Down'])
        ax.set_yticklabels(['Range', 'Up', 'Down'])
        
        # Add annotations
        for i in range(len(cm)):
            for j in range(len(cm)):
                text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Count')
    
    ax.set_xlabel('Predicted Regime', fontsize=12)
    ax.set_ylabel('True Regime', fontsize=12)
    ax.set_title('Confusion Matrix: True vs Predicted Regimes', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def create_confidence_histogram(df: pd.DataFrame, output_path: str):
    """Create prediction confidence histogram."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate correctness
    correct = df['regime_numeric'] == df['ml_prediction']
    
    # Plot histograms
    ax.hist(df[correct]['ml_confidence'], bins=50, alpha=0.7, 
            label='Correct Predictions', color='green', edgecolor='black')
    ax.hist(df[~correct]['ml_confidence'], bins=50, alpha=0.7, 
            label='Incorrect Predictions', color='red', edgecolor='black')
    
    ax.set_xlabel('Prediction Confidence', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Prediction Confidence Histogram: Correct vs Incorrect', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add vertical line at 70%
    ax.axvline(x=0.7, color='blue', linestyle='--', linewidth=2, label='70% Threshold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def create_all_visualizations(predictions_path: str, output_dir: str = None):
    """Create all visualizations."""
    print("=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    if output_dir is None:
        output_dir = config.PROCESSED_DATA_PATH
    
    # Load predictions
    print(f"\nLoading predictions from: {predictions_path}")
    df = pd.read_csv(predictions_path, parse_dates=['timestamp'], index_col='timestamp')
    print(f"  Samples: {len(df):,}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Sample data for timeline (too many points can be slow)
    sample_size = min(5000, len(df))
    df_sample = df.tail(sample_size)
    
    create_heuristic_timeline(df_sample, os.path.join(output_dir, 'heuristic_regime_timeline.png'))
    create_ml_timeline(df_sample, os.path.join(output_dir, 'ml_regime_timeline.png'))
    create_distribution_comparison(df, os.path.join(output_dir, 'regime_distribution_comparison.png'))
    create_confusion_heatmap(df, os.path.join(output_dir, 'confusion_matrix_heatmap.png'))
    create_confidence_histogram(df, os.path.join(output_dir, 'prediction_confidence_histogram.png'))
    
    print("\n" + "=" * 80)
    print("ALL VISUALIZATIONS CREATED")
    print("=" * 80)


if __name__ == "__main__":
    predictions_path = os.path.join(config.PROCESSED_DATA_PATH, 'regime_predictions.csv')
    create_all_visualizations(predictions_path)

