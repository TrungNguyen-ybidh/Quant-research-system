"""
Training Module for Regime Classification Model

This module implements the training pipeline for the regime classification model.
It loads training and validation data, trains the neural network, and saves
the best model based on validation accuracy.

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import json
import os
import sys
from typing import Dict, List, Tuple, Any
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.regime_model import create_model, RegimeClassifier
from src.data_preparation import get_feature_columns
from src.config_manager import get_model_paths, get_setting


def load_data(data_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load and prepare data for training.
    
    Args:
        data_path: Path to CSV file with features and labels
        
    Returns:
        Tuple of (DataFrame, feature_columns)
    """
    df = pd.read_csv(data_path)
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    
    # Remove rows with missing values
    df = df.dropna(subset=feature_cols + ['regime_numeric'])
    
    return df, feature_cols


def prepare_tensors(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert DataFrame to PyTorch tensors.
    
    Args:
        df: DataFrame with features and labels
        feature_cols: List of feature column names
        
    Returns:
        Tuple of (features_tensor, labels_tensor)
    """
    # Extract features and labels
    features = df[feature_cols].fillna(0).values.astype(np.float32)
    labels = df['regime_numeric'].values.astype(np.int64)
    
    # Convert to PyTorch tensors
    features_tensor = torch.from_numpy(features)
    labels_tensor = torch.from_numpy(labels)
    
    return features_tensor, labels_tensor


def add_noise_to_features(features: torch.Tensor, noise_level: float = 0.02) -> torch.Tensor:
    """
    Add random noise to features for data augmentation.
    
    Adds ±noise_level (default ±2%) random variation to each feature value.
    This helps the model learn to be robust to noisy data.
    
    Args:
        features: Feature tensor of shape (batch_size, num_features)
        noise_level: Noise level as fraction (0.02 = ±2% variation)
        
    Returns:
        Augmented feature tensor with same shape
    """
    # Generate random noise: uniform distribution in [-noise_level, +noise_level]
    noise = torch.empty_like(features).uniform_(-noise_level, noise_level)
    
    # Apply noise multiplicatively: feature * (1 + noise)
    # This ensures noise is proportional to feature magnitude
    augmented_features = features * (1.0 + noise)
    
    return augmented_features


def train_epoch(model: RegimeClassifier, dataloader: DataLoader, 
                criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device,
                use_augmentation: bool = True, augmentation_prob: float = 0.5, 
                noise_level: float = 0.02) -> Tuple[float, float]:
    """
    Train the model for one epoch with optional data augmentation.
    
    Args:
        model: Neural network model
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device (CPU or CUDA)
        use_augmentation: Whether to use data augmentation (default: True)
        augmentation_prob: Probability of augmenting each sample (default: 0.5 = 50%)
        noise_level: Noise level as fraction (default: 0.02 = ±2% variation)
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)
        
        # Apply data augmentation to training batch
        if use_augmentation:
            batch_size = features.size(0)
            
            # Determine which samples to augment (50% of batch)
            num_augment = int(batch_size * augmentation_prob)
            augment_indices = torch.randperm(batch_size)[:num_augment]
            
            # Create augmented features
            augmented_features = features.clone()
            if len(augment_indices) > 0:
                # Add noise to selected samples
                augmented_features[augment_indices] = add_noise_to_features(
                    features[augment_indices], noise_level=noise_level
                )
            
            # Use augmented features for training
            features = augmented_features
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate_epoch(model: RegimeClassifier, dataloader: DataLoader, 
                   criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """
    Validate the model for one epoch.
    
    Args:
        model: Neural network model
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device (CPU or CUDA)
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def train_model(train_path: str, val_path: str, config_path: str = None, 
                model_save_path: str = None, history_save_path: str = None,
                asset_config: Dict[str, Any] = None) -> Dict:
    """
    Main training function.
    
    Args:
        train_path: Path to training CSV
        val_path: Path to validation CSV
        config_path: Path to model config JSON (defaults to models/model_config.json)
        model_save_path: Path to save model (defaults to models/regime_classifier.pth)
        history_save_path: Path to save training history (defaults to models/training_history.json)
        
    Returns:
        Dictionary with training results
    """
    print("=" * 80)
    print("REGIME CLASSIFICATION MODEL TRAINING")
    print("=" * 80)
    
    if asset_config:
        asset_name = get_setting(asset_config, 'asset.name')
        symbol = get_setting(asset_config, 'asset.symbol')
        print(f"Asset: {asset_name} ({symbol})")
    
    # Set default paths
    if config_path is None:
        config_path = os.path.join('models', 'model_config.json')
    
    model_paths = get_model_paths(asset_config) if asset_config else None
    
    if model_save_path is None:
        model_save_path = model_paths['model'] if model_paths else os.path.join('models', 'regime_classifier.pth')
    
    if history_save_path is None:
        history_save_path = model_paths['history'] if model_paths else os.path.join('models', 'training_history.json')
    
    # Load config
    print(f"\nLoading model config from: {config_path}")
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    
    print(f"  Model architecture: {model_config['model']['hidden_sizes']}")
    print(f"  Learning rate: {model_config['training']['learning_rate']}")
    print(f"  Batch size: {model_config['training']['batch_size']}")
    print(f"  Epochs: {model_config['training']['epochs']}")
    
    # Load data
    print(f"\nLoading training data from: {train_path}")
    train_df, feature_cols = load_data(train_path)
    print(f"  Training samples: {len(train_df):,}")
    print(f"  Features: {len(feature_cols)}")
    
    print(f"\nLoading validation data from: {val_path}")
    val_df, _ = load_data(val_path)
    print(f"  Validation samples: {len(val_df):,}")
    
    # Prepare tensors
    print("\nPreparing tensors...")
    train_features, train_labels = prepare_tensors(train_df, feature_cols)
    val_features, val_labels = prepare_tensors(val_df, feature_cols)
    
    print(f"  Training features shape: {train_features.shape}")
    print(f"  Validation features shape: {val_features.shape}")
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    
    batch_size = model_config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Initialize model
    print("\nInitializing model...")
    input_size = len(feature_cols)
    model = create_model(input_size, model_config['model'])
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    learning_rate = model_config['training']['learning_rate']
    weight_decay = model_config['training'].get('weight_decay', 0.0001)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    # Data augmentation settings
    use_augmentation = model_config['training'].get('use_augmentation', True)
    augmentation_prob = model_config['training'].get('augmentation_prob', 0.5)
    noise_level = model_config['training'].get('augmentation_noise_level', 0.02)
    
    if use_augmentation:
        print(f"\nData Augmentation: ENABLED")
        print(f"  Augmentation probability: {augmentation_prob*100:.0f}% of samples per batch")
        print(f"  Noise level: ±{noise_level*100:.0f}% variation per feature")
        print(f"  Purpose: Improve robustness to noisy data")
    else:
        print(f"\nData Augmentation: DISABLED")
    
    num_epochs = model_config['training']['epochs']
    patience = model_config['training'].get('patience', 10)
    early_stopping = model_config['training'].get('early_stopping', True)
    
    best_val_accuracy = 0.0
    best_epoch = 0
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'epochs': [],
        'best_epoch': 0,
        'best_val_accuracy': 0.0
    }
    
    for epoch in range(num_epochs):
        # Train (with augmentation if enabled)
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            use_augmentation=use_augmentation, augmentation_prob=augmentation_prob,
            noise_level=noise_level
        )
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Calculate train-val gap
        gap = train_acc - val_acc
        print(f"  Train-Val Gap: {gap:.2f}%")
        
        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'model_config': model_config['model'],
                'feature_cols': feature_cols
            }, model_save_path)
            
            print(f"  ✓ Best model saved (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if early_stopping and patience_counter >= patience:
            print(f"\nEarly stopping triggered (patience: {patience})")
            break
        
        print()
    
    # Update history
    history['best_epoch'] = best_epoch
    history['best_val_accuracy'] = best_val_accuracy
    
    # Save training history
    print(f"\nSaving training history to: {history_save_path}")
    with open(history_save_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Total epochs: {len(history['epochs'])}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Final train accuracy: {history['train_accuracy'][-1]:.2f}%")
    print(f"Final validation accuracy: {history['val_accuracy'][-1]:.2f}%")
    print(f"Final train-val gap: {history['train_accuracy'][-1] - history['val_accuracy'][-1]:.2f}%")
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    final_gap = history['train_accuracy'][-1] - history['val_accuracy'][-1]
    
    if final_gap < 5:
        print("✓ Good: Train-validation gap < 5% (no overfitting)")
    elif final_gap < 10:
        print("⚠ Moderate: Train-validation gap 5-10% (slight overfitting)")
    else:
        print("✗ Warning: Train-validation gap > 10% (overfitting detected)")
    
    if best_val_accuracy >= 65:
        print(f"✓ Good: Validation accuracy {best_val_accuracy:.2f}% >= 65%")
    elif best_val_accuracy >= 60:
        print(f"⚠ Moderate: Validation accuracy {best_val_accuracy:.2f}% (target: 65-70%)")
    else:
        print(f"✗ Warning: Validation accuracy {best_val_accuracy:.2f}% < 60% (underfitting)")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    
    return {
        'history': history,
        'best_epoch': best_epoch,
        'best_val_accuracy': best_val_accuracy,
        'model_save_path': model_save_path,
        'history_save_path': history_save_path
    }


def generate_training_summary(history_path: str, summary_path: str = None,
                              asset_config: Dict[str, Any] = None):
    """
    Generate training summary text file.
    
    Args:
        history_path: Path to training history JSON
        summary_path: Path to save summary (defaults to models/training_summary.txt)
    """
    if summary_path is None:
        if asset_config:
            summary_path = get_model_paths(asset_config)['summary']
        else:
            summary_path = os.path.join('models', 'training_summary.txt')
    
    print(f"\nGenerating training summary...")
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("REGIME CLASSIFICATION MODEL TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("TRAINING RESULTS\n")
        f.write("-" * 80 + "\n\n")
        
        f.write(f"Total Epochs: {len(history['epochs'])}\n")
        f.write(f"Best Epoch: {history['best_epoch']}\n")
        f.write(f"Best Validation Accuracy: {history['best_val_accuracy']:.2f}%\n")
        f.write(f"Final Train Accuracy: {history['train_accuracy'][-1]:.2f}%\n")
        f.write(f"Final Validation Accuracy: {history['val_accuracy'][-1]:.2f}%\n")
        f.write(f"Final Train-Val Gap: {history['train_accuracy'][-1] - history['val_accuracy'][-1]:.2f}%\n\n")
        
        f.write("TRAINING PROGRESS\n")
        f.write("-" * 80 + "\n\n")
        
        f.write("Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Gap\n")
        f.write("-" * 80 + "\n")
        
        for i, epoch in enumerate(history['epochs']):
            train_loss = history['train_loss'][i]
            train_acc = history['train_accuracy'][i]
            val_loss = history['val_loss'][i]
            val_acc = history['val_accuracy'][i]
            gap = train_acc - val_acc
            
            f.write(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:9.2f}% | {val_loss:8.4f} | {val_acc:7.2f}% | {gap:5.2f}%\n")
        
        f.write("\n")
        f.write("ANALYSIS\n")
        f.write("-" * 80 + "\n\n")
        
        final_gap = history['train_accuracy'][-1] - history['val_accuracy'][-1]
        
        if final_gap < 5:
            f.write("✓ Good: Train-validation gap < 5% (no overfitting)\n")
        elif final_gap < 10:
            f.write("⚠ Moderate: Train-validation gap 5-10% (slight overfitting)\n")
        else:
            f.write("✗ Warning: Train-validation gap > 10% (overfitting detected)\n")
        
        if history['best_val_accuracy'] >= 65:
            f.write(f"✓ Good: Validation accuracy {history['best_val_accuracy']:.2f}% >= 65%\n")
        elif history['best_val_accuracy'] >= 60:
            f.write(f"⚠ Moderate: Validation accuracy {history['best_val_accuracy']:.2f}% (target: 65-70%)\n")
        else:
            f.write(f"✗ Warning: Validation accuracy {history['best_val_accuracy']:.2f}% < 60% (underfitting)\n")
    
    print(f"✓ Saved training summary to: {summary_path}")


if __name__ == "__main__":
    # Set paths
    train_path = os.path.join(config.PROCESSED_DATA_PATH, 'regime_train.csv')
    val_path = os.path.join(config.PROCESSED_DATA_PATH, 'regime_validation.csv')
    
    # Train model
    results = train_model(train_path, val_path)
    
    # Generate summary
    generate_training_summary(results['history_save_path'])
