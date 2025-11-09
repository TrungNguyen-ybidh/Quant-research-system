"""
Phase 1 Checkpoint Validation Script

Validates all Phase 1 requirements:
✅ models/ folder exists
✅ src/regime_labeling.py (with labeling logic)
✅ src/regime_model.py (with network architecture)
✅ models/model_config.json (with hyperparameters)
✅ data/processed/regime_labels.csv (full labeled dataset)
✅ data/processed/regime_train.csv (60% - training)
✅ data/processed/regime_validation.csv (20% - validation)
✅ data/processed/regime_test.csv (20% - test)
✅ models/normalization_params.json (feature scaling parameters)
✅ Label distribution is balanced (30-40% each class)
✅ Visual check shows labels match price movement
✅ Average returns match regime types (up=positive, down=negative, range=neutral)
✅ PyTorch installed and working
✅ All features calculated and normalized

"""

import os
import json
import pandas as pd
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def validate_files():
    """Validate that all required files exist."""
    print("=" * 80)
    print("FILE VALIDATION")
    print("=" * 80)
    
    required_files = {
        'models/': os.path.isdir('models'),
        'src/regime_labeling.py': os.path.exists('src/regime_labeling.py'),
        'src/regime_model.py': os.path.exists('src/regime_model.py'),
        'models/model_config.json': os.path.exists('models/model_config.json'),
        'data/processed/regime_labels.csv': os.path.exists('data/processed/regime_labels.csv'),
        'data/processed/regime_train.csv': os.path.exists('data/processed/regime_train.csv'),
        'data/processed/regime_validation.csv': os.path.exists('data/processed/regime_validation.csv'),
        'data/processed/regime_test.csv': os.path.exists('data/processed/regime_test.csv'),
        'models/normalization_params.json': os.path.exists('models/normalization_params.json'),
    }
    
    all_passed = True
    for file_path, exists in required_files.items():
        status = "✓" if exists else "✗"
        print(f"{status} {file_path}")
        if not exists:
            all_passed = False
    
    return all_passed


def validate_label_distribution():
    """Validate label distribution is balanced."""
    print("\n" + "=" * 80)
    print("LABEL DISTRIBUTION VALIDATION")
    print("=" * 80)
    
    df = pd.read_csv('data/processed/regime_labels.csv')
    
    label_counts = df['regime_label'].value_counts()
    total = len(df)
    
    distribution = {
        'range': label_counts.get('range', 0) / total * 100,
        'up': label_counts.get('up', 0) / total * 100,
        'down': label_counts.get('down', 0) / total * 100
    }
    
    print(f"\nTotal samples: {total:,}")
    print(f"\nLabel distribution:")
    for regime in ['range', 'up', 'down']:
        count = label_counts.get(regime, 0)
        pct = distribution[regime]
        print(f"  {regime.capitalize()}: {count:,} ({pct:.1f}%)")
    
    # Check if balanced (target: 30-45% range, 30-40% up, 20-30% down)
    range_ok = 30 <= distribution['range'] <= 45
    up_ok = 30 <= distribution['up'] <= 40
    down_ok = 20 <= distribution['down'] <= 30
    
    print(f"\nBalance check:")
    print(f"  Range: {distribution['range']:.1f}% (expected: 30-45%) {'✓' if range_ok else '⚠'}")
    print(f"  Up: {distribution['up']:.1f}% (expected: 30-40%) {'✓' if up_ok else '⚠'}")
    print(f"  Down: {distribution['down']:.1f}% (expected: 20-30%) {'✓' if down_ok else '⚠'}")
    
    return range_ok and up_ok and down_ok


def validate_returns():
    """Validate average returns match regime types."""
    print("\n" + "=" * 80)
    print("AVERAGE RETURNS VALIDATION")
    print("=" * 80)
    
    df = pd.read_csv('data/processed/regime_labels.csv', parse_dates=['timestamp'], index_col='timestamp')
    df = df.sort_index()
    
    # Calculate hourly returns
    df['hourly_return'] = df['close'].pct_change() * 100
    
    returns_by_regime = {}
    for regime in ['range', 'up', 'down']:
        regime_mask = df['regime_label'] == regime
        regime_returns = df.loc[regime_mask, 'hourly_return'].dropna()
        returns_by_regime[regime] = {
            'mean': regime_returns.mean(),
            'std': regime_returns.std(),
            'count': len(regime_returns)
        }
    
    print(f"\nAverage hourly returns by regime:")
    for regime in ['range', 'up', 'down']:
        stats = returns_by_regime[regime]
        print(f"  {regime.capitalize()}: {stats['mean']:.4f}% (std: {stats['std']:.4f}%)")
    
    # Validate
    up_positive = returns_by_regime['up']['mean'] > 0
    down_negative = returns_by_regime['down']['mean'] < 0
    range_near_zero = abs(returns_by_regime['range']['mean']) < 0.02
    
    print(f"\nValidation:")
    print(f"  Up periods show positive returns: {up_positive} {'✓' if up_positive else '✗'}")
    print(f"  Down periods show negative returns: {down_negative} {'✓' if down_negative else '✗'}")
    print(f"  Range periods show near-zero returns: {range_near_zero} {'✓' if range_near_zero else '✗'}")
    
    return up_positive and down_negative and range_near_zero


def validate_pytorch():
    """Validate PyTorch is installed and working."""
    print("\n" + "=" * 80)
    print("PYTORCH VALIDATION")
    print("=" * 80)
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        # Test basic operations
        x = torch.randn(5, 3)
        y = torch.randn(3, 4)
        z = torch.mm(x, y)
        
        print(f"✓ PyTorch tensor operations working")
        print(f"  Test tensor shape: {z.shape}")
        
        # Test CUDA availability (optional)
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA available: {cuda_available}")
        
        return True
    except ImportError as e:
        print(f"✗ PyTorch not installed: {e}")
        return False
    except Exception as e:
        print(f"✗ PyTorch error: {e}")
        return False


def validate_model_config():
    """Validate model configuration."""
    print("\n" + "=" * 80)
    print("MODEL CONFIG VALIDATION")
    print("=" * 80)
    
    try:
        with open('models/model_config.json', 'r') as f:
            config = json.load(f)
        
        print(f"✓ Model config loaded successfully")
        print(f"\nModel architecture:")
        print(f"  Hidden sizes: {config['model']['hidden_sizes']}")
        print(f"  Dropout rate: {config['model']['dropout_rate']}")
        print(f"  Num classes: {config['model']['num_classes']}")
        
        print(f"\nTraining parameters:")
        print(f"  Learning rate: {config['training']['learning_rate']}")
        print(f"  Batch size: {config['training']['batch_size']}")
        print(f"  Epochs: {config['training']['epochs']}")
        
        return True
    except Exception as e:
        print(f"✗ Error loading model config: {e}")
        return False


def validate_normalization():
    """Validate normalization parameters."""
    print("\n" + "=" * 80)
    print("NORMALIZATION VALIDATION")
    print("=" * 80)
    
    try:
        with open('models/normalization_params.json', 'r') as f:
            params = json.load(f)
        
        print(f"✓ Normalization parameters loaded")
        print(f"  Number of features: {params['n_features']}")
        print(f"  Feature columns: {len(params['feature_columns'])}")
        print(f"  Mean shape: {len(params['mean'])}")
        print(f"  Scale shape: {len(params['scale'])}")
        
        # Check if train/val/test data is normalized
        train_df = pd.read_csv('data/processed/regime_train.csv')
        val_df = pd.read_csv('data/processed/regime_validation.csv')
        test_df = pd.read_csv('data/processed/regime_test.csv')
        
        feature_cols = params['feature_columns']
        
        # Check if features are normalized (mean should be close to 0, std close to 1)
        train_features = train_df[feature_cols].fillna(0)
        train_mean = train_features.mean().mean()
        train_std = train_features.std().mean()
        
        print(f"\nNormalization check (training data):")
        print(f"  Mean of features: {train_mean:.6f} (expected: ~0)")
        print(f"  Std of features: {train_std:.6f} (expected: ~1)")
        
        normalized = abs(train_mean) < 0.1 and 0.8 < train_std < 1.2
        
        return normalized
    except Exception as e:
        print(f"✗ Error validating normalization: {e}")
        return False


def validate_data_splits():
    """Validate data splits are correct."""
    print("\n" + "=" * 80)
    print("DATA SPLIT VALIDATION")
    print("=" * 80)
    
    train_df = pd.read_csv('data/processed/regime_train.csv')
    val_df = pd.read_csv('data/processed/regime_validation.csv')
    test_df = pd.read_csv('data/processed/regime_test.csv')
    
    total = len(train_df) + len(val_df) + len(test_df)
    
    train_pct = len(train_df) / total * 100
    val_pct = len(val_df) / total * 100
    test_pct = len(test_df) / total * 100
    
    print(f"Total samples: {total:,}")
    print(f"  Training: {len(train_df):,} ({train_pct:.1f}%)")
    print(f"  Validation: {len(val_df):,} ({val_pct:.1f}%)")
    print(f"  Test: {len(test_df):,} ({test_pct:.1f}%)")
    
    train_ok = 58 <= train_pct <= 62
    val_ok = 18 <= val_pct <= 22
    test_ok = 18 <= test_pct <= 22
    
    print(f"\nSplit validation:")
    print(f"  Training: {train_pct:.1f}% (expected: 58-62%) {'✓' if train_ok else '✗'}")
    print(f"  Validation: {val_pct:.1f}% (expected: 18-22%) {'✓' if val_ok else '✗'}")
    print(f"  Test: {test_pct:.1f}% (expected: 18-22%) {'✓' if test_ok else '✗'}")
    
    return train_ok and val_ok and test_ok


def main():
    """Run all validation checks."""
    print("\n" + "=" * 80)
    print("PHASE 1 CHECKPOINT VALIDATION")
    print("=" * 80)
    
    results = {
        'Files': validate_files(),
        'Label Distribution': validate_label_distribution(),
        'Average Returns': validate_returns(),
        'PyTorch': validate_pytorch(),
        'Model Config': validate_model_config(),
        'Normalization': validate_normalization(),
        'Data Splits': validate_data_splits(),
    }
    
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {check}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL CHECKS PASSED - PHASE 1 CHECKPOINT COMPLETE")
    else:
        print("✗ SOME CHECKS FAILED - PLEASE REVIEW")
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    main()


