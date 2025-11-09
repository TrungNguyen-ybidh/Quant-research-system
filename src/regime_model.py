"""
Regime Model Module for Machine Learning Model Definition

This module defines the neural network architecture for regime classification.
Uses PyTorch to build a feedforward neural network for classifying market regimes
into "up", "down", or "range" categories.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class RegimeClassifier(nn.Module):
    """
    Neural network for regime classification.
    
    Architecture:
    - Input layer: Number of features (technical indicators)
    - Hidden layers: Fully connected layers with ReLU activation
    - Output layer: 3 classes (range=0, up=1, down=2)
    - Dropout: Applied after hidden layers for regularization
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int], 
                 dropout_rate: float = 0.3, num_classes: int = 3):
        """
        Initialize the regime classifier.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes (e.g., [128, 64, 32])
            dropout_rate: Dropout probability (default: 0.3)
            num_classes: Number of output classes (default: 3 for range/up/down)
        """
        super(RegimeClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        
        # Build layers
        layers = []
        prev_size = input_size
        
        # Create hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        return self.model(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Probability tensor of shape (batch_size, num_classes)
        """
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Predicted class indices of shape (batch_size,)
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)


def create_model(input_size: int, config: Dict) -> RegimeClassifier:
    """
    Create a regime classifier model from configuration.
    
    Args:
        input_size: Number of input features
        config: Configuration dictionary with model hyperparameters
        
    Returns:
        Initialized RegimeClassifier model
    """
    hidden_sizes = config.get('hidden_sizes', [128, 64, 32])
    dropout_rate = config.get('dropout_rate', 0.3)
    num_classes = config.get('num_classes', 3)
    
    model = RegimeClassifier(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        dropout_rate=dropout_rate,
        num_classes=num_classes
    )
    
    return model


def get_model_info(model: RegimeClassifier) -> Dict:
    """
    Get information about the model architecture.
    
    Args:
        model: RegimeClassifier model
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'input_size': model.input_size,
        'hidden_sizes': model.hidden_sizes,
        'dropout_rate': model.dropout_rate,
        'num_classes': model.num_classes,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params
    }


if __name__ == "__main__":
    """
    Test the model architecture.
    """
    # Example configuration
    config = {
        'hidden_sizes': [128, 64, 32],
        'dropout_rate': 0.3,
        'num_classes': 3
    }
    
    # Create model with example input size (14 features)
    input_size = 14
    model = create_model(input_size, config)
    
    print("Model Architecture:")
    print(model)
    print("\nModel Information:")
    info = get_model_info(model)
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, input_size)
    output = model(x)
    
    print(f"\nTest Forward Pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Test predictions
    probs = model.predict_proba(x)
    preds = model.predict(x)
    
    print(f"  Probabilities shape: {probs.shape}")
    print(f"  Predictions shape: {preds.shape}")
    print(f"  Sample predictions: {preds[:5].tolist()}")
    
    print("\n✓ Model architecture test complete!")
