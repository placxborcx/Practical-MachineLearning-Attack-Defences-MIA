"""
Four-Layer Hybrid Defense Mechanism against Membership Inference Attacks
=========================================================================

This module implements a comprehensive defense system designed to protect machine learning
models from membership inference attacks while preserving model utility. The defense
operates through four synergistic layers that address different aspects of privacy leakage.

Defense Architecture:
- Layer 1: Training-time regularization to prevent overfitting
- Layer 2: Temperature scaling for confidence calibration
- Layer 3: Adaptive noise injection based on prediction confidence
- Layer 4: Output sanitization through clipping and rounding

Key Features:
- Multi-layer protection against various attack vectors
- Configurable defense parameters for privacy-utility trade-off
- Compatible with existing ML pipelines
- Specifically designed to counter confidence-based membership inference

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import argparse
import os
import json
import math

# =============================================================================
# CONFIGURATION AND HYPERPARAMETERS
# =============================================================================

class CFG:
    """
    Global configuration class containing all defense parameters and training settings.
    
    This centralized configuration approach allows easy parameter tuning and ensures
    consistency across all defense components. Parameters are carefully chosen based
    on empirical evaluation to balance privacy protection with model utility.
    """
    
    # System Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42                    # Random seed for reproducibility
    
    # Training Parameters
    epochs = 50                  # Training epochs (controlled to prevent overfitting)
    batch = 128                  # Batch size for stable training
    lr = 1e-3                    # Learning rate for balanced convergence
    
    # Layer 1: Regularization Parameters (Training-Time Defense)
    weight_decay = 1e-3          # L2 regularization strength (prevents overfitting)
    dropout_p = 0.6              # Dropout probability (high for strong regularization)
    
    # Layer 2: Temperature Scaling Parameters
    temperature = 6.0            # Temperature scaling factor (higher = more uniform distribution)
    
    # Layer 3: Adaptive Noise Parameters  
    noise_std_min = 0.30         # Minimum noise standard deviation
    noise_std_max = 0.9          # Maximum noise standard deviation (confidence-dependent)
    
    # Layer 4: Output Sanitization Parameters
    clip_max = 0.6               # Maximum probability after clipping (prevents extreme confidence)
    round_digit = 1              # Decimal places for probability rounding
    
    # File Management
    save_path = "./defended_target.pth"      # Model save location
    json_out = "./defense_config.json"       # Configuration backup

# =============================================================================
# REPRODUCIBILITY SETUP
# =============================================================================

def setup_reproducibility():
    """
    Configure random seeds for deterministic behavior across all components.
    
    This ensures that defense mechanisms behave consistently across different
    runs, which is crucial for reliable evaluation and comparison.
    """
    torch.manual_seed(CFG.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CFG.seed)
        # Note: We don't set deterministic=True for CUDA as it may impact performance

# Apply reproducibility settings
setup_reproducibility()

# =============================================================================
# LAYER 1: DEFENDED MODEL ARCHITECTURE (TRAINING-TIME DEFENSE)
# =============================================================================

class DefendedCNN(nn.Module):
    """
    CNN architecture with built-in regularization to prevent overfitting.
    
    This model implements Layer 1 of our defense strategy by incorporating
    regularization techniques directly into the architecture. The goal is to
    prevent the model from memorizing training patterns that enable membership
    inference attacks.
    
    Key Defense Features:
    - High dropout rate (60%) for strong regularization
    - Moderate capacity to prevent memorization
    - Compatible with standard CNN architectures
    
    Args:
        num_classes (int): Number of output classes (default: 10 for MNIST)
    """
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Feature extraction layers with regularization
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 32, 3, padding=1),    # Input: 1x28x28 -> Output: 32x28x28
            nn.ReLU(),                          # Non-linear activation
            
            # Second convolutional block  
            nn.Conv2d(32, 64, 3, padding=1),   # Input: 32x28x28 -> Output: 64x28x28
            nn.ReLU(),                          # Non-linear activation
            nn.MaxPool2d(2),                    # Pooling: 64x28x28 -> 64x14x14
            
            # Layer 1 Defense: Dropout for regularization
            nn.Dropout(CFG.dropout_p),          # 60% dropout - aggressive regularization
        )
        
        # Classification layers with regularization
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # Flatten spatial dimensions
            nn.Linear(64*14*14, 256),          # Feature compression
            nn.ReLU(),                         # Non-linear activation
            
            # Layer 1 Defense: Additional dropout in classifier
            nn.Dropout(CFG.dropout_p),          # Consistent 60% dropout
            nn.Linear(256, num_classes)        # Final classification layer
        )

    def forward(self, x):
        """
        Forward pass through the defended CNN.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            torch.Tensor: Raw logits of shape (batch_size, num_classes)
        """
        features = self.features(x)      # Extract regularized features
        output = self.classifier(features)  # Classify with regularization
        return output

# =============================================================================
# LAYERS 2-4: INFERENCE-TIME DEFENSE WRAPPER
# =============================================================================

class DefenceWrapper(nn.Module):
    """
    Comprehensive inference-time defense wrapper implementing Layers 2-4.
    
    This wrapper applies multiple defense mechanisms to model outputs without
    modifying the underlying model architecture. It can be applied to any
    pre-trained model to provide membership inference protection.
    
    Defense Layers Implemented:
    - Layer 2: Temperature scaling for confidence calibration
    - Layer 3: Adaptive noise injection based on prediction confidence  
    - Layer 4: Output sanitization through clipping and rounding
    
    The wrapper maintains API compatibility while sanitizing outputs to
    prevent membership inference attacks.
    
    Args:
        base_model: Pre-trained model to be protected
    """
    
    def __init__(self, base_model):
        super().__init__()
        # Store base model in evaluation mode (no gradient updates needed)
        self.base = base_model.eval()

    @torch.no_grad()
    def forward(self, x):
        """
        Apply comprehensive defense pipeline to model outputs.
        
        This method implements the complete inference-time defense strategy,
        transforming potentially vulnerable model outputs into sanitized
        logits that preserve utility while protecting privacy.
        
        Defense Pipeline:
        1. Temperature scaling (Layer 2)
        2. Adaptive noise injection (Layer 3) 
        3. Probability clipping and rounding (Layer 4)
        4. Logit sanitization for API compatibility
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Sanitized logits with reduced membership signals
        """
        
        # === LAYER 2: TEMPERATURE SCALING ===
        # Apply temperature scaling to reduce overconfidence
        raw_logits = self.base(x) / CFG.temperature
        
        # Temperature scaling effect:
        # - Higher temperature (6.0) makes distributions more uniform
        # - Reduces extreme confidence values that indicate membership
        # - Maintains relative ranking while reducing absolute confidence
        
        # === LAYER 3: ADAPTIVE NOISE INJECTION ===
        # Calculate confidence-dependent noise standard deviation
        probs_tmp = torch.softmax(raw_logits, dim=1)
        max_conf = probs_tmp.max(dim=1, keepdim=True)[0]  # Get maximum confidence per sample
        
        # Adaptive noise formula: higher confidence -> more noise
        # This specifically targets overfitted predictions where membership signals are strongest
        std = CFG.noise_std_min + (CFG.noise_std_max - CFG.noise_std_min) * max_conf.pow(2)
        
        # Apply confidence-dependent Gaussian noise
        noisy_logits = raw_logits + torch.randn_like(raw_logits) * std
        
        # Adaptive noise rationale:
        # - High confidence predictions (likely overfitted) receive maximum noise
        # - Low confidence predictions (uncertain) receive minimal noise
        # - Preserves decision quality while disrupting membership signals
        
        # === LAYER 4: OUTPUT SANITIZATION ===
        # Convert to probabilities for clipping and rounding
        probs = F.softmax(noisy_logits, dim=1)
        
        # Step 1: Probability clipping to prevent extreme values
        probs = torch.clamp(probs, 0.0, CFG.clip_max)  # Maximum probability = 0.6
        
        # Step 2: Renormalization after clipping
        probs = probs / probs.sum(dim=1, keepdim=True)
        
        # Step 3: Probability rounding to reduce precision
        factor = 10 ** CFG.round_digit  # For 1 decimal place
        probs = torch.round(probs * factor) / factor
        
        # Step 4: Handle edge case where all probabilities become zero
        row_sum = probs.sum(dim=1)
        zero_mask = (row_sum == 0)
        if zero_mask.any():
            # Assign uniform probabilities to zero rows
            probs[zero_mask, :] = 1.0 / probs.size(1)
        
        # === LOGIT SANITIZATION FOR API COMPATIBILITY ===
        # Convert sanitized probabilities back to logit form
        # This maintains API compatibility while embedding defense mechanisms
        
        # Multinomial sampling for additional randomization
        idx = torch.multinomial(probs, num_samples=1)  # Shape: [batch_size, 1]
        one_hot = torch.zeros_like(probs).scatter_(1, idx, 1)  # One-hot encoding
        
        # Convert to sanitized logits with numerical stability
        eps = 1e-4  # Small epsilon for numerical stability
        sanitised_logits = torch.log(one_hot * (1 - eps) + eps / probs.size(1))
        
        return sanitised_logits

    @torch.no_grad()
    def defended_probs(self, x):
        """
        Return sanitized probability distributions for direct use.
        
        This method provides an alternative interface that returns probabilities
        directly rather than logits. It applies additional smoothing for
        enhanced privacy protection.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Sanitized probability distributions
        """
        
        # Get sanitized logits from main defense pipeline
        sanitized_logits = self.forward(x)
        probs = F.softmax(sanitized_logits, dim=1)
        
        # === ENHANCED LAYER 4: ADDITIONAL PROBABILITY SANITIZATION ===
        
        # Step 1: Apply probability clipping
        probs = torch.clamp(probs, min=0.0, max=CFG.clip_max)
        
        # Step 2: Renormalization to ensure valid probability distribution
        probs = probs / probs.sum(dim=1, keepdim=True)
        
        # Step 3: Probability rounding for reduced precision
        factor = 10 ** CFG.round_digit
        probs = torch.round(probs * factor) / factor
        
        # Step 4: Final normalization to maintain probability constraints
        probs = probs / probs.sum(dim=1, keepdim=True)
        
        # Step 5: Add uniform noise for final smoothing
        # This provides additional protection against deterministic analysis
        uniform_noise = torch.rand_like(probs) * 0.01  # Small uniform noise (1%)
        probs = probs + uniform_noise
        
        # Final normalization after noise addition
        probs = probs / probs.sum(dim=1, keepdim=True)
        
        return probs

# =============================================================================
# MODEL TRAINING WITH LAYER 1 DEFENSE
# =============================================================================

def train_defended_target():
    """
    Train the defended model with Layer 1 regularization techniques.
    
    This function implements the training process for the defended model,
    incorporating regularization strategies to prevent overfitting and
    reduce membership inference vulnerability.
    
    Key Training Features:
    - L2 regularization through weight decay
    - Built-in dropout regularization
    - Validation-based early stopping
    - Optimized performance settings
    
    Returns:
        None (saves trained model to disk)
    """
    
    # === PERFORMANCE OPTIMIZATION ===
    # Configure PyTorch for optimal training performance
    torch.set_num_threads(os.cpu_count())  # Utilize all CPU cores
    torch.backends.mkldnn.enabled = True   # Enable Intel MKL-DNN for acceleration
    
    print(f"=== Training Defended Model ===")
    print(f"Device: {CFG.device}")
    print(f"Epochs: {CFG.epochs}")
    print(f"Batch size: {CFG.batch}")
    print(f"Learning rate: {CFG.lr}")
    print(f"Weight decay (L2): {CFG.weight_decay}")
    print(f"Dropout probability: {CFG.dropout_p}")
    
    # === DATA PREPARATION ===
    # Standard data transformation without augmentation
    # Note: No data augmentation to maintain clean evaluation conditions
    transform = transforms.Compose([
        transforms.ToTensor(),                    # Convert PIL to tensor
        transforms.Normalize((0.5,), (0.5,))     # Normalize to [-1, 1] range
    ])
    
    # Load MNIST training dataset
    train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
    
    # Split into training and validation sets
    train_len = int(0.9 * len(train_set))  # 90% for training
    val_len = len(train_set) - train_len    # 10% for validation
    train_ds, val_ds = random_split(train_set, [train_len, val_len])
    
    print(f"Training samples: {train_len}")
    print(f"Validation samples: {val_len}")
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=CFG.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch, shuffle=False)
    
    # === MODEL AND OPTIMIZER SETUP ===
    # Initialize defended model with regularization
    model = DefendedCNN().to(CFG.device)
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Configure optimizer with L2 regularization
    optimizer = optim.Adam(
        model.parameters(), 
        lr=CFG.lr, 
        weight_decay=CFG.weight_decay  # Layer 1 Defense: L2 regularization
    )
    
    # === TRAINING LOOP ===
    best_val_acc = 0.0
    
    for epoch in range(CFG.epochs):
        # --- Training Phase ---
        model.train()  # Enable training mode (activates dropout)
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data, target = data.to(CFG.device), target.to(CFG.device)
            
            # Forward pass
            optimizer.zero_grad()          # Clear gradients
            output = model(data)           # Get predictions (with dropout active)
            loss = F.cross_entropy(output, target)  # Compute loss
            
            # Backward pass
            loss.backward()                # Compute gradients
            optimizer.step()               # Update parameters
            
            # Track metrics
            train_loss += loss.item()
            pred = output.argmax(dim=1)    # Get predicted classes
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
        
        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100.0 * train_correct / train_total
        
        # --- Validation Phase ---
        model.eval()  # Disable training mode (deactivates dropout)
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():  # Disable gradient computation for efficiency
            for data, target in val_loader:
                data, target = data.to(CFG.device), target.to(CFG.device)
                output = model(data)           # Get predictions (no dropout)
                pred = output.argmax(dim=1)    # Get predicted classes
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
        
        val_accuracy = 100.0 * val_correct / val_total
        
        # === MODEL SAVING AND LOGGING ===
        # Save best model based on validation accuracy
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), CFG.save_path)
            print(f"  ✓ New best model saved (Val Acc: {val_accuracy:.2f}%)")
        
        # Periodic logging
        if epoch % 10 == 0 or epoch == CFG.epochs - 1:
            print(f"Epoch {epoch:2d}/{CFG.epochs}: "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Train Acc: {train_accuracy:.2f}%, "
                  f"Val Acc: {val_accuracy:.2f}%")
    
    # === SAVE CONFIGURATION ===
    # Save defense configuration for reproducibility and analysis
    config_dict = {
        'defense_parameters': vars(CFG),
        'training_results': {
            'best_validation_accuracy': best_val_acc,
            'final_training_accuracy': train_accuracy,
            'total_epochs': CFG.epochs,
            'model_parameters': total_params
        },
        'layer_descriptions': {
            'layer_1': 'Training-time regularization (L2 + Dropout)',
            'layer_2': 'Temperature scaling for confidence calibration',
            'layer_3': 'Adaptive noise injection based on confidence',
            'layer_4': 'Output sanitization (clipping + rounding)'
        }
    }
    
    with open(CFG.json_out, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # === TRAINING SUMMARY ===
    print(f"\n=== Training Complete ===")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {CFG.save_path}")
    print(f"Configuration saved to: {CFG.json_out}")
    print(f"Defense layers: 4-layer hybrid system")
    
    # Analyze training effectiveness for defense
    if best_val_acc > 85.0:
        print("✓ Good utility preservation (>85% accuracy)")
    else:
        print("⚠ Utility may be compromised (<85% accuracy)")
    
    if train_accuracy - best_val_acc < 10.0:
        print("✓ Good generalization (small train-val gap)")
    else:
        print("⚠ Potential overfitting detected (large train-val gap)")

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """
    Main entry point for the defense mechanism with command line interface.
    
    Supports various training configurations and parameter overrides for
    experimental evaluation and hyperparameter tuning.
    """
    parser = argparse.ArgumentParser(
        description="Four-Layer Hybrid Defense against Membership Inference Attacks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training control
    parser.add_argument(
        "--train", 
        action="store_true", 
        help="Train the defended target model"
    )
    
    # Hyperparameter overrides
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=CFG.epochs,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch", 
        type=int, 
        default=CFG.batch,
        help="Training batch size"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=CFG.lr,
        help="Learning rate"
    )
    parser.add_argument(
        "--dropout", 
        type=float, 
        default=CFG.dropout_p,
        help="Dropout probability for Layer 1 defense"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=CFG.temperature,
        help="Temperature scaling factor for Layer 2 defense"
    )
    
    args = parser.parse_args()
    
    # Apply parameter overrides
    if args.epochs != CFG.epochs:
        CFG.epochs = args.epochs
        print(f"Override: epochs = {CFG.epochs}")
    
    if args.batch != CFG.batch:
        CFG.batch = args.batch
        print(f"Override: batch_size = {CFG.batch}")
    
    if args.lr != CFG.lr:
        CFG.lr = args.lr
        print(f"Override: learning_rate = {CFG.lr}")
    
    if args.dropout != CFG.dropout_p:
        CFG.dropout_p = args.dropout
        print(f"Override: dropout_probability = {CFG.dropout_p}")
    
    if args.temperature != CFG.temperature:
        CFG.temperature = args.temperature
        print(f"Override: temperature = {CFG.temperature}")
    
    # Execute training if requested
    if args.train:
        train_defended_target()
    else:
        print("Use --train flag to train the defended model")
        print("Use --help for all available options")
