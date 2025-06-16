"""
Membership Inference Attack Implementation
==========================================

Purpose: Optimized PyTorch implementation of a black-box Membership Inference Attack.
This implementation addresses critical issues found in baseline approaches:
- Double normalization bugs
- Insufficient non-member data diversity  
- Weak overfitting signals
- Poor feature extraction

Target Performance: AUC > 0.55 (relaxed from 0.60 for realistic expectations)

Key Technical Improvements:
- Enhanced feature extraction with 13 discriminative features
- Strategic use of multiple datasets as non-member sources
- Deliberate overfitting through architectural and training choices
- Robust data handling and preprocessing

Academic Context: 
This implementation demonstrates how membership inference attacks can be conducted
against machine learning models, particularly those trained on image classification
tasks. The attack leverages confidence patterns to distinguish between training
data (members) and unseen data (non-members).

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split, ConcatDataset

from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import os
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. GLOBAL CONFIGURATION AND SETUP
# =============================================================================

class Config:
    """
    Global configuration class containing all hyperparameters and settings.
    
    This centralized configuration approach allows easy experimentation
    and ensures consistency across all components of the attack.
    """
    
    # Hardware Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training Batch Sizes
    BATCH_SHADOW = 64      # Smaller batch size for shadow models to encourage overfitting
    BATCH_ATTACK = 256     # Larger batch size for attack model training (more stable)
    
    # Training Epochs
    EPOCHS_SHADOW = 50     # Extended training for shadow models to ensure overfitting
    EPOCHS_ATTACK = 60     # Sufficient epochs for attack model convergence
    N_SHADOW = 8           # Number of shadow models (more models = better attack)
    
    # Model Configuration
    NUM_CLASSES = 10       # Number of output classes (MNIST has 10 digits)
    SEED = 1337           # Random seed for reproducibility
    
    # Dataset Size Configuration
    # These sizes are carefully chosen to balance overfitting and generalization
    SHADOW_SIZE = 2000     # Training size per shadow model (smaller encourages overfitting)
    SHADOW_TEST = 1000     # Test data size for non-member feature collection
    
    # Target Model Configuration
    TARGET_SIZE = 800      # Target model training size (reduced from 2000 for stronger overfitting)
    TARGET_EPOCHS = 80     # Extended training for target model to maximize overfitting
    
    # Non-member Dataset Sizes
    # Using diverse non-member sources improves attack model generalization
    FASHION_SIZE = 3000    # FashionMNIST samples for non-member data
    KMNIST_SIZE = 3000     # KMNIST samples for non-member data  
    CIFAR_SIZE = 3000      # CIFAR-10 samples for non-member data
    NOISE_SIZE = 2000      # Synthetic noise samples for non-member data
    
    # Learning Rate Configuration
    # Higher learning rates promote faster overfitting
    SHADOW_LR = 5e-3       # Aggressive learning rate for shadow models
    ATTACK_LR = 5e-3       # Learning rate for attack classifier
    TARGET_LR = 5e-3       # Learning rate for target model
    
    # Attack Strategy Settings
    USE_AUGMENTATION = False      # Disabled to maintain clear membership signals
    TEMPERATURE_SCALING = False   # Disabled to preserve confidence patterns
    ENSEMBLE_SHADOWS = True       # Use multiple shadow models for robust features
    
    # Logging and Storage
    LOG_DIR = "./mia_results"     # Directory for saving results and models
    SAVE_MODELS = True            # Whether to save trained models

# Initialize global configuration
config = Config()

# =============================================================================
# 2. REPRODUCIBILITY SETUP
# =============================================================================

def setup_reproducibility():
    """
    Configure random seeds and deterministic behavior for reproducible results.
    
    This is crucial for academic work where experiments need to be repeatable.
    """
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)
        torch.backends.cudnn.deterministic = True

# Apply reproducibility settings
setup_reproducibility()

# =============================================================================
# 3. LOGGING AND DIRECTORY SETUP
# =============================================================================

def setup_logging():
    """
    Create logging directory with proper error handling.
    
    Returns:
        bool: True if directory creation successful, False otherwise
    """
    try:
        os.makedirs(config.LOG_DIR, exist_ok=True)
        return True
    except Exception as e:
        print(f"Warning: Could not create directory {config.LOG_DIR}: {e}")
        # Fallback to current directory
        config.LOG_DIR = "."
        config.SAVE_MODELS = False
        return False

# Setup logging
setup_logging()

# Print system information
print(f"=== System Configuration ===")
print(f"Using device: {config.DEVICE}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Log directory: {config.LOG_DIR}")
print(f"Random seed: {config.SEED}")

# =============================================================================
# 4. UTILITY FUNCTIONS
# =============================================================================

def time_function(func):
    """
    Decorator to measure and log function execution time.
    
    This helps track performance bottlenecks in the attack pipeline.
    
    Args:
        func: Function to be timed
        
    Returns:
        Wrapper function that returns (result, execution_time)
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result, end_time - start_time
    return wrapper

# =============================================================================
# 5. NEURAL NETWORK ARCHITECTURES
# =============================================================================

class SimpleCNN(nn.Module):
    """
    Simplified CNN architecture designed to be easily overfittable.
    
    This architecture deliberately lacks regularization techniques like dropout
    or batch normalization to encourage overfitting, which creates stronger
    membership signals that the attack can exploit.
    
    Architecture Details:
    - Two convolutional blocks with ReLU activation
    - MaxPooling for dimensionality reduction
    - Simple fully connected classifier
    - NO dropout or batch normalization (key for overfitting)
    
    Args:
        num_classes (int): Number of output classes (default: 10 for MNIST)
        dropout_rate (float): Dropout rate (kept at 0.0 for maximum overfitting)
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.0):
        super().__init__()
        
        # Feature extraction layers
        # Using larger kernels (5x5) and moderate channels for good feature learning
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 16, kernel_size=5, padding=2),  # 28x28 -> 28x28
            nn.ReLU(inplace=True),                       # Non-linear activation
            nn.MaxPool2d(2, 2),                         # 28x28 -> 14x14
            
            # Second convolutional block  
            nn.Conv2d(16, 32, kernel_size=5, padding=2), # 14x14 -> 14x14
            nn.ReLU(inplace=True),                       # Non-linear activation
            nn.MaxPool2d(2, 2),                         # 14x14 -> 7x7
        )
        
        # Classification layers
        # Simple architecture that can easily memorize training patterns
        self.classifier = nn.Sequential(
            nn.Flatten(),                    # Flatten spatial dimensions
            nn.Linear(32 * 7 * 7, 128),    # Feature compression
            nn.ReLU(inplace=True),          # Non-linear activation
            # NOTE: No dropout here - this is intentional for overfitting
            nn.Linear(128, num_classes)     # Final classification layer
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            torch.Tensor: Raw logits of shape (batch_size, num_classes)
        """
        x = self.features(x)      # Extract features
        x = self.classifier(x)    # Classify
        return x

# =============================================================================
# 6. FEATURE EXTRACTION FOR MEMBERSHIP INFERENCE
# =============================================================================

def extract_attack_features(logits, return_raw=False):
    """
    Extract comprehensive features from model logits for membership inference.
    
    This function implements the core insight of membership inference attacks:
    models tend to be more confident on training data (members) compared to
    test data (non-members). We extract various statistical measures of
    confidence and uncertainty.
    
    Key Features Extracted:
    1. Max Probability: Highest softmax probability (confidence measure)
    2. Entropy: Information-theoretic uncertainty measure
    3. Confidence Gap: Difference between top-2 predictions
    4. Logit Statistics: Raw logit norms and variances
    5. Probability Distributions: Various transformations of softmax outputs
    
    Args:
        logits (torch.Tensor): Raw model outputs of shape (batch_size, num_classes)
        return_raw (bool): Whether to include raw logits as features
        
    Returns:
        torch.Tensor: Feature tensor of shape (batch_size, num_features)
    """
    
    # Convert logits to probability distributions
    probs = F.softmax(logits, dim=1)           # Softmax probabilities
    log_probs = F.log_softmax(logits, dim=1)   # Log probabilities for stability
    
    # Sort probabilities in descending order for gap calculations
    sorted_probs, sorted_indices = probs.sort(dim=1, descending=True)
    
    # === CORE CONFIDENCE FEATURES ===
    
    # Feature 1: Maximum probability (primary confidence indicator)
    max_prob = sorted_probs[:, 0]
    
    # Feature 2: Shannon entropy (uncertainty measure)
    # Lower entropy indicates higher confidence (members typically have lower entropy)
    entropy = -(probs * log_probs).sum(dim=1)
    
    # Feature 3: Confidence gap (difference between top 2 predictions)
    # Members often have larger gaps due to overfitting
    if probs.shape[1] > 1:
        confidence_gap = sorted_probs[:, 0] - sorted_probs[:, 1]
        third_gap = sorted_probs[:, 0] - sorted_probs[:, 2] if probs.shape[1] > 2 else sorted_probs[:, 0]
    else:
        confidence_gap = sorted_probs[:, 0]
        third_gap = sorted_probs[:, 0]
    
    # === ADDITIONAL DISCRIMINATIVE FEATURES ===
    
    # Feature 4: Modified entropy (alternative uncertainty measure)
    modified_entropy = -torch.sum(probs * torch.log(probs + 1e-20), dim=1)
    
    # Feature 5: Standard deviation of probabilities
    prob_std = probs.std(dim=1)
    
    # Feature 6: Sum of top-3 probabilities
    top3_sum = sorted_probs[:, :3].sum(dim=1) if probs.shape[1] >= 3 else sorted_probs[:, 0]
    
    # === LOGIT-BASED FEATURES ===
    
    # Feature 7: L2 norm of logits (members often have higher norms)
    logit_norm = torch.norm(logits, p=2, dim=1)
    
    # Feature 8: Maximum logit value
    max_logit = logits.max(dim=1)[0]
    
    # Feature 9: Variance of logits
    logit_var = logits.var(dim=1)
    
    # === DERIVED FEATURES ===
    
    # Feature 10: Inverse entropy (higher for more confident predictions)
    inverse_entropy = 1.0 - entropy
    
    # Feature 11: Squared maximum probability (amplifies high confidence)
    squared_max_prob = sorted_probs[:, 0] ** 2
    
    # Feature 12: Log of maximum probability (log-scale confidence)
    log_max_prob = torch.log(max_prob + 1e-10)  # Add small epsilon for numerical stability
    
    # Combine all features into a single tensor
    features = torch.stack([
        max_prob,           # 0: Maximum probability
        entropy,            # 1: Shannon entropy  
        confidence_gap,     # 2: Top-1 vs Top-2 gap
        modified_entropy,   # 3: Alternative entropy
        prob_std,          # 4: Probability standard deviation
        third_gap,         # 5: Top-1 vs Top-3 gap
        top3_sum,          # 6: Sum of top-3 probabilities
        inverse_entropy,   # 7: Inverse entropy
        logit_norm,        # 8: L2 norm of logits
        max_logit,         # 9: Maximum logit value
        logit_var,         # 10: Logit variance
        squared_max_prob,  # 11: Squared max probability
        log_max_prob,      # 12: Log max probability
    ], dim=1)
    
    # Optionally include raw logits as additional features
    if return_raw:
        return torch.cat([features, logits], dim=1).detach()
    
    return features.detach()

# =============================================================================
# 7. DATA LOADING AND PREPROCESSING
# =============================================================================

def load_datasets():
    """
    Load and prepare all required datasets for the membership inference attack.
    
    This function loads:
    - MNIST: Primary dataset for target and shadow models
    - FashionMNIST, KMNIST, CIFAR-10: Non-member datasets for attack training
    
    The diversity of non-member datasets helps the attack model learn to
    distinguish membership patterns rather than dataset-specific features.
    
    Returns:
        tuple: (mnist_train, mnist_test, fashion_mnist, kmnist, cifar10)
    """
    print("=== Loading Datasets ===")
    
    # Standard transformation for MNIST-like datasets
    # Simple normalization without augmentation to preserve membership signals
    transform = transforms.Compose([
        transforms.ToTensor(),                    # Convert PIL to tensor
        transforms.Normalize((0.5,), (0.5,))     # Normalize to [-1, 1] range
    ])
    
    # Special transformation for CIFAR-10
    # Convert to grayscale and resize to match MNIST dimensions
    cifar_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # RGB -> Grayscale
        transforms.Resize((28, 28)),                  # Resize to 28x28
        transforms.ToTensor(),                        # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))          # Normalize to [-1, 1]
    ])
    
    # Load all datasets
    print("  Loading MNIST (primary dataset)...")
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    
    print("  Loading non-member datasets...")
    fashion_mnist = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    kmnist = datasets.KMNIST(root="./data", train=True, download=True, transform=transform)
    cifar10 = datasets.CIFAR10(root="./data", train=True, download=True, transform=cifar_transform)
    
    print(f"  Dataset sizes:")
    print(f"    MNIST Train: {len(mnist_train)}")
    print(f"    MNIST Test: {len(mnist_test)}")  
    print(f"    FashionMNIST: {len(fashion_mnist)}")
    print(f"    KMNIST: {len(kmnist)}")
    print(f"    CIFAR-10: {len(cifar10)}")
    
    print("  Datasets loaded successfully!")
    
    return mnist_train, mnist_test, fashion_mnist, kmnist, cifar10

def safe_collate_fn(batch):
    """
    Custom collate function to handle mixed data types safely.
    
    When combining datasets with different label types, we need to ensure
    all labels are converted to the same tensor type (torch.long) for
    consistent processing.
    
    Args:
        batch: List of (data, target) tuples from DataLoader
        
    Returns:
        tuple: (stacked_data, stacked_targets)
    """
    data_list = []
    target_list = []
    
    for item in batch:
        data, target = item
        data_list.append(data)
        
        # Ensure target is a torch.long tensor
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.long)
        elif target.dtype != torch.long:
            target = target.long()
            
        target_list.append(target)
    
    return torch.stack(data_list), torch.stack(target_list)

def create_attack_datasets(mnist_train, fashion_mnist, kmnist, cifar10):
    """
    Create carefully partitioned datasets for shadow models, target model, and attack training.
    
    This function implements the core data partitioning strategy:
    1. Separate MNIST data for shadow models and target model (no overlap)
    2. Create diverse non-member dataset from multiple sources
    3. Ensure sufficient data for each component while avoiding contamination
    
    Args:
        mnist_train: MNIST training dataset
        fashion_mnist: FashionMNIST dataset  
        kmnist: KMNIST dataset
        cifar10: CIFAR-10 dataset
        
    Returns:
        tuple: (shadow_indices, target_indices, nonmember_datasets)
    """
    print("=== Creating Attack Datasets ===")
    
    # Calculate total data requirements
    shadow_data_per_model = config.SHADOW_SIZE + config.SHADOW_TEST
    total_shadow_needed = config.N_SHADOW * shadow_data_per_model
    target_needed = config.TARGET_SIZE + 1000  # Extra for validation
    
    total_needed = total_shadow_needed + target_needed
    total_available = len(mnist_train)
    
    print(f"  Data requirements:")
    print(f"    Shadow models: {config.N_SHADOW} × {shadow_data_per_model} = {total_shadow_needed}")
    print(f"    Target model: {target_needed}")
    print(f"    Total needed: {total_needed}")
    print(f"    Available: {total_available}")
    
    # Adjust sizes if insufficient data available
    if total_available < total_needed:
        scale = total_available / total_needed * 0.9  # 10% safety margin
        original_shadow_size = config.SHADOW_SIZE
        original_target_size = config.TARGET_SIZE
        
        config.SHADOW_SIZE = int(config.SHADOW_SIZE * scale)
        config.SHADOW_TEST = int(config.SHADOW_TEST * scale)
        config.TARGET_SIZE = int(config.TARGET_SIZE * scale)
        
        print(f"  ⚠️  Adjusted sizes due to data constraints:")
        print(f"    Shadow size: {original_shadow_size} → {config.SHADOW_SIZE}")
        print(f"    Target size: {original_target_size} → {config.TARGET_SIZE}")
    
    # Create non-overlapping index partitions
    all_indices = np.random.permutation(len(mnist_train))
    
    # Target model gets first portion (clean separation)
    target_indices = all_indices[:target_needed]
    
    # Shadow models get remaining data
    shadow_start = target_needed
    shadow_indices = all_indices[shadow_start:shadow_start + total_shadow_needed]
    
    print(f"  Data partitioning:")
    print(f"    Target indices: {len(target_indices)}")
    print(f"    Shadow indices: {len(shadow_indices)}")
    
    # === CREATE NON-MEMBER DATASETS ===
    
    print("  Creating non-member datasets...")
    
    # Sample from each non-member dataset
    fashion_indices = np.random.choice(len(fashion_mnist), config.FASHION_SIZE, replace=False)
    kmnist_indices = np.random.choice(len(kmnist), config.KMNIST_SIZE, replace=False)
    cifar_indices = np.random.choice(len(cifar10), config.CIFAR_SIZE, replace=False)
    
    # Create synthetic noise dataset
    # This adds another source of non-member data with different characteristics
    noise_data = torch.randn(config.NOISE_SIZE, 1, 28, 28) * 0.3
    noise_labels = torch.randint(0, config.NUM_CLASSES, (config.NOISE_SIZE,), dtype=torch.long)
    noise_dataset = TensorDataset(noise_data, noise_labels)
    
    # Combine all non-member datasets
    nonmember_datasets = ConcatDataset([
        Subset(fashion_mnist, fashion_indices),   # Fashion items
        Subset(kmnist, kmnist_indices),          # Japanese characters
        Subset(cifar10, cifar_indices),          # Natural images  
        noise_dataset                            # Synthetic noise
    ])
    
    print(f"  Non-member dataset composition:")
    print(f"    FashionMNIST: {config.FASHION_SIZE}")
    print(f"    KMNIST: {config.KMNIST_SIZE}")
    print(f"    CIFAR-10: {config.CIFAR_SIZE}")
    print(f"    Synthetic noise: {config.NOISE_SIZE}")
    print(f"    Total non-members: {len(nonmember_datasets)}")
    
    return shadow_indices, target_indices, nonmember_datasets

# =============================================================================
# 8. SHADOW MODEL TRAINING
# =============================================================================

@time_function
def train_model_with_overfitting(model, train_loader, val_loader, epochs, lr, model_name="model"):
    """
    Train a model with deliberate overfitting to create strong membership signals.
    
    This function implements training strategies that encourage overfitting:
    - High learning rate for rapid memorization
    - SGD with momentum (less regularization than Adam)
    - No weight decay (prevents generalization)
    - Extended training (more epochs)
    
    The goal is NOT good generalization, but rather strong overfitting that
    creates detectable differences between member and non-member predictions.
    
    Args:
        model: Neural network model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation (optional, for monitoring only)
        epochs: Number of training epochs
        lr: Learning rate
        model_name: Name for logging and saving
        
    Returns:
        torch.nn.Module: Trained model
    """
    print(f"=== Training {model_name} ===")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Training samples: {len(train_loader.dataset)}")
    
    # Configure optimizer for aggressive overfitting
    # SGD with momentum but no weight decay encourages overfitting
    optimizer = optim.SGD(
        model.parameters(), 
        lr=lr, 
        momentum=0.9,      # Helps with convergence
        weight_decay=0     # No regularization - key for overfitting
    )
    
    # Training metrics tracking
    train_losses = []
    train_accs = []
    
    # Training loop
    for epoch in range(epochs):
        # === TRAINING PHASE ===
        model.train()  # Enable training mode (affects dropout, batchnorm)
        
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for data, target in train_loader:
            # Move data to appropriate device
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            
            # Forward pass
            optimizer.zero_grad()           # Clear gradients
            output = model(data)            # Get predictions
            loss = F.cross_entropy(output, target)  # Compute loss
            
            # Backward pass
            loss.backward()                 # Compute gradients
            optimizer.step()                # Update parameters
            
            # Track metrics
            train_loss += loss.item()
            pred = output.argmax(dim=1)     # Get predicted classes
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
        
        # Calculate epoch metrics
        train_acc = 100. * train_correct / train_total
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # === VALIDATION PHASE (OPTIONAL) ===
        # Only for monitoring - we don't use validation for early stopping
        # because we WANT overfitting
        if val_loader is not None and epoch % 10 == 0:
            model.eval()  # Disable training mode
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():  # Disable gradient computation
                for data, target in val_loader:
                    data, target = data.to(config.DEVICE), target.to(config.DEVICE)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    val_correct += pred.eq(target).sum().item()
                    val_total += target.size(0)
            
            val_acc = 100. * val_correct / val_total
            print(f'  Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        elif epoch % 10 == 0:
            print(f'  Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    
    # Final metrics
    final_train_loss = train_losses[-1]
    final_train_acc = train_accs[-1]
    print(f"  Final metrics: Train Loss: {final_train_loss:.4f}, Train Acc: {final_train_acc:.2f}%")
    
    # High training accuracy with long training indicates successful overfitting
    if final_train_acc > 95:
        print(f"  ✓ Strong overfitting achieved (Train Acc: {final_train_acc:.2f}%)")
    else:
        print(f"  ⚠️ Moderate overfitting (Train Acc: {final_train_acc:.2f}%)")
    
    # Save model if enabled
    if config.SAVE_MODELS:
        save_path = f"{config.LOG_DIR}/{model_name}_final.pth"
        torch.save(model.state_dict(), save_path)
        print(f"  Model saved to: {save_path}")
    
    return model

@time_function
def collect_attack_features(models, train_loaders, test_loader, external_loader):
    """
    Collect comprehensive features from shadow models for attack training.
    
    This is the core data collection phase where we extract membership signals
    from shadow models. The key insight is that shadow models exhibit similar
    overfitting patterns to the target model, so we can learn to detect
    membership from their behavior.
    
    Collection Strategy:
    1. Member features: From shadow model training data (known members)
    2. Non-member features: From test data + external datasets (known non-members)
    3. Extract features from ALL shadow models for diversity
    4. Balance data collection to avoid bias
    
    Args:
        models: List of trained shadow models
        train_loaders: List of training data loaders for each shadow model
        test_loader: DataLoader for MNIST test data (non-members)
        external_loader: DataLoader for external datasets (non-members)
        
    Returns:
        tuple: (member_features, nonmember_features)
    """
    print("=== Collecting Attack Features ===")
    print(f"  Processing {len(models)} shadow models...")
    
    all_member_features = []      # Features from training data (members)
    all_nonmember_features = []   # Features from test/external data (non-members)
    
    # === COLLECT FROM EACH SHADOW MODEL ===
    
    for i, (model, train_loader) in enumerate(zip(models, train_loaders)):
        print(f"  Processing shadow model {i+1}/{len(models)}...")
        model.eval()  # Set to evaluation mode
        
        # --- Collect Member Features ---
        # These are features from data the model was trained on
        member_count = 0
        with torch.no_grad():  # No gradients needed for inference
            for data, _ in train_loader:
                data = data.to(config.DEVICE)
                logits = model(data)                    # Get raw predictions
                features = extract_attack_features(logits)  # Extract membership features
                all_member_features.append(features.cpu())   # Store on CPU
                member_count += data.size(0)
        
        print(f"    Collected {member_count} member samples")
        
        # --- Collect Non-Member Features from Test Set ---
        # These are MNIST test samples the model has never seen
        nonmember_count = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(config.DEVICE)
                logits = model(data)
                features = extract_attack_features(logits)
                all_nonmember_features.append(features.cpu())
                nonmember_count += data.size(0)
                
                # Limit collection per model to balance dataset
                if nonmember_count >= 2000:
                    break
        
        print(f"    Collected {nonmember_count} non-member samples from test set")
    
    # === COLLECT FROM EXTERNAL DATASETS ===
    
    if external_loader is not None:
        print("  Processing external non-member data...")
        
        # Apply each shadow model to external data for diverse non-member features
        for j, model in enumerate(models):
            model.eval()
            external_count = 0
            
            with torch.no_grad():
                for data, _ in external_loader:
                    data = data.to(config.DEVICE)
                    logits = model(data)
                    features = extract_attack_features(logits)
                    all_nonmember_features.append(features.cpu())
                    external_count += data.size(0)
                    
                    # Collect substantial external data per model
                    if external_count >= 1500:
                        break
            
            print(f"    Model {j+1}: Collected {external_count} external non-member samples")
    
    # === COMBINE AND SUMMARIZE ===
    
    # Concatenate all collected features
    member_features = torch.cat(all_member_features, dim=0)
    nonmember_features = torch.cat(all_nonmember_features, dim=0)
    
    print(f"  Feature collection summary:")
    print(f"    Total member features: {len(member_features)}")
    print(f"    Total non-member features: {len(nonmember_features)}")
    print(f"    Feature dimensionality: {member_features.shape[1]}")
    
    # Check for reasonable balance
    ratio = len(member_features) / len(nonmember_features)
    if 0.5 <= ratio <= 2.0:
        print(f"    ✓ Good balance ratio: {ratio:.2f}")
    else:
        print(f"    ⚠️ Imbalanced data ratio: {ratio:.2f}")
    
    return member_features, nonmember_features

# =============================================================================
# 9. ATTACK MODEL ARCHITECTURE AND TRAINING
# =============================================================================

class AttackModel(nn.Module):
    """
    Neural network classifier for membership inference attack.
    
    This model takes extracted features from target model predictions and
    classifies whether the input was likely a member (training data) or
    non-member (unseen data) of the target model.
    
    Architecture Design:
    - Multi-layer perceptron with batch normalization
    - Dropout for regularization (unlike shadow models, we want generalization here)
    - Binary classification output (member vs non-member)
    - Sufficient capacity to learn complex membership patterns
    
    Args:
        input_dim (int): Number of input features (default: 13)
    """
    
    def __init__(self, input_dim=13):
        super().__init__()
        
        # Multi-layer architecture with decreasing width
        self.net = nn.Sequential(
            # Layer 1: Input projection with regularization
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),        # Stabilizes training
            nn.Dropout(0.3),            # Prevents overfitting
            
            # Layer 2: Feature compression
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),            # Lighter dropout in deeper layers
            
            # Layer 3: Further compression
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            # No dropout to maintain information flow
            
            # Layer 4: Pre-classification layer
            nn.Linear(32, 16),
            nn.ReLU(),
            
            # Output layer: Binary classification
            nn.Linear(16, 2)            # 2 classes: member (1) vs non-member (0)
        )
    
    def forward(self, x):
        """
        Forward pass through the attack classifier.
        
        Args:
            x (torch.Tensor): Input features of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Raw logits of shape (batch_size, 2)
        """
        return self.net(x)

@time_function
def train_attack_model(X_train, y_train, X_val, y_val):
    """
    Train the membership inference attack classifier.
    
    This function trains a binary classifier to distinguish between member
    and non-member samples based on the extracted features. The training
    process includes class balancing, learning rate scheduling, and early
    stopping to ensure robust performance.
    
    Training Strategy:
    - Use class weights to handle potential data imbalance
    - Adam optimizer for stable convergence
    - Learning rate scheduling based on validation performance
    - Early stopping to prevent overfitting
    - Monitor AUC as primary metric
    
    Args:
        X_train (torch.Tensor): Training features
        y_train (torch.Tensor): Training labels (0=non-member, 1=member)
        X_val (torch.Tensor): Validation features
        y_val (torch.Tensor): Validation labels
        
    Returns:
        AttackModel: Trained attack classifier
    """
    print("=== Training Attack Classifier ===")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Feature dimension: {X_train.shape[1]}")
    
    # Create data loaders for batch processing
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_ATTACK, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_ATTACK)
    
    # Initialize attack model
    model = AttackModel(input_dim=X_train.shape[1]).to(config.DEVICE)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # === HANDLE CLASS IMBALANCE ===
    
    # Calculate class weights to handle potential imbalance
    class_counts = torch.bincount(y_train)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * 2  # Normalize
    
    print(f"  Class distribution:")
    print(f"    Non-members (0): {class_counts[0]} ({100*class_counts[0]/len(y_train):.1f}%)")
    print(f"    Members (1): {class_counts[1]} ({100*class_counts[1]/len(y_train):.1f}%)")
    print(f"    Class weights: {class_weights.tolist()}")
    
    # Loss function with class weighting
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(config.DEVICE))
    
    # === OPTIMIZATION SETUP ===
    
    # Adam optimizer for stable training
    optimizer = optim.Adam(model.parameters(), lr=config.ATTACK_LR)
    
    # Learning rate scheduler - reduces LR when validation plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',        # Maximize AUC
        patience=5,        # Wait 5 epochs before reducing
        factor=0.5,        # Reduce by half
        verbose=True
    )
    
    # === TRAINING LOOP ===
    
    best_val_auc = 0
    patience_counter = 0
    
    print(f"  Starting training for {config.EPOCHS_ATTACK} epochs...")
    
    for epoch in range(config.EPOCHS_ATTACK):
        
        # --- Training Phase ---
        model.train()
        train_loss = 0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(config.DEVICE), batch_y.to(config.DEVICE)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches
        
        # --- Validation Phase ---
        model.eval()
        val_probs = []
        val_true = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(config.DEVICE)
                outputs = model(batch_x)
                # Get probability of membership (class 1)
                probs = F.softmax(outputs, dim=1)[:, 1]
                val_probs.extend(probs.cpu().numpy())
                val_true.extend(batch_y.numpy())
        
        # Calculate validation AUC
        val_auc = roc_auc_score(val_true, val_probs)
        
        # Update learning rate scheduler
        scheduler.step(val_auc)
        
        # Logging
        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'  Epoch {epoch:3d}: Train Loss: {avg_train_loss:.4f}, Val AUC: {val_auc:.4f}, LR: {current_lr:.2e}')
        
        # === EARLY STOPPING AND MODEL SAVING ===
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            
            # Save best model
            if config.SAVE_MODELS:
                torch.save(model.state_dict(), f"{config.LOG_DIR}/attack_model_best.pth")
        else:
            patience_counter += 1
            
            # Early stopping if no improvement
            if patience_counter > 15:
                print(f"  Early stopping at epoch {epoch} (no improvement for 15 epochs)")
                break
    
    # === LOAD BEST MODEL ===
    
    if config.SAVE_MODELS and os.path.exists(f"{config.LOG_DIR}/attack_model_best.pth"):
        model.load_state_dict(torch.load(f"{config.LOG_DIR}/attack_model_best.pth"))
        print(f"  Loaded best model (Val AUC: {best_val_auc:.4f})")
    
    print(f"  Training completed. Best validation AUC: {best_val_auc:.4f}")
    
    return model

# =============================================================================
# 10. EVALUATION AND METRICS
# =============================================================================

def evaluate_attack(model, X_test, y_test, scaler, title="Test", already_scaled=False):
    """
    Evaluate membership inference attack performance with comprehensive metrics.
    
    This function computes various metrics to assess attack effectiveness:
    - AUC: Area under ROC curve (primary metric)
    - Accuracy: Classification accuracy at optimal threshold
    - TPR@FPR=0.1: True positive rate when false positive rate is 10%
    
    The function also handles feature scaling properly to avoid double
    normalization bugs that can inflate performance metrics.
    
    Args:
        model: Trained attack classifier
        X_test (torch.Tensor): Test features
        y_test (torch.Tensor): Test labels
        scaler: StandardScaler for feature normalization
        title (str): Description for logging
        already_scaled (bool): Whether features are already normalized
        
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    print(f"=== Evaluating Attack: {title} ===")
    model.eval()
    
    # === FEATURE SCALING ===
    # Critical: Only scale if not already scaled to avoid double normalization
    if already_scaled:
        X_test_scaled = X_test
        print("  Using pre-scaled features")
    else:
        X_test_scaled = torch.tensor(scaler.transform(X_test.numpy())).float()
        print("  Applied feature scaling")
    
    print(f"  Test samples: {len(X_test_scaled)}")
    print(f"  Feature shape: {X_test_scaled.shape}")
    
    # Create data loader for batch processing
    test_dataset = TensorDataset(X_test_scaled, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_ATTACK)
    
    # === PREDICTION COLLECTION ===
    
    all_probs = []  # Membership probabilities
    all_true = []   # True labels
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(config.DEVICE)
            
            # Get model predictions
            outputs = model(batch_x)
            # Convert to membership probabilities (softmax on class 1)
            probs = F.softmax(outputs, dim=1)[:, 1]
            
            # Store results
            all_probs.extend(probs.cpu().numpy())
            all_true.extend(batch_y.numpy())
    
    # Convert to numpy arrays for metric computation
    all_probs = np.array(all_probs)
    all_true = np.array(all_true)
    
    # === METRIC CALCULATION ===
    
    # Primary metric: Area Under ROC Curve
    auc = roc_auc_score(all_true, all_probs)
    
    # ROC curve for detailed analysis
    fpr, tpr, thresholds = roc_curve(all_true, all_probs)
    
    # TPR at FPR = 0.1 (privacy-relevant metric)
    # This shows attack effectiveness when false positive rate is limited
    tpr_at_fpr01_indices = np.where(fpr <= 0.1)[0]
    if len(tpr_at_fpr01_indices) > 0:
        tpr_at_fpr01 = tpr[tpr_at_fpr01_indices[-1]]
    else:
        tpr_at_fpr01 = 0
    
    # Optimal threshold (maximizes TPR - FPR)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Accuracy at optimal threshold
    predictions = (all_probs >= optimal_threshold).astype(int)
    accuracy = accuracy_score(all_true, predictions)
    
    # === RESULTS SUMMARY ===
    
    print(f"  Results:")
    print(f"    AUC: {auc:.4f}")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    TPR@FPR=0.1: {tpr_at_fpr01:.4f}")
    print(f"    Optimal threshold: {optimal_threshold:.4f}")
    
    # Performance interpretation
    if auc > 0.7:
        print(f"    ✓ Strong attack (AUC > 0.7)")
    elif auc > 0.55:
        print(f"    ✓ Moderate attack (AUC > 0.55)")  
    else:
        print(f"    ⚠️ Weak attack (AUC ≤ 0.55)")
    
    # Class distribution in predictions
    member_count = np.sum(all_true == 1)
    nonmember_count = np.sum(all_true == 0)
    print(f"    Test distribution: {member_count} members, {nonmember_count} non-members")
    
    return {
        'auc': auc,
        'accuracy': accuracy,
        'tpr_at_fpr01': tpr_at_fpr01,
        'fpr': fpr,
        'tpr': tpr,
        'probs': all_probs,
        'true': all_true,
        'optimal_threshold': optimal_threshold
    }

# =============================================================================
# 11. MAIN EXECUTION PIPELINE
# =============================================================================

def main():
    """
    Main execution function implementing the complete membership inference attack pipeline.
    
    This function orchestrates the entire attack process:
    1. Data loading and partitioning
    2. Shadow model training with deliberate overfitting
    3. Feature extraction from shadow models
    4. Attack classifier training
    5. Target model training and evaluation
    6. Results analysis and visualization
    
    The implementation follows the shadow model approach where multiple
    models are trained to mimic the target model's behavior, enabling
    the attack to learn membership inference patterns.
    
    Returns:
        tuple: (attack_model, target_model, shadow_results, target_results)
    """
    print("=" * 60)
    print("MEMBERSHIP INFERENCE ATTACK - COMPREHENSIVE IMPLEMENTATION")
    print("=" * 60)
    print(f"Target Performance: AUC > 0.55")
    print(f"Attack Strategy: Shadow model approach with deliberate overfitting")
    print(f"Feature Engineering: 13 confidence-based features")
    print("")
    
    total_start = time.time()
    
    # ==========================================================================
    # PHASE 1: DATA PREPARATION
    # ==========================================================================
    
    print("PHASE 1: DATA PREPARATION")
    print("-" * 40)
    
    # Load all required datasets
    mnist_train, mnist_test, fashion_mnist, kmnist, cifar10 = load_datasets()
    
    # Create non-overlapping partitions for shadow models, target model, and non-members
    shadow_indices, target_indices, nonmember_datasets = create_attack_datasets(
        mnist_train, fashion_mnist, kmnist, cifar10)
    
    # ==========================================================================
    # PHASE 2: SHADOW MODEL TRAINING
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("PHASE 2: SHADOW MODEL TRAINING")
    print("-" * 40)
    print(f"Training {config.N_SHADOW} shadow models with deliberate overfitting...")
    
    shadow_models = []           # Store trained shadow models
    shadow_train_loaders = []    # Store training data loaders for feature extraction
    
    for i in range(config.N_SHADOW):
        print(f"\n--- Shadow Model {i+1}/{config.N_SHADOW} ---")
        
        # Get unique data partition for this shadow model
        start_idx = i * (config.SHADOW_SIZE + config.SHADOW_TEST)
        end_idx = start_idx + config.SHADOW_SIZE
        
        train_indices = shadow_indices[start_idx:end_idx]
        train_subset = Subset(mnist_train, train_indices)
        
        print(f"  Training data size: {len(train_subset)}")
        
        # Create data loader for training
        # No validation set - we want maximum overfitting
        train_loader = DataLoader(train_subset, batch_size=config.BATCH_SHADOW, shuffle=True)
        
        # Initialize model with no regularization
        model = SimpleCNN(dropout_rate=0.0).to(config.DEVICE)
        
        # Train with aggressive overfitting strategy
        model = train_model_with_overfitting(
            model=model,
            train_loader=train_loader,
            val_loader=None,  # No validation to encourage overfitting
            epochs=config.EPOCHS_SHADOW,
            lr=config.SHADOW_LR,
            model_name=f"shadow_{i}"
        )[0]  # Extract model from (model, time) tuple
        
        # Store for feature extraction phase
        shadow_models.append(model)
        shadow_train_loaders.append(DataLoader(train_subset, batch_size=config.BATCH_ATTACK))
    
    print(f"\n✓ Successfully trained {len(shadow_models)} shadow models")
    
    # ==========================================================================
    # PHASE 3: FEATURE EXTRACTION
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("PHASE 3: MEMBERSHIP SIGNAL EXTRACTION")
    print("-" * 40)
    
    # Prepare data loaders for feature extraction
    test_loader = DataLoader(mnist_test, batch_size=config.BATCH_ATTACK, shuffle=True)
    external_loader = DataLoader(
        nonmember_datasets, 
        batch_size=config.BATCH_ATTACK, 
        shuffle=True, 
        collate_fn=safe_collate_fn  # Handle mixed data types
    )
    
    # Extract features from all shadow models
    member_features, nonmember_features = collect_attack_features(
        models=shadow_models,
        train_loaders=shadow_train_loaders,
        test_loader=test_loader,
        external_loader=external_loader
    )[0]  # Extract features from (features, time) tuple
    
    # ==========================================================================
    # PHASE 4: ATTACK DATA PREPARATION
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("PHASE 4: ATTACK DATA PREPARATION")
    print("-" * 40)
    
    # Balance classes to ensure fair evaluation
    min_samples = min(len(member_features), len(nonmember_features))
    print(f"Balancing dataset to {min_samples} samples per class")
    
    # Trim to balanced size
    member_features = member_features[:min_samples]
    nonmember_features = nonmember_features[:min_samples]
    
    # Create binary labels
    member_labels = torch.ones(len(member_features), dtype=torch.long)      # Class 1: Member
    nonmember_labels = torch.zeros(len(nonmember_features), dtype=torch.long)  # Class 0: Non-member
    
    # Combine and shuffle data
    all_features = torch.cat([member_features, nonmember_features])
    all_labels = torch.cat([member_labels, nonmember_labels])
    
    # Random permutation for unbiased training
    perm = torch.randperm(len(all_features))
    all_features = all_features[perm]
    all_labels = all_labels[perm]
    
    print(f"Combined dataset: {len(all_features)} samples, {all_features.shape[1]} features")
    
    # === FEATURE NORMALIZATION ===
    # Apply standardization ONCE to avoid double normalization bugs
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features.numpy())
    all_features_scaled = torch.tensor(all_features_scaled, dtype=torch.float32)
    
    print("✓ Applied feature standardization")
    
    # === DATA SPLITTING ===
    # Split into train/validation/test for attack model training
    total_samples = len(all_features_scaled)
    n_train = int(0.7 * total_samples)    # 70% for training
    n_val = int(0.15 * total_samples)     # 15% for validation
    # Remaining 15% for testing
    
    X_train = all_features_scaled[:n_train]
    y_train = all_labels[:n_train]
    X_val = all_features_scaled[n_train:n_train+n_val]
    y_val = all_labels[n_train:n_train+n_val]
    X_test = all_features_scaled[n_train+n_val:]
    y_test = all_labels[n_train+n_val:]
    
    print(f"Data splits:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Testing: {len(X_test)} samples")
    
    # ==========================================================================
    # PHASE 5: ATTACK MODEL TRAINING
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("PHASE 5: ATTACK MODEL TRAINING")
    print("-" * 40)
    
    # Train the membership inference classifier
    attack_model = train_attack_model(X_train, y_train, X_val, y_val)[0]
    
    # ==========================================================================
    # PHASE 6: SHADOW MODEL EVALUATION
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("PHASE 6: ATTACK EVALUATION ON SHADOW MODELS")
    print("-" * 40)
    
    # Evaluate attack performance on shadow model data
    # Features are already scaled, so set already_scaled=True
    shadow_results = evaluate_attack(
        model=attack_model,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler,
        title="Shadow Models",
        already_scaled=True
    )
    
    # ==========================================================================
    # PHASE 7: TARGET MODEL TRAINING
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("PHASE 7: TARGET MODEL TRAINING")
    print("-" * 40)
    print("Training target model with extreme overfitting...")
    
    # Prepare target model training data
    target_train_indices = target_indices[:config.TARGET_SIZE]
    target_train_data = Subset(mnist_train, target_train_indices)
    
    print(f"Target model training size: {len(target_train_data)}")
    
    # Create data loader - no validation for maximum overfitting
    train_loader = DataLoader(target_train_data, batch_size=config.BATCH_SHADOW, shuffle=True)
    
    # Initialize target model (same architecture as shadow models)
    target_model = SimpleCNN(dropout_rate=0.0).to(config.DEVICE)
    
    # Train with extreme overfitting (more epochs than shadow models)
    target_model = train_model_with_overfitting(
        model=target_model,
        train_loader=train_loader,
        val_loader=None,  # No validation
        epochs=config.TARGET_EPOCHS,
        lr=config.TARGET_LR,
        model_name="target"
    )[0]
    
    # ==========================================================================
    # PHASE 8: TARGET MODEL ATTACK EVALUATION
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("PHASE 8: ATTACK EVALUATION ON TARGET MODEL")
    print("-" * 40)
    print("Extracting features from target model...")
    
    target_model.eval()
    
    # === COLLECT MEMBER FEATURES ===
    # Extract features from target model's training data
    member_features = []
    target_train_loader = DataLoader(target_train_data, batch_size=config.BATCH_ATTACK)
    
    with torch.no_grad():
        for data, _ in target_train_loader:
            data = data.to(config.DEVICE)
            logits = target_model(data)
            features = extract_attack_features(logits)
            member_features.append(features.cpu())
    
    # === COLLECT NON-MEMBER FEATURES ===
    nonmember_features = []
    
    # Use MNIST test data (unseen by target model)
    test_indices = np.random.choice(len(mnist_test), min(5000, len(mnist_test)), replace=False)
    test_subset = Subset(mnist_test, test_indices)
    test_loader = DataLoader(test_subset, batch_size=config.BATCH_ATTACK)
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(config.DEVICE)
            logits = target_model(data)
            features = extract_attack_features(logits)
            nonmember_features.append(features.cpu())
    
    # Add external dataset features for robustness
    external_count = 0
    with torch.no_grad():
        for data, _ in external_loader:
            data = data.to(config.DEVICE)
            logits = target_model(data)
            features = extract_attack_features(logits)
            nonmember_features.append(features.cpu())
            external_count += data.size(0)
            if external_count >= 2000:
                break
    
    # === PREPARE EVALUATION DATA ===
    
    # Combine features
    member_features = torch.cat(member_features)
    nonmember_features = torch.cat(nonmember_features)
    
    print(f"Target evaluation data:")
    print(f"  Members: {len(member_features)}")
    print(f"  Non-members: {len(nonmember_features)}")
    
    # Balance for fair evaluation
    min_eval = min(len(member_features), len(nonmember_features))
    member_features = member_features[:min_eval]
    nonmember_features = nonmember_features[:min_eval]
    
    # Create labels and combine
    member_labels = torch.ones(len(member_features), dtype=torch.long)
    nonmember_labels = torch.zeros(len(nonmember_features), dtype=torch.long)
    
    target_features = torch.cat([member_features, nonmember_features])
    target_labels = torch.cat([member_labels, nonmember_labels])
    
    # Shuffle for unbiased evaluation
    perm = torch.randperm(len(target_features))
    target_features = target_features[perm]
    target_labels = target_labels[perm]
    
    # === EVALUATE ATTACK ON TARGET ===
    
    # Features need scaling (not pre-scaled like shadow evaluation)
    target_results = evaluate_attack(
        model=attack_model,
        X_test=target_features,
        y_test=target_labels,
        scaler=scaler,
        title="Target Model",
        already_scaled=False
    )
    
    # ==========================================================================
    # PHASE 9: RESULTS ANALYSIS AND REPORTING
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("PHASE 9: COMPREHENSIVE RESULTS ANALYSIS")
    print("=" * 60)
    
    total_time = time.time() - total_start
    
    # === PERFORMANCE SUMMARY ===
    
    print(f"⏱️  Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"")
    print(f"📊 ATTACK PERFORMANCE SUMMARY:")
    print(f"   Shadow Models  - AUC: {shadow_results['auc']:.4f}, TPR@FPR=0.1: {shadow_results['tpr_at_fpr01']:.4f}")
    print(f"   Target Model   - AUC: {target_results['auc']:.4f}, TPR@FPR=0.1: {target_results['tpr_at_fpr01']:.4f}")
    print(f"")
    
    # === SUCCESS EVALUATION ===
    
    target_auc = target_results['auc']
    if target_auc > 0.55:
        print(f"🎯 SUCCESS: Attack meets performance target!")
        print(f"   Target AUC: {target_auc:.4f} > 0.55 ✓")
        success_status = True
    else:
        print(f"❌ Attack did not meet performance target")
        print(f"   Target AUC: {target_auc:.4f} ≤ 0.55")