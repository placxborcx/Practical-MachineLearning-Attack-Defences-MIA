"""
Purpose: Optimized PyTorch implementation of a black-box Membership Inference Attack.
Enhanced with improved transferability, timing metrics, and comprehensive analysis.
Target: AUC > 0.60, TPR@FPR=0.1 > 0.20
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



# -----------------------------
# 1. Enhanced Global Configuration
# -----------------------------
class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SHADOW = 128
    BATCH_ATTACK = 256
    EPOCHS_SHADOW = 20  # Increased for better overfitting
    EPOCHS_ATTACK = 40  # Increased for better convergence
    N_SHADOW = 6  # Increased for better generalization
    NUM_CLASSES = 10
    SEED = 1337
    
    # Data configuration
    SHADOW_SIZE = 3000     # Smaller to encourage overfitting
    SHADOW_TEST = 800      # Smaller test set
    
    # Target model configuration
    TARGET_SIZE = 3000     # Match shadow model size
    TARGET_EPOCHS = 25     # More epochs for target to encourage overfitting
    
    # Non-member dataset sizes
    FASHION_SIZE = 2000
    KMNIST_SIZE = 2000
    CIFAR_SIZE = 2000
    NOISE_SIZE = 1000
    
    # Learning rates
    SHADOW_LR = 2e-3       # Higher LR for faster convergence
    ATTACK_LR = 1e-2
    TARGET_LR = 2e-3       
    
    # Attack improvements
    USE_AUGMENTATION = False  # Disable augmentation for clearer membership signal
    TEMPERATURE_SCALING = False
    ENSEMBLE_SHADOWS = True
    
    # Logging
    LOG_DIR = "./mia_results"
    SAVE_MODELS = True

config = Config()

# Set random seeds for reproducibility
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.SEED)
    torch.backends.cudnn.deterministic = True

# Create logging directory with proper error handling
try:
    os.makedirs(config.LOG_DIR, exist_ok=True)
except Exception as e:
    print(f"Warning: Could not create directory {config.LOG_DIR}: {e}")
    config.LOG_DIR = "."  # Use current directory as fallback
    config.SAVE_MODELS = False

print(f"Using device: {config.DEVICE}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Log directory: {config.LOG_DIR}")

# Timing decorator
def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result, end_time - start_time
    return wrapper

# -----------------------------
# 2. Simplified CNN Architecture (easier to overfit)
# -----------------------------
class SimpleCNN(nn.Module):
    """Simplified CNN that's easier to overfit for stronger membership signal"""
    def __init__(self, num_classes=10, dropout_rate=0.1):  # Less dropout
        super().__init__()
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second conv block
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# -----------------------------
# 3. Optimized Feature Extraction
# -----------------------------
def extract_attack_features(logits):
    """Extract features optimized for membership inference"""
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    
    # Sort probabilities
    sorted_probs, _ = probs.sort(dim=1, descending=True)
    
    # Core features that distinguish members from non-members
    max_prob = sorted_probs[:, 0]
    
    # Entropy - members typically have lower entropy
    entropy = -(probs * log_probs).sum(dim=1)
    
    # Confidence gap - members typically have larger gaps
    if probs.shape[1] > 1:
        confidence_gap = sorted_probs[:, 0] - sorted_probs[:, 1]
    else:
        confidence_gap = sorted_probs[:, 0]
    
    # Modified entropy (more sensitive to high confidence)
    modified_entropy = -torch.sum(probs * torch.log(probs + 1e-20), dim=1)
    
    # Standard deviation of probabilities
    prob_std = probs.std(dim=1)
    
    # Create feature vector
    features = torch.stack([
        max_prob,
        entropy,
        confidence_gap,
        modified_entropy,
        prob_std,
        sorted_probs[:, 0],  # Highest prob again (important feature)
        1.0 - entropy,  # Inverse entropy
    ], dim=1)
    
    return features.detach()

# -----------------------------
# 4. Data Loading
# -----------------------------
def load_datasets():
    """Load and prepare all datasets"""
    print("Loading datasets...")
    
    # Standard transform (no augmentation for clearer signal)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # CIFAR transform
    cifar_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load datasets
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    fashion_mnist = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    kmnist = datasets.KMNIST(root="./data", train=True, download=True, transform=transform)
    cifar10 = datasets.CIFAR10(root="./data", train=True, download=True, transform=cifar_transform)
    
    print(f"Datasets loaded successfully")
    
    return mnist_train, mnist_test, fashion_mnist, kmnist, cifar10

def safe_collate_fn(batch):
    """Custom collate function to handle mixed data types"""
    data_list = []
    target_list = []
    
    for item in batch:
        data, target = item
        data_list.append(data)
        
        # Ensure target is a tensor
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.long)
        elif target.dtype != torch.long:
            target = target.long()
            
        target_list.append(target)
    
    return torch.stack(data_list), torch.stack(target_list)



def create_attack_datasets(mnist_train, fashion_mnist, kmnist, cifar10):
    """Create datasets for shadow models and target"""
    print("Creating attack datasets...")
    
    # Calculate total needed
    shadow_data_per_model = config.SHADOW_SIZE + config.SHADOW_TEST
    total_shadow_needed = config.N_SHADOW * shadow_data_per_model
    target_needed = config.TARGET_SIZE + 1000  # Extra for validation
    
    total_needed = total_shadow_needed + target_needed
    total_available = len(mnist_train)
    
    if total_available < total_needed:
        # Adjust sizes if needed
        scale = total_available / total_needed * 0.9
        config.SHADOW_SIZE = int(config.SHADOW_SIZE * scale)
        config.SHADOW_TEST = int(config.SHADOW_TEST * scale)
        config.TARGET_SIZE = int(config.TARGET_SIZE * scale)
        print(f"Adjusted sizes - Shadow: {config.SHADOW_SIZE}, Target: {config.TARGET_SIZE}")
    
    # Split indices
    all_indices = np.random.permutation(len(mnist_train))
    
    # Target data comes first
    target_indices = all_indices[:target_needed]
    
    # Shadow data
    shadow_start = target_needed
    shadow_indices = all_indices[shadow_start:shadow_start + total_shadow_needed]
    
    # Create non-member datasets
    fashion_indices = np.random.choice(len(fashion_mnist), config.FASHION_SIZE, replace=False)
    kmnist_indices = np.random.choice(len(kmnist), config.KMNIST_SIZE, replace=False)
    cifar_indices = np.random.choice(len(cifar10), config.CIFAR_SIZE, replace=False)
    
    # Create noise dataset with proper tensor types
    noise_data = torch.randn(config.NOISE_SIZE, 1, 28, 28) * 0.3
    # Convert to same format as other datasets - targets as long tensors  
    noise_labels = torch.randint(0, config.NUM_CLASSES, (config.NOISE_SIZE,), dtype=torch.long)
    noise_dataset = TensorDataset(noise_data, noise_labels)
    
    # Combine non-member datasets
    nonmember_datasets = ConcatDataset([
        Subset(fashion_mnist, fashion_indices),
        Subset(kmnist, kmnist_indices),
        Subset(cifar10, cifar_indices),
        noise_dataset
    ])
    
    print(f"Shadow indices: {len(shadow_indices)}, Target indices: {len(target_indices)}")
    
    return shadow_indices, target_indices, nonmember_datasets

# -----------------------------
# 5. Shadow Model Training
# -----------------------------
@time_function
def train_model_with_overfitting(model, train_loader, val_loader, epochs, lr, model_name="model"):
    """Train model to deliberately overfit on training data"""
    print(f"Training {model_name}...")
    
    # Use SGD for more overfitting
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    
    best_train_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for data, target in train_loader:
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
        
        train_acc = 100. * train_correct / train_total
        train_loss /= len(train_loader)
        
        # Validation (just for monitoring)
        if val_loader is not None:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(config.DEVICE), target.to(config.DEVICE)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    val_correct += pred.eq(target).sum().item()
                    val_total += target.size(0)
            val_acc = 100. * val_correct / val_total
        else:
            val_acc = 0
        
        if epoch % 5 == 0:
            print(f'  Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # Save best model based on training loss (to get most overfitted model)
        if train_loss < best_train_loss and config.SAVE_MODELS:
            best_train_loss = train_loss
            torch.save(model.state_dict(), f"{config.LOG_DIR}/{model_name}_best.pth")
    
    # Load best model
    if config.SAVE_MODELS and os.path.exists(f"{config.LOG_DIR}/{model_name}_best.pth"):
        model.load_state_dict(torch.load(f"{config.LOG_DIR}/{model_name}_best.pth"))
    
    return model

@time_function
def collect_attack_features(models, train_loaders, test_loader, external_loader):
    """Collect features for attack model training"""
    print("Collecting features from models...")
    
    all_member_features = []
    all_nonmember_features = []
    
    # Collect from each shadow model
    for i, (model, train_loader) in enumerate(zip(models, train_loaders)):
        print(f"  Processing shadow model {i}...")
        model.eval()
        
        # Member features (from training data)
        with torch.no_grad():
            for data, _ in train_loader:
                data = data.to(config.DEVICE)
                logits = model(data)
                features = extract_attack_features(logits)
                all_member_features.append(features.cpu())
        
        # Non-member features (from test data)
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(config.DEVICE)
                logits = model(data)
                features = extract_attack_features(logits)
                all_nonmember_features.append(features.cpu())
                break  # Only use part of test data per model
    
    # Add external non-member data
    if external_loader is not None:
        print("  Processing external non-member data...")
        for model in models[:2]:  # Use first 2 models for external data
            model.eval()
            with torch.no_grad():
                for data, _ in external_loader:
                    data = data.to(config.DEVICE)
                    logits = model(data)
                    features = extract_attack_features(logits)
                    all_nonmember_features.append(features.cpu())
    
    # Combine features
    member_features = torch.cat(all_member_features, dim=0)
    nonmember_features = torch.cat(all_nonmember_features, dim=0)
    
    print(f"Collected {len(member_features)} member and {len(nonmember_features)} non-member features")
    
    return member_features, nonmember_features

# -----------------------------
# 6. Attack Model
# -----------------------------
class AttackModel(nn.Module):
    """Simple but effective attack model"""
    def __init__(self, input_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, 2)
        )
    
    def forward(self, x):
        return self.net(x)

@time_function
def train_attack_model(X_train, y_train, X_val, y_val):
    """Train the attack classifier"""
    print("Training attack classifier...")
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_ATTACK, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_ATTACK)
    
    # Initialize model
    model = AttackModel(input_dim=X_train.shape[1]).to(config.DEVICE)
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=config.ATTACK_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    best_val_auc = 0
    patience_counter = 0
    
    for epoch in range(config.EPOCHS_ATTACK):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(config.DEVICE), batch_y.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = F.cross_entropy(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_probs = []
        val_true = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(config.DEVICE)
                outputs = model(batch_x)
                probs = F.softmax(outputs, dim=1)[:, 1]
                val_probs.extend(probs.cpu().numpy())
                val_true.extend(batch_y.numpy())
        
        val_auc = roc_auc_score(val_true, val_probs)
        scheduler.step(val_auc)
        
        if epoch % 10 == 0:
            print(f'  Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, Val AUC: {val_auc:.4f}')
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            if config.SAVE_MODELS:
                torch.save(model.state_dict(), f"{config.LOG_DIR}/attack_model_best.pth")
        else:
            patience_counter += 1
            if patience_counter > 10:
                print(f"  Early stopping at epoch {epoch}")
                break
    
    # Load best model
    if config.SAVE_MODELS and os.path.exists(f"{config.LOG_DIR}/attack_model_best.pth"):
        model.load_state_dict(torch.load(f"{config.LOG_DIR}/attack_model_best.pth"))
    
    print(f"Best validation AUC: {best_val_auc:.4f}")
    return model

# -----------------------------
# 7. Evaluation
# -----------------------------
def evaluate_attack(model, X_test, y_test, scaler, title="Test"):
    """Evaluate attack performance"""
    model.eval()
    
    # Scale features
    X_test_scaled = torch.tensor(scaler.transform(X_test.numpy())).float()
    test_dataset = TensorDataset(X_test_scaled, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_ATTACK)
    
    all_probs = []
    all_true = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(config.DEVICE)
            outputs = model(batch_x)
            probs = F.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_true.extend(batch_y.numpy())
    
    all_probs = np.array(all_probs)
    all_true = np.array(all_true)
    
    # Calculate metrics
    auc = roc_auc_score(all_true, all_probs)
    fpr, tpr, thresholds = roc_curve(all_true, all_probs)
    
    # TPR at FPR = 0.1
    tpr_at_fpr01 = tpr[np.where(fpr <= 0.1)[0][-1]] if np.any(fpr <= 0.1) else 0
    
    # Optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    predictions = (all_probs >= optimal_threshold).astype(int)
    accuracy = accuracy_score(all_true, predictions)
    
    print(f"\n[{title}] Results:")
    print(f"  AUC: {auc:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  TPR@FPR=0.1: {tpr_at_fpr01:.4f}")
    
    return {
        'auc': auc,
        'accuracy': accuracy,
        'tpr_at_fpr01': tpr_at_fpr01,
        'fpr': fpr,
        'tpr': tpr,
        'probs': all_probs,
        'true': all_true
    }

# -----------------------------
# 8. Main Execution
# -----------------------------
def main():
    """Main execution function"""
    print("=== Membership Inference Attack ===")
    print(f"Target: AUC > 0.60, TPR@FPR=0.1 > 0.20\n")
    
    total_start = time.time()
    
    # Load data
    mnist_train, mnist_test, fashion_mnist, kmnist, cifar10 = load_datasets()
    shadow_indices, target_indices, nonmember_datasets = create_attack_datasets(
        mnist_train, fashion_mnist, kmnist, cifar10)
    
    # Train shadow models
    print("\n=== Training Shadow Models ===")
    shadow_models = []
    shadow_train_loaders = []
    
    for i in range(config.N_SHADOW):
        # Get data for this shadow model
        start_idx = i * (config.SHADOW_SIZE + config.SHADOW_TEST)
        end_idx = start_idx + config.SHADOW_SIZE
        
        train_indices = shadow_indices[start_idx:end_idx]
        train_subset = Subset(mnist_train, train_indices)
        
        # Small validation set
        val_size = int(0.1 * len(train_subset))
        train_size = len(train_subset) - val_size
        train_data, val_data = random_split(train_subset, [train_size, val_size])
        
        train_loader = DataLoader(train_data, batch_size=config.BATCH_SHADOW, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=config.BATCH_SHADOW)
        
        # Train shadow model
        model = SimpleCNN().to(config.DEVICE)
        model = train_model_with_overfitting(
            model, train_loader, val_loader, 
            config.EPOCHS_SHADOW, config.SHADOW_LR, f"shadow_{i}"
        )[0]  # Extract model from tuple
        
        shadow_models.append(model)
        shadow_train_loaders.append(DataLoader(train_subset, batch_size=config.BATCH_ATTACK))
    
    # Collect features
    print("\n=== Collecting Features ===")
    test_loader = DataLoader(mnist_test, batch_size=config.BATCH_ATTACK, shuffle=True)
    external_loader = DataLoader(nonmember_datasets, batch_size=config.BATCH_ATTACK, 
                           shuffle=True, collate_fn=safe_collate_fn)
    
    member_features, nonmember_features = collect_attack_features(
        shadow_models, shadow_train_loaders, test_loader, external_loader
    )[0]  # Extract from tuple
    
    # Prepare attack data
    print("\n=== Preparing Attack Data ===")
    # Balance classes
    min_samples = min(len(member_features), len(nonmember_features))
    member_features = member_features[:min_samples]
    nonmember_features = nonmember_features[:min_samples]
    
    # Create labels
    member_labels = torch.ones(len(member_features), dtype=torch.long)
    nonmember_labels = torch.zeros(len(nonmember_features), dtype=torch.long)
    
    # Combine and shuffle
    all_features = torch.cat([member_features, nonmember_features])
    all_labels = torch.cat([member_labels, nonmember_labels])
    
    perm = torch.randperm(len(all_features))
    all_features = all_features[perm]
    all_labels = all_labels[perm]
    
    # Scale features
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features.numpy())
    all_features_scaled = torch.tensor(all_features_scaled, dtype=torch.float32)
    
    # Split data
    n_train = int(0.7 * len(all_features_scaled))
    n_val = int(0.15 * len(all_features_scaled))
    
    X_train = all_features_scaled[:n_train]
    y_train = all_labels[:n_train]
    X_val = all_features_scaled[n_train:n_train+n_val]
    y_val = all_labels[n_train:n_train+n_val]
    X_test = all_features_scaled[n_train+n_val:]
    y_test = all_labels[n_train+n_val:]
    
    # Train attack model
    print("\n=== Training Attack Model ===")
    attack_model = train_attack_model(X_train, y_train, X_val, y_val)[0]
    
    # Evaluate on shadow data
    print("\n=== Evaluating Attack ===")
    shadow_results = evaluate_attack(attack_model, X_test, y_test, scaler, "Shadow Models")
    
    # Train target model
    print("\n=== Training Target Model ===")
    target_train_indices = target_indices[:config.TARGET_SIZE]
    target_train_data = Subset(mnist_train, target_train_indices)
    
    # Split for validation
    val_size = int(0.1 * len(target_train_data))
    train_size = len(target_train_data) - val_size
    train_data, val_data = random_split(target_train_data, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SHADOW, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.BATCH_SHADOW)
    
    target_model = SimpleCNN().to(config.DEVICE)
    target_model = train_model_with_overfitting(
        target_model, train_loader, val_loader,
        config.TARGET_EPOCHS, config.TARGET_LR, "target"
    )[0]
    
    # Evaluate on target model
    print("\n=== Evaluating on Target Model ===")
    
    # Extract features from target model
    target_model.eval()
    
    # Member features (training data)
    member_features = []
    target_train_loader = DataLoader(target_train_data, batch_size=config.BATCH_ATTACK)
    with torch.no_grad():
        for data, _ in target_train_loader:
            data = data.to(config.DEVICE)
            logits = target_model(data)
            features = extract_attack_features(logits)
            member_features.append(features.cpu())
    
    # Non-member features (test data)
    nonmember_features = []
    test_subset = Subset(mnist_test, np.random.choice(len(mnist_test), 3000, replace=False))
    test_loader = DataLoader(test_subset, batch_size=config.BATCH_ATTACK)
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(config.DEVICE)
            logits = target_model(data)
            features = extract_attack_features(logits)
            nonmember_features.append(features.cpu())
    
    # Combine features
    member_features = torch.cat(member_features)
    nonmember_features = torch.cat(nonmember_features)
    
    # Create labels
    member_labels = torch.ones(len(member_features), dtype=torch.long)
    nonmember_labels = torch.zeros(len(nonmember_features), dtype=torch.long)
    
    # Combine
    target_features = torch.cat([member_features, nonmember_features])
    target_labels = torch.cat([member_labels, nonmember_labels])
    
    # Evaluate
    target_results = evaluate_attack(attack_model, target_features, target_labels, scaler, "Target Model")
    
    # Summary
    total_time = time.time() - total_start
    print(f"\n=== Summary ===")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"\nShadow Models - AUC: {shadow_results['auc']:.4f}, TPR@FPR=0.1: {shadow_results['tpr_at_fpr01']:.4f}")
    print(f"Target Model - AUC: {target_results['auc']:.4f}, TPR@FPR=0.1: {target_results['tpr_at_fpr01']:.4f}")
    
    # Check if we met the targets
    if target_results['auc'] > 0.60 and target_results['tpr_at_fpr01'] > 0.20:
        print("\n✓ SUCCESS: Attack meets performance targets!")
    else:
        print("\n✗ Attack did not meet all performance targets")
    
    # Save results
    results = {
        "config": {
            "n_shadow": config.N_SHADOW,
            "shadow_size": config.SHADOW_SIZE,
            "target_size": config.TARGET_SIZE,
            "epochs_shadow": config.EPOCHS_SHADOW,
            "epochs_attack": config.EPOCHS_ATTACK,
            "epochs_target": config.TARGET_EPOCHS,
        },
        "results": {
            "shadow_test": {
                "auc": float(shadow_results['auc']),
                "accuracy": float(shadow_results['accuracy']),
                "tpr_at_fpr01": float(shadow_results['tpr_at_fpr01']),
            },
            "target_model": {
                "auc": float(target_results['auc']),
                "accuracy": float(target_results['accuracy']),
                "tpr_at_fpr01": float(target_results['tpr_at_fpr01']),
            }
        },
        "timing": {
            "total_seconds": total_time,
            "total_minutes": total_time / 60,
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "success": target_results['auc'] > 0.60 and target_results['tpr_at_fpr01'] > 0.20
    }
    
    # Save to JSON
    with open(f"{config.LOG_DIR}/attack_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # Plot results
    plot_attack_results(shadow_results, target_results)
    
    return attack_model, target_model, shadow_results, target_results

def plot_attack_results(shadow_results, target_results):
    """Plot ROC curves and score distributions"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curves
    axes[0].plot(shadow_results['fpr'], shadow_results['tpr'], 
                label=f"Shadow (AUC={shadow_results['auc']:.3f})", linewidth=2)
    axes[0].plot(target_results['fpr'], target_results['tpr'], 
                label=f"Target (AUC={target_results['auc']:.3f})", linewidth=2)
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0].axvline(x=0.1, color='red', linestyle=':', alpha=0.7, label='FPR=0.1')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Score Distributions for Target Model
    member_scores = target_results['probs'][target_results['true'] == 1]
    nonmember_scores = target_results['probs'][target_results['true'] == 0]
    
    axes[1].hist(nonmember_scores, bins=30, alpha=0.7, label='Non-members', density=True, color='blue')
    axes[1].hist(member_scores, bins=30, alpha=0.7, label='Members', density=True, color='red')
    axes[1].set_xlabel('Attack Score')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Target Model Score Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{config.LOG_DIR}/attack_results.png", dpi=300, bbox_inches='tight')
    plt.show()

# -----------------------------
# 9. Analysis Functions
# -----------------------------
def analyze_feature_importance(attack_model, scaler, feature_names=None):
    """Analyze which features are most important for the attack"""
    if feature_names is None:
        feature_names = [
            "Max Probability",
            "Entropy", 
            "Confidence Gap",
            "Modified Entropy",
            "Probability Std",
            "Top Probability",
            "Inverse Entropy"
        ]
    
    # Create random inputs
    n_samples = 1000
    X = torch.randn(n_samples, len(feature_names))
    X_scaled = torch.tensor(scaler.transform(X.numpy())).float()
    
    attack_model.eval()
    X_scaled.requires_grad = True
    
    # Get gradients
    outputs = attack_model(X_scaled.to(config.DEVICE))
    outputs[:, 1].sum().backward()
    
    # Average absolute gradients
    importance = X_scaled.grad.abs().mean(dim=0).cpu().numpy()
    
    # Plot
    plt.figure(figsize=(8, 6))
    indices = np.argsort(importance)[::-1]
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
    plt.ylabel('Importance Score')
    plt.title('Feature Importance for Membership Inference')
    plt.tight_layout()
    plt.savefig(f"{config.LOG_DIR}/feature_importance.png", dpi=300)
    plt.show()

def analyze_overfitting_impact(target_model, attack_model, scaler, mnist_train, mnist_test):
    """Analyze how model overfitting affects attack success"""
    print("\n=== Analyzing Overfitting Impact ===")
    
    # Train models with different levels of overfitting
    overfitting_levels = [5, 10, 20, 40]  # Different epoch counts
    attack_results = []
    
    for epochs in overfitting_levels:
        print(f"\nTraining model with {epochs} epochs...")
        
        # Train a new model
        model = SimpleCNN().to(config.DEVICE)
        subset = Subset(mnist_train, np.random.choice(len(mnist_train), 3000, replace=False))
        train_loader = DataLoader(subset, batch_size=config.BATCH_SHADOW, shuffle=True)
        
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        for epoch in range(epochs):
            model.train()
            for data, target in train_loader:
                data, target = data.to(config.DEVICE), target.to(config.DEVICE)
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
        
        # Evaluate attack
        model.eval()
        
        # Extract features
        member_features = []
        with torch.no_grad():
            for data, _ in train_loader:
                data = data.to(config.DEVICE)
                logits = model(data)
                features = extract_attack_features(logits)
                member_features.append(features.cpu())
        
        test_loader = DataLoader(mnist_test, batch_size=config.BATCH_ATTACK)
        nonmember_features = []
        count = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(config.DEVICE)
                logits = model(data)
                features = extract_attack_features(logits)
                nonmember_features.append(features.cpu())
                count += 1
                if count >= 10:  # Use subset of test data
                    break
        
        # Combine features
        member_features = torch.cat(member_features)
        nonmember_features = torch.cat(nonmember_features)
        
        # Balance
        min_samples = min(len(member_features), len(nonmember_features))
        member_features = member_features[:min_samples]
        nonmember_features = nonmember_features[:min_samples]
        
        # Create dataset
        features = torch.cat([member_features, nonmember_features])
        labels = torch.cat([torch.ones(min_samples), torch.zeros(min_samples)]).long()
        
        # Evaluate
        results = evaluate_attack(attack_model, features, labels, scaler, f"Epochs={epochs}")
        attack_results.append((epochs, results['auc']))
    
    # Plot results
    plt.figure(figsize=(8, 6))
    epochs_list, auc_list = zip(*attack_results)
    plt.plot(epochs_list, auc_list, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=0.6, color='red', linestyle='--', label='Target AUC')
    plt.xlabel('Training Epochs')
    plt.ylabel('Attack AUC')
    plt.title('Attack Success vs Model Overfitting')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{config.LOG_DIR}/overfitting_analysis.png", dpi=300)
    plt.show()

# Run the attack
if __name__ == "__main__":
    # Run main attack
    attack_model, target_model, shadow_results, target_results = main()
    
    # Additional analyses
    print("\n=== Running Additional Analyses ===")
    
    # Load scaler (you'd need to save this in main() for reuse)
    # For now, we'll skip the additional analyses in automatic execution
    
    print("\nAttack completed! Check results in", config.LOG_DIR)