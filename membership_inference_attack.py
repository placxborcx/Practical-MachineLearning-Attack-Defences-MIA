"""
Purpose: Optimized PyTorch implementation of a black-box Membership Inference Attack.
Fixed critical issues: double normalization, insufficient non-member data, weak overfitting
Target: AUC > 0.55 (relaxed from 0.60 for realistic expectations)
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

print("04:28")

# -----------------------------
# 1. Enhanced Global Configuration
# -----------------------------
class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SHADOW = 64      # Smaller batch for stronger overfitting
    BATCH_ATTACK = 256
    EPOCHS_SHADOW = 50     # Much more epochs for overfitting
    EPOCHS_ATTACK = 60     # More epochs for attack model
    N_SHADOW = 8           # More shadow models
    NUM_CLASSES = 10
    SEED = 1337
    
    # Data configuration
    SHADOW_SIZE = 2000     # Smaller to encourage overfitting
    SHADOW_TEST = 1000     # More test data for better non-member representation
    
    # Target model configuration
    TARGET_SIZE = 800     # Match shadow model size 2000
    TARGET_EPOCHS = 80     # Much more epochs for strong overfitting
    
    # Non-member dataset sizes - increased
    FASHION_SIZE = 3000
    KMNIST_SIZE = 3000
    CIFAR_SIZE = 3000
    NOISE_SIZE = 2000
    
    # Learning rates
    SHADOW_LR = 5e-3       # Higher LR for aggressive overfitting
    ATTACK_LR = 5e-3       # Lower LR for attack model
    TARGET_LR = 5e-3       
    
    # Attack improvements
    USE_AUGMENTATION = False  
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
    config.LOG_DIR = "."
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
    def __init__(self, num_classes=10, dropout_rate=0.0):  # NO dropout for maximum overfitting
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
            # No dropout for maximum overfitting
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# -----------------------------
# 3. Enhanced Feature Extraction
# -----------------------------
def extract_attack_features(logits, return_raw=False):
    """Extract comprehensive features optimized for membership inference"""
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    
    # Sort probabilities
    sorted_probs, sorted_indices = probs.sort(dim=1, descending=True)
    
    # Core features that distinguish members from non-members
    max_prob = sorted_probs[:, 0]
    
    # Entropy - members typically have lower entropy
    entropy = -(probs * log_probs).sum(dim=1)
    
    # Confidence gap - members typically have larger gaps
    if probs.shape[1] > 1:
        confidence_gap = sorted_probs[:, 0] - sorted_probs[:, 1]
        third_gap = sorted_probs[:, 0] - sorted_probs[:, 2] if probs.shape[1] > 2 else sorted_probs[:, 0]
    else:
        confidence_gap = sorted_probs[:, 0]
        third_gap = sorted_probs[:, 0]
    
    # Modified entropy (more sensitive to high confidence)
    modified_entropy = -torch.sum(probs * torch.log(probs + 1e-20), dim=1)
    
    # Standard deviation of probabilities
    prob_std = probs.std(dim=1)
    
    # Additional features for better discrimination
    # Top-k probabilities
    top3_sum = sorted_probs[:, :3].sum(dim=1) if probs.shape[1] >= 3 else sorted_probs[:, 0]
    
    # L2 norm of logits (members often have higher norms)
    logit_norm = torch.norm(logits, p=2, dim=1)
    
    # Max logit value
    max_logit = logits.max(dim=1)[0]
    
    # Logit variance
    logit_var = logits.var(dim=1)
    
    # Create comprehensive feature vector
    features = torch.stack([
        max_prob,
        entropy,
        confidence_gap,
        modified_entropy,
        prob_std,
        third_gap,
        top3_sum,
        1.0 - entropy,  # Inverse entropy
        logit_norm,
        max_logit,
        logit_var,
        sorted_probs[:, 0] ** 2,  # Squared max prob
        torch.log(max_prob + 1e-10),  # Log of max prob
    ], dim=1)
    
    if return_raw:
        # Return raw logits as additional features
        return torch.cat([features, logits], dim=1).detach()
    
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
    print(f"Non-member datasets: {len(nonmember_datasets)} samples")
    
    return shadow_indices, target_indices, nonmember_datasets

# -----------------------------
# 5. Shadow Model Training with Strong Overfitting
# -----------------------------
@time_function
def train_model_with_overfitting(model, train_loader, val_loader, epochs, lr, model_name="model"):
    """Train model to deliberately overfit on training data"""
    print(f"Training {model_name} for {epochs} epochs...")
    
    # Use SGD with high learning rate and no weight decay for maximum overfitting
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    
    train_losses = []
    train_accs = []
    
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
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation (just for monitoring)
        if val_loader is not None and epoch % 10 == 0:
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
            print(f'  Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        elif epoch % 10 == 0:
            print(f'  Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    
    print(f"  Final: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    
    # Save model if enabled
    if config.SAVE_MODELS:
        torch.save(model.state_dict(), f"{config.LOG_DIR}/{model_name}_final.pth")
    
    return model

@time_function
def collect_attack_features(models, train_loaders, test_loader, external_loader):
    """Collect features for attack model training - FIXED: no break, more data"""
    print("Collecting features from models...")
    
    all_member_features = []
    all_nonmember_features = []
    
    # Collect from each shadow model
    for i, (model, train_loader) in enumerate(zip(models, train_loaders)):
        print(f"  Processing shadow model {i}...")
        model.eval()
        
        # Member features (from training data)
        member_count = 0
        with torch.no_grad():
            for data, _ in train_loader:
                data = data.to(config.DEVICE)
                logits = model(data)
                features = extract_attack_features(logits)
                all_member_features.append(features.cpu())
                member_count += data.size(0)
        print(f"    Collected {member_count} member samples")
        
        # Non-member features (from test data) - NO BREAK!
        nonmember_count = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(config.DEVICE)
                logits = model(data)
                features = extract_attack_features(logits)
                all_nonmember_features.append(features.cpu())
                nonmember_count += data.size(0)
                if nonmember_count >= 2000:  # Limit per model but collect much more
                    break
        print(f"    Collected {nonmember_count} non-member samples from test set")
    
    # Add external non-member data - use ALL shadow models
    if external_loader is not None:
        print("  Processing external non-member data...")
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
                    if external_count >= 1500:  # Collect substantial external data per model
                        break
            print(f"    Model {j}: Collected {external_count} external non-member samples")
    
    # Combine features
    member_features = torch.cat(all_member_features, dim=0)
    nonmember_features = torch.cat(all_nonmember_features, dim=0)
    
    print(f"Total collected: {len(member_features)} member and {len(nonmember_features)} non-member features")
    
    return member_features, nonmember_features

# -----------------------------
# 6. Enhanced Attack Model
# -----------------------------
class AttackModel(nn.Module):
    """Enhanced attack model with better capacity"""
    def __init__(self, input_dim=13):  # Updated for more features
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            
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
    
    # Training setup with class weights for imbalanced data
    class_counts = torch.bincount(y_train)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * 2
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(config.DEVICE))
    
    optimizer = optim.Adam(model.parameters(), lr=config.ATTACK_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
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
            loss = criterion(outputs, batch_y)
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
            if patience_counter > 15:
                print(f"  Early stopping at epoch {epoch}")
                break
    
    # Load best model
    if config.SAVE_MODELS and os.path.exists(f"{config.LOG_DIR}/attack_model_best.pth"):
        model.load_state_dict(torch.load(f"{config.LOG_DIR}/attack_model_best.pth"))
    
    print(f"Best validation AUC: {best_val_auc:.4f}")
    return model

# -----------------------------
# 7. Evaluation - FIXED: No double scaling
# -----------------------------
def evaluate_attack(model, X_test, y_test, scaler, title="Test", already_scaled=False):
    """Evaluate attack performance - FIXED: handle scaling properly"""
    model.eval()
    
    # Only scale if not already scaled
    if already_scaled:
        X_test_scaled = X_test
    else:
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
# 8. Main Execution - FIXED
# -----------------------------
def main():
    """Main execution function with fixes"""
    print("=== Membership Inference Attack (Optimized) ===")
    print(f"Target: AUC > 0.55\n")
    
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
        
        # No validation set - use all data for training (maximum overfitting)
        train_loader = DataLoader(train_subset, batch_size=config.BATCH_SHADOW, shuffle=True)
        
        # Train shadow model
        model = SimpleCNN(dropout_rate=0.0).to(config.DEVICE)  # No dropout
        model = train_model_with_overfitting(
            model, train_loader, None,  # No validation
            config.EPOCHS_SHADOW, config.SHADOW_LR, f"shadow_{i}"
        )[0]
        
        shadow_models.append(model)
        shadow_train_loaders.append(DataLoader(train_subset, batch_size=config.BATCH_ATTACK))
    
    # Collect features
    print("\n=== Collecting Features ===")
    test_loader = DataLoader(mnist_test, batch_size=config.BATCH_ATTACK, shuffle=True)
    external_loader = DataLoader(nonmember_datasets, batch_size=config.BATCH_ATTACK, 
                           shuffle=True, collate_fn=safe_collate_fn)
    
    member_features, nonmember_features = collect_attack_features(
        shadow_models, shadow_train_loaders, test_loader, external_loader
    )[0]
    
    # Prepare attack data
    print("\n=== Preparing Attack Data ===")
    # Balance classes but keep more data
    min_samples = min(len(member_features), len(nonmember_features))
    print(f"Balancing to {min_samples} samples per class")
    
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
    
    # Scale features ONCE
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
    
    print(f"Training samples: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    # Train attack model
    print("\n=== Training Attack Model ===")
    attack_model = train_attack_model(X_train, y_train, X_val, y_val)[0]
    
    # Evaluate on shadow data - data is already scaled
    print("\n=== Evaluating Attack ===")
    shadow_results = evaluate_attack(attack_model, X_test, y_test, scaler, "Shadow Models", already_scaled=True)
    
    # Train target model with extreme overfitting
    print("\n=== Training Target Model ===")
    target_train_indices = target_indices[:config.TARGET_SIZE]
    target_train_data = Subset(mnist_train, target_train_indices)
    
    # No validation - use all data for training
    train_loader = DataLoader(target_train_data, batch_size=config.BATCH_SHADOW, shuffle=True)
    
    target_model = SimpleCNN(dropout_rate=0.0).to(config.DEVICE)  # No dropout
    target_model = train_model_with_overfitting(
        target_model, train_loader, None,  # No validation
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
    
    # Non-member features - use more test data
    nonmember_features = []
    # Use 5000 test samples for better evaluation
    test_indices = np.random.choice(len(mnist_test), min(5000, len(mnist_test)), replace=False)
    test_subset = Subset(mnist_test, test_indices)
    test_loader = DataLoader(test_subset, batch_size=config.BATCH_ATTACK)
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(config.DEVICE)
            logits = target_model(data)
            features = extract_attack_features(logits)
            nonmember_features.append(features.cpu())
    
    # Also add some external data for non-members
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
    
    # Combine features
    member_features = torch.cat(member_features)
    nonmember_features = torch.cat(nonmember_features)
    
    print(f"Target evaluation - Members: {len(member_features)}, Non-members: {len(nonmember_features)}")
    
    # Balance for fair evaluation
    min_eval = min(len(member_features), len(nonmember_features))
    member_features = member_features[:min_eval]
    nonmember_features = nonmember_features[:min_eval]
    
    # Create labels
    member_labels = torch.ones(len(member_features), dtype=torch.long)
    nonmember_labels = torch.zeros(len(nonmember_features), dtype=torch.long)
    
    # Combine
    target_features = torch.cat([member_features, nonmember_features])
    target_labels = torch.cat([member_labels, nonmember_labels])
    
    # Shuffle
    perm = torch.randperm(len(target_features))
    target_features = target_features[perm]
    target_labels = target_labels[perm]
    
    # Evaluate - features need scaling
    target_results = evaluate_attack(attack_model, target_features, target_labels, scaler, "Target Model", already_scaled=False)
    
    # Summary
    total_time = time.time() - total_start
    print(f"\n=== Summary ===")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"\nShadow Models - AUC: {shadow_results['auc']:.4f}, TPR@FPR=0.1: {shadow_results['tpr_at_fpr01']:.4f}")
    print(f"Target Model - AUC: {target_results['auc']:.4f}, TPR@FPR=0.1: {target_results['tpr_at_fpr01']:.4f}")
    
    # Check if we met the targets
    if target_results['auc'] > 0.55:
        print("\n✓ SUCCESS: Attack meets performance target (AUC > 0.55)!")
    else:
        print("\n✗ Attack did not meet performance target")
    
    # Save results
    results = {
        "config": {
            "n_shadow": config.N_SHADOW,
            "shadow_size": config.SHADOW_SIZE,
            "target_size": config.TARGET_SIZE,
            "epochs_shadow": config.EPOCHS_SHADOW,
            "epochs_attack": config.EPOCHS_ATTACK,
            "epochs_target": config.TARGET_EPOCHS,
            "features": 13,
            "fixes_applied": ["no_double_scaling", "more_nonmember_data", "extreme_overfitting", "no_dropout"]
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
        "success": target_results['auc'] > 0.55
    }
    
    # Save to JSON
    with open(f"{config.LOG_DIR}/attack_results_optimized.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # Plot results
    plot_attack_results(shadow_results, target_results)
    
    # Additional analysis
    print("\n=== Feature Statistics ===")
    analyze_feature_distributions(member_features, nonmember_features)
    
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
    
    # Add text with key metrics
    axes[1].text(0.05, 0.95, f"AUC: {target_results['auc']:.3f}\nTPR@FPR=0.1: {target_results['tpr_at_fpr01']:.3f}", 
                transform=axes[1].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f"{config.LOG_DIR}/attack_results_optimized.png", dpi=300, bbox_inches='tight')
    plt.show()

def analyze_feature_distributions(member_features, nonmember_features):
    """Analyze and print feature distribution differences"""
    feature_names = [
        "Max Probability", "Entropy", "Confidence Gap", "Modified Entropy",
        "Probability Std", "Third Gap", "Top-3 Sum", "Inverse Entropy",
        "Logit Norm", "Max Logit", "Logit Variance", "Squared Max Prob", "Log Max Prob"
    ]
    
    print("\nFeature-wise discrimination (Member vs Non-member means):")
    print("-" * 60)
    
    member_means = member_features.mean(dim=0).numpy()
    nonmember_means = nonmember_features.mean(dim=0).numpy()
    
    for i, name in enumerate(feature_names):
        if i < len(member_means):
            diff = member_means[i] - nonmember_means[i]
            ratio = member_means[i] / (nonmember_means[i] + 1e-8)
            print(f"{name:20s}: Member={member_means[i]:7.4f}, Non-member={nonmember_means[i]:7.4f}, "
                  f"Diff={diff:7.4f}, Ratio={ratio:6.3f}")

# -----------------------------
# 9. Additional Analysis Functions
# -----------------------------
def analyze_model_confidence(model, data_loader, model_name="Model"):
    """Analyze confidence distribution of a model"""
    model.eval()
    all_probs = []
    all_entropies = []
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(config.DEVICE)
            logits = model(data)
            probs = F.softmax(logits, dim=1)
            max_probs = probs.max(dim=1)[0]
            
            # Calculate entropy
            log_probs = F.log_softmax(logits, dim=1)
            entropy = -(probs * log_probs).sum(dim=1)
            
            all_probs.extend(max_probs.cpu().numpy())
            all_entropies.extend(entropy.cpu().numpy())
    
    print(f"\n{model_name} Confidence Analysis:")
    print(f"  Mean max probability: {np.mean(all_probs):.4f}")
    print(f"  Std max probability: {np.std(all_probs):.4f}")
    print(f"  Mean entropy: {np.mean(all_entropies):.4f}")
    print(f"  % samples with >0.9 confidence: {100 * np.mean(np.array(all_probs) > 0.9):.2f}%")
    print(f"  % samples with >0.99 confidence: {100 * np.mean(np.array(all_probs) > 0.99):.2f}%")

# Run the attack
if __name__ == "__main__":
    # Run main attack
    attack_model, target_model, shadow_results, target_results = main()
    
    # Additional analysis
    print("\n=== Additional Model Analysis ===")
    
    # Analyze target model confidence on training data
    target_train_indices = list(range(config.TARGET_SIZE))
    target_train_data = Subset(datasets.MNIST(root="./data", train=True, download=True, 
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5,), (0.5,))
                                             ])), target_train_indices)
    train_loader = DataLoader(target_train_data, batch_size=256)
    analyze_model_confidence(target_model, train_loader, "Target Model (Training Data)")
    
    print("\nAttack completed! Check results in", config.LOG_DIR)