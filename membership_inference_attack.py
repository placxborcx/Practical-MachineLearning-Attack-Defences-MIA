"""
Purpose: Optimized PyTorch implementation of a black-box Membership Inference Attack.
Enhanced with improved transferability, timing metrics, and comprehensive analysis.
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
    EPOCHS_SHADOW = 15
    EPOCHS_ATTACK = 25  # Increased for better convergence
    N_SHADOW = 4  # Reduced back to 4 to ensure enough data
    NUM_CLASSES = 10
    SEED = 1337
    
    # Data configuration
    MNIST_AUX_RATIO = 0.5  # Reduced to leave more data for target
    SHADOW_SIZE = 5000     # Reduced to match target better
    SHADOW_TEST = 1000     
    
    # Target model configuration (matching shadow models)
    TARGET_SIZE = 5000     # Match shadow model size for better transferability
    TARGET_EPOCHS = 15     # Match shadow model epochs
    
    # Non-member dataset sizes
    FASHION_SIZE = 3000
    KMNIST_SIZE = 2000
    CIFAR_SIZE = 2000
    NOISE_SIZE = 1000
    
    # Learning rates
    SHADOW_LR = 1e-3
    ATTACK_LR = 5e-3
    TARGET_LR = 1e-3       # Match shadow model LR
    
    # Validation
    VAL_SPLIT = 0.2
    
    # Attack improvements
    USE_AUGMENTATION = True
    TEMPERATURE_SCALING = True
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

# Create logging directory
os.makedirs(config.LOG_DIR, exist_ok=True)

print(f"Using device: {config.DEVICE}")
print(f"CUDA available: {torch.cuda.is_available()}")

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
# 2. Enhanced CNN Architecture
# -----------------------------
class EnhancedCNN(nn.Module):
    """Enhanced CNN with dropout and batch normalization for better generalization"""
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate/2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        features = self.conv_layers(x)
        return self.classifier(features)

# -----------------------------
# 3. Enhanced Feature Engineering with Temperature Calibration
# -----------------------------
class TemperatureScaling(nn.Module):
    """Temperature scaling for calibration"""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, logits):
        return logits / self.temperature

def extract_comprehensive_features(logits, true_labels=None, temperature=1.0):
    """Extract comprehensive features for membership inference with improvements"""
    # Apply temperature scaling for better calibration
    if config.TEMPERATURE_SCALING:
        temperature = 1.5  # Empirically better for transferability
    
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=1)
    
    # Basic probability features
    max_prob, predicted_class = probs.max(dim=1)
    sorted_probs, _ = probs.sort(dim=1, descending=True)
    
    # Uncertainty features
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1)
    
    # Confidence features
    confidence_gap = sorted_probs[:, 0] - sorted_probs[:, 1]
    top3_sum = sorted_probs[:, :3].sum(dim=1)
    top5_sum = sorted_probs[:, :5].sum(dim=1)
    
    # Statistical features
    prob_variance = probs.var(dim=1)
    prob_std = probs.std(dim=1)
    
    # Modified Entropy (Renyi entropy)
    alpha = 2.0
    renyi_entropy = -torch.log((probs ** alpha).sum(dim=1) + 1e-12) / (alpha - 1)
    
    # Loss-based features
    ce_loss = F.cross_entropy(scaled_logits, predicted_class, reduction='none')
    
    # KL divergence from uniform distribution
    uniform_dist = torch.ones_like(probs) / probs.size(1)
    kl_uniform = F.kl_div(torch.log(probs + 1e-12), uniform_dist, reduction='none').sum(dim=1)
    
    # Margin-based features
    margin = sorted_probs[:, 0] - sorted_probs[:, -1]
    
    # Combine all features
    features = torch.stack([
        max_prob,           # 0: Maximum probability
        entropy,            # 1: Shannon entropy
        ce_loss,            # 2: Cross-entropy loss
        confidence_gap,     # 3: Confidence gap
        top3_sum,           # 4: Sum of top-3 probabilities
        top5_sum,           # 5: Sum of top-5 probabilities
        prob_variance,      # 6: Probability variance
        prob_std,           # 7: Probability standard deviation
        kl_uniform,         # 8: KL divergence from uniform
        renyi_entropy,      # 9: Renyi entropy
        margin,             # 10: Margin
        sorted_probs[:, 0], # 11: Highest probability
        sorted_probs[:, 1], # 12: Second highest probability
    ], dim=1)
    
    return features.detach()

# -----------------------------
# 4. Enhanced Data Loading with Augmentation
# -----------------------------
def get_augmentation_transform():
    """Get augmentation transforms for better generalization"""
    return transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

def load_datasets():
    """Load and prepare all datasets with proper transforms"""
    print("Loading datasets...")
    
    # Standard transform
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Augmented transform for shadow models
    aug_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.RandomRotation(5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load datasets
    try:
        mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        
        # Load with augmentation for shadow training
        mnist_train_aug = datasets.MNIST(root="./data", train=True, download=True, transform=aug_transform)
        
        fashion_mnist = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        kmnist = datasets.KMNIST(root="./data", train=True, download=True, transform=transform)
        cifar10 = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        
        print(f"MNIST train size: {len(mnist_train)}")
        print(f"Fashion-MNIST size: {len(fashion_mnist)}")
        print(f"KMNIST size: {len(kmnist)}")
        print(f"CIFAR-10 size: {len(cifar10)}")
        
    except Exception as e:
        print(f"Error loading datasets: {e}")
        raise
    
    return mnist_train, mnist_test, mnist_train_aug, fashion_mnist, kmnist, cifar10

def create_auxiliary_datasets(mnist_train, mnist_train_aug, fashion_mnist, kmnist, cifar10):
    """Create auxiliary datasets with improved distribution matching"""
    print("Creating auxiliary datasets...")
    
    # Calculate data requirements more carefully
    shadow_data_per_model = config.SHADOW_SIZE + config.SHADOW_TEST
    total_shadow_data_needed = config.N_SHADOW * shadow_data_per_model
    
    # Reserve data for target model
    target_reserve_size = config.TARGET_SIZE + 2000  # Extra for validation
    
    # Ensure we have enough total data
    total_mnist = len(mnist_train)
    print(f"Total MNIST data available: {total_mnist}")
    print(f"Target reserve needed: {target_reserve_size}")
    print(f"Shadow data needed: {total_shadow_data_needed}")
    
    # Check if we have enough data
    min_required = target_reserve_size + total_shadow_data_needed
    if total_mnist < min_required:
        print(f"Warning: Not enough data! Need {min_required}, have {total_mnist}")
        # Adjust sizes proportionally
        scale_factor = total_mnist / min_required * 0.9  # 0.9 for safety margin
        config.SHADOW_SIZE = int(config.SHADOW_SIZE * scale_factor)
        config.SHADOW_TEST = int(config.SHADOW_TEST * scale_factor)
        config.TARGET_SIZE = int(config.TARGET_SIZE * scale_factor)
        
        # Recalculate
        shadow_data_per_model = config.SHADOW_SIZE + config.SHADOW_TEST
        total_shadow_data_needed = config.N_SHADOW * shadow_data_per_model
        target_reserve_size = config.TARGET_SIZE + 2000
        
        print(f"Adjusted sizes - Shadow: {config.SHADOW_SIZE}, Test: {config.SHADOW_TEST}, Target: {config.TARGET_SIZE}")
    
    # Create indices ensuring no overlap
    all_indices = np.random.permutation(len(mnist_train))
    
    # Split indices
    target_indices = all_indices[:target_reserve_size]
    aux_start = target_reserve_size
    aux_end = aux_start + total_shadow_data_needed
    aux_indices = all_indices[aux_start:aux_end]
    
    print(f"Auxiliary data indices: {len(aux_indices)}")
    print(f"Target data indices: {len(target_indices)}")
    
    # Create auxiliary subsets
    mnist_aux = Subset(mnist_train_aug if config.USE_AUGMENTATION else mnist_train, aux_indices)
    
    # Create non-member datasets with better diversity
    fashion_indices = np.random.choice(len(fashion_mnist), 
                                     min(config.FASHION_SIZE, len(fashion_mnist)), 
                                     replace=False)
    kmnist_indices = np.random.choice(len(kmnist), 
                                    min(config.KMNIST_SIZE, len(kmnist)), 
                                    replace=False)
    cifar_indices = np.random.choice(len(cifar10), 
                                   min(config.CIFAR_SIZE, len(cifar10)), 
                                   replace=False)
    
    # Create synthetic noise dataset with more realistic patterns
    def create_noise_dataset(n_samples):
        data = []
        labels = []
        for _ in range(n_samples):
            # Mix of different noise patterns
            noise_type = np.random.choice(['gaussian', 'uniform', 'salt_pepper'])
            if noise_type == 'gaussian':
                img = torch.randn(1, 28, 28) * 0.3
            elif noise_type == 'uniform':
                img = torch.rand(1, 28, 28) * 2 - 1
            else:  # salt and pepper
                img = torch.randn(1, 28, 28) * 0.1
                mask = torch.rand(1, 28, 28) > 0.9
                img[mask] = torch.randint(-1, 2, (mask.sum(),)).float()
            
            data.append(img)
            labels.append(torch.randint(0, config.NUM_CLASSES, (1,)).item())
        
        data = torch.stack(data)
        labels = torch.tensor(labels)
        return TensorDataset(data, labels)
    
    noise_dataset = create_noise_dataset(config.NOISE_SIZE)
    
    # Combine non-member datasets
    nonmember_datasets = ConcatDataset([
        Subset(fashion_mnist, fashion_indices),
        Subset(kmnist, kmnist_indices),
        Subset(cifar10, cifar_indices),
        noise_dataset
    ])
    
    print(f"Non-member dataset size: {len(nonmember_datasets)}")
    
    return mnist_aux, nonmember_datasets, aux_indices, target_indices

# -----------------------------
# 5. Enhanced Shadow Model Training
# -----------------------------
@time_function
def train_shadow_model(model, train_loader, val_loader, model_id):
    """Train a single shadow model with improved regularization"""
    print(f"Training shadow model {model_id}...")
    
    optimizer = optim.Adam(model.parameters(), lr=config.SHADOW_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS_SHADOW)
    
    best_val_acc = 0
    patience = 7
    patience_counter = 0
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(config.EPOCHS_SHADOW):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            
            # Add L2 regularization
            l2_lambda = 1e-4
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = loss + l2_lambda * l2_norm
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_total += target.size(0)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(config.DEVICE), target.to(config.DEVICE)
                output = model(data)
                val_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                val_total += target.size(0)
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        train_loss /= len(train_loader)
        val_loss /= val_total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if epoch % 5 == 0:
            print(f'  Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            if config.SAVE_MODELS:
                torch.save(model.state_dict(), f"{config.LOG_DIR}/shadow_model_{model_id}_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break
        
        scheduler.step()
    
    # Load best model
    if config.SAVE_MODELS:
        model.load_state_dict(torch.load(f"{config.LOG_DIR}/shadow_model_{model_id}_best.pth"))
    
    print(f"Shadow model {model_id} training completed. Best val acc: {best_val_acc:.2f}%")
    return (train_losses, val_losses, train_accs, val_accs)

@time_function
def collect_shadow_features(shadow_models, mnist_aux, nonmember_datasets, aux_indices):
    """Collect features with ensemble predictions"""
    print("Collecting features from shadow models...")
    
    all_member_features = []
    all_nonmember_features = []
    
    for i, model in enumerate(shadow_models):
        print(f"Collecting features from shadow model {i}...")
        model.eval()
        
        # Prepare shadow model's train/test split
        start_idx = i * (config.SHADOW_SIZE + config.SHADOW_TEST)
        end_train = start_idx + config.SHADOW_SIZE
        end_test = start_idx + config.SHADOW_SIZE + config.SHADOW_TEST
        
        # Member data (shadow training set)
        shadow_train_indices = aux_indices[start_idx:end_train]
        member_subset = Subset(mnist_aux.dataset, shadow_train_indices)
        member_loader = DataLoader(member_subset, batch_size=config.BATCH_ATTACK, shuffle=False)
        
        # Non-member data (shadow test set)
        shadow_test_indices = aux_indices[end_train:end_test]
        nonmember_subset = Subset(mnist_aux.dataset, shadow_test_indices)
        nonmember_loader = DataLoader(nonmember_subset, batch_size=config.BATCH_ATTACK, shuffle=False)
        
        # Collect member features
        with torch.no_grad():
            for data, _ in member_loader:
                data = data.to(config.DEVICE)
                logits = model(data)
                features = extract_comprehensive_features(logits)
                all_member_features.append(features.cpu())
        
        # Collect non-member features
        with torch.no_grad():
            for data, _ in nonmember_loader:
                data = data.to(config.DEVICE)
                logits = model(data)
                features = extract_comprehensive_features(logits)
                all_nonmember_features.append(features.cpu())
    
    # Collect features from external non-member datasets
    if config.ENSEMBLE_SHADOWS:
        print("Collecting external non-member features with ensemble...")
        external_loader = DataLoader(nonmember_datasets, batch_size=config.BATCH_ATTACK, shuffle=False)
        
        # Use ensemble of shadow models for better features
        for data, _ in external_loader:
            data = data.to(config.DEVICE)
            ensemble_features = []
            
            for model in shadow_models:
                model.eval()
                with torch.no_grad():
                    logits = model(data)
                    features = extract_comprehensive_features(logits)
                    ensemble_features.append(features)
            
            # Average features across models
            avg_features = torch.stack(ensemble_features).mean(dim=0)
            all_nonmember_features.append(avg_features.cpu())
    
    # Combine all features
    member_features = torch.cat(all_member_features, dim=0)
    nonmember_features = torch.cat(all_nonmember_features, dim=0)
    
    print(f"Collected {len(member_features)} member features")
    print(f"Collected {len(nonmember_features)} non-member features")
    
    return member_features, nonmember_features

# -----------------------------
# 6. Enhanced Attack Classifier
# -----------------------------
class ImprovedAttackMLP(nn.Module):
    """Improved MLP with residual connections for better gradient flow"""
    def __init__(self, input_dim=13, hidden_dims=[128, 64, 32], dropout_rate=0.3):
        super().__init__()
        
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # First layer
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            layer = nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            )
            self.hidden_layers.append(layer)
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dims[-1], 2)
        
        # Skip connection
        self.skip = nn.Linear(input_dim, hidden_dims[-1])
    
    def forward(self, x):
        # Input normalization
        x = self.input_bn(x)
        identity = self.skip(x)
        
        # First layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
        
        # Add skip connection
        x = x + identity
        x = F.relu(x)
        
        # Output
        return self.fc_out(x)

@time_function
def train_attack_classifier(X_train, y_train, X_val, y_val):
    """Train attack classifier with improved optimization"""
    print("Training attack classifier...")
    
    # Create data loaders with data augmentation
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_ATTACK, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_ATTACK, shuffle=False)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = ImprovedAttackMLP(input_dim=input_dim).to(config.DEVICE)
    
    # Optimizer with learning rate scheduling
    optimizer = optim.AdamW(model.parameters(), lr=config.ATTACK_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config.ATTACK_LR,
        epochs=config.EPOCHS_ATTACK,
        steps_per_epoch=len(train_loader)
    )
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_val_auc = 0
    patience = 10
    patience_counter = 0
    
    train_losses, val_losses = [], []
    train_aucs, val_aucs = [], []
    
    for epoch in range(config.EPOCHS_ATTACK):
        # Training phase
        model.train()
        train_loss = 0
        train_probs, train_labels = [], []
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(config.DEVICE), batch_y.to(config.DEVICE)
            
            # Add noise to inputs for regularization
            if epoch < config.EPOCHS_ATTACK // 2:
                noise = torch.randn_like(batch_x) * 0.01
                batch_x = batch_x + noise
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            probs = F.softmax(outputs, dim=1)[:, 1]
            train_probs.extend(probs.detach().cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_probs, val_labels = [], []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(config.DEVICE), batch_y.to(config.DEVICE)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                probs = F.softmax(outputs, dim=1)[:, 1]
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        train_auc = roc_auc_score(train_labels, train_probs)
        val_auc = roc_auc_score(val_labels, val_probs)
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)
        
        if epoch % 5 == 0:
            print(f'  Epoch {epoch}: Train Loss: {train_losses[-1]:.4f}, Train AUC: {train_auc:.4f}, '
                  f'Val Loss: {val_losses[-1]:.4f}, Val AUC: {val_auc:.4f}')
        
        # Early stopping and best model saving
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            if config.SAVE_MODELS:
                torch.save(model.state_dict(), f"{config.LOG_DIR}/attack_classifier_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break
    
    print(f"Attack classifier training completed. Best val AUC: {best_val_auc:.4f}")
    return (model, train_losses, val_losses, train_aucs, val_aucs)

# -----------------------------
# 7. Evaluation Functions
# -----------------------------
def comprehensive_evaluation(model, X_test, y_test, scaler, title="Test"):
    """Comprehensive evaluation with multiple metrics"""
    model.eval()
    
    X_test_scaled = torch.tensor(scaler.transform(X_test.numpy())).float()
    dataset = TensorDataset(X_test_scaled, y_test)
    loader = DataLoader(dataset, batch_size=config.BATCH_ATTACK, shuffle=False)
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(config.DEVICE)
            outputs = model(batch_x)
            probs = F.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    auc = roc_auc_score(all_labels, all_probs)
    
    # Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(all_labels, all_probs)
    
    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(all_labels, all_probs)
    
    # Find optimal threshold
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = roc_thresholds[optimal_idx]
    
    # Calculate metrics at optimal threshold
    predictions = (all_probs >= optimal_threshold).astype(int)
    accuracy = accuracy_score(all_labels, predictions)
    
    # TPR at FPR = 0.1
    target_fpr = 0.1
    fpr_idx = np.where(fpr <= target_fpr)[0]
    if len(fpr_idx) > 0:
        tpr_at_01_fpr = tpr[fpr_idx[-1]]
    else:
        tpr_at_01_fpr = 0
    
    results = {
        'title': title,
        'auc': auc,
        'accuracy': accuracy,
        'optimal_threshold': optimal_threshold,
        'tpr_at_01_fpr': tpr_at_01_fpr,
        'all_probs': all_probs,
        'all_labels': all_labels,
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall
    }
    
    print(f"[{title}] AUC: {auc:.4f}, Accuracy: {accuracy:.4f}, "
          f"TPR@FPR=0.1: {tpr_at_01_fpr:.4f}")
    
    return results

@time_function
def evaluate_on_target_model(attack_model, target_model, target_train_data, target_test_data, scaler):
    """Evaluate the attack on the actual target model"""
    print("\n=== Evaluating on Target Model ===")
    target_model.eval()
    
    # Prepare data loaders
    target_train_loader = DataLoader(target_train_data, batch_size=config.BATCH_ATTACK, shuffle=False)
    target_test_loader = DataLoader(target_test_data, batch_size=config.BATCH_ATTACK, shuffle=False)
    
    # Extract features from target model
    def extract_features_from_loader(model, loader):
        features_list = []
        
        with torch.no_grad():
            for data, _ in loader:
                data = data.to(config.DEVICE)
                logits = model(data)
                features = extract_comprehensive_features(logits)
                features_list.append(features.cpu())
                
        return torch.cat(features_list, dim=0)
    
    print("Extracting features from target model...")
    member_features = extract_features_from_loader(target_model, target_train_loader)
    nonmember_features = extract_features_from_loader(target_model, target_test_loader)
    
    # Create labels
    member_labels = torch.ones(len(member_features), dtype=torch.long)
    nonmember_labels = torch.zeros(len(nonmember_features), dtype=torch.long)
    
    # Combine features and labels
    all_features = torch.cat([member_features, nonmember_features], dim=0)
    all_labels = torch.cat([member_labels, nonmember_labels], dim=0)
    
    # Evaluate
    results = comprehensive_evaluation(attack_model, all_features, all_labels, scaler, "Target Model")
    return results

def plot_results(results_list, save_path=None):
    """Plot comprehensive results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC curves
    for results in results_list:
        axes[0, 0].plot(results['fpr'], results['tpr'], 
                       label=f"{results['title']} (AUC={results['auc']:.3f})")
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Precision-Recall curves
    for results in results_list:
        axes[0, 1].plot(results['recall'], results['precision'], 
                       label=f"{results['title']}")
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Score distributions
    for i, results in enumerate(results_list[:2]):
        member_scores = results['all_probs'][results['all_labels'] == 1]
        nonmember_scores = results['all_probs'][results['all_labels'] == 0]
        
        axes[1, i].hist(nonmember_scores, bins=30, alpha=0.7, label='Non-members', density=True)
        axes[1, i].hist(member_scores, bins=30, alpha=0.7, label='Members', density=True)
        axes[1, i].axvline(results['optimal_threshold'], color='red', linestyle='--', 
                          label=f'Threshold={results["optimal_threshold"]:.3f}')
        axes[1, i].set_xlabel('Attack Score')
        axes[1, i].set_ylabel('Density')
        axes[1, i].set_title(f'{results["title"]} - Score Distribution')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# -----------------------------
# 8. Main Execution with Timing
# -----------------------------
def main():
    """Main execution function with comprehensive timing"""
    print("=== Enhanced Membership Inference Attack Implementation ===")
    print(f"Configuration: {config.N_SHADOW} shadow models, "
          f"{config.SHADOW_SIZE} train + {config.SHADOW_TEST} test per shadow")
    
    total_start_time = time.time()
    timing_results = {}
    
    # Load datasets
    load_start = time.time()
    mnist_train, mnist_test, mnist_train_aug, fashion_mnist, kmnist, cifar10 = load_datasets()
    mnist_aux, nonmember_datasets, aux_indices, target_indices = create_auxiliary_datasets(
        mnist_train, mnist_train_aug, fashion_mnist, kmnist, cifar10)
    timing_results['data_loading'] = time.time() - load_start
    
    # Train shadow models
    print("\n=== Training Shadow Models ===")
    shadow_models = []
    shadow_training_times = []
    
    # Verify we have enough data
    total_aux_needed = config.N_SHADOW * (config.SHADOW_SIZE + config.SHADOW_TEST)
    if len(aux_indices) < total_aux_needed:
        print(f"Warning: Not enough auxiliary data. Have {len(aux_indices)}, need {total_aux_needed}")
        print("Adjusting shadow model configuration...")
        available_per_shadow = len(aux_indices) // config.N_SHADOW
        config.SHADOW_SIZE = int(available_per_shadow * 0.8)
        config.SHADOW_TEST = int(available_per_shadow * 0.2)
        print(f"New shadow size: {config.SHADOW_SIZE}, test size: {config.SHADOW_TEST}")
    
    for i in range(config.N_SHADOW):
        # Prepare data for this shadow model
        start_idx = i * (config.SHADOW_SIZE + config.SHADOW_TEST)
        end_train = start_idx + config.SHADOW_SIZE
        end_test = start_idx + config.SHADOW_SIZE + config.SHADOW_TEST
        
        # Ensure we don't exceed available indices
        if end_test > len(aux_indices):
            print(f"Skipping shadow model {i} - insufficient data")
            break
        
        # Create train/val split for shadow model
        shadow_train_indices = aux_indices[start_idx:end_train]
        shadow_train_subset = Subset(mnist_aux.dataset, shadow_train_indices)
        
        # Check if we have data
        if len(shadow_train_subset) == 0:
            print(f"No data available for shadow model {i}")
            break
        
        # Split training data for validation
        train_size = int(0.8 * len(shadow_train_subset))
        val_size = len(shadow_train_subset) - train_size
        
        # Ensure we have at least some data for both splits
        if train_size == 0 or val_size == 0:
            print(f"Insufficient data for train/val split in shadow model {i}")
            train_size = max(1, int(0.8 * len(shadow_train_subset)))
            val_size = max(1, len(shadow_train_subset) - train_size)
        
        train_subset, val_subset = random_split(shadow_train_subset, [train_size, val_size])
        
        train_loader = DataLoader(train_subset, batch_size=config.BATCH_SHADOW, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config.BATCH_SHADOW, shuffle=False)
        
        # Initialize and train shadow model
        shadow_model = EnhancedCNN(num_classes=config.NUM_CLASSES).to(config.DEVICE)
        (train_losses, val_losses, train_accs, val_accs), train_time = train_shadow_model(
            shadow_model, train_loader, val_loader, i)
        
        shadow_models.append(shadow_model)
        shadow_training_times.append(train_time)
        
    # Update number of successfully trained shadow models
    config.N_SHADOW = len(shadow_models)
    print(f"\nSuccessfully trained {config.N_SHADOW} shadow models")
    
    if config.N_SHADOW < 2:
        print("Error: Need at least 2 shadow models for attack")
        return None, None, None, None, None
    
    timing_results['shadow_training'] = sum(shadow_training_times)
    timing_results['avg_shadow_training'] = np.mean(shadow_training_times)
    
    # Collect features from shadow models
    print("\n=== Collecting Features from Shadow Models ===")
    (member_features, nonmember_features), feature_time = collect_shadow_features(
        shadow_models, mnist_aux, nonmember_datasets, aux_indices)
    timing_results['feature_extraction'] = feature_time
    
    # Prepare attack training data
    print("\n=== Preparing Attack Classifier Data ===")
    prep_start = time.time()
    
    # Create labels
    member_labels = torch.ones(len(member_features), dtype=torch.long)
    nonmember_labels = torch.zeros(len(nonmember_features), dtype=torch.long)
    
    # Balance the dataset
    min_samples = min(len(member_features), len(nonmember_features))
    print(f"Balancing dataset to {min_samples} samples per class")
    
    member_features = member_features[:min_samples]
    member_labels = member_labels[:min_samples]
    nonmember_features = nonmember_features[:min_samples]
    nonmember_labels = nonmember_labels[:min_samples]
    
    # Combine features and labels
    all_features = torch.cat([member_features, nonmember_features], dim=0)
    all_labels = torch.cat([member_labels, nonmember_labels], dim=0)
    
    # Shuffle the data
    perm = torch.randperm(len(all_features))
    all_features = all_features[perm]
    all_labels = all_labels[perm]
    
    # Feature scaling
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features.numpy())
    all_features_scaled = torch.tensor(all_features_scaled, dtype=torch.float32)
    
    # Split into train/val/test
    n_samples = len(all_features_scaled)
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)
    n_test = n_samples - n_train - n_val
    
    X_train = all_features_scaled[:n_train]
    y_train = all_labels[:n_train]
    X_val = all_features_scaled[n_train:n_train+n_val]
    y_val = all_labels[n_train:n_train+n_val]
    X_test = all_features_scaled[n_train+n_val:]
    y_test = all_labels[n_train+n_val:]
    
    print(f"Attack dataset sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    timing_results['data_preparation'] = time.time() - prep_start
    
    # Train attack classifier
    print("\n=== Training Attack Classifier ===")
    (attack_model, train_losses, val_losses, train_aucs, val_aucs), attack_time = train_attack_classifier(
        X_train, y_train, X_val, y_val)
    timing_results['attack_training'] = attack_time
    
    # Evaluate attack classifier
    print("\n=== Evaluating Attack Classifier ===")
    eval_start = time.time()
    
    # Load best model
    if config.SAVE_MODELS:
        attack_model.load_state_dict(torch.load(f"{config.LOG_DIR}/attack_classifier_best.pth"))
    
    # Evaluate on test set
    test_results = comprehensive_evaluation(attack_model, X_test, y_test, scaler, "Attack Test Set")
    
    # Train target model with similar settings to shadow models
    print("\n=== Training Target Model ===")
    target_model = EnhancedCNN(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    
    # Use reserved MNIST data for target model
    target_train_subset = Subset(mnist_train, target_indices[:config.TARGET_SIZE])
    
    # Split for training and validation
    train_size = int(0.8 * len(target_train_subset))
    val_size = len(target_train_subset) - train_size
    train_subset, val_subset = random_split(target_train_subset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=config.BATCH_SHADOW, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config.BATCH_SHADOW, shuffle=False)
    
    # Train target model with same settings as shadow models
    (_, _, _, _), target_train_time = train_shadow_model(
        target_model, train_loader, val_loader, "target")
    timing_results['target_training'] = target_train_time
    
    # Evaluate attack on target model
    target_results, target_eval_time = evaluate_on_target_model(
        attack_model, target_model, target_train_subset, mnist_test, scaler)
    
    timing_results['evaluation'] = time.time() - eval_start
    timing_results['total_time'] = time.time() - total_start_time
    
    # Plot all results
    print("\n=== Plotting Results ===")
    plot_results([test_results, target_results], save_path=f"{config.LOG_DIR}/mia_results.png")
    
    # Save summary results with timing
    summary = {
        "config": {
            "n_shadow": config.N_SHADOW,
            "shadow_size": config.SHADOW_SIZE,
            "shadow_test": config.SHADOW_TEST,
            "target_size": config.TARGET_SIZE,
            "epochs_shadow": config.EPOCHS_SHADOW,
            "epochs_attack": config.EPOCHS_ATTACK,
            "use_augmentation": config.USE_AUGMENTATION,
            "temperature_scaling": config.TEMPERATURE_SCALING,
            "ensemble_shadows": config.ENSEMBLE_SHADOWS,
        },
        "results": {
            "attack_test": {
                "auc": float(test_results['auc']),
                "accuracy": float(test_results['accuracy']),
                "tpr_at_01_fpr": float(test_results['tpr_at_01_fpr']),
            },
            "target_model": {
                "auc": float(target_results['auc']),
                "accuracy": float(target_results['accuracy']),
                "tpr_at_01_fpr": float(target_results['tpr_at_01_fpr']),
            }
        },
        "timing": {
            "data_loading": timing_results['data_loading'],
            "shadow_training_total": timing_results['shadow_training'],
            "shadow_training_avg": timing_results['avg_shadow_training'],
            "feature_extraction": timing_results['feature_extraction'],
            "data_preparation": timing_results['data_preparation'],
            "attack_training": timing_results['attack_training'],
            "target_training": timing_results['target_training'],
            "evaluation": timing_results['evaluation'],
            "total_time": timing_results['total_time'],
        },
        "efficiency_metrics": {
            "total_time_minutes": timing_results['total_time'] / 60,
            "attack_prep_time": (timing_results['shadow_training'] + 
                               timing_results['feature_extraction'] + 
                               timing_results['attack_training']) / 60,
            "inference_time_per_sample": timing_results['evaluation'] / (len(target_train_subset) + len(mnist_test)),
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(f"{config.LOG_DIR}/summary_results.json", "w") as f:
        json.dump(summary, f, indent=4)
    
    print("\n=== Attack Summary ===")
    print(f"Attack Test Set - AUC: {test_results['auc']:.4f}, "
          f"Accuracy: {test_results['accuracy']:.4f}")
    print(f"Target Model - AUC: {target_results['auc']:.4f}, "
          f"Accuracy: {target_results['accuracy']:.4f}")
    
    print("\n=== Timing Summary ===")
    print(f"Total execution time: {timing_results['total_time']:.2f} seconds "
          f"({timing_results['total_time']/60:.2f} minutes)")
    print(f"Shadow model training: {timing_results['shadow_training']:.2f} seconds")
    print(f"Feature extraction: {timing_results['feature_extraction']:.2f} seconds")
    print(f"Attack training: {timing_results['attack_training']:.2f} seconds")
    print(f"Target evaluation: {timing_results['evaluation']:.2f} seconds")
    print(f"Inference time per sample: {summary['efficiency_metrics']['inference_time_per_sample']*1000:.2f} ms")
    
    print(f"\nResults saved to {config.LOG_DIR}/")
    
    return attack_model, target_model, test_results, target_results, timing_results

# -----------------------------
# 9. Additional Analysis Functions
# -----------------------------
def analyze_hyperparameter_impact(base_config):
    """Analyze impact of different hyperparameters on attack success"""
    print("\n=== Analyzing Hyperparameter Impact ===")
    
    results = {}
    
    # Test different numbers of shadow models
    shadow_counts = [2, 4, 6, 8]
    for n_shadow in shadow_counts:
        print(f"\nTesting with {n_shadow} shadow models...")
        config.N_SHADOW = n_shadow
        # Run simplified version of main
        # ... (simplified execution)
        # results[f'shadow_{n_shadow}'] = auc_score
    
    return results

def analyze_data_requirements():
    """Analyze how data requirements affect attack performance"""
    print("\n=== Analyzing Data Requirements ===")
    
    data_sizes = [1000, 2500, 5000, 10000]
    results = {}
    
    for size in data_sizes:
        print(f"\nTesting with {size} samples per shadow model...")
        config.SHADOW_SIZE = size
        # Run simplified version
        # results[f'size_{size}'] = auc_score
    
    return results

# Run the main function
if __name__ == "__main__":
    attack_model, target_model, test_results, target_results, timing_results = main()
    
    print("\nMembership Inference Attack completed successfully!")
    print("Check the results in ./mia_results/")