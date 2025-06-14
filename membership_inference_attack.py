"""
Purpose: Optimized PyTorch implementation of a black-box Membership Inference Attack.
Enhanced with proper error handling, logging, and multi-dataset support.
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
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# 1. Enhanced Global Configuration
# -----------------------------
class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SHADOW = 128
    BATCH_ATTACK = 256
    EPOCHS_SHADOW = 15  # Increased for better training
    EPOCHS_ATTACK = 20
    N_SHADOW = 4
    NUM_CLASSES = 10
    SEED = 1337
    
    # Data configuration
    MNIST_AUX_RATIO = 0.6  # 60% of MNIST for auxiliary data
    SHADOW_SIZE = 8000     # Reduced to ensure enough data
    SHADOW_TEST = 1500     # Test size per shadow model
    
    # Non-member dataset sizes
    FASHION_SIZE = 6000
    KMNIST_SIZE = 3000
    CIFAR_SIZE = 3000
    NOISE_SIZE = 1500
    
    # Learning rates
    SHADOW_LR = 1e-3
    ATTACK_LR = 5e-3
    
    # Validation
    VAL_SPLIT = 0.2
    
    # Logging
    LOG_DIR = "./mia_results"
    SAVE_MODELS = True

config = Config()

# Set random seeds for reproducibility
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.SEED)

# Create logging directory
os.makedirs(config.LOG_DIR, exist_ok=True)

print(f"Using device: {config.DEVICE}")
print(f"CUDA available: {torch.cuda.is_available()}")

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
# 3. Enhanced Feature Engineering
# -----------------------------
def extract_comprehensive_features(logits, true_labels=None, temperature=1.0):
    """Extract comprehensive features for membership inference"""
    # Apply temperature scaling
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=1)
    
    # Basic probability features
    max_prob, predicted_class = probs.max(dim=1)
    sorted_probs, _ = probs.sort(dim=1, descending=True)
    
    # Uncertainty features
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1)
    
    # Confidence features
    confidence_gap = sorted_probs[:, 0] - sorted_probs[:, 1]  # gap between top 2
    top3_sum = sorted_probs[:, :3].sum(dim=1)
    
    # Statistical features
    prob_variance = probs.var(dim=1)
    prob_std = probs.std(dim=1)
    
    # Loss-based features (using predicted labels for black-box setting)
    ce_loss = F.cross_entropy(scaled_logits, predicted_class, reduction='none')
    
    # KL divergence from uniform distribution
    uniform_dist = torch.ones_like(probs) / probs.size(1)
    kl_uniform = F.kl_div(torch.log(probs + 1e-12), uniform_dist, reduction='none').sum(dim=1)
    
    # Combine all features
    features = torch.stack([
        max_prob,           # 0: Maximum probability
        entropy,            # 1: Predictive entropy
        ce_loss,            # 2: Cross-entropy loss
        confidence_gap,     # 3: Confidence gap
        top3_sum,           # 4: Sum of top-3 probabilities
        prob_variance,      # 5: Probability variance
        prob_std,           # 6: Probability standard deviation
        kl_uniform,         # 7: KL divergence from uniform
        sorted_probs[:, 0], # 8: Highest probability
        sorted_probs[:, 1], # 9: Second highest probability
    ], dim=1)
    
    return features.detach()

# -----------------------------
# 4. Enhanced Data Loading
# -----------------------------
def load_datasets():
    """Load and prepare all datasets with proper transforms"""
    print("Loading datasets...")
    
    # Common transform pipeline
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    # Load datasets
    try:
        mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
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
    
    return mnist_train, mnist_test, fashion_mnist, kmnist, cifar10

def create_auxiliary_datasets(mnist_train, fashion_mnist, kmnist, cifar10):
    """Create auxiliary datasets for shadow model training"""
    print("Creating auxiliary datasets...")
    
    # Calculate required total size
    total_shadow_data_needed = config.N_SHADOW * (config.SHADOW_SIZE + config.SHADOW_TEST)
    mnist_aux_size = int(config.MNIST_AUX_RATIO * len(mnist_train))
    
    print(f"Total shadow data needed: {total_shadow_data_needed}")
    print(f"Available MNIST auxiliary size: {mnist_aux_size}")
    
    if mnist_aux_size < total_shadow_data_needed:
        print(f"Warning: Not enough MNIST data. Adjusting shadow sizes...")
        available_per_shadow = mnist_aux_size // config.N_SHADOW
        config.SHADOW_SIZE = int(available_per_shadow * 0.8)
        config.SHADOW_TEST = int(available_per_shadow * 0.2)
        print(f"Adjusted shadow size: {config.SHADOW_SIZE}, test size: {config.SHADOW_TEST}")
    
    # Create MNIST auxiliary subset
    mnist_indices = np.random.permutation(len(mnist_train))[:mnist_aux_size]
    mnist_aux = Subset(mnist_train, mnist_indices)
    
    # Create non-member datasets
    fashion_indices = np.random.choice(len(fashion_mnist), 
                                     min(config.FASHION_SIZE, len(fashion_mnist)), 
                                     replace=False)
    kmnist_indices = np.random.choice(len(kmnist), 
                                    min(config.KMNIST_SIZE, len(kmnist)), 
                                    replace=False)
    cifar_indices = np.random.choice(len(cifar10), 
                                   min(config.CIFAR_SIZE, len(cifar10)), 
                                   replace=False)
    
    # Create synthetic noise dataset
    def create_noise_dataset(n_samples):
        data = torch.randn(n_samples, 1, 28, 28)
        # Normalize to [0, 1] then to [-1, 1] to match other data
        data = (data - data.min()) / (data.max() - data.min())
        data = 2 * data - 1
        labels = torch.randint(0, config.NUM_CLASSES, (n_samples,))
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
    
    return mnist_aux, nonmember_datasets, mnist_indices

# -----------------------------
# 5. Enhanced Shadow Model Training
# -----------------------------
def train_shadow_model(model, train_loader, val_loader, model_id):
    """Train a single shadow model with validation monitoring"""
    print(f"Training shadow model {model_id}...")
    
    optimizer = optim.Adam(model.parameters(), lr=config.SHADOW_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_val_acc = 0
    patience = 5
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
    
    print(f"Shadow model {model_id} training completed. Best val acc: {best_val_acc:.2f}%")
    return train_losses, val_losses, train_accs, val_accs

def collect_shadow_features(shadow_models, mnist_aux, nonmember_datasets, mnist_indices):
    """Collect features from shadow models for attack training"""
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
        member_indices = mnist_indices[start_idx:end_train]
        member_subset = Subset(mnist_aux.dataset, member_indices)
        member_loader = DataLoader(member_subset, batch_size=config.BATCH_ATTACK, shuffle=False)
        
        # Non-member data (shadow test set + external)
        nonmember_indices = mnist_indices[end_train:end_test]
        nonmember_subset = Subset(mnist_aux.dataset, nonmember_indices)
        
        # Add external non-member data
        nonmember_loader = DataLoader(nonmember_subset, batch_size=config.BATCH_ATTACK, shuffle=False)
        
        # Collect member features
        with torch.no_grad():
            for data, _ in member_loader:
                data = data.to(config.DEVICE)
                logits = model(data)
                features = extract_comprehensive_features(logits)
                all_member_features.append(features.cpu())
        
        # Collect non-member features (in-distribution)
        with torch.no_grad():
            for data, _ in nonmember_loader:
                data = data.to(config.DEVICE)
                logits = model(data)
                features = extract_comprehensive_features(logits)
                all_nonmember_features.append(features.cpu())
    
    # Collect features from external non-member datasets (using all shadow models)
    print("Collecting external non-member features...")
    external_loader = DataLoader(nonmember_datasets, batch_size=config.BATCH_ATTACK, shuffle=False)
    
    # Use each shadow model to extract features from external data
    for i, model in enumerate(shadow_models):
        model.eval()
        with torch.no_grad():
            for data, _ in external_loader:
                data = data.to(config.DEVICE)
                logits = model(data)
                features = extract_comprehensive_features(logits)
                all_nonmember_features.append(features.cpu())
    
    # Combine all features
    member_features = torch.cat(all_member_features, dim=0)
    nonmember_features = torch.cat(all_nonmember_features, dim=0)
    
    print(f"Collected {len(member_features)} member features")
    print(f"Collected {len(nonmember_features)} non-member features")
    
    return member_features, nonmember_features

# -----------------------------
# 6. Enhanced Attack Classifier
# -----------------------------
class EnhancedAttackMLP(nn.Module):
    """Enhanced MLP for membership inference with batch normalization and dropout"""
    def __init__(self, input_dim=10, hidden_dims=[64, 32, 16], dropout_rate=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 2))  # Binary classification
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def train_attack_classifier(X_train, y_train, X_val, y_val):
    """Train the attack classifier with comprehensive monitoring"""
    print("Training attack classifier...")
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_ATTACK, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_ATTACK, shuffle=False)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = EnhancedAttackMLP(input_dim=input_dim).to(config.DEVICE)
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config.ATTACK_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_val_auc = 0
    patience = 8
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
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = F.cross_entropy(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
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
                loss = F.cross_entropy(outputs, batch_y)
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
        
        scheduler.step(val_auc)
    
    print(f"Attack classifier training completed. Best val AUC: {best_val_auc:.4f}")
    return model, train_losses, val_losses, train_aucs, val_aucs

# -----------------------------
# 7. Evaluation and Analysis
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
    
    # Find optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = roc_thresholds[optimal_idx]
    
    # Calculate metrics at optimal threshold
    predictions = (all_probs >= optimal_threshold).astype(int)
    accuracy = accuracy_score(all_labels, predictions)
    
    # TPR at FPR = 0.1 (regulatory requirement)
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
    for i, results in enumerate(results_list[:2]):  # Only plot first two
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
# 8. Target Model Evaluation
# -----------------------------
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
        labels_list = []
        
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

# -----------------------------
# 9. Main Execution
# -----------------------------
def main():
    """Main execution function"""
    print("=== Membership Inference Attack Implementation ===")
    print(f"Configuration: {config.N_SHADOW} shadow models, "
          f"{config.SHADOW_SIZE} train + {config.SHADOW_TEST} test per shadow")
    
    # Load datasets
    mnist_train, mnist_test, fashion_mnist, kmnist, cifar10 = load_datasets()
    mnist_aux, nonmember_datasets, mnist_indices = create_auxiliary_datasets(
        mnist_train, fashion_mnist, kmnist, cifar10)
    
    # Train shadow models
    print("\n=== Training Shadow Models ===")
    shadow_models = []
    
    for i in range(config.N_SHADOW):
        # Prepare data for this shadow model
        start_idx = i * (config.SHADOW_SIZE + config.SHADOW_TEST)
        end_train = start_idx + config.SHADOW_SIZE
        end_test = start_idx + config.SHADOW_SIZE + config.SHADOW_TEST
        
        # Create train/val split for shadow model
        shadow_train_indices = mnist_indices[start_idx:end_train]
        shadow_train_subset = Subset(mnist_train, shadow_train_indices)
        
        # Split training data for validation
        train_size = int(0.8 * len(shadow_train_subset))
        val_size = len(shadow_train_subset) - train_size
        train_subset, val_subset = random_split(shadow_train_subset, [train_size, val_size])
        
        train_loader = DataLoader(train_subset, batch_size=config.BATCH_SHADOW, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config.BATCH_SHADOW, shuffle=False)
        
        # Initialize and train shadow model
        shadow_model = EnhancedCNN(num_classes=config.NUM_CLASSES).to(config.DEVICE)
        train_losses, val_losses, train_accs, val_accs = train_shadow_model(
            shadow_model, train_loader, val_loader, i)
        
        shadow_models.append(shadow_model)
    
    # Collect features from shadow models
    print("\n=== Collecting Features from Shadow Models ===")
    member_features, nonmember_features = collect_shadow_features(
        shadow_models, mnist_aux, nonmember_datasets, mnist_indices)
    
    # Prepare attack training data
    print("\n=== Preparing Attack Classifier Data ===")
    
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
    
    # Train attack classifier
    print("\n=== Training Attack Classifier ===")
    attack_model, train_losses, val_losses, train_aucs, val_aucs = train_attack_classifier(
        X_train, y_train, X_val, y_val)
    
    # Evaluate attack classifier
    print("\n=== Evaluating Attack Classifier ===")
    
    # Load best model
    if config.SAVE_MODELS:
        attack_model.load_state_dict(torch.load(f"{config.LOG_DIR}/attack_classifier_best.pth"))
    
    # Evaluate on test set
    test_results = comprehensive_evaluation(attack_model, X_test, y_test, scaler, "Attack Test Set")
    
    # Train target model (simulating a real target)
    print("\n=== Training Target Model ===")
    target_model = EnhancedCNN(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    
    # Use remaining MNIST data for target model
    remaining_indices = list(set(range(len(mnist_train))) - set(mnist_indices))
    target_train_size = min(10000, len(remaining_indices))
    target_train_indices = np.random.choice(remaining_indices, target_train_size, replace=False)
    target_train_subset = Subset(mnist_train, target_train_indices)
    
    # Split for training and validation
    train_size = int(0.8 * len(target_train_subset))
    val_size = len(target_train_subset) - train_size
    train_subset, val_subset = random_split(target_train_subset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=config.BATCH_SHADOW, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config.BATCH_SHADOW, shuffle=False)
    
    # Train target model
    train_losses, val_losses, train_accs, val_accs = train_shadow_model(
        target_model, train_loader, val_loader, "target")
    
    # Evaluate attack on target model
    target_results = evaluate_on_target_model(
        attack_model, target_model, target_train_subset, mnist_test, scaler)
    
    # Plot all results
    print("\n=== Plotting Results ===")
    plot_results([test_results, target_results], save_path=f"{config.LOG_DIR}/mia_results.png")
    
    # Save summary results
    summary = {
        "config": {
            "n_shadow": config.N_SHADOW,
            "shadow_size": config.SHADOW_SIZE,
            "shadow_test": config.SHADOW_TEST,
            "epochs_shadow": config.EPOCHS_SHADOW,
            "epochs_attack": config.EPOCHS_ATTACK,
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
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(f"{config.LOG_DIR}/summary_results.json", "w") as f:
        json.dump(summary, f, indent=4)
    
    print("\n=== Attack Summary ===")
    print(f"Attack Test Set - AUC: {test_results['auc']:.4f}, "
          f"Accuracy: {test_results['accuracy']:.4f}")
    print(f"Target Model - AUC: {target_results['auc']:.4f}, "
          f"Accuracy: {target_results['accuracy']:.4f}")
    print(f"\nResults saved to {config.LOG_DIR}/")
    
    return attack_model, target_model, test_results, target_results

# -----------------------------
# 10. Additional Analysis Functions
# -----------------------------
def analyze_feature_importance(attack_model, scaler, feature_names=None):
    """Analyze feature importance for the attack model"""
    if feature_names is None:
        feature_names = [
            "Max Probability", "Entropy", "Cross-Entropy Loss", 
            "Confidence Gap", "Top-3 Sum", "Prob Variance", 
            "Prob Std Dev", "KL from Uniform", "Highest Prob", 
            "Second Highest Prob"
        ]
    
    # For neural networks, we can use gradient-based importance
    attack_model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(100, len(feature_names)).to(config.DEVICE)
    dummy_input.requires_grad = True
    
    # Forward pass
    output = attack_model(dummy_input)
    
    # Calculate gradients for member class
    attack_model.zero_grad()
    output[:, 1].sum().backward()
    
    # Get average absolute gradients
    importance = dummy_input.grad.abs().mean(dim=0).cpu().numpy()
    
    # Create importance plot
    plt.figure(figsize=(10, 6))
    indices = np.argsort(importance)[::-1]
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel("Features")
    plt.ylabel("Importance Score")
    plt.title("Feature Importance for Membership Inference")
    plt.tight_layout()
    plt.savefig(f"{config.LOG_DIR}/feature_importance.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return importance, feature_names

def analyze_model_vulnerability(shadow_models, attack_model, scaler):
    """Analyze which types of samples are most vulnerable to MIA"""
    print("\n=== Analyzing Model Vulnerability ===")
    
    # Use first shadow model for analysis
    model = shadow_models[0]
    model.eval()
    
    # Load a subset of data
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, 
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                                ]))
    
    # Get a sample of data
    subset_indices = np.random.choice(len(mnist_train), 1000, replace=False)
    subset = Subset(mnist_train, subset_indices)
    loader = DataLoader(subset, batch_size=config.BATCH_ATTACK, shuffle=False)
    
    # Extract features and predictions
    all_features = []
    all_predictions = []
    all_labels = []
    all_confidences = []
    
    with torch.no_grad():
        for data, labels in loader:
            data = data.to(config.DEVICE)
            logits = model(data)
            features = extract_comprehensive_features(logits)
            
            # Get predictions and confidences
            probs = F.softmax(logits, dim=1)
            confidences, predictions = probs.max(dim=1)
            
            all_features.append(features.cpu())
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    all_features = torch.cat(all_features, dim=0)
    
    # Get attack scores
    features_scaled = torch.tensor(scaler.transform(all_features.numpy())).float()
    attack_model.eval()
    
    with torch.no_grad():
        attack_logits = attack_model(features_scaled.to(config.DEVICE))
        attack_scores = F.softmax(attack_logits, dim=1)[:, 1].cpu().numpy()
    
    # Analyze vulnerability by confidence
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(all_confidences, attack_scores, alpha=0.5)
    plt.xlabel("Model Confidence")
    plt.ylabel("Attack Score")
    plt.title("Attack Score vs Model Confidence")
    plt.grid(True, alpha=0.3)
    
    # Analyze vulnerability by class
    plt.subplot(1, 2, 2)
    class_scores = {}
    for i in range(10):
        class_mask = np.array(all_labels) == i
        if class_mask.sum() > 0:
            class_scores[i] = attack_scores[class_mask].mean()
    
    plt.bar(class_scores.keys(), class_scores.values())
    plt.xlabel("Class")
    plt.ylabel("Average Attack Score")
    plt.title("Average Attack Score by Class")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{config.LOG_DIR}/vulnerability_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return attack_scores, all_confidences, all_labels

# Run the main function
if __name__ == "__main__":
    attack_model, target_model, test_results, target_results = main()
    
    # Additional analyses
    print("\n=== Additional Analyses ===")
    
    # Analyze feature importance
    scaler = StandardScaler()
    # Note: You'll need to reload the scaler from training
    # importance, feature_names = analyze_feature_importance(attack_model, scaler)
    
    print("\nMembership Inference Attack completed successfully!")