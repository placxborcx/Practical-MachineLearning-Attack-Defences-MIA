"""
Defense Verification Script
Verify that the defense is working correctly and measure utility loss
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# Import your modules
from membership_inference_attack import SimpleCNN
from MIA_defense_mechanism import DefendedCNN,DefenceWrapper, CFG as DEF


def verify_defense_components():
    """Verify each defense component is working"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
    #base = SimpleCNN().to(device)
    base = DefendedCNN().to(device)
    base.load_state_dict(torch.load("defended_target.pth", map_location=device))
    
    # Configure defense
    DEF.temperature = 6.0   # 3.0-4
    DEF.noise_std_min = 0.30    #0.2
    DEF.noise_std_max = 1.2    #0.6
    DEF.clip_max = 0.60   #085-75-
    DEF.round_digit = 1
    
    defended = DefenceWrapper(base).to(device).eval()
    
    # Test with sample data
    test_input = torch.randn(10, 1, 28, 28).to(device)
    
    # Get outputs at different stages
    with torch.no_grad():
        # Base model output
        base_logits = base(test_input)
        base_probs = F.softmax(base_logits, dim=1)
        
        # After temperature scaling
        temp_logits = base_logits / DEF.temperature
        temp_probs = F.softmax(temp_logits, dim=1)
        
        # Full defense
        defended_logits = defended.forward(test_input)
        defended_probs = defended.defended_probs(test_input)
    
    print("=== Defense Component Verification ===")
    print(f"Base model - Max prob: {base_probs.max(dim=1)[0].mean():.4f}")
    print(f"After temperature - Max prob: {temp_probs.max(dim=1)[0].mean():.4f}")
    print(f"After full defense - Max prob: {defended_probs.max(dim=1)[0].mean():.4f}")
    print(f"Max prob after clipping: {defended_probs.max():.4f} (should be ≤ {DEF.clip_max})")
    
    # Check rounding
    unique_values = torch.unique(defended_probs * 10)
    print(f"Unique probability values (x10): {len(unique_values)} values")
    
    return base, defended

def evaluate_model_utility(base_model, defended_model):
    """Evaluate the utility (accuracy) of defended model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Evaluate base model
    base_model.eval()
    base_correct = 0
    base_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = base_model(data)
            _, predicted = torch.max(outputs.data, 1)
            base_total += target.size(0)
            base_correct += (predicted == target).sum().item()
    
    base_accuracy = 100 * base_correct / base_total
    
    # Evaluate defended model
    defended_correct = 0
    defended_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            probs = defended_model.defended_probs(data)
            _, predicted = torch.max(probs, 1)
            defended_total += target.size(0)
            defended_correct += (predicted == target).sum().item()
    
    defended_accuracy = 100 * defended_correct / defended_total
    
    print(f"\n=== Model Utility Evaluation ===")
    print(f"Base model accuracy: {base_accuracy:.2f}%")
    print(f"Defended model accuracy: {defended_accuracy:.2f}%")
    print(f"Accuracy loss: {base_accuracy - defended_accuracy:.2f}%")
    
    return base_accuracy, defended_accuracy

def analyze_confidence_distribution(base_model, defended_model):
    """Analyze how defense affects confidence distribution"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load training data (members)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST("./data", train=True, transform=transform)
    member_indices = list(range(5000))  # Same as in attack
    member_loader = DataLoader(Subset(train_dataset, member_indices), 
                              batch_size=256, shuffle=False)
    
    base_confidences = []
    defended_confidences = []
    
    with torch.no_grad():
        for data, _ in member_loader:
            data = data.to(device)
            
            # Base model
            base_logits = base_model(data)
            base_probs = F.softmax(base_logits, dim=1)
            base_conf = base_probs.max(dim=1)[0]
            base_confidences.extend(base_conf.cpu().numpy())
            
            # Defended model
            defended_probs = defended_model.defended_probs(data)
            defended_conf = defended_probs.max(dim=1)[0]
            defended_confidences.extend(defended_conf.cpu().numpy())
    
    # Plot distributions
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(base_confidences, bins=50, alpha=0.7, label='Base Model', density=True)
    plt.hist(defended_confidences, bins=50, alpha=0.7, label='Defended Model', density=True)
    plt.xlabel('Confidence (Max Probability)')
    plt.ylabel('Density')
    plt.title('Confidence Distribution on Training Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot([base_confidences, defended_confidences], labels=['Base', 'Defended'])
    plt.ylabel('Confidence')
    plt.title('Confidence Comparison')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('defense_confidence_analysis.png', dpi=300)
    plt.show()
    
    print(f"\n=== Confidence Analysis ===")
    print(f"Base model - Mean confidence: {np.mean(base_confidences):.4f}")
    print(f"Defended model - Mean confidence: {np.mean(defended_confidences):.4f}")
    print(f"Base model - % >0.9 confidence: {100 * np.mean(np.array(base_confidences) > 0.9):.2f}%")
    print(f"Defended model - % >0.9 confidence: {100 * np.mean(np.array(defended_confidences) > 0.9):.2f}%")

def test_parameter_sensitivity():
    """Test how sensitive the defense is to parameter changes"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #base = SimpleCNN().to(device)
    base = DefendedCNN().to(device)
    base.load_state_dict(torch.load("defended_target.pth", map_location=device))
    
    # Test different parameter combinations
    param_tests = [
        {"temperature": 2.0, "clip_max": 0.9, "noise_std_min": 0.1, "noise_std_max": 0.3},
        {"temperature": 3.0, "clip_max": 0.85, "noise_std_min": 0.2, "noise_std_max": 0.6},
        {"temperature": 4.0, "clip_max": 0.8, "noise_std_min": 0.3, "noise_std_max": 0.8},
    ]
    
    print("\n=== Parameter Sensitivity Analysis ===")
    
    test_input = torch.randn(100, 1, 28, 28).to(device)
    
    for i, params in enumerate(param_tests):
        # Update defense parameters
        for key, value in params.items():
            setattr(DEF, key, value)
        
        defended = DefenceWrapper(base).to(device).eval()
        
        with torch.no_grad():
            probs = defended.defended_probs(test_input)
            max_probs = probs.max(dim=1)[0]
        
        print(f"\nTest {i+1}: {params}")
        print(f"  Mean max prob: {max_probs.mean():.4f}")
        print(f"  Std max prob: {max_probs.std():.4f}")

def evaluate_actual_defense_auc():
    """Actually evaluate the AUC under defense using the pre-trained attack model"""
    from membership_inference_attack import (
        extract_attack_features, evaluate_attack, AttackModel, Config as ATKCFG
    )
    from sklearn.preprocessing import StandardScaler
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load defended model
    # base = SimpleCNN().to(device)
    base = DefendedCNN().to(device)
    base.load_state_dict(torch.load("defended_target.pth", map_location=device))
    
    # Apply defense parameters
    DEF.temperature = 6.0
    DEF.noise_std_min = 0.30
    DEF.noise_std_max = 0.80
    DEF.clip_max = 0.60
    DEF.round_digit = 0
    
    defended = DefenceWrapper(base).to(device).eval()
    
    # Load datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_train = datasets.MNIST("./data", train=True, transform=transform)
    mnist_test = datasets.MNIST("./data", train=False, transform=transform)
    
    # Use same indices as in attack
    member_idx = list(range(0, 5000))
    nonmem_idx = list(range(5000))
    
    member_loader = DataLoader(Subset(mnist_train, member_idx), batch_size=256)
    nonmem_loader = DataLoader(Subset(mnist_test, nonmem_idx), batch_size=256)
    
    print("\n=== Collecting Defense Features ===")
    member_feats, nonmem_feats = [], []
    
    with torch.no_grad():
        for xb, _ in member_loader:
            xb = xb.to(device)
            logits = defended.forward(xb)
            member_feats.append(extract_attack_features(logits).cpu())
            
        for xb, _ in nonmem_loader:
            xb = xb.to(device)
            logits = defended.forward(xb)
            nonmem_feats.append(extract_attack_features(logits).cpu())
    
    member_feats = torch.cat(member_feats)
    nonmem_feats = torch.cat(nonmem_feats)
    
    # Balance
    min_len = min(len(member_feats), len(nonmem_feats))
    member_feats = member_feats[:min_len]
    nonmem_feats = nonmem_feats[:min_len]
    
    # Create labels
    y = torch.cat([torch.ones(min_len), torch.zeros(min_len)]).long()
    X = torch.cat([member_feats, nonmem_feats])
    
    # Load attack model
    attack = AttackModel(input_dim=X.shape[1]).to(device)
    attack.load_state_dict(torch.load(f"{ATKCFG.LOG_DIR}/attack_model_best.pth", map_location=device))
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = torch.tensor(scaler.fit_transform(X)).float()
    
    # Evaluate
    results = evaluate_attack(attack, X_scaled, y, scaler, title="Defended Model", already_scaled=True)
    
    return results["auc"]

if __name__ == "__main__":
    print("Running defense verification...\n")
    
    # 1. Verify defense components
    base_model, defended_model = verify_defense_components()
    
    # 2. Evaluate utility
    base_acc, def_acc = evaluate_model_utility(base_model, defended_model)
    
    # 3. Analyze confidence distributions
    analyze_confidence_distribution(base_model, defended_model)
    
    # 4. Test parameter sensitivity
    test_parameter_sensitivity()
    
    # 5. Evaluate actual defense AUC
    actual_auc = evaluate_actual_defense_auc()
    
    print("\n=== Summary ===")
    print(f"Defense reduces AUC to: {actual_auc:.4f} (target: 0.50)")
    print(f"Utility loss: {base_acc - def_acc:.2f}%")
    print(f"Defense is {'SUCCESSFUL' if actual_auc <= 0.53 and def_acc > 85 else 'NEEDS IMPROVEMENT'}")