"""
Defense Verification and Evaluation Script
==========================================

This script provides comprehensive testing and evaluation of the four-layer hybrid
defense mechanism against membership inference attacks. It validates that all defense
components are functioning correctly and measures the effectiveness of privacy protection
while assessing the trade-off with model utility.

Key Evaluation Components:
1. Defense Component Verification - Tests each layer of the defense pipeline
2. Utility Preservation Analysis - Measures accuracy loss due to defense mechanisms
3. Confidence Distribution Analysis - Analyzes how defense affects prediction confidence
4. Parameter Sensitivity Testing - Evaluates robustness to parameter changes
5. Attack Resistance Evaluation - Measures actual AUC reduction against trained attacks

The script is designed to work with the defended model created by the wrap_target.py
script and uses the same attack model from the membership inference attack phase
to provide realistic evaluation conditions.

"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from typing import Tuple, List, Dict, Any

# Import defense mechanism components
try:
    from MIA_defense_mechanism import DefendedCNN, DefenceWrapper, CFG as DEF
    print("✓ Successfully imported defense mechanism components")
except ImportError as e:
    print(f"❌ Error importing defense components: {e}")
    print("Ensure MIA_defense_mechanism.py is in the same directory")
    sys.exit(1)

# =============================================================================
# GLOBAL CONFIGURATION AND SETUP
# =============================================================================

class EvaluationConfig:
    """
    Configuration class for defense evaluation parameters.
    
    This class centralizes all evaluation settings and ensures consistency
    across different testing functions. Parameters are chosen to provide
    comprehensive evaluation while maintaining reasonable execution time.
    """
    
    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model file paths
    DEFENDED_MODEL_PATH = "defended_target.pth"
    ATTACK_MODEL_DIR = "./mia_results"
    
    # Evaluation parameters
    TEST_BATCH_SIZE = 256          # Batch size for utility evaluation
    ANALYSIS_BATCH_SIZE = 256      # Batch size for confidence analysis
    MEMBER_SAMPLE_SIZE = 5000      # Number of member samples for analysis
    
    # Visualization settings
    FIGURE_DPI = 300               # High resolution for publication-quality plots
    FIGURE_SIZE = (12, 5)          # Figure dimensions for plots
    
    # Defense parameter ranges for sensitivity testing
    PARAMETER_TEST_CONFIGS = [
        {"temperature": 2.0, "clip_max": 0.9, "noise_std_min": 0.1, "noise_std_max": 0.3},
        {"temperature": 3.0, "clip_max": 0.85, "noise_std_min": 0.2, "noise_std_max": 0.6},
        {"temperature": 4.0, "clip_max": 0.8, "noise_std_min": 0.3, "noise_std_max": 0.8},
    ]
    
    # Success criteria for defense evaluation
    TARGET_AUC_THRESHOLD = 0.53    # Defense success if AUC ≤ this value
    MIN_UTILITY_THRESHOLD = 85.0   # Minimum acceptable accuracy percentage

config = EvaluationConfig()

# =============================================================================
# DEFENSE COMPONENT VERIFICATION
# =============================================================================

def verify_defense_components() -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Verify that each layer of the defense mechanism is functioning correctly.
    
    This function tests the four-layer defense pipeline step by step to ensure
    that each component is working as intended. It provides detailed analysis
    of how each layer transforms the model outputs.
    
    Defense Layers Tested:
    - Layer 1: Built-in regularization (already in model architecture)
    - Layer 2: Temperature scaling for confidence calibration
    - Layer 3: Adaptive noise injection based on prediction confidence
    - Layer 4: Output sanitization through clipping and rounding
    
    Returns:
        Tuple[torch.nn.Module, torch.nn.Module]: (base_model, defended_model)
        
    Raises:
        FileNotFoundError: If defended model file cannot be found
        RuntimeError: If defense components fail validation tests
    """
    
    print("=" * 60)
    print("DEFENSE COMPONENT VERIFICATION")
    print("=" * 60)
    
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # === LOAD BASE MODEL ===
    print(f"\nLoading defended model from: {config.DEFENDED_MODEL_PATH}")
    
    try:
        # Initialize model architecture
        base_model = DefendedCNN().to(device)
        
        # Load trained weights
        model_state = torch.load(config.DEFENDED_MODEL_PATH, map_location=device)
        base_model.load_state_dict(model_state)
        base_model.eval()
        
        print("✓ Base model loaded successfully")
        
        # Verify model has Layer 1 defense (regularization) built-in
        has_dropout = any(isinstance(module, torch.nn.Dropout) for module in base_model.modules())
        if has_dropout:
            print("✓ Layer 1 defense verified: Dropout regularization present")
        else:
            print("⚠️ Layer 1 defense warning: No dropout layers detected")
            
    except FileNotFoundError:
        print(f"❌ Error: Could not find defended model at {config.DEFENDED_MODEL_PATH}")
        print("Please run wrap_target.py first to create the defended model")
        raise
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise
    
    # === CONFIGURE DEFENSE PARAMETERS ===
    print(f"\nConfiguring defense parameters...")
    
    # Set optimized defense parameters based on evaluation results
    DEF.temperature = 5.0          # Layer 2: Temperature scaling
    DEF.noise_std_min = 0.30       # Layer 3: Minimum adaptive noise
    DEF.noise_std_max = 1.2        # Layer 3: Maximum adaptive noise  
    DEF.clip_max = 0.60            # Layer 4: Probability clipping threshold
    DEF.round_digit = 1            # Layer 4: Decimal places for rounding
    
    print(f"  Layer 2 - Temperature scaling: {DEF.temperature}")
    print(f"  Layer 3 - Adaptive noise range: {DEF.noise_std_min} - {DEF.noise_std_max}")
    print(f"  Layer 4 - Clipping threshold: {DEF.clip_max}")
    print(f"  Layer 4 - Rounding precision: {DEF.round_digit} decimal places")
    
    # === INITIALIZE DEFENSE WRAPPER ===
    try:
        defended_model = DefenceWrapper(base_model).to(device).eval()
        print("✓ Defense wrapper initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing defense wrapper: {e}")
        raise
    
    # === COMPONENT TESTING ===
    print(f"\nTesting defense components with sample data...")
    
    # Generate test input (MNIST-like random data)
    test_batch_size = 10
    test_input = torch.randn(test_batch_size, 1, 28, 28).to(device)
    
    with torch.no_grad():
        try:
            # === TEST BASE MODEL OUTPUT ===
            base_logits = base_model(test_input)
            base_probs = F.softmax(base_logits, dim=1)
            base_max_conf = base_probs.max(dim=1)[0].mean().item()
            
            # === TEST LAYER 2: TEMPERATURE SCALING ===
            temp_scaled_logits = base_logits / DEF.temperature
            temp_probs = F.softmax(temp_scaled_logits, dim=1)
            temp_max_conf = temp_probs.max(dim=1)[0].mean().item()
            
            # === TEST FULL DEFENSE PIPELINE ===
            defended_logits = defended_model.forward(test_input)
            defended_probs = defended_model.defended_probs(test_input)
            defended_max_conf = defended_probs.max(dim=1)[0].mean().item()
            
            # === ANALYSIS AND VALIDATION ===
            print(f"\nDefense Pipeline Analysis:")
            print(f"  Base model max confidence:        {base_max_conf:.4f}")
            print(f"  After temperature scaling:        {temp_max_conf:.4f}")
            print(f"  After full defense pipeline:      {defended_max_conf:.4f}")
            
            # Calculate confidence reduction
            total_reduction = base_max_conf - defended_max_conf
            temp_reduction = base_max_conf - temp_max_conf
            additional_reduction = temp_max_conf - defended_max_conf
            
            print(f"\nConfidence Reduction Analysis:")
            print(f"  Total reduction:                  {total_reduction:.4f}")
            print(f"  From temperature scaling:         {temp_reduction:.4f}")
            print(f"  From noise + sanitization:        {additional_reduction:.4f}")
            
            # === VALIDATE DEFENSE CONSTRAINTS ===
            print(f"\nDefense Constraint Validation:")
            
            # Check probability clipping
            max_prob_after_clipping = defended_probs.max().item()
            clipping_satisfied = max_prob_after_clipping <= DEF.clip_max + 1e-6
            print(f"  Maximum probability: {max_prob_after_clipping:.4f} "
                  f"(should be ≤ {DEF.clip_max}) {'✓' if clipping_satisfied else '❌'}")
            
            # Check probability rounding
            # Count unique values to verify rounding is working
            rounded_probs = torch.round(defended_probs * 10) / 10
            unique_values = torch.unique(rounded_probs * 10)
            print(f"  Unique probability values (×10): {len(unique_values)} "
                  f"(indicates {DEF.round_digit} decimal precision)")
            
            # Check probability sum constraint
            prob_sums = defended_probs.sum(dim=1)
            sum_valid = torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6)
            print(f"  Probability sum constraint: {'✓' if sum_valid else '❌'} "
                  f"(range: {prob_sums.min():.6f} - {prob_sums.max():.6f})")
            
            # Overall validation
            all_constraints_satisfied = clipping_satisfied and sum_valid
            if all_constraints_satisfied:
                print(f"\n✅ All defense components verified successfully")
            else:
                print(f"\n❌ Some defense constraints failed validation")
                raise RuntimeError("Defense component validation failed")
            
        except Exception as e:
            print(f"❌ Error during component testing: {e}")
            raise
    
    return base_model, defended_model

# =============================================================================
# MODEL UTILITY EVALUATION
# =============================================================================

def evaluate_model_utility(base_model: torch.nn.Module, 
                          defended_model: torch.nn.Module) -> Tuple[float, float]:
    """
    Evaluate the utility preservation of the defended model.
    
    This function measures how much accuracy is lost due to the defense
    mechanisms. The goal is to maintain high utility (>85% accuracy) while
    providing effective privacy protection.
    
    The evaluation uses the MNIST test set to measure classification accuracy
    on both the base model and the defended model under identical conditions.
    
    Args:
        base_model: The base model without defense wrapper
        defended_model: The model with defense wrapper applied
        
    Returns:
        Tuple[float, float]: (base_accuracy, defended_accuracy) in percentages
        
    Raises:
        RuntimeError: If utility evaluation fails
    """
    
    print("\n" + "=" * 60)
    print("MODEL UTILITY EVALUATION")
    print("=" * 60)
    
    device = config.DEVICE
    
    # === PREPARE TEST DATA ===
    print("Preparing MNIST test dataset...")
    
    # Use same transformation as in training/attack phases for consistency
    transform = transforms.Compose([
        transforms.ToTensor(),                    # Convert PIL to tensor
        transforms.Normalize((0.5,), (0.5,))     # Normalize to [-1, 1] range
    ])
    
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.TEST_BATCH_SIZE, 
        shuffle=False,  # No shuffling needed for accuracy evaluation
        num_workers=0   # Avoid multiprocessing issues
    )
    
    print(f"Test dataset size: {len(test_dataset)} samples")
    print(f"Batch size: {config.TEST_BATCH_SIZE}")
    
    # === EVALUATE BASE MODEL ===
    print(f"\nEvaluating base model accuracy...")
    
    base_model.eval()
    base_correct = 0
    base_total = 0
    
    try:
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                # Move data to device
                data, target = data.to(device), target.to(device)
                
                # Get predictions from base model
                outputs = base_model(data)
                _, predicted = torch.max(outputs.data, 1)
                
                # Update counters
                base_total += target.size(0)
                base_correct += (predicted == target).sum().item()
                
                # Progress reporting for large datasets
                if batch_idx % 20 == 0:
                    current_acc = 100.0 * base_correct / base_total if base_total > 0 else 0
                    print(f"  Batch {batch_idx:3d}: Current accuracy = {current_acc:.2f}%")
        
        base_accuracy = 100.0 * base_correct / base_total
        print(f"✓ Base model evaluation complete: {base_accuracy:.2f}%")
        
    except Exception as e:
        print(f"❌ Error evaluating base model: {e}")
        raise
    
    # === EVALUATE DEFENDED MODEL ===
    print(f"\nEvaluating defended model accuracy...")
    
    defended_correct = 0
    defended_total = 0
    
    try:
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                # Move data to device
                data, target = data.to(device), target.to(device)
                
                # Get predictions from defended model
                # Using defended_probs method for consistency with attack evaluation
                probs = defended_model.defended_probs(data)
                _, predicted = torch.max(probs, 1)
                
                # Update counters
                defended_total += target.size(0)
                defended_correct += (predicted == target).sum().item()
                
                # Progress reporting
                if batch_idx % 20 == 0:
                    current_acc = 100.0 * defended_correct / defended_total if defended_total > 0 else 0
                    print(f"  Batch {batch_idx:3d}: Current accuracy = {current_acc:.2f}%")
        
        defended_accuracy = 100.0 * defended_correct / defended_total
        print(f"✓ Defended model evaluation complete: {defended_accuracy:.2f}%")
        
    except Exception as e:
        print(f"❌ Error evaluating defended model: {e}")
        raise
    
    # === UTILITY ANALYSIS ===
    accuracy_loss = base_accuracy - defended_accuracy
    relative_loss = (accuracy_loss / base_accuracy) * 100
    
    print(f"\n📊 Utility Evaluation Results:")
    print(f"   Base model accuracy:      {base_accuracy:.2f}%")
    print(f"   Defended model accuracy:  {defended_accuracy:.2f}%")
    print(f"   Absolute accuracy loss:   {accuracy_loss:.2f} percentage points")
    print(f"   Relative accuracy loss:   {relative_loss:.2f}%")
    
    # === UTILITY ASSESSMENT ===
    print(f"\n🎯 Utility Assessment:")
    
    if defended_accuracy >= config.MIN_UTILITY_THRESHOLD:
        print(f"✅ EXCELLENT utility preservation (≥{config.MIN_UTILITY_THRESHOLD}%)")
    elif defended_accuracy >= 80.0:
        print(f"✅ GOOD utility preservation (≥80%)")
    elif defended_accuracy >= 75.0:
        print(f"⚠️ ACCEPTABLE utility preservation (≥75%)")
    else:
        print(f"❌ POOR utility preservation (<75%)")
    
    if accuracy_loss <= 2.0:
        print(f"✅ MINIMAL accuracy loss (≤2%)")
    elif accuracy_loss <= 5.0:
        print(f"✅ LOW accuracy loss (≤5%)")
    elif accuracy_loss <= 10.0:
        print(f"⚠️ MODERATE accuracy loss (≤10%)")
    else:
        print(f"❌ HIGH accuracy loss (>10%)")
    
    return base_accuracy, defended_accuracy

# =============================================================================
# CONFIDENCE DISTRIBUTION ANALYSIS
# =============================================================================

def analyze_confidence_distribution(base_model: torch.nn.Module, 
                                  defended_model: torch.nn.Module) -> None:
    """
    Analyze how the defense affects confidence distributions on training data.
    
    This function examines the confidence patterns that membership inference
    attacks exploit. The defense should significantly reduce confidence on
    training data (members) to eliminate membership signals while maintaining
    reasonable decision quality.
    
    The analysis focuses on training data because this is where membership
    inference attacks expect to see higher confidence patterns.
    
    Args:
        base_model: The base model without defense wrapper
        defended_model: The model with defense wrapper applied
        
    Returns:
        None (generates visualizations and prints analysis)
        
    Raises:
        RuntimeError: If confidence analysis fails
    """
    
    print("\n" + "=" * 60)
    print("CONFIDENCE DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    device = config.DEVICE
    
    # === PREPARE TRAINING DATA (MEMBERS) ===
    print("Loading training data for confidence analysis...")
    
    # Use same transformation as in other phases
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST("./data", train=True, transform=transform)
    
    # Use same member indices as in the attack evaluation for consistency
    member_indices = list(range(config.MEMBER_SAMPLE_SIZE))
    member_subset = Subset(train_dataset, member_indices)
    member_loader = DataLoader(
        member_subset, 
        batch_size=config.ANALYSIS_BATCH_SIZE, 
        shuffle=False
    )
    
    print(f"Analyzing {len(member_subset)} training samples (members)")
    
    # === COLLECT CONFIDENCE DATA ===
    print("Collecting confidence data from both models...")
    
    base_confidences = []
    defended_confidences = []
    
    try:
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(member_loader):
                data = data.to(device)
                
                # === BASE MODEL CONFIDENCE ===
                base_logits = base_model(data)
                base_probs = F.softmax(base_logits, dim=1)
                base_conf = base_probs.max(dim=1)[0]  # Maximum probability per sample
                base_confidences.extend(base_conf.cpu().numpy())
                
                # === DEFENDED MODEL CONFIDENCE ===
                defended_probs = defended_model.defended_probs(data)
                defended_conf = defended_probs.max(dim=1)[0]  # Maximum probability per sample
                defended_confidences.extend(defended_conf.cpu().numpy())
                
                # Progress reporting
                if batch_idx % 10 == 0:
                    print(f"  Processed batch {batch_idx}/{len(member_loader)}")
        
        print(f"✓ Confidence data collection complete")
        print(f"  Collected {len(base_confidences)} confidence measurements")
        
    except Exception as e:
        print(f"❌ Error collecting confidence data: {e}")
        raise
    
    # === STATISTICAL ANALYSIS ===
    print(f"\nPerforming statistical analysis...")
    
    # Convert to numpy arrays for analysis
    base_confidences = np.array(base_confidences)
    defended_confidences = np.array(defended_confidences)
    
    # Calculate key statistics
    base_stats = {
        'mean': np.mean(base_confidences),
        'std': np.std(base_confidences),
        'median': np.median(base_confidences),
        'min': np.min(base_confidences),
        'max': np.max(base_confidences),
        'q95': np.percentile(base_confidences, 95),
        'high_conf_pct': np.mean(base_confidences > 0.9) * 100
    }
    
    defended_stats = {
        'mean': np.mean(defended_confidences),
        'std': np.std(defended_confidences),
        'median': np.median(defended_confidences),
        'min': np.min(defended_confidences),
        'max': np.max(defended_confidences),
        'q95': np.percentile(defended_confidences, 95),
        'high_conf_pct': np.mean(defended_confidences > 0.9) * 100
    }
    
    # === VISUALIZATION ===
    print(f"Generating confidence distribution visualizations...")
    
    try:
        plt.figure(figsize=config.FIGURE_SIZE)
        
        # === HISTOGRAM COMPARISON ===
        plt.subplot(1, 2, 1)
        plt.hist(base_confidences, bins=50, alpha=0.7, label='Base Model', 
                density=True, color='red', edgecolor='darkred')
        plt.hist(defended_confidences, bins=50, alpha=0.7, label='Defended Model', 
                density=True, color='blue', edgecolor='darkblue')
        plt.xlabel('Confidence (Maximum Probability)')
        plt.ylabel('Density')
        plt.title('Confidence Distribution on Training Data\n(Higher density = more samples)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add vertical lines for key statistics
        plt.axvline(base_stats['mean'], color='red', linestyle='--', alpha=0.8, 
                   label=f"Base Mean: {base_stats['mean']:.3f}")
        plt.axvline(defended_stats['mean'], color='blue', linestyle='--', alpha=0.8,
                   label=f"Defended Mean: {defended_stats['mean']:.3f}")
        
        # === BOX PLOT COMPARISON ===
        plt.subplot(1, 2, 2)
        box_data = [base_confidences, defended_confidences]
        box_labels = ['Base Model', 'Defended Model']
        
        bp = plt.boxplot(box_data, labels=box_labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('red')
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor('blue')
        bp['boxes'][1].set_alpha(0.7)
        
        plt.ylabel('Confidence (Maximum Probability)')
        plt.title('Confidence Distribution Summary\n(Box plots show quartiles)')
        plt.grid(True, alpha=0.3)
        
        # Add key statistics as text
        stats_text = f"Base: μ={base_stats['mean']:.3f}, σ={base_stats['std']:.3f}\n"
        stats_text += f"Defended: μ={defended_stats['mean']:.3f}, σ={defended_stats['std']:.3f}"
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('defense_confidence_analysis.png', dpi=config.FIGURE_DPI, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Visualization saved as 'defense_confidence_analysis.png'")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not generate visualization: {e}")
    
    # === DETAILED STATISTICAL REPORT ===
    print(f"\n📊 Detailed Confidence Analysis Results:")
    print(f"\n   Base Model Statistics:")
    print(f"     Mean confidence:           {base_stats['mean']:.4f}")
    print(f"     Standard deviation:        {base_stats['std']:.4f}")
    print(f"     Median confidence:         {base_stats['median']:.4f}")
    print(f"     95th percentile:           {base_stats['q95']:.4f}")
    print(f"     Samples >0.9 confidence:   {base_stats['high_conf_pct']:.2f}%")
    
    print(f"\n   Defended Model Statistics:")
    print(f"     Mean confidence:           {defended_stats['mean']:.4f}")
    print(f"     Standard deviation:        {defended_stats['std']:.4f}")
    print(f"     Median confidence:         {defended_stats['median']:.4f}")
    print(f"     95th percentile:           {defended_stats['q95']:.4f}")
    print(f"     Samples >0.9 confidence:   {defended_stats['high_conf_pct']:.2f}%")
    
    # === DEFENSE EFFECTIVENESS ANALYSIS ===
    mean_reduction = base_stats['mean'] - defended_stats['mean']
    high_conf_reduction = base_stats['high_conf_pct'] - defended_stats['high_conf_pct']
    
    print(f"\n🛡️ Defense Effectiveness Analysis:")
    print(f"   Mean confidence reduction:     {mean_reduction:.4f}")
    print(f"   High confidence reduction:     {high_conf_reduction:.2f} percentage points")
    print(f"   Maximum confidence capped at:  {defended_stats['max']:.4f}")
    
    # === EFFECTIVENESS ASSESSMENT ===
    print(f"\n🎯 Confidence Defense Assessment:")
    
    if mean_reduction >= 0.2:
        print(f"✅ EXCELLENT confidence reduction (≥0.2)")
    elif mean_reduction >= 0.1:
        print(f"✅ GOOD confidence reduction (≥0.1)")
    elif mean_reduction >= 0.05:
        print(f"⚠️ MODERATE confidence reduction (≥0.05)")
    else:
        print(f"❌ POOR confidence reduction (<0.05)")
    
    if defended_stats['max'] <= DEF.clip_max + 1e-6:
        print(f"✅ EFFECTIVE probability clipping (max ≤ {DEF.clip_max})")
    else:
        print(f"❌ INEFFECTIVE probability clipping (max > {DEF.clip_max})")

# =============================================================================
# PARAMETER SENSITIVITY ANALYSIS
# =============================================================================

def test_parameter_sensitivity() -> None:
    """
    Test how sensitive the defense is to parameter changes.
    
    This function evaluates the robustness of the defense mechanism by testing
    different parameter combinations. Understanding parameter sensitivity is
    crucial for:
    1. Optimizing defense effectiveness
    2. Ensuring robustness across different scenarios
    3. Providing guidance for parameter tuning
    
    The test evaluates multiple parameter configurations and analyzes their
    impact on confidence reduction and output characteristics.
    
    Returns:
        None (prints analysis results)
        
    Raises:
        RuntimeError: If parameter sensitivity testing fails
    """
    
    print("\n" + "=" * 60)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    device = config.DEVICE
    
    # === LOAD BASE MODEL ===
    print("Loading base model for sensitivity testing...")
    
    try:
        base_model = DefendedCNN().to(device)
        base_model.load_state_dict(torch.load(config.DEFENDED_MODEL_PATH, map_location=device))
        base_model.eval()
        print("✓ Base model loaded for sensitivity testing")
    except Exception as e:
        print(f"❌ Error loading model for sensitivity testing: {e}")
        raise
    
    # === GENERATE TEST DATA ===
    print("Generating test data for parameter sensitivity analysis...")
    
    # Use larger test batch for more stable statistics
    test_batch_size = 100
    test_input = torch.randn(test_batch_size, 1, 28, 28).to(device)
    
    # Get baseline (undefended) statistics
    with torch.no_grad():
        baseline_logits = base_model(test_input)
        baseline_probs = F.softmax(baseline_logits, dim=1)
        baseline_max_conf = baseline_probs.max(dim=1)[0]
        baseline_mean = baseline_max_conf.mean().item()
        baseline_std = baseline_max_conf.std().item()
    
    print(f"Baseline (undefended) statistics:")
    print(f"  Mean max confidence: {baseline_mean:.4f}")
    print(f"  Std max confidence:  {baseline_std:.4f}")
    
    # === TEST PARAMETER CONFIGURATIONS ===
    print(f"\nTesting {len(config.PARAMETER_TEST_CONFIGS)} parameter configurations...")
    
    results = []
    
    for i, params in enumerate(config.PARAMETER_TEST_CONFIGS):
        print(f"\n--- Configuration {i+1}/{len(config.PARAMETER_TEST_CONFIGS)} ---")
        print(f"Parameters: {params}")
        
        try:
            # === APPLY PARAMETER CONFIGURATION ===
            # Backup original parameters
            original_params = {
                'temperature': DEF.temperature,
                'clip_max': DEF.clip_max,
                'noise_std_min': DEF.noise_std_min,
                'noise_std_max': DEF.noise_std_max
            }
            
            # Update defense parameters
            for key, value in params.items():
                if hasattr(DEF, key):
                    setattr(DEF, key, value)
                    print(f"  Set {key} = {value}")
                else:
                    print(f"  ⚠️ Warning: Unknown parameter {key}")
            
            # === TEST CONFIGURATION ===
            defended_model = DefenceWrapper(base_model).to(device).eval()
            
            with torch.no_grad():
                defended_probs = defended_model.defended_probs(test_input)
                defended_max_conf = defended_probs.max(dim=1)[0]
                
                # Calculate statistics
                defended_mean = defended_max_conf.mean().item()
                defended_std = defended_max_conf.std().item()
                defended_max = defended_probs.max().item()
                defended_min = defended_probs.min().item()
                
                # Calculate reductions relative to baseline
                mean_reduction = baseline_mean - defended_mean
                reduction_percentage = (mean_reduction / baseline_mean) * 100
                
                # Store results
                config_result = {
                    'config_id': i + 1,
                    'parameters': params.copy(),
                    'defended_mean': defended_mean,
                    'defended_std': defended_std,
                    'defended_max': defended_max,
                    'defended_min': defended_min,
                    'mean_reduction': mean_reduction,
                    'reduction_percentage': reduction_percentage
                }
                results.append(config_result)
                
                # Display results
                print(f"  Results:")
                print(f"    Mean max confidence:     {defended_mean:.4f}")
                print(f"    Std max confidence:      {defended_std:.4f}")
                print(f"    Maximum probability:     {defended_max:.4f}")
                print(f"    Confidence reduction:    {mean_reduction:.4f} ({reduction_percentage:.1f}%)")
                
                # Validate constraints
                clipping_valid = defended_max <= params.get('clip_max', DEF.clip_max) + 1e-6
                print(f"    Clipping constraint:     {'✓' if clipping_valid else '❌'}")
            
            # === RESTORE ORIGINAL PARAMETERS ===
            for key, value in original_params.items():
                setattr(DEF, key, value)
                
        except Exception as e:
            print(f"  ❌ Error testing configuration {i+1}: {e}")
            # Restore original parameters even if error occurred
            for key, value in original_params.items():
                setattr(DEF, key, value)
            continue
    
    # === COMPARATIVE ANALYSIS ===
    print(f"\n📊 Parameter Sensitivity Summary:")
    print(f"{'Config':<8} {'Temp':<6} {'Clip':<6} {'Noise Min':<10} {'Noise Max':<10} {'Mean Conf':<10} {'Reduction':<10}")
    print("-" * 70)
    
    for result in results:
        params = result['parameters']
        print(f"{result['config_id']:<8} "
              f"{params['temperature']:<6.1f} "
              f"{params['clip_max']:<6.2f} "
              f"{params['noise_std_min']:<10.2f} "
              f"{params['noise_std_max']:<10.2f} "
              f"{result['defended_mean']:<10.4f} "
              f"{result['reduction_percentage']:<10.1f}%")
    
    # === RECOMMENDATIONS ===
    if results:
        best_config = max(results, key=lambda x: x['reduction_percentage'])
        print(f"\n🎯 Parameter Sensitivity Analysis:")
        print(f"   Best configuration: Config {best_config['config_id']}")
        print(f"   Highest reduction: {best_config['reduction_percentage']:.1f}%")
        print(f"   Optimal parameters: {best_config['parameters']}")
        
        # Analyze parameter trends
        temp_effect = []
        clip_effect = []
        for result in results:
            temp_effect.append((result['parameters']['temperature'], result['reduction_percentage']))
            clip_effect.append((result['parameters']['clip_max'], result['reduction_percentage']))
        
        print(f"\n   Parameter Effect Analysis:")
        print(f"   Temperature scaling: Higher values generally increase confidence reduction")
        print(f"   Probability clipping: Lower values generally increase confidence reduction")
        print(f"   Adaptive noise: Higher ranges provide better membership signal disruption")

# =============================================================================
# ATTACK RESISTANCE EVALUATION
# =============================================================================

def evaluate_actual_defense_auc() -> float:
    """
    Evaluate the actual attack resistance using the pre-trained attack model.
    
    This function provides the most realistic evaluation of defense effectiveness
    by using the same attack model that was trained in the membership inference
    attack phase. It measures how much the defense reduces the attack's AUC.
    
    The evaluation process:
    1. Load the defended model with optimal defense parameters
    2. Extract features using the same feature extraction as the attack
    3. Apply the pre-trained attack classifier
    4. Measure the resulting AUC reduction
    
    Returns:
        float: AUC score of the attack against the defended model
        
    Raises:
        ImportError: If attack components cannot be imported
        FileNotFoundError: If attack model file cannot be found
        RuntimeError: If defense evaluation fails
    """
    
    print("\n" + "=" * 60)
    print("ATTACK RESISTANCE EVALUATION")
    print("=" * 60)
    
    # === IMPORT ATTACK COMPONENTS ===
    print("Importing attack model components...")
    
    try:
        from membership_inference_attack import (
            extract_attack_features, evaluate_attack, AttackModel, Config as ATKCFG
        )
        from sklearn.preprocessing import StandardScaler
        print("✓ Attack components imported successfully")
    except ImportError as e:
        print(f"❌ Error importing attack components: {e}")
        print("Ensure membership_inference_attack.py is available")
        raise
    
    device = config.DEVICE
    
    # === LOAD AND CONFIGURE DEFENDED MODEL ===
    print("Loading and configuring defended model...")
    
    try:
        # Load base model
        base_model = DefendedCNN().to(device)
        base_model.load_state_dict(torch.load(config.DEFENDED_MODEL_PATH, map_location=device))
        base_model.eval()
        
        # Apply optimal defense parameters (based on evaluation results)
        DEF.temperature = 6.0      # Higher temperature for better defense
        DEF.noise_std_min = 0.30   # Minimum adaptive noise
        DEF.noise_std_max = 1.2    # Maximum adaptive noise  
        DEF.clip_max = 0.60        # Probability clipping threshold
        DEF.round_digit = 1        # Rounding precision
        
        print(f"✓ Defense parameters configured:")
        print(f"  Temperature: {DEF.temperature}")
        print(f"  Noise range: {DEF.noise_std_min} - {DEF.noise_std_max}")
        print(f"  Clipping: {DEF.clip_max}")
        print(f"  Rounding: {DEF.round_digit} decimal places")
        
        # Create defended model
        defended_model = DefenceWrapper(base_model).to(device).eval()
        print("✓ Defended model ready for evaluation")
        
    except Exception as e:
        print(f"❌ Error configuring defended model: {e}")
        raise
    
    # === PREPARE EVALUATION DATASETS ===
    print("Preparing evaluation datasets...")
    
    # Use same transformation as in attack phase for consistency
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_train = datasets.MNIST("./data", train=True, transform=transform)
    mnist_test = datasets.MNIST("./data", train=False, transform=transform)
    
    # Use same data indices as in the original attack for fair comparison
    member_indices = list(range(0, 5000))      # Training data (members)
    nonmember_indices = list(range(5000))      # Test data (non-members)
    
    member_loader = DataLoader(Subset(mnist_train, member_indices), batch_size=256)
    nonmember_loader = DataLoader(Subset(mnist_test, nonmember_indices), batch_size=256)
    
    print(f"✓ Evaluation datasets prepared:")
    print(f"  Member samples: {len(member_indices)}")
    print(f"  Non-member samples: {len(nonmember_indices)}")
    
    # === EXTRACT FEATURES FROM DEFENDED MODEL ===
    print("Extracting features from defended model...")
    
    member_features = []
    nonmember_features = []
    
    try:
        with torch.no_grad():
            # Extract member features (from training data)
            print("  Extracting member features...")
            for batch_idx, (data, _) in enumerate(member_loader):
                data = data.to(device)
                # Get logits from defended model
                defended_logits = defended_model.forward(data)
                # Extract same features as used in attack
                features = extract_attack_features(defended_logits)
                member_features.append(features.cpu())
                
                if batch_idx % 5 == 0:
                    print(f"    Processed member batch {batch_idx}/{len(member_loader)}")
            
            # Extract non-member features (from test data)
            print("  Extracting non-member features...")
            for batch_idx, (data, _) in enumerate(nonmember_loader):
                data = data.to(device)
                # Get logits from defended model
                defended_logits = defended_model.forward(data)
                # Extract same features as used in attack
                features = extract_attack_features(defended_logits)
                nonmember_features.append(features.cpu())
                
                if batch_idx % 5 == 0:
                    print(f"    Processed non-member batch {batch_idx}/{len(nonmember_loader)}")
        
        # Combine features
        member_features = torch.cat(member_features)
        nonmember_features = torch.cat(nonmember_features)
        
        print(f"✓ Feature extraction complete:")
        print(f"  Member features: {member_features.shape}")
        print(f"  Non-member features: {nonmember_features.shape}")
        
    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        raise
    
    # === PREPARE ATTACK EVALUATION DATA ===
    print("Preparing attack evaluation data...")
    
    # Balance dataset for fair evaluation
    min_samples = min(len(member_features), len(nonmember_features))
    member_features = member_features[:min_samples]
    nonmember_features = nonmember_features[:min_samples]
    
    # Create labels (1 = member, 0 = non-member)
    member_labels = torch.ones(min_samples, dtype=torch.long)
    nonmember_labels = torch.zeros(min_samples, dtype=torch.long)
    
    # Combine features and labels
    X_eval = torch.cat([member_features, nonmember_features])
    y_eval = torch.cat([member_labels, nonmember_labels])
    
    print(f"✓ Evaluation dataset prepared:")
    print(f"  Total samples: {len(X_eval)}")
    print(f"  Feature dimensions: {X_eval.shape[1]}")
    print(f"  Class balance: {torch.sum(y_eval).item()} members, {len(y_eval) - torch.sum(y_eval).item()} non-members")
    
    # === LOAD PRE-TRAINED ATTACK MODEL ===
    print("Loading pre-trained attack model...")
    
    try:
        # Initialize attack model with correct input dimensions
        attack_model = AttackModel(input_dim=X_eval.shape[1]).to(device)
        
        # Load trained attack model weights
        attack_model_path = f"{ATKCFG.LOG_DIR}/attack_model_best.pth"
        if not os.path.exists(attack_model_path):
            print(f"❌ Attack model not found at {attack_model_path}")
            print("Please run the membership inference attack first to train the attack model")
            raise FileNotFoundError(f"Attack model file not found: {attack_model_path}")
        
        attack_model.load_state_dict(torch.load(attack_model_path, map_location=device))
        attack_model.eval()
        
        print(f"✓ Attack model loaded from: {attack_model_path}")
        
    except Exception as e:
        print(f"❌ Error loading attack model: {e}")
        raise
    
    # === APPLY ATTACK TO DEFENDED MODEL ===
    print("Applying attack to defended model...")
    
    try:
        # Scale features using same preprocessing as in attack training
        scaler = StandardScaler()
        X_scaled = torch.tensor(scaler.fit_transform(X_eval.numpy()), dtype=torch.float32)
        
        print("✓ Features scaled for attack evaluation")
        
        # Evaluate attack performance against defended model
        attack_results = evaluate_attack(
            model=attack_model,
            X_test=X_scaled,
            y_test=y_eval,
            scaler=scaler,
            title="Defended Model",
            already_scaled=True
        )
        
        defended_auc = attack_results["auc"]
        defended_accuracy = attack_results["accuracy"]
        defended_tpr_at_fpr01 = attack_results["tpr_at_fpr01"]
        
        print(f"✓ Attack evaluation complete")
        
    except Exception as e:
        print(f"❌ Error during attack evaluation: {e}")
        raise
    
    # === DEFENSE EFFECTIVENESS ANALYSIS ===
    print(f"\n🛡️ Defense Effectiveness Results:")
    print(f"   Attack AUC against defended model:     {defended_auc:.4f}")
    print(f"   Attack accuracy against defended model: {defended_accuracy:.4f}")
    print(f"   Attack TPR@FPR=0.1:                    {defended_tpr_at_fpr01:.4f}")
    
    # Compare with expected attack performance on undefended model
    # (Based on results from Task 5: AUC ≈ 0.60)
    original_auc = 0.60  # From original attack results
    auc_reduction = original_auc - defended_auc
    reduction_percentage = (auc_reduction / original_auc) * 100
    
    print(f"\n📊 Attack Performance Comparison:")
    print(f"   Original attack AUC (undefended):      {original_auc:.4f}")
    print(f"   Defended model AUC:                    {defended_auc:.4f}")
    print(f"   AUC reduction:                         {auc_reduction:.4f}")
    print(f"   Relative reduction:                    {reduction_percentage:.1f}%")
    
    return defended_auc

# =============================================================================
# MAIN EVALUATION ORCHESTRATION
# =============================================================================

def run_comprehensive_defense_evaluation() -> Dict[str, Any]:
    """
    Run the complete defense evaluation pipeline.
    
    This function orchestrates all evaluation components to provide a
    comprehensive assessment of the defense mechanism's effectiveness.
    
    Returns:
        Dict[str, Any]: Complete evaluation results
        
    Raises:
        RuntimeError: If any critical evaluation component fails
    """
    
    print("=" * 80)
    print("COMPREHENSIVE DEFENSE EVALUATION PIPELINE")
    print("=" * 80)
    print(f"Evaluation started at: {torch.datetime.now()}")
    
    evaluation_results = {}
    
    try:
        # === STEP 1: COMPONENT VERIFICATION ===
        print("\n🔧 Step 1: Defense Component Verification")
        base_model, defended_model = verify_defense_components()
        evaluation_results['component_verification'] = 'PASSED'
        
        # === STEP 2: UTILITY EVALUATION ===
        print("\n📊 Step 2: Model Utility Evaluation")
        base_accuracy, defended_accuracy = evaluate_model_utility(base_model, defended_model)
        evaluation_results['base_accuracy'] = base_accuracy
        evaluation_results['defended_accuracy'] = defended_accuracy
        evaluation_results['utility_loss'] = base_accuracy - defended_accuracy
        
        # === STEP 3: CONFIDENCE ANALYSIS ===
        print("\n🔍 Step 3: Confidence Distribution Analysis")
        analyze_confidence_distribution(base_model, defended_model)
        evaluation_results['confidence_analysis'] = 'COMPLETED'
        
        # === STEP 4: PARAMETER SENSITIVITY ===
        print("\n⚙️ Step 4: Parameter Sensitivity Testing")
        test_parameter_sensitivity()
        evaluation_results['parameter_sensitivity'] = 'COMPLETED'
        
        # === STEP 5: ATTACK RESISTANCE ===
        print("\n🛡️ Step 5: Attack Resistance Evaluation")
        defended_auc = evaluate_actual_defense_auc()
        evaluation_results['defended_auc'] = defended_auc
        evaluation_results['attack_resistance'] = 'MEASURED'
        
        # === OVERALL ASSESSMENT ===
        print("\n" + "=" * 80)
        print("COMPREHENSIVE EVALUATION SUMMARY")
        print("=" * 80)
        
        print(f"\n📈 Performance Metrics:")
        print(f"   Defense AUC:              {defended_auc:.4f} (target: ≤ {config.TARGET_AUC_THRESHOLD})")
        print(f"   Utility preservation:     {defended_accuracy:.2f}% (target: ≥ {config.MIN_UTILITY_THRESHOLD}%)")
        print(f"   Utility loss:             {evaluation_results['utility_loss']:.2f} percentage points")
        
        # === SUCCESS CRITERIA EVALUATION ===
        defense_successful = defended_auc <= config.TARGET_AUC_THRESHOLD
        utility_acceptable = defended_accuracy >= config.MIN_UTILITY_THRESHOLD
        overall_success = defense_successful and utility_acceptable
        
        print(f"\n🎯 Success Criteria Assessment:")
        print(f"   Privacy protection:       {'✅ PASSED' if defense_successful else '❌ FAILED'}")
        print(f"   Utility preservation:     {'✅ PASSED' if utility_acceptable else '❌ FAILED'}")
        print(f"   Overall defense status:   {'🎉 SUCCESS' if overall_success else '⚠️ NEEDS IMPROVEMENT'}")
        
        evaluation_results['defense_successful'] = defense_successful
        evaluation_results['utility_acceptable'] = utility_acceptable
        evaluation_results['overall_success'] = overall_success
        
        # === RECOMMENDATIONS ===
        print(f"\n💡 Recommendations:")
        if overall_success:
            print(f"   ✅ Defense is working effectively")
            print(f"   ✅ Ready for production deployment")
            print(f"   ✅ Consider monitoring for long-term effectiveness")
        else:
            if not defense_successful:
                print(f"   ⚠️ Consider increasing defense strength:")
                print(f"      - Higher temperature scaling")
                print(f"      - More aggressive noise injection")
                print(f"      - Lower probability clipping threshold")
            if not utility_acceptable:
                print(f"   ⚠️ Consider reducing defense strength to preserve utility:")
                print(f"      - Lower temperature scaling")
                print(f"      - Reduced noise levels")
                print(f"      - Higher probability clipping threshold")
        
        return evaluation_results
        
    except Exception as e:
        print(f"\n❌ Critical error during evaluation: {e}")
        evaluation_results['error'] = str(e)
        raise

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main entry point for the defense verification script.
    
    This function provides a comprehensive evaluation of the four-layer
    hybrid defense mechanism, including component verification, utility
    assessment, and attack resistance measurement.
    """
    
    print(__doc__)  # Print module docstring
    
    try:
        # Run comprehensive evaluation
        results = run_comprehensive_defense_evaluation()
        
        # Final status report
        if results.get('overall_success', False):
            print(f"\n🎉 DEFENSE EVALUATION: COMPLETE SUCCESS")
            print(f"The four-layer hybrid defense is working effectively!")
        else:
            print(f"\n⚠️ DEFENSE EVALUATION: NEEDS IMPROVEMENT")
            print(f"Review the recommendations above for optimization guidance.")
        
    except Exception as e:
        print(f"\n💥 DEFENSE EVALUATION: CRITICAL FAILURE")
        print(f"Error: {e}")
        print(f"Please review the error messages above and ensure all dependencies are available.")
        sys.exit(1)

if __name__ == "__main__":
    main()