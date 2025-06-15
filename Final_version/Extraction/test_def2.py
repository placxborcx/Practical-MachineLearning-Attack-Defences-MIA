"""
Fixed Comprehensive Defense Testing Script
==========================================

This module provides comprehensive testing and debugging capabilities for defense
mechanisms against model extraction attacks. It includes multiple testing modes:

Key Features:
- Direct defense activation testing with aggressive configurations
- Configuration validation and debugging utilities
- Comprehensive defense effectiveness evaluation
- Normal usage impact assessment
- Output modification analysis with statistical metrics
- Defense statistics monitoring and reporting

Testing Modes:
1. Direct Activation Test: Verifies defense mechanisms trigger properly
2. Configuration Debug: Validates defense parameter application
3. Comprehensive Evaluation: Full defense effectiveness assessment
4. Normal Usage Impact: Measures legitimate user experience degradation

Author: ML Security Research Team
License: MIT
Version: 2.0 (Fixed with Enhanced Debugging)
"""

#----------------------------------
# IMPORTS AND DEPENDENCIES
#----------------------------------

import json
import time
import torch
import numpy as np
from typing import Dict, List, Tuple
from torch.utils.data import TensorDataset

# Import extraction attack components
from extraction_attack import (
    CFG, get_loaders, load_victim, BlackBoxAPI,
    build_query_set, train_surrogate, accuracy, agreement,
    SurrogateNet, DEVICE
)

# Import defense mechanism (with fallback for compatibility)
try:
    from defense_mechanism_fixed import DefendedBlackBoxAPI, DefenseMechanism
except ImportError:
    from defense_mechanism import DefendedBlackBoxAPI, DefenseMechanism

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configuration constants
IMG_SHAPE = (1, 28, 28)  # (C, H, W) for MNIST dataset
C, H, W = IMG_SHAPE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#----------------------------------
# CONFIGURATION DEBUGGING UTILITIES
#----------------------------------

def debug_defense_config_application(defense_config, config_name):
    """
    Debug utility to verify defense configuration is properly applied.
    
    This function creates a defended API with the given configuration and
    verifies that all parameters are correctly set in the defense mechanism.
    It provides detailed comparison between expected and actual values.
    
    Args:
        defense_config (Dict): Defense configuration parameters to test
        config_name (str): Name of the configuration for logging
        
    Returns:
        DefendedBlackBoxAPI: Configured defended API instance
    """
    print(f"\n--- Debug Config Application: {config_name} ---")
    
    # Create defended API with configuration
    victim = load_victim()
    defended_api = DefendedBlackBoxAPI(victim, defense_config)
    
    # Verify configuration was applied correctly
    actual_config = defended_api.defense.config
    print("Expected vs Actual Config:")
    
    for key, expected_value in defense_config.items():
        actual_value = actual_config.get(key, "NOT_FOUND")
        match_status = "✓" if actual_value == expected_value else "✗"
        print(f"  {key}: {expected_value} -> {actual_value} {match_status}")
    
    return defended_api


#----------------------------------
# DIRECT DEFENSE ACTIVATION TESTING
#----------------------------------

def test_defense_activation_directly():
    """
    Direct test of defense activation with manual queries and aggressive settings.
    
    This function tests the defense mechanism with extremely aggressive configurations
    to ensure that defense components (pattern detection, OOD detection, perturbation)
    are working correctly. It uses repetitive queries to trigger pattern detection
    and monitors defense statistics in real-time.
    
    Key Testing Strategies:
    1. Aggressive configuration with low thresholds
    2. Repetitive queries to trigger pattern detection
    3. Real-time monitoring of defense statistics
    4. Output modification verification
    """
    print("\n" + "="*60)
    print("DIRECT DEFENSE ACTIVATION TEST")
    print("="*60)
    
    # Setup model and data
    victim = load_victim()
    train_loader, test_loader = get_loaders(CFG["batch_size"])
    
    # Extremely aggressive configuration to guarantee defense activation
    aggressive_config = {
        'baseline_suspicion': 0.5,     # Very high baseline (normal: 0.15)
        'perturb_threshold': 0.1,      # Very low threshold (normal: 0.25)
        'block_threshold': 0.6,        # Lower block threshold (normal: 0.70)
        'ood_threshold': 0.5,          # Lower OOD threshold (normal: 0.70)
        'top_k': 2,                    # More restrictive output (normal: 3)
        'base_noise_scale': 0.1,       # Higher base noise (normal: 0.02)
        'max_noise_scale': 0.5,        # Much higher max noise (normal: 0.30)
    }
    
    # Create and initialize defended API
    defended_api = DefendedBlackBoxAPI(victim, aggressive_config)
    defended_api.fit_defense(train_loader)
    
    print("\nTesting with aggressive configuration:")
    print(f"Config: {aggressive_config}")
    
    # Test with same data multiple times to trigger pattern detection
    test_data = next(iter(test_loader))[0][:5].to(DEVICE)
    
    print("\nQuerying same data multiple times to trigger defenses:")
    
    # Run multiple queries with the same data to trigger pattern detection
    for i in range(10):
        # Query the defended API
        output = defended_api.query(test_data, logits=True)
        
        # Get current defense statistics
        stats = defended_api.get_defense_report()
        
        # Display real-time monitoring information
        print(f"Query {i+1}: Avg Suspicion={stats.get('average_suspicion', 0):.3f}, "
              f"Perturb Rate={stats.get('perturb_rate', 0)*100:.1f}%, "
              f"Block Rate={stats.get('block_rate', 0)*100:.1f}%")
        
        # Verify output modification by comparing with original
        with torch.no_grad():
            original_output = victim(test_data)
            original_logprobs = torch.log_softmax(original_output, dim=1)
            
            # Calculate difference between defended and original outputs
            diff = torch.abs(output - original_logprobs).mean().item()
            print(f"         Output difference from original: {diff:.6f}")


#----------------------------------
# COMPREHENSIVE DEFENSE EVALUATION
#----------------------------------

def test_single_defense_configuration_fixed(
    victim_model, 
    train_loader, 
    test_loader,
    defense_config, 
    config_name,
    n_queries=3000  # Reduced for debugging
):
    """
    Fixed comprehensive test of a single defense configuration.
    
    This function performs a thorough evaluation of a defense configuration by:
    1. Testing baseline attack without defense
    2. Testing attack against the defense mechanism
    3. Analyzing defense statistics and effectiveness
    4. Measuring impact on normal usage
    5. Comparing output modifications
    
    Args:
        victim_model: The target model to protect
        train_loader: Training data for defense initialization
        test_loader: Test data for evaluation
        defense_config (Dict): Defense configuration to test
        config_name (str): Name of the configuration for logging
        n_queries (int): Number of queries for the extraction attack
        
    Returns:
        Dict: Comprehensive results including baseline, defended, and analysis metrics
    """
    
    print(f"\n{'='*60}")
    print(f"Testing: {config_name}")
    print('='*60)
    
    # Debug: Verify configuration is applied correctly
    defended_api = debug_defense_config_application(defense_config, config_name)
    defended_api.fit_defense(train_loader)
    
    #--- 1. BASELINE ATTACK (NO DEFENSE) ---
    print("\n1. Baseline Attack (No Defense):")
    undefended_api = BlackBoxAPI(victim_model)
    
    # Run baseline extraction attack
    t_start = time.perf_counter()
    qset_undefended = build_query_set(undefended_api, n_queries)
    surrogate_undefended = train_surrogate(qset_undefended)
    baseline_time = time.perf_counter() - t_start
    
    # Evaluate baseline attack success
    baseline_acc = accuracy(surrogate_undefended, test_loader)
    baseline_agr = agreement(victim_model, surrogate_undefended, test_loader)
    
    print(f"  Surrogate Accuracy: {baseline_acc*100:.2f}%")
    print(f"  Agreement Rate: {baseline_agr*100:.2f}%")
    print(f"  Attack Time: {baseline_time:.2f}s")
    
    #--- 2. DEFENDED ATTACK ---
    print(f"\n2. Attack Against {config_name} Defense:")
    
    # Reset defended API to ensure clean state
    defended_api = DefendedBlackBoxAPI(victim_model, defense_config)
    defended_api.fit_defense(train_loader)
    
    # Run attack against defended model with monitoring
    t_start = time.perf_counter()
    
    print("   Building query set with defense monitoring...")
    qset_defended = build_query_set_with_monitoring(defended_api, n_queries)
    
    surrogate_defended = train_surrogate(qset_defended)
    defended_time = time.perf_counter() - t_start
    
    # Evaluate defended attack results
    defended_acc = accuracy(surrogate_defended, test_loader)
    defended_agr = agreement(victim_model, surrogate_defended, test_loader)
    
    print(f"  Surrogate Accuracy: {defended_acc*100:.2f}%")
    print(f"  Agreement Rate: {defended_agr*100:.2f}%")
    print(f"  Attack Time: {defended_time:.2f}s")
    
    #--- 3. DEFENSE STATISTICS ANALYSIS ---
    defense_report = defended_api.get_defense_report()
    print(f"\n3. Defense Statistics:")
    for key, value in defense_report.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    #--- 4. OUTPUT MODIFICATION ANALYSIS ---
    print(f"\n4. Output Comparison Test:")
    test_batch = next(iter(test_loader))[0][:5].to(DEVICE)
    
    with torch.no_grad():
        # Get original and defended outputs
        original_output = victim_model(test_batch)
        defended_output = defended_api.query(test_batch, logits=True)
        
        # Convert to probability distributions for comparison
        original_probs = torch.softmax(original_output, dim=1)
        defended_probs = torch.exp(defended_output)  # defended_output is log_softmax
        
        # Calculate statistical differences
        kl_div = torch.nn.functional.kl_div(defended_output, original_probs, 
                                          reduction='batchmean').item()
        l1_diff = torch.abs(original_probs - defended_probs).mean().item()
        max_diff = torch.abs(original_probs - defended_probs).max().item()
        
        print(f"  KL Divergence: {kl_div:.6f}")
        print(f"  L1 Difference: {l1_diff:.6f}")
        print(f"  Max Prob Diff: {max_diff:.6f}")
    
    #--- 5. NORMAL USAGE IMPACT ASSESSMENT ---
    print(f"\n5. Normal Usage Impact:")
    normal_acc = test_normal_usage_accuracy(victim_model, defended_api, test_loader, n_samples=500)
    victim_acc = accuracy(victim_model, test_loader)
    
    print(f"  Victim Model Accuracy: {victim_acc*100:.2f}%")
    print(f"  Normal Usage Accuracy: {normal_acc*100:.2f}%")
    print(f"  Accuracy Drop: {(victim_acc - normal_acc)*100:.2f}%")
    
    #--- 6. DEFENSE EFFECTIVENESS ANALYSIS ---
    print(f"\n6. Defense Effectiveness:")
    
    # Calculate reduction in attack success
    if baseline_acc > 0:
        acc_reduction = (baseline_acc - defended_acc) / baseline_acc * 100
    else:
        acc_reduction = 0
    
    if baseline_agr > 0:
        agr_reduction = (baseline_agr - defended_agr) / baseline_agr * 100
    else:
        agr_reduction = 0
    
    print(f"  Accuracy Reduction: {acc_reduction:.2f}%")
    print(f"  Agreement Reduction: {agr_reduction:.2f}%")
    
    # Overall effectiveness score
    effectiveness_score = (acc_reduction + agr_reduction) / 2
    print(f"  Overall Effectiveness Score: {effectiveness_score:.2f}%")
    
    # Warning for ineffective defenses
    if abs(baseline_acc - defended_acc) < 0.001 and abs(baseline_agr - defended_agr) < 0.001:
        print("  ⚠️  WARNING: No significant difference detected between baseline and defended!")
        print("     This suggests the defense is not activating properly.")
    
    # Return comprehensive results
    return {
        'baseline': {
            'accuracy': float(baseline_acc),
            'agreement': float(baseline_agr),
            'time': baseline_time
        },
        'defended': {
            'accuracy': float(defended_acc),
            'agreement': float(defended_agr),
            'time': defended_time
        },
        'defense_stats': defense_report,
        'normal_usage': {
            'accuracy': float(normal_acc),
            'drop': float(victim_acc - normal_acc)
        },
        'effectiveness': {
            'acc_reduction': float(acc_reduction),
            'agr_reduction': float(agr_reduction),
            'score': float(effectiveness_score)
        },
        'output_analysis': {
            'kl_divergence': float(kl_div),
            'l1_difference': float(l1_diff)
        }
    }


#----------------------------------
# QUERY SET GENERATION WITH MONITORING
#----------------------------------

def build_query_set_with_monitoring(api, n_queries):
    """
    Build query set while monitoring defense activation in real-time.
    
    This function creates a synthetic query set for extraction attacks while
    providing detailed monitoring of defense mechanism activation. It uses
    multiple query strategies to trigger different defense components.
    
    Query Strategies:
    1. Random queries (30%): Standard random inputs
    2. Repeated queries (30%): Identical queries to trigger pattern detection
    3. Semi-structured queries (40%): Slightly modified inputs for OOD detection
    
    Args:
        api: The defended API to query
        n_queries (int): Total number of queries to generate
        
    Returns:
        TensorDataset: Dataset containing queries and corresponding labels
    """
    print(f"   Building {n_queries} queries...")
    
    # Initialize storage for queries and labels
    queries = []
    labels = []
    
    # Set up monitoring intervals
    check_intervals = [100, 500, 1000, 2000, n_queries]
    next_check = 0
    
    # Configure batch processing
    batch_size = 32
    n_batches = n_queries // batch_size
    
    # Generate queries in batches with different strategies
    for batch_idx in range(n_batches):
        
        # Select query generation strategy based on batch index
        if batch_idx % 3 == 0:
            #--- Random Queries Strategy (30% of batches) ---
            # Standard random inputs to establish baseline behavior
            x = torch.randn(batch_size, C, H, W, device=DEVICE)
            
        elif batch_idx % 3 == 1:
            #--- Repeated Queries Strategy (30% of batches) ---
            # Repeated queries with small variations to trigger pattern detection
            base_query = torch.randn(1, C, H, W, device=DEVICE)
            x = base_query.repeat(batch_size, 1, 1, 1) + torch.randn(batch_size, C, H, W, device=DEVICE) * 0.1
            
        else:
            #--- Semi-Structured Queries Strategy (40% of batches) ---
            # Structured queries with correlated channels to trigger OOD detection
            x = torch.randn(batch_size, C, H, W, device=DEVICE)
            # Add small correlated noise to make queries slightly suspicious
            x += torch.randn_like(x) * 0.05
        
        # Query the defended API
        y_pred = api.query(x, logits=True)
        
        # Store query and response
        queries.append(x.cpu())
        
        # Convert log-probabilities to probabilities for labels
        probs = torch.exp(y_pred).cpu()     
        labels.append(probs)
        
        # Monitor defense statistics at specified intervals
        current_queries = (batch_idx + 1) * batch_size
        if next_check < len(check_intervals) and current_queries >= check_intervals[next_check]:
            stats = api.get_defense_report()
            print(f"     Query {current_queries}: Suspicion={stats.get('average_suspicion',0):.3f}, "
                  f"Perturb={stats.get('perturb_rate',0)*100:.1f}%, "
                  f"Block={stats.get('block_rate',0)*100:.1f}%")
            
            next_check += 1
    
    # Handle remaining queries (if n_queries not divisible by batch_size)
    remaining_queries = n_queries - n_batches * batch_size
    if remaining_queries > 0:
        x = torch.randn(remaining_queries, C, H, W, device=DEVICE)
        y_pred = api.query(x, logits=True)
        queries.append(x.cpu())
        labels.append(torch.exp(y_pred).cpu())
    
    # Combine all queries and labels into a dataset
    X = torch.cat(queries)   # shape: [n_queries, C, H, W]
    Y = torch.cat(labels)    # shape: [n_queries, num_classes]
    
    return TensorDataset(X, Y)


#----------------------------------
# NORMAL USAGE TESTING
#----------------------------------

def test_normal_usage_accuracy(victim_model, defended_api, test_loader, n_samples=1000):
    """
    Test accuracy impact on normal (legitimate) queries.
    
    This function measures how the defense mechanism affects the accuracy
    for legitimate users by testing the defended API on normal test data
    and comparing predictions with ground truth labels.
    
    Args:
        victim_model: Original undefended model
        defended_api: Defended API wrapper
        test_loader: Test data loader with legitimate samples
        n_samples (int): Number of samples to test
        
    Returns:
        float: Accuracy of defended model on legitimate queries
    """
    correct = 0
    total = 0
    
    victim_model.eval()
    
    with torch.no_grad():
        for data, labels in test_loader:
            # Stop when we've tested enough samples
            if total >= n_samples:
                break
                
            # Process batch (limited by remaining samples needed)
            batch_size = min(len(data), n_samples - total)
            data = data[:batch_size].to(DEVICE)
            labels = labels[:batch_size]
            
            # Get defended predictions
            defended_output = defended_api.query(data, logits=True)
            defended_pred = defended_output.argmax(dim=1).cpu()
            
            # Count correct predictions
            correct += (defended_pred == labels).sum().item()
            total += batch_size
    
    return correct / total


#----------------------------------
# MAIN TESTING ORCHESTRATION
#----------------------------------

def run_comprehensive_defense_test_fixed():
    """
    Execute comprehensive defense testing with multiple configurations.
    
    This is the main testing function that orchestrates a complete evaluation
    of the defense mechanism. It tests multiple defense configurations with
    varying aggressiveness levels and provides detailed analysis of:
    
    1. Defense effectiveness against extraction attacks
    2. Impact on normal user experience
    3. Statistical analysis of output modifications
    4. Trade-off analysis between security and usability
    
    The function saves detailed results to JSON for further analysis.
    """
    
    print("\n" + "="*80)
    print("FIXED COMPREHENSIVE DEFENSE TESTING")
    print("="*80)
    
    # First run direct activation test to verify defense is working
    test_defense_activation_directly()
    
    # Setup models and data loaders
    victim = load_victim()
    train_loader, test_loader = get_loaders(CFG["batch_size"])
    
    # Define test configurations with increasing defense strength
    test_configs = {
        'no_defense': {
            # Essentially disable all defenses for baseline comparison
            "baseline_suspicion": 0.0,    # No baseline suspicion
            "block_threshold": 1.0,       # Never block queries
            "perturb_threshold": 1.0,     # Never perturb responses
            "top_k": 100,                 # No controlled release (all logits)
            "base_noise_scale": 0.0,      # No noise injection
        },
        'light_defense': {
            # Light defense that should trigger occasionally
            "baseline_suspicion": 0.15,   # Low baseline suspicion
            "block_threshold": 0.85,      # High blocking threshold
            "perturb_threshold": 0.70,    # High perturbation threshold
            "top_k": 5,                   # Release top-5 logits
            "base_noise_scale": 0.01,     # Minimal noise
            "ood_threshold": 0.80,        # High OOD threshold
        },
        'moderate_defense': {
            # Moderate defense - should trigger more often
            "baseline_suspicion": 0.25,   # Medium baseline suspicion
            "block_threshold": 0.70,      # Medium blocking threshold
            "perturb_threshold": 0.50,    # Medium perturbation threshold
            "top_k": 3,                   # Release top-3 logits
            "base_noise_scale": 0.03,     # Moderate base noise
            "max_noise_scale": 0.20,      # Moderate max noise
            "ood_threshold": 0.70,        # Medium OOD threshold
            "temperature_base": 1.8,      # Temperature scaling
        },
        'aggressive_defense': {
            # Aggressive defense - should trigger frequently
            "baseline_suspicion": 0.35,   # High baseline suspicion
            "block_threshold": 0.60,      # Low blocking threshold
            "perturb_threshold": 0.30,    # Low perturbation threshold
            "top_k": 2,                   # Release only top-2 logits
            "base_noise_scale": 0.05,     # High base noise
            "max_noise_scale": 0.30,      # High max noise
            "ood_threshold": 0.60,        # Low OOD threshold
            "temperature_base": 2.5,      # Strong temperature scaling
        }
    }
    
    # Store results for all configurations
    all_results = {}
    
    # Test each configuration with error handling
    for config_name, config in test_configs.items():
        try:
            print(f"\n--- Testing Configuration: {config_name} ---")
            results = test_single_defense_configuration_fixed(
                victim, 
                train_loader, 
                test_loader,
                config, 
                config_name,
                n_queries=2000  # Reduced for debugging efficiency
            )
            all_results[config_name] = results
            
        except Exception as e:
            print(f"Error testing {config_name}: {e}")
            import traceback
            traceback.print_exc()
    
    #--- COMPREHENSIVE RESULTS SUMMARY ---
    print("\n" + "="*80)
    print("FIXED DEFENSE TESTING SUMMARY")
    print("="*80)
    
    # Create formatted summary table
    print("\n{:<18} | {:>8} | {:>8} | {:>8} | {:>8} | {:>10} | {:>8}".format(
        "Configuration", "Surr.Acc", "Agreemnt", "Norm.Acc", "Effectiv", "KL-Div", "L1-Diff"
    ))
    print("-" * 90)
    
    # Display results for each configuration
    for config_name, results in all_results.items():
        surr_acc = results['defended']['accuracy'] * 100
        agreement = results['defended']['agreement'] * 100
        normal_acc = results['normal_usage']['accuracy'] * 100
        effectiveness = results['effectiveness']['score']
        kl_div = results['output_analysis']['kl_divergence']
        l1_diff = results['output_analysis']['l1_difference']
        
        print("{:<18} | {:>8.2f}% | {:>8.2f}% | {:>8.2f}% | {:>8.2f}% | {:>10.6f} | {:>8.6f}".format(
            config_name, surr_acc, agreement, normal_acc, effectiveness, kl_div, l1_diff
        ))
    
    # Save comprehensive results to file
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'victim_accuracy': float(accuracy(victim, test_loader)),
        'configurations': test_configs,
        'results': all_results
    }
    
    with open('defense_test_results_fixed.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nDetailed results saved to defense_test_results_fixed.json")
    
    return all_results


#----------------------------------
# COMMAND LINE INTERFACE
#----------------------------------

if __name__ == "__main__":
    import argparse
    
    # Command line argument parser
    parser = argparse.ArgumentParser(
        description="Comprehensive Defense Testing Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Testing Modes:
  Default:     Run comprehensive defense evaluation with all configurations
  --debug:     Run debug test to verify defense activation  
  --direct:    Run direct activation test only with aggressive settings

Examples:
  python test_def2.py                    # Full comprehensive testing
  python test_def2.py --debug           # Debug defense activation
  python test_def2.py --direct          # Direct activation test only
        """
    )
    
    parser.add_argument("--debug", action="store_true", 
                       help="Run debug test to check defense activation")
    parser.add_argument("--direct", action="store_true",
                       help="Run direct activation test only")
    
    args = parser.parse_args()
    
    # Execute based on command line arguments
    if args.debug:
        print("Running defense activation debug test...")
        test_defense_activation_directly()
    elif args.direct:
        print("Running direct activation test...")
        test_defense_activation_directly()
    else:
        print("Running comprehensive defense testing...")
        run_comprehensive_defense_test_fixed()