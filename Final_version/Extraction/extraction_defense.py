# enhanced_defense_evaluation.py
"""
Enhanced Defense Evaluation with Normal Usage Impact Assessment
==============================================================
Evaluates both attack prevention effectiveness and impact on legitimate users.

This module provides comprehensive evaluation tools for defense mechanisms against
model extraction attacks, including:
- Attack prevention effectiveness measurement
- Normal usage impact assessment
- Defense transparency testing
- Trade-off analysis between security and usability

Author: Generated for ML Security Research
License: MIT
"""

#----------------------------------
# IMPORTS AND DEPENDENCIES
#----------------------------------

import json
import time
import torch
import numpy as np
from typing import Dict, List, Tuple

# Import from extraction_attack - make sure all needed functions are imported
from extraction_attack import (
    CFG, get_loaders, load_victim, BlackBoxAPI,
    build_query_set, train_surrogate, accuracy, agreement,
    SurrogateNet, DEVICE
)

from defense_mechanism import DefendedBlackBoxAPI, DefenseMechanism

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#----------------------------------
# NORMAL USAGE IMPACT EVALUATION
#----------------------------------

def evaluate_normal_usage_impact(victim_model, defended_api, test_loader, num_samples=1000):
    """
    Evaluate the impact of defense mechanism on normal/legitimate queries.
    
    This function measures how defense mechanisms affect the performance of legitimate
    users by comparing predictions from defended and undefended models on normal test data.
    
    Args:
        victim_model (torch.nn.Module): Original undefended model
        defended_api (DefendedBlackBoxAPI): Defended API wrapper
        test_loader (torch.utils.data.DataLoader): Test data loader for evaluation
        num_samples (int, optional): Number of samples to test. Defaults to 1000.
        
    Returns:
        Dict: Dictionary containing evaluation metrics including:
            - undefended_accuracy: Accuracy of original model
            - defended_accuracy: Accuracy of defended model
            - accuracy_drop: Difference in accuracy (undefended - defended)
            - prediction_agreement: Rate of agreement between models
            - class_impacts: Per-class impact analysis
            - samples_tested: Actual number of samples evaluated
    """
    print("\n[Normal Usage Evaluation] Testing impact on legitimate queries...")
    
    # Get device from victim model
    device = next(victim_model.parameters()).device
    
    # Initialize result containers
    undefended_predictions = []
    defended_predictions = []
    true_labels = []
    
    sample_count = 0
    victim_model.eval()  # Set to evaluation mode
    
    # Create undefended API for fair comparison
    undefended_api = BlackBoxAPI(victim_model)
    
    # Evaluate models without gradient computation (faster)
    with torch.no_grad():
        for data, labels in test_loader:
            # Check if we've reached the desired number of samples
            if sample_count >= num_samples:
                break
                
            # Determine batch size for this iteration
            batch_size = min(len(data), num_samples - sample_count)
            data = data[:batch_size].to(device)
            labels = labels[:batch_size]
            
            # Get undefended predictions (using BlackBoxAPI for consistency)
            undefended_output = undefended_api.query(data, logits=True)
            undefended_pred = undefended_output.argmax(dim=1).cpu()
            
            # Get defended predictions (through defended API)
            defended_output = defended_api.query(data, logits=True)
            defended_pred = defended_output.argmax(dim=1).cpu()
            
            # Store predictions and true labels
            undefended_predictions.extend(undefended_pred.numpy())
            defended_predictions.extend(defended_pred.numpy())
            true_labels.extend(labels.numpy())
            
            sample_count += batch_size
    
    # Convert lists to numpy arrays for easier computation
    undefended_predictions = np.array(undefended_predictions)
    defended_predictions = np.array(defended_predictions)
    true_labels = np.array(true_labels)
    
    # Calculate overall accuracy metrics
    undefended_accuracy = (undefended_predictions == true_labels).mean()
    defended_accuracy = (defended_predictions == true_labels).mean()
    
    # Calculate agreement between defended and undefended models
    prediction_agreement = (undefended_predictions == defended_predictions).mean()
    
    # Calculate accuracy drop due to defense
    accuracy_drop = undefended_accuracy - defended_accuracy
    
    # Debug output for verification
    print(f"  Debug - Undefended correct: {(undefended_predictions == true_labels).sum()}/{len(true_labels)}")
    print(f"  Debug - Defended correct: {(defended_predictions == true_labels).sum()}/{len(true_labels)}")
    print(f"  Debug - Predictions differ: {(undefended_predictions != defended_predictions).sum()}")
    
    # Per-class impact analysis (assuming 10 classes for CIFAR-10)
    class_impacts = {}
    for class_id in range(10):
        class_mask = true_labels == class_id
        if class_mask.sum() > 0:  # Only analyze classes that exist in the sample
            undefended_class_acc = (undefended_predictions[class_mask] == true_labels[class_mask]).mean()
            defended_class_acc = (defended_predictions[class_mask] == true_labels[class_mask]).mean()
            class_impacts[class_id] = {
                'undefended_acc': float(undefended_class_acc),
                'defended_acc': float(defended_class_acc),
                'accuracy_drop': float(undefended_class_acc - defended_class_acc)
            }
    
    # Return comprehensive evaluation results
    return {
        'undefended_accuracy': float(undefended_accuracy),
        'defended_accuracy': float(defended_accuracy),
        'accuracy_drop': float(accuracy_drop),
        'prediction_agreement': float(prediction_agreement),
        'class_impacts': class_impacts,
        'samples_tested': sample_count
    }


#----------------------------------
# COMPREHENSIVE DEFENSE EVALUATION
#----------------------------------

def comprehensive_defense_evaluation_with_normal_usage():
    """
    Enhanced evaluation including normal usage impact analysis.
    
    This function performs a comprehensive evaluation of defense mechanisms by testing
    multiple defense configurations and measuring both their effectiveness against
    attacks and their impact on legitimate users.
    
    The evaluation includes:
    1. Baseline (no defense) performance measurement
    2. Multiple defense configurations with varying strength levels
    3. Normal usage impact assessment for each configuration
    4. Attack prevention effectiveness testing
    5. Trade-off analysis between security and usability
    
    Returns:
        Dict: Complete evaluation results for all tested configurations
    """
    
    # Setup model and data loaders
    device = torch.device(CFG["device"])
    victim = load_victim()
    train_loader, test_loader = get_loaders(CFG["batch_size"])
    
    # Define defense configurations with increasing strength levels
    defense_configs = {
        'baseline': None,  # No defense for comparison
        'low': {
            'base_noise_scale': 0.005,      # Minimal noise injection
            'max_noise_scale': 0.05,        # Low maximum noise
            'block_threshold': 0.9,         # High threshold (less blocking)
            'ood_threshold': 0.9,           # High OOD threshold
            'perturb_threshold': 0.4,       # Moderate perturbation threshold
            'top_k': 10  # Allow all logits for minimal protection
        },
        'medium': {
            'base_noise_scale': 0.01,       # Moderate noise injection
            'max_noise_scale': 0.1,         # Moderate maximum noise
            'block_threshold': 0.7,         # Medium threshold
            'ood_threshold': 0.85,          # Medium OOD threshold
            'perturb_threshold': 0.3,       # Lower perturbation threshold
            'top_k': 5  # Restrict to top-5 logits
        },
        'high': {
            'base_noise_scale': 0.02,       # High noise injection
            'max_noise_scale': 0.2,         # High maximum noise
            'block_threshold': 0.5,         # Low threshold (more blocking)
            'ood_threshold': 0.8,           # Lower OOD threshold
            'deception_probability': 0.4,   # Add deception mechanism
            'perturb_threshold': 0.2,       # Low perturbation threshold
            'top_k': 1  # Only top-1 logit (maximum protection)
        },
        'adaptive': {
            'base_noise_scale': 0.01,       # Adaptive configuration
            'max_noise_scale': 0.15,        # Balanced noise levels
            'block_threshold': 0.6,         # Balanced threshold
            'ood_threshold': 0.85,          # Balanced OOD threshold
            'perturb_threshold': 0.25,      # Balanced perturbation threshold
            'top_k': 3  # Top-3 logits for balanced protection
        }
    }
    
    results = {}
    
    # Get baseline victim accuracy for reference
    print("\n[Baseline Evaluation]")
    victim_accuracy = accuracy(victim, test_loader)
    print(f"Original Victim Model Accuracy: {victim_accuracy*100:.2f}%")
    
    # Test each defense configuration
    for config_name, config in defense_configs.items():
        print(f"\n{'='*60}")
        print(f"Testing defense configuration: {config_name.upper()}")
        print('='*60)
        
        # Handle baseline case (no defense)
        if config_name == 'baseline':
            # For baseline, just run undefended attack
            api = BlackBoxAPI(victim)
            
            print(f"\nAttack Test (No Defense):")
            n_queries = 5000
            
            # Run extraction attack to establish baseline
            t_start = time.perf_counter()
            qset = build_query_set(api, n_queries)
            surrogate = train_surrogate(qset)
            attack_time = time.perf_counter() - t_start
            
            # Evaluate baseline attack success
            surrogate_acc = accuracy(surrogate, test_loader)
            agr_rate = agreement(victim, surrogate, test_loader)
            
            print(f"  Surrogate Accuracy: {surrogate_acc*100:.2f}%")
            print(f"  Agreement Rate: {agr_rate*100:.2f}%")
            
            # Store baseline results
            results[config_name] = {
                'normal_usage': {
                    'undefended_accuracy': float(victim_accuracy),
                    'defended_accuracy': float(victim_accuracy),
                    'accuracy_drop': 0.0,
                    'prediction_agreement': 1.0
                },
                'attack_prevention': {
                    'surrogate_accuracy': float(surrogate_acc),
                    'agreement_rate': float(agr_rate),
                    'attack_time': attack_time
                }
            }
            continue
            
        # Create defended API with current configuration
        defended_api = DefendedBlackBoxAPI(victim, config)
        defended_api.fit_defense(train_loader)
        
        # 1. Normal usage impact evaluation
        print(f"\n--- Normal Usage Impact Assessment ---")
        normal_usage_results = evaluate_normal_usage_impact(
            victim, defended_api, test_loader, num_samples=2000
        )
        
        print(f"\nNormal Usage Impact:")
        print(f"  Original Accuracy: {normal_usage_results['undefended_accuracy']*100:.2f}%")
        print(f"  Defended Accuracy: {normal_usage_results['defended_accuracy']*100:.2f}%")
        print(f"  Accuracy Drop: {normal_usage_results['accuracy_drop']*100:.2f}%")
        print(f"  Prediction Agreement: {normal_usage_results['prediction_agreement']*100:.2f}%")
        
        # 2. Attack prevention evaluation
        print(f"\n--- Attack Prevention Assessment ---")
        print(f"Attack Prevention Test:")
        n_queries = 5000
        
        # Run extraction attack against defended API
        t_start = time.perf_counter()
        qset = build_query_set(defended_api, n_queries)
        surrogate = train_surrogate(qset)
        attack_time = time.perf_counter() - t_start
        
        # Evaluate attack success against defended model
        surrogate_acc = accuracy(surrogate, test_loader)
        agr_rate = agreement(victim, surrogate, test_loader)
        
        print(f"  Surrogate Accuracy: {surrogate_acc*100:.2f}%")
        print(f"  Agreement Rate: {agr_rate*100:.2f}%")
        
        # Store comprehensive results for this configuration
        results[config_name] = {
            'normal_usage': normal_usage_results,
            'attack_prevention': {
                'surrogate_accuracy': float(surrogate_acc),
                'agreement_rate': float(agr_rate),
                'attack_time': attack_time
            },
            'defense_report': defended_api.get_defense_report()
        }
    
    #----------------------------------
    # RESULTS ANALYSIS AND SUMMARY
    #----------------------------------
    
    print("\n" + "="*60)
    print("DEFENSE EVALUATION SUMMARY")
    print("="*60)
    
    # Get baseline results for comparison
    baseline_results = results.get('baseline', {})
    if baseline_results:
        baseline_acc = baseline_results['attack_prevention']['surrogate_accuracy']
        baseline_agr = baseline_results['attack_prevention']['agreement_rate']
    else:
        baseline_acc = victim_accuracy
        baseline_agr = 1.0
    
    # Trade-off analysis table
    print("\nDefense Trade-off Analysis:")
    print("Config    | Normal Acc Drop | Surrogate Acc | Agreement | Trade-off Score")
    print("-" * 75)
    
    for config_name in ['low', 'medium', 'high', 'adaptive']:
        if config_name in results:
            normal_drop = results[config_name]['normal_usage']['accuracy_drop'] * 100
            surrogate_acc = results[config_name]['attack_prevention']['surrogate_accuracy'] * 100
            agreement_rate = results[config_name]['attack_prevention']['agreement_rate'] * 100
            
            # Trade-off score calculation:
            # Higher score is better (lower surrogate accuracy is good, lower normal drop is good)
            # Penalize normal usage impact more heavily (factor of 2)
            trade_off_score = (100 - surrogate_acc) - (2 * normal_drop)
            
            print(f"{config_name:9} | {normal_drop:15.2f}% | {surrogate_acc:13.2f}% | {agreement_rate:9.2f}% | {trade_off_score:15.2f}")
    
    # Save comprehensive results to file
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'victim_accuracy': float(victim_accuracy),
        'configurations': defense_configs,
        'results': results
    }
    
    with open('enhanced_defense_evaluation.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nDetailed results saved to enhanced_defense_evaluation.json")
    
    return results


#----------------------------------
# DEFENSE TRANSPARENCY TESTING
#----------------------------------

def test_defense_transparency():
    """
    Test how transparent the defense is to normal users.
    
    This function evaluates the consistency of defense mechanisms by running
    multiple identical legitimate queries and checking if the results are
    consistent. A good defense should be deterministic for legitimate users
    while still protecting against attacks.
    
    The test measures:
    - Query consistency: Whether identical queries return identical results
    - Defense determinism: How predictable the defense is for normal usage
    - User experience impact: Whether defense introduces unwanted randomness
    
    Returns:
        float: Consistency rate (0.0 to 1.0, where 1.0 is perfectly consistent)
    """
    print("\n" + "="*60)
    print("DEFENSE TRANSPARENCY TEST")
    print("="*60)
    
    # Setup model and data
    device = torch.device(CFG["device"])
    victim = load_victim()
    train_loader, test_loader = get_loaders(CFG["batch_size"])
    
    # Test configuration (medium strength for balanced testing)
    test_config = {
        'base_noise_scale': 0.01,
        'max_noise_scale': 0.1,
        'block_threshold': 0.7,
        'ood_threshold': 0.85
    }
    
    # Create defended API
    defended_api = DefendedBlackBoxAPI(victim, test_config)
    defended_api.fit_defense(train_loader)
    
    print("\nTesting query consistency for legitimate users...")
    
    # Test consistency across multiple identical queries
    consistency_results = []
    
    for data, labels in test_loader:
        # Limit to 10 test batches for efficiency
        if len(consistency_results) >= 10:
            break
            
        # Use first 10 samples from each batch
        data = data[:10].to(device)
        labels = labels[:10]
        
        # Query the same data 5 times to test consistency
        predictions_list = []
        for i in range(5):
            output = defended_api.query(data, logits=True)
            pred = output.argmax(dim=1).cpu()
            predictions_list.append(pred)
        
        # Check if all predictions are identical (consistency check)
        first_pred = predictions_list[0]
        consistency = all((pred == first_pred).all() for pred in predictions_list[1:])
        consistency_results.append(consistency)
        
        # Log any inconsistencies for debugging
        if not consistency:
            print(f"  Inconsistency detected in batch {len(consistency_results)}")
    
    # Calculate overall consistency rate
    consistency_rate = sum(consistency_results) / len(consistency_results)
    print(f"\nQuery Consistency Rate: {consistency_rate*100:.2f}%")
    print("(100% means defense is deterministic for legitimate queries)")
    
    return consistency_rate


#----------------------------------
# MAIN EXECUTION AND CLI INTERFACE
#----------------------------------

if __name__ == "__main__":
    import argparse
    
    # Command line argument parser
    parser = argparse.ArgumentParser(
        description="Enhanced Defense Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhanced_defense_evaluation.py                    # Run default evaluation
  python enhanced_defense_evaluation.py --full            # Run full evaluation with normal usage impact
  python enhanced_defense_evaluation.py --transparency    # Test defense transparency only
        """
    )
    
    parser.add_argument("--full", action="store_true", 
                       help="Run full evaluation with normal usage impact assessment")
    parser.add_argument("--transparency", action="store_true",
                       help="Test defense transparency and consistency")
    
    args = parser.parse_args()
    
    # Execute based on command line arguments
    if args.transparency:
        print("Testing defense transparency...")
        test_defense_transparency()
    elif args.full:
        print("Running comprehensive defense evaluation...")
        comprehensive_defense_evaluation_with_normal_usage()
    else:
        # Run enhanced evaluation by default
        print("Running enhanced defense evaluation...")
        comprehensive_defense_evaluation_with_normal_usage()