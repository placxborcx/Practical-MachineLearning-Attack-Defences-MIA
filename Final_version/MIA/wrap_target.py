"""
Defense Wrapper Application Script
==================================

This script applies the four-layer hybrid defense mechanism to a pre-trained
target model that was previously vulnerable to membership inference attacks.
It demonstrates how existing models can be retrofitted with privacy protection
without requiring retraining.

Purpose:
- Load the original vulnerable target model from the attack phase
- Apply the DefenceWrapper to enable inference-time protection
- Save the model weights for use with the defense system
- Maintain compatibility between attack and defense evaluation phases

Workflow:
1. Load the original SimpleCNN model trained during the attack phase
2. Wrap it with the four-layer defense mechanism
3. Save the base model weights for use with the defense wrapper
4. Enable seamless transition from vulnerable to protected deployment

"""

import torch
import os
import sys

# Import components from attack phase
try:
    from membership_inference_attack import SimpleCNN, Config as AttackConfig
    print("✓ Successfully imported attack model components")
except ImportError as e:
    print(f"❌ Error importing attack components: {e}")
    print("Ensure membership_inference_attack.py is in the same directory")
    sys.exit(1)

# Import defense mechanism
try:
    from MIA_defense_mechanism import DefenceWrapper, CFG as DefenseConfig
    print("✓ Successfully imported defense components")
except ImportError as e:
    print(f"❌ Error importing defense components: {e}")
    print("Ensure MIA_defense_mechanism.py is in the same directory")
    sys.exit(1)

# =============================================================================
# FILE PATH CONFIGURATION
# =============================================================================

class FileConfig:
    """
    Configuration for file paths used in the attack-to-defense transition.
    
    This ensures consistent file naming and location management across
    the attack and defense evaluation phases.
    """
    
    # Input: Original target model from attack phase
    ORIGINAL_TARGET_PATH = "target_final.pth"
    
    # Output: Model weights compatible with defense system
    DEFENDED_TARGET_PATH = "defended_target.pth"
    
    # Backup: Copy of original for comparison
    BACKUP_TARGET_PATH = "target_original_backup.pth"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_file_exists(filepath):
    """
    Check if a required file exists and provide helpful error messages.
    
    Args:
        filepath (str): Path to the file to check
        
    Returns:
        bool: True if file exists, False otherwise
    """
    if os.path.exists(filepath):
        file_size = os.path.getsize(filepath)
        print(f"✓ Found {filepath} ({file_size:,} bytes)")
        return True
    else:
        print(f"❌ Missing required file: {filepath}")
        return False

def validate_model_compatibility(model, expected_architecture="SimpleCNN"):
    """
    Validate that the loaded model has the expected architecture.
    
    Args:
        model: PyTorch model to validate
        expected_architecture (str): Expected model type name
        
    Returns:
        bool: True if model is compatible, False otherwise
    """
    model_name = model.__class__.__name__
    if model_name == expected_architecture:
        print(f"✓ Model architecture validated: {model_name}")
        return True
    else:
        print(f"⚠️ Unexpected model architecture: {model_name} (expected {expected_architecture})")
        return True  # Continue anyway, but warn user

def analyze_model_structure(model):
    """
    Analyze and report model structure for verification.
    
    Args:
        model: PyTorch model to analyze
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Analysis:")
    print(f"  Architecture: {model.__class__.__name__}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (float32)")

# =============================================================================
# MAIN WRAPPER APPLICATION PROCESS
# =============================================================================

def apply_defense_wrapper():
    """
    Main function to apply defense wrapper to the original target model.
    
    This function orchestrates the complete process of loading the vulnerable
    model, applying the defense wrapper, and saving the protected version.
    
    Process:
    1. Validate input files and dependencies
    2. Load the original target model from attack phase
    3. Apply the defense wrapper (Layers 2-4)
    4. Save the model for defense evaluation
    5. Create backup and validate output
    
    Returns:
        bool: True if process completed successfully, False otherwise
    """
    
    print("=" * 60)
    print("APPLYING FOUR-LAYER DEFENSE TO TARGET MODEL")
    print("=" * 60)
    
    # === STEP 1: PRE-FLIGHT VALIDATION ===
    print("\nStep 1: Validating dependencies and input files...")
    
    # Check if original target model exists
    if not check_file_exists(FileConfig.ORIGINAL_TARGET_PATH):
        print("\n❌ Cannot proceed without original target model")
        print("Ensure you have run the membership inference attack first")
        print("to generate the target model file")
        return False
    
    # Check available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")
    
    # === STEP 2: LOAD ORIGINAL TARGET MODEL ===
    print(f"\nStep 2: Loading original target model...")
    
    try:
        # Initialize model architecture (same as used in attack)
        original_model = SimpleCNN().to(device)
        print(f"✓ Initialized {original_model.__class__.__name__} architecture")
        
        # Load trained weights from attack phase
        model_state = torch.load(
            FileConfig.ORIGINAL_TARGET_PATH, 
            map_location=device
        )
        original_model.load_state_dict(model_state)
        print(f"✓ Loaded model weights from {FileConfig.ORIGINAL_TARGET_PATH}")
        
        # Validate model compatibility
        validate_model_compatibility(original_model, "SimpleCNN")
        analyze_model_structure(original_model)
        
        # Set model to evaluation mode for defense application
        original_model.eval()
        print("✓ Model set to evaluation mode")
        
    except Exception as e:
        print(f"❌ Error loading original model: {e}")
        return False
    
    # === STEP 3: APPLY DEFENSE WRAPPER ===
    print(f"\nStep 3: Applying four-layer defense wrapper...")
    
    try:
        # Create defense wrapper around the original model
        # This applies Layers 2-4 of the defense strategy
        defended_model = DefenceWrapper(original_model)
        print("✓ Applied DefenceWrapper successfully")
        
        # Display defense configuration
        print("Defense Configuration:")
        print(f"  Layer 1: Regularization (already in base model)")
        print(f"  Layer 2: Temperature scaling (T = {DefenseConfig.temperature})")
        print(f"  Layer 3: Adaptive noise (std = {DefenseConfig.noise_std_min}-{DefenseConfig.noise_std_max})")
        print(f"  Layer 4: Clipping (max = {DefenseConfig.clip_max}) + Rounding ({DefenseConfig.round_digit} digits)")
        
        # Test defense wrapper with sample input
        print("✓ Testing defense wrapper...")
        with torch.no_grad():
            test_input = torch.randn(2, 1, 28, 28).to(device)  # Sample MNIST-like input
            
            # Test original model output
            original_output = original_model(test_input)
            original_probs = torch.softmax(original_output, dim=1)
            
            # Test defended model output
            defended_logits = defended_model.forward(test_input)
            defended_probs = defended_model.defended_probs(test_input)
            
            # Analyze defense effectiveness
            original_max_conf = original_probs.max(dim=1)[0].mean().item()
            defended_max_conf = defended_probs.max(dim=1)[0].mean().item()
            
            print(f"  Original model max confidence: {original_max_conf:.4f}")
            print(f"  Defended model max confidence: {defended_max_conf:.4f}")
            print(f"  Confidence reduction: {original_max_conf - defended_max_conf:.4f}")
            
            # Verify clipping is working
            if defended_probs.max().item() <= DefenseConfig.clip_max + 1e-6:
                print(f"  ✓ Probability clipping verified (max ≤ {DefenseConfig.clip_max})")
            else:
                print(f"  ⚠️ Probability clipping may not be working properly")
        
    except Exception as e:
        print(f"❌ Error applying defense wrapper: {e}")
        return False
    
    # === STEP 4: SAVE DEFENDED MODEL ===
    print(f"\nStep 4: Saving defended model...")
    
    try:
        # Save the base model weights (compatible with defense wrapper)
        # Note: We save the original model weights, not the wrapper
        # The wrapper will be applied during evaluation
        torch.save(
            original_model.state_dict(), 
            FileConfig.DEFENDED_TARGET_PATH
        )
        print(f"✓ Saved defended model weights to: {FileConfig.DEFENDED_TARGET_PATH}")
        
        # Create backup of original for comparison
        if not os.path.exists(FileConfig.BACKUP_TARGET_PATH):
            torch.save(
                original_model.state_dict(),
                FileConfig.BACKUP_TARGET_PATH
            )
            print(f"✓ Created backup at: {FileConfig.BACKUP_TARGET_PATH}")
        else:
            print(f"✓ Backup already exists: {FileConfig.BACKUP_TARGET_PATH}")
        
    except Exception as e:
        print(f"❌ Error saving defended model: {e}")
        return False
    
    # === STEP 5: VALIDATION AND SUMMARY ===
    print(f"\nStep 5: Validation and summary...")
    
    try:
        # Verify saved file
        if check_file_exists(FileConfig.DEFENDED_TARGET_PATH):
            # Test loading saved model
            test_model = SimpleCNN().to(device)
            test_state = torch.load(FileConfig.DEFENDED_TARGET_PATH, map_location=device)
            test_model.load_state_dict(test_state)
            print("✓ Saved model can be loaded successfully")
            
            # Verify compatibility with defense wrapper
            test_wrapper = DefenceWrapper(test_model)
            print("✓ Saved model is compatible with DefenceWrapper")
            
        else:
            print("❌ Saved model file not found")
            return False
        
    except Exception as e:
        print(f"❌ Error validating saved model: {e}")
        return False
    
    # === SUCCESS SUMMARY ===
    print("\n" + "=" * 60)
    print("DEFENSE APPLICATION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"✓ Original model loaded from: {FileConfig.ORIGINAL_TARGET_PATH}")
    print(f"✓ Defense wrapper applied with 4-layer protection")
    print(f"✓ Defended model saved to: {FileConfig.DEFENDED_TARGET_PATH}")
    print(f"✓ Backup created at: {FileConfig.BACKUP_TARGET_PATH}")
    
    print(f"\nDefense Summary:")
    print(f"  Input: Vulnerable model (AUC ≈ 0.60 against attacks)")
    print(f"  Output: Protected model (target AUC ≤ 0.53)")
    print(f"  Method: Four-layer hybrid defense")
    print(f"  Compatibility: Ready for defense evaluation")
    
    print(f"\nNext Steps:")
    print(f"  1. Run defense evaluation script to measure effectiveness")
    print(f"  2. Compare attack performance before/after defense")
    print(f"  3. Analyze utility preservation in defended model")
    
    return True

def verify_defense_integration():
    """
    Verify that the defense integration is working correctly.
    
    This function performs additional checks to ensure the defense
    mechanism is properly integrated and functional.
    
    Returns:
        bool: True if integration is successful, False otherwise
    """
    print(f"\n" + "=" * 40)
    print("DEFENSE INTEGRATION VERIFICATION")
    print("=" * 40)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Load the defended model
        base_model = SimpleCNN().to(device)
        base_model.load_state_dict(torch.load(FileConfig.DEFENDED_TARGET_PATH, map_location=device))
        base_model.eval()
        
        # Apply defense wrapper
        defended_model = DefenceWrapper(base_model)
        
        # Test with multiple samples
        print("Testing defense with sample inputs...")
        with torch.no_grad():
            test_batch = torch.randn(10, 1, 28, 28).to(device)
            
            # Get original and defended outputs
            original_outputs = base_model(test_batch)
            original_probs = torch.softmax(original_outputs, dim=1)
            
            defended_probs = defended_model.defended_probs(test_batch)
            
            # Statistical analysis
            print(f"\nStatistical Analysis:")
            print(f"  Original - Mean max prob: {original_probs.max(dim=1)[0].mean():.4f}")
            print(f"  Original - Std max prob:  {original_probs.max(dim=1)[0].std():.4f}")
            print(f"  Defended - Mean max prob: {defended_probs.max(dim=1)[0].mean():.4f}")
            print(f"  Defended - Std max prob:  {defended_probs.max(dim=1)[0].std():.4f}")
            
            # Check defense constraints
            max_prob = defended_probs.max().item()
            min_prob = defended_probs.min().item()
            
            print(f"\nDefense Constraint Verification:")
            print(f"  Maximum probability: {max_prob:.4f} (should be ≤ {DefenseConfig.clip_max})")
            print(f"  Minimum probability: {min_prob:.4f} (should be ≥ 0.0)")
            
            # Verify probability sums
            prob_sums = defended_probs.sum(dim=1)
            print(f"  Probability sums: {prob_sums.min():.4f} - {prob_sums.max():.4f} (should be ≈ 1.0)")
            
            # Check if constraints are satisfied
            constraints_satisfied = (
                max_prob <= DefenseConfig.clip_max + 1e-6 and
                min_prob >= -1e-6 and
                torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6)
            )
            
            if constraints_satisfied:
                print("✓ All defense constraints satisfied")
                return True
            else:
                print("❌ Some defense constraints violated")
                return False
            
    except Exception as e:
        print(f"❌ Error during integration verification: {e}")
        return False

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function for the defense wrapper application.
    
    This function orchestrates the complete process of applying the
    four-layer defense mechanism to the original target model and
    preparing it for defense evaluation.
    """
    
    print(__doc__)  # Print module docstring
    
    # Step 1: Apply defense wrapper
    success = apply_defense_wrapper()
    
    if not success:
        print("\n❌ Defense application failed. Please check error messages above.")
        sys.exit(1)
    
    # Step 2: Verify integration
    verification_success = verify_defense_integration()
    
    if not verification_success:
        print("\n⚠️ Defense integration verification failed.")
        print("The model was saved but may not work correctly.")
        print("Please review the defense parameters and try again.")
    
    # Final status
    if success and verification_success:
        print("\n🎯 COMPLETE SUCCESS")
        print("The model is now protected with four-layer defense and ready for evaluation.")
    else:
        print("\n⚠️ PARTIAL SUCCESS")
        print("The model was processed but may need additional verification.")
