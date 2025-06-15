# defense_mechanism_fixed.py
"""
Fixed Defense Mechanism Against Model-Extraction Attacks
========================================================

This module implements a comprehensive multi-layer defense system against model
extraction attacks. The defense combines multiple detection and mitigation techniques:

Key Features:
- Out-of-distribution (OOD) query detection
- Query pattern analysis and duplicate detection
- Suspicion scoring with multiple signals
- Controlled output release with noise injection
- Deceptive perturbation based on threat level
- Rate limiting and fingerprinting

Key Fixes:
1. Lower baseline suspicion and thresholds for better sensitivity
2. Improved OOD detection using multiple features
3. Better controlled output release
4. Enhanced query pattern analysis

"""

#----------------------------------
# IMPORTS AND DEPENDENCIES
#----------------------------------

from __future__ import annotations
import logging
import hashlib
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import entropy
from sklearn.covariance import EllipticEnvelope

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


#----------------------------------
# CORE DEFENSE MECHANISM CLASS
#----------------------------------

class DefenseMechanism:
    """
    Multi-layer defense wrapper for a victim model against extraction attacks.
    
    This class implements a comprehensive defense system that combines multiple
    detection and mitigation techniques to protect machine learning models from
    extraction attacks while maintaining usability for legitimate users.
    
    Defense Layers:
    1. OOD Detection: Identifies queries from unusual input distributions
    2. Pattern Analysis: Detects repetitive or systematic query patterns
    3. Suspicion Scoring: Combines multiple signals to assess threat level
    4. Controlled Release: Limits information exposure in model outputs
    5. Deceptive Perturbation: Adds noise to responses based on threat level
    """

    def __init__(
        self,
        victim_model: nn.Module,
        device: torch.device,
        defense_config: Optional[Dict] = None,
    ):
        """
        Initialize the defense mechanism with a victim model.
        
        Args:
            victim_model (nn.Module): The target model to protect
            device (torch.device): Device for computation (CPU/GPU)
            defense_config (Optional[Dict]): Custom defense configuration parameters
        """
        self.victim_model = victim_model
        self.device = device

        # Fixed configuration with optimized thresholds for better sensitivity
        self.config: Dict = {
            #--- Controlled Release Configuration ---
            "top_k": 3,                   # Release only top-3 logits with noise
            
            #--- Suspicion Baseline & Detection Thresholds ---
            "baseline_suspicion": 0.15,   # Increased from 0.05 for better sensitivity
            "ood_threshold": 0.70,        # Lowered from 0.85 for more aggressive detection
            "ood_contamination": 0.10,    # Expected proportion of outliers in training
            "entropy_threshold": 2.0,     # Absolute entropy threshold for confidence detection
            "perturb_threshold": 0.25,    # Lowered from 0.40 for earlier perturbation
            "block_threshold": 0.70,      # Increased from 0.60 for more blocking
            
            #--- Noise and Perturbation Scales ---
            "base_noise_scale": 0.02,     # Increased base noise for better protection
            "max_noise_scale": 0.30,      # Increased maximum noise level
            "temperature_base": 1.5,      # Base temperature for scaling logits
            
            #--- Pattern Analysis Window Configuration ---
            "sequence_length": 100,       # Number of queries to remember for pattern detection
            "similarity_threshold": 0.85, # Threshold for detecting similar queries
            "query_rate_window": 10,      # Time window for rate limiting (seconds)
            "max_queries_per_window": 50, # Maximum queries allowed per time window
        }
        
        # Update configuration with any custom parameters
        if defense_config:
            self.config.update(defense_config)

        # Initialize defense components
        self._initialize_components()

        #--- Enhanced Statistics Tracking ---
        self.total_queries = 0          # Total number of queries processed
        self.blocked_queries = 0        # Number of queries blocked due to high suspicion
        self.perturbed_responses = 0    # Number of responses that were perturbed
        
        # Query history for pattern analysis (limited size for memory efficiency)
        self.query_history: deque = deque(maxlen=self.config["sequence_length"])
        
        # Fingerprint tracking for duplicate detection
        self.query_fingerprints: defaultdict[str, int] = defaultdict(int)
        
        # Timestamp tracking for rate limiting
        self.query_timestamps: deque = deque(maxlen=self.config["query_rate_window"])
        
        # Track suspicion scores for analysis and debugging
        self.suspicion_history: deque = deque(maxlen=100)

        logger.info("Fixed Defense mechanism initialized with enhanced sensitivity")

    def _initialize_components(self) -> None:
        """
        Initialize internal defense components.
        
        Sets up the feature extractor for OOD detection and prepares
        placeholders for the OOD detector and training statistics.
        """
        # Create feature extractor for accessing internal model representations
        self.feature_extractor = self._create_feature_extractor()
        
        # Initialize OOD detector (will be fitted during training)
        self.ood_detector: Optional[EllipticEnvelope] = None
        
        # Training data statistics for normalization
        self.training_features_mean = None
        self.training_features_std = None

    def _create_feature_extractor(self) -> nn.Module:
        """
        Create a feature extractor that exposes penultimate layer activations.
        
        This wrapper around the victim model captures the features from the
        last hidden layer before the final classification layer. These features
        are used for OOD detection and other analysis.
        
        Returns:
            nn.Module: Feature extractor that returns both features and predictions
        """

        class _Extractor(nn.Module):
            """Internal feature extractor class."""
            
            def __init__(self, base: nn.Module):
                super().__init__()
                self.base = base
                self._features = None
                self._hook = None

                # Find the last Linear layer for feature extraction
                self._last_linear = None
                for m in reversed(list(base.modules())):
                    if isinstance(m, nn.Linear):
                        self._last_linear = m
                        break

            def forward(self, x):
                """Forward pass that captures features and returns predictions."""
                def hook_fn(_, inp, __):
                    # Capture input to the last linear layer
                    self._features = inp[0] if isinstance(inp, tuple) else inp

                # Register hook to capture features
                if self._hook is not None:
                    self._hook.remove()
                self._hook = self._last_linear.register_forward_hook(hook_fn)

                # Run forward pass
                out = self.base(x)
                
                # Clean up hook
                self._hook.remove()
                
                # Return both features and model output
                return self._features, out

        return _Extractor(self.victim_model).to(self.device)


    #----------------------------------
    # OUT-OF-DISTRIBUTION DETECTION
    #----------------------------------

    @torch.no_grad()
    def fit_ood_detector(self, train_loader) -> None:
        """
        Fit the out-of-distribution (OOD) detector on training data.
        
        This method trains an EllipticEnvelope detector on features extracted
        from legitimate training data. The detector learns the normal distribution
        of features and can identify queries that come from different distributions.
        
        Args:
            train_loader: DataLoader containing legitimate training samples
        """
        feats = []
        self.feature_extractor.eval()
        
        logger.info("Fitting OOD detector on training data...")
        
        # Extract features from training data
        for idx, (x, _) in enumerate(train_loader):
            if idx >= 50:  # Use more samples for better statistics
                break
            f, _ = self.feature_extractor(x.to(self.device))
            feats.append(f.cpu().numpy())
        
        # Concatenate all features
        feats = np.concatenate(feats, axis=0)
        
        # Store training statistics for normalization
        self.training_features_mean = feats.mean(axis=0)
        self.training_features_std = feats.std(axis=0) + 1e-8  # Add epsilon for numerical stability
        
        # Fit the OOD detector using EllipticEnvelope
        self.ood_detector = EllipticEnvelope(
            contamination=self.config["ood_contamination"], 
            random_state=42
        ).fit(feats)
        
        logger.info(f"OOD detector fitted on {len(feats)} training samples")

    @torch.no_grad()
    def _detect_ood(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect out-of-distribution queries using multiple signals.
        
        This method combines multiple approaches to detect OOD queries:
        1. EllipticEnvelope-based detection on extracted features
        2. Distance-based detection from training distribution
        
        Args:
            x (torch.Tensor): Input batch to analyze
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - OOD scores (higher = more likely OOD)
                - Binary OOD predictions (True = OOD)
        """
        # If OOD detector not fitted, return no detections
        if self.ood_detector is None:
            return np.zeros(len(x)), np.zeros(len(x), dtype=bool)

        # Extract features for OOD analysis
        feats, _ = self.feature_extractor(x)
        feats_np = feats.cpu().numpy()
        
        # Standard OOD detection using EllipticEnvelope
        ood_scores = -self.ood_detector.score_samples(feats_np)  # Negative for intuitive scoring
        is_ood = self.ood_detector.predict(feats_np) == -1
        
        # Enhanced detection: distance from training distribution
        if self.training_features_mean is not None:
            # Normalize features using training statistics
            normalized_feats = (feats_np - self.training_features_mean) / self.training_features_std
            
            # Calculate Euclidean distance from origin in normalized space
            distance_scores = np.linalg.norm(normalized_feats, axis=1)
            
            # Combine OOD scores with distance scores (weighted combination)
            ood_scores = 0.7 * ood_scores + 0.3 * (distance_scores / (distance_scores.mean() + 1e-8))
        
        return ood_scores, is_ood


    #----------------------------------
    # QUERY PATTERN ANALYSIS
    #----------------------------------

    def _analyze_query_patterns(self, x: torch.Tensor) -> float:
        """
        Enhanced pattern analysis with multiple behavioral signals.
        
        This method analyzes query patterns to detect systematic extraction attempts.
        It combines multiple signals:
        1. Duplicate detection: Identifies repeated queries
        2. Query diversity: Measures entropy of query patterns
        3. Query rate: Detects unusually high query rates
        
        Args:
            x (torch.Tensor): Input batch to analyze
            
        Returns:
            float: Pattern suspicion score (0.0 = normal, 1.0 = highly suspicious)
        """
        # Generate fingerprints for current queries
        fps = self._generate_fingerprints(x)
        
        # 1. Duplicate Detection Analysis
        # Count how many queries in current batch are duplicates
        dup_count = sum(self.query_fingerprints[fp] > 1 for fp in fps)
        dup_ratio = dup_count / len(fps) if len(fps) > 0 else 0.0
        
        # 2. Query Diversity Analysis (entropy of fingerprints)
        if len(self.query_fingerprints) > 10:
            fp_counts = np.array(list(self.query_fingerprints.values()))
            fp_probs = fp_counts / fp_counts.sum()
            fp_entropy = entropy(fp_probs)
            
            # Low entropy indicates repetitive queries (suspicious)
            max_entropy = np.log(len(self.query_fingerprints) + 1)
            diversity_score = 1.0 - (fp_entropy / max_entropy)
        else:
            diversity_score = 0.0
        
        # Update fingerprint counts for future analysis
        for fp in fps:
            self.query_fingerprints[fp] += 1
        
        # 3. Query Rate Analysis (CPU/GPU compatible)
        # Record timestamp for rate limiting
        if torch.cuda.is_available():
            # GPU: use CUDA events for precise timing
            evt = torch.cuda.Event(enable_timing=True)
            evt.record()
            self.query_timestamps.append(evt)
        else:
            # CPU: use high-resolution timer
            self.query_timestamps.append(time.perf_counter())

        # Check if query rate exceeds threshold
        if len(self.query_timestamps) >= self.config["max_queries_per_window"]:
            rate_score = 0.5  # High rate detected
        else:
            rate_score = 0.0  # Normal rate

        # Combine all pattern signals with weights
        pattern_score = (
            0.4 * dup_ratio +      # Duplicate queries weight
            0.4 * diversity_score + # Low diversity weight  
            0.2 * rate_score       # High rate weight
        )
        
        return min(pattern_score, 1.0)  # Clamp to [0, 1]

    @staticmethod
    def _generate_fingerprints(x: torch.Tensor) -> List[str]:
        """
        Generate compact fingerprints for query duplicate detection.
        
        Creates hash-based fingerprints from input tensors to efficiently
        detect duplicate or near-duplicate queries without storing the
        full input data.
        
        Args:
            x (torch.Tensor): Input batch
            
        Returns:
            List[str]: List of fingerprint strings, one per input sample
        """
        fps = []
        flat = x.cpu().numpy().reshape(len(x), -1)
        
        for row in flat:
            # Aggressive downsampling for better duplicate detection
            # Take every 5th element and round to 1 decimal place
            sample = np.round(row[::5], 1).tobytes()
            
            # Generate shorter hash for efficiency (16 chars sufficient)
            fps.append(hashlib.md5(sample).hexdigest()[:16])
            
        return fps


    #----------------------------------
    # SUSPICION SCORING SYSTEM
    #----------------------------------

    def _calculate_suspicion_score(
        self, x: torch.Tensor, ood_scores: np.ndarray, pattern_score: float
    ) -> float:
        """
        Enhanced suspicion scoring with multiple behavioral signals.
        
        Combines multiple detection signals to produce a comprehensive suspicion
        score that indicates the likelihood of an extraction attack:
        
        1. OOD component: Unusual input distributions
        2. Pattern component: Systematic query patterns  
        3. Entropy component: Suspiciously confident predictions
        4. Uniformity component: Unnaturally uniform confidence levels
        
        Args:
            x (torch.Tensor): Input batch
            ood_scores (np.ndarray): Out-of-distribution scores
            pattern_score (float): Query pattern suspicion score
            
        Returns:
            float: Overall suspicion score (0.0 = normal, 1.0 = highly suspicious)
        """
        # 1. OOD Component (more sensitive threshold)
        ood_detection_rate = (ood_scores > self.config["ood_threshold"]).mean()
        ood_component = ood_detection_rate * 0.3
        
        # 2. Pattern Component  
        pattern_component = pattern_score * 0.3
        
        # 3. Entropy Analysis (sophisticated confidence analysis)
        with torch.no_grad():
            # Get model predictions for entropy analysis
            _, logits = self.feature_extractor(x)
            probs = F.softmax(logits, dim=1)
            
            # Calculate average entropy across batch
            batch_entropy = -(probs * torch.log(probs + 1e-8)).sum(1).mean().item()
            
            # Detect suspiciously low entropy (overconfident predictions)
            if batch_entropy < self.config["entropy_threshold"]:
                entropy_component = 0.2  # Suspicious overconfidence
            else:
                entropy_component = 0.0  # Normal confidence levels
            
            # 4. Confidence Uniformity Analysis (new detection signal)
            # Extract maximum probabilities (confidence scores)
            max_probs = probs.max(dim=1)[0]
            confidence_std = max_probs.std().item()
            
            # Detect unnaturally uniform confidence (sign of systematic querying)
            if confidence_std < 0.1:  # Very low variance in confidence
                uniformity_component = 0.2  # Suspicious uniformity
            else:
                uniformity_component = 0.0  # Normal confidence variation
        
        # Combine all components with baseline suspicion
        total_score = (
            self.config["baseline_suspicion"] +  # Base suspicion level
            ood_component +                       # OOD detection
            pattern_component +                   # Pattern analysis
            entropy_component +                   # Entropy analysis
            uniformity_component                  # Uniformity analysis
        )
        
        # Store score for debugging and analysis
        self.suspicion_history.append(total_score)
        
        # Clamp final score to valid range
        return min(total_score, 1.0)


    #----------------------------------
    # OUTPUT PROTECTION MECHANISMS
    #----------------------------------

    def _controlled_release(self, outputs: torch.Tensor, k: int) -> torch.Tensor:
        """
        Implement controlled information release with top-k filtering.
        
        This method limits the information exposed in model outputs by:
        1. Keeping only the top-k logits unchanged
        2. Adding noise to non-top-k logits  
        3. Ensuring the top-1 prediction remains identical for usability
        
        Parameters:
            outputs (torch.Tensor): Raw logits from victim model (batch, num_classes)
            k (int): Number of top logits to preserve (1 ≤ k ≤ num_classes)
            
        Returns:
            torch.Tensor: Log-softmax distribution with controlled information release
        """
        num_classes = outputs.size(1)

        # Safety check: if k is out of range, release full distribution
        if k is None or k >= num_classes:
            return F.log_softmax(outputs, dim=1)

        # Normal controlled-release implementation
        
        # 1. Identify top-k logits for each sample in the batch
        topk_vals, topk_idx = torch.topk(outputs, k=k, dim=1)

        # 2. Create controlled version by adding noise to non-top-k entries
        noise_scale = 0.1  # Moderate noise level for non-top-k logits
        controlled = outputs.clone()
        noise = torch.randn_like(outputs) * noise_scale

        # Create mask: False for top-k positions, True for others
        mask = torch.ones_like(outputs, dtype=torch.bool)
        mask.scatter_(1, topk_idx, False)  # Set False where we keep original logits
        
        # Add noise only to non-top-k logits
        controlled[mask] += noise[mask]

        # 3. Guarantee top-1 logit exactly matches original (for usability)
        top1_idx = topk_idx[:, 0].unsqueeze(1)  # (batch, 1)
        controlled.scatter_(1, top1_idx, topk_vals[:, 0].unsqueeze(1))

        # Return log-softmax for consistent output format
        return F.log_softmax(controlled, dim=1)

    def _apply_deceptive_perturbation(
        self, outputs: torch.Tensor, suspicion: float
    ) -> torch.Tensor:
        """
        Apply graduated perturbation based on suspicion level.
        
        This method applies increasingly strong perturbations to model outputs
        based on the calculated suspicion score. Higher suspicion leads to
        more aggressive perturbation to mislead potential attackers.
        
        Perturbation techniques:
        1. Temperature scaling to flatten confidence
        2. Gaussian noise injection  
        3. Mixing with uniform distribution for high suspicion
        
        Args:
            outputs (torch.Tensor): Model outputs to perturb
            suspicion (float): Suspicion score (0.0 to 1.0)
            
        Returns:
            torch.Tensor: Perturbed log-softmax outputs
        """
        # Skip perturbation for low suspicion queries
        if suspicion < self.config["perturb_threshold"]:
            return outputs
        
        # Calculate perturbation strength based on suspicion level
        perturb_strength = (suspicion - self.config["perturb_threshold"]) / (
            1.0 - self.config["perturb_threshold"]
        )
        
        # 1. Temperature Scaling (flatten overconfident predictions)
        T = self.config["temperature_base"] + 2.0 * perturb_strength
        scaled_logits = outputs / T
        
        # 2. Calibrated Noise Injection
        # Scale noise intensity based on suspicion level
        noise_scale = self.config["base_noise_scale"] + (
            self.config["max_noise_scale"] - self.config["base_noise_scale"]
        ) * perturb_strength
        
        noise = torch.randn_like(outputs) * noise_scale
        
        # 3. Mix with Uniform Distribution for High Suspicion Queries
        if suspicion > 0.5:
            # Create uniform logits (equal probability for all classes)
            uniform_logits = torch.ones_like(outputs) * (-np.log(outputs.size(1)))
            
            # Mix scaled+noisy logits with uniform distribution
            alpha = 0.3 * perturb_strength  # Mixing coefficient
            perturbed = (1 - alpha) * (scaled_logits + noise) + alpha * uniform_logits
        else:
            # For moderate suspicion, just apply scaling and noise
            perturbed = scaled_logits + noise
        
        # Update statistics
        self.perturbed_responses += outputs.size(0)
        
        # Return log-softmax for consistent output format
        return F.log_softmax(perturbed, dim=1)


    #----------------------------------
    # MAIN DEFENSE COORDINATION
    #----------------------------------

    def defend(self, x: torch.Tensor) -> torch.Tensor:
        """
        Main defense entry point that coordinates all protection mechanisms.
        
        This method orchestrates the complete defense pipeline:
        1. OOD detection and pattern analysis
        2. Suspicion score calculation
        3. Query blocking for high-threat queries
        4. Controlled output release
        5. Adaptive perturbation based on threat level
        
        Args:
            x (torch.Tensor): Input batch to process
            
        Returns:
            torch.Tensor: Defended model outputs (log-softmax)
        """
        # Update query statistics
        self.total_queries += x.size(0)
        
        # --- Detection Phase ---
        # Perform OOD detection
        ood_scores, _ = self._detect_ood(x)
        
        # Analyze query patterns
        pattern_score = self._analyze_query_patterns(x)
        
        # Calculate overall suspicion score
        suspicion = self._calculate_suspicion_score(x, ood_scores, pattern_score)
        
        # Debug logging for suspicion monitoring
        if len(self.suspicion_history) % 10 == 0:
            avg_suspicion = (
                float(np.mean(self.suspicion_history))
                if self.suspicion_history else 0.0
            )
            logger.debug(f"Average suspicion score: {avg_suspicion:.3f}")
        
        # --- Response Generation Phase ---
        # Get victim model outputs
        with torch.no_grad():
            _, raw_outputs = self.feature_extractor(x)
        
        # Apply controlled information release
        controlled_outputs = self._controlled_release(raw_outputs, self.config["top_k"])
        
        # --- Threat Response Phase ---
        # Handle high-suspicion queries (potential blocking)
        if suspicion > self.config["block_threshold"]:
            self.blocked_queries += x.size(0)
            logger.warning(f"High suspicion query detected (score: {suspicion:.3f})")
            
            # Return heavily perturbed output for high-threat queries
            return self._apply_deceptive_perturbation(controlled_outputs, 1.0)
        
        # Apply graduated perturbation based on suspicion level
        defended_outputs = self._apply_deceptive_perturbation(controlled_outputs, suspicion)
        
        return defended_outputs

    def get_defense_summary(self) -> Dict:
        """
        Get comprehensive defense statistics and performance metrics.
        
        Returns:
            Dict: Dictionary containing defense performance statistics including:
                - Query processing statistics
                - Detection and blocking rates  
                - Pattern analysis results
                - Average suspicion levels
        """
        avg_suspicion = np.mean(list(self.suspicion_history)) if self.suspicion_history else 0
        
        return {
            "total_queries": self.total_queries,
            "blocked_queries": self.blocked_queries,
            "perturbed_responses": self.perturbed_responses,
            "unique_query_patterns": len(self.query_fingerprints),
            "average_suspicion": float(avg_suspicion),
            "block_rate": self.blocked_queries / max(1, self.total_queries),
            "perturb_rate": self.perturbed_responses / max(1, self.total_queries),
        }


#----------------------------------
# DEFENDED API WRAPPER
#----------------------------------

class DefendedBlackBoxAPI:
    """
    Drop-in replacement for BlackBoxAPI with integrated defense mechanisms.
    
    This class provides a compatible interface for existing code while adding
    comprehensive protection against model extraction attacks. It wraps the
    original model with the DefenseMechanism class and provides the same
    query interface as the original BlackBoxAPI.
    
    Key Features:
    - Compatible with existing BlackBoxAPI usage
    - Automatic defense activation for all queries
    - Comprehensive logging and statistics
    - Easy integration with existing ML pipelines
    """

    def __init__(self, victim: nn.Module, defense_config: Optional[Dict] = None):
        """
        Initialize defended API wrapper.
        
        Args:
            victim (nn.Module): The model to protect
            defense_config (Optional[Dict]): Custom defense configuration
        """
        self.victim = victim
        self.device = next(victim.parameters()).device
        self.defense = DefenseMechanism(victim, self.device, defense_config)
        self.api_calls = 0

    @torch.no_grad()
    def query(self, x: torch.Tensor, *, logits: bool = False) -> torch.Tensor:
        """
        Process query through defense mechanism.
        
        This method provides the same interface as the original BlackBoxAPI
        but routes all queries through the comprehensive defense system.
        
        Args:
            x (torch.Tensor): Input batch to query
            logits (bool): If True, return log-probabilities; if False, return probabilities
            
        Returns:
            torch.Tensor: Model predictions (defended)
        """
        self.api_calls += 1
        
        # Route query through defense mechanism
        defended_outputs = self.defense.defend(x.to(self.device))
        
        # Return in requested format (logits or probabilities)
        return defended_outputs if logits else torch.exp(defended_outputs)

    def fit_defense(self, train_loader) -> None:
        """
        Fit the defense mechanisms using training data.
        
        This method must be called before using the defended API to properly
        initialize the OOD detector and other components that require training data.
        
        Args:
            train_loader: DataLoader containing legitimate training samples
        """
        self.defense.fit_ood_detector(train_loader)

    def get_defense_report(self) -> Dict:
        """
        Get comprehensive defense performance report.
        
        Returns:
            Dict: Complete defense statistics including API usage and protection metrics
        """
        # Get base defense statistics
        report = self.defense.get_defense_summary()
        
        # Add API-specific statistics
        report["api_calls"] = self.api_calls
        
        return report