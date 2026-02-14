"""
Representation Quality Analysis Hooks

Hooks for analyzing quality and redundancy of encoder representations.
Includes feature extraction, CKA similarity computation, and effective rank calculation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import defaultdict

from .base_hooks import BaseFeatureHook


class FeatureExtractor(BaseFeatureHook):
    """
    Extract and store intermediate features from encoders.
    
    Captures features before fusion to analyze representation quality
    and redundancy between modalities.
    """
    
    def __init__(self, name: str = "feature_extractor", store_all: bool = True):
        super().__init__(name, store_all)
        self.feature_shapes = {}
    
    def attach_to_encoders(
        self,
        vision_encoder: Optional[nn.Module] = None,
        language_encoder: Optional[nn.Module] = None,
        proprio_encoder: Optional[nn.Module] = None,
        custom_encoders: Optional[Dict[str, nn.Module]] = None
    ):
        """Attach to multiple encoders."""
        if vision_encoder is not None:
            self.attach(vision_encoder, "vision_features")
        
        if language_encoder is not None:
            self.attach(language_encoder, "language_features")
        
        if proprio_encoder is not None:
            self.attach(proprio_encoder, "proprio_features")
        
        if custom_encoders:
            for name, encoder in custom_encoders.items():
                self.attach(encoder, f"{name}_features")
    
    def get_stacked_features(self, modality: str) -> Optional[torch.Tensor]:
        """
        Get stacked features across all samples for a modality.
        
        Args:
            modality: Name of modality (e.g., "vision_features")
        
        Returns:
            Stacked tensor of shape (n_samples, n_features) or None
        """
        if modality not in self.features:
            return None
        
        features_list = self.features[modality]
        if not features_list:
            return None
        
        # Handle different feature shapes
        try:
            # Flatten spatial dimensions if needed
            flattened = []
            for feat in features_list:
                if len(feat.shape) > 2:
                    # (batch, seq, dim) or (batch, h, w, dim) -> (batch, -1)
                    feat = feat.reshape(feat.shape[0], -1)
                elif len(feat.shape) == 2:
                    # Already (batch, dim)
                    pass
                else:
                    # (dim,) -> (1, dim)
                    feat = feat.unsqueeze(0)
                
                # Take mean across batch if batched
                if feat.shape[0] > 1:
                    feat = feat.mean(dim=0, keepdim=True)
                
                flattened.append(feat)
            
            stacked = torch.cat(flattened, dim=0)
            self.feature_shapes[modality] = stacked.shape
            return stacked
        except Exception as e:
            print(f"Error stacking features for {modality}: {e}")
            return None
    
    def get_feature_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics about extracted features."""
        stats = {}
        
        for modality in self.features.keys():
            stacked = self.get_stacked_features(modality)
            if stacked is not None:
                stats[modality] = {
                    "shape": tuple(stacked.shape),
                    "mean": float(stacked.mean()),
                    "std": float(stacked.std()),
                    "min": float(stacked.min()),
                    "max": float(stacked.max()),
                    "n_samples": stacked.shape[0],
                    "n_features": stacked.shape[1] if len(stacked.shape) > 1 else 1
                }
        
        return stats


class CKASimilarityAnalyzer:
    """
    Compute Centered Kernel Alignment (CKA) similarity between representations.
    
    Measures redundancy between different encoder outputs.
    High CKA indicates redundant information.
    """
    
    def __init__(self, feature_extractor: FeatureExtractor):
        self.feature_extractor = feature_extractor
    
    @staticmethod
    def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Compute linear CKA between two feature matrices.
        
        Args:
            X: Feature matrix 1, shape (n_samples, n_features_1)
            Y: Feature matrix 2, shape (n_samples, n_features_2)
        
        Returns:
            CKA similarity score between 0 and 1
        """
        # Center the features
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)
        
        # Compute CKA
        X_gram = X @ X.T
        Y_gram = Y @ Y.T
        
        numerator = torch.norm(X_gram @ Y_gram, p='fro')**2
        denominator = torch.norm(X_gram, p='fro') * torch.norm(Y_gram, p='fro')
        
        if denominator == 0:
            return 0.0
        
        cka = numerator / denominator
        return float(cka)
    
    @staticmethod
    def rbf_cka(X: torch.Tensor, Y: torch.Tensor, sigma: Optional[float] = None) -> float:
        """
        Compute RBF (Gaussian) kernel CKA.
        
        Args:
            X: Feature matrix 1
            Y: Feature matrix 2
            sigma: RBF bandwidth (auto-computed if None)
        
        Returns:
            CKA similarity score
        """
        def rbf_kernel(X, sigma):
            # Compute pairwise squared distances
            X_sq = (X**2).sum(dim=1, keepdim=True)
            dist = X_sq + X_sq.T - 2 * X @ X.T
            
            # Compute RBF kernel
            if sigma is None:
                sigma = torch.median(dist[dist > 0])
            
            K = torch.exp(-dist / (2 * sigma**2))
            return K
        
        if sigma is None:
            # Auto-select sigma based on median heuristic
            X_sigma = torch.median((X**2).sum(dim=1))
            Y_sigma = torch.median((Y**2).sum(dim=1))
            sigma = (X_sigma + Y_sigma) / 2
        
        K_X = rbf_kernel(X, sigma)
        K_Y = rbf_kernel(Y, sigma)
        
        # Center kernels
        n = K_X.shape[0]
        H = torch.eye(n) - torch.ones(n, n) / n
        K_X = H @ K_X @ H
        K_Y = H @ K_Y @ H
        
        numerator = torch.norm(K_X @ K_Y, p='fro')**2
        denominator = torch.norm(K_X, p='fro') * torch.norm(K_Y, p='fro')
        
        if denominator == 0:
            return 0.0
        
        return float(numerator / denominator)
    
    def compute_all_pairwise_cka(
        self,
        kernel: str = "linear"
    ) -> Dict[Tuple[str, str], float]:
        """
        Compute CKA between all pairs of modalities.
        
        Args:
            kernel: "linear" or "rbf"
        
        Returns:
            Dict mapping (modality1, modality2) to CKA score
        """
        modalities = list(self.feature_extractor.features.keys())
        cka_scores = {}
        
        for i, mod1 in enumerate(modalities):
            for mod2 in modalities[i+1:]:
                feat1 = self.feature_extractor.get_stacked_features(mod1)
                feat2 = self.feature_extractor.get_stacked_features(mod2)
                
                if feat1 is not None and feat2 is not None:
                    # Ensure same number of samples
                    min_samples = min(feat1.shape[0], feat2.shape[0])
                    feat1 = feat1[:min_samples]
                    feat2 = feat2[:min_samples]
                    
                    if kernel == "linear":
                        score = self.linear_cka(feat1, feat2)
                    else:
                        score = self.rbf_cka(feat1, feat2)
                    
                    cka_scores[(mod1, mod2)] = score
        
        return cka_scores
    
    def get_redundancy_report(self) -> Dict[str, Any]:
        """Get comprehensive redundancy analysis."""
        cka_linear = self.compute_all_pairwise_cka(kernel="linear")
        cka_rbf = self.compute_all_pairwise_cka(kernel="rbf")
        
        return {
            "cka_linear": cka_linear,
            "cka_rbf": cka_rbf,
            "high_redundancy_pairs": {
                k: v for k, v in cka_linear.items() if v > 0.7
            }
        }


class EffectiveRankCalculator:
    """
    Compute effective rank of encoder representations.
    
    Measures intrinsic dimensionality - how many dimensions are actually
    being used vs. total dimensions available.
    """
    
    def __init__(self, feature_extractor: FeatureExtractor):
        self.feature_extractor = feature_extractor
    
    @staticmethod
    def compute_effective_rank(features: torch.Tensor) -> Tuple[float, np.ndarray]:
        """
        Compute effective rank using participation ratio.
        
        Args:
            features: Feature matrix, shape (n_samples, n_features)
        
        Returns:
            Tuple of (effective_rank, eigenvalues)
        """
        # Compute covariance matrix
        features_np = features.numpy()
        cov = np.cov(features_np.T)
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Filter near-zero
        
        if len(eigenvalues) == 0:
            return 0.0, np.array([])
        
        # Effective rank via participation ratio
        effective_rank = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
        
        return float(effective_rank), eigenvalues
    
    @staticmethod
    def compute_stable_rank(features: torch.Tensor) -> float:
        """
        Compute stable rank (Frobenius norm / spectral norm).
        
        Args:
            features: Feature matrix
        
        Returns:
            Stable rank value
        """
        features_np = features.numpy()
        
        # Compute singular values
        _, singular_values, _ = np.linalg.svd(features_np, full_matrices=False)
        
        if len(singular_values) == 0:
            return 0.0
        
        # Stable rank = ||A||_F^2 / ||A||_2^2
        frobenius_norm_sq = (singular_values ** 2).sum()
        spectral_norm_sq = singular_values[0] ** 2
        
        if spectral_norm_sq == 0:
            return 0.0
        
        stable_rank = frobenius_norm_sq / spectral_norm_sq
        return float(stable_rank)
    
    def analyze_all_modalities(self) -> Dict[str, Dict[str, Any]]:
        """
        Compute effective rank for all modalities.
        
        Returns:
            Dict with effective rank statistics for each modality
        """
        results = {}
        
        for modality in self.feature_extractor.features.keys():
            features = self.feature_extractor.get_stacked_features(modality)
            
            if features is not None and features.shape[0] > 1:
                effective_rank, eigenvalues = self.compute_effective_rank(features)
                stable_rank = self.compute_stable_rank(features)
                
                total_dims = features.shape[1] if len(features.shape) > 1 else 1
                utilization_pct = (effective_rank / total_dims) * 100 if total_dims > 0 else 0
                
                results[modality] = {
                    "effective_rank": effective_rank,
                    "stable_rank": stable_rank,
                    "total_dimensions": total_dims,
                    "utilization_percentage": utilization_pct,
                    "eigenvalue_spectrum": eigenvalues.tolist(),
                    "top_5_eigenvalues": eigenvalues[-5:][::-1].tolist() if len(eigenvalues) >= 5 else eigenvalues[::-1].tolist()
                }
        
        return results
    
    def get_dimensionality_report(self) -> Dict[str, Any]:
        """Get comprehensive dimensionality analysis."""
        analysis = self.analyze_all_modalities()
        
        # Find most and least utilized
        if analysis:
            utilizations = {k: v["utilization_percentage"] for k, v in analysis.items()}
            most_utilized = max(utilizations, key=utilizations.get)
            least_utilized = min(utilizations, key=utilizations.get)
        else:
            most_utilized = None
            least_utilized = None
        
        return {
            "modality_analysis": analysis,
            "most_utilized": most_utilized,
            "least_utilized": least_utilized,
            "summary": {
                mod: {
                    "effective_rank": data["effective_rank"],
                    "utilization_pct": data["utilization_percentage"]
                }
                for mod, data in analysis.items()
            }
        }


class RepresentationQualityAnalyzer:
    """
    High-level analyzer combining feature extraction, CKA, and effective rank.
    """
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor(store_all=True)
        self.cka_analyzer = CKASimilarityAnalyzer(self.feature_extractor)
        self.rank_calculator = EffectiveRankCalculator(self.feature_extractor)
    
    def setup(
        self,
        vision_encoder: Optional[nn.Module] = None,
        language_encoder: Optional[nn.Module] = None,
        proprio_encoder: Optional[nn.Module] = None,
        custom_encoders: Optional[Dict[str, nn.Module]] = None
    ):
        """Setup feature extraction hooks."""
        self.feature_extractor.attach_to_encoders(
            vision_encoder, language_encoder, proprio_encoder, custom_encoders
        )
    
    def reset(self):
        """Reset feature storage."""
        self.feature_extractor.reset()
    
    def remove_hooks(self):
        """Remove all hooks."""
        self.feature_extractor.remove()
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get complete representation quality report."""
        return {
            "feature_statistics": self.feature_extractor.get_feature_stats(),
            "redundancy_analysis": self.cka_analyzer.get_redundancy_report(),
            "dimensionality_analysis": self.rank_calculator.get_dimensionality_report()
        }
    
    def print_summary(self):
        """Print human-readable summary."""
        report = self.get_comprehensive_report()
        
        print("=" * 80)
        print("REPRESENTATION QUALITY ANALYSIS")
        print("=" * 80)
        
        # Feature stats
        print("\n### Feature Statistics ###")
        for modality, stats in report["feature_statistics"].items():
            print(f"\n{modality}:")
            print(f"  Shape: {stats['shape']}")
            print(f"  Samples: {stats['n_samples']}, Features: {stats['n_features']}")
            print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
        
        # CKA redundancy
        print("\n### CKA Redundancy Analysis ###")
        for (mod1, mod2), score in report["redundancy_analysis"]["cka_linear"].items():
            print(f"CKA({mod1}, {mod2}): {score:.4f}")
        
        # Effective rank
        print("\n### Effective Rank Analysis ###")
        for modality, data in report["dimensionality_analysis"]["summary"].items():
            print(f"{modality}:")
            print(f"  Effective Rank: {data['effective_rank']:.2f}")
            print(f"  Utilization: {data['utilization_pct']:.1f}%")
        
        print("=" * 80)
