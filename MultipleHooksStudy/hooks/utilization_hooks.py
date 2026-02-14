"""
Downstream Utilization Analysis Hooks

Hooks for measuring how encoder features are utilized downstream.
Includes attention weight tracking, feature similarity across layers,
and mutual information estimation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import defaultdict

from .base_hooks import BaseAttentionHook, BaseFeatureHook


class AttentionWeightTracker(BaseAttentionHook):
    """
    Track attention weights to measure modality utilization.
    
    Measures what percentage of attention mass goes to each modality
    (vision, language, proprio) in cross-attention or self-attention layers.
    """
    
    def __init__(self, name: str = "attention_tracker", store_all: bool = True):
        super().__init__(name, store_all)
        self.modality_ranges: Dict[str, Tuple[int, int]] = {}
    
    def set_modality_ranges(self, ranges: Dict[str, Tuple[int, int]]):
        """
        Set token index ranges for each modality.
        
        Args:
            ranges: Dict mapping modality name to (start_idx, end_idx) tuple
                   e.g., {"vision": (0, 256), "language": (256, 512), "proprio": (512, 520)}
        """
        self.modality_ranges = ranges
    
    def compute_modality_attention(
        self,
        layer_name: str
    ) -> Optional[Dict[str, float]]:
        """
        Compute attention mass to each modality for a specific layer.
        
        Args:
            layer_name: Name of attention layer
        
        Returns:
            Dict mapping modality names to attention percentages
        """
        if layer_name not in self.attention_weights:
            return None
        
        attn_weights_list = self.attention_weights[layer_name]
        if not attn_weights_list or not self.modality_ranges:
            return None
        
        # Average over all stored attention matrices
        avg_attention = {}
        
        for attn in attn_weights_list:
            # attn shape: (batch, heads, seq, seq) or (batch, seq, seq)
            if len(attn.shape) == 4:
                # Average over heads
                attn = attn.mean(dim=1)
            
            # Average over batch
            if attn.shape[0] > 1:
                attn = attn.mean(dim=0)
            else:
                attn = attn.squeeze(0)
            
            # attn is now (seq, seq) - query × key attention
            # Sum over query dimension to get total attention to each key position
            attn_to_keys = attn.sum(dim=0)  # (seq,)
            
            # Compute attention mass to each modality
            total_attention = attn_to_keys.sum()
            
            for modality, (start, end) in self.modality_ranges.items():
                modality_attention = attn_to_keys[start:end].sum()
                if modality not in avg_attention:
                    avg_attention[modality] = []
                avg_attention[modality].append(
                    float(modality_attention / total_attention) if total_attention > 0 else 0.0
                )
        
        # Average over samples
        result = {
            modality: np.mean(values) * 100  # Convert to percentage
            for modality, values in avg_attention.items()
        }
        
        return result
    
    def get_modality_attention_report(self) -> Dict[str, Dict[str, float]]:
        """Get attention distribution across all tracked layers."""
        report = {}
        
        for layer_name in self.attention_weights.keys():
            modality_attn = self.compute_modality_attention(layer_name)
            if modality_attn:
                report[layer_name] = modality_attn
        
        return report


class FeatureSimilarityTracker:
    """
    Track feature similarity across layers.
    
    High similarity between input and output features indicates
    the features are not being transformed (underutilized).
    """
    
    def __init__(self):
        self.feature_hooks: Dict[str, BaseFeatureHook] = {}
        self.layer_names: List[str] = []
    
    def attach_to_layers(
        self,
        layers: List[nn.Module],
        layer_names: List[str]
    ):
        """Attach feature extraction hooks to multiple layers."""
        assert len(layers) == len(layer_names)
        
        for layer, name in zip(layers, layer_names):
            hook = BaseFeatureHook(name=f"{name}_features", store_all=True)
            hook.attach(layer, name)
            self.feature_hooks[name] = hook
            self.layer_names.append(name)
    
    @staticmethod
    def cosine_similarity(feat1: torch.Tensor, feat2: torch.Tensor) -> float:
        """Compute cosine similarity between two feature sets."""
        # Flatten if needed
        feat1_flat = feat1.reshape(feat1.shape[0], -1)
        feat2_flat = feat2.reshape(feat2.shape[0], -1)
        
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            feat1_flat, feat2_flat, dim=1
        )
        
        return float(similarity.mean())
    
    def compute_layer_to_layer_similarity(self) -> Dict[Tuple[str, str], float]:
        """
        Compute similarity between consecutive layers.
        
        Returns:
            Dict mapping (layer_i, layer_j) to similarity score
        """
        similarities = {}
        
        for i in range(len(self.layer_names) - 1):
            layer1_name = self.layer_names[i]
            layer2_name = self.layer_names[i + 1]
            
            hook1 = self.feature_hooks[layer1_name]
            hook2 = self.feature_hooks[layer2_name]
            
            feat1_list = hook1.features.get(layer1_name, [])
            feat2_list = hook2.features.get(layer2_name, [])
            
            if feat1_list and feat2_list:
                # Compute similarity for each pair
                sims = []
                for feat1, feat2 in zip(feat1_list, feat2_list):
                    if feat1.shape == feat2.shape:
                        sim = self.cosine_similarity(feat1, feat2)
                        sims.append(sim)
                
                if sims:
                    similarities[(layer1_name, layer2_name)] = np.mean(sims)
        
        return similarities
    
    def find_stagnant_layers(self, threshold: float = 0.95) -> List[Tuple[str, str]]:
        """
        Find layer pairs with very high similarity (features not transforming).
        
        Args:
            threshold: Similarity threshold for "stagnant"
        
        Returns:
            List of (layer_i, layer_j) tuples with similarity > threshold
        """
        similarities = self.compute_layer_to_layer_similarity()
        stagnant = []
        
        for (layer1, layer2), sim in similarities.items():
            if sim > threshold:
                stagnant.append((layer1, layer2))
        
        return stagnant
    
    def remove_all_hooks(self):
        """Remove all feature hooks."""
        for hook in self.feature_hooks.values():
            hook.remove()
        self.feature_hooks.clear()


class MutualInformationEstimator:
    """
    Estimate mutual information between encoder features and actions.
    
    Uses MINE (Mutual Information Neural Estimation) to quantify
    how much action-relevant information is in each encoder.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """
        Initialize MINE network.
        
        Args:
            input_dim: Dimension of concatenated (features, actions)
            hidden_dim: Hidden layer dimension
        """
        self.statistics_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.optimizer = torch.optim.Adam(self.statistics_network.parameters(), lr=0.001)
    
    def estimate_mi(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 256
    ) -> float:
        """
        Estimate mutual information between features and actions.
        
        Args:
            features: Feature tensor, shape (n_samples, feature_dim)
            actions: Action tensor, shape (n_samples, action_dim)
            epochs: Number of training epochs
            batch_size: Batch size for training
        
        Returns:
            Estimated mutual information in nats
        """
        n_samples = features.shape[0]
        
        for epoch in range(epochs):
            # Sample batch
            indices = np.random.choice(n_samples, min(batch_size, n_samples), replace=False)
            feat_batch = features[indices]
            action_batch = actions[indices]
            
            # Joint distribution
            joint = torch.cat([feat_batch, action_batch], dim=1)
            joint_scores = self.statistics_network(joint)
            
            # Marginal distribution (shuffle actions)
            shuffled_indices = np.random.permutation(len(indices))
            action_shuffled = action_batch[shuffled_indices]
            marginal = torch.cat([feat_batch, action_shuffled], dim=1)
            marginal_scores = self.statistics_network(marginal)
            
            # MINE lower bound
            mi_estimate = joint_scores.mean() - torch.log(torch.exp(marginal_scores).mean() + 1e-10)
            loss = -mi_estimate
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Final estimate
        with torch.no_grad():
            joint = torch.cat([features, actions], dim=1)
            joint_scores = self.statistics_network(joint)
            
            # Shuffle for marginal
            shuffled_idx = np.random.permutation(n_samples)
            marginal = torch.cat([features, actions[shuffled_idx]], dim=1)
            marginal_scores = self.statistics_network(marginal)
            
            mi = joint_scores.mean() - torch.log(torch.exp(marginal_scores).mean() + 1e-10)
        
        return float(mi)


class DownstreamUtilizationAnalyzer:
    """
    High-level analyzer for downstream feature utilization.
    """
    
    def __init__(self):
        self.attention_tracker = None
        self.similarity_tracker = None
        self.mi_estimators: Dict[str, MutualInformationEstimator] = {}
    
    def setup_attention_tracking(
        self,
        attention_layers: Dict[str, nn.Module],
        modality_ranges: Optional[Dict[str, Tuple[int, int]]] = None
    ):
        """Setup attention weight tracking."""
        self.attention_tracker = AttentionWeightTracker(store_all=True)
        
        for layer_name, layer_module in attention_layers.items():
            self.attention_tracker.attach(layer_module, layer_name)
        
        if modality_ranges:
            self.attention_tracker.set_modality_ranges(modality_ranges)
    
    def setup_similarity_tracking(
        self,
        layers: List[nn.Module],
        layer_names: List[str]
    ):
        """Setup feature similarity tracking."""
        self.similarity_tracker = FeatureSimilarityTracker()
        self.similarity_tracker.attach_to_layers(layers, layer_names)
    
    def estimate_information_content(
        self,
        features_dict: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        epochs: int = 100
    ) -> Dict[str, float]:
        """
        Estimate MI between each modality's features and actions.
        
        Args:
            features_dict: Dict mapping modality names to feature tensors
            actions: Action tensor
            epochs: MINE training epochs
        
        Returns:
            Dict mapping modality names to MI estimates
        """
        mi_estimates = {}
        
        for modality, features in features_dict.items():
            # Flatten features if needed
            if len(features.shape) > 2:
                features = features.reshape(features.shape[0], -1)
            
            # Initialize MINE if not exists
            if modality not in self.mi_estimators:
                input_dim = features.shape[1] + actions.shape[1]
                self.mi_estimators[modality] = MutualInformationEstimator(input_dim)
            
            # Estimate MI
            mi = self.mi_estimators[modality].estimate_mi(features, actions, epochs)
            mi_estimates[modality] = mi
        
        return mi_estimates
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get complete utilization analysis."""
        report = {}
        
        if self.attention_tracker:
            report["attention_distribution"] = (
                self.attention_tracker.get_modality_attention_report()
            )
        
        if self.similarity_tracker:
            report["layer_similarity"] = (
                self.similarity_tracker.compute_layer_to_layer_similarity()
            )
            report["stagnant_layers"] = (
                self.similarity_tracker.find_stagnant_layers()
            )
        
        return report
    
    def cleanup(self):
        """Remove all hooks."""
        if self.attention_tracker:
            self.attention_tracker.remove()
        if self.similarity_tracker:
            self.similarity_tracker.remove_all_hooks()
