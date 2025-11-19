"""
OCTO model adapter for VLA benchmarking.

This adapter wraps the OCTO model to provide a standardized interface
for evaluation on various benchmarks.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import jax
import jax.numpy as jnp

# Add OCTO to path
octo_path = Path(__file__).parent.parent / "models" / "octo"
if str(octo_path) not in sys.path:
    sys.path.insert(0, str(octo_path))

from octo.model.octo_model import OctoModel
from adapters.base_adapter import BaseAdapter


class OctoAdapter(BaseAdapter):
    """
    Adapter for OCTO (Open X-Embodiment Transformer for Control) models.
    
    OCTO is a generalist robot policy trained on diverse robot manipulation data.
    It supports:
    - Multiple camera inputs
    - Language-conditioned tasks
    - Goal-image-conditioned tasks
    - Variable action spaces
    """
    
    def __init__(
        self,
        model_path: str = "hf://rail-berkeley/octo-small-1.5",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize OCTO adapter.
        
        Args:
            model_path: HuggingFace model identifier or local path
                      Default: "hf://rail-berkeley/octo-small-1.5"
                      Options: "hf://rail-berkeley/octo-base-1.5"
            config: Optional configuration dict with keys:
                - 'horizon': Action prediction horizon (default: 1)
                - 'action_ensemble': Number of action samples (default: 1)
                - 'temperature': Sampling temperature (default: 1.0)
        """
        super().__init__(model_path, config)
        
        # Default config
        self.horizon = self.config.get('horizon', 1)
        self.action_ensemble = self.config.get('action_ensemble', 1)
        self.temperature = self.config.get('temperature', 1.0)
        
        # Normalization statistics (will be loaded with model)
        self.action_mean = None
        self.action_std = None
        
        # Model-specific state
        self.task_encoding = None
    
    def load_model(self) -> None:
        """Load OCTO model from HuggingFace or local path."""
        try:
            print(f"Loading OCTO model from: {self.model_path}")
            self.model = OctoModel.load_pretrained(self.model_path)
            
            # Get normalization stats from model
            if hasattr(self.model, 'dataset_statistics'):
                stats = self.model.dataset_statistics
                if 'action' in stats:
                    self.action_mean = np.array(stats['action']['mean'])
                    self.action_std = np.array(stats['action']['std'])
            
            self._is_loaded = True
            print(f"✅ OCTO model loaded successfully!")
            print(f"   Model info: {self.model.get_pretty_spec()}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load OCTO model: {e}")
    
    def preprocess_observation(
        self,
        observation: Dict[str, np.ndarray],
        task_description: Optional[str] = None,
        goal_image: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Preprocess observation for OCTO model.
        
        Args:
            observation: Dict with keys:
                - 'image_primary': Primary camera RGB image (H, W, 3), values in [0, 255]
                - 'image_wrist': Optional wrist camera image (H, W, 3)
                - 'state': Robot state (proprioceptive info)
            task_description: Natural language task description
            goal_image: Goal image for goal-conditioned tasks
        
        Returns:
            Preprocessed observation dict for OCTO
        """
        from PIL import Image
        
        preprocessed = {}
        
        # Process images - OCTO expects:
        # - image_primary: 256x256
        # - image_wrist: 128x128  
        # - values in [0, 1] range
        # - history window (we'll add this in predict_action)
        
        if 'image_primary' in observation:
            img = observation['image_primary']
            if img.max() > 1.0:
                img = img.astype(np.uint8)
            else:
                img = (img * 255).astype(np.uint8)
            
            # Resize to 256x256
            img_pil = Image.fromarray(img)
            img_pil = img_pil.resize((256, 256), Image.BILINEAR)
            img = np.array(img_pil).astype(np.float32) / 255.0
            preprocessed['image_primary'] = img
        
        if 'image_wrist' in observation:
            img = observation['image_wrist']
            if img.max() > 1.0:
                img = img.astype(np.uint8)
            else:
                img = (img * 255).astype(np.uint8)
                
            # Resize to 128x128
            img_pil = Image.fromarray(img)
            img_pil = img_pil.resize((128, 128), Image.BILINEAR)
            img = np.array(img_pil).astype(np.float32) / 255.0
            preprocessed['image_wrist'] = img
        
        # Process proprioceptive state
        if 'state' in observation:
            preprocessed['proprio'] = observation['state']
        
        # Store task conditioning
        if task_description is not None:
            preprocessed['task_description'] = task_description
        
        if goal_image is not None:
            if goal_image.max() > 1.0:
                goal_image = goal_image.astype(np.uint8)
            else:
                goal_image = (goal_image * 255).astype(np.uint8)
            
            # Resize goal image to 256x256
            img_pil = Image.fromarray(goal_image)
            img_pil = img_pil.resize((256, 256), Image.BILINEAR)
            goal_image = np.array(img_pil).astype(np.float32) / 255.0
            preprocessed['goal_image'] = goal_image
        
        return preprocessed
    
    def predict_action(
        self,
        preprocessed_obs: Dict[str, Any],
        **kwargs
    ) -> np.ndarray:
        """
        Predict action using OCTO model.
        
        Args:
            preprocessed_obs: Preprocessed observation from preprocess_observation()
            **kwargs: Override default parameters:
                - 'horizon': Action prediction horizon
                - 'temperature': Sampling temperature
                - 'action_ensemble': Number of samples to average
        
        Returns:
            Predicted action array, shape (action_dim,) or (horizon, action_dim)
        """
        # Get task conditioning
        task_description = preprocessed_obs.get('task_description', None)
        goal_image = preprocessed_obs.get('goal_image', None)
        
        # Create observation dict for OCTO
        # OCTO expects shape (batch, history_window, H, W, C)
        # We duplicate the current observation for history window = 2
        obs_dict = {
            'image_primary': jnp.array(preprocessed_obs['image_primary'])[None, None, ...],  # (1, 1, 256, 256, 3)
            'timestep_pad_mask': jnp.array([[True, True]])  # (1, 2) - both timesteps are valid
        }
        # Duplicate to create history window of 2
        obs_dict['image_primary'] = jnp.repeat(obs_dict['image_primary'], 2, axis=1)  # (1, 2, 256, 256, 3)
        
        if 'image_wrist' in preprocessed_obs:
            obs_dict['image_wrist'] = jnp.array(preprocessed_obs['image_wrist'])[None, None, ...]
            obs_dict['image_wrist'] = jnp.repeat(obs_dict['image_wrist'], 2, axis=1)  # (1, 2, 128, 128, 3)
        
        if 'proprio' in preprocessed_obs:
            obs_dict['proprio'] = jnp.array(preprocessed_obs['proprio'])[None]
        
        # Get parameters
        rng_seed = kwargs.get('rng_seed', 0)
        
        # Prepare task conditioning
        task = None
        if task_description is not None:
            # Tokenize language instruction using OCTO's tokenizer
            task = self.model.create_tasks(texts=[task_description])
        elif goal_image is not None:
            # Goal-conditioned: convert goal image to task dict
            task = {'image_primary': jnp.array(goal_image)[None]}
        
        # Run inference
        try:
            # Get unnormalization statistics for action denormalization
            # Try to use dataset-specific statistics if available
            dataset_name = kwargs.get('dataset_name', 'bridge_dataset')
            unnorm_stats = self.model.dataset_statistics.get(dataset_name, {}).get("action", None)
            
            if task is not None:
                # Task-conditioned (language or goal)
                actions = self.model.sample_actions(
                    obs_dict,
                    task,
                    unnormalization_statistics=unnorm_stats,
                    rng=jax.random.PRNGKey(rng_seed)
                )
            else:
                # No task conditioning
                actions = self.model.sample_actions(
                    obs_dict,
                    unnormalization_statistics=unnorm_stats,
                    rng=jax.random.PRNGKey(rng_seed)
                )
            
            # Convert to numpy and remove batch dimension
            actions = np.array(actions)
            
            # Remove batch dimension if present
            if actions.ndim == 3:  # (batch, horizon, action_dim)
                actions = actions[0]  # (horizon, action_dim)
            
            # OCTO predicts multiple timesteps, take first action for execution
            if actions.ndim == 2:  # (horizon, action_dim)
                actions = actions[0]  # (action_dim,)
            
            return actions
            
        except Exception as e:
            raise RuntimeError(f"OCTO inference failed: {e}")
    
    def postprocess_action(
        self,
        action: np.ndarray,
        observation: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Postprocess OCTO action prediction.
        
        OCTO outputs normalized actions, so we denormalize them here.
        
        Args:
            action: Raw action from OCTO model
            observation: Optional original observation
        
        Returns:
            Denormalized action ready for execution
        """
        # Denormalize if statistics available
        if self.action_mean is not None and self.action_std is not None:
            action = action * self.action_std + self.action_mean
        
        # Clip to reasonable ranges (safety)
        # These are common ranges for robot manipulation
        action = np.clip(action, -1.0, 1.0)
        
        return action
    
    def reset(self) -> None:
        """Reset OCTO adapter state."""
        # OCTO is not recurrent, but we reset task encoding
        self.task_encoding = None
    
    @property
    def action_dim(self) -> int:
        """Get action dimension from loaded model."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")
        
        # OCTO action dimension is typically 7 (6 DoF + gripper)
        # but can vary - check from model spec
        if hasattr(self.model, 'action_dim'):
            return self.model.action_dim
        else:
            # Default for most OCTO models
            return 7
    
    @property
    def action_space(self) -> Dict[str, Any]:
        """Get OCTO action space info."""
        return {
            'dim': self.action_dim,
            'low': -1.0,  # OCTO typically uses normalized actions
            'high': 1.0,
            'type': 'continuous'
        }
    
    def __repr__(self) -> str:
        model_type = "octo-small" if "small" in self.model_path else "octo-base"
        return f"OctoAdapter(model='{model_type}', loaded={self._is_loaded})"
