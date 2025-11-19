"""
Base adapter class for VLA models.

This module provides an abstract base class for adapting different VLA models
to a common interface for benchmarking.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class BaseAdapter(ABC):
    """
    Abstract base class for VLA model adapters.
    
    An adapter wraps a VLA model and provides a standardized interface for:
    - Loading models and checkpoints
    - Preprocessing observations
    - Running inference (action prediction)
    - Postprocessing actions
    
    This allows different VLA architectures to be benchmarked uniformly.
    """
    
    def __init__(self, model_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the adapter.
        
        Args:
            model_path: Path to the model checkpoint or identifier
            config: Optional configuration dictionary for model-specific settings
        """
        self.model_path = model_path
        self.config = config or {}
        self.model = None
        self._is_loaded = False
    
    @abstractmethod
    def load_model(self) -> None:
        """
        Load the model from the specified path.
        
        This method should:
        1. Load the model architecture
        2. Load pretrained weights
        3. Set the model to evaluation mode
        4. Move model to appropriate device (CPU/GPU)
        
        Raises:
            FileNotFoundError: If model checkpoint not found
            RuntimeError: If model loading fails
        """
        pass
    
    @abstractmethod
    def preprocess_observation(
        self,
        observation: Dict[str, np.ndarray],
        task_description: Optional[str] = None,
        goal_image: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Preprocess raw observation data for model input.
        
        Args:
            observation: Dictionary containing observation data
                - 'image': RGB image(s) from camera(s), shape (H, W, 3) or (N, H, W, 3)
                - 'state': Robot state (joint positions, gripper state, etc.)
                - Additional modality-specific keys
            task_description: Optional natural language task description
            goal_image: Optional goal image for goal-conditioned tasks
        
        Returns:
            Preprocessed observation dict ready for model input
        """
        pass
    
    @abstractmethod
    def predict_action(
        self,
        preprocessed_obs: Dict[str, Any],
        **kwargs
    ) -> np.ndarray:
        """
        Predict action(s) given preprocessed observation.
        
        Args:
            preprocessed_obs: Preprocessed observation from preprocess_observation()
            **kwargs: Additional model-specific parameters (temperature, num_samples, etc.)
        
        Returns:
            Predicted action(s) as numpy array
            - For single action: shape (action_dim,)
            - For action sequence: shape (horizon, action_dim)
            - For action distribution: shape (num_samples, action_dim)
        """
        pass
    
    @abstractmethod
    def postprocess_action(
        self,
        action: np.ndarray,
        observation: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Postprocess predicted action for execution.
        
        This may include:
        - Denormalization
        - Clipping to valid ranges
        - Coordinate frame transformations
        - Temporal filtering
        
        Args:
            action: Raw action prediction from the model
            observation: Optional original observation for context
        
        Returns:
            Action ready for execution on the robot/environment
        """
        pass
    
    def reset(self) -> None:
        """
        Reset the adapter state (if stateful).
        
        This should be called at the beginning of each episode.
        Can be overridden by models that maintain internal state
        (e.g., recurrent models, temporal models).
        """
        pass
    
    def get_action(
        self,
        observation: Dict[str, np.ndarray],
        task_description: Optional[str] = None,
        goal_image: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """
        High-level interface: Get action from raw observation (end-to-end).
        
        This is a convenience method that chains:
        1. preprocess_observation()
        2. predict_action()
        3. postprocess_action()
        
        Args:
            observation: Raw observation dictionary
            task_description: Optional task description
            goal_image: Optional goal image
            **kwargs: Additional parameters for predict_action()
        
        Returns:
            Action ready for execution
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess
        preprocessed_obs = self.preprocess_observation(
            observation,
            task_description=task_description,
            goal_image=goal_image
        )
        
        # Predict
        raw_action = self.predict_action(preprocessed_obs, **kwargs)
        
        # Postprocess
        action = self.postprocess_action(raw_action, observation)
        
        return action
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Return the action dimension of the model."""
        pass
    
    @property
    @abstractmethod
    def action_space(self) -> Dict[str, Any]:
        """
        Return information about the action space.
        
        Returns:
            Dictionary with keys:
                - 'dim': Action dimension
                - 'low': Lower bounds (if applicable)
                - 'high': Upper bounds (if applicable)
                - 'type': 'continuous' or 'discrete'
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_path='{self.model_path}', loaded={self._is_loaded})"
