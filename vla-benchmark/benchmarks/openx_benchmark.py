"""
Open X-Embodiment benchmark for VLA evaluation.

This benchmark evaluates models on the Open X-Embodiment dataset,
using OCTO's data loading utilities for proper dataset access.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import tensorflow_datasets as tfds

# Add OCTO to path for dataset utilities
octo_path = Path(__file__).parent.parent / "models" / "octo"
if str(octo_path) not in sys.path:
    sys.path.insert(0, str(octo_path))

try:
    from octo.data.oxe import make_oxe_dataset_kwargs, OXE_DATASET_CONFIGS
    from octo.data.dataset import make_single_dataset
    OXE_AVAILABLE = True
except ImportError:
    OXE_AVAILABLE = False
    print("Warning: OCTO data utilities not available. Install OCTO to use OpenX benchmark.")

from benchmarks.base_benchmark import BaseBenchmark


class OpenXBenchmark(BaseBenchmark):
    """
    Open X-Embodiment benchmark for offline evaluation.
    
    This benchmark:
    - Loads trajectories from Open X-Embodiment datasets (e.g., Bridge V2)
    - Evaluates action prediction accuracy
    - Computes metrics like MSE, L1, and success criteria
    
    Note: This is an OFFLINE benchmark - it compares predicted actions
    to ground truth actions from the dataset, not live robot execution.
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        dataset_name: str = "fractal20220817_data",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize OpenX benchmark.
        
        Args:
            data_path: Path to OXE data (default: data/open-x for local project data)
            dataset_name: Name of the OXE dataset (see OXE_DATASET_CONFIGS)
                Available: fractal20220817_data, bridge_dataset, kuka, etc.
            config: Optional config with keys:
                - 'max_episodes_per_task': Max episodes to evaluate per task
                - 'action_mse_threshold': MSE threshold for "success"
                - 'use_language': Whether to use language instructions
        """
        # Default to project data directory if not specified
        if data_path is None:
            script_dir = Path(__file__).parent
            data_path = str(script_dir.parent / "data" / "open-x")
        
        super().__init__(
            benchmark_name="OpenX",
            data_path=data_path,
            config=config
        )
        
        if not OXE_AVAILABLE:
            raise RuntimeError(
                "OCTO data utilities not available. Please install OCTO:\n"
                "  cd models/octo && pip install -e ."
            )
        
        # Validate dataset name
        if dataset_name not in OXE_DATASET_CONFIGS:
            available = list(OXE_DATASET_CONFIGS.keys())
            raise ValueError(
                f"Dataset '{dataset_name}' not found in OXE configs.\n"
                f"Available datasets: {available[:10]}...\n"
                f"See models/octo/octo/data/oxe/oxe_dataset_configs.py for full list"
            )
        
        self.dataset_name = dataset_name
        self.max_episodes = self.config.get('max_episodes_per_task', 100)
        self.action_threshold = self.config.get('action_mse_threshold', 0.1)
        self.use_language = self.config.get('use_language', True)
        
        self.dataset = None
        self.iterator = None
        self.episodes = []
    
    def setup(self) -> None:
        """Load Open X-Embodiment dataset using OCTO's data utilities.

        Behavior:
        - If config['use_synthetic_debug'] is True, build a small synthetic
          episode set (fast, no downloads) suitable for pipeline testing.
        - Otherwise, attempt to load the dataset from TFDS data_dir. We do
          NOT automatically invoke a large download; if the dataset is absent
          we raise a clear error with next steps.
        """

        # Synthetic debug mode: create small synthetic episodes to exercise the
        # pipeline without downloading large datasets. Useful for fast testing.
        if self.config.get('use_synthetic_debug', False):
            print("Using synthetic debug dataset (no downloads). Generating small episodes...")
            # Create simple synthetic episodes matching OCTO data format
            num_eps = min(self.max_episodes, 8)
            synthetic_eps = []
            for i in range(num_eps):
                # Each episode has 10 timesteps
                traj_len = 10
                
                # Create observations in OCTO format: {'image_primary': (traj_len, H, W, C)}
                obs_dict = {
                    'image_primary': np.zeros((traj_len, 128, 128, 3), dtype=np.uint8),
                }
                
                # Create actions: (traj_len, action_dim)
                action = np.zeros((traj_len, 7), dtype=np.float32)
                
                # Create task dict with language instruction
                task_dict = {
                    'language_instruction': b'synthetic debug task'
                }
                
                # Episode in OCTO format
                synthetic_eps.append({
                    'observation': obs_dict,
                    'action': action,
                    'task': task_dict
                })

            self.episodes = synthetic_eps
            self._is_initialized = True
            print(f"✅ Created {len(self.episodes)} synthetic episodes for debug/testing")
            return

        # Normal mode: use OCTO's custom OXE data loader
        try:
            print(f"Loading {self.dataset_name} using OCTO data loader (data_dir={self.data_path})...")

            # Use OCTO's make_oxe_dataset_kwargs to get the proper config
            # This handles dataset-specific transformations and standardization
            dataset_kwargs = make_oxe_dataset_kwargs(
                self.dataset_name,
                self.data_path
            )
            
            print(f"Dataset config: {dataset_kwargs['name']}")
            
            # Try to create the dataset using OCTO's loader
            # Note: Some OXE datasets (like bridge_dataset) don't have registered TFDS builders
            # and need to be loaded directly from their directory
            try:
                self.dataset = make_single_dataset(
                    dataset_kwargs,
                    train=True,
                    traj_transform_kwargs=dict(
                        window_size=1,  # No history window for evaluation
                        goal_relabeling_strategy=None,  # No goal relabeling
                    ),
                    frame_transform_kwargs=dict(
                        resize_size=(256, 256),  # Standard size for OCTO
                    ),
                )
            except Exception as e:
                # If standard TFDS loading fails, try loading from directory
                # This handles datasets like bridge_dataset that exist as TFRecords
                # but don't have a registered TFDS builder class
                error_msg = str(e)
                if "not found" in error_msg.lower() or "no builder" in error_msg.lower():
                    print(f"Standard TFDS loading failed, trying directory-based loading...")
                    print(f"Error was: {error_msg[:200]}")
                    
                    # Find the dataset directory
                    from pathlib import Path
                    data_path = Path(self.data_path)
                    dataset_dir = data_path / self.dataset_name
                    
                    # Find the version directory (usually 0.0.1 or 0.1.0)
                    version_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
                    if not version_dirs:
                        raise RuntimeError(
                            f"Dataset directory {dataset_dir} exists but contains no version subdirectories.\n"
                            f"Expected structure: {self.data_path}/{self.dataset_name}/VERSION/"
                        )
                    
                    version_dir = version_dirs[0]  # Use first/oldest version
                    print(f"Loading from directory: {version_dir}")
                    
                    # Load using TFDS builder_from_directory
                    import tensorflow_datasets as tfds
                    builder = tfds.builder_from_directory(str(version_dir))
                    print(f"✅ Loaded builder: {builder.info.name if hasattr(builder.info, 'name') else 'unknown'}")
                    
                    # Now we need to manually create the dataset since make_single_dataset won't work
                    # Load the raw dataset
                    raw_dataset = builder.as_dataset(split='train')
                    
                    # We'll need to manually apply OCTO's transforms
                    # For now, convert to our internal format directly
                    print("Converting dataset to episodes (this may take a while)...")
                    self.episodes = self._load_episodes_from_tfds_builder(
                        raw_dataset, 
                        dataset_name=self.dataset_name
                    )
                    self._is_initialized = True
                    print(f"✅ Loaded {len(self.episodes)} episodes from {self.dataset_name}")
                    return
                else:
                    raise

            # Convert OCTO dataset format to episodes
            print("Loading episodes from dataset...")
            self.episodes = self._load_episodes_from_octo_dataset(self.dataset)

            self._is_initialized = True
            print(f"✅ Loaded {len(self.episodes)} episodes from {self.dataset_name}")

        except Exception as e:
            # Surface a helpful message for the common case (dataset not present
            # or download blocked). The caller can switch to synthetic debug or
            # provide a local TFDS data_dir.
            raise RuntimeError(
                f"Failed to load OpenX dataset '{self.dataset_name}': {e}\n\n"
                "Hints:\n"
                " - Use the synthetic debug dataset for fast tests: set config['use_synthetic_debug']=True or run with --use-debug.\n"
                " - List available OXE dataset names: scripts/octo/list_oxe_datasets.py\n"
                " - If you need the full dataset, download it into your TFDS data_dir or provide data_path pointing to local TFDS data.\n"
            )
    
    def _load_episodes_from_octo_dataset(self, dataset) -> List[Dict[str, Any]]:
        """
        Load episodes from OCTO dataset format.
        
        OCTO dataset format (from make_single_dataset):
        - Yields trajectories with keys: 'observation', 'action', 'task'
        - 'observation': dict with keys like 'image_primary', 'image_wrist', etc.
        - 'action': array of shape (traj_len, window_size, action_dim)
        - 'task': dict with 'language_instruction' and other task info
        
        Our internal format:
        - 'observation': dict with timestep-indexed observations
        - 'action': array of actions over time (traj_len, action_dim)
        - 'task': task metadata
        """
        episodes = []
        iterator = dataset.iterator()
        
        print(f"Loading up to {self.max_episodes} episodes...")
        for i in range(self.max_episodes):
            try:
                traj = next(iterator)
                
                # OCTO format already has the structure we need
                # Just need to squeeze out the window dimension from actions
                # Actions shape: (traj_len, window_size, action_dim) -> (traj_len, action_dim)
                action = traj['action']
                if len(action.shape) == 3 and action.shape[1] == 1:
                    # Squeeze out window dimension
                    action = action.squeeze(1)
                
                # Convert to numpy if needed
                if hasattr(action, 'numpy'):
                    action = action.numpy()
                
                # Process observations - convert to numpy
                obs_dict = {}
                for key, value in traj['observation'].items():
                    if hasattr(value, 'numpy'):
                        obs_dict[key] = value.numpy()
                    else:
                        obs_dict[key] = np.array(value)
                    
                    # Squeeze window dimension from observations too
                    if len(obs_dict[key].shape) > 2 and obs_dict[key].shape[1] == 1:
                        obs_dict[key] = obs_dict[key].squeeze(1)
                
                # Process task info
                task_dict = {}
                for key, value in traj['task'].items():
                    if hasattr(value, 'numpy'):
                        task_dict[key] = value.numpy()
                    else:
                        task_dict[key] = value
                
                episode = {
                    'observation': obs_dict,
                    'action': action,
                    'task': task_dict
                }
                
                episodes.append(episode)
                
                if (i + 1) % 10 == 0:
                    print(f"  Loaded {i + 1}/{self.max_episodes} episodes...")
                    
            except StopIteration:
                print(f"  Reached end of dataset after {i} episodes")
                break
            except Exception as e:
                print(f"  Warning: Error loading episode {i}: {e}")
                continue
        
        return episodes
    
    def _load_episodes_from_tfds_builder(self, dataset, dataset_name: str) -> List[Dict[str, Any]]:
        """
        Load episodes from a TFDS builder's raw dataset.
        
        This is used for datasets like bridge_dataset that don't have a registered TFDS builder
        but exist as downloaded TFRecords that can be loaded with builder_from_directory.
        
        Args:
            dataset: Raw TFDS dataset from builder.as_dataset()
            dataset_name: Name of the dataset (for logging)
        
        Returns:
            List of episodes in our internal format
        """
        import tensorflow as tf
        from octo.data.oxe.oxe_dataset_configs import OXE_DATASET_CONFIGS
        
        # Get OXE config for this dataset to know how to map observation keys
        oxe_config = OXE_DATASET_CONFIGS.get(dataset_name, {})
        image_obs_keys = oxe_config.get("image_obs_keys", {})
        
        episodes = []
        print(f"Loading up to {self.max_episodes} episodes from raw TFDS...")
        
        for i, episode in enumerate(dataset.take(self.max_episodes)):
            try:
                # RLDS format: episode has 'steps'
                steps = list(episode['steps'])
                
                # Collect observations and actions
                observations = []
                actions = []
                
                for step in steps:
                    # Extract observation
                    obs = {}
                    for key, value in step['observation'].items():
                        if hasattr(value, 'numpy'):
                            obs[key] = value.numpy()
                        else:
                            obs[key] = np.array(value)
                    observations.append(obs)
                    
                    # Extract action
                    action = step['action']
                    if hasattr(action, 'numpy'):
                        action = action.numpy()
                    else:
                        action = np.array(action)
                    actions.append(action)
                
                # Stack into trajectory format
                action_array = np.stack(actions)
                
                # Consolidate observations into timestep-indexed dicts
                obs_dict = {}
                if observations:
                    # Get all unique keys across observations
                    all_keys = set()
                    for obs in observations:
                        all_keys.update(obs.keys())
                    
                    # Stack each key
                    for key in all_keys:
                        values = [obs.get(key, None) for obs in observations]
                        # Filter out None values
                        values = [v for v in values if v is not None]
                        if values:
                            try:
                                obs_dict[key] = np.stack(values)
                            except:
                                obs_dict[key] = np.array(values)
                
                # Map raw observation keys to standardized OCTO keys
                # This mimics what OCTO's standardization transforms do
                standardized_obs = {}
                
                # Map image observations
                if image_obs_keys:
                    # Map primary image
                    if image_obs_keys.get("primary"):
                        raw_key = image_obs_keys["primary"]
                        if raw_key in obs_dict:
                            standardized_obs["image_primary"] = obs_dict[raw_key]
                    
                    # Map secondary image
                    if image_obs_keys.get("secondary"):
                        raw_key = image_obs_keys["secondary"]
                        if raw_key in obs_dict:
                            standardized_obs["image_secondary"] = obs_dict[raw_key]
                    
                    # Map wrist image
                    if image_obs_keys.get("wrist"):
                        raw_key = image_obs_keys["wrist"]
                        if raw_key in obs_dict:
                            standardized_obs["image_wrist"] = obs_dict[raw_key]
                
                # Keep proprioception (state) with its original key
                if "state" in obs_dict:
                    standardized_obs["state"] = obs_dict["state"]
                
                # If no mapping config available, keep all original keys
                if not standardized_obs:
                    standardized_obs = obs_dict
                
                # Extract task info (language instruction if available)
                task_dict = {}
                if 'language_instruction' in episode:
                    lang = episode['language_instruction']
                    if hasattr(lang, 'numpy'):
                        task_dict['language_instruction'] = lang.numpy()
                    else:
                        task_dict['language_instruction'] = lang
                elif steps and 'language_instruction' in steps[0]:
                    lang = steps[0]['language_instruction']
                    if hasattr(lang, 'numpy'):
                        task_dict['language_instruction'] = lang.numpy()
                    else:
                        task_dict['language_instruction'] = lang
                
                episode_data = {
                    'observation': standardized_obs,  # Use standardized keys
                    'action': action_array,
                    'task': task_dict
                }
                
                episodes.append(episode_data)
                
                if (i + 1) % 10 == 0:
                    print(f"  Loaded {i + 1}/{self.max_episodes} episodes...")
                    
            except Exception as e:
                print(f"  Warning: Error loading episode {i}: {e}")
                continue
        
        return episodes
    
    def _convert_rlds_to_episodes(self, dataset) -> List[Dict[str, Any]]:
        """
        Convert RLDS format episodes to our internal format.
        (DEPRECATED: Use _load_episodes_from_octo_dataset instead)
        
        RLDS format:
        - Each episode has 'steps' (a dataset of steps)
        - Each step has 'observation', 'action', 'reward', etc.
        
        Our format:
        - 'observation': dict with timestep-indexed observations
        - 'action': array of actions over time
        - 'task': task metadata (language instruction, etc.)
        """
        import tensorflow as tf
        
        episodes = []
        for i, episode in enumerate(dataset.take(self.max_episodes)):
            # Extract all steps from this episode
            steps = list(episode['steps'])
            
            # Initialize structures
            observations = {
                'image_primary': [],
                'image_wrist': [],
                'proprio': []
            }
            actions = []
            task_info = {}
            language_instruction = None
            
            # Iterate through timesteps
            for step in steps:
                # Extract action - handle both tensor, numpy, and dict formats
                action = step['action']
                
                # If action is a dict, apply OCTO's standardization transforms
                # Different datasets have different action formats - we follow OCTO's conventions
                if isinstance(action, dict):
                    # Check if this is an RT-1/Kuka/Fractal style action
                    # Standard format: world_vector (3) + rotation_delta (3) + gripper (1) = 7D
                    if 'world_vector' in action and 'rotation_delta' in action:
                        world_vec = action['world_vector']
                        rot_delta = action['rotation_delta']
                        gripper = action.get('gripper_closedness_action', None)
                        
                        if hasattr(world_vec, 'numpy'):
                            world_vec = world_vec.numpy()
                        if hasattr(rot_delta, 'numpy'):
                            rot_delta = rot_delta.numpy()
                        if gripper is not None and hasattr(gripper, 'numpy'):
                            gripper = gripper.numpy()
                        elif gripper is None:
                            gripper = np.array([0.0])  # Default closed gripper
                        
                        # Concatenate: world_vector(3) + rotation_delta(3) + gripper(1) = 7D
                        action = np.concatenate([
                            np.array(world_vec).flatten(),
                            np.array(rot_delta).flatten(),
                            np.array(gripper).flatten()
                        ])
                    else:
                        # Fallback: concatenate all components
                        action_components = []
                        for key in sorted(action.keys()):
                            component = action[key]
                            if hasattr(component, 'numpy'):
                                component = component.numpy()
                            action_components.append(np.array(component).flatten())
                        action = np.concatenate(action_components)
                elif hasattr(action, 'numpy'):
                    action = action.numpy()
                
                actions.append(np.array(action))
                
                # Extract observations
                obs = step['observation']
                
                # Handle images - different datasets use different key names
                # Try various common image keys
                for img_key in ['image', 'image_primary', 'rgb', 'pixels']:
                    if img_key in obs:
                        img = obs[img_key]
                        if hasattr(img, 'numpy'):
                            img = img.numpy()
                        observations['image_primary'].append(img)
                        break
                
                for wrist_key in ['wrist_image', 'image_wrist', 'wrist_rgb']:
                    if wrist_key in obs:
                        wrist_img = obs[wrist_key]
                        if hasattr(wrist_img, 'numpy'):
                            wrist_img = wrist_img.numpy()
                        observations['image_wrist'].append(wrist_img)
                        break
                
                # Handle state/proprioception
                for state_key in ['state', 'proprio', 'robot_state', 'qpos']:
                    if state_key in obs:
                        state = obs[state_key]
                        if hasattr(state, 'numpy'):
                            state = state.numpy()
                        observations['proprio'].append(state)
                        break
                
                # Extract language instruction from first observation
                if language_instruction is None:
                    for lang_key in ['natural_language_instruction', 'language_instruction', 'task']:
                        if lang_key in obs:
                            lang = obs[lang_key]
                            if hasattr(lang, 'numpy'):
                                lang = lang.numpy()
                            if isinstance(lang, bytes):
                                language_instruction = lang.decode('utf-8')
                            else:
                                language_instruction = str(lang)
                            break
            
            # Store language instruction in task info
            if language_instruction:
                task_info['language_instruction'] = language_instruction
            
            # Also check episode-level attributes
            if 'attributes' in episode:
                attrs = episode['attributes']
                if 'natural_language_instruction' in attrs and language_instruction is None:
                    lang = attrs['natural_language_instruction']
                    if hasattr(lang, 'numpy'):
                        lang = lang.numpy()
                    if isinstance(lang, bytes):
                        task_info['language_instruction'] = lang.decode('utf-8')
                    else:
                        task_info['language_instruction'] = str(lang)
            
            # Convert lists to numpy arrays
            episode_dict = {
                'observation': {k: np.array(v) if v else None for k, v in observations.items()},
                'action': np.array(actions),
                'task': task_info
            }
            
            # Remove None/empty values
            episode_dict['observation'] = {k: v for k, v in episode_dict['observation'].items() if v is not None and len(v) > 0}
            
            episodes.append(episode_dict)
        
        return episodes
    
    def get_task_list(self) -> List[Dict[str, Any]]:
        """
        Get list of tasks.
        
        For OpenX, we treat the dataset as a single "task" with multiple episodes.
        This can be extended to group by language instruction or scene.
        """
        return [{
            'task_id': 'openx_eval',
            'task_name': f'{self.dataset_name} Evaluation',
            'task_description': 'Offline action prediction on Open X-Embodiment data',
            'dataset_path': str(self.data_path) if self.data_path else 'tfds',
            'num_episodes': len(self.episodes)
        }]
    
    def load_episode(self, task_id: str, episode_idx: int) -> Dict[str, Any]:
        """
        Load a specific episode from the dataset.
        
        Args:
            task_id: Task identifier (ignored for single-task benchmark)
            episode_idx: Episode index
        
        Returns:
            Episode data with observations, actions, and metadata
        """
        if episode_idx >= len(self.episodes):
            raise IndexError(f"Episode {episode_idx} out of range (max: {len(self.episodes)})")
        
        traj = self.episodes[episode_idx]
        
        # OCTO data format:
        # traj['observation'] contains images and state
        # traj['action'] contains actions
        # traj['task'] contains language instructions
        
        observations = []
        actions = []
        
        # Get trajectory length
        traj_len = len(traj['action'])
        
        for t in range(traj_len):
            # Extract observation at timestep t
            obs_dict = {}
            
            # Primary image (required)
            if 'image_primary' in traj['observation']:
                # Shape: (traj_len, window_size, H, W, C)
                image = traj['observation']['image_primary'][t]
                if len(image.shape) == 4:  # Has window dimension
                    image = image[0]  # Take first in window
                obs_dict['image_primary'] = np.array(image)
            
            # Secondary/wrist images (optional)
            if 'image_wrist' in traj['observation']:
                image = traj['observation']['image_wrist'][t]
                if len(image.shape) == 4:
                    image = image[0]
                obs_dict['image_wrist'] = np.array(image)
            
            # Proprioceptive state (optional)
            if 'proprio' in traj['observation']:
                obs_dict['state'] = np.array(traj['observation']['proprio'][t])
            
            observations.append(obs_dict)
            actions.append(np.array(traj['action'][t]))
        
        # Extract language instruction
        task_description = None
        if self.use_language and 'language_instruction' in traj['task']:
            lang = traj['task']['language_instruction']
            if isinstance(lang, bytes):
                task_description = lang.decode('utf-8')
            elif isinstance(lang, str):
                task_description = lang
            else:
                task_description = str(lang)
        
        episode_data = {
            'observations': observations,
            'actions': np.array(actions),  # (T, action_dim)
            'task_description': task_description,
            'episode_length': traj_len,
            'episode_idx': episode_idx,
            'dataset_name': self.dataset_name
        }
        
        return episode_data
    
    def evaluate_episode(
        self,
        adapter,
        episode_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate model on a single episode (offline).
        
        Args:
            adapter: Model adapter
            episode_data: Episode data from load_episode()
            **kwargs: Additional parameters
        
        Returns:
            Episode results with metrics and predictions
        """
        observations = episode_data['observations']
        ground_truth_actions = episode_data['actions']
        task_description = episode_data['task_description']
        
        predicted_actions = []
        inference_times = []
        
        # Predict action for each observation
        import time
        for obs in observations:
            start_time = time.time()
            
            predicted_action = adapter.get_action(
                obs,
                task_description=task_description,
                **kwargs
            )
            
            inference_time = time.time() - start_time
            
            predicted_actions.append(predicted_action)
            inference_times.append(inference_time)
        
        predicted_actions = np.array(predicted_actions)
        
        # Compute metrics
        metrics = self.compute_metrics(
            predicted_actions,
            ground_truth_actions,
            episode_data
        )
        
        # Determine success (MSE below threshold)
        success = metrics['action_mse'] < self.action_threshold
        
        result = {
            'success': success,
            'metrics': metrics,
            'predicted_actions': predicted_actions.tolist(),  # For JSON serialization
            'execution_time': np.mean(inference_times),
            'total_steps': len(observations)
        }
        
        return result
    
    def compute_metrics(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        episode_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            predictions: Predicted actions (T, action_dim)
            ground_truth: Ground truth actions (T, action_dim)
            episode_data: Episode data for context
        
        Returns:
            Dictionary of metrics
        """
        # Ensure same shape
        min_len = min(len(predictions), len(ground_truth))
        predictions = predictions[:min_len]
        ground_truth = ground_truth[:min_len]
        
        # Mean Squared Error
        mse = np.mean((predictions - ground_truth) ** 2)
        
        # Mean Absolute Error (L1)
        mae = np.mean(np.abs(predictions - ground_truth))
        
        # Per-dimension MSE
        mse_per_dim = np.mean((predictions - ground_truth) ** 2, axis=0)
        
        # Cosine similarity
        norm_pred = predictions / (np.linalg.norm(predictions, axis=1, keepdims=True) + 1e-8)
        norm_gt = ground_truth / (np.linalg.norm(ground_truth, axis=1, keepdims=True) + 1e-8)
        cosine_sim = np.mean(np.sum(norm_pred * norm_gt, axis=1))
        
        metrics = {
            'action_mse': float(mse),
            'action_mae': float(mae),
            'cosine_similarity': float(cosine_sim),
            'mse_per_dim_mean': float(np.mean(mse_per_dim)),
            'mse_per_dim_std': float(np.std(mse_per_dim)),
        }
        
        return metrics
    
    def __repr__(self) -> str:
        return f"OpenXBenchmark(dataset='{self.dataset_name}', episodes={len(self.episodes)}, initialized={self._is_initialized})"
