"""
Bridge/OpenX data collector for gradient analysis.

Collects real observations from Bridge V2 and other OpenX datasets.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import tensorflow_datasets as tfds

from scripts.data_collectors.base_collector import BenchmarkDataCollector


class BridgeDataCollector(BenchmarkDataCollector):
    """
    Collects observations from Bridge V2 and OpenX datasets.
    
    Bridge/OpenX provides:
    - RGB images from robot cameras
    - Proprioceptive state (joint positions, end-effector pose)
    - Natural language task instructions
    """
    
    def __init__(
        self,
        dataset_name: str = "bridge_dataset",
        split: str = "train[:1000]",  # Limit to avoid loading entire dataset
        image_key: str = "observation/image_0",
        state_key: str = "observation/state",
        language_key: str = "observation/natural_language_instruction",
        image_size: int = 224,
        output_dir: str = "/content/benchmark_observations",
        num_samples: int = 10,
        seed: int = 42
    ):
        """
        Initialize Bridge/OpenX data collector.
        
        Args:
            dataset_name: TFDS dataset name (e.g., 'bridge_dataset', 'fractal20220817_data')
            split: Dataset split to load
            image_key: Key for image in observation dict
            state_key: Key for state in observation dict
            language_key: Key for language instruction
            image_size: Target image resolution
            output_dir: Directory to save observations
            num_samples: Number of samples to collect
            seed: Random seed
        """
        super().__init__(
            benchmark_name=f"openx_{dataset_name}",
            output_dir=output_dir,
            num_samples=num_samples,
            seed=seed
        )
        
        self.dataset_name = dataset_name
        self.split = split
        self.image_key = image_key
        self.state_key = state_key
        self.language_key = language_key
        self.image_size = image_size
        
        self.dataset = None
        self.dataset_iterator = None
        self.current_episode = None
        self.episode_step = 0
    
    def setup_environment(self) -> None:
        """Setup Bridge/OpenX dataset."""
        import tensorflow as tf
        
        print(f"   Loading OpenX dataset: {self.dataset_name}")
        print(f"   Split: {self.split}")
        
        try:
            # Load dataset
            builder = tfds.builder(self.dataset_name)
            self.dataset = builder.as_dataset(
                split=self.split,
                shuffle_files=True,
                read_config=tfds.ReadConfig(shuffle_seed=self.seed)
            )
            
            # Create iterator
            self.dataset_iterator = iter(self.dataset)
            
            print(f"   Dataset loaded successfully")
            
            # Get first episode to check structure
            self.current_episode = next(self.dataset_iterator)
            print(f"   Episode structure: {self.current_episode.keys()}")
            
            # Check if it's RLDS format (has 'steps')
            if 'steps' in self.current_episode:
                print(f"   RLDS format detected (episodic)")
                self.episode_step = 0
            else:
                print(f"   Flat format detected (step-level)")
                
        except Exception as e:
            print(f"   Error loading dataset: {e}")
            print(f"   Falling back to placeholder data")
            self.dataset = None
    
    def _extract_from_step(self, step: Dict) -> Dict[str, np.ndarray]:
        """Extract observation from a single step."""
        import tensorflow as tf
        
        observation = {}
        
        # Extract image
        if self.image_key in step:
            image = step[self.image_key].numpy()
        elif 'observation' in step and self.image_key.split('/')[-1] in step['observation']:
            image = step['observation'][self.image_key.split('/')[-1]].numpy()
        else:
            # Try common alternatives
            for key in ['image', 'image_0', 'rgb']:
                if key in step:
                    image = step[key].numpy()
                    break
                elif 'observation' in step and key in step['observation']:
                    image = step['observation'][key].numpy()
                    break
            else:
                image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Resize if needed
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            import cv2
            image = cv2.resize(image, (self.image_size, self.image_size))
        
        observation['image'] = image
        
        # Extract state
        if self.state_key in step:
            state = step[self.state_key].numpy()
        elif 'observation' in step and self.state_key.split('/')[-1] in step['observation']:
            state = step['observation'][self.state_key.split('/')[-1]].numpy()
        else:
            # Try common alternatives
            for key in ['state', 'robot_state', 'proprio']:
                if key in step:
                    state = step[key].numpy()
                    break
                elif 'observation' in step and key in step['observation']:
                    state = step['observation'][key].numpy()
                    break
            else:
                state = np.zeros(7, dtype=np.float32)  # Default 7-dim state
        
        observation['robot_state'] = state.astype(np.float32)
        
        # Extract language instruction
        if self.language_key in step:
            language = step[self.language_key].numpy().decode('utf-8')
        elif 'observation' in step and self.language_key.split('/')[-1] in step['observation']:
            language = step['observation'][self.language_key.split('/')[-1]].numpy().decode('utf-8')
        else:
            # Try common alternatives
            for key in ['language_instruction', 'instruction', 'task']:
                if key in step:
                    lang_value = step[key].numpy()
                    language = lang_value.decode('utf-8') if isinstance(lang_value, bytes) else str(lang_value)
                    break
                elif 'observation' in step and key in step['observation']:
                    lang_value = step['observation'][key].numpy()
                    language = lang_value.decode('utf-8') if isinstance(lang_value, bytes) else str(lang_value)
                    break
            else:
                language = f"{self.dataset_name}_task"
        
        observation['task_description'] = language
        
        return observation
    
    def collect_observation(self) -> Dict[str, np.ndarray]:
        """Collect one observation from Bridge/OpenX."""
        if self.dataset is None:
            # Return placeholder if dataset failed to load
            return {
                'image': np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8),
                'robot_state': np.zeros(7, dtype=np.float32),
                'task_description': f"{self.dataset_name}_placeholder"
            }
        
        try:
            # Check if RLDS episodic format
            if 'steps' in self.current_episode:
                # Get next step from current episode
                steps = list(self.current_episode['steps'])
                
                if self.episode_step >= len(steps):
                    # Move to next episode
                    self.current_episode = next(self.dataset_iterator)
                    self.episode_step = 0
                    steps = list(self.current_episode['steps'])
                
                step = steps[self.episode_step]
                self.episode_step += 1
                
                return self._extract_from_step(step)
            else:
                # Flat format - each item is a step
                step = self.current_episode
                self.current_episode = next(self.dataset_iterator)
                
                return self._extract_from_step(step)
                
        except StopIteration:
            # Reset iterator
            self.dataset_iterator = iter(self.dataset)
            self.current_episode = next(self.dataset_iterator)
            self.episode_step = 0
            return self.collect_observation()
    
    def get_observation_spec(self) -> Dict[str, Dict[str, Any]]:
        """Get Bridge/OpenX observation specification."""
        return {
            'image': {
                'shape': (self.image_size, self.image_size, 3),
                'dtype': 'uint8',
                'description': 'RGB image from robot camera'
            },
            'robot_state': {
                'shape': (-1,),  # Variable dimension depending on robot
                'dtype': 'float32',
                'description': 'Proprioceptive state (joint positions, end-effector pose, etc.)'
            },
            'task_description': {
                'type': 'string',
                'description': 'Natural language task instruction'
            }
        }


def main():
    """Test Bridge/OpenX data collection."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect Bridge/OpenX observations')
    parser.add_argument('--dataset', type=str, default='bridge_dataset',
                       help='TFDS dataset name')
    parser.add_argument('--split', type=str, default='train[:100]',
                       help='Dataset split')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of observations to collect')
    parser.add_argument('--output-dir', type=str, default='/content/benchmark_observations',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    collector = BridgeDataCollector(
        dataset_name=args.dataset,
        split=args.split,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    dataset_path = collector.collect_dataset()
    
    # Load and visualize one sample
    print("\nLoading dataset for verification...")
    data = collector.load_dataset(dataset_path)
    
    if len(data['image']) > 0:
        sample = {key: val[0] for key, val in data.items()}
        collector.visualize_sample(sample)


if __name__ == '__main__':
    main()
