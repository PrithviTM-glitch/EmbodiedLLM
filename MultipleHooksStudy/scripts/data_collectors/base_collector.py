"""
Base class for collecting real observations from benchmark environments.

This provides a standardized interface for collecting observations that will
be used in gradient flow analysis across different VLA models.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import h5py
import json


class BenchmarkDataCollector(ABC):
    """
    Abstract base class for benchmark data collectors.
    
    Collects N observation samples from a benchmark environment and saves them
    in a standardized format for use in gradient analysis.
    """
    
    def __init__(
        self,
        benchmark_name: str,
        output_dir: str = "/content/benchmark_observations",
        num_samples: int = 10,
        seed: int = 42
    ):
        """
        Initialize data collector.
        
        Args:
            benchmark_name: Name of benchmark (e.g., "libero", "metaworld", "bridge")
            output_dir: Directory to save collected observations
            num_samples: Number of observation samples to collect
            seed: Random seed for reproducibility
        """
        self.benchmark_name = benchmark_name
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.seed = seed
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        np.random.seed(seed)
    
    @abstractmethod
    def setup_environment(self) -> None:
        """
        Setup the benchmark environment.
        
        This should initialize the environment, load any required datasets,
        and prepare for data collection.
        """
        pass
    
    @abstractmethod
    def collect_observation(self) -> Dict[str, np.ndarray]:
        """
        Collect a single observation from the environment.
        
        Returns:
            Dictionary containing:
                - 'image': RGB image(s) as np.ndarray (H, W, 3) or (N, H, W, 3)
                - 'state': Proprioceptive state as np.ndarray (state_dim,)
                - 'task_description': Optional text description of task
                - Any other benchmark-specific observations
        """
        pass
    
    @abstractmethod
    def get_observation_spec(self) -> Dict[str, Dict[str, Any]]:
        """
        Get specification of observation format.
        
        Returns:
            Dictionary describing observation structure:
            {
                'image': {'shape': (224, 224, 3), 'dtype': 'uint8'},
                'state': {'shape': (7,), 'dtype': 'float32'},
                ...
            }
        """
        pass
    
    def collect_dataset(self, tasks: Optional[List[str]] = None) -> str:
        """
        Collect full dataset of observations.
        
        Args:
            tasks: Optional list of specific tasks to collect from
        
        Returns:
            Path to saved dataset file
        """
        print(f"🔬 Collecting {self.num_samples} observations from {self.benchmark_name}")
        print("="*60)
        
        # Setup environment
        print("[1/3] Setting up environment...")
        self.setup_environment()
        print("✅ Environment ready")
        
        # Collect observations
        print(f"\n[2/3] Collecting observations...")
        observations = []
        
        for i in range(self.num_samples):
            try:
                obs = self.collect_observation()
                observations.append(obs)
                
                if (i + 1) % 5 == 0:
                    print(f"   Collected {i + 1}/{self.num_samples} observations")
            
            except Exception as e:
                print(f"⚠️  Error collecting observation {i}: {e}")
                continue
        
        print(f"✅ Collected {len(observations)} observations")
        
        # Save dataset
        print(f"\n[3/3] Saving dataset...")
        output_path = self._save_dataset(observations)
        print(f"✅ Dataset saved to: {output_path}")
        print("="*60)
        
        return output_path
    
    def _save_dataset(self, observations: List[Dict[str, np.ndarray]]) -> str:
        """
        Save collected observations to HDF5 file.
        
        Args:
            observations: List of observation dictionaries
        
        Returns:
            Path to saved file
        """
        output_file = self.output_dir / f"{self.benchmark_name}_observations.hdf5"
        
        with h5py.File(output_file, 'w') as f:
            # Metadata
            f.attrs['benchmark'] = self.benchmark_name
            f.attrs['num_samples'] = len(observations)
            f.attrs['seed'] = self.seed
            f.attrs['observation_spec'] = json.dumps(self.get_observation_spec())
            
            # Create datasets for each observation type
            if observations:
                sample_obs = observations[0]
                
                for key in sample_obs.keys():
                    # Stack observations along batch dimension
                    data_list = [obs[key] for obs in observations]
                    
                    # Handle different data types
                    if isinstance(data_list[0], str):
                        # String data (e.g., task descriptions)
                        dt = h5py.string_dtype(encoding='utf-8')
                        f.create_dataset(key, data=data_list, dtype=dt)
                    else:
                        # Numeric data
                        data_array = np.stack(data_list, axis=0)
                        f.create_dataset(key, data=data_array, compression='gzip')
        
        return str(output_file)
    
    def load_dataset(self, dataset_path: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Load a previously saved dataset.
        
        Args:
            dataset_path: Path to dataset file. If None, uses default path.
        
        Returns:
            Dictionary containing all observations
        """
        if dataset_path is None:
            dataset_path = self.output_dir / f"{self.benchmark_name}_observations.hdf5"
        
        observations = {}
        
        with h5py.File(dataset_path, 'r') as f:
            # Load metadata
            print(f"Loading {f.attrs['benchmark']} dataset:")
            print(f"  Samples: {f.attrs['num_samples']}")
            print(f"  Seed: {f.attrs['seed']}")
            
            # Load observation data
            for key in f.keys():
                observations[key] = f[key][:]
        
        return observations
    
    def visualize_sample(self, observation: Dict[str, np.ndarray]) -> None:
        """
        Visualize a sample observation (for debugging).
        
        Args:
            observation: Single observation dictionary
        """
        print("\nObservation Summary:")
        print("-" * 40)
        
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                if value.size < 20:
                    print(f"    values: {value}")
            else:
                print(f"  {key}: {value}")
