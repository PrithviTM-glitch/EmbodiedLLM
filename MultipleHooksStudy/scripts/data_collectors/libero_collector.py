"""
LIBERO data collector for gradient analysis.

Collects real observations from LIBERO benchmark tasks.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys

# Add LIBERO to path
libero_path = '/content/LIBERO'
if str(libero_path) not in sys.path:
    sys.path.insert(0, libero_path)

from scripts.data_collectors.base_collector import BenchmarkDataCollector


class LIBERODataCollector(BenchmarkDataCollector):
    """
    Collects observations from LIBERO benchmark.
    
    LIBERO provides:
    - RGB images from multiple cameras (agentview, eye_in_hand)
    - Proprioceptive state (gripper, joint positions, end-effector pose)
    - Language-conditioned tasks
    """
    
    def __init__(
        self,
        benchmark_subset: str = "libero_90",
        task_suite: str = "libero_spatial",
        camera_names: List[str] = None,
        image_size: int = 224,
        output_dir: str = "/content/benchmark_observations",
        num_samples: int = 10,
        seed: int = 42
    ):
        """
        Initialize LIBERO data collector.
        
        Args:
            benchmark_subset: LIBERO benchmark subset (libero_90, libero_10, etc.)
            task_suite: Task suite name (libero_spatial, libero_object, etc.)
            camera_names: List of camera names to use
            image_size: Image resolution (H, W)
            output_dir: Directory to save observations
            num_samples: Number of samples to collect
            seed: Random seed
        """
        super().__init__(
            benchmark_name=f"libero_{task_suite}",
            output_dir=output_dir,
            num_samples=num_samples,
            seed=seed
        )
        
        self.benchmark_subset = benchmark_subset
        self.task_suite = task_suite
        self.camera_names = camera_names or ['agentview']
        self.image_size = image_size
        
        self.env = None
        self.benchmark = None
        self.current_task_id = 0
    
    def setup_environment(self) -> None:
        """Setup LIBERO environment."""
        print(f"   Loading LIBERO benchmark: {self.benchmark_subset}")
        
        from libero.libero import get_libero_path
        from libero.libero.benchmark import get_benchmark
        from libero.libero.envs import OffScreenRenderEnv
        
        # Get benchmark
        self.benchmark = get_benchmark(self.benchmark_subset)(self.task_suite)
        n_tasks = self.benchmark.n_tasks
        
        print(f"   Benchmark: {n_tasks} tasks available")
        print(f"   Task suite: {self.task_suite}")
        
        # Initialize first task
        task = self.benchmark.get_task(0)
        task_description = task.language
        task_bddl_file = task.problem_folder / task.bddl_file
        
        print(f"   Task 0: {task_description}")
        
        # Create environment
        env_args = {
            "bddl_file_name": str(task_bddl_file),
            "camera_heights": self.image_size,
            "camera_widths": self.image_size,
            "camera_names": self.camera_names,
        }
        
        self.env = OffScreenRenderEnv(**env_args)
        self.env.seed(self.seed)
        print(f"   Environment created with cameras: {self.camera_names}")
    
    def collect_observation(self) -> Dict[str, np.ndarray]:
        """Collect one observation from LIBERO."""
        # Reset environment to get initial state
        if self.current_task_id >= self.benchmark.n_tasks:
            self.current_task_id = 0
        
        # Randomly reset to get diverse states
        obs = self.env.reset()
        
        # Extract observations
        observation = {}
        
        # Images from cameras
        images = []
        for camera_name in self.camera_names:
            cam_key = f"{camera_name}_image"
            if cam_key in obs:
                img = obs[cam_key]
                # LIBERO images are (H, W, 3), may need flipping
                if img.shape[0] > 0:
                    images.append(img)
        
        if images:
            # Stack multiple camera views or use first one
            observation['image'] = images[0] if len(images) == 1 else np.concatenate(images, axis=1)
        
        # Proprioceptive state
        state_components = []
        
        # Gripper state
        if 'gripper_states' in obs:
            state_components.append(obs['gripper_states'].flatten())
        
        # Joint positions
        if 'joint_states' in obs:
            state_components.append(obs['joint_states'].flatten())
        
        # End-effector state
        if 'ee_states' in obs:
            state_components.append(obs['ee_states'].flatten())
        
        if state_components:
            observation['robot_state'] = np.concatenate(state_components)
        
        # Task description
        task = self.benchmark.get_task(self.current_task_id)
        observation['task_description'] = task.language
        
        # Rotate through tasks
        self.current_task_id = (self.current_task_id + 1) % self.benchmark.n_tasks
        
        return observation
    
    def get_observation_spec(self) -> Dict[str, Dict[str, Any]]:
        """Get LIBERO observation specification."""
        return {
            'image': {
                'shape': (self.image_size, self.image_size * len(self.camera_names), 3),
                'dtype': 'uint8',
                'description': 'RGB images from cameras (concatenated if multiple)'
            },
            'robot_state': {
                'shape': (None,),  # Variable size depending on robot
                'dtype': 'float32',
                'description': 'Concatenated [gripper_state, joint_positions, ee_pose]'
            },
            'task_description': {
                'type': 'string',
                'description': 'Natural language task instruction'
            }
        }
    
    def __del__(self):
        """Cleanup."""
        if self.env is not None:
            self.env.close()


def main():
    """Test LIBERO data collection."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect LIBERO observations')
    parser.add_argument('--benchmark', type=str, default='libero_90',
                       help='LIBERO benchmark subset')
    parser.add_argument('--task-suite', type=str, default='libero_spatial',
                       help='Task suite name')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of observations to collect')
    parser.add_argument('--output-dir', type=str, default='/content/benchmark_observations',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    collector = LIBERODataCollector(
        benchmark_subset=args.benchmark,
        task_suite=args.task_suite,
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
