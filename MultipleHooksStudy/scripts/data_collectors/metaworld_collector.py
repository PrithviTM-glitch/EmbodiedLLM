"""
MetaWorld data collector for gradient analysis.

Collects real observations from MetaWorld benchmark tasks.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import gymnasium as gym

from scripts.data_collectors.base_collector import BenchmarkDataCollector


class MetaWorldDataCollector(BenchmarkDataCollector):
    """
    Collects observations from MetaWorld benchmark.
    
    MetaWorld provides:
    - RGB images from camera
    - Proprioceptive state (end-effector pose, object positions, etc.)
    - 50 diverse manipulation tasks (MT50)
    """
    
    def __init__(
        self,
        benchmark_version: str = "ML10",  # MT50, ML10, ML1, etc
        task_names: Optional[List[str]] = None,
        image_size: int = 224,
        output_dir: str = "/content/benchmark_observations",
        num_samples: int = 10,
        seed: int = 42,
        use_goal_observable: bool = True
    ):
        """
        Initialize MetaWorld data collector.
        
        Args:
            benchmark_version: MetaWorld benchmark (MT50, ML10, ML1)
            task_names: Specific tasks to sample from (None = all tasks)
            image_size: Image resolution (H, W)
            output_dir: Directory to save observations
            num_samples: Number of samples to collect
            seed: Random seed
            use_goal_observable: Whether to use goal-observable variant
        """
        super().__init__(
            benchmark_name=f"metaworld_{benchmark_version.lower()}",
            output_dir=output_dir,
            num_samples=num_samples,
            seed=seed
        )
        
        self.benchmark_version = benchmark_version
        self.task_names = task_names
        self.image_size = image_size
        self.use_goal_observable = use_goal_observable
        
        self.benchmark = None
        self.envs = {}
        self.task_list = []
        self.current_task_idx = 0
    
    def setup_environment(self) -> None:
        """Setup MetaWorld environment."""
        import metaworld
        
        print(f"   Loading MetaWorld benchmark: {self.benchmark_version}")
        
        # Get benchmark
        if self.benchmark_version == "MT50":
            self.benchmark = metaworld.MT50(seed=self.seed)
        elif self.benchmark_version == "ML10":
            self.benchmark = metaworld.ML10(seed=self.seed)
        elif self.benchmark_version == "ML1":
            self.benchmark = metaworld.ML1("pick-place-v3", seed=self.seed)
        else:
            raise ValueError(f"Unknown benchmark: {self.benchmark_version}")
        
        # Get task list
        if self.task_names:
            self.task_list = [(name, env_cls) for name, env_cls in self.benchmark.train_classes.items() 
                             if name in self.task_names]
        else:
            self.task_list = list(self.benchmark.train_classes.items())
        
        print(f"   Benchmark: {len(self.task_list)} tasks available")
        
        # Create environments for each task
        for task_name, env_cls in self.task_list[:5]:  # Create first 5 to save memory
            # Add goal-observable suffix if requested
            env_name = task_name
            if self.use_goal_observable and not task_name.endswith("-goal-observable"):
                # Try goal-observable version
                try:
                    env = env_cls(render_mode='rgb_array', 
                                 camera_name='corner',
                                 width=self.image_size,
                                 height=self.image_size)
                    env._partially_observable = False  # Make goal observable
                except:
                    env = env_cls(render_mode='rgb_array',
                                 camera_name='corner',
                                 width=self.image_size,
                                 height=self.image_size)
            else:
                env = env_cls(render_mode='rgb_array',
                             camera_name='corner',
                             width=self.image_size,
                             height=self.image_size)
            
            env.seed(self.seed)
            self.envs[task_name] = env
            
        print(f"   Created {len(self.envs)} environments")
    
    def collect_observation(self) -> Dict[str, np.ndarray]:
        """Collect one observation from MetaWorld."""
        # Select task (round-robin through available tasks)
        task_name, _ = self.task_list[self.current_task_idx]
        
        # Create env if not exists
        if task_name not in self.envs:
            _, env_cls = self.task_list[self.current_task_idx]
            env = env_cls(render_mode='rgb_array',
                         camera_name='corner', 
                         width=self.image_size,
                         height=self.image_size)
            env.seed(self.seed + self.current_task_idx)
            self.envs[task_name] = env
        
        env = self.envs[task_name]
        
        # Get a task from the benchmark
        task = metaworld.Task(env_name=task_name)
        env.set_task(task)
        
        # Reset to get observation
        obs, info = env.reset()
        
        # Render image
        image = env.render()
        
        observation = {}
        
        # Image
        if image is not None:
            observation['image'] = image  # Already (H, W, 3)
        else:
            # Fallback: create dummy image if rendering fails
            observation['image'] = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Proprioceptive state
        # MetaWorld obs is a vector containing:
        # [ee_pos (3), ee_quat (4), gripper_state (1), obj_pos (3), obj_quat (4), ...]
        observation['robot_state'] = obs.astype(np.float32)
        
        # Task name as description
        observation['task_description'] = task_name
        
        # Rotate through tasks
        self.current_task_idx = (self.current_task_idx + 1) % len(self.task_list)
        
        return observation
    
    def get_observation_spec(self) -> Dict[str, Dict[str, Any]]:
        """Get MetaWorld observation specification."""
        return {
            'image': {
                'shape': (self.image_size, self.image_size, 3),
                'dtype': 'uint8',
                'description': 'RGB image from corner camera'
            },
            'robot_state': {
                'shape': (39,),  # Standard MetaWorld observation dimension
                'dtype': 'float32',
                'description': 'State vector [ee_pos, ee_quat, gripper, obj_pos, obj_quat, goal, ...]'
            },
            'task_description': {
                'type': 'string',
                'description': 'Task name (e.g., reach-v3, push-v3, pick-place-v3)'
            }
        }
    
    def __del__(self):
        """Cleanup."""
        for env in self.envs.values():
            env.close()


def main():
    """Test MetaWorld data collection."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect MetaWorld observations')
    parser.add_argument('--benchmark', type=str, default='ML10',
                       choices=['MT50', 'ML10', 'ML1'],
                       help='MetaWorld benchmark')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of observations to collect')
    parser.add_argument('--output-dir', type=str, default='/content/benchmark_observations',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    collector = MetaWorldDataCollector(
        benchmark_version=args.benchmark,
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
