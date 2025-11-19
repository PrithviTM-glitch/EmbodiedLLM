"""
Base benchmark class for evaluating VLA models.

This module provides an abstract base class for benchmarking VLA models
on different datasets and tasks.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import json
import time
from datetime import datetime
import numpy as np


class BaseBenchmark(ABC):
    """
    Abstract base class for VLA benchmarks.
    
    A benchmark defines:
    - How to load and prepare evaluation data
    - Evaluation metrics and success criteria
    - How to run evaluation episodes
    - How to collect and save results
    """
    
    def __init__(
        self,
        benchmark_name: str,
        data_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the benchmark.
        
        Args:
            benchmark_name: Name of the benchmark (e.g., 'openx', 'libero90')
            data_path: Path to benchmark dataset
            config: Optional configuration dictionary
        """
        self.benchmark_name = benchmark_name
        self.data_path = data_path
        self.config = config or {}
        self.results = []
        self._is_initialized = False
    
    @abstractmethod
    def setup(self) -> None:
        """
        Set up the benchmark environment/dataset.
        
        This should:
        1. Load/download dataset if needed
        2. Initialize evaluation environment(s)
        3. Prepare task configurations
        4. Set up any required resources
        
        Raises:
            RuntimeError: If setup fails
        """
        pass
    
    @abstractmethod
    def get_task_list(self) -> List[Dict[str, Any]]:
        """
        Get list of tasks to evaluate.
        
        Returns:
            List of task dictionaries, each containing:
                - 'task_id': Unique identifier
                - 'task_name': Human-readable name
                - 'task_description': Natural language description (if applicable)
                - 'dataset_path': Path to task-specific data
                - Additional task-specific metadata
        """
        pass
    
    @abstractmethod
    def load_episode(self, task_id: str, episode_idx: int) -> Dict[str, Any]:
        """
        Load a specific evaluation episode.
        
        Args:
            task_id: Task identifier
            episode_idx: Episode index within the task
        
        Returns:
            Episode data dictionary containing:
                - 'observations': List/array of observations
                - 'actions': Ground truth actions (if available)
                - 'task_description': Task description
                - 'initial_state': Initial robot/environment state
                - Additional episode metadata
        """
        pass
    
    @abstractmethod
    def evaluate_episode(
        self,
        adapter,
        episode_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate model on a single episode.
        
        Args:
            adapter: Model adapter instance
            episode_data: Episode data from load_episode()
            **kwargs: Additional evaluation parameters
        
        Returns:
            Results dictionary containing:
                - 'success': Boolean indicating task success
                - 'metrics': Dict of evaluation metrics
                - 'predicted_actions': Actions predicted by the model
                - 'execution_time': Time taken for inference
                - Additional episode-specific results
        """
        pass
    
    @abstractmethod
    def compute_metrics(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        episode_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics for predictions.
        
        Args:
            predictions: Predicted actions
            ground_truth: Ground truth actions
            episode_data: Episode data for context
        
        Returns:
            Dictionary of metrics (e.g., MSE, success rate, etc.)
        """
        pass
    
    def run_evaluation(
        self,
        adapter,
        num_episodes_per_task: Optional[int] = None,
        tasks: Optional[List[str]] = None,
        save_results: bool = True,
        results_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run full benchmark evaluation.
        
        Args:
            adapter: Model adapter to evaluate
            num_episodes_per_task: Number of episodes per task (None = all)
            tasks: List of task IDs to evaluate (None = all)
            save_results: Whether to save results to disk
            results_dir: Directory to save results
            **kwargs: Additional parameters passed to evaluate_episode()
        
        Returns:
            Aggregated results dictionary
        """
        if not self._is_initialized:
            raise RuntimeError("Benchmark not initialized. Call setup() first.")
        
        if not adapter.is_loaded:
            raise RuntimeError("Adapter model not loaded. Call adapter.load_model() first.")
        
        # Get tasks to evaluate
        all_tasks = self.get_task_list()
        if tasks is not None:
            all_tasks = [t for t in all_tasks if t['task_id'] in tasks]
        
        print(f"\n{'='*80}")
        print(f"Running {self.benchmark_name} Benchmark")
        print(f"Model: {adapter.__class__.__name__}")
        print(f"Tasks: {len(all_tasks)}")
        print(f"{'='*80}\n")
        
        task_results = []
        start_time = time.time()
        
        # Evaluate each task
        for task_idx, task in enumerate(all_tasks):
            print(f"\n[{task_idx+1}/{len(all_tasks)}] Evaluating task: {task['task_name']}")
            print(f"  Task ID: {task['task_id']}")
            
            task_result = self._evaluate_task(
                adapter,
                task,
                num_episodes_per_task,
                **kwargs
            )
            task_results.append(task_result)
        
        total_time = time.time() - start_time
        
        # Aggregate results
        aggregated_results = self._aggregate_results(task_results, total_time)
        
        # Save results
        if save_results:
            self._save_results(aggregated_results, results_dir)
        
        # Print summary
        self._print_summary(aggregated_results)
        
        return aggregated_results
    
    def _evaluate_task(
        self,
        adapter,
        task: Dict[str, Any],
        num_episodes: Optional[int],
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate a single task."""
        task_id = task['task_id']
        
        # Determine number of episodes
        max_episodes = task.get('num_episodes', 100)
        num_episodes = num_episodes if num_episodes is not None else max_episodes
        num_episodes = min(num_episodes, max_episodes)
        
        episode_results = []
        
        for ep_idx in range(num_episodes):
            # Reset adapter state
            adapter.reset()
            
            # Load episode
            episode_data = self.load_episode(task_id, ep_idx)
            
            # Evaluate
            import time
            ep_start = time.time()
            result = self.evaluate_episode(adapter, episode_data, **kwargs)
            ep_time = time.time() - ep_start
            
            result['episode_idx'] = ep_idx
            episode_results.append(result)
            
            # Progress - print after each episode
            successes = sum(r['success'] for r in episode_results)
            success_rate = successes / len(episode_results)
            
            # Estimate time remaining
            avg_time_per_ep = sum(time.time() - ep_start for _ in episode_results) / len(episode_results)
            remaining_eps = num_episodes - (ep_idx + 1)
            eta_seconds = avg_time_per_ep * remaining_eps
            eta_str = f"{int(eta_seconds//60)}m {int(eta_seconds%60)}s" if eta_seconds > 60 else f"{int(eta_seconds)}s"
            
            status = "✅" if result['success'] else "❌"
            print(f"    {status} Episode {ep_idx+1}/{num_episodes} | "
                  f"Steps: {result.get('total_steps', '?'):3d} | "
                  f"MSE: {result['metrics'].get('action_mse', 0):.4f} | "
                  f"Time: {ep_time:.1f}s | "
                  f"Success: {successes}/{ep_idx+1} ({success_rate:.1%}) | "
                  f"ETA: {eta_str}")
        
        # Aggregate task metrics
        task_result = {
            'task_id': task_id,
            'task_name': task['task_name'],
            'num_episodes': len(episode_results),
            'episode_results': episode_results,
            'success_rate': sum(r['success'] for r in episode_results) / len(episode_results),
            'avg_metrics': self._average_metrics([r['metrics'] for r in episode_results])
        }
        
        return task_result
    
    def _aggregate_results(
        self,
        task_results: List[Dict[str, Any]],
        total_time: float
    ) -> Dict[str, Any]:
        """Aggregate results across all tasks."""
        total_episodes = sum(t['num_episodes'] for t in task_results)
        total_successes = sum(
            sum(r['success'] for r in t['episode_results'])
            for t in task_results
        )
        
        aggregated = {
            'benchmark_name': self.benchmark_name,
            'timestamp': datetime.now().isoformat(),
            'total_tasks': len(task_results),
            'total_episodes': total_episodes,
            'total_successes': total_successes,
            'overall_success_rate': total_successes / total_episodes if total_episodes > 0 else 0.0,
            'total_time_seconds': total_time,
            'avg_time_per_episode': total_time / total_episodes if total_episodes > 0 else 0.0,
            'task_results': task_results,
            'per_task_success_rates': {
                t['task_id']: t['success_rate'] for t in task_results
            }
        }
        
        return aggregated
    
    def _average_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Average metrics across episodes."""
        if not metrics_list:
            return {}
        
        avg_metrics = {}
        keys = metrics_list[0].keys()
        
        for key in keys:
            values = [m[key] for m in metrics_list if key in m]
            avg_metrics[key] = np.mean(values) if values else 0.0
        
        return avg_metrics
    
    def _save_results(
        self,
        results: Dict[str, Any],
        results_dir: Optional[str] = None
    ) -> None:
        """Save results to JSON file."""
        if results_dir is None:
            results_dir = f"results/{self.benchmark_name}"
        
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = results_path / f"{self.benchmark_name}_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Results saved to: {filename}")
    
    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Print evaluation summary."""
        print(f"\n{'='*80}")
        print(f"🎯 BENCHMARK RESULTS: {self.benchmark_name}")
        print(f"{'='*80}")
        
        # Overall stats
        total_time_min = results['total_time_seconds'] / 60
        print(f"\n📊 Overall Statistics:")
        print(f"   Total Episodes:     {results['total_episodes']}")
        print(f"   Successful:         {results['total_successes']} ✅")
        print(f"   Failed:             {results['total_episodes'] - results['total_successes']} ❌")
        print(f"   Success Rate:       {results['overall_success_rate']:.1%}")
        print(f"   Total Time:         {total_time_min:.1f} minutes ({results['total_time_seconds']:.1f}s)")
        print(f"   Avg Time/Episode:   {results['avg_time_per_episode']:.1f}s")
        
        # Per-task breakdown
        if len(results['task_results']) > 0:
            print(f"\n📋 Task Details:")
            for task_result in results['task_results']:
                task_name = task_result['task_name']
                success_rate = task_result['success_rate']
                num_eps = task_result['num_episodes']
                successes = int(success_rate * num_eps)
                
                # Get average metrics
                avg_metrics = task_result.get('avg_metrics', {})
                mse = avg_metrics.get('action_mse', 0)
                mae = avg_metrics.get('action_mae', 0)
                cosine = avg_metrics.get('cosine_similarity', 0)
                
                print(f"\n   Task: {task_name}")
                print(f"      Episodes:        {successes}/{num_eps} successful ({success_rate:.1%})")
                print(f"      MSE:             {mse:.6f}")
                print(f"      MAE:             {mae:.6f}")
                print(f"      Cosine Sim:      {cosine:.4f}")
        
        print(f"\n{'='*80}\n")
    
    @property
    def is_initialized(self) -> bool:
        """Check if benchmark is initialized."""
        return self._is_initialized
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.benchmark_name}', initialized={self._is_initialized})"
