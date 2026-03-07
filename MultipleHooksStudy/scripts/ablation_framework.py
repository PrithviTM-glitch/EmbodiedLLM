#!/usr/bin/env python3
"""
Ablation Framework - Reusable Components

Provides utilities for running server-based ablation studies on VLA models.
This framework supports zero-injection at the state encoder level and 
measures task success rates with and without state information.

Key Components:
1. AblationServer: WebSocket server that serves model with ablation hook
2. run_ablation_trial: Execute single trial with ablated or baseline model
3. compare_results: Statistical comparison of baseline vs ablated performance

Usage:
    from scripts.ablation_framework import AblationServer, run_ablation_trial
    
    # Create server with ablation
    server = AblationServer(model, state_encoder, port=8765)
    server.start_background()
    
    # Run trials
    results = run_ablation_trial(env, num_episodes=50)
"""

import asyncio
import websockets
import torch
import json
import numpy as np
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import threading


class AblationServer:
    """
    WebSocket server for serving VLA model with state encoder ablation.
    
    The server intercepts state encoder output and optionally zeros it out,
    allowing comparison of model performance with/without state information.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        state_encoder: torch.nn.Module,
        port: int = 8765,
        enable_ablation: bool = False,
        device: str = 'cuda'
    ):
        """
        Initialize ablation server.
        
        Args:
            model: The VLA model to serve
            state_encoder: The state encoder module to ablate
            port: WebSocket server port
            enable_ablation: Whether to enable zero injection
            device: Device to run model on
        """
        self.model = model
        self.state_encoder = state_encoder
        self.port = port
        self.enable_ablation = enable_ablation
        self.device = device
        
        self.ablation_hook = None
        self.server_task = None
        
        # Attach ablation hook
        if self.enable_ablation:
            self._attach_ablation_hook()
    
    def _attach_ablation_hook(self):
        """Attach forward hook that zeros out state encoder output."""
        def zero_output_hook(module, input, output):
            if self.enable_ablation:
                return torch.zeros_like(output)
            return output
        
        self.ablation_hook = self.state_encoder.register_forward_hook(zero_output_hook)
        print(f"✅ Ablation hook attached (enabled={self.enable_ablation})")
    
    def set_ablation(self, enable: bool):
        """Enable or disable ablation dynamically."""
        self.enable_ablation = enable
        if enable and self.ablation_hook is None:
            self._attach_ablation_hook()
    
    async def handle_inference(self, websocket):
        """Handle inference request from client."""
        async for message in websocket:
            try:
                # Parse request
                request = json.loads(message)
                observation = request['observation']
                
                # Convert to tensors
                images = torch.tensor(observation['image']).to(self.device).float()
                state = torch.tensor(observation['state']).to(self.device).float()
                
                # Forward pass through model
                with torch.no_grad():
                    action = self.model(images, state)
                
                # Send response
                response = {
                    'action': action.cpu().numpy().tolist(),
                    'ablated': self.enable_ablation
                }
                await websocket.send(json.dumps(response))
                
            except Exception as e:
                error_response = {'error': str(e)}
                await websocket.send(json.dumps(error_response))
    
    async def start(self):
        """Start WebSocket server (async)."""
        print(f"Starting ablation server on port {self.port}")
        print(f"Ablation enabled: {self.enable_ablation}")
        
        async with websockets.serve(self.handle_inference, "localhost", self.port):
            await asyncio.Future()  # Run forever
    
    def start_background(self):
        """Start server in background thread."""
        def run_server():
            asyncio.run(self.start())
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        print(f"✅ Server started in background on port {self.port}")
    
    def cleanup(self):
        """Remove hooks and cleanup."""
        if self.ablation_hook is not None:
            self.ablation_hook.remove()
            self.ablation_hook = None


def run_ablation_trial(
    benchmark_name: str,
    task_name: str,
    num_episodes: int,
    server_port: int,
    enable_ablation: bool = False,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run ablation trial on a specific benchmark task.
    
    Args:
        benchmark_name: 'libero', 'metaworld', or 'vlabench'
        task_name: Specific task within benchmark
        num_episodes: Number of episodes to run
        server_port: Port where model server is running
        enable_ablation: Whether this is ablated or baseline trial
        seed: Random seed for reproducibility
    
    Returns:
        Dict with trial results including success rate
    """
    import websocket  # pip install websocket-client
    
    # Connect to server
    ws = websocket.WebSocket()
    ws.connect(f"ws://localhost:{server_port}")
    
    # Import appropriate benchmark
    if benchmark_name == 'libero':
        from scripts.data_collectors.libero_collector import create_libero_env
        env = create_libero_env(task_name=task_name, seed=seed)
    elif benchmark_name == 'metaworld':
        from scripts.data_collectors.metaworld_collector import create_metaworld_env
        env = create_metaworld_env(task_name=task_name, seed=seed)
    elif benchmark_name == 'vlabench':
        # VLABench integration (to be implemented)
        raise NotImplementedError("VLABench integration pending")
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    
    # Run episodes
    successes = []
    episode_lengths = []
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 500:  # Max 500 steps per episode
            # Prepare observation
            request = {
                'observation': {
                    'image': obs['image'].tolist(),
                    'state': obs['state'].tolist()
                }
            }
            
            # Get action from server
            ws.send(json.dumps(request))
            response = json.loads(ws.recv())
            action = np.array(response['action'])
            
            # Step environment
            obs, reward, done, info = env.step(action)
            steps += 1
        
        # Record results
        success = info.get('success', reward > 0)
        successes.append(success)
        episode_lengths.append(steps)
    
    ws.close()
    
    # Compute statistics
    success_rate = np.mean(successes)
    avg_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    
    return {
        'benchmark': benchmark_name,
        'task': task_name,
        'num_episodes': num_episodes,
        'ablated': enable_ablation,
        'success_rate': float(success_rate),
        'avg_episode_length': float(avg_length),
        'std_episode_length': float(std_length),
        'successes': successes,
        'episode_lengths': episode_lengths
    }


def compare_results(baseline_results: Dict, ablated_results: Dict) -> Dict[str, Any]:
    """
    Compare baseline and ablated performance.
    
    Args:
        baseline_results: Results from baseline trial
        ablated_results: Results from ablated trial
    
    Returns:
        Statistical comparison including effect size
    """
    baseline_sr = baseline_results['success_rate']
    ablated_sr = ablated_results['success_rate']
    
    # Compute drop
    absolute_drop = baseline_sr - ablated_sr
    relative_drop = (absolute_drop / baseline_sr) * 100 if baseline_sr >  0 else 0
    
    # Statistical test (t-test on success indicators)
    from scipy import stats
    baseline_successes = baseline_results['successes']
    ablated_successes = ablated_results['successes']
    
    t_stat, p_value = stats.ttest_ind(baseline_successes, ablated_successes)
    
    # Cohen's d (effect size)
    baseline_mean = np.mean(baseline_successes)
    ablated_mean = np.mean(ablated_successes)
    pooled_std = np.sqrt(
        (np.std(baseline_successes)**2 + np.std(ablated_successes)**2) / 2
    )
    cohens_d = (baseline_mean - ablated_mean) / pooled_std if pooled_std > 0 else 0
    
    return {
        'baseline_success_rate': float(baseline_sr),
        'ablated_success_rate': float(ablated_sr),
        'absolute_drop': float(absolute_drop),
        'relative_drop_percent': float(relative_drop),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'statistically_significant': p_value < 0.05,
        'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
    }


def save_results(results: Dict, output_path: str):
    """Save results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to: {output_path}")


def print_results_summary(comparison: Dict):
    """Print human-readable results summary."""
    print("\n" + "="*60)
    print("ABLATION STUDY RESULTS")
    print("="*60)
    print(f"Baseline Success Rate: {comparison['baseline_success_rate']:.1%}")
    print(f"Ablated Success Rate:  {comparison['ablated_success_rate']:.1%}")
    print(f"Absolute Drop:         {comparison['absolute_drop']:.1%}")
    print(f"Relative Drop:         {comparison['relative_drop_percent']:.1f}%")
    print(f"P-value:               {comparison['p_value']:.4f}")
    print(f"Cohen's d:             {comparison['cohens_d']:.3f} ({comparison['effect_size']})")
    print(f"Significant:           {'✓ YES' if comparison['statistically_significant'] else '✗ NO'}")
    print("="*60)
