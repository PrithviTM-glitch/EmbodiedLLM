"""
Pi0 Flow Matching Loss Implementation

Source: https://github.com/physical-intelligence/openpi
- src/openpi/models/pi0.py#L213-214: return jnp.mean(jnp.square(v_t - u_t), axis=-1)
- src/openpi/models_pytorch/pi0_pytorch.py#L369: F.mse_loss(u_t, v_t, reduction="none")

Flow matching loss: L^τ(θ) = E[||v_θ(A_t^τ, o_t) - u(A_t^τ|A_t)||²]
where q(A_t^τ|A_t) = N(τ·A_t, (1-τ)·I)
"""

import torch
import torch.nn.functional as F
import numpy as np


def pi0_flow_matching_loss(model, observation, action_gt, return_components=False):
    """
    Compute Pi0 flow matching loss for gradient analysis.
    
    Args:
        model: Pi0 model instance
        observation: Dict containing observation data (images, state, language)
        action_gt: Ground truth actions tensor, shape (batch, action_horizon, action_dim)
        return_components: If True, return intermediate values for debugging
    
    Returns:
        loss: Flow matching loss tensor, shape (batch, action_horizon)
        components (optional): Dict with intermediate values
    """
    device = action_gt.device
    batch_size = action_gt.shape[0]
    
    # Sample noise from standard normal
    noise = torch.randn_like(action_gt)
    
    # Sample time from Beta(1.5, 1) distribution, following openpi implementation
    # time ∈ (0.001, 0.999) to avoid numerical issues at boundaries
    time = torch.from_numpy(
        np.random.beta(1.5, 1, batch_size)
    ).to(device).float() * 0.999 + 0.001
    
    # Expand time for broadcasting: (batch,) -> (batch, 1, 1)
    time_expanded = time[:, None, None]
    
    # Compute noisy actions: A_t^τ = τ·A_t + (1-τ)·ε
    # This follows the probability path q(A_t^τ|A_t) = N(τ·A_t, (1-τ)·I)
    x_t = time_expanded * action_gt + (1 - time_expanded) * noise
    
    # Compute target velocity: u_t = ε - A_t
    # The model predicts velocity field v_θ that should match this target
    u_t = noise - action_gt
    
    # Forward pass through model to predict velocity
    # Note: This requires adapting the model interface to accept (observation, x_t, time)
    # and return predicted velocity v_t
    try:
        # Try standard forward interface
        v_t = model.forward_with_time(observation, x_t, time)
    except AttributeError:
        # Fallback: use model's compute_loss method but extract v_t
        # This might need model-specific adaptation
        raise NotImplementedError(
            "Model must implement forward_with_time(observation, x_t, time) "
            "that returns predicted velocity v_t"
        )
    
    # Compute MSE loss between predicted and target velocity
    # Loss shape: (batch, action_horizon, action_dim) -> (batch, action_horizon)
    loss_per_step = F.mse_loss(v_t, u_t, reduction='none').mean(dim=-1)
    
    if return_components:
        components = {
            'noise': noise,
            'time': time,
            'x_t': x_t,
            'u_t': u_t,
            'v_t': v_t,
            'loss_per_step': loss_per_step
        }
        return loss_per_step, components
    
    return loss_per_step


def pi0_flow_matching_loss_simple(predicted_velocity, target_velocity):
    """
    Simplified version for when velocity components are already computed.
    
    Args:
        predicted_velocity: v_θ from model, shape (batch, horizon, action_dim)
        target_velocity: u_t ground truth, shape (batch, horizon, action_dim)
    
    Returns:
        loss: MSE loss, shape (batch, horizon)
    """
    return F.mse_loss(predicted_velocity, target_velocity, reduction='none').mean(dim=-1)


def compute_flow_matching_components(action_gt, time_schedule='beta'):
    """
    Utility to compute flow matching noise schedule components.
    
    Args:
        action_gt: Ground truth actions, shape (batch, horizon, action_dim)
        time_schedule: 'beta' (default) or 'uniform'
    
    Returns:
        Dict with noise, time, x_t, u_t
    """
    batch_size = action_gt.shape[0]
    device = action_gt.device
    
    # Sample noise
    noise = torch.randn_like(action_gt)
    
    # Sample time
    if time_schedule == 'beta':
        time = torch.from_numpy(
            np.random.beta(1.5, 1, batch_size)
        ).to(device).float() * 0.999 + 0.001
    elif time_schedule == 'uniform':
        time = torch.rand(batch_size, device=device) * 0.998 + 0.001
    else:
        raise ValueError(f"Unknown time_schedule: {time_schedule}")
    
    time_expanded = time[:, None, None]
    
    # Compute noisy actions and target velocity
    x_t = time_expanded * action_gt + (1 - time_expanded) * noise
    u_t = noise - action_gt
    
    return {
        'noise': noise,
        'time': time,
        'x_t': x_t,
        'u_t': u_t
    }
