"""
Evo-1 Flow Matching Loss Implementation

Source: docs/PROPRIOCEPTIVE_UNDERUTILIZATION_THEORY.md
Flow matching loss: L^τ(θ) = E[||v_θ(A_t^τ, z_t, s_t) - u(A_t^τ|A_t)||²]
where A_t^τ = τ·A_t + (1-τ)·ε

Similar to Pi0 but adapted for Evo-1's architecture which includes
visual embeddings (z_t) and state (s_t) in the conditioning.
"""

import torch
import torch.nn.functional as F
import numpy as np


def evo1_flow_matching_loss(model, observation, action_gt, return_components=False):
    """
    Compute Evo-1 flow matching loss for gradient analysis.
    
    Args:
        model: Evo-1 model instance
        observation: Dict containing observation data (images, state, language)
        action_gt: Ground truth actions, shape (batch, action_horizon, action_dim)
        return_components: If True, return intermediate values
    
    Returns:
        loss: Flow matching loss tensor, shape (batch, action_horizon)
        components (optional): Dict with intermediate values
    """
    device = action_gt.device
    batch_size = action_gt.shape[0]
    
    # Sample noise from standard normal
    noise = torch.randn_like(action_gt)
    
    # Sample time from Beta(1.5, 1) distribution
    # Following the same schedule as Pi0 for consistency
    time = torch.from_numpy(
        np.random.beta(1.5, 1, batch_size)
    ).to(device).float() * 0.999 + 0.001
    
    # Expand time for broadcasting
    time_expanded = time[:, None, None]
    
    # Compute noisy actions: A_t^τ = τ·A_t + (1-τ)·ε
    x_t = time_expanded * action_gt + (1 - time_expanded) * noise
    
    # Compute target velocity: u_t = ε - A_t
    u_t = noise - action_gt
    
    # Forward pass through Evo-1 model
    # Evo-1 takes visual embeddings z_t, state s_t, and noisy action A_t^τ
    # Returns predicted velocity v_θ(A_t^τ, z_t, s_t)
    try:
        v_t = model.forward_with_time(observation, x_t, time)
    except AttributeError:
        raise NotImplementedError(
            "Model must implement forward_with_time(observation, x_t, time) "
            "that returns predicted velocity v_t. "
            "For Evo-1, this should condition on visual embeddings z_t and state s_t."
        )
    
    # Compute MSE loss between predicted and target velocity
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


def evo1_flow_matching_loss_simple(predicted_velocity, target_velocity):
    """
    Simplified version for when velocity components are already computed.
    
    Args:
        predicted_velocity: v_θ from model, shape (batch, horizon, action_dim)
        target_velocity: u_t ground truth, shape (batch, horizon, action_dim)
    
    Returns:
        loss: MSE loss, shape (batch, horizon)
    """
    return F.mse_loss(predicted_velocity, target_velocity, reduction='none').mean(dim=-1)


def evo1_encode_observations(model, observation):
    """
    Encode observations for Evo-1 model.
    
    Evo-1 uses:
    - Visual encoder to get z_t (image embeddings)
    - State encoder (CategorySpecificMLP in action_head) to process s_t
    
    Args:
        model: Evo-1 model instance
        observation: Dict with 'images', 'state', 'language'
    
    Returns:
        Dict with encoded components needed for forward pass
    """
    # Extract visual embeddings z_t
    if hasattr(model, 'vision_encoder'):
        z_t = model.vision_encoder(observation['images'])
    else:
        raise AttributeError("Evo-1 model must have vision_encoder")
    
    # State s_t will be processed by state_encoder in action_head
    s_t = observation['state']
    
    # Language encoding if needed
    if 'language' in observation and hasattr(model, 'language_encoder'):
        lang_embed = model.language_encoder(observation['language'])
    else:
        lang_embed = None
    
    return {
        'visual_embeddings': z_t,
        'state': s_t,
        'language_embeddings': lang_embed
    }


def compute_evo1_flow_matching_components(action_gt, time_schedule='beta'):
    """
    Utility to compute flow matching noise schedule components for Evo-1.
    Same as Pi0 flow matching components.
    
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
