"""
RDT-1B Diffusion Loss Implementation

Source: https://github.com/thu-ml/roboticsdiffusiontransformer
- models/rdt_runner.py#L205-219: F.mse_loss(pred, target)

Diffusion loss: L(θ) = MSE(a_t, f_θ(ℓ, o_t, √(ᾱ^k)·a_t + √(1-ᾱ^k)·ε, k))
"""

import torch
import torch.nn.functional as F


def rdt_diffusion_loss(model, observation, action_gt, pred_type='epsilon', return_components=False):
    """
    Compute RDT-1B diffusion loss for gradient analysis.
    
    Args:
        model: RDT model instance with noise_scheduler
        observation: Dict containing observation data
        action_gt: Ground truth actions, shape (batch, horizon, action_dim)
        pred_type: 'epsilon' (predict noise) or 'sample' (predict clean action)
        return_components: If True, return intermediate values
    
    Returns:
        loss: Diffusion loss tensor
        components (optional): Dict with intermediate values
    """
    device = action_gt.device
    batch_size = action_gt.shape[0]
    
    # Sample noise from standard normal
    noise = torch.randn_like(action_gt)
    
    # Sample random diffusion timesteps uniformly from [0, num_train_timesteps)
    timesteps = torch.randint(
        0, 
        model.num_train_timesteps, 
        (batch_size,), 
        device=device
    ).long()
    
    # Add noise to clean actions according to noise schedule
    # noisy_action = √(ᾱ_t) * action_gt + √(1 - ᾱ_t) * noise
    noisy_action = model.noise_scheduler.add_noise(action_gt, noise, timesteps)
    
    # Forward pass through model
    # Model learns to predict either noise (epsilon) or clean sample
    try:
        pred = model.forward_with_timesteps(observation, noisy_action, timesteps)
    except AttributeError:
        raise NotImplementedError(
            "Model must implement forward_with_timesteps(observation, noisy_action, timesteps) "
            "that returns predicted noise or sample"
        )
    
    # Determine target based on prediction type
    if pred_type == 'epsilon':
        target = noise
    elif pred_type == 'sample':
        target = action_gt
    else:
        raise ValueError(f"Unknown pred_type: {pred_type}. Use 'epsilon' or 'sample'")
    
    # Compute MSE loss
    loss = F.mse_loss(pred, target, reduction='mean')
    
    if return_components:
        components = {
            'noise': noise,
            'timesteps': timesteps,
            'noisy_action': noisy_action,
            'pred': pred,
            'target': target,
            'loss': loss
        }
        return loss, components
    
    return loss


def rdt_diffusion_loss_simple(pred, target):
    """
    Simplified version for when predictions and targets are already computed.
    
    Args:
        pred: Model prediction (noise or sample), shape (batch, horizon, action_dim)
        target: Ground truth target, shape (batch, horizon, action_dim)
    
    Returns:
        loss: MSE loss scalar
    """
    return F.mse_loss(pred, target, reduction='mean')


class DDPMNoiseScheduler:
    """
    Simplified DDPM noise scheduler for RDT.
    Based on thu-ml/RoboticsDiffusionTransformer implementation.
    """
    
    def __init__(self, num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2'):
        self.num_train_timesteps = num_train_timesteps
        self.beta_schedule = beta_schedule
        
        # Compute beta schedule
        if beta_schedule == 'linear':
            betas = torch.linspace(0.0001, 0.02, num_train_timesteps)
        elif beta_schedule == 'squaredcos_cap_v2':
            betas = self._betas_for_alpha_bar(num_train_timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")
        
        self.betas = betas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def _betas_for_alpha_bar(self, num_diffusion_timesteps, max_beta=0.999):
        """
        Create a beta schedule that discretizes the given alpha_t_bar function.
        """
        def alpha_bar(time_step):
            return torch.cos((time_step + 0.008) / 1.008 * torch.pi / 2) ** 2
        
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return torch.tensor(betas)
    
    def add_noise(self, original_samples, noise, timesteps):
        """
        Add noise to samples according to diffusion schedule.
        
        Args:
            original_samples: Clean samples, shape (batch, ...)
            noise: Noise tensor, same shape as original_samples
            timesteps: Timestep indices, shape (batch,)
        
        Returns:
            noisy_samples: Noised samples
        """
        alphas_cumprod = self.alphas_cumprod.to(original_samples.device)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        
        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples


def create_noise_scheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2'):
    """
    Factory function to create noise scheduler.
    """
    return DDPMNoiseScheduler(num_train_timesteps, beta_schedule)
