# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flow-matching utilities for XVLA action generation.

Implements conditional flow-matching with various time schedules.
"""

import math
from typing import Callable

import torch
import torch.nn as nn


class FlowMatchingSampler:
    """Flow-matching sampler for continuous action generation.
    
    Uses conditional flow-matching to generate actions by iteratively
    denoising from a Gaussian distribution.
    """
    
    def __init__(
        self,
        num_steps: int = 10,
        sigma_min: float = 0.001,
        sigma_max: float = 1.0,
        rho: float = 7.0,
        time_schedule: str = "lognorm",
    ):
        """Initialize flow-matching sampler.
        
        Args:
            num_steps: Number of denoising steps
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
            rho: Schedule parameter for time steps
            time_schedule: Type of time schedule ("lognorm", "uniform", "cosine")
        """
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.time_schedule = time_schedule
        
        # Create time schedule
        self.timesteps = self._create_timesteps()
    
    def _create_timesteps(self) -> torch.Tensor:
        """Create time schedule for denoising.
        
        Returns:
            Tensor of timesteps from 1.0 to 0.0
        """
        if self.time_schedule == "uniform":
            return torch.linspace(1.0, 0.0, self.num_steps + 1)[:-1]
        
        elif self.time_schedule == "lognorm":
            # Log-normal schedule
            step_indices = torch.arange(self.num_steps)
            t = (
                self.sigma_max ** (1 / self.rho)
                + step_indices
                / (self.num_steps - 1)
                * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
            ) ** self.rho
            # Reverse and normalize to [0, 1]
            t = t.flip(0)
            return t / self.sigma_max
        
        elif self.time_schedule == "cosine":
            # Cosine schedule
            steps = torch.arange(self.num_steps + 1)
            alphas = torch.cos(((steps / self.num_steps) + 0.008) / 1.008 * math.pi / 2) ** 2
            t = torch.sqrt(alphas / alphas[0])
            return t[:-1]
        
        else:
            raise ValueError(f"Unknown time schedule: {self.time_schedule}")
    
    def get_schedule(self, device: torch.device) -> torch.Tensor:
        """Get time schedule on specified device.
        
        Args:
            device: Target device
            
        Returns:
            Timesteps tensor on device
        """
        return self.timesteps.to(device)
    
    @staticmethod
    def sample_z0(batch_size: int, action_dim: int, device: torch.device) -> torch.Tensor:
        """Sample initial noise from standard Gaussian.
        
        Args:
            batch_size: Batch size
            action_dim: Action dimension
            device: Target device
            
        Returns:
            Sampled noise [batch_size, action_dim]
        """
        return torch.randn(batch_size, action_dim, device=device)
    
    def step(
        self,
        z_t: torch.Tensor,
        v_t: torch.Tensor,
        t: float,
        t_next: float,
    ) -> torch.Tensor:
        """Single Euler step for flow-matching.
        
        dz = v_t * dt where dt = t_next - t
        
        Args:
            z_t: Current noisy state
            v_t: Predicted vector field
            t: Current time
            t_next: Next time
            
        Returns:
            Denoised state at t_next
        """
        dt = t_next - t
        z_next = z_t + v_t * dt
        return z_next
    
    def sample(
        self,
        vector_field_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        batch_size: int,
        action_dim: int,
        device: torch.device,
        cond: dict = None,
    ) -> torch.Tensor:
        """Sample actions using flow-matching.
        
        Args:
            vector_field_fn: Function that predicts vector field v_t
            batch_size: Batch size
            action_dim: Action dimension
            device: Target device
            cond: Conditioning information
            
        Returns:
            Sampled actions [batch_size, action_dim]
        """
        # Start from noise
        z = self.sample_z0(batch_size, action_dim, device)
        timesteps = self.get_schedule(device)
        
        # Iterative denoising
        for i in range(len(timesteps)):
            t = timesteps[i]
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else 0.0
            
            # Predict vector field
            t_batch = torch.full((batch_size,), t, device=device)
            v_t = vector_field_fn(z, t_batch, cond)
            
            # Euler step
            z = self.step(z, v_t, t, t_next)
        
        return z


class TimeEmbedding(nn.Module):
    """Time embedding for flow-matching.
    
    Embeds continuous time values into high-dimensional vectors
    using sinusoidal position embeddings.
    """
    
    def __init__(self, dim: int, max_period: float = 10000.0):
        """Initialize time embedding.
        
        Args:
            dim: Embedding dimension (must be even)
            max_period: Maximum period for sinusoidal embeddings
        """
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed time values.
        
        Args:
            t: Time values [batch_size] in range [0, 1]
            
        Returns:
            Time embeddings [batch_size, dim]
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, dtype=torch.float32, device=t.device)
            / half_dim
        )
        
        # t: [batch] -> [batch, 1] * [half_dim] -> [batch, half_dim]
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        return embedding


def compute_flow_matching_loss(
    model: Callable,
    actions: torch.Tensor,
    observations: torch.Tensor,
    sigma_min: float = 0.001,
    sigma_max: float = 1.0,
) -> torch.Tensor:
    """Compute conditional flow-matching loss.
    
    L = E[||v_theta(z_t, t, obs) - u_t(z_t)||^2]
    
    where u_t is the ground truth vector field.
    
    Args:
        model: Model that predicts vector field
        actions: Ground truth actions [batch, action_dim]
        observations: Observations for conditioning
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
        
    Returns:
        Flow-matching loss
    """
    batch_size = actions.shape[0]
    device = actions.device
    
    # Sample time uniformly
    t = torch.rand(batch_size, device=device) * (sigma_max - sigma_min) + sigma_min
    
    # Sample noise
    z0 = torch.randn_like(actions)
    
    # Interpolate: z_t = (1 - t) * z0 + t * actions
    t_expanded = t.view(-1, 1)
    z_t = (1 - t_expanded) * z0 + t_expanded * actions
    
    # Ground truth vector field
    u_t = actions - z0  # dz_t/dt
    
    # Predict vector field
    v_t = model(z_t, t, observations)
    
    # MSE loss
    loss = torch.nn.functional.mse_loss(v_t, u_t)
    
    return loss
