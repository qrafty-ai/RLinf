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

"""SoftPromptedTransformer policy head for XVLA.

Implements the policy transformer with domain-specific soft prompts
for multi-domain action generation.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlinf.models.embodiment.xvla.flow_matching import TimeEmbedding


class SoftPromptHub(nn.Module):
    """Domain-specific soft prompt embeddings.
    
    Maintains learnable soft prompts for multiple domains to enable
    multi-domain training with domain-specific conditioning.
    """
    
    def __init__(
        self,
        num_domains: int,
        len_soft_prompts: int,
        dim: int,
    ):
        """Initialize soft prompt hub.
        
        Args:
            num_domains: Number of different domains
            len_soft_prompts: Length of soft prompt sequence
            dim: Embedding dimension
        """
        super().__init__()
        self.num_domains = num_domains
        self.len_soft_prompts = len_soft_prompts
        self.dim = dim
        
        # Learnable soft prompts [num_domains, len_soft_prompts, dim]
        self.soft_prompts = nn.Parameter(
            torch.randn(num_domains, len_soft_prompts, dim) * 0.02
        )
    
    def forward(self, domain_ids: Optional[torch.Tensor] = None, batch_size: int = 1) -> torch.Tensor:
        """Get soft prompts for given domains.
        
        Args:
            domain_ids: Domain indices [batch_size] or None for domain 0
            batch_size: Batch size when domain_ids is None (default: 1)
            
        Returns:
            Soft prompts [batch_size, len_soft_prompts, dim]
        """
        if domain_ids is None:
            # Default to domain 0 for all batch items
            domain_ids = torch.zeros(batch_size, dtype=torch.long, device=self.soft_prompts.device)
        else:
            batch_size = domain_ids.shape[0]
        
        # Index soft prompts
        prompts = self.soft_prompts[domain_ids]  # [batch, len_soft_prompts, dim]
        return prompts


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional soft prompt conditioning."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        """Initialize attention.
        
        Args:
            dim: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input [batch, seq_len, dim]
            mask: Attention mask [batch, seq_len, seq_len]
            
        Returns:
            Output [batch, seq_len, dim]
        """
        batch_size, seq_len, dim = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [batch, heads, seq, seq]
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Combine heads
        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, dim)
        out = self.proj(out)
        out = self.dropout(out)
        
        return out


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        """Initialize block.
        
        Args:
            dim: Model dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim ratio
            dropout: Dropout rate
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward with residual connections."""
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class SoftPromptedTransformer(nn.Module):
    """Transformer policy head with soft prompts for multi-domain action generation.
    
    This is the core policy network that predicts flow-matching vector fields
    for action generation, conditioned on visual-language features and domain.
    """
    
    def __init__(
        self,
        hidden_size: int,
        multi_modal_input_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        num_domains: int = 30,
        dim_action: int = 20,
        dim_proprio: int = 0,
        len_soft_prompts: int = 32,
        dim_time: int = 32,
        dropout: float = 0.0,
    ):
        """Initialize transformer.
        
        Args:
            hidden_size: Hidden dimension
            multi_modal_input_size: Size of visual-language features from Florence2
            depth: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim ratio
            num_domains: Number of domains for soft prompts
            dim_action: Action dimension
            dim_proprio: Proprioception dimension (0 if not used)
            len_soft_prompts: Length of soft prompt sequence
            dim_time: Time embedding dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.dim_action = dim_action
        self.dim_proprio = dim_proprio
        
        # Project multi-modal input to hidden size
        self.input_proj = nn.Linear(multi_modal_input_size, hidden_size)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            TimeEmbedding(dim_time),
            nn.Linear(dim_time, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        # Action input projection
        total_action_dim = dim_action + dim_proprio
        self.action_in_proj = nn.Linear(total_action_dim, hidden_size)
        
        # Soft prompt hub for domain conditioning
        self.soft_prompt_hub = SoftPromptHub(
            num_domains=num_domains,
            len_soft_prompts=len_soft_prompts,
            dim=hidden_size,
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(hidden_size)
        
        # Action output projection (predicts vector field)
        self.action_out_proj = nn.Linear(hidden_size, dim_action)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        multi_modal_features: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        domain_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict vector field for flow-matching.
        
        Args:
            z_t: Current noisy action [batch, action_dim]
            t: Time [batch]
            multi_modal_features: Visual-language features [batch, seq, dim]
            proprio: Proprioception [batch, dim_proprio] or None
            domain_ids: Domain indices [batch] or None
            
        Returns:
            Predicted vector field [batch, action_dim]
        """
        batch_size = z_t.shape[0]
        
        # 1. Get soft prompts for domain
        soft_prompts = self.soft_prompt_hub(domain_ids, batch_size=batch_size)  # [batch, len_soft_prompts, hidden]
        
        # 2. Project multi-modal features
        mm_features = self.input_proj(multi_modal_features)  # [batch, seq, hidden]
        
        # 3. Time embedding
        t_embed = self.time_embed(t)  # [batch, hidden]
        
        # 4. Action embedding (with proprio if available)
        if proprio is not None and self.dim_proprio > 0:
            z_input = torch.cat([z_t, proprio], dim=-1)
        else:
            z_input = z_t
        z_embed = self.action_in_proj(z_input)  # [batch, hidden]
        
        # 5. Build sequence: [soft_prompts, mm_features, time_embed, action_embed]
        # Expand time and action to sequence length 1
        t_embed = t_embed.unsqueeze(1)  # [batch, 1, hidden]
        z_embed = z_embed.unsqueeze(1)  # [batch, 1, hidden]
        
        sequence = torch.cat([
            soft_prompts,     # [batch, len_soft_prompts, hidden]
            mm_features,      # [batch, seq, hidden]
            t_embed,          # [batch, 1, hidden]
            z_embed,          # [batch, 1, hidden]
        ], dim=1)  # [batch, len_soft_prompts + seq + 2, hidden]
        
        # 6. Transformer forward
        for block in self.blocks:
            sequence = block(sequence)
        
        sequence = self.norm(sequence)
        
        # 7. Extract action output (last position)
        action_features = sequence[:, -1, :]  # [batch, hidden]
        vector_field = self.action_out_proj(action_features)  # [batch, action_dim]
        
        return vector_field


class ValueHead(nn.Module):
    """Value head for PPO (optional)."""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        """Initialize value head.
        
        Args:
            input_dim: Input dimension (Florence2 projection dim)
            hidden_dim: Hidden dimension
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict value.
        
        Args:
            features: Visual-language features [batch, input_dim]
            
        Returns:
            Value estimate [batch, 1]
        """
        return self.net(features)
