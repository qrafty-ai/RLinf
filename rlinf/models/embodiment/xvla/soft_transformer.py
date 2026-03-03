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

def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 100) -> torch.Tensor:
    """Create sinusoidal timestep embeddings.

    Args:
        t: Timesteps [batch]
        dim: Output embedding dimension
        max_period: Controls minimum frequency

    Returns:
        Embeddings [batch, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=t.dtype, device=t.device)
        / half
    )
    args = t[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class DomainAwareLinear(nn.Module):
    """Per-domain linear projection.
    
    Stores separate weights for each domain, enabling domain-specific
    transformations. Used for action encoder/decoder in LeRobot XVLA.
    
    Weight shape: [num_domains, input_size * output_size] (flattened per-domain weights)
    Bias shape: [num_domains, output_size]
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_domains: int = 30,
    ):
        """Initialize domain-aware linear.
        
        Args:
            input_size: Input feature dimension
            output_size: Output feature dimension
            num_domains: Number of domains
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_domains = num_domains
        
        # Per-domain weights stored as flattened [num_domains, input_size * output_size]
        # This matches LeRobot checkpoint format
        self.fc = nn.Embedding(num_domains, input_size * output_size)
        self.bias = nn.Embedding(num_domains, output_size)
        
        # Initialize
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.bias.weight)
    
    def forward(self, x: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        """Forward pass with domain-specific weights.
        
        Args:
            x: Input tensor [batch, ..., input_size]
            domain_id: Domain indices [batch]
            
        Returns:
            Output tensor [batch, ..., output_size]
        """
        # Handle sequence input
        squeeze_seq = False
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, input_size]
            squeeze_seq = True
        
        batch_size, seq_len, _ = x.shape
        
        # Get per-domain weights and biases
        weight = self.fc(domain_id).view(batch_size, self.input_size, self.output_size)
        bias = self.bias(domain_id).view(batch_size, self.output_size)
        
        # Apply linear transformation with domain-specific weights
        # x: [batch, seq, input_size], weight: [batch, input_size, output_size]
        y = torch.matmul(x, weight) + bias.unsqueeze(1)  # [batch, seq, output_size]
        
        if squeeze_seq:
            y = y.squeeze(1)
        
        return y


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
        
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
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
            nn.Linear(mlp_dim, dim),
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
    
    Supports LeRobot-compatible architecture:
    - Action encoder/decoder are always domain-aware
    - VLM/aux projections are optionally domain-aware (use_hetero_proj)
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
        max_len_seq: int = 512,
        dropout: float = 0.1,
        use_hetero_proj: bool = False,
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
            max_len_seq: Maximum sequence length for positional embeddings
            use_hetero_proj: Use per-domain visual projections
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.dim_action = dim_action
        self.dim_proprio = dim_proprio
        self.dim_time = dim_time
        self.len_soft_prompts = len_soft_prompts
        self.use_hetero_proj = use_hetero_proj
        self.max_len_seq = max_len_seq

        # Visual projections (LeRobot naming)
        if use_hetero_proj:
            self.vlm_proj = DomainAwareLinear(
                multi_modal_input_size, hidden_size, num_domains=num_domains
            )
            self.aux_visual_proj = DomainAwareLinear(
                multi_modal_input_size, hidden_size, num_domains=num_domains
            )
        else:
            self.vlm_proj = nn.Linear(multi_modal_input_size, hidden_size)
            self.aux_visual_proj = nn.Linear(multi_modal_input_size, hidden_size)

        # Learned positional embeddings (LeRobot)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len_seq, hidden_size), requires_grad=True)
        nn.init.normal_(self.pos_emb, std=0.02)

        # Action encoder/decoder are domain-aware in LeRobot
        total_action_dim = dim_action + dim_proprio + dim_time
        self.action_encoder = DomainAwareLinear(
            total_action_dim, hidden_size, num_domains=num_domains
        )
        self.action_decoder = DomainAwareLinear(
            hidden_size, dim_action, num_domains=num_domains
        )

        # Domain soft prompts
        if len_soft_prompts > 0:
            self.soft_prompt_hub = nn.Embedding(num_domains, len_soft_prompts * hidden_size)
            nn.init.normal_(self.soft_prompt_hub.weight, std=0.02)
        else:
            self.soft_prompt_hub = None

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(hidden_size)

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
        aux_visual_inputs: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        domain_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict vector field for flow-matching.
        
        Args:
            z_t: Current noisy action [batch, action_dim] or [batch, chunk, action_dim]
            t: Time [batch]
            multi_modal_features: Visual-language features [batch, seq, dim]
            aux_visual_inputs: Additional visual tokens [batch, seq_aux, dim] or None
            proprio: Proprioception [batch, dim_proprio] or None
            domain_ids: Domain indices [batch] or None
            
        Returns:
            Predicted vector field [batch, action_dim] or [batch, chunk, action_dim]
        """
        batch_size = z_t.shape[0]

        squeeze_actions = False
        if z_t.dim() == 2:
            z_t = z_t.unsqueeze(1)
            squeeze_actions = True

        _, num_actions, _ = z_t.shape
        
        # Default domain_ids to 0 if not provided
        if domain_ids is None:
            domain_ids = torch.zeros(batch_size, dtype=torch.long, device=z_t.device)
        
        # 1. Prepare visual features
        if aux_visual_inputs is None:
            aux_visual_inputs = multi_modal_features.new_zeros(
                batch_size, 0, multi_modal_features.shape[-1]
            )

        if self.use_hetero_proj:
            mm_features = self.vlm_proj(multi_modal_features, domain_ids)
            aux_features = self.aux_visual_proj(aux_visual_inputs, domain_ids)
        else:
            mm_features = self.vlm_proj(multi_modal_features)
            aux_features = self.aux_visual_proj(aux_visual_inputs)

        # 2. Encode action tokens from noisy action + proprio + timestep embedding
        if proprio is None:
            proprio_features = z_t.new_zeros((batch_size, self.dim_proprio))
        else:
            proprio_features = proprio
        time_tokens = timestep_embedding(t, self.dim_time).to(dtype=z_t.dtype)
        time_tokens = time_tokens.unsqueeze(1).expand(batch_size, num_actions, self.dim_time)
        proprio_tokens = proprio_features.unsqueeze(1).expand(
            batch_size, num_actions, proprio_features.shape[-1]
        )
        action_tokens = torch.cat([z_t, proprio_tokens, time_tokens], dim=-1)

        action_features = self.action_encoder(action_tokens, domain_ids)  # [batch, chunk, hidden]

        # 3. Build sequence: action + primary visual + aux visual
        sequence = torch.cat([action_features, mm_features, aux_features], dim=1)

        # 4. Add positional embedding
        seq_len = sequence.shape[1]
        if seq_len > self.pos_emb.shape[1]:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len_seq={self.pos_emb.shape[1]}"
            )
        sequence = sequence + self.pos_emb[:, :seq_len, :]

        # 5. Append soft prompts
        if self.soft_prompt_hub is not None:
            soft_prompts = self.soft_prompt_hub(domain_ids)
            soft_prompts = soft_prompts.view(batch_size, self.len_soft_prompts, self.hidden_size)
            sequence = torch.cat([sequence, soft_prompts], dim=1)

        # 6. Transformer forward
        for block in self.blocks:
            sequence = block(sequence)

        sequence = self.norm(sequence)

        # 7. Decode action segment only
        action_features = sequence[:, :num_actions, :]  # [batch, chunk, hidden]
        vector_field = self.action_decoder(action_features, domain_ids)

        if squeeze_actions:
            vector_field = vector_field.squeeze(1)

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
