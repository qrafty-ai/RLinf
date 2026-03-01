# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch Florence-2 model - Standalone implementation for RLinf.

Adapted from LeRobot's XVLA implementation:
https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/xvla/modeling_florence2.py

This implementation removes transformers library dependencies while maintaining
full compatibility with the original Florence-2 architecture.
"""

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from torch import nn

from .configuration_florence2 import Florence2Config, Florence2LanguageConfig, Florence2VisionConfig

# Optional flash attention support
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


# =============================================================================
# Utility Functions
# =============================================================================

def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Args:
        x: Input tensor.
        drop_prob: Probability of dropping a path.
        training: Whether in training mode.
        scale_by_keep: Whether to scale by keep probability.

    Returns:
        Tensor with stochastic depth applied.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


def _prepare_4d_attention_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None) -> torch.Tensor:
    """Prepare a 2D or 3D attention mask to 4D for scaled dot product attention.
    
    Args:
        mask: 2D mask [batch_size, src_len] or 3D mask [batch_size, tgt_len, src_len]
        dtype: Data type of the mask
        tgt_len: Target length for broadcasting
        
    Returns:
        4D attention mask [batch_size, 1, tgt_len, src_len] that can broadcast to [batch_size, num_heads, tgt_len, src_len]
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len or src_len
    
    # Convert to 4D: [batch_size, 1, tgt_len, src_len]
    # First expand to [batch_size, 1, src_len], then expand tgt dimension
    mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, src_len]
    mask = mask.expand(bsz, 1, tgt_len, src_len)  # [batch_size, 1, tgt_len, src_len]
    
    # Convert 0s to large negative values for softmax
    mask = mask.to(dtype)
    mask = (1.0 - mask) * torch.finfo(dtype).min
    
    return mask


# =============================================================================
# DropPath Module
# =============================================================================

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self) -> str:
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


# =============================================================================
# Position Embeddings
# =============================================================================

class LearnedAbsolutePositionEmbedding2D(nn.Module):
    """Learned absolute 2D position embeddings.

    This module learns positional embeddings up to a fixed maximum size.
    Useful for encoding spatial positions in images.

    Args:
        embedding_dim: Dimension of the embeddings.
        num_pos: Maximum number of positions to support.
    """

    def __init__(self, embedding_dim: int = 256, num_pos: int = 50):
        super().__init__()
        self.row_embeddings = nn.Embedding(num_pos, embedding_dim // 2)
        self.column_embeddings = nn.Embedding(num_pos, embedding_dim - (embedding_dim // 2))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            pixel_values: Input tensor of shape (batch_size, height, width, num_channels).

        Returns:
            Position embeddings of shape (batch_size, height, width, embedding_dim).
        """
        if len(pixel_values.shape) != 4:
            raise ValueError("pixel_values must be a 4D tensor")
        height, width = pixel_values.shape[1:3]
        width_values = torch.arange(width, device=pixel_values.device)
        height_values = torch.arange(height, device=pixel_values.device)
        x_emb = self.column_embeddings(width_values)
        y_emb = self.row_embeddings(height_values)
        # (height, width, embedding_dim)
        pos = torch.cat(
            [x_emb.unsqueeze(0).repeat(height, 1, 1), y_emb.unsqueeze(1).repeat(1, width, 1)], dim=-1
        )
        # (embedding_dim, height, width)
        pos = pos.permute(2, 0, 1)
        pos = pos.unsqueeze(0)
        # (batch_size, embedding_dim, height, width)
        pos = pos.repeat(pixel_values.shape[0], 1, 1, 1)
        # (batch_size, height, width, embedding_dim)
        pos = pos.permute(0, 2, 3, 1)
        return pos


class PositionalEmbeddingCosine1D(nn.Module):
    """Cosine 1D positional embeddings.

    This class implements sinusoidal positional encoding following:
    https://pytorch.org/tutorials/beginner/translation_transformer.html

    Args:
        embed_dim: The dimension of the embeddings.
        max_seq_len: The maximum length to precompute the positional encodings.
    """

    def __init__(self, embed_dim: int = 512, max_seq_len: int = 1024) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        # Generate the sinusoidal arrays
        factor = math.log(10000)
        denominator = torch.exp(-factor * torch.arange(0, self.embed_dim, 2) / self.embed_dim)
        frequencies = torch.arange(0, self.max_seq_len).reshape(self.max_seq_len, 1) * denominator
        pos_idx_to_embed = torch.zeros((self.max_seq_len, self.embed_dim))
        pos_idx_to_embed[:, 0::2] = torch.sin(frequencies)
        pos_idx_to_embed[:, 1::2] = torch.cos(frequencies)
        self.register_buffer("pos_idx_to_embed", pos_idx_to_embed)

    def forward(self, seq_embeds: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            seq_embeds: Sequence embeddings of shape [T, D] or [B, T, D].

        Returns:
            Positional embeddings with same shape as input.
        """
        shape_len = len(seq_embeds.shape)
        assert 2 <= shape_len <= 3
        len_seq = seq_embeds.size(-2)
        assert len_seq <= self.max_seq_len
        pos_embeds = self.pos_idx_to_embed[0:len_seq, :]
        if shape_len == 3:
            pos_embeds = pos_embeds.view((1, pos_embeds.size(0), pos_embeds.size(1)))
        return pos_embeds


class LearnedAbsolutePositionEmbedding1D(nn.Module):
    """Learnable absolute 1D positional embeddings.

    Args:
        embedding_dim: The dimension of the embeddings.
        num_pos: The maximum number of positions.
    """

    def __init__(self, embedding_dim: int = 512, num_pos: int = 1024) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(num_pos, embedding_dim)
        self.num_pos = num_pos

    def forward(self, seq_embeds: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            seq_embeds: Sequence embeddings of shape [T, D] or [B, T, D].

        Returns:
            Positional embeddings with same shape as input.
        """
        shape_len = len(seq_embeds.shape)
        assert 2 <= shape_len <= 3
        len_seq = seq_embeds.size(-2)
        assert len_seq <= self.num_pos
        pos_embeds = self.embeddings(torch.arange(len_seq, device=seq_embeds.device))
        if shape_len == 3:
            pos_embeds = pos_embeds.view((1, pos_embeds.size(0), pos_embeds.size(1)))
        return pos_embeds


class Florence2LearnedPositionalEmbedding(nn.Embedding):
    """Learned positional embeddings with offset.

    Florence2 offsets embedding ids by 2 to handle padding correctly.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Input token ids of shape [bsz, seqlen].
            past_key_values_length: Length of past key values for caching.

        Returns:
            Positional embeddings.
        """
        bsz, seq_len = input_ids.shape[:2]
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        ).expand(bsz, -1)
        return super().forward(positions + self.offset)


# =============================================================================
# Utility Modules
# =============================================================================

class MySequential(nn.Sequential):
    """Sequential module that handles tuple inputs."""

    def forward(self, *inputs):
        for module in self._modules.values():
            inputs = module(*inputs) if isinstance(inputs, tuple) else module(inputs)
        return inputs


class PreNorm(nn.Module):
    """Pre-normalization wrapper with optional drop path."""

    def __init__(self, norm: Optional[nn.Module], fn: nn.Module, drop_path: Optional[nn.Module] = None):
        super().__init__()
        self.norm = norm
        self.fn = fn
        self.drop_path = drop_path

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, Any]:
        shortcut = x
        if self.norm is not None:
            x, size = self.fn(self.norm(x), *args, **kwargs)
        else:
            x, size = self.fn(x, *args, **kwargs)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = shortcut + x
        return x, size


class Mlp(nn.Module):
    """MLP module with configurable activation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type = nn.GELU,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.net = nn.Sequential(
            OrderedDict([
                ("fc1", nn.Linear(in_features, hidden_features)),
                ("act", act_layer()),
                ("fc2", nn.Linear(hidden_features, out_features)),
            ])
        )

    def forward(self, x: torch.Tensor, size: Any) -> Tuple[torch.Tensor, Any]:
        return self.net(x), size


class DepthWiseConv2d(nn.Module):
    """Depthwise separable 2D convolution."""

    def __init__(
        self,
        dim_in: int,
        kernel_size: int,
        padding: int,
        stride: int,
        bias: bool = True,
    ):
        super().__init__()
        self.dw = nn.Conv2d(
            dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride, bias=bias
        )

    def forward(self, x: torch.Tensor, size: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        batch_size, num_tokens, channels = x.shape
        height, width = size
        assert num_tokens == height * width

        x = self.dw(x.transpose(1, 2).view(batch_size, channels, height, width))
        size = (x.size(-2), x.size(-1))
        x = x.flatten(2).transpose(1, 2)
        return x, size


# =============================================================================
# DaViT Vision Components
# =============================================================================

class ConvEmbed(nn.Module):
    """Image to Patch Embedding using convolution.

    Args:
        patch_size: Size of the patch.
        in_chans: Number of input channels.
        embed_dim: Embedding dimension.
        stride: Stride of the convolution.
        padding: Padding of the convolution.
        norm_layer: Normalization layer.
        pre_norm: Whether to apply normalization before convolution.
    """

    def __init__(
        self,
        patch_size: int = 7,
        in_chans: int = 3,
        embed_dim: int = 64,
        stride: int = 4,
        padding: int = 2,
        norm_layer: Optional[type] = None,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        dim_norm = in_chans if pre_norm else embed_dim
        self.norm = norm_layer(dim_norm) if norm_layer else None
        self.pre_norm = pre_norm

    def forward(self, x: torch.Tensor, size: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        height, width = size
        if len(x.size()) == 3:
            if self.norm is not None and self.pre_norm:
                x = self.norm(x)
            x = rearrange(x, "b (h w) c -> b c h w", h=height, w=width)

        x = self.proj(x)
        _, _, height, width = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.norm is not None and not self.pre_norm:
            x = self.norm(x)

        return x, (height, width)


class ChannelAttention(nn.Module):
    """Channel attention mechanism for DaViT.

    Groups channels and applies attention within each group.
    """

    def __init__(self, dim: int, groups: int = 8, qkv_bias: bool = True):
        super().__init__()
        self.groups = groups
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, size: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        batch_size, num_tokens, channels = x.shape

        qkv = (
            self.qkv(x)
            .reshape(batch_size, num_tokens, 3, self.groups, channels // self.groups)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * (float(num_tokens) ** -0.5)
        attention = q.transpose(-1, -2) @ k
        attention = attention.softmax(dim=-1)
        x = (attention @ v.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(batch_size, num_tokens, channels)
        x = self.proj(x)
        return x, size


class ChannelBlock(nn.Module):
    """Channel block with attention and MLP.

    Args:
        dim: Input dimension.
        groups: Number of groups for channel attention.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim.
        qkv_bias: Whether to use bias in QKV projection.
        drop_path_rate: Drop path rate.
        act_layer: Activation layer.
        norm_layer: Normalization layer.
        conv_at_attn: Whether to apply conv before attention.
        conv_at_ffn: Whether to apply conv before FFN.
    """

    def __init__(
        self,
        dim: int,
        groups: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        act_layer: type = nn.GELU,
        norm_layer: type = nn.LayerNorm,
        conv_at_attn: bool = True,
        conv_at_ffn: bool = True,
    ):
        super().__init__()
        drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.conv1 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1)) if conv_at_attn else None
        self.channel_attn = PreNorm(
            norm_layer(dim), ChannelAttention(dim, groups=groups, qkv_bias=qkv_bias), drop_path
        )
        self.conv2 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1)) if conv_at_ffn else None
        self.ffn = PreNorm(
            norm_layer(dim),
            Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer),
            drop_path,
        )

    def forward(self, x: torch.Tensor, size: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if self.conv1 is not None:
            x, size = self.conv1(x, size)
        x, size = self.channel_attn(x, size)

        if self.conv2 is not None:
            x, size = self.conv2(x, size)
        x, size = self.ffn(x, size)

        return x, size


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition into windows.

    Args:
        x: Input tensor of shape (B, H, W, C).
        window_size: Size of each window.

    Returns:
        Windows of shape (B*num_windows, window_size, window_size, C).
    """
    batch_size, height, width, channels = x.shape
    x = x.view(batch_size, height // window_size, window_size, width // window_size, window_size, channels)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, channels)
    return windows


def window_reverse(windows: torch.Tensor, batch_size: int, window_size: int, height: int, width: int) -> torch.Tensor:
    """Reverse window partitioning.

    Args:
        windows: Windows tensor of shape (B*num_windows, window_size, window_size, C).
        batch_size: Original batch size.
        window_size: Size of each window.
        height: Original height.
        width: Original width.

    Returns:
        Reconstructed tensor of shape (B, H, W, C).
    """
    x = windows.view(batch_size, height // window_size, width // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, height, width, -1)
    return x


class WindowAttention(nn.Module):
    """Window-based multi-head self attention.

    Args:
        dim: Input dimension.
        num_heads: Number of attention heads.
        window_size: Size of attention window.
        qkv_bias: Whether to use bias in QKV projection.
    """

    def __init__(self, dim: int, num_heads: int, window_size: int, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = float(head_dim) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, size: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        height, width = size
        batch_size, seq_len, channels = x.shape
        assert seq_len == height * width, "input feature has wrong size"

        x = x.view(batch_size, height, width, channels)

        # Pad if needed
        pad_l = pad_t = 0
        pad_r = (self.window_size - width % self.window_size) % self.window_size
        pad_b = (self.window_size - height % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, height_padded, width_padded, _ = x.shape

        # Partition into windows
        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size * self.window_size, channels)

        # Self-attention within windows
        batch_windows, num_tokens, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch_windows, num_tokens, 3, self.num_heads, channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_windows, num_tokens, channels)
        x = self.proj(x)

        # Merge windows
        x = x.view(-1, self.window_size, self.window_size, channels)
        x = window_reverse(x, batch_size, self.window_size, height_padded, width_padded)

        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :height, :width, :].contiguous()

        x = x.view(batch_size, height * width, channels)
        return x, size


class SpatialBlock(nn.Module):
    """Spatial block with window attention and MLP.

    Args:
        dim: Input dimension.
        num_heads: Number of attention heads.
        window_size: Size of attention window.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim.
        qkv_bias: Whether to use bias in QKV projection.
        drop_path_rate: Drop path rate.
        act_layer: Activation layer.
        norm_layer: Normalization layer.
        conv_at_attn: Whether to apply conv before attention.
        conv_at_ffn: Whether to apply conv before FFN.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        act_layer: type = nn.GELU,
        norm_layer: type = nn.LayerNorm,
        conv_at_attn: bool = True,
        conv_at_ffn: bool = True,
    ):
        super().__init__()
        drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.conv1 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1)) if conv_at_attn else None
        self.window_attn = PreNorm(
            norm_layer(dim), WindowAttention(dim, num_heads, window_size, qkv_bias=qkv_bias), drop_path
        )
        self.conv2 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1)) if conv_at_ffn else None
        self.ffn = PreNorm(
            norm_layer(dim),
            Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer),
            drop_path,
        )

    def forward(self, x: torch.Tensor, size: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if self.conv1 is not None:
            x, size = self.conv1(x, size)
        x, size = self.window_attn(x, size)

        if self.conv2 is not None:
            x, size = self.conv2(x, size)
        x, size = self.ffn(x, size)
        return x, size


class DaViT(nn.Module):
    """DaViT: Dual-Attention Vision Transformer.

    Combines spatial window attention with channel attention for hierarchical
    feature extraction in vision tasks.

    Args:
        in_chans: Number of input image channels.
        num_classes: Number of classes for classification head.
        depths: Number of blocks at each stage.
        patch_size: Patch size of convolution in different stages.
        patch_stride: Patch stride of convolution in different stages.
        patch_padding: Patch padding of convolution in different stages.
        patch_prenorm: Whether to apply norm before convolution layer.
        embed_dims: Patch embedding dimension in different stages.
        num_heads: Number of spatial attention heads in different stages.
        num_groups: Number of channel groups in different stages.
        window_size: Window size.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        qkv_bias: Whether to add a learnable bias to query, key, value.
        drop_path_rate: Stochastic depth rate.
        norm_layer: Normalization layer.
        enable_checkpoint: Whether to enable checkpointing.
        conv_at_attn: Whether to perform depthwise convolution before attention.
        conv_at_ffn: Whether to perform depthwise convolution before ffn.
    """

    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        depths: Tuple[int, ...] = (1, 1, 3, 1),
        patch_size: Tuple[int, ...] = (7, 2, 2, 2),
        patch_stride: Tuple[int, ...] = (4, 2, 2, 2),
        patch_padding: Tuple[int, ...] = (3, 0, 0, 0),
        patch_prenorm: Tuple[bool, ...] = (False, False, False, False),
        embed_dims: Tuple[int, ...] = (64, 128, 192, 256),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        num_groups: Tuple[int, ...] = (3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.1,
        norm_layer: type = nn.LayerNorm,
        enable_checkpoint: bool = False,
        conv_at_attn: bool = True,
        conv_at_ffn: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.num_stages = len(self.embed_dims)
        self.enable_checkpoint = enable_checkpoint
        assert self.num_stages == len(self.num_heads) == len(self.num_groups)

        num_stages = len(embed_dims)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths) * 2)]

        depth_offset = 0
        convs = []
        blocks = []
        for i in range(num_stages):
            conv_embed = ConvEmbed(
                patch_size=patch_size[i],
                stride=patch_stride[i],
                padding=patch_padding[i],
                in_chans=in_chans if i == 0 else self.embed_dims[i - 1],
                embed_dim=self.embed_dims[i],
                norm_layer=norm_layer,
                pre_norm=patch_prenorm[i],
            )
            convs.append(conv_embed)

            block = MySequential(
                *[
                    MySequential(
                        OrderedDict([
                            (
                                "spatial_block",
                                SpatialBlock(
                                    embed_dims[i],
                                    num_heads[i],
                                    window_size,
                                    drop_path_rate=dpr[depth_offset + j * 2],
                                    qkv_bias=qkv_bias,
                                    mlp_ratio=mlp_ratio,
                                    conv_at_attn=conv_at_attn,
                                    conv_at_ffn=conv_at_ffn,
                                ),
                            ),
                            (
                                "channel_block",
                                ChannelBlock(
                                    embed_dims[i],
                                    num_groups[i],
                                    drop_path_rate=dpr[depth_offset + j * 2 + 1],
                                    qkv_bias=qkv_bias,
                                    mlp_ratio=mlp_ratio,
                                    conv_at_attn=conv_at_attn,
                                    conv_at_ffn=conv_at_ffn,
                                ),
                            ),
                        ])
                    )
                    for j in range(depths[i])
                ]
            )
            blocks.append(block)
            depth_offset += depths[i] * 2

        self.convs = nn.ModuleList(convs)
        self.blocks = nn.ModuleList(blocks)
        self.norms = norm_layer(self.embed_dims[-1])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

    @property
    def dim_out(self) -> int:
        """Output dimension."""
        return self.embed_dims[-1]

    def forward_features_unpool(self, x: torch.Tensor) -> torch.Tensor:
        """Forward until avg pooling.

        Args:
            x: Input image tensor.

        Returns:
            Feature tensor before pooling.
        """
        input_size = (x.size(2), x.size(3))
        for conv, block in zip(self.convs, self.blocks, strict=False):
            x, input_size = conv(x, input_size)
            if self.enable_checkpoint:
                x, input_size = checkpoint.checkpoint(block, x, input_size)
            else:
                x, input_size = block(x, input_size)
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward features with pooling.

        Args:
            x: Input image tensor.

        Returns:
            Pooled feature tensor.
        """
        x = self.forward_features_unpool(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        x = self.norms(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image tensor.

        Returns:
            Output logits.
        """
        x = self.forward_features(x)
        x = self.head(x)
        return x

    @classmethod
    def from_config(cls, config: Florence2VisionConfig) -> "DaViT":
        """Create DaViT from configuration.

        Args:
            config: Vision configuration.

        Returns:
            DaViT model.
        """
        return cls(
            depths=tuple(config.depths),
            embed_dims=tuple(config.dim_embed),
            num_heads=tuple(config.num_heads),
            num_groups=tuple(config.num_groups),
            patch_size=tuple(config.patch_size),
            patch_stride=tuple(config.patch_stride),
            patch_padding=tuple(config.patch_padding),
            patch_prenorm=tuple(config.patch_prenorm),
            drop_path_rate=config.drop_path_rate,
            window_size=config.window_size,
        )


# =============================================================================
# BART Language Model Components
# =============================================================================

def _get_unpad_data(attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Get unpadded data indices for flash attention.

    Args:
        attention_mask: Attention mask tensor.

    Returns:
        Tuple of (indices, cu_seqlens, max_seqlen).
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int) -> torch.Tensor:
    """Shift input ids one token to the right.

    Args:
        input_ids: Input token ids.
        pad_token_id: Pad token id.
        decoder_start_token_id: Decoder start token id.

    Returns:
        Shifted input ids.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("pad_token_id has to be defined.")
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class Florence2ScaledWordEmbedding(nn.Embedding):
    """Word embeddings with optional scaling.

    Args:
        num_embeddings: Number of embeddings.
        embedding_dim: Dimension of embeddings.
        padding_idx: Padding index.
        embed_scale: Scale factor for embeddings.
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: Optional[float] = 1.0
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale


class Florence2Attention(nn.Module):
    """Multi-headed attention mechanism.

    Implements the attention mechanism from 'Attention Is All You Need'.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        is_decoder: Whether this is a decoder attention.
        bias: Whether to use bias in projections.
        is_causal: Whether to use causal attention.
        config: Language configuration.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[Florence2LanguageConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass.

        Args:
            hidden_states: Hidden states tensor.
            key_value_states: Key/value states for cross-attention.
            past_key_value: Past key/value states for caching.
            attention_mask: Attention mask.
            layer_head_mask: Layer head mask.
            output_attentions: Whether to output attention weights.

        Returns:
            Tuple of (attention output, attention weights, past key value).
        """
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling

        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class Florence2SdpaAttention(Florence2Attention):
    """Attention using PyTorch's scaled dot product attention (SDPA)."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass using SDPA."""
        if output_attentions or layer_head_mask is not None:
            return super().forward(
                hidden_states,
                key_value_states=key_value_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )

        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)

        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        query_states = self._shape(query_states, tgt_len, bsz)
        is_causal = bool(self.is_causal and attention_mask is None and tgt_len > 1)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, None, past_key_value


class Florence2FlashAttention2(Florence2Attention):
    """Flash Attention 2 implementation.

    Requires flash-attn package to be installed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not FLASH_ATTN_AVAILABLE:
            raise ImportError("Flash Attention 2 requires flash-attn package. Install with: pip install flash-attn")

    def _reshape(self, tensor: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass using Flash Attention 2."""
        if output_attentions:
            raise ValueError("Florence2FlashAttention2 does not support output_attentions")

        is_cross_attention = key_value_states is not None
        bsz, q_len, _ = hidden_states.size()

        query_states = self._reshape(self.q_proj(hidden_states), -1, bsz)

        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            key_states = past_key_value[0].transpose(1, 2)
            value_states = past_key_value[1].transpose(1, 2)
        elif is_cross_attention:
            key_states = self._reshape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._reshape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._reshape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._reshape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0].transpose(1, 2), key_states], dim=1)
            value_states = torch.cat([past_key_value[1].transpose(1, 2), value_states], dim=1)
        else:
            key_states = self._reshape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._reshape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            past_key_value = (key_states.transpose(1, 2), value_states.transpose(1, 2))

        query_states = query_states.to(torch.float16)
        key_states = key_states.to(torch.float16)
        value_states = value_states.to(torch.float16)

        attn_output = flash_attn_func(
            query_states, key_states, value_states, dropout_p=self.dropout, causal=self.is_causal
        )

        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.out_proj(attn_output)

        return attn_output, None, past_key_value


FLORENCE2_ATTENTION_CLASSES = {
    "eager": Florence2Attention,
    "sdpa": Florence2SdpaAttention,
    "flash_attention_2": Florence2FlashAttention2 if FLASH_ATTN_AVAILABLE else Florence2SdpaAttention,
}


class Florence2EncoderLayer(nn.Module):
    """Florence2 encoder layer.

    Args:
        config: Language configuration.
    """

    def __init__(self, config: Florence2LanguageConfig):
        super().__init__()
        self.embed_dim = config.d_model

        attn_class = FLORENCE2_ATTENTION_CLASSES.get(config._attn_implementation, Florence2Attention)
        self.self_attn = attn_class(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = self._get_activation(config.activation_function)
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def _get_activation(self, activation: str):
        """Get activation function by name."""
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
            "gelu_new": nn.GELU(),
        }
        return activations.get(activation, nn.GELU())

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            hidden_states: Hidden states.
            attention_mask: Attention mask.
            layer_head_mask: Layer head mask.
            output_attentions: Whether to output attention weights.

        Returns:
            Tuple of (hidden states, attention weights).
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        return hidden_states, attn_weights


class Florence2DecoderLayer(nn.Module):
    """Florence2 decoder layer with cross-attention.

    Args:
        config: Language configuration.
    """

    def __init__(self, config: Florence2LanguageConfig):
        super().__init__()
        self.embed_dim = config.d_model

        attn_class = FLORENCE2_ATTENTION_CLASSES.get(config._attn_implementation, Florence2Attention)
        self.self_attn = attn_class(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )
        self.dropout = config.dropout
        self.activation_fn = self._get_activation(config.activation_function)
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = attn_class(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def _get_activation(self, activation: str):
        """Get activation function by name."""
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
            "gelu_new": nn.GELU(),
        }
        return activations.get(activation, nn.GELU())

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass.

        Args:
            hidden_states: Hidden states.
            attention_mask: Attention mask.
            encoder_hidden_states: Encoder hidden states for cross-attention.
            encoder_attention_mask: Encoder attention mask.
            layer_head_mask: Layer head mask.
            cross_attn_layer_head_mask: Cross-attention layer head mask.
            past_key_value: Past key/value states.
            output_attentions: Whether to output attention weights.
            use_cache: Whether to use caching.

        Returns:
            Tuple of (hidden states, self attention weights, cross attention weights, present key value).
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            present_key_value = present_key_value + cross_attn_present_key_value

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights) if encoder_hidden_states is not None else (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs[0] if len(outputs) == 1 else outputs


class Florence2Encoder(nn.Module):
    """Florence2 encoder.

    Args:
        config: Language configuration.
        embed_tokens: Embedding layer.
    """

    def __init__(self, config: Florence2LanguageConfig, embed_tokens: Optional[nn.Module] = None):
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = embed_tokens
        self.embed_positions = Florence2LearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([Florence2EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> "BaseModelOutput":
        """Forward pass.

        Args:
            input_ids: Input token ids.
            attention_mask: Attention mask.
            inputs_embeds: Input embeddings.
            head_mask: Head mask.
            output_attentions: Whether to output attention weights.
            output_hidden_states: Whether to output hidden states.

        Returns:
            BaseModelOutput with last hidden state, hidden states, and attentions.
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_shape = input.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input = inputs_embeds[:, :, 0]
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input)
        inputs_embeds = inputs_embeds + embed_pos
        inputs_embeds = self.layernorm_embedding(inputs_embeds)
        hidden_states = F.dropout(inputs_embeds, p=self.dropout, training=self.training)

        # Convert attention mask to the format expected by scaled_dot_product_attention
        # attn_mask should be True/False where True means "attend" and False means "don't attend"
        if attention_mask is not None:
            # attention_mask is [batch, seq_len] with 1 for attend, 0 for don't attend
            # Convert to bool: True for attend, False for don't attend
            attention_mask = attention_mask.bool()
            # Expand to 4D: [batch, 1, seq_len, seq_len] for broadcasting in attention
            # We need [batch, 1, tgt_len, src_len] where tgt_len = src_len = seq_len for self-attention
            seq_len = attention_mask.size(1)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(-1, -1, seq_len, seq_len)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            dropout_probability = torch.rand([])
            skip_the_layer = self.training and (dropout_probability < self.layerdrop)
            if not skip_the_layer:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    output_attentions=output_attentions,
                )
                hidden_states = layer_outputs[0]
            else:
                layer_outputs = (None, None)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class Florence2Decoder(nn.Module):
    """Florence2 decoder.

    Args:
        config: Language configuration.
        embed_tokens: Embedding layer.
    """

    def __init__(self, config: Florence2LanguageConfig, embed_tokens: Optional[nn.Module] = None):
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = embed_tokens
        self.embed_positions = Florence2LearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([Florence2DecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> "BaseModelOutputWithPastAndCrossAttentions":
        """Forward pass.

        Args:
            input_ids: Input token ids.
            attention_mask: Attention mask.
            encoder_hidden_states: Encoder hidden states.
            encoder_attention_mask: Encoder attention mask.
            head_mask: Head mask.
            cross_attn_head_mask: Cross-attention head mask.
            past_key_values: Past key values.
            inputs_embeds: Input embeddings.
            use_cache: Whether to use caching.
            output_attentions: Whether to output attention weights.
            output_hidden_states: Whether to output hidden states.

        Returns:
            BaseModelOutputWithPastAndCrossAttentions.
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_shape = input.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, 0]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        positions = self.embed_positions(input, past_key_values_length)
        inputs_embeds = inputs_embeds + positions
        inputs_embeds = self.layernorm_embedding(inputs_embeds)
        hidden_states = F.dropout(inputs_embeds, p=self.dropout, training=self.training)

        # Prepare attention masks for scaled dot product attention
        if attention_mask is not None:
            attention_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.bool().unsqueeze(1).unsqueeze(2)

        decoder_layers = self.layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(decoder_layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                cross_attn_layer_head_mask=(
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                ),
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


# =============================================================================
# Model Outputs
# =============================================================================

@dataclass
class BaseModelOutput:
    """Base model output.

    Args:
        last_hidden_state: Last hidden state.
        hidden_states: All hidden states if output_hidden_states=True.
        attentions: All attentions if output_attentions=True.
    """
    last_hidden_state: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


@dataclass
class BaseModelOutputWithPastAndCrossAttentions:
    """Base model output with past and cross attentions.

    Args:
        last_hidden_state: Last hidden state.
        past_key_values: Past key values if use_cache=True.
        hidden_states: All hidden states if output_hidden_states=True.
        attentions: All self attentions if output_attentions=True.
        cross_attentions: All cross attentions if output_attentions=True.
    """
    last_hidden_state: torch.Tensor
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.Tensor, ...]] = None


@dataclass
class Seq2SeqModelOutput:
    """Sequence-to-sequence model output.

    Args:
        last_hidden_state: Last decoder hidden state.
        past_key_values: Past key values.
        decoder_hidden_states: Decoder hidden states.
        decoder_attentions: Decoder attentions.
        cross_attentions: Cross attentions.
        encoder_last_hidden_state: Encoder last hidden state.
        encoder_hidden_states: Encoder hidden states.
        encoder_attentions: Encoder attentions.
    """
    last_hidden_state: torch.Tensor
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None
    decoder_hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.Tensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.Tensor, ...]] = None
    encoder_last_hidden_state: Optional[torch.Tensor] = None
    encoder_hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    encoder_attentions: Optional[Tuple[torch.Tensor, ...]] = None


@dataclass
class Seq2SeqLMOutput:
    """Sequence-to-sequence language model output.

    Args:
        loss: Language modeling loss.
        logits: Prediction logits.
        past_key_values: Past key values.
        decoder_hidden_states: Decoder hidden states.
        decoder_attentions: Decoder attentions.
        cross_attentions: Cross attentions.
        encoder_last_hidden_state: Encoder last hidden state.
        encoder_hidden_states: Encoder hidden states.
        encoder_attentions: Encoder attentions.
    """
    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None
    decoder_hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.Tensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.Tensor, ...]] = None
    encoder_last_hidden_state: Optional[torch.Tensor] = None
    encoder_hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    encoder_attentions: Optional[Tuple[torch.Tensor, ...]] = None


@dataclass
class ModelOutput:
    """Generic model output."""
    pass


# =============================================================================
# Florence2 Models
# =============================================================================

class Florence2VisionModel(nn.Module):
    """Florence2 vision model based on DaViT.

    Args:
        config: Vision configuration.
    """

    def __init__(self, config: Florence2VisionConfig):
        super().__init__()
        self.config = config
        self.model = DaViT.from_config(config)

        image_pos_embed = config.image_pos_embed
        if image_pos_embed["type"] == "learned_abs_2d":
            # Use the last dimension of dim_embed (actual DaViT output dimension)
            # not projection_dim (which is the target dimension after projection)
            actual_output_dim = config.dim_embed[-1]
            self.image_pos_embed = LearnedAbsolutePositionEmbedding2D(
                embedding_dim=actual_output_dim,
                num_pos=image_pos_embed["max_pos_embeddings"],
            )
        else:
            raise NotImplementedError(f"Unknown image_pos_embed type: {image_pos_embed['type']}")

        visual_temporal_embedding = config.visual_temporal_embedding
        if visual_temporal_embedding["type"] == "COSINE":
            # Use actual DaViT output dimension
            actual_output_dim = config.dim_embed[-1]
            self.visual_temporal_embed = PositionalEmbeddingCosine1D(
                embed_dim=actual_output_dim,
                max_seq_len=visual_temporal_embedding["max_temporal_embeddings"],
            )
        else:
            raise NotImplementedError(
                f"Unknown visual_temporal_embedding type: {visual_temporal_embedding['type']}"
            )

        self.image_feature_source = config.image_feature_source

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            pixel_values: Input images of shape (B, C, H, W) or (B, T, C, H, W).

        Returns:
            Image features.
        """
        batch_size = pixel_values.shape[0]
        if len(pixel_values.shape) == 5:
            # (B, T, C, H, W) -> (B*T, C, H, W)
            temporal_dim = pixel_values.shape[1]
            pixel_values = pixel_values.reshape(-1, *pixel_values.shape[2:])
        else:
            temporal_dim = 1

        # Get image features from DaViT
        image_embeds = self.model.forward_features_unpool(pixel_values)
        _, num_tokens, dim = image_embeds.shape

        # Add 2D spatial position embeddings
        height = width = int(num_tokens**0.5)
        image_embeds = image_embeds.reshape(batch_size * temporal_dim, height, width, dim)
        pos_embeds = self.image_pos_embed(image_embeds)
        image_embeds = image_embeds + pos_embeds
        image_embeds = image_embeds.reshape(batch_size * temporal_dim, num_tokens, dim)

        if temporal_dim > 1 and "temporal_avg_pool" in self.image_feature_source:
            # (B, T, num_tokens, dim)
            image_embeds = image_embeds.reshape(batch_size, temporal_dim, num_tokens, dim)
            # Add temporal embeddings - need to reshape to [B, T, D] for the embedding layer
            # Then broadcast back to [B, T, N, D]
            # Average over spatial dimension first to get [B, T, D]
            temp_for_embed = image_embeds.mean(dim=2)  # [B, T, D]
            temporal_embeds = self.visual_temporal_embed(temp_for_embed)  # [B, T, D]
            # Broadcast back to [B, T, N, D]
            temporal_embeds = temporal_embeds.unsqueeze(2).expand(-1, -1, num_tokens, -1)
            image_embeds = image_embeds + temporal_embeds
            # Average pool over time
            image_embeds = image_embeds.mean(dim=1)

        if "spatial_avg_pool" in self.image_feature_source:
            image_embeds = image_embeds.mean(dim=1)

        return image_embeds


class Florence2LanguageModel(nn.Module):
    """Florence2 language model based on BART.

    Args:
        config: Language configuration.
    """

    def __init__(self, config: Florence2LanguageConfig):
        super().__init__()
        self.config = config
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = Florence2ScaledWordEmbedding(
            vocab_size, config.d_model, padding_idx, embed_scale=math.sqrt(config.d_model) if config.scale_embedding else 1.0
        )
        self.encoder = Florence2Encoder(config, self.shared)
        self.decoder = Florence2Decoder(config, self.shared)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Seq2SeqModelOutput:
        """Forward pass.

        Args:
            input_ids: Input token ids.
            attention_mask: Attention mask.
            decoder_input_ids: Decoder input token ids.
            decoder_attention_mask: Decoder attention mask.
            encoder_outputs: Precomputed encoder outputs.
            past_key_values: Past key values for caching.
            inputs_embeds: Input embeddings.
            decoder_inputs_embeds: Decoder input embeddings.
            use_cache: Whether to use caching.
            output_attentions: Whether to output attention weights.
            output_hidden_states: Whether to output hidden states.

        Returns:
            Seq2SeqModelOutput.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class Florence2Model(nn.Module):
    """Florence2 multimodal model.

    Combines DaViT vision encoder with BART language model.

    Args:
        config: Florence2 configuration.
    """

    def __init__(self, config: Florence2Config):
        super().__init__()
        self.config = config

        if config.vision_config is not None:
            self.vision_model = Florence2VisionModel(config.vision_config)
            vision_projection_dim = config.vision_config.projection_dim
        else:
            self.vision_model = None
            vision_projection_dim = 0

        if config.text_config is not None:
            self.language_model = Florence2LanguageModel(config.text_config)
            text_config = config.text_config
        else:
            self.language_model = None
            text_config = None

        if vision_projection_dim > 0 and text_config is not None:
            # The DaViT output dimension is the last element of dim_embed (typically 2048)
            # not the projection_dim (which is 1024)
            vision_output_dim = config.vision_config.dim_embed[-1] if config.vision_config else vision_projection_dim
            self.vision_projection = nn.Linear(vision_output_dim, text_config.d_model)
        else:
            self.vision_projection = None

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Seq2SeqModelOutput:
        """Forward pass.

        Args:
            pixel_values: Input images.
            input_ids: Input token ids.
            attention_mask: Attention mask.
            decoder_input_ids: Decoder input token ids.
            decoder_attention_mask: Decoder attention mask.
            encoder_outputs: Precomputed encoder outputs.
            past_key_values: Past key values.
            inputs_embeds: Input embeddings.
            decoder_inputs_embeds: Decoder input embeddings.
            use_cache: Whether to use caching.
            output_attentions: Whether to output attention weights.
            output_hidden_states: Whether to output hidden states.

        Returns:
            Seq2SeqModelOutput.
        """
        use_cache = use_cache if use_cache is not None else self.config.text_config.use_cache if self.config.text_config else False
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        if pixel_values is not None and self.vision_model is not None:
            image_embeds = self.vision_model(pixel_values)
            if self.vision_projection is not None:
                image_embeds = self.vision_projection(image_embeds)

            # Expand to sequence length
            batch_size = image_embeds.shape[0]
            if len(image_embeds.shape) == 2:
                image_embeds = image_embeds.unsqueeze(1)

            if inputs_embeds is not None:
                inputs_embeds = torch.cat([image_embeds, inputs_embeds], dim=1)
            else:
                inputs_embeds = image_embeds

            if attention_mask is not None:
                image_attention_mask = torch.ones(
                    (batch_size, image_embeds.shape[1]), dtype=attention_mask.dtype, device=attention_mask.device
                )
                attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)

        return self.language_model(
            input_ids=None if inputs_embeds is not None else input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )


class Florence2ForConditionalGeneration(nn.Module):
    """Florence2 for conditional generation.

    Complete Florence-2 model with language modeling head.

    Args:
        config: Florence2 configuration.
    """

    def __init__(self, config: Florence2Config):
        super().__init__()
        self.config = config
        self.model = Florence2Model(config)

        if config.text_config is not None:
            self.lm_head = nn.Linear(config.text_config.d_model, config.vocab_size, bias=False)
        else:
            self.lm_head = None

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Seq2SeqLMOutput:
        """Forward pass for conditional generation.

        Args:
            pixel_values: Input images.
            input_ids: Input token ids.
            attention_mask: Attention mask.
            decoder_input_ids: Decoder input token ids.
            decoder_attention_mask: Decoder attention mask.
            encoder_outputs: Precomputed encoder outputs.
            past_key_values: Past key values.
            inputs_embeds: Input embeddings.
            decoder_inputs_embeds: Decoder input embeddings.
            labels: Target labels for computing loss.
            use_cache: Whether to use caching.
            output_attentions: Whether to output attention weights.
            output_hidden_states: Whether to output hidden states.

        Returns:
            Seq2SeqLMOutput.
        """
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                labels, self.config.text_config.pad_token_id, self.config.text_config.decoder_start_token_id
            )

        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        lm_logits = self.lm_head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        **kwargs,
    ) -> dict:
        """Prepare inputs for generation.

        Args:
            decoder_input_ids: Decoder input token ids.
            past_key_values: Past key values.
            attention_mask: Attention mask.
            decoder_attention_mask: Decoder attention mask.
            encoder_outputs: Encoder outputs.
            **kwargs: Additional arguments.

        Returns:
            Dictionary of model inputs.
        """
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        if decoder_attention_mask is not None and past_key_values is not None:
            decoder_attention_mask = decoder_attention_mask[:, -1:]

        return {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "pixel_values": kwargs.get("pixel_values", None),
        }


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    "DropPath",
    "LearnedAbsolutePositionEmbedding2D",
    "PositionalEmbeddingCosine1D",
    "LearnedAbsolutePositionEmbedding1D",
    "Florence2LearnedPositionalEmbedding",
    "ConvEmbed",
    "ChannelAttention",
    "ChannelBlock",
    "WindowAttention",
    "SpatialBlock",
    "DaViT",
    "Florence2Attention",
    "Florence2SdpaAttention",
    "Florence2FlashAttention2",
    "Florence2EncoderLayer",
    "Florence2Encoder",
    "Florence2DecoderLayer",
    "Florence2Decoder",
    "Florence2VisionModel",
    "Florence2LanguageModel",
    "Florence2Model",
    "Florence2ForConditionalGeneration",
    "BaseModelOutput",
    "BaseModelOutputWithPastAndCrossAttentions",
    "Seq2SeqModelOutput",
    "Seq2SeqLMOutput",
]
