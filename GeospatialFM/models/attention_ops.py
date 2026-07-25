from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp, DropPath, use_fused_attn

from .low_rank_attention import AttentionBranch, LowDimPool, LowRankAttention
from .pos_chan_embed import rope_rotate_half

__all__ = [
    'SpatialSpectralAttention',
    'LESSAttention',
    'FullSpatialSpectralAttention',
    'AdditiveSpatialSpectralAttention',
    'build_attention',
    'LayerScale',
    'LowRankBlock',
]


class SpatialSpectralAttention(nn.Module):
    """
    Common interface for the block-level spatial-spectral attention operator.

    forward(x, spatial_mask=None, rope=None) -> Tensor
        x: [B, C, HW, D] (C and HW here are whatever the caller's grid axis sizes
           currently are, e.g. C+1/HW+1 when CLS tokens are folded in).
        spatial_mask: optional perception-field mask, [HW, HW] or None.
        rope: optional (sin_hw, cos_hw, sin_c, cos_c) tuple from
           RopePositionChannelEmbedding, or None.
        returns: [B, C, HW, D], same shape as x.
    """
    pass


class LESSAttention(SpatialSpectralAttention):
    """
    Arm A: the existing LESS (Kronecker) attention. Thin composition of the
    unmodified LowDimPool + LowRankAttention classes behind the unified interface;
    no math differs from the original LowRankBlock.forward.
    """
    def __init__(
        self,
        dim: int,
        channel_dim: int,
        spatial_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        rank: int = 1,
        skip_pool: bool = False,
    ) -> None:
        super().__init__()
        self.low_dim_pool = LowDimPool(
            dim=dim,
            channel_dim=channel_dim,
            spatial_dim=spatial_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            skip_pool=skip_pool,
        )
        self.channel_norm = norm_layer(channel_dim)
        self.spatial_norm = norm_layer(spatial_dim)
        self.attn = LowRankAttention(
            dim=dim,
            channel_dim=channel_dim,
            spatial_dim=spatial_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            rank=rank,
        )

    def forward(self, x: torch.Tensor, spatial_mask: torch.Tensor = None, rope: tuple = None) -> torch.Tensor:
        x_c, x_s, _ = self.low_dim_pool(x)
        return self.attn(self.channel_norm(x_c), self.spatial_norm(x_s), spatial_mask, rope)


class AdditiveSpatialSpectralAttention(SpatialSpectralAttention):
    """
    Arm C: *identical* channel/spatial branch attentions as arm A -- same
    LowDimPool, same AttentionBranch class, same channel_dim/spatial_dim (so
    c_head_dim/s_head_dim keep arm A's low-rank widths, e.g. 4 and 16 for ViT-S,
    NOT widened to D_h). This isolates the fusion mechanism specifically, rather
    than also widening the branch attentions (an earlier version of this class
    set channel_dim=spatial_dim=dim, which inflated LowDimPool's projections and
    the branch attentions themselves and produced a ~3x arm A param count -- this
    version's param delta over arm A is just P_c/P_s, i.e. genuinely "slight").

    Fusion: each branch's per-head output is projected *up* to the full head_dim
    D_h only at the fusion step, then broadcast-added:

        Y[c, n] = P_c(Y_C[c]) + P_s(Y_S[n]),   P_c: R^c_head_dim -> R^D_h,
                                                 P_s: R^s_head_dim -> R^D_h

    P_c/P_s are shared across heads (a single small matrix broadcast over the
    heads dimension, mirroring how the Kronecker fusion also has no per-head-
    distinct fusion parameters). `fusion_init_scale` rescales P_c/P_s's
    initialization so that arm C's post-fusion activation std at initialization
    matches arm A's (see ablation_sanity_checks.py check #4) -- the two fusion
    mechanisms are algebraically different (product vs. sum) so matching
    variance requires an explicit scale, not just "sensible" init.
    """
    def __init__(
        self,
        dim: int,
        channel_dim: int,
        spatial_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        rank: int = 1,
        skip_pool: bool = False,
        fusion_init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        assert channel_dim % num_heads == 0, 'channel_dim should be divisible by num_heads'
        assert spatial_dim % num_heads == 0, 'spatial_dim should be divisible by num_heads'
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.c_head_dim = channel_dim // num_heads
        self.s_head_dim = spatial_dim // num_heads
        self.rank = rank

        self.low_dim_pool = LowDimPool(
            dim=dim,
            channel_dim=channel_dim,
            spatial_dim=spatial_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            skip_pool=skip_pool,
        )
        self.channel_norm = norm_layer(channel_dim)
        self.spatial_norm = norm_layer(spatial_dim)

        self.channel_branch = AttentionBranch(channel_dim, num_heads, self.c_head_dim, qkv_bias, qk_norm, attn_drop, norm_layer, rank)
        self.spatial_branch = AttentionBranch(spatial_dim, num_heads, self.s_head_dim, qkv_bias, qk_norm, attn_drop, norm_layer, rank)

        self.P_c = nn.Linear(self.c_head_dim, self.head_dim)
        self.P_s = nn.Linear(self.s_head_dim, self.head_dim)
        with torch.no_grad():
            # Scale the whole affine map (weight AND bias) so that P(x) -> scale*P(x)
            # exactly, uniformly rescaling arm C's post-fusion output to match arm A's
            # activation std at init (see ablation_sanity_checks.check_fusion_init_scale_calibration).
            self.P_c.weight.mul_(fusion_init_scale)
            self.P_c.bias.mul_(fusion_init_scale)
            self.P_s.weight.mul_(fusion_init_scale)
            self.P_s.bias.mul_(fusion_init_scale)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_grid: torch.Tensor, spatial_mask: torch.Tensor = None, rope: tuple = None) -> torch.Tensor:
        x_c, x_s, _ = self.low_dim_pool(x_grid)
        C = x_c.shape[1]
        HW = x_s.shape[1]

        sin_hw, cos_hw, sin_c, cos_c = rope if rope is not None else (None, None, None, None)

        xc = self.channel_branch(self.channel_norm(x_c), sin=sin_c, cos=cos_c)  # B, rank, num_heads, C, c_head_dim
        xs = self.spatial_branch(self.spatial_norm(x_s), spatial_mask, sin=sin_hw, cos=cos_hw)  # B, rank, num_heads, HW, s_head_dim

        xc = xc.sum(dim=1) / self.rank  # B, num_heads, C, c_head_dim
        xs = xs.sum(dim=1) / self.rank  # B, num_heads, HW, s_head_dim

        Yc = self.P_c(xc)  # B, num_heads, C, head_dim
        Ys = self.P_s(xs)  # B, num_heads, HW, head_dim

        Y = Yc.unsqueeze(3) + Ys.unsqueeze(2)  # B, num_heads, C, HW, head_dim
        B = Y.shape[0]
        Y = Y.permute(0, 2, 3, 1, 4).reshape(B, C, HW, -1)  # B, C, HW, D
        Y = self.proj(Y)
        Y = self.proj_drop(Y)
        return Y


def _apply_axis_rope(x: torch.Tensor, sin: Optional[torch.Tensor], cos: Optional[torch.Tensor], axis: int) -> torch.Tensor:
    """
    Rotate a sub-block of the head dimension of `x` ([B, num_heads, C, HW, W])
    along one grid axis (axis=2 for the channel axis, axis=3 for the position
    axis), skipping index 0 along that axis (the channel-CLS row when axis=2, the
    spatial-CLS column when axis=3). sin/cos: [B, L, W] where L == x.shape[axis]-1.

    Reuses the exact rotate-half formula from pos_chan_embed.rope_apply; only the
    broadcasting is generalized here to the extra grid axis that arm A's per-branch
    tensors never had.
    """
    if sin is None:
        return x
    L = x.shape[axis] - 1
    x_cls, x_rest = x.split([1, L], dim=axis)
    view_shape = [x.shape[0]] + [1] * (x.dim() - 2) + [x.shape[-1]]
    view_shape[axis] = L
    sin = sin.reshape(view_shape)
    cos = cos.reshape(view_shape)
    x_rot = x_rest * cos + rope_rotate_half(x_rest) * sin
    return torch.cat([x_cls, x_rot], dim=axis)


class FullSpatialSpectralAttention(SpatialSpectralAttention):
    """
    Arm B: standard full softmax attention over the flattened (channel, position)
    token grid (paper Appendix E, Algorithm 2). No low-rank branch factorization,
    no Kronecker fusion -- a single Q/K/V, softmax over all C*HW tokens.

    Joint RoPE: partitions the head dimension D_h into a spatial (u,v) sub-block
    of width `spatial_split_dim` and a spectral (lambda) sub-block of width
    `spectral_split_dim` (spatial_split_dim + spectral_split_dim == D_h). Each
    sub-block is rotated independently -- spatial sub-block using (sin_hw, cos_hw)
    broadcast over the channel axis, spectral sub-block using (sin_c, cos_c)
    broadcast over the position axis -- reusing RopePositionChannelEmbedding's
    existing frequency construction unmodified (see model config: rope is built
    with rope_spatial_split_dim/rope_channel_split_dim instead of the per-branch
    spatial_dim/channel_dim that arms A/C use). Rotation is applied while the grid
    is still 2D-indexed (before the C/HW flatten), so each axis's CLS-skip is
    independent: real tokens (c>=1, n>=1) get both rotations; the channel-CLS row
    (c=0) gets spatial rotation only; the spatial-CLS column (n=0) gets spectral
    rotation only; the global CLS (0,0) gets neither.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim % 2 == 0, 'head_dim must be even for RoPE rotate-half'
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.spatial_split_dim = self.head_dim // 2
        self.spectral_split_dim = self.head_dim - self.spatial_split_dim
        assert self.spatial_split_dim % 2 == 0 and self.spectral_split_dim % 2 == 0, \
            'both rope sub-blocks must be even for rotate-half'

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def apply_joint_rope(self, q: torch.Tensor, k: torch.Tensor, rope: tuple):
        sin_hw, cos_hw, sin_c, cos_c = rope

        def rotate(t: torch.Tensor) -> torch.Tensor:
            t_spa, t_spec = t.split([self.spatial_split_dim, self.spectral_split_dim], dim=-1)
            t_spa = _apply_axis_rope(t_spa, sin_hw, cos_hw, axis=3)   # rotate along the HW (position) axis
            t_spec = _apply_axis_rope(t_spec, sin_c, cos_c, axis=2)  # rotate along the C (channel) axis
            return torch.cat([t_spa, t_spec], dim=-1)

        return rotate(q), rotate(k)

    def forward(self, x: torch.Tensor, spatial_mask: torch.Tensor = None, rope: tuple = None) -> torch.Tensor:
        if spatial_mask is not None:
            raise NotImplementedError(
                'perception_field_mask is not supported for attn_type="full" in this ablation '
                '(not exercised by the shared base config -- fail loudly rather than silently ignore it)'
            )
        B, C, HW, D = x.shape
        qkv = self.qkv(x).reshape(B, C, HW, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv.unbind(0)  # B, num_heads, C, HW, head_dim

        if rope is not None:
            q, k = self.apply_joint_rope(q, k, rope)

        q, k = self.q_norm(q), self.k_norm(k)

        q = q.reshape(B, self.num_heads, C * HW, self.head_dim)
        k = k.reshape(B, self.num_heads, C * HW, self.head_dim)
        v = v.reshape(B, self.num_heads, C * HW, self.head_dim)

        if self.fused_attn:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v

        out = out.reshape(B, self.num_heads, C, HW, self.head_dim)
        out = out.permute(0, 2, 3, 1, 4).reshape(B, C, HW, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


def build_attention(
    attn_type: str,
    *,
    dim: int,
    channel_dim: int,
    spatial_dim: int,
    num_heads: int,
    rank: int = 1,
    qkv_bias: bool = False,
    qk_norm: bool = False,
    attn_drop: float = 0.,
    proj_drop: float = 0.,
    norm_layer: nn.Module = nn.LayerNorm,
    skip_pool: bool = False,
    fusion_init_scale: float = 1.0,
) -> SpatialSpectralAttention:
    if attn_type == 'less':
        return LESSAttention(
            dim=dim, channel_dim=channel_dim, spatial_dim=spatial_dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_norm=qk_norm, attn_drop=attn_drop, proj_drop=proj_drop,
            norm_layer=norm_layer, rank=rank, skip_pool=skip_pool,
        )
    elif attn_type == 'additive':
        return AdditiveSpatialSpectralAttention(
            dim=dim, channel_dim=channel_dim, spatial_dim=spatial_dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_norm=qk_norm, attn_drop=attn_drop, proj_drop=proj_drop,
            norm_layer=norm_layer, rank=rank, skip_pool=skip_pool, fusion_init_scale=fusion_init_scale,
        )
    elif attn_type == 'full':
        return FullSpatialSpectralAttention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
            attn_drop=attn_drop, proj_drop=proj_drop, norm_layer=norm_layer,
        )
    else:
        raise ValueError(f'Unknown attn_type: {attn_type!r}, expected one of "less", "full", "additive"')


class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class LowRankBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            channel_dim: int,
            spatial_dim: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            init_values: Optional[float] = None,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
            skip_pool: bool = False,
            rank: int = 1,
            use_rope_embed: bool = False,
            attn_type: str = 'less',
            fusion_init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = build_attention(
            attn_type,
            dim=dim,
            channel_dim=channel_dim,
            spatial_dim=spatial_dim,
            num_heads=num_heads,
            rank=rank,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            skip_pool=skip_pool,
            fusion_init_scale=fusion_init_scale,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.use_rope_embed = use_rope_embed

    def forward(self, x: torch.Tensor, spatial_mask: torch.Tensor = None, pos_chan_embedding: torch.Tensor = None) -> torch.Tensor:
        rope = pos_chan_embedding if self.use_rope_embed else None

        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), spatial_mask, rope)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
