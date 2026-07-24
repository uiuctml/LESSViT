"""
Ported from https://github.com/Jingtao-Li-CVer/HyperFree (HyperFree/modeling/image_encoder.py)

Only the backbone (`ImageEncoderViT`) is ported here -- the prompt encoder / mask decoder /
SAM wrapper from the upstream repo are not needed since GeospatialFM only uses HyperFree as a
feature-extracting encoder baseline, the same way it uses DOFA/DINOv3/SpecViT/etc.

Dead code from upstream that isn't reachable from `ImageEncoderViT.forward` (the unused
`PatchEmbed`/`to_2tuple`/`trunc_normal_` helpers, the torch<1.8 `container_abcs` shim) has been
dropped for clarity; the numerics of the ported code are otherwise unchanged.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type

from .common import LayerNorm2d, MLPBlock
from .scale_aware_PE import get_2d_sincos_pos_embed_with_resolution
from .spectral_process_utils import (
    spectral_wavelength,
    weight_bank_wavelength,
    input_wavelengths_hy,
    input_wavelengths_mu,
    find_corresponding_indices,
    generate_random_indices,
)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        merge_indexs=[3, 6, 8, 11],  # for Vit-b version
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
            merge_indexs (list or None): Block indices after which patch merging is applied.
                Pass None to keep a single-scale feature map throughout (used by the
                GeospatialFM wrapper so the output grid always matches img_size / patch_size).
        """
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.patch_size = patch_size
        self.out_chans = out_chans

        self.pos_embed_mlp = MLP(self.embed_dim, self.embed_dim // 2, self.embed_dim, 3, sigmoid_output=False)

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.contras_modules = nn.ModuleList()
        for i in range(2):
            block = Block(
                dim=256,
                num_heads=8,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=16,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.contras_modules.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

        self.nm_dis = 10
        self.Band_feature_indices_hy, self.unmatch_indices_hy, self.point_bank_indices_hy = find_corresponding_indices(input_wavelengths_hy, spectral_wavelength, self.nm_dis)
        self.Band_feature_indices_mu, self.unmatch_indices_mu, self.point_bank_indices_mu = find_corresponding_indices(input_wavelengths_mu, spectral_wavelength, self.nm_dis)
        self.weight_bank_data_indices_hy, _, self.weight_bank_indices_hy = find_corresponding_indices(input_wavelengths_hy, weight_bank_wavelength, self.nm_dis)

        self.point_spectral_weight_bank_w = nn.Parameter(torch.randn((self.embed_dim, len(spectral_wavelength), patch_size, patch_size)))
        self.point_spectral_weight_bank_b = nn.Parameter(torch.randn(self.embed_dim))
        self.block_spectral_weight_bank_w = nn.Parameter(torch.randn((self.embed_dim, len(weight_bank_wavelength), patch_size, patch_size)))
        self.block_spectral_weight_bank_b = nn.Parameter(torch.randn(self.embed_dim))

        self.merge_indexs = merge_indexs
        self.global_attn_indexes = global_attn_indexes
        self.multi_scale_convs = nn.ModuleList([
            PatchMerging(dim=embed_dim)
            for i in range(len(self.merge_indexs))]) if self.merge_indexs is not None else None

    def convert_semantic_feature(self, backbone_features):
        backbone_features = backbone_features.permute((0, 2, 3, 1))

        for i, blk in enumerate(self.contras_modules):
            backbone_features = blk(backbone_features)

        contras_features = (backbone_features.permute(0, 3, 1, 2))
        return contras_features

    def find_indices_not_in_A(self, A, B):
        set_A = set(A)
        result_indices = []
        for index, element in enumerate(B):
            if element not in set_A:
                result_indices.append(index)
        return result_indices

    def forward(self, x: torch.Tensor, test_mode=False, input_wavelength=None, GSD=None) -> torch.Tensor:
        """
        Args:
            x (tensor): input image with [B, C, H, W].
            test_mode (bool): If true, all the input channels would be used.
                              If false, we would randomly select 40 channels for each iteration
            input_wavelength: list, storing wavelengths (in nm) for each hyperspectral channel
            GSD: ground sampling distance (m/pixel). list, such as [1.0] or tensor, such as torch.tensor([1.0])

        Returns:  multi-stage backbone features
        """

        is_hy = False
        is_mu = False

        if x.shape[1] < 20:
            is_mu = True
        else:
            is_hy = True

        if x.shape[1] == 224 and not test_mode:
            x = x[:, 4:224, :, :]

        if input_wavelength is not None and is_hy:
            input_wavelengths_hy = input_wavelength
            self.Band_feature_indices_hy, self.unmatch_indices_hy, self.point_bank_indices_hy = find_corresponding_indices(input_wavelengths_hy, spectral_wavelength, self.nm_dis)
            self.weight_bank_data_indices_hy, _, self.weight_bank_indices_hy = find_corresponding_indices(input_wavelengths_hy, weight_bank_wavelength, self.nm_dis)
        elif input_wavelength is not None and is_mu:
            input_wavelengths_mu = input_wavelength
            self.Band_feature_indices_mu, self.unmatch_indices_mu, self.point_bank_indices_mu = find_corresponding_indices(input_wavelengths_mu, spectral_wavelength, self.nm_dis)

        if is_hy:
            if not test_mode:
                random_indices = generate_random_indices(len(self.weight_bank_data_indices_hy) - 1, 40)
                random_indices.sort()
                indices = [self.Band_feature_indices_hy, self.point_bank_indices_hy, np.array(self.weight_bank_data_indices_hy)[random_indices].tolist(), np.array(self.weight_bank_indices_hy)[random_indices].tolist()]
            else:
                indices = [self.Band_feature_indices_hy, self.point_bank_indices_hy, self.weight_bank_data_indices_hy, self.weight_bank_indices_hy]
            block_indices = self.find_indices_not_in_A(indices[0], indices[2])
            indices[2] = np.array(indices[2])[block_indices].tolist()
            indices[3] = np.array(indices[3])[block_indices].tolist()
            self.last_indices = indices
        elif is_mu:
            indices = [self.Band_feature_indices_mu, self.point_bank_indices_mu, [], []]
            self.last_indices = indices

        if GSD is None:
            GSD = [1.0]
        if not torch.is_tensor(GSD):
            GSD = torch.tensor(GSD)

        point_feature = F.conv2d(
            x[:, indices[0], :, :],
            weight=self.point_spectral_weight_bank_w[:, indices[1], :, :],
            bias=self.point_spectral_weight_bank_b,
            stride=(self.patch_size, self.patch_size),
            padding=(0, 0)
        )

        if len(indices[2]) > 0:
            block_feature = F.conv2d(
                x[:, indices[2], :, :],
                weight=self.block_spectral_weight_bank_w[:, indices[3], :, :],
                bias=self.block_spectral_weight_bank_b,
                stride=(self.patch_size, self.patch_size),
            )

        if len(indices[2]) > 0:
            x_feature = point_feature + block_feature
        else:
            x_feature = point_feature

        scale_aware_pos_embed = get_2d_sincos_pos_embed_with_resolution(self.embed_dim, int(self.img_size / self.patch_size), GSD, device=x.device)
        scale_aware_pos_embed = self.pos_embed_mlp(scale_aware_pos_embed)
        scale_aware_pos_embed = scale_aware_pos_embed.reshape((x.shape[0], int(self.img_size / self.patch_size), int(self.img_size / self.patch_size), self.embed_dim))

        x_feature = x_feature.permute((0, 2, 3, 1))
        x_feature = x_feature + scale_aware_pos_embed

        self.multi_stage_features = []

        multi_scale_merge_index = 0
        for i, blk in enumerate(self.blocks):
            if self.patch_size <= 8:
                x_feature = torch.utils.checkpoint.checkpoint(blk, x_feature, use_reentrant=True)
            else:
                x_feature = blk(x_feature)

            if self.merge_indexs is not None:
                if i in [self.merge_indexs[0], self.global_attn_indexes[0], self.global_attn_indexes[2]]:
                    self.multi_stage_features.append(x_feature.permute(0, 3, 1, 2))

                if i in self.merge_indexs:
                    x_feature = self.multi_scale_convs[multi_scale_merge_index](x_feature)
                    multi_scale_merge_index += 1
            elif i in self.global_attn_indexes:
                self.multi_stage_features.append(x_feature.permute(0, 3, 1, 2))

        x_feature = self.neck(x_feature.permute(0, 3, 1, 2))
        self.multi_stage_features.append(x_feature)

        return self.multi_stage_features


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H, W, C
        """
        B, H, W, C = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        x = x.view(B, H // 2, W // 2, C)

        return x
