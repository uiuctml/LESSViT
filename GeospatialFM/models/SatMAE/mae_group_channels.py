# --------------------------------------------------------
# Masked Autoencoder with Grouped Channels for Hugging Face
# Based on: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import torch
import torch.nn as nn
from functools import partial
from typing import Optional, Tuple, Union, List

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import ModelOutput
from timm.models.vision_transformer import PatchEmbed, Block

from .pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid


class MaskedAutoencoderGroupChannelViTConfig(PretrainedConfig):
    """
    Configuration class for MaskedAutoencoderGroupChannelViT.
    
    This configuration defines the architecture parameters for a Masked Autoencoder
    with Vision Transformer backbone that handles grouped channels.
    """
    
    model_type = "masked_autoencoder_group_channel_vit"
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        spatial_mask: bool = False,
        channel_groups: Tuple[Tuple[int, ...], ...] = ((0, 1, 2, 6), (3, 4, 5, 7), (8, 9)),
        channel_embed: int = 256,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        decoder_channel_embed: int = 128,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_pix_loss: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.spatial_mask = spatial_mask
        self.channel_groups = channel_groups
        self.channel_embed = channel_embed
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.decoder_channel_embed = decoder_channel_embed
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_pix_loss = norm_pix_loss
        
        # Calculate derived parameters
        self.num_groups = len(channel_groups)
        self.num_patches = (img_size // patch_size) ** 2
        
        # Validate channel groups
        all_channels = set()
        for group in channel_groups:
            for channel in group:
                if channel in all_channels:
                    raise ValueError(f"Channel {channel} appears in multiple groups")
                all_channels.add(channel)
       
class MaskedAutoencoderGroupChannelViT(PreTrainedModel):
    """
    Masked Autoencoder with VisionTransformer backbone for grouped channels.
    
    This model implements a masked autoencoder that processes multi-channel images
    by grouping channels and applying separate patch embeddings for each group.
    """
    
    config_class = MaskedAutoencoderGroupChannelViTConfig
    
    def __init__(self, config: MaskedAutoencoderGroupChannelViTConfig):
        super().__init__(config)
        
        self.config = config
        self.in_c = config.in_chans
        self.patch_size = config.patch_size
        self.channel_groups = config.channel_groups
        self.spatial_mask = config.spatial_mask
        num_groups = len(config.channel_groups)

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = nn.ModuleList([
            PatchEmbed(config.img_size, config.patch_size, len(group), config.embed_dim)
            for group in config.channel_groups
        ])
        num_patches = self.patch_embed[0].num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.embed_dim - config.channel_embed),
            requires_grad=False
        )  # fixed sin-cos embedding
        self.channel_embed = nn.Parameter(
            torch.zeros(1, num_groups, config.channel_embed), 
            requires_grad=False
        )

        self.blocks = nn.ModuleList([
            Block(config.embed_dim, config.num_heads, config.mlp_ratio, 
                  qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(config.depth)
        ])
        self.norm = nn.LayerNorm(config.embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(config.embed_dim, config.decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.decoder_embed_dim - config.decoder_channel_embed),
            requires_grad=False
        )  # fixed sin-cos embedding
        # Extra channel for decoder to represent special place for cls token
        self.decoder_channel_embed = nn.Parameter(
            torch.zeros(1, num_groups + 1, config.decoder_channel_embed),
            requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList([
            Block(config.decoder_embed_dim, config.decoder_num_heads, config.mlp_ratio, 
                  qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(config.decoder_depth)
        ])

        self.decoder_norm = nn.LayerNorm(config.decoder_embed_dim)

        self.decoder_pred = nn.ModuleList([
            nn.Linear(config.decoder_embed_dim, len(group) * config.patch_size**2)
            for group in config.channel_groups
        ])
        # --------------------------------------------------------------------------

        self.norm_pix_loss = config.norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights using the same strategy as the original implementation."""
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], 
            int(self.patch_embed[0].num_patches ** .5),
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        channel_embed = get_1d_sincos_pos_embed_from_grid(
            self.channel_embed.shape[-1],
            torch.arange(len(self.channel_groups)).numpy()
        )
        self.channel_embed.data.copy_(torch.from_numpy(channel_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed[0].num_patches ** .5), 
            cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        dec_channel_embed = get_1d_sincos_pos_embed_from_grid(
            self.decoder_channel_embed.shape[-1],
            torch.arange(len(self.channel_groups) + 1).numpy()
        )
        self.decoder_channel_embed.data.copy_(torch.from_numpy(dec_channel_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        for patch_embed in self.patch_embed:
            w = patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights for linear and layer norm layers."""
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs, p, c):
        """
        Convert images to patches.
        
        Args:
            imgs: (N, C, H, W)
            p: Patch embed patch size
            c: Num channels
        Returns:
            x: (N, L, C*patch_size**2)
        """
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * c))
        return x

    def unpatchify(self, x, p, c):
        """
        Convert patches back to images.
        
        Args:
            x: (N, L, C*patch_size**2)
            p: Patch embed patch size
            c: Num channels
        Returns:
            imgs: (N, C, H, W)
        """
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, c, p, p))
        x = torch.einsum('nhwcpq->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        
        Args:
            x: [N, L, D], sequence
            mask_ratio: ratio of patches to mask
        Returns:
            x_masked: masked sequence
            mask: binary mask (0 is keep, 1 is remove)
            ids_restore: indices to restore original order
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        """
        Forward pass through the encoder.
        
        Args:
            x: (N, C, H, W) input images
            mask_ratio: ratio of patches to mask
        Returns:
            x: encoded features
            mask: binary mask
            ids_restore: indices to restore original order
        """
        b, c, h, w = x.shape

        x_c_embed = []
        for i, group in enumerate(self.channel_groups):
            x_c = x[:, group, :, :]
            x_c_embed.append(self.patch_embed[i](x_c))  # (N, L, D)

        x = torch.stack(x_c_embed, dim=1)  # (N, G, L, D)
        _, G, L, D = x.shape

        # add channel embed
        channel_embed = self.channel_embed.unsqueeze(2)  # (1, G, 1, cD)
        pos_embed = self.pos_embed[:, 1:, :].unsqueeze(1)  # (1, 1, L, pD)

        # Channel embed same across (x,y) position, and pos embed same across channel (c)
        channel_embed = channel_embed.expand(-1, -1, pos_embed.shape[2], -1)  # (1, G, L, cD)
        pos_embed = pos_embed.expand(-1, channel_embed.shape[1], -1, -1)  # (1, G, L, pD)
        pos_channel = torch.cat((pos_embed, channel_embed), dim=-1)  # (1, G, L, D)

        # add pos embed w/o cls token
        x = x + pos_channel  # (N, G, L, D)

        if self.spatial_mask:
            # Mask spatial location across all channels (i.e. spatial location as either all/no channels)
            x = x.permute(0, 2, 1, 3).reshape(b, L, -1)  # (N, L, G*D)
            x, mask, ids_restore = self.random_masking(x, mask_ratio)  # (N, 0.25*L, G*D)
            x = x.view(b, x.shape[1], G, D).permute(0, 2, 1, 3).reshape(b, -1, D)  # (N, 0.25*G*L, D)
            mask = mask.repeat(1, G)  # (N, G*L)
            mask = mask.view(b, G, L)
        else:
            # Independently mask each channel (i.e. spatial location has subset of channels visible)
            x, mask, ids_restore = self.random_masking(x.view(b, -1, D), mask_ratio)  # (N, 0.25*G*L, D)
            mask = mask.view(b, G, L)

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (N, G*L + 1, D)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """
        Forward pass through the decoder.
        
        Args:
            x: encoded features
            ids_restore: indices to restore original order
        Returns:
            x: reconstructed patches
        """
        # embed tokens
        x = self.decoder_embed(x)  # (N, 1 + G*0.25*L, D)

        # append mask tokens to sequence
        G = len(self.channel_groups)
        if self.spatial_mask:
            N, L = ids_restore.shape

            x_ = x[:, 1:, :].view(N, G, -1, x.shape[2]).permute(0, 2, 1, 3)  # (N, 0.25*L, G, D)
            _, ml, _, D = x_.shape
            x_ = x_.reshape(N, ml, G * D)  # (N, 0.25*L, G*D)

            mask_tokens = self.mask_token.repeat(N, L - ml, G)
            x_ = torch.cat((x_, mask_tokens), dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[2]))  # (N, L, G*D)
            x_ = x_.view(N, L, G, D).permute(0, 2, 1, 3).reshape(N, -1, D)  # (N, G*L, D)
            x = torch.cat((x[:, :1, :], x_), dim=1)  # append cls token  (N, 1 + G*L, D)
        else:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token  (N, 1 + c*L, D)

        # add pos and channel embed
        channel_embed = self.decoder_channel_embed[:, :-1, :].unsqueeze(2)  # (1, G, 1, cD)
        pos_embed = self.decoder_pos_embed[:, 1:, :].unsqueeze(1)  # (1, 1, L, pD)

        channel_embed = channel_embed.expand(-1, -1, pos_embed.shape[2], -1)  # (1, G, L, cD)
        pos_embed = pos_embed.expand(-1, channel_embed.shape[1], -1, -1)  # (1, G, L, pD)
        pos_channel = torch.cat((pos_embed, channel_embed), dim=-1)  # (1, G, L, D)
        pos_channel = pos_channel.view(1, -1, pos_channel.shape[-1])  # (1, G*L, D)

        extra = torch.cat((self.decoder_pos_embed[:, :1, :],
                           self.decoder_channel_embed[:, -1:, :]), dim=-1)  # (1, 1, D)

        pos_channel = torch.cat((extra, pos_channel), dim=1)  # (1, 1+G*L, D)
        x = x + pos_channel  # (N, 1+G*L, D)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # remove cls token
        x = x[:, 1:, :]

        # Separate channel axis
        N, GL, D = x.shape
        x = x.view(N, G, GL//G, D)

        # predictor projection
        x_c_patch = []
        for i, group in enumerate(self.channel_groups):
            x_c = x[:, i]  # (N, L, D)
            dec = self.decoder_pred[i](x_c)  # (N, L, g_c * p^2)
            dec = dec.view(N, x_c.shape[1], -1, int(self.patch_size**2))  # (N, L, g_c, p^2)
            dec = torch.einsum('nlcp->nclp', dec)  # (N, g_c, L, p^2)
            x_c_patch.append(dec)

        x = torch.cat(x_c_patch, dim=1)  # (N, c, L, p**2)
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        Compute reconstruction loss.
        
        Args:
            imgs: [N, c, H, W] target images
            pred: [N, L, c*p*p] predicted patches
            mask: [N, L] binary mask (0 is keep, 1 is remove)
        Returns:
            loss: reconstruction loss
        """
        target = self.patchify(imgs, self.patch_embed[0].patch_size[0], self.in_c)  # (N, L, C*P*P)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        N, L, _ = target.shape
        target = target.view(N, L, self.in_c, -1)  # (N, L, C, p^2)
        target = torch.einsum('nlcp->nclp', target)  # (N, C, L, p^2)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, C, L], mean loss per patch

        total_loss, num_removed = 0., 0.
        for i, group in enumerate(self.channel_groups):
            group_loss = loss[:, group, :].mean(dim=1)  # (N, L)
            total_loss += (group_loss * mask[:, i]).sum()
            num_removed += mask[:, i].sum()  # mean loss on removed patches

        return total_loss/num_removed

    def forward(
        self,
        optical: torch.Tensor,
        mask_ratio: float = 0.75,
        return_dict: bool = True,
        **kwargs
    ):
        """
        Forward pass of the model.
        
        Args:
            pixel_values: Input images of shape (batch_size, num_channels, height, width)
            mask_ratio: Ratio of patches to mask (default: 0.75)
            return_dict: Whether to return a ModelOutput or tuple
        Returns:
            ModelOutput or tuple containing loss, reconstruction, and mask
        """
        latent, mask, ids_restore = self.forward_encoder(optical, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, C, L, p*p]
        loss = self.forward_loss(optical, pred, mask)
        
        if not return_dict:
            return (loss, pred, mask)
        
        return dict(
            loss=loss,
            reconstruction=pred,
            mask=mask,
            hidden_states=(latent,)
        )

    def get_input_embeddings(self):
        """Get input embeddings (patch embeddings)."""
        return self.patch_embed

    def set_input_embeddings(self, value):
        """Set input embeddings (patch embeddings)."""
        self.patch_embed = value
