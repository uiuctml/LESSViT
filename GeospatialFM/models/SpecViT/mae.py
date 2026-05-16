from functools import partial
from typing import Any, Dict, Optional, Union
import logging

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from transformers import PreTrainedModel, PretrainedConfig

from .specvit import SpecVisionTransformer


logger = logging.getLogger(__name__)

SELECTED_CHANNEL_IDX = [0, 3, 5, 8, 10, 13, 15, 18, 21, 23, 26, 28, 31, 33, 36, 38, 41, 44, 46, 49, 51, 54, 56, 59, 62, 64, 67, 69, 72, 74, 77, 79, 82, 85, 87, 90, 92, 95, 97, 100, 101, 102, 104, 105, 106, 107, 109, 110, 111, 112, 114, 115, 116, 117, 119, 120, 121, 123, 124, 125, 126, 128, 129, 130, 131, 133, 134, 135, 136, 138, 139, 140, 142, 143, 144, 145, 147, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 162, 163, 164, 166, 167, 168, 169, 171, 172, 173, 174, 176, 177, 178, 179, 181, 182, 183, 185, 186, 187, 188, 190, 191, 192, 193, 195, 196, 197, 198, 200, 201]

class SpecViTMAEConfig(PretrainedConfig):
    model_type = "specvit_mae"

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        patch_size: int = 16,
        input_size: int = 128,
        reduced_channels: int = 128,
        mask_ratio: float = 0.75,
        channel_mask_ratio: float = 0.5,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        init_values: Optional[float] = None,
        norm_pix_loss: bool = True,
        in_channels: int = len(SELECTED_CHANNEL_IDX),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.input_size = input_size
        self.reduced_channels = reduced_channels
        self.mask_ratio = mask_ratio
        self.channel_mask_ratio = channel_mask_ratio
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.drop_path_rate = drop_path_rate
        self.drop_path_uniform = drop_path_uniform
        self.init_values = init_values
        self.norm_pix_loss = norm_pix_loss
        self.in_channels = in_channels


class SpecViTMAEDecoder(nn.Module):
    def __init__(self, config: SpecViTMAEConfig, num_patches: int):
        super().__init__()
        self.config = config
        self.num_patches = num_patches
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.decoder_embed = nn.Linear(config.embed_dim, config.decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.decoder_embed_dim)
        )

        if config.drop_path_uniform:
            dpr = [config.drop_path_rate] * config.decoder_depth
        else:
            dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.decoder_depth)]

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=config.decoder_embed_dim,
                    num_heads=config.decoder_num_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias,
                    qk_norm=config.qk_norm,
                    proj_drop=config.proj_drop,
                    attn_drop=config.attn_drop,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(config.decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(config.decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            config.decoder_embed_dim,
            config.patch_size**2 * config.in_channels,
            bias=True,
        )

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.mask_token, std=0.02)
        torch.nn.init.normal_(self.decoder_pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor, ids_restore: torch.Tensor):
        x = self.decoder_embed(x)

        B, L_keep_plus_cls, D = x.shape
        num_keep = L_keep_plus_cls - 1

        mask_tokens = self.mask_token.expand(B, self.num_patches - num_keep, -1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed
        hidden_states = []
        for block in self.decoder_blocks:
            x = block(x)
            hidden_states.append(x.detach().cpu())
        x = self.decoder_norm(x)
        x = self.decoder_pred(x[:, 1:, :])

        B, HW, _ = x.shape
        x = x.reshape(B, HW, self.config.in_channels, self.config.patch_size**2)
        x = x.permute(0, 2, 1, 3)
        return x, hidden_states

    def forward_target(self, imgs: torch.Tensor):
        target = self.patchify(imgs)
        if self.config.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6).sqrt()
        return target

    def patchify(self, imgs: torch.Tensor):
        p = self.config.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, p))
        x = torch.einsum("bchpwq->bchwpq", x)
        x = x.reshape(shape=(imgs.shape[0], imgs.shape[1], h * w, p**2))
        return x


class SpecViTMAE(PreTrainedModel):
    config_class = SpecViTMAEConfig
    main_input_name = "optical"
    supports_gradient_checkpointing = True

    def __init__(self, config: SpecViTMAEConfig):
        super().__init__(config)
        self.config = config
        self.encoder = SpecVisionTransformer(
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            token_patch_size=config.patch_size,
            patch_size=config.input_size,
            reduced_channels=config.reduced_channels,
            num_classes=0,
        )

        self.patch_embed = self.encoder.vit_core.patch_embed
        self.cls_token = self.encoder.vit_core.cls_token
        self.pos_embed = self.encoder.vit_core.pos_embed
        self.pos_drop = self.encoder.vit_core.pos_drop
        self.patch_drop = self.encoder.vit_core.patch_drop
        self.norm_pre = self.encoder.vit_core.norm_pre
        self.blocks = self.encoder.vit_core.blocks
        self.norm = self.encoder.vit_core.norm
        self.num_patches = self.patch_embed.num_patches

        self.decoder = SpecViTMAEDecoder(config, num_patches=self.num_patches)
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "set_grad_checkpointing"):
            module.set_grad_checkpointing(enable=value)
        elif hasattr(module, "grad_checkpointing"):
            module.grad_checkpointing = value

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def random_masking(self, x: torch.Tensor, mask_ratio: float):
        B, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def random_channel_masking(self, imgs: torch.Tensor, channel_mask_ratio: float):
        B, C, _, _ = imgs.shape
        if channel_mask_ratio <= 0:
            channel_mask = torch.zeros(B, C, device=imgs.device, dtype=imgs.dtype)
            return imgs, channel_mask

        len_keep = max(1, int(C * (1 - channel_mask_ratio)))
        noise = torch.rand(B, C, device=imgs.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        keep_mask = torch.zeros(B, C, device=imgs.device, dtype=imgs.dtype)
        keep_mask.scatter_(1, ids_keep, 1.0)
        masked_imgs = imgs * keep_mask.unsqueeze(-1).unsqueeze(-1)
        channel_mask = 1.0 - keep_mask
        return masked_imgs, channel_mask

    def forward_encoder(self, optical: torch.Tensor, mask_ratio: float, channel_mask_ratio: float):
        optical, channel_mask = self.random_channel_masking(optical, channel_mask_ratio)
        x = self.encoder.spectral_adapter(optical)
        x = self.patch_embed(x)
        x = self.encoder.vit_core._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        x, pos_mask, ids_restore = self.random_masking(x[:, 1:, :], mask_ratio)
        cls_token = x.new_zeros(optical.shape[0], 1, self.config.embed_dim)
        cls_token[:] = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat([cls_token.expand(optical.shape[0], -1, -1), x], dim=1)
        x = self.pos_drop(x)

        hidden_states = []
        for block in self.blocks:
            x = block(x)
            hidden_states.append(x.detach().cpu())
        x = self.norm(x)
        return x, channel_mask, pos_mask, ids_restore, hidden_states

    def forward(
        self,
        optical: torch.Tensor,
        radar: Optional[torch.Tensor] = None,
        optical_channel_wv: Optional[torch.Tensor] = None,
        radar_channel_wv: Optional[torch.Tensor] = None,
        mask_ratio: Optional[float] = None,
        channel_mask_ratio: Optional[float] = None,
        spatial_resolution: Optional[Union[int, float]] = None,
        modal: Optional[str] = None,
    ):
        del radar, optical_channel_wv, radar_channel_wv, spatial_resolution

        if modal not in (None, "optical"):
            raise ValueError(f"SpecViT MAE only supports optical pretraining, but got modal={modal}")

        if optical.shape[1] != self.config.in_channels:
            raise ValueError(
                f"Expected {self.config.in_channels} optical channels, but got {optical.shape[1]}"
            )

        mask_ratio = self.config.mask_ratio if mask_ratio is None else mask_ratio
        channel_mask_ratio = (
            self.config.channel_mask_ratio if channel_mask_ratio is None else channel_mask_ratio
        )

        target = self.decoder.forward_target(optical)
        latent, channel_mask, pos_mask, ids_restore, encoder_hidden_states = self.forward_encoder(
            optical=optical,
            mask_ratio=mask_ratio,
            channel_mask_ratio=channel_mask_ratio,
        )
        recon, decoder_hidden_states = self.decoder(latent, ids_restore)

        return {
            "target": target,
            "optical_recon": recon,
            "optical_channel_mask": channel_mask,
            "optical_pos_mask": pos_mask,
            "optical_hidden_states": encoder_hidden_states + decoder_hidden_states,
        }

    def estimate_tokens(self, input_dict: Dict[str, Union[torch.Tensor, Any]]) -> int:
        if "optical" in input_dict:
            return input_dict["optical"].numel()
        if not hasattr(self, "warnings_issued"):
            self.warnings_issued = {}
        if "estimate_tokens" not in self.warnings_issued:
            logger.warning(
                "Could not estimate the number of tokens of the input, floating-point operations will not be computed"
            )
            self.warnings_issued["estimate_tokens"] = True
        return 0
