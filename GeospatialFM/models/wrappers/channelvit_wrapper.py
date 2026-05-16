from GeospatialFM.models.ChannelViT import *
from transformers import PretrainedConfig, PreTrainedModel
import glob
import os
from loguru import logger
import torch

class ChannelViTConfig(PretrainedConfig):
    model_type = "channelvit"

    def __init__(self, 
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 img_size=[128],
                 in_chans=202,
                 **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.img_size = img_size
        self.in_chans = in_chans

class ChannelViTEncoder(ChannelVisionTransformer):
    config_class = ChannelViTConfig
    model_type = "channelvit"
    def __init__(self, config: ChannelViTConfig):
        super().__init__(
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            img_size=config.img_size,
            in_chans=config.in_chans,
        )
        
    def forward_encoder(self, x, **kwargs):
        channels = kwargs.get("channels")
        if channels is None:
            channels = list(range(x.shape[1]))
        extra_tokens = {"channels": [channels]}
        features = super().get_intermediate_layers(x, extra_tokens=extra_tokens, n=1)[-1]
        cls_token, patch_tokens = features[:, :1], features[:, 1:]
        batch_size, num_tokens, embed_dim = patch_tokens.shape
        num_channels = x.shape[1]
        if num_tokens % num_channels != 0:
            raise ValueError(
                f"ChannelViT produced {num_tokens} patch tokens for {num_channels} channels; "
                "expected the token count to be divisible by the channel count."
            )
        patch_tokens = patch_tokens.reshape(
            batch_size,
            num_channels,
            num_tokens // num_channels,
            embed_dim,
        )
        pooled_patch_tokens = patch_tokens.mean(dim=1)
        # print(pooled_patch_tokens.shape)
        # assert False
        return torch.cat([cls_token, pooled_patch_tokens], dim=1)
    
    def load_pretrained_weights(self, pretrained_model_dir):
        print("not loading anything")
        pass
