from GeospatialFM.models.HyperFree.image_encoder import ImageEncoderViT
from transformers import PretrainedConfig, PreTrainedModel
import torch
import torch.nn.functional as F
import glob
import os
from loguru import logger


class HyperFreeConfig(PretrainedConfig):
    model_type = "hyperfree"

    def __init__(self,
                 img_size=128,
                 patch_size=16,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 # `out_chans` is the dimension of the neck's output (what forward_encoder
                 # actually returns), not `embed_dim` (the ViT body's internal width). It's
                 # set to 768 -- rather than the 256 used by the released SAM-style
                 # checkpoint's neck -- so it lines up with the shared `--embed_dim` CLI
                 # default that sizes the task head in downstream_models.py; get_encoder()
                 # never threads CLI args into baseline configs, so this can't be fixed up
                 # from the command line the way it can for lessvit. Pretrained backbone
                 # blocks still load fine; only the neck's own conv weights fall back to
                 # random init as a result. Pass --embed_dim 256 if you'd rather keep the
                 # pretrained neck instead.
                 out_chans=768,
                 qkv_bias=True,
                 use_rel_pos=True,
                 rel_pos_zero_init=True,
                 window_size=14,
                 global_attn_indexes=(2, 5, 8, 11),
                 **kwargs):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.out_chans = out_chans
        self.qkv_bias = qkv_bias
        self.use_rel_pos = use_rel_pos
        self.rel_pos_zero_init = rel_pos_zero_init
        self.window_size = window_size
        self.global_attn_indexes = global_attn_indexes


class HyperFreeEncoder(ImageEncoderViT):
    config_class = HyperFreeConfig
    model_type = "hyperfree"

    def __init__(self, config: HyperFreeConfig):
        grid_size = config.img_size // config.patch_size
        # HyperFree's released vit_b build hardcodes window_size=14, tuned for its own
        # 512x512-patch training data (a 32x32 token grid at patch_size=16). This benchmark's
        # crops are 128x128 (8x8 tokens), so window_partition would pad every windowed block
        # up to 14x14=196 slots with only 64 real tokens -- ~67% zero-padding that isn't
        # masked out of attention, corrupting every windowed block's output. That's not a
        # capacity disadvantage inherent to HyperFree, it's a padding artifact this benchmark
        # would never have exercised at its native resolution, so disable windowing (fall
        # back to full attention on every block) whenever the configured grid can't fill even
        # one real window -- this removes the artifact without changing what pixels the model
        # sees, unlike upsampling the input to chase HyperFree's native resolution would.
        effective_window_size = config.window_size if grid_size > config.window_size else 0
        super().__init__(
            img_size=config.img_size,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            out_chans=config.out_chans,
            qkv_bias=config.qkv_bias,
            use_rel_pos=config.use_rel_pos,
            rel_pos_zero_init=config.rel_pos_zero_init,
            window_size=effective_window_size,
            global_attn_indexes=config.global_attn_indexes,
            # Keep a single-scale feature map (no patch merging) so the output grid always
            # matches img_size / patch_size, like the other wrapped encoders (DOFA, DINOv3, ...).
            merge_indexs=None,
        )

    def forward_encoder(self, x, wave_list=None, spatial_resolution=None, **kwargs):
        # HyperFree's weight-bank patch embed expects wavelengths in nm; upstream callers
        # (GeospatialFM.models.downstream_models) hand us `wave_list` already converted to um.
        # downstream_models.py derives wave_list via `optical_channel_wv.squeeze(dim=0)`,
        # which only squeezes away the batch dim when batch_size == 1; for batch_size > 1
        # it stays [B, C] (every row identical -- all samples in a batch share one channel
        # config), so tolist() gives a list of per-sample lists instead of a flat list.
        if wave_list is not None and len(wave_list) > 0 and isinstance(wave_list[0], (list, tuple)):
            wave_list = wave_list[0]
        input_wavelength = None if wave_list is None else [w * 1000.0 for w in wave_list]
        gsd = 1.0 if spatial_resolution is None else float(spatial_resolution)
        # The scale-aware positional embedding reshapes to (batch_size, grid, grid, embed_dim),
        # so GSD must carry one resolution per batch element, not a single scalar.
        gsd_per_sample = [gsd] * x.shape[0]

        multi_stage_features = super().forward(x, test_mode=True, input_wavelength=input_wavelength, GSD=gsd_per_sample)
        # HyperFree has no CLS token (it's a SAM-style encoder); use the final (post-neck)
        # feature map as patch tokens and a mean-pooled token in its place, so the output
        # matches the [CLS, patches] convention the other encoder wrappers return.
        features = multi_stage_features[-1]  # [B, out_chans, H, W]
        batch_size, channels, height, width = features.shape
        patch_tokens = features.flatten(2).transpose(1, 2)  # [B, H*W, out_chans]
        cls_token = patch_tokens.mean(dim=1, keepdim=True)
        return torch.cat([cls_token, patch_tokens], dim=1)

    def _load_and_resize(self, state_dict):
        # Ported from HyperFree/build_HyperFree.py::load_and_resize_params so mismatched
        # pos_embed / rel_pos / weight_bank shapes (e.g. from a different img_size or
        # patch_size than the released checkpoint) get resized instead of failing to load.
        model_dict = self.state_dict()
        for key, value in state_dict.items():
            if key not in model_dict:
                continue
            if value.shape != model_dict[key].shape:
                if "pos_embed" in key:
                    value = F.interpolate(
                        value.permute((0, 3, 1, 2)),
                        size=(model_dict[key].shape[1], model_dict[key].shape[2]),
                        mode="nearest",
                    ).permute((0, 2, 3, 1))
                elif "rel_pos" in key:
                    value = F.interpolate(
                        value.unsqueeze(0).unsqueeze(0),
                        size=(model_dict[key].shape[0], model_dict[key].shape[1]),
                    ).squeeze(0).squeeze(0)
                elif "weight_bank" in key:
                    value = F.interpolate(
                        value,
                        size=(model_dict[key].shape[2], model_dict[key].shape[3]),
                        mode="nearest",
                    )
                else:
                    # Shape mismatch we don't know how to resize (e.g. embed_dim doesn't
                    # match between config and checkpoint); keep the current init instead.
                    continue
            model_dict[key] = value
        super().load_state_dict(model_dict, strict=False)

    def load_pretrained_weights(self, pretrained_model_dir):
        pretrained_model_paths = glob.glob(os.path.join(pretrained_model_dir, "*.pth"))
        pretrained_model_paths.sort()
        pretrained_model_path = pretrained_model_paths[-1]

        state_dict = torch.load(pretrained_model_path, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict):
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]

        # The released checkpoint is a full Sam model (image_encoder + prompt_encoder +
        # mask_decoder); keep only the backbone weights we actually instantiate.
        encoder_state_dict = {
            key[len("image_encoder."):]: value
            for key, value in state_dict.items()
            if key.startswith("image_encoder.")
        }
        if not encoder_state_dict:
            encoder_state_dict = state_dict

        self._load_and_resize(encoder_state_dict)
        # logger.info("Load pretrained HyperFree Encoder successfully!")
