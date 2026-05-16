from GeospatialFM.models.spatial_spectral_low_rank_vit import SpatialSpectralLowRankViTEncoder, SpatialSpectralLowRankViTConfig
from GeospatialFM.models.wrappers.specvit_wrapper import SpecViTEncoder, SpecViTConfig
from GeospatialFM.models.wrappers.dinov3_wrapper import DINOv3Encoder, DINOv3Config
from GeospatialFM.models.wrappers.dofa_wrapper import DOFAEncoder, DOFAConfig
from GeospatialFM.models.wrappers.spatsigma_wrapper import SpatSigmaClsEncoder, SpatSigmaSegEncoder, SpatSigmaConfig
from GeospatialFM.models.wrappers.channelvit_wrapper import ChannelViTEncoder, ChannelViTConfig

ENCODER_CONFIGS = {
    "lessvit": SpatialSpectralLowRankViTConfig,
    "specvit": SpecViTConfig,
    "dinov3": DINOv3Config,
    "dofa": DOFAConfig,
    "spatsigma": SpatSigmaConfig,
    "channelvit": ChannelViTConfig,
}

ENCODER_MODELS = {
    "lessvit": SpatialSpectralLowRankViTEncoder,
    "specvit": SpecViTEncoder,
    "dinov3": DINOv3Encoder,
    "dofa": DOFAEncoder,
    "spatsigma_cls": SpatSigmaClsEncoder,
    "spatsigma_seg": SpatSigmaSegEncoder,
    "channelvit": ChannelViTEncoder,
}