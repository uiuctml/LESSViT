import torch.nn as nn


class SpectralAdapter(nn.Sequential):
    """Processes hyperspectral data by reducing spectral dimensionality.
    
    Uses 1D convolutions along the spectral dimension to extract features
    while preserving spatial dimensions. Converts hyperspectral input
    to 128 feature channels for standard 2D models.
    """
    def __init__(self):
        """Three 1D conv blocks (conv->batchnorm->relu) followed by adaptive pooling."""
        super(SpectralAdapter, self).__init__(
            nn.Conv3d(
                1, 32, kernel_size=(7, 1, 1), stride=(5, 1, 1), padding=(1, 0, 0)
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.Conv3d(
                32, 64, kernel_size=(7, 1, 1), stride=(5, 1, 1), padding=(1, 0, 0)
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            
            nn.Conv3d(
                64, 128, kernel_size=(5, 1, 1), stride=(3, 1, 1), padding=(1, 0, 0)
            ),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool3d((1, None, None))
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, depth, height, width]
        Returns:
            Output tensor [batch_size, 128, height, width]
        """
        x = x.unsqueeze(1)  # Add channel dimension
        x = super(SpectralAdapter, self).forward(x)
        x = x.squeeze(2)  # Remove the depth dimension
        return x


class RGBSpectralAdapter(nn.Sequential):
    """Adapter for 3-channel inputs that preserves the SpecViT 128-channel interface.

    This version uses standard 2D convolutions instead of spectral-depth Conv3d layers,
    so it can process RGB-like inputs of shape [B, 3, H, W].
    """

    def __init__(self):
        super(RGBSpectralAdapter, self).__init__(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, 3, height, width]
        Returns:
            Output tensor [batch_size, 128, height, width]
        """
        if x.shape[1] != 3:
            raise ValueError(f"RGBSpectralAdapter expects 3 input channels, but got {x.shape[1]}")
        return super(RGBSpectralAdapter, self).forward(x)
