"""Neural network models for semantic segmentation surrogate training."""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two successive Convolution-BatchNorm-ReLU blocks with 3x3 kernels."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize DoubleConv module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C_in, H, W).

        Returns:
            Output tensor of shape (B, C_out, H, W).
        """
        return self.conv(x)


class UNetSurrogate(nn.Module):
    """Compact U-Net used as a surrogate model for semantic segmentation taggants.

    ``depth`` is the number of 2x downsamplings. Each extra level doubles the receptive field
    (depth=3: 92 px, depth=4: 188 px, depth=5: 380 px) and quadruples the parameter count for a
    fixed ``up_channels``. Parameter names (``down1``, ``pool1``, ``up1``, ``conv_up1``, ...) are the
    same as the original fixed-depth model, so ``depth=3`` checkpoints load unchanged.
    """

    def __init__(self, in_channels: int, num_classes: int, up_channels: int = 32, depth: int = 3) -> None:
        """Initialize UNetSurrogate.

        Args:
            in_channels: Number of channels in input image (e.g., 3 for RGB).
            num_classes: Number of semantic segmentation classes.
            up_channels: Base channel dimension; level ``i`` has ``up_channels * 2**i`` channels.
            depth: Number of downsampling steps (default 3, the original architecture).
        """
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.depth = depth
        channels = [up_channels * 2 ** i for i in range(depth + 1)]

        # Encoder: down1 .. down{depth+1}, pool1 .. pool{depth}
        for i in range(depth + 1):
            setattr(self, f"down{i + 1}", DoubleConv(in_channels if i == 0 else channels[i - 1], channels[i]))
            if i < depth:
                setattr(self, f"pool{i + 1}", nn.MaxPool2d(2))

        # Decoder: up1 is the deepest upsampling (bottleneck -> level depth-1)
        for j in range(depth):
            level = depth - 1 - j
            setattr(self, f"up{j + 1}", nn.ConvTranspose2d(channels[level + 1], channels[level], kernel_size=2, stride=2))
            setattr(self, f"conv_up{j + 1}", DoubleConv(2 * channels[level], channels[level]))

        self.out = nn.Conv2d(up_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-Net encoder, skip connections, decoder, and classifier.

        Args:
            x: Input image tensor of shape (B, in_channels, H, W); H and W must be divisible by 2**depth.

        Returns:
            Logits tensor of shape (B, num_classes, H, W).
        """
        skips = []
        for i in range(self.depth + 1):
            x = getattr(self, f"down{i + 1}")(x)
            if i < self.depth:
                skips.append(x)
                x = getattr(self, f"pool{i + 1}")(x)

        for j in range(self.depth):
            x = getattr(self, f"up{j + 1}")(x)
            x = torch.cat([x, skips.pop()], dim=1)
            x = getattr(self, f"conv_up{j + 1}")(x)

        return self.out(x)
