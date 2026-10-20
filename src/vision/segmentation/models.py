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
    """Compact U-Net architecture used as a surrogate model for semantic segmentation taggants."""

    def __init__(self, in_channels: int, num_classes: int, up_channels: int = 32) -> None:
        """Initialize UNetSurrogate.

        Args:
            in_channels: Number of channels in input image (e.g., 3 for RGB).
            num_classes: Number of semantic segmentation classes.
            up_channels: Base channel dimension for encoder/decoder (default: 32).
        """
        super().__init__()
        self.down1 = DoubleConv(in_channels, up_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(up_channels, 2 * up_channels)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(2 * up_channels, 4 * up_channels)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(4 * up_channels, 8 * up_channels)

        self.up1 = nn.ConvTranspose2d(8 * up_channels, 4 * up_channels, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(8 * up_channels, 4 * up_channels)
        self.up2 = nn.ConvTranspose2d(4 * up_channels, 2 * up_channels, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(4 * up_channels, 2 * up_channels)
        self.up3 = nn.ConvTranspose2d(2 * up_channels, up_channels, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(2 * up_channels, up_channels)

        self.out = nn.Conv2d(up_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-Net encoder, skip connections, decoder, and classifier.

        Args:
            x: Input image tensor of shape (B, in_channels, H, W).

        Returns:
            Logits tensor of shape (B, num_classes, H, W).
        """
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        bottleneck = self.down4(self.pool3(d3))

        u1 = self.up1(bottleneck)
        u1 = torch.cat([u1, d3], dim=1)
        u1 = self.conv_up1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv_up2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, d1], dim=1)
        u3 = self.conv_up3(u3)

        return self.out(u3)
