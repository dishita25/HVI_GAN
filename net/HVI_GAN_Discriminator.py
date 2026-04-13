"""
Two 70x70 PatchGAN discriminators - one for the RGB colour space and one for
the HVI colour space.  They share the same network structure but operate on
different input domains.

Architecture (classic 70x70 PatchGAN, Isola et al. 2017):
    Conv(k4, s2) → LeakyReLU
    Conv(k4, s2) + InstanceNorm → LeakyReLU
    Conv(k4, s2) + InstanceNorm → LeakyReLU
    Conv(k4, s1) + InstanceNorm → LeakyReLU
    Conv(k4, s1) → 1-channel patch output (no sigmoid - used with LSGAN)

The design is deliberately identical for both domains so that the training
pipeline can treat them symmetrically.
"""

import torch
import torch.nn as nn


def _disc_block(in_ch: int, out_ch: int, stride: int = 2, use_norm: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=not use_norm),
    ]
    if use_norm:
        layers.append(nn.InstanceNorm2d(out_ch, affine=True))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


# Shared PatchGAN backbone
class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN discriminator.

    Args:
        in_channels (int): Number of input channels (3 for RGB or HVI).
        base_channels (int): Feature channels in the first conv layer.
                             Subsequent layers double until `max_channels`.
        max_channels (int): Channel cap.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64, max_channels: int = 512):
        super().__init__()

        ch = base_channels

        # Layer 1 – no instance-norm on the first layer (common practice)
        self.layer1 = _disc_block(in_channels, ch, stride=2, use_norm=False)

        # Layers 2-4 – stride-2 with norm
        self.layer2 = _disc_block(ch, min(ch * 2, max_channels), stride=2)
        ch = min(ch * 2, max_channels)

        self.layer3 = _disc_block(ch, min(ch * 2, max_channels), stride=2)
        ch = min(ch * 2, max_channels)

        self.layer4 = _disc_block(ch, min(ch * 2, max_channels), stride=1)
        ch = min(ch * 2, max_channels)

        # Final conv → 1-channel patch score (no activation; used with LSGAN)
        self.out = nn.Conv2d(ch, 1, kernel_size=4, stride=1, padding=1, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor (B, C, H, W).
        Returns:
            Patch-level scores (B, 1, H', W').  No sigmoid is applied;
            callers should use MSE-based LSGAN objectives.
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.out(x)


class RGBDiscriminator(PatchGANDiscriminator):
    """Discriminates between real and generated images in RGB space."""

    def __init__(self, base_channels: int = 64, max_channels: int = 512):
        super().__init__(
            in_channels=3,
            base_channels=base_channels,
            max_channels=max_channels,
        )


class HVIDiscriminator(PatchGANDiscriminator):
    """Discriminates between real and generated images in HVI space."""

    def __init__(self, base_channels: int = 64, max_channels: int = 512):
        super().__init__(
            in_channels=3,
            base_channels=base_channels,
            max_channels=max_channels,
        )
