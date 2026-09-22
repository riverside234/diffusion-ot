"""Small RGB PatchGAN with a 70x70 receptive field (Isola et al., CVPR 2017).

Architecture reference: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
Three stride-2 convolutions and two stride-1 convolutions, all 4x4. Here we
use spectral normalization without instance/batch normalization, retaining
absolute pixel appearance cues. This is an unconditional target-domain critic.
"""
import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


class RGBPatchDiscriminator(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        if isinstance(base_channels, bool) or not isinstance(base_channels, int) or base_channels < 1:
            raise ValueError("PatchGAN base_channels must be a positive integer.")
        channels = [3, base_channels, 2 * base_channels, 4 * base_channels, 8 * base_channels, 1]
        layers = []
        for i, stride in enumerate((2, 2, 2, 1, 1)):
            layers.append(spectral_norm(nn.Conv2d(channels[i], channels[i + 1], 4, stride, padding=1)))
            if i < 4:
                layers.append(nn.LeakyReLU(.2))
        self.layers = nn.Sequential(*layers)

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[-2:]) < 24:
            raise ValueError("RGB PatchGAN requires [batch, 3, height, width] with height/width >= 24.")
        # Both real and fake RGB are in [0, 1]; keep the same fixed scaling.
        with torch.autocast(device_type=images.device.type, enabled=False):
            return self.layers(images.float() * 2 - 1)
