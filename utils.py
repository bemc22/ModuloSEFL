import torch
import torch.nn.functional as F

import torch.nn as nn

def grad_y(y):
    Mdx_y = F.pad(torch.diff(y, 1, dim=-1), (1, 0))
    Mdy_y = F.pad(torch.diff(y, 1, dim=-2), (0, 0, 1, 0))
    return Mdx_y, Mdy_y


class ToneMapping(nn.Module):
    def __init__(self, exposure=1.0, factor=0.2,):
        super().__init__()

        self.register_buffer("exposure", torch.tensor(exposure))
        self.register_buffer("factor", torch.tensor(factor))

    def forward(self, x):
        # Apply exposure adjustment
        x = x * self.exposure

        # Calculate luminance (ITU-R BT.709 coefficients)
        luminance = (
              0.2126 * x[:, 0:1, :, :]
            + 0.7152 * x[:, 1:2, :, :]
            + 0.0722 * x[:, 2:3, :, :]
        )

        # Apply tone-mapping curve to luminance
        tonemapped_lum = luminance / (luminance + self.factor)

        # Calculate scaling factor while avoiding division by zero
        scale = tonemapped_lum / (luminance + 1e-6)

        # Apply scale to each color channel
        return x * scale