import math
import torch
import torch.nn as nn
import torch.nn.init as init


class wdsr(nn.Module):
    """
    WDSR model.

    Reference: `Wide Activation for Efficient and Accurate Image Super-Resolution <https://arxiv.org/abs/1808.08718>`_.

    Adapted from `here <https://github.com/yjn870/WDSR-pytorch/tree/master>`_.
    """

    def __init__(
        self,
        scale,
        num_filters=32,
        num_res_blocks=16,
        res_block_expansion=6,
        num_channels=1,
    ):
        super(wdsr, self).__init__()
        if type(scale) is tuple:
            scale = scale[0]
        kernel_size = 3
        skip_kernel_size = 5
        weight_norm = torch.nn.utils.weight_norm
        num_outputs = scale * scale * num_channels

        body = []
        conv = weight_norm(nn.Conv2d(num_channels, num_filters, kernel_size, padding=kernel_size // 2))
        init.ones_(conv.weight_g)
        init.zeros_(conv.bias)
        body.append(conv)
        for _ in range(num_res_blocks):
            body.append(
                Block(
                    num_filters,
                    kernel_size,
                    res_block_expansion,
                    weight_norm=weight_norm,
                    res_scale=1 / math.sqrt(num_res_blocks),
                )
            )
        conv = weight_norm(nn.Conv2d(num_filters, num_outputs, kernel_size, padding=kernel_size // 2))
        init.ones_(conv.weight_g)
        init.zeros_(conv.bias)
        body.append(conv)
        self.body = nn.Sequential(*body)

        skip = []
        if num_channels != num_outputs:
            conv = weight_norm(
                nn.Conv2d(
                    num_channels,
                    num_outputs,
                    skip_kernel_size,
                    padding=skip_kernel_size // 2,
                )
            )
            init.ones_(conv.weight_g)
            init.zeros_(conv.bias)
            skip.append(conv)
        self.skip = nn.Sequential(*skip)

        shuf = []
        if scale > 1:
            shuf.append(nn.PixelShuffle(scale))
        self.shuf = nn.Sequential(*shuf)

    def forward(self, x):
        x = self.body(x) + self.skip(x)
        x = self.shuf(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        num_residual_units,
        kernel_size,
        width_multiplier=1,
        weight_norm=torch.nn.utils.weight_norm,
        res_scale=1,
    ):
        super(Block, self).__init__()
        body = []
        conv = weight_norm(
            nn.Conv2d(
                num_residual_units,
                int(num_residual_units * width_multiplier),
                kernel_size,
                padding=kernel_size // 2,
            )
        )
        init.constant_(conv.weight_g, 2.0)
        init.zeros_(conv.bias)
        body.append(conv)
        body.append(nn.ReLU(True))
        conv = weight_norm(
            nn.Conv2d(
                int(num_residual_units * width_multiplier),
                num_residual_units,
                kernel_size,
                padding=kernel_size // 2,
            )
        )
        init.constant_(conv.weight_g, res_scale)
        init.zeros_(conv.bias)
        body.append(conv)

        self.body = nn.Sequential(*body)

    def forward(self, x):
        x = self.body(x) + x
        return x
