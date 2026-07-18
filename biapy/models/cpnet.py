"""
Cellpose ``CPnet`` architecture made available inside BiaPy.

The network definition (``batchconv``, ``batchconv0``, ``resdown``, ``downsample``,
``batchconvstyle``, ``resup``, ``make_style``, ``upsample`` and ``CPnet``) is copied
**verbatim** from Cellpose 3.x (``cellpose/resnet_torch.py``) so the model is
numerically identical to Cellpose's "classic" res-U-Net:

    Copyright © 2023 Howard Hughes Medical Institute,
    Authored by Carsen Stringer and Marius Pachitariu.
    https://github.com/MouseLand/cellpose  (BSD-3-Clause)

Only Cellpose's file-I/O helpers (``CPnet.save_model`` / ``CPnet.load_model``) are
dropped, since BiaPy handles checkpointing itself. The single BiaPy-specific addition
is the :class:`CPnet_BiaPy` wrapper at the bottom, which maps BiaPy's model-building
arguments (``image_shape``, ``feature_maps``, ``output_channels``, ``k_size`` ...) onto
``CPnet``'s constructor and returns a single ``(B, nout, ...)`` logits tensor, as the
rest of BiaPy expects.

This is a *fixed* architecture (4 resolution levels, residual + style + additive skips,
BatchNorm pre-activation blocks, nearest-neighbour upsampling). Most BiaPy MODEL options
(``NORMALIZATION``, ``ACTIVATION``, ``UPSAMPLE_LAYER``, ``CONV_LAYERS``, ``Z_DOWN`` /
``YX_DOWN``, ``CONV_BLOCK_ORDER`` ...) do not apply and are ignored. To reproduce
Cellpose exactly set ``MODEL.FEATURE_MAPS: [32, 64, 128, 256]`` and ``MODEL.KERNEL_SIZE: 3``.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================================
# Verbatim Cellpose CPnet (cellpose/resnet_torch.py) -----------------------------------------
# ============================================================================================


def batchconv(in_channels, out_channels, sz, conv_3D=False):
    conv_layer = nn.Conv3d if conv_3D else nn.Conv2d
    batch_norm = nn.BatchNorm3d if conv_3D else nn.BatchNorm2d
    return nn.Sequential(
        batch_norm(in_channels, eps=1e-5, momentum=0.05),
        nn.ReLU(inplace=True),
        conv_layer(in_channels, out_channels, sz, padding=sz // 2),
    )


def batchconv0(in_channels, out_channels, sz, conv_3D=False):
    conv_layer = nn.Conv3d if conv_3D else nn.Conv2d
    batch_norm = nn.BatchNorm3d if conv_3D else nn.BatchNorm2d
    return nn.Sequential(
        batch_norm(in_channels, eps=1e-5, momentum=0.05),
        conv_layer(in_channels, out_channels, sz, padding=sz // 2),
    )


class resdown(nn.Module):

    def __init__(self, in_channels, out_channels, sz, conv_3D=False):
        super().__init__()
        self.conv = nn.Sequential()
        self.proj = batchconv0(in_channels, out_channels, 1, conv_3D)
        for t in range(4):
            if t == 0:
                self.conv.add_module("conv_%d" % t,
                                     batchconv(in_channels, out_channels, sz, conv_3D))
            else:
                self.conv.add_module("conv_%d" % t,
                                     batchconv(out_channels, out_channels, sz, conv_3D))

    def forward(self, x):
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        x = x + self.conv[3](self.conv[2](x))
        return x


class downsample(nn.Module):

    def __init__(self, nbase, sz, conv_3D=False, max_pool=True):
        super().__init__()
        self.down = nn.Sequential()
        if max_pool:
            self.maxpool = nn.MaxPool3d(2, stride=2) if conv_3D else nn.MaxPool2d(
                2, stride=2)
        else:
            self.maxpool = nn.AvgPool3d(2, stride=2) if conv_3D else nn.AvgPool2d(
                2, stride=2)
        for n in range(len(nbase) - 1):
            self.down.add_module("res_down_%d" % n,
                                 resdown(nbase[n], nbase[n + 1], sz, conv_3D))

    def forward(self, x):
        xd = []
        for n in range(len(self.down)):
            if n > 0:
                y = self.maxpool(xd[n - 1])
            else:
                y = x
            xd.append(self.down[n](y))
        return xd


class batchconvstyle(nn.Module):

    def __init__(self, in_channels, out_channels, style_channels, sz, conv_3D=False):
        super().__init__()
        self.concatenation = False
        self.conv = batchconv(in_channels, out_channels, sz, conv_3D)
        self.full = nn.Linear(style_channels, out_channels)

    def forward(self, style, x, mkldnn=False, y=None):
        if y is not None:
            x = x + y
        feat = self.full(style)
        for k in range(len(x.shape[2:])):
            feat = feat.unsqueeze(-1)
        if mkldnn:
            x = x.to_dense()
            y = (x + feat).to_mkldnn()
        else:
            y = x + feat
        y = self.conv(y)
        return y


class resup(nn.Module):

    def __init__(self, in_channels, out_channels, style_channels, sz, conv_3D=False):
        super().__init__()
        self.concatenation = False
        self.conv = nn.Sequential()
        self.conv.add_module("conv_0",
                             batchconv(in_channels, out_channels, sz, conv_3D=conv_3D))
        self.conv.add_module(
            "conv_1",
            batchconvstyle(out_channels, out_channels, style_channels, sz,
                           conv_3D=conv_3D))
        self.conv.add_module(
            "conv_2",
            batchconvstyle(out_channels, out_channels, style_channels, sz,
                           conv_3D=conv_3D))
        self.conv.add_module(
            "conv_3",
            batchconvstyle(out_channels, out_channels, style_channels, sz,
                           conv_3D=conv_3D))
        self.proj = batchconv0(in_channels, out_channels, 1, conv_3D=conv_3D)

    def forward(self, x, y, style, mkldnn=False):
        x = self.proj(x) + self.conv[1](style, self.conv[0](x), y=y, mkldnn=mkldnn)
        x = x + self.conv[3](style, self.conv[2](style, x, mkldnn=mkldnn),
                             mkldnn=mkldnn)
        return x


class make_style(nn.Module):

    def __init__(self, conv_3D=False):
        super().__init__()
        self.flatten = nn.Flatten()
        self.avg_pool = F.avg_pool3d if conv_3D else F.avg_pool2d

    def forward(self, x0):
        style = self.avg_pool(x0, kernel_size=x0.shape[2:])
        style = self.flatten(style)
        style = style / torch.sum(style**2, axis=1, keepdim=True)**.5
        return style


class upsample(nn.Module):

    def __init__(self, nbase, sz, conv_3D=False):
        super().__init__()
        self.upsampling = nn.Upsample(scale_factor=2, mode="nearest")
        self.up = nn.Sequential()
        for n in range(1, len(nbase)):
            self.up.add_module("res_up_%d" % (n - 1),
                               resup(nbase[n], nbase[n - 1], nbase[-1], sz, conv_3D))

    def forward(self, style, xd, mkldnn=False):
        x = self.up[-1](xd[-1], xd[-1], style, mkldnn=mkldnn)
        for n in range(len(self.up) - 2, -1, -1):
            if mkldnn:
                x = self.upsampling(x.to_dense()).to_mkldnn()
            else:
                x = self.upsampling(x)
            x = self.up[n](x, xd[n], style, mkldnn=mkldnn)
        return x


class CPnet(nn.Module):
    """
    CPnet is the Cellpose neural network model used for cell segmentation and image restoration.

    Args:
        nbase (list): List of integers representing the number of channels in each layer of the downsample path.
        nout (int): Number of output channels.
        sz (int): Size of the convolution kernels.
        mkldnn (bool, optional): Whether to use MKL-DNN acceleration. Defaults to False.
        conv_3D (bool, optional): Whether to use 3D convolution. Defaults to False.
        max_pool (bool, optional): Whether to use max pooling. Defaults to True.
        diam_mean (float, optional): Mean diameter of the cells. Defaults to 30.0.
    """

    def __init__(self, nbase, nout, sz, mkldnn=False, conv_3D=False, max_pool=True,
                 diam_mean=30.):
        super().__init__()
        self.nchan = nbase[0]
        self.nbase = nbase
        self.nout = nout
        self.sz = sz
        self.residual_on = True
        self.style_on = True
        self.concatenation = False
        self.conv_3D = conv_3D
        self.mkldnn = mkldnn if mkldnn is not None else False
        self.downsample = downsample(nbase, sz, conv_3D=conv_3D, max_pool=max_pool)
        nbaseup = nbase[1:]
        nbaseup.append(nbaseup[-1])
        self.upsample = upsample(nbaseup, sz, conv_3D=conv_3D)
        self.make_style = make_style(conv_3D=conv_3D)
        self.output = batchconv(nbaseup[0], nout, 1, conv_3D=conv_3D)
        self.diam_mean = nn.Parameter(data=torch.ones(1) * diam_mean,
                                      requires_grad=False)
        self.diam_labels = nn.Parameter(data=torch.ones(1) * diam_mean,
                                        requires_grad=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, data):
        if self.mkldnn:
            data = data.to_mkldnn()
        T0 = self.downsample(data)
        if self.mkldnn:
            style = self.make_style(T0[-1].to_dense())
        else:
            style = self.make_style(T0[-1])
        style0 = style
        if not self.style_on:
            style = style * 0
        T1 = self.upsample(style, T0, self.mkldnn)
        T1 = self.output(T1)
        if self.mkldnn:
            T0 = [t0.to_dense() for t0 in T0]
            T1 = T1.to_dense()
        return T1, style0, T0


# ============================================================================================
# BiaPy wrapper ------------------------------------------------------------------------------
# ============================================================================================


class CPnet_BiaPy(nn.Module):
    """
    BiaPy-facing wrapper around Cellpose's :class:`CPnet`.

    It adapts BiaPy's model-building arguments to ``CPnet``'s constructor and exposes a
    standard BiaPy forward that returns the segmentation logits as a single
    ``(B, nout, Y, X)`` (2D) or ``(B, nout, Z, Y, X)`` (3D) tensor.

    Parameters
    ----------
    image_shape : tuple, optional
        Input patch shape as ``(Y, X, C)`` (2D) or ``(Z, Y, X, C)`` (3D). The number of
        input channels fed to the network is ``C`` (Cellpose normally uses 2; here it
        follows whatever BiaPy provides).
    feature_maps : list of int, optional
        Channels per encoder level. Cellpose uses ``[32, 64, 128, 256]`` (4 levels); set
        ``MODEL.FEATURE_MAPS`` to this to match Cellpose exactly.
    output_channels : list of int, optional
        Output channels per head. ``CPnet`` has a single output head, so the total number
        of output channels is ``sum(output_channels)`` (e.g. ``[3]`` for Cellpose flows
        ``F, Gh, Gv``).
    k_size : int, optional
        Convolution kernel size (Cellpose uses 3).
    diam_mean : float, optional
        Mean object diameter stored on the network (``PROBLEM.INSTANCE_SEG.CELLPOSE.DIAM_MEAN``).
        Only bookkeeping in BiaPy (rescaling is done by BiaPy's data pipeline), kept for fidelity.
    max_pool : bool, optional
        Use max pooling (Cellpose default) instead of average pooling for downsampling.
    contrast : bool, optional
        Not supported by CPnet; raises if enabled.
    """

    def __init__(
        self,
        image_shape=(256, 256, 1),
        feature_maps=[32, 64, 128, 256],
        output_channels=[3],
        k_size=3,
        diam_mean=30.0,
        max_pool=True,
        output_channel_info=None,
        contrast=False,
        return_one_tensor=False,
        **kwargs,  # swallow BiaPy MODEL options that CPnet's fixed architecture does not use
    ):
        super().__init__()
        if contrast:
            raise NotImplementedError(
                "The CPnet (Cellpose) model does not implement the contrastive-learning head; "
                "set LOSS.CONTRAST.ENABLE=False to use it."
            )
        ndim = len(image_shape) - 1
        if ndim not in (2, 3):
            raise ValueError(f"'image_shape' must be 2D (Y,X,C) or 3D (Z,Y,X,C), got {image_shape}")

        if isinstance(output_channels, (list, tuple)):
            nout = int(sum(output_channels))
        else:
            nout = int(output_channels)
        if nout < 1:
            raise ValueError(f"CPnet needs at least one output channel, got output_channels={output_channels}")

        nchan = int(image_shape[-1])
        nbase = [nchan] + [int(f) for f in feature_maps]

        self.ndim = ndim
        self.nout = nout
        self.return_one_tensor = return_one_tensor
        # The encoder applies (len(feature_maps) - 1) 2x poolings, so each spatial dim must be
        # divisible by 2**(len(feature_maps) - 1) (e.g. 8 for [32,64,128,256]; 224 -> 28 is fine).
        # BiaPy resizes the output back if it ever mismatches.
        self.cpnet = CPnet(
            nbase, nout, int(k_size), mkldnn=False, conv_3D=(ndim == 3),
            max_pool=max_pool, diam_mean=float(diam_mean),
        )

    def forward(self, x):
        # CPnet returns (segmentation_logits, style, encoder_feats); BiaPy consumes the logits.
        return self.cpnet(x)[0]
