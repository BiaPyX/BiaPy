
"""
This module contains various head architectures for neural networks,
including the Atrous Spatial Pyramid Pooling (ASPP) module and a Projection Head
for self-supervised learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from biapy.models.blocks import get_norm_2d, get_norm_3d

class ASPP(nn.Module):
    """
    Implements the Atrous Spatial Pyramid Pooling (ASPP) module.

    ASPP captures multi-scale contextual information by employing parallel atrous
    convolutions with different dilation rates. This allows the model to
    effectively enlarge the receptive field and capture context at various scales.

    References:
        DeepLabv3 work: `Rethinking Atrous Convolution for Semantic Image Segmentation <https://arxiv.org/abs/1706.05587>`_.
        DeepLabv3+ work: `Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation <https://arxiv.org/abs/1802.02611>`_.
        Code adapted from `here <https://github.com/rishikksh20/ResUnet/blob/master/core/modules.py>`_.

    Parameters
    ----------
    conv : Type[nn.Conv2d | nn.Conv3d]
        The convolutional layer type to use (e.g., `nn.Conv2d` for 2D, `nn.Conv3d` for 3D).
    in_dims : int
        Number of input channels.
    out_dims : int
        Number of output channels for each atrous convolution block and the final output.
    norm : str, optional
        Normalization layer type to use within each atrous convolution block.
        Options include `'bn'` (BatchNorm), `'sync_bn'` (SyncBatchNorm),
        `'in'` (InstanceNorm), `'gn'` (GroupNorm), or `'none'` (no normalization).
        Defaults to "none".
    rate : list of int, optional
        A list of integers specifying the dilation rates for the parallel atrous
        convolutions. Defaults to `[6, 12, 18]`.
    """

    def __init__(self, conv, in_dims, out_dims, norm="none", rate=[6, 12, 18]):
        """
        Initialize the Atrous Spatial Pyramid Pooling (ASPP) module.

        Sets up parallel atrous convolutional blocks with different dilation rates
        and a final 1x1 convolution for combining their outputs. Each atrous block
        includes a convolution, ReLU activation, and an optional normalization layer.

        Parameters
        ----------
        conv : Type[nn.Conv2d | nn.Conv3d]
            The convolutional layer type to use (e.g., `nn.Conv2d` for 2D, `nn.Conv3d` for 3D).
        in_dims : int
            Number of input channels to the ASPP module.
        out_dims : int
            Number of output channels for each individual atrous convolution block,
            and also the final output channels of the ASPP module.
        norm : str, optional
            Normalization layer type to use after the convolution and before ReLU
            in each atrous block. Options include `'bn'`, `'sync_bn'`, `'in'`,
            `'gn'`, or `'none'`. Defaults to "none".
        rate : list of int, optional
            A list of dilation rates to be used for the parallel atrous convolutions.
            The length of this list determines the number of parallel atrous blocks.
            Defaults to `[6, 12, 18]`.
        """
        super(ASPP, self).__init__()

        block = [
            conv(in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]),
            nn.ReLU(inplace=True),
        ]
        if norm != "none":
            if conv == nn.Conv2d:
                block.append(get_norm_2d(norm, out_dims))
            else:
                block.append(get_norm_3d(norm, out_dims))
        self.aspp_block1 = nn.Sequential(*block)
        block = [
            conv(in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]),
            nn.ReLU(inplace=True),
        ]
        if norm != "none":
            if conv == nn.Conv2d:
                block.append(get_norm_2d(norm, out_dims))
            else:
                block.append(get_norm_3d(norm, out_dims))
        self.aspp_block2 = nn.Sequential(*block)
        block = [
            conv(in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]),
            nn.ReLU(inplace=True),
        ]
        if norm != "none":
            if conv == nn.Conv2d:
                block.append(get_norm_2d(norm, out_dims))
            else:
                block.append(get_norm_3d(norm, out_dims))
        self.aspp_block3 = nn.Sequential(*block)

        self.output = conv(len(rate) * out_dims, out_dims, 1)

    def forward(self, x):
        """
        Perform the forward pass of the ASPP module.

        The input tensor `x` is processed by multiple parallel atrous convolutions
        (each with a different dilation rate), and their outputs are concatenated
        along the channel dimension. A final 1x1 convolution is then applied to
        reduce the channel count to `out_dims`.

        Parameters
        ----------
        x : torch.Tensor
            The input feature tensor. Its shape should be (batch_size, in_dims, D, H, W)
            for 3D data or (batch_size, in_dims, H, W) for 2D data.

        Returns
        -------
        torch.Tensor
            The output tensor after ASPP processing. Its shape will be
            (batch_size, out_dims, D, H, W) or (batch_size, out_dims, H, W),
            depending on the input dimensionality.
        """
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)


class ProjectionHead(nn.Module):
    """
    Implements a projection head for self-supervised learning, designed to project input features into a lower-dimensional space and normalize the output.

    This module can configure its projection layer to be either a simple linear
    layer or a convolutional MLP (Multi-Layer Perceptron) structure, and supports
    different normalization types.

    Parameters
    ----------
    ndim : int
        Number of dimensions of the input data. Supports 2 (for 2D data) or 3 (for 3D data).
    in_channels : int
        Number of input feature channels.
    proj_dim : int, optional
        The desired dimension of the projected output features. Defaults to 256.
    proj : str, optional
        Specifies the type of projection layer to use.

        - 'linear': Uses a single 1x1 convolutional layer (equivalent to a linear projection).
        - 'convmlp': Employs a convolutional MLP structure, consisting of a 1x1 convolution, batch normalization, ReLU activation, and another 1x1 convolution.
        
        Defaults to 'convmlp'.
    bn_type : str, optional
        Defines the type of batch normalization to apply within the 'convmlp' projection.

        - 'sync_bn': Synchronized Batch Normalization.
        - 'none': No batch normalization is applied.
        
        Defaults to 'sync_bn'.
    """

    def __init__(self, ndim, in_channels, proj_dim=256, proj="convmlp", bn_type="sync_bn"):
        """
        Initialize the ProjectionHead module with specified dimensions, input channels, projection type, and normalization settings.

        The appropriate convolutional and normalization functions (2D or 3D) are selected
        based on the `ndim` parameter.

        Parameters
        ----------
        ndim : int
            Number of dimensions of the input data (2 for 2D, 3 for 3D).
        in_channels : int
            Number of input channels for the projection head.
        proj_dim : int, optional
            Dimension of the projected output. Defaults to 256.
        proj : str, optional
            Type of projection to use. Options are 'linear' or 'convmlp'.
            'linear' uses a simple 1x1 convolution. 'convmlp' uses a sequence of
            convolution, batch normalization, ReLU, and another convolution.
            Defaults to 'convmlp'.
        bn_type : str, optional
            Type of batch normalization to use if `proj` is 'convmlp'.
            Options are 'sync_bn' or 'none'. Defaults to 'sync_bn'.
        """
        super(ProjectionHead, self).__init__()
        self.ndim = ndim
        if self.ndim == 3:
            self.conv_call = nn.Conv3d
            self.norm_func = get_norm_3d
        else:
            self.conv_call = nn.Conv2d
            self.norm_func = get_norm_2d

        if proj == "linear":
            self.proj = self.conv_call(in_channels, proj_dim, kernel_size=1)
        elif proj == "convmlp":
            self.proj = nn.Sequential(
                self.conv_call(in_channels, in_channels, kernel_size=1),
                self.norm_func(bn_type, in_channels),
                nn.ReLU(inplace=True),
                self.conv_call(in_channels, proj_dim, kernel_size=1),
            )

    def forward(self, x):
        """
        Perform the forward pass through the projection head.

        The input tensor `x` is first passed through the configured projection layer,
        and then the output is L2-normalized along the channel dimension.

        Parameters
        ----------
        x : torch.Tensor
            The input feature tensor. Its shape should be (batch_size, in_channels, D, H, W)
            for 3D data or (batch_size, in_channels, H, W) for 2D data.

        Returns
        -------
        torch.Tensor
            The L2-normalized projected output tensor. Its shape will be
            (batch_size, proj_dim, D, H, W) or (batch_size, proj_dim, H, W),
            depending on `ndim`.
        """
        return F.normalize(self.proj(x), p=2, dim=1)
    
class PSP(nn.Module):
    """
    Implements a Pyramid Scene Parsing (PSP) module.

    PSP captures multi-scale contextual information by applying pooled contexts
    at different grid sizes, projecting them with 1x1 convolutions, and
    upsampling them back to the original feature resolution before concatenation.

    References:
        PSPNet work: `Pyramid Scene Parsing Network <https://arxiv.org/abs/1612.01105>`_.

    Parameters
    ----------
    conv : Type[nn.Conv2d | nn.Conv3d]
        Convolutional layer type (e.g., nn.Conv2d for 2D, nn.Conv3d for 3D).
    in_dims : int
        Number of input channels.
    out_dims : int
        Number of output channels for each pooled branch and for the final output.
    norm : str, optional
        Normalization layer type: 'bn', 'sync_bn', 'in', 'gn', or 'none'.
        Defaults to "none".
    pool_sizes : list of int, optional
        Output sizes for the adaptive pooling branches. Typical values are
        [1, 2, 3, 6] for 2D PSP. Defaults to [1, 2, 3, 6].
    """

    def __init__(
        self,
        conv,
        in_dims,
        out_dims,
        norm: str = "none",
        pool_sizes = [1, 2, 3, 6],
    ):
        super().__init__()

        self.conv = conv
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.norm = norm
        self.pool_sizes = pool_sizes

        # Create one pooled branch per pool size
        self.stages = nn.ModuleList(
            [self._make_stage(ps) for ps in pool_sizes]
        )

        # Bottleneck to fuse original + pooled features
        # input channels = in_dims (original) + len(pool_sizes) * out_dims
        self.bottleneck = conv(
            in_dims + len(pool_sizes) * out_dims,
            out_dims,
            kernel_size=1,
            bias=False,
        )

        if norm != "none":
            if conv == nn.Conv2d:
                self.bottleneck_norm = get_norm_2d(norm, out_dims)
            else:
                self.bottleneck_norm = get_norm_3d(norm, out_dims)
        else:
            self.bottleneck_norm = None

        self.relu = nn.ReLU(inplace=True)

    def _make_stage(self, pool_size: int) -> nn.Module:
        """
        Create one PSP branch: AdaptiveAvgPool -> 1x1 conv -> ReLU (+ norm).
        """
        if self.conv == nn.Conv2d:
            pool = nn.AdaptiveAvgPool2d(output_size=pool_size)
        else:
            pool = nn.AdaptiveAvgPool3d(output_size=pool_size)

        block = [
            pool,
            self.conv(self.in_dims, self.out_dims, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        ]

        if self.norm != "none":
            if self.conv == nn.Conv2d:
                block.append(get_norm_2d(self.norm, self.out_dims))
            else:
                block.append(get_norm_3d(self.norm, self.out_dims))

        return nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PSP module.

        Input
        -----
        x : Tensor
            Shape (B, in_dims, H, W) for 2D or (B, in_dims, D, H, W) for 3D.

        Output
        ------
        Tensor
            Shape (B, out_dims, H, W) or (B, out_dims, D, H, W), same spatial
            size as the input feature map.
        """
        input_size = x.shape[2:]  # (H, W) or (D, H, W)

        # Start with the original feature map
        priors = [x]

        # Add each pooled/upsampled context branch
        for stage in self.stages:
            out = stage(x)  # pooled and projected, small spatial size

            # Upsample back to input_size
            if x.dim() == 4:  # 2D: (B, C, H, W)
                mode = "bilinear"
            else:             # 3D: (B, C, D, H, W)
                mode = "trilinear"

            out = F.interpolate(
                out,
                size=input_size,
                mode=mode,
                align_corners=False,
            )
            priors.append(out)

        # Concatenate original + pooled features
        out = torch.cat(priors, dim=1)

        # Bottleneck projection
        out = self.bottleneck(out)
        if self.bottleneck_norm is not None:
            out = self.bottleneck_norm(out)
        out = self.relu(out)

        return out
    

class SpatialGatherModule(nn.Module):
    """
    Aggregate context for each class based on the predicted segmentation
    probability map (coarse segmentation).

    Args
    ----
    num_classes : int
        Number of semantic classes.
    scale : float
        Scaling factor applied to the probabilities before softmax.
    """

    def __init__(self, num_classes: int, scale: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, feats: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        feats : Tensor
            Feature map from the backbone/head, shape (B, C, H, W).
        probs : Tensor
            Coarse class score map, shape (B, K, H, W).

        Returns
        -------
        context : Tensor
            Object region representations, shape (B, C, K, 1).
        """
        b, c, h, w = feats.shape
        _, k, _, _ = probs.shape
        assert k == self.num_classes, "probs channels must equal num_classes"

        # B x C x HW
        feats = feats.view(b, c, -1)
        # B x K x HW
        probs = probs.view(b, k, -1)

        # Normalize probabilities along spatial dimension
        probs = F.softmax(self.scale * probs, dim=2)  # B x K x HW

        # context: B x C x K = B x C x HW @ B x HW x K
        context = torch.bmm(feats, probs.permute(0, 2, 1))

        # reshape to (B, C, K, 1) so it can be treated as a small "feature map"
        context = context.unsqueeze(-1)

        return context


class ObjectAttentionBlock2D(nn.Module):
    """
    Object context attention block.
    Takes pixel-level features and object-region context, and returns a
    context-augmented feature map.

    Args
    ----
    in_channels : int
        Number of input feature channels.
    key_channels : int
        Number of channels in the key/query space.
    norm : str
        Normalization type ('bn', 'sync_bn', 'in', 'gn', 'none').
    """

    def __init__(self, in_channels: int, key_channels: int, norm: str = "bn"):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels

        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1, bias=False),
            get_norm_2d(norm, key_channels),
            nn.ReLU(inplace=True),
        )

        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1, bias=False),
            get_norm_2d(norm, key_channels),
            nn.ReLU(inplace=True),
        )

        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1, bias=False),
            get_norm_2d(norm, key_channels),
            nn.ReLU(inplace=True),
        )

        self.f_up = nn.Sequential(
            nn.Conv2d(key_channels, in_channels, kernel_size=1, bias=False),
            get_norm_2d(norm, in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, obj_context: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Pixel-level features, shape (B, C, H, W).
        obj_context : Tensor
            Object-region context, shape (B, C, K, 1) from SpatialGatherModule.

        Returns
        -------
        context_enhanced : Tensor
            Context-augmented features, shape (B, C, H, W).
        """
        b, c, h, w = x.shape
        _, _, k, _ = obj_context.shape  # K = number of classes

        # Query: pixel features
        query = self.f_pixel(x)                         # B x key_channels x H x W
        query = query.view(b, self.key_channels, -1)    # B x key_channels x N
        query = query.permute(0, 2, 1)                  # B x N x key_channels

        # Key & value: object-region features
        key = self.f_object(obj_context)                # B x key_channels x K x 1
        key = key.view(b, self.key_channels, -1)        # B x key_channels x K

        value = self.f_down(obj_context)                # B x key_channels x K x 1
        value = value.view(b, self.key_channels, -1)    # B x key_channels x K
        value = value.permute(0, 2, 1)                  # B x K x key_channels

        # Attention: (B x N x K)
        sim_map = torch.bmm(query, key)                 # B x N x K
        sim_map = sim_map * (self.key_channels ** -0.5)
        sim_map = F.softmax(sim_map, dim=-1)

        # Context: (B x N x key_channels)
        context = torch.bmm(sim_map, value)             # B x N x key_channels
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(b, self.key_channels, h, w)

        # Map back to in_channels
        context = self.f_up(context)

        return context

class OCRHead(nn.Module):
    """
    OCR head for HRNet-style semantic segmentation.

    References:
        `Object-Contextual Representations for Semantic Segmentation <https://arxiv.org/abs/1809.00916>`_.

    Pattern:
        1) 3x3 conv to get mid-level features.
        2) 1x1 conv to get coarse class scores (auxiliary logits).
        3) SpatialGatherModule to build object-region representations.
        4) ObjectAttentionBlock2D to inject object context into pixel features.
        5) 1x1 bottleneck conv to produce final OCR features.

    Parameters
    ----------
    conv : Type[nn.Conv2d]
        Convolution layer type (typically nn.Conv2d).
    in_dims : int
        Number of input channels from the HRNet fused features.
    out_dims : int
        Number of output channels of the OCR feature map (e.g. 256).
    num_classes : int
        Number of segmentation classes.
    norm : str, optional
        Normalization type: 'bn', 'sync_bn', 'in', 'gn', or 'none'.
    key_dims : int, optional
        Number of channels in the key/query space for attention.
        Default is 256; often set to out_dims // 2 or out_dims.
    scale : float, optional
        Scale factor used in SpatialGatherModule.
    """

    def __init__(
        self,
        conv,
        in_dims: int,
        out_dims: int,
        num_classes: int,
        norm: str = "none",
        key_dims: int = 256,
        scale: float = 1.0,
    ):
        super().__init__()

        assert conv is nn.Conv2d, "Current OCRHead implementation is 2D-only."

        # 1) 3x3 conv -> norm -> ReLU (reduce/reshape backbone features)
        block = [
            conv(in_dims, out_dims, kernel_size=3, stride=1, padding=1, bias=False)
        ]
        if norm != "none":
            block.append(get_norm_2d(norm, out_dims))
        block.append(nn.ReLU(inplace=True))
        self.conv3x3 = nn.Sequential(*block)

        # 2) Coarse classifier (auxiliary logits)
        self.aux_classifier = conv(
            out_dims, num_classes, kernel_size=1, stride=1, padding=0, bias=True
        )

        # 3) Spatial gather
        self.spatial_gather = SpatialGatherModule(num_classes=num_classes, scale=scale)

        # 4) Object context attention block
        # key_dims defaults to out_dims if not specified
        if key_dims is None:
            key_dims = out_dims
        self.object_context_block = ObjectAttentionBlock2D(
            in_channels=out_dims,
            key_channels=key_dims,
            norm=norm,
        )

        # 5) Final bottleneck conv
        bottleneck = [
            conv(out_dims, out_dims, kernel_size=1, stride=1, padding=0, bias=False)
        ]
        if norm != "none":
            bottleneck.append(get_norm_2d(norm, out_dims))
        bottleneck.append(nn.ReLU(inplace=True))
        self.bottleneck = nn.Sequential(*bottleneck)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : Tensor
            Fused HRNet feature map, shape (B, in_dims, H, W).

        Returns
        -------
        ocr_feats : Tensor
            OCR-refined features, shape (B, out_dims, H, W).
            Typically fed into a 1x1 classifier + upsample.
        aux_logits : Tensor
            Coarse class logits, shape (B, num_classes, H, W).
            Can be used with an auxiliary loss.
        """
        feats = self.conv3x3(x)                  # (B, out_dims, H, W)

        # Coarse segmentation prediction
        aux_logits = self.aux_classifier(feats)  # (B, num_classes, H, W)

        # Build object-region context
        context = self.spatial_gather(feats, aux_logits)  # (B, out_dims, K, 1)

        # Inject object context
        ocr_context = self.object_context_block(feats, context)  # (B, out_dims, H, W)

        # Final bottleneck
        ocr_feats = self.bottleneck(ocr_context)  # (B, out_dims, H, W)

        # Let's return only the OCR features for now
        # return ocr_feats, aux_logits
        return ocr_feats
