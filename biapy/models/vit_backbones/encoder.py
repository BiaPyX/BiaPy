"""
ViT encoder of 'unetr', 'dpt' and 'vit_readout', built on the model itself (``patch_embed``, ``cls_token``,
``reg_token``, ``pos_embed``, ``blocks`` and, per architecture, ``ln_pre``, ``norm``, ``neck``) so parameter names
match across architectures and pretrained weights load into all of them. Supports overlapping tokens, stochastic
depth, gradient checkpointing and the SAM 3, DINOv3 and Cellpose-SAM blocks.
"""

from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from timm.layers import trunc_normal_

from biapy.models.vit_backbones.cellpose_sam import CELLPOSE_SAM_VIT_PARAMS, build_cellpose_sam_neck
from biapy.models.vit_backbones.common import (
    VIT_BACKBONES,
    PatchEmbed,
    build_vit_blocks,
    check_vit_backbone,
)
from biapy.models.vit_backbones.dinov3 import DINOV3_VIT_PARAMS
from biapy.models.vit_backbones.sam3 import SAM3_VIT_PARAMS


class ViTEncoderMixin:
    """Build with ``_build_vit_encoder``; run with ``forward_tokens`` or ``forward_features``."""

    def _build_vit_encoder(
        self,
        arch: str,
        input_shape: Tuple[int, ...],
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        vit_model: str = "custom",
        token_stride: int = -1,
        drop_path_rate: float = 0.0,
        grad_checkpointing: bool = False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        final_norm: bool = True,
        pre_norm: bool = False,
        trainable_pos_embed: bool = True,
        preset_patch_size: Optional[int] = None,
    ):
        """
        Build the ViT encoder.

        Parameters
        ----------
        arch : str
            Architecture name, to check ``vit_model``.

        input_shape : Tuple[int, ...]
            ``(y, x, channels)`` or ``(z, y, x, channels)``.

        patch_size, embed_dim, depth, num_heads, mlp_ratio : int, int, int, int, float
            ViT geometry, only used with ``vit_model`` = "custom".

        vit_model : str, optional
            "custom" or one of ``VIT_BACKBONES``.

        token_stride : int, optional
            Patch embedding stride (``-1`` = token size).

        drop_path_rate : float, optional
            Stochastic depth of the last block (linear from 0).

        grad_checkpointing : bool, optional
            Recompute the blocks in the backward pass.

        norm_layer : optional
            LayerNorm of the plain ViT blocks and final norm, unless the backbone sets its epsilon.

        final_norm : bool, optional
            Create a final ``norm``.

        pre_norm : bool, optional
            Create ``ln_pre`` (SAM 3's; identity for the rest).

        trainable_pos_embed : bool, optional
            Learn the position embedding.

        preset_patch_size : int, optional
            Token size overriding the backbone's (UNETR).
        """
        if len(input_shape) not in [3, 4]:
            raise ValueError(f"'input_shape' needs to be (y, x, channels) or (z, y, x, channels). Provided: {input_shape}")
        ndim = len(input_shape) - 1
        check_vit_backbone(vit_model, arch, ndim, input_shape[-1])

        blocks_kind = "timm"
        init_values = None
        if vit_model != "custom":
            preset = VIT_BACKBONES[vit_model]
            patch_size = preset_patch_size if preset_patch_size is not None else preset["patch_size"]
            embed_dim = preset["embed_dim"]
            depth = preset["depth"]
            num_heads = preset["num_heads"]
            mlp_ratio = preset["mlp_ratio"]
            init_values = preset.get("init_values")
            if "norm_eps" in preset:
                norm_layer = partial(nn.LayerNorm, eps=preset["norm_eps"])
            blocks_kind = preset["blocks"]
            print(f"Building {type(self).__name__}'s ViT backbone as '{vit_model}': {preset}")
        token_stride = patch_size if token_stride <= 0 else token_stride

        self.input_shape = input_shape
        self.embed_dim = embed_dim
        self.depth = depth
        self.token_stride = token_stride
        self.grad_checkpointing = grad_checkpointing

        self.patch_embed = PatchEmbed(
            img_size=input_shape[0],
            patch_size=patch_size,
            in_chans=input_shape[-1],
            ndim=ndim,
            embed_dim=embed_dim,
            stride=token_stride,
        )
        self.grid_size = self.patch_embed.grid_size
        grid = (self.grid_size, self.grid_size)
        drop_path = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.feature_dim = embed_dim
        self.neck = None
        self.reg_token = None

        if blocks_kind == "cellpose_sam":
            # No class token, grid-shaped position embedding, no final norm, SAM's neck
            self.cls_token = None
            self.pos_embed = nn.Parameter(torch.zeros(1, self.grid_size, self.grid_size, embed_dim))
            self.blocks = build_vit_blocks(
                blocks_kind, embed_dim, depth, num_heads, mlp_ratio, grid, 0, drop_path=drop_path
            )
            self.norm = nn.Identity()
            self.neck = build_cellpose_sam_neck()
            self.feature_dim = CELLPOSE_SAM_VIT_PARAMS["out_chans"]
        else:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            num_prefix_tokens = 1
            if blocks_kind == "dinov3":
                # RoPE instead of a position embedding, plus register tokens
                self.reg_token = nn.Parameter(torch.zeros(1, DINOV3_VIT_PARAMS["num_register_tokens"], embed_dim))
                num_prefix_tokens += self.reg_token.shape[1]
                self.pos_embed = None
            else:
                self.pos_embed = nn.Parameter(
                    torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim), requires_grad=trainable_pos_embed
                )
            self.blocks = build_vit_blocks(
                blocks_kind,
                embed_dim,
                depth,
                num_heads,
                mlp_ratio,
                grid,
                num_prefix_tokens,
                norm_layer=norm_layer,
                init_values=init_values,
                drop_path=drop_path,
            )
            if final_norm:
                self.norm = norm_layer(embed_dim)
        if pre_norm:
            self.ln_pre = (
                nn.LayerNorm(embed_dim, eps=SAM3_VIT_PARAMS["norm_eps"]) if blocks_kind == "sam3" else nn.Identity()
            )
        print(
            f"{type(self).__name__}'s ViT: {depth} blocks, {patch_size}x{patch_size} tokens with stride "
            f"{token_stride} ({'x'.join([str(self.grid_size)] * ndim)} token grid), stochastic depth up to "
            f"{drop_path_rate}"
        )

    def _init_vit_encoder_weights(self):
        """Initialize the encoder as timm's ViT."""
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.reg_token is not None:
            nn.init.normal_(self.reg_token, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)

        def _init(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.blocks.apply(_init)
        if getattr(self, "norm", None) is not None:
            self.norm.apply(_init)

    @property
    def num_prefix_tokens(self) -> int:
        """Class and register tokens before the patch tokens."""
        if self.cls_token is None:
            return 0
        return 1 + (0 if self.reg_token is None else self.reg_token.shape[1])

    def _run_blocks(self, x: torch.Tensor, layers: List[int]) -> List[torch.Tensor]:
        """Run the blocks and return the (unnormalized) output of ``layers`` (1-indexed)."""
        outs = []
        for i, blk in enumerate(self.blocks):
            if self.grad_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
            if i + 1 in layers:
                outs.append(x)
        return outs

    def embed_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Images ``(B, C, (Z), Y, X)`` to the block input: class, register and patch tokens with position embedding."""
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        if getattr(self, "ln_pre", None) is not None:
            x = self.ln_pre(x)
        if self.reg_token is not None:
            x = torch.cat((x[:, :1], self.reg_token.expand(B, -1, -1), x[:, 1:]), dim=1)
        return x

    def forward_tokens(self, x: torch.Tensor, layers: List[int], normalize: bool = True) -> List[torch.Tensor]:
        """
        Tokens of the ``layers`` blocks (1-indexed): one ``(B, 1 + N, embed_dim)`` tensor each, class token first,
        register tokens dropped, optionally with the final norm.
        """
        num_reg = 0 if self.reg_token is None else self.reg_token.shape[1]
        outs = []
        for y in self._run_blocks(self.embed_tokens(x), layers):
            if normalize:
                y = self.norm(y)
            outs.append(torch.cat((y[:, :1], y[:, 1 + num_reg :]), dim=1) if num_reg > 0 else y)
        return outs

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Last block as a ``(B, feature_dim, h, w)`` map: normalized patch tokens, or the neck output (Cellpose-SAM)."""
        if self.neck is None:
            return self.tokens_to_map(self.forward_tokens(x, [self.depth])[0][:, 1:])

        x = self.patch_embed(x).reshape(x.shape[0], self.grid_size, self.grid_size, -1) + self.pos_embed
        x = self._run_blocks(x, [self.depth])[0]
        return self.neck(x.permute(0, 3, 1, 2))

    def tokens_to_map(self, tokens: torch.Tensor) -> torch.Tensor:
        """``(B, N, D)`` patch tokens to a ``(B, D, (d), h, w)`` map."""
        B, N, D = tokens.shape
        return tokens.transpose(1, 2).reshape(B, D, *([self.grid_size] * self.patch_embed.ndim))
