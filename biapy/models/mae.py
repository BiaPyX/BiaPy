"""
This file implements the Masked Autoencoder (MAE) model with a Vision Transformer (ViT) backbone, as described in the paper "Masked Autoencoders Are Scalable Vision Learners" (https://arxiv.org/abs/2111.06377).

The MAE model is designed for self-supervised pre-training of Vision Transformers
by reconstructing masked-out patches of an image. It consists of an encoder that
processes visible patches and a lightweight decoder that reconstructs the original
image from the encoder's latent representation and mask tokens.

Key components and functionalities include:

Classes:

- ``MaskedAutoencoderViT``: The main MAE model, encompassing the encoder and decoder.

Functions:

- ``mae_vit_base_patch16_dec512d8b``: Factory function for a base MAE-ViT model.
- ``mae_vit_large_patch16_dec512d8b``: Factory function for a large MAE-ViT model.
- ``mae_vit_huge_patch14_dec512d8b``: Factory function for a huge MAE-ViT model.

The implementation supports both 2D and 3D image inputs, different masking strategies
(random and grid), and provides methods for patching/unpatching images,
forward passes through encoder/decoder, and loss calculation.

References:

- Masked Autoencoders Are Scalable Vision Learners: https://arxiv.org/abs/2111.06377
- timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
- DeiT: https://github.com/facebookresearch/deit
"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from biapy.models.tr_layers import PatchEmbed


class MaskedAutoencoderViT(nn.Module):
    """
    Masked Autoencoder (MAE) with Vision Transformer (ViT) backbone.

    This model implements the architecture proposed in "Masked Autoencoders Are Scalable
    Vision Learners" for self-supervised pre-training by reconstructing masked image patches.
    It comprises an encoder to process unmasked patches and a decoder to reconstruct
    the full image, including the masked regions.

    Reference: `Masked Autoencoders Are Scalable Vision Learners <https://arxiv.org/abs/2111.06377>`_.

    Parameters
    ----------
    img_size : int, optional
        Size of the input image (height and width for 2D, or depth, height, and width for 3D,
        assuming square/cubic dimensions). Defaults to 224.

    patch_size : int, optional
        Size of the square/cubic patch (token) that the image is divided into. Defaults to 16.

    in_chans : int, optional
        Number of input image channels (e.g., 3 for RGB, 1 for grayscale). Defaults to 3.

    ndim : int, optional
        Number of input dimensions, 2 for 2D images (H, W) or 3 for 3D images (D, H, W). Defaults to 2.

    embed_dim : int, optional
        Dimensionality of the embedding space for the Vision Transformer encoder. Defaults to 1024.

    depth : int, optional
        Number of transformer encoder blocks (layers). Defaults to 24.

    num_heads : int, optional
        Number of attention heads in the multi-head attention layer of the encoder. Defaults to 16.

    mlp_ratio : float, optional
        Ratio of the hidden dimension of the MLP block to the `embed_dim`. Defaults to 4.0.

    decoder_embed_dim : int, optional
        Dimensionality of the embedding space for the MAE decoder. Defaults to 512.

    decoder_depth : int, optional
        Number of transformer decoder blocks (layers). Defaults to 8.

    decoder_num_heads : int, optional
        Number of attention heads in the multi-head attention layer of the decoder. Defaults to 16.

    norm_layer : Torch layer, optional
        Normalization layer to use throughout the model (e.g., `nn.LayerNorm`). Defaults to `nn.LayerNorm`.

    norm_pix_loss : bool, optional
        If True, normalize pixel values (mean 0, variance 1) per patch before computing the
        reconstruction loss. This helps stabilize training. Defaults to False.

    masking_type : str, optional
        Type of masking strategy to apply. Can be "random" for random patch masking
        or "grid" for structured grid masking. Defaults to "random".

    mask_ratio : float, optional
        Percentage of the input image patches to mask out. Value between 0 and 1.
        Only applicable when `masking_type` is "random". Defaults to 0.5.

    device : Torch device, optional
        The device (e.g., 'cuda', 'cpu') where the model parameters and input tensors
        will be stored. Defaults to None, inferring from input.

    Returns
    -------
    model : nn.Module
        The MAE model.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        ndim=2,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        masking_type="random",
        mask_ratio=0.5,
        device=None,
    ):
        """
        Initialize the Masked Autoencoder Vision Transformer (MAE-ViT) model.

        Sets up the encoder and decoder components of the MAE. The encoder
        transforms image patches into a latent representation, while the decoder
        reconstructs the original image from this latent space and mask tokens.
        Includes parameters for configuring the transformer architecture,
        masking strategy, and loss normalization.

        Parameters
        ----------
        img_size : int, optional
            Side length of the square/cubic input image. Defaults to 224.
        patch_size : int, optional
            Side length of the square/cubic patches. Defaults to 16.
        in_chans : int, optional
            Number of input image channels. Defaults to 3.
        ndim : int, optional
            Number of spatial dimensions of the input (2 for 2D, 3 for 3D). Defaults to 2.
        embed_dim : int, optional
            Dimensionality of the patch embeddings and transformer blocks in the encoder. Defaults to 1024.
        depth : int, optional
            Number of transformer encoder blocks. Defaults to 24.
        num_heads : int, optional
            Number of attention heads in the encoder's multi-head attention. Defaults to 16.
        mlp_ratio : float, optional
            Ratio of the hidden dimension in the MLP block to `embed_dim` for both encoder and decoder. Defaults to 4.0.
        decoder_embed_dim : int, optional
            Dimensionality of the embeddings in the decoder. Defaults to 512.
        decoder_depth : int, optional
            Number of transformer decoder blocks. Defaults to 8.
        decoder_num_heads : int, optional
            Number of attention heads in the decoder's multi-head attention. Defaults to 16.
        norm_layer : Type[nn.Module], optional
            The normalization layer class to use (e.g., `nn.LayerNorm`). Defaults to `nn.LayerNorm`.
        norm_pix_loss : bool, optional
            If True, normalize pixel values per patch (mean 0, variance 1) before computing the reconstruction loss. Defaults to False.
        masking_type : str, optional
            Specifies the masking strategy: "random" for random patch dropout, or "grid" for a structured checkerboard-like mask. Defaults to "random".
        mask_ratio : float, optional
            The proportion of patches to mask when `masking_type` is "random". Value between 0 and 1. Defaults to 0.5.
        device : Optional[torch.device], optional
            The device on which to place the model and its parameters. If None, it will be inferred. Defaults to None.

        Raises
        ------
        AssertionError
            If `masking_type` is not "random" or "grid".
        """
        super().__init__()
        self.ndim = ndim
        self.in_chans = in_chans
        self.mask_ratio = mask_ratio
        self.masking_type = masking_type

        assert masking_type in ["random", "grid"]
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            ndim=self.ndim,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**self.ndim * in_chans, bias=True
        )  # decoder to patch
        # --------------------------------------------------------------------------
        if masking_type == "random":
            self.masking_func = self.random_masking
        else:
            self.masking_func = self.grid_masking

            # Define grid mask, as it doesn't change over epochs
            D, L = embed_dim, self.patch_embed.num_patches
            if self.ndim == 2:
                self.mask = torch.zeros([img_size // patch_size, img_size // patch_size], device=device)
                self.mask[::2, ::2] = 1
                self.mask[1::2, 1::2] = 1
            else:
                self.mask = torch.zeros(
                    [
                        img_size // patch_size,
                        img_size // patch_size,
                        img_size // patch_size,
                    ],
                    device=device,
                )
                self.mask[::2, ::2, ::2] = 1
                self.mask[1::2, 1::2, 1::2] = 1
            self.mask = self.mask.flatten()
            self.ids_keep = torch.argsort(self.mask)[: L // 2].unsqueeze(-1).repeat(1, 1, D)
            self.mask = self.mask.unsqueeze(0)
            self.ids_restore = torch.argsort(torch.argsort(self.mask))

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize the weights of the model's layers.

        This method applies specific initialization strategies to different types of layers
        within the MAE model, including:
        - Truncated normal initialization for positional embeddings (`pos_embed`, `decoder_pos_embed`).
        - Xavier uniform initialization for the patch embedding projection (`patch_embed.proj.weight`).
        - Normal initialization for the class token (`cls_token`) and mask token (`mask_token`).
        - Calls `_init_weights` to initialize `nn.Linear` and `nn.LayerNorm` layers.
        """
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize weights for `nn.Linear` and `nn.LayerNorm` layers.

        This is a helper function typically called by `initialize_weights` using `model.apply()`.
        It applies Xavier uniform initialization to `nn.Linear` weights (with bias set to 0 if present)
        and sets `nn.LayerNorm` weights to 1.0 and biases to 0.

        Parameters
        ----------
        m : nn.Module
            A module within the network to initialize.
        """
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        Convert an input image into a sequence of non-overlapping patches.

        This function is the inverse of `unpatchify`. It rearranges the pixel data
        from a standard image tensor format into a sequence of flattened patch vectors.

        Parameters
        ----------
        imgs : Tensor
            Input images.

            - For 2D: `(N, C, H, W)`, where `N` is batch size, `C` are channels,
              `H` is height, and `W` is width.
            - For 3D: `(N, C, Z, H, W)`, where `N` is batch size, `C` are channels,
              `Z` is depth, `H` is height, and `W` is width.

        Returns
        -------
        x : Torch tensor
            Flattened image patches.

            - For 2D: `(N, L, patch_size**2 * C)`, where `L` is the total number
              of patches (`(H*W)/(p*p)`).
            - For 3D: `(N, L, patch_size**3 * C)`, where `L` is the total number
              of patches (`(Z*H*W)/(p*p*p)`).

        """
        p = self.patch_embed.patch_size

        if self.ndim == 2:
            assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
            h = w = imgs.shape[2] // p
            x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
            x = torch.einsum("nchpwq->nhwpqc", x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_chans))
        else:
            assert imgs.shape[2] == imgs.shape[3] == imgs.shape[4] and imgs.shape[2] % p == 0
            d = h = w = imgs.shape[2] // p
            x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, d, p, h, p, w, p))
            x = torch.einsum("ncdahpwq->ndhwapqc", x)
            x = x.reshape(shape=(imgs.shape[0], d * h * w, p**3 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        Reconstruct an image from a sequence of flattened patches.

        This function is the inverse of `patchify`. It takes a batch of flattened
        patches and reshapes them back into standard image tensor format.

        Parameters
        ----------
        x : Tensor
            Input patches.

            - For 2D: `(N, L, patch_size**2 * C)`, where `N` is batch size, `L` is
              the number of patches, and `C` are channels.
            - For 3D: `(N, L, patch_size**3 * C)`, where `N` is batch size, `L` is
              the number of patches, and `C` are channels.

        Returns
        -------
        imgs : Torch tensor
            Reconstructed images.

            - For 2D: `(N, C, H, W)`.
            - For 3D: `(N, C, Z, H, W)`.

        """
        p = self.patch_embed.patch_size
        if self.ndim == 2:
            h = w = int(x.shape[1] ** 0.5)
            assert h * w == x.shape[1]
            x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
            x = torch.einsum("nhwpqc->nchpwq", x)
            imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        else:
            d = h = w = int(round(x.shape[1] ** 0.333333))
            assert d * h * w == x.shape[1]
            x = x.reshape(shape=(x.shape[0], d, h, w, p, p, p, self.in_chans))
            x = torch.einsum("ndhwapqc->ncdahpwq", x)
            imgs = x.reshape(shape=(x.shape[0], self.in_chans, d * p, h * p, h * p))

        return imgs

    def random_masking(self, x):
        """
        Perform per-sample random masking of input patches.

        This method randomly selects a subset of patches to keep (visible) and
        masks out the rest. The selection is done by shuffling patch indices
        based on random noise.

        Parameters
        ----------
        x : Tensor
            Input patches with shape `(N, L, D)`, where `N` is the batch size,
            `L` is the number of patches, and `D` is the embedding dimension.

        Returns
        -------
        x_masked : Tensor
            The input patches with masked patches removed, shape `(N, L_keep, D)`.
        mask : Tensor
            A binary mask tensor of shape `(N, L)`, where 0 indicates a kept (visible)
            patch and 1 indicates a removed (masked) patch.
        ids_restore : Tensor
            Indices to restore the original order of patches, shape `(N, L)`.
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def grid_masking(self, x):
        """
        Perform grid-based masking for input patches.

        This method applies a pre-defined checkerboard-like grid mask to the
        input patches, ensuring a structured masking pattern.

        Parameters
        ----------
        x : Tensor
            Input patches with shape `(N, L, D)`, where `N` is the batch size,
            `L` is the number of patches, and `D` is the embedding dimension.

        Returns
        -------
        x_masked : Tensor
            The input patches with masked patches removed based on the grid pattern,
            shape `(N, L_keep, D)`.
        mask : Tensor
            A binary mask tensor of shape `(N, L)`, where 0 indicates a kept (visible)
            patch and 1 indicates a removed (masked) patch.
        ids_restore : Tensor
            Indices to restore the original order of patches, shape `(N, L)`.
        """
        N, L, D = x.shape  # batch, length, dim

        mask = self.mask.repeat(N, 1)
        x_masked = torch.gather(x, dim=1, index=self.ids_keep.repeat(N, 1, 1))
        return x_masked, mask, self.ids_restore.repeat(N, 1)

    def forward_encoder(self, x):
        """
        Perform the forward pass through the MAE encoder.

        This method first embeds the input image into patches, adds positional
        embeddings, applies masking, appends the class token, and then processes
        the resulting sequence through a series of Transformer encoder blocks.

        Parameters
        ----------
        x : Tensor
            Input image tensor. Its shape depends on `ndim`:

            - For 2D: `(N, C, H, W)`
            - For 3D: `(N, C, Z, H, W)`

        Returns
        -------
        latent : Tensor
            The latent representation produced by the encoder, typically
            ` (N, L_keep + 1, embed_dim)` where `L_keep` is the number of visible patches.
        mask : Tensor
            A binary mask indicating which patches were kept (0) or removed (1),
            shape `(N, L)`.
        ids_restore : Tensor
            Indices to restore the original patch order, shape `(N, L)`.
        """
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.masking_func(x)
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """
        Perform the forward pass through the MAE decoder.

        The decoder takes the encoder's latent representation, appends mask tokens,
        restores the original patch order, adds decoder positional embeddings, and
        then processes the sequence through a series of Transformer decoder blocks
        to predict the original pixel values of all patches.

        Parameters
        ----------
        x : Tensor
            Latent representation from the encoder, shape `(N, L_keep + 1, embed_dim)`.
        ids_restore : Tensor
            Indices to restore the original patch order, shape `(N, L)`.

        Returns
        -------
        x : Tensor
            The reconstructed patches, shape `(N, L, patch_size**ndim * in_chans)`.
        """
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        Calculate the MAE reconstruction loss.

        The loss is computed only on the masked patches. Optionally, pixel values
        can be normalized per patch before loss calculation.

        Parameters
        ----------
        imgs : Tensor
            Original input images.
            - For 2D: `(N, C, H, W)`.
            - For 3D: `(N, C, Z, H, W)`.

        pred : Tensor
            Predicted patches from the decoder, shape `(N, L, patch_size**ndim * C)`.

        mask : Tensor
            A binary mask indicating which patches were masked (1) or visible (0),
            shape `(N, L)`.

        Returns
        -------
        loss : Tensor
            The calculated mean squared error (MSE) loss, averaged only over the
            masked patches.
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs) -> dict:
        """
        Perform the complete forward pass of the Masked Autoencoder.

        This method orchestrates the full MAE process: encoding visible patches,
        decoding the full image, and calculating the reconstruction loss.

        Parameters
        ----------
        imgs : Tensor
            Input images.

            - For 2D: `(N, C, H, W)`.
            - For 3D: `(N, C, Z, H, W)`.

        Returns
        -------
        dict
            A dictionary containing:

            - "loss": The calculated reconstruction loss (Tensor).
            - "pred": The predicted full patch sequence from the decoder (Tensor),
                      shape `(N, L, patch_size**ndim * C)`.
            - "mask": The binary mask used during masking (Tensor), shape `(N, L)`.
            
        """
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)

        return { "loss": loss, "pred": pred, "mask": mask}

    def save_images(self, _x, _y, _mask, dtype):
        """
        Generate and prepare images for visualization/saving from MAE outputs.

        This method reconstructs the predicted image, creates a masked version of
        the original input, and generates an image where visible patches from
        the original are combined with reconstructed masked patches.

        Parameters
        ----------
        _x : Torch tensor
            Original input images.
            - For 2D: `(N, C, H, W)`.
            - For 3D: `(N, C, Z, H, W)`.

        _y : Torch tensor
            MAE model's predicted patches, shape `(N, L, patch_size**ndim * C)`.

        _mask : Torch tensor
            Binary mask indicating masked (1) and visible (0) patches, shape `(N, L)`.

        dtype : Numpy dtype
            The desired NumPy data type for the output images.

        Returns
        -------
        pred : 4D/5D Numpy array
            The fully reconstructed images (from decoder predictions), converted to NumPy.
            - For 2D: `(N, H, W, C)`.
            - For 3D: `(N, Z, H, W, C)`.
        p_mask : 4D/5D Numpy array
            The original input images with only the visible (unmasked) patches remaining,
            converted to NumPy.
            - For 2D: `(N, H, W, C)`.
            - For 3D: `(N, Z, H, W, C)`.
        pred_visi : 4D/5D Numpy array
            The image where the visible patches are from the original input, and the
            masked regions are filled with the decoder's predictions, converted to NumPy.
            - For 2D: `(N, H, W, C)`.
            - For 3D: `(N, Z, H, W, C)`.
        """
        pred = np.zeros(_x.shape, dtype=dtype)
        p_mask = np.zeros(_x.shape, dtype=dtype)
        pred_visi = np.zeros(_x.shape, dtype=dtype)
        for i in range(len(_x)):
            y = self.unpatchify(_y[i].unsqueeze(dim=0))[0]
            y = y.detach().cpu()

            # visualize the mask
            mask = _mask[i].unsqueeze(dim=0).detach()
            mask = mask.unsqueeze(-1).repeat(
                1, 1, self.patch_embed.patch_size**self.ndim * self.in_chans
            )  # (N, H*W, p*p*3)
            mask = self.unpatchify(mask)[0]  # 1 is removing, 0 is keeping
            mask = mask.detach().cpu()
            x = _x[i].detach().cpu()

            # masked image
            im_masked = x * (1 - mask)

            # MAE reconstruction pasted with visible patches
            im_paste = x * (1 - mask) + y * mask

            pred[i] = y.numpy()
            p_mask[i] = im_masked.numpy()
            pred_visi[i] = im_paste.numpy()

        if self.ndim == 2:
            return (
                pred.transpose((0, 2, 3, 1)),
                p_mask.transpose((0, 2, 3, 1)),
                pred_visi.transpose((0, 2, 3, 1)),
            )
        else:
            return (
                pred.transpose((0, 2, 3, 4, 1)),
                p_mask.transpose((0, 2, 3, 4, 1)),
                pred_visi.transpose((0, 2, 3, 4, 1)),
            )


def mae_vit_base_patch16_dec512d8b(**kwargs):
    """
    Create a Masked Autoencoder ViT (MAE-ViT) model with a Base-sized encoder and a decoder with 512 embedding dimensions and 8 blocks.

    This function serves as a convenient constructor for a specific MAE-ViT
    configuration, often used as a standard baseline.

    Parameters
    ----------
    **kwargs
        Arbitrary keyword arguments to be passed to the `MaskedAutoencoderViT`
        constructor. This allows overriding default parameters like `img_size`,
        `in_chans`, `norm_pix_loss`, `masking_type`, `mask_ratio`, or `device`.

    Returns
    -------
    model : MaskedAutoencoderViT
        An initialized MAE-ViT model configured as a base variant.
    """
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    """
    Create a Masked Autoencoder ViT (MAE-ViT) model with a Large-sized encoder and a decoder with 512 embedding dimensions and 8 blocks.

    This function provides a constructor for a larger MAE-ViT configuration,
    suitable for tasks requiring more capacity.

    Parameters
    ----------
    **kwargs
        Arbitrary keyword arguments to be passed to the `MaskedAutoencoderViT`
        constructor. This allows overriding default parameters like `img_size`,
        `in_chans`, `norm_pix_loss`, `masking_type`, `mask_ratio`, or `device`.

    Returns
    -------
    model : MaskedAutoencoderViT
        An initialized MAE-ViT model configured as a large variant.
    """
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    """
    Create a Masked Autoencoder ViT (MAE-ViT) model with a Huge-sized encoder and a decoder with 512 embedding dimensions and 8 blocks.

    This function provides a constructor for the largest MAE-ViT configuration,
    designed for tasks demanding maximum model capacity.

    Parameters
    ----------
    **kwargs
        Arbitrary keyword arguments to be passed to the `MaskedAutoencoderViT`
        constructor. This allows overriding default parameters like `img_size`,
        `in_chans`, `norm_pix_loss`, `masking_type`, `mask_ratio`, or `device`.

    Returns
    -------
    model : MaskedAutoencoderViT
        An initialized MAE-ViT model configured as a huge variant.
    """
    model = MaskedAutoencoderViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocksyy
