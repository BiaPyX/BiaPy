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
    Mask autoenconder (MAE) with VisionTransformer (ViT) backbone.

    Reference: `Masked Autoencoders Are Scalable Vision Learners <https://arxiv.org/abs/2111.06377>`_.

    Parameters
    ----------
    img_size : int, optional
        Size of the input image.

    patch_size : int, optional
        Size of the input size or token size for the transformer.

    in_chans : int, optional
        Number of channels.

    ndim : int, optional
        Number of input dimensions.

    embed_dim : int, optional
        Size of the transformer embedding.

    depth : int, optional
        Number of layers of the transformer.

    num_heads : int, optional
        Number of heads in the multi-head attention layer.

    mlp_ratio : float, optional
        Size of the dense layers of the final classifier. This value will mutiply ``embed_dim``.

    decoder_embed_dim : int, optional
        Size of the transformer embedding in the decoder.

    decoder_depth: int, optional
        Number of layers of the decoder.

    decoder_num_heads : int, optional
        Number of heads in the multi-head attention layer in the decoder.

    norm_layer : Torch layer, optional
        Normalization layer to use in the model.

    norm_pix_loss : bool, optional
        Use (per-patch) normalized pixels as targets for computing loss

    mask_ratio : float, optional
        Percentage of the input image to mask. Value between 0 and 1.

    device : Torch device
        Device used.

    Returns
    -------
    model : Torch model
        MAE model.
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
        Initialize layer weigths.
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
        Initialize nn.Linear and nn.LayerNor layer's weights.
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
        Create patches from input image. Opposite function of :func:`~unpatchify`.

        Parameters
        ----------
        imgs : Tensor
            Input images. In 2D: ``(N, C, H, W)``, in 3D: ``(N, C, Z, H, W)``.
            Where ``N`` is the batch size, ``C`` are the channels, ``Z`` image depth,
            ``H`` image height and ``W`` image's width.

        Returns
        -------
        x : Torch tensor
            MAE model. in 2D: ``(N, L, patch_size**2 *C)`` in 3D: ``(N, L, patch_size**3 *C)``.
            Where ``N`` is the batch size, ``L`` is the multiplication of dimension (i.e. ``Z``,
            ``H`` and ``W``) and ``C`` are the channels.
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
        Create original image shape from input patches. Opposite function of :func:`~patchify`.

        Parameters
        ----------
        x : Tensor
            Input images. In 2D: ``(N, L, patch_size**2 *C)``, in 3D: ``(N, L, patch_size**3 *C)``.
            Where ``N`` is the batch size, ``L`` is the multiplication of dimension (i.e. ``Z``,
            ``H`` and ``W``) and ``C`` are the channels.

        Returns
        -------
        imgs : Torch tensor
            MAE model. in 2D: ``(N, C, H, W)`` in 3D: ``(N, C, Z, H, W)``. Where ``N`` is the batch size,
            ``C`` are the channels, ``Z`` image depth, ``H`` image height and ``W`` image's width.
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
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.

        Parameters
        ----------
        x : Tensor
            Input images. Is shape is ``(N, L, D)`` shape. Where ``N`` is the batch size,
            ``L`` is the multiplication of dimension (i.e. ``Z``, ``H`` and ``W``) and
            ``D`` is ``embed_dim``.
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
        Perform grid masking for each sample.

        Parameters
        ----------
        x : Tensor
            Input images. Is shape is ``(N, L, D)`` shape. Where ``N`` is the batch size,
            ``L`` is the multiplication of dimension (i.e. ``Z``, ``H`` and ``W``) and
            ``D`` is ``embed_dim``.
        """
        N, L, D = x.shape  # batch, length, dim

        mask = self.mask.repeat(N, 1)
        x_masked = torch.gather(x, dim=1, index=self.ids_keep.repeat(N, 1, 1))
        return x_masked, mask, self.ids_restore.repeat(N, 1)

    def forward_encoder(self, x):
        """
        Encoder forward pass.
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
        Decoder forward pass.
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
        MAE loss calculation.

        Parameters
        ----------
        imgs : Tensor
            Input images. In 2D: ``(N, C, H, W)``, in 3D: ``(N, C, Z, H, W)``. Where ``N`` is the batch size,
            ``C`` are the channels, ``Z`` image depth, ``H`` image height and ``W`` image's width.

        pred : Tensor
            Predictions. In 2D: ``(N, L, patch_size**2 *C)``, in 3D: ``(N, L, patch_size**3 *C)``.
            Where ``N`` is the batch size, ``L`` is the multiplication of dimension (i.e. ``Z``,
            ``H`` and ``W``) and ``C`` are the channels.

        mask : 2d array
            Information of which patches will be retain and masked. Shape is: ``(N, L)`` where ``0`` is keep
            and ``1`` is remove.

        Returns
        -------
        loss : Tensor
            Calculated loss on masked patches only.
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
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)

        return { "loss": loss, "pred": pred, "mask": mask}

    def save_images(self, _x, _y, _mask, dtype):
        """
        Save images from MAE.

        Parameters
        ----------
        _x : Torch tensor
            Input images. In 2D: ``(N, C, H, W)``, in 3D: ``(N, C, Z, H, W)``. Where ``N`` is the batch size,
            ``C`` are the channels, ``Z`` image depth, ``H`` image height and ``W`` image's width.

        _y : Torch tensor
            MAE model prediction. In 2D: ``(N, L, patch_size**2 *C)``, in 3D: ``(N, L, patch_size**3 *C)``.
            Where ``N`` is the batch size, ``L`` is the multiplication of dimension (i.e. ``Z``,
            ``H`` and ``W``) and ``C`` represents the channels.

        _mask : 2d array
            Information of which patches will be retain and masked. Shape is: ``(N, L)`` where ``0`` is keep
            and ``1`` is remove.

        dtype : Numpy dtype
            Dtype to save the images.

        Returns
        -------
        pred : 4D/5D Numpy array
            Predicted images converted to Numpy. In 2D: ``(N, H, W, C)``, in 3D: ``(N, Z, H, W, C)``. Where ``N`` is the batch size,
            ``C`` are the channels, ``Z`` image depth, ``H`` image height and ``W`` image's width.

        p_mask : 4D/5D Numpy array
            Predicted images's mask. In 2D: ``(N, H, W, C)``, in 3D: ``(N, Z, H, W, C)``. Where ``N`` is the batch size,
            ``C`` are the channels, ``Z`` image depth, ``H`` image height and ``W`` image's width.

        pred_visi : 4D/5D Numpy array
            Predicted image with visible patches. In 2D: ``(N, H, W, C)``, in 3D: ``(N, Z, H, W, C)``. Where ``N`` is the batch size,
            ``C`` are the channels, ``Z`` image depth, ``H`` image height and ``W`` image's width.
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
