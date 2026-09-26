"""
Dedicated data generator for the membrane-repair problem (PROBLEM.IMAGE_TO_IMAGE.MEMBRANE_REPAIR).

Reuses ``PairBaseDataGenerator.apply_transform`` (geometric warps, content augmentors, and online
regeneration of the GT affinity ('A') channel from the warped instance-label ('I') channel) via
``super().apply_transform(...)``, and adds on top: the membrane corruption augmentors
(``biapy.data.generators.membrane_augmentors``) and the derived-channel computation
(``biapy.data.membrane_channels``) that replaces ``SOURCE_CHANNELS`` with ``DERIVED_CHANNELS``.

The raw on-disk X file only has ``len(SOURCE_CHANNELS)`` channels; ``DATA.PATCH_SIZE`` reflects
that. ``SOURCE_CHANNELS`` are warped/corruption-augmented like any other X data, then replaced --
not appended to -- by the channels derived from them, recomputed every call (never cached to
disk). This mixin is the one place X's width changes: raw ``SOURCE_CHANNELS`` width in, derived-
only width out.
"""
import copy
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from numpy.typing import NDArray

from biapy.data.generators.pair_data_2D_generator import Pair2DImageDataGenerator
from biapy.data.generators.pair_data_3D_generator import Pair3DImageDataGenerator
from biapy.data.membrane_channels import derive_membrane_input_channels, source_channel_offsets
from biapy.data.pre_processing import channel_physical_offsets, labels_into_channels
from biapy.data.generators.membrane_augmentors import (
    artifact_corruption,
    skeleton_perturbation,
    slice_dropout,
    spurious_bridge,
    synthetic_gap,
)


class MembraneRepairGeneratorMixin:
    """
    Adds membrane-repair-specific channel bookkeeping, corruption augmentors and derived-channel
    expansion on top of a ``Pair2DImageDataGenerator``/``Pair3DImageDataGenerator`` base class.
    """

    def __init__(
        self,
        source_channels: List[str] = ["membrane"],
        derived_channels: List[str] = ["skeleton_dt", "hessian_blob", "meijering"],
        derived_channels_extra_opts: Dict = {},
        slice_dropout_aug: Dict = {},
        gap_aug: Dict = {},
        bridge_aug: Dict = {},
        artifact_aug: Dict = {},
        skeleton_perturb_aug: Dict = {},
        ignore_value: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize the membrane-repair generator mixin.

        Parameters
        ----------
        source_channels : list of str, optional
            Ordered raw, on-disk input channel names, e.g. ``["membrane"]``, ``["raw"]``,
            ``["membrane", "raw"]`` or ``["raw", "membrane"]``. Resolved by name, not position.

        derived_channels : list of str, optional
            Ordered derived channel names appended after ``source_channels`` to build the
            network input (see ``biapy.data.membrane_channels.derive_membrane_input_channels``).

        derived_channels_extra_opts : dict, optional
            Per-derived-channel options, e.g. ``{"skeleton_dt": {"clamp_px": 10}}``.

        slice_dropout_aug : dict, optional
            ``{"enable": bool, "prob": float}``. Each source channel's z-slices are independently
            zeroed with probability ``"prob"``; derived channels are never touched (see
            ``slice_dropout`` in ``biapy.data.generators.membrane_augmentors``).

        gap_aug, bridge_aug, artifact_aug, skeleton_perturb_aug : dict, optional
            Corruption-augmentor configs, each with an ``"enable"`` bool, a ``"prob"`` float and
            augmentor-specific range keys (see ``biapy.data.generators.membrane_augmentors``).
            ``artifact_aug``'s blob artifact affects every channel, not just the membrane channel.

        ignore_value : int, optional
            Ignore label (``LOSS.IGNORE_INDEX``) in the membrane channel. Fed to the model as 0; the
            affinities between two ignored voxels are set to this value.

        **kwargs : dict
            Forwarded to ``Pair2DImageDataGenerator``/``Pair3DImageDataGenerator``. Must include
            ``shape`` set to ``DATA.PATCH_SIZE``, channel count ``len(source_channels)``.
        """
        raw_channels = kwargs["shape"][-1]
        if raw_channels != len(source_channels):
            raise ValueError(
                f"Generator's raw-facing channel count ({raw_channels}) must equal "
                f"len(SOURCE_CHANNELS) = {len(source_channels)} (SOURCE_CHANNELS={source_channels})."
            )

        super().__init__(**kwargs)

        self.source_channels = list(source_channels)
        self.derived_channels = list(derived_channels)
        self.derived_channels_extra_opts = dict(derived_channels_extra_opts)
        self.slice_dropout_aug = dict(slice_dropout_aug)
        self.gap_aug = dict(gap_aug)
        self.bridge_aug = dict(bridge_aug)
        self.artifact_aug = dict(artifact_aug)
        self.skeleton_perturb_aug = dict(skeleton_perturb_aug)

        self.channel_offsets = source_channel_offsets(self.source_channels, self.derived_channels)
        self.membrane_idx = self.channel_offsets.get("membrane")
        self.droppable_channel_idxs = tuple(self.channel_offsets[name] for name in self.source_channels)

        # The corruption augmentors below all paint into the membrane channel; without one
        # (raw-only SOURCE_CHANNELS) they have nothing to corrupt.
        for name, aug in (
            ("gap_aug", self.gap_aug),
            ("bridge_aug", self.bridge_aug),
            ("artifact_aug", self.artifact_aug),
            ("skeleton_perturb_aug", self.skeleton_perturb_aug),
        ):
            if aug.get("enable") and self.membrane_idx is None:
                raise ValueError(f"{name} requires 'membrane' in SOURCE_CHANNELS.")

        self.ignore_value = ignore_value
        self.ignore_mask_col = None
        if self.ignore_value is not None:
            if self.membrane_idx is None:
                raise ValueError("An ignore value (LOSS.IGNORE_INDEX) requires 'membrane' in SOURCE_CHANNELS.")
            if "A" not in self.data_channels:
                raise ValueError("An ignore value (LOSS.IGNORE_INDEX) requires 'A' in DATA_CHANNELS.")
            # Ignore mask travels as an extra binary mask channel through the augmentations
            self.ignore_mask_col = self.Y_channels
            self.mask_norm = copy.deepcopy(self.mask_norm)
            self.mask_norm["per_channel_info"][self.ignore_mask_col] = {"type": "bin", "div": False}

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate one pair of data, always deriving the input channels.

        Overrides ``PairBaseDataGenerator.__getitem__``, which only calls ``apply_transform`` when
        ``self.da or self.do_cellpose_rescale`` -- but the derived channels are the actual network
        input, not optional augmentation, so they must be computed every time.

        Cutmix and Noise2Void support (present in the base ``__getitem__``) are dropped here --
        neither applies to membrane repair.
        """
        _rng_state = None
        if self.val:
            _seed = (int(self.seed) + index) % (2**31 - 1)
            _rng_state = (np.random.get_state(), random.getstate())
            random.seed(_seed)
            np.random.seed(_seed)

        img, mask = self.load_sample(index, geom_enlarge=True)
        img, mask = self.apply_transform(
            img, mask, e_im=None, e_mask=None,
            sample_resolution=self.resolution_for_affinities(index),
        )

        # Drop the instance-label regeneration source: it must never reach the model.
        if self.instance_channel is not None:
            mask = np.delete(mask, self.instance_channel, axis=-1)

        if img.dtype == np.uint16:
            img = torch.from_numpy(img.astype(np.float32))
        else:
            img = torch.from_numpy(img.copy())
        mask = torch.from_numpy(mask.copy())

        if _rng_state is not None:
            np.random.set_state(_rng_state[0])
            random.setstate(_rng_state[1])

        return img, mask

    def apply_transform(
        self, image: NDArray, mask: NDArray, e_im, e_mask,
        diam_factor: float = 1.0, sample_resolution: Optional[Tuple[float, ...]] = None,
    ):
        """
        Apply the shared geometric/content augmentations and GT affinity regeneration (inherited,
        see the module docstring), then the membrane-repair-specific corruption augmentors,
        channel dropout and derived-channel computation, in that order.

        Parameters
        ----------
        image : 3D/4D Numpy array
            Raw source-channel image, ``(y, x, len(SOURCE_CHANNELS))`` in 2D or
            ``(z, y, x, len(SOURCE_CHANNELS))`` in 3D.

        mask : 3D/4D Numpy array
            GT channel stack (affinities + the instance-label regeneration source).

        e_im, e_mask
            Unused here (cutmix donor); forwarded for signature compatibility.

        diam_factor : float, optional
            Unused here (Cellpose-only); forwarded for signature compatibility.

        sample_resolution : tuple of float, optional
            Per-sample ``(z, y, x)`` resolution; forwarded to the inherited ``apply_transform`` for
            'A'-channel regeneration (see ``PairBaseDataGenerator.resolution_for_affinities``).

        Returns
        -------
        image : 3D/4D Numpy array
            Replaced by ``len(DERIVED_CHANNELS)`` channels (the ``SOURCE_CHANNELS`` used to compute
            them are dropped, not appended to).

        mask : 3D/4D Numpy array
            GT channel stack, warped and with 'A' regenerated (unchanged shape).
        """
        if self.ignore_value is not None:
            ignored = image[..., self.membrane_idx] == self.ignore_value
            image[..., self.membrane_idx][ignored] = 0
            mask = np.concatenate([mask, ignored[..., None].astype(mask.dtype)], axis=-1)

        image, mask = super().apply_transform(
            image, mask, e_im, e_mask,
            diam_factor=diam_factor, sample_resolution=sample_resolution,
        )

        if self.ignore_value is not None:
            ignored = mask[..., self.ignore_mask_col] > 0.5
            mask = np.delete(mask, self.ignore_mask_col, axis=-1)
            mask = self._ignore_affinities(mask, ignored, sample_resolution)

        if self.gap_aug.get("enable") and self.da:
            image = synthetic_gap(
                image,
                self.membrane_idx,
                self.ndim,
                prob=float(self.gap_aug.get("prob", 0.5)),
                length_range=tuple(self.gap_aug.get("length_range", (0.3, 1.0))),
                thickness_range=tuple(self.gap_aug.get("thickness_range", (4, 9))),
                n_lines=tuple(self.gap_aug.get("n_lines", (1, 3))),
            )

        if self.bridge_aug.get("enable") and self.da:
            image = spurious_bridge(
                image,
                self.membrane_idx,
                self.ndim,
                prob=float(self.bridge_aug.get("prob", 0.3)),
                length_range=tuple(self.bridge_aug.get("length_range", (0.3, 1.0))),
                thickness_range=tuple(self.bridge_aug.get("thickness_range", (4, 9))),
                n_lines=tuple(self.bridge_aug.get("n_lines", (1, 3))),
            )

        if self.artifact_aug.get("enable") and self.da:
            image = artifact_corruption(
                image,
                self.membrane_idx,
                self.ndim,
                prob=float(self.artifact_aug.get("prob", 0.1)),
                band_prob=float(self.artifact_aug.get("band_prob", 0.5)),
                band_thickness_range=tuple(self.artifact_aug.get("band_thickness_range", (50, 70))),
                blob_size_range=tuple(self.artifact_aug.get("blob_size_range", (0.1, 0.3))),
                blob_n_range=tuple(self.artifact_aug.get("blob_n_range", (1, 3))),
            )

        if self.skeleton_perturb_aug.get("enable") and self.da:
            image = skeleton_perturbation(
                image,
                self.membrane_idx,
                self.ndim,
                prob=float(self.skeleton_perturb_aug.get("prob", 0.3)),
                radius_range=tuple(self.skeleton_perturb_aug.get("radius_range", (1, 2))),
            )

        # Must run before 'derive' below: dropping a source channel after deriving would leave its
        # information intact in the still-derived-from-it channels.
        if self.slice_dropout_aug.get("enable") and self.da:
            image = slice_dropout(
                image, self.droppable_channel_idxs, float(self.slice_dropout_aug.get("prob", 0.3)), self.ndim
            )

        # Derive replaces the source channels with the derived channels -- the model's actual input.
        # self.resolution is stored (x, y, z) (see PairBaseDataGenerator.__init__'s reordering), but
        # derive_membrane_input_channels expects (z, y, x) -- reverse it back.
        image = derive_membrane_input_channels(
            image,
            self.source_channels,
            self.derived_channels,
            self.derived_channels_extra_opts,
            ndim=self.ndim,
            resolution=tuple(reversed(self.resolution)),
        )

        return image, mask

    def _ignore_affinities(
        self, mask: NDArray, ignored: NDArray, sample_resolution: Optional[Tuple[float, ...]] = None
    ) -> NDArray:
        """
        Set to ``ignore_value`` every 'A' target whose two voxels are both ignored.

        Parameters
        ----------
        mask : 3D/4D Numpy array
            GT channel stack (without the ignore mask channel).

        ignored : 2D/3D Numpy array of bool
            Ignore mask, same spatial shape as ``mask``.

        sample_resolution : tuple of float, optional
            Per-sample ``(z, y, x)`` resolution, as used to regenerate 'A'.

        Returns
        -------
        mask : 3D/4D Numpy array
            ``mask`` with the ignored affinity entries set to ``ignore_value``.
        """
        if not ignored.any():
            return mask
        a_opts = dict(self.channel_extra_opts.get("A", {}))
        a_opts["widen_borders"] = 0
        kwargs = {}
        if self.ndim == 3:
            kwargs["resolution"] = sample_resolution if sample_resolution is not None else self.default_resolution_zyx
        both_ignored = labels_into_channels(
            ignored[..., None].astype(np.int32), mode=["A"], channel_extra_opts={"A": a_opts}, **kwargs
        ) > 0.5
        a_start = channel_physical_offsets(self.data_channels, self.channel_extra_opts)["A"]
        a_block = mask[..., a_start : a_start + both_ignored.shape[-1]]
        a_block[both_ignored] = self.ignore_value
        return mask


class Membrane2DRepairDataGenerator(MembraneRepairGeneratorMixin, Pair2DImageDataGenerator):
    """2D membrane-repair data generator. See ``MembraneRepairGeneratorMixin``."""


class Membrane3DRepairDataGenerator(MembraneRepairGeneratorMixin, Pair3DImageDataGenerator):
    """3D membrane-repair data generator. See ``MembraneRepairGeneratorMixin``."""
