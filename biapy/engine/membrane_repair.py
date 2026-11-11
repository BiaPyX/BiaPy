"""
Membrane repair (PROBLEM.INSTANCE_SEG.MEMBRANE_REPAIR): instance segmentation from an upstream membrane prediction.

``Membrane_Repair_Workflow`` is an ``Instance_Segmentation_Workflow`` (same targets, decoding, matching and
post-processing) whose model input is ``DERIVED_CHANNELS``, computed from the on-disk ``SOURCE_CHANNELS`` after the
membrane corruption augmentors (``Membrane2D/3DRepairDataGenerator``) or, at test time, in ``process_test_sample``.
``DATA.PATCH_SIZE`` keeps ``len(SOURCE_CHANNELS)``; the derived width is only shown to the model builder and the
test-time patching. On top of INSTANCE_SEG it adds the affinity losses (``LOSS.TYPE`` 'BCE'/'MALIS'/'CLDICE'/'SVOX')
and the SKOOTS target (``LOSS.TYPE`` ['SKOOTS'], ``INSTANCE_CREATION_PROCESS`` 'skoots').
"""
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from biapy.data.data_manipulation import check_binary_masks
from biapy.data.membrane_channels import derive_membrane_input_channels
from biapy.data.post_processing.skoots import skoots_instances
from biapy.data.pre_processing import affinity_offsets_from_opts, hvz_skeleton_scale, load_resolution_stats
from biapy.engine.base_workflow import Base_Workflow
from biapy.engine.instance_seg import Instance_Segmentation_Workflow as _Instance_Segmentation_Workflow
from biapy.engine.malis_loss import MalisLoss
from biapy.engine.metrics import (
    CLDiceComponentPenaltyLoss,
    SkootsLoss,
    SkootsMetrics,
    WeightedBCEAffinityLoss,
    WeightedCompositeLoss,
    loss_encapsulation,
    multiple_metrics,
    skoots_offset_axes,
)
from biapy.engine.supervoxel_loss import SupervoxelLoss


class Membrane_Repair_Workflow(_Instance_Segmentation_Workflow):
    """Instance segmentation with derived membrane input channels and the affinity/SKOOTS losses."""

    def __init__(self, cfg, job_identifier, device, system_dict, args, **kwargs):
        """Initialize the workflow and check that the "membrane" source channel is binary."""
        super().__init__(cfg, job_identifier, device, system_dict, args, **kwargs)

        # "membrane", if present in SOURCE_CHANNELS, must be a binary mask
        source_channels = list(cfg.PROBLEM.INSTANCE_SEG.MEMBRANE_REPAIR.SOURCE_CHANNELS)
        membrane_idx = source_channels.index("membrane") if "membrane" in source_channels else None
        if membrane_idx is not None:
            ignore = [cfg.LOSS.IGNORE_INDEX] if cfg.LOSS.IGNORE_INDEX != -1 else []
            is_3d = cfg.PROBLEM.NDIM == "3D"
            if cfg.TRAIN.ENABLE and cfg.DATA.TRAIN.CHECK_DATA:
                check_binary_masks(cfg.DATA.TRAIN.PATH, is_3d=is_3d, channel=membrane_idx, extra_allowed_values=ignore)
                if not cfg.DATA.VAL.FROM_TRAIN:
                    check_binary_masks(cfg.DATA.VAL.PATH, is_3d=is_3d, channel=membrane_idx, extra_allowed_values=ignore)
            if cfg.TEST.ENABLE and cfg.DATA.TEST.CHECK_DATA:
                check_binary_masks(cfg.DATA.TEST.PATH, is_3d=is_3d, channel=membrane_idx)
        # Keep ignored membrane voxels out of the train/val normalization
        if cfg.LOSS.IGNORE_INDEX != -1 and membrane_idx is not None:
            self.norm_module["ignore_value"] = cfg.LOSS.IGNORE_INDEX
            self.norm_module["ignore_channels"] = [membrane_idx]

    def prepare_model(self):
        """
        Build the model with the derived input width (``len(DERIVED_CHANNELS)``) and, if ``CLASS_SET_CHANNELS`` is
        set, STUNet's permutation-invariant class-set encoder; both overrides are undone after the build.
        """
        cfg = self.cfg
        mr = cfg.PROBLEM.INSTANCE_SEG.MEMBRANE_REPAIR
        raw_patch = tuple(cfg.DATA.PATCH_SIZE)
        raw_class_set_idxs = list(cfg.MODEL.STUNET.CLASS_SET_CHANNEL_IDXS)
        raw_class_set_out = cfg.MODEL.STUNET.CLASS_SET_OUT_CHANNELS
        raw_class_set_pooling = cfg.MODEL.STUNET.CLASS_SET_POOLING

        was_frozen = cfg.is_frozen()
        cfg.defrost()
        cfg.DATA.PATCH_SIZE = raw_patch[:-1] + (len(mr.DERIVED_CHANNELS),)
        if mr.CLASS_SET_CHANNELS:
            derived = list(mr.DERIVED_CHANNELS)
            cfg.MODEL.STUNET.CLASS_SET_CHANNEL_IDXS = [derived.index(ch) for ch in mr.CLASS_SET_CHANNELS]
            cfg.MODEL.STUNET.CLASS_SET_OUT_CHANNELS = mr.CLASS_SET_ENCODER.OUT_CHANNELS
            cfg.MODEL.STUNET.CLASS_SET_POOLING = mr.CLASS_SET_ENCODER.POOLING
        try:
            super().prepare_model()
        finally:
            cfg.DATA.PATCH_SIZE = raw_patch
            cfg.MODEL.STUNET.CLASS_SET_CHANNEL_IDXS = raw_class_set_idxs
            cfg.MODEL.STUNET.CLASS_SET_OUT_CHANNELS = raw_class_set_out
            cfg.MODEL.STUNET.CLASS_SET_POOLING = raw_class_set_pooling
            if was_frozen:
                cfg.freeze()

    def define_activations_and_channels(self):
        """INSTANCE_SEG's; the SKOOTS loss needs 'sigmoid' ('F'/'P'/'C') and 'tanh' (offsets) applied in training too."""
        super().define_activations_and_channels()
        if [str(n).upper() for n in self.cfg.LOSS.TYPE] == ["SKOOTS"]:
            self.head_activations = [
                "sigmoid" if ch in ("F", "P", "C") else "tanh"
                for ch in self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS
                if ch != "I"
            ]

    def define_metrics(self):
        """
        INSTANCE_SEG's per-channel losses when ``LOSS.TYPE`` is empty, the SKOOTS loss for ['SKOOTS'], else the
        affinity composite of 'BCE'/'MALIS'/'CLDICE'/'SVOX' weighted by ``LOSS.WEIGHTS`` (a term with weight 0 is
        not built, so MALIS's and the supervoxel loss's CPU cost is avoided when ablated off).
        """
        names = [str(n).upper() for n in self.cfg.LOSS.TYPE]
        if not names:
            return super().define_metrics()
        if names == ["SKOOTS"]:
            return self._define_skoots_metrics()

        inst = self.cfg.PROBLEM.INSTANCE_SEG
        by_name = dict(zip(names, list(self.cfg.LOSS.WEIGHTS)))
        ignore_index = self.cfg.LOSS.IGNORE_INDEX if self.cfg.LOSS.IGNORE_INDEX != -1 else None
        weighted_sub_losses = []
        if by_name.get("BCE", 0.0) > 0:
            bce_loss = WeightedBCEAffinityLoss(
                class_rebalance_within_channels=inst.CLASS_REBALANCE_WITHIN_CHANNELS, ignore_index=ignore_index
            ).to(self.device)
            weighted_sub_losses.append((by_name["BCE"], bce_loss))
        if by_name.get("MALIS", 0.0) > 0:
            a_opts = dict(inst.DATA_CHANNELS_EXTRA_OPTS[0]).get("A", {})
            resolution = self._malis_resolution() if a_opts.get("units", "voxel") == "physical_nm" else None
            offsets = affinity_offsets_from_opts(a_opts, ndim=self.dims, resolution=resolution)
            weighted_sub_losses.append((by_name["MALIS"], MalisLoss(offsets=offsets, ignore_index=ignore_index)))
        if by_name.get("CLDICE", 0.0) > 0:
            weighted_sub_losses.append(
                (by_name["CLDICE"], CLDiceComponentPenaltyLoss(ignore_index=ignore_index).to(self.device))
            )
        if by_name.get("SVOX", 0.0) > 0:
            weighted_sub_losses.append((by_name["SVOX"], SupervoxelLoss(ignore_index=ignore_index).to(self.device)))
        self.loss = loss_encapsulation(WeightedCompositeLoss(weighted_sub_losses).to(self.device))

        # Single IoU over all the affinity channels
        self.train_metric_names = ["IoU (A channels)"]
        self.train_metric_best = ["max"]
        self.test_metric_names = ["IoU (A channels)"]
        self.train_metrics, self.test_metrics = [
            [
                multiple_metrics(
                    num_classes=self.cfg.DATA.N_CLASSES,
                    metric_names=["IoU (A channels)"],
                    device=device,
                    out_channels=["A"],
                    model_source=self.cfg.MODEL.SOURCE,
                    ignore_index=self.cfg.LOSS.IGNORE_INDEX,
                    ndim=self.dims,
                )
            ]
            for device in (self.device, self.test_device)
        ]

    def _define_skoots_metrics(self):
        """SKOOTS loss and metrics."""
        cfg = self.cfg
        inst = cfg.PROBLEM.INSTANCE_SEG
        sk = cfg.LOSS.SKOOTS
        channels = [ch for ch in inst.DATA_CHANNELS if ch != "I"]
        scale = hvz_skeleton_scale(dict(inst.DATA_CHANNELS_EXTRA_OPTS[0]), self.dims)
        epochs = cfg.TRAIN.EPOCHS
        skoots_loss = SkootsLoss(
            channels=channels,
            ndim=self.dims,
            scale=scale,
            channel_weights=list(inst.DATA_CHANNEL_WEIGHTS),
            initial_sigma=list(sk.INITIAL_SIGMA),
            sigma_decay=[[m, int(f * epochs)] for m, f in sk.SIGMA_DECAY],
            tversky={
                "foreground": tuple(sk.FOREGROUND_TVERSKY),
                "skeleton": tuple(sk.SKELETON_TVERSKY),
                "embed": tuple(sk.EMBED_TVERSKY),
                "contour": tuple(sk.CONTOUR_TVERSKY),
            },
            start_epochs=[int(f * epochs) for f in sk.START_FRACTIONS],
        ).to(self.device)
        self.loss = loss_encapsulation(
            WeightedCompositeLoss([(float(cfg.LOSS.WEIGHTS[0]), skoots_loss)]).to(self.device)
        )
        self.train_metrics = [SkootsMetrics(channels, ndim=self.dims, scale=scale)]
        self.train_metric_names = list(self.train_metrics[0].names)
        self.train_metric_best = list(self.train_metrics[0].best)
        self.test_metrics = [SkootsMetrics(channels, ndim=self.dims, scale=scale)]
        self.test_metric_names = list(self.test_metrics[0].names)

    def loss_description(self) -> str:
        """``LOSS.TYPE``'s terms when set, else INSTANCE_SEG's per-channel losses."""
        if self.cfg.LOSS.TYPE:
            return Base_Workflow.loss_description(self)
        return super().loss_description()

    def _malis_resolution(self) -> Tuple[float, ...]:
        """
        Single (z, y, x) training resolution used to turn nm affinity offsets into MALIS voxel offsets: from
        ``DATA.TRAIN.RESOLUTION`` or, if unset, the ``resolution.json`` of the train/val data.
        """
        cfg = self.cfg
        if cfg.DATA.TRAIN.RESOLUTION[0] != -1:
            return tuple(cfg.DATA.TRAIN.RESOLUTION)
        paths = [cfg.DATA.TRAIN.PATH] + ([] if cfg.DATA.VAL.FROM_TRAIN else [cfg.DATA.VAL.PATH])
        found = set()
        for path in paths:
            found.update(load_resolution_stats(path).values())
        if len(found) != 1:
            raise ValueError(
                "MALIS with 'units': 'physical_nm' affinities needs a single training resolution (set "
                f"'DATA.TRAIN.RESOLUTION' or a resolution.json next to the data), found: {sorted(found)}. "
                "Use voxel units for data with mixed resolutions."
            )
        return found.pop()

    def _derive_input(self, x: NDArray) -> NDArray:
        """``SOURCE_CHANNELS`` -> ``DERIVED_CHANNELS`` of one channel-last sample."""
        mr = self.cfg.PROBLEM.INSTANCE_SEG.MEMBRANE_REPAIR
        return derive_membrane_input_channels(
            x,
            list(mr.SOURCE_CHANNELS),
            list(mr.DERIVED_CHANNELS),
            dict(mr.DERIVED_CHANNELS_EXTRA_OPTS[0]),
            ndim=self.dims,
            resolution=self.resolution,
        )

    def prepare_bmz_data(self, img: NDArray):
        """
        Replace ``img``'s ``SOURCE_CHANNELS`` with ``DERIVED_CHANNELS`` before ``Base_Workflow.prepare_bmz_data``
        calls the model.

        Parameters
        ----------
        img : 4D/5D Numpy array
            Image in torch axes order, ``(b, c, y, x)`` or ``(b, c, z, y, x)``, with ``SOURCE_CHANNELS`` channels.
        """
        expanded = np.stack([self._derive_input(sample) for sample in np.moveaxis(img, 1, -1)])
        super().prepare_bmz_data(np.moveaxis(expanded, -1, 1))

    def process_test_sample(self):
        """
        Replace X's ``SOURCE_CHANNELS`` with ``DERIVED_CHANNELS`` and run INSTANCE_SEG's test step with
        ``DATA.PATCH_SIZE`` set to the derived width. Saved predictions (``TEST.REUSE_PREDICTIONS``) need no input.
        """
        if self.current_sample.get("discard", False) or self.cfg.TEST.REUSE_PREDICTIONS:
            return super().process_test_sample()
        cfg = self.cfg
        self.current_sample["X"] = np.expand_dims(self._derive_input(self.current_sample["X"][0]), 0)
        raw_patch = tuple(cfg.DATA.PATCH_SIZE)
        was_frozen = cfg.is_frozen()
        cfg.defrost()
        cfg.DATA.PATCH_SIZE = raw_patch[:-1] + (len(cfg.PROBLEM.INSTANCE_SEG.MEMBRANE_REPAIR.DERIVED_CHANNELS),)
        try:
            return super().process_test_sample()
        finally:
            cfg.DATA.PATCH_SIZE = raw_patch
            if was_frozen:
                cfg.freeze()

    def _create_instance_labels(self, pred: NDArray, save_dir: Optional[str] = None, verbose: bool = False):
        """INSTANCE_SEG's, plus the 'skoots' ``INSTANCE_CREATION_PROCESS`` (``pred`` is ``(Z, Y, X, C)``)."""
        inst = self.cfg.PROBLEM.INSTANCE_SEG
        if inst.INSTANCE_CREATION_PROCESS != "skoots":
            return super()._create_instance_labels(pred, save_dir=save_dir, verbose=verbose)
        channels = [ch for ch in inst.DATA_CHANNELS if ch != "I"]
        p = pred if self.dims == 3 else pred[0]
        sk = inst.MEMBRANE_REPAIR.SKOOTS
        labels = skoots_instances(
            foreground=p[..., channels.index("F")],
            skeleton=p[..., channels.index("P")],
            offsets=p[..., [channels.index(c) for c in skoots_offset_axes(self.dims)]],
            scale=hvz_skeleton_scale(dict(inst.DATA_CHANNELS_EXTRA_OPTS[0]), self.dims),
            fg_th=sk.FG_TH,
            skeleton_th=sk.SKELETON_TH,
            n_steps=sk.N_STEPS,
        ).astype(np.uint32)
        return labels if labels.ndim == 3 else labels[None]
