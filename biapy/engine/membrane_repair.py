"""
Membrane-repair sub-problem of the IMAGE_TO_IMAGE workflow (PROBLEM.IMAGE_TO_IMAGE.MEMBRANE_REPAIR).

Trains a dataset-agnostic network that repairs membrane-segmentation errors coming out of an
upstream foundation-model + GMM pipeline. This module holds:

- ``prepare_membrane_repair_gt``: offline step that expands the raw GT instance-label folder into
  the affinity (+ raw-label regeneration source) channel stack, into a dedicated cache directory,
  repointing ``DATA.*.GT_PATH`` at it.

  X is *not* prepared offline, unlike Y: the derived channels (skeleton-DT/Hessian/Meijering) must
  be recomputed after every corruption augmentor runs anyway. ``DATA.PATCH_SIZE`` stays
  ``len(SOURCE_CHANNELS)`` (the raw on-disk channel count); the model's expanded input width
  (``len(SOURCE_CHANNELS) + len(DERIVED_CHANNELS)``) is presented to the model builder only for
  the duration of ``prepare_model``.
- ``Membrane_Repair_Workflow``: the workflow subclass wiring the membrane-repair loss options,
  running the Y preparation step above, presenting the expanded input width to the model builder,
  and fixing up ``process_test_sample`` for affinity output.
"""
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from skimage.transform import resize
from tqdm import tqdm

from biapy.data.data_manipulation import check_binary_masks, read_img_as_ndarray, save_tif
from biapy.data.membrane_channels import derive_membrane_input_channels
from biapy.data.post_processing.affinity_agglomeration import watershed_and_agglomerate_affinities
from biapy.data.post_processing.post_processing import apply_label_refinement, watershed_by_channels
from biapy.data.pre_processing import affinity_offsets_from_opts, labels_into_channels
from biapy.engine.base_workflow import Base_Workflow
from biapy.engine.image_to_image import Image_to_Image_Workflow as _Image_to_Image_Workflow
from biapy.engine.malis_loss import MalisLoss
from biapy.engine.metrics import CLDiceComponentPenaltyLoss, WeightedBCEAffinityLoss, multiple_metrics
from biapy.engine.supervoxel_loss import SupervoxelLoss
from biapy.utils.matching import build_tp_fp_fn_report, matching, wrapper_matching_dataset_lazy
from biapy.utils.misc import (
    crop_border_numpy,
    crop_border_tensor,
    get_rank,
    get_world_size,
    is_dist_avail_and_initialized,
    is_main_process,
    os_walk_clean,
    to_pytorch_format,
)


def _channel_prep_suffix(names: List[str], extra_opts: Dict) -> str:
    """
    Build a filesystem-safe suffix encoding channel names + their options.

    Appending this to a raw data path gives a cache directory that's always different from the raw
    path, and changes whenever the channel config changes, so a stale cache is never reused.

    Parameters
    ----------
    names : list of str
        Ordered channel names.

    extra_opts : dict
        Per-channel options, e.g. ``{"A": {"z_affinities": [1]}}``.

    Returns
    -------
    suffix : str
        E.g. ``"_A.z_affinities-1_I"``.
    """
    suffix = ""
    for ch in names:
        suffix += f"_{ch}"
        for entry, val in extra_opts.get(ch, {}).items():
            val_str = (
                str(val).replace(" ", "").replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace(",", "-")
            )
            suffix += f".{entry}-{val_str}"
    return suffix


def prepare_membrane_repair_gt(cfg, out_dir: str, data_type: str = "train") -> None:
    """
    Expand the raw GT instance-label folder into the channel stack the generator consumes.

    For every GT label image in ``DATA.<TRAIN|VAL|TEST>.GT_PATH``, computes
    ``PROBLEM.IMAGE_TO_IMAGE.MEMBRANE_REPAIR.DATA_CHANNELS`` (affinities + the virtual raw-label
    channel) via ``labels_into_channels`` and writes the result to ``out_dir``.

    Only supports plain (non-Zarr/H5) label images.

    Parameters
    ----------
    cfg : YACS configuration
        Running configuration.

    out_dir : str
        Directory to write the prepared channel stack to.

    data_type : str, optional
        Which split to prepare: ``"train"``, ``"val"`` or ``"test"``.
    """
    assert data_type in ["train", "val", "test"]
    tag = data_type.upper()
    data_cfg = getattr(cfg.DATA, tag)
    mr = cfg.PROBLEM.IMAGE_TO_IMAGE.MEMBRANE_REPAIR

    data_channels = list(mr.DATA_CHANNELS)
    channel_extra_opts = dict(mr.DATA_CHANNELS_EXTRA_OPTS[0])

    try:
        gt_files = next(os_walk_clean(data_cfg.GT_PATH))[2]
    except StopIteration:
        raise ValueError(f"No GT label images found in {data_cfg.GT_PATH}")

    os.makedirs(out_dir, exist_ok=True)
    print(f"Creating membrane-repair Y_{data_type} channels {data_channels} in {out_dir} . . .")

    rank, world_size = get_rank(), get_world_size()
    for i in tqdm(range(rank, len(gt_files), world_size), disable=not is_main_process()):
        img_path = os.path.join(data_cfg.GT_PATH, gt_files[i])
        img = read_img_as_ndarray(img_path, is_3d=(cfg.PROBLEM.NDIM == "3D"))
        if img.shape[-1] != 1:
            raise ValueError(
                "Expected membrane-repair GT images to have a single channel containing the "
                f"instance labels, but got shape {img.shape}. Check the image file: {img_path}"
            )

        channels = labels_into_channels(img, mode=data_channels, channel_extra_opts=channel_extra_opts)

        save_tif(
            np.expand_dims(channels, 0),
            data_dir=out_dir,
            filenames=[gt_files[i]],
            verbose=False,
        )


class _MembraneRepairCombinedLoss(nn.Module):
    """
    Combines the enabled ``MEMBRANE_REPAIR_AFFINITY`` sub-losses per ``LOSS.WEIGHTS``.

    Follows the ``(inputs, targets) -> {"losses": [tensor]}`` contract ``train_engine.py`` expects
    from ``self.loss`` regardless of workflow.
    """

    def __init__(self, weighted_sub_losses: List[Tuple[float, nn.Module]]):
        """
        Initialize the combined loss.

        Parameters
        ----------
        weighted_sub_losses : list of (float, nn.Module)
            ``(weight, sub_loss)`` pairs; only entries with a positive weight are ever
            constructed by the caller, so every entry here is actually computed.
        """
        super().__init__()
        self.sub_losses = nn.ModuleList([m for _, m in weighted_sub_losses])
        self.weights = [w for w, _ in weighted_sub_losses]

    def forward(self, inputs, targets):
        """Compute the weighted sum of the enabled sub-losses."""
        if isinstance(inputs, dict):
            inputs = inputs["pred"]
        total = inputs.new_zeros(())
        for w, m in zip(self.weights, self.sub_losses):
            total = total + w * m(inputs, targets)
        return {"losses": [total]}


class Membrane_Repair_Workflow(_Image_to_Image_Workflow):
    """
    Membrane-repair sub-problem of the IMAGE_TO_IMAGE workflow.

    A thin subclass rather than plain ``IMAGE_TO_IMAGE`` config because:

    - ``process_test_sample`` must not rescale predictions back into X's raw intensity domain (they
      are ``[0,1]`` affinity probabilities), and must expand raw X to the model's input width
      before cropping/prediction.
    - GT preparation: Y is a GT instance-label volume expanded into affinity (+ raw-label
      regeneration source) channels before training (see ``prepare_membrane_repair_gt``).
    - Model input width: ``SOURCE_CHANNELS + DERIVED_CHANNELS``, while ``DATA.PATCH_SIZE`` only
      has ``SOURCE_CHANNELS`` -- see ``prepare_model``.
    """

    def __init__(self, cfg, job_identifier, device, system_dict, args, **kwargs):
        """
        Initialize the Membrane_Repair_Workflow.

        ``Base_Workflow.__init__`` sizes the model's output head from
        ``cfg.PROBLEM.IMAGE_TO_IMAGE.OUTPUT_CHANNELS``, so that (and ``DATA.PATCH_SIZE``) must be
        validated/derived *before* it runs, not after. GT preparation only needs ``self.cfg`` to
        exist, so it runs after.
        """
        Membrane_Repair_Workflow._validate_membrane_repair_channel_counts(cfg)
        Membrane_Repair_Workflow._derive_membrane_repair_output_channels(cfg)

        Base_Workflow.__init__(self, cfg, job_identifier, device, system_dict, args, **kwargs)

        # Set before _prepare_membrane_repair_gt (which may populate them) so print_stats/
        # _postprocess_and_evaluate_membrane_repair always find well-defined attributes, even when
        # no GT is available.
        self.original_test_mask_path = None
        self.test_gt_filenames = []
        self.all_matching_stats_merge_patches = []
        self.all_matching_stats_merge_patches_post = []
        self.stats["inst_stats_merge_patches"] = None
        self.stats["inst_stats_merge_patches_post"] = None

        self._prepare_membrane_repair_gt()

        # SOURCE_CHANNELS[0] (membrane) must be a binary mask.
        if cfg.TRAIN.ENABLE and cfg.DATA.TRAIN.CHECK_DATA:
            check_binary_masks(cfg.DATA.TRAIN.PATH, is_3d=(cfg.PROBLEM.NDIM == "3D"), channel=0)
            if not cfg.DATA.VAL.FROM_TRAIN:
                check_binary_masks(cfg.DATA.VAL.PATH, is_3d=(cfg.PROBLEM.NDIM == "3D"), channel=0)
        if cfg.TEST.ENABLE and cfg.DATA.TEST.CHECK_DATA:
            check_binary_masks(cfg.DATA.TEST.PATH, is_3d=(cfg.PROBLEM.NDIM == "3D"), channel=0)

        self.cfg.freeze()
        self.mask_path = self.cfg.DATA.TRAIN.GT_PATH
        self.is_y_mask = True
        self.norm_module["target_type"] = "mask"
        self.test_norm_module["target_type"] = "mask"

    @staticmethod
    def _validate_membrane_repair_channel_counts(cfg):
        """
        Validate (never rewrite) ``DATA.PATCH_SIZE``'s channel count against
        ``len(SOURCE_CHANNELS)`` -- the raw, on-disk channel count; derived channels are computed
        in memory, never written to disk. ``staticmethod`` since it must run before
        ``Base_Workflow.__init__``.
        """
        mr = cfg.PROBLEM.IMAGE_TO_IMAGE.MEMBRANE_REPAIR
        expected = len(mr.SOURCE_CHANNELS)
        if cfg.DATA.PATCH_SIZE[-1] != expected:
            raise ValueError(
                f"DATA.PATCH_SIZE's channel count ({cfg.DATA.PATCH_SIZE[-1]}) must equal "
                f"len(SOURCE_CHANNELS) = {expected} (SOURCE_CHANNELS={list(mr.SOURCE_CHANNELS)}) "
                "-- the raw, on-disk channel count you provide. The derived channels "
                f"({list(mr.DERIVED_CHANNELS)}) are computed in memory and are not part of what "
                "you configure DATA.PATCH_SIZE with."
            )

    @staticmethod
    def _derive_membrane_repair_output_channels(cfg):
        """
        Compute ``PROBLEM.IMAGE_TO_IMAGE.OUTPUT_CHANNELS`` from ``DATA_CHANNELS_EXTRA_OPTS["A"]``.

        The affinity width is implied by the configured offset lists (3 channels -- z, y, x -- per
        offset pair in 3D), so this is always derived, never a user-provided value.
        """
        mr = cfg.PROBLEM.IMAGE_TO_IMAGE.MEMBRANE_REPAIR
        a_opts = dict(mr.DATA_CHANNELS_EXTRA_OPTS[0]).get("A", {})
        channels_per_offset = 3 if cfg.PROBLEM.NDIM == "3D" else 2
        n_offsets = len(a_opts.get("z_affinities" if cfg.PROBLEM.NDIM == "3D" else "y_affinities", [1]))
        cfg.PROBLEM.IMAGE_TO_IMAGE.OUTPUT_CHANNELS = n_offsets * channels_per_offset

    def _prepare_membrane_repair_gt(self):
        """
        Prepare the GT affinity channel cache (skipping it if already present) and repoint
        ``DATA.<TRAIN|VAL|TEST>.GT_PATH`` at it.

        The cache directory is the raw ``GT_PATH`` plus a config-derived suffix (see
        ``_channel_prep_suffix``), never ``GT_PATH`` itself, otherwise the "already prepared?"
        ``os.path.isdir`` check would always be true and silently skip preparation.
        """
        cfg = self.cfg
        mr = cfg.PROBLEM.IMAGE_TO_IMAGE.MEMBRANE_REPAIR

        malis_enabled = (
            cfg.LOSS.TYPE == "MEMBRANE_REPAIR_AFFINITY" and len(cfg.LOSS.WEIGHTS) > 1 and cfg.LOSS.WEIGHTS[1] > 0
        )
        y_extra_opts = dict(mr.DATA_CHANNELS_EXTRA_OPTS[0])
        if malis_enabled:
            a_opts = y_extra_opts.get("A", {})
            if a_opts.get("widen_borders", 1) != 0:
                raise ValueError(
                    "MALIS (LOSS.WEIGHTS[1] > 0) requires PROBLEM.IMAGE_TO_IMAGE.MEMBRANE_REPAIR."
                    "DATA_CHANNELS_EXTRA_OPTS[0]['A']['widen_borders'] == 0 to reconstruct GT "
                    "instance labels correctly from the target affinities -- see "
                    "biapy/engine/malis_loss.py's module docstring."
                )

        y_suffix = "_mrepair" + _channel_prep_suffix(list(mr.DATA_CHANNELS), y_extra_opts)

        def _prepare_and_repoint(tag: str) -> None:
            data_cfg = getattr(cfg.DATA, tag)
            cache_dir = data_cfg.GT_PATH + y_suffix
            if not os.path.isdir(cache_dir):
                if is_dist_avail_and_initialized():
                    torch.distributed.barrier()
                prepare_membrane_repair_gt(cfg, cache_dir, data_type=tag.lower())
            if data_cfg.GT_PATH != cache_dir:
                opts.extend([f"DATA.{tag}.GT_PATH", cache_dir])

        opts = []
        if cfg.TRAIN.ENABLE:
            _prepare_and_repoint("TRAIN")

        if cfg.TRAIN.ENABLE and not cfg.DATA.VAL.FROM_TRAIN:
            _prepare_and_repoint("VAL")

        if cfg.TEST.ENABLE and cfg.DATA.TEST.LOAD_GT:
            # Captured *before* the repoint below: instance evaluation needs the raw GT instance
            # labels, not the prepared affinity stack DATA.TEST.GT_PATH is about to point to.
            self.original_test_mask_path = cfg.DATA.TEST.GT_PATH
            self.test_gt_filenames = next(os_walk_clean(cfg.DATA.TEST.GT_PATH))[2]
            _prepare_and_repoint("TEST")

        if opts:
            cfg.merge_from_list(opts)

    def prepare_model(self):
        """
        Present the model builder with the expanded input width, then restore ``DATA.PATCH_SIZE``.

        ``build_model`` reads ``cfg.DATA.PATCH_SIZE[-1]`` to size the model's input layer and needs
        the real network width (``SOURCE_CHANNELS + DERIVED_CHANNELS``), but every other part of
        BiaPy must keep seeing the raw ``SOURCE_CHANNELS`` count that matches what's on disk. So
        ``cfg.DATA.PATCH_SIZE`` is mutated only for the duration of ``Base_Workflow.prepare_model()``,
        then restored.
        """
        cfg = self.cfg
        mr = cfg.PROBLEM.IMAGE_TO_IMAGE.MEMBRANE_REPAIR
        raw_patch = tuple(cfg.DATA.PATCH_SIZE)
        expanded_patch = raw_patch[:-1] + (len(mr.SOURCE_CHANNELS) + len(mr.DERIVED_CHANNELS),)

        was_frozen = cfg.is_frozen()
        cfg.defrost()
        cfg.DATA.PATCH_SIZE = expanded_patch
        try:
            super().prepare_model()
        finally:
            cfg.DATA.PATCH_SIZE = raw_patch
            if was_frozen:
                cfg.freeze()

    def define_activations_and_channels(self):
        """
        Force ``ce_sigmoid`` activation on every output channel, regardless of
        ``PROBLEM.IMAGE_TO_IMAGE.OUTPUT_CHANNEL_ACT``: skipped at training time (loss gets raw
        logits, for ``BCEWithLogitsLoss``'s numerically-stable convention) and applied as a real
        sigmoid at inference.
        """
        super().define_activations_and_channels()
        self.head_activations = ["ce_sigmoid"] * len(self.head_activations)

    def define_metrics(self):
        """
        Define ``LOSS.TYPE == "MEMBRANE_REPAIR_AFFINITY"``; delegate anything else to
        ``Image_to_Image_Workflow``.

        Combines up to 4 independently-toggleable sub-losses via ``LOSS.WEIGHTS = [w_bce,
        w_malis, w_cldice, w_svox]``: a weight of 0 skips *constructing* that term, not just
        multiplying its result by 0, since MALIS's and the supervoxel loss's CPU cost must be
        avoidable when ablated off.
        """
        if self.cfg.LOSS.TYPE != "MEMBRANE_REPAIR_AFFINITY":
            super().define_metrics()
            return

        weights = list(self.cfg.LOSS.WEIGHTS) + [0.0, 0.0, 0.0, 0.0]
        w_bce, w_malis, w_cldice, w_svox = weights[0], weights[1], weights[2], weights[3]

        weighted_sub_losses = []
        if w_bce > 0:
            weighted_sub_losses.append((w_bce, WeightedBCEAffinityLoss().to(self.device)))
        if w_malis > 0:
            mr = self.cfg.PROBLEM.IMAGE_TO_IMAGE.MEMBRANE_REPAIR
            a_opts = dict(mr.DATA_CHANNELS_EXTRA_OPTS[0]).get("A", {})
            offsets = affinity_offsets_from_opts(a_opts, ndim=self.dims)
            weighted_sub_losses.append((w_malis, MalisLoss(offsets=offsets)))
        if w_cldice > 0:
            weighted_sub_losses.append((w_cldice, CLDiceComponentPenaltyLoss().to(self.device)))
        if w_svox > 0:
            weighted_sub_losses.append((w_svox, SupervoxelLoss().to(self.device)))

        if not weighted_sub_losses:
            raise ValueError(
                "LOSS.WEIGHTS must have at least one positive entry "
                "([w_bce, w_malis, w_cldice, w_svox]) when LOSS.TYPE == 'MEMBRANE_REPAIR_AFFINITY'"
            )

        self.loss = _MembraneRepairCombinedLoss(weighted_sub_losses).to(self.device)

        # Single aggregate IoU across all predicted affinity channels. Requires metric_calculation
        # to be overridden below -- Image_to_Image_Workflow's version would silently drop this
        # unrecognized metric name.
        self.train_metric_names = ["IoU (A channels)"]
        self.train_metric_best = ["max"]
        self.train_metrics = [
            multiple_metrics(
                num_classes=self.cfg.DATA.N_CLASSES,
                metric_names=self.train_metric_names,
                device=self.device,
                out_channels=["A"],
                model_source=self.cfg.MODEL.SOURCE,
                ignore_index=self.cfg.LOSS.IGNORE_INDEX,
                ndim=self.dims,
            )
        ]
        self.test_metric_names = ["IoU (A channels)"]
        self.test_metrics = [
            multiple_metrics(
                num_classes=self.cfg.DATA.N_CLASSES,
                metric_names=self.test_metric_names,
                device=self.test_device,
                out_channels=["A"],
                model_source=self.cfg.MODEL.SOURCE,
                ignore_index=self.cfg.LOSS.IGNORE_INDEX,
                ndim=self.dims,
            )
        ]

    def metric_calculation(
        self,
        output,
        targets,
        train: bool = True,
        metric_logger=None,
    ) -> Dict:
        """
        Compute the metrics defined in ``define_metrics``: calls each ``multiple_metrics``-style
        callable and merges its returned ``{name: value}`` dict.
        ``Image_to_Image_Workflow.metric_calculation`` is not reused here: it branches on a fixed
        set of image-quality metric names and would silently skip an unrecognized metric like ours.

        Parameters
        ----------
        output : NDArray or torch.Tensor
            Model prediction (raw logits during training; torchmetrics' binary ``JaccardIndex``
            auto-applies sigmoid to out-of-``[0,1]`` values, so no explicit activation is needed
            here -- see the ``ce_sigmoid`` convention in ``define_activations_and_channels``).

        targets : NDArray or torch.Tensor
            Ground truth affinities.

        train : bool, optional
            Whether to use ``self.train_metrics`` or ``self.test_metrics``.

        metric_logger : MetricLogger, optional
            Updated in place with each metric's value, if given.

        Returns
        -------
        out_metrics : dict
            Value of each computed metric.
        """
        device = self.device if train else self.test_device
        if isinstance(output, np.ndarray):
            _output = to_pytorch_format(output.copy(), self.axes_order, device, dtype=self.loss_dtype)
        else:
            _output = output if train else output.clone()

        if isinstance(targets, np.ndarray):
            _targets = to_pytorch_format(targets.copy(), self.axes_order, device, dtype=self.loss_dtype)
        else:
            _targets = targets if train else targets.clone()

        # Exclude the border region from the metric computation (see 'TEST.EVAL_BORDER_CROP').
        # Train-time patches are small already and never carry this crop.
        if not train and self.cfg.TEST.EVAL_BORDER_CROP:
            border = list(self.cfg.TEST.EVAL_BORDER_CROP)
            _output = crop_border_tensor(_output, border)
            _targets = crop_border_tensor(_targets, border)

        out_metrics = {}
        list_to_use = self.train_metrics if train else self.test_metrics
        with torch.no_grad():
            for metric in list_to_use:
                val = metric(_output, _targets)
                for m_name, m_val in val.items():
                    if isinstance(m_val, torch.Tensor):
                        v = m_val.item() if not torch.isnan(m_val) else 0
                    else:
                        v = m_val
                    out_metrics[m_name] = v
                    if metric_logger:
                        metric_logger.meters[m_name].update(v)
        return out_metrics

    def prepare_bmz_data(self, img: NDArray):
        """
        Expand ``img``'s raw ``SOURCE_CHANNELS`` width to the model's full input width before
        ``Base_Workflow.prepare_bmz_data``'s BMZ-export bookkeeping calls the model directly,
        bypassing ``process_test_sample`` (and its own X-derivation) entirely.

        Parameters
        ----------
        img : 4D/5D Numpy array
            Image in torch-format axes already, i.e. ``(b, c, y, x)`` for 2D or
            ``(b, c, z, y, x)`` for 3D, at the raw ``SOURCE_CHANNELS`` width.
        """
        mr = self.cfg.PROBLEM.IMAGE_TO_IMAGE.MEMBRANE_REPAIR
        channel_last = np.moveaxis(img, 1, -1)  # (b, c, ...) -> (b, ..., c)
        expanded = np.stack(
            [
                derive_membrane_input_channels(
                    sample,
                    list(mr.SOURCE_CHANNELS),
                    list(mr.DERIVED_CHANNELS),
                    dict(mr.DERIVED_CHANNELS_EXTRA_OPTS[0]),
                    ndim=self.dims,
                    resolution=self.resolution,
                )
                for sample in channel_last
            ]
        )
        expanded = np.moveaxis(expanded, -1, 1)  # (b, ..., c) -> (b, c, ...)
        super().prepare_bmz_data(expanded)

    def process_test_sample(self):
        """
        Expand X to the model's input width, then delegate crop/predict/merge to
        ``Image_to_Image_Workflow.process_test_sample`` with ``DATA.PATCH_SIZE`` and the
        undo-normalization step both temporarily adjusted for the duration of that call.

        ``test_pair_data_generator`` loads only the raw ``SOURCE_CHANNELS``-width X, so the loaded
        image is expanded here before any patch cropping, and ``DATA.PATCH_SIZE`` is temporarily
        widened to match for the duration of the ``super()`` call (same scoped-swap as
        ``prepare_model``).

        The undo-normalization step is neutralized with a temporary identity norm dict
        (``max_val_to_div=1, min_val_to_div=0, orig_dtype="float32"``), reducing
        ``undo_norm_range01`` to a no-op clip on our already-``[0,1]`` sigmoid output.

        If ``TEST.REUSE_PREDICTIONS`` is set, the model call is skipped entirely and the merged
        raw prediction is instead read back from ``PATHS.RESULT_DIR.PER_IMAGE`` (where a prior run
        saved it, since ``Image_to_Image_Workflow.process_test_sample`` always writes it there --
        ``TEST.SAVE_MODEL_RAW_OUTPUT`` defaults to ``True``), under the same filename convention
        ``save_tif`` uses. ``Image_to_Image_Workflow.process_test_sample`` never checks this flag
        itself (unlike ``Base_Workflow.process_test_sample``), so it is handled locally here.

        Finally converts the merged prediction into instances and evaluates them against GT (see
        ``_postprocess_and_evaluate_membrane_repair``). Since
        ``Image_to_Image_Workflow.process_test_sample`` never calls the generic
        ``after_merge_patches`` hook, the merged prediction is retrieved via a temporarily-forced
        ``self.return_prediction=True``, restored immediately after.
        """
        assert self.model
        if "discard" in self.current_sample and self.current_sample["discard"]:
            return super().process_test_sample()

        cfg = self.cfg
        original_return_prediction = self.return_prediction

        if cfg.TEST.REUSE_PREDICTIONS:
            result = None
            test_file = os.path.join(
                cfg.PATHS.RESULT_DIR.PER_IMAGE,
                os.path.splitext(self.current_sample["X_filename"])[0] + ".tif",
            )
            pred = read_img_as_ndarray(test_file, is_3d=(cfg.PROBLEM.NDIM == "3D"))
            pred = np.expand_dims(pred, 0)
            if pred.dtype == np.dtype("uint16"):
                pred = pred.astype(np.float32)
            self._predictions.append({"role": "raw", "data": np.array(pred)})

            if self.current_sample["Y"] is not None:
                if self.current_sample["Y"].dtype == np.dtype("uint16"):
                    self.current_sample["Y"] = self.current_sample["Y"].astype(np.float32)
                metric_values = self.metric_calculation(output=pred, targets=self.current_sample["Y"], train=False)
                for metric in metric_values:
                    key = str(metric).lower()
                    if key not in self.stats["merge_patches"]:
                        self.stats["merge_patches"][key] = 0
                    self.stats["merge_patches"][key] += metric_values[metric]
                    self.current_sample_metrics[key] = metric_values[metric]
        else:
            mr = cfg.PROBLEM.IMAGE_TO_IMAGE.MEMBRANE_REPAIR
            x = self.current_sample["X"]
            expanded = derive_membrane_input_channels(
                x[0],
                list(mr.SOURCE_CHANNELS),
                list(mr.DERIVED_CHANNELS),
                dict(mr.DERIVED_CHANNELS_EXTRA_OPTS[0]),
                ndim=self.dims,
                resolution=self.resolution,
            )
            self.current_sample["X"] = np.expand_dims(expanded, 0)

            raw_patch = tuple(cfg.DATA.PATCH_SIZE)
            expanded_patch = raw_patch[:-1] + (len(mr.SOURCE_CHANNELS) + len(mr.DERIVED_CHANNELS),)
            was_frozen = cfg.is_frozen()
            cfg.defrost()
            cfg.DATA.PATCH_SIZE = expanded_patch

            original_x_norm = self.current_sample["X_norm"]
            n_out = cfg.PROBLEM.IMAGE_TO_IMAGE.OUTPUT_CHANNELS
            self.current_sample["X_norm"] = {
                "type": "div",
                "orig_dtype": "float32",
                "per_channel_info": {str(c): {"max_val_to_div": 1, "min_val_to_div": 0} for c in range(n_out)},
            }
            self.return_prediction = True
            try:
                result = super().process_test_sample()
            finally:
                self.current_sample["X_norm"] = original_x_norm
                cfg.DATA.PATCH_SIZE = raw_patch
                if was_frozen:
                    cfg.freeze()
                self.return_prediction = original_return_prediction

        if self._predictions:
            pred = self._predictions[-1]["data"]
            if not original_return_prediction:
                self._predictions.pop()
            self._postprocess_and_evaluate_membrane_repair(pred)
        return result

    def _matching_stats_and_report(self, _Y: NDArray, pred_labels: NDArray, suffix: str = "") -> Tuple[Dict, ...]:
        """
        Compute per-threshold matching stats between `_Y` and `pred_labels`, save the GT-association/FP
        CSVs and (for the configured thresholds) the TP/FP/FN color map, then return the stats.

        Shared between the raw-prediction evaluation and the post-processing (instance refinement)
        evaluation in ``_postprocess_and_evaluate_membrane_repair`` -- `suffix` (``""`` or
        ``"_post-proc"``) is what tells their output files apart.

        Parameters
        ----------
        _Y : NDArray
            Ground truth instance label image.
        pred_labels : NDArray
            Predicted instance label image, same shape as `_Y`.
        suffix : str, optional
            Inserted right before ``_th_{thr}`` in every saved filename, e.g. ``"_post-proc"``.

        Returns
        -------
        tuple of dict
            One ``matching()`` stats dict per threshold in ``cfg.TEST.MATCHING_STATS_THS``.
        """
        cfg = self.cfg
        log_tag = " (post-processing)" if suffix else ""
        print(f"Calculating matching stats{log_tag} . . .")
        results = matching(_Y, pred_labels, thresh=cfg.TEST.MATCHING_STATS_THS, report_matches=True)

        diff_ths_colored_img = abs(len(cfg.TEST.MATCHING_STATS_THS_COLORED_IMG) - len(cfg.TEST.MATCHING_STATS_THS))
        colored_img_ths = cfg.TEST.MATCHING_STATS_THS_COLORED_IMG + [-1] * diff_ths_colored_img

        for i, r_stats in enumerate(results):
            thr = r_stats["thresh"]
            want_colored_img = colored_img_ths[i] != -1 and colored_img_ths[i] == thr
            if want_colored_img:
                print("Creating the image with a summary of TPs (green), FPs (blue) and FNs (red) . . .")
            colored_result, df_gt_assoc, df_fp = build_tp_fp_fn_report(
                gt_labels=_Y,
                pred_labels=pred_labels,
                r_stats=r_stats,
                build_color_map=want_colored_img,
            )

            if self.save_to_disk:
                os.makedirs(cfg.PATHS.RESULT_DIR.INST_ASSOC_POINTS, exist_ok=True)
                df_gt_assoc.to_csv(
                    os.path.join(
                        cfg.PATHS.RESULT_DIR.INST_ASSOC_POINTS,
                        os.path.splitext(self.current_sample["X_filename"])[0]
                        + f"{suffix}_th_{thr}_gt_assoc.csv",
                    ),
                    index=False,
                )
                df_fp.to_csv(
                    os.path.join(
                        cfg.PATHS.RESULT_DIR.INST_ASSOC_POINTS,
                        os.path.splitext(self.current_sample["X_filename"])[0] + f"{suffix}_th_{thr}_fp.csv",
                    ),
                    index=False,
                )
                if want_colored_img:
                    save_tif(
                        np.expand_dims(colored_result, 0),
                        cfg.PATHS.RESULT_DIR.INST_ASSOC_POINTS,
                        [os.path.splitext(self.current_sample["X_filename"])[0] + f"{suffix}_th_{thr}.tif"],
                        verbose=cfg.TEST.VERBOSE,
                    )
            del colored_result, df_gt_assoc, df_fp

            for key in ["matched_scores", "matched_tps", "matched_pairs", "pred_ids", "gt_ids"]:
                r_stats.pop(key, None)
            print(f"DatasetMatching{log_tag}: {r_stats}")

        return results

    def _postprocess_and_evaluate_membrane_repair(self, pred: NDArray):
        """
        Convert predicted affinities into instance labels, save them, and (if GT is available)
        evaluate them against the raw GT instance labels.

        Method selected by ``PROBLEM.IMAGE_TO_IMAGE.MEMBRANE_REPAIR.POSTPROCESS.METHOD``:
        ``"watershed"`` (single min-affinity watershed, see ``watershed_by_channels``) or
        ``"agglomeration"`` (default, see ``biapy/data/post_processing/affinity_agglomeration.py``).

        If ``TEST.EVAL_BORDER_CROP`` is set, the saved/agglomerated instance labels are left
        untouched, but a border-cropped copy of both the predicted labels and the GT is used for
        matching, so only the center of the image is evaluated (see ``TEST.EVAL_BORDER_CROP``'s
        docstring in config.py).

        Parameters
        ----------
        pred : NDArray
            Merged, sigmoid-activated affinity prediction, ``(1, ..., C)`` or ``(..., C)``.
        """
        cfg = self.cfg
        mr = cfg.PROBLEM.IMAGE_TO_IMAGE.MEMBRANE_REPAIR
        pp = mr.POSTPROCESS
        if pred.shape[0] == 1:
            pred = pred[0]

        # 'pred_labels'/'_Y' are always (z, y, x) below (a leading z=1 axis is inserted for 2D),
        # while 'TEST.EVAL_BORDER_CROP' follows PROBLEM.NDIM's convention ([y, x] for 2D) -- pad
        # it with a leading 0 so indices line up.
        border_crop = list(cfg.TEST.EVAL_BORDER_CROP)
        eval_border_crop = (([0] + border_crop) if cfg.PROBLEM.NDIM == "2D" else border_crop) if border_crop else []

        if pp.METHOD == "agglomeration":
            a_opts = dict(mr.DATA_CHANNELS_EXTRA_OPTS[0]).get("A", {})
            offsets = affinity_offsets_from_opts(a_opts, ndim=self.dims)
            pred_labels = watershed_and_agglomerate_affinities(
                data=pred,
                offsets=offsets,
                fragment_seed_th=pp.FRAGMENT_SEED_TH,
                fragment_growth_th=pp.FRAGMENT_GROWTH_TH,
                merge_threshold=pp.MERGE_TH,
                merge_quantile=pp.MERGE_QUANTILE,
                min_edge_voxels=pp.MIN_EDGE_VOXELS,
                resolution=self.resolution,
                watershed_by_2d_slices=False,
                verbose=cfg.TEST.VERBOSE,
            ).astype(np.uint32)
        else:
            pred_labels = watershed_by_channels(
                data=pred,
                channels=["A"],
                seed_channels=["A"],
                seed_channel_ths=["auto"],
                topo_surface_channel="",
                growth_mask_channels=["A"],
                growth_mask_channel_ths=["auto"],
                resolution=self.resolution,
                watershed_by_2d_slices=False,
                save_dir=None,
                verbose=cfg.TEST.VERBOSE,
            ).astype(np.uint32)
        if pred_labels.ndim == 2:
            pred_labels = np.expand_dims(pred_labels, 0)

        if self.save_to_disk:
            save_tif(
                np.expand_dims(np.expand_dims(pred_labels, -1), 0),
                cfg.PATHS.RESULT_DIR.PER_IMAGE_INSTANCES,
                [self.current_sample["X_filename"]],
                verbose=cfg.TEST.VERBOSE,
            )

        matching_enabled = (
            cfg.TEST.MATCHING_STATS
            and (cfg.DATA.TEST.LOAD_GT or cfg.DATA.TEST.USE_VAL_AS_TEST)
            and self.original_test_mask_path is not None
        )

        _Y = None
        if matching_enabled:
            test_file = os.path.join(self.original_test_mask_path, self.current_sample["X_filename"])
            if not os.path.exists(test_file):
                print(
                    "WARNING: The image seems to have a different name than its GT mask file. Using "
                    "the mask file that's in the same position (within the mask files list) as the "
                    "image is in its own list. Check if it is correct!"
                )
                test_file = os.path.join(self.original_test_mask_path, self.test_gt_filenames[self.f_numbers[0]])

            _Y = read_img_as_ndarray(test_file, is_3d=(cfg.PROBLEM.NDIM == "3D")).squeeze()
            if _Y.ndim == 2:
                _Y = np.expand_dims(_Y, 0)
            if _Y.dtype == np.float32:
                _Y = _Y.astype(np.uint32)
            elif _Y.dtype == np.float64:
                _Y = _Y.astype(np.uint64)

            if pred_labels.shape != _Y.shape:
                pred_labels = resize(pred_labels, _Y.shape, order=0)

            eval_Y, eval_pred_labels = _Y, pred_labels
            if eval_border_crop:
                eval_Y = crop_border_numpy(_Y, eval_border_crop, has_channel_axis=False)
                eval_pred_labels = crop_border_numpy(pred_labels, eval_border_crop, has_channel_axis=False)

            results = self._matching_stats_and_report(eval_Y, eval_pred_labels, suffix="")
            self.all_matching_stats_merge_patches.append(results)
            for i, r_per_th in enumerate(results):
                prefix = str(r_per_th["thresh"]) + " TH " if "thresh" in r_per_th else f"TH{i} "
                for mkey in [
                    "fp", "tp", "fn", "precision", "recall", "accuracy", "f1", "n_true", "n_pred",
                    "mean_true_score", "mean_matched_score", "panoptic_quality",
                ]:
                    if mkey in r_per_th:
                        self.current_sample_metrics[prefix + mkey] = r_per_th[mkey]

        ###################
        # Post-processing #
        ###################
        if cfg.TEST.POST_PROCESSING.INSTANCE_REFINEMENT.ENABLE:
            pred_labels = apply_label_refinement(
                pred_labels,
                is_3d=cfg.PROBLEM.NDIM == "3D",
                operations=cfg.TEST.POST_PROCESSING.INSTANCE_REFINEMENT.OPERATIONS,
                values=cfg.TEST.POST_PROCESSING.INSTANCE_REFINEMENT.VALUES,
            )

            if self.save_to_disk:
                save_tif(
                    np.expand_dims(np.expand_dims(pred_labels, -1), 0),
                    cfg.PATHS.RESULT_DIR.PER_IMAGE_POST_PROCESSING,
                    [self.current_sample["X_filename"]],
                    verbose=cfg.TEST.VERBOSE,
                )

            if matching_enabled:
                eval_pred_labels_post = pred_labels
                if eval_border_crop:
                    eval_pred_labels_post = crop_border_numpy(pred_labels, eval_border_crop, has_channel_axis=False)
                results_post_proc = self._matching_stats_and_report(eval_Y, eval_pred_labels_post, suffix="_post-proc")
                self.all_matching_stats_merge_patches_post.append(results_post_proc)
                for i, r_per_th in enumerate(results_post_proc):
                    prefix = str(r_per_th["thresh"]) + " TH (post)" if "thresh" in r_per_th else f"TH{i} (post) "
                    for mkey in [
                        "fp", "tp", "fn", "precision", "recall", "accuracy", "f1", "n_true", "n_pred",
                        "mean_true_score", "mean_matched_score", "panoptic_quality",
                    ]:
                        if mkey in r_per_th:
                            self.current_sample_metrics[prefix + mkey] = r_per_th[mkey]

    def normalize_stats(self, image_counter):
        """
        Aggregate the accumulated per-image matching results into final dataset-level stats.

        Parameters
        ----------
        image_counter : int
            Number of images (unused directly here; ``wrapper_matching_dataset_lazy`` recomputes
            the aggregate from the raw per-image ``matching()`` results, not an average of
            per-image scalars).
        """
        super().normalize_stats(image_counter)
        if (self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST) and self.cfg.TEST.MATCHING_STATS:
            if len(self.all_matching_stats_merge_patches) > 0:
                self.stats["inst_stats_merge_patches"] = wrapper_matching_dataset_lazy(
                    self.all_matching_stats_merge_patches,
                    self.cfg.TEST.MATCHING_STATS_THS,
                    by_image=self.cfg.TEST.MATCHING_STATS_BY_IMAGE,
                )
            if len(self.all_matching_stats_merge_patches_post) > 0:
                self.stats["inst_stats_merge_patches_post"] = wrapper_matching_dataset_lazy(
                    self.all_matching_stats_merge_patches_post,
                    self.cfg.TEST.MATCHING_STATS_THS,
                    by_image=self.cfg.TEST.MATCHING_STATS_BY_IMAGE,
                )

    def print_stats(self, image_counter):
        """
        Print the aggregated instance-matching stats.

        Parameters
        ----------
        image_counter : int
            Number of images to call ``normalize_stats`` with.
        """
        super().print_stats(image_counter)
        if self.cfg.TEST.MATCHING_STATS and (self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST):
            print("Membrane-repair instance metrics (affinities -> watershed -> matching):")
            for i in range(len(self.cfg.TEST.MATCHING_STATS_THS)):
                print("IoU TH={}".format(self.cfg.TEST.MATCHING_STATS_THS[i]))
                if self.stats["inst_stats_merge_patches"]:
                    print(f"      {self.stats['inst_stats_merge_patches'][i]}")
                if self.cfg.TEST.POST_PROCESSING.INSTANCE_REFINEMENT.ENABLE:
                    print("IoU (post-processing) TH={}".format(self.cfg.TEST.MATCHING_STATS_THS[i]))
                    if self.stats["inst_stats_merge_patches_post"]:
                        print(f"      {self.stats['inst_stats_merge_patches_post'][i]}")
