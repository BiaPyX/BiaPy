"""
Cellpose/Omnipose test-phase helpers for the instance-segmentation workflow.

This module isolates the Cellpose test-time input-rescale logic (resolving the per-image diameter —
optionally with a cheap first-pass "double inference" estimate — and rescaling the image before the
network) so the main instance-segmentation workflow (:mod:`biapy.engine.instance_seg`) stays
manageable. It follows Cellpose's default ``resample=True`` strategy: rescale the input by
``diam_mean/diameter``, run the network, then the workflow resizes the flows back to native and runs
the dynamics there.

The functionality is exposed as a mixin (:class:`CellposeTestPhaseMixin`) that the workflow inherits,
so the methods run with the workflow instance as ``self`` and rely on the following attributes
provided by ``Base_Workflow`` / the instance-segmentation workflow:

- ``self.cfg``: the BiaPy configuration.
- ``self.current_sample``: dict of the test sample currently being processed.
- ``self.dims``: ``2`` or ``3``.
- ``self.axes_order_back``: axis order used by :func:`to_numpy_format`.
- ``self.model_call_func``: runs the model on a batch/patch.
- ``self.cellpose_diameter``: training-set median diameter (used as prior/fallback), or ``None``.

The mixin only reads/writes ``self.current_sample`` and never overrides workflow methods, so it can
be combined with the workflow without name clashes. The prediction is rescaled back to the native
resolution downstream in ``Instance_Segmentation_Workflow._create_instance_labels`` using the
``cellpose_orig_shape`` / ``cellpose_resized_shape`` keys set here.
"""
import torch
import numpy as np
from typing import Tuple
from numpy.typing import NDArray
from skimage.transform import resize

from biapy.utils.misc import is_main_process, to_numpy_format
from biapy.data.data_manipulation import pad_and_reflect
from biapy.data.post_processing.gradient_tracking import _estimate_cell_radius


class CellposeTestPhaseMixin:
    """Cellpose double-inference auto-diameter logic for the instance-segmentation test phase."""

    def _cellpose_test_rescale_active(self) -> bool:
        """
        Whether the Cellpose test-time input rescale should run for the current test sample.

        Active for a non-torchvision flow (Gv/Gh/Gz) workflow when a (non-discarded) sample is loaded
        and it is not the by-chunks path. The actual rescale factor is decided later by
        :meth:`_resolve_cellpose_test_diameter` (which may return ``None`` ⇒ run at native resolution).
        """
        c = self.cfg.PROBLEM.INSTANCE_SEG
        return (
            self.cfg.MODEL.SOURCE != "torchvision"
            and not self.cfg.TEST.BY_CHUNKS.ENABLE
            and any(ch in c.DATA_CHANNELS for ch in ("Gv", "Gh", "Gz"))
            and isinstance(getattr(self, "current_sample", None), dict)
            and self.current_sample.get("X") is not None
            and not self.current_sample.get("discard", False)
        )

    def _resolve_cellpose_test_diameter(self):
        """
        Resolve the native cell diameter (pixels) used to rescale the current test image.

        Mirrors Cellpose's single ``rescale = diam_mean / diameter``. The diameter comes from, in order:

        - ``CELLPOSE.DIAMETER`` when it is > 0 (used directly, no first pass);
        - a cheap first-pass estimate (:meth:`_estimate_cellpose_diameter_first_pass`) when
          ``DIAMETER == 0`` and ``TEST_DOUBLE_INFERENCE`` is enabled — Cellpose's per-image size step;
        - the training-set median (``self.cellpose_diameter``) as the final fallback.

        Returns the diameter in pixels, or ``None`` if none is available (⇒ run at native resolution).
        """
        diam_cfg = self.cfg.PROBLEM.INSTANCE_SEG.CELLPOSE.DIAMETER
        if diam_cfg is not None and diam_cfg > 0:
            return float(diam_cfg)
        if self.cfg.PROBLEM.INSTANCE_SEG.CELLPOSE.TEST_DOUBLE_INFERENCE:
            try:
                est = self._estimate_cellpose_diameter_first_pass()
            except Exception as e:  # never let the estimate crash inference
                print("WARNING: Cellpose first-pass diameter estimate failed ({}); using fallback.".format(e))
                est = None
            if est and est > 0:
                return float(est)
        if self.cellpose_diameter and self.cellpose_diameter > 0:
            return float(self.cellpose_diameter)
        return None

    def _cellpose_native_view(self) -> NDArray:
        """Return the native (pre-reflect) content of ``current_sample["X"]`` as ``(1, [z,] y, x, c)``."""
        X = self.current_sample["X"]
        if self.cfg.DATA.REFLECT_TO_COMPLETE_SHAPE and "reflected_orig_shape" in self.current_sample:
            ros = self.current_sample["reflected_orig_shape"]  # ([z,] y, x, c)
            if self.dims == 2:
                return X[:, -ros[0]:, -ros[1]:, :]
            return X[:, -ros[0]:, -ros[1]:, -ros[2]:, :]
        return X

    def _cellpose_center_patch(self, x: NDArray, pshape) -> Tuple[NDArray, tuple]:
        """Centre-crop or reflect-pad ``x`` (``[z,] y, x, c``) to one patch; return patch + valid slices."""
        out = np.zeros(tuple(pshape) + (x.shape[-1],), dtype=x.dtype)
        src, dst = [], []
        for d in range(len(pshape)):
            a, p = x.shape[d], pshape[d]
            if a >= p:
                s = (a - p) // 2
                src.append(slice(s, s + p))
                dst.append(slice(0, p))
            else:
                o = (p - a) // 2
                src.append(slice(0, a))
                dst.append(slice(o, o + a))
        out[tuple(dst) + (slice(None),)] = x[tuple(src) + (slice(None),)]
        return out, tuple(dst) + (slice(None),)

    def _cellpose_foreground(self, pred: NDArray) -> NDArray:
        """Foreground mask of a raw flow prediction (F/M/B channel if present, else flow magnitude)."""
        ch = list(self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS)
        fg_c = next((c for c in ("F", "M", "B") if c in ch), "")
        thr = float(self.cfg.PROBLEM.INSTANCE_SEG.CELLPOSE.FG_THRESH)
        if fg_c:
            v = pred[..., ch.index(fg_c)]
            return (v < thr) if fg_c == "B" else (v > thr)
        mag = np.zeros(pred.shape[:-1], dtype=np.float32)
        for c in ("Gz", "Gv", "Gh"):
            if c in ch:
                mag = mag + pred[..., ch.index(c)] ** 2
        return np.sqrt(mag) > 1.0

    def _estimate_cellpose_diameter_first_pass(self):
        """
        Estimate the native cell diameter (pixels) of the current test image with one cheap prediction.

        The native image is rescaled toward the training scale (``DIAM_MEAN / training_median`` as a
        prior), a single central patch is run through the network, and the median object size of the
        predicted foreground is measured with :func:`_estimate_cell_radius`, then mapped back to the
        native resolution. Returns ``None`` if the estimate is unreliable.
        """
        X = self._cellpose_native_view()
        is_3d = self.dims == 3
        patch = self.cfg.DATA.PATCH_SIZE
        diam_mean = float(self.cfg.PROBLEM.INSTANCE_SEG.CELLPOSE.DIAM_MEAN)
        prior = self.cellpose_diameter
        r1 = (diam_mean / prior) if (prior and prior > 0) else 1.0

        x = np.ascontiguousarray(X[0]).astype(np.float32)  # ([z,] y, x, c)
        if abs(r1 - 1.0) > 1e-3:
            if is_3d:
                tgt = (x.shape[0], max(1, int(round(x.shape[1] * r1))), max(1, int(round(x.shape[2] * r1))), x.shape[-1])
            else:
                tgt = (max(1, int(round(x.shape[0] * r1))), max(1, int(round(x.shape[1] * r1))), x.shape[-1])
            x = resize(x, tgt, order=1, mode="reflect", clip=True, preserve_range=True,
                       anti_aliasing=(r1 < 1.0)).astype(np.float32)

        x_patch, valid = self._cellpose_center_patch(x, tuple(patch[:-1]))
        with torch.no_grad():
            p = self.model_call_func(np.expand_dims(x_patch, 0))
        if isinstance(p, list):
            p = p[0]
        if isinstance(p, dict):
            p = p["pred"]
        p = to_numpy_format(p, self.axes_order_back)[0]  # ([z,] y, x, c_out)
        p = p[valid]

        radius, _ = _estimate_cell_radius(self._cellpose_foreground(p), is_3d)
        if radius is None or radius <= 0:
            return None
        return 2.0 * float(radius) / r1

    def _apply_cellpose_test_rescale(self):
        """Resolve the per-image diameter and rescale ``current_sample`` in-plane for inference."""
        native_diam = self._resolve_cellpose_test_diameter()
        if not native_diam or native_diam <= 0:
            if is_main_process():
                print("[Cellpose test rescale] no diameter available; running at native resolution.")
            return

        diam_mean = float(self.cfg.PROBLEM.INSTANCE_SEG.CELLPOSE.DIAM_MEAN)
        factor = float(min(4.0, max(0.25, diam_mean / native_diam)))  # clamp against bad estimates
        if abs(factor - 1.0) <= 1e-3:
            return

        try:
            is_3d = self.dims == 3
            iy, ix = (1, 2) if is_3d else (0, 1)
            reflect = self.cfg.DATA.REFLECT_TO_COMPLETE_SHAPE and "reflected_orig_shape" in self.current_sample

            # Native (pre-reflect) content of X, recovered from the bottom-right of the reflected image.
            X_native = self._cellpose_native_view()
            native_shape = tuple(X_native.shape[1:])  # ([z,] y, x, c)

            def _rescale_native(arr_native, order):
                a = arr_native[0]
                tgt = list(a.shape)
                tgt[iy] = max(1, int(round(a.shape[iy] * factor)))
                tgt[ix] = max(1, int(round(a.shape[ix] * factor)))
                r = resize(a, tuple(tgt), order=order, mode="reflect", clip=True, preserve_range=True,
                           anti_aliasing=(order != 0 and factor < 1.0)).astype(arr_native.dtype)
                # Re-pad so the rescaled image is at least PATCH_SIZE again (down-scaling can shrink it
                # below one patch, which breaks the tiler). Content stays at the bottom-right.
                r = pad_and_reflect(r, self.cfg.DATA.PATCH_SIZE, verbose=False)
                return np.expand_dims(r, 0), tuple(r.shape)

            X_new, x_padded_shape = _rescale_native(X_native, 1)
            resized_shape = list(native_shape)  # rescaled-native spatial (the valid, un-padded region)
            resized_shape[iy] = max(1, int(round(native_shape[iy] * factor)))
            resized_shape[ix] = max(1, int(round(native_shape[ix] * factor)))
            resized_shape = tuple(resized_shape)

            self.current_sample["X"] = X_new
            if self.current_sample.get("Y") is not None:
                Y_native = self.current_sample["Y"]
                if reflect:
                    Y_native = (Y_native[:, -native_shape[0]:, -native_shape[1]:, :] if not is_3d
                                else Y_native[:, -native_shape[0]:, -native_shape[1]:, -native_shape[2]:, :])
                self.current_sample["Y"], _ = _rescale_native(Y_native, 0)

            self.current_sample["cellpose_orig_shape"] = native_shape        # -> prediction resized back to this
            self.current_sample["cellpose_resized_shape"] = resized_shape    # valid region inside the padded X
            self.current_sample["cellpose_rescale_factor"] = factor
            self.current_sample["cellpose_diameter"] = native_diam
            self.current_sample["cellpose_diam_mean"] = diam_mean

            # When the generator reflected, tell the base per-patch path to crop the merged prediction
            # to the rescaled-native (valid) region; _create_instance_labels then resizes it to native.
            if reflect:
                self.current_sample["reflected_orig_shape"] = resized_shape
        except Exception as e:
            print("WARNING: Cellpose double-inference rescale failed ({}); running at native resolution.".format(e))
            for k in ("cellpose_orig_shape", "cellpose_resized_shape", "cellpose_rescale_factor",
                      "cellpose_diameter", "cellpose_diam_mean"):
                self.current_sample.pop(k, None)
