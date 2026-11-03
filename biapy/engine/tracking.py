"""
Tracking workflow: instance segmentation of each frame of a 2D/3D time-lapse, then linking of the instances across
frames.

Selected with ``PROBLEM.TYPE = "INSTANCE_SEG"`` and ``PROBLEM.TRACKING.ENABLE``. ``DATA.TEST.PATH`` is one sequence
(one image per frame). The instances come from the model, as in instance segmentation, or from
``PROBLEM.TRACKING.MASKS_PATH``, and are linked by ``PROBLEM.TRACKING.LINKER_MODEL``.
"""

import os

import numpy as np

from biapy.data.post_processing.tracking import (
    association_scores,
    greedy_linking,
    links_to_ctc,
    normalize_movie,
    object_features,
)
from biapy.data.tracks_io import read_ctc_tracks, read_frames, write_ctc_tracks, write_geff
from biapy.engine.instance_seg import Instance_Segmentation_Workflow as _Instance_Segmentation_Workflow
from biapy.models.trackastra import load_trackastra
from biapy.utils.misc import os_walk_clean
from biapy.engine.metrics import tracking_metrics


class Tracking_Workflow(_Instance_Segmentation_Workflow):
    """
    Instance segmentation workflow whose test phase also links the instances of the frames into tracks, saved in
    ``PATHS.RESULT_DIR.TRACKING`` in the Cell Tracking Challenge result format and as GEFF (``tracks.geff``).
    """

    def __init__(self, cfg, job_identifier, device, system_dict, args, **kwargs):
        super().__init__(cfg, job_identifier, device, system_dict, args, **kwargs)
        self.frame_instances = {}

    def test(self, image=None, gt=None):
        """Segment the frames (unless ``PROBLEM.TRACKING.MASKS_PATH`` is set) and link their instances."""
        if self.cfg.PROBLEM.TRACKING.MASKS_PATH:
            self.link_frames()
        else:
            super().test(image=image, gt=gt)

    def instance_seg_process(self, pred, filenames, *args, **kwargs):
        """Instance segmentation of a frame, keeping its instances for the linking."""
        results = super().instance_seg_process(pred, filenames, *args, **kwargs)
        labels = self.last_instance_labels
        self.frame_instances[filenames[0]] = labels.reshape(labels.shape[-self.dims :])
        return results

    def after_all_images(self):
        """Link the instances of all the frames."""
        super().after_all_images()
        self.link_frames()

    def link_frames(self):
        """Link the instances of the frames of ``DATA.TEST.PATH``, save the tracks and evaluate them."""
        print("Linking the instances across frames . . .")
        cfg = self.cfg.PROBLEM.TRACKING
        frames = read_frames(self.cfg.DATA.TEST.PATH, self.dims == 3)
        if cfg.MASKS_PATH:
            masks = read_frames(cfg.MASKS_PATH, self.dims == 3)
        else:
            names = next(os_walk_clean(self.cfg.DATA.TEST.PATH))[2]
            missing = [f for f in names if f not in self.frame_instances]
            if missing:
                raise ValueError(f"No instances were created for these frames: {missing}")
            masks = np.stack([self.frame_instances[f] for f in names])
        if masks.shape != frames.shape:
            raise ValueError(f"Frames {frames.shape} and their masks {masks.shape} must have the same shape")

        linker = load_trackastra(cfg.LINKER_MODEL, self.device)
        # 2D frames are one-slice volumes for 3D linkers
        expand = linker.coord_dim == 3 and self.dims == 2
        f, m = (frames[:, None], masks[:, None]) if expand else (frames, masks)
        timepoints, labels, coords, features = object_features(m, normalize_movie(f))
        scores = association_scores(linker, timepoints, coords, features, self.device)
        coords = coords[:, -self.dims :]
        links = greedy_linking(scores, timepoints, coords, cfg.ALLOW_DIVISIONS)
        tracks, label_maps = links_to_ctc(links, timepoints, labels, len(frames))

        out_dir = self.cfg.PATHS.RESULT_DIR.TRACKING
        tracked = write_ctc_tracks(out_dir, masks, tracks, label_maps)
        nodes = dict.fromkeys(n for link in links for n in link)
        write_geff(
            os.path.join(out_dir, "tracks.geff"),
            links,
            timepoints,
            coords,
            {n: label_maps[timepoints[n]][labels[n]] for n in nodes},
        )
        print(f"Number of instances: {len(timepoints)}")
        print(f"Number of tracks: {len(tracks)}")
        print(f"Number of divisions: {int(np.count_nonzero(tracks[:, 3])) // 2}")
        print(f"Tracks saved in {out_dir} (CTC format and tracks.geff)")

        if cfg.GT_PATH:
            gt_masks, gt_tracks = read_ctc_tracks(cfg.GT_PATH, self.dims == 3)
            if gt_masks.shape != tracked.shape:
                raise ValueError(f"Ground truth {gt_masks.shape} and tracked masks {tracked.shape} differ in shape")
            metrics = tracking_metrics(gt_masks, gt_tracks, tracked, tracks)
            print("Tracking metrics (against 'PROBLEM.TRACKING.GT_PATH'):")
            for k in ("DET", "TRA", "link_precision", "link_recall", "link_f1", "division_precision",
                      "division_recall", "division_f1"):
                print(f"  Tracking {k}: {metrics[k]}")
            print("  AOGM errors: " + ", ".join(f"{k}={metrics[k]}" for k in ("NS", "FN", "FP", "ED", "EA", "EC")))
