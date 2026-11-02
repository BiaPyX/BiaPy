"""
Tracking workflow: instance segmentation of each frame of a 2D time-lapse, then linking of the instances across frames.

Selected with ``PROBLEM.TYPE = "INSTANCE_SEG"`` and ``PROBLEM.TRACKING.ENABLE``. ``DATA.TEST.PATH`` is one sequence
(one image per frame). The instances come from the model, as in instance segmentation, or from
``PROBLEM.TRACKING.MASKS_PATH``, and are linked by ``PROBLEM.TRACKING.LINKER_MODEL``.
"""

import os

import numpy as np
import tifffile
from skimage.util import map_array

from biapy.data.data_manipulation import read_img_as_ndarray
from biapy.data.post_processing.tracking import (
    association_scores,
    greedy_linking,
    links_to_ctc,
    normalize_movie,
    object_features,
)
from biapy.engine.instance_seg import Instance_Segmentation_Workflow as _Instance_Segmentation_Workflow
from biapy.models.trackastra import load_trackastra
from biapy.utils.misc import os_walk_clean


class Tracking_Workflow(_Instance_Segmentation_Workflow):
    """
    Instance segmentation workflow whose test phase also links the instances of the frames into tracks, saved in the
    Cell Tracking Challenge result format in ``PATHS.RESULT_DIR.TRACKING``.
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
        self.frame_instances[filenames[0]] = self.last_instance_labels.squeeze()
        return results

    def after_all_images(self):
        """Link the instances of all the frames."""
        super().after_all_images()
        self.link_frames()

    def _read_sequence(self, folder: str) -> np.ndarray:
        return np.stack(
            [read_img_as_ndarray(os.path.join(folder, f), is_3d=False).squeeze() for f in next(os_walk_clean(folder))[2]]
        )

    def link_frames(self):
        """Link the instances of the frames of ``DATA.TEST.PATH`` and save the tracks."""
        print("Linking the instances across frames . . .")
        frame_names = next(os_walk_clean(self.cfg.DATA.TEST.PATH))[2]
        frames = self._read_sequence(self.cfg.DATA.TEST.PATH)
        if self.cfg.PROBLEM.TRACKING.MASKS_PATH:
            masks = self._read_sequence(self.cfg.PROBLEM.TRACKING.MASKS_PATH)
        else:
            missing = [f for f in frame_names if f not in self.frame_instances]
            if missing:
                raise ValueError(f"No instances were created for these frames: {missing}")
            masks = np.stack([self.frame_instances[f] for f in frame_names])
        if frames.ndim != 3 or masks.shape != frames.shape:
            raise ValueError(
                f"Tracking needs single-channel 2D frames and one mask per frame of the same size. Frames: "
                f"{frames.shape}, masks: {masks.shape}"
            )

        timepoints, labels, coords, features = object_features(masks, normalize_movie(frames))
        linker = load_trackastra(self.cfg.PROBLEM.TRACKING.LINKER_MODEL, self.device)
        scores = association_scores(linker, timepoints, coords, features, self.device)
        links = greedy_linking(scores, timepoints, coords, self.cfg.PROBLEM.TRACKING.ALLOW_DIVISIONS)
        tracks, label_maps = links_to_ctc(links, timepoints, labels, len(frames))

        out_dir = self.cfg.PATHS.RESULT_DIR.TRACKING
        os.makedirs(out_dir, exist_ok=True)
        dtype = np.uint16 if len(tracks) < 2**16 else np.uint32
        digits = max(3, len(str(len(frames) - 1)))
        for t, (mask, label_map) in enumerate(zip(masks, label_maps)):
            tracked = np.zeros(mask.shape, dtype=dtype)
            if label_map:
                map_array(mask, np.array(list(label_map)), np.array(list(label_map.values()), dtype=dtype), out=tracked)
            tifffile.imwrite(os.path.join(out_dir, f"mask{t:0{digits}d}.tif"), tracked, compression="zlib")
        np.savetxt(os.path.join(out_dir, "res_track.txt"), tracks, fmt="%d")

        print(f"Number of instances: {len(timepoints)}")
        print(f"Number of tracks: {len(tracks)}")
        print(f"Number of divisions: {int(np.count_nonzero(tracks[:, 3])) // 2}")
        print(f"Tracks saved in {out_dir}")
