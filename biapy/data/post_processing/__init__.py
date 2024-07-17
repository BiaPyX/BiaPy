import numpy as np

from biapy.engine.metrics import jaccard_index_numpy, voc_calculation
from biapy.data.post_processing.post_processing import apply_median_filtering


def apply_post_processing(cfg, data, Y=None):
    """
    Create training and validation generators.

    Parameters
    ----------
    cfg : YACS CN object
        Configuration.

    data : 4D/5D Numpy array
        Data to apply post_proccessing. E.g. ``(num_of_images, y, x, channels)`` for 2D and
        ``(num_of_images, z, y, x, channels)`` for 3D.

    Y : 4D/5D Numpy array, optional
        Data GT to calculate the metrics. E.g. ``(num_of_images, y, x, channels)`` for 2D and
        ``(num_of_images, z, y, x, channels)`` for 3D.

    Returns
    -------
    iou_post : float
        Foreground IoU of ``data`` compared with ``Y`` after post-processing.

    ov_iou_post : float
        Overall IoU of ``data`` compared with ``Y`` after post-processing.
    """

    print("Applying post-processing . . .")

    for f, val in zip(
        cfg.TEST.POST_PROCESSING.MEDIAN_FILTER_AXIS,
        cfg.TEST.POST_PROCESSING.MEDIAN_FILTER_SIZE,
    ):
        data = apply_median_filtering(data, axes=f, mf_size=val)

    if Y is not None:
        iou_post = jaccard_index_numpy(
            (Y > 0.5).astype(np.uint8), (data > 0.5).astype(np.uint8)
        )
        ov_iou_post = voc_calculation(
            (Y > 0.5).astype(np.uint8), (data > 0.5).astype(np.uint8), iou_post
        )
    else:
        iou_post, ov_iou_post = 0, 0

    return data, iou_post, ov_iou_post
