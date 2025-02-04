from biapy.data.post_processing.post_processing import apply_median_filtering


def apply_post_processing(cfg, data):
    """
    Create training and validation generators.

    Parameters
    ----------
    cfg : YACS CN object
        Configuration.

    data : 4D/5D Numpy array
        Data to apply post_proccessing. E.g. ``(num_of_images, y, x, channels)`` for 2D and
        ``(num_of_images, z, y, x, channels)`` for 3D.

    Returns
    -------
    data : 4D/5D Numpy array
        Data to apply post_proccessing. E.g. ``(num_of_images, y, x, channels)`` for 2D and
        ``(num_of_images, z, y, x, channels)`` for 3D.
    """

    print("Applying post-processing . . .")

    for f, val in zip(
        cfg.TEST.POST_PROCESSING.MEDIAN_FILTER_AXIS,
        cfg.TEST.POST_PROCESSING.MEDIAN_FILTER_SIZE,
    ):
        data = apply_median_filtering(data, axes=f, mf_size=val)

    return data
