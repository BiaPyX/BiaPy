"""
SKOOTS post-processing: instances from the 'F', 'P' and 'H'/'V'(/'Z') (type 'skeleton') predictions.

Adapted from `SKOOTS <https://github.com/indzhykulianlab/skoots>`_ (``skoots/lib/eval.py``).
"""
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import grey_dilation, label
from typing import Sequence


def skoots_instances(
    foreground: NDArray,
    skeleton: NDArray,
    offsets: NDArray,
    scale: Sequence[float],
    fg_th: float = 0.5,
    skeleton_th: float = 0.8,
    n_steps: int = 2,
) -> NDArray:
    """
    Label the skeleton connected components and give each voxel the label its offsets point to.

    Parameters
    ----------
    foreground : 2D/3D Numpy array
        'F' probability, e.g. ``(z, y, x)``.

    skeleton : 2D/3D Numpy array
        'P' probability, same shape.

    offsets : 3D/4D Numpy array
        'H'/'V'(/'Z') tanh outputs in axis order (``(z,) y, x``) as the last axis.

    scale : sequence of float
        Voxels per axis (``(z,) y, x``) of a tanh output of 1.

    fg_th : float, optional
        Foreground threshold.

    skeleton_th : float, optional
        Skeleton threshold, applied after the dilations.

    n_steps : int, optional
        Times each voxel is moved by the offsets.

    Returns
    -------
    instances : 2D/3D Numpy array
        Instance labels, same shape as ``foreground``.
    """
    ndim = foreground.ndim
    shape = foreground.shape
    fg = foreground > fg_th

    # One full and three in-plane dilations, as SKOOTS
    skel = grey_dilation(skeleton.astype(np.float32) * fg, size=(3,) * ndim)
    in_plane = (1, 3, 3) if ndim == 3 else (3, 3)
    for _ in range(3):
        skel = grey_dilation(skel, size=in_plane)
    skel_labels, _ = label(skel > skeleton_th)

    scale_arr = np.asarray(list(scale)[-ndim:], dtype=np.float32).reshape((ndim,) + (1,) * ndim)
    # Offsets store 'voxel - skeleton', so move against them
    vec = -np.moveaxis(offsets.astype(np.float32), -1, 0) * scale_arr * fg

    def _index(emb: NDArray):
        return tuple(np.clip(np.round(emb[a]), 0, shape[a] - 1).astype(np.intp) for a in range(ndim))

    emb = np.indices(shape, dtype=np.float32) + vec
    for _ in range(n_steps - 1):
        emb += vec[(slice(None),) + _index(emb)]
    return skel_labels[_index(emb)]
