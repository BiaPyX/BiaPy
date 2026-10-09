"""
nnU-Net-like planning of BiaPy's U-Net ("unet" architecture) from the dataset fingerprint.

Follows nnU-Net v2's ExperimentPlanner (https://github.com/MIC-DKFZ/nnUNet):

* The patch starts as large as nnU-Net's (256^3 voxels in 3D, 2048^2 pixels in 2D) with the aspect ratio of
  the voxel spacing, clipped to the median image shape.
* The network topology follows the spacing: an axis is only downsampled while its (current) spacing is less
  than twice the finest one and the feature maps are at least 8 voxels long along it, and its kernels are 1
  wide (e.g. 1x3x3) until downsampling the other axes brings their spacing within a factor of 2. So
  anisotropic stacks are first downsampled in XY only.
* While the network does not fit in the GPU memory with the minimum batch size (2), the patch is shrunk along
  the axis that is largest relative to the median image shape.
* The batch size is then the largest that fits, but not covering more than 5% of the dataset per step, and
  at least 2. So large patches are preferred over large batches (in 2D too: nnU-Net's reference batch size of
  12 only sets the scale of its memory estimate, not the batch the patch must allow).
* Feature maps start at 32 and double on every level, up to 320 (3D) or 512 (2D).

Y and X can be downsampled differently (``MODEL.YX_DOWN`` takes ``[y, x]`` pairs), as nnU-Net does with the
long axis of very elongated images. BiaPy differences taken into account: the Z kernels can only be 1 or 3
wide (``MODEL.ISOTROPY``), and BiaPy trains in FP32 on the GPU it has, so the memory is estimated with a model fitted to measurements of BiaPy's U-Net (see
estimate_vram_bytes()) instead of nnU-Net's reference values for its mixed-precision training on 8 GB, and
the batch size is rounded down (not to the nearest) so the estimate is not exceeded. BiaPy does not resample
the images to a common spacing either, so the median spacing and shape of the images are used as they are.
"""
import math

import numpy as np

BASE_FEATURES = 32
MAX_FEATURES = {2: 512, 3: 320}
MIN_FEATURE_MAP_SIZE = 4
INITIAL_PATCH_VOXELS = {2: 2048**2, 3: 256**3}
MIN_BATCH = 2
MAX_DATASET_COVERED = 0.05
CONVS_PER_LEVEL = 2
# Above this ratio between the coarsest and finest spacing nnU-Net treats 3D data as a stack of 2D slices
# for the spatial augmentations
ANISOTROPY_THRESHOLD = 3

# Memory model of a BiaPy U-Net training step in FP32, fitted to the peak memory PyTorch reserved with
# several patch sizes, depths and batch sizes (2D and 3D, AdamW, CE+Dice loss) on an RTX 2080 Ti
# (fits: 1.24 and 1.02 bytes per FP32 activation in 2D and 3D, rounded up so it errs on the safe side).
# See tests/calibrate_unet_memory.py
VRAM_BYTES_PER_ACTIVATION = {2: 4.0 * 1.3, 3: 4.0 * 1.1}
VRAM_BYTES_PER_PARAMETER = 16.0  # weights + gradients + 2 AdamW states
VRAM_FIXED_BYTES = 0.1 * 1024**3
# Part of the GPU memory not used for planning: PyTorch's CUDA context, cuDNN workspaces, fragmentation
VRAM_USABLE_FRACTION = 0.9
VRAM_SAFETY_BYTES = 0.75 * 1024**3


def get_topology(spacing, patch_size):
    """
    Downsampling and kernel per level, as nnU-Net's get_pool_and_conv_props().

    Parameters
    ----------
    spacing : sequence of float
        ``(z, y, x)`` or ``(y, x)`` voxel spacing (only the ratios matter).
    patch_size : sequence of int
        Patch size in the same order.

    Returns
    -------
    topology : dict
        * ``pools``: per downsampling, factor of each axis (``[[z, y, x], ...]``).
        * ``conv_kernels``: per level (one more than ``pools``), kernel size of each axis.
        * ``divisor``: what each patch axis needs to be divisible by.
        * ``patch_size``: ``patch_size`` rounded up to the divisor.
    """
    dim = len(spacing)
    cur_spacing = [float(s) for s in spacing]
    cur_size = [float(s) for s in patch_size]
    kernel = [1] * dim
    pools, conv_kernels = [], []
    n_pool = [0] * dim
    while True:
        valid = [i for i in range(dim) if cur_size[i] >= 2 * MIN_FEATURE_MAP_SIZE]
        if not valid:
            break
        min_spacing = min(cur_spacing[i] for i in valid)
        valid = [i for i in valid if cur_spacing[i] / min_spacing < 2]
        if len(valid) == 1 and cur_size[valid[0]] < 3 * MIN_FEATURE_MAP_SIZE:
            break
        if not valid:
            break
        for d in range(dim):
            if kernel[d] != 3 and cur_spacing[d] / min(cur_spacing) < 2:
                kernel[d] = 3
        pool = [1] * dim
        for v in valid:
            pool[v] = 2
            n_pool[v] += 1
            cur_spacing[v] *= 2
            cur_size[v] = math.ceil(cur_size[v] / 2)
        pools.append(pool)
        conv_kernels.append(list(kernel))
    conv_kernels.append([3] * dim)  # bottleneck
    divisor = [2**n for n in n_pool]
    padded = [int(math.ceil(p / d) * d) for p, d in zip(patch_size, divisor)]
    return {"pools": pools, "conv_kernels": conv_kernels, "divisor": divisor, "patch_size": padded}


def feature_maps(n_levels, ndim):
    return [min(BASE_FEATURES * 2**i, MAX_FEATURES[ndim]) for i in range(n_levels)]


def _level_sizes(patch_size, pools):
    sizes = [list(patch_size)]
    for p in pools:
        sizes.append([int(math.ceil(s / f)) for s, f in zip(sizes[-1], p)])
    return [int(np.prod(s)) for s in sizes]


def count_activations(patch_size, topology, in_channels, out_channels, fmaps, convs=CONVS_PER_LEVEL):
    """
    Elements of the tensors kept for the backward pass of BiaPy's U-Net for one sample. Each conv unit
    (conv + norm + activation) keeps 3 tensors, the decoder also keeps the concatenations and upsamplings.
    """
    sizes = _level_sizes(patch_size, topology["pools"])
    depth = len(fmaps) - 1
    n = in_channels * sizes[0]
    for i in range(depth):
        n += convs * 3 * fmaps[i] * sizes[i]  # encoder convs
        n += fmaps[i] * sizes[i + 1]  # max pooling
        n += 3 * fmaps[i] * sizes[i]  # transposed conv + norm + act
        n += 2 * fmaps[i] * sizes[i]  # concatenation
        n += convs * 3 * fmaps[i] * sizes[i]  # decoder convs
    n += convs * 3 * fmaps[-1] * sizes[-1]  # bottleneck
    n += 3 * out_channels * sizes[0]  # head, softmax and loss
    return n


def count_parameters(topology, in_channels, out_channels, fmaps, convs=CONVS_PER_LEVEL):
    """Parameters of BiaPy's U-Net (conv weights, biases and norm affine parameters)."""
    kernels = [int(np.prod(k)) for k in topology["conv_kernels"]]
    depth = len(fmaps) - 1
    n, prev = 0, in_channels
    for i in range(depth + 1):
        for c in range(convs):
            cin = prev if c == 0 else fmaps[i]
            n += cin * fmaps[i] * kernels[i] + 3 * fmaps[i]
        prev = fmaps[i]
    for i in range(depth - 1, -1, -1):
        n += fmaps[i + 1] * fmaps[i] * int(np.prod(topology["pools"][i])) + 3 * fmaps[i]
        for c in range(convs):
            cin = 2 * fmaps[i] if c == 0 else fmaps[i]
            n += cin * fmaps[i] * kernels[i] + 3 * fmaps[i]
    n += fmaps[0] * out_channels + out_channels
    return n


def estimate_vram_bytes(patch_size, topology, in_channels, out_channels, batch_size):
    """Estimated peak GPU memory of a training step."""
    ndim = len(patch_size)
    fmaps = feature_maps(len(topology["conv_kernels"]), ndim)
    act = count_activations(patch_size, topology, in_channels, out_channels, fmaps)
    params = count_parameters(topology, in_channels, out_channels, fmaps)
    return VRAM_BYTES_PER_ACTIVATION[ndim] * act * batch_size + VRAM_BYTES_PER_PARAMETER * params + VRAM_FIXED_BYTES


def initial_patch_size(spacing, median_shape):
    """nnU-Net's starting patch: INITIAL_PATCH_VOXELS with the aspect ratio of the spacing, clipped to the data."""
    ndim = len(spacing)
    tmp = 1 / np.array(spacing, dtype=float)
    tmp = tmp * (INITIAL_PATCH_VOXELS[ndim] / np.prod(tmp)) ** (1 / ndim)
    return [int(min(round(t), m)) for t, m in zip(tmp, median_shape)]


def plan_unet(spacing, median_shape, in_channels, out_channels, dataset_voxels, vram_bytes):
    """
    Plans patch size, topology and batch size.

    Parameters
    ----------
    spacing : sequence of float
        ``(z, y, x)`` or ``(y, x)`` voxel spacing.
    median_shape : sequence of int
        Median image shape, same order.
    in_channels, out_channels : int
        Channels of the images and of the output (classes).
    dataset_voxels : float
        Voxels of all the training images, to limit the batch size.
    vram_bytes : float
        GPU memory available.

    Returns
    -------
    plan : dict
    """
    ndim = len(spacing)
    budget = max(VRAM_USABLE_FRACTION * vram_bytes - VRAM_SAFETY_BYTES, 0.25 * vram_bytes)
    patch = initial_patch_size(spacing, median_shape)
    topo = get_topology(spacing, patch)
    patch = topo["patch_size"]

    def estimate(p, t, bs):
        return estimate_vram_bytes(p, t, in_channels, out_channels, bs)

    while estimate(patch, topo, MIN_BATCH) > budget:
        # Shrink the axis that is largest relative to the data (as nnU-Net)
        axis = int(np.argsort([p / m for p, m in zip(patch, median_shape)])[-1])
        tmp = list(patch)
        tmp[axis] -= topo["divisor"][axis]
        divisor = get_topology(spacing, tmp)["divisor"]
        patch = list(patch)
        patch[axis] -= divisor[axis]
        if patch[axis] < MIN_FEATURE_MAP_SIZE:
            patch[axis] = MIN_FEATURE_MAP_SIZE
            if all(p <= 2 * MIN_FEATURE_MAP_SIZE for p in patch):
                break
        topo = get_topology(spacing, patch)
        patch = topo["patch_size"]

    per_sample = estimate(patch, topo, 1) - estimate(patch, topo, 0)
    batch = int((budget - estimate(patch, topo, 0)) // per_sample) if per_sample > 0 else MIN_BATCH
    batch_5_percent = int(round(dataset_voxels * MAX_DATASET_COVERED / np.prod(patch, dtype=np.float64)))
    batch = max(min(batch, batch_5_percent), MIN_BATCH)

    n_levels = len(topo["conv_kernels"])
    return {
        "patch_size": [int(p) for p in patch],
        "batch_size": int(batch),
        "feature_maps": feature_maps(n_levels, ndim),
        "pools": topo["pools"],
        "conv_kernels": topo["conv_kernels"],
        "vram_estimate_gb": round(estimate(patch, topo, batch) / 1024**3, 2),
        "vram_budget_gb": round(budget / 1024**3, 2),
        "anisotropic": bool(max(spacing) / min(spacing) > ANISOTROPY_THRESHOLD),
    }


def plan_to_biapy(plan):
    """BiaPy MODEL.* variables of a plan from plan_unet()."""
    ndim = len(plan["patch_size"])
    n_levels = len(plan["feature_maps"])
    model = {
        "ARCHITECTURE": "unet",
        "FEATURE_MAPS": plan["feature_maps"],
        "CONV_LAYERS": [CONVS_PER_LEVEL] * n_levels,
        "KERNEL_SIZE": 3,
        "DROPOUT_VALUES": [0.0] * n_levels,
        "NORMALIZATION": "in",
        "ACTIVATION": "leaky_relu",
        "UPSAMPLE_LAYER": "convtranspose",
        # One number when Y and X are downsampled alike, [y, x] otherwise
        "YX_DOWN": [p[-1] if p[-2] == p[-1] else [p[-2], p[-1]] for p in plan["pools"]],
    }
    if ndim == 3:
        model["Z_DOWN"] = [p[0] for p in plan["pools"]]
        model["ISOTROPY"] = [k[0] == 3 for k in plan["conv_kernels"]]
    return model
