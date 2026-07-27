"""
Equivariance round-trip tests for the representation-aware TTA.

How to run
----------
No data, no weights and no GPU are needed: everything is synthetic and runs on CPU in a few seconds.
From the repository root, with the BiaPy environment active (``conda activate BiaPy_env``)::

    pytest tests/test_tta_equivariance.py -q                 # whole suite (~10 s, 33 tests)
    pytest tests/test_tta_equivariance.py -v                 # one line per test, useful to see names
    pytest tests/test_tta_equivariance.py -k flow -v         # only the flow/Cellpose/Omnipose ones
    pytest tests/test_tta_equivariance.py::test_hovernet_maps_round_trip    # a single test
    pytest tests/test_tta_equivariance.py -x -q              # stop at the first failure

Without activating the environment, call its interpreter directly::

    ~/miniforge3/envs/BiaPy_env/bin/python -m pytest tests/test_tta_equivariance.py -q

Useful ``-k`` selectors: ``flow`` (Cellpose/Omnipose), ``affinit``, ``stardist``, ``hovernet``,
``embedseg``, ``degrade`` (orientations dropped when a representation is not equivariant).

The idea needs no trained model. A perfectly accurate model for an instance representation *is*
:func:`labels_into_channels`, so a fake ``pred_func`` that runs the real target generation on
whatever image it is handed behaves exactly like an oracle network. Feeding such an oracle through
:func:`ensemble_predictions` must return the representation of the un-augmented image.

That makes the tests compare the TTA channel remap against the code that *defines* each
representation rather than against a docstring: a wrong flow sign, a ray grid off by one index, a
swapped HoVer axis or a mis-rolled affinity all fail here.

Two representations are only *approximately* equivariant at the source, for reasons that have
nothing to do with TTA:

- Cellpose/Omnipose flows are normalized per cell by their maximum magnitude, and the diffusion seed
  pixel is picked with integer arithmetic that is not flip-symmetric. At the handful of pixels right
  at a cell centre the gradient is ~0, so the normalized direction there is degenerate and can point
  anywhere. Everything else matches to float precision.
- HoVerNet builds its coordinate grid as ``np.arange(1, extent + 1) - round(centroid)`` (verbatim
  upstream HoVer-Net), a one-based grid against a rounded centre. Mirroring the labels shifts that
  convention by two units, so the normalized maps differ by roughly ``2 / cell_extent``.

For those two, the tests assert on a robust error *and* on the margin over not remapping at all,
which stays enormous; :func:`test_hovernet_remap_is_exact_on_an_equivariant_field` additionally
pins the HoVer remap itself against an analytically equivariant field, where it is exact.
"""

import numpy as np
import pytest
import torch

from biapy.data.post_processing.post_processing import ensemble_predictions
from biapy.data.post_processing.tta import (
    AxisTransform,
    build_axis_transform_group,
    build_tta_spec,
    parse_model_output_channel_names,
)
from biapy.data.pre_processing import affinity_channel_names, labels_into_channels

DEVICE = torch.device("cpu")
AXES_ORDER = {2: (0, 3, 1, 2), 3: (0, 4, 1, 2, 3)}
AXES_ORDER_BACK = {2: (0, 2, 3, 1), 3: (0, 2, 3, 4, 1)}


# --------------------------------------------------------------------------------------------- #
# Synthetic label images                                                                          #
# --------------------------------------------------------------------------------------------- #
def _blob_labels_2d(shape=(64, 64), n=6, seed=0):
    """Build a 2D instance-label image of well-separated discs."""
    rng = np.random.default_rng(seed)
    lab = np.zeros(shape, np.int32)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    placed, ident = [], 1
    while len(placed) < n:
        cy = int(rng.integers(10, shape[0] - 10))
        cx = int(rng.integers(10, shape[1] - 10))
        r = int(rng.integers(4, 7))
        if any((cy - py) ** 2 + (cx - px) ** 2 < (r + pr + 4) ** 2 for py, px, pr in placed):
            continue
        lab[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = ident
        placed.append((cy, cx, r))
        ident += 1
    return lab


def _blob_labels_3d(shape=(20, 36, 36), n=4, seed=0):
    """Build a 3D instance-label volume of well-separated balls."""
    rng = np.random.default_rng(seed)
    lab = np.zeros(shape, np.int32)
    zz, yy, xx = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    placed, ident = [], 1
    while len(placed) < n:
        cz = int(rng.integers(6, shape[0] - 6))
        cy = int(rng.integers(8, shape[1] - 8))
        cx = int(rng.integers(8, shape[2] - 8))
        r = int(rng.integers(3, 5))
        if any((cz - pz) ** 2 + (cy - py) ** 2 + (cx - px) ** 2 < (r + pr + 4) ** 2 for pz, py, px, pr in placed):
            continue
        lab[(zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = ident
        placed.append((cz, cy, cx, r))
        ident += 1
    return lab


# --------------------------------------------------------------------------------------------- #
# Oracle plumbing                                                                                 #
# --------------------------------------------------------------------------------------------- #
def _gt_channel_names(data_channels, channel_extra_opts):
    """Physical channel names of the array :func:`labels_into_channels` produces, in order."""
    names = []
    for ch in data_channels:
        if ch in ("E_sigma", "E_seediness"):
            continue
        if ch == "R":
            names += ["R_{}".format(j) for j in range(int(channel_extra_opts["R"]["nrays"]))]
        elif ch == "A":
            names += affinity_channel_names(channel_extra_opts["A"])
        else:
            names.append(ch)
    return names


def _generate(labels, data_channels, channel_extra_opts, out_names):
    """Run the real target generation and keep only the model's output channels, in output order."""
    gt_names = _gt_channel_names(data_channels, channel_extra_opts)
    take = [gt_names.index(nm) for nm in out_names]
    gt = labels_into_channels(
        np.ascontiguousarray(labels)[..., None].astype(np.int32),
        mode=list(data_channels),
        channel_extra_opts=channel_extra_opts,
    )
    return gt[..., take].astype(np.float32)


def _oracle(data_channels, channel_extra_opts, out_names):
    """A ``pred_func`` returning the exact representation of whatever image it is handed."""

    def pred_func(batch):
        outs = [
            _generate(np.rint(batch[b, ..., 0]).astype(np.int32), data_channels, channel_extra_opts, out_names)
            for b in range(batch.shape[0])
        ]
        arr = np.stack(outs, 0)  # (B, spatial..., C)
        ndim = arr.ndim - 2
        order = (0, ndim + 1) + tuple(range(1, ndim + 1))  # -> (B, C, spatial...)
        return torch.from_numpy(np.ascontiguousarray(np.transpose(arr, order)))

    return pred_func


def _run_tta(labels, data_channels, channel_extra_opts, out_names, ndim, group="full", mode="mean"):
    """Run the oracle through the real TTA and return ``(tta_output, reference)``."""
    spec = build_tta_spec(out_names, ndim=ndim, channel_extra_opts=channel_extra_opts)
    out = ensemble_predictions(
        labels.astype(np.float32)[..., None],
        pred_func=_oracle(data_channels, channel_extra_opts, out_names),
        axes_order_back=AXES_ORDER_BACK[ndim],
        axes_order=AXES_ORDER[ndim],
        device=DEVICE,
        ndim=ndim,
        batch_size_value=4,
        mode=mode,
        tta_spec=spec,
        group=group,
    )
    arr = out["pred"] if isinstance(out, dict) else out
    arr = np.moveaxis(arr.numpy(), 1, -1)[0]  # (1, C, spatial...) -> (spatial..., C)
    return arr, _generate(labels, data_channels, channel_extra_opts, out_names)


def _remap_errors(labels, data_channels, channel_extra_opts, out_names, ndim):
    """
    Per-orientation error of the channel remap, with and without it.

    Returns ``(worst_mean_with_remap, worst_mean_without_remap, worst_outlier_count)`` measured over
    the foreground, where "without" means undoing only the pixels and leaving the channel values in
    the augmented frame -- the mistake this whole module exists to avoid.
    """
    spec = build_tta_spec(out_names, ndim=ndim, channel_extra_opts=channel_extra_opts)
    ref = _generate(labels, data_channels, channel_extra_opts, out_names)
    fg = labels > 0
    with_remap = without_remap = 0.0
    outliers = 0
    for t in spec.filter_orientations(build_axis_transform_group(ndim, "full"))[0]:
        aug = np.rint(t.apply(labels[..., None].astype(np.float32))[..., 0]).astype(np.int32)
        base = t.inverse.apply(_generate(aug, data_channels, channel_extra_opts, out_names)).copy()
        remapped = base.copy()
        spec.remap_channels(remapped, t)
        with_remap = max(with_remap, float(np.abs(remapped - ref)[fg].mean()))
        without_remap = max(without_remap, float(np.abs(base - ref)[fg].mean()))
        outliers = max(outliers, int(((np.abs(remapped - ref) > 0.05) & fg[..., None]).sum()))
    return with_remap, without_remap, outliers


# --------------------------------------------------------------------------------------------- #
# AxisTransform algebra                                                                           #
# --------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("ndim", [2, 3])
def test_transform_inverse_round_trips(ndim):
    """Applying a transform then its inverse must be the identity, spatially and on vectors."""
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(5, 6, 7)[:ndim] + (3,))
    vecs = rng.normal(size=(20, ndim))
    for t in build_axis_transform_group(ndim, "full"):
        assert np.array_equal(t.inverse.apply(t.apply(arr)), arr), t.describe()
        assert np.allclose(t.inverse.transform_vectors(t.transform_vectors(vecs)), vecs), t.describe()


def test_group_sizes():
    """The orientation sets must have the classic sizes: 8/16 for full, 4/8 for flips."""
    assert len(build_axis_transform_group(2, "full")) == 8
    assert len(build_axis_transform_group(2, "flips")) == 4
    assert len(build_axis_transform_group(3, "full")) == 16
    assert len(build_axis_transform_group(3, "flips")) == 8
    assert len(build_axis_transform_group(3, "none")) == 1
    # Z is never permuted onto Y/X: no 3D orientation may move axis 0.
    assert all(t.perm[0] == 0 for t in build_axis_transform_group(3, "full"))
    # The identity is always first, so orientation 0 is the plain prediction.
    assert build_axis_transform_group(3, "full")[0].is_identity


def test_rot90_matches_numpy():
    """``AxisTransform((1, 0), (-1, 1))`` must reproduce ``np.rot90`` exactly."""
    arr = np.random.default_rng(0).normal(size=(8, 8, 2))
    assert np.array_equal(AxisTransform((1, 0), (-1, 1)).apply(arr), np.rot90(arr, 1, axes=(0, 1)))


# --------------------------------------------------------------------------------------------- #
# Representations whose targets are exactly equivariant                                           #
# --------------------------------------------------------------------------------------------- #
def test_scalar_channels_round_trip_exactly():
    """Binary/contour channels are invariant, so the oracle must come back untouched."""
    labels = _blob_labels_2d()
    out, ref = _run_tta(labels, ["B", "C"], {}, ["B", "C"], ndim=2)
    assert np.allclose(out, ref, atol=1e-5)


def test_normalized_distance_round_trips_exactly():
    """The per-instance normalized distance 'D' is a scalar field and must be untouched."""
    labels = _blob_labels_2d()
    out, ref = _run_tta(labels, ["B", "D"], {}, ["B", "D"], ndim=2)
    assert np.allclose(out, ref, atol=1e-5)


def test_stardist_rays_round_trip_exactly():
    """StarDist rays permute cyclically; with nrays=32 all 8 orientations are exact."""
    labels = _blob_labels_2d()
    opts = {"R": {"nrays": 32}}
    names = ["B"] + ["R_{}".format(j) for j in range(32)]
    spec = build_tta_spec(names, ndim=2, channel_extra_opts=opts)
    assert len(spec.filter_orientations(build_axis_transform_group(2, "full"))[0]) == 8

    out, ref = _run_tta(labels, ["B", "R"], opts, names, ndim=2)
    assert np.allclose(out, ref, atol=1e-4)

    # And the permutation is doing real work: without it the rays would be badly wrong.
    _, without, _ = _remap_errors(labels, ["B", "R"], opts, names, ndim=2)
    assert without > 0.1


def test_affinities_round_trip_exactly_3d():
    """Affinities need a spatial roll when an axis is reflected; the roll must be exact."""
    labels = _blob_labels_3d()
    opts = {"A": {"z_affinities": [1], "y_affinities": [1], "x_affinities": [1], "widen_borders": 0}}
    names = ["Az_1", "Ay_1", "Ax_1"]
    with_remap, without_remap, _ = _remap_errors(labels, ["A"], opts, names, ndim=3)
    assert with_remap == 0.0
    assert without_remap > 0.1

    out, ref = _run_tta(labels, ["A"], opts, names, ndim=3)
    # The roll cannot recover the single border slice the reversed offset falls outside of -- the
    # same slice seg2aff_pni itself fills by broadcasting -- so compare the interior.
    inner = (slice(1, -1),) * 3
    assert np.allclose(out[inner], ref[inner], atol=1e-4)


def test_affinity_channel_order_is_interleaved_by_offset():
    """
    The affinity names must be one ``(z, y, x)`` triple per offset index, not grouped by axis.

    This is the order :func:`labels_into_channels` writes and the order the affinity watershed
    assumes when it takes ``data[..., [0, 1, 2]]``, so the declared model output-channel metadata
    has to agree with it.
    """
    assert affinity_channel_names({}) == ["Az_1", "Ay_1", "Ax_1"]
    assert affinity_channel_names(
        {"z_affinities": [1, 2], "y_affinities": [1, 3], "x_affinities": [1, 3]}
    ) == ["Az_1", "Ay_1", "Ax_1", "Az_2", "Ay_3", "Ax_3"]
    with pytest.raises(ValueError):
        affinity_channel_names({"z_affinities": [1, 2], "y_affinities": [1], "x_affinities": [1]})


def test_affinity_channel_names_match_the_generated_content():
    """
    Each declared affinity name must sit on the column that actually holds that (axis, offset) map.

    Checks the naming against :func:`seg2aff_pni` directly, so it pins the *content* of the layout
    rather than just its self-consistency.
    """
    from biapy.utils.util import seg2aff_pni

    labels = _blob_labels_3d()
    opts = {"z_affinities": [1, 2], "y_affinities": [1, 2], "x_affinities": [1, 2], "widen_borders": 0}
    names = affinity_channel_names(opts)
    gt = labels_into_channels(labels[..., None].astype(np.int32), mode=["A"], channel_extra_opts={"A": opts})

    axis_row = {"z": 0, "y": 1, "x": 2}
    for col, name in enumerate(names):
        axis, off = name[1], int(name.split("_")[1])
        kwargs = {"dz": 1, "dy": 1, "dx": 1}
        kwargs["d" + axis] = off
        expected = seg2aff_pni(labels, **kwargs)[axis_row[axis]]
        assert np.array_equal(gt[..., col], expected), "{} is not on column {}".format(name, col)


def test_multi_offset_affinities_round_trip_exactly_3d():
    """Several offsets per axis must remap correctly, which needs the layouts to agree."""
    labels = _blob_labels_3d()
    opts = {"z_affinities": [1, 2], "y_affinities": [1, 2], "x_affinities": [1, 2], "widen_borders": 0}
    names = affinity_channel_names(opts)
    assert len(names) == 6
    with_remap, without_remap, _ = _remap_errors(labels, ["A"], {"A": opts}, names, ndim=3)
    assert with_remap == 0.0
    assert without_remap > 0.1


def test_affinities_with_mismatched_offsets_degrade():
    """Swapping Y and X needs an affinity with the same offset on the other axis."""
    spec = build_tta_spec(["Az_1", "Ay_1", "Ax_2"], ndim=3)
    kept, reasons = spec.filter_orientations(build_axis_transform_group(3, "full"))
    assert len(kept) == 8 and all(not t.permutes_axes for t in kept)
    assert reasons


# --------------------------------------------------------------------------------------------- #
# Representations whose targets are only approximately equivariant                                #
# --------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "tag, data_channels, opts, names, ndim",
    [
        ("cellpose 2D", ["B", "Gv", "Gh"], {}, ["B", "Gv", "Gh"], 2),
        ("cellpose 3D", ["B", "Gz", "Gv", "Gh"], {}, ["B", "Gz", "Gv", "Gh"], 3),
        ("omnipose 2D", ["B", "Db", "Gv", "Gh"], {"Db": {"val_type": "omnipose"}}, ["B", "Db", "Gv", "Gh"], 2),
    ],
)
def test_flow_fields_round_trip(tag, data_channels, opts, names, ndim):
    """
    Flow vectors must be sign-corrected on every reflected axis and swapped on every rotation.

    The residual comes from the cell-centre pixels, where the flow is normalized by a ~0 gradient
    and its direction is therefore degenerate; it is a property of the flow *construction*, not of
    the remap (see the module docstring).
    """
    labels = _blob_labels_2d() if ndim == 2 else _blob_labels_3d()
    with_remap, without_remap, outliers = _remap_errors(labels, data_channels, opts, names, ndim)
    n_fg = int((labels > 0).sum())
    assert with_remap < 0.02, tag
    assert without_remap > 20 * with_remap, tag
    assert outliers < 0.05 * n_fg, tag

    out, ref = _run_tta(labels, data_channels, opts, names, ndim=ndim)
    assert float(np.abs(out - ref)[labels > 0].mean()) < 0.02, tag


def test_hovernet_remap_is_exact_on_an_equivariant_field():
    """
    Pin the HoVer remap itself: on an exactly equivariant HoVer-like field it must be perfect.

    Built independently of the TTA code as ``coord - centroid`` normalized per instance and per sign
    half, i.e. what HoVerNet's maps are up to its one-based grid convention.
    """
    labels = _blob_labels_2d()

    def hover(lab):
        out = np.zeros(lab.shape + (2,), np.float32)
        yy, xx = np.mgrid[0 : lab.shape[0], 0 : lab.shape[1]]
        for i in np.unique(lab):
            if i == 0:
                continue
            m = lab == i
            for c, grid in ((0, yy), (1, xx)):
                v = (grid[m] - grid[m].mean()).astype(np.float32)
                if (v < 0).any():
                    v[v < 0] /= -v[v < 0].min()
                if (v > 0).any():
                    v[v > 0] /= v[v > 0].max()
                out[..., c][m] = v
        return out

    spec = build_tta_spec(["V", "H"], ndim=2)
    ref, fg = hover(labels), labels > 0
    for t in build_axis_transform_group(2, "full"):
        aug = np.rint(t.apply(labels[..., None].astype(np.float32))[..., 0]).astype(np.int32)
        restored = t.inverse.apply(hover(aug)).copy()
        spec.remap_channels(restored, t)
        assert np.allclose(restored[fg], ref[fg], atol=1e-5), t.describe()


def test_hovernet_maps_round_trip():
    """
    HoVerNet's V/H maps are a vector field and must be swapped/negated like one.

    The residual is HoVer-Net's own one-based coordinate grid against a rounded centroid, which is
    not mirror-symmetric (see the module docstring); the remap still beats leaving the channels
    alone by a wide margin.
    """
    labels = _blob_labels_2d()
    with_remap, without_remap, _ = _remap_errors(labels, ["B", "V", "H"], {}, ["B", "V", "H"], ndim=2)
    assert with_remap < 0.25
    assert without_remap > 3 * with_remap


# --------------------------------------------------------------------------------------------- #
# EmbedSeg                                                                                        #
# --------------------------------------------------------------------------------------------- #
def test_embedseg_offsets_and_sigmas_remap():
    """EmbedSeg offsets are signed vectors and sigmas per-axis magnitudes: both must remap."""
    ndim = 2
    names = ["E_offset_0", "E_offset_1", "E_sigma_0", "E_sigma_1", "E_seediness"]
    spec = build_tta_spec(names, ndim=ndim)
    canonical = np.random.default_rng(3).normal(size=(24, 24, len(names))).astype(np.float32)
    canonical[..., 2:4] = np.abs(canonical[..., 2:4])  # sigmas are magnitudes

    for t in build_axis_transform_group(ndim, "full"):
        # Emulate an equivariant model: the prediction on the augmented image is the canonical one
        # moved spatially, with its channels re-expressed in the augmented frame. EmbedSeg stores
        # its components in Cartesian (x, y) order, hence the ``[::-1]`` round trip.
        augmented = t.apply(canonical).copy()
        for lo, hi, signed in ((0, 2, True), (2, 4, False)):
            comps = augmented[..., lo:hi][..., ::-1]  # (x, y) -> spatial-axis order (y, x)
            moved = t.transform_vectors(comps)
            augmented[..., lo:hi] = (moved if signed else np.abs(moved))[..., ::-1]
        restored = t.inverse.apply(augmented).copy()
        spec.remap_channels(restored, t)
        assert np.allclose(restored, canonical, atol=1e-5), t.describe()


def test_embedseg_anisotropy_drops_only_the_rotations():
    """EmbedSeg coordinates carry the voxel spacing, so unequal in-plane scales block the swaps."""
    names = ["E_offset_0", "E_offset_1", "E_sigma_0", "E_sigma_1", "E_seediness"]

    spec = build_tta_spec(names, ndim=3, anisotropy=(5.0, 1.0, 1.0))
    assert len(spec.filter_orientations(build_axis_transform_group(3, "full"))[0]) == 16

    spec = build_tta_spec(names, ndim=3, anisotropy=(1.0, 1.0, 2.0))
    kept, reasons = spec.filter_orientations(build_axis_transform_group(3, "full"))
    assert len(kept) == 8 and all(not t.permutes_axes for t in kept)
    assert reasons


# --------------------------------------------------------------------------------------------- #
# Degradation                                                                                     #
# --------------------------------------------------------------------------------------------- #
def test_stardist_rays_not_multiple_of_four_degrade_to_flips():
    """With nrays=30 the 90 degree rotations are not exact, so only the flips survive."""
    names = ["B"] + ["R_{}".format(j) for j in range(30)]
    spec = build_tta_spec(names, ndim=2, channel_extra_opts={"R": {"nrays": 30}})
    kept, reasons = spec.filter_orientations(build_axis_transform_group(2, "full"))
    assert len(kept) == 4 and all(not t.permutes_axes for t in kept)
    assert reasons


def test_stardist_rays_3d_degrade_to_identity():
    """3D rays lie on a Fibonacci sphere, which no orientation maps onto itself."""
    names = ["B"] + ["R_{}".format(j) for j in range(96)]
    spec = build_tta_spec(names, ndim=3, channel_extra_opts={"R": {"nrays": 96}})
    kept, reasons = spec.filter_orientations(build_axis_transform_group(3, "full"))
    assert [t.is_identity for t in kept] == [True]
    assert reasons


def test_2d_flows_keep_all_orientations():
    """Nothing about 2D flows blocks an orientation, so the full group must survive."""
    spec = build_tta_spec(["B", "Gv", "Gh"], ndim=2)
    assert len(spec.filter_orientations(build_axis_transform_group(2, "full"))[0]) == 8


def test_3d_flows_without_a_z_component_keep_all_orientations():
    """A missing Gz only matters if Z were permuted onto Y/X, which never happens."""
    spec = build_tta_spec(["B", "Gv", "Gh"], ndim=3)
    assert len(spec.filter_orientations(build_axis_transform_group(3, "full"))[0]) == 16


# --------------------------------------------------------------------------------------------- #
# Reduction mode                                                                                  #
# --------------------------------------------------------------------------------------------- #
def test_min_max_skips_signed_vector_channels():
    """``min``/``max`` must not touch flow/offset channels, but must still apply to the others."""
    assert build_tta_spec(["B", "Gv", "Gh"], ndim=2).mode_reducible_channels == [0]
    assert build_tta_spec(["B", "C"], ndim=2).mode_reducible_channels == [0, 1]
    assert build_tta_spec(["B", "V", "H"], ndim=2).mode_reducible_channels == [0]
    # E_sigma is a magnitude per axis and E_seediness a probability, so min/max stays meaningful.
    names = ["E_offset_0", "E_offset_1", "E_sigma_0", "E_sigma_1", "E_seediness"]
    assert build_tta_spec(names, ndim=2).mode_reducible_channels == [2, 3, 4]


def test_flows_are_averaged_even_under_min_mode():
    """With mode='min' the flow channels must still be averaged, so they still round-trip."""
    labels = _blob_labels_2d()
    out_min, ref = _run_tta(labels, ["B", "Gv", "Gh"], {}, ["B", "Gv", "Gh"], ndim=2, mode="min")
    out_mean, _ = _run_tta(labels, ["B", "Gv", "Gh"], {}, ["B", "Gv", "Gh"], ndim=2, mode="mean")
    assert np.array_equal(out_min[..., 1:], out_mean[..., 1:])  # flows: untouched by the mode
    assert float(np.abs(out_min - ref)[labels > 0].mean()) < 0.02


# --------------------------------------------------------------------------------------------- #
# Plumbing                                                                                        #
# --------------------------------------------------------------------------------------------- #
def test_parse_model_output_channel_names_drops_class_head():
    """The classification head is a separate model output and must not enter the spec."""
    assert parse_model_output_channel_names(["Gv+Gh+B", "class"]) == ["Gv", "Gh", "B"]


def test_spec_covers_every_channel_once():
    """Every declared output channel must land in exactly one group."""
    names = ["B", "Db_bin0", "Db_bin1", "Gv", "Gh", "R_0", "R_1", "R_2", "R_3"]
    spec = build_tta_spec(names, ndim=2, channel_extra_opts={"R": {"nrays": 4}})
    assert sorted(c for g in spec.groups for c in g.channels) == list(range(len(names)))
    assert spec.n_channels == len(names)


def test_non_square_input_is_padded_and_cropped():
    """A non-square input must come back at its original shape."""
    labels = _blob_labels_2d((48, 64))
    out, ref = _run_tta(labels, ["B", "Gv", "Gh"], {}, ["B", "Gv", "Gh"], ndim=2)
    assert out.shape == ref.shape == (48, 64, 3)


def test_scalar_workflow_without_a_spec():
    """No spec at all (semantic seg, denoising, SSL) must still work and average correctly."""
    img = np.random.default_rng(1).normal(size=(32, 32, 1)).astype(np.float32)

    def pred_func(batch):
        return torch.from_numpy(np.ascontiguousarray(np.transpose(batch.astype(np.float32), (0, 3, 1, 2))))

    out = ensemble_predictions(
        img,
        pred_func=pred_func,
        axes_order_back=AXES_ORDER_BACK[2],
        axes_order=AXES_ORDER[2],
        device=DEVICE,
        ndim=2,
        batch_size_value=8,
    )
    # An identity model ensembled over the full group returns the input unchanged.
    assert np.allclose(np.moveaxis(out.numpy(), 1, -1)[0], img, atol=1e-5)


def test_extra_model_outputs_are_ensembled():
    """A second spatial head (e.g. 'class') must be un-augmented and averaged too, not dropped."""
    img = np.random.default_rng(4).normal(size=(24, 24, 1)).astype(np.float32)

    def pred_func(batch):
        t = torch.from_numpy(np.ascontiguousarray(np.transpose(batch.astype(np.float32), (0, 3, 1, 2))))
        return {"pred": t, "class": t * 2.0}

    out = ensemble_predictions(
        img,
        pred_func=pred_func,
        axes_order_back=AXES_ORDER_BACK[2],
        axes_order=AXES_ORDER[2],
        device=DEVICE,
        ndim=2,
        batch_size_value=4,
    )
    assert isinstance(out, dict) and set(out) == {"pred", "class"}
    assert np.allclose(np.moveaxis(out["pred"].numpy(), 1, -1)[0], img, atol=1e-5)
    assert np.allclose(np.moveaxis(out["class"].numpy(), 1, -1)[0], img * 2.0, atol=1e-5)


@pytest.mark.parametrize("group, expected", [("none", 1), ("flips", 4), ("full", 8)])
def test_group_selects_the_number_of_forward_passes(group, expected):
    """``TEST.AUGMENTATION_GROUP`` must control how many times the model is called."""
    calls = []

    def pred_func(batch):
        calls.append(batch.shape[0])
        return torch.from_numpy(np.ascontiguousarray(np.transpose(batch.astype(np.float32), (0, 3, 1, 2))))

    ensemble_predictions(
        np.random.default_rng(2).normal(size=(16, 16, 1)).astype(np.float32),
        pred_func=pred_func,
        axes_order_back=AXES_ORDER_BACK[2],
        axes_order=AXES_ORDER[2],
        device=DEVICE,
        ndim=2,
        group=group,
    )
    assert sum(calls) == expected
