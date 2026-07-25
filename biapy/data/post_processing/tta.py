"""
Representation-aware test-time augmentation (TTA) primitives.

Test-time augmentation predicts the same image in several orientations and averages the results.
That is trivially correct while every predicted channel is a *scalar* field (foreground, contours,
per-instance-normalized distances...), because un-transforming the prediction spatially is enough.
It is **wrong** for the geometry-derived representations BiaPy supports -- Cellpose/Omnipose flows,
HoVerNet maps, StarDist rays, EmbedSeg offsets/sigmas, affinities -- whose *values* encode a
direction or an axis and therefore have to be remapped along with the pixels.

This module provides the two pieces needed to do that once, for every representation:

1. :class:`AxisTransform` -- the orientations used by TTA (90 degree rotations, flips and their
   compositions) are exactly the **signed axis permutations**: output axis ``a`` takes input axis
   ``perm[a]``, optionally reversed (``sign[a] == -1``). The very same ``(perm, sign)`` pair drives
   the spatial transform *and* the channel remap, so the two can never disagree.

2. :class:`TTASpec` -- a declarative description of what each physical output channel *is*
   (scalar, vector component, per-axis magnitude, ray, affinity), built from the model output
   channel names. It knows how to remap the channels for a given :class:`AxisTransform`, which
   orientations it supports at all, and which channels may be reduced with a ``min``/``max``
   ensemble mode instead of a plain mean.

Not every representation supports every orientation. StarDist rays are only exactly permutable when
the ray directions map onto each other (true for the 2D uniform-angle grid when ``nrays % 4 == 0``,
never true for the 3D Fibonacci sphere); EmbedSeg coordinates carry the voxel anisotropy, so
swapping Y and X is only valid on isotropic in-plane data. Rather than silently producing garbage,
:meth:`TTASpec.filter_orientations` drops the orientations a spec cannot represent exactly and
reports why.

See Also
--------
biapy.data.post_processing.post_processing.ensemble_predictions : the single TTA entry point.
"""

import re
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "AxisTransform",
    "ChannelGroup",
    "ScalarChannels",
    "VectorChannels",
    "RayChannels",
    "AffinityChannels",
    "TTASpec",
    "build_axis_transform_group",
    "parse_model_output_channel_names",
    "build_tta_spec",
]

#: Valid values of ``TEST.AUGMENTATION_GROUP``.
TTA_GROUPS = ("auto", "full", "flips", "none")


# --------------------------------------------------------------------------------------------- #
# The group elements                                                                              #
# --------------------------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AxisTransform:
    """
    A signed axis permutation: the general form of a 90 degree rotation / flip / their compositions.

    Arrays are handled in ``(spatial..., channels)`` layout, so the spatial axes are
    ``0 .. ndim-1`` (``(y, x)`` in 2D, ``(z, y, x)`` in 3D).

    The transform maps input coordinates ``u`` to output coordinates ``v`` by
    ``v[a] = sign[a] * u[perm[a]]`` (in centred coordinates), i.e. **output axis ``a`` comes from
    input axis ``perm[a]``, reversed when ``sign[a] == -1``**. Every 90 degree rotation, every axis
    flip and every composition of them is of this form, and nothing else is:

    - a flip of axis ``a`` is ``perm = identity``, ``sign[a] = -1``;
    - ``np.rot90(arr, 1, axes=(0, 1))`` is ``perm = (1, 0)``, ``sign = (-1, +1)``.

    The same ``(perm, sign)`` acts on a vector-valued channel triple by
    ``v_out[a] = sign[a] * v_in[perm[a]]`` (:meth:`transform_vectors`), which is what makes a single
    generic TTA possible.

    Parameters
    ----------
    perm : tuple of int
        ``perm[a]`` is the input spatial axis that becomes output spatial axis ``a``. Must be a
        permutation of ``range(ndim)``.

    sign : tuple of int
        ``+1`` or ``-1`` per output spatial axis; ``-1`` reverses that axis.
    """

    perm: Tuple[int, ...]
    sign: Tuple[int, ...]

    def __post_init__(self):
        """Validate that ``perm`` is a permutation and ``sign`` only holds +-1."""
        if sorted(self.perm) != list(range(len(self.perm))):
            raise ValueError("'perm' must be a permutation of range(ndim); got {}".format(self.perm))
        if len(self.sign) != len(self.perm):
            raise ValueError("'sign' and 'perm' must have the same length")
        if any(s not in (1, -1) for s in self.sign):
            raise ValueError("'sign' entries must be +1 or -1; got {}".format(self.sign))

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions the transform acts on."""
        return len(self.perm)

    @property
    def is_identity(self) -> bool:
        """Whether the transform leaves everything untouched."""
        return self.perm == tuple(range(self.ndim)) and all(s == 1 for s in self.sign)

    @property
    def permutes_axes(self) -> bool:
        """Whether the transform moves any axis onto another one (i.e. it is not flips-only)."""
        return self.perm != tuple(range(self.ndim))

    @classmethod
    def identity(cls, ndim: int) -> "AxisTransform":
        """Build the identity transform for ``ndim`` spatial dimensions."""
        return cls(tuple(range(ndim)), (1,) * ndim)

    @property
    def inverse(self) -> "AxisTransform":
        """
        The transform undoing this one.

        With ``perm_inv[perm[a]] = a``, composing requires ``sign_inv[b] = sign[perm_inv[b]]``
        (signs are their own inverse), which is what this returns.
        """
        perm_inv = [0] * self.ndim
        for a, p in enumerate(self.perm):
            perm_inv[p] = a
        sign_inv = tuple(self.sign[perm_inv[b]] for b in range(self.ndim))
        return AxisTransform(tuple(perm_inv), sign_inv)

    def apply(self, arr: NDArray) -> NDArray:
        """
        Apply the transform spatially to an array laid out as ``(spatial..., channels)``.

        Only pixels are moved; channel *values* are untouched (see :meth:`transform_vectors` and
        :meth:`TTASpec.remap_channels` for that half).

        Parameters
        ----------
        arr : NDArray
            Array with ``ndim`` leading spatial axes and a trailing channel axis.

        Returns
        -------
        NDArray
            Transformed array. Contiguous, so it can be stacked and fed to a model directly.
        """
        if arr.ndim != self.ndim + 1:
            raise ValueError(
                "Expected an array with {} spatial axes + 1 channel axis; got shape {}".format(self.ndim, arr.shape)
            )
        out = np.transpose(arr, self.perm + (self.ndim,))
        flip_axes = tuple(a for a in range(self.ndim) if self.sign[a] < 0)
        if flip_axes:
            out = np.flip(out, axis=flip_axes)
        return np.ascontiguousarray(out)

    def transform_vectors(self, vecs: NDArray) -> NDArray:
        """
        Apply the transform to vectors expressed in spatial-axis order.

        Parameters
        ----------
        vecs : NDArray
            Array of shape ``(..., ndim)`` whose last axis holds the components in spatial-axis
            order (``(y, x)`` in 2D, ``(z, y, x)`` in 3D).

        Returns
        -------
        NDArray
            ``out[..., a] = sign[a] * vecs[..., perm[a]]``.
        """
        vecs = np.asarray(vecs)
        out = np.empty_like(vecs)
        for a in range(self.ndim):
            out[..., a] = self.sign[a] * vecs[..., self.perm[a]]
        return out

    def describe(self) -> str:
        """Return a short human-readable description, e.g. ``"y<-+x, x<--y"``."""
        names = ("y", "x") if self.ndim == 2 else ("z", "y", "x")
        return ", ".join(
            "{}<-{}{}".format(names[a], "+" if self.sign[a] > 0 else "-", names[self.perm[a]])
            for a in range(self.ndim)
        )


def build_axis_transform_group(
    ndim: int,
    level: str = "full",
    interchangeable_axes: Optional[Sequence[int]] = None,
) -> List[AxisTransform]:
    """
    Enumerate the TTA orientations as signed axis permutations.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions (2 or 3).

    level : str, optional
        - ``"full"``: every permutation of ``interchangeable_axes`` crossed with every combination
          of axis flips. 8 orientations in 2D (the dihedral group of the square, i.e. the classic
          "8 rotations and flips"), 16 in 3D (4 in-plane rotations x 3 independent flips).
        - ``"flips"``: axis flips only, no permutation. 4 orientations in 2D, 8 in 3D. This is what
          Cellpose does upstream.
        - ``"none"``: the identity alone (TTA disabled).

    interchangeable_axes : sequence of int, optional
        Spatial axes that may be permuted onto each other. Defaults to ``(0, 1)`` in 2D and
        ``(1, 2)`` in 3D, i.e. **Z is never swapped with Y/X**: volumes are typically anisotropic
        along Z, so an out-of-plane rotation would not be a symmetry of the data.

    Returns
    -------
    list of AxisTransform
        The orientations, identity first, in a deterministic order.
    """
    if ndim not in (2, 3):
        raise ValueError("ndim must be 2 or 3; got {}".format(ndim))
    if level not in ("full", "flips", "none"):
        raise ValueError("level must be one of 'full', 'flips', 'none'; got '{}'".format(level))

    if level == "none":
        return [AxisTransform.identity(ndim)]

    if interchangeable_axes is None:
        interchangeable_axes = (0, 1) if ndim == 2 else (1, 2)
    inter = tuple(sorted(interchangeable_axes))

    if level == "flips":
        perms = [tuple(range(ndim))]
    else:
        perms = []
        for sub in itertools.permutations(inter):
            p = list(range(ndim))
            for slot, src in zip(inter, sub):
                p[slot] = src
            perms.append(tuple(p))

    out = []
    for signs in itertools.product((1, -1), repeat=ndim):
        for p in perms:
            out.append(AxisTransform(p, signs))
    # Identity first so orientation 0 always corresponds to the plain prediction.
    out.sort(key=lambda t: (not t.is_identity,))
    return out


# --------------------------------------------------------------------------------------------- #
# Channel groups: what each output channel *is*                                                   #
# --------------------------------------------------------------------------------------------- #
class ChannelGroup:
    """
    Base class describing how a set of physical output channels behaves under an orientation change.

    Sub-classes implement :meth:`supports` (can this orientation be represented exactly?) and
    :meth:`remap` (rewrite the channels of an already spatially-restored prediction).

    Attributes
    ----------
    channels : tuple of int
        Physical output-channel indices covered by this group.

    mode_reducible : bool
        Whether the ensemble may be reduced with ``min``/``max`` instead of a mean. ``False`` for
        signed vector fields: a component-wise minimum of several flow fields is not a flow field,
        it just biases every vector towards the negative axis direction.
    """

    channels: Tuple[int, ...] = ()
    mode_reducible: bool = True
    name: str = "channels"

    def supports(self, t: AxisTransform) -> Optional[str]:
        """
        Check whether the group can be exactly remapped for orientation ``t``.

        Returns
        -------
        str or None
            ``None`` when supported, otherwise a short reason used to explain the degradation.
        """
        return None

    def remap(self, pred: NDArray, t: AxisTransform) -> None:
        """
        Rewrite this group's channels in place.

        ``pred`` has already been un-transformed spatially, so only the channel *values* still live
        in the augmented frame. ``t`` is the orientation that was applied to the **input image**;
        for an equivariant representation the predicted values are therefore ``t`` applied to the
        canonical ones, and undoing them means applying ``t.inverse``.

        Parameters
        ----------
        pred : NDArray
            Single prediction, ``(spatial..., channels)``, float, modified in place.

        t : AxisTransform
            Orientation the prediction was produced in.
        """

    def describe(self) -> str:
        """Short description used in the TTA log line."""
        return "{}[{}]".format(self.name, len(self.channels))


@dataclass
class ScalarChannels(ChannelGroup):
    """
    Channels whose value does not encode any direction: nothing to do beyond the spatial un-transform.

    Covers binary/probability channels (``B``, ``C``, ``M``, ``P``, ``F*``, ``T``), per-instance
    normalized distances (``D``, normalized ``Db``/``Dc``/``Dn``), the discretized ``Db`` bins, the
    EmbedSeg seediness map and the classification head.
    """

    channels: Tuple[int, ...] = ()
    mode_reducible: bool = True
    name: str = "scalar"


@dataclass
class VectorChannels(ChannelGroup):
    """
    Channels forming a vector field, one component per spatial axis.

    Covers Cellpose/Omnipose flows (``Gv``/``Gh``/``Gz``), HoVerNet maps (``V``/``H``/``Z``) and
    EmbedSeg offsets (``E_offset_*``) when ``signed`` is ``True``, and EmbedSeg per-axis sigmas
    (``E_sigma_*``) when it is ``False`` (a magnitude per axis: it permutes with the axes but never
    changes sign).

    Under ``t`` the components transform exactly like the coordinates:
    ``v_out[a] = sign[a] * v_in[perm[a]]``.

    Attributes
    ----------
    axis_channels : tuple of (int or None)
        Physical channel index per spatial axis, ``None`` when that component is not predicted
        (e.g. 2D flows have no Z component).

    signed : bool
        ``True`` for true vectors (sign flips on a reflected axis), ``False`` for per-axis
        magnitudes such as ``E_sigma``.

    axis_scale : tuple of float, optional
        Physical scale each component is expressed in. Two axes may only be swapped when their
        scales match -- EmbedSeg coordinates are built as ``index * spacing_axis / norm``, so on
        anisotropic data swapping Y and X would compare values on different scales.
    """

    axis_channels: Tuple[Optional[int], ...] = ()
    signed: bool = True
    axis_scale: Optional[Tuple[float, ...]] = None
    name: str = "vector"

    @property
    def channels(self) -> Tuple[int, ...]:  # type: ignore[override]
        """Physical channels held by this group."""
        return tuple(c for c in self.axis_channels if c is not None)

    @property
    def mode_reducible(self) -> bool:  # type: ignore[override]
        """Signed vector fields must be averaged; per-axis magnitudes may use min/max."""
        return not self.signed

    def supports(self, t: AxisTransform) -> Optional[str]:
        """Reject orientations that would move a predicted component onto a missing one."""
        ti = t.inverse
        for a in range(t.ndim):
            src = ti.perm[a]
            if (self.axis_channels[a] is None) != (self.axis_channels[src] is None):
                return "{} is not predicted on every axis it would be permuted onto".format(self.name)
            if self.axis_scale is not None and src != a:
                if not np.isclose(self.axis_scale[a], self.axis_scale[src]):
                    return "{} components use different physical scales on axes {} and {} " "(anisotropic data)".format(
                        self.name, a, src
                    )
        return None

    def remap(self, pred: NDArray, t: AxisTransform) -> None:
        """Undo the orientation on the components: ``v[a] = t.inverse.sign[a] * v[t.inverse.perm[a]]``."""
        ti = t.inverse
        src = [pred[..., c].copy() if c is not None else None for c in self.axis_channels]
        for a, dst in enumerate(self.axis_channels):
            if dst is None:
                continue
            comp = src[ti.perm[a]]
            assert comp is not None  # guaranteed by supports()
            pred[..., dst] = comp if (ti.sign[a] > 0 or not self.signed) else -comp

    def describe(self) -> str:
        """Short description used in the TTA log line."""
        return "{}{}[{}]".format(self.name, "" if self.signed else "(unsigned)", len(self.channels))


@dataclass(eq=False)
class RayChannels(ChannelGroup):
    """
    StarDist radial distances: one channel per fixed ray direction.

    A ray channel is a scalar (a distance) but it is *labelled* by a direction, so an orientation
    change permutes the channels. The permutation is found by transforming the ray directions
    themselves and matching them back onto the original set -- no assumption about the angular grid
    is baked in, which means the check is exact by construction and automatically rejects
    orientations the grid does not admit.

    In practice: the 2D uniform-angle grid admits flips for any even ``nrays`` and the 90 degree
    rotations when ``nrays % 4 == 0`` (the default 32 qualifies); the 3D Fibonacci sphere admits
    none, so 3D StarDist degrades to no TTA rather than to wrong TTA.

    Attributes
    ----------
    start : int
        Physical index of the first ray channel.

    dirs : NDArray
        ``(nrays, ndim)`` unit ray directions in **spatial-axis order** (``(y, x)`` / ``(z, y, x)``),
        i.e. already converted from the Cartesian order :func:`generate_rays` returns.
    """

    start: int = 0
    dirs: NDArray = field(default_factory=lambda: np.zeros((0, 2), np.float32))
    name: str = "rays"
    _perm_cache: Dict[AxisTransform, Optional[NDArray]] = field(default_factory=dict, repr=False)

    @property
    def channels(self) -> Tuple[int, ...]:  # type: ignore[override]
        """Physical channels held by this group."""
        return tuple(range(self.start, self.start + len(self.dirs)))

    def _ray_permutation(self, t: AxisTransform) -> Optional[NDArray]:
        """
        Map each augmented-frame ray ``j`` to the canonical ray it corresponds to.

        A ray pointing along ``d`` in the augmented image measures, in the original image, the
        distance along ``t.inverse(d)``. Returns ``dest`` with ``dest[j] = k`` such that
        ``dirs[k] == t.inverse(dirs[j])``, or ``None`` when some transformed direction is not part
        of the ray set (so no exact permutation exists).
        """
        if t in self._perm_cache:
            return self._perm_cache[t]
        result: Optional[NDArray] = None
        if len(self.dirs):
            target = t.inverse.transform_vectors(self.dirs)  # (nrays, ndim)
            # Cosine between every transformed direction and every original one; an exact match is a
            # dot product of 1 for unit vectors.
            dots = target @ self.dirs.T  # (nrays, nrays)
            dest = np.argmax(dots, axis=1)
            if np.allclose(dots[np.arange(len(dest)), dest], 1.0, atol=1e-4) and len(np.unique(dest)) == len(dest):
                result = dest.astype(np.int64)
        self._perm_cache[t] = result
        return result

    def supports(self, t: AxisTransform) -> Optional[str]:
        """Reject orientations that do not map the ray direction set onto itself."""
        if len(self.dirs) == 0:
            return None
        if self._ray_permutation(t) is None:
            return "the {} ray directions are not mapped onto each other by this orientation".format(len(self.dirs))
        return None

    def remap(self, pred: NDArray, t: AxisTransform) -> None:
        """Scatter each augmented-frame ray channel back onto the canonical direction it measures."""
        dest = self._ray_permutation(t)
        if dest is None:
            raise RuntimeError("remap() called with an unsupported orientation")
        s, n = self.start, len(self.dirs)
        block = pred[..., s : s + n].copy()
        pred[..., s + dest] = block

    def describe(self) -> str:
        """Short description used in the TTA log line."""
        return "rays[{}]".format(len(self.dirs))


@dataclass
class AffinityChannels(ChannelGroup):
    """
    Affinity channels: binary "same instance as the voxel ``d`` steps back along axis ``a``" maps.

    Affinities are the one representation where the channel remap is not purely algebraic. With
    ``aff_{a,d}(p) = [L(p) == L(p - d e_a)] and L(p) > 0`` (see
    :func:`biapy.utils.util.seg2aff_pni`), reflecting axis ``a`` turns offset ``+d`` into ``-d``,
    and ``aff_{a,-d}(p) = aff_{a,+d}(p + d e_a)`` -- the same map **shifted by ``d`` voxels**. So an
    orientation that reverses an axis requires a spatial roll on top of the channel permutation.

    Attributes
    ----------
    layout : dict
        ``(spatial_axis, offset) -> physical channel index``.
    """

    layout: Dict[Tuple[int, int], int] = field(default_factory=dict)
    name: str = "affinities"

    @property
    def channels(self) -> Tuple[int, ...]:  # type: ignore[override]
        """Physical channels held by this group."""
        return tuple(sorted(self.layout.values()))

    def supports(self, t: AxisTransform) -> Optional[str]:
        """Reject orientations that would need an affinity offset the configuration does not have."""
        for (axis, off) in self.layout:
            if (t.perm[axis], off) not in self.layout:
                return "no affinity with offset {} along axis {} to receive axis {} " "under this orientation".format(
                    off, t.perm[axis], axis
                )
        return None

    def remap(self, pred: NDArray, t: AxisTransform) -> None:
        """Permute the affinity channels and roll the ones whose offset got reversed."""
        src = {key: pred[..., ch].copy() for key, ch in self.layout.items()}
        for (axis, off), block in src.items():
            dst_axis = t.perm[axis]
            dst = self.layout[(dst_axis, off)]
            if t.sign[axis] > 0:
                pred[..., dst] = block
            else:
                # aff_{b,-d} = aff_{b,+d} shifted by -d, so undo it with a +d roll along axis b and
                # rebuild the border the roll wrapped around by broadcasting the first valid slice
                # (what seg2aff_pni does when it pads the missing starting border).
                rolled = np.roll(block, shift=off, axis=dst_axis)
                if 0 < off < rolled.shape[dst_axis]:
                    lead = [slice(None)] * rolled.ndim
                    lead[dst_axis] = slice(0, off)
                    first_valid = [slice(None)] * rolled.ndim
                    first_valid[dst_axis] = slice(off, off + 1)
                    rolled[tuple(lead)] = rolled[tuple(first_valid)]
                pred[..., dst] = rolled

    def describe(self) -> str:
        """Short description used in the TTA log line."""
        return "affinities[{}]".format(len(self.layout))


# --------------------------------------------------------------------------------------------- #
# The spec                                                                                        #
# --------------------------------------------------------------------------------------------- #
@dataclass
class TTASpec:
    """
    Description of every physical output channel, driving the generic TTA.

    Build it with :func:`build_tta_spec`. A ``None`` spec (or an all-scalar one) reproduces the
    classic "8 rotations and flips" scalar ensemble exactly.

    Attributes
    ----------
    ndim : int
        Number of spatial dimensions.

    n_channels : int
        Number of physical output channels described.

    groups : list of ChannelGroup
        One entry per channel group; every channel belongs to exactly one.
    """

    ndim: int
    n_channels: int
    groups: List[ChannelGroup] = field(default_factory=list)

    @property
    def is_scalar_only(self) -> bool:
        """Whether every channel is a plain scalar field (classic TTA is then exactly correct)."""
        return all(isinstance(g, ScalarChannels) for g in self.groups)

    @property
    def mode_reducible_channels(self) -> List[int]:
        """Channels a ``min``/``max`` ensemble mode may be applied to (see :class:`ChannelGroup`)."""
        out: List[int] = []
        for g in self.groups:
            if g.mode_reducible:
                out.extend(g.channels)
        return sorted(out)

    def filter_orientations(self, orientations: Sequence[AxisTransform]) -> Tuple[List[AxisTransform], List[str]]:
        """
        Keep only the orientations every group can represent exactly.

        Parameters
        ----------
        orientations : sequence of AxisTransform
            Candidate orientations, typically from :func:`build_axis_transform_group`.

        Returns
        -------
        kept : list of AxisTransform
            Supported orientations. Always contains at least the identity.

        reasons : list of str
            De-duplicated explanations for the dropped orientations, for logging.
        """
        kept, reasons = [], []
        for t in orientations:
            why = None
            for g in self.groups:
                why = g.supports(t)
                if why:
                    break
            if why is None:
                kept.append(t)
            elif why not in reasons:
                reasons.append(why)
        if not kept:
            kept = [AxisTransform.identity(self.ndim)]
        return kept, reasons

    def remap_channels(self, pred: NDArray, t: AxisTransform) -> None:
        """
        Rewrite the channels of a spatially-restored prediction in place.

        Parameters
        ----------
        pred : NDArray
            Single prediction, ``(spatial..., channels)``, already un-transformed spatially.

        t : AxisTransform
            The orientation the prediction was produced in.
        """
        if t.is_identity:
            return
        if pred.shape[-1] != self.n_channels:
            raise ValueError(
                "TTA spec describes {} output channels but the model returned {}".format(
                    self.n_channels, pred.shape[-1]
                )
            )
        for g in self.groups:
            g.remap(pred, t)

    def describe(self) -> str:
        """Human-readable summary used in the TTA log line."""
        return " + ".join(g.describe() for g in self.groups) if self.groups else "scalar[0]"


# --------------------------------------------------------------------------------------------- #
# Building the spec from the model output channel names                                           #
# --------------------------------------------------------------------------------------------- #
#: ``Az_1``, ``Ay_2``, ``Ax_1``, ...
_AFFINITY_RE = re.compile(r"^A([zyx])_(-?\d+)$")
#: ``R_0`` ... ``R_31``
_RAY_RE = re.compile(r"^R_(\d+)$")
#: ``E_offset_0`` / ``E_sigma_1``
_EMBED_RE = re.compile(r"^(E_offset|E_sigma)_(\d+)$")

#: Flow / HoVer style single-channel vector components -> the axis letter they live on.
_VECTOR_COMPONENT_AXIS = {
    "Gz": "z",
    "Gv": "y",
    "Gh": "x",
    "Z": "z",
    "V": "y",
    "H": "x",
}


def _axis_index(letter: str, ndim: int) -> Optional[int]:
    """Spatial-axis index of an axis letter, or ``None`` when the axis does not exist in ``ndim``."""
    order = ("y", "x") if ndim == 2 else ("z", "y", "x")
    return order.index(letter) if letter in order else None


def parse_model_output_channel_names(model_output_channel_info: Sequence[str]) -> List[str]:
    """
    Flatten the per-head ``model_output_channel_info`` into one name per physical ``pred`` channel.

    Workflows build ``model_output_channel_info`` as one ``"+"``-joined string per output head (e.g.
    ``["Gv+Gh+B", "class"]``), and the model's ``pred`` output is the concatenation of the heads.
    The classification head is returned separately by the model, so it is excluded here.

    Parameters
    ----------
    model_output_channel_info : sequence of str
        Per-head channel descriptions.

    Returns
    -------
    list of str
        One channel name per physical channel of ``pred``, in order.
    """
    names: List[str] = []
    for head in model_output_channel_info:
        if head == "class":
            continue  # separate model output, handled as a scalar field by the ensembler
        names.extend([c for c in head.split("+") if c])
    return names


def build_tta_spec(
    channel_names: Sequence[str],
    ndim: int,
    channel_extra_opts: Optional[Dict] = None,
    anisotropy: Optional[Sequence[float]] = None,
) -> TTASpec:
    """
    Build the TTA channel spec from the model's physical output channel names.

    The names come from ``model_output_channel_info`` (see
    :func:`parse_model_output_channel_names`), which is the same metadata used to size the model
    heads -- so the spec cannot drift from the actual output layout.

    Parameters
    ----------
    channel_names : sequence of str
        One name per physical output channel, e.g. ``["Gv", "Gh", "B"]``,
        ``["B", "R_0", ..., "R_31"]``, ``["E_offset_0", "E_offset_1", "E_sigma_0", "E_sigma_1",
        "E_seediness"]`` or ``["Az_1", "Ay_1", "Ax_1"]``. Unknown names are treated as scalars.

    ndim : int
        Number of spatial dimensions (2 or 3).

    channel_extra_opts : dict, optional
        ``PROBLEM.INSTANCE_SEG.DATA_CHANNELS_EXTRA_OPTS[0]``; only ``R`` is read, to rebuild the
        exact ray directions.

    anisotropy : sequence of float, optional
        Voxel spacing in spatial-axis order (``(y, x)`` / ``(z, y, x)``). Used to decide whether the
        EmbedSeg coordinate axes are on the same scale and may therefore be swapped. ``None``
        assumes isotropic.

    Returns
    -------
    TTASpec
        Spec covering every channel exactly once.
    """
    from biapy.data.pre_processing import generate_rays

    channel_extra_opts = channel_extra_opts or {}
    names = list(channel_names)
    n = len(names)
    groups: List[ChannelGroup] = []
    claimed: set = set()

    # --- Flow / HoVer vector components (one channel per axis, named individually) --------------
    for family, members in (("flow", ("Gz", "Gv", "Gh")), ("hover", ("Z", "V", "H"))):
        present = {m: names.index(m) for m in members if m in names}
        if not present:
            continue
        axis_channels: List[Optional[int]] = [None] * ndim
        for m, idx in present.items():
            ax = _axis_index(_VECTOR_COMPONENT_AXIS[m], ndim)
            if ax is not None:
                axis_channels[ax] = idx
                claimed.add(idx)
            else:
                # e.g. a 'Z'/'Gz' channel declared on 2D data: it carries no in-plane direction, so
                # treat it as a scalar rather than silently dropping it from the spec.
                groups.append(ScalarChannels(channels=(idx,)))
                claimed.add(idx)
        if any(c is not None for c in axis_channels):
            groups.append(VectorChannels(axis_channels=tuple(axis_channels), signed=True, name=family))

    # --- EmbedSeg offsets / sigmas (``E_offset_i`` with i in Cartesian [x, y, z] order) ---------
    embed: Dict[str, Dict[int, int]] = {"E_offset": {}, "E_sigma": {}}
    for i, nm in enumerate(names):
        m = _EMBED_RE.match(nm)
        if m:
            embed[m.group(1)][int(m.group(2))] = i
    cart_to_axis = {0: "x", 1: "y", 2: "z"}
    scale = None
    if anisotropy is not None and len(anisotropy) == ndim:
        scale = tuple(float(s) for s in anisotropy)
    for fam, comps in embed.items():
        if not comps:
            continue
        axis_channels = [None] * ndim
        for cart_i, ch in comps.items():
            ax = _axis_index(cart_to_axis.get(cart_i, "?"), ndim)
            if ax is None:
                groups.append(ScalarChannels(channels=(ch,)))
            else:
                axis_channels[ax] = ch
            claimed.add(ch)
        if any(c is not None for c in axis_channels):
            groups.append(
                VectorChannels(
                    axis_channels=tuple(axis_channels),
                    signed=(fam == "E_offset"),
                    axis_scale=scale,
                    name=fam,
                )
            )

    # --- StarDist rays --------------------------------------------------------------------------
    ray_idx = sorted((int(m.group(1)), i) for i, nm in enumerate(names) if (m := _RAY_RE.match(nm)))
    if ray_idx:
        positions = [i for _, i in ray_idx]
        if positions != list(range(positions[0], positions[0] + len(positions))):
            raise ValueError("StarDist ray channels must be contiguous; got positions {}".format(positions))
        nrays = len(positions)
        cfg_nrays = int(channel_extra_opts.get("R", {}).get("nrays", nrays))
        if cfg_nrays != nrays:
            raise ValueError(
                "'R' declares nrays={} but {} ray output channels were found".format(cfg_nrays, nrays)
            )
        # generate_rays returns Cartesian [x, y(, z)]; radial_distances reverses it to index order,
        # so do the same here to stay in spatial-axis order.
        dirs = np.asarray(generate_rays(n_rays=nrays, ndim=ndim), dtype=np.float64)[:, ::-1].copy()
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
        groups.append(RayChannels(start=positions[0], dirs=dirs))
        claimed.update(positions)

    # --- Affinities ----------------------------------------------------------------------------
    layout: Dict[Tuple[int, int], int] = {}
    for i, nm in enumerate(names):
        m = _AFFINITY_RE.match(nm)
        if m:
            ax = _axis_index(m.group(1), ndim)
            if ax is None:
                # A z-affinity declared on 2D data has no axis to live on; keep it a scalar so the
                # channel is still accounted for, and let supports() decide nothing else.
                groups.append(ScalarChannels(channels=(i,)))
            else:
                layout[(ax, int(m.group(2)))] = i
            claimed.add(i)
    if layout:
        groups.append(AffinityChannels(layout=layout))

    # --- Everything else is a scalar field ------------------------------------------------------
    rest = tuple(i for i in range(n) if i not in claimed)
    if rest:
        groups.append(ScalarChannels(channels=rest))

    covered = sorted(c for g in groups for c in g.channels)
    if covered != list(range(n)):
        raise ValueError(
            "TTA spec does not cover every output channel exactly once (covered {} of {}); "
            "channel names were {}".format(len(covered), n, names)
        )
    return TTASpec(ndim=ndim, n_channels=n, groups=groups)
