"""
OME-Zarr (OME-NGFF) support.

BiaPy reads any Zarr (a plain array, a group of arrays, or an OME-Zarr 0.4/0.5 image) and writes all the Zarr files it
creates as OME-Zarr images (https://ngff.openmicroscopy.org/specifications/0.5/): a group with the ``multiscales``
metadata (axes and voxel size) and one array per resolution level, the full resolution one at ``"0"``. Their axes are
ordered as the specification requires: time, channel and then the spatial ones (``[T]C[Z]YX``).

OME-Zarr 0.5 uses Zarr format 3 and 0.4 Zarr format 2 (still the only one some viewers read).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import zarr

# Order of the axes in an OME-Zarr image: time, channel and then the spatial axes
NGFF_AXES_ORDER = "TCZYX"
SUPPORTED_VERSIONS = ("0.4", "0.5")
DEFAULT_VERSION = "0.5"
_AXIS_TYPES = {"T": "time", "C": "channel", "Z": "space", "Y": "space", "X": "space"}


@dataclass
class OmeZarrImage:
    """
    Information of an OME-Zarr image, from its ``multiscales`` metadata.

    Attributes
    ----------
    version : str
        OME-Zarr version, e.g. ``"0.5"``.
    axes : str
        Axes of the arrays as BiaPy names them, e.g. ``"CZYX"``.
    shape : tuple of int
        Shape of the level read.
    scale : tuple of float
        Size of a voxel of the level read along each axis, in ``units``.
    units : tuple of str or None
        Unit of each axis (e.g. ``"micrometer"``), ``None`` if not given.
    translation : tuple of float
        Position of the first voxel of the level read along each axis.
    level : int
        Resolution level read (``0`` is the full resolution).
    levels : list of str
        Path of each resolution level, from the highest resolution to the lowest.
    name : str
        Name of the image.
    labels : list of str
        Label images found in its ``labels`` group.
    channel_names : list of str
        Name of each channel (from the ``omero`` metadata).
    """

    version: str
    axes: str
    shape: Tuple[int, ...]
    scale: Tuple[float, ...]
    units: Tuple[Optional[str], ...]
    translation: Tuple[float, ...]
    level: int = 0
    levels: List[str] = field(default_factory=list)
    name: str = ""
    labels: List[str] = field(default_factory=list)
    channel_names: List[str] = field(default_factory=list)

    def voxel_size(self, axes: str = "ZYX") -> Tuple[Optional[float], ...]:
        """Size of a voxel along ``axes``, in microns (``None`` for the axes without a known length unit)."""
        from biapy.data.data_manipulation import _UNIT_TO_MICRONS

        sizes = []
        for a in axes:
            if a not in self.axes:
                sizes.append(None)
                continue
            i = self.axes.index(a)
            factor = _UNIT_TO_MICRONS.get(str(self.units[i]).lower()) if self.units[i] else None
            sizes.append(self.scale[i] * factor if factor is not None else None)
        return tuple(sizes)

    def resolution(self, axes: str = "ZYX") -> Tuple[float, ...]:
        """
        Voxel size along ``axes`` relative to the X one, as BiaPy takes the resolution (e.g. ``(10, 1, 1)`` for a
        voxel of 40x4x4 nm). Axes not present in the image are 1.
        """
        x = self.scale[self.axes.index("X")] if "X" in self.axes else 1.0
        return tuple(
            float(self.scale[self.axes.index(a)]) / x if a in self.axes and x > 0 else 1.0 for a in axes
        )

    def summary(self) -> List[str]:
        """Lines describing the image, to show the user what BiaPy takes from its metadata."""
        spatial = "".join(a for a in self.axes if a in "ZYX")
        lines = [
            "OME-Zarr {}, resolution level {} of {}".format(self.version, self.level, max(1, len(self.levels))),
            "axes: {} ({})".format(
                self.axes, ", ".join("{}: {}".format(a.lower(), _AXIS_TYPES[a]) for a in self.axes)
            ),
            "shape: {}".format(self.shape),
        ]
        units = [self.units[self.axes.index(a)] for a in spatial]
        scale = " x ".join("{:g}".format(self.scale[self.axes.index(a)]) for a in spatial)
        unit = units[0] if units and all(u == units[0] for u in units) else None
        lines.append(
            "voxel size ({}): {}{} -> resolution used: {}".format(
                ", ".join(spatial.lower()),
                scale,
                " " + unit if unit else (" (" + ", ".join(str(u) for u in units) + ")" if any(units) else ""),
                tuple(round(v, 4) for v in self.resolution(spatial)),
            )
        )
        if self.channel_names:
            lines.append("channels: {}".format(", ".join(self.channel_names)))
        if self.labels:
            lines.append("labels: {}".format(", ".join(self.labels)))
        return lines


def ngff_axes_order(axes: str, with_channel: bool = True) -> str:
    """
    Order the given axes as OME-Zarr requires: time, channel and then the spatial ones (e.g. ``"ZYXC"`` -> ``"CZYX"``).

    Parameters
    ----------
    axes : str
        Axes, e.g. ``"ZYXC"``.

    with_channel : bool, optional
        Whether to add the channel axis if ``axes`` does not have it.

    Returns
    -------
    axes : str
        Axes in OME-Zarr order.
    """
    return "".join(a for a in NGFF_AXES_ORDER if a in axes or (with_channel and a == "C"))


def _ome_attrs(group: zarr.Group) -> Dict:
    """OME metadata of a group: under ``"ome"`` in 0.5, the attributes themselves before."""
    attrs = dict(group.attrs)
    return attrs["ome"] if isinstance(attrs.get("ome"), dict) else attrs


def _axis_letter(axis) -> str:
    """BiaPy name of an OME-Zarr axis (a dict since 0.4, its name in 0.3)."""
    name = (axis if isinstance(axis, str) else axis.get("name", "")).lower()
    atype = None if isinstance(axis, str) else axis.get("type")
    if atype == "time" or name in ("t", "time"):
        return "T"
    if atype == "channel" or name in ("c", "ch", "channel"):
        return "C"
    if name in ("z", "y", "x"):
        return name.upper()
    raise ValueError(
        "OME-Zarr axis {} is not supported: BiaPy works with 't' (time), 'c' (channel), 'z', 'y' and 'x' axes".format(
            axis
        )
    )


def _transforms(transforms: Optional[Sequence[Dict]], ndim: int) -> Tuple[List[float], List[float]]:
    """Scale and translation of a list of ``coordinateTransformations``."""
    scale, translation = [1.0] * ndim, [0.0] * ndim
    for t in transforms or []:
        if t.get("type") == "scale" and "scale" in t:
            scale = [float(v) for v in t["scale"]]
        elif t.get("type") == "translation" and "translation" in t:
            translation = [float(v) for v in t["translation"]]
    return scale, translation


def find_image_group(node) -> Optional[zarr.Group]:
    """
    Group of the OME-Zarr image in ``node``: the node itself if it holds ``multiscales`` metadata, or the first image
    of a ``bioformats2raw`` layout (``"0"``). ``None`` if it is not an OME-Zarr image.
    """
    if not isinstance(node, zarr.Group):
        return None
    ome = _ome_attrs(node)
    if ome.get("multiscales"):
        return node
    if "bioformats2raw.layout" in ome and "0" in node:
        return find_image_group(node["0"])
    if ome.get("plate") or ome.get("well"):
        raise ValueError("OME-Zarr plates (high-content screening) are not supported. Use one of its images instead")
    return None


def read_image(group: zarr.Group, level: int = 0) -> Tuple[OmeZarrImage, zarr.Array]:
    """
    Read the metadata of an OME-Zarr image and open one of its resolution levels.

    Parameters
    ----------
    group : zarr.Group
        Group of the image (see :func:`find_image_group`).

    level : int, optional
        Resolution level to open.

    Returns
    -------
    info : OmeZarrImage
        Information of the image.

    data : zarr.Array
        Array of the level.
    """
    ome = _ome_attrs(group)
    multiscales = ome["multiscales"][0]
    version = str(ome.get("version", multiscales.get("version", "0.4")))
    datasets = multiscales["datasets"]
    if not 0 <= level < len(datasets):
        raise ValueError(
            "Resolution level {} not found in the OME-Zarr image {} ({} levels)".format(level, group.path, len(datasets))
        )
    data = group[datasets[level]["path"]]
    axes_meta = multiscales.get("axes")
    if axes_meta is None:  # Before 0.3 the axes were always 'tczyx'
        axes_meta = list("tczyx")[-data.ndim :]
    axes = "".join(_axis_letter(a) for a in axes_meta)
    if len(axes) != data.ndim:
        raise ValueError(
            "The OME-Zarr image {} has {} axes in its metadata but its data has {} dimensions".format(
                group.path, len(axes), data.ndim
            )
        )
    scale, translation = _transforms(datasets[level].get("coordinateTransformations"), data.ndim)
    # Transformations of the whole multiscales are applied after the ones of each level
    gscale, gtranslation = _transforms(multiscales.get("coordinateTransformations"), data.ndim)
    scale = [s * g for s, g in zip(scale, gscale)]
    translation = [t * g + gt for t, g, gt in zip(translation, gscale, gtranslation)]
    labels = []
    if "labels" in group:
        labels = [str(x) for x in _ome_attrs(group["labels"]).get("labels", [])]
    channels = ome.get("omero", {}).get("channels", []) if isinstance(ome.get("omero"), dict) else []
    info = OmeZarrImage(
        version=version,
        axes=axes,
        shape=tuple(int(s) for s in data.shape),
        scale=tuple(scale),
        units=tuple(None if isinstance(a, str) else a.get("unit") for a in axes_meta),
        translation=tuple(translation),
        level=level,
        levels=[str(d["path"]) for d in datasets],
        name=str(multiscales.get("name", "")),
        labels=labels,
        channel_names=[str(c.get("label", "")) for c in channels if c.get("label")],
    )
    return info, data


def _open_node(path: str, data_path: str = "", mode: str = "r"):
    """Open a Zarr file and go to ``data_path`` within it (``"a.b.c"`` or ``"a/b/c"``)."""
    node = zarr.open(path, mode=mode)
    for part in data_path.replace(".", "/").strip("/").split("/"):
        if part:
            node = node[part]
    return node


def image_info(path: str, data_path: str = "", level: int = 0) -> Optional[OmeZarrImage]:
    """
    Information of the OME-Zarr image at ``data_path`` within the Zarr file ``path``.

    Parameters
    ----------
    path : str
        Path to the Zarr file.

    data_path : str, optional
        Path within the file, e.g. ``"labels.nuclei"``. The image itself if empty.

    level : int, optional
        Resolution level.

    Returns
    -------
    info : OmeZarrImage or None
        Information of the image, ``None`` if it is not an OME-Zarr image (or not a Zarr file at all).
    """
    if not str(path).endswith(".zarr") or not os.path.exists(path):
        return None
    try:
        node = _open_node(path, data_path)
    except (KeyError, FileNotFoundError):
        return None
    group = find_image_group(node)
    return read_image(group, level)[0] if group is not None else None


def resolve_axes(
    path: str, configured_axes: str, data_path: str = "", use_metadata: bool = True
) -> Tuple[str, Optional[OmeZarrImage]]:
    """
    Axes of an image: from its metadata if it is an OME-Zarr image, ``configured_axes`` otherwise.

    Parameters
    ----------
    path : str
        Path to the image.

    configured_axes : str
        Axes set by the user (e.g. ``DATA.TEST.INPUT_IMG_AXES_ORDER``).

    data_path : str, optional
        Path of the image within the Zarr file.

    use_metadata : bool, optional
        Whether to take the axes from the OME-Zarr metadata (``DATA.OME_ZARR.IGNORE_METADATA`` disables it).

    Returns
    -------
    axes : str
        Axes of the image.

    info : OmeZarrImage or None
        Information of the image if it is an OME-Zarr image whose metadata is used, ``None`` otherwise.
    """
    info = image_info(path, data_path) if use_metadata else None
    return (info.axes, info) if info is not None else (configured_axes, None)


_reported = set()


def report_ome_zarr_images(images: Dict[str, OmeZarrImage], where: str, ignored: Sequence[str]):
    """
    Print the information BiaPy takes from the metadata of OME-Zarr images, once per image. Images with the same
    axes and voxel size as one already described are just listed.

    Parameters
    ----------
    images : dict
        Information of each image, by file name.

    where : str
        Folder of the images.

    ignored : list of str
        Variables not used for these images, as their metadata gives the information.
    """
    from biapy.utils.misc import is_main_process

    images = {k: v for k, v in images.items() if (where, k) not in _reported}
    if not images or not is_main_process():
        return
    _reported.update((where, k) for k in images)
    print(
        "WARNING: OME-Zarr images found in {}. Their axes and voxel size are taken from their metadata, so {} are not "
        "used for them (set 'DATA.OME_ZARR.IGNORE_METADATA' to use them instead):".format(where, ", ".join(ignored))
    )
    groups: Dict[Tuple, List[str]] = {}
    for name, info in images.items():
        groups.setdefault((info.version, info.axes, info.scale, info.units), []).append(name)
    for names in groups.values():
        print("  {}:".format(names[0]))
        for line in images[names[0]].summary():
            print("    " + line)
        if len(names) > 1:
            others = names[1:] if len(names) <= 4 else [names[1], "...", names[-1]]
            print("  {} ({} images): same axes and voxel size".format(", ".join(others), len(names) - 1))


def find_array(node, level: int = 0) -> zarr.Array:
    """
    Array of a Zarr node: a level of its OME-Zarr image, the node itself if it is an array, or the first array found
    in it otherwise (sorted by key).

    Parameters
    ----------
    node : zarr.Group or zarr.Array
        Node.

    level : int, optional
        Resolution level, if it is an OME-Zarr image.

    Returns
    -------
    data : zarr.Array
        Array.
    """
    if isinstance(node, zarr.Array):
        return node
    group = find_image_group(node)
    if group is not None:
        return read_image(group, level)[1]
    while isinstance(node, zarr.Group):
        keys = sorted(node.keys())
        if not keys:
            raise ValueError("No array found in the Zarr group '{}'".format(node.path))
        node = node[keys[0]]
    return node


def open_array(path: str, mode: str = "r", data_path: str = "", level: int = 0) -> zarr.Array:
    """
    Open the array of a Zarr file: a level of its OME-Zarr image, the array itself if the file is a plain array (e.g.
    created by BiaPy before it wrote OME-Zarr), or the first array found in it otherwise.

    Parameters
    ----------
    path : str
        Path to the Zarr file.

    mode : str, optional
        Mode to open the file with.

    data_path : str, optional
        Path within the file, e.g. ``"labels.nuclei"``.

    level : int, optional
        Resolution level, if it is an OME-Zarr image.

    Returns
    -------
    data : zarr.Array
        Array.
    """
    return find_array(_open_node(path, data_path, mode=mode), level)


def to_biapy_order(data: np.ndarray, axes: str, is_3d: bool) -> np.ndarray:
    """
    Reorder an image to the order BiaPy works with, ``(z, y, x, c)`` in 3D and ``(y, x, c)`` in 2D, adding the
    channel axis if missing. The time axis, and the z axis in 2D, must have size 1.

    Parameters
    ----------
    data : NDArray
        Image.

    axes : str
        Axes of ``data``, e.g. ``"CZYX"``.

    is_3d : bool
        Whether the image is 3D.

    Returns
    -------
    data : NDArray
        Reordered image.
    """
    target = "ZYXC" if is_3d else "YXC"
    for i, a in enumerate(axes):
        if a not in target and data.shape[i] != 1:
            raise ValueError(
                "Axis '{}' of the image (axes {}, shape {}) has size {}, but it must be 1 to process the image in "
                "{}".format(a, axes, data.shape, data.shape[i], "3D" if is_3d else "2D")
            )
    data = data[tuple(slice(None) if a in target else 0 for a in axes)]
    axes = "".join(a for a in axes if a in target)
    for a in target:
        if a not in axes:
            data, axes = np.expand_dims(data, -1), axes + a
    return np.transpose(data, [axes.index(a) for a in target])


def _version(version) -> str:
    """OME-Zarr version as a string, e.g. ``"0.5"`` (``DATA.OME_ZARR.VERSION`` is a number in the configuration)."""
    version = str(float(version)) if isinstance(version, (int, float)) else str(version)
    if version not in SUPPORTED_VERSIONS:
        raise ValueError("OME-Zarr version {} not supported. Options: {}".format(version, SUPPORTED_VERSIONS))
    return version


def multiscales_metadata(
    axes: str,
    scales: Sequence[Sequence[float]],
    units: Optional[Sequence[Optional[str]]] = None,
    name: str = "",
    version: str | float = DEFAULT_VERSION,
) -> Dict:
    """
    Attributes of the group of an OME-Zarr image.

    Parameters
    ----------
    axes : str
        Axes of the arrays, in OME-Zarr order (see :func:`ngff_axes_order`).

    scales : list of list of float
        Voxel size of each resolution level (``"0"``, ``"1"``...), one value per axis.

    units : list of str or None, optional
        Unit of each axis (``None`` for no unit).

    name : str, optional
        Name of the image.

    version : str or float, optional
        OME-Zarr version.

    Returns
    -------
    attrs : dict
        Attributes.
    """
    version = _version(version)
    if axes != ngff_axes_order(axes, with_channel="C" in axes):
        raise ValueError("Axes {} are not in OME-Zarr order ({})".format(axes, ngff_axes_order(axes, "C" in axes)))
    units = units or [None] * len(axes)
    axes_meta = []
    for a, unit in zip(axes, units):
        axis = {"name": a.lower(), "type": _AXIS_TYPES[a]}
        if unit and a != "C":
            axis["unit"] = unit
        axes_meta.append(axis)
    multiscale = {
        "name": name,
        "axes": axes_meta,
        "datasets": [
            {"path": str(i), "coordinateTransformations": [{"type": "scale", "scale": [float(v) for v in s]}]}
            for i, s in enumerate(scales)
        ],
    }
    if version == "0.4":
        return {"multiscales": [{"version": version, **multiscale}]}
    return {"ome": {"version": version, "multiscales": [multiscale]}}


def _array_kwargs(axes: str, version: str) -> Dict:
    """Arguments to create the arrays of an OME-Zarr image: 0.5 names their dimensions, 0.4 nests their chunks."""
    if version == "0.5":
        return {"dimension_names": [a.lower() for a in axes]}
    return {"chunk_key_encoding": {"name": "v2", "separator": "/"}}


def default_chunks(shape: Sequence[int], dtype: str) -> Tuple[int, ...]:
    """Chunk shape of about 4 MB for an image in OME-Zarr axes order, with few channels and time points per chunk."""
    from biapy.data.data_3D_manipulation import pick_chunks

    # pick_chunks keeps the last dimension small, which is meant for channels
    return tuple(pick_chunks(tuple(shape)[::-1], dtype)[::-1])


def create_image(
    path: str,
    shape: Sequence[int],
    axes: str,
    dtype: str,
    chunks: Sequence[int],
    scale: Optional[Sequence[float]] = None,
    units: Optional[Sequence[Optional[str]]] = None,
    version: str | float = DEFAULT_VERSION,
) -> zarr.Array:
    """
    Create an OME-Zarr image with one resolution level, or open it if another process/rank created it already (all
    the ranks of a job, and the jobs of a split prediction, write into the same file).

    Parameters
    ----------
    path : str
        Path to the Zarr file.

    shape : tuple of int
        Shape of the image.

    axes : str
        Axes of the image, in OME-Zarr order (see :func:`ngff_axes_order`).

    dtype : str
        Data type.

    chunks : tuple of int
        Chunk shape.

    scale : tuple of float, optional
        Voxel size along each axis (1 if not given).

    units : tuple of str or None, optional
        Unit of each axis.

    version : str or float, optional
        OME-Zarr version (``"0.5"`` or ``"0.4"``).

    Returns
    -------
    data : zarr.Array
        Full resolution array of the image, open for writing.
    """
    shape = tuple(int(s) for s in shape)
    version = _version(version)
    attrs = multiscales_metadata(axes, [scale or [1.0] * len(axes)], units, os.path.splitext(os.path.basename(path))[0], version)
    zarr_format = 3 if version == "0.5" else 2
    array_kwargs = _array_kwargs(axes, version)

    last_err = None
    for _ in range(20):
        try:
            group = zarr.open_group(path, mode="w-", zarr_format=zarr_format, attributes=attrs)
            return group.create_array(
                "0", shape=shape, chunks=tuple(int(c) for c in chunks), dtype=dtype, **array_kwargs
            )
        except Exception as e:
            last_err = e
        # Created by another rank/process (or left over from a previous run)
        try:
            node = zarr.open(path, mode="r+")
        except FileNotFoundError:
            time.sleep(0.05)
            continue
        if isinstance(node, zarr.Array) or (find_image_group(node) is None and len(node.attrs) > 0):
            raise RuntimeError(
                "{} is not an OME-Zarr image. It is likely left over from a run of an older BiaPy version: remove it "
                "and re-run.".format(path)
            )
        try:
            data = open_array(path, mode="r+")
        except (KeyError, FileNotFoundError, ValueError):
            # The other process has not finished creating it
            time.sleep(0.05)
            continue
        if tuple(data.shape) != shape:
            raise RuntimeError(
                f"Existing Zarr at {path} has shape {data.shape}, but this run expects {shape}. It is likely left "
                "over from a previous run with a different 'DATA.PATCH_SIZE' or "
                "'DATA.PREPROCESS.ZOOM.ZOOM_FACTOR'. Remove it and re-run."
            )
        return data
    raise RuntimeError(f"Could not create/open the OME-Zarr image {path}. Last error: {last_err}")


def output_scale(
    axes: str, shape: Sequence[int], source: Optional[OmeZarrImage]
) -> Tuple[Optional[List[float]], Optional[List[Optional[str]]]]:
    """
    Voxel size and units of an image created from the OME-Zarr image ``source``: the spatial axes keep its physical
    extent (so the voxel size changes if the image was resized, e.g. zoomed or upsampled) and units.

    Parameters
    ----------
    axes : str
        Axes of the image created.

    shape : tuple of int
        Shape of the image created.

    source : OmeZarrImage or None
        Image it was created from.

    Returns
    -------
    scale : list of float or None
        Voxel size along each axis, ``None`` if there is no source.

    units : list of str or None
        Units of each axis, ``None`` if there is no source.
    """
    if source is None:
        return None, None
    scale, units = [], []
    for a, size in zip(axes, shape):
        if a in source.axes and a not in "C":
            i = source.axes.index(a)
            scale.append(source.scale[i] * source.shape[i] / max(1, int(size)))
            units.append(source.units[i])
        else:
            scale.append(1.0)
            units.append(None)
    return scale, units


def write_image(
    path: str,
    data: np.ndarray,
    axes: str,
    scale: Optional[Sequence[float]] = None,
    units: Optional[Sequence[Optional[str]]] = None,
    version: str | float = DEFAULT_VERSION,
    chunks: Optional[Sequence[int]] = None,
    pyramid_levels: int = 1,
):
    """
    Write a whole array as an OME-Zarr image, transposing it to OME-Zarr axes order.

    Parameters
    ----------
    path : str
        Path to the Zarr file to create. It must not exist.

    data : NDArray
        Data to write.

    axes : str
        Axes of ``data``, e.g. ``"ZYXC"``.

    scale : tuple of float, optional
        Voxel size along each axis of ``data``.

    units : tuple of str or None, optional
        Unit of each axis of ``data``.

    version : str or float, optional
        OME-Zarr version.

    chunks : tuple of int, optional
        Chunk shape, along the axes of ``data``. Picked automatically if not given.

    pyramid_levels : int, optional
        Number of resolution levels to write (see :func:`build_pyramid`).
    """
    out_axes = ngff_axes_order(axes, with_channel="C" in axes)
    order = [axes.index(a) for a in out_axes]
    data = np.transpose(data, order)
    if chunks is None:
        chunks = default_chunks(data.shape, str(data.dtype))
    else:
        chunks = [chunks[i] for i in order]
    out = create_image(
        path,
        data.shape,
        out_axes,
        str(data.dtype),
        chunks,
        scale=[scale[i] for i in order] if scale is not None else None,
        units=[units[i] for i in order] if units is not None else None,
        version=version,
    )
    out[...] = data
    if pyramid_levels > 1:
        build_pyramid(path, pyramid_levels)


def downscale_factors(axes: str, shape: Sequence[int], scale: Sequence[float]) -> Tuple[int, ...]:
    """
    Factor to reduce each axis by to create the next resolution level: 2 along the spatial axes whose voxel size is
    at most twice the smallest one (so anisotropic images, e.g. 40x4x4 nm, are first reduced only in y and x until
    their voxels are about isotropic), 1 along the rest.

    Parameters
    ----------
    axes : str
        Axes of the level, e.g. ``"CZYX"``.

    shape : tuple of int
        Shape of the level.

    scale : tuple of float
        Voxel size of the level along each axis.

    Returns
    -------
    factors : tuple of int
        Factor of each axis. All 1 if the level can not be reduced anymore.
    """
    spatial = [i for i, a in enumerate(axes) if a in "ZYX" and shape[i] > 1]
    smallest = min((float(scale[i]) for i in spatial), default=0.0)
    return tuple(2 if i in spatial and float(scale[i]) <= 2 * smallest else 1 for i in range(len(axes)))


def _downsample(data: np.ndarray, factors: Sequence[int]) -> np.ndarray:
    """
    Reduce ``data`` by ``factors``: averaging the voxels of each block for floating point data, taking one of them
    for the rest (labels must keep their values). Incomplete blocks at the end are averaged with what they have.
    """
    if not np.issubdtype(data.dtype, np.floating):
        return data[tuple(slice(None, None, f) for f in factors)]
    pad = [(0, -s % f) for s, f in zip(data.shape, factors)]
    padded = np.pad(data.astype(np.float64), pad, mode="edge") if any(p[1] for p in pad) else data
    shape = [v for s, f in zip(padded.shape, factors) for v in (s // f, f)]
    return padded.reshape(shape).mean(axis=tuple(range(1, len(shape), 2))).astype(data.dtype)


def build_pyramid(path: str, levels: int):
    """
    Create the lower resolution levels of an OME-Zarr image from its full resolution one (``"0"``), each half the
    size of the previous along the axes given by :func:`downscale_factors`, and update its ``multiscales`` metadata.
    Levels left by a previous run are replaced (or removed if ``levels`` is lower now). Fewer levels are created if
    the image gets too small. The image is processed chunk by chunk, so it does not need to fit in memory.

    Parameters
    ----------
    path : str
        Path to the OME-Zarr image.

    levels : int
        Number of resolution levels the image must have, including the full resolution one.
    """
    group = zarr.open_group(path, mode="r+")
    info, src = read_image(group, 0)
    if info.levels[0] != "0":
        raise ValueError("Can not build the pyramid of {}: its full resolution level is not '0'".format(path))
    scales = [list(info.scale)]
    for level in range(1, max(1, int(levels))):
        factors = downscale_factors(info.axes, src.shape, scales[-1])
        if all(f == 1 for f in factors):
            break
        shape = tuple(-(-s // f) for s, f in zip(src.shape, factors))
        dst = group.create_array(
            str(level),
            shape=shape,
            chunks=tuple(min(c, s) for c, s in zip(group["0"].chunks, shape)),
            dtype=src.dtype,
            overwrite=True,
            **_array_kwargs(info.axes, info.version),
        )
        for block in np.ndindex(*dst.cdata_shape):
            out = tuple(slice(b * c, min((b + 1) * c, s)) for b, c, s in zip(block, dst.chunks, shape))
            block_src = src[tuple(slice(o.start * f, o.stop * f) for o, f in zip(out, factors))]
            dst[out] = _downsample(np.asarray(block_src), factors)
        src = dst
        scales.append([v * f for v, f in zip(scales[-1], factors)])
    for stale in info.levels[len(scales) :]:
        del group[stale]
    _update_ome_attrs(group, multiscales_metadata(info.axes, scales, info.units, info.name, info.version))


def _update_ome_attrs(group: zarr.Group, attrs: Dict):
    """Update the OME metadata of a group keeping the rest of it (e.g. ``image-label`` or ``omero``)."""
    if "ome" in attrs:
        attrs = {"ome": {**dict(group.attrs).get("ome", {}), **attrs["ome"]}}
    group.attrs.update(attrs)


def write_label(
    path: str,
    name: str,
    data: np.ndarray,
    axes: str,
    scale: Optional[Sequence[float]] = None,
    units: Optional[Sequence[Optional[str]]] = None,
    pyramid_levels: int = 1,
):
    """
    Add a label image (e.g. the ground truth of the image) to the ``labels`` group of an OME-Zarr image, where BiaPy
    finds it with ``DATA.*.INPUT_ZARR_MULTIPLE_DATA_GT_PATH`` set to ``"labels.<name>"``.

    Parameters
    ----------
    path : str
        Path to the OME-Zarr image.

    name : str
        Name of the label image.

    data : NDArray
        Labels. They must be integers.

    axes : str
        Axes of ``data``, e.g. ``"ZYX"``.

    scale : tuple of float, optional
        Voxel size along each axis of ``data``.

    units : tuple of str or None, optional
        Unit of each axis of ``data``.

    pyramid_levels : int, optional
        Number of resolution levels to write (see :func:`build_pyramid`).
    """
    info = image_info(path)
    if info is None:
        raise ValueError("{} is not an OME-Zarr image".format(path))
    if not np.issubdtype(data.dtype, np.integer):
        raise ValueError("Label images must be of an integer type, but '{}' is {}".format(name, data.dtype))
    labels = zarr.open_group(path, mode="r+").require_group("labels")
    names = [n for n in _ome_attrs(labels).get("labels", []) if n != name] + [name]
    meta = {"labels": names}
    _update_ome_attrs(labels, {"ome": {"version": info.version, **meta}} if info.version == "0.5" else meta)
    label_path = os.path.join(path, "labels", name)
    write_image(label_path, data, axes, scale, units, info.version, pyramid_levels=pyramid_levels)
    image_label = {"source": {"image": "../../"}}
    if info.version == "0.5":
        _update_ome_attrs(zarr.open_group(label_path, mode="r+"), {"ome": {"image-label": image_label}})
    else:
        zarr.open_group(label_path, mode="r+").attrs.update({"image-label": {"version": info.version, **image_label}})
