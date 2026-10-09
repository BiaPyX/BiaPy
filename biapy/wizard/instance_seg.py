"""
Strategy for the instance segmentation workflow.

* Representation of the instances (what the network learns to predict and the instances are rebuilt from):
  the Wizard offers the ones that suit the shape of the instances in the training masks
  (propose_representations()), and the one chosen is translated into BiaPy's channels
  (representation_to_biapy()).
* Training: as for semantic segmentation (nnU-Net-like U-Net, patch and batch size, foreground
  oversampling, no elastic deformations), with the losses BiaPy sets for each channel.
"""
from . import common

# Representations offered in the Wizard: id -> name, description, BiaPy channels ("Gz"/"Z" are added in 3D) and
# how BiaPy builds the instances from them ("creation", set explicitly: BiaPy 3.7.1 picks "watershed" for 'R'
# when left empty, as its "Gv" check is an "if" instead of an "elif" in check_configuration.py)
REPRESENTATIONS = {
    "F_C": {
        "name": "Foreground + contour",
        "description": "Binary mask of the instances plus their contours, which separate touching instances. "
        "General purpose.",
        "channels": ["F", "C"],
    },
    "F_P": {
        "name": "Foreground + central point",
        "description": "Binary mask plus a point in the center of each instance, used as seed to grow it. For "
        "compact, roundish instances.",
        "channels": ["F", "P"],
        "extra": {"P": {"type": "centroid"}},
    },
    "F_Db": {
        "name": "Foreground + distance to the border",
        "description": "Binary mask plus, inside each instance, the distance to its border, which peaks once in "
        "the center of convex instances.",
        "channels": ["F", "Db"],
    },
    "F_S": {
        "name": "Foreground + skeleton",
        "description": "Binary mask plus the skeleton of each instance, as seed. For elongated, thin or branched "
        "instances (e.g. neurites, filaments, bacteria).",
        "channels": ["F", "P"],
        "extra": {"P": {"type": "skeleton"}},
    },
    "Db_R": {
        "name": "Distance to the border + radial distances (StarDist)",
        "description": "Each instance as a polygon/polyhedron from its center: distances to its border along "
        "fixed rays. Very good for crowded star-convex instances such as nuclei.",
        "channels": ["Db", "R"],
        "creation": "stardist",
    },
    "F_G": {
        "name": "Foreground + flows (Cellpose)",
        "description": "Binary mask plus a flow field that points to the center of each instance; pixels "
        "following it to the same center form an instance. Handles irregular shapes.",
        "channels": ["F", "Gv", "Gh"],
        "channels_3d_extra": ["Gz"],
        "creation": "gradient-flow",
    },
    "F_HV": {
        "name": "Foreground + horizontal and vertical distances (HoVer-Net)",
        "description": "Binary mask plus the horizontal and vertical distances of each pixel to the center of "
        "its instance; their sharp changes separate touching instances. Designed for nuclei.",
        "channels": ["F", "H", "V"],
        "channels_3d_extra": ["Z"],
    },
    "A": {
        "name": "Affinities",
        "description": "For each voxel, whether its neighbours along Z, Y and X belong to the same instance; the "
        "instances are built by merging the voxels joined by high affinities. For dense 3D instances that fill the "
        "volume, such as neurons in electron microscopy.",
        "channels": ["A"],
        "creation": "agglomeration",
        "only_3d": True,
    },
}

# Below this number of instances the shape is not analyzed: all representations are offered
MIN_INSTANCES = 10
# Compact instances: convex enough and not elongated, so that a central point or a distance peak marks them
COMPACT_SOLIDITY = 0.85
COMPACT_MAX_ELONGATED = 0.3  # max. fraction of elongated instances
CENTROID_INSIDE = 0.95  # min. fraction of instances whose centroid falls inside them
# Elongated / branched / irregular: the skeleton represents them better than a point
SKELETON_ELONGATED = 0.3  # min. fraction of elongated instances
SKELETON_BRANCHED = 0.2  # min. fraction of branched instances
SKELETON_SOLIDITY = 0.75  # max. median solidity
# StarDist assumes star-convex instances
STAR_CONVEX = 0.95
# Cellpose's flows struggle with thin branched structures
FLOWS_MAX_BRANCHED = 0.2
# Crowded: many instances touch each other
CROWDED = 0.3
# Instances this thin (inscribed radius in pixels) are mostly contour
THIN_RADIUS = 2.0
# Radial distances predicted per pixel with BiaPy's default number of rays (memory planning)
STARDIST_RAYS = {2: 32, 3: 96}
# Dense instances (affinities, 3D only): they fill most of the volume and nearly all touch others, as neurons in
# electron microscopy
DENSE_FOREGROUND = 0.5  # min. median foreground fraction
DENSE_TOUCHING = 0.8  # min. fraction of instances touching others


def _pct(v):
    return "{:.0%}".format(v)


def describe_morphology(fp):
    """One-line summary of the shape of the instances of a masks fingerprint."""
    m = fp.get("morphology")
    if not m:
        return "{} instances found".format(fp.get("n_instances", 0))
    return (
        "{} instances measured: median radius {:.1f} px, median solidity {:.2f}, {} elongated, {} branched, "
        "{} star-convex, {} touching others; they cover {} of the images".format(
            m["n_measured"], m["inscribed_radius_median"], m["solidity_median"], _pct(m["elongated_fraction"]),
            _pct(m["branched_fraction"]), _pct(m["star_convex_fraction"]), _pct(fp.get("touching_fraction", 0.0)),
            _pct(fp["foreground_fraction"]["median"]),
        )
    )


def propose_representations(fp, ndim):
    """
    Representations that suit the instances of a masks fingerprint (fingerprint_instance_masks()).

    Recommended, in this order: affinities for dense 3D instances, the skeleton for elongated or branched
    instances, StarDist for star-convex ones, Cellpose's flows for crowded instances that are not star-convex,
    and foreground + contour otherwise.

    Returns
    -------
    options : list of dict
        One per representation, in REPRESENTATIONS order: ``id``, ``available`` (offered in the Wizard),
        ``recommended`` (exactly one) and ``reason`` (based on the data).
    """
    m = fp.get("morphology")
    if not m or m["n_measured"] < MIN_INSTANCES:
        n = m["n_measured"] if m else fp.get("n_instances", 0)
        reason = "Only {} instances found: too few to analyze their shape.".format(n)
        return [
            {"id": rid, "available": ndim == 3 or not REPRESENTATIONS[rid].get("only_3d"), "recommended": rid == "F_C",
             "reason": reason if ndim == 3 or not REPRESENTATIONS[rid].get("only_3d") else "Only for 3D data."}
            for rid in REPRESENTATIONS
        ]

    touching = fp.get("touching_fraction", 0.0)
    solidity = "median solidity {:.2f}".format(m["solidity_median"])
    elong = "{} elongated".format(_pct(m["elongated_fraction"]))
    compact = m["solidity_median"] >= COMPACT_SOLIDITY and m["elongated_fraction"] < COMPACT_MAX_ELONGATED
    centroid_ok = m["centroid_inside_fraction"] >= CENTROID_INSIDE
    elongated = m["elongated_fraction"] >= SKELETON_ELONGATED
    branched = m["branched_fraction"] >= SKELETON_BRANCHED
    irregular = m["solidity_median"] < SKELETON_SOLIDITY
    star = m["star_convex_fraction"] >= STAR_CONVEX
    flows_ok = m["branched_fraction"] < FLOWS_MAX_BRANCHED
    thin = m["inscribed_radius_median"] < THIN_RADIUS

    opts = {}
    opts["F_C"] = (True, "Works for any shape." + (
        " Your instances are very thin (median radius {:.1f} px), so contours will cover much of them.".format(
            m["inscribed_radius_median"]) if thin else ""))
    if compact and centroid_ok:
        opts["F_P"] = (True, "Your instances are compact ({}, {}) with the centroid inside.".format(solidity, elong))
    else:
        opts["F_P"] = (False, "Needs compact instances with the centroid inside ({}, {}, centroid inside {}).".format(
            solidity, elong, _pct(m["centroid_inside_fraction"])))
    opts["F_Db"] = (compact, ("Your instances are compact ({}, {}): the distance peaks once in each." if compact else
                              "Needs compact instances, otherwise the distance peaks several times in each ({}, {}).")
                    .format(solidity, elong))
    if elongated or branched or irregular:
        why = [w for w, c in ((elong, elongated), ("{} branched".format(_pct(m["branched_fraction"])), branched),
                              (solidity, irregular)) if c]
        opts["F_S"] = (True, "Your instances are elongated, branched or irregular ({}).".format(", ".join(why)))
    else:
        opts["F_S"] = (False, "For elongated, branched or irregular instances; yours are not ({}, {}, {} branched).".format(
            solidity, elong, _pct(m["branched_fraction"])))
    opts["Db_R"] = (star, ("{} of your instances are star-convex, as StarDist assumes." if star else
                           "Needs star-convex instances; {} of yours are.").format(_pct(m["star_convex_fraction"])))
    opts["F_G"] = (flows_ok, "Handles irregular and crowded instances." if flows_ok else
                   "The flows struggle with branched instances ({} of yours).".format(_pct(m["branched_fraction"])))
    opts["F_HV"] = (compact and centroid_ok,
                    "Your instances are compact ({}), like the nuclei HoVer-Net was designed for.".format(solidity)
                    if compact and centroid_ok else
                    "Designed for compact instances such as nuclei ({}, {}).".format(solidity, elong))

    fg = fp["foreground_fraction"]["median"]
    dense = fg >= DENSE_FOREGROUND and touching >= DENSE_TOUCHING
    if ndim != 3:
        opts["A"] = (False, "Only for 3D data (dense instances such as neurons in electron microscopy).")
    elif dense:
        opts["A"] = (True, "Your instances are dense: they cover {} of the volume and {} touch others, as in "
                           "electron microscopy.".format(_pct(fg), _pct(touching)))
    else:
        opts["A"] = (False, "For dense instances that fill most of the volume; yours cover {} of it and {} touch "
                            "others.".format(_pct(fg), _pct(touching)))

    if opts["A"][0]:
        rec = "A"
    elif opts["F_S"][0] and (elongated or branched):
        rec = "F_S"
    elif star:
        rec = "Db_R"
    elif touching >= CROWDED and flows_ok:
        rec = "F_G"
    else:
        rec = "F_C"
    return [
        {"id": rid, "available": opts[rid][0], "recommended": rid == rec, "reason": opts[rid][1]} for rid in REPRESENTATIONS
    ]


def representation_to_biapy(rid, ndim):
    """BiaPy DATA_CHANNELS and DATA_CHANNELS_EXTRA_OPTS of a representation."""
    r = REPRESENTATIONS[rid]
    channels = list(r["channels"]) + (list(r.get("channels_3d_extra", [])) if ndim == 3 else [])
    return channels, [dict(r.get("extra", {}))]


def instance_creation(rid):
    """BiaPy's INSTANCE_CREATION_PROCESS for a representation."""
    return REPRESENTATIONS[rid].get("creation", "watershed")


def representation_of(channels, extra, ndim):
    """Id of the representation with these BiaPy channels and options, or None."""
    for rid in REPRESENTATIONS:
        ch, ex = representation_to_biapy(rid, ndim)
        if list(channels) == ch and (dict(extra[0]) if extra else {}) == ex[0]:
            return rid
    return None


def output_channels(channels, ndim, n_classes):
    """
    Channels predicted by the network (each ray of 'R' is one, 'A' has one per axis with BiaPy's default
    offsets), plus the classes if any.
    """
    n = sum(STARDIST_RAYS[ndim] if c == "R" else ndim if c == "A" else 1 for c in channels)
    return n + (n_classes if n_classes > 2 else 0)


def plan_instance_seg(cfg, sample_info, vram_bytes, vram_source):
    """Applies the strategy to ``cfg``. Returns a report as plan_semantic_seg()."""
    reason = common.not_plannable_reason(cfg)
    if reason:
        return {"applied": False, "reason": reason}
    raw_fp = common.fingerprint(sample_info, "DATA.TRAIN.PATH", "images")
    mask_fp = common.fingerprint(sample_info, "DATA.TRAIN.GT_PATH", "instance_masks")
    if raw_fp is None or mask_fp is None or not mask_fp.get("n_instances"):
        return {"applied": False, "reason": "the training data was not analyzed"}

    ndim = 3 if cfg["PROBLEM"]["NDIM"] == "3D" else 2
    lines = []
    spacing, spacing_source = common.choose_spacing(raw_fp, mask_fp.get("objects"), ndim)
    common.describe_data(raw_fp, spacing, spacing_source, lines)
    m = mask_fp.get("morphology", {})
    lines.append(
        "Instances: {} (median {:.0f} per image), median size {}, {} touching others".format(
            mask_fp["n_instances"], mask_fp["instances_per_image"]["median"],
            common.fmt(mask_fp["objects"]["instances"]["extent_median"]), _pct(mask_fp.get("touching_fraction", 0)),
        )
    )
    if m:
        lines.append(
            "Shape: median solidity {:.2f}, {} elongated, {} branched, {} star-convex".format(
                m["solidity_median"], _pct(m["elongated_fraction"]), _pct(m["branched_fraction"]),
                _pct(m["star_convex_fraction"]),
            )
        )

    inst = cfg["PROBLEM"].setdefault("INSTANCE_SEG", {})
    channels = inst.get("DATA_CHANNELS")
    if isinstance(channels, str):  # BiaPy-GUI's old format, e.g. "BC"
        channels = ["F", "C"]
    rep = representation_of(channels, inst.get("DATA_CHANNELS_EXTRA_OPTS"), ndim)
    if rep:
        lines.append("Representation: {} (channels {})".format(REPRESENTATIONS[rep]["name"], ", ".join(channels)))
    n_classes = int(cfg["DATA"].get("N_CLASSES", 2))
    plan = common.apply_unet_plan(
        cfg, raw_fp, spacing, output_channels(channels, ndim, n_classes), vram_bytes, vram_source, lines
    )
    if plan is None:
        return {"applied": False, "reason": "the images are too small to plan a U-Net", "lines": lines}
    lines.append("Losses: BiaPy's default for each channel; no elastic deformations")
    common.warn_objects_larger_than_patch(mask_fp.get("objects"), plan["patch_size"], lines, what="{}")
    common.foreground_oversampling(cfg, raw_fp, mask_fp["foreground_fraction"]["median"], plan["patch_size"], lines)
    return {"applied": True, "lines": lines, "plan": plan, "spacing": spacing, "spacing_source": spacing_source}
