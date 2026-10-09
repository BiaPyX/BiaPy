"""
Check of the wizard's interface with the front-ends (BiaPy-GUI, the Fiji plugin): what ``biapy.wizard.INTERFACE``
lists, with the keys of the results the front-ends get, must match the snapshot in ``tests/wizard_api.json``.

When they differ the test fails and says what changed:

* Additions (an optional argument, a result key, a function...): front-ends keep working. Update the snapshot.
* Anything else (something removed or renamed, an argument that becomes required...): front-ends written for the
  current version would break. Increase ``API_VERSION`` in ``biapy/wizard/__init__.py``, update the snapshot and
  the front-ends (they accept only the version they were written for).

Not everything is checked: the meaning of the values, or of the answers the functions get, can change without the
snapshot noticing. Such a change also needs ``API_VERSION`` increased.

How to run
----------
With the BiaPy environment active, from the repository root (no GPU, network or data needed)::

    pytest tests/test_wizard_api.py -q
    python tests/test_wizard_api.py --update    # writes tests/wizard_api.json from the current code
"""
import importlib
import inspect
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_wizard as tw  # noqa: E402  (synthetic datasets)

SNAPSHOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wizard_api.json")
# Results the front-ends pass along without reading them (only the wizard does): their keys are not part of the
# interface, except for the ones listed
OPAQUE = {"fingerprint": ("representations", "morphology_summary"), "plan": ()}


def _resolve(name):
    module, attr = name.rsplit(".", 1)
    return getattr(importlib.import_module("biapy.wizard." + module), attr)


def signatures():
    """Kind of each object of INTERFACE and, for functions, their parameters: [name, kind, has a default]."""
    import biapy.wizard

    out = {}
    for name in biapy.wizard.INTERFACE:
        obj = _resolve(name)
        if inspect.isclass(obj):
            out[name] = {"kind": "class"}
        elif callable(obj):
            params = [
                [p.name, p.kind.name, p.default is not inspect.Parameter.empty]
                for p in inspect.signature(obj).parameters.values()
            ]
            out[name] = {"kind": "function", "parameters": params}
        else:
            out[name] = {"kind": type(obj).__name__}
    return out


def _paths(obj, prefix, out):
    """Paths of the keys of a result: "a/b" for nested dicts, "a/[]/c" for the dicts of a list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = "{}/{}".format(prefix, k)
            out.add(path)
            if k in OPAQUE:
                v = {kk: vv for kk, vv in v.items() if kk in OPAQUE[k]} if isinstance(v, dict) else None
            _paths(v, path, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            if isinstance(v, (dict, list, tuple)):
                _paths(v, prefix + "/[]", out)


def results(tmp_path):
    """Keys of the results of the INTERFACE functions on small synthetic datasets."""
    from biapy.wizard.config_builder import build_config, to_jsonable
    from biapy.wizard.data_check import check_folder
    from biapy.wizard.instance_seg import REPRESENTATIONS
    from biapy.wizard.preview import preview

    out = set()
    masks = tw._instances(tmp_path, "round")
    raw = tw._raw_like(tmp_path, masks)
    points = tw._points(tmp_path, False)
    classes = tmp_path / "classes"
    for c in ("a", "b"):
        (classes / c).mkdir(parents=True)
        tw._write_2d(str(classes / c / "0.tif"), np.zeros((32, 32), np.uint8))

    checks = [
        ("raw images", raw, "DATA.TRAIN.PATH", "INSTANCE_SEG"),
        ("instance masks", masks, "DATA.TRAIN.GT_PATH", "INSTANCE_SEG"),
        ("detection points", points, "DATA.TRAIN.GT_PATH", "DETECTION"),
        ("classification", str(classes), "DATA.TRAIN.PATH", "CLASSIFICATION"),
        ("error", str(tmp_path / "missing"), "DATA.TRAIN.PATH", "SEMANTIC_SEG"),
    ]
    for name, folder, key, workflow in checks:
        _paths(to_jsonable(check_folder(folder, key, workflow, "2D")), "check_folder({})".format(name), out)
    _paths(preview("INSTANCE_SEG", "2D", raw, masks), "preview(instance masks)", out)
    _paths(preview("DETECTION", "2D", raw, points), "preview(detection points)", out)

    answers = tw._answers("INSTANCE_SEG", "2D", [("DATA.TRAIN.PATH", raw), ("DATA.TRAIN.GT_PATH", masks)],
                          instance_representation="Db_R")
    _, report = build_config(answers, tw.GPU)
    _paths(to_jsonable(report), "build_config/report", out)
    # The representation ids are in the front-ends' questions
    _paths({k: dict(v) for k, v in REPRESENTATIONS.items()}, "REPRESENTATIONS", out)
    return sorted(out)


def current(tmp_path):
    import biapy.wizard

    return {"API_VERSION": biapy.wizard.API_VERSION, "signatures": signatures(), "results": results(tmp_path)}


def compare(old, new):
    """Differences between two snapshots: (breaking, additions), as lists of messages."""
    breaking, additions = [], []
    for name, o in old["signatures"].items():
        n = new["signatures"].get(name)
        if n is None:
            breaking.append("{} removed from INTERFACE".format(name))
            continue
        if o["kind"] != n["kind"]:
            breaking.append("{} is now a {} (it was a {})".format(name, n["kind"], o["kind"]))
            continue
        if o["kind"] != "function":
            continue
        old_params = {p[0]: (i, p) for i, p in enumerate(o["parameters"])}
        new_params = {p[0]: (i, p) for i, p in enumerate(n["parameters"])}
        for pname, (i, p) in old_params.items():
            if pname not in new_params:
                breaking.append("{}: argument '{}' removed or renamed".format(name, pname))
                continue
            j, q = new_params[pname]
            if p[2] and not q[2]:
                breaking.append("{}: argument '{}' is now required".format(name, pname))
            if i != j and p[1] != "KEYWORD_ONLY":
                breaking.append("{}: argument '{}' moved from position {} to {}".format(name, pname, i, j))
        for pname, (j, q) in new_params.items():
            if pname not in old_params:
                if q[2] or q[1] in ("VAR_POSITIONAL", "VAR_KEYWORD"):
                    additions.append("{}: new optional argument '{}'".format(name, pname))
                else:
                    breaking.append("{}: new required argument '{}'".format(name, pname))
    for name in new["signatures"]:
        if name not in old["signatures"]:
            additions.append("{} added to INTERFACE".format(name))
    old_results, new_results = set(old["results"]), set(new["results"])
    breaking += ["result key removed: {}".format(p) for p in sorted(old_results - new_results)]
    additions += ["result key added: {}".format(p) for p in sorted(new_results - old_results)]
    return breaking, additions


UPDATE = "python tests/test_wizard_api.py --update"


def test_wizard_api(tmp_path):
    with open(SNAPSHOT) as f:
        old = json.load(f)
    new = current(tmp_path)
    breaking, additions = compare(old, new)
    if not breaking and not additions and old["API_VERSION"] == new["API_VERSION"]:
        return
    changes = "\n".join("  - " + m for m in breaking + additions) or "  (none besides API_VERSION)"
    if breaking and new["API_VERSION"] <= old["API_VERSION"]:
        pytest.fail(
            "The wizard's interface changed in a way that breaks the front-ends (BiaPy-GUI, the Fiji plugin):\n{}\n"
            "Increase API_VERSION in biapy/wizard/__init__.py, run '{}' and update the front-ends.".format(changes, UPDATE)
        )
    pytest.fail("The wizard's interface changed:\n{}\nUpdate tests/wizard_api.json with '{}'.".format(changes, UPDATE))


def update():
    import tempfile
    from pathlib import Path

    new = current(Path(tempfile.mkdtemp()))
    if os.path.exists(SNAPSHOT):
        with open(SNAPSHOT) as f:
            old = json.load(f)
        breaking, _ = compare(old, new)
        if new["API_VERSION"] < old["API_VERSION"] or (breaking and new["API_VERSION"] == old["API_VERSION"]):
            sys.exit(
                "Not updated: these changes break the front-ends, increase API_VERSION first:\n"
                + "\n".join("  - " + m for m in breaking)
            )
    with open(SNAPSHOT, "w") as f:
        json.dump(new, f, indent=1, sort_keys=True)
        f.write("\n")
    print("{} written (API_VERSION {})".format(SNAPSHOT, new["API_VERSION"]))


if __name__ == "__main__":
    if sys.argv[1:] == ["--update"]:
        update()
    else:
        sys.exit("Usage: python tests/test_wizard_api.py --update  (run the check with pytest)")
