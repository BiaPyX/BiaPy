"""
Search of the BioImage Model Zoo for the models BiaPy can use.

It is shared by BiaPy's notebooks and BiaPy-Fiji, so that they list the same models with the same information (name,
cover, representation...) and a fix or addition here reaches all of them:

* :func:`list_compatible_bmz_models`: the models compatible with a workflow, with what is shown of each one.
* :func:`bmz_models_html`: those models as an HTML table (e.g. for ``IPython.display.HTML`` in a notebook).
"""
import contextlib
import html
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional

from biapy.models import BMZ_HYPHA_FILES_URL, check_bmz_model_compatibility, find_bmz_models

# Page of a model in the BioImage Model Zoo website
BMZ_MODEL_PAGE_URL = "https://bioimage.io/#/artifacts/{}"

WORKFLOW_NAMES = {
    "SEMANTIC_SEG": "Semantic segmentation",
    "INSTANCE_SEG": "Instance segmentation",
    "DETECTION": "Object detection",
    "DENOISING": "Image denoising",
    "SUPER_RESOLUTION": "Super-resolution",
    "SELF_SUPERVISED": "Self-supervised learning",
    "CLASSIFICATION": "Image classification",
    "IMAGE_TO_IMAGE": "Image to image",
}


def bmz_cover_url(model: Dict) -> str:
    """
    URL of the first cover image of a model listed by :func:`biapy.models.find_bmz_models`.

    Parameters
    ----------
    model : dict
        Model as listed by :func:`biapy.models.find_bmz_models`.

    Returns
    -------
    url : str
        URL of the cover, or an empty string if the model has none.
    """
    manifest = (model.get("raw") or {}).get("manifest") or {}
    covers = manifest.get("covers") or []
    cover = covers[0] if covers else ""
    # A file name, or {"source": <file name or URL>, "sha256": ...} since format 0.5.12
    if isinstance(cover, dict):
        cover = cover.get("source") or ""
    cover = str(cover)
    if not cover or cover.startswith(("http://", "https://")):
        return cover
    return BMZ_HYPHA_FILES_URL.format(model.get("nickname")) + cover


def instance_representation_name(channels: List[str], ndim: str) -> str:
    """
    Name of the Wizard's instance representation with these BiaPy channels, e.g. "Foreground + contour" for
    ``["F", "C"]``.

    Parameters
    ----------
    channels : list of str
        BiaPy channels (``PROBLEM.INSTANCE_SEG.DATA_CHANNELS``).

    ndim : str
        ``"2D"`` or ``"3D"``.

    Returns
    -------
    name : str
        Name of the representation, or an empty string if none of the Wizard's has these channels.
    """
    from biapy.wizard.instance_seg import REPRESENTATIONS

    for r in REPRESENTATIONS.values():
        ch = list(r["channels"]) + (list(r.get("channels_3d_extra", [])) if ndim == "3D" else [])
        if ch == list(channels):
            return r.get("name", "")
    return ""


def representation_label(info: Dict) -> str:
    """
    Instance representation of a model found by :func:`list_compatible_bmz_models`, e.g.
    "Foreground + flows (Cellpose) (F, Gv, Gh)", or just its channels when it is not one of the Wizard's.

    Parameters
    ----------
    info : dict
        Model as returned by :func:`list_compatible_bmz_models`.

    Returns
    -------
    label : str
        The representation, or an empty string in other workflows than instance segmentation.
    """
    if not info.get("channels"):
        return ""
    codes = ", ".join(info["channels"])
    return f"{info['representation']} ({codes})" if info.get("representation") else f"Channels {codes}"


def bmz_model_info(model: Dict, compat: Dict) -> Dict:
    """
    What is shown of a compatible model.

    Parameters
    ----------
    model : dict
        Model as listed by :func:`biapy.models.find_bmz_models`.

    compat : dict
        Its report from :func:`biapy.models.check_bmz_model_compatibility`.

    Returns
    -------
    info : dict
        ``id`` (its nickname, what ``MODEL.BMZ.SOURCE_MODEL_ID`` takes), ``nickname``, ``nickname_icon`` (emoji),
        ``name``, ``description``, ``tags``, ``cover`` (URL), ``url`` (its page in the BioImage Model Zoo),
        ``workflow_type`` and ``ndim`` of the model, ``opts`` (configuration it imposes), and in instance segmentation
        the BiaPy ``channels`` it outputs and the name of that ``representation`` (empty if it is not one of the
        Wizard's).
    """
    manifest = (model.get("raw") or {}).get("manifest") or {}
    nickname = model.get("nickname") or model.get("alias") or ""
    workflow_info = compat.get("workflow_info") or {}
    opts = compat.get("opts") or {}
    channels = [str(c) for c in opts.get("PROBLEM.INSTANCE_SEG.DATA_CHANNELS") or []]
    ndim = workflow_info.get("ndim", "")
    representation = ""
    if channels:
        try:
            representation = instance_representation_name(channels, ndim)
        except Exception:
            pass
    return {
        "id": nickname,
        "nickname": nickname,
        "nickname_icon": manifest.get("id_emoji", ""),
        "name": manifest.get("name", ""),
        "description": manifest.get("description", ""),
        "tags": [str(t) for t in manifest.get("tags") or []],
        "cover": bmz_cover_url(model),
        "url": BMZ_MODEL_PAGE_URL.format(nickname),
        "workflow_type": workflow_info.get("workflow_type", ""),
        "ndim": ndim,
        "opts": opts,
        "channels": channels,
        "representation": representation,
    }


def _sort_key(info: Dict):
    """Grouped by representation (alphabetically, those that are not one of the Wizard's last), then by nickname."""
    label = representation_label(info)
    return (not label, not info.get("representation"), label.lower(), info["nickname"])


def list_compatible_bmz_models(
    workflow: str = "all",
    ndim: str = "all",
    nclasses="all",
    progress: bool = False,
    quiet: bool = True,
    max_workers: int = 8,
    callback: Optional[Callable[[int, int, Optional[Dict], str, str], None]] = None,
) -> List[Dict]:
    """
    The models of the BioImage Model Zoo that BiaPy can use for a workflow.

    Models are checked in parallel, as most of the time is spent waiting for the network.

    Parameters
    ----------
    workflow : str
        Workflow (``PROBLEM.TYPE``), or ``"all"``.

    ndim : str
        ``"2D"``, ``"3D"`` or ``"all"``.

    nclasses : int or str
        Classes of the data (``DATA.N_CLASSES``), or ``"all"``.

    progress : bool
        Whether to show a progress bar (tqdm, also in notebooks).

    quiet : bool
        Whether to hide what the checks print about each model (and bioimageio's download bars). It hides everything
        printed meanwhile, also to stderr, so leave it ``False`` if ``callback`` prints.

    max_workers : int
        Models checked at the same time.

    callback : callable, optional
        Called after each model is checked as ``callback(done, total, info, nickname, reason)``: ``info`` is what
        :func:`bmz_model_info` returns if the model is compatible, otherwise ``None`` and ``reason`` says why.

    Returns
    -------
    models : list of dict
        The compatible models (see :func:`bmz_model_info`), grouped by instance representation and then sorted by
        nickname.
    """
    workflow_specs = {"workflow_type": workflow, "ndim": ndim, "nclasses": nclasses}
    models = find_bmz_models()
    total = len(models)

    def check(model):
        nickname = model.get("nickname") or model.get("alias") or "?"
        try:
            compat = check_bmz_model_compatibility(model, workflow_specs=workflow_specs)
        except Exception as e:
            return nickname, None, str(e)
        if compat["error"]:
            return nickname, None, str(compat["reason_message"]).strip()
        return nickname, bmz_model_info(model, compat), ""

    bar = None
    if progress:
        from tqdm.auto import tqdm

        bar = tqdm(total=total, desc="Analyzing models from the Zoo")
    out = []
    with contextlib.ExitStack() as silence:
        if quiet:  # the notes of the checks and the download bars of bioimageio (the progress bar keeps its output)
            silence.enter_context(contextlib.redirect_stdout(io.StringIO()))
            silence.enter_context(contextlib.redirect_stderr(io.StringIO()))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(check, m) for m in models]
            for done, fut in enumerate(as_completed(futures), start=1):
                nickname, info, reason = fut.result()
                if info is not None:
                    out.append(info)
                if bar is not None:
                    bar.update(1)
                if callback is not None:
                    callback(done, total, info, nickname, reason)
    if bar is not None:
        bar.close()
    return sorted(out, key=_sort_key)


def bmz_models_html(models: List[Dict], columns: int = 3) -> str:
    """
    The models found by :func:`list_compatible_bmz_models` as HTML: a title and a table of them (name, nickname,
    workflow, description, representation and cover), with a header for each instance representation.

    Parameters
    ----------
    models : list of dict
        Models as returned by :func:`list_compatible_bmz_models`.

    columns : int
        Models per row.

    Returns
    -------
    html : str
        The HTML, e.g. for ``IPython.display.HTML``.
    """
    if not models:
        return "<h1>No BMZ models compatible with BiaPy were found for this task.</h1>"
    e = html.escape
    out = [f"<h1>List of models that can be used in BiaPy ({len(models)}):</h1>"]
    out.append("<table style='width:100%; table-layout:fixed'>")
    grouped = any(representation_label(m) for m in models)
    group, col = None, 0
    for m in sorted(models, key=_sort_key):
        label = representation_label(m)
        if grouped and label != group:
            if col:
                out.append("</tr>")
                col = 0
            group = label
            n = sum(1 for o in models if representation_label(o) == label)
            title = f"{e(label or 'Other workflows')} &mdash; {n} model{'s' if n != 1 else ''}"
            out.append(
                f"<tr><td colspan='{columns}' style='text-align:left'><h3 style='color:#2196f3; margin:16px 0 0 0;"
                f" border-bottom:2px solid #c5eafa'>{title}</h3></td></tr>"
            )
        if col == 0:
            out.append("<tr>")
        workflow = WORKFLOW_NAMES.get(m["workflow_type"], m["workflow_type"])
        cover = f"<img src='{e(m['cover'])}' style='max-height:150px; max-width:100%'>" if m["cover"] else ""
        rep = f"<p><b>Representation:</b> {e(label)}</p>" if label else ""
        out.append(
            f"<td style='width:{100 // columns}%; vertical-align:top; text-align:left'>"
            f"<p style='color:#2196f3'>{e(m['name'])}</p>"
            f"<p>Nickname: <b>{e(m['nickname'])}</b> {e(m['nickname_icon'])}</p>"
            f"<p>{e(workflow)} ({e(m['ndim'])})</p>"
            f"<details><summary>Description</summary><p>{e(m['description'])}</p></details>"
            f"{rep}{cover}</td>"
        )
        col += 1
        if col == columns:
            out.append("</tr>")
            col = 0
    if col:
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)
