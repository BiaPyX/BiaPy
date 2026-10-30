"""
Tests of the ViT backbones ('MODEL.VIT_MODEL' / 'MODEL.UNETR_VIT_MODEL') of the 'vit', 'unetr', 'dpt' and 'vit_readout'
architectures (biapy/models/vit_backbones, biapy/models/vit_decoders): every backbone builds with every architecture and
dimensionality it supports, the rest are rejected, and the model source exported to the BioImage Model Zoo rebuilds
the same network.

How to run
----------
No GPU, data, network or pretrained weights needed: the foundation backbones (ViT-L/H, SAM 3, DINOv3, Cellpose-SAM...)
are built on PyTorch's 'meta' device, which checks their layers, parameter names and output shapes without allocating
or computing anything. From the repository root, with the BiaPy environment active::

    pytest tests/test_vit_backbones.py -q
    pytest tests/test_vit_backbones.py -k dinov3 -v      # only one backbone

The pretrained weights themselves are loaded in the end-to-end checks of tests/run_checks.py.
"""
import types

import pytest
import torch

import biapy.models
from biapy.config.config import Config
from biapy.models import adapt_bmz_model_kwargs, build_model
from biapy.models.vit_backbones.common import UNETR_PATCH_SIZE, VIT_BACKBONES, check_vit_backbone

ARCHS = ["vit", "unetr", "dpt", "vit_readout"]
# 'dpt' and 'vit_readout' are 2D only, whatever the backbone
ARCHS_3D = ["vit", "unetr"]
N_CLASSES = 3


def _token_size(arch, vit_model):
    if vit_model == "custom":
        return 16 if arch == "vit" else 8
    if arch == "unetr":
        return UNETR_PATCH_SIZE.get(vit_model, VIT_BACKBONES[vit_model]["patch_size"])
    return VIT_BACKBONES[vit_model]["patch_size"]


def _cases():
    """(arch, vit_model, ndim, in_chans) of every supported combination."""
    cases = []
    for arch in ARCHS:
        cases += [(arch, "custom", 2, 3)]
        if arch in ARCHS_3D:
            cases += [(arch, "custom", 3, 1)]
        for vit_model, preset in VIT_BACKBONES.items():
            if arch not in preset["archs"]:
                continue
            for ndim in preset["ndims"]:
                if ndim == 2 or arch in ARCHS_3D:
                    cases.append((arch, vit_model, ndim, preset["in_chans"] or 3))
    return cases


def _build(arch, vit_model, ndim, in_chans, device, **model_opts):
    """Model built by BiaPy as in a workflow, with its BMZ export source and kwargs."""
    cfg = Config("/tmp", "test_vit_backbones").get_cfg_defaults()
    cfg.PROBLEM.TYPE = "CLASSIFICATION" if arch == "vit" else "INSTANCE_SEG"
    cfg.PROBLEM.NDIM = f"{ndim}D"
    cfg.DATA.PATCH_SIZE = (4 * _token_size(arch, vit_model),) * ndim + (in_chans,)
    cfg.DATA.N_CLASSES = N_CLASSES
    cfg.MODEL.ARCHITECTURE = arch
    if arch == "unetr":
        cfg.MODEL.UNETR_VIT_MODEL = vit_model
    else:
        cfg.MODEL.VIT_MODEL = vit_model
    # Small "custom" ViT
    cfg.MODEL.VIT_TOKEN_SIZE = _token_size(arch, "custom")
    cfg.MODEL.VIT_EMBED_DIM = 96
    cfg.MODEL.VIT_NUM_LAYERS = 4
    cfg.MODEL.VIT_NUM_HEADS = 4
    for key, value in model_opts.items():
        node = cfg.MODEL
        *path, last = key.split(".")
        for p in path:
            node = node[p]
        node[last] = value

    if arch == "vit":
        outputs = ([N_CLASSES], ["class"], ["ce_softmax"] * N_CLASSES)
    else:
        outputs = ([3], ["F+Gv+Gh"], ["ce_sigmoid", "linear", "linear"])
    model, callable_name, sources, imports, _, kwargs, _ = build_model(cfg, *outputs, torch.device(device))
    return cfg, model.eval(), callable_name, sources, imports, kwargs


def _bmz_model(callable_name, sources, imports, kwargs):
    """Network built from the 'model.py' and kwargs that the BMZ export writes (see BiaPy.export_model_to_bmz)."""
    code = "\n".join(imports) + "\n\n" + "\n\n".join(sources.values())
    module = types.ModuleType("bmz_model")
    exec(compile(code, "model.py", "exec"), module.__dict__)
    kwargs = adapt_bmz_model_kwargs(adapt_bmz_model_kwargs(kwargs, model_to_consume=False), model_to_consume=True)
    return getattr(module, callable_name)(**kwargs).eval()


def _pred(out):
    return out["pred"] if isinstance(out, dict) else out


def _check_output(cfg, arch, out):
    batch = out.shape[0]
    if arch == "vit":
        assert tuple(out.shape) == (batch, N_CLASSES)
    else:
        assert tuple(out.shape) == (batch, 3) + tuple(cfg.DATA.PATCH_SIZE[:-1])


@pytest.fixture
def meta_build(monkeypatch):
    """
    Lets 'build_model' build on the 'meta' device, and faster:
      - 'torch.linspace' on the CPU unless told otherwise, as the ViTs (BiaPy's and timm's) read their stochastic depth
        rates with '.item()', which can not be called on 'meta' tensors.
      - No model summary: it is just printed, and a forward pass on 'meta' takes ~1 s with the large ViTs.
    """
    linspace = torch.linspace
    monkeypatch.setattr(torch, "linspace", lambda *args, **kwargs: linspace(*args, **{"device": "cpu", **kwargs}))
    monkeypatch.setattr(biapy.models, "summary", lambda *args, **kwargs: None)


@pytest.mark.parametrize("arch,vit_model,ndim,in_chans", _cases())
def test_backbone_builds(arch, vit_model, ndim, in_chans, meta_build):
    # Built on 'meta', so the foundation backbones cost nothing. The network rebuilt from the BMZ source must have the
    # same parameters. It runs the same code as BiaPy's, so only its forward pass is checked (each one takes ~1 s on
    # 'meta' with the large ViTs)
    with torch.device("meta"):
        cfg, model, callable_name, sources, imports, kwargs = _build(arch, vit_model, ndim, in_chans, "meta")
        bmz_model = _bmz_model(callable_name, sources, imports, kwargs)
        x = torch.empty((1, in_chans) + tuple(cfg.DATA.PATCH_SIZE[:-1]), device="meta")
        with torch.no_grad():
            _check_output(cfg, arch, _pred(bmz_model(x)))
    shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    assert shapes == {k: tuple(v.shape) for k, v in bmz_model.state_dict().items()}


@pytest.mark.parametrize(
    "arch,ndim,model_opts",
    [
        ("vit", 2, {}),
        ("vit", 3, {}),
        ("unetr", 2, {}),
        ("unetr", 3, {}),
        ("dpt", 2, {}),
        ("dpt", 2, {"VIT_TOKEN_STRIDE": 4, "VIT_DROP_PATH_RATE": 0.1, "DPT.READOUT": "add"}),
        ("vit_readout", 2, {"VIT_READOUT.FEATURES": 3, "VIT_READOUT.REFINE_LAYERS": 0, "VIT_READOUT.INPUT_SKIP": False}),
        ("vit_readout", 2, {"VIT_TOKEN_STRIDE": 4, "VIT_READOUT.REFINE_LAYERS": 2, "VIT_READOUT.INPUT_SKIP": True}),
    ],
)
def test_bmz_source_reproduces_output(arch, ndim, model_opts):
    # With real weights (a small "custom" ViT on CPU): the exported network gives the same output
    torch.manual_seed(0)
    in_chans = 3 if ndim == 2 else 1
    cfg, model, callable_name, sources, imports, kwargs = _build(arch, "custom", ndim, in_chans, "cpu", **model_opts)
    bmz_model = _bmz_model(callable_name, sources, imports, kwargs)
    bmz_model.load_state_dict(model.state_dict())

    x = torch.rand((2, in_chans) + tuple(cfg.DATA.PATCH_SIZE[:-1]))
    with torch.no_grad():
        out = _pred(model(x))
        _check_output(cfg, arch, out)
        assert torch.allclose(out, _pred(bmz_model(x)), atol=1e-5)


def _unsupported():
    cases = []
    for vit_model, preset in VIT_BACKBONES.items():
        in_chans = preset["in_chans"] or 3
        cases += [(vit_model, arch, 2, in_chans) for arch in ARCHS + ["mae"] if arch not in preset["archs"]]
        if 3 not in preset["ndims"]:
            cases.append((vit_model, preset["archs"][0], 3, in_chans))
        if preset["in_chans"] is not None:
            cases.append((vit_model, preset["archs"][0], 2, preset["in_chans"] + 2))
    cases.append(("not_a_backbone", "vit", 2, 3))
    return cases


@pytest.mark.parametrize("vit_model,arch,ndim,in_chans", _unsupported())
def test_unsupported_combination_rejected(vit_model, arch, ndim, in_chans):
    with pytest.raises(ValueError):
        check_vit_backbone(vit_model, arch, ndim, in_chans)
