"""The cache of the BioImage Model Zoo search (biapy.models.bmz_search.CompatibilityCache), without the network."""
import biapy
from biapy.models import bmz_search


def _zoo(monkeypatch, changed):
    """A zoo of two models (``changed``: last change of each) and the checks made, counted by model."""
    models = [{"nickname": n, "raw": {"last_modified": t, "manifest": {"name": n}}} for n, t in changed.items()]
    monkeypatch.setattr(bmz_search, "find_bmz_models", lambda: models)
    checked = []

    def check(model, workflow_specs=None):
        checked.append(model["nickname"])
        if model["nickname"] == "bad":
            return {"error": True, "reason_message": "no", "opts": {}, "workflow_info": {}}
        return {"error": False, "reason_message": "", "opts": {"DATA.PATCH_SIZE": (64, 64, 1)},
                "workflow_info": {"workflow_type": "SEMANTIC_SEG", "ndim": "2D"}}

    monkeypatch.setattr(bmz_search, "check_bmz_model_compatibility", check)
    return checked


def test_cache(monkeypatch, tmp_path):
    cache = str(tmp_path / "sub" / "bmz.json")
    checked = _zoo(monkeypatch, {"good": 1, "bad": 1})
    first = bmz_search.list_compatible_bmz_models(cache=cache)
    assert sorted(checked) == ["bad", "good"] and len(first) == 1

    checked.clear()
    again = bmz_search.list_compatible_bmz_models(cache=cache)
    assert checked == [] and again == first  # everything from the cache
    assert again[0]["opts"]["DATA.PATCH_SIZE"] == (64, 64, 1)  # still a tuple, as the configuration needs
    assert bmz_search.list_compatible_bmz_models(cache=cache, nchannels=3) == []  # channels filtered after the cache

    checked = _zoo(monkeypatch, {"good": 2, "bad": 1})  # "good" changed in the zoo
    bmz_search.list_compatible_bmz_models(cache=cache)
    assert checked == ["good"]
    checked.clear()
    bmz_search.list_compatible_bmz_models(cache=cache, workflow="DENOISING")  # another search: checked for it
    assert sorted(checked) == ["bad", "good"]

    checked.clear()
    monkeypatch.setattr(biapy, "__version__", "0.0.1")  # another BiaPy: its checks may differ
    bmz_search.list_compatible_bmz_models(cache=cache)
    assert sorted(checked) == ["bad", "good"]


def test_unreadable_cache(monkeypatch, tmp_path):
    cache = tmp_path / "bmz.json"
    cache.write_text("{not json")
    checked = _zoo(monkeypatch, {"good": 1})
    assert len(bmz_search.list_compatible_bmz_models(cache=str(cache))) == 1 and checked == ["good"]
