"""biapy.models.download_bmz_model, without the network."""
import pytest
import requests

import biapy.models as models


class _Response:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.data


def _zoo(monkeypatch, files):
    """A zoo model with ``files`` ({name: bytes}); the files downloaded are counted."""
    got = []
    listing = [{"name": n, "size": len(d), "last_modified": 1} for n, d in files.items()]
    monkeypatch.setattr(models, "bmz_model_files", lambda model_id, timeout=30: listing)
    monkeypatch.setattr(models, "find_bmz_description_url", lambda model_id, timeout=30: "https://x/files/rdf.yaml")

    def get(url, stream=False, timeout=30):
        got.append(url.rsplit("/", 1)[-1])
        return _Response(files[got[-1]])

    monkeypatch.setattr(models.requests, "get", get)
    return got, listing


def test_download(monkeypatch, tmp_path):
    folder = str(tmp_path / "m")
    got, listing = _zoo(monkeypatch, {"rdf.yaml": b"type: model", "weights.pt": b"0" * 10})
    progress = []
    path = models.download_bmz_model("m", folder, callback=lambda d, t: progress.append((d, t)))
    assert path == str(tmp_path / "m" / "rdf.yaml") and sorted(got) == ["rdf.yaml", "weights.pt"]
    assert progress[-1] == (21, 21) and models.downloaded_bmz_model("m", folder) == path

    got.clear()
    assert models.download_bmz_model("m", folder) == path and got == []  # kept

    listing[1]["last_modified"] = 2  # a new version of the weights in the zoo
    models.download_bmz_model("m", folder)
    assert got == ["weights.pt"]

    # Stopped halfway (e.g. canceled): what was downloaded is kept, the model is not usable until the rest is there
    listing[0]["last_modified"] = listing[1]["last_modified"] = 3
    get = models.requests.get

    def fails(url, stream=False, timeout=30):
        if url.endswith("weights.pt"):
            raise requests.ConnectionError()
        return get(url, stream, timeout)

    monkeypatch.setattr(models.requests, "get", fails)
    got.clear()
    with pytest.raises(requests.ConnectionError):
        models.download_bmz_model("m", folder)
    assert models.downloaded_bmz_model("m", folder) is None
    monkeypatch.setattr(models.requests, "get", get)
    got.clear()
    assert models.download_bmz_model("m", folder) == path and got == ["weights.pt"]

    def offline(model_id, timeout=30):
        raise requests.ConnectionError()

    monkeypatch.setattr(models, "bmz_model_files", offline)
    assert models.download_bmz_model("m", folder) == path  # the copy downloaded before
    with pytest.raises(requests.ConnectionError):
        models.download_bmz_model("other", str(tmp_path / "other"))
    assert models.downloaded_bmz_model("other", str(tmp_path / "other")) is None


def test_local_model_utf8(tmp_path):
    """A model downloaded is read as UTF-8 whatever the system's encoding (e.g. its emoji with cp1252 on Windows)."""
    import os
    import subprocess
    import sys

    rdf = tmp_path / "rdf.yaml"
    rdf.write_text("type: model\nid_emoji: \U0001F98F\n", encoding="utf-8")
    env = dict(os.environ, LC_ALL="POSIX", PYTHONCOERCECLOCALE="0", PYTHONUTF8="0")
    code = "import sys; from biapy.models import read_local_bmz_model as r; print(r(sys.argv[1])['raw']['manifest']['type'])"
    out = subprocess.run([sys.executable, "-c", code, str(rdf)], env=env, capture_output=True, text=True, encoding="utf-8")
    assert out.stdout.strip().endswith("model"), out.stderr[-500:]
