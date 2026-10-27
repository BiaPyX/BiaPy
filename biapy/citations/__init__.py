"""
Works to cite when using BiaPy with a given configuration.

The references, and the rules that decide when each of them applies (e.g. the U-Net paper when
``MODEL.ARCHITECTURE`` is ``"unet"``), are in ``references.yaml``, next to this file. This module evaluates the
rules on a configuration and gives the citations in several formats:

- :func:`get_citations`: the :class:`Citation` list of a configuration (BiaPy first).
- :func:`citations_to_bibtex` / :func:`citations_to_markdown`: as a ``.bib`` file or a Markdown list.
- :func:`write_citations`: writes ``CITATIONS.md`` and ``citations.bib`` (done at the end of every run).
- :meth:`Citation.to_dict`: as JSON for the front-ends (e.g. the Fiji plugin).

The BioImage Model Zoo export uses them too, so the model's ``cite`` field and the references of its
documentation are the same as the ones of the run.
"""

import os
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml

REFERENCES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "references.yaml")

__all__ = [
    "Citation",
    "load_references",
    "get_citations",
    "citations_to_bibtex",
    "citations_to_markdown",
    "citations_summary",
    "write_citations",
]


@dataclass
class Citation:
    """
    A work to cite and why.

    Parameters
    ----------
    id : str
        Its id in ``references.yaml`` (or ``"extra_<n>"`` for the ones given apart, e.g. a BioImage Model Zoo
        model's own references).

    reasons : list of str
        Why it is cited (e.g. "Model architecture: U-Net"). Several rules can cite the same work.

    entry : dict
        Its fields in ``references.yaml``: ``type``, ``title``, ``authors``, ``year``, ``venue``, ``volume``,
        ``pages``, ``doi``, ``arxiv``, ``url``, ``code``. Extra citations may only have ``text`` and ``doi``/``url``.

    group : str, optional
        For extra citations that go together, shown as one item: e.g. "The references of the BioImage Model Zoo
        model 'affable-shark'", whose ``text`` are then their roles (e.g. "training library", "data").

    group_url : str, optional
        Where to read about the group (e.g. the model's page).
    """

    id: str
    reasons: List[str] = field(default_factory=list)
    entry: Dict[str, Any] = field(default_factory=dict)
    group: Optional[str] = None
    group_url: Optional[str] = None

    @property
    def title(self) -> str:
        return self.entry.get("title") or self.entry.get("text") or self.id

    @property
    def doi(self) -> Optional[str]:
        return self.entry.get("doi")

    @property
    def url(self) -> Optional[str]:
        """Where to read it: its DOI, arXiv page or URL."""
        if self.entry.get("doi"):
            return "https://doi.org/" + self.entry["doi"]
        if self.entry.get("arxiv"):
            return "https://arxiv.org/abs/" + self.entry["arxiv"]
        return self.entry.get("url")

    def text(self) -> str:
        """
        One-line reference, e.g. "Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical
        Image Segmentation. Medical Image Computing and Computer Assisted Intervention (MICCAI), 2015.
        https://doi.org/10.1007/978-3-319-24574-4_28".
        """
        e = self.entry
        if "title" not in e:
            return " ".join(x for x in (e.get("text", self.id), self.url or "") if x)
        authors = [_short_author(a) for a in e.get("authors", [])]
        if len(authors) > 6:
            authors = authors[:3] + ["et al"]
        venue = e.get("venue", "")
        if e.get("volume"):
            venue += " " + str(e["volume"])
        if e.get("pages"):
            venue += (":" if e.get("volume") else ", ") + str(e["pages"])
        parts = [", ".join(authors), _end(e["title"]), ", ".join(str(x) for x in (venue, e.get("year")) if x)]
        out = " ".join(_end(p) for p in parts if p)
        return out + (" " + self.url if self.url else "")

    def bibtex_key(self) -> str:
        """E.g. "ronneberger2015unet" (the first author's family name, the year and the first word of the title)."""
        e = self.entry
        family = (e.get("authors") or [self.id])[0].split(",")[0]
        word = next((w for w in re.findall(r"[A-Za-z0-9]+", e.get("title", "")) if len(w) > 3), "")
        return _ascii(family + str(e.get("year", "")) + word).lower() or self.id

    def bibtex(self, key: Optional[str] = None) -> str:
        """BibTeX entry, with the reasons it is cited as a comment above it."""
        e = self.entry
        kind = e.get("type", "misc")
        fields: List[tuple] = []
        if "title" in e:
            fields.append(("title", "{" + e["title"] + "}"))
            fields.append(("author", " and ".join(e.get("authors", []))))
            if kind == "article":
                fields.append(("journal", e.get("venue")))
            elif kind == "inproceedings":
                fields.append(("booktitle", e.get("venue")))
            elif kind == "preprint" and e.get("arxiv"):
                fields += [("eprint", e["arxiv"]), ("archivePrefix", "arXiv")]
            elif e.get("venue"):
                fields.append(("howpublished", e["venue"]))
            fields += [("year", e.get("year")), ("volume", e.get("volume")), ("pages", e.get("pages"))]
        else:
            fields.append(("note", e.get("text")))
        fields += [("doi", e.get("doi")), ("url", self.url)]
        entry_type = {"article": "article", "inproceedings": "inproceedings"}.get(kind, "misc")
        lines = ["% " + r for r in self.reasons]
        lines.append("@%s{%s," % (entry_type, key or self.bibtex_key()))
        for k, v in fields:
            if v not in (None, ""):
                v = str(v).replace("--", "-").replace("-", "--") if k == "pages" else str(v)
                # DOIs and URLs are taken verbatim
                lines.append("  %s = {%s}," % (k, v if k in ("doi", "url", "eprint") else _bibtex_escape(v)))
        lines.append("}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """For the front-ends (JSON): id, reasons, title, text, url, doi, code, bibtex, group and group_url."""
        return {
            "id": self.id,
            "reasons": list(self.reasons),
            "title": self.title,
            "text": self.text(),
            "url": self.url,
            "doi": self.doi,
            "code": self.entry.get("code"),
            "bibtex": self.bibtex(),
            "group": self.group,
            "group_url": self.group_url,
        }


@lru_cache(maxsize=None)
def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for rule in data["rules"]:
        for ref in rule["cite"]:
            if ref not in data["references"]:
                raise ValueError(f"{path}: the rule '{rule['reason']}' cites '{ref}', which is not in 'references'")
    return data


def load_references(path: Optional[str] = None) -> Dict[str, Any]:
    """The contents of ``references.yaml`` ("references" and "rules")."""
    return _load(path or REFERENCES_FILE)


def get_citations(
    cfg: Any,
    extra: Optional[Iterable[Dict[str, Any]]] = None,
    path: Optional[str] = None,
) -> List[Citation]:
    """
    Works to cite for a configuration, in the order of the rules (BiaPy first), each with the reasons it is cited.

    Parameters
    ----------
    cfg : CfgNode or dict
        Complete configuration (defaults included, as after ``validate_config``): the rules check variables that a
        YAML file usually leaves out (e.g. ``MODEL.ARCHITECTURE``).

    extra : list of dicts, optional
        Citations given apart, added at the end unless they are already there (same DOI or title): e.g. the
        references of a BioImage Model Zoo model, as ``{"text": ..., "doi": ..., "url": ..., "reason": ...}``, plus
        ``"group"`` and ``"group_url"`` for the ones shown together (see :class:`Citation`).

    path : str, optional
        Another references file (by default the one of BiaPy).

    Returns
    -------
    citations : list of :class:`Citation`
    """
    data = load_references(path)
    out: Dict[str, Citation] = {}
    for rule in data["rules"]:
        if not _holds(cfg, rule.get("when")):
            continue
        for ref in rule["cite"]:
            c = out.setdefault(ref, Citation(ref, [], data["references"][ref]))
            if rule["reason"] not in c.reasons:
                c.reasons.append(rule["reason"])
    citations = list(out.values())
    for i, e in enumerate(extra or []):
        e = dict(e)
        text = str(e.get("text") or "").strip()
        doi = re.sub(r"^.*?(?=10\.)", "", str(e["doi"])).lower() if e.get("doi") else None
        if any((doi and c.doi == doi) or (text and c.title.lower() == text.lower()) for c in citations):
            continue
        entry = {"text": text}
        if doi:
            entry["doi"] = doi
        if e.get("url"):
            entry["url"] = str(e["url"])
        citations.append(
            Citation(f"extra_{i}", [e.get("reason") or "Reference of the model"], entry, e.get("group"), e.get("group_url"))
        )
    return citations


def _items(citations: Sequence[Citation]) -> List[List[Citation]]:
    """The citations as the items to show: alone, or together with the rest of their group (where its first is)."""
    items: List[List[Citation]] = []
    groups: Dict[str, List[Citation]] = {}
    for c in citations:
        if c.group is None:
            items.append([c])
        elif c.group in groups:
            groups[c.group].append(c)
        else:
            groups[c.group] = [c]
            items.append(groups[c.group])
    return items


def citations_to_bibtex(citations: Sequence[Citation]) -> str:
    """The citations as the contents of a ``.bib`` file (unique keys)."""
    keys: Dict[str, int] = {}
    out = []
    for c in citations:
        key = c.bibtex_key()
        keys[key] = keys.get(key, 0) + 1
        out.append(c.bibtex(key if keys[key] == 1 else f"{key}{chr(ord('a') + keys[key] - 1)}"))
    return "\n\n".join(out) + "\n"


def citations_to_markdown(citations: Sequence[Citation], title: str = "How to cite") -> str:
    """The citations as a numbered Markdown list, with why each one is cited and the code BiaPy adapted."""
    lines = [
        f"# {title}",
        "",
        "If you use these results in your work, please cite the following works (the BibTeX entries are in "
        "citations.bib):",
        "",
    ]
    for i, item in enumerate(_items(citations), start=1):
        c = item[0]
        if c.group is not None:
            lines.append(f"{i}. {c.group}" + (f": {c.group_url}" if c.group_url else ""))
            # Their texts are usually their roles (e.g. "training library"): "role: link"
            lines += [f"    - {g.title}: {g.url}" if g.url and "title" not in g.entry else f"    - {g.text()}" for g in item]
            continue
        lines.append(f"{i}. {c.text()}")
        lines.append(f"    - Cited for: {'; '.join(c.reasons)}")
        if c.entry.get("code"):
            lines.append(f"    - BiaPy's implementation is adapted from: {c.entry['code']}")
    return "\n".join(lines) + "\n"


def citations_summary(citations: Sequence[Citation]) -> str:
    """Short text for the console: a line per work with why it is cited, and one per group of works."""
    lines = []
    for item in _items(citations):
        c = item[0]
        if c.group is None:
            lines.append(f"  - {c.title} ({'; '.join(c.reasons)})")
        else:
            n = "{} work{}".format(len(item), "s" if len(item) > 1 else "")
            lines.append(f"  - {c.group} ({n}" + (f": {c.group_url})" if c.group_url else ")"))
    return "\n".join(lines)


def write_citations(
    cfg: Any,
    out_dir: str,
    extra: Optional[Iterable[Dict[str, Any]]] = None,
) -> List[Citation]:
    """
    Writes ``CITATIONS.md`` and ``citations.bib`` with the works to cite for a configuration into ``out_dir``.

    Returns
    -------
    citations : list of :class:`Citation`
        The citations written.
    """
    citations = get_citations(cfg, extra=extra)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "CITATIONS.md"), "w", encoding="utf-8") as f:
        f.write(citations_to_markdown(citations))
    with open(os.path.join(out_dir, "citations.bib"), "w", encoding="utf-8") as f:
        f.write(citations_to_bibtex(citations))
    return citations


###########
# Helpers #
###########
_MISSING = object()


def _value(cfg: Any, key: str) -> Any:
    """Value of a dotted path (numbers index lists) in a CfgNode/dict, or ``_MISSING``."""
    v = cfg
    for part in key.split("."):
        if isinstance(v, (list, tuple)) and part.isdigit() and int(part) < len(v):
            v = v[int(part)]
        elif isinstance(v, dict) and part in v:
            v = v[part]
        else:
            return _MISSING
    return v


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, str) and isinstance(b, str):
        return a.lower() == b.lower()
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    return a == b


def _as_list(v: Any) -> list:
    if v is _MISSING or v is None:
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _matches(v: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        (op, arg), = expected.items()
        if op == "contains":
            return any(_same(x, arg) for x in _as_list(v))
        if op == "contains_any":
            return any(_same(x, a) for x in _as_list(v) for a in arg)
        if op == "not_in":
            return not any(_same(v, a) for a in arg)
        if op == "startswith":
            return isinstance(v, str) and v.lower().startswith(str(arg).lower())
        raise ValueError(f"Unknown operator '{op}' in references.yaml")
    if v is _MISSING:
        return False
    if isinstance(expected, list):
        return any(_same(v, x) for x in expected)
    return _same(v, expected)


def _holds(cfg: Any, cond: Any) -> bool:
    if cond is None:
        return True
    for key, expected in cond.items():
        if key == "any":
            ok = any(_holds(cfg, c) for c in expected)
        elif key == "not":
            ok = not _holds(cfg, expected)
        else:
            ok = _matches(_value(cfg, key), expected)
        if not ok:
            return False
    return True


def _short_author(a: str) -> str:
    """"Ronneberger, Olaf" -> "Ronneberger O" (names without a comma, e.g. organizations, as they are)."""
    if ", " not in a:
        return a
    family, given = a.split(", ", 1)
    initials = "".join(p[0] for p in re.split(r"[\s\-\.]+", given) if p)
    return f"{family} {initials}"


def _end(s: str) -> str:
    s = s.strip()
    return s if s.endswith((".", "?", "!")) else s + "."


def _ascii(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode())


def _bibtex_escape(s: str) -> str:
    return re.sub(r"(?<!\\)([&%#_])", r"\\\1", s)
