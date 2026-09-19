"""The kernel index (docs/registry.md): one TOML per family under
tools/kernels/index/, read here by the generators.

A family is one parameter ABI: a handwritten `.cu`, one vendored template
instance, or one dtype combination of a code generator's kernels (every tile
variant of trtllm-gen's `Bmm_MxE4m3_MxE2m1MxE4m3` shares a `KernelParams`).
`[family]` records where the bytes came from and what proves the ABI;
`[[variant]]` is one artifact: its sha256 (the identity a manifest pins),
the entries it defines, its launch geometry when the upstream launcher
fixes it, and tags parsed from its name; `[[pick]]` is a measured choice of
variant for one (op, workload shape), never made without a bench report.

A generator asks `variant(family, **defines)` for a handwritten build or a
one-artifact family and `pick(family, op, shape)` for a generated one; both
return the `module` dict a launch spells (`cubin` = the blob's registry
ref, `sha256`, `label` = the manifest module name). Nothing here touches
the network: the index is the truth at generation time, the bytes are
fetched by the runtime from the blob store or its cache.
"""
import dataclasses
import json
import os
import pathlib
import re
import tomllib

REPO = pathlib.Path(__file__).resolve().parents[2]
INDEX = REPO / "tools" / "kernels" / "index"
BLOB_REPO = "Pegainfer/kern-kernels"
FAMILY_KEYS = ["name", "kind", "sm", "abi", "upstream", "license", "license_blob", "toolchain", "rebuild",
               "abi_source", "abi_capture", "imported"]
VARIANT_KEYS = ["name", "sha256", "entries", "defines", "launch", "tags"]
PICK_KEYS = ["op", "shape", "variant", "report", "measured"]


def source(sha256):
    """The registry ref of a blob: the runtime fetches it into its cache by this URL, checks the sha, keeps the bytes."""
    return f"hf:{BLOB_REPO}/blobs/{sha256}"


def cache_dir():
    """Where the runtime keeps fetched blobs; the import tools write there first so a local build runs before any upload."""
    env = os.environ.get("KERN_CACHE_DIR")
    return pathlib.Path(env) if env else pathlib.Path.home() / ".cache" / "kern"


def blob_path(sha256):
    return cache_dir() / "blobs" / sha256


def path(family):
    return INDEX / f"{family}.toml"


def families():
    return sorted(p.stem for p in INDEX.glob("*.toml"))


def load(family):
    p = path(family)
    if not p.exists():
        raise KeyError(f"kernel family `{family}`: no {p.relative_to(REPO)}; import it (tools/kernels/import_*.py)")
    return tomllib.loads(p.read_text())


def variant_name(family, defines):
    return family + "".join(f"+{k}={v}" for k, v in sorted(defines.items()))


@dataclasses.dataclass(frozen=True)
class Variant:
    family: str
    name: str
    sha256: str
    entries: tuple
    launch: dict | None
    tags: dict
    defines: dict

    @property
    def module(self):
        """The fields a launch spells for its artifact; `normalize` hoists them into the manifest's `modules` table under `label`."""
        return {"cubin": source(self.sha256), "sha256": self.sha256, "label": self.name}

    @property
    def entry(self):
        (e,) = self.entries
        return e


def _variant(family, v):
    return Variant(family, v["name"], v["sha256"], tuple(v.get("entries", ())), v.get("launch"), v.get("tags", {}),
                   v.get("defines", {}))


def variant(family, **defines):
    """The family's build named by `defines` (`variant("k3_residual", LAND_BF16=1)` is `k3_residual+LAND_BF16=1`), the plain one without."""
    name = variant_name(family, defines)
    for v in load(family).get("variant", []):
        if v["name"] == name:
            return _variant(family, v)
    raise KeyError(f"kernel family `{family}`: no variant `{name}` in the index")


def pick(family, op, shape):
    """The variant a bench chose for `op` on workload `shape`; a family with many variants is never picked by hand."""
    doc = load(family)
    for p in doc.get("pick", []):
        if (p["op"], p["shape"]) == (op, shape):
            return variant_by_name(family, p["variant"], doc)
    raise KeyError(f"kernel family `{family}`: no pick for op `{op}` on shape `{shape}`; run kern bench and record one")


def variant_by_name(family, name, doc=None):
    for v in (doc or load(family)).get("variant", []):
        if v["name"] == name:
            return _variant(family, v)
    raise KeyError(f"kernel family `{family}`: no variant `{name}`")


# --- writing: the import tools land a normalized document, the same bytes for the same facts

_BARE = re.compile(r"^[A-Za-z0-9_-]+$")


def _key(k):
    return k if _BARE.match(k) else json.dumps(k)


def _value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{_key(k)} = {_value(x)}" for k, x in v.items()) + " }"
    raise TypeError(f"no TOML form for {type(v).__name__}")


def _table(d, keys):
    ordered = [k for k in keys if k in d] + [k for k in sorted(d) if k not in keys]
    return "".join(f"{_key(k)} = {_value(d[k])}\n" for k in ordered if d[k] is not None)


def dumps(doc):
    out = ["[family]\n" + _table(doc["family"], FAMILY_KEYS)]
    for v in sorted(doc.get("variant", []), key=lambda v: v["name"]):
        out.append("[[variant]]\n" + _table(v, VARIANT_KEYS))
    for p in sorted(doc.get("pick", []), key=lambda p: (p["op"], p["shape"])):
        out.append("[[pick]]\n" + _table(p, PICK_KEYS))
    return "\n".join(out)


def save(doc):
    INDEX.mkdir(parents=True, exist_ok=True)
    p = path(doc["family"]["name"])
    p.write_text(dumps(doc))
    assert tomllib.loads(p.read_text()) == _normal(doc), "the document does not round-trip"
    return p


def _normal(doc):
    """The document as `dumps` lays it out: no None values, variants and picks in their sorted order."""
    d = _strip_none(doc)
    if "variant" in d:
        d["variant"] = sorted(d["variant"], key=lambda v: v["name"])
    if "pick" in d:
        d["pick"] = sorted(d["pick"], key=lambda p: (p["op"], p["shape"]))
    return d


def _strip_none(x):
    if isinstance(x, dict):
        return {k: _strip_none(v) for k, v in x.items() if v is not None}
    if isinstance(x, (list, tuple)):
        return [_strip_none(v) for v in x]
    return x


def upsert_variant(doc, v):
    """Replace the variant of the same name or add it; the document stays sorted on save."""
    rest = [x for x in doc.get("variant", []) if x["name"] != v["name"]]
    doc["variant"] = rest + [v]
    return doc
