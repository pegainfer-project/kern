"""The blob store side of the registry (docs/registry.md): bytes go into the
runtime's cache first (`blobs/<sha256>`, so a fresh build runs locally
before anything is published), then to `Pegainfer/kern-kernels` with the
same layout. Blobs are immutable and never deleted; a wrong one is
superseded by a new sha the manifests move to.
"""
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import index  # noqa: E402


def sha256_of(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def put(path):
    """Copy a file into the cache under its sha256; returns the sha."""
    sha = sha256_of(path)
    dst = index.blob_path(sha)
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f"{sha}.tmp.{os.getpid()}")
        shutil.copyfile(path, tmp)
        os.rename(tmp, dst)
    return sha


def token():
    t = os.environ.get("HF_TOKEN")
    if t:
        return t
    p = pathlib.Path.home() / ".cache" / "huggingface" / "token"
    return p.read_text().strip() if p.exists() else None


def url(sha):
    return f"https://huggingface.co/{index.BLOB_REPO}/resolve/main/blobs/{sha}"


def exists_remote(sha):
    req = urllib.request.Request(url(sha), method="HEAD")
    if t := token():
        req.add_header("Authorization", f"Bearer {t}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 200
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 404):
            return False
        raise


def upload(shas, message, extra_files=None):
    """Publish cached blobs (and optional {repo path: local path} files, e.g. licenses/) in one commit with the hf CLI."""
    shas = sorted(set(shas))
    with tempfile.TemporaryDirectory() as d:
        stage = pathlib.Path(d)
        (stage / "blobs").mkdir()
        for sha in shas:
            src = index.blob_path(sha)
            assert src.exists(), f"blob {sha} is not in the cache; import it first"
            shutil.copyfile(src, stage / "blobs" / sha)
        for rel, local in (extra_files or {}).items():
            dst = stage / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local, dst)
        env = dict(os.environ)
        if t := token():
            env["HF_TOKEN"] = t
        subprocess.run(["hf", "upload", index.BLOB_REPO, str(stage), ".", "--commit-message", message],
                       check=True, env=env)
    return shas
