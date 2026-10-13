"""The kern runtime for a Python host.

`kern.Runtime` is the binding. `kern.host` reads a manifest's buffers into
torch views and binds an engine's tensors to host states; `kern.group`
connects a manifest's ranks over a `torch.distributed` process group. Both
are engine-neutral: what an engine calls its cache or its batch lives in
that engine's adapter.
"""
from kern._kern import Runtime, __version__

__all__ = ["Runtime", "__version__"]
