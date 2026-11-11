"""Every kernel a host's process runs, named by the manifest call that launched it.

`start()` subscribes the process to CUPTI's hardware event system. It must
run before the process makes its first CUDA context, which for an engine
means before it loads the model, and it needs libcupti 13.3 or newer as the
first `libcupti.so` on the library path. Called again it does nothing, or
raises what kept it from starting.

`record(rt, root, rank)` then keeps every kernel under `root`, in the files
kern-serve's `--trace-dir` writes (docs/serve.md, "Tracing"). A program the
host enqueues is a step. So is every program in a CUDA graph torch captured,
each time torch replays the graph: the runtime kept which call added each
kernel node while the host captured, and torch's graph hooks tell it when
the graph is launched. The host's own kernels, inside its graphs or outside
them, are rows no step claims.

The files being written are closed, and so readable, at every quarter hour
and whenever `<root>/rotate` is touched. A kernel reaches them up to about a
second after it ran. An engine ends its processes with a signal nobody
handles, which loses the pair being written: touch the file before stopping
the engine.
"""
import weakref

import torch

from kern._kern import trace_start as start

__all__ = ["start", "record"]


def record(rt, root: str, rank: int = 0) -> None:
    """Record this process's kernels under `root` as `rank`, and trace every CUDA graph torch
    captures from now on as the programs `rt` enqueued into it."""
    graphs = torch.cuda.graphs
    if not hasattr(graphs, "register_graph_replay_start_hook"):
        raise RuntimeError(f"tracing a host's CUDA graphs needs torch.cuda.graphs' capture and replay hooks, "
                           f"which torch {torch.__version__} does not have")
    rt.record(root, rank)
    taken = weakref.WeakKeyDictionary()

    def captured(graph):
        programs = rt.captured()
        if programs is None:
            taken.pop(graph, None)
        else:
            taken[graph] = programs

    def launching(graph):
        if (programs := taken.get(graph)) is not None:
            programs.launching()

    def launched(graph):
        if (programs := taken.get(graph)) is not None:
            programs.launched()

    graphs.register_graph_capture_end_hook(captured)
    graphs.register_graph_replay_start_hook(launching)
    graphs.register_graph_replay_end_hook(launched)
