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

The files being written are closed, and so readable, at every quarter hour,
whenever `<root>/rotate` is touched, and when the interpreter exits. A kernel
reaches them up to about a second after it ran. An engine that ends its
processes with a signal nobody handles loses the pair being written: touch
the file before stopping such an engine.
"""
import atexit
import warnings
import weakref

import torch

from kern._kern import trace_finish, trace_start as start

__all__ = ["start", "record"]


def heard(hook):
    """`hook`, saying the first thing it raises: torch drops what a graph hook raises, and a
    graph the tracer lost is traced as the host's own kernels."""
    said = []

    def run(graph):
        try:
            hook(graph)
        except Exception as e:
            if not said:
                said.append(e)
                warnings.warn(f"kern.trace: {hook.__name__} failed, so CUDA graphs may be traced as the host's own "
                              f"kernels: {e!r}", RuntimeWarning)

    return run


def record(rt, root: str, rank: int = 0) -> None:
    """Record this process's kernels under `root` as `rank`, and trace every CUDA graph torch
    captures from now on as the programs `rt` enqueued into it."""
    graphs = torch.cuda.graphs
    if not hasattr(graphs, "register_graph_replay_start_hook"):
        raise RuntimeError(f"tracing a host's CUDA graphs needs torch.cuda.graphs' capture and replay hooks, "
                           f"which torch {torch.__version__} does not have")
    rt.record(root, rank)
    atexit.register(trace_finish)
    taken = weakref.WeakKeyDictionary()

    def captured(graph):
        programs = rt.captured(graph.raw_cuda_graph())
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

    graphs.register_graph_capture_end_hook(heard(captured))
    graphs.register_graph_replay_start_hook(heard(launching))
    graphs.register_graph_replay_end_hook(heard(launched))
