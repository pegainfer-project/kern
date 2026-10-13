"""A manifest's ranks, one per process, over a `torch.distributed` group.

An engine already has a process group spanning the ranks (vLLM's and
SGLang's tensor-parallel group, each with a gloo `cpu_group`); kern rides
it instead of a channel of its own. `connect` gathers every member's offer
and maps the peers; `start` runs what the manifest runs once and then waits
for every member, because a rank that ran its setup must not push into a
peer whose setup (a Lamport stage poisoned to -0.0, say) has not run yet:
the poison would overwrite what arrived and the reader would wait forever.

The group spans every topology group of the manifest, member `i` of it
being member `i` of each; a manifest whose groups are smaller than the
world needs a group per topology group, which no engine asks for yet.
"""
import torch.distributed as dist

from kern.host import once


def connect(rt, m: dict, group) -> None:
    """Gather every member's offer over `group` and connect `rt` to them all."""
    me, world = dist.get_rank(group), dist.get_world_size(group)
    groups = m.get("topology", {}).get("groups", {})
    wrong = {g: (rt.rank(g), n) for g, n in groups.items() if (rt.rank(g), n) != (me, world)}
    if wrong:
        raise ValueError(f"process group rank {me} of {world}; the runtime's (index, size) per group: {wrong}")
    offers = [None] * world
    dist.all_gather_object(offers, rt.offer(), group=group)
    rt.connect({g: offers for g in groups})


def start(rt, m: dict, group=None) -> None:
    """Run the manifest's once programs, then wait at `group`'s barrier."""
    for name in once(m):
        rt.run(name, {v: 1 for v in m.get("vars", {})})
    rt.synchronize()
    if group is not None:
        dist.barrier(group=group)
