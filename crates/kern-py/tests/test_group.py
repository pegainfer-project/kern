"""Two ranks, one process each, meet over a gloo group and all-reduce.

The manifest is the decode path's peer all-reduce (`peer_allreduce.cu`'s
Lamport one-shot) over a `w` group of two: three exported buffers whose
handles every rank maps, a `once` program that poisons the Lamport stages,
and the reduction, eager and inside a torch CUDA graph. Each rank loads its
place in the topology, `kern.group.connect` trades offers over the
process group, `kern.group.start` runs the setup and waits for the other
rank; every element of every round is then an exact sum of small integers.

Needs two free GPUs, nvcc, and the `kern` wheel installed:

    python3 -m unittest discover -s crates/kern-py/tests -p test_group.py
"""
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

H = 7168
ROWS = 64
WORLD = 2
SRC = Path(__file__).resolve().parents[3] / "tools" / "kernels-src" / "peer_allreduce.cu"


def cubin(out: Path) -> str:
    """nvcc the all-reduce for two ranks into `out`, named by its sha; the sha."""
    tmp = out / "peer_allreduce.cubin.tmp"
    arch = os.environ.get("KERN_SM", "sm_103a")
    subprocess.run(["nvcc", "-cubin", f"-arch={arch}", f"-DNRANKS={WORLD}", "-o", str(tmp), str(SRC)], check=True)
    sha = hashlib.sha256(tmp.read_bytes()).hexdigest()
    tmp.rename(out / f"peer_allreduce-{sha[:12]}.cubin")
    return sha


def manifest(sha: str) -> dict:
    stage = WORLD * ROWS * H * 4
    carry = lambda dt, n: {"dtype": dt, "shape": [n], "kind": "carry", "export": True}
    peers = lambda of: {"dtype": "u64", "shape": [WORLD], "kind": "peer", "of": of, "group": "w"}
    reduce = {"op": "allreduce", "args": [
        {"buf": "x"}, {"buf": "y"}, {"buf": "comm"}, {"buf": "comm_peers"}, {"buf": "flags"}, {"buf": "flag_peers"},
        {"buf": "lamport"}, {"buf": "lamport_peers"}, {"buf": "state"}, {"buf": "err"}, {"buf": "blocks"},
        {"rank": "w"}, {"var": "rows"}, {"i32": H}, {"i64": stage}, {"i32": 1}, {"i64": 2_000_000_000}]}
    return {
        "schema_version": 7,
        "model": "peer-allreduce-2",
        "vars": {"rows": {"max": ROWS}},
        "topology": {"groups": {"w": WORLD}},
        "buffers": {
            "x": {"dtype": "f32", "shape": ["rows", H], "kind": "input"},
            "y": {"dtype": "f32", "shape": ["rows", H], "kind": "output"},
            "comm": carry("u8", 2 * WORLD * ROWS * H * 4),
            "comm_peers": peers("comm"),
            "flags": carry("i32", WORLD * 256),
            "flag_peers": peers("flags"),
            "lamport": carry("u8", 3 * stage),
            "lamport_peers": peers("lamport"),
            "state": {"dtype": "i32", "shape": [8], "kind": "carry"},
            "err": {"dtype": "i32", "shape": [1], "kind": "output"},
            "blocks": {"dtype": "i32", "shape": [WORLD + 1], "kind": "input"},
        },
        "modules": {"peer_allreduce": {"source": "peer_allreduce.cubin", "sha256": sha}},
        "ops": {
            "allreduce": {
                "params": ["in buffer<f32>", "out buffer<f32>", "inout buffer<u8>", "in buffer<u64>",
                           "inout buffer<i32>", "in buffer<u64>", "inout buffer<u8>", "in buffer<u64>",
                           "inout buffer<i32>", "out buffer<i32>", "in buffer<i32>",
                           "i32", "i32", "i32", "i64", "i32", "i64"],
                "impl": {"launches": [{"module": "peer_allreduce", "entry": "kern_peer_allreduce_f32",
                                       "block": [H // 4 // 8, 1, 1], "grid": [152, 1, 1], "cluster": [8, 1, 1]}]},
            },
            "lamport_init": {
                "params": ["inout buffer<u8>", "i64"],
                "impl": {"launches": [{"module": "peer_allreduce", "entry": "kern_peer_lamport_init",
                                       "block": [256, 1, 1], "grid": [256, 1, 1]}]},
            },
        },
        "programs": {
            "init": {"once": True, "calls": [{"op": "lamport_init", "args": [{"buf": "lamport"}, {"i64": 3 * stage}]}]},
            "ar": {"calls": [reduce]},
        },
    }


def value(r: int, i: torch.Tensor, round: int) -> torch.Tensor:
    """Rank `r`'s input at element `i`: small integers, exact in any summation order."""
    return (((r + 1) * (i * 7 + round)) % 13).float()


def rank(r: int, root: str, out) -> None:
    import kern
    from kern import group
    from kern.host import region

    torch.cuda.set_device(r)
    dist.init_process_group("gloo", init_method=f"file://{root}/store", rank=r, world_size=WORLD)
    try:
        path = f"{root}/m.json"
        m = json.load(open(path))
        rt = kern.Runtime(path, root, [], r, {"w": (r, WORLD)})
        group.connect(rt, m, dist.group.WORLD)
        group.start(rt, m, dist.group.WORLD)
        x, y, blocks, err = (region(rt, m, n) for n in ("x", "y", "blocks", "err"))
        graphs = {}

        def call(how: str, rows: int) -> None:
            if how == "eager":
                torch.cuda.synchronize()
                rt.run("ar", {"rows": rows})
                return
            if rows not in graphs:
                graphs[rows] = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graphs[rows]):
                    rt.enqueue_after("ar", {"rows": rows}, torch.cuda.current_stream().cuda_stream)
            graphs[rows].replay()
            torch.cuda.synchronize()

        bad = []
        for round, (how, rows) in enumerate([(h, n) for n in (1, 7, 64) for h in ("eager", "graph", "graph", "eager")]):
            bl = [q * rows // WORLD for q in range(WORLD + 1)]
            # "Own rows first": local row j of rank q is group row (bl[q] + j) mod rows.
            i = torch.arange(rows * H, device="cuda")
            local = lambda q: ((i // H + bl[q]) % rows) * H + i % H
            blocks.copy_(torch.tensor(bl, dtype=torch.int32))
            x[:rows].copy_(value(r, local(r), round).view(rows, H))
            call(how, rows)
            want = sum(value(q, local(r), round) for q in range(WORLD)).view(rows, H)
            wrong = int((y[:rows] != want).sum())
            if wrong or int(err[0]):
                bad.append(f"{how} rows={rows}: {wrong} wrong elements, err {int(err[0])}")
        out.put((r, bad))
    except Exception as e:
        out.put((r, [f"{type(e).__name__}: {e}"]))
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(torch.cuda.device_count() >= WORLD and shutil.which("nvcc"), "needs two GPUs and nvcc")
class Group(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="kern-group-")
        sha = cubin(Path(self.root))
        Path(self.root, "m.json").write_text(json.dumps(manifest(sha)))

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_two_ranks_connect_over_gloo_and_all_reduce_exactly(self):
        ctx = mp.get_context("spawn")
        out = ctx.Queue()
        procs = [ctx.Process(target=rank, args=(r, self.root, out)) for r in range(WORLD)]
        for p in procs:
            p.start()
        results = dict(out.get(timeout=300) for _ in procs)
        for p in procs:
            p.join(timeout=60)
        self.assertEqual(results, {r: [] for r in range(WORLD)})

    def test_a_rank_without_fabric_handles_refuses_to_offer_and_names_the_imex_channel(self):
        import kern

        os.environ["KERN_NO_FABRIC"] = "1"
        try:
            rt = kern.Runtime(f"{self.root}/m.json", self.root, [], 0, {"w": (0, WORLD)})
            with self.assertRaisesRegex(RuntimeError, "imex-channels"):
                rt.offer()
        finally:
            del os.environ["KERN_NO_FABRIC"]


if __name__ == "__main__":
    unittest.main()
