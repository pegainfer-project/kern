"""The e2e driver's verdicts, off the GPU: what excuses a divergence, what
a hit must pair with, what an empty run is worth."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e2e  # noqa: E402


class Scripted:
    """An oracle whose answers are `want` from any context and whose
    near-tie verdicts are given by the context length they fall at."""

    exact = False
    available = True

    def __init__(self, prompt: list[int], want: list[int], ties: set[int]):
        self.prompt, self.want, self.ties, self.probes = prompt, want, ties, 0

    def near_tie(self, context, a, b, rows):
        self.probes += 1
        return "near-tie: scripted" if len(context) in self.ties else "confident flip: scripted"

    def run(self, ids, steps, rows=None):
        return self.want[len(ids) - len(self.prompt) :][:steps]


class SameTest(unittest.TestCase):
    prompt = [1, 2, 3]
    want = [10, 11, 12, 13, 14, 15]

    def test_a_near_tie_excuses_one_token_and_the_rest_is_still_compared(self):
        o = Scripted(self.prompt, self.want, ties={len(self.prompt) + 2})
        s = e2e.Same(o, 1)
        s.add("x", self.prompt, [10, 11, 99, 13, 14, 77], self.want)
        self.assertEqual((s.ok, s.excused, o.probes), (False, 1, 2))

    def test_a_near_tie_followed_by_the_oracles_own_continuation_passes(self):
        o = Scripted(self.prompt, self.want, ties={len(self.prompt) + 2})
        s = e2e.Same(o, 1)
        s.add("x", self.prompt, [10, 11, 99, 13, 14, 15], self.want)
        self.assertEqual((s.ok, s.excused, s.same), (True, 1, 0))

    def test_near_ties_are_bounded_per_answer(self):
        o = Scripted(self.prompt, self.want, ties=set(range(3, 20)))
        s = e2e.Same(o, 1)
        s.add("x", self.prompt, [90, 91, 92, 93, 94, 95], self.want)
        self.assertEqual((s.ok, s.excused <= e2e.EXCUSED_MAX), (False, True))

    def test_an_exact_oracle_excuses_nothing(self):
        class Exact(Scripted):
            exact = True

            def near_tie(self, context, a, b, rows):
                return e2e.Reference.near_tie(self, context, a, b, rows)

        s = e2e.Same(Exact(self.prompt, self.want, set()), 1)
        s.add("x", self.prompt, [10, 11, 99, 13, 14, 15], self.want)
        self.assertEqual((s.ok, s.excused), (False, 0))


class UlpTest(unittest.TestCase):
    def test_ulp_is_the_top_logits_own_bf16_ulp(self):
        self.assertEqual([e2e.ulp_of(x) for x in (0.25, 1.0, 20.0, 3e-5, -6.0)], [2**-9, 2**-7, 2**-3, 2**-23, 2**-5])


class VerdictTest(unittest.TestCase):
    def test_a_miss_pairs_with_its_own_first_turn_not_kept(self):
        floor = lambda n: n
        unkept = {"req-2"}
        self.assertFalse(e2e.turn2_hit([(0, 90, ["req-1", "req-13"])], floor, unkept))
        self.assertTrue(e2e.turn2_hit([(96, 90, ["req-0", "req-12"]), (0, 90, ["req-1", "req-13"])], floor, {"req-1", "req-13"}))
        self.assertFalse(e2e.turn2_hit([(0, 90, ["req-1", "req-13"])], floor, {"req-1", "req-13"}))
        self.assertFalse(e2e.turn2_hit([(80, 90, ["req-0", "req-12"])], floor, set()))

    def test_acceptance_is_held_to_the_unloaded_rate(self):
        self.assertEqual([e2e.acceptance_holds(b, u) for b, u in ((48, 47), (48, 43.2), (48, 40), (None, 30), (30, None), (0, 0))], [True, True, False, False, False, False])
        self.assertEqual(e2e.accept_rate([{"accept_pct": 20, "steps": 100}, {"accept_pct": 40, "steps": 300}, {"steps": 50}]), 35.0)
        self.assertIsNone(e2e.accept_rate([{"steps": 50}]))
        self.assertEqual([e2e.acceptance_holds(b, b, (40, 60)) for b in (48, 33, 70)], [True, False, False])

    def test_a_single_rank_target_without_an_oracle_is_not_a_pass(self):
        self.assertEqual([e2e.oracle_gate(r, avail) for r, avail in ((1, True), (1, False), (4, False))], [True, False, None])


class DealtTest(unittest.TestCase):
    def test_a_group_is_asked_short_page_crossing_chunked_and_many_rows(self):
        for w in (2, 8):
            cases = e2e.dealt_cases(w, 16, 2048, 64)
            by = lambda sc: [c for c in cases if c.scenario == sc]
            span = 16 * w
            short = {len(c.ids) for c in by("short")}
            crossing = [c for c in by("pages") if len(c.ids) // span < (len(c.ids) + c.max_tokens) // span]
            self.assertEqual((min(short) < w, max(short) > w), (True, True))
            self.assertEqual((bool(crossing), max(len(c.ids) for c in by("pages")) > 2048, len(by("many_rows"))), (True, True, 64))
            self.assertEqual({1, 64} <= {c.max_tokens for c in by("short")}, True)
            self.assertEqual((len(set(cases)), all(0 <= i < 256 for c in cases for i in c.ids)), (len(cases), True))

    def test_gpus_name_their_node_and_a_dealt_target_finds_its_twin(self):
        self.assertEqual(e2e.gpus_of("0,3,node2:1"), [e2e.Gpu(0), e2e.Gpu(3), e2e.Gpu(1, "node2")])
        mk = lambda name, ranks, model, dealt: e2e.Target(name, Path("k.toml"), Path(__file__), Path("."), [], None, ranks, True, model, dealt)
        twin, other, d = mk("twin", 1, "m", False), mk("other", 1, "x", False), mk("d", 2, "m", True)
        self.assertEqual((e2e.twin_of(d, [other, d, twin]), e2e.twin_of(twin, [d, twin]), e2e.twin_of(d, [other, d])), (twin, None, None))

    def test_exactness_names_the_first_divergence(self):
        self.assertEqual(e2e.exact([("a", [1, 2], [1, 2])]), (True, "1/1 identical"))
        ok, note = e2e.exact([("a", [1, 2], [1, 2]), ("b", [1, 3], [1, 2])])
        self.assertEqual((ok, note.startswith("1/2 identical; b: diverge at token 1")), (False, True))


class SummaryTest(unittest.TestCase):
    def test_nothing_run_is_not_a_pass(self):
        skipped = e2e.Report(target="t", skipped="missing: kern")
        _, code = e2e.summarize([skipped])
        self.assertEqual(code, 1)

    def test_a_gated_failure_fails_and_reports_do_not(self):
        ok = e2e.Report(target="a")
        ok.checks = [e2e.Check("c", True), e2e.Check("i", None)]
        bad = e2e.Report(target="b")
        bad.checks = [e2e.Check("c", False)]
        self.assertEqual((e2e.summarize([ok])[1], e2e.summarize([ok, bad])[1]), (0, 1))


if __name__ == "__main__":
    unittest.main()
