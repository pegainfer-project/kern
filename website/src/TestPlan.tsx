import { useEffect, useState } from "react";
import "./verify-plan.css";

// One `kern test` run as one figure: the stretch of a 646-call decode program
// around the call the candidate changed, the report's sections in order, and
// the report itself. Every line is that run's own output; the beat only moves
// the eye from section to section.

type Phase = "diff" | "record" | "local" | "logits" | "noise" | "perf";
type Beat = Phase | "verdict";

const BEATS: { id: Beat; name: string; fact: string; ms: number; bad?: boolean }[] = [
  { id: "diff", name: "Find what changed", fact: "1 span of 646 calls", ms: 1800 },
  { id: "record", name: "Record A", fact: "22 runs · 1.3 MB kept", ms: 3000 },
  { id: "local", name: "Replay B on A's inputs", fact: "0/21 bit-identical", ms: 2800, bad: true },
  { id: "logits", name: "Run B free", fact: "KL 2.95e-1 · limit 1e-2", ms: 2800, bad: true },
  { id: "noise", name: "A against itself", fact: "1/1 clean · KL 0", ms: 2400 },
  { id: "perf", name: "Time both", fact: "tpot 9.575 → 9.599 ms", ms: 1800 },
  { id: "verdict", name: "Verdict", fact: "exit 1 · 10.1 s", ms: 3200, bad: true },
];

const REPORT: [Beat, string][] = [
  ["diff", "diff      decode 1 span of 646 calls (645 shared) · gdn_conv gdn.line_index, qkvz → qkvz"],
  ["record", "record    A: 22 runs · 1 spans kept (1.3 MB in 41 pieces) · state images 12687.8 MB · workload 376.6ms   2.5s"],
  ["local", "local     0/21 bit-identical · output next_token bit-identical · state gdn 131311250 of 308281344 bytes differ (42.59%) · state kv 872548 of 268435456 bytes differ (0.33%)"],
  ["local", "local     ✗ decode step 0 A[16..17) B[16..17) state gdn: 54177/61440 bytes of the write-set differ · wrote 55 KB outside A's write-set"],
  ["logits", "logits    17/22 argmax agree (2 within the KL limit, 3 beyond) · max KL 2.95e-1 at step 11 (limit 1e-2) · top-20 differs on 18 rows, overlap down to 6/20 at step 19   366.5ms"],
  ["logits", "logits    ✗ flip step 11: A 57839 → B 198 · A's margin 0.0938 · KL 2.95e-1 · A's token is B's #5"],
  ["noise", "noise     A against itself end to end: 22 rows · max KL 0.00e0 at step 19 · 0 argmax flips"],
  ["perf", "perf      decode rows 1 · eager 12.404 ms → 12.511 ms +0.9% · tpot 9.575 ms → 9.599 ms +0.2% (104 → 104 tok/s) · 1 span 6.0 µs → 5.7 µs −5.9%   1.2s"],
  ["verdict", "FAIL      B changes the argmax end-to-end at step 11: A 57839 → B 198, KL 2.95e-1 above the limit 1e-2 (A's margin 0.0938, A's token is B's #5)   10.1s"],
];

const MID = 150;
const PILL = { w: 112, h: 34 };
const X = { q1: 70, span: 204, q2: 436, step: 570, logits: 790, sample: 924, out: 1066 };
const SPAN = { w: 210, h: 96 };
const STEP = { w: 130, h: 60 };
const SAMPLE = { w: 120, h: 60 };

type Flow = { id: string; name: string; tag?: string; x: number; type: string; kind: "workspace" | "output"; on: Phase[]; note: Partial<Record<Phase, string>> };

const FLOWS: Flow[] = [
  {
    id: "q1",
    name: "qkvz",
    tag: "1",
    x: X.q1,
    type: "bf16 [tokens, 16384]",
    kind: "workspace",
    on: ["diff", "record", "local", "noise"],
    note: { diff: "frontier · read", record: "kept before the span", local: "A's bytes go into B", noise: "A's bytes, put back" },
  },
  {
    id: "q2",
    name: "qkvz",
    tag: "2",
    x: X.q2,
    type: "bf16 [tokens, 16384]",
    kind: "workspace",
    on: ["diff", "record", "local", "noise"],
    note: { diff: "frontier · written", record: "kept: the write-set", local: "compared with A's", noise: "A against itself" },
  },
  {
    id: "logits",
    name: "logits",
    tag: "3",
    x: X.logits,
    type: "bf16 [seqs, 248320]",
    kind: "workspace",
    on: ["record", "logits"],
    note: { record: "read after every run", logits: "KL(A ‖ B), row by row" },
  },
  { id: "out", name: "next_token", x: X.out, type: "i64 [seqs]", kind: "output", on: ["record", "logits"], note: { logits: "17/22 agree" } },
];

// The spine, left to right: from, to, and the sections that walk it.
const EDGES: [number, number, Phase[]][] = [
  [X.q1 + PILL.w, X.span, ["diff", "record", "local", "noise"]],
  [X.span + SPAN.w, X.q2, ["diff", "record", "local", "noise"]],
  [X.q2 + PILL.w, X.step, ["record", "logits"]],
  [X.logits + PILL.w, X.sample, ["record", "logits"]],
  [X.sample + SAMPLE.w, X.out, ["record", "logits"]],
];

// Where a section's finding lands: x, y, the section, and whether it is a difference.
const FOUND: [number, number, Phase, boolean][] = [
  [X.span + SPAN.w / 2 + 54, 228, "local", true],
  [X.q2 + PILL.w - 6, MID - PILL.h / 2, "local", true],
  [X.logits + PILL.w - 6, MID - PILL.h / 2, "logits", true],
  [X.q2 + PILL.w - 6, MID - PILL.h / 2, "noise", false],
  [X.logits + PILL.w - 6, MID - PILL.h / 2, "noise", false],
];

// What A's recording keeps, under the thing it was taken from: x centre, width,
// where the copy leaves the plan, the label, when the run reaches it (seconds
// into the record beat), the sections that write it back into the plan, and
// the sections that use it at all.
const SHELF = 306;
const KEPT: { id: string; cx: number; w: number; from: number; label: string; at: number; back: Phase[]; on: Phase[] }[] = [
  { id: "q1", cx: X.q1 + PILL.w / 2, w: 150, from: 190, label: "qkvz · before the span", at: 0.2, back: ["local", "noise"], on: ["record", "local", "noise"] },
  { id: "state", cx: X.span + SPAN.w / 2, w: 168, from: 272, label: "state · every run's start", at: 0, back: ["local", "noise"], on: ["record", "local", "noise"] },
  { id: "q2", cx: X.q2 + PILL.w / 2, w: 150, from: 190, label: "qkvz · what A wrote", at: 0.75, back: [], on: ["record", "local", "noise"] },
  { id: "logits", cx: X.logits + PILL.w / 2, w: 172, from: 190, label: "logits · after every run", at: 1.3, back: [], on: ["record", "logits", "noise"] },
];

const RUNS: Partial<Record<Beat, string>> = { record: "A", logits: "B" };

const SIDE: Partial<Record<Beat, string>> = { record: "A runs", local: "B runs", noise: "A, twice", perf: "A 6.0 µs · B 5.7 µs" };

// The two sides of the span: letter, row offset, the argument, and the sections that run that side.
const SIDES: [string, number, string, Beat[]][] = [
  ["A", -3, "offset 512", ["diff", "record", "noise"]],
  ["B", 22, "no offset", ["diff", "local"]],
];

const state = (beat: Beat, mine: Phase[]) => (beat === "verdict" ? "" : mine.includes(beat) ? " is-on" : " is-dim");

function Plan({ beat }: { beat: Beat }) {
  const mid = X.span + SPAN.w / 2;
  return (
    <svg className="vp-svg tp-svg" viewBox="0 0 1200 344" role="img" aria-label="The decode program around call 16, left to right: the reference and the candidate differ in one argument of that call. kern test finds the call, replays the candidate on the reference's inputs, and compares the logits at the end.">
      <defs>
        <marker id="tp-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 L8 4 L0 8 z" className="vp-arrowhead" />
        </marker>
      </defs>

      <g className={`vp-elide${state(beat, ["record", "logits"])}`}>
        <line x1="8" y1={MID} x2={X.q1 - 1} y2={MID} markerEnd="url(#tp-arrow)" />
        <text x="8" y={MID - 10} className="vp-mono vp-faint">#0 … #15</text>
        <line x1={X.step + STEP.w} y1={MID} x2={X.logits - 1} y2={MID} markerEnd="url(#tp-arrow)" />
        <text x={(X.step + STEP.w + X.logits) / 2} y={MID - 10} textAnchor="middle" className="vp-mono vp-faint">#18 … #644</text>
      </g>

      {EDGES.map(([from, to, mine]) => (
        <g key={from} className={`vp-edge${state(beat, mine)}`}>
          <line x1={from} y1={MID} x2={to - 1} y2={MID} markerEnd="url(#tp-arrow)" />
        </g>
      ))}

      {FLOWS.map((f) => {
        const cx = f.x + PILL.w / 2;
        const lead = f.tag ? 34 : 14;
        return (
          <g key={f.id} className={`vp-flow vp-${f.kind}${state(beat, f.on)}`}>
            <rect x={f.x} y={MID - PILL.h / 2} width={PILL.w} height={PILL.h} rx={PILL.h / 2} />
            {f.tag && (
              <g>
                <circle cx={f.x + 18} cy={MID} r="10" className="vp-tag" />
                <text x={f.x + 18} y={MID + 4} textAnchor="middle" className="vp-mono vp-tag-text">{f.tag}</text>
              </g>
            )}
            <text x={f.x + lead} y={MID + 4} className="vp-mono vp-strong">{f.name}</text>
            <text x={cx} y={MID + PILL.h / 2 + 16} textAnchor="middle" className="vp-mono vp-faint">{f.type}</text>
            {beat !== "verdict" && f.note[beat] && (
              <text x={cx} y={MID - PILL.h / 2 - 10} textAnchor="middle" className="vp-mono vp-note is-hot">{f.note[beat]}</text>
            )}
          </g>
        );
      })}

      <g className={`vp-side${state(beat, ["diff", "local"])}`}>
        <path d={`M${mid} 66 V${MID - SPAN.h / 2 - 3}`} markerEnd="url(#tp-arrow)" />
        <rect x={mid - 84} y="30" width="168" height="36" rx="4" />
        <text x={mid - 74} y="45" className="vp-mono vp-strong">gdn.line_index</text>
        <text x={mid - 74} y="58" className="vp-mono vp-faint">i32 [48, seqs] → gdn</text>
        <text x={mid + 84} y="24" textAnchor="end" className="vp-mono vp-kind">arg #3 · differs in B</text>
      </g>

      <g className={`vp-state${state(beat, ["record", "local", "noise"])}`}>
        <path d={`M${mid} ${MID + SPAN.h / 2 + 3} V227`} markerStart="url(#tp-arrow)" markerEnd="url(#tp-arrow)" />
        <path d={`M${mid + 54} 248 H${X.step + STEP.w / 2} V${MID + STEP.h / 2 + 3}`} markerEnd="url(#tp-arrow)" />
        <rect x={mid - 54} y="228" width="108" height="42" rx="4" />
        <text x={mid - 44} y="245" className="vp-mono vp-strong">state gdn</text>
        <text x={mid - 44} y="259" className="vp-mono vp-faint">147 MiB / seq</text>
      </g>

      <g className={`vp-kernel${state(beat, ["record", "perf"])}`}>
        <rect x={X.step} y={MID - STEP.h / 2} width={STEP.w} height={STEP.h} rx="5" />
        <rect x={X.step} y={MID - STEP.h / 2} width="36" height={STEP.h} rx="5" className="vp-shared-bg" />
        <text x={X.step + 18} y={MID + 4} textAnchor="middle" className="vp-mono vp-callno">#17</text>
        <text x={X.step + 46} y={MID - 4} className="vp-mono vp-op">gdn_step</text>
        <text x={X.step + 46} y={MID + 14} className="vp-mono vp-row">same in B</text>
      </g>

      <g className={`vp-kernel${state(beat, ["record", "logits", "perf"])}`}>
        <rect x={X.sample} y={MID - SAMPLE.h / 2} width={SAMPLE.w} height={SAMPLE.h} rx="5" />
        <rect x={X.sample} y={MID - SAMPLE.h / 2} width="40" height={SAMPLE.h} rx="5" className="vp-shared-bg" />
        <text x={X.sample + 20} y={MID + 4} textAnchor="middle" className="vp-mono vp-callno">#645</text>
        <text x={X.sample + 50} y={MID - 4} className="vp-mono vp-op">argmax</text>
        <text x={X.sample + 50} y={MID + 14} className="vp-mono vp-row">same in B</text>
      </g>

      <g className={`vp-kernel vp-span${state(beat, ["diff", "record", "local", "noise", "perf"])}${beat === "diff" || beat === "local" || beat === "noise" || beat === "perf" ? " tp-pulse" : ""}`}>
        <rect x={X.span} y={MID - SPAN.h / 2} width={SPAN.w} height={SPAN.h} rx="5" />
        <rect x={X.span} y={MID - SPAN.h / 2} width="40" height={SPAN.h} rx="5" className="vp-letter-bg" />
        <text x={X.span + 20} y={MID + 4} textAnchor="middle" className="vp-mono vp-callno vp-callno-on">#16</text>
        <text x={X.span + 52} y={MID - 27} className="vp-mono vp-op">gdn_conv</text>
        <text x={X.span + SPAN.w - 10} y={MID - 27} textAnchor="end" className="vp-mono vp-row is-hot">the span</text>
        {SIDES.map(([side, y, arg, on]) => (
          <g key={side} className={`tp-side tp-side-${side.toLowerCase()}${on.includes(beat) ? " is-hot" : ""}`}>
            <rect x={X.span + 52} y={MID + y - 9} width="18" height="18" rx="4" />
            <text x={X.span + 61} y={MID + y + 4} textAnchor="middle" className="vp-mono tp-side-letter">{side}</text>
            <text x={X.span + 78} y={MID + y + 4} className="vp-mono vp-row">
              arg #3 · <tspan className="tp-side-arg">{arg}</tspan>
            </text>
          </g>
        ))}
        {SIDE[beat] && <text x={X.span} y={MID - SPAN.h / 2 - 8} className="vp-mono vp-kind vp-kind-on">{SIDE[beat]}</text>}
      </g>

      <g className={`tp-shelf${beat === "diff" ? " is-empty" : ""}`}>
        {KEPT.map((k) => (
          <g key={k.id} className={`tp-kept${beat === "record" ? " is-filling" : ""}${beat === "diff" ? "" : state(beat, k.on)}`} style={{ "--at": `${k.at + 0.4}s` } as React.CSSProperties}>
            <line x1={k.cx} y1={k.from} x2={k.cx} y2={SHELF - 1} />
            <rect x={k.cx - k.w / 2} y={SHELF} width={k.w} height="26" rx="4" className="tp-kept-empty" />
            <g className="tp-kept-full">
              <rect x={k.cx - k.w / 2} y={SHELF} width={k.w} height="26" rx="4" />
              <text x={k.cx} y={SHELF + 17} textAnchor="middle" className="vp-mono">{k.label}</text>
            </g>
          </g>
        ))}
        <text x="1192" y={SHELF + 10} textAnchor="end" className="vp-mono vp-kind tp-shelf-name">A's recording, kept on the GPU</text>
        <text x="1192" y={SHELF + 23} textAnchor="end" className="vp-mono vp-faint">1.3 MB of span data · 12.7 GB of state</text>
      </g>

      <g key={beat} className="tp-motion">
        {RUNS[beat] && (
          <g className={`tp-run tp-chip-${RUNS[beat]!.toLowerCase()}`}>
            <rect x="0" y={MID - 10} width="20" height="20" rx="5" />
            <text x="10" y={MID + 4} textAnchor="middle" className="vp-mono">{RUNS[beat]}</text>
          </g>
        )}
        {beat === "record" &&
          KEPT.map((k) => (
            <circle key={k.id} cx={k.cx} cy={k.from} r="4.5" className="tp-copy" style={{ "--at": `${k.at}s`, "--d": `${SHELF - k.from}px` } as React.CSSProperties} />
          ))}
        {beat !== "verdict" &&
          KEPT.filter((k) => k.back.includes(beat as Phase)).map((k) => (
            <circle key={k.id} cx={k.cx} cy={SHELF} r="4.5" className="tp-copy" style={{ "--at": "0.1s", "--d": `${k.from - SHELF}px` } as React.CSSProperties} />
          ))}
      </g>

      {FOUND.filter(([, , phase, differs]) => (beat === "verdict" ? differs : beat === phase)).map(([x, y, , differs]) => (
        <g key={`${beat}-${x}-${y}`} className={`vp-mark${differs ? "" : " is-same"}${beat === "verdict" ? "" : " tp-pop"}`}>
          <circle cx={x} cy={y} r="9" />
          <text x={x} y={y + 3.5} textAnchor="middle" className="vp-mono">{differs ? "✗" : "="}</text>
        </g>
      ))}
    </svg>
  );
}

const still = () => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

export default function TestPlan() {
  const [at, setAt] = useState(BEATS.length - 1);
  const [playing, setPlaying] = useState(() => !still());
  const [held, setHeld] = useState(false);

  useEffect(() => {
    if (!playing || held) return;
    const next = window.setTimeout(() => setAt((i) => (i + 1) % BEATS.length), BEATS[at].ms);
    return () => window.clearTimeout(next);
  }, [playing, held, at]);

  const beat = BEATS[at].id;
  return (
    <div className="tp" onMouseEnter={() => setHeld(true)} onMouseLeave={() => setHeld(false)}>
      <div className="vp-plan-bar">
        <span className="tp-legend">
          <span>program decode · 646 calls</span>
          <span><i className="tp-chip tp-chip-a">A</i> reference</span>
          <span><i className="tp-chip tp-chip-b">B</i> candidate · differs in call #16</span>
        </span>
        <span>Qwen3.8-27B · 1 GPU · 10.1 s</span>
      </div>

      <div className="vp-plan-scroll tp-canvas">
        <Plan beat={beat} />
      </div>

      <div className="tp-beats">
        {BEATS.map((b, i) => (
          <button
            key={b.id}
            type="button"
            aria-pressed={at === i}
            className={`tp-beat${at === i ? " active" : ""}${b.bad ? " is-bad" : ""}`}
            onClick={() => {
              setAt(i);
              setPlaying(false);
            }}
          >
            <span className="tp-beat-tag">{b.id === "verdict" ? "FAIL" : `${i + 1} · ${b.id}`}</span>
            <span className="tp-beat-name">{b.name}</span>
            <span className="tp-beat-fact">
              {b.bad ? "✗ " : "✓ "}
              {b.fact}
            </span>
          </button>
        ))}
      </div>

      <div className="tp-report-bar">
        <span>$ kern test · 9 of the report's 28 lines</span>
        <button type="button" onClick={() => setPlaying((p) => !p)}>
          {playing ? "❚❚ pause" : "▶ play"}
        </button>
      </div>
      <ul className={`tp-report${beat === "verdict" ? "" : " is-stepping"}`}>
        {REPORT.map(([phase, line]) => {
          const cut = line.indexOf(" ");
          const bad = phase === "verdict" || line.includes("✗");
          return (
            <li key={line} className={`${phase === beat ? "is-on" : ""}${bad ? " is-bad" : ""}`}>
              <b>{line.slice(0, cut)}</b>
              {line.slice(cut)}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
