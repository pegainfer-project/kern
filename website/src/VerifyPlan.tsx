import { useState } from "react";
import "./verify-plan.css";

// The plan of one program of a four-kernel manifest, drawn the way a query
// planner draws a plan: operators, what flows between them, and what the
// verifier knows about each. Every diagnostic below is `kern verify`'s own
// line for that manifest with one edit.

type Check = "wiring" | "dataflow" | "bounds" | "pins" | "index" | "unused" | "protocol";
type Lens = "all" | Check;

const CHECKS: { id: Check; name: string; hint: string; rule: string; refused: string[] }[] = [
  {
    id: "wiring",
    name: "Arguments fit parameters",
    hint: "arity · kind · dtype · scalar range",
    rule: "Every call against its op's interface: argument count, kind, dtype, and that a var's whole range fits the scalar it is passed as.",
    refused: [
      "program `decode` call #1: arg #0: buffer `token_ids` has dtype i64 but param expects bf16",
      "program `decode` call #3: op `argmax` takes 3 params, got 2 args",
      "program `decode` call #0: arg #3: var `tokens` max 4096 exceeds u8 range",
    ],
  },
  {
    id: "dataflow",
    name: "Reads follow writes",
    hint: "per program, in call order",
    rule: "In call order: nothing is read before it is written, nothing writes an input or a weight, and every output is written.",
    refused: [
      "program `decode` call #0: arg #0: buffer `x` is read before ever being written",
      "program `decode` call #0: arg #2: op writes to read-only weight buffer `embed.w`",
      "output buffer `next_token` is never written by any program",
    ],
  },
  {
    id: "bounds",
    name: "Sizes at the corners",
    hint: "bytes and geometry at var min / max",
    rule: "Byte sizes and launch geometry, evaluated where each var is smallest and largest.",
    refused: [
      "op `argmax` launch #0 (argmax_rows): block [2048, 1, 1] exceeds 1024 threads or has a zero dim",
      "buffer `logits`: byte size overflows u64 at var upper bounds",
    ],
  },
  {
    id: "pins",
    name: "Kernels by hash",
    hint: "every module names a sha256",
    rule: "A module is its sha256. The runtime loads by hash, never by file name.",
    refused: ["module `c`: sha256 `c0ffee` is not 64 hex chars"],
  },
  {
    id: "index",
    name: "Indices name their state",
    hint: "slot and page tables resolve",
    rule: "A buffer of indices declares what it indexes, and the target must exist.",
    refused: ["buffer `block_table` domain: `index_into` unknown buffer/state `kvx`"],
  },
  {
    id: "unused",
    name: "Nothing left over",
    hint: "no dead buffer, op or var",
    rule: "Every declaration is used by some program.",
    refused: ["buffer `spare` is never used by any program", "op `argmax` is never called"],
  },
  {
    id: "protocol",
    name: "A serving contract",
    hint: "fills and row shapes",
    rule: "What a driver feeds and reads back: fills, page tables, and one program per row shape.",
    refused: [
      "program `decode_wide` and `decode` take the same rows per sequence; a row shape names one program, and what differs with the batch is a launch's `when`",
    ],
  },
];

const PASS = `$ kern verify figure.json
 INFO figure.json: \`figure\` schema v7, verified
  rows      \`tokens\` <= 4096
  groups    \`seqs\` <= 128
  fill      tokens      \`next_token\` i64 [seqs]
  fill      slot        \`slot_mapping\` i64 [tokens]
  fill      token       \`token_ids\` i64 [tokens]
  pages     \`block_table\` [seqs, 64]
  forward   \`decode\`: <= 128 sequences of 1 rows, hands back \`next_token\`
  forward   \`prefill\`: <= 1 sequences of rows as fed, state only`;

const SPINE = 270;
const NODE = { x: 150, w: 240, h: 66 };
const TILE = { x: 175, w: 190, h: 40 };
const SIDE = { x: 424, w: 168 };
const Y = { ids: 40, a: 106, x: 196, b: 260, h: 350, c: 414, logits: 504, d: 568, out: 658 };

type Kernel = {
  letter: string;
  op: string;
  call: number;
  y: number;
  entry: string;
  module: string;
  geometry: string;
  corner: string;
};

const KERNELS: Kernel[] = [
  { letter: "A", op: "embed", call: 0, y: Y.a, entry: "embed_rows", module: "a", geometry: "block 256 · grid [tokens, 1, 1]", corner: "1 … 4096 blocks" },
  { letter: "B", op: "rms_norm", call: 1, y: Y.b, entry: "rms_norm", module: "b", geometry: "block 256 · grid [tokens, 1, 1]", corner: "1 … 4096 blocks" },
  { letter: "C", op: "attend", call: 2, y: Y.c, entry: "attend_paged", module: "c", geometry: "block 128 · grid [seqs, 4, 1]", corner: "4 … 512 blocks" },
  { letter: "D", op: "argmax", call: 3, y: Y.d, entry: "argmax_rows", module: "d", geometry: "block 1024 · grid [seqs, 1, 1]", corner: "1 … 128 blocks" },
];

type Flow = { name: string; tag?: string; y: number; type: string; size: string; life: string; fill?: string; kind: "input" | "workspace" | "output" };

const FLOWS: Flow[] = [
  { name: "token_ids", y: Y.ids, type: "i64 [tokens]", size: "≤ 32 KiB at tokens = 4096", life: "fed by the driver · read by #0", fill: "fill: token", kind: "input" },
  { name: "x", tag: "1", y: Y.x, type: "bf16 [tokens, 256]", size: "≤ 2.0 MiB at tokens = 4096", life: "written by #0 · read by #1", kind: "workspace" },
  { name: "h", tag: "2", y: Y.h, type: "bf16 [tokens, 256]", size: "≤ 2.0 MiB at tokens = 4096", life: "written by #1 · read by #2", kind: "workspace" },
  { name: "logits", tag: "3", y: Y.logits, type: "bf16 [seqs, 32000]", size: "≤ 7.8 MiB at seqs = 128", life: "written by #2 · read by #3", kind: "workspace" },
  { name: "next_token", y: Y.out, type: "i64 [seqs]", size: "≤ 1 KiB at seqs = 128", life: "written by #3 · read by the driver", fill: "fill: tokens", kind: "output" },
];

// The arguments on the spine, top to bottom: where the segment starts and ends, and the port it is.
const PORTS: [number, number, string][] = [
  [Y.ids + TILE.h, Y.a, "arg #0 · in buffer<i64>"],
  [Y.a + NODE.h, Y.x, "arg #2 · out buffer<bf16>"],
  [Y.x + TILE.h, Y.b, "arg #0 · in buffer<bf16>"],
  [Y.b + NODE.h, Y.h, "arg #2 · out buffer<bf16>"],
  [Y.h + TILE.h, Y.c, "arg #0 · in buffer<bf16>"],
  [Y.c + NODE.h, Y.logits, "arg #4 · out buffer<bf16>"],
  [Y.logits + TILE.h, Y.d, "arg #0 · in buffer<bf16>"],
  [Y.d + NODE.h, Y.out, "arg #1 · out buffer<i64>"],
];

type SideInput = { name: string; y: number; type: string; kind: string; into: number; lens: Check[] };

const SIDES: SideInput[] = [
  { name: "embed.w", y: Y.a + 13, type: "bf16 [32000, 256]", kind: "weight", into: Y.a + 31, lens: ["dataflow", "wiring"] },
  { name: "norm.w", y: Y.b + 13, type: "bf16 [256]", kind: "weight", into: Y.b + 31, lens: ["dataflow", "wiring"] },
  { name: "slot_mapping", y: Y.c - 16, type: "i64 [tokens] → kv", kind: "input · fill: slot", into: Y.c + 16, lens: ["index", "protocol", "wiring"] },
  { name: "block_table", y: Y.c + 44, type: "i32 [seqs, 64] → kv", kind: "input · pages", into: Y.c + 50, lens: ["index", "protocol", "wiring"] },
];

// Where each lens's diagnostics land on the plan, in the order of `refused`.
const MARKS: Record<Check, [number, number][]> = {
  wiring: [[SPINE + 16, Y.b - 4], [NODE.x + NODE.w, Y.d], [NODE.x + NODE.w, Y.a]],
  dataflow: [[TILE.x + TILE.w, Y.x], [SIDE.x + SIDE.w, Y.a + 13], [TILE.x + TILE.w, Y.out]],
  bounds: [[NODE.x + NODE.w, Y.d], [TILE.x + TILE.w, Y.logits]],
  pins: [[NODE.x + NODE.w, Y.c]],
  index: [[SIDE.x + SIDE.w, Y.c + 44]],
  unused: [[118, Y.x + 2], [NODE.x + NODE.w, Y.d]],
  protocol: [[118, 14]],
};

function state(lens: Lens, ...mine: Check[]) {
  if (lens === "all") return "";
  return mine.includes(lens) ? " is-on" : " is-dim";
}

function Plan({ lens }: { lens: Lens }) {
  const note = (f: Flow) => (lens === "dataflow" ? f.life : lens === "protocol" && f.fill ? f.fill : f.size);
  return (
    <svg className="vp-svg" viewBox="0 0 600 712" role="img" aria-label="The decode program of a four-kernel manifest as a plan: kernels A to D, the buffers between them, and what kern verify checks on each.">
      <defs>
        <marker id="vp-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 L8 4 L0 8 z" className="vp-arrowhead" />
        </marker>
      </defs>

      <g className={`vp-program${state(lens, "protocol")}`}>
        <rect x="10" y="4" width="108" height="20" rx="3" />
        <text x="18" y="18" className="vp-mono vp-strong">program decode</text>
        <text x="128" y="18" className="vp-mono vp-faint">batch {"{"} groups: 128, rows: 1 {"}"} · vars tokens ≤ 4096, seqs ≤ 128</text>
      </g>

      {PORTS.map(([from, to, label]) => (
        <g key={from} className={`vp-edge${state(lens, "wiring", "dataflow")}`}>
          <line x1={SPINE} y1={from} x2={SPINE} y2={to - 1} markerEnd="url(#vp-arrow)" />
          <text x={SPINE - 10} y={(from + to) / 2 + 3} textAnchor="end" className={`vp-mono vp-port${lens === "wiring" ? " is-shown" : ""}`}>{label}</text>
        </g>
      ))}

      {FLOWS.map((f) => (
        <g key={f.name} className={`vp-flow vp-${f.kind}${state(lens, "wiring", "dataflow", "bounds", ...(f.fill ? (["protocol"] as Check[]) : []))}`}>
          <rect x={TILE.x} y={f.y} width={TILE.w} height={TILE.h} rx="20" />
          {f.tag && (
            <g>
              <circle cx={TILE.x + 21} cy={f.y + 20} r="10" className="vp-tag" />
              <text x={TILE.x + 21} y={f.y + 24} textAnchor="middle" className="vp-mono vp-tag-text">{f.tag}</text>
            </g>
          )}
          <text x={TILE.x + (f.tag ? 38 : 18)} y={f.y + 17} className="vp-mono vp-strong">{f.name}</text>
          <text x={TILE.x + (f.tag ? 38 : 18)} y={f.y + 31} className="vp-mono vp-faint">{f.type}</text>
          <text x={TILE.x + TILE.w + 14} y={f.y + 24} className={`vp-mono vp-note${lens === "bounds" || lens === "dataflow" || (lens === "protocol" && f.fill) ? " is-hot" : ""}`}>{note(f)}</text>
        </g>
      ))}

      {SIDES.map((s) => (
        <g key={s.name} className={`vp-side${state(lens, ...s.lens)}`}>
          <path d={`M${SIDE.x} ${s.y + 18} H${SIDE.x - 16} V${s.into} H${NODE.x + NODE.w + 3}`} markerEnd="url(#vp-arrow)" />
          <rect x={SIDE.x} y={s.y} width={SIDE.w} height="36" rx="4" />
          <text x={SIDE.x + 10} y={s.y + 15} className="vp-mono vp-strong">{s.name}</text>
          <text x={SIDE.x + 10} y={s.y + 28} className="vp-mono vp-faint">{s.type}</text>
          <text x={SIDE.x + SIDE.w} y={s.y - 4} textAnchor="end" className="vp-mono vp-kind">{s.kind}</text>
        </g>
      ))}

      <g className={`vp-state${state(lens, "index", "dataflow", "wiring")}`}>
        <path d={`M118 ${Y.c + 33} H${NODE.x - 3}`} markerStart="url(#vp-arrow)" markerEnd="url(#vp-arrow)" />
        <rect x="10" y={Y.c + 10} width="108" height="46" rx="4" />
        <text x="20" y={Y.c + 29} className="vp-mono vp-strong">state kv</text>
        <text x="20" y={Y.c + 44} className="vp-mono vp-faint">512 B / token</text>
        <text x="10" y={Y.c + 4} className="vp-mono vp-kind">inout · paged</text>
      </g>

      {lens === "unused" && (
        <g className="vp-ghost">
          <rect x="10" y={Y.x + 2} width="108" height="36" rx="4" />
          <text x="20" y={Y.x + 17} className="vp-mono vp-strong">spare</text>
          <text x="20" y={Y.x + 30} className="vp-mono vp-faint">f32 [tokens, 8]</text>
        </g>
      )}

      {KERNELS.map((k) => (
        <g key={k.letter} className={`vp-kernel${state(lens, "wiring", "bounds", "pins", "unused")}`}>
          <rect x={NODE.x} y={k.y} width={NODE.w} height={NODE.h} rx="5" />
          <rect x={NODE.x} y={k.y} width="28" height={NODE.h} rx="5" className="vp-letter-bg" />
          <text x={NODE.x + 14} y={k.y + 38} textAnchor="middle" className="vp-letter">{k.letter}</text>
          <text x={NODE.x + 40} y={k.y + 20} className="vp-mono vp-op">{k.op}</text>
          <text x={NODE.x + NODE.w - 10} y={k.y + 20} textAnchor="end" className="vp-mono vp-faint">call #{k.call}</text>
          <text x={NODE.x + 40} y={k.y + 37} className={`vp-mono vp-row${lens === "pins" ? " is-hot" : ""}`}>{k.entry} · module {k.module} · sha256</text>
          <text x={NODE.x + 40} y={k.y + 53} className={`vp-mono vp-row${lens === "bounds" ? " is-hot" : ""}`}>{lens === "bounds" ? `${k.geometry.split(" · ")[0]} · ${k.corner}` : k.geometry}</text>
        </g>
      ))}

      {lens !== "all" &&
        MARKS[lens].map(([x, y], i) => (
          <g key={i} className="vp-mark">
            <circle cx={x} cy={y} r="9" />
            <text x={x} y={y + 3.5} textAnchor="middle" className="vp-mono">{i + 1}</text>
          </g>
        ))}
    </svg>
  );
}

export default function VerifyPlan() {
  const [lens, setLens] = useState<Lens>("all");
  const active = CHECKS.find((c) => c.id === lens);
  return (
    <div className="vp">
      <div className="vp-plan">
        <div className="vp-plan-bar">
          <span>PLAN · figure.json</span>
          <span>4 kernels · 3 workspace buffers · 1 state</span>
        </div>
        <div className="vp-plan-scroll">
          <Plan lens={lens} />
        </div>
      </div>

      <div className="vp-panel">
        <div className="vp-checks" role="tablist" aria-label="What kern verify checks">
          <button type="button" role="tab" aria-selected={lens === "all"} className={`vp-check${lens === "all" ? " active" : ""}`} onClick={() => setLens("all")}>
            <span className="vp-check-name">The whole plan</span>
            <span className="vp-check-hint">7 checks, one pass, no GPU</span>
          </button>
          {CHECKS.map((c) => (
            <button key={c.id} type="button" role="tab" aria-selected={lens === c.id} className={`vp-check${lens === c.id ? " active" : ""}`} onClick={() => setLens(c.id)}>
              <span className="vp-check-name">{c.name}</span>
              <span className="vp-check-hint">{c.hint}</span>
            </button>
          ))}
        </div>

        <div className="vp-detail">
          {active ? (
            <>
              <p className="vp-rule">{active.rule}</p>
              <div className="vp-refused-head">Refused with</div>
              <ol className="vp-refused">
                {active.refused.map((line) => (
                  <li key={line}>{line}</li>
                ))}
              </ol>
            </>
          ) : (
            <>
              <p className="vp-rule">
                A manifest is a plan: kernels, the buffers between them, the state they share. <code>kern verify</code> reports every
                diagnostic in one pass, before anything touches a GPU.
              </p>
              <pre className="vp-pass">{PASS}</pre>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
