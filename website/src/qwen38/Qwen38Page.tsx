import { useEffect } from "react";
import { DIVERGE, MARGINS } from "./margins";

const REPO = "https://github.com/pegainfer-project/kern";
// the last revision that carried the record, its manifests and its generator
const TREE = `${REPO}/blob/a0101a6e9df2e0f4f670231fca885f7ea3333ddb`;
const DOC = `${TREE}/docs/qwen38-bringup.md`;

// ---------------------------------------------------------------- timeline
const T0 = 12 * 60 + 2; // 12:02Z
const T1 = 14 * 60 + 27; // 14:27Z
const min = (hhmm: string) => {
  const [h, m] = hhmm.split(":").map(Number);
  return h * 60 + m;
};
const X = (hhmm: string) => ((min(hhmm) - T0) / (T1 - T0)) * 1000;

type Seg = { s: string; e: string; l: string; stage: 1 | 2 | 3; lane?: number };
const SEGS: Seg[] = [
  { s: "12:02", e: "12:10", l: "capture", stage: 1 },
  { s: "12:12", e: "12:40", l: "kernel ABIs", stage: 1 },
  { s: "12:31", e: "12:54", l: "weights · generator", stage: 1, lane: 1 },
  { s: "12:55", e: "13:20", l: "first end-to-end", stage: 1 },
  { s: "13:20", e: "13:37", l: "bisect", stage: 1 },
  { s: "13:38", e: "13:45", l: "✓", stage: 1 },
  { s: "13:45", e: "14:00", l: "spec capture · draft", stage: 2 },
  { s: "14:00", e: "14:16", l: "six programs", stage: 2 },
  { s: "14:16", e: "14:20", l: "✓", stage: 2 },
  { s: "14:20", e: "14:27", l: "docs", stage: 3 },
];
const EVENTS = [
  { t: "13:37", l: "the one human intervention", kind: "human", row: 0, end: false },
  { t: "13:14", l: "GPU node taken → moved", kind: "ext", row: 1, end: false },
  { t: "13:26", l: "taken again → moved", kind: "ext", row: 2, end: false },
  { t: "14:16", l: "first speculative tokens", kind: "milestone", row: 1, end: true },
];

function Timeline() {
  const laneY = [70, 104];
  return (
    <svg className="q-timeline" viewBox="0 0 1000 200" role="img" aria-label="Wall-clock timeline from 12:02 to 14:27 UTC: Stage 1 target model accepted at 13:45, Stage 2 speculative decoding accepted at 14:20, evidence written by 14:27; two GPU-node moves and one human intervention marked">
      {[12 * 60 + 30, 13 * 60, 13 * 60 + 30, 14 * 60].map((t) => {
        const x = ((t - T0) / (T1 - T0)) * 1000;
        const label = `${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}Z`;
        return (
          <g key={t}>
            <line x1={x} x2={x} y1={56} y2={132} className="q-tick" />
            <text x={x} y={150} className="q-mono q-tick-label" textAnchor="middle">{label}</text>
          </g>
        );
      })}
      {SEGS.map((s) => {
        const x = X(s.s);
        const w = X(s.e) - x;
        const y = laneY[s.lane ?? 0];
        return (
          <g key={s.l + s.s} className={`q-seg q-stage-${s.stage}`}>
            <rect x={x} y={y - 12} width={w} height={24} />
            <text x={x + 5} y={y + 4} className="q-mono q-seg-label">{s.l}</text>
          </g>
        );
      })}
      {/* stage brackets */}
      <g className="q-mono q-bracket">
        <text x={X("12:02")} y={22}>STAGE 1 · TARGET MODEL</text>
        <text x={X("13:45") + 4} y={22}>STAGE 2 · SPECULATION</text>
        <text x={X("14:20") + 4} y={22} textAnchor="start">3 · DOCS</text>
        <line x1={X("12:02")} x2={X("13:45") - 4} y1={30} y2={30} />
        <line x1={X("13:45")} x2={X("14:20") - 4} y1={30} y2={30} />
        <line x1={X("14:20")} x2={1000} y1={30} y2={30} />
      </g>
      {/* commits */}
      {["13:45", "14:20", "14:27"].map((t) => (
        <g key={t} className="q-commit">
          <circle cx={X(t)} cy={40} r={4} />
        </g>
      ))}
      {EVENTS.map((e) => {
        const x = X(e.t);
        const y = 168 + e.row * 15;
        return (
          <g key={e.t} className={`q-event q-event-${e.kind}`}>
            <line x1={x} x2={x} y1={56} y2={y - 10} />
            <circle cx={x} cy={y - 10} r={e.kind === "human" ? 5 : 3} />
            <text x={e.end ? x - 9 : x + 9} y={y - 6} className="q-mono" textAnchor={e.end ? "end" : "start"}>{e.l}</text>
          </g>
        );
      })}
    </svg>
  );
}

// ------------------------------------------------------------ cost split
const COST = [
  { l: "runtime + schema", n: 49, cls: "runtime" },
  { l: "generator", n: 1417, cls: "gen" },
  { l: "six kernels", n: 360, cls: "kern" },
  { l: "one-off tooling", n: 1000, cls: "tool" },
];
function CostBar() {
  const total = COST.reduce((a, c) => a + c.n, 0);
  let x = 0;
  return (
    <svg className="q-cost" viewBox="0 0 1000 120" role="img" aria-label="Lines of code by location: runtime and schema 49, generator 1417, six kernels 360, one-off tooling about 1000">
      {COST.map((c) => {
        const w = Math.max((c.n / total) * 1000, 6);
        const el = (
          <g key={c.l} className={`q-cost-${c.cls}`}>
            <rect x={x} y={20} width={w - 2} height={44} />
            {c.cls !== "runtime" && (
              <text x={c.cls === "tool" ? x + w - 8 : x + 8} y={94} className={`q-mono q-cost-label${c.cls === "tool" ? " q-cost-end" : ""}`}>
                {`${c.l.toUpperCase()} · ${c.n.toLocaleString()}`}
              </text>
            )}
          </g>
        );
        x += w;
        return el;
      })}
      <g className="q-cost-runtime">
        <path d="M6 66 L6 100" />
        <text x={14} y={112} className="q-mono q-cost-label q-cost-callout">RUNTIME + SCHEMA · 49 LINES</text>
      </g>
    </svg>
  );
}

// --------------------------------------------------------- architecture
function Sketch() {
  return (
    <svg className="q-sketch" viewBox="0 0 1000 420" role="img" aria-label="Two vLLM captures feed one generator, which writes two manifests plus pinned kernels and six handwritten ones; the unchanged kern runtime executes them on the GPU">
      <defs>
        <filter id="rough" x="-2%" y="-2%" width="104%" height="104%">
          <feTurbulence type="fractalNoise" baseFrequency="0.035" numOctaves="2" seed="7" />
          <feDisplacementMap in="SourceGraphic" scale="1.8" />
        </filter>
        <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M1 1 L9 5 L1 9" fill="none" stroke="currentColor" strokeWidth="1.6" />
        </marker>
      </defs>
      <g filter="url(#rough)" className="q-lines">
        {/* captures */}
        <rect x={20} y={70} width={230} height={62} />
        <rect x={20} y={170} width={230} height={62} />
        {/* generator */}
        <rect x={310} y={95} width={240} height={112} className="q-fill-blue" />
        {/* outputs */}
        <rect x={620} y={40} width={220} height={54} />
        <rect x={620} y={110} width={220} height={54} />
        <rect x={620} y={190} width={220} height={54} />
        <rect x={620} y={260} width={220} height={54} className="q-fill-green" />
        {/* runtime */}
        <rect x={880} y={40} width={110} height={274} className="q-fill-ink" />
        {/* arrows */}
        <path d="M250 101 C280 101, 285 130, 310 132" markerEnd="url(#ah)" />
        <path d="M250 201 C280 201, 285 172, 310 170" markerEnd="url(#ah)" />
        <path d="M550 130 C590 130, 590 67, 620 67" markerEnd="url(#ah)" />
        <path d="M550 145 C590 145, 590 137, 620 137" markerEnd="url(#ah)" />
        <path d="M550 165 C590 165, 590 217, 620 217" markerEnd="url(#ah)" />
        <path d="M550 180 C590 180, 590 287, 620 287" markerEnd="url(#ah)" />
        {[67, 137, 217, 287].map((y) => (
          <path key={y} d={`M840 ${y} L880 ${y}`} markerEnd="url(#ah)" />
        ))}
        {/* weights, bypassing the generator */}
        <rect x={310} y={300} width={240} height={54} />
        <path d="M550 327 C700 327, 800 345, 880 305" markerEnd="url(#ah)" />
        {/* GPU */}
        <path d="M935 314 L935 372" markerEnd="url(#ah)" />
        <rect x={880} y={372} width={110} height={36} />
      </g>
      <g className="q-mono q-sketch-text">
        <text x={30} y={95}>vLLM 0.28 · CAPTURE</text>
        <text x={30} y={117} className="q-dim">target · 200 cubins</text>
        <text x={30} y={195}>vLLM 0.28 · CAPTURE</text>
        <text x={30} y={217} className="q-dim">speculative · 173 cubins</text>

        <text x={322} y={125} className="q-on-blue q-bold">GENERATOR</text>
        <text x={322} y={147} className="q-on-blue">gen_qwen35.py · 1,417 lines</text>
        <text x={322} y={169} className="q-on-blue">every line of model</text>
        <text x={322} y={191} className="q-on-blue">knowledge lives here</text>

        <text x={630} y={63}>qwen3.8-27b.json</text>
        <text x={630} y={83} className="q-dim">prefill · decode</text>
        <text x={630} y={133}>qwen3.8-27b-dflash2.json</text>
        <text x={630} y={153} className="q-dim">6 programs · 40 ops</text>
        <text x={630} y={213}>25 PINNED CUBINS</text>
        <text x={630} y={233} className="q-dim">sha256 · from both captures</text>
        <text x={630} y={283} className="q-bold">6 HANDWRITTEN KERNELS</text>
        <text x={630} y={303}>&lt; 150 lines each</text>

        <text x={890} y={66} className="q-on-ink q-bold">RUNTIME</text>
        <text x={890} y={88} className="q-on-ink">unchanged</text>
        <text x={890} y={110} className="q-on-ink">+49 lines</text>
        <text x={890} y={150} className="q-on-ink q-dim-light">no model</text>
        <text x={890} y={170} className="q-on-ink q-dim-light">branches</text>
        <text x={890} y={190} className="q-on-ink q-dim-light">no GDN</text>
        <text x={890} y={210} className="q-on-ink q-dim-light">no DFlash</text>
        <text x={890} y={230} className="q-on-ink q-dim-light">no Qwen</text>

        <text x={322} y={323}>WEIGHTS · 54 GB</text>
        <text x={322} y={343} className="q-dim">target + draft, bound by name</text>
        <text x={890} y={396}>GB300 · TP1</text>
      </g>
    </svg>
  );
}

// -------------------------------------------------- checkpoint pages inset
function Pages() {
  const acc = 3; // illustrative: anchor + 3 drafts accepted -> resume at page 3
  return (
    <svg className="q-pages" viewBox="0 0 520 250" role="img" aria-label="Each GDN layer keeps eight checkpoint pages; a verify pass writes one page per row, the next round resumes from the last accepted row and overwrites the rest — rollback costs nothing">
      <g filter="url(#rough)" className="q-lines">
        {Array.from({ length: 8 }, (_, i) => (
          <rect key={i} x={20 + i * 60} y={60} width={54} height={64} className={i <= acc ? "q-fill-green" : ""} />
        ))}
        <path d={`M${20 + acc * 60 + 27} 124 L${20 + acc * 60 + 27} 172 L47 172 L47 128`} markerEnd="url(#ah)" />
        <path d="M20 40 L500 40" markerEnd="url(#ah)" className="q-thin" />
      </g>
      <g className="q-mono q-sketch-text">
        <text x={20} y={26}>VERIFY · 8 ROWS → 8 STATE PAGES PER GDN LAYER</text>
        {Array.from({ length: 8 }, (_, i) => (
          <text key={i} x={47 + i * 60} y={98} textAnchor="middle" fontSize={i === 0 ? 10 : 12}>{i === 0 ? "anchor" : `d${i}`}</text>
        ))}
        <text x={20} y={205}>next round resumes at page {acc} = last accepted row</text>
        <text x={20} y={227} className="q-dim">rejected pages are simply overwritten · no rollback kernels</text>
      </g>
    </svg>
  );
}

// ----------------------------------------------------------- throughput
function Bars({ a, b, la, lb, unit, max }: { a: number; b: number; la: string; lb: string; unit: string; max: number }) {
  return (
    <div className="q-bars">
      <div className="q-bar-row q-bar-kern">
        <span className="q-mono">{la}</span>
        <i style={{ width: `${(a / max) * 100}%` }} />
        <strong>{a}</strong>
      </div>
      <div className="q-bar-row q-bar-ref">
        <span className="q-mono">{lb}</span>
        <i style={{ width: `${(b / max) * 100}%` }} />
        <strong>{b}</strong>
      </div>
      <small className="q-mono">{unit}</small>
    </div>
  );
}

// -------------------------------------------------------------- margins
function Margins() {
  const W = 1000;
  const rowH = 46;
  const clip = 4;
  return (
    <svg className="q-margins" viewBox={`0 0 ${W} ${MARGINS.length * rowH + 30}`} role="img" aria-label="Top-2 logit margin at each of 200 decode steps for five prompts; every position where a kern configuration diverged from the reference has a margin of at most one bf16 quantum">
      {MARGINS.map((row, r) => {
        const base = r * rowH + rowH - 6;
        return (
          <g key={r}>
            <text x={0} y={base - 14} className="q-mono q-dim">P{r}</text>
            {row.map((m, i) => {
              const h = (Math.min(m, clip) / clip) * (rowH - 14);
              return <rect key={i} x={30 + i * ((W - 30) / 200)} y={base - h} width={3.2} height={Math.max(h, 0.6)} className="q-m" />;
            })}
            {(DIVERGE[r] ?? []).map((p) => (
              <g key={p} className="q-m-div">
                <rect x={30 + p * ((W - 30) / 200) - 1} y={base - (rowH - 14) - 4} width={5.2} height={rowH - 10} />
                <text x={30 + p * ((W - 30) / 200) + 8} y={base - (rowH - 14) + 2} className="q-mono">{row[p] === 0 ? "0.0" : String(row[p])}</text>
              </g>
            ))}
          </g>
        );
      })}
      <text x={30} y={MARGINS.length * rowH + 22} className="q-mono q-dim">step 0</text>
      <text x={W} y={MARGINS.length * rowH + 22} className="q-mono q-dim" textAnchor="end">step 200 · bar height = top-2 margin, clipped at 4</text>
    </svg>
  );
}

export default function Qwen38Page() {
  // client-rendered: a deep link's target does not exist at the browser's own anchor jump
  useEffect(() => {
    const id = decodeURIComponent(window.location.hash.slice(1));
    if (!id) return;
    const frame = requestAnimationFrame(() => {
      document.getElementById(id)?.scrollIntoView({ behavior: "instant", block: "start" });
    });
    return () => cancelAnimationFrame(frame);
  }, []);
  return (
    <main className="q-page">
      <header className="site-header">
        <a className="wordmark" href="/" aria-label="Kern home">
          KERN<span className="wordmark-dot">■</span>
        </a>
        <nav aria-label="Primary navigation">
          <a href="#cost">COST</a>
          <a href="#how">HOW</a>
          <a href="#speed">SPEED</a>
          <a href="#ties">TIES</a>
          <a href={DOC} target="_blank" rel="noreferrer">TIMELINE ↗</a>
          <a className="github-link" href={REPO} target="_blank" rel="noreferrer">GITHUB ↗</a>
        </nav>
      </header>

      <section className="q-hero" id="top">
        <p className="eyebrow">BRING-UP LOG · 2026-08-31 · ONE AGENT, UNATTENDED</p>
        <div className="q-hero-grid">
          <div className="q-hero-main">
            <strong className="q-giant">+49</strong>
            <span className="q-giant-label">lines of runtime changed</span>
          </div>
          <div className="q-hero-side">
            <div>
              <strong>2h 18m</strong>
              <span>to Qwen3.8-27B running on kern —<br />a 64-layer hybrid linear-attention model<br />plus its DFlash2 speculative draft</span>
            </div>
            <div>
              <strong>1</strong>
              <span>human intervention</span>
            </div>
          </div>
        </div>
        <Timeline />
      </section>

      <section className="q-section" id="cost">
        <div className="section-number">01 / WHERE THE COST WENT</div>
        <h2>THE MODEL LANDED<br />IN THE GENERATOR.</h2>
        <CostBar />
        <ul className="q-list q-runtime-list">
          <li><code>State.bytes</code><span>a state may be fixed-size, not per-token</span></li>
          <li><code>verify</code><span>bounds-check offsets into such a state</span></li>
          <li><code>load_weights(&amp;[blob])</code><span>weights may arrive in several files</span></li>
          <li><code>spec</code><span>a speculative manifest declares its block size</span></li>
        </ul>
        <p className="q-aside">Those are the 49 lines. None of them says GDN, DFlash, or Qwen.</p>
      </section>

      <section className="q-section" id="how">
        <div className="section-number">02 / HOW</div>
        <h2>TWO CAPTURES IN.<br />TWO MANIFESTS OUT.</h2>
        <Sketch />
        <div className="q-how-pair">
          <Pages />
          <p className="q-aside">
            Under speculation the target's recurrent layers use the reference stack's own
            speculative kernels: one checkpoint page per verified row. The next round
            resumes from the last accepted row. That is the whole rollback story.
          </p>
        </div>
      </section>

      <section className="q-section" id="speed">
        <div className="section-number">03 / SPEED</div>
        <h2>PLAIN DECODE 85%.<br />SPECULATIVE AT PARITY.</h2>
        <div className="q-speed">
          <Bars a={81} b={95} la="KERN" lb="vLLM 0.28" unit="DECODE · tok/s" max={100} />
          <Bars a={178} b={176} la="KERN" lb="vLLM 0.28" unit="DFLASH2 SPECULATIVE · tok/s" max={220} />
          <div className="q-stat">
            <strong>2.61</strong>
            <span>tokens per round<br /><em>2.46 for the reference draft</em></span>
          </div>
        </div>
        <p className="q-foot q-mono">batch 1 · one GB300 · CUDA graphs on both sides · 5 prose prompts × 400 tokens · the 15% gap is 742 unfused launches per step, not arithmetic</p>
      </section>

      <section className="q-section" id="ties">
        <div className="section-number">04 / TIES</div>
        <h2>EVERY DIVERGENCE<br />WAS A BF16 TIE.</h2>
        <Margins />
        <div className="q-ties-grid">
          <div className="q-stat">
            <strong>1</strong>
            <span>GEMM · M=43, N=96<br /><em>4 of 4,128 values off by 1 ulp — a cuBLAS algorithm choice, the one launch not pinned by sha256</em></span>
          </div>
          <div className="q-stat q-stat-tight">
            <strong>≤&nbsp;0.125</strong>
            <span>top-2 margin at every divergence<br /><em>one bf16 quantum · median margin 1.1–2.0</em></span>
          </div>
          <p className="q-aside">
            Outputs were not byte-identical, so the agent bisected. The reference stack's own
            graph/eager and speculative/plain paths flip at the same positions.
          </p>
        </div>
      </section>

      <section className="q-section q-caveats" id="read">
        <div className="section-number">05 / READ IT YOURSELF</div>
        <ul className="q-list">
          <li><code>caveat</code><span>the agent started from a written brief with the environment facts filled in</span></li>
          <li><code>caveat</code><span>acceptance was relaxed from bit-exact to near-tie agreement — that was the one intervention</span></li>
          <li><code>caveat</code><span>single GPU, batch 1; two GPU-node moves mid-run are in the timeline</span></li>
        </ul>
        <div className="q-links">
          <a href={DOC} target="_blank" rel="noreferrer">FULL TIMELINE, UTC ↗</a>
          <a href={`${TREE}/examples/qwen3.8-27b-dflash2.json`} target="_blank" rel="noreferrer">THE SPECULATIVE MANIFEST ↗</a>
          <a href={`${TREE}/tools/gen_qwen35.py`} target="_blank" rel="noreferrer">THE GENERATOR ↗</a>
          <a href="/">← KERN</a>
        </div>
      </section>
    </main>
  );
}
