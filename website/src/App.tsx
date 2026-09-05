import { useEffect, useState } from "react";

type VerifyMode = "type" | "dataflow" | "abi";

const verifyExamples: Record<
  VerifyMode,
  { phase: string; source: string; target: string; error: string }
> = {
  type: {
    phase: "MANIFEST / TYPE",
    source: "q · buffer<bf16>",
    target: "arg 2 · buffer<fp8e4m3>",
    error: "buffer `q` has dtype bf16\nbut param expects fp8e4m3",
  },
  dataflow: {
    phase: "MANIFEST / DATAFLOW",
    source: "segm_out · unreadied",
    target: "reduce · in buffer<f32>",
    error: "scratch `segm_out` is read\nbefore any step wrote it",
  },
  abi: {
    phase: "LOAD / CUBIN ABI",
    source: "declared · [8, 8, 4]",
    target: "loaded · [8, 16, 4]",
    error: "no loaded instance matches\ndeclared param layout",
  },
};

function Arrow({ className = "" }: { className?: string }) {
  return (
    <svg className={`arrow ${className}`} viewBox="0 0 120 34" aria-hidden="true">
      <path d="M3 19 C34 12, 72 24, 109 15" />
      <path d="M99 7 L110 15 L100 26" />
    </svg>
  );
}

function Header() {
  return (
    <header className="site-header">
      <a className="wordmark" href="#top" aria-label="Kern home">
        KERN<span className="wordmark-dot">■</span>
      </a>
      <nav aria-label="Primary navigation">
        <a href="#swap">SWAP</a>
        <a href="#loop">LOOP</a>
        <a href="#evidence">EVIDENCE</a>
        <a href="#proof">PROOF</a>
        <a href="/schema/">SCHEMA</a>
        <a href="/perf/">PERF</a>
        <a href="/qwen38/">+49</a>
        <a
          className="github-link"
          href="https://github.com/pegainfer-project/kern"
          target="_blank"
          rel="noreferrer"
        >
          GITHUB ↗
        </a>
      </nav>
    </header>
  );
}

function HeroDiagram() {
  return (
    <div
      className="hero-diagram"
      aria-label="An agent writes a kernel implementation; kern verifies it and ships it into the runtime; bad implementations are refused before the first launch"
    >
      <div className="agent-node">
        <strong>✎ AGENT</strong>
        <small>writes an impl · 03:14 AM</small>
      </div>
      <Arrow className="hero-arrow-zero" />
      <div className="artifact-stack">
        <div className="file-sheet file-sheet-back">WEIGHTS</div>
        <div className="file-sheet file-sheet-mid">KERNELS</div>
        <div className="file-sheet file-sheet-front">
          <span>MANIFEST</span>
          <span className="file-code">kernel silu_mul</span>
          <span className="file-code">impl · hf:…/activation</span>
          <span className="file-code">sha256 · 73748b54…</span>
        </div>
      </div>
      <Arrow className="hero-arrow-one" />
      <div className="verify-stamp">
        <span className="verify-check">✓</span>
        <span>VERIFIED</span>
      </div>
      <div className="reject-tag">
        ✗ BAD IMPL
        <small>REFUSED · NO LAUNCH</small>
      </div>
      <Arrow className="hero-arrow-two" />
      <div className="runtime-chip">
        <span>THIN</span>
        <strong>RUNTIME</strong>
        <div className="chip-pins" aria-hidden="true" />
      </div>
      <svg className="loop-return" viewBox="0 0 340 90" aria-hidden="true">
        <path d="M330 14 C260 88, 90 88, 12 30" />
        <path d="M22 44 L10 28 L28 24" />
      </svg>
      <span className="loop-return-label">the loop runs unattended</span>
    </div>
  );
}

function Hero() {
  return (
    <section className="hero" id="top">
      <Header />
      <div className="hero-copy">
        <p className="eyebrow">VERIFIED GPU PROGRAMS · MODEL-AGNOSTIC RUNTIME</p>
        <h1>
          MACHINES WRITE
          <span>KERNELS</span>
          NOW.
        </h1>
        <div className="hero-bottomline">
          <p>
            Shipping one still takes a human review cycle.
            <br />
            kern closes the loop.
          </p>
          <a href="#swap">SEE ONE SHIP ↓</a>
        </div>
      </div>
      <HeroDiagram />
      <div className="hero-scale">
        <strong>&lt;3K</strong>
        <span>RUST SOURCE LINES<br />END TO END</span>
        <small>447 run · 655 runtime · 1,537 schema + verifier</small>
      </div>
      <div className="hero-proof">
        <strong>92%</strong>
        <span>of vLLM decode throughput</span>
        <small>Qwen3-4B · batch 1 · single GB300 · 377 vs 409 tok/s</small>
      </div>
    </section>
  );
}

function TheSwap() {
  return (
    <section className="section swap-section" id="swap">
      <div className="section-number">01 / THE SWAP</div>
      <div className="swap-header">
        <h2>
          SWAP THE
          <br />
          KERNEL.
        </h2>
        <div className="swap-header-side">
          <p className="swap-note">Nothing else changes.</p>
          <p className="swap-story">
            Three diff lines point one op at a kernel from the
            Hugging&nbsp;Face hub. The old entry is erased, the new module is
            verified into its slot — the rest of the program never moves.
          </p>
          <a className="swap-schema-link" href="/schema/">
            REGISTRY REFS IN THE SCHEMA →
          </a>
        </div>
      </div>

      <div
        className="swap-stage"
        aria-label="A three-line JSON diff swaps one implementation: the old engine entry is erased from the silu_mul op, a verified Hugging Face module drops in, and no other call is touched"
      >
        <div className="diff-window swap-diff">
          <div className="swap-diff-title">examples/qwen3-4b.json</div>
          <div className="diff-line context">
            <b> </b>
            <code>"silu_mul": {"{"} "impl": {"{"} "launches": [{"{"}</code>
          </div>
          <div className="diff-line removed">
            <b>-</b>
            <code>"entry": "_ZN4vllm18act_and_mul_kernel…packed_silu…"</code>
          </div>
          <div className="diff-line added">
            <b>+</b>
            <code>"module": "activation",</code>
          </div>
          <div className="diff-line added">
            <b>+</b>
            <code>"entry": "_ZN4vllm18act_and_mul_kernel…"</code>
          </div>
          <div className="diff-line context">
            <b> </b>
            <code>"modules": {"{"}</code>
          </div>
          <div className="diff-line added">
            <b>+</b>
            <code>"activation": {"{"} "source": "hf:kernels-community/activation/…",</code>
          </div>
          <div className="diff-line added">
            <b>+</b>
            <code>                "sha256": "73748b54…b1fe49aa" {"}"}</code>
          </div>
        </div>

        <div className="swap-flight" aria-hidden="true">
          <svg className="flight-paths" viewBox="0 0 250 400">
            <path className="flight-erase" d="M5 127 C12 260, 70 345, 226 278" />
            <path className="flight-erase" d="M212 292 L228 277 L208 268" />
            <path className="flight-ship" d="M202 200 C226 208, 240 226, 246 246" />
            <path className="flight-ship" d="M248 228 L246 247 L233 235" />
          </svg>
          <div className="flight-block">
            <span className="flight-kicker">NEW MODULE</span>
            <strong>hf:kernels-community/activation</strong>
            <small>sha256 · 73748b54…b1fe49aa</small>
            <i className="flight-verified">✓ VERIFIED</i>
          </div>
          <span className="flight-label label-erase">ERASED</span>
          <span className="flight-label label-ship">SWAPS IN</span>
        </div>

        <div className="swap-pipeline">
          <span className="pipeline-title">FORWARD PROGRAM · CALL ORDER</span>
          <div className="pipe-block">
            <code>rms_norm</code>
            <i>=</i>
          </div>
          <div className="pipe-block">
            <code>qkv_proj</code>
            <i>=</i>
          </div>
          <div className="pipe-block">
            <code>attention</code>
            <i>=</i>
          </div>
          <div className="pipe-block swap-slot">
            <code>silu_mul</code>
            <span className="old-impl">_ZN4vllm18act_and_mul_kernelI…</span>
            <div className="eraser" />
            <span className="shaving shaving-a" />
            <span className="shaving shaving-b" />
            <span className="shaving shaving-c" />
          </div>
          <div className="pipe-block">
            <code>down_proj</code>
            <i>=</i>
          </div>
          <div className="pipeline-foot">
            <b>0</b>
            <span>
              OTHER CALLS
              <br />
              TOUCHED
            </span>
          </div>
        </div>
      </div>

      <div className="swap-stamps">
        <span className="swap-stamp">NO TORCH</span>
        <span className="swap-stamp">NO PYTHON</span>
        <span className="swap-stamp stamp-green">BYTE-IDENTICAL</span>
      </div>
    </section>
  );
}

function TheLoop() {
  return (
    <section className="section loop-section" id="loop">
      <div className="section-number">02 / THE LOOP</div>
      <div className="loop-header">
        <h2>
          A KERNEL CHANGE IS
          <br />
          AN ENGINE CHANGE.
          <br />
          <span>NOT HERE.</span>
        </h2>
      </div>

      <div className="lanes">
        <div className="lane lane-elsewhere">
          <span className="lane-label">MONOLITHIC ENGINE</span>
          <div className="lane-steps">
            <i>patch the engine</i>
            <em>→</em>
            <i>open a PR</i>
            <em>→</em>
            <i>prove accuracy</i>
            <em>→</em>
            <i>CI every model</i>
            <em>→</em>
            <i>wait for review</i>
          </div>
          <strong className="lane-cost">WEEKS</strong>
        </div>
        <div className="lane lane-kern">
          <span className="lane-label">KERN</span>
          <div className="lane-steps">
            <i>swap the impl</i>
            <em>→</em>
            <i>
              verify <b>ms</b>
            </i>
            <em>→</em>
            <i>
              attest <b>7 s</b>
            </i>
          </div>
          <strong className="lane-cost cost-green">SHIPPED</strong>
        </div>
      </div>

      <div className="loop-thesis">
        <p>
          Agents can generate a thousand kernels a night. The scarce thing is a
          loop that can verify one and ship it with <b>no human in it</b> — a
          typed interface, a millisecond verifier, a byte-level oracle, a
          content-addressed registry.
        </p>
        <p className="loop-punch">The engine goes back to being an engine.</p>
      </div>
    </section>
  );
}

// Slightly hand-drawn rectangle: each edge bows by a pixel or so.
function sketchRect(x: number, y: number, w: number, h: number, bow = 1.4) {
  return [
    `M${x} ${y}`,
    `Q${x + w / 2} ${y - bow} ${x + w} ${y}`,
    `Q${x + w + bow} ${y + h / 2} ${x + w} ${y + h}`,
    `Q${x + w / 2} ${y + h + bow} ${x} ${y + h}`,
    `Q${x - bow} ${y + h / 2} ${x} ${y}`,
    "Z",
  ].join(" ");
}

const stages = ["rms_norm", "qkv_proj", "attention", "silu_mul", "down_proj", "argmax"];
const CUT = 3;

function ProgramRow({ y, tone }: { y: number; tone: "a" | "b" }) {
  return (
    <g className={`ev-row ev-row-${tone}`}>
      {stages.map((name, i) => {
        const x = 40 + i * 98;
        const cut = i === CUT;
        return (
          <g key={name} className={cut ? "ev-block ev-cut" : "ev-block"}>
            <path d={sketchRect(x, y, 88, 44)} />
            <text x={x + 44} y={y + 27}>{name}</text>
          </g>
        );
      })}
    </g>
  );
}

function EvidenceDiagram() {
  const cutX = 40 + CUT * 98 + 44; // centre of the swapped block
  const checks = [
    { y: 150, name: "NOISE", what: "A re-run vs A", result: "clean" },
    { y: 186, name: "LOCAL", what: "B vs A", result: "bit-identical" },
    { y: 222, name: "FUZZ ×6", what: "synth inputs", result: "±0 only" },
    { y: 258, name: "TIME", what: "cut alone", result: "−24%" },
  ];
  return (
    <svg
      className="evidence-diagram"
      viewBox="0 0 810 440"
      role="img"
      aria-label="A and B run one prompt in lockstep; at the swapped call the frontier inputs and A's outputs are snapshotted; every later check — noise floor, local compare, fuzz, timing — replays only that cut from the snapshot"
    >
      <text className="ev-label" x={40} y={28}>A · REFERENCE · hf:kernels-community/activation</text>
      <ProgramRow y={40} tone="a" />
      <text className="ev-label" x={40} y={428}>B · CANDIDATE · kernels/module_59.cubin (mined vLLM)</text>
      <ProgramRow y={356} tone="b" />

      {/* tap: A's cut → snapshot */}
      <path className="ev-wire ev-wire-blue" d={`M${cutX} 86 C${cutX - 3} 120, ${cutX + 4} 140, ${cutX} 172`} />
      <path className="ev-wire ev-wire-blue" d={`M${cutX - 7} 160 L${cutX} 173 L${cutX + 8} 161`} />
      <text className="ev-tag ev-tag-blue" x={cutX - 16} y={128} textAnchor="end">TAP</text>
      <text className="ev-tag" x={cutX - 16} y={143} textAnchor="end">frontier in · A out</text>

      {/* snapshot */}
      <g className="ev-snapshot">
        <path d={sketchRect(cutX - 78, 176, 156, 58, 1.8)} />
        <text x={cutX} y={200}>SNAPSHOT</text>
        <text className="ev-snapshot-sub" x={cutX} y={220}>35.7 MB · 72 cuts</text>
      </g>

      {/* replay: snapshot → B's cut */}
      <path className="ev-wire ev-wire-blue" d={`M${cutX} 236 C${cutX + 3} 280, ${cutX - 4} 310, ${cutX} 352`} />
      <path className="ev-wire ev-wire-blue" d={`M${cutX - 7} 340 L${cutX} 353 L${cutX + 8} 341`} />
      <text className="ev-tag ev-tag-blue" x={cutX - 16} y={300} textAnchor="end">REPLAY</text>
      <text className="ev-tag" x={cutX - 16} y={315} textAnchor="end">only the cut</text>

      {/* the four checks fan out of the snapshot */}
      {checks.map((c) => (
        <g key={c.name} className="ev-check">
          <path className="ev-wire" d={`M${cutX + 80} 205 C${cutX + 110} 205, ${cutX + 112} ${c.y - 4}, ${cutX + 134} ${c.y - 4}`} />
          <text className="ev-check-name" x={cutX + 142} y={c.y}>{c.name}</text>
          <text className="ev-check-what" x={cutX + 210} y={c.y}>{c.what}</text>
          <text className="ev-check-result" x={cutX + 312} y={c.y}>✓ {c.result}</text>
        </g>
      ))}

      {/* the rest of the program never runs again */}
      <text className="ev-tag" x={40} y={206}>one prompt · 17 tokens</text>
      <text className="ev-tag" x={40} y={221}>the model runs once.</text>
      <text className="ev-tag" x={40} y={236}>everything else replays the cut.</text>
    </svg>
  );
}

const timeline = [
  { name: "LOAD A+B", ms: 2400, tone: "muted", note: "weights, once" },
  { name: "TAP", ms: 130, tone: "blue", note: "the only full run" },
  { name: "NOISE", ms: 16, tone: "blue", note: "72 cuts" },
  { name: "FUZZ", ms: 1200, tone: "blue", note: "72 cuts × 6" },
  { name: "PERF", ms: 1800, tone: "blue", note: "eager + graph + sweep" },
  { name: "", ms: 1450, tone: "faint", note: "tokenizer, report" },
];
const TOTAL_MS = 7000;

function Timeline() {
  return (
    <div className="ev-timeline" role="img" aria-label="Seven seconds from swap to verdict: 2.4 s loading both programs, 130 ms tapping one prompt, 16 ms noise floor, 1.2 s fuzz, 1.8 s timing">
      <div className="ev-bar">
        {timeline.map((t, i) => (
          <i key={i} className={`ev-seg ev-seg-${t.tone}`} style={{ flexGrow: t.ms }} />
        ))}
      </div>
      <div className="ev-ticks">
        {timeline.filter((t) => t.name).map((t) => (
          <div key={t.name} className={`ev-tick ev-tick-${t.tone}`}>
            <b>{t.name}</b>
            <span>{t.ms >= 1000 ? `${(t.ms / 1000).toFixed(1)} s` : `${t.ms} ms`}</span>
            <small>{t.note}</small>
          </div>
        ))}
        <div className="ev-tick ev-tick-total">
          <b>SWAP → VERDICT</b>
          <span>{(TOTAL_MS / 1000).toFixed(1)} s</span>
          <small>one GB300</small>
        </div>
      </div>
    </div>
  );
}

function Evidence() {
  return (
    <section className="section evidence-section" id="evidence">
      <div className="section-number">03 / THE EVIDENCE</div>
      <div className="evidence-header">
        <div className="evidence-hero">
          <h2>
            SWAP VERIFIED
            <br />
            IN
          </h2>
          <strong className="evidence-time">7<i>s</i></strong>
        </div>
        <div className="evidence-header-side">
          <pre className="evidence-cmd">
            <b>$</b> kern-attest --a qwen3-4b.json{"\n"}
            {"             "}--b qwen3-4b-silu-mined.json
          </pre>
          <p>
            Diff, noise floor, bit-compare, six fuzz distributions, timing.
            <br />
            The old program is the oracle. No thresholds anywhere.
          </p>
        </div>
      </div>

      <Timeline />
      <p className="ev-timeline-why">
        The model runs <b>once</b>. Every check after the tap replays only the swapped cut —
        cost follows the cut, not the model.
      </p>

      <div className="evidence-stage">
        <EvidenceDiagram />
        <div className="evidence-verdict">
          <span className="verdict-kicker">VERDICT</span>
          <strong>PASS</strong>
          <p>value-identical at every cut</p>
          <small>only signed zeros differ · exit 0</small>
          <ol className="verdict-ladder">
            <li><b>0</b> PASS · bit / value / within noise</li>
            <li><b>2</b> INCONCLUSIVE · beyond noise, no oracle here</li>
            <li><b>1</b> FAIL · crash or out of domain</li>
          </ol>
        </div>
      </div>

      <div className="evidence-metrics">
        <div>
          <strong>72</strong>
          <span>CUTS CHECKED</span>
          <small>36 prefill + 36 decode</small>
        </div>
        <div>
          <strong>130<i>ms</i></strong>
          <span>OF REAL MODEL TIME</span>
          <small>one prompt, tapped once</small>
        </div>
        <div>
          <strong>6</strong>
          <span>FUZZ DISTRIBUTIONS</span>
          <small>uniform · normal · laplace · outliers · edge · special</small>
        </div>
        <div>
          <strong>0</strong>
          <span>TOKENS GENERATED</span>
          <small>bit-identical cuts imply it</small>
        </div>
      </div>
    </section>
  );
}

function Verifier() {
  const [mode, setMode] = useState<VerifyMode>("type");
  const example = verifyExamples[mode];

  return (
    <section className="section verifier-section">
      <div className="section-number">04 / CRASH EARLY</div>
      <div className="verifier-header">
        <h2>BAD DECLARATIONS<br /><span>STOP HERE.</span></h2>
        <p>Why the loop runs unattended.</p>
      </div>
      <div className="verifier-console">
        <div className="verify-tabs" role="group" aria-label="Verifier failure example">
          {(["type", "dataflow", "abi"] as VerifyMode[]).map((item) => (
            <button
              key={item}
              className={mode === item ? "active" : ""}
              onClick={() => setMode(item)}
              aria-pressed={mode === item}
            >
              {item.toUpperCase()}
            </button>
          ))}
        </div>
        <div className="broken-wire">
          <div className="wire-end">
            <span>SOURCE</span>
            <strong>{example.source}</strong>
          </div>
          <svg viewBox="0 0 330 80" aria-hidden="true">
            <path className="wire-left" d="M3 39 C75 20, 112 62, 148 39" />
            <path className="wire-right" d="M181 39 C224 13, 274 58, 326 33" />
            <path className="wire-break" d="M151 24 L178 57 M178 23 L151 58" />
          </svg>
          <div className="wire-end wire-target">
            <span>TARGET</span>
            <strong>{example.target}</strong>
          </div>
        </div>
        <div className="diagnostic">
          <span>{example.phase}</span>
          <pre>{example.error}</pre>
          <b>EXECUTION REFUSED</b>
        </div>
      </div>
      <div className="trust-boundary">
        <span>PROVES</span> declaration consistency
        <i />
        <span>TRUSTS</span> kernel behavior
      </div>
    </section>
  );
}

function Artifact() {
  return (
    <section className="section artifact-section" id="artifact">
      <div className="section-number">05 / THE ARTIFACT</div>
      <div className="artifact-title">
        <h2>A MODEL IS<br />A PROGRAM.</h2>
        <p>Everything the runtime needs.<br />Nothing about the model architecture.</p>
      </div>
      <div className="program-blueprint">
        <div className="blueprint-node source-node">
          <span className="node-kicker">PROVIDER</span>
          <strong>manifest.json</strong>
          <strong>kernels/*.cubin</strong>
          <strong>weights</strong>
        </div>
        <Arrow />
        <div className="manifest-window">
          <div className="manifest-rail">
            <span>V3</span>
            <span>QWEN3-4B</span>
          </div>
          <div className="manifest-body">
            <div><b>1</b><span>var</span></div>
            <div><b>1</b><span>opaque state</span></div>
            <div><b>310</b><span>buffers</span></div>
            <div><b>4</b><span>modules</span></div>
            <div><b>12</b><span>op interfaces</span></div>
            <div className="manifest-programs"><b>2</b><span>programs</span></div>
          </div>
        </div>
        <Arrow />
        <div className="blueprint-node executor-node">
          <span className="node-kicker">RUNTIME KNOWS</span>
          <strong>buffers</strong>
          <strong>state bytes</strong>
          <strong>calls</strong>
          <span className="executor-no">NO MODEL BRANCHES</span>
        </div>
      </div>
    </section>
  );
}

function Proof() {
  return (
    <section className="section proof-section" id="proof">
      <div className="section-number">06 / MEASURED</div>
      <h2>REAL MODEL.<br />REAL KERNELS.</h2>
      <div className="benchmark benchmark-decode">
        <div className="benchmark-label">
          <span>DECODE</span>
          <small>tok/s · higher is better</small>
        </div>
        <div className="bar-row">
          <span>KERN</span>
          <div className="bar-track"><i style={{ width: "92%" }} /></div>
          <strong>377</strong>
        </div>
        <div className="bar-row baseline">
          <span>vLLM</span>
          <div className="bar-track"><i style={{ width: "100%" }} /></div>
          <strong>409</strong>
        </div>
      </div>
      <div className="proof-pair">
        <div className="proof-stat prefill-stat">
          <span>CHUNKED PREFILL</span>
          <strong>37×</strong>
          <p><b>~60 ms</b> vs 2.18 s</p>
          <small>709-token prompt · chunked vs repeated decode path</small>
        </div>
        <div className="proof-stat spec-stat">
          <span>SPECULATIVE DECODE</span>
          <strong>2.4×</strong>
          <p><b>948</b> vs 388 tok/s</p>
          <small>32-token prompt · byte-equal greedy output in this run</small>
        </div>
        <div className="proof-stat kernels-stat">
          <span>NEW KERNELS IT TOOK</span>
          <strong>0</strong>
          <p><b>6</b> programs, same schema</p>
          <small>DSpark speculative decoding composed from existing kernels</small>
        </div>
        <a className="proof-stat model-stat" href="/qwen38/">
          <span>SECOND MODEL · QWEN3.8-27B →</span>
          <strong>+49</strong>
          <p><b>lines</b> of runtime + schema</p>
          <small>hybrid GDN + attention, plus its DFlash2 draft · decode 81 vs 95 tok/s, speculative 178 vs 176 · the rest lives in a 1.4k-line generator</small>
        </a>
      </div>
      <p className="benchmark-footnote">
        Repository measurements · Qwen3-4B (last card: Qwen3.8-27B, docs/qwen38-bringup.md) · batch 1 · single GB300. Each comparison uses its stated control.
      </p>
    </section>
  );
}

function Footer() {
  return (
    <footer>
      <div>
        <span className="footer-mark">KERN■</span>
        <h2>THE RUNTIME<br />DOESN'T NEED<br />THE MODEL.</h2>
      </div>
      <div className="footer-links">
        <a href="https://github.com/pegainfer-project/kern" target="_blank" rel="noreferrer">SOURCE ↗</a>
        <a href="/schema/">SCHEMA →</a>
        <a href="#top">BACK TO TOP ↑</a>
      </div>
    </footer>
  );
}

export default function App() {
  // The page is client-rendered: a deep link like /#evidence arrives before
  // the sections exist, so the browser's own anchor jump finds nothing.
  useEffect(() => {
    const id = decodeURIComponent(window.location.hash.slice(1));
    if (!id) return;
    const frame = requestAnimationFrame(() => {
      document.getElementById(id)?.scrollIntoView({ behavior: "instant", block: "start" });
    });
    return () => cancelAnimationFrame(frame);
  }, []);

  return (
    <main>
      <Hero />
      <TheSwap />
      <TheLoop />
      <Evidence />
      <Verifier />
      <Artifact />
      <Proof />
      <Footer />
    </main>
  );
}
