import { useEffect, useState } from "react";
import { END, LIVE, PHASES, POINTS, RUNS, STAIR } from "./data";

const REPO = "https://github.com/pegainfer-project/kern";
const DOC = `${REPO}/blob/master/docs/vllm-rsi-example.md`;

// ------------------------------------------------------------ progress
const T0 = PHASES[0].s;
const W = 1000;
const H = 360;
const PAD = { l: 52, r: 16, t: 34, b: 34 };
const Y0 = 1.5;
const Y1 = -6.5;
const x = (t: number) => PAD.l + ((t - T0) / (END - T0)) * (W - PAD.l - PAD.r);
const y = (v: number) => PAD.t + ((Y0 - Math.min(Y0, v)) / (Y0 - Y1)) * (H - PAD.t - PAD.b);
const clock = (t: number) => new Date(t * 1000).toISOString().slice(11, 16);

function Progress() {
  const [hover, setHover] = useState<number | null>(null);
  const steps = LIVE.map((p, i) => {
    const next = i + 1 < LIVE.length ? LIVE[i + 1].t : END;
    return `M${x(p.t)} ${y(p.y)} L${x(next)} ${y(p.y)}` + (i + 1 < LIVE.length ? ` L${x(next)} ${y(LIVE[i + 1].y)}` : "");
  });
  const h = hover === null ? null : POINTS[hover];
  return (
    <svg className="r-progress" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="GPU cost relative to the start over one night: every agent measurement as a dot, the served manifest as a step line falling to minus 5.3 percent">
      {PHASES.map((p) => (
        <g key={`${p.round}${p.kind}`} className={`r-phase r-phase-${p.kind}`}>
          <rect x={x(p.s)} y={PAD.t} width={x(p.e) - x(p.s)} height={H - PAD.t - PAD.b} />
          <text x={x(p.s) + 4} y={PAD.t - 8} className="q-mono">{p.kind === "serve" ? `R${p.round} SERVE` : `R${p.round} AGENTS`}</text>
        </g>
      ))}
      {[0, -2, -4, -6].map((v) => (
        <g key={v}>
          <line x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} className={v === 0 ? "r-zero" : "r-grid"} />
          <text x={PAD.l - 8} y={y(v) + 4} textAnchor="end" className="q-mono q-dim">{v === 0 ? "0%" : `${v}%`}</text>
        </g>
      ))}
      {Array.from({ length: 6 }, (_, i) => Math.ceil(T0 / 3600) * 3600 + i * 7200).filter((t) => t < END).map((t) => (
        <text key={t} x={x(t)} y={H - 10} textAnchor="middle" className="q-mono q-dim">{clock(t)}Z</text>
      ))}
      {POINTS.map((p, i) => (
        <circle
          key={i}
          cx={x(p.t)}
          cy={y(p.y)}
          r={p.kind === "full" ? 4.5 : 3.2}
          className={`r-pt r-pt-${p.kind}`}
          onMouseEnter={() => setHover(i)}
          onMouseLeave={() => setHover(null)}
        />
      ))}
      {steps.map((d, i) => <path key={i} d={d} className="r-live" />)}
      {LIVE.slice(1).map((p) => (
        <g key={p.t} className="r-deploy">
          <circle cx={x(p.t)} cy={y(p.y)} r={6} />
        </g>
      ))}
      <text x={W - PAD.r} y={y(LIVE[LIVE.length - 1].y) - 10} textAnchor="end" className="q-mono r-live-label">SERVED · −5.3%</text>
      {h && (
        <text x={Math.min(x(h.t) + 10, W - 180)} y={y(h.y) - 10} className="q-mono">
          {`R${h.round} ${h.run} · ${h.kind === "full" ? "full workload" : "sub-bench, projected"} · ${h.y.toFixed(2)}%`}
        </text>
      )}
    </svg>
  );
}

// ------------------------------------------------------------ the loop
const STEPS = [
  { k: "SERVE", v: "vLLM + kern takes 1 h of AgentX traffic; the plugin logs every step's shape" },
  { k: "PRICE", v: "the shapes become a weighted kern bench workload; its cost is GPU seconds" },
  { k: "SPLIT", v: "an orchestrator ranks ops by what leaving them out saves, and hands 3 agents one each" },
  { k: "OPTIMIZE", v: "2 h per agent, own GPU, own sub-bench, a 35 s correctness gate" },
  { k: "GATE", v: "logits vs main, every admitted batch size, −1% on the same GPU" },
  { k: "DEPLOY", v: "merge, regenerate the served manifest, restart; the next hour measures it" },
];

function Loop() {
  return (
    <ol className="r-loop">
      {STEPS.map((s, i) => (
        <li key={s.k}>
          <span className="q-mono r-loop-n">{String(i + 1).padStart(2, "0")}</span>
          <strong>{s.k}</strong>
          <span>{s.v}</span>
        </li>
      ))}
    </ol>
  );
}

// ------------------------------------------------------------ servers
const SERVERS = [
  { l: "vLLM", req: 1310, tok: 291.6, itl: 30.0, cls: "ref" },
  { l: "vLLM + kern · start", req: 1446, tok: 325.4, itl: 23.6, cls: "mid" },
  { l: "vLLM + kern · after the loop", req: 1528, tok: 336.5, itl: 21.9, cls: "kern" },
];

function Servers() {
  const max = 1600;
  return (
    <div className="r-servers">
      {SERVERS.map((s) => (
        <div key={s.l} className={`r-server r-server-${s.cls}`}>
          <span className="q-mono">{s.l.toUpperCase()}</span>
          <i style={{ width: `${(s.req / max) * 100}%` }} />
          <strong>{s.req.toLocaleString()}</strong>
          <small className="q-mono">{`${s.tok} tok/s · ITL p50 ${s.itl} ms`}</small>
        </div>
      ))}
      <p className="q-foot q-mono">requests served in one hour · Qwen3.8-27B · one GB300 each, side by side · AgentX replay, concurrency 24, same seed · same scheduler limits · 0 errors</p>
    </div>
  );
}

// ------------------------------------------------------------ staircase
function Stair() {
  const s0 = STAIR[0].cost;
  return (
    <ul className="q-list r-stair">
      {STAIR.map((s, i) => (
        <li key={s.label}>
          <code>{i === 0 ? "—" : `${((s.cost / STAIR[i - 1].cost - 1) * 100).toFixed(2)}%`}</code>
          <span>
            {s.label} <em className="r-dim">{i === 0 ? `· ${s.cost.toFixed(0)} GPU s for the traced hour` : `· ${s.cost.toFixed(0)} GPU s · ${((s.cost / s0 - 1) * 100).toFixed(2)}% in all`}</em>
          </span>
        </li>
      ))}
    </ul>
  );
}

const LESSONS = [
  ["guard", "a gemm 2.7% faster on the trace was 38% slower at 128 sequences; the gate now times every batch size the server admits"],
  ["when", "a manifest launch can run only in a row range, so that kernel shipped for ≤ 16 rows and cuBLASLt kept the rest"],
  ["--ablate", "time each program with one op left out: small ops' shares were inflated about twofold"],
  ["merge", "wins on neighbouring ops overlap (−0.95% and −0.66% gave −1.11%): trial-merge mid-round, draw runs on kernel boundaries"],
  ["time", "agents stopped 8–16 min early believing a cycle takes longer than it does; now the brief states it (< 5 min)"],
];

export default function RsiPage() {
  useEffect(() => {
    const id = decodeURIComponent(window.location.hash.slice(1));
    if (!id) return;
    const frame = requestAnimationFrame(() => document.getElementById(id)?.scrollIntoView({ behavior: "instant", block: "start" }));
    return () => cancelAnimationFrame(frame);
  }, []);
  const runs = Object.values(RUNS);
  const kernelCommits = runs.reduce((a, r) => a + r.kernel, 0);
  return (
    <main className="q-page">
      <header className="site-header">
        <a className="wordmark" href="/" aria-label="Kern home">
          KERN<span className="wordmark-dot">■</span>
        </a>
        <nav aria-label="Primary navigation">
          <a href="#night">NIGHT</a>
          <a href="#loop">LOOP</a>
          <a href="#servers">SERVERS</a>
          <a href="#learned">LEARNED</a>
          <a href={DOC} target="_blank" rel="noreferrer">WRITE-UP ↗</a>
          <a className="github-link" href={REPO} target="_blank" rel="noreferrer">GITHUB ↗</a>
        </nav>
      </header>

      <section className="q-hero r-hero" id="top">
        <p className="eyebrow">OPTIMIZATION LOOP · 2026-09-25 → 26 · VLLM + KERN · ONE NIGHT</p>
        <div className="q-hero-grid">
          <div className="q-hero-main">
            <strong className="q-giant">+17%</strong>
            <span className="q-giant-label">requests per GPU-hour over stock vLLM, on real agent traffic</span>
          </div>
          <div className="q-hero-side">
            <div>
              <strong>−5.3%</strong>
              <span>GPU time for the same traffic,<br />found by agents in three rounds</span>
            </div>
            <div>
              <strong>{runs.length}</strong>
              <span>agent runs of two hours<br /><em>{kernelCommits} kernel commits, 3 deploys</em></span>
            </div>
          </div>
        </div>
      </section>

      <section className="q-section" id="night">
        <div className="section-number">01 / THE NIGHT</div>
        <h2>EVERY DOT IS AN<br />AGENT'S MEASUREMENT.</h2>
        <Progress />
        <p className="q-foot q-mono">cost of the traced traffic relative to the start · dots: agent scores projected onto the full workload (filled: full-workload runs) · blue line: the manifest being served · green: deploys</p>
      </section>

      <section className="q-section" id="loop">
        <div className="section-number">02 / THE LOOP</div>
        <h2>THE TRAFFIC WRITES<br />THE BENCHMARK.</h2>
        <Loop />
        <p className="q-aside">
          The benchmark is the served hour itself: step shapes counted by vLLM, each priced by kern's
          bench on the GPU. On a 5-minute check it accounted for 294 of the 308 GPU-busy seconds, and
          each bench step matched its later serving hour (ITL p50 −4.8% after a −3.1% bench win).
        </p>
      </section>

      <section className="q-section" id="servers">
        <div className="section-number">03 / THREE SERVERS, ONE HOUR</div>
        <h2>SAME SCHEDULER.<br />DIFFERENT FORWARD PASS.</h2>
        <Servers />
        <Stair />
      </section>

      <section className="q-section" id="learned">
        <div className="section-number">04 / WHAT THE LOOP LEARNED</div>
        <h2>MOST FIXES WERE<br />TO THE LOOP ITSELF.</h2>
        <ul className="q-list">
          {LESSONS.map(([k, v]) => (
            <li key={k}><code>{k}</code><span>{v}</span></li>
          ))}
        </ul>
        <p className="q-aside">
          By round 3 bf16 decode sat at 85–98% of its memory-bandwidth floor and three agents found
          about 0.1% each. The next lever changes what a step reads: fewer bytes, or more tokens per read.
        </p>
      </section>

      <section className="q-section q-caveats" id="read">
        <div className="section-number">05 / READ IT YOURSELF</div>
        <ul className="q-list">
          <li><code>caveat</code><span>one GPU per server, one model, one traffic source (AgentX replay)</span></li>
          <li><code>caveat</code><span>stock vLLM keeps the better tails: ITL p99 62 vs 75 ms, TTFT p99 9.4 vs 10.5 s</span></li>
          <li><code>caveat</code><span>the orchestrator is itself an agent; a human chose when to stop and ruled out quantization</span></li>
        </ul>
        <div className="q-links">
          <a href={DOC} target="_blank" rel="noreferrer">THE WRITE-UP ↗</a>
          <a href={`${REPO}/blob/master/docs/vllm.md`} target="_blank" rel="noreferrer">VLLM + KERN ↗</a>
          <a href="/qwen38/">QWEN3.8 BRING-UP</a>
          <a href="/">← KERN</a>
        </div>
      </section>
    </main>
  );
}
