import { useEffect, useState } from "react";
import liveDark from "./assets/live-status-dark.png";
import liveLight from "./assets/live-status-light.png";
import BenchPlan from "./BenchPlan";
import TestPlan from "./TestPlan";
import VerifyPlan from "./VerifyPlan";

type Theme = "light" | "dusk" | "ink";
type CliMode = "serve" | "vllm" | "sglang";
type ManifestTab = "structure" | "states" | "parallel" | "routing";
type QuickTab = "serve" | "vllm" | "sglang";
type SchFocus =
  | "all"
  | "pkg"
  | "rt"
  | "serve"
  | "observe"
  | "analyze"
  | "optimize"
  | "redeploy";

const THEME_ORDER: Theme[] = ["light", "dusk", "ink"];
const THEME_LABELS: Record<Theme, string> = {
  light: "Paper",
  dusk: "Dusk",
  ink: "Ink",
};

const HERO_CLI: Record<CliMode, { label: string; cmd: string }> = {
  serve: {
    label: "kern-serve & CLI",
    cmd: "curl -fsSL https://kern-baa.pages.dev/install.sh | sh",
  },
  vllm: {
    label: "vLLM",
    cmd: "uv pip install kern-llm && vllm serve Qwen/Qwen3-4B -O.model_impl=kern",
  },
  sglang: {
    label: "SGLang",
    cmd: "uv pip install kern-llm && KERN_MANIFEST=manifest.json python -m kern_sglang.launch_server ...",
  },
};

const MANIFEST_TABS: Record<
  ManifestTab,
  {
    tab: string;
    title: string;
    desc: string;
    code: string;
    points: { key: string; body: string }[];
  }
> = {
  structure: {
    tab: "1. Four-Part Manifest",
    title: "Forward is a verified static dataflow graph",
    desc: "Every weight, workspace tensor, state, kernel binary and launch step is declared up front and verified at load, before touching a GPU.",
    code: `{
  "schema_version": 9,
  "model": { "vocab_size": 151936, "eos_token_ids": [151645] },
  "buffers": {
    "lm_head": { "source": "weight", "dtype": "bf16", "shape": [151936, 2560] },
    "h0":      { "source": "scratch", "dtype": "bf16", "shape": ["T", 2560] }
  },
  "kernels": {
    "rmsnorm": { "cubin": "kernels/rmsnorm_sm100.cubin", "sha256": "6ac5f6..." }
  },
  "programs": {
    "decode":  { "axes": { "T": { "max": 512 } }, "steps": [ /* flat launch list */ ] },
    "prefill": { "axes": { "T": { "max": 16384 } }, "steps": [ /* flat launch list */ ] }
  }
}`,
    points: [
      {
        key: "buffers",
        body: "Weights, derived tensors and scratch buffers, sized by symbolic axes.",
      },
      {
        key: "kernels",
        body: "Compiled cubins pinned by SHA-256, with typed signatures.",
      },
      {
        key: "programs",
        body: "Named DAGs lowered once at load into flat launch lists.",
      },
    ],
  },
  states: {
    tab: "2. Hybrid States & Pool",
    title: "Paged KV + recurrent state in one pure-host pool",
    desc: "Hybrid architectures mix attention layers (state grows per token) with recurrent layers like GDN or KDA (fixed per sequence). The manifest declares bytes; kern-pool manages the rest.",
    code: `"states": {
  "layer_0_gdn": {
    "kind": "recurrent",
    "bytes_per_seq": 2097152          // 2 MiB slot per sequence (conv + SSM state)
  },
  "layer_3_kv": {
    "kind": "kv",
    "bytes_per_token": 4096,          // paged KV blocks + zero-copy radix sharing
    "evictable": true                 // eligible for DRAM & io_uring NVMe tiers
  }
}`,
    points: [
      {
        key: "bytes_per_token",
        body: "Paged KV blocks, shared across matching prompts through a radix prefix index.",
      },
      {
        key: "bytes_per_seq",
        body: "Recurrent state, snapshotted at page boundaries so hybrid models hit the prefix cache too.",
      },
      {
        key: "evictable",
        body: "Cold pages spill to host DRAM and local NVMe.",
      },
    ],
  },
  parallel: {
    tab: "3. Parallelism as Data",
    title: "TP, EP, DCP, and pipeline stages without engine branches",
    desc: "Tensor, expert and decode-context parallelism are shard rules on weights and collective ops in the graph. kern cut splits a manifest into pipeline stages.",
    code: `"parallel": { "world_size": 8 },

// Sharded weight slice per rank:
"qkv_w": {
  "source": "weight", "dtype": "fp8_e4m3",
  "shape": [768, 7168],
  "shard": { "axis": 0, "world": 8 }
},

// Explicit collective in the step list:
{ "id": "o_reduce", "op": "all_reduce", "reduce": "sum", "buf": "attn_out" }`,
    points: [
      {
        key: "shard",
        body: "Each rank loads only its weight slice.",
      },
      {
        key: "all_reduce / peer",
        body: "NCCL collectives and peer-mapped buffers are ordinary graph steps.",
      },
      {
        key: "kern cut",
        body: "Per-stage manifests; the basis of prefill/decode disaggregation.",
      },
    ],
  },
  routing: {
    tab: "4. Shape Routing (when)",
    title: "Different batch shapes dispatch specialized kernels",
    desc: "A kernel tuned for T=1 decode often regresses at T=64 or 32k context. Steps carry guards on runtime axes, so one shape is specialized without touching the others.",
    code: `"steps": [
  {
    "id": "mlp_gate_up_small_t",
    "when": "T <= 16",
    "op": "kernel",
    "kernel": "fused_silu_gemv_sm100",
    "writes": ["mlp_inter"]
  },
  {
    "id": "mlp_gate_up_large_t",
    "when": "T > 16",
    "op": "cublas_gemm",
    "writes": ["mlp_inter"]
  }
]`,
    points: [
      {
        key: "when",
        body: "Evaluated once at step launch; no graph recompilation.",
      },
      {
        key: "Safe specialization",
        body: "Tune T <= 16 decode without risking large-chunk prefill.",
      },
      {
        key: "Verified coverage",
        body: "Both branches are verified at load.",
      },
    ],
  },
};

const QUICKSTART_TABS: Record<
  QuickTab,
  {
    label: string;
    title: string;
    desc: string;
    code: string;
    note: string;
  }
> = {
  serve: {
    label: "1 · kern-serve & CLI",
    title: "Native Rust serving & multi-node P/D disaggregation",
    desc: "Install the prebuilt release, fetch kernels from the registry, and launch an OpenAI-compatible server — from one GPU to multi-node prefill/decode clusters.",
    code: `# Install kern + kern-serve binaries (CUDA 12.9+ / 13.0)
curl -fsSL https://kern-baa.pages.dev/install.sh | sh

# Pull compiled cubins & run a local prompt
kern kernels fetch qwen3.8-27b
kern run qwen3.8-27b --prompt "Explain speculative decoding in three sentences."

# Launch OpenAI-compatible serving (/v1/chat/completions, tool calls, prefix cache)
kern server --manifest examples/qwen3.8-27b/manifest.json --port 8000`,
    note: "Prefix caching, hybrid KV + recurrent slots, pipeline prefill and P/D disaggregation included.",
  },
  vllm: {
    label: "2 · Inside vLLM",
    title: "Drop-in C-ABI model backend for vLLM",
    desc: "Keep vLLM's frontend, tokenizer, guided decoding and scheduler; replace the model forward with a kern package via one flag.",
    code: `uv pip install kern-llm

# Resolve a package from kern.toml or Hugging Face Hub
KERN_TARGET=qwen3.8-27b \\
  vllm serve Qwen/Qwen3.5-27B -O.model_impl=kern`,
    note: "Runs inside vLLM's CUDA graph capture, on vLLM's own KV cache tensor and stream.",
  },
  sglang: {
    label: "3 · Inside SGLang",
    title: "RadixAttention + custom manifests in SGLang",
    desc: "Register kern as SGLang's model runner: SGLang's RadixCache and orchestration, with agent-tuned kernel packages.",
    code: `uv pip install kern-llm

KERN_MANIFEST=/path/to/manifest.json \\
  python -m kern_sglang.launch_server \\
  --model-path Qwen/Qwen3-4B \\
  --attention-backend flashinfer`,
    note: "Cut Kimi-K3 TP8/DCP8/EP8 decode step time from 37.2 ms to 29.3 ms.",
  },
};

// Two buckets of a served.toml that `kern workload` wrote from a Qwen3.8-27B trace.
const SERVED_TOML = [
  {
    bucket: "# prefill · seqs<=1,tokens<=1024 · kv<=64k · 3314 steps · 1274.0 s",
    sweep: "[[sweep]]\ngroups = [1]\nchunk = [1024]\ncontext = [46502]\nweight = 3314",
  },
  {
    bucket: "# round · seqs=16,tokens=256 · kv<=512k · 6024 steps · 1187.9 s",
    sweep: "[[sweep]]\ngroups = [16]\nrows = [16]\ncontext = [24818]\nweight = 6024",
  },
];

const SCH_FOOTNOTES: Record<SchFocus, string> = {
  all: "Continual loop · Hover any stage · Observe → Analyze → Optimize → Redeploy",
  pkg: "Model package · manifest.json + kernels + weights in your own repository",
  rt: "kern runtime · Verifies the package in ms and executes flat launch lists",
  serve: "Unified serving · Native Rust clusters on kern-serve, or inside vLLM & SGLang",
  observe: "01 Observe · Zero-sync CUPTI hardware timestamps stream to Parquet",
  analyze: "02 Analyze · kern workload distills live steps into weighted shape × KV buckets",
  optimize: "03 Optimize · Agents fuse & tune kernels against 7s span tests and trace pricing",
  redeploy: "04 Redeploy · Ship the updated package back to production with zero engine forks",
};

function HeroSchematic() {
  const [focus, setFocus] = useState<SchFocus>("all");

  const zoneCls = (z: Exclude<SchFocus, "all">) =>
    `k-sch-zone ${focus === "all" ? "" : focus === z ? "is-lit" : "is-dim"}`;

  const wireCls = (zones: Exclude<SchFocus, "all">[], extra = "") => {
    if (focus === "all") return `k-sch-wire ${extra}`;
    return zones.includes(focus)
      ? `k-sch-wire ${extra} is-lit`
      : `k-sch-wire ${extra} is-dim`;
  };

  return (
    <div className="k-hero-stage" onMouseLeave={() => setFocus("all")}>
      <div className="k-hero-aura" aria-hidden="true" />

      <svg
        className="k-schematic-svg"
        viewBox="0 0 580 416"
        role="img"
        aria-label="kern continual loop: model package verified by kern, served side-by-side on kern-serve, vLLM, and SGLang, and continuously improved through Observe, Analyze, Optimize, and Redeploy"
      >
        {/* Subtle floating spatial orbit ring & precision crosshairs (no outer frame box) */}
        <rect
          x="54"
          y="41"
          width="472"
          height="332"
          rx="44"
          className="k-sch-orbit-bg"
        />
        {[
          [32, 24],
          [548, 24],
          [290, 294],
        ].map(([cx, cy], idx) => (
          <g key={idx} stroke="var(--border-strong)" strokeWidth="1" opacity="0.55">
            <line x1={cx - 4} y1={cy} x2={cx + 4} y2={cy} />
            <line x1={cx} y1={cy - 4} x2={cx} y2={cy + 4} />
          </g>
        ))}

        {/* Invisible full-cycle motion track for glowing orbital pulses */}
        <path
          id="continual-orbit"
          d="M 290 41 L 290 113 L 165 201 L 462 201 L 462 295 L 462 357 Q 462 373 446 373 L 108 373 Q 92 373 92 357 L 92 295 L 54 295 L 54 57 Q 54 41 70 41 Z"
          fill="none"
          stroke="none"
        />

        {/* Wires: Package -> kern Runtime */}
        <path
          d="M 169 60 C 169 76, 246 72, 246 86"
          className={wireCls(["pkg", "rt"], "is-main")}
        />
        <path
          d="M 301 60 L 301 86"
          className={wireCls(["pkg", "rt"], "is-main")}
        />
        <path
          d="M 422 60 C 422 76, 336 72, 336 86"
          className={wireCls(["pkg", "rt"])}
        />

        {/* Wires: kern Runtime -> Unified Serve Dock */}
        <path
          d="M 256 140 C 256 154, 165 150, 165 176"
          className={wireCls(["rt", "serve"], "is-main")}
        />
        <path
          d="M 301 140 L 301 176"
          className={wireCls(["rt", "serve"])}
        />
        <path
          d="M 336 140 C 336 154, 426 150, 426 176"
          className={wireCls(["rt", "serve"])}
        />

        {/* Continual Loop Wires: Serve -> Observe -> Analyze -> Optimize -> Redeploy -> Package */}
        <path
          d="M 462 238 L 462 268"
          className={wireCls(["serve", "observe"], "is-loop")}
        />
        <path
          d="M 472 322 V 357 Q 472 373 456 373"
          className={wireCls(["observe", "analyze"], "is-loop")}
        />
        <path
          d="M 304 373 L 276 373"
          className={wireCls(["analyze", "optimize"], "is-loop")}
        />
        <path
          d="M 124 373 H 108 Q 92 373 92 357 V 322"
          className={wireCls(["optimize", "redeploy"], "is-loop")}
        />
        <path
          d="M 54 268 V 57 Q 54 41 70 41 H 102"
          className={wireCls(["redeploy", "pkg"], "is-loop")}
        />

        {/* Directional chevrons along the loop */}
        <path
          d="M 458 253 L 462 260 L 466 253"
          className={wireCls(["serve", "observe"], "is-main")}
        />
        <path
          d="M 293 369 L 286 373 L 293 377"
          className={wireCls(["analyze", "optimize"], "is-main")}
        />
        <path
          d="M 88 338 L 92 331 L 96 338"
          className={wireCls(["optimize", "redeploy"], "is-main")}
        />
        <path
          d="M 87 37 L 94 41 L 87 45"
          className={wireCls(["redeploy", "pkg"], "is-main")}
        />

        {/* Animated pulses orbiting the full closed flywheel */}
        <circle r="4" fill="#1f4fff" className="k-pulse-dot">
          <animateMotion dur="7.5s" repeatCount="indefinite">
            <mpath href="#continual-orbit" />
          </animateMotion>
        </circle>
        <circle
          r="4.2"
          fill="#b8ff2e"
          stroke="#0b0f17"
          strokeWidth="1"
          className="k-pulse-dot"
        >
          <animateMotion dur="7.5s" begin="-3.75s" repeatCount="indefinite">
            <mpath href="#continual-orbit" />
          </animateMotion>
        </circle>

        {/* TOP: MODEL PACKAGE (3 floating pills, x=102..478, centered at 290) */}
        <g className={zoneCls("pkg")} onMouseEnter={() => setFocus("pkg")}>
          <text x="290" y="13" textAnchor="middle" className="k-sch-cap">
            MODEL PACKAGE
          </text>

          <rect x="105" y="25" width="134" height="38" rx="10" className="k-sch-shadow" />
          <rect x="102" y="22" width="134" height="38" rx="10" className="k-sch-card" />
          <circle cx="118" cy="41" r="4" fill="#1f4fff" />
          <text x="175" y="45" textAnchor="middle" className="k-sch-label">
            manifest.json
          </text>

          <rect x="249" y="25" width="110" height="38" rx="10" className="k-sch-shadow" />
          <rect x="246" y="22" width="110" height="38" rx="10" className="k-sch-card" />
          <circle cx="262" cy="41" r="4" fill="#e05a1b" />
          <text x="307" y="45" textAnchor="middle" className="k-sch-label">
            kernels/
          </text>

          <rect x="369" y="25" width="112" height="38" rx="10" className="k-sch-shadow" />
          <rect x="366" y="22" width="112" height="38" rx="10" className="k-sch-card" />
          <circle cx="382" cy="41" r="4" fill="#08875b" />
          <text x="428" y="45" textAnchor="middle" className="k-sch-label">
            weights
          </text>
        </g>

        {/* CORE: KERN SILICON CHIP (x=194..386, y=86..140) */}
        <g className={zoneCls("rt")} onMouseEnter={() => setFocus("rt")}>
          {[98, 113, 128].map((py) => (
            <g key={py}>
              <rect x="187" y={py - 2.5} width="7" height="5" rx="1.5" fill="#64748b" />
              <rect x="386" y={py - 2.5} width="7" height="5" rx="1.5" fill="#64748b" />
            </g>
          ))}
          <rect x="198" y="90" width="192" height="54" rx="13" className="k-sch-shadow" />
          <rect x="194" y="86" width="192" height="54" rx="13" className="k-sch-chip" />

          <rect
            x="212"
            y="107"
            width="10"
            height="12"
            fill="#b8ff2e"
            stroke="#0b0f17"
            strokeWidth="1"
          />
          <text
            x="228"
            y="118"
            fill="#f8fafc"
            fontFamily="Space Grotesk, Inter, sans-serif"
            fontSize="17"
            fontWeight="700"
          >
            kern
          </text>

          <rect
            x="280"
            y="101"
            width="92"
            height="24"
            rx="6"
            fill="#151d2a"
            stroke="#334155"
          />
          <text
            x="326"
            y="117"
            textAnchor="middle"
            fill="#cbd5e1"
            fontFamily="IBM Plex Mono, monospace"
            fontSize="9.8"
            fontWeight="600"
          >
            0 model code
          </text>
        </g>

        {/* UNIFIED SIDE-BY-SIDE SERVING BOX: [ kern-serve | vLLM | SGLang ] */}
        <g className={zoneCls("serve")} onMouseEnter={() => setFocus("serve")}>
          <text x="98" y="157" className="k-sch-cap">
            SERVE
          </text>
          <rect
            x="86"
            y="164"
            width="408"
            height="74"
            rx="15"
            className="k-sch-dock"
          />

          {/* 1. kern-serve (primary, left) */}
          <rect x="101" y="179" width="134" height="50" rx="10" className="k-sch-shadow" />
          <rect x="98" y="176" width="134" height="50" rx="10" className="k-sch-serve" />
          <text x="165" y="198" textAnchor="middle" className="k-sch-serve-txt">
            kern-serve
          </text>
          {[0, 1, 2, 3].map((i) => (
            <rect
              key={i}
              x={128 + i * 20}
              y="208"
              width="14"
              height="6.5"
              rx="2"
              className={i < 2 ? "k-sch-rank is-hot" : "k-sch-rank"}
            />
          ))}

          {/* 2. vLLM (side-by-side middle) */}
          <rect x="247" y="179" width="114" height="50" rx="10" className="k-sch-shadow" />
          <rect x="244" y="176" width="114" height="50" rx="10" className="k-sch-card" />
          <text x="301" y="205" textAnchor="middle" className="k-sch-label">
            vLLM
          </text>

          {/* 3. SGLang (side-by-side right) */}
          <rect x="373" y="179" width="112" height="50" rx="10" className="k-sch-shadow" />
          <rect x="370" y="176" width="112" height="50" rx="10" className="k-sch-card" />
          <text x="426" y="205" textAnchor="middle" className="k-sch-label">
            SGLang
          </text>
        </g>

        {/* CONTINUAL LOOP STAGE 01: OBSERVE */}
        <g
          className={zoneCls("observe")}
          onMouseEnter={() => setFocus("observe")}
        >
          <rect x="399" y="271" width="152" height="54" rx="12" className="k-sch-shadow" />
          <rect x="396" y="268" width="152" height="54" rx="12" className="k-sch-card" />
          <circle cx="414" cy="287" r="3.8" fill="#08875b" />
          <text x="426" y="291" className="k-sch-label">
            Observe
          </text>
          <text x="534" y="290" textAnchor="end" className="k-sch-step-num">
            01
          </text>
          <text x="426" y="309" className="k-sch-sub">
            CUPTI · Parquet
          </text>
        </g>

        {/* CONTINUAL LOOP STAGE 02: ANALYZE */}
        <g
          className={zoneCls("analyze")}
          onMouseEnter={() => setFocus("analyze")}
        >
          <rect x="307" y="349" width="152" height="54" rx="12" className="k-sch-shadow" />
          <rect x="304" y="346" width="152" height="54" rx="12" className="k-sch-card" />
          <circle cx="322" cy="365" r="3.8" fill="#e05a1b" />
          <text x="334" y="369" className="k-sch-label">
            Analyze
          </text>
          <text x="442" y="368" textAnchor="end" className="k-sch-step-num">
            02
          </text>
          <text x="334" y="387" className="k-sch-sub">
            workload shapes
          </text>
        </g>

        {/* CONTINUAL LOOP STAGE 03: OPTIMIZE */}
        <g
          className={zoneCls("optimize")}
          onMouseEnter={() => setFocus("optimize")}
        >
          <rect x="127" y="349" width="152" height="54" rx="12" className="k-sch-shadow" />
          <rect x="124" y="346" width="152" height="54" rx="12" className="k-sch-card" />
          <circle cx="142" cy="365" r="3.8" fill="#1f4fff" />
          <text x="154" y="369" className="k-sch-label">
            Optimize
          </text>
          <text x="262" y="368" textAnchor="end" className="k-sch-step-num">
            03
          </text>
          <text x="154" y="387" className="k-sch-sub">
            7s test · bench
          </text>
        </g>

        {/* CONTINUAL LOOP STAGE 04: REDEPLOY */}
        <g
          className={zoneCls("redeploy")}
          onMouseEnter={() => setFocus("redeploy")}
        >
          <rect x="35" y="271" width="152" height="54" rx="12" className="k-sch-shadow" />
          <rect x="32" y="268" width="152" height="54" rx="12" className="k-sch-card" />
          <circle cx="50" cy="287" r="3.8" fill="#84cc16" />
          <text x="62" y="291" className="k-sch-label">
            Redeploy
          </text>
          <text x="170" y="290" textAnchor="end" className="k-sch-step-num">
            04
          </text>
          <text x="62" y="309" className="k-sch-sub">
            hot-swap package
          </text>
        </g>
      </svg>

      <div className="k-stage-whisper">{SCH_FOOTNOTES[focus]}</div>
    </div>
  );
}

export default function App() {
  const [theme, setTheme] = useState<Theme>(() => {
    if (typeof window === "undefined") return "light";
    const saved = window.localStorage.getItem("kern-home-theme") as Theme | null;
    return saved && THEME_ORDER.includes(saved) ? saved : "light";
  });
  const [cliMode, setCliMode] = useState<CliMode>("serve");
  const [copied, setCopied] = useState(false);
  const [manifestTab, setManifestTab] = useState<ManifestTab>("structure");
  const [quickTab, setQuickTab] = useState<QuickTab>("serve");

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    window.localStorage.setItem("kern-home-theme", theme);
  }, [theme]);

  const cycleTheme = () => {
    const next = THEME_ORDER[(THEME_ORDER.indexOf(theme) + 1) % THEME_ORDER.length];
    setTheme(next);
  };

  const copyCommand = (text: string) => {
    navigator.clipboard?.writeText(text);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  };

  const activeManifest = MANIFEST_TABS[manifestTab];
  const activeQuick = QUICKSTART_TABS[quickTab];

  return (
    <div>
      {/* STICKY TOPBAR */}
      <header className="k-topbar">
        <div className="container k-topbar-inner">
          <a href="#" className="k-brand" aria-label="kern home" translate="no">
            <span className="k-brand-square" aria-hidden="true" />
            <span className="k-brand-name">kern</span>
          </a>

          <div className="k-topbar-actions">
            <nav className="k-nav" aria-label="Primary navigation">
              <a href="/docs/">Docs</a>
              <a href="/studio/" className="k-nav-studio">
                <span className="k-dot-live" aria-hidden="true" />
                Studio
              </a>
            </nav>
            <button
              type="button"
              className="k-btn-ghost"
              onClick={cycleTheme}
              title="Switch color theme"
            >
              <span className="k-theme-swatch" aria-hidden="true" />
              {THEME_LABELS[theme]}
            </button>
            <a
              href="https://github.com/pegainfer-project/kern"
              target="_blank"
              rel="noreferrer"
              className="k-btn-dark"
            >
              GitHub ↗
            </a>
          </div>
        </div>
      </header>

      {/* HERO */}
      <section className="k-hero">
        <div className="container k-hero-grid">
          <div>
            <div className="k-kicker-pill">
              <i aria-hidden="true" />
              MODEL-AGNOSTIC GPU RUNTIME · BUILT IN RUST
            </div>
            <h1>
              An inference engine per model, per GPU, <span>per workload.</span>
            </h1>
            <p className="k-hero-lead">
              <strong>kern</strong> takes the model out of the inference engine.
              Each model ships as a package — manifest, kernels, weights — that
              agents verify and tune in seconds, on <strong>kern-serve</strong>{" "}
              or inside <strong>vLLM</strong> and <strong>SGLang</strong>.
            </p>

            <div className="k-hero-ctas">
              <a href="#quickstart" className="k-btn-primary">
                Deploy with kern-serve →
              </a>
              <a href="/studio/" className="k-btn-secondary">
                Open Manifest Studio ↗
              </a>
            </div>

            <div className="k-cli-wrap">
              <div className="k-cli-top">
                <span className="k-cli-label">QUICK INSTALL</span>
                <div className="k-cli-switch" role="tablist" aria-label="Install target">
                  {(Object.keys(HERO_CLI) as CliMode[]).map((m) => (
                    <button
                      key={m}
                      type="button"
                      role="tab"
                      aria-selected={cliMode === m}
                      className={`k-cli-pill ${cliMode === m ? "active" : ""}`}
                      onClick={() => setCliMode(m)}
                    >
                      {HERO_CLI[m].label}
                    </button>
                  ))}
                </div>
              </div>
              <div className="k-cli-bar">
                <div className="k-cli-cmd">
                  <span className="k-cli-prompt">$</span>
                  <span>{HERO_CLI[cliMode].cmd}</span>
                </div>
                <button
                  type="button"
                  className="k-copy-btn"
                  onClick={() => copyCommand(HERO_CLI[cliMode].cmd)}
                >
                  {copied ? "Copied" : "Copy"}
                </button>
              </div>
            </div>
          </div>

          <HeroSchematic />
        </div>
      </section>

      {/* PRODUCTION PROOF CARDS (3-UP) */}
      <section className="k-outcomes-band">
        <div className="container">
          <div className="k-story-grid">
            <article className="k-story-card is-featured">
              <span className="k-story-tag">KERN-SERVE · 24× GB300 P/D</span>
              <h3>Full 896-expert Kimi-K3 disaggregated at SLA</h3>
              <p>
                Two 8-stage pipeline-parallel prefill pods feed an EP8 decode
                tier on <code>kern-serve</code>.
              </p>
              <div className="k-story-metrics">
                <div className="k-story-metric">
                  <strong>0.85 req/s</strong>
                  <span>+81% over 1P1D baseline</span>
                </div>
                <div className="k-story-metric">
                  <strong>23.4 ms</strong>
                  <span>ITL p50 (0.99 s TTFT p50)</span>
                </div>
              </div>
            </article>

            <article className="k-story-card">
              <span className="k-story-tag">KERN-SERVE · DAY-0 BRING-UP</span>
              <h3>Frontier models serving hours after weight release</h3>
              <p>
                Agents assemble a standalone model repo without waiting on
                upstream framework releases.
              </p>
              <div className="k-story-metrics">
                <div className="k-story-metric">
                  <strong>7 hrs</strong>
                  <span>DeepSeek-V4.1-Flash (GB300/B300)</span>
                </div>
                <div className="k-story-metric">
                  <strong>2.5 hrs</strong>
                  <span>Qwen3.8-27B + draft (<a href="/qwen38/">log ↗</a>)</span>
                </div>
              </div>
            </article>

            <article className="k-story-card">
              <span className="k-story-tag">CONTINUAL AGENT LOOP</span>
              <h3>One overnight loop lifts every host engine</h3>
              <p>
                Three agents tuned Qwen3.8-27B against 154,849 traced steps on
                GB300 (<a href="/rsi/">report ↗</a>).
              </p>
              <div className="k-story-metrics">
                <div className="k-story-metric">
                  <strong>+17%</strong>
                  <span>vLLM requests (−27% ITL p50)</span>
                </div>
                <div className="k-story-metric">
                  <strong>−21%</strong>
                  <span>Kimi-K3 step time in SGLang</span>
                </div>
              </div>
            </article>
          </div>
        </div>
      </section>

      {/* 01 · WHY KERN */}
      <section id="why" className="k-band">
        <div className="container">
          <div className="k-section-head">
            <div className="k-eyebrow">01 · Why Kern</div>
            <h2>
              Open-source engines try to be one-size-fits-all — and drown in
              complexity.
            </h2>
            <p>
              Covering{" "}
              <code>
                models × precisions × GPUs × parallelism × decoding × workloads
              </code>{" "}
              in one repository turns popular engines into 1.4M–2M lines of
              Python. Tuning one model there means upstream PRs that stall,
              forks that break on every rebase, and coding agents lost in
              nested <code>if/else</code> dispatch.
            </p>
          </div>

          <div className="k-compare-grid">
            <div className="k-compare-col">
              <span className="k-compare-tag tag-old">
                ONE-SIZE-FITS-ALL ENGINE
              </span>
              <h3>300 models coupled in 2M lines of Python</h3>
              <ul className="k-compare-list">
                <li>
                  <span className="k-li-mark k-li-bad">×</span>
                  <span>
                    <b>Fork &amp; PR gridlock:</b> 4 in 5 upstream PRs go
                    unreviewed; private forks conflict on every rebase.
                  </span>
                </li>
                <li>
                  <span className="k-li-mark k-li-bad">×</span>
                  <span>
                    <b>Opaque to agents:</b> kernel dispatch hides behind flags
                    and runtime Python branches.
                  </span>
                </li>
                <li>
                  <span className="k-li-mark k-li-bad">×</span>
                  <span>
                    <b>Hours to validate:</b> precision drift shows up only
                    after slow end-to-end suites.
                  </span>
                </li>
              </ul>
              <pre className="k-mini-code">{`# Which kernel actually ran on this GPU?
if arch >= SM100 and head_dim in (64, 128):
    if kv_dtype == "fp8" and not use_mla:
        backend = select_attn_v3(...)
elif tp_size > 1 and env.ENABLE_CUSTOM_ALLREDUCE:
    ... # 2,000,000 lines across 300 models`}</pre>
            </div>

            <div className="k-compare-col is-kern">
              <span className="k-compare-tag tag-new">
                KERN MODEL PACKAGE
              </span>
              <h3>Engine shared once. Every model in its own repo.</h3>
              <ul className="k-compare-list">
                <li>
                  <span className="k-li-mark k-li-good">✓</span>
                  <span>
                    <b>Zero rebase tax:</b> your model package is its own Git
                    repo of JSON + cubins.
                  </span>
                </li>
                <li>
                  <span className="k-li-mark k-li-good">✓</span>
                  <span>
                    <b>Explicit static graph:</b> every buffer, state and
                    launch is readable data.
                  </span>
                </li>
                <li>
                  <span className="k-li-mark k-li-good">✓</span>
                  <span>
                    <b>Deploy anywhere:</b> the same package runs on{" "}
                    <code>kern-serve</code>, vLLM, or SGLang.
                  </span>
                </li>
              </ul>
              <pre className="k-mini-code">{`# Verified in ms · Oracle-tested in 7s · Served natively
$ kern verify manifest.json
$ kern test --reference manifest.json --manifest candidate.json
$ kern server --manifest candidate.json --port 8000`}</pre>
            </div>
          </div>
        </div>
      </section>

      {/* 02 · THE MANIFEST */}
      <section id="architecture" className="k-band k-band-blueprint">
        <div className="container">
          <div className="k-section-head">
            <div className="k-eyebrow">02 · The Manifest</div>
            <h2>The runtime contains zero model code — and never will.</h2>
            <p>
              Architecture tricks, hybrid states, multi-GPU collectives and
              shape predicates live in a declarative{" "}
              <a
                href="/schema/"
                style={{ color: "var(--brand)", fontWeight: 600 }}
              >
                JSON Schema ↗
              </a>
              . <code>kern-runtime</code> verifies it at load, then executes
              flat launch lists blindly.
            </p>
          </div>

          <div className="k-manifest-explorer">
            <div className="k-me-code-pane">
              <div className="k-me-tabs" role="tablist" aria-label="Manifest concepts">
                {(Object.keys(MANIFEST_TABS) as ManifestTab[]).map((k) => (
                  <button
                    key={k}
                    type="button"
                    role="tab"
                    aria-selected={manifestTab === k}
                    className={`k-me-tab ${manifestTab === k ? "active" : ""}`}
                    onClick={() => setManifestTab(k)}
                  >
                    {MANIFEST_TABS[k].tab}
                  </button>
                ))}
              </div>
              <pre className="k-me-pre">
                <code>{activeManifest.code}</code>
              </pre>
            </div>

            <div className="k-me-info-pane">
              <span className="k-story-tag">{activeManifest.tab}</span>
              <h3>{activeManifest.title}</h3>
              <p>{activeManifest.desc}</p>
              <div className="k-me-points">
                {activeManifest.points.map((pt) => (
                  <div key={pt.key} className="k-me-point">
                    <code>{pt.key}</code> — {pt.body}
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="k-studio-banner">
            <div className="k-studio-banner-copy">
              <strong>
                <span className="k-dot-live" aria-hidden="true" />
                Inspect real production graphs in Manifest Studio
              </strong>
              <span>
                Browse buffers, state budgets, folded layer loops and launch
                signatures in your browser.
              </span>
            </div>
            <a href="/studio/" className="k-btn-primary">
              Launch Studio (/studio/) ↗
            </a>
          </div>
        </div>
      </section>

      {/* 03 · CONTINUAL LOOP */}
      <section id="harness" className="k-band">
        <div className="container">
          <div className="k-section-head">
            <div className="k-eyebrow">03 · Continual Loop</div>
            <h2>Observe, analyze, optimize, redeploy — closed in minutes.</h2>
            <p>Four steps. Each is one command, and each hands the next a file.</p>
          </div>

          <div className="k-loop">
            <div className="k-loop-stage">
              <div className="k-loop-rail">
                <div className="k-loop-n">01</div>
                <div className="k-loop-verb">Observe</div>
                <span className="k-loop-cmd">kern-serve --trace</span>
              </div>
              <div className="k-loop-body">
                <p className="k-loop-lead">
                  <strong>Always on, on live traffic.</strong> The GPU timestamps
                  every kernel of every served step in hardware, with nothing
                  synchronized between launches, and each rank streams the
                  records to Parquet as it serves. Prompts, token IDs and
                  completions are never written.{" "}
                  <a href="/docs/reference/traces.html">Trace format ↗</a>
                </p>
                <div className="k-loop-facts">
                  <div className="k-loop-fact">
                    <strong>+0.06%</strong>
                    <span>TTFT with every kernel recorded, against tracing off</span>
                  </div>
                  <div className="k-loop-fact">
                    <strong>0.28 core</strong>
                    <span>of CPU for the trace thread; output tok/s unchanged</span>
                  </div>
                  <div className="k-loop-fact">
                    <strong>100 µs</strong>
                    <span>between samples of the GPU's own counters</span>
                  </div>
                </div>
                <pre className="k-loop-code">
                  {"traces/date=2026-11-11/rank=0/\n"}
                  {"  101500.steps.parquet     "}
                  <span className="c"># program, shape, vars, kv, start_ns, end_ns{"\n"}</span>
                  {"  101500.kernels.parquet   "}
                  <span className="c"># every kernel execution of every step{"\n"}</span>
                  {"  101500.counters.parquet  "}
                  <span className="c"># the GPU's own counters, every 100 µs</span>
                </pre>
                <a className="k-live" href="https://api.kern-llm.org/status/" target="_blank" rel="noreferrer">
                  <div className="vp-plan-bar">
                    <span>Qwen3.8-27B-FP8 on kern-serve · its trace, published as it serves</span>
                    <span>snapshot 2026-11-18 07:46 UTC</span>
                  </div>
                  {[
                    ["light", liveLight],
                    ["dark", liveDark],
                  ].map(([theme, src]) => (
                    <img
                      key={theme}
                      className={`k-live-${theme}`}
                      src={src}
                      width={1656}
                      height={534}
                      loading="lazy"
                      alt="329 ms of a serving GPU: prefill and decode steps split into stages and kernels, above its memory bandwidth and tensor-pipe utilization sampled every 100 µs"
                    />
                  ))}
                  <span className="k-live-open">Open the live status ↗</span>
                </a>
              </div>
            </div>

            <div className="k-loop-stage">
              <div className="k-loop-rail">
                <div className="k-loop-n">02</div>
                <div className="k-loop-verb">Analyze</div>
                <span className="k-loop-cmd">kern workload</span>
              </div>
              <div className="k-loop-body">
                <p className="k-loop-lead">
                  <code>kern workload</code> folds live traffic into weighted{" "}
                  <code>shape × KV</code> buckets on the CPU.
                </p>
                <div className="k-loop-pair">
                  <div>
                    <pre className="k-loop-code">
                      {"kern workload --manifest manifest.json \\\n"}
                      {"  --trace traces/date=2026-11-11/rank=0 \\\n"}
                      {"  --out served.toml\n\n"}
                      {"kern bench demo --workload served.toml --out bench.json"}
                    </pre>
                    <div className="k-loop-facts">
                      <div className="k-loop-fact">
                        <strong>261,672 → 157</strong>
                        <span>online steps folded into buckets</span>
                      </div>
                      <div className="k-loop-fact">
                        <strong>6 min</strong>
                        <span>of replay for 7.3 hours of traffic</span>
                      </div>
                      <div className="k-loop-fact">
                        <strong>0.90%</strong>
                        <span>between replayed and traced GPU cost</span>
                      </div>
                    </div>
                  </div>
                  <pre className="k-loop-code">
                    <span className="k-loop-file">served.toml · one sweep a bucket{"\n\n"}</span>
                    {SERVED_TOML.map((b) => (
                      <span key={b.bucket}>
                        <span className="c">{b.bucket}</span>
                        {`\n${b.sweep}\n\n`}
                      </span>
                    ))}
                    <span className="c"># …</span>
                  </pre>
                </div>
              </div>
            </div>

            <div className="k-loop-stage">
              <div className="k-loop-rail">
                <div className="k-loop-n">03</div>
                <div className="k-loop-verb">Optimize</div>
                <span className="k-loop-cmd">verify · test · bench</span>
              </div>
              <div className="k-loop-body">
                <p className="k-loop-lead">
                  A coding agent changes a kernel or a launch. Three gates
                  answer, cheapest first.
                </p>

                <div className="k-gate">
                  <div className="k-gate-head">
                    <span className="k-gate-step">GATE 01 · STATIC</span>
                    <h3>kern verify</h3>
                    <span className="k-gate-speed">~5 ms · No GPU</span>
                    <p>
                      Four kernels, seven checks. Pick a check to see what it
                      reads and what <code>kern verify</code> prints when it
                      refuses.
                    </p>
                  </div>
                  <VerifyPlan />
                </div>

                <div className="k-gate">
                  <div className="k-gate-head">
                    <span className="k-gate-step">GATE 02 · NUMERICS</span>
                    <h3>kern test</h3>
                    <span className="k-gate-speed">~10 s · 1 GPU</span>
                    <p>
                      B is the reference A with one argument of call #16
                      changed. Every type still fits, so{" "}
                      <code>kern verify</code> passes it;{" "}
                      <code>kern test</code> finds the call and fails it on the
                      logits.
                    </p>
                  </div>
                  <TestPlan />
                  <p className="k-gate-foot">
                    An overnight run of this loop, change by change:{" "}
                    <a href="/rsi/">Overnight RSI Report ↗</a>
                  </p>
                </div>

                <div className="k-gate">
                  <div className="k-gate-head">
                    <span className="k-gate-step">GATE 03 · PRICING</span>
                    <h3>kern bench</h3>
                    <span className="k-gate-speed">Minutes · Trace-weighted</span>
                    <p>
                      What a faster kernel is worth depends on the shape it is
                      priced at. The same decode step, measured at six shapes:
                    </p>
                  </div>
                  <BenchPlan />
                </div>
              </div>
            </div>

            <div className="k-loop-stage">
              <div className="k-loop-rail">
                <div className="k-loop-n">04</div>
                <div className="k-loop-verb">Redeploy</div>
                <span className="k-loop-cmd">kern server</span>
              </div>
              <div className="k-loop-body">
                <p className="k-loop-lead">
                  The manifest that clears the gates is the next one served,
                  and its traffic is the next trace.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 04 · MODELS & QUICK START */}
      <section id="models" className="k-band k-band-blueprint">
        <div className="container">
          <div className="k-section-head">
            <div className="k-eyebrow">04 · Ecosystem &amp; Quick Start</div>
            <h2>Production model packages, ready to serve or fork.</h2>
            <p>
              Each model family lives in its own repository. Run it on{" "}
              <code>kern-serve</code>, or load it inside vLLM and SGLang.
            </p>
          </div>

          <div className="k-models-wrap">
            <table className="k-models-table">
              <thead>
                <tr>
                  <th>Model Family</th>
                  <th>Architecture &amp; Parallelism</th>
                  <th>Target GPUs</th>
                  <th>Serving Paths</th>
                  <th>Package</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>
                    <strong style={{ color: "var(--text-1)" }}>
                      Kimi-K3 (Full 896E)
                    </strong>
                  </td>
                  <td>MoE + MLA · PP8 prefill + TP8/DCP8/EP8 decode</td>
                  <td>GB300 NVL72, B300</td>
                  <td>
                    <div className="k-host-pills">
                      <span className="is-primary-host">kern-serve P/D</span>
                      <span>SGLang</span>
                    </div>
                  </td>
                  <td>
                    <a
                      href="https://github.com/pegainfer-project/kern"
                      target="_blank"
                      rel="noreferrer"
                    >
                      kern-k3 ↗
                    </a>
                  </td>
                </tr>
                <tr>
                  <td>
                    <strong style={{ color: "var(--text-1)" }}>
                      DeepSeek-V4.1-Flash
                    </strong>
                  </td>
                  <td>MLA + MoE · Tool calling &amp; prefix cache</td>
                  <td>GB300, HGX B300</td>
                  <td>
                    <div className="k-host-pills">
                      <span className="is-primary-host">kern-serve</span>
                      <span>vLLM</span>
                    </div>
                  </td>
                  <td>
                    <a
                      href="https://github.com/pegainfer-project/kern"
                      target="_blank"
                      rel="noreferrer"
                    >
                      kern-dsv41-flash ↗
                    </a>
                  </td>
                </tr>
                <tr>
                  <td>
                    <strong style={{ color: "var(--text-1)" }}>
                      Qwen3.8-27B + MTP Draft
                    </strong>
                  </td>
                  <td>Hybrid GDN + Paged KV · Speculative decode</td>
                  <td>GB300, B200, H100</td>
                  <td>
                    <div className="k-host-pills">
                      <span className="is-primary-host">kern-serve</span>
                      <span>vLLM</span>
                      <span>SGLang</span>
                    </div>
                  </td>
                  <td>
                    <a href="/qwen38/">bring-up log ↗</a>
                  </td>
                </tr>
                <tr>
                  <td>
                    <strong style={{ color: "var(--text-1)" }}>
                      Dense Qwen3 (0.6B – 32B)
                    </strong>
                  </td>
                  <td>GQA Transformer · BF16 / FP8</td>
                  <td>SM80 / SM90 / SM100 / SM120</td>
                  <td>
                    <div className="k-host-pills">
                      <span className="is-primary-host">kern-serve</span>
                      <span>vLLM</span>
                      <span>SGLang</span>
                    </div>
                  </td>
                  <td>
                    <a
                      href="https://github.com/pegainfer-project/kern-qwen3"
                      target="_blank"
                      rel="noreferrer"
                    >
                      kern-qwen3 ↗
                    </a>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          <div id="quickstart" className="k-qs-card">
            <div className="k-qs-header">
              <div className="k-qs-tabs" role="tablist" aria-label="Deployment mode">
                {(Object.keys(QUICKSTART_TABS) as QuickTab[]).map((t) => (
                  <button
                    key={t}
                    type="button"
                    role="tab"
                    aria-selected={quickTab === t}
                    className={`k-qs-tab ${quickTab === t ? "active" : ""}`}
                    onClick={() => setQuickTab(t)}
                  >
                    {QUICKSTART_TABS[t].label}
                  </button>
                ))}
              </div>
              <button
                type="button"
                className="k-btn-ghost"
                onClick={() => copyCommand(activeQuick.code)}
              >
                {copied ? "Copied" : "Copy commands"}
              </button>
            </div>

            <div className="k-qs-body">
              <pre className="k-qs-code">
                <code>{activeQuick.code}</code>
              </pre>
              <div className="k-qs-side">
                <h4>{activeQuick.title}</h4>
                <p>{activeQuick.desc}</p>
                <p style={{ fontSize: "12.5px", color: "var(--text-3)" }}>
                  {activeQuick.note}
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="k-footer">
        <div className="container k-footer-inner">
          <div className="k-footer-left">
            <a href="#" className="k-brand">
              <span className="k-brand-square" aria-hidden="true" />
              <span className="k-brand-name">kern</span>
            </a>
            <span className="k-footer-tagline">
              Model-agnostic GPU runtime in Rust · Built by agents, built for
              agents.
            </span>
          </div>

          <div className="k-footer-links">
            <a href="/docs/">Docs</a>
            <a href="/studio/">Manifest Studio</a>
            <a href="/schema/">Schema</a>
            <a href="/rsi/">RSI Report</a>
            <a href="/qwen38/">Qwen3.8 Bring-up</a>
            <a
              href="https://github.com/pegainfer-project/kern"
              target="_blank"
              rel="noreferrer"
            >
              GitHub ↗
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
