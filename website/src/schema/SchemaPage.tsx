import type { ReactNode } from "react";
import schemaRaw from "../../../schema/manifest-v4.schema.json?raw";
import minimalRaw from "../../../examples/minimal.json?raw";

const schema: any = JSON.parse(schemaRaw);
const defs: Record<string, any> = schema.$defs ?? {};

/* ---------------------------------------------------------------- helpers */

const FORMAT_NAMES: Record<string, string> = {
  uint64: "u64",
  uint32: "u32",
  int32: "i32",
  uint8: "u8",
  float: "f32",
  double: "f64",
};

function refName(ref: string): string {
  return ref.split("/").pop() ?? ref;
}

/** Render a JSON-Schema fragment as a compact type signature. */
function Sig({ s, depth = 0 }: { s: any; depth?: number }): ReactNode {
  if (!s || typeof s !== "object") return "?";
  if (s.$ref) {
    const name = refName(s.$ref);
    return (
      <a className="tref" href={`#t-${name}`}>
        {name}
      </a>
    );
  }
  if (s.anyOf) {
    return (s.anyOf as any[]).map((v, i) => (
      <span key={i}>
        {i > 0 && <span className="sig-sep"> | </span>}
        <Sig s={v} depth={depth + 1} />
      </span>
    ));
  }
  if (s.enum) return (s.enum as string[]).map((e) => JSON.stringify(e)).join(" | ");
  if (s.oneOf) return (s.oneOf as any[]).map((v) => JSON.stringify(v.const)).join(" | ");
  switch (s.type) {
    case "integer":
      return FORMAT_NAMES[s.format] ?? "int";
    case "number":
      return FORMAT_NAMES[s.format] ?? "number";
    case "string":
      return s.pattern ? "string (patterned)" : "string";
    case "boolean":
      return "bool";
    case "array": {
      if (s.prefixItems) {
        return (
          <>
            [
            {(s.prefixItems as any[]).map((v, i) => (
              <span key={i}>
                {i > 0 && ", "}
                <Sig s={v} depth={depth + 1} />
              </span>
            ))}
            ]
          </>
        );
      }
      const n = s.minItems != null && s.minItems === s.maxItems ? s.minItems : null;
      return (
        <>
          [<Sig s={s.items} depth={depth + 1} />
          {n != null ? `; ${n}` : ""}]
        </>
      );
    }
    case "object": {
      if (s.additionalProperties && typeof s.additionalProperties === "object") {
        return (
          <>
            map&lt;name, <Sig s={s.additionalProperties} depth={depth + 1} />&gt;
          </>
        );
      }
      if (s.properties) {
        const keys = Object.keys(s.properties);
        return (
          <>
            {"{ "}
            {keys.map((k, i) => (
              <span key={k}>
                {i > 0 && ", "}
                {k}: <Sig s={s.properties[k]} depth={depth + 1} />
              </span>
            ))}
            {" }"}
          </>
        );
      }
      return "object";
    }
  }
  return "?";
}

function fmtDefault(v: any): string {
  if (v && typeof v === "object" && Object.keys(v).length === 0) return "{}";
  return JSON.stringify(v);
}

/* ------------------------------------------------------------- reference */

const GROUPS: { label: string; names: string[] }[] = [
  { label: "root", names: ["Manifest", "Module", "Topology"] },
  {
    label: "declarations",
    names: ["Var", "State", "Buffer", "Fill", "Dim", "DType", "BufferKind", "Domain", "Bound"],
  },
  {
    label: "ops",
    names: [
      "Op",
      "ParamType",
      "Impl",
      "Scratch",
      "Launch",
      "KernelLaunch",
      "ExternLaunch",
      "LaunchArg",
      "Pack",
      "Field",
      "TensorMap",
      "TmaDType",
    ],
  },
  { label: "programs", names: ["Program", "Batch", "Call", "Arg", "Expr"] },
];

// Future-proof: anything the schema gains that the curated groups don't
// know yet still renders, at the end.
const grouped = new Set(GROUPS.flatMap((g) => g.names).concat("Manifest"));
const leftovers = Object.keys(defs)
  .filter((n) => !grouped.has(n))
  .sort();
const allGroups = leftovers.length
  ? GROUPS.concat({ label: "other", names: leftovers })
  : GROUPS;

function defFor(name: string): any {
  if (name === "Manifest") {
    const { $defs: _d, $schema: _s, $id: _i, title: _t, ...root } = schema;
    return { description: "The root object — the file itself.", ...root };
  }
  return defs[name];
}

function kindOf(d: any): string {
  if (d.enum || d.oneOf) return "closed enum";
  if (d.pattern) return "patterned string";
  if (d.anyOf) return "union (untagged)";
  if (d.type === "object") return "object";
  return d.type ?? "";
}

function PropertyRows({ d }: { d: any }) {
  const req: string[] = d.required ?? [];
  const props = d.properties ?? {};
  return (
    <div className="prop-table" role="table">
      {Object.keys(props).map((k) => {
        const p = props[k];
        const required = req.includes(k);
        return (
          <div className="prop-row" role="row" key={k}>
            <span className="prop-name">{k}</span>
            <span className="prop-sig">
              <Sig s={p} />
            </span>
            <span className={required ? "prop-req" : "prop-opt"}>
              {required
                ? "required"
                : p.default !== undefined
                  ? `default ${fmtDefault(p.default)}`
                  : "optional"}
            </span>
            {p.description && <span className="prop-desc">{p.description}</span>}
          </div>
        );
      })}
    </div>
  );
}

function DefSection({ name }: { name: string }) {
  const d = defFor(name);
  if (!d) return null;
  return (
    <section className="def" id={`t-${name}`}>
      <header className="def-head">
        <h3>{name}</h3>
        <span className="def-kind">{kindOf(d)}</span>
      </header>
      {d.description && <p className="def-desc">{d.description}</p>}

      {d.enum && (
        <div className="enum-chips">
          {(d.enum as string[]).map((e) => (
            <code key={e}>{e}</code>
          ))}
        </div>
      )}

      {d.oneOf && (
        <div className="variant-list">
          {(d.oneOf as any[]).map((v) => (
            <div className="variant-row" key={v.const}>
              <span className="variant-sig">
                <code>{v.const}</code>
              </span>
              {v.description && <span className="prop-desc">{v.description}</span>}
            </div>
          ))}
        </div>
      )}

      {d.pattern && (
        <div className="pattern-block">
          <span className="pattern-label">pattern</span>
          <code>{d.pattern}</code>
        </div>
      )}

      {d.anyOf && (
        <div className="variant-list">
          {(d.anyOf as any[]).map((v, i) => (
            <div className="variant-row" key={i}>
              <span className="variant-sig">
                <Sig s={v} />
              </span>
              {v.description && <span className="prop-desc">{v.description}</span>}
            </div>
          ))}
        </div>
      )}

      {d.properties && <PropertyRows d={d} />}

      {d.type === "object" &&
        !d.properties &&
        d.additionalProperties &&
        typeof d.additionalProperties === "object" && (
          <p className="def-desc">
            <code className="inline-sig">
              map&lt;name, <Sig s={d.additionalProperties} />&gt;
            </code>
          </p>
        )}

      {d.additionalProperties === false && (
        <p className="def-strict">unknown fields rejected</p>
      )}
    </section>
  );
}

/* --------------------------------------------------------------- diagram */

function sketch(x: number, y: number, w: number, h: number): string {
  // Deterministic hand-drawn jitter.
  const j = (n: number) => {
    const v = Math.sin(n * 12.9898 + x + y) * 43758.5453;
    return (v - Math.floor(v)) * 2.6 - 1.3;
  };
  const px = (n: number, s: number) => (n + j(s)).toFixed(1);
  return [
    `M ${px(x, 1)} ${px(y, 2)}`,
    `C ${px(x + w * 0.35, 3)} ${px(y, 4)}, ${px(x + w * 0.7, 5)} ${px(y, 6)}, ${px(x + w, 7)} ${px(y, 8)}`,
    `C ${px(x + w, 9)} ${px(y + h * 0.4, 10)}, ${px(x + w, 11)} ${px(y + h * 0.7, 12)}, ${px(x + w, 13)} ${px(y + h, 14)}`,
    `C ${px(x + w * 0.65, 15)} ${px(y + h, 16)}, ${px(x + w * 0.3, 17)} ${px(y + h, 18)}, ${px(x, 19)} ${px(y + h, 20)}`,
    `C ${px(x, 21)} ${px(y + h * 0.6, 22)}, ${px(x, 23)} ${px(y + h * 0.3, 24)}, ${px(x, 1)} ${px(y, 2)}`,
  ].join(" ");
}

function head(x: number, y: number, angle: number): string {
  const a = (angle * Math.PI) / 180;
  const p = (r: number, da: number) =>
    `${(x - r * Math.cos(a + da)).toFixed(1)} ${(y - r * Math.sin(a + da)).toFixed(1)}`;
  return `M ${p(11, -0.42)} L ${x} ${y} L ${p(11, 0.42)}`;
}

function WireDiagram() {
  return (
    <figure className="wire-figure">
      <div className="wire-scroll">
        <svg
          className="wire-diagram"
          viewBox="0 0 1060 610"
          role="img"
          aria-label="Structure of a kern manifest: one file containing vars, states, buffers, modules, ops and programs; a program is a call list plus the batch shape a serving loop drives it at; calls bind to op interfaces; implementations launch entries of modules pinned by sha256, local or from a registry."
        >
          {/* manifest file */}
          <path className="box" d={sketch(42, 92, 230, 330)} />
          <text className="t-title" x={46} y={76}>
            manifest.json
          </text>
          <text className="t-row" x={66} y={130}>
            vars
          </text>
          <text className="t-row" x={66} y={166}>
            states
          </text>
          <text className="t-row" x={66} y={202}>
            buffers
          </text>
          <text className="t-row" x={66} y={238}>
            modules
          </text>
          <text className="t-row t-strong" x={66} y={292}>
            ops
          </text>
          <text className="t-row t-strong" x={66} y={336}>
            programs
          </text>
          <text className="t-note" x={46} y={452}>
            one file · the whole contract
          </text>

          {/* kernel */}
          <path className="box" d={sketch(402, 72, 280, 230)} />
          <text className="t-title" x={420} y={102}>
            op
          </text>
          <text className="t-row t-blue" x={420} y={136}>
            interface — typed params
          </text>
          <path className="box box-inner" d={sketch(420, 154, 244, 128)} />
          <text className="t-label" x={434} y={178}>
            impl
          </text>
          <text className="t-row-sm" x={434} y={206}>
            scratch — private
          </text>
          <text className="t-row-sm" x={434} y={232}>
            launches[] · entry · block · grid
          </text>
          <text className="t-row-sm" x={434} y={258}>
            module @ sha256
          </text>

          {/* program */}
          <path className="box" d={sketch(772, 122, 252, 180)} />
          <text className="t-title" x={790} y={152}>
            program
          </text>
          <text className="t-row" x={790} y={186}>
            calls[] — in order
          </text>
          <text className="t-row-sm t-blue" x={790} y={216}>
            attn(q, kv, out, tokens)
          </text>
          <text className="t-row-sm t-blue" x={790} y={242}>
            gemm(x, w, y)
          </text>
          <text className="t-note" x={790} y={282}>
            batch · groups × rows · no control flow
          </text>

          {/* impl sources */}
          <path className="box" d={sketch(432, 470, 158, 54)} />
          <text className="t-row-sm" x={448} y={502}>
            kernels/*.cubin
          </text>
          <path className="box box-registry" d={sketch(628, 470, 250, 54)} />
          <text className="t-row-sm" x={644} y={502}>
            hf:org/repo/path
          </text>
          <text className="t-note" x={432} y={556}>
            modules · sha256 = identity, transport untrusted
          </text>

          {/* containment: ops -> op, programs -> program */}
          <path className="wire" d="M 274 292 C 320 280, 350 230, 398 196" />
          <path className="wire" d={head(400, 194, -34)} />
          <path className="wire" d="M 274 336 C 440 400, 620 396, 766 240" />
          <path className="wire" d={head(769, 236, -44)} />

          {/* call binds to interface */}
          <path className="wire wire-blue" d="M 786 210 C 740 190, 724 160, 690 138" />
          <path className="wire wire-blue" d={head(686, 136, -145)} />
          <text className="t-note t-blue" x={696} y={110}>
            binds to the interface
          </text>

          {/* args reference buffers/vars (over the top) */}
          <path
            className="wire wire-blue wire-dash"
            d="M 800 118 C 640 30, 420 26, 278 226"
          />
          <path className="wire wire-blue" d={head(276, 230, 125)} />
          <text className="t-note t-blue" x={330} y={40}>
            args reference buffers &amp; vars
          </text>

          {/* launch module -> sources */}
          <path className="wire" d="M 520 286 C 512 340, 508 400, 508 464" />
          <path className="wire" d={head(508, 468, 92)} />
          <path className="wire" d="M 560 286 C 640 350, 700 400, 736 464" />
          <path className="wire" d={head(738, 468, 62)} />
        </svg>
      </div>
      <figcaption className="wire-caption">
        call sites bind to the <span className="c-blue">interface</span> — swapping an
        implementation, even one fetched from a{" "}
        <span className="c-green">registry</span>, touches only its{" "}
        <span className="mono">impl</span> block.
      </figcaption>
    </figure>
  );
}

/* --------------------------------------------------------------- example */

const STEPS: { key: string; type: string; text: string }[] = [
  { key: "vars", type: "Var", text: "the caller passes tokens on every call, at most 1024; it is the only per-call number in the file." },
  { key: "states", type: "State", text: "kv is 256 bytes per token slot; the runtime provisions it and never looks inside." },
  { key: "buffers", type: "Buffer", text: "x is written by the caller, w is bound from the weights file, y is read back; shapes use tokens." },
  { key: "modules", type: "Module", text: "the compiled code, identified by sha256; toy.cubin is a label." },
  { key: "ops", type: "Op", text: "scale is an interface — five typed, directional params — plus an impl: one launch of scale_rows from toy, tokens blocks of 64 threads." },
  { key: "programs", type: "Program", text: "step calls scale once, binding x, w, y, kv and tokens to its params in order. It declares no batch, so no serving loop drives it; a harness stages x by name." },
];

function MinimalExample() {
  return (
    <section className="example" aria-label="A complete minimal manifest">
      <p className="kicker">A COMPLETE MANIFEST · TOY</p>
      <h2>Six sections. One op. One call.</h2>
      <div className="example-grid">
        <pre className="example-json">
          <code>{minimalRaw.trim()}</code>
        </pre>
        <ol className="example-steps">
          {STEPS.map((s) => (
            <li key={s.key}>
              <a className="example-key" href={`#t-${s.type}`}>
                {s.key}
              </a>
              <span>{s.text}</span>
            </li>
          ))}
        </ol>
      </div>
      <p className="example-note">
        The runtime verifies every reference and type at load, then runs the call
        list as written. It does not know what <code>scale</code> computes.
      </p>
    </section>
  );
}

/* -------------------------------------------------------------- protocol */

const FILLS: { const: string; description: string }[] = defs.Fill?.oneOf ?? [];
const inputs = FILLS.filter((f) => f.description.startsWith("Input:"));
const outputs = FILLS.filter((f) => f.description.startsWith("Output:"));

function FillList({ items }: { items: typeof FILLS }) {
  return (
    <ul className="fill-list">
      {items.map((f) => (
        <li key={f.const}>
          <code>{f.const}</code>
          <span>{f.description.replace(/^(Input|Output): /, "")}</span>
        </li>
      ))}
    </ul>
  );
}

function Protocol() {
  const batch = defs.Batch?.properties ?? {};
  return (
    <section className="protocol" aria-label="The serving protocol">
      <p className="kicker">SERVING PROTOCOL · THREE WORDS</p>
      <h2>The loop reads fill, batch and once. Nothing else.</h2>
      <p className="protocol-lead">
        How a serving loop drives a manifest — which buffer takes the token, which
        program takes how many sequences of how many rows, where the produced token
        is read back — is declared in the file, not known by the caller. Every axis
        is derived from these declarations: the runtime never reads a buffer or
        program by name.
      </p>
      <div className="protocol-grid">
        <div className="protocol-card">
          <h3>
            <a className="tref" href="#t-Fill">fill</a>
            <span>on a buffer</span>
          </h3>
          <p>{defs.Fill?.description}</p>
          <p className="fill-head">inputs the loop writes</p>
          <FillList items={inputs} />
          <p className="fill-head">outputs the loop reads</p>
          <FillList items={outputs} />
        </div>
        <div className="protocol-card">
          <h3>
            <a className="tref" href="#t-Batch">batch</a>
            <span>on a program</span>
          </h3>
          <p>{defs.Batch?.description}</p>
          <ul className="fill-list">
            {Object.keys(batch).map((k) => (
              <li key={k}>
                <code>{k}</code>
                <span>{batch[k].description}</span>
              </li>
            ))}
          </ul>
          <p className="protocol-note">
            One code path, four shapes: prefill is 1 group of <code>tokens</code>{" "}
            rows, decode is b groups of 1 row, a speculative round is b groups of r
            rows cut by <code>count</code>, a span is a decode step where one group
            carries a run of <code>span</code> rows.
          </p>
        </div>
        <div className="protocol-card">
          <h3>
            <a className="tref" href="#t-Program">once</a>
            <span>on a program</span>
          </h3>
          <p>{defs.Program?.properties?.once?.description}</p>
          <p className="protocol-note">
            A program with neither <code>batch</code> nor <code>once</code> is not
            driven by a loop at all: a single layer under test, a barrier, a
            harness's material.
          </p>
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ page */

const RAW_URL = "/schema/manifest-v4.schema.json";
const TYPES_URL =
  "https://github.com/pegainfer-project/kern/blob/master/crates/kern-manifest/src/types.rs";

export default function SchemaPage() {
  const typeCount = Object.keys(defs).length + 1; // + root
  const exprForms = (defs.Expr?.anyOf ?? []).length;
  const fillRoles = (defs.Fill?.oneOf ?? []).length;
  return (
    <div className="schema-page" id="top">
      <header className="site-header">
        <a className="wordmark" href="/" aria-label="Kern home">
          KERN<span className="wordmark-dot">■</span>
        </a>
        <nav aria-label="Primary navigation">
          <a href={RAW_URL}>RAW JSON ↓</a>
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

      <main>
        <section className="schema-hero">
          <p className="kicker">WIRE FORMAT · JSON SCHEMA 2020-12</p>
          <h1>
            manifest <span className="v-chip">v4</span>
          </h1>
          <p className="hero-line">
            One JSON file is the entire contract between a model provider and the
            runtime. The parser is the law — this page is rendered from{" "}
            <a className="tref" href={RAW_URL}>
              its schema
            </a>
            , which CI keeps generated from{" "}
            <a className="tref" href={TYPES_URL} target="_blank" rel="noreferrer">
              the Rust types
            </a>
            .
          </p>

          <div className="fact-strip">
            <div className="fact">
              <span className="fact-num">{typeCount}</span>
              <span className="fact-cap">types — the whole format</span>
            </div>
            <div className="fact">
              <span className="fact-num">0</span>
              <span className="fact-cap">unknown fields tolerated, anywhere</span>
            </div>
            <div className="fact">
              <span className="fact-num">{exprForms}</span>
              <span className="fact-cap">expression forms — the entire scalar language</span>
            </div>
            <div className="fact">
              <span className="fact-num">{fillRoles}</span>
              <span className="fact-cap">fill roles — the entire serving protocol</span>
            </div>
          </div>

        </section>

        <MinimalExample />

        <Protocol />

        <section className="schema-hero schema-diagram">
          <WireDiagram />
        </section>

        <section className="reference">
          <aside className="type-index" aria-label="Type index">
            {allGroups.map((g) => (
              <div className="index-group" key={g.label}>
                <span className="index-label">{g.label}</span>
                {g.names.map((n) => (
                  <a key={n} href={`#t-${n}`}>
                    {n}
                  </a>
                ))}
              </div>
            ))}
          </aside>

          <div className="defs">
            {allGroups.map((g) => (
              <div key={g.label}>
                <h2 className="group-title">{g.label}</h2>
                {g.names.map((n) => (
                  <DefSection key={n} name={n} />
                ))}
              </div>
            ))}
          </div>
        </section>

        <footer className="schema-footer">
          <pre className="curl-block">
            <code>curl -s https://kern-baa.pages.dev{RAW_URL}</code>
          </pre>
          <p>
            ground truth: <code>crates/kern-manifest</code> · schema golden-checked in
            CI · validate any manifest against it before shipping
          </p>
        </footer>
      </main>
    </div>
  );
}
