import { useState } from "react";
import "./verify-plan.css";

// Where one program's time goes at six shapes: `kern bench` on a Qwen3.8-27B
// decode step, one GB300, 12 graph replays a shape. A bar is the step's p50;
// it is split by each op's share of kernel time, the report's `mix`.

const OPS = [
  { id: "gemm", name: "gemm", calls: 305 },
  { id: "attn", name: "attn", calls: 16 },
  { id: "gdn", name: "gdn_step", calls: 48 },
  { id: "other", name: "8 other ops", calls: 277 },
] as const;

// Per shape: sequences, context tokens, the step's p50 in ms, and each op's kernel time in µs, in OPS order.
const SHAPES: { seqs: number; context: string; ms: number; us: [number, number, number, number] }[] = [
  { seqs: 1, context: "64", ms: 9.65, us: [9400, 189, 271, 999] },
  { seqs: 1, context: "262,080", ms: 12.291, us: [9404, 2829, 271, 1005] },
  { seqs: 16, context: "64", ms: 11.096, us: [10028, 372, 820, 1222] },
  { seqs: 16, context: "131,008", ms: 30.053, us: [10035, 19322, 818, 1227] },
  { seqs: 128, context: "64", ms: 19.004, us: [11321, 1851, 5606, 1775] },
  { seqs: 128, context: "16,320", ms: 36.942, us: [11312, 19798, 5604, 1777] },
];

const AXIS = 40;
const TICKS = [0, 10, 20, 30, 40];

const sum = (xs: readonly number[]) => xs.reduce((a, b) => a + b, 0);
const time = (us: number) => (us >= 1000 ? `${(us / 1000).toFixed(2)} ms` : `${us} µs`);
const shape = (s: (typeof SHAPES)[number]) => `${s.seqs} ${s.seqs === 1 ? "sequence" : "sequences"} · ${s.context} tokens deep`;

export default function BenchPlan() {
  const [hover, setHover] = useState<[number, number] | null>(null);
  const readout = hover
    ? (() => {
        const [r, o] = hover;
        const s = SHAPES[r];
        return `${shape(s)} · ${OPS[o].name} ×${OPS[o].calls} · ${time(s.us[o])} · ${((100 * s.us[o]) / sum(s.us)).toFixed(1)}% of kernel time`;
      })()
    : "Hover a segment for its calls and time.";

  return (
    <figure className="bp">
      <div className="vp-plan-bar">
        <span>program decode · 646 calls · one step, six shapes</span>
        <span>NVIDIA GB300 · 12 graph replays a shape</span>
      </div>

      <div className="bp-body">
        <div className="bp-legend">
          {OPS.map((op) => (
            <span key={op.id}>
              <i className={`bp-${op.id}`} />
              {op.name} <em>×{op.calls}</em>
            </span>
          ))}
        </div>

        <div className="bp-rows">
          {SHAPES.map((s, r) => {
            const total = sum(s.us);
            return (
              <div key={shape(s)} className="bp-row">
                <div className="bp-label">
                  <strong>
                    {s.seqs} {s.seqs === 1 ? "sequence" : "sequences"}
                  </strong>
                  <span>{s.context} tokens deep</span>
                </div>
                <div className="bp-track">
                  {TICKS.slice(1).map((t) => (
                    <i key={t} className="bp-grid" style={{ left: `${(100 * t) / AXIS}%` }} />
                  ))}
                  <div className="bp-bar" style={{ width: `${(100 * s.ms) / AXIS}%` }}>
                    {OPS.map((op, o) => {
                      const share = s.us[o] / total;
                      return (
                        <div
                          key={op.id}
                          className={`bp-seg bp-${op.id}${hover && hover[0] === r && hover[1] === o ? " is-hover" : ""}`}
                          style={{ flexGrow: s.us[o] }}
                          tabIndex={0}
                          aria-label={`${shape(s)}: ${op.name}, ${time(s.us[o])}, ${(100 * share).toFixed(1)}% of kernel time`}
                          onMouseEnter={() => setHover([r, o])}
                          onMouseLeave={() => setHover(null)}
                          onFocus={() => setHover([r, o])}
                          onBlur={() => setHover(null)}
                        >
                          {op.id !== "other" && <span>{Math.round(100 * share)}%</span>}
                        </div>
                      );
                    })}
                  </div>
                  <span className="bp-tip" style={{ left: `${(100 * s.ms) / AXIS}%` }}>
                    {s.ms.toFixed(2)} ms
                  </span>
                </div>
              </div>
            );
          })}
          <div className="bp-row bp-axis" aria-hidden="true">
            <div className="bp-label" />
            <div className="bp-track">
              {TICKS.map((t) => (
                <span key={t} style={{ left: `${(100 * t) / AXIS}%` }}>
                  {t === 0 ? "0" : `${t} ms`}
                </span>
              ))}
            </div>
          </div>
        </div>

        <p className="bp-readout" aria-live="polite">
          {readout}
        </p>
      </div>

      <figcaption className="bp-caption">
        Matrix multiply costs 9 to 11 ms at every shape. Attention grows with context and <code>gdn_step</code> with the batch, so which
        kernel is worth optimizing depends on the traffic.
      </figcaption>

      <div className="bp-table">
      <table>
        <caption>Step time and kernel time by op, per shape</caption>
        <thead>
          <tr>
            <th scope="col">shape</th>
            <th scope="col">step p50</th>
            {OPS.map((op) => (
              <th key={op.id} scope="col">
                {op.name}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {SHAPES.map((s) => (
            <tr key={shape(s)}>
              <th scope="row">{shape(s)}</th>
              <td>{s.ms.toFixed(3)} ms</td>
              {s.us.map((us, o) => (
                <td key={OPS[o].id}>{time(us)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      </div>
    </figure>
  );
}
