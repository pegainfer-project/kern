import { useEffect, useMemo, useState } from "react";
import type { Anchor, Call, Case, Catalog, Dataset, Op, Record as Run, Series } from "./types";

const DATA = "/perf/data/";
// Standalone exports embed exactly the same evidence files as the hosted view.
type Files = {[file:string]:unknown};
const packed=document.getElementById("perf-evidence-gzip")?.textContent;
let embedded:Files|null=null;
async function unpack<T>(bytes:Uint8Array<ArrayBuffer>):Promise<T> {
  if(bytes[0]===0x1f&&bytes[1]===0x8b)return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).json();
  return JSON.parse(new TextDecoder().decode(bytes));
}
const ready=packed?unpack<Files>(Uint8Array.from(atob(packed),c=>c.charCodeAt(0))).then(files=>{embedded=files;}):Promise.resolve();
async function evidence<T>(file:string):Promise<T> {
  await ready;
  if(embedded && file in embedded)return embedded[file] as T;
  const response=await fetch(DATA+file);
  if(!response.ok)throw Error("Could not load exported measurements.");
  return unpack<T>(new Uint8Array(await response.arrayBuffer()));
}
function EvidenceLink({file,children}:{file:string;children:React.ReactNode}) {
  const url=useMemo(()=>embedded&&file in embedded?URL.createObjectURL(new Blob([JSON.stringify(embedded[file])],{type:"application/json"})):DATA+file,[file]);
  useEffect(()=>()=>{if(url.startsWith("blob:"))URL.revokeObjectURL(url);},[url]);
  return <a href={url} download={embedded?file.replace(/\.gz$/,""):file}>{children}</a>;
}
const colors=["#b7f576","#80bfff","#d3a5ff","#f5c878","#79daca","#ee9eb4","#becbe0"];
const number=(n:number,d=2)=>n.toLocaleString("en-US",{maximumFractionDigits:d,minimumFractionDigits:d});
const us=(n:number)=>n>=1000?`${number(n/1000)} ms`:`${number(n)} µs`;
const pct=(n:number)=>`${number(n*100,1)}%`;
const color=(name:string)=>colors[[...name].reduce((s,c)=>s+c.charCodeAt(0),0)%colors.length];
const context=(r:Run)=>r.scenario.context.reduce((a,b)=>a+b,0)/r.scenario.context.length;
const title=(r:Run)=>`${r.program} · B${r.scenario.batch} · Q${r.scenario.query} · KV ${r.scenario.context.length>1?"mixed":context(r).toLocaleString()}`;

function Spark({series}:{series:Series}) {
  const v=series.samples_us, lo=Math.min(...v),hi=Math.max(...v),span=Math.max(hi-lo,0.001);
  return <svg viewBox="0 0 150 30" className="spark" role="img" aria-label={`Ordered timing samples, ${us(lo)} to ${us(hi)}`}><path d={v.map((x,i)=>`${i?"L":"M"}${i/(v.length-1)*148+1},${28-(x-lo)/span*26}`).join(" ")} fill="none" stroke="currentColor" strokeWidth="1.5"/></svg>;
}

function Distribution({sample}:{sample:Case}) {
  const values=[...sample.cold.samples_us,...sample.warm.samples_us];
  const max=Math.max(...values)*1.12,min=Math.max(0,Math.min(...values)*0.85),span=max-min;
  const y=(v:number)=>170-(v-min)/span*140;
  return <svg className="distribution" viewBox="0 0 650 210" role="img" aria-label="All cold and warm latency samples in acquisition order. Slow samples are retained.">
    {[0,.5,1].map(f=><g key={f}><line x1="65" x2="625" y1={y(min+span*f)} y2={y(min+span*f)} className="chart-grid"/><text x="55" y={y(min+span*f)+4} textAnchor="end">{number(min+span*f,1)} µs</text></g>)}
    {(["cold","warm"] as const).map(mode=><g key={mode} style={{color:mode==="cold"?"#80bfff":"#b7f576"}}>
      <path d={sample[mode].samples_us.map((v,i)=>`${i?"L":"M"}${70+i/(sample[mode].samples_us.length-1)*550},${y(v)}`).join(" ")} stroke="currentColor" strokeWidth="1" opacity=".45" fill="none"/>
      {sample[mode].samples_us.map((v,i)=><circle key={i} cx={70+i/(sample[mode].samples_us.length-1)*550} cy={y(v)} r="3" fill="currentColor"><title>{mode} sample {i+1}: {us(v)}</title></circle>)}
      <line x1="65" x2="625" y1={y(sample[mode].stats.p50)} y2={y(sample[mode].stats.p50)} stroke="currentColor" strokeDasharray="4 5" opacity=".6"/>
    </g>)}
    <text x="70" y="200">01</text><text x="345" y="200" textAnchor="middle">SAMPLE ORDER · COLD / WARM ALTERNATED</text><text x="620" y="200" textAnchor="end">{sample.cold.stats.n}</text>
  </svg>;
}

function Heatmap({runs,selected,onSelect}:{runs:Run[];selected:Run;onSelect:(r:Run)=>void}) {
  const steps=runs.filter(r=>r.scenario.kind==="step" && r.scenario.context.length===1);
  const batches=[...new Set(steps.map(r=>r.scenario.batch))].sort((a,b)=>a-b);
  const contexts=[...new Set(steps.map(context))].sort((a,b)=>a-b);
  const times=steps.map(r=>r.graph.stats.p50),lo=Math.min(...times),hi=Math.max(...times);
  return <div className="heatmap" style={{"--batch-count":batches.length} as React.CSSProperties}>
    <span className="axis">KV / B</span>{batches.map(b=><span className="axis" key={b}>{b}</span>)}
    {contexts.map(l=><div className="heat-row" key={l}><span className="axis">{l.toLocaleString()}</span>{batches.map(b=>{
      const r=steps.find(x=>x.scenario.batch===b&&context(x)===l);
      if(!r)return <span className="heat-missing" key={b} title="Not measured">—</span>;
      const f=(r.graph.stats.p50-lo)/Math.max(hi-lo,1);
      return <button key={b} className={`heat-cell ${r===selected?"selected":""}`} style={{background:`hsl(${145-f*65} 46% ${20+f*19}%)`}} onClick={()=>onSelect(r)} aria-label={`Batch ${b}, context ${l}, ${us(r.graph.stats.p50)}`} aria-pressed={r===selected}>
        {number(r.graph.stats.p50/1000,2)}<span>{r.scenario.holdout?"HOLDOUT":"ms"}</span>
      </button>;
    })}</div>)}
  </div>;
}

function Program({run,onCall,selected}:{run:Run;onCall:(c:Call)=>void;selected:number}) {
  const layers=useMemo(()=>{
    const result=new Map<string,Call[]>();
    for(const c of run.calls){const name=c.label?.match(/^l(?:ayer)?[._]?(\d+)[._]/)?.[1];const group=name!=null?`L${name.padStart(2,"0")}`:"I/O";result.set(group,[...(result.get(group)??[]),c]);}
    return [...result.entries()].sort(([a],[b])=>a==="I/O"?-1:b==="I/O"?1:a.localeCompare(b));
  },[run]);
  const max=Math.max(...layers.map(([,calls])=>calls.reduce((s,c)=>s+c.in_program.stats.p50,0)));
  return <div className="program-scroll">{layers.map(([label,calls])=><div className="layer" key={label}><span>{label}</span><div className="layer-track">{calls.map(c=><button key={c.index} onClick={()=>onCall(c)} aria-label={`${c.label??c.op}, call ${c.index}`} title={`${c.label??c.op} · ${us(c.in_program.stats.p50)}`} className={c.index===selected?"active":""} style={{width:`${c.in_program.stats.p50/max*100}%`,background:color(c.op)}}/> )}</div><span>{us(calls.reduce((s,c)=>s+c.in_program.stats.p50,0))}</span></div>)}</div>;
}

function Hardware({data}:{data:Dataset}) {
  const anchor=(name:string)=>data.calibration_before.find(a=>a.name===name)!;
  const bandwidth=(a:Anchor)=>a.traffic_bytes/a.timing.stats.p50/1e3;
  const gemm=data.calibration_before.find(a=>a.name==="bf16_gemm_4096");
  return <section className="panel hardware"><div className="panel-head"><div><span className="eyebrow">HARDWARE ANCHORS</span><h2>Measured on this GPU</h2></div><span className="tag">{data.hardware.sm_count} SMs</span></div>
    <div className="anchor-grid">{[["SM read",anchor("sm_read")],["SM copy · read + write",anchor("sm_copy")],["D2D · read + write",anchor("d2d_copy")],["L2 resident read",anchor("l2_read")]].map(([label,a])=><div key={String(label)}><span>{String(label)}</span><strong>{number(bandwidth(a as Anchor)/1000)} <small>TB/s</small></strong><em>{number(data.calibration_drift_pct[(a as Anchor).name],1)}% duration drift</em></div>)}</div>
    <div className="hardware-footer"><span>L2 <b>{data.hardware.l2_bytes/2**20} MiB</b></span><span>Eviction buffer <b>{data.hardware.eviction_bytes/2**20} MiB</b></span><span>Evicted / resident probe <b>{number(data.eviction_ratio)}×</b></span><span>Empty kernel <b>{us(anchor("empty_kernel").timing.stats.p50)}</b></span>{gemm&&<span>BF16 GEMM · 4096³ <b>{number((gemm.flops??0)/gemm.timing.stats.p50/1e9)} PFLOP/s</b></span>}</div>
    <p className="note">Local-device reference throughput, not a universal kernel ceiling. Copy counts N bytes read + N written. Eviction is empirically checked; profiler counters are separate diagnostics.</p>
  </section>;
}

function Inspector({sample,op,run,caseIndex,callIndex,onCase,speedup,onSpeedup}:{sample:Case;op:Op;run:Run;caseIndex:number;callIndex:number;onCase:(n:number)=>void;speedup:number;onSpeedup:(n:number)=>void}) {
  const saving=op.share*(1-1/speedup),projected=run.graph.stats.p50*(1-saving);
  const call=run.calls[callIndex];
  return <section className="panel inspector" id="inspector"><div className="panel-head"><div><span className="eyebrow">OPERATOR INSPECTOR</span><h2>{op.name}</h2></div><span className={`tag ${op.variable?"amber":""}`}>{op.variable?"VARIATION OBSERVED":"SAMPLES RETAINED"}</span></div>
    <label className="case-label">CALL CONFIGURATION<select aria-label="Call configuration" value={caseIndex} onChange={e=>onCase(Number(e.target.value))}>{op.case_indices.map(i=><option value={i} key={i}>{run.calls[run.cases[i].representative_call].label??`Call ${run.cases[i].representative_call}`} · {run.cases[i].id.slice(0,8)}</option>)}</select></label>
    <div className="timing-pair">{(["cold","warm"] as const).map(mode=><div className={mode} key={mode}><span><i/>{mode==="cold"?"COLD L2":"WARM REPLAY"}</span><strong>{us(sample[mode].stats.p50)}</strong><small>p10–p90 {us(sample[mode].stats.p10)} – {us(sample[mode].stats.p90)}</small><small>max {us(sample[mode].stats.max)} · CV {pct(sample[mode].stats.cv)}</small></div>)}</div>
    <Distribution sample={sample}/><p className="note">Dashed lines: medians. Every dot is retained, including slow tails. Warm mode primes the op and restores its writes before timing; cold mode restores, then evicts. Variability alone does not identify its cause.</p>
    <div className="in-program"><div><span className="eyebrow">IN THE REAL PROGRAM · {call.label??`CALL ${call.index}`}</span><strong>{us(call.in_program.stats.p50)}</strong><small>p10–p90 {us(call.in_program.stats.p10)} – {us(call.in_program.stats.p90)} · CV {pct(call.in_program.stats.cv)} · max {us(call.in_program.stats.max)}</small></div><Spark series={call.in_program}/></div>
    <div className="whatif"><div><span className="eyebrow">WHAT IF THIS OP WERE FASTER?</span><strong>{number(speedup,1)}×</strong></div><input aria-label="Hypothetical operator speedup" type="range" min="1" max="4" step="0.1" value={speedup} onChange={e=>onSpeedup(Number(e.target.value))}/><p>Projected step <b>{us(projected)}</b><span>−{pct(saving)}</span></p><small>All {op.count} calls · fixed workload · unchanged dependencies and cache interactions. A scenario, not an achieved speedup.</small></div>
    <details><summary>Call arguments & launch entries</summary><pre>{JSON.stringify({call:call.label,index:call.index,signature:sample.signature,launches:call.launches},null,2)}</pre></details>
  </section>;
}

export function PerfPage() {
  const [catalog,setCatalog]=useState<Catalog>([]),[model,setModel]=useState("");
  const [data,setData]=useState<Dataset|null>(null),[error,setError]=useState("");
  const [scenario,setScenario]=useState(""),[operator,setOperator]=useState(""),[caseIndex,setCaseIndex]=useState(0);
  const [selectedCall,setSelectedCall]=useState<number|null>(null);
  const [speedup,setSpeedup]=useState(2),[search,setSearch]=useState(""),[view,setView]=useState<"time"|"variation">("time");
  useEffect(()=>{evidence<Catalog>("index.json").then(c=>{if(!c.length)throw Error("No exported measurements.");setCatalog(c);setModel(c[0].file);}).catch(e=>setError(String(e)));},[]);
  useEffect(()=>{if(!model)return;let cancelled=false;setData(null);setError("");evidence<Dataset>(model).then(d=>{if(!cancelled){setData(d);setScenario(d.scenarios.find(r=>r.scenario.id==="step-b4-l512")?.scenario.id??d.scenarios[0].scenario.id);setOperator("");setCaseIndex(0);setSelectedCall(null);}}).catch(e=>{if(!cancelled)setError(String(e));});return()=>{cancelled=true;};},[model]);
  const run=data?.scenarios.find(r=>r.scenario.id===scenario)??data?.scenarios[0];
  const op=run?.ops.find(o=>o.name===operator)??run?.ops[0];
  const selectedCase=op?.case_indices.includes(caseIndex)?caseIndex:op?.case_indices[0]??0;
  const sample=run?.cases[selectedCase];
  const callIndex=selectedCall!==null&&run?.calls[selectedCall]?.case===selectedCase?selectedCall:sample?.representative_call??0;
  const current=catalog.find(c=>c.file===model);
  const choose=(r:Run)=>{setScenario(r.scenario.id);setCaseIndex(0);setSelectedCall(null);};
  const onCall=(c:Call)=>{setOperator(c.op);setCaseIndex(c.case);setSelectedCall(c.index);};
  const onCase=(n:number)=>{setCaseIndex(n);setSelectedCall(null);};
  const ops=run?.ops.filter(o=>o.name.toLowerCase().includes(search.toLowerCase())).sort((a,b)=>view==="time"?b.share-a.share:b.max_cv-a.max_cv)??[];
  return <div className="atlas"><header className="atlas-header"><a href={packed?"https://kern-baa.pages.dev/":"/"} className="atlas-logo">kern<span>●</span></a><span className="header-divider"/><a href={packed?"#":"/perf/"}>PERFORMANCE ATLAS</a><nav><a href={packed?"https://kern-baa.pages.dev/schema/":"/schema/"}>Schema</a><a href="https://github.com/pegainfer-project/kern">GitHub ↗</a></nav></header>
    <main><div className="atlas-hero"><div><div className="eyebrow"><span className="live-dot"/> SINGLE GPU · REAL MEASUREMENTS</div><h1>A model, down to<br/><span>the microsecond.</span></h1><p>Explore every operator. See what the cache changes.<br/>Find where a faster kernel would actually matter.</p></div><div className="hero-aside"><span className="eyebrow">MANIFEST → MEASURE → UNDERSTAND</span><div className="hero-glyph" aria-hidden="true">{Array.from({length:48},(_,i)=><span key={i} style={{height:12+(i*37%71),background:colors[i%colors.length],opacity:.25+(i%5)*.15}}/>)}</div><span>Portable evidence. No hidden slow samples.</span></div></div>
      <div className="toolbar"><label>MODEL<select aria-label="Model" value={model} onChange={e=>setModel(e.target.value)}>{catalog.map(c=><option key={c.file} value={c.file}>{c.model}</option>)}</select></label>{run&&data&&<><label className="scenario-select">WORKLOAD<select aria-label="Workload" value={run.scenario.id} onChange={e=>choose(data.scenarios.find(r=>r.scenario.id===e.target.value)!)}>{data.scenarios.map(r=><option value={r.scenario.id} key={r.scenario.id}>{title(r)}{r.scenario.holdout?" · holdout":""}</option>)}</select></label><div className="toolbar-links"><EvidenceLink file={model}>Evidence JSON ↓</EvidenceLink>{current&&<EvidenceLink file={current.quick}>AI quick view ↓</EvidenceLink>}</div></>}</div>
      {error?<div className="empty-state" role="alert">{error}</div>:!data||!run||!op||!sample?<div className="empty-state" role="status">Loading measured profiles…</div>:<>
        <div className="context-line"><span>{data.hardware.device} · {new Date(data.created_unix*1000).toISOString().slice(0,10)} · manifest {data.manifest_sha256.slice(0,12)}</span><span>{data.coverage.scenarios} workloads / {data.coverage.distinct_ops} ops / {data.coverage.call_observations.toLocaleString()} call observations</span></div>
        <section className="metrics"><div><span>MEASURED PROGRAM</span><strong>{us(run.graph.stats.p50)}</strong><small>{run.scenario.kind==="step"?"GPU step time · no queue or host loop":"GPU prompt/extend chunk · not serving TTFT"}</small></div><div><span>OBSERVED VARIATION</span><strong>{us(run.graph.stats.p90)}</strong><small>p90 · median {us(run.graph.stats.p50)} · {run.graph.stats.n} samples</small><Spark series={run.graph}/></div><div><span>OP-COST PREDICTION</span><strong>{run.prediction?us(run.prediction.us):"—"}</strong><small>{run.prediction?`${number(run.prediction.error_pct,1)}% error · ${run.scenario.holdout?"held-out program time":"calibration point"}`:"No calibration available"}</small></div><div><span>CALL COVERAGE</span><strong>{run.calls.length}<small> / {run.calls.length}</small></strong><small>{run.cases.length} measured configurations · output check passed</small></div></section>
        <div className="top-grid"><section className="panel"><div className="panel-head"><div><span className="eyebrow">DECODE LANDSCAPE</span><h2>Batch × context</h2></div><span className="tag">CLICK TO EXPLORE</span></div><Heatmap runs={data.scenarios} selected={run} onSelect={choose}/><div className="heat-legend"><span>Lower latency</span><i/><span>Higher latency</span></div><p className="note">Each filled cell is a measured GPU step. Missing cells stay empty. Holdouts were excluded from whole-program calibration; their op costs are measured.</p></section>
        <section className="panel composition"><div className="panel-head"><div><span className="eyebrow">COMPOSITION CHECK</span><h2>Does the sum predict the step?</h2></div></div>{[["Σ cold op medians",run.cold_sum_us,"#80bfff"],["Σ warm op medians",run.warm_sum_us,"#b7f576"],["Calibrated prediction",run.prediction?.us??0,"#d3a5ff"],["Measured graph",run.graph.stats.p50,"#ffffff"]].map(([label,value,c])=><div className="compare-row" key={String(label)}><div><span>{String(label)}</span><b>{us(Number(value))}</b></div><div className="compare-track"><i style={{width:`${Number(value)/Math.max(run.cold_sum_us,run.warm_sum_us,run.prediction?.us??0,run.graph.stats.p50)*100}%`,background:String(c)}}/></div></div>)}<p className="note">Measured with natural reuse between calls, cold L2 at program entry. Event-per-call instrumentation would add {pct(run.instrumentation_ratio-1)} to the graph time. The prediction learns a cold/warm mixture using training workloads only.</p>{run.prediction&&<p className="residual">Training residual envelope <b>{us(run.prediction.range_us[0])} – {us(run.prediction.range_us[1])}</b><small>Not a latency percentile or confidence interval.</small></p>}</section></div>
        <div className="detail-grid"><div><section className="panel operators"><div className="panel-head"><div><span className="eyebrow">OPTIMIZATION MAP</span><h2>Where the time goes</h2></div><span className="tag">{run.ops.length} OPS</span></div><div className="table-tools"><input aria-label="Filter operators" placeholder="Find an operator…" value={search} onChange={e=>setSearch(e.target.value)}/><select aria-label="Sort operators" value={view} onChange={e=>setView(e.target.value as "time"|"variation")}><option value="time">Time contribution</option><option value="variation">Sample variation</option></select></div><div className="op-table"><div className="op-table-head"><span>OPERATOR</span><span>CALLS</span><span>SHARE</span><span>AT 2×</span></div>{ops.map(o=><button className={`op-row ${o.name===op.name?"selected":""}`} onClick={()=>{setOperator(o.name);onCase(o.case_indices[0]);}} key={o.name}><span className="op-name"><i style={{background:color(o.name)}}/>{o.name}{o.variable&&<em title="CV > 10% or p90/median > 1.15 in at least one configuration">∿</em>}<span className="op-bar" style={{width:`${o.share*100}%`,background:color(o.name)}}/></span><span>{o.count}</span><span>{pct(o.share)}</span><span>−{us(o.saving_at_2x_us)}</span></button>)}{!ops.length&&<p className="note">No operators match this filter.</p>}</div><p className="note">Shares use uninstrumented program GPU activity durations, scaled to the actual graph time. “At 2×” is a hypothetical saving across all calls. ∿ marks observed variability, not a diagnosis.</p></section>
          <section className="panel program"><div className="panel-head"><div><span className="eyebrow">PROGRAM EXPLORER</span><h2>{run.program}</h2></div><span className="tag">CALL → OP → LAUNCH</span></div><Program run={run} selected={callIndex} onCall={onCall}/><p className="note">Each segment is one call; width encodes GPU activity duration. Click to inspect its implementation and samples. Layer labels come from the manifest.</p></section></div>
          <Inspector sample={sample} op={op} run={run} caseIndex={selectedCase} callIndex={callIndex} onCase={onCase} speedup={speedup} onSpeedup={setSpeedup}/></div>
        <Hardware data={data}/>
        {data.repeat_check&&<div className="repeat-check"><span className="eyebrow">INDEPENDENT UNTRACED REPEAT</span><p>{data.repeat_check.scenarios} workloads · median absolute difference <b>{number(data.repeat_check.median_abs_delta_pct,2)}%</b> · maximum <b>{number(data.repeat_check.max_abs_delta_pct,2)}%</b> · token outputs agree.</p><small>Same measurement protocol, separate program-only run. Includes run-to-run variability; not a pure measurement of tracer overhead.</small></div>}
        <section className="panel validation"><div className="panel-head"><div><span className="eyebrow">HELD-OUT PROGRAM TIMES</span><h2>Prediction errors stay visible</h2></div><span className="tag">NO FIT ON THESE TARGET TIMES</span></div><div className="validation-table"><div><span>WORKLOAD</span><span>MEASURED</span><span>PREDICTED</span><span>ERROR</span></div>{data.scenarios.filter(r=>r.scenario.holdout).map(r=><button onClick={()=>choose(r)} key={r.scenario.id}><span>{title(r)}</span><span>{us(r.graph.stats.p50)}</span><span>{r.prediction?us(r.prediction.us):"—"}</span><span className={Math.abs(r.prediction?.error_pct??0)>10?"warning":"good"}>{r.prediction?`${number(r.prediction.error_pct,1)}%`:"—"}</span></button>)}</div><p className="note">This validates composition from measured op costs. It does not establish accuracy on unseen shapes, request queueing, speculative acceptance, or multi-GPU execution.</p></section>
        <details className="method"><summary>Measurement protocol & limits</summary><p>Every batch sequence has a private state lease and a real prose prefix. Declared writes are restored before each op sample, including opaque state. Eviction and restoration are outside timing brackets. Cold/warm order alternates, and every timed sample is retained. Whole-program outputs are checked against the profiled call trajectory.</p><p>Full-program timing uses CUDA events around the entire graph. Per-call attribution uses GPU activity timestamps from those same runs, with no events between calls. Isolated implementation sequences are checked against every program call. Hardware anchors are measured before and after the sweep. Variability can reflect the kernel, cache conditions, clocks or the measurement setup; this demo flags it without assigning an unsupported cause.</p><pre>{JSON.stringify(data.protocol,null,2)}</pre></details>
      </>}
    </main><footer><a href={packed?"https://kern-baa.pages.dev/":"/"} className="atlas-logo">kern<span>●</span></a><span>A model is a program. Its performance should be inspectable.</span><a href="https://github.com/pegainfer-project/kern">Source & methodology ↗</a></footer></div>;
}
