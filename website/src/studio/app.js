/* KERN Manifest Studio — Interactive Blueprint Engine */
(function () {
  "use strict";

  // ==================== CONSTANTS & UTILITIES ====================
  const DTYPE_BITS = {
    u4: 4,
    u8: 8, i8: 8, fp8e4m3: 8, fp8e5m2: 8, fp8e8m0: 8,
    f16: 16, bf16: 16, i16: 16, u16: 16,
    f32: 32, i32: 32, u32: 32,
    f64: 64, i64: 64, u64: 64,
  };

  const MEM_ALIGN = 256;

  // High-clarity cobalt, copper & emerald stage palette inspired by susun-blog.com
  const SIG_PALETTE = [
    "#1f4fff", "#d96226", "#0d9466", "#c97d10",
    "#0884c7", "#6554d9", "#d13b56", "#129488",
  ];

  function esc(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function trunc(s, maxLen) {
    const str = String(s ?? "");
    return str.length > maxLen ? str.slice(0, maxLen - 1) + "…" : str;
  }

  function formatBytes(bytes) {
    if (bytes == null || !Number.isFinite(bytes) || bytes < 0) return "—";
    if (bytes === 0) return "0 B";
    const units = ["B", "KiB", "MiB", "GiB", "TiB"];
    let v = bytes;
    let u = 0;
    while (v >= 1024 && u < units.length - 1) {
      v /= 1024;
      u++;
    }
    return (v >= 100 || u === 0 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2)) + " " + units[u];
  }

  function formatExpr(e) {
    if (e == null) return "";
    if (typeof e === "number" || typeof e === "string") return String(e);
    if (typeof e === "object") {
      if (e.ceil_div) return `⌈${formatExpr(e.ceil_div[0])}/${formatExpr(e.ceil_div[1])}⌉`;
      if (e.mul) return `${formatExpr(e.mul[0])}×${formatExpr(e.mul[1])}`;
      if (e.add) return `(${formatExpr(e.add[0])}+${formatExpr(e.add[1])})`;
      if (e.clamp) return `clamp(${formatExpr(e.clamp[0])}, ${e.clamp[1]}, ${e.clamp[2]})`;
    }
    return JSON.stringify(e);
  }

  function evalExpr(e, vars) {
    if (e == null) return 0;
    if (typeof e === "number") return e;
    if (typeof e === "string") return vars[e] ?? 1;
    if (typeof e === "object") {
      if (e.ceil_div) {
        const a = evalExpr(e.ceil_div[0], vars);
        const b = evalExpr(e.ceil_div[1], vars) || 1;
        return Math.ceil(a / b);
      }
      if (e.mul) return evalExpr(e.mul[0], vars) * evalExpr(e.mul[1], vars);
      if (e.add) return evalExpr(e.add[0], vars) + evalExpr(e.add[1], vars);
      if (e.clamp) {
        const v = evalExpr(e.clamp[0], vars);
        const start = Number(e.clamp[1]) || 0;
        const len = Number(e.clamp[2]) || 0;
        return Math.min(Math.max(v - start, 0), len);
      }
    }
    return 0;
  }

  function parseParamSpec(spec) {
    const s = String(spec || "").trim();
    let m = /^(in|out|inout)\s+buffer<([^>]+)>$/.exec(s);
    if (m) return { dir: m[1], kind: "buffer", dtype: m[2], raw: s };
    m = /^(in|out|inout)\s+state$/.exec(s);
    if (m) return { dir: m[1], kind: "state", dtype: "state", raw: s };
    m = /^bytes<(\d+)>$/.exec(s);
    if (m) return { dir: "in", kind: "bytes", size: Number(m[1]), dtype: `bytes<${m[1]}>`, raw: s };
    return { dir: "scalar", kind: "scalar", dtype: s, raw: s };
  }

  function evalBufferBytes(buf, vars) {
    if (!buf || !buf.shape) return 0;
    const bits = DTYPE_BITS[buf.dtype] || 8;
    let elems = 1;
    for (const d of buf.shape) {
      elems *= Math.max(0, evalExpr(d, vars));
    }
    return Math.ceil((elems * bits) / 8);
  }

  function seqSlotsOf(manifest) {
    let maxCols = 0;
    for (const b of Object.values(manifest.buffers || {})) {
      if (b.kind !== "input") continue;
      const target = b.domain && b.domain.index_into;
      const st = target && manifest.states && manifest.states[target];
      if (!st || st.bytes_per_seq == null) continue;
      const dim1 = b.shape && b.shape[1];
      if (typeof dim1 === "string" && manifest.vars && manifest.vars[dim1]) {
        maxCols = Math.max(maxCols, Number(manifest.vars[dim1].max) || 1);
      } else if (typeof dim1 === "number") {
        maxCols = Math.max(maxCols, dim1);
      }
    }
    return (maxCols > 0 ? maxCols : 1) + 2;
  }

  function evalStateBytes(state, manifest, vars) {
    if (!state) return 0;
    if (state.bytes != null) return Number(state.bytes);
    if (state.bytes_per_token != null) {
      const tokCap = vars.tokens || (manifest.vars?.tokens?.max) || 1024;
      return Number(state.bytes_per_token) * tokCap;
    }
    if (state.bytes_per_seq != null) {
      const activeSeqs = vars.seqs != null ? (vars.seqs + 2) : seqSlotsOf(manifest);
      return Number(state.bytes_per_seq) * activeSeqs;
    }
    return 0;
  }

  function isLaunchActive(launch, vars) {
    if (!launch.when) return true;
    const v = vars[launch.when.var];
    if (v == null) return true;
    if (launch.when.min != null && v < launch.when.min) return false;
    if (launch.when.max != null && v > launch.when.max) return false;
    return true;
  }

  function classifyOp(opName, label, opDef) {
    const s = `${opName} ${label || ""}`.toLowerCase();
    const launches = opDef?.impl?.launches || [];
    if (launches.some((l) => String(l.entry || "").includes("nccl") || String(l.entry || "").includes("allreduce")) ||
        s.includes("allreduce") || s.includes("allgather") || s.includes("alltoall") || s.includes("peer")) {
      return { cat: "comm", color: "var(--cat-comm)", tag: "COMM" };
    }
    if (s.includes("attn") || s.includes("attention") || s.includes("fmha") || s.includes("gdn") ||
        s.includes("ssm") || s.includes("recurrent") || s.includes("conv") || s.includes("mla") || s.includes("kv_write")) {
      return { cat: "attn", color: "var(--cat-attn)", tag: "ATTN/SSM" };
    }
    if (s.includes("gemm") || s.includes("linear") || s.includes("proj") || s.includes("qkv") ||
        s.includes("gate_up") || s.includes("down") || s.includes("lm_head") || s.includes("expert") || s.includes("moe") ||
        launches.some((l) => String(l.entry || "").includes("cublas"))) {
      return { cat: "gemm", color: "var(--cat-gemm)", tag: "LINEAR" };
    }
    if (s.includes("norm") || s.includes("silu") || s.includes("act") || s.includes("rope") ||
        s.includes("add") || s.includes("residual") || s.includes("embed") || s.includes("merge") || s.includes("finish")) {
      return { cat: "norm", color: "var(--cat-norm)", tag: "NORM/ACT" };
    }
    if (s.includes("plan") || s.includes("sample") || s.includes("argmax") || s.includes("splice") ||
        s.includes("code") || s.includes("verify") || s.includes("iota") || s.includes("fill")) {
      return { cat: "plan", color: "var(--cat-plan)", tag: "PLAN/CTRL" };
    }
    return { cat: "misc", color: "var(--cat-misc)", tag: "OP" };
  }

  function kindColor(kind) {
    switch (kind) {
      case "input": return "var(--kind-input)";
      case "output": return "var(--kind-output)";
      case "workspace": return "var(--kind-workspace)";
      case "carry": return "var(--kind-carry)";
      case "state": return "var(--kind-state)";
      case "peer": return "var(--kind-peer)";
      case "source": return "var(--kind-source)";
      default: return "var(--kind-weight)";
    }
  }

  // ==================== APPLICATION STATE ====================
  const state = {
    presets: [],
    currentPresetId: "",
    meta: null,
    manifest: null,
    derivedConstSet: new Set(),
    vars: {},
    activeView: "graph",      // "graph" | "memory" | "protocol"
    activeSubtab: "stages",   // "stages" | "buffers"
    programName: "",
    analysis: null,           // cached per program
    scopeMode: "macro",       // "macro" | "stage" | "slice"
    focusedStageIdx: 0,
    expandedStageIdxs: new Set(),
    unfoldedFamilyKeys: new Set(),
    sliceStart: 0,
    sliceEnd: 28,
    showBuffers: false,
    showStates: true,
    showInputs: true,
    showWeights: false,
    foldRepeated: true,
    orient: "TB",             // "TB" | "LR"
    traceMode: "both",
    searchQuery: "",
    bufKindFilter: "all",
    selected: null,           // { type: "call"|"resource"|"stage"|"fold", ... }
    selectedNodeId: null,
    layoutResult: null,
    zoom: { k: 1, x: 40, y: 40 },
    // Memory & Peak Budget state
    memGpuGiB: 80,
    memSeqSlots: 16,
    memKvTokens: 32768,
    memFilterCat: "all",      // "all" | "weights" | "activations" | "seq_states" | "kv_states" | "staging"
    memGroupLayers: true,
    memSearch: "",
  };

  // ==================== SSA & STAGE PROGRAM ANALYZER ====================
  function extractStagePrefix(label) {
    if (!label) return "";
    const clean = String(label).trim();
    const layersMatch = /^(?:model\.)?(?:language_model\.)?layers\.(\d+)/.exec(clean);
    if (layersMatch) return `layers.${layersMatch[1]}`;
    const parts = clean.split(".");
    if (parts.length <= 1) return "";
    if (parts.length > 2 && /^[a-z]*\d+$/i.test(parts[1]) && parts[1].length <= 10) {
      return `${parts[0]}.${parts[1]}`;
    }
    return parts[0];
  }

  function stageStem(baseName, isGrouped) {
    if (!isGrouped || !/\d/.test(baseName)) return "";
    if (/^(?:layers\.|layer\.?|l)\d+$/i.test(baseName)) return "layer.#";
    return baseName.replace(/\d+/g, "#");
  }

  function normalizeOpForSignature(opName) {
    return String(opName || "").replace(/(^|[._])(l|layer|layers|draft)\.?(\d+)(?=[._]|$)/gi, "$1$2*");
  }

  function analyzeProgram(manifest, programName) {
    const prog = manifest.programs[programName];
    const calls = prog?.calls || [];
    const buffers = manifest.buffers || {};
    const states = manifest.states || {};
    const ops = manifest.ops || {};

    // Extract contiguous stages with sandwich-gap coalescing
    const rawPrefixes = calls.map((c) => extractStagePrefix(c.label));
    const resolvedStageNames = rawPrefixes.slice();

    for (let i = 0; i < calls.length; i++) {
      if (resolvedStageNames[i]) continue;
      // Look backward and forward within a bounded window for matching stage prefix
      let prevP = "";
      for (let p = i - 1; p >= Math.max(0, i - 8); p--) {
        if (resolvedStageNames[p]) { prevP = resolvedStageNames[p]; break; }
      }
      if (!prevP) continue;
      let nextP = "";
      for (let q = i + 1; q < Math.min(calls.length, i + 10); q++) {
        if (rawPrefixes[q]) { nextP = rawPrefixes[q]; break; }
      }
      if (nextP === prevP) {
        resolvedStageNames[i] = prevP;
      }
    }

    // Build contiguous stage objects
    const stages = [];
    for (let i = 0; i < calls.length; i++) {
      const sName = resolvedStageNames[i] || calls[i].label || `${calls[i].op}#${i}`;
      const isGrouped = Boolean(resolvedStageNames[i]);
      const prev = stages[stages.length - 1];
      if (isGrouped && prev && prev.isGrouped && prev.baseName === sName) {
        prev.endCall = i;
        prev.callIndices.push(i);
      } else {
        stages.push({
          idx: stages.length,
          baseName: sName,
          name: sName,
          isGrouped,
          startCall: i,
          endCall: i,
          callIndices: [i],
        });
      }
    }

    // Disambiguate duplicate stage names (e.g. forward l0 vs late commit l0)
    const nameCounts = {};
    stages.forEach((st) => { nameCounts[st.baseName] = (nameCounts[st.baseName] || 0) + 1; });
    const seenCounts = {};
    stages.forEach((st) => {
      seenCounts[st.baseName] = (seenCounts[st.baseName] || 0) + 1;
      if (nameCounts[st.baseName] > 1 && st.isGrouped) {
        const suffix = seenCounts[st.baseName] > 1 ? ` (pass ${seenCounts[st.baseName]})` : "";
        st.displayName = st.baseName + suffix;
      } else {
        st.displayName = st.baseName;
      }
    });

    // Compute structural signature for each stage (100% model-agnostic)
    const sigMap = new Map();
    stages.forEach((st) => {
      const sigStr = st.callIndices.map((ci) => normalizeOpForSignature(calls[ci].op)).join("|");
      if (!sigMap.has(sigStr)) {
        sigMap.set(sigStr, sigMap.size);
      }
      st.sigId = sigMap.get(sigStr);
      st.sigColor = st.callIndices.length > 1 ? SIG_PALETTE[st.sigId % SIG_PALETTE.length] : "#82938f";
    });

    // Map callIdx -> stageIdx
    const callToStage = new Int32Array(calls.length);
    stages.forEach((st) => {
      for (const ci of st.callIndices) callToStage[ci] = st.idx;
    });

    // Pre-scan buffers written inside this program so read-only `carry` constants (from `derive`) are recognized
    const writtenBufsInProg = new Set();
    calls.forEach((call) => {
      const opDef = ops[call.op];
      if (!opDef) return;
      const pSpecs = (opDef.params || []).map(parseParamSpec);
      (call.args || []).forEach((arg, ai) => {
        const dir = pSpecs[ai]?.dir;
        if (arg?.buf != null && (dir === "out" || dir === "inout")) {
          writtenBufsInProg.add(arg.buf);
        }
      });
    });

    const isOnceProg = Boolean(prog?.once);
    function isConstBuffer(bKind, bName) {
      if (bKind === "weight" || bKind === "source") return true;
      if (!isOnceProg && bKind === "carry" && !writtenBufsInProg.has(bName) && state.derivedConstSet.has(bName)) {
        return true;
      }
      return false;
    }

    // Two-phase SSA version tracking across all calls
    const curVer = new Map();       // resourceKey -> { ver, writerCall, offset }
    const resourceAccesses = new Map(); // resourceKey -> Array<{ callIdx, dir, ver, offset }>
    const selfUpdateCounts = new Map(); // resourceKey -> count of in+out self-updates (residual spine detector!)

    function resKey(isState, name) {
      return (isState ? "state:" : "buf:") + name;
    }

    const callInfos = calls.map((call, ci) => {
      const opDef = ops[call.op] || { params: [], impl: { launches: [] } };
      const pSpecs = (opDef.params || []).map(parseParamSpec);
      const args = call.args || [];
      const reads = [];
      const writes = [];
      const scalars = [];
      const readResourcesThisCall = new Set();

      // Phase 1: Record all reads against PRE-CALL SSA versions
      args.forEach((arg, ai) => {
        const spec = pSpecs[ai] || { dir: "scalar", kind: "scalar", dtype: "?" };
        if (arg && (arg.buf != null || arg.state != null)) {
          const isState = arg.state != null;
          const name = isState ? arg.state : arg.buf;
          const rk = resKey(isState, name);
          const bMeta = isState ? states[name] : buffers[name];
          const bKind = isState ? "state" : (bMeta?.kind || "workspace");
          const offset = arg.offset || 0;

          if (spec.dir === "in" || spec.dir === "inout") {
            const cur = curVer.get(rk) || { ver: 0, writerCall: -1, offset: 0 };
            reads.push({
              argIdx: ai,
              dir: spec.dir,
              isState,
              name,
              rk,
              kind: bKind,
              isConst: !isState && isConstBuffer(bKind, name),
              dtype: isState ? "state" : (bMeta?.dtype || spec.dtype),
              ver: cur.ver,
              producerCall: cur.writerCall,
              offset,
            });
            readResourcesThisCall.add(rk);
            if (!resourceAccesses.has(rk)) resourceAccesses.set(rk, []);
            resourceAccesses.get(rk).push({ callIdx: ci, dir: "read", ver: cur.ver, offset });
          }
        } else {
          scalars.push({ argIdx: ai, spec, arg });
        }
      });

      // Phase 2: Advance SSA versions for all writes (`out` / `inout`)
      args.forEach((arg, ai) => {
        const spec = pSpecs[ai] || { dir: "scalar", kind: "scalar", dtype: "?" };
        if (arg && (arg.buf != null || arg.state != null) && (spec.dir === "out" || spec.dir === "inout")) {
          const isState = arg.state != null;
          const name = isState ? arg.state : arg.buf;
          const rk = resKey(isState, name);
          const bMeta = isState ? states[name] : buffers[name];
          const bKind = isState ? "state" : (bMeta?.kind || "workspace");
          const offset = arg.offset || 0;

          const prev = curVer.get(rk) || { ver: 0, writerCall: -1 };
          const nextVer = prev.ver + 1;
          curVer.set(rk, { ver: nextVer, writerCall: ci, offset });

          if (!isState && readResourcesThisCall.has(rk)) {
            selfUpdateCounts.set(rk, (selfUpdateCounts.get(rk) || 0) + 1);
          }

          writes.push({
            argIdx: ai,
            dir: spec.dir,
            isState,
            name,
            rk,
            kind: bKind,
            dtype: isState ? "state" : (bMeta?.dtype || spec.dtype),
            ver: nextVer,
            offset,
          });
          if (!resourceAccesses.has(rk)) resourceAccesses.set(rk, []);
          resourceAccesses.get(rk).push({ callIdx: ci, dir: "write", ver: nextVer, offset });
        }
      });

      return {
        idx: ci,
        call,
        label: call.label || `${call.op}#${ci}`,
        opName: call.op,
        opDef,
        cat: classifyOp(call.op, call.label, opDef),
        stageIdx: callToStage[ci],
        reads,
        writes,
        scalars,
      };
    });

    // Identify primary residual spine buffer(s): workspace/carry buffers self-updated >= 2 times
    const residualKeys = new Set();
    for (const [rk, count] of selfUpdateCounts.entries()) {
      if (count >= 2) residualKeys.add(rk);
    }

    return {
      programName,
      prog,
      calls,
      callInfos,
      stages,
      callToStage,
      resourceAccesses,
      residualKeys,
    };
  }

  // ==================== CUSTOM SUGIYAMA LAYERED LAYOUT ENGINE ====================
  function computeSugiyamaLayout(nodes, edges, orient) {
    if (!nodes.length) return { nodes: [], edges: [], width: 400, height: 300 };

    const isLR = orient === "LR";
    const nodeMap = new Map();
    nodes.forEach((n, idx) => {
      n._order = idx;
      n.rank = 0;
      nodeMap.set(n.id, n);
    });

    const validEdges = edges.filter((e) => nodeMap.has(e.src) && nodeMap.has(e.dst) && e.src !== e.dst);
    const outAdj = new Map();
    const inAdj = new Map();
    nodes.forEach((n) => { outAdj.set(n.id, []); inAdj.set(n.id, []); });

    // Cycle-safe DAG edge insertion (respecting natural order on ties)
    validEdges.forEach((e) => {
      outAdj.get(e.src).push(e);
      inAdj.get(e.dst).push(e);
    });

    // Kahn topological sort + cycle breaking
    const inDeg = new Map();
    nodes.forEach((n) => inDeg.set(n.id, inAdj.get(n.id).length));
    const ready = nodes.filter((n) => inDeg.get(n.id) === 0).sort((a, b) => a._order - b._order);
    const topo = [];
    const visited = new Set();

    while (topo.length < nodes.length) {
      let cur = ready.shift();
      if (!cur) {
        cur = nodes.find((n) => !visited.has(n.id));
      }
      if (!cur || visited.has(cur.id)) continue;
      visited.add(cur.id);
      topo.push(cur);
      for (const e of outAdj.get(cur.id)) {
        if (visited.has(e.dst)) {
          e._backEdge = true;
          continue;
        }
        const d = inDeg.get(e.dst) - 1;
        inDeg.set(e.dst, d);
        if (d === 0) ready.push(nodeMap.get(e.dst));
      }
    }

    // Longest-path rank assignment
    topo.forEach((n) => {
      let r = 0;
      for (const e of inAdj.get(n.id)) {
        if (e._backEdge) continue;
        const p = nodeMap.get(e.src);
        r = Math.max(r, p.rank + 1);
      }
      n.rank = r;
    });

    // Pull leaf/floating input pills closer to their first consumer so inputs don't float 20 ranks away!
    for (let i = topo.length - 1; i >= 0; i--) {
      const n = topo[i];
      if (inAdj.get(n.id).filter((e) => !e._backEdge).length === 0 && outAdj.get(n.id).length > 0) {
        let minChildRank = Infinity;
        for (const e of outAdj.get(n.id)) {
          if (!e._backEdge) minChildRank = Math.min(minChildRank, nodeMap.get(e.dst).rank);
        }
        if (Number.isFinite(minChildRank) && minChildRank > 0) {
          n.rank = minChildRank - 1;
        }
      }
    }

    // Normalize ranks to start at 0
    let minR = Infinity;
    let maxR = 0;
    nodes.forEach((n) => {
      if (n.rank < minR) minR = n.rank;
      if (n.rank > maxR) maxR = n.rank;
    });
    if (!Number.isFinite(minR)) minR = 0;
    nodes.forEach((n) => { n.rank -= minR; });
    maxR -= minR;

    // Insert virtual bend nodes for edges spanning > 1 rank (creates clean highway channels!)
    const allNodes = nodes.slice();
    const rankBuckets = Array.from({ length: maxR + 1 }, () => []);
    nodes.forEach((n) => rankBuckets[n.rank].push(n));

    const routedSegments = [];
    let vCounter = 0;

    validEdges.forEach((e) => {
      const u = nodeMap.get(e.src);
      const v = nodeMap.get(e.dst);
      if (e._backEdge || v.rank <= u.rank + 1) {
        e.waypoints = [u, v];
        routedSegments.push({ u: u.id, v: v.id });
        return;
      }
      // Cap virtual chain if span is huge (> 40 ranks) to keep layout snappy
      const span = v.rank - u.rank;
      if (span > 45) {
        e.waypoints = [u, v];
        routedSegments.push({ u: u.id, v: v.id });
        return;
      }
      const chain = [u];
      for (let r = u.rank + 1; r < v.rank; r++) {
        const vn = {
          id: `__virt_${vCounter++}`,
          isVirtual: true,
          isResidual: e.isResidual,
          rank: r,
          w: e.isResidual ? 26 : 16,
          h: e.isResidual ? 20 : 14,
          _order: e.isResidual ? -100 : u._order, // Keep residual highway consistently on one side!
        };
        allNodes.push(vn);
        nodeMap.set(vn.id, vn);
        rankBuckets[r].push(vn);
        chain.push(vn);
      }
      chain.push(v);
      e.waypoints = chain;
      for (let k = 0; k < chain.length - 1; k++) {
        routedSegments.push({ u: chain[k].id, v: chain[k + 1].id });
      }
    });

    // Build segment adjacency for barycenter crossing minimization
    const segParents = new Map();
    const segChildren = new Map();
    allNodes.forEach((n) => { segParents.set(n.id, []); segChildren.set(n.id, []); });
    routedSegments.forEach((s) => {
      segChildren.get(s.u)?.push(s.v);
      segParents.get(s.v)?.push(s.u);
    });

    // Initial deterministic order per rank
    rankBuckets.forEach((bucket) => {
      bucket.sort((a, b) => a._order - b._order);
      bucket.forEach((n, idx) => { n._pos = idx; });
    });

    // 8 sweeps of barycenter ordering
    for (let iter = 0; iter < 8; iter++) {
      // Top-down
      for (let r = 1; r <= maxR; r++) {
        const bucket = rankBuckets[r];
        bucket.forEach((n) => {
          const ps = segParents.get(n.id);
          if (ps && ps.length) {
            let sum = 0;
            for (const pid of ps) sum += nodeMap.get(pid)._pos;
            n._bary = sum / ps.length;
          } else {
            n._bary = n._pos;
          }
        });
        bucket.sort((a, b) => (a._bary - b._bary) || (a._order - b._order));
        bucket.forEach((n, idx) => { n._pos = idx; });
      }
      // Bottom-up
      for (let r = maxR - 1; r >= 0; r--) {
        const bucket = rankBuckets[r];
        bucket.forEach((n) => {
          const cs = segChildren.get(n.id);
          if (cs && cs.length) {
            let sum = 0;
            for (const cid of cs) sum += nodeMap.get(cid)._pos;
            n._bary = sum / cs.length;
          } else {
            n._bary = n._pos;
          }
        });
        bucket.sort((a, b) => (a._bary - b._bary) || (a._order - b._order));
        bucket.forEach((n, idx) => { n._pos = idx; });
      }
    }

    // Coordinate assignment along Primary (rank) and Cross (within-rank) axes
    const RANK_GAP = isLR ? 48 : 30;
    const NODE_GAP = isLR ? 22 : 24;

    // Primary axis (y in TB, x in LR)
    let primaryCursor = 48;
    const rankCenterPrimary = new Float64Array(maxR + 1);
    for (let r = 0; r <= maxR; r++) {
      const bucket = rankBuckets[r];
      let maxThickness = 32;
      for (const n of bucket) {
        const thick = isLR ? n.w : n.h;
        if (thick > maxThickness) maxThickness = thick;
      }
      rankCenterPrimary[r] = primaryCursor + maxThickness / 2;
      for (const n of bucket) {
        if (isLR) n.cx = rankCenterPrimary[r];
        else n.cy = rankCenterPrimary[r];
      }
      primaryCursor += maxThickness + RANK_GAP;
    }

    // Initial centered Cross axis placement
    rankBuckets.forEach((bucket) => {
      let totalSpan = 0;
      bucket.forEach((n, i) => {
        totalSpan += (isLR ? n.h : n.w) + (i > 0 ? NODE_GAP : 0);
      });
      let cursor = -totalSpan / 2;
      bucket.forEach((n) => {
        const span = isLR ? n.h : n.w;
        n._cross = cursor + span / 2;
        cursor += span + NODE_GAP;
      });
    });

    // 6-pass damped neighbor alignment so 1-to-1 spines align laser-straight without overlaps!
    function resolveBucketOverlaps(bucket) {
      for (let i = 1; i < bucket.length; i++) {
        const prev = bucket[i - 1];
        const cur = bucket[i];
        const minDist = ((isLR ? prev.h : prev.w) + (isLR ? cur.h : cur.w)) / 2 + NODE_GAP;
        if (cur._cross - prev._cross < minDist) {
          cur._cross = prev._cross + minDist;
        }
      }
    }

    for (let pass = 0; pass < 6; pass++) {
      // Align children to parents
      for (let r = 1; r <= maxR; r++) {
        const bucket = rankBuckets[r];
        bucket.forEach((n) => {
          const ps = segParents.get(n.id);
          if (ps && ps.length) {
            let sum = 0;
            for (const pid of ps) sum += nodeMap.get(pid)._cross;
            const target = sum / ps.length;
            n._cross = n._cross * 0.3 + target * 0.7;
          }
        });
        resolveBucketOverlaps(bucket);
      }
      // Align parents to children (right-to-left overlap resolution)
      for (let r = maxR - 1; r >= 0; r--) {
        const bucket = rankBuckets[r];
        bucket.forEach((n) => {
          const cs = segChildren.get(n.id);
          if (cs && cs.length) {
            let sum = 0;
            for (const cid of cs) sum += nodeMap.get(cid)._cross;
            const target = sum / cs.length;
            n._cross = n._cross * 0.35 + target * 0.65;
          }
        });
        for (let i = bucket.length - 2; i >= 0; i--) {
          const cur = bucket[i];
          const next = bucket[i + 1];
          const minDist = ((isLR ? cur.h : cur.w) + (isLR ? next.h : next.w)) / 2 + NODE_GAP;
          if (next._cross - cur._cross < minDist) {
            cur._cross = next._cross - minDist;
          }
        }
      }
    }

    // Shift Cross coordinates so minimum is +48px margin
    let minCross = Infinity;
    let maxCross = -Infinity;
    allNodes.forEach((n) => {
      const half = (isLR ? n.h : n.w) / 2;
      if (n._cross - half < minCross) minCross = n._cross - half;
      if (n._cross + half > maxCross) maxCross = n._cross + half;
    });
    const shift = 48 - (Number.isFinite(minCross) ? minCross : 0);
    allNodes.forEach((n) => {
      const c = n._cross + shift;
      if (isLR) {
        n.cy = c;
        n.x = n.cx - n.w / 2;
        n.y = n.cy - n.h / 2;
      } else {
        n.cx = c;
        n.x = n.cx - n.w / 2;
        n.y = n.cy - n.h / 2;
      }
    });

    // Build smooth SVG Bezier spline paths for edges
    validEdges.forEach((e) => {
      const pts = e.waypoints.map((wp, idx) => {
        if (idx === 0) {
          return isLR ? { x: wp.x + wp.w, y: wp.cy } : { x: wp.cx, y: wp.y + wp.h };
        }
        if (idx === e.waypoints.length - 1) {
          return isLR ? { x: wp.x, y: wp.cy } : { x: wp.cx, y: wp.y };
        }
        return { x: wp.cx, y: wp.cy };
      });
      e.path = buildSmoothSpline(pts, isLR);
      e.midPoint = pts[Math.floor(pts.length / 2)];
    });

    const totalW = (isLR ? primaryCursor : (maxCross + shift)) + 64;
    const totalH = (isLR ? (maxCross + shift) : primaryCursor) + 64;

    return {
      nodes,
      edges: validEdges,
      width: Math.max(400, totalW),
      height: Math.max(300, totalH),
    };
  }

  function buildSmoothSpline(pts, isLR) {
    if (pts.length < 2) return "";
    if (pts.length === 2) {
      const [a, b] = pts;
      if (isLR) {
        const dx = Math.max(28, Math.abs(b.x - a.x) * 0.45);
        return `M ${a.x.toFixed(1)} ${a.y.toFixed(1)} C ${(a.x + dx).toFixed(1)} ${a.y.toFixed(1)}, ${(b.x - dx).toFixed(1)} ${b.y.toFixed(1)}, ${b.x.toFixed(1)} ${b.y.toFixed(1)}`;
      }
      const dy = Math.max(24, Math.abs(b.y - a.y) * 0.45);
      return `M ${a.x.toFixed(1)} ${a.y.toFixed(1)} C ${a.x.toFixed(1)} ${(a.y + dy).toFixed(1)}, ${b.x.toFixed(1)} ${(b.y - dy).toFixed(1)}, ${b.x.toFixed(1)} ${b.y.toFixed(1)}`;
    }
    // Multi-waypoint smooth cubic curve through virtual channel waypoints
    let d = `M ${pts[0].x.toFixed(1)} ${pts[0].y.toFixed(1)}`;
    for (let i = 0; i < pts.length - 1; i++) {
      const p0 = pts[Math.max(0, i - 1)];
      const p1 = pts[i];
      const p2 = pts[i + 1];
      const p3 = pts[Math.min(pts.length - 1, i + 2)];
      const tension = 0.22;
      const cp1x = p1.x + (p2.x - p0.x) * tension;
      const cp1y = p1.y + (p2.y - p0.y) * tension;
      const cp2x = p2.x - (p3.x - p1.x) * tension;
      const cp2y = p2.y - (p3.y - p1.y) * tension;
      d += ` C ${cp1x.toFixed(1)} ${cp1y.toFixed(1)}, ${cp2x.toFixed(1)} ${cp2y.toFixed(1)}, ${p2.x.toFixed(1)} ${p2.y.toFixed(1)}`;
    }
    return d;
  }

  // ==================== GRAPH MODEL BUILDER (MACRO / STAGE / SLICE) ====================
  function buildGraphModel() {
    const an = state.analysis;
    const manifest = state.manifest;
    if (!an || !manifest) return { nodes: [], edges: [], stageBoxes: [] };

    if (state.scopeMode === "stage") {
      const st = an.stages[state.focusedStageIdx] || an.stages[0];
      if (!st) return { nodes: [], edges: [], stageBoxes: [] };
      return buildCallRangeSubgraph(st.startCall, st.endCall, st);
    }

    if (state.scopeMode === "slice") {
      const s = Math.max(0, Math.min(an.calls.length - 1, state.sliceStart | 0));
      const e = Math.max(s, Math.min(an.calls.length - 1, state.sliceEnd | 0));
      return buildCallRangeSubgraph(s, e, null);
    }

    // MACRO PIPELINE MODE: Folded super-blocks + Stage cards + Inline Expanded stages!
    return buildMacroGraph();
  }

  function buildCallRangeSubgraph(startCall, endCall, stageObj) {
    const an = state.analysis;
    const manifest = state.manifest;
    const nodes = [];
    const edges = [];
    const pillCreated = new Set();

    function ensureTensorNode(acc, isEntry) {
      const { isState, name, rk, kind, isConst, dtype, ver, offset } = acc;
      if (isConst && !state.showWeights) return null;
      if (isState && !state.showStates) return null;
      if ((kind === "input" || kind === "peer") && !state.showInputs) return null;

      // Deduplicate external inputs/constants by rk; versioned SSA pills by rk@ver
      const nid = (isConst || kind === "input" || kind === "peer" || (isEntry && ver === 0))
        ? `res:${rk}`
        : `ssa:${rk}@v${ver}`;

      if (!pillCreated.has(nid)) {
        pillCreated.add(nid);
        const meta = isState ? manifest.states[name] : manifest.buffers[name];
        const byteVal = isState
          ? evalStateBytes(meta, manifest, state.vars)
          : evalBufferBytes(meta, state.vars);
        const shapeTxt = isState
          ? (meta?.bytes_per_token ? `${formatBytes(meta.bytes_per_token)}/tok` : meta?.bytes_per_seq ? `${formatBytes(meta.bytes_per_seq)}/seq` : formatBytes(meta?.bytes))
          : `[${(meta?.shape || []).map((d) => formatExpr(d)).join(", ")}]`;

        nodes.push({
          id: nid,
          nodeType: isState ? "state" : "buffer",
          isState,
          name,
          rk,
          ver,
          offset,
          kind,
          dtype,
          shapeTxt,
          bytesTxt: formatBytes(byteVal),
          fill: meta?.fill || null,
          isExternalIn: isEntry,
          isResidual: an.residualKeys.has(rk),
          w: 204,
          h: 44,
        });
      }
      return nid;
    }

    // Track last writer call inside range when showBuffers is false
    const directProducerNode = new Map();

    for (let ci = startCall; ci <= endCall; ci++) {
      const info = an.callInfos[ci];
      if (!info) continue;

      const weightPins = [];
      for (const r of info.reads) {
        if (r.isConst && !state.showWeights) {
          weightPins.push(r.name);
        }
      }

      const launches = info.opDef?.impl?.launches || [];
      const activeL = launches.filter((l) => isLaunchActive(l, state.vars)).length;
      const hasWhen = launches.some((l) => l.when);
      const hasPack = launches.some((l) => (l.args || []).some((a) => a && a.pack));
      const hasPdl = launches.some((l) => l.pdl);

      const callNodeId = `call:${ci}`;
      const extraPinRows = (weightPins.length > 0 || info.scalars.length > 0) ? 1 : 0;
      nodes.push({
        id: callNodeId,
        nodeType: "call",
        callIdx: ci,
        info,
        weightPins,
        launchesTotal: launches.length,
        launchesActive: activeL,
        hasWhen,
        hasPack,
        hasPdl,
        w: 248,
        h: extraPinRows ? 76 : 58,
      });

      // Wire Reads
      for (const r of info.reads) {
        if (r.isConst && !state.showWeights) continue;
        if (r.isState && !state.showStates) continue;
        if ((r.kind === "input" || r.kind === "peer") && !state.showInputs) continue;

        const isRes = an.residualKeys.has(r.rk);
        if (state.showBuffers || r.producerCall < startCall || r.kind === "input" || r.isConst || r.kind === "peer") {
          const tid = ensureTensorNode(r, r.producerCall < startCall);
          if (tid) {
            edges.push({
              src: tid,
              dst: callNodeId,
              rk: r.rk,
              label: r.offset ? `+${formatBytes(r.offset)}` : "",
              isResidual: isRes,
              isState: r.isState,
            });
          }
        } else {
          const srcNode = directProducerNode.get(`${r.rk}@v${r.ver}`);
          if (srcNode) {
            const bMeta = manifest.buffers[r.name];
            const dims = bMeta?.shape ? `[${bMeta.shape.map((d) => evalExpr(d, state.vars)).join("×")}]` : "";
            edges.push({
              src: srcNode,
              dst: callNodeId,
              rk: r.rk,
              label: `${trunc(r.name, 13)} ${r.dtype}${dims}`,
              isResidual: isRes,
              isState: r.isState,
            });
          }
        }
      }

      // Wire Writes
      for (const w of info.writes) {
        if (w.isState && !state.showStates) continue;
        const isRes = an.residualKeys.has(w.rk);

        // Check if this written version is read outside the range or is an output buffer
        const accesses = an.resourceAccesses.get(w.rk) || [];
        const escapesRange = w.kind === "output" || accesses.some((a) => a.dir === "read" && a.ver === w.ver && a.callIdx > endCall);

        if (state.showBuffers || escapesRange) {
          const tid = ensureTensorNode(w, false);
          if (tid) {
            edges.push({
              src: callNodeId,
              dst: tid,
              rk: w.rk,
              label: w.offset ? `+${formatBytes(w.offset)}` : "",
              isResidual: isRes,
              isState: w.isState,
            });
          }
        }
        directProducerNode.set(`${w.rk}@v${w.ver}`, callNodeId);
      }

      // Stream synchronization edges (`after` lanes)
      if (info.call.after && Array.isArray(info.call.after)) {
        for (const laneId of info.call.after) {
          for (let prev = ci - 1; prev >= startCall; prev--) {
            const prevLane = an.callInfos[prev]?.call.lane ?? 0;
            if (prevLane === laneId) {
              edges.push({
                src: `call:${prev}`,
                dst: callNodeId,
                rk: `lane:${laneId}`,
                label: `sync:L${laneId}`,
                isResidual: false,
                isState: false,
              });
              break;
            }
          }
        }
      }
    }

    return { nodes, edges, stageBoxes: [], stageObj };
  }

  function buildMacroGraph() {
    const an = state.analysis;
    const manifest = state.manifest;
    const stages = an.stages;
    const isLargeProg = an.calls.length > 20;

    // Partition stages into compact Macro Units so even a 1,200-call program fits on ONE screen:
    // 1. Explicitly expanded stage -> renders its internal calls inline.
    // 2. When `foldRepeated` is ON:
    //    a) Indexed stage families (`l0..l63`, `draft.0..draft.4`, `verify.l0..verify.l63`) fold into a single Family Super-Card
    //       (unless unfolded via `unfoldedFamilyKeys`), exposing clickable pattern chips (`l0 ×48`, `l3 ×16`) right on the card!
    //    b) If a family IS unfolded, consecutive identical-signature stages still fold into `×N` blocks.
    //    c) Consecutive single-call glue stages (>=2 in large programs) coalesce into a compact `Chain` card.
    const units = [];
    let i = 0;
    while (i < stages.length) {
      const st = stages[i];
      if (state.expandedStageIdxs.has(st.idx)) {
        units.push({ kind: "expanded", stages: [st], startCall: st.startCall, endCall: st.endCall });
        i++;
        continue;
      }

      if (state.foldRepeated) {
        const stem = stageStem(st.baseName, st.isGrouped);
        const famKey = `${stem}@${st.idx}`;
        if (stem && !state.unfoldedFamilyKeys.has(famKey)) {
          let j = i + 1;
          while (
            j < stages.length &&
            !state.expandedStageIdxs.has(stages[j].idx) &&
            stageStem(stages[j].baseName, stages[j].isGrouped) === stem
          ) {
            j++;
          }
          if (j - i >= 2) {
            units.push({
              kind: "fold",
              foldKind: "family",
              famKey,
              stages: stages.slice(i, j),
              startCall: stages[i].startCall,
              endCall: stages[j - 1].endCall,
            });
            i = j;
            continue;
          }
        }

        // Exact structural signature repeat (used when a family is unfolded or non-indexed stages repeat)
        if (st.callIndices.length >= 2) {
          let j = i + 1;
          while (
            j < stages.length &&
            !state.expandedStageIdxs.has(stages[j].idx) &&
            stages[j].sigId === st.sigId &&
            stages[j].callIndices.length === st.callIndices.length
          ) {
            j++;
          }
          if (j - i >= 2) {
            units.push({
              kind: "fold",
              foldKind: "exact",
              famKey: "",
              stages: stages.slice(i, j),
              startCall: stages[i].startCall,
              endCall: stages[j - 1].endCall,
            });
            i = j;
            continue;
          }
        }

        // Coalesce consecutive short non-family glue stages (<=2 calls) in large programs
        if (isLargeProg && st.callIndices.length <= 2 && !stem) {
          let j = i + 1;
          while (
            j < stages.length &&
            !state.expandedStageIdxs.has(stages[j].idx) &&
            stages[j].callIndices.length <= 2 &&
            !stageStem(stages[j].baseName, stages[j].isGrouped)
          ) {
            j++;
          }
          if (j - i >= 2) {
            units.push({
              kind: "fold",
              foldKind: "chain",
              famKey: `chain@${st.idx}`,
              stages: stages.slice(i, j),
              startCall: stages[i].startCall,
              endCall: stages[j - 1].endCall,
            });
            i = j;
            continue;
          }
        }
      }

      units.push({ kind: "stage", stages: [st], startCall: st.startCall, endCall: st.endCall });
      i++;
    }

    // Map every callIdx -> owner nodeId in the macro graph
    const callOwnerNode = new Array(an.calls.length);
    const unitPrimaryNodeId = new Array(units.length);
    const nodes = [];
    const edges = [];
    const pillCreated = new Set();

    units.forEach((u, uidx) => {
      if (u.kind === "expanded") {
        const st = u.stages[0];
        unitPrimaryNodeId[uidx] = `call:${st.startCall}`;
        for (let ci = st.startCall; ci <= st.endCall; ci++) {
          const info = an.callInfos[ci];
          const nid = `call:${ci}`;
          callOwnerNode[ci] = nid;
          const launches = info.opDef?.impl?.launches || [];
          const weightPins = info.reads.filter((r) => r.isConst).map((r) => r.name);
          nodes.push({
            id: nid,
            nodeType: "call",
            callIdx: ci,
            stageIdx: st.idx,
            inExpandedStage: st.idx,
            info,
            weightPins,
            launchesTotal: launches.length,
            launchesActive: launches.filter((l) => isLaunchActive(l, state.vars)).length,
            hasWhen: launches.some((l) => l.when),
            hasPack: launches.some((l) => (l.args || []).some((a) => a && a.pack)),
            hasPdl: launches.some((l) => l.pdl),
            w: 244,
            h: weightPins.length || info.scalars.length ? 72 : 54,
          });
        }
      } else if (u.kind === "fold") {
        const first = u.stages[0];
        const last = u.stages[u.stages.length - 1];
        const nid = `fold:${first.idx}_${last.idx}`;
        unitPrimaryNodeId[uidx] = nid;
        for (let ci = u.startCall; ci <= u.endCall; ci++) callOwnerNode[ci] = nid;

        // Summarize distinct structural patterns inside this fold group
        const patMap = new Map();
        u.stages.forEach((s) => {
          if (!patMap.has(s.sigId)) {
            patMap.set(s.sigId, { repStage: s, count: 0, color: s.sigColor });
          }
          patMap.get(s.sigId).count++;
        });
        const patterns = Array.from(patMap.values());
        const sampleOps = u.foldKind === "chain"
          ? u.stages.slice(0, 4).map((s) => s.displayName)
          : first.callIndices.slice(0, 4).map((ci) => an.callInfos[ci].opName);

        nodes.push({
          id: nid,
          nodeType: "fold",
          foldKind: u.foldKind,
          famKey: u.famKey,
          unitIdx: uidx,
          stages: u.stages,
          patterns,
          title: u.foldKind === "chain"
            ? `${first.displayName} → ${last.displayName}`
            : `${first.baseName} … ${last.baseName}`,
          repeatCount: u.stages.length,
          totalCalls: u.endCall - u.startCall + 1,
          sigColor: u.foldKind === "chain" ? "var(--cat-norm)" : first.sigColor,
          sampleOps,
          w: 284,
          h: patterns.length > 1 && u.foldKind !== "chain" ? 94 : 76,
        });
      } else {
        const st = u.stages[0];
        if (st.callIndices.length === 1) {
          const ci = st.startCall;
          const info = an.callInfos[ci];
          const nid = `call:${ci}`;
          unitPrimaryNodeId[uidx] = nid;
          callOwnerNode[ci] = nid;
          const launches = info.opDef?.impl?.launches || [];
          const weightPins = info.reads.filter((r) => r.isConst).map((r) => r.name);
          nodes.push({
            id: nid,
            nodeType: "call",
            callIdx: ci,
            stageIdx: st.idx,
            info,
            weightPins,
            launchesTotal: launches.length,
            launchesActive: launches.filter((l) => isLaunchActive(l, state.vars)).length,
            hasWhen: launches.some((l) => l.when),
            hasPack: launches.some((l) => (l.args || []).some((a) => a && a.pack)),
            hasPdl: launches.some((l) => l.pdl),
            w: 244,
            h: weightPins.length || info.scalars.length ? 74 : 56,
          });
        } else {
          const nid = `stage:${st.idx}`;
          unitPrimaryNodeId[uidx] = nid;
          for (let ci = st.startCall; ci <= st.endCall; ci++) callOwnerNode[ci] = nid;
          const sampleOps = st.callIndices.slice(0, 4).map((ci) => an.callInfos[ci].opName);
          nodes.push({
            id: nid,
            nodeType: "stage",
            stage: st,
            stageIdx: st.idx,
            sigColor: st.sigColor,
            sampleOps,
            w: 256,
            h: 72,
          });
        }
      }
    });

    // Deduplicate edges between macro owner nodes
    const edgeDedup = new Map();
    function addMacroEdge(src, dst, rk, label, isResidual, isState) {
      if (!src || !dst || src === dst) return;
      const key = `${src}->${dst}:${isResidual ? "R" : isState ? "S" : "D"}`;
      if (edgeDedup.has(key)) {
        const ex = edgeDedup.get(key);
        if (label && !ex.labels.includes(label) && ex.labels.length < 3) {
          ex.labels.push(label);
          ex.label = ex.labels.join(", ");
        }
        return;
      }
      const e = { src, dst, rk, label, labels: label ? [label] : [], isResidual, isState };
      edgeDedup.set(key, e);
      edges.push(e);
    }

    function ensureExternalPill(acc) {
      const { isState, name, rk, kind, isConst, dtype, ver } = acc;
      if (isConst && !state.showWeights) return null;
      if (isState && !state.showStates) return null;
      if ((kind === "input" || kind === "peer") && !state.showInputs) return null;
      const nid = (isConst || kind === "input" || kind === "peer" || kind === "carry" || isState)
        ? `res:${rk}`
        : `ssa:${rk}@v${ver}`;
      if (!pillCreated.has(nid)) {
        pillCreated.add(nid);
        const meta = isState ? manifest.states[name] : manifest.buffers[name];
        const byteVal = isState
          ? evalStateBytes(meta, manifest, state.vars)
          : evalBufferBytes(meta, state.vars);
        const shapeTxt = isState
          ? (meta?.bytes_per_token ? `${formatBytes(meta.bytes_per_token)}/tok` : meta?.bytes_per_seq ? `${formatBytes(meta.bytes_per_seq)}/seq` : formatBytes(meta?.bytes))
          : `[${(meta?.shape || []).map((d) => formatExpr(d)).join(", ")}]`;
        nodes.push({
          id: nid,
          nodeType: isState ? "state" : "buffer",
          isState,
          name,
          rk,
          ver,
          kind,
          dtype,
          shapeTxt,
          bytesTxt: formatBytes(byteVal),
          fill: meta?.fill || null,
          isResidual: an.residualKeys.has(rk),
          w: 198,
          h: 42,
        });
      }
      return nid;
    }

    // Collect unexpanded external inputs, external carry-ins, and state pool usage so Macro never fans out horizontally!
    const foldedAuxInputs = new Map();     // rk -> { acc, firstOwner }
    const foldedPrimaryInputs = new Map(); // rk -> { acc, firstOwner }
    const foldedCarryIns = new Map();      // rk -> { acc, firstOwner }
    const foldedStateUsage = new Map();    // stateRk -> { acc, ownerCounts: Map<ownerId, count> }
    let interUnitEdgeCount = 0;

    for (let ci = 0; ci < an.calls.length; ci++) {
      const info = an.callInfos[ci];
      const dstOwner = callOwnerNode[ci];
      const inExpanded = state.expandedStageIdxs.has(info.stageIdx);

      for (const r of info.reads) {
        if (r.isConst && !state.showWeights) continue;
        if (r.isState && !state.showStates) continue;
        if ((r.kind === "input" || r.kind === "peer") && !state.showInputs) continue;

        const isRes = an.residualKeys.has(r.rk);

        if (r.isState) {
          if (inExpanded) {
            const pid = ensureExternalPill(r);
            addMacroEdge(pid, dstOwner, r.rk, "", false, true);
          } else {
            if (!foldedStateUsage.has(r.rk)) {
              foldedStateUsage.set(r.rk, { acc: r, ownerCounts: new Map() });
            }
            const su = foldedStateUsage.get(r.rk);
            su.ownerCounts.set(dstOwner, (su.ownerCounts.get(dstOwner) || 0) + 1);
          }
          continue;
        }

        if (r.producerCall < 0 || r.kind === "input" || r.kind === "peer" || r.isConst) {
          if (inExpanded || !isLargeProg) {
            const pid = ensureExternalPill(r);
            addMacroEdge(pid, dstOwner, r.rk, "", isRes, false);
          } else if (r.kind === "input" || r.kind === "peer") {
            const fill = manifest.buffers[r.name]?.fill;
            if (fill === "token" || r.kind === "peer") {
              if (!foldedPrimaryInputs.has(r.rk)) foldedPrimaryInputs.set(r.rk, { acc: r, firstOwner: dstOwner });
            } else {
              if (!foldedAuxInputs.has(r.rk)) foldedAuxInputs.set(r.rk, { acc: r, firstOwner: dstOwner });
            }
          } else if (r.kind === "carry") {
            if (!foldedCarryIns.has(r.rk)) foldedCarryIns.set(r.rk, { acc: r, firstOwner: dstOwner });
          } else if (r.isConst && state.showWeights && foldedAuxInputs.size < 6) {
            const pid = ensureExternalPill(r);
            addMacroEdge(pid, dstOwner, r.rk, "", false, false);
          }
        } else {
          const srcOwner = callOwnerNode[r.producerCall];
          if (srcOwner === dstOwner) continue;
          interUnitEdgeCount++;

          if (inExpanded && state.showBuffers) {
            const pid = ensureExternalPill(r);
            addMacroEdge(srcOwner, pid, r.rk, "", isRes, false);
            addMacroEdge(pid, dstOwner, r.rk, "", isRes, false);
          } else {
            addMacroEdge(srcOwner, dstOwner, r.rk, r.name, isRes, false);
          }
        }
      }

      for (const w of info.writes) {
        if (w.isState && !inExpanded && state.showStates) {
          if (!foldedStateUsage.has(w.rk)) {
            foldedStateUsage.set(w.rk, { acc: w, ownerCounts: new Map() });
          }
          const su = foldedStateUsage.get(w.rk);
          su.ownerCounts.set(dstOwner, (su.ownerCounts.get(dstOwner) || 0) + 1);
          continue;
        }
        if (w.kind === "output") {
          const pid = ensureExternalPill(w);
          addMacroEdge(dstOwner, pid, w.rk, "", false, false);
        } else if (inExpanded && state.showBuffers && !w.isState) {
          const accesses = an.resourceAccesses.get(w.rk) || [];
          const internalConsumers = accesses.filter(
            (a) => a.dir === "read" && a.ver === w.ver && callOwnerNode[a.callIdx] !== dstOwner && an.callInfos[a.callIdx]?.stageIdx === info.stageIdx
          );
          if (internalConsumers.length > 0) {
            const pid = ensureExternalPill(w);
            addMacroEdge(dstOwner, pid, w.rk, "", an.residualKeys.has(w.rk), false);
            for (const c of internalConsumers) {
              addMacroEdge(pid, callOwnerNode[c.callIdx], w.rk, "", an.residualKeys.has(w.rk), false);
            }
          }
        }
      }
    }

    // Materialize clean boundary nodes for folded Macro mode
    for (const { acc, firstOwner } of foldedPrimaryInputs.values()) {
      const pid = ensureExternalPill(acc);
      addMacroEdge(pid, firstOwner, acc.rk, "", false, false);
    }

    if (foldedAuxInputs.size === 1) {
      const { acc, firstOwner } = foldedAuxInputs.values().next().value;
      const pid = ensureExternalPill(acc);
      addMacroEdge(pid, firstOwner, acc.rk, "", false, false);
    } else if (foldedAuxInputs.size > 1) {
      const list = Array.from(foldedAuxInputs.values());
      const gid = "grp:inputs";
      const sampleNames = list.slice(0, 3).map((x) => x.acc.name).join(", ");
      nodes.push({
        id: gid,
        nodeType: "group_pill",
        title: `Batch & Sampling Inputs (${list.length})`,
        subtitle: sampleNames + (list.length > 3 ? ` +${list.length - 3}` : ""),
        kind: "input",
        members: list.map((x) => x.acc),
        w: 224,
        h: 44,
      });
      addMacroEdge(gid, list[0].firstOwner, gid, "", false, false);
    }

    if (foldedCarryIns.size > 0 && foldedCarryIns.size <= 2) {
      for (const { acc, firstOwner } of foldedCarryIns.values()) {
        const pid = ensureExternalPill(acc);
        addMacroEdge(pid, firstOwner, acc.rk, "", false, false);
      }
    } else if (foldedCarryIns.size > 2) {
      const list = Array.from(foldedCarryIns.values());
      const gid = "grp:carry_in";
      const sampleNames = list.slice(0, 3).map((x) => x.acc.name).join(", ");
      nodes.push({
        id: gid,
        nodeType: "group_pill",
        title: `Step Carry-In (${list.length})`,
        subtitle: sampleNames + (list.length > 3 ? ` +${list.length - 3}` : ""),
        kind: "carry",
        members: list.map((x) => x.acc),
        w: 214,
        h: 44,
      });
      addMacroEdge(gid, list[0].firstOwner, gid, "", false, false);
    }

    // Wire each persistent State pool only to its top 1–2 primary Macro units
    for (const { acc, ownerCounts } of foldedStateUsage.values()) {
      const pid = ensureExternalPill(acc);
      if (!pid) continue;
      const topOwners = Array.from(ownerCounts.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 2);
      for (const [ownerId] of topOwners) {
        addMacroEdge(pid, ownerId, acc.rk, "", false, true);
      }
    }

    // If a program consists of independent derivations with zero inter-unit dataflow (e.g. `derive`), stack units vertically
    if (interUnitEdgeCount === 0 && units.length > 1) {
      for (let k = 0; k < units.length - 1; k++) {
        addMacroEdge(unitPrimaryNodeId[k], unitPrimaryNodeId[k + 1], "seq:order", "", false, false);
      }
    }

    return { nodes, edges, stageBoxes: [] };
  }

  // ==================== SVG RENDERING ====================
  const svgEl = document.getElementById("blueprint-svg");
  const viewportG = document.getElementById("viewport-g");
  const minimapSvg = document.getElementById("minimap-svg");
  const canvasWrap = document.getElementById("canvas-wrap");

  function renderGraph(autoFit) {
    const t0 = performance.now();
    const model = buildGraphModel();

    // Remove redundant direct edges inside expanded stages when buffer pill path exists
    const pillIntermediates = new Set();
    model.edges.forEach((e) => {
      if (e.dst.startsWith("ssa:")) pillIntermediates.add(`${e.src}|${e.rk}`);
    });
    const cleanEdges = model.edges.filter((e) => {
      if (!e.src.startsWith("ssa:") && !e.dst.startsWith("ssa:") && pillIntermediates.has(`${e.src}|${e.rk}`)) {
        return false;
      }
      return true;
    });

    const layout = computeSugiyamaLayout(model.nodes, cleanEdges, state.orient);
    state.layoutResult = layout;

    // Compute bounding boxes for expanded stages in Macro mode
    const stageBounds = new Map();
    if (state.scopeMode === "macro") {
      layout.nodes.forEach((n) => {
        if (n.inExpandedStage != null) {
          const b = stageBounds.get(n.inExpandedStage) || { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };
          b.minX = Math.min(b.minX, n.x - 18);
          b.minY = Math.min(b.minY, n.y - 34);
          b.maxX = Math.max(b.maxX, n.x + n.w + 18);
          b.maxY = Math.max(b.maxY, n.y + n.h + 16);
          stageBounds.set(n.inExpandedStage, b);
        }
      });
    }

    let html = "";

    // 1. Expanded Stage Container Boxes
    for (const [stIdx, b] of stageBounds.entries()) {
      const st = state.analysis.stages[stIdx];
      if (!st) continue;
      const bw = b.maxX - b.minX;
      const bh = b.maxY - b.minY;
      html += `<g class="g-stage-box" data-stage-idx="${stIdx}">
        <rect class="stage-bg" x="${b.minX.toFixed(1)}" y="${b.minY.toFixed(1)}" width="${bw.toFixed(1)}" height="${bh.toFixed(1)}" rx="12"/>
        <g class="stage-header-pill" data-collapse-stage="${stIdx}" transform="translate(${(b.minX + 12).toFixed(1)}, ${(b.minY + 8).toFixed(1)})">
          <rect width="188" height="20" rx="5" fill="${st.sigColor}" opacity="0.22"/>
          <text x="8" y="14" fill="var(--text-1)" font-size="11" font-weight="700" class="mono">▼ ${esc(st.displayName)} (${st.callIndices.length} calls · click to fold)</text>
        </g>
      </g>`;
    }

    // 2. Edges (Wires)
    layout.edges.forEach((e, ei) => {
      const cls = ["g-edge"];
      let marker = "url(#arrow-default)";
      if (e.isResidual) {
        cls.push("residual");
        marker = "url(#arrow-residual)";
      } else if (e.isState) {
        cls.push("state-edge");
        marker = "url(#arrow-state)";
      }
      html += `<g class="${cls.join(" ")}" data-edge-idx="${ei}" data-src="${esc(e.src)}" data-dst="${esc(e.dst)}">
        <path class="wire-hit" d="${e.path}"/>
        <path class="wire-line" d="${e.path}" marker-end="${marker}"/>`;
      if (e.label && e.midPoint) {
        const lw = Math.min(184, e.label.length * 6.1 + 14);
        html += `<g transform="translate(${(e.midPoint.x - lw / 2).toFixed(1)}, ${(e.midPoint.y - 9).toFixed(1)})">
          <rect width="${lw.toFixed(1)}" height="17" rx="4" fill="var(--bg-surface)" stroke="var(--border-subtle)"/>
          <text x="${(lw / 2).toFixed(1)}" y="12" text-anchor="middle" fill="var(--text-2)" font-size="9.6" class="mono">${esc(trunc(e.label, 27))}</text>
        </g>`;
      }
      html += `</g>`;
    });

    // 3. Nodes
    layout.nodes.forEach((n) => {
      if (n.nodeType === "call") {
        html += renderCallCardSvg(n);
      } else if (n.nodeType === "buffer" || n.nodeType === "state") {
        html += renderTensorPillSvg(n);
      } else if (n.nodeType === "stage") {
        html += renderStageCardSvg(n);
      } else if (n.nodeType === "fold") {
        html += renderFoldCardSvg(n);
      } else if (n.nodeType === "group_pill") {
        html += renderGroupPillSvg(n);
      }
    });

    viewportG.innerHTML = html;

    // Update breadcrumb HUD
    renderBreadcrumb();

    // Telemetry
    const dt = (performance.now() - t0).toFixed(1);
    const activeLaunches = countActiveProgramLaunches();
    document.getElementById("graph-telemetry").textContent =
      `${layout.nodes.length} nodes · ${layout.edges.length} wires · ${activeLaunches} active GPU launches · layout ${dt}ms`;

    renderMinimap();
    if (autoFit) {
      fitGraphToScreen();
    } else {
      applyZoomTransform();
    }

    if (state.selectedNodeId) {
      applyHighlightCone(state.selectedNodeId);
    } else if (state.searchQuery) {
      applySearchHighlight(state.searchQuery);
    }
  }

  function renderCallCardSvg(n) {
    const info = n.info;
    const cat = info.cat;
    const isSel = state.selectedNodeId === n.id ? " selected" : "";
    const badges = [];
    if (n.launchesTotal > 1) badges.push(`${n.launchesActive}/${n.launchesTotal}L`);
    if (n.hasWhen) badges.push("when");
    if (n.hasPack) badges.push("TMA/pack");
    if (n.hasPdl) badges.push("PDL");
    if (info.call.lane != null) badges.push(`lane:${info.call.lane}`);

    let pinSummary = "";
    if (n.weightPins.length) {
      pinSummary += `W: ${trunc(n.weightPins[0], 20)}${n.weightPins.length > 1 ? ` +${n.weightPins.length - 1}` : ""}`;
    }
    if (info.scalars.length) {
      const sTxt = info.scalars.slice(0, 2).map((s) => {
        const a = s.arg;
        if (a.var) return `${a.var}=${state.vars[a.var] ?? "?"}`;
        if (a.expr) return formatExpr(a.expr);
        if (a.rank) return `rank(${a.rank})`;
        const k = Object.keys(a)[0];
        return `${a[k]}`;
      }).join(", ");
      pinSummary += (pinSummary ? " · " : "") + sTxt;
    }

    return `<g class="g-node node-call${isSel}" data-node-id="${esc(n.id)}" data-call-idx="${n.callIdx}" transform="translate(${n.x.toFixed(1)}, ${n.y.toFixed(1)})">
      <rect class="card-body" width="${n.w}" height="${n.h}" rx="8"/>
      <path d="M 0 8 A 8 8 0 0 1 8 0 L ${n.w - 8} 0 A 8 8 0 0 1 ${n.w} 8 L ${n.w} 23 L 0 23 Z" fill="var(--bg-node-header)"/>
      <rect x="0" y="0" width="4.5" height="${n.h}" rx="2" fill="${cat.color}"/>
      <text x="11" y="15.5" fill="var(--text-1)" font-size="11.8" font-weight="700" class="mono">${esc(trunc(info.label, 24))}</text>
      <text x="${n.w - 8}" y="15" text-anchor="end" fill="var(--text-3)" font-size="10" font-weight="700" class="mono">#${n.callIdx}</text>
      <text x="11" y="41" fill="var(--text-2)" font-size="11" class="mono">${esc(trunc(info.opName, 22))}</text>
      ${badges.length ? `<g transform="translate(${n.w - 8}, 30)">
        <text text-anchor="end" y="11" fill="${cat.color}" font-size="9.8" font-weight="700" class="mono">${esc(badges.join("·"))}</text>
      </g>` : ""}
      ${pinSummary ? `<line x1="8" y1="50" x2="${n.w - 8}" y2="50" stroke="var(--border-subtle)"/>
      <text x="11" y="65" fill="var(--text-3)" font-size="10.2" class="mono">${esc(trunc(pinSummary, 34))}</text>` : ""}
    </g>`;
  }

  function renderTensorPillSvg(n) {
    const col = kindColor(n.kind);
    const isSel = state.selectedNodeId === n.id ? " selected" : "";
    const kindBadge = n.fill ? `${n.kind}:${n.fill}` : n.kind;
    return `<g class="g-node node-tensor${isSel}" data-node-id="${esc(n.id)}" data-rk="${esc(n.rk)}" transform="translate(${n.x.toFixed(1)}, ${n.y.toFixed(1)})">
      <rect class="card-body" width="${n.w}" height="${n.h}" rx="21" style="fill:var(--bg-pill)"/>
      <circle cx="15" cy="${n.h / 2}" r="4.5" fill="${col}"/>
      <text x="26" y="17.5" fill="var(--text-1)" font-size="11.2" font-weight="700" class="mono">${esc(trunc(n.name, 20))}${n.ver > 0 && n.kind !== "input" && n.kind !== "weight" ? ` v${n.ver}` : ""}</text>
      <text x="${n.w - 12}" y="17" text-anchor="end" fill="${col}" font-size="9.6" font-weight="700" class="mono">${esc(trunc(kindBadge.toUpperCase(), 11))}</text>
      <text x="26" y="33" fill="var(--text-2)" font-size="10.2" class="mono">${esc(n.dtype)} ${esc(trunc(n.shapeTxt, 16))} · ${esc(n.bytesTxt)}</text>
    </g>`;
  }

  function renderGroupPillSvg(n) {
    const col = kindColor(n.kind);
    const isSel = state.selectedNodeId === n.id ? " selected" : "";
    return `<g class="g-node node-tensor${isSel}" data-node-id="${esc(n.id)}" data-group-node="${esc(n.id)}" transform="translate(${n.x.toFixed(1)}, ${n.y.toFixed(1)})">
      <rect x="3" y="3" width="${n.w}" height="${n.h - 2}" rx="12" fill="var(--bg-pill)" stroke="var(--border-node)" opacity="0.72"/>
      <rect class="card-body" width="${n.w}" height="${n.h - 2}" rx="12" style="fill:var(--bg-pill)"/>
      <circle cx="15" cy="${(n.h - 2) / 2}" r="4.5" fill="${col}"/>
      <text x="26" y="17" fill="var(--text-1)" font-size="11.2" font-weight="700" class="mono">${esc(trunc(n.title, 26))}</text>
      <text x="26" y="32" fill="var(--text-2)" font-size="10" class="mono">${esc(trunc(n.subtitle, 29))}</text>
    </g>`;
  }

  function renderStageCardSvg(n) {
    const st = n.stage;
    const isSel = state.selectedNodeId === n.id ? " selected" : "";
    return `<g class="g-node node-stage${isSel}" data-node-id="${esc(n.id)}" data-stage-idx="${st.idx}" transform="translate(${n.x.toFixed(1)}, ${n.y.toFixed(1)})">
      <rect class="card-body" width="${n.w}" height="${n.h}" rx="10"/>
      <rect x="0" y="0" width="5" height="${n.h}" rx="2.5" fill="${n.sigColor}"/>
      <text x="13" y="21" fill="var(--text-1)" font-size="12.8" font-weight="800" class="mono">${esc(trunc(st.displayName, 20))}</text>
      <g data-expand-stage="${st.idx}" transform="translate(${n.w - 68}, 7)">
        <rect width="60" height="19" rx="5" fill="var(--brand-soft)" stroke="var(--brand-bright)" stroke-width="0.9"/>
        <text x="30" y="13" text-anchor="middle" fill="var(--brand-bright)" font-size="10" font-weight="700">Inspect →</text>
      </g>
      <text x="13" y="40" fill="var(--text-2)" font-size="10.8" font-weight="500">${st.callIndices.length} calls · #${st.startCall}–#${st.endCall}</text>
      <text x="13" y="58" fill="var(--text-3)" font-size="10.2" class="mono">${esc(trunc(n.sampleOps.join(" → "), 33))}</text>
    </g>`;
  }

  function renderFoldCardSvg(n) {
    const firstIdx = n.stages[0].idx;
    const isSel = state.selectedNodeId === n.id ? " selected" : "";
    const badgeLabel = n.foldKind === "chain" ? `${n.repeatCount} OPS` : `×${n.repeatCount} STAGES`;

    // Render clickable pattern representative chips when multiple patterns exist inside a family (e.g. GDN x48 + Attn x16)
    let patRow = "";
    if (n.patterns && n.patterns.length > 1 && n.foldKind !== "chain") {
      let cx = 13;
      patRow = n.patterns.slice(0, 3).map((p) => {
        const txt = `${p.repStage.baseName} ×${p.count}`;
        const w = txt.length * 6.5 + 14;
        const g = `<g data-focus-stage="${p.repStage.idx}" transform="translate(${cx.toFixed(1)}, 63)">
          <rect width="${w.toFixed(1)}" height="18" rx="4" fill="var(--bg-surface-2)" stroke="${p.color}" stroke-width="1.15"/>
          <text x="${(w / 2).toFixed(1)}" y="12.5" text-anchor="middle" fill="var(--text-1)" font-size="9.8" font-weight="700" class="mono">${esc(txt)}</text>
        </g>`;
        cx += w + 6;
        return g;
      }).join("");
      if (n.famKey) {
        patRow += `<g data-unfold-fam="${esc(n.famKey)}" transform="translate(${(n.w - 58).toFixed(1)}, 63)">
          <rect width="48" height="18" rx="4" fill="var(--bg-surface-3)"/>
          <text x="24" y="12.5" text-anchor="middle" fill="var(--text-2)" font-size="9.6" font-weight="700">Unfold</text>
        </g>`;
      }
    }

    return `<g class="g-node node-fold${isSel}" data-node-id="${esc(n.id)}" data-fold-node="${esc(n.id)}" data-stage-idx="${firstIdx}" transform="translate(${n.x.toFixed(1)}, ${n.y.toFixed(1)})">
      <rect x="6" y="6" width="${n.w}" height="${n.h - 4}" rx="10" fill="var(--bg-node)" stroke="var(--border-node)" opacity="0.58"/>
      <rect x="3" y="3" width="${n.w}" height="${n.h - 4}" rx="10" fill="var(--bg-node)" stroke="var(--border-node)" opacity="0.82"/>
      <rect class="card-body" width="${n.w}" height="${n.h - 4}" rx="10"/>
      <rect x="0" y="0" width="5.5" height="${n.h - 4}" rx="2.5" fill="${n.sigColor}"/>
      <text x="13" y="21" fill="var(--text-1)" font-size="12.8" font-weight="800" class="mono">${esc(trunc(n.title, 21))}</text>
      <g data-focus-stage="${firstIdx}" transform="translate(${n.w - 78}, 7)">
        <rect width="70" height="19" rx="5" fill="${n.sigColor}" fill-opacity="0.15" stroke="${n.sigColor}" stroke-width="0.9"/>
        <text x="35" y="13" text-anchor="middle" fill="var(--text-1)" font-size="10" font-weight="800" class="mono">${esc(badgeLabel)}</text>
      </g>
      <text x="13" y="40" fill="var(--text-2)" font-size="10.8" font-weight="500">${n.totalCalls} calls (#${n.stages[0].startCall}–#${n.stages[n.stages.length - 1].endCall}) · dblclick to drill in</text>
      <text x="13" y="56" fill="var(--text-3)" font-size="10" class="mono">${esc(trunc(n.sampleOps.join(" → "), 36))}</text>
      ${patRow}
    </g>`;
  }

  function renderBreadcrumb() {
    const bc = document.getElementById("stage-breadcrumb");
    const an = state.analysis;
    if (!an || state.scopeMode !== "stage") {
      bc.classList.add("hidden");
      return;
    }
    bc.classList.remove("hidden");
    const st = an.stages[state.focusedStageIdx] || an.stages[0];
    bc.innerHTML = `
      <button class="crumb-btn" data-bc-action="macro">← Macro</button>
      <button class="crumb-btn" data-bc-action="prev" ${st.idx === 0 ? "disabled" : ""}>◀</button>
      <span class="mono" style="color:var(--brand-bright)">${esc(st.displayName)}</span>
      <span style="color:var(--text-3)">(${st.idx + 1}/${an.stages.length} · ${st.callIndices.length} calls)</span>
      <button class="crumb-btn" data-bc-action="next" ${st.idx >= an.stages.length - 1 ? "disabled" : ""}>▶</button>
    `;
  }

  // ==================== PAN, ZOOM & MINIMAP ====================
  function applyZoomTransform() {
    const { k, x, y } = state.zoom;
    viewportG.setAttribute("transform", `translate(${x.toFixed(1)}, ${y.toFixed(1)}) scale(${k.toFixed(4)})`);
    document.getElementById("zoom-readout").textContent = `${Math.round(k * 100)}%`;
    updateMinimapViewport();
  }

  function fitGraphToScreen() {
    const lay = state.layoutResult;
    if (!lay) return;
    const rect = canvasWrap.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const padX = 52;
    const padY = 74;
    const availW = Math.max(200, rect.width - padX * 2);
    const availH = Math.max(200, rect.height - padY * 2);
    const fitW = availW / lay.width;
    const fitH = availH / lay.height;
    let k = Math.min(fitW, fitH, 1.08);
    k = Math.max(0.14, k);

    // Keep vertical blueprints near crisp 1:1 readable scale (~88%-100%) and top-align when taller than viewport
    if (state.orient === "TB" && k < 0.88 && fitW >= 0.80) {
      k = Math.min(1.0, Math.max(0.88, Math.min(fitW * 0.94, availW / Math.max(lay.width, 560))));
      state.zoom = {
        k,
        x: (rect.width - lay.width * k) / 2,
        y: 58,
      };
    } else if (state.orient === "TB" && lay.height > lay.width * 2.2 && k < 0.68) {
      k = Math.min(0.95, Math.max(0.68, fitW * 0.92));
      state.zoom = {
        k,
        x: (rect.width - lay.width * k) / 2,
        y: 58,
      };
    } else {
      state.zoom = {
        k,
        x: (rect.width - lay.width * k) / 2,
        y: (rect.height - lay.height * k) / 2 + 12,
      };
    }
    applyZoomTransform();
  }

  function renderMinimap() {
    const lay = state.layoutResult;
    if (!lay) return;
    minimapSvg.setAttribute("viewBox", `0 0 ${lay.width} ${lay.height}`);
    let h = "";
    lay.edges.forEach((e) => {
      h += `<path d="${e.path}" fill="none" stroke="${e.isResidual ? "var(--edge-residual)" : "var(--edge-default)"}" stroke-width="4" opacity="0.45"/>`;
    });
    lay.nodes.forEach((n) => {
      const col = n.nodeType === "call"
        ? n.info.cat.color
        : n.nodeType === "stage" || n.nodeType === "fold"
          ? n.sigColor
          : kindColor(n.kind);
      h += `<rect x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" rx="6" fill="${col}" opacity="0.85"/>`;
    });
    h += `<rect id="minimap-vp" x="0" y="0" width="100" height="100" fill="var(--brand-soft)" stroke="var(--brand-bright)" stroke-width="${Math.max(4, lay.width / 45)}"/>`;
    minimapSvg.innerHTML = h;
    updateMinimapViewport();
  }

  function updateMinimapViewport() {
    const vp = document.getElementById("minimap-vp");
    const lay = state.layoutResult;
    if (!vp || !lay) return;
    const rect = canvasWrap.getBoundingClientRect();
    const { k, x, y } = state.zoom;
    const vx = -x / k;
    const vy = -y / k;
    const vw = rect.width / k;
    const vh = rect.height / k;
    vp.setAttribute("x", vx.toFixed(1));
    vp.setAttribute("y", vy.toFixed(1));
    vp.setAttribute("width", Math.max(40, vw).toFixed(1));
    vp.setAttribute("height", Math.max(40, vh).toFixed(1));
  }

  // ==================== CONNECTIVITY CONE & SEARCH HIGHLIGHT ====================
  function clearHighlights() {
    svgEl.classList.remove("dimmed");
    viewportG.querySelectorAll(".hl-on, .hl-up, .hl-down, .hl-cur, .selected").forEach((el) => {
      el.classList.remove("hl-on", "hl-up", "hl-down", "hl-cur", "selected");
    });
  }

  function applyHighlightCone(nodeId) {
    clearHighlights();
    const lay = state.layoutResult;
    if (!lay || !nodeId) return;
    const nodeEls = new Map();
    viewportG.querySelectorAll(".g-node").forEach((el) => nodeEls.set(el.dataset.nodeId, el));
    const targetEl = nodeEls.get(nodeId);
    if (!targetEl) return;

    targetEl.classList.add("selected", "hl-on", "hl-cur");
    svgEl.classList.add("dimmed");

    // Keep expanded stage boxes visible
    viewportG.querySelectorAll(".g-stage-box").forEach((el) => el.classList.add("hl-on"));

    const mode = state.traceMode;
    const upAdj = new Map();
    const downAdj = new Map();
    lay.edges.forEach((e) => {
      if (!downAdj.has(e.src)) downAdj.set(e.src, []);
      downAdj.get(e.src).push(e.dst);
      if (!upAdj.has(e.dst)) upAdj.set(e.dst, []);
      upAdj.get(e.dst).push(e.src);
    });

    function bfs(start, adj) {
      const seen = new Set();
      const q = (adj.get(start) || []).slice();
      while (q.length) {
        const cur = q.shift();
        if (seen.has(cur)) continue;
        seen.add(cur);
        for (const nxt of adj.get(cur) || []) q.push(nxt);
      }
      return seen;
    }

    const wantUp = mode === "both" || mode === "up";
    const wantDown = mode === "both" || mode === "down";
    const upSet = wantUp ? bfs(nodeId, upAdj) : new Set();
    const downSet = wantDown ? bfs(nodeId, downAdj) : new Set();

    upSet.forEach((id) => nodeEls.get(id)?.classList.add("hl-on", "hl-up"));
    downSet.forEach((id) => nodeEls.get(id)?.classList.add("hl-on", "hl-down"));

    viewportG.querySelectorAll(".g-edge").forEach((el) => {
      const s = el.dataset.src;
      const d = el.dataset.dst;
      const pLine = el.querySelector("path.wire-line");
      if (wantUp && (d === nodeId || upSet.has(d)) && (s === nodeId || upSet.has(s))) {
        el.classList.add("hl-on", "hl-up");
        pLine?.setAttribute("marker-end", "url(#arrow-up)");
      } else if (wantDown && (s === nodeId || downSet.has(s)) && (d === nodeId || downSet.has(d))) {
        el.classList.add("hl-on", "hl-down");
        pLine?.setAttribute("marker-end", "url(#arrow-down)");
      }
    });
  }

  function applySearchHighlight(query) {
    clearHighlights();
    const q = String(query || "").trim().toLowerCase();
    if (!q) return;
    let hits = 0;
    viewportG.querySelectorAll(".g-node").forEach((el) => {
      const txt = (el.textContent || "").toLowerCase() + " " + (el.dataset.nodeId || "").toLowerCase();
      if (txt.includes(q)) {
        el.classList.add("hl-on", "hl-cur");
        hits++;
      }
    });
    if (hits > 0) svgEl.classList.add("dimmed");
  }

  // ==================== RIGHT INSPECTOR DRAWER ====================
  const inspectorEl = document.getElementById("inspector-content");

  function renderInspector() {
    const sel = state.selected;
    const manifest = state.manifest;
    const an = state.analysis;
    if (!manifest || !an) return;

    if (!sel) {
      renderProgramOverviewInspector();
      return;
    }
    if (sel.type === "call") {
      renderCallInspector(an.callInfos[sel.callIdx]);
      return;
    }
    if (sel.type === "resource") {
      renderResourceInspector(sel.rk);
      return;
    }
    if (sel.type === "stage") {
      renderStageInspector(an.stages[sel.stageIdx]);
      return;
    }
    if (sel.type === "fold") {
      renderFoldInspector(sel.nodeId);
      return;
    }
    if (sel.type === "group_pill") {
      renderGroupInspector(sel.nodeId);
      return;
    }
  }

  function renderGroupInspector(nodeId) {
    const grp = state.layoutResult?.nodes.find((n) => n.id === nodeId);
    if (!grp) { renderProgramOverviewInspector(); return; }
    const m = state.manifest;
    const members = grp.members || [];
    inspectorEl.innerHTML = `
      <div class="insp-hero">
        <div class="insp-kicker"><span class="dot" style="background:${kindColor(grp.kind)}"></span><span>GROUPED BOUNDARY RESOURCES</span></div>
        <h2 class="insp-title">${esc(grp.title)}</h2>
        <div class="insp-sub">
          <span class="tiny-tag">${members.length} ${esc(grp.kind)} buffer(s)</span>
          <span class="tiny-tag">Coalesced in Macro View</span>
        </div>
      </div>
      <div class="insp-card">
        <div class="insp-card-title"><span>Member Buffers (${members.length})</span></div>
        <div class="port-list" style="max-height:380px;overflow-y:auto">
          ${members.map((acc) => {
            const meta = m.buffers[acc.name];
            const bytes = evalBufferBytes(meta, state.vars);
            return `<div class="port-row clickable" data-jump-rk="${esc(acc.rk)}">
              <span class="mono" style="font-weight:700;flex:1">${esc(acc.name)}</span>
              ${meta?.fill ? `<span class="tiny-tag" style="color:var(--kind-input)">fill:${esc(meta.fill)}</span>` : ""}
              <span class="mono" style="color:var(--text-2);font-size:10.5px">${esc(acc.dtype)} · ${formatBytes(bytes)}</span>
            </div>`;
          }).join("")}
        </div>
      </div>
    `;
    bindInspectorJumpLinks();
  }

  function renderFoldInspector(nodeId) {
    const foldNode = state.layoutResult?.nodes.find((n) => n.id === nodeId);
    if (!foldNode) { renderProgramOverviewInspector(); return; }
    const pats = foldNode.patterns || [];

    inspectorEl.innerHTML = `
      <div class="insp-hero">
        <div class="insp-kicker"><span class="stage-sig-dot" style="background:${foldNode.sigColor}"></span><span>FOLDED STAGE GROUP</span></div>
        <h2 class="insp-title">${esc(foldNode.title)}</h2>
        <div class="insp-sub">
          <span class="tiny-tag">${foldNode.repeatCount} stages</span>
          <span class="tiny-tag">${foldNode.totalCalls} calls</span>
          <span class="tiny-tag">${pats.length} structural pattern(s)</span>
        </div>
        <div style="display:flex;gap:8px;margin-top:6px">
          <button class="btn-mini" id="insp-drill-first">Inspect ${esc(foldNode.stages[0].displayName)} Blueprint →</button>
          ${foldNode.famKey ? `<button class="btn-ghost" id="insp-unfold-fam">Unfold Group in Macro</button>` : ""}
        </div>
      </div>

      ${pats.length > 0 && foldNode.foldKind !== "chain" ? `<div class="insp-card">
        <div class="insp-card-title"><span>Structural Block Patterns (${pats.length})</span></div>
        <div class="port-list">
          ${pats.map((p) => `<div class="port-row clickable" data-jump-stage="${p.repStage.idx}">
            <span class="stage-sig-dot" style="background:${p.color}"></span>
            <span class="mono" style="flex:1;font-weight:700">${esc(p.repStage.displayName)} template (${p.repStage.callIndices.length} calls)</span>
            <span class="tiny-tag">×${p.count} stages →</span>
          </div>`).join("")}
        </div>
      </div>` : ""}

      <div class="insp-card">
        <div class="insp-card-title"><span>Stages in Group (${foldNode.stages.length})</span></div>
        <div class="port-list" style="max-height:310px;overflow-y:auto">
          ${foldNode.stages.map((s) => `<div class="port-row clickable" data-jump-stage="${s.idx}">
            <span class="stage-sig-dot" style="background:${s.sigColor}"></span>
            <span class="mono" style="flex:1;font-weight:600">${esc(s.displayName)}</span>
            <span class="mono" style="color:var(--text-3);font-size:11px">${s.callIndices.length} calls (#${s.startCall}–#${s.endCall})</span>
          </div>`).join("")}
        </div>
      </div>
    `;

    document.getElementById("insp-drill-first")?.addEventListener("click", () => {
      if (foldNode.foldKind === "chain") {
        state.sliceStart = foldNode.stages[0].startCall;
        state.sliceEnd = foldNode.stages[foldNode.stages.length - 1].endCall;
        document.getElementById("slice-start").value = String(state.sliceStart);
        document.getElementById("slice-end").value = String(state.sliceEnd);
        setScopeMode("slice");
      } else {
        setScopeMode("stage", foldNode.stages[0].idx);
      }
    });
    document.getElementById("insp-unfold-fam")?.addEventListener("click", () => {
      if (foldNode.famKey === `chain@${foldNode.stages[0].idx}`) {
        foldNode.stages.forEach((s) => state.expandedStageIdxs.add(s.idx));
      } else {
        state.unfoldedFamilyKeys.add(foldNode.famKey);
      }
      renderGraph(true);
    });
    inspectorEl.querySelectorAll("[data-jump-stage]").forEach((el) => {
      el.addEventListener("click", () => {
        const stIdx = Number(el.dataset.jumpStage);
        setScopeMode("stage", stIdx);
      });
    });
  }

  function renderProgramOverviewInspector() {
    const m = state.manifest;
    const an = state.analysis;
    const prog = an.prog;
    const batchStr = prog.batch
      ? `groups: ${prog.batch.groups}, rows: ${formatExpr(prog.batch.rows)}${prog.batch.context ? `, ctx: ${prog.batch.context}` : ""}`
      : "harness / standalone";

    inspectorEl.innerHTML = `
      <div class="insp-hero">
        <div class="insp-kicker"><span>ARTIFACT &amp; PROGRAM SUMMARY</span></div>
        <h2 class="insp-title">${esc(m.model)} · ${esc(an.programName)}</h2>
        <div class="insp-sub">
          <span class="tiny-tag">schema v${m.schema_version}</span>
          ${prog.graph ? `<span class="tiny-tag cuda-graph">CUDA Graph</span>` : ""}
          ${prog.once ? `<span class="tiny-tag once">Run Once</span>` : ""}
          ${prog.derive ? `<span class="tiny-tag once">Derive</span>` : ""}
          ${m.chat?.tool_parser ? `<span class="tiny-tag">parser: ${esc(m.chat.tool_parser)}</span>` : ""}
        </div>
      </div>

      <div class="insp-card">
        <div class="insp-card-title"><span>Calling Contract</span></div>
        <div class="port-list">
          <div class="port-row"><span style="color:var(--text-3)">Batch Shape</span><span class="mono">${esc(batchStr)}</span></div>
          <div class="port-row"><span style="color:var(--text-3)">Total Calls</span><span class="mono">${an.calls.length} calls across ${an.stages.length} stages</span></div>
          <div class="port-row"><span style="color:var(--text-3)">Residual Spine</span><span class="mono">${esc(Array.from(an.residualKeys).map((k) => k.slice(4)).join(", ") || "none")}</span></div>
          ${m.topology ? `<div class="port-row"><span style="color:var(--text-3)">SPMD Groups</span><span class="mono">${esc(JSON.stringify(m.topology.groups))}</span></div>` : ""}
          ${m.cut ? `<div class="port-row"><span style="color:var(--text-3)">Pipeline Cut</span><span class="mono">stage ${m.cut.stage}/${m.cut.stages} (${trunc(m.cut.id, 10)})</span></div>` : ""}
        </div>
      </div>

      <div class="insp-card">
        <div class="insp-card-title"><span>Quick Interaction Guide</span></div>
        <div style="font-size:12px;color:var(--text-2);line-height:1.55">
          • <b>Click any Op Card</b> to inspect its typed interface, internal <code>impl</code> micro-graph, <code>when</code> guards, cubin SHA-256, and TMA/struct <code>pack</code> byte-map.<br>
          • <b>Click ⊕ Expand</b> on any Stage block (or double-click a Stage) to open its full SSA tensor-wire blueprint.<br>
          • <b>Drag the topbar <code>vars</code> sliders</b> to live-evaluate symbolic shapes, VRAM sizes, launch grids, and conditional kernels.
        </div>
      </div>
    `;
  }

  function renderStageInspector(st) {
    if (!st) return;
    const an = state.analysis;
    const opCounts = {};
    st.callIndices.forEach((ci) => {
      const op = an.callInfos[ci].opName;
      opCounts[op] = (opCounts[op] || 0) + 1;
    });

    const isExp = state.expandedStageIdxs.has(st.idx);
    inspectorEl.innerHTML = `
      <div class="insp-hero">
        <div class="insp-kicker"><span class="stage-sig-dot" style="background:${st.sigColor}"></span><span>STAGE BLOCK</span></div>
        <h2 class="insp-title">${esc(st.displayName)}</h2>
        <div class="insp-sub">
          <span class="tiny-tag">Calls #${st.startCall} – #${st.endCall} (${st.callIndices.length})</span>
          <span class="tiny-tag">Pattern Family #${st.sigId + 1}</span>
        </div>
        <div style="display:flex;gap:8px;margin-top:6px">
          <button class="btn-mini" id="insp-focus-stage">Open Stage Blueprint →</button>
          <button class="btn-ghost" id="insp-toggle-inline">${isExp ? "Collapse Inline" : "⊕ Expand in Macro"}</button>
        </div>
      </div>

      <div class="insp-card">
        <div class="insp-card-title"><span>Calls in Stage (${st.callIndices.length})</span></div>
        <div class="port-list">
          ${st.callIndices.map((ci) => {
            const inf = an.callInfos[ci];
            return `<div class="port-row clickable" data-jump-call="${ci}">
              <span class="mono" style="color:var(--text-3)">#${ci}</span>
              <span class="mono" style="flex:1;font-weight:600">${esc(inf.label)}</span>
              <span class="tiny-tag" style="border-color:${inf.cat.color}">${esc(trunc(inf.opName, 18))}</span>
            </div>`;
          }).join("")}
        </div>
      </div>
    `;

    document.getElementById("insp-focus-stage")?.addEventListener("click", () => {
      setScopeMode("stage", st.idx);
    });
    document.getElementById("insp-toggle-inline")?.addEventListener("click", () => {
      if (state.expandedStageIdxs.has(st.idx)) state.expandedStageIdxs.delete(st.idx);
      else state.expandedStageIdxs.add(st.idx);
      renderSidebarStages();
      renderGraph(false);
      renderStageInspector(st);
    });
    bindInspectorJumpLinks();
  }

  function renderCallInspector(info) {
    if (!info) return;
    const m = state.manifest;
    const op = info.opDef;
    const pSpecs = (op.params || []).map(parseParamSpec);
    const args = info.call.args || [];

    // Build interface argument rows
    const argRows = args.map((arg, ai) => {
      const spec = pSpecs[ai] || { dir: "scalar", dtype: "?" };
      const dirCls = spec.dir === "in" ? "dir-in" : spec.dir === "out" ? "dir-out" : spec.dir === "inout" ? "dir-inout" : "dir-scalar";
      if (arg.buf != null || arg.state != null) {
        const isState = arg.state != null;
        const name = isState ? arg.state : arg.buf;
        const rk = (isState ? "state:" : "buf:") + name;
        const meta = isState ? m.states[name] : m.buffers[name];
        const bytes = isState ? evalStateBytes(meta, m, state.vars) : evalBufferBytes(meta, state.vars);
        const offBadge = arg.offset ? ` <span class="tiny-tag" style="color:var(--kind-state)">+${formatBytes(arg.offset)}</span>` : "";
        return `<div class="port-row clickable" data-jump-rk="${esc(rk)}">
          <span class="dir-badge ${dirCls}">P${ai} ${esc(spec.dir)}</span>
          <span class="mono" style="flex:1;font-weight:600;overflow:hidden;text-overflow:ellipsis">${esc(name)}${offBadge}</span>
          <span class="mono" style="color:var(--text-2);font-size:10.5px">${esc(isState ? "state" : meta?.dtype || spec.dtype)} · ${formatBytes(bytes)}</span>
        </div>`;
      }
      let valTxt = "";
      if (arg.var) valTxt = `var(${arg.var}) = ${state.vars[arg.var] ?? "?"}`;
      else if (arg.expr) valTxt = `${formatExpr(arg.expr)} = ${evalExpr(arg.expr, state.vars)}`;
      else if (arg.rank) valTxt = `rank(${arg.rank}) [group=${m.topology?.groups?.[arg.rank] ?? "?"}]`;
      else {
        const k = Object.keys(arg)[0];
        valTxt = `${k}: ${arg[k]}`;
      }
      return `<div class="port-row">
        <span class="dir-badge dir-scalar">P${ai} ${esc(spec.dtype)}</span>
        <span class="mono" style="color:var(--text-1)">${esc(valTxt)}</span>
      </div>`;
    }).join("");

    // Render Op `impl` Micro-Graph SVG!
    const microSvg = buildOpMicroGraphSvg(info.opName, op);

    // Launches details + ABI `pack` byte-maps
    const launches = op.impl?.launches || [];
    const launchCards = launches.map((l, li) => {
      const active = isLaunchActive(l, state.vars);
      const isExtern = String(l.entry || "").startsWith("extern:");
      const modMeta = l.module ? m.modules?.[l.module] : null;
      const gridStr = l.grid ? `[${l.grid.map((g) => `${formatExpr(g)}=${evalExpr(g, state.vars)}`).join(", ")}]` : "—";
      const blkStr = l.block ? `[${l.block.join(", ")}]` : "—";
      const whenStr = l.when ? `when ${l.when.var} ∈ [${l.when.min ?? 1}, ${l.when.max ?? "∞"}]` : null;

      // Detect `pack` struct arguments
      const packHtml = (l.args || []).map((la, lai) => {
        if (!la || !la.pack) return "";
        return renderPackByteMap(la.pack, lai);
      }).join("");

      return `<div class="launch-card ${active ? "active-launch" : "skipped-launch"}">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:6px">
          <span class="mono" style="font-weight:800;font-size:11.5px;color:var(--text-1)">L${li} · ${esc(l.entry)}</span>
          <span class="tiny-tag" style="color:${active ? "#10b981" : "var(--text-3)"}">${active ? "● FIRES" : "○ SKIPPED"}</span>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:5px;font-size:10.5px;color:var(--text-2)" class="mono">
          ${l.module ? `<span class="tiny-tag">mod: ${esc(l.module)}</span>` : `<span class="tiny-tag">builtin extern</span>`}
          ${whenStr ? `<span class="tiny-tag" style="color:var(--kind-state)">${esc(whenStr)}</span>` : ""}
          ${l.pdl ? `<span class="tiny-tag" style="color:#38bdf8">PDL</span>` : ""}
          ${l.cluster ? `<span class="tiny-tag">cluster [${l.cluster.join("×")}]</span>` : ""}
          ${l.shared_mem ? `<span class="tiny-tag">smem ${formatBytes(evalExpr(l.shared_mem, state.vars))}</span>` : ""}
        </div>
        ${!isExtern ? `<div class="mono" style="font-size:10.5px;color:var(--text-2)">grid ${esc(gridStr)} × block ${esc(blkStr)}</div>` : ""}
        ${modMeta ? `<div class="sha-box" title="Pinned Module Artifact">${esc(modMeta.source)}<br><span style="color:var(--text-3)">sha256: ${esc(modMeta.sha256.slice(0, 24))}…</span></div>` : ""}
        ${packHtml}
      </div>`;
    }).join("");

    inspectorEl.innerHTML = `
      <div class="insp-hero">
        <div class="insp-kicker">
          <span class="stage-sig-dot" style="background:${info.cat.color}"></span>
          <span>CALL #${info.idx} · ${esc(info.cat.tag)}</span>
        </div>
        <h2 class="insp-title">${esc(info.label)}</h2>
        <div class="insp-sub">
          <span class="tiny-tag">op: <strong class="mono">${esc(info.opName)}</strong></span>
          <span class="tiny-tag">stage: ${esc(state.analysis.stages[info.stageIdx]?.displayName || "—")}</span>
          ${info.call.lane != null ? `<span class="tiny-tag cuda-graph">lane: ${esc(info.call.lane)}</span>` : ""}
          ${info.call.after ? `<span class="tiny-tag once">after: ${esc(info.call.after.join(","))}</span>` : ""}
        </div>
      </div>

      <div class="insp-card">
        <div class="insp-card-title"><span>Op Impl Micro-Graph</span><span>${launches.length} launch(es)</span></div>
        <div class="micro-svg-wrap">${microSvg}</div>
      </div>

      <div class="insp-card">
        <div class="insp-card-title"><span>Call Interface Bindings (${args.length})</span></div>
        <div class="port-list">${argRows}</div>
      </div>

      <div class="insp-card">
        <div class="insp-card-title"><span>Kernel Launches &amp; ABI</span></div>
        <div class="port-list">${launchCards}</div>
      </div>
    `;

    bindInspectorJumpLinks();
  }

  function buildOpMicroGraphSvg(opName, opDef) {
    const pSpecs = (opDef.params || []).map(parseParamSpec);
    const scratchEntries = Object.entries(opDef.impl?.scratch || {});
    const launches = opDef.impl?.launches || [];

    const inParams = [];
    const outParams = [];
    pSpecs.forEach((p, idx) => {
      if (p.kind === "scalar") return;
      if (p.dir === "in" || p.dir === "inout") inParams.push({ idx, p });
      if (p.dir === "out" || p.dir === "inout") outParams.push({ idx, p });
    });

    const rows = Math.max(inParams.length, launches.length + scratchEntries.length, outParams.length, 1);
    const W = 336;
    const H = Math.max(96, rows * 30 + 24);

    const pInPos = new Map();
    inParams.forEach((item, i) => {
      pInPos.set(item.idx, { x: 54, y: 20 + i * 26 });
    });
    const pOutPos = new Map();
    outParams.forEach((item, i) => {
      pOutPos.set(item.idx, { x: W - 54, y: 20 + i * 26 });
    });

    const launchPos = launches.map((l, i) => ({
      x: W / 2,
      y: 22 + i * 34,
      active: isLaunchActive(l, state.vars),
      l,
    }));
    const scratchPos = new Map();
    scratchEntries.forEach(([sname, smeta], i) => {
      scratchPos.set(sname, {
        x: W / 2,
        y: 24 + launches.length * 34 + i * 26,
        smeta,
      });
    });

    let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" style="display:block">`;

    // Draw micro-wires from Launch args
    launches.forEach((l, li) => {
      const lp = launchPos[li];
      const lSpecs = (l.params || opDef.params || []).map(parseParamSpec);
      const lArgs = l.args || (opDef.params || []).map((_, pi) => ({ param: pi }));
      const strokeOp = lp.active ? "0.75" : "0.22";

      lArgs.forEach((a, ai) => {
        const dir = lSpecs[ai]?.dir || "in";
        let refParam = a?.param;
        if (a?.pack) {
          for (const f of a.pack.fields || []) {
            if (f.param != null && pInPos.has(f.param)) {
              const sp = pInPos.get(f.param);
              svg += `<path d="M ${sp.x + 22} ${sp.y} C ${sp.x + 55} ${sp.y}, ${lp.x - 75} ${lp.y}, ${lp.x - 58} ${lp.y}" fill="none" stroke="var(--kind-input)" stroke-width="1.2" opacity="${strokeOp}"/>`;
            }
            if (f.tensormap && f.tensormap.param != null && pInPos.has(f.tensormap.param)) {
              const sp = pInPos.get(f.tensormap.param);
              svg += `<path d="M ${sp.x + 22} ${sp.y} C ${sp.x + 55} ${sp.y}, ${lp.x - 75} ${lp.y}, ${lp.x - 58} ${lp.y}" fill="none" stroke="#f472b6" stroke-width="1.5" opacity="${strokeOp}"/>`;
            }
            if (f.scratch && scratchPos.has(f.scratch)) {
              const sc = scratchPos.get(f.scratch);
              svg += `<path d="M ${sc.x - 45} ${sc.y} L ${lp.x - 45} ${lp.y + 8}" fill="none" stroke="var(--kind-carry)" stroke-dasharray="3 2" stroke-width="1.2" opacity="${strokeOp}"/>`;
            }
          }
        }
        if (refParam != null) {
          if ((dir === "in" || dir === "inout") && pInPos.has(refParam)) {
            const sp = pInPos.get(refParam);
            svg += `<path d="M ${sp.x + 22} ${sp.y} C ${sp.x + 52} ${sp.y}, ${lp.x - 75} ${lp.y}, ${lp.x - 58} ${lp.y}" fill="none" stroke="var(--kind-input)" stroke-width="1.2" opacity="${strokeOp}"/>`;
          }
          if ((dir === "out" || dir === "inout") && pOutPos.has(refParam)) {
            const dp = pOutPos.get(refParam);
            svg += `<path d="M ${lp.x + 58} ${lp.y} C ${lp.x + 75} ${lp.y}, ${dp.x - 52} ${dp.y}, ${dp.x - 22} ${dp.y}" fill="none" stroke="var(--kind-output)" stroke-width="1.3" opacity="${strokeOp}"/>`;
          }
        }
        if (a?.scratch && scratchPos.has(a.scratch)) {
          const sc = scratchPos.get(a.scratch);
          svg += `<line x1="${lp.x}" y1="${lp.y + 10}" x2="${sc.x}" y2="${sc.y - 9}" stroke="var(--kind-carry)" stroke-dasharray="3 2" stroke-width="1.3" opacity="${strokeOp}"/>`;
        }
      });
    });

    // Draw inParam pills
    inParams.forEach(({ idx, p }) => {
      const pos = pInPos.get(idx);
      svg += `<g transform="translate(${pos.x - 48}, ${pos.y - 9})">
        <rect width="70" height="18" rx="4" fill="var(--bg-surface-2)" stroke="var(--kind-input)" stroke-width="1"/>
        <text x="35" y="12.5" text-anchor="middle" fill="var(--text-1)" font-size="9.2" class="mono">P${idx}:${esc(trunc(p.dtype, 6))}</text>
      </g>`;
    });

    // Draw Launch boxes
    launchPos.forEach((lp, li) => {
      const bCol = lp.active ? "#10b981" : "var(--border-strong)";
      const op = lp.active ? "1" : "0.45";
      svg += `<g transform="translate(${lp.x - 58}, ${lp.y - 11})" opacity="${op}">
        <rect width="116" height="22" rx="5" fill="var(--bg-surface-2)" stroke="${bCol}" stroke-width="1.3"/>
        <text x="58" y="14.5" text-anchor="middle" fill="var(--text-1)" font-size="9.5" font-weight="700" class="mono">L${li}:${esc(trunc(lp.l.entry.replace("extern:", ""), 13))}</text>
      </g>`;
    });

    // Draw Scratch pills
    for (const [sname, sc] of scratchPos.entries()) {
      svg += `<g transform="translate(${sc.x - 52}, ${sc.y - 9})">
        <rect width="104" height="18" rx="9" fill="var(--bg-surface)" stroke="var(--kind-carry)" stroke-dasharray="3 2"/>
        <text x="52" y="12.5" text-anchor="middle" fill="var(--kind-carry)" font-size="9" font-weight="700" class="mono">scratch:${esc(trunc(sname, 8))}</text>
      </g>`;
    }

    // Draw outParam pills
    outParams.forEach(({ idx, p }) => {
      const pos = pOutPos.get(idx);
      svg += `<g transform="translate(${pos.x - 22}, ${pos.y - 9})">
        <rect width="70" height="18" rx="4" fill="var(--bg-surface-2)" stroke="var(--kind-output)" stroke-width="1"/>
        <text x="35" y="12.5" text-anchor="middle" fill="var(--text-1)" font-size="9.2" class="mono">P${idx}:${esc(trunc(p.dtype, 6))}</text>
      </g>`;
    });

    svg += `</svg>`;
    return svg;
  }

  function renderPackByteMap(pack, argIdx) {
    const total = Number(pack.size) || 64;
    const fields = (pack.fields || []).slice().sort((a, b) => (a.at || 0) - (b.at || 0));

    function fieldWidth(f) {
      if (f.width != null) return Number(f.width);
      if (f.tensormap) return 128;
      if (f.param != null || f.scratch != null || f.i64 != null) return 8;
      if (f.u8 != null) return 1;
      return 4;
    }

    function fieldDesc(f) {
      if (f.tensormap) {
        const tm = f.tensormap;
        return { label: `TMA CUtensorMap (P${tm.param} ${tm.dtype} box[${(tm.box || []).join("×")}] swz=${tm.swizzle})`, color: "#ec4899" };
      }
      if (f.param != null) return { label: `param[${f.param}] ptr/val`, color: "#38bdf8" };
      if (f.scratch != null) return { label: `scratch.${f.scratch} ptr`, color: "#c084fc" };
      if (f.var != null) return { label: `var(${f.var})=${state.vars[f.var] ?? "?"}`, color: "#fbbf24" };
      if (f.expr != null) return { label: `expr=${evalExpr(f.expr, state.vars)}`, color: "#fb923c" };
      const k = ["i32", "i64", "f32", "u8"].find((t) => f[t] != null);
      return { label: k ? `${k}: ${f[k]}` : "field", color: "#64748b" };
    }

    // Build proportional visual segments
    let segsHtml = "";
    let cursor = 0;
    const rowsHtml = [];

    fields.forEach((f) => {
      const at = Number(f.at) || 0;
      const w = fieldWidth(f);
      if (at > cursor) {
        const padPct = ((at - cursor) / total) * 100;
        segsHtml += `<div class="pack-seg" style="width:${padPct.toFixed(2)}%;background:transparent" title="zero-pad [${cursor}..${at})"></div>`;
      }
      const pct = Math.max(0.8, (w / total) * 100);
      const fd = fieldDesc(f);
      segsHtml += `<div class="pack-seg" style="width:${pct.toFixed(2)}%;background:${fd.color}" title="[+${at}..+${at + w}B] ${esc(fd.label)}"></div>`;
      cursor = Math.max(cursor, at + w);
      rowsHtml.push(`<div style="display:flex;justify-content:space-between;font-size:10px" class="mono">
        <span style="color:${fd.color}">+${at}..${at + w}B</span>
        <span style="color:var(--text-2)">${esc(trunc(fd.label, 34))}</span>
      </div>`);
    });
    if (cursor < total) {
      segsHtml += `<div class="pack-seg" style="width:${(((total - cursor) / total) * 100).toFixed(2)}%;background:transparent"></div>`;
    }

    return `<div class="pack-ruler" style="margin-top:6px;padding-top:6px;border-top:1px dashed var(--border-subtle)">
      <div class="mono" style="font-size:10px;font-weight:700;color:var(--brand-bright)">ABI Struct Pack (arg #${argIdx} · bytes&lt;${total}&gt; · ${fields.length} fields)</div>
      <div class="pack-bar">${segsHtml}</div>
      <div style="max-height:110px;overflow-y:auto;display:flex;flex-direction:column;gap:2px">${rowsHtml.slice(0, 18).join("")}${rowsHtml.length > 18 ? `<div style="font-size:10px;color:var(--text-3)">… +${rowsHtml.length - 18} more byte fields</div>` : ""}</div>
    </div>`;
  }

  function renderResourceInspector(rk) {
    const m = state.manifest;
    const an = state.analysis;
    const isState = rk.startsWith("state:");
    const name = rk.slice(isState ? 6 : 4);
    const meta = isState ? m.states[name] : m.buffers[name];
    if (!meta) return;

    const kind = isState ? "state" : meta.kind;
    const bytes = isState ? evalStateBytes(meta, m, state.vars) : evalBufferBytes(meta, state.vars);
    const accesses = an.resourceAccesses.get(rk) || [];
    const writers = accesses.filter((a) => a.dir === "write");
    const readers = accesses.filter((a) => a.dir === "read");

    let domainHtml = "";
    if (meta.domain) {
      const d = meta.domain;
      if (d.index_into) {
        const targetIsState = Boolean(m.states?.[d.index_into]);
        const trk = (targetIsState ? "state:" : "buf:") + d.index_into;
        domainHtml = `<div class="port-row clickable" data-jump-rk="${esc(trk)}">
          <span style="color:var(--text-3)">Domain Index Into</span>
          <span class="mono" style="color:var(--brand-bright);font-weight:700">${esc(d.index_into)}${d.stride ? ` (stride ${d.stride})` : ""} →</span>
        </div>`;
      } else {
        domainHtml = `<div class="port-row">
          <span style="color:var(--text-3)">Domain Bounds</span>
          <span class="mono">[${formatExpr(d.min ?? "−∞")}, ${formatExpr(d.max ?? "+∞")}]${d.monotone ? " · monotone" : ""}</span>
        </div>`;
      }
    }

    inspectorEl.innerHTML = `
      <div class="insp-hero">
        <div class="insp-kicker"><span class="dot" style="background:${kindColor(kind)}"></span><span>${esc(kind.toUpperCase())} RESOURCE</span></div>
        <h2 class="insp-title">${esc(name)}</h2>
        <div class="insp-sub">
          ${!isState ? `<span class="tiny-tag">${esc(meta.dtype)}</span>` : `<span class="tiny-tag">VMM Persistent State</span>`}
          <span class="tiny-tag"><strong>${formatBytes(bytes)}</strong> at current vars</span>
          ${meta.fill ? `<span class="tiny-tag cuda-graph">fill: ${esc(meta.fill)}</span>` : ""}
          ${meta.export ? `<span class="tiny-tag once">export: VMM IPC</span>` : ""}
        </div>
      </div>

      <div class="insp-card">
        <div class="insp-card-title"><span>Declaration &amp; Geometry</span></div>
        <div class="port-list">
          ${!isState ? `<div class="port-row"><span style="color:var(--text-3)">Symbolic Shape</span><span class="mono">[${(meta.shape || []).map(formatExpr).join(", ")}]</span></div>
          <div class="port-row"><span style="color:var(--text-3)">Evaluated Shape</span><span class="mono">[${(meta.shape || []).map((d) => evalExpr(d, state.vars)).join(" × ")}]</span></div>` : ""}
          ${isState && meta.bytes_per_token != null ? `<div class="port-row"><span style="color:var(--text-3)">Paging</span><span class="mono">${formatBytes(meta.bytes_per_token)} / token</span></div>` : ""}
          ${isState && meta.bytes_per_seq != null ? `<div class="port-row"><span style="color:var(--text-3)">Per-Sequence Slot</span><span class="mono">${formatBytes(meta.bytes_per_seq)} / seq</span></div>` : ""}
          ${meta.shard ? `<div class="port-row"><span style="color:var(--text-3)">SPMD Shard</span><span class="mono">group ${esc(meta.shard.group)} by ${esc(meta.shard.by)}</span></div>` : ""}
          ${meta.of ? `<div class="port-row"><span style="color:var(--text-3)">Peer Target</span><span class="mono">${esc(meta.of)} (group ${esc(meta.group)})</span></div>` : ""}
          ${domainHtml}
          ${meta.bind ? `<div class="port-row"><span style="color:var(--text-3)">Checkpoint Bind</span><span class="mono" style="font-size:10.5px">${esc(meta.bind.map((b) => b.tensor).join(", "))}</span></div>` : ""}
        </div>
      </div>

      <div class="insp-card">
        <div class="insp-card-title"><span>SSA Read/Write Timeline in \`${esc(an.programName)}\`</span><span>${writers.length}W / ${readers.length}R</span></div>
        <div class="port-list" style="max-height:320px;overflow-y:auto">
          ${accesses.length === 0 ? `<div style="color:var(--text-3);font-size:12px">Not accessed in \`${esc(an.programName)}\`.</div>` : accesses.slice(0, 80).map((a) => {
            const inf = an.callInfos[a.callIdx];
            return `<div class="port-row clickable" data-jump-call="${a.callIdx}">
              <span class="dir-badge ${a.dir === "write" ? "dir-out" : "dir-in"}">${a.dir === "write" ? "WRITE" : "READ"} v${a.ver}</span>
              <span class="mono" style="flex:1">#${a.callIdx} ${esc(trunc(inf.label, 22))}</span>
              ${a.offset ? `<span class="tiny-tag">+${formatBytes(a.offset)}</span>` : ""}
            </div>`;
          }).join("")}
          ${accesses.length > 80 ? `<div style="font-size:11px;color:var(--text-3);padding:4px">… +${accesses.length - 80} more accesses</div>` : ""}
        </div>
      </div>
    `;

    bindInspectorJumpLinks();
  }

  function bindInspectorJumpLinks() {
    inspectorEl.querySelectorAll("[data-jump-call]").forEach((el) => {
      el.addEventListener("click", () => {
        const ci = Number(el.dataset.jumpCall);
        jumpToCall(ci);
      });
    });
    inspectorEl.querySelectorAll("[data-jump-rk]").forEach((el) => {
      el.addEventListener("click", () => {
        const rk = el.dataset.jumpRk;
        state.selected = { type: "resource", rk };
        renderInspector();
        renderSidebarBuffers();
      });
    });
  }

  function jumpToCall(ci) {
    const an = state.analysis;
    if (!an || !an.callInfos[ci]) return;
    const stIdx = an.callInfos[ci].stageIdx;
    if (state.scopeMode === "macro" && !state.expandedStageIdxs.has(stIdx) && an.stages[stIdx].callIndices.length > 1) {
      setScopeMode("stage", stIdx);
    } else if (state.scopeMode === "stage" && state.focusedStageIdx !== stIdx) {
      setScopeMode("stage", stIdx);
    }
    const nid = `call:${ci}`;
    state.selectedNodeId = nid;
    state.selected = { type: "call", callIdx: ci };
    applyHighlightCone(nid);
    renderInspector();
    // Center node smoothly in viewport
    const n = state.layoutResult?.nodes.find((x) => x.id === nid);
    if (n) {
      const rect = canvasWrap.getBoundingClientRect();
      state.zoom.x = rect.width / 2 - n.cx * state.zoom.k;
      state.zoom.y = rect.height / 2 - n.cy * state.zoom.k;
      applyZoomTransform();
    }
  }

  // ==================== VIEW 2: MEMORY & PEAK BUDGET ====================
  const HEADROOM_BYTES = 1024 * 1024 * 1024;       // 1 GiB (`kern_runtime::HEADROOM`)
  const CUBLAS_WORKSPACE_BYTES = 32 * 1024 * 1024; // 32 MiB (`cublas::Blas`)
  const GPU_PRESETS_GIB = [24, 48, 80, 96, 141, 192];

  function gcd(a, b) {
    let x = Math.abs(Math.trunc(a));
    let y = Math.abs(Math.trunc(b));
    while (y !== 0) {
      const t = x % y;
      x = y;
      y = t;
    }
    return x || 1;
  }

  function lcm(a, b) {
    if (!a || !b) return 1;
    return Math.abs(Math.trunc(a / gcd(a, b)) * Math.trunc(b));
  }

  function computePageUnit(manifest) {
    let unit = 1;
    const states = manifest.states || {};
    for (const b of Object.values(manifest.buffers || {})) {
      if (b.kind !== "input") continue;
      const target = b.domain?.index_into;
      const st = target && states[target];
      if (!st || !(st.bytes_per_token > 0)) continue;
      const stride = Number(b.domain?.stride) || 1;
      if (stride > 0) unit = lcm(unit, stride);
    }
    return Math.max(1, unit);
  }

  function align256(bytes) {
    if (!bytes || bytes <= 0) return 0;
    return Math.ceil(bytes / MEM_ALIGN) * MEM_ALIGN;
  }

  // Exact port of `kern_runtime::derive::plan` (`crates/kern-runtime/src/derive.rs`):
  // first-fit 256B-aligned staging windows across `once` derive calls.
  function computeDeriveStaging(manifest, varsMax) {
    const buffers = manifest.buffers || {};
    const sourceBytes = new Map();
    for (const [name, b] of Object.entries(buffers)) {
      if (b.kind === "source") {
        sourceBytes.set(name, evalBufferBytes(b, varsMax));
      }
    }
    if (sourceBytes.size === 0) return { bytes: 0, items: [] };

    const progNames = Object.keys(manifest.programs || {}).sort();
    const deriveCalls = [];
    for (const pn of progNames) {
      const p = manifest.programs[pn];
      if (!p?.once) continue;
      const touchesSource = (p.calls || []).some((c) =>
        (c.args || []).some((a) => a?.buf && sourceBytes.has(a.buf))
      );
      if (touchesSource) {
        for (const c of p.calls || []) deriveCalls.push(c);
      }
    }

    const live = new Map();
    deriveCalls.forEach((c, k) => {
      for (const a of c.args || []) {
        if (a?.buf && sourceBytes.has(a.buf)) {
          const cur = live.get(a.buf);
          if (!cur) live.set(a.buf, { name: a.buf, first: k, last: k, bytes: sourceBytes.get(a.buf) });
          else cur.last = k;
        }
      }
    });

    const windows = Array.from(live.values());
    const order = windows.map((_, idx) => idx).sort((i, j) => (windows[i].first - windows[j].first) || (i - j));
    const at = new Array(windows.length).fill(0);
    const placed = [];

    for (const i of order) {
      const w = windows[i];
      const taken = placed
        .filter((j) => windows[j].last >= w.first)
        .map((j) => [at[j], at[j] + windows[j].bytes])
        .sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
      let lo = 0;
      for (const [a, b] of taken) {
        if (lo + w.bytes <= a) break;
        lo = Math.max(lo, align256(b));
      }
      at[i] = lo;
      placed.push(i);
    }

    let total = 0;
    const items = windows.map((w, idx) => {
      total = Math.max(total, at[idx] + w.bytes);
      const b = buffers[w.name];
      return {
        name: w.name,
        rk: `buf:${w.name}`,
        cat: "staging",
        subKind: "derive source",
        formula: `${b.dtype}[${(b.shape || []).map(formatExpr).join(", ")}] · derive #${w.first}..#${w.last} @ +${formatBytes(at[idx])}`,
        bytes: w.bytes,
      };
    });

    return { bytes: total, items };
  }

  function computeMemoryBudget(manifest) {
    const varsMax = {};
    for (const [k, v] of Object.entries(manifest.vars || {})) {
      varsMax[k] = Number(v.max) || 1;
    }

    const pageUnit = computePageUnit(manifest);
    const items = [];
    let weightsBytes = 0;
    let weightCount = 0;
    let derivedCount = 0;
    let hostWeightBytes = 0;

    let activationsBytes = 0;
    let actBufCount = 0;
    let scratchCount = 0;

    for (const [name, b] of Object.entries(manifest.buffers || {})) {
      if (b.kind === "source") continue;
      const bytes = evalBufferBytes(b, varsMax);
      const shapeStr = `${b.dtype}[${(b.shape || []).map(formatExpr).join(", ")}]`;

      if (b.placement === "host") {
        hostWeightBytes += bytes;
        continue;
      }

      const isDerivedWeight = b.kind === "carry" && state.derivedConstSet.has(name);
      if (b.kind === "weight" || isDerivedWeight) {
        weightsBytes += bytes;
        if (b.kind === "weight") weightCount++;
        else derivedCount++;
        items.push({
          name,
          rk: `buf:${name}`,
          cat: "weights",
          subKind: b.kind === "weight" ? "weight" : "once table",
          formula: shapeStr,
          bytes,
        });
      } else {
        activationsBytes += bytes;
        actBufCount++;
        items.push({
          name,
          rk: `buf:${name}`,
          cat: "activations",
          subKind: b.kind,
          formula: `${shapeStr} (@ max)`,
          bytes,
        });
      }
    }

    // Op-private `impl.scratch` allocations (`compile::resolve_ops` sizes every op's scratch at `vars_max`)
    for (const [opName, op] of Object.entries(manifest.ops || {})) {
      for (const [scName, sc] of Object.entries(op.impl?.scratch || {})) {
        const bytes = evalBufferBytes(sc, varsMax);
        activationsBytes += bytes;
        scratchCount++;
        items.push({
          name: `${opName} · ${scName}`,
          rk: null,
          cat: "activations",
          subKind: "op scratch",
          formula: `${sc.dtype}[${(sc.shape || []).map(formatExpr).join(", ")}] (@ max)`,
          bytes,
        });
      }
    }

    // Fixed cuBLASLt workspace (`Blas::new` in `Runtime::load`)
    activationsBytes += CUBLAS_WORKSPACE_BYTES;
    items.push({
      name: "runtime::cublaslt_workspace",
      rk: null,
      cat: "activations",
      subKind: "runtime",
      formula: "fixed 32 MiB stream workspace",
      bytes: CUBLAS_WORKSPACE_BYTES,
    });

    // States: fixed, per-sequence (`slots + 1`), and paged (`pages * pageUnit`)
    let slotUnitBytes = 0;
    let tokenUnitBytes = 0;
    let fixedStateBytes = 0;
    for (const s of Object.values(manifest.states || {})) {
      if (s.source && s.source !== "owned") continue;
      if (s.bytes_per_token > 0) tokenUnitBytes += Number(s.bytes_per_token);
      else if (s.bytes_per_seq > 0) slotUnitBytes += Number(s.bytes_per_seq);
      else if (s.bytes > 0) fixedStateBytes += Number(s.bytes);
    }

    const activeSlots = slotUnitBytes > 0 ? Math.max(1, Math.trunc(state.memSeqSlots)) : 0;
    const totalProvisionedSlots = slotUnitBytes > 0 ? activeSlots + 1 : 0; // +1 for reserved null slot 0

    const reqTokens = tokenUnitBytes > 0 ? Math.max(pageUnit, Math.trunc(state.memKvTokens)) : 0;
    const pages = tokenUnitBytes > 0 ? Math.floor(reqTokens / pageUnit) : 0;
    const effectiveTokens = pages * pageUnit;

    let seqStatesBytes = 0;
    let kvStatesBytes = 0;

    for (const [sname, s] of Object.entries(manifest.states || {})) {
      if (s.source && s.source !== "owned") continue;
      if (s.bytes_per_token > 0) {
        const bpt = Number(s.bytes_per_token);
        const bytes = bpt * effectiveTokens;
        kvStatesBytes += bytes;
        items.push({
          name: `state.${sname}`,
          rk: `state:${sname}`,
          cat: "kv_states",
          subKind: "paged kv",
          formula: `${formatBytes(bpt)}/tok × ${effectiveTokens.toLocaleString()} tok (${pages.toLocaleString()} pages)`,
          bytes,
        });
      } else if (s.bytes_per_seq > 0) {
        const bps = Number(s.bytes_per_seq);
        const bytes = bps * totalProvisionedSlots;
        seqStatesBytes += bytes;
        items.push({
          name: `state.${sname}`,
          rk: `state:${sname}`,
          cat: "seq_states",
          subKind: "seq state",
          formula: `${formatBytes(bps)}/slot × ${totalProvisionedSlots} slots (${activeSlots} + slot 0)`,
          bytes,
        });
      } else if (s.bytes > 0) {
        const bytes = Number(s.bytes);
        seqStatesBytes += bytes;
        items.push({
          name: `state.${sname}`,
          rk: `state:${sname}`,
          cat: "seq_states",
          subKind: "fixed state",
          formula: `fixed ${formatBytes(bytes)}`,
          bytes,
        });
      }
    }

    const deriveStaging = computeDeriveStaging(manifest, varsMax);
    const servingPeakBytes = weightsBytes + activationsBytes + seqStatesBytes + kvStatesBytes;
    const loadPeakBytes = servingPeakBytes + deriveStaging.bytes;

    const gpuBytes = state.memGpuGiB * 1024 * 1024 * 1024;
    const fixedBeforeKv = weightsBytes + activationsBytes + seqStatesBytes + deriveStaging.bytes;
    const pageBytes = tokenUnitBytes * pageUnit;
    const availForKv = Math.max(0, gpuBytes - HEADROOM_BYTES - fixedBeforeKv);
    const maxFitPages = pageBytes > 0 ? Math.floor(availForKv / pageBytes) : 0;
    const maxFitTokens = maxFitPages * pageUnit;

    return {
      varsMax,
      pageUnit,
      gpuBytes,
      weightsBytes,
      weightCount,
      derivedCount,
      hostWeightBytes,
      activationsBytes,
      actBufCount,
      scratchCount,
      seqStatesBytes,
      slotUnitBytes,
      activeSlots,
      totalProvisionedSlots,
      fixedStateBytes,
      kvStatesBytes,
      tokenUnitBytes,
      pages,
      effectiveTokens,
      deriveStagingBytes: deriveStaging.bytes,
      servingPeakBytes,
      loadPeakBytes,
      fixedBeforeKv,
      maxFitTokens,
      items: [...items, ...deriveStaging.items],
    };
  }

  function initMemoryDefaults(manifest) {
    state.memFilterCat = "all";
    state.memSearch = "";
    const states = Object.values(manifest.states || {});
    const hasSeq = states.some((s) => (!s.source || s.source === "owned") && s.bytes_per_seq > 0);
    const hasKv = states.some((s) => (!s.source || s.source === "owned") && s.bytes_per_token > 0);
    const pageUnit = computePageUnit(manifest);

    state.memSeqSlots = hasSeq ? Math.max(1, Math.min(16, seqSlotsOf(manifest) - 1)) : 0;
    const tokMax = Number(manifest.vars?.tokens?.max) || 32768;
    state.memKvTokens = hasKv ? Math.max(pageUnit, Math.min(32768, Math.max(tokMax, 8192))) : 0;

    // Pick a realistic initial GPU tier so opening view shows meaningful headroom
    const probe = computeMemoryBudget(manifest);
    const neededGiB = (probe.loadPeakBytes + HEADROOM_BYTES) / (1024 ** 3);
    state.memGpuGiB = GPU_PRESETS_GIB.find((g) => g >= neededGiB) || 192;
  }

  function foldLayerAllocations(items) {
    const groups = new Map();
    const out = [];
    const layerRe = /^(.*?\blayers?)\.(\d+)\.(.+)$/i;

    for (const it of items) {
      const m = layerRe.exec(it.name);
      if (!m) {
        out.push({ ...it, count: 1 });
        continue;
      }
      const prefix = m[1];
      const idx = Number(m[2]);
      const suffix = m[3];
      const key = `${it.cat}|${it.subKind}|${prefix}|${suffix}|${it.formula}`;
      let g = groups.get(key);
      if (!g) {
        g = {
          ...it,
          prefix,
          suffix,
          minIdx: idx,
          maxIdx: idx,
          unitBytes: it.bytes,
          bytes: 0,
          count: 0,
        };
        groups.set(key, g);
        out.push(g);
      }
      g.minIdx = Math.min(g.minIdx, idx);
      g.maxIdx = Math.max(g.maxIdx, idx);
      g.bytes += it.bytes;
      g.count++;
    }

    for (const g of groups.values()) {
      if (g.count > 1) {
        g.name = `${g.prefix}.[${g.minIdx}..${g.maxIdx}].${g.suffix}`;
        g.formula = `${g.formula} × ${g.count} (${formatBytes(g.unitBytes)} each)`;
      }
    }
    return out;
  }

  function catColor(cat) {
    switch (cat) {
      case "weights": return "var(--cat-gemm)";
      case "activations": return "var(--kind-workspace)";
      case "seq_states": return "var(--kind-state)";
      case "kv_states": return "var(--kind-output)";
      case "staging": return "var(--kind-carry)";
      default: return "var(--text-3)";
    }
  }

  function renderMemoryTableRows(budget) {
    const tbody = document.getElementById("mem-breakdown-tbody");
    const countEl = document.getElementById("mem-table-count");
    if (!tbody) return;

    let list = budget.items;
    if (state.memFilterCat !== "all") {
      if (state.memFilterCat === "states") {
        list = list.filter((it) => it.cat === "seq_states" || it.cat === "kv_states");
      } else {
        list = list.filter((it) => it.cat === state.memFilterCat);
      }
    }

    if (state.memGroupLayers) {
      list = foldLayerAllocations(list);
    } else {
      list = list.map((it) => ({ ...it, count: 1 }));
    }

    const q = state.memSearch.trim().toLowerCase();
    if (q) {
      list = list.filter((it) =>
        it.name.toLowerCase().includes(q) ||
        it.subKind.toLowerCase().includes(q) ||
        it.formula.toLowerCase().includes(q)
      );
    }

    list.sort((a, b) => (b.bytes - a.bytes) || a.name.localeCompare(b.name));
    if (countEl) countEl.textContent = `${list.length} item${list.length === 1 ? "" : "s"}`;

    const denom = Math.max(1, budget.loadPeakBytes);
    const shown = list.slice(0, 120);

    if (shown.length === 0) {
      tbody.innerHTML = `<tr><td colspan="5" style="color:var(--text-3);padding:16px 10px;text-align:center">No allocations match the current filter.</td></tr>`;
      return;
    }

    tbody.innerHTML = shown.map((it) => {
      const pct = (it.bytes / denom) * 100;
      const col = catColor(it.cat);
      const clickable = Boolean(it.rk);
      return `<tr class="${clickable ? "clickable" : ""}" ${clickable ? `data-mem-jump-rk="${esc(it.rk)}" title="Click to inspect in Dataflow Blueprint"` : ""}>
        <td class="mono" style="font-weight:700">
          ${esc(it.name)}
          ${it.count > 1 ? `<span class="tiny-tag mono" style="margin-left:6px">×${it.count}</span>` : ""}
        </td>
        <td><span class="tiny-tag" style="color:${col}">${esc(it.subKind)}</span></td>
        <td class="mono" style="color:var(--text-2);font-size:11.5px">${esc(it.formula)}</td>
        <td class="mono" style="font-weight:800;text-align:right;white-space:nowrap">${formatBytes(it.bytes)}</td>
        <td style="width:140px">
          <div class="share-bar-cell">
            <div class="share-bar-bg">
              <div class="share-bar-fill" style="width:${Math.min(100, Math.max(1.2, pct)).toFixed(1)}%;background:${col}"></div>
            </div>
            <span class="mono" style="font-size:11px;color:var(--text-2)">${pct >= 0.1 ? pct.toFixed(1) + "%" : "<0.1%"}</span>
          </div>
        </td>
      </tr>`;
    }).join("") + (list.length > shown.length
      ? `<tr><td colspan="5" style="color:var(--text-3);font-size:11px;text-align:center;padding:10px">Showing top ${shown.length} of ${list.length} allocations by size</td></tr>`
      : "");

    tbody.querySelectorAll("[data-mem-jump-rk]").forEach((row) => {
      row.addEventListener("click", () => {
        const rk = row.dataset.memJumpRk;
        state.selected = { type: "resource", rk };
        state.activeView = "graph";
        document.querySelectorAll(".view-tab").forEach((b) => b.classList.toggle("active", b.dataset.view === "graph"));
        document.getElementById("view-graph").classList.add("active");
        document.getElementById("view-memory").classList.remove("active");
        document.getElementById("view-protocol").classList.remove("active");
        renderGraph(false);
        renderInspector();
      });
    });
  }

  function renderMemoryDashboard() {
    const container = document.getElementById("memory-dashboard");
    const m = state.manifest;
    if (!container || !m) return;

    const b = computeMemoryBudget(m);
    const peak = b.loadPeakBytes;
    const gpu = b.gpuBytes;
    const pctOfGpu = gpu > 0 ? (peak / gpu) * 100 : 0;

    let verdictCls = "ok";
    let verdictText = "";
    if (peak > gpu) {
      verdictCls = "oom";
      verdictText = `OOM · Exceeds ${state.memGpuGiB} GiB by +${formatBytes(peak - gpu)}`;
    } else if (gpu - peak < HEADROOM_BYTES) {
      verdictCls = "tight";
      verdictText = `Tight · Only ${formatBytes(gpu - peak)} free (< 1 GiB runtime headroom)`;
    } else {
      verdictCls = "ok";
      verdictText = `Fits · ${formatBytes(gpu - peak)} free`;
    }

    const scaleMax = Math.max(gpu, peak, 1);
    const wPct = (b.weightsBytes / scaleMax) * 100;
    const aPct = (b.activationsBytes / scaleMax) * 100;
    const sPct = (b.seqStatesBytes / scaleMax) * 100;
    const kPct = (b.kvStatesBytes / scaleMax) * 100;
    const dPct = (b.deriveStagingBytes / scaleMax) * 100;
    const limitLeftPct = peak > gpu ? (gpu / scaleMax) * 100 : null;

    const varsMaxSummary = Object.entries(b.varsMax)
      .slice(0, 3)
      .map(([k, v]) => `${k}=${v}`)
      .join(", ");

    let subNote = "";
    if (b.fixedBeforeKv > gpu) {
      subNote = `Fixed weights + buffers + sequence slots (${formatBytes(b.fixedBeforeKv)}) already exceed ${state.memGpuGiB} GiB before allocating any KV pages.`;
    } else if (b.tokenUnitBytes > 0) {
      subNote = `With 1 GiB headroom, ${state.memGpuGiB} GiB fits up to ${b.maxFitTokens.toLocaleString()} KV tokens (${Math.floor(b.maxFitTokens / b.pageUnit).toLocaleString()} pages × ${b.pageUnit} tok).`;
    } else {
      subNote = `No paged KV state declared in manifest; peak is fixed once sequence slots are chosen.`;
    }
    if (b.deriveStagingBytes > 0) {
      subNote += ` · Steady-state serving is ${formatBytes(b.servingPeakBytes)} (+${formatBytes(b.deriveStagingBytes)} temporary derive staging freed after load).`;
    }

    const denomPeak = Math.max(1, peak);
    const pctStr = (x) => `${((x / denomPeak) * 100).toFixed(1)}%`;

    container.innerHTML = `
      <div class="mem-peak-card">
        <div class="mem-peak-top">
          <div style="display:flex;flex-direction:column;gap:5px">
            <span class="kpi-label">Peak Device Memory (Runtime::load)</span>
            <div class="mem-peak-headline">
              <span class="mem-peak-number mono">${formatBytes(peak)}</span>
              <span class="mem-peak-denom mono">/ ${state.memGpuGiB} GiB (${pctOfGpu.toFixed(1)}%)</span>
              <span class="mem-verdict ${verdictCls}">${esc(verdictText)}</span>
            </div>
            <div style="font-size:12px;color:var(--text-2)">${esc(subNote)}</div>
          </div>

          <div class="mem-controls-row">
            <div class="seg-pills" title="Target GPU VRAM per rank">
              ${GPU_PRESETS_GIB.map((g) => `<button class="seg-pill ${g === state.memGpuGiB ? "active" : ""}" data-gpu-gib="${g}">${g}G</button>`).join("")}
            </div>

            ${b.slotUnitBytes > 0 ? `
              <div class="mem-field" title="Concurrent sequence slots (Runtime::load provisions slots + 1 for null slot 0)">
                <label for="mem-slots-in">Seq Slots</label>
                <input id="mem-slots-in" class="mem-num-input" type="number" min="1" max="4096" step="1" value="${b.activeSlots}">
              </div>
            ` : ""}

            ${b.tokenUnitBytes > 0 ? `
              <div class="mem-field" title="Provisioned KV cache tokens (rounded down to whole ${b.pageUnit}-token pages)">
                <label for="mem-tokens-in">KV Tokens</label>
                <input id="mem-tokens-in" class="mem-num-input" style="width:86px" type="number" min="${b.pageUnit}" step="${Math.max(b.pageUnit, 1024)}" value="${b.effectiveTokens || state.memKvTokens}">
                <button id="mem-fill-gpu" class="btn-mini" title="Set KV tokens to fill remaining ${state.memGpuGiB} GiB minus 1 GiB headroom">Max Fit</button>
              </div>
            ` : ""}
          </div>
        </div>

        <div class="mem-ruler-wrap">
          <div class="mem-ruler-track">
            <div class="mem-ruler-seg" style="width:${wPct.toFixed(3)}%;background:var(--cat-gemm)" title="Weights & Once Tables: ${formatBytes(b.weightsBytes)}"></div>
            <div class="mem-ruler-seg" style="width:${aPct.toFixed(3)}%;background:var(--kind-workspace)" title="Buffers & Scratch (@ vars.max): ${formatBytes(b.activationsBytes)}"></div>
            <div class="mem-ruler-seg" style="width:${sPct.toFixed(3)}%;background:var(--kind-state)" title="Per-Seq & Fixed States: ${formatBytes(b.seqStatesBytes)}"></div>
            <div class="mem-ruler-seg" style="width:${kPct.toFixed(3)}%;background:var(--kind-output)" title="Paged KV Pool: ${formatBytes(b.kvStatesBytes)}"></div>
            ${b.deriveStagingBytes > 0 ? `<div class="mem-ruler-seg staging" style="width:${dPct.toFixed(3)}%" title="Temporary Load-Time Derive Staging: ${formatBytes(b.deriveStagingBytes)}"></div>` : ""}
          </div>
          ${limitLeftPct !== null ? `
            <div class="mem-limit-marker" style="left:${limitLeftPct.toFixed(2)}%">
              <span class="mem-limit-tag mono">${state.memGpuGiB} GiB LIMIT</span>
            </div>
          ` : ""}
        </div>

        <div class="mem-legend-row">
          <div class="mem-legend-items">
            <span><i class="mem-leg-dot" style="background:var(--cat-gemm)"></i>Weights <strong>${formatBytes(b.weightsBytes)}</strong></span>
            <span><i class="mem-leg-dot" style="background:var(--kind-workspace)"></i>Buffers &amp; Scratch <strong>${formatBytes(b.activationsBytes)}</strong></span>
            <span><i class="mem-leg-dot" style="background:var(--kind-state)"></i>Seq States <strong>${formatBytes(b.seqStatesBytes)}</strong></span>
            <span><i class="mem-leg-dot" style="background:var(--kind-output)"></i>Paged KV <strong>${formatBytes(b.kvStatesBytes)}</strong></span>
            ${b.deriveStagingBytes > 0 ? `<span><i class="mem-leg-dot" style="background:var(--kind-carry)"></i>Load Staging <strong>+${formatBytes(b.deriveStagingBytes)}</strong></span>` : ""}
          </div>
          ${b.hostWeightBytes > 0 ? `<span class="mono" style="font-size:11px;color:var(--text-3)">Host-pinned weights (off-GPU): ${formatBytes(b.hostWeightBytes)}</span>` : ""}
        </div>
      </div>

      <div class="kpi-grid">
        <div class="kpi-card clickable-kpi ${state.memFilterCat === "weights" ? "active-filter" : ""}" data-mem-card="weights">
          <span class="kpi-label" style="color:var(--cat-gemm)">1 · Weights &amp; Tables</span>
          <span class="kpi-value mono">${formatBytes(b.weightsBytes)}</span>
          <span class="kpi-sub">${pctStr(b.weightsBytes)} of peak · ${b.weightCount} weights${b.derivedCount ? ` + ${b.derivedCount} once` : ""}</span>
        </div>

        <div class="kpi-card clickable-kpi ${state.memFilterCat === "activations" ? "active-filter" : ""}" data-mem-card="activations">
          <span class="kpi-label" style="color:var(--kind-workspace)">2 · Buffers &amp; Scratch</span>
          <span class="kpi-value mono">${formatBytes(b.activationsBytes)}</span>
          <span class="kpi-sub">${pctStr(b.activationsBytes)} · sized @ max (${esc(varsMaxSummary || "static")})</span>
        </div>

        <div class="kpi-card clickable-kpi ${state.memFilterCat === "seq_states" ? "active-filter" : ""}" data-mem-card="seq_states">
          <span class="kpi-label" style="color:var(--kind-state)">3 · Per-Seq States</span>
          <span class="kpi-value mono">${formatBytes(b.seqStatesBytes)}</span>
          <span class="kpi-sub">${b.slotUnitBytes > 0 ? `${formatBytes(b.slotUnitBytes)}/slot × ${b.totalProvisionedSlots} slots` : (b.fixedStateBytes > 0 ? `Fixed ${formatBytes(b.fixedStateBytes)}` : "None declared")}</span>
        </div>

        <div class="kpi-card clickable-kpi ${state.memFilterCat === "kv_states" ? "active-filter" : ""}" data-mem-card="kv_states">
          <span class="kpi-label" style="color:var(--kind-output)">4 · Paged KV Pool</span>
          <span class="kpi-value mono">${formatBytes(b.kvStatesBytes)}</span>
          <span class="kpi-sub">${b.tokenUnitBytes > 0 ? `${formatBytes(b.tokenUnitBytes)}/tok × ${b.effectiveTokens.toLocaleString()} tok` : "None declared"}</span>
        </div>
      </div>

      <div class="panel-box">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px">
          <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
            <h3 style="margin:0">Top Memory Allocations</h3>
            <span id="mem-table-count" class="pill-count mono">0</span>
            <div class="seg-pills">
              <button class="seg-pill ${state.memFilterCat === "all" ? "active" : ""}" data-mem-cat="all">All</button>
              <button class="seg-pill ${state.memFilterCat === "weights" ? "active" : ""}" data-mem-cat="weights">Weights</button>
              <button class="seg-pill ${state.memFilterCat === "activations" ? "active" : ""}" data-mem-cat="activations">Buffers &amp; Scratch</button>
              <button class="seg-pill ${state.memFilterCat === "states" ? "active" : ""}" data-mem-cat="states">States</button>
              ${b.deriveStagingBytes > 0 ? `<button class="seg-pill ${state.memFilterCat === "staging" ? "active" : ""}" data-mem-cat="staging">Load Staging</button>` : ""}
            </div>
          </div>

          <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
            <label style="display:inline-flex;align-items:center;gap:5px;font-size:11.5px;color:var(--text-2);cursor:pointer">
              <input id="mem-fold-layers" type="checkbox" ${state.memGroupLayers ? "checked" : ""}>
              <span>Fold <code>layers.*</code></span>
            </label>
            <input id="mem-search-input" class="filter-input" style="margin:0;width:190px" type="search" placeholder="Filter by name..." value="${esc(state.memSearch)}">
          </div>
        </div>

        <table class="data-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Type</th>
              <th>Sizing (@ vars.max / capacity)</th>
              <th style="text-align:right">Bytes</th>
              <th>Share of Peak</th>
            </tr>
          </thead>
          <tbody id="mem-breakdown-tbody"></tbody>
        </table>
      </div>
    `;

    renderMemoryTableRows(b);

    container.querySelectorAll("[data-gpu-gib]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.memGpuGiB = Number(btn.dataset.gpuGib) || 80;
        renderMemoryDashboard();
      });
    });

    const slotsIn = document.getElementById("mem-slots-in");
    if (slotsIn) {
      slotsIn.addEventListener("change", () => {
        state.memSeqSlots = Math.max(1, Math.min(4096, Number(slotsIn.value) || 1));
        renderMemoryDashboard();
      });
    }

    const tokIn = document.getElementById("mem-tokens-in");
    if (tokIn) {
      tokIn.addEventListener("change", () => {
        state.memKvTokens = Math.max(b.pageUnit, Number(tokIn.value) || b.pageUnit);
        renderMemoryDashboard();
      });
    }

    const fillBtn = document.getElementById("mem-fill-gpu");
    if (fillBtn) {
      fillBtn.addEventListener("click", () => {
        state.memKvTokens = Math.max(b.pageUnit, b.maxFitTokens);
        renderMemoryDashboard();
      });
    }

    container.querySelectorAll("[data-mem-card]").forEach((card) => {
      card.addEventListener("click", () => {
        const c = card.dataset.memCard;
        state.memFilterCat = state.memFilterCat === c ? "all" : c;
        renderMemoryDashboard();
      });
    });

    container.querySelectorAll("[data-mem-cat]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.memFilterCat = btn.dataset.memCat;
        renderMemoryDashboard();
      });
    });

    const foldChk = document.getElementById("mem-fold-layers");
    if (foldChk) {
      foldChk.addEventListener("change", () => {
        state.memGroupLayers = foldChk.checked;
        renderMemoryTableRows(b);
      });
    }

    const searchIn = document.getElementById("mem-search-input");
    if (searchIn) {
      searchIn.addEventListener("input", () => {
        state.memSearch = searchIn.value;
        renderMemoryTableRows(b);
      });
    }
  }

  // ==================== VIEW 3: SERVING PROTOCOL & OPS CATALOG ====================
  function renderProtocolDashboard() {
    const container = document.getElementById("protocol-dashboard");
    const m = state.manifest;
    if (!m) return;

    const fillInputs = Object.entries(m.buffers || {}).filter(([, b]) => b.kind === "input" && b.fill);
    const fillOutputs = Object.entries(m.buffers || {}).filter(([, b]) => b.kind === "output" && b.fill);
    const progs = Object.entries(m.programs || {});
    const opsEntries = Object.entries(m.ops || {});
    const modEntries = Object.entries(m.modules || {});

    // Count calls per op across all programs
    const opCallFreq = {};
    for (const [, p] of progs) {
      for (const c of p.calls || []) opCallFreq[c.op] = (opCallFreq[c.op] || 0) + 1;
    }

    container.innerHTML = `
      <div class="panel-box">
        <h3><span>Serving Protocol Boundary (<code>Fill</code> Roles ↔ Programs)</span></h3>
        <div style="display:grid;grid-template-columns:1fr 1.3fr 1fr;gap:16px;align-items:start">
          <div class="insp-card">
            <div class="insp-card-title"><span style="color:var(--kind-input)">Caller Fill Inputs (${fillInputs.length})</span></div>
            <div class="port-list">
              ${fillInputs.map(([n, b]) => `<div class="port-row">
                <span class="tiny-tag" style="color:var(--kind-input)">fill:${esc(b.fill)}</span>
                <span class="mono" style="font-weight:700">${esc(n)}</span>
                <span class="mono" style="color:var(--text-3)">${esc(b.dtype)}[${(b.shape || []).map(formatExpr).join(",")}]</span>
              </div>`).join("") || `<div style="color:var(--text-3)">No fill-annotated inputs</div>`}
            </div>
          </div>

          <div class="insp-card">
            <div class="insp-card-title"><span style="color:var(--brand-bright)">Declared Programs (${progs.length})</span></div>
            <div class="port-list">
              ${progs.map(([pn, p]) => `<div class="port-row">
                <span class="mono" style="font-weight:800">${esc(pn)}</span>
                <span class="mono" style="color:var(--text-2)">${(p.calls || []).length} calls</span>
                <span>
                  ${p.batch ? `<span class="tiny-tag">${p.batch.groups}×${formatExpr(p.batch.rows)}</span>` : ""}
                  ${p.graph ? `<span class="tiny-tag cuda-graph">graph</span>` : ""}
                  ${p.once ? `<span class="tiny-tag once">once</span>` : ""}
                </span>
              </div>`).join("")}
            </div>
          </div>

          <div class="insp-card">
            <div class="insp-card-title"><span style="color:var(--kind-output)">Caller Fill Outputs (${fillOutputs.length})</span></div>
            <div class="port-list">
              ${fillOutputs.map(([n, b]) => `<div class="port-row">
                <span class="tiny-tag" style="color:var(--kind-output)">fill:${esc(b.fill)}</span>
                <span class="mono" style="font-weight:700">${esc(n)}</span>
                <span class="mono" style="color:var(--text-3)">${esc(b.dtype)}[${(b.shape || []).map(formatExpr).join(",")}]</span>
              </div>`).join("") || `<div style="color:var(--text-3)">Outputs read directly by buffer name</div>`}
            </div>
          </div>
        </div>
      </div>

      <div class="panel-box">
        <h3><span>Operators Catalog (${opsEntries.length} ops)</span></h3>
        <table class="data-table">
          <thead><tr><th>Operator</th><th>Category</th><th>Typed Interface</th><th>Launches &amp; Scratch</th><th>Calls</th></tr></thead>
          <tbody>
            ${opsEntries.map(([oname, op]) => {
              const cat = classifyOp(oname, "", op);
              const ls = op.impl?.launches || [];
              const scCount = Object.keys(op.impl?.scratch || {}).length;
              return `<tr>
                <td class="mono" style="font-weight:700">${esc(oname)}</td>
                <td><span class="tiny-tag" style="border-color:${cat.color};color:${cat.color}">${esc(cat.tag)}</span></td>
                <td class="mono" style="font-size:11px;color:var(--text-2)">(${(op.params || []).map(esc).join(", ")})</td>
                <td class="mono" style="font-size:11px">${ls.length} launch${ls.length !== 1 ? "es" : ""}${scCount ? ` · ${scCount} scratch` : ""}${ls.some((l) => l.when) ? " · when-gated" : ""}</td>
                <td class="mono" style="font-weight:700">${opCallFreq[oname] || 0}</td>
              </tr>`;
            }).join("")}
          </tbody>
        </table>
      </div>

      <div class="panel-box">
        <h3><span>Hash-Pinned Kernel Modules (${modEntries.length} cubins)</span></h3>
        <table class="data-table">
          <thead><tr><th>Module Key</th><th>Artifact Source</th><th>Pinned SHA-256 Digest</th></tr></thead>
          <tbody>
            ${modEntries.map(([mname, mod]) => `<tr>
              <td class="mono" style="font-weight:700">${esc(mname)}</td>
              <td class="mono" style="color:var(--brand-bright)">${esc(mod.source)}</td>
              <td class="mono" style="color:var(--text-2);font-size:11px">${esc(mod.sha256)}</td>
            </tr>`).join("")}
          </tbody>
        </table>
      </div>
    `;
  }

  // ==================== SIDEBAR & TOPBAR UI SYNC ====================
  function countActiveProgramLaunches() {
    const an = state.analysis;
    if (!an) return 0;
    let total = 0;
    for (const inf of an.callInfos) {
      for (const l of inf.opDef?.impl?.launches || []) {
        if (isLaunchActive(l, state.vars)) total++;
      }
    }
    return total;
  }

  function renderTopbarVars() {
    const bar = document.getElementById("vars-bar");
    const m = state.manifest;
    if (!m || !m.vars) { bar.innerHTML = ""; return; }

    bar.innerHTML = Object.entries(m.vars).map(([vname, vdef]) => {
      const maxVal = Number(vdef.max) || 1024;
      const curVal = state.vars[vname] ?? Math.min(maxVal, vname === "tokens" ? 128 : 4);
      state.vars[vname] = curVal;
      return `<div class="var-slider-chip">
        <span class="var-k">${esc(vname)}</span>
        <input type="range" min="1" max="${maxVal}" value="${curVal}" data-var-slider="${esc(vname)}">
        <span class="var-v" id="var-val-${esc(vname)}">${curVal}</span>
      </div>`;
    }).join("");

    bar.querySelectorAll("[data-var-slider]").forEach((inp) => {
      inp.addEventListener("input", () => {
        const vn = inp.dataset.varSlider;
        state.vars[vn] = Number(inp.value);
        const lbl = document.getElementById(`var-val-${vn}`);
        if (lbl) lbl.textContent = inp.value;
        if (state.activeView === "graph") {
          renderGraph(false);
          renderInspector();
          renderSidebarBuffers();
        } else if (state.activeView === "memory") {
          renderMemoryDashboard();
        }
      });
    });

    const badgesEl = document.getElementById("artifact-badges");
    const nb = Object.keys(m.buffers || {}).length;
    const no = Object.keys(m.ops || {}).length;
    const nm = Object.keys(m.modules || {}).length;
    badgesEl.innerHTML = `
      <span class="meta-badge"><strong>${esc(m.model)}</strong></span>
      <span class="meta-badge"><strong>${no}</strong> ops · <strong>${nm}</strong> cubins · <strong>${nb}</strong> bufs</span>
    `;
  }

  function renderSidebarPrograms() {
    const listEl = document.getElementById("program-list");
    const m = state.manifest;
    const entries = Object.entries(m?.programs || {});
    document.getElementById("program-count-badge").textContent = String(entries.length);

    listEl.innerHTML = entries.map(([pname, p]) => {
      const active = pname === state.programName ? " active" : "";
      const bTxt = p.batch ? `batch ${p.batch.groups}×${formatExpr(p.batch.rows)}` : "";
      return `<button class="program-item${active}" data-prog="${esc(pname)}">
        <div class="prog-row-top">
          <span class="prog-name">${esc(pname)}</span>
          <span class="prog-calls">${(p.calls || []).length} calls</span>
        </div>
        <div class="prog-row-tags">
          ${bTxt ? `<span class="tiny-tag">${esc(bTxt)}</span>` : ""}
          ${p.graph ? `<span class="tiny-tag cuda-graph">CUDA Graph</span>` : ""}
          ${p.once ? `<span class="tiny-tag once">once</span>` : ""}
          ${p.derive ? `<span class="tiny-tag once">derive</span>` : ""}
        </div>
      </button>`;
    }).join("");

    listEl.querySelectorAll("[data-prog]").forEach((btn) => {
      btn.addEventListener("click", () => selectProgram(btn.dataset.prog));
    });
  }

  function renderSidebarStages() {
    const listEl = document.getElementById("stage-list");
    const an = state.analysis;
    if (!an) return;
    const q = (document.getElementById("stage-filter").value || "").trim().toLowerCase();

    listEl.innerHTML = an.stages
      .filter((st) => !q || st.displayName.toLowerCase().includes(q))
      .map((st) => {
        const isAct = (state.scopeMode === "stage" && state.focusedStageIdx === st.idx) ? " active" : "";
        const isExp = state.expandedStageIdxs.has(st.idx);
        return `<div class="stage-row${isAct}" data-stage-row="${st.idx}">
          <span class="stage-sig-dot" style="background:${st.sigColor}" title="Structural Pattern #${st.sigId + 1}"></span>
          <span class="stage-row-name">${esc(st.displayName)}</span>
          <span class="stage-row-meta">${st.callIndices.length}c</span>
          ${state.scopeMode === "macro" && st.callIndices.length > 1
            ? `<span class="stage-expand-pill ${isExp ? "expanded" : ""}" data-row-expand="${st.idx}" title="Expand/collapse inline on Macro canvas">${isExp ? "▼ open" : "⊕"}</span>`
            : ""}
        </div>`;
      }).join("");

    listEl.querySelectorAll("[data-stage-row]").forEach((row) => {
      row.addEventListener("click", (ev) => {
        const expBtn = ev.target.closest("[data-row-expand]");
        const idx = Number(row.dataset.stageRow);
        if (expBtn) {
          ev.stopPropagation();
          if (state.expandedStageIdxs.has(idx)) state.expandedStageIdxs.delete(idx);
          else state.expandedStageIdxs.add(idx);
          renderSidebarStages();
          renderGraph(false);
          return;
        }
        setScopeMode("stage", idx);
      });
    });
  }

  function renderSidebarBuffers() {
    const m = state.manifest;
    const an = state.analysis;
    if (!m || !an) return;

    const kinds = ["all", "state", "input", "output", "workspace", "carry", "peer", "weight"];
    const pillsEl = document.getElementById("buffer-kind-pills");
    pillsEl.innerHTML = kinds.map((k) =>
      `<button class="kind-chip ${state.bufKindFilter === k ? "active" : ""}" data-kchip="${k}">${k}</button>`
    ).join("");
    pillsEl.querySelectorAll("[data-kchip]").forEach((b) => {
      b.addEventListener("click", () => {
        state.bufKindFilter = b.dataset.kchip;
        renderSidebarBuffers();
      });
    });

    const q = (document.getElementById("buffer-filter").value || "").trim().toLowerCase();
    const items = [];
    for (const [sname, st] of Object.entries(m.states || {})) {
      if (state.bufKindFilter !== "all" && state.bufKindFilter !== "state") continue;
      if (q && !sname.toLowerCase().includes(q)) continue;
      items.push({ rk: `state:${sname}`, name: sname, kind: "state", dtype: "state", bytes: evalStateBytes(st, m, state.vars) });
    }
    for (const [bname, buf] of Object.entries(m.buffers || {})) {
      if (state.bufKindFilter !== "all" && state.bufKindFilter !== buf.kind) continue;
      if (q && !bname.toLowerCase().includes(q) && !(buf.dtype || "").toLowerCase().includes(q)) continue;
      items.push({ rk: `buf:${bname}`, name: bname, kind: buf.kind, dtype: buf.dtype, bytes: evalBufferBytes(buf, state.vars) });
    }

    const listEl = document.getElementById("sidebar-buffer-list");
    listEl.innerHTML = items.slice(0, 140).map((it) => {
      const accCount = (an.resourceAccesses.get(it.rk) || []).length;
      const isAct = state.selected?.type === "resource" && state.selected.rk === it.rk ? " active" : "";
      return `<div class="buf-row${isAct}" data-buf-rk="${esc(it.rk)}">
        <div class="buf-row-top">
          <span class="buf-row-name">${esc(it.name)}</span>
          <span class="tiny-tag" style="color:${kindColor(it.kind)}">${esc(it.kind)}</span>
        </div>
        <div class="buf-row-sub">
          <span>${esc(it.dtype)} · ${formatBytes(it.bytes)}</span>
          <span>${accCount} ops</span>
        </div>
      </div>`;
    }).join("");

    listEl.querySelectorAll("[data-buf-rk]").forEach((el) => {
      el.addEventListener("click", () => {
        const rk = el.dataset.bufRk;
        state.selected = { type: "resource", rk };
        renderSidebarBuffers();
        renderInspector();
        // Also highlight on canvas if visible
        const matchNode = state.layoutResult?.nodes.find((n) => n.rk === rk);
        if (matchNode) {
          state.selectedNodeId = matchNode.id;
          applyHighlightCone(matchNode.id);
        }
      });
    });
  }

  function setScopeMode(mode, stageIdx) {
    state.scopeMode = mode;
    if (stageIdx != null) state.focusedStageIdx = stageIdx;
    document.querySelectorAll(".scope-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.scope === mode);
    });
    document.getElementById("slice-controls").classList.toggle("hidden", mode !== "slice");
    state.selectedNodeId = null;
    if (mode === "stage") {
      state.selected = { type: "stage", stageIdx: state.focusedStageIdx };
    } else {
      state.selected = null;
    }
    renderSidebarStages();
    renderGraph(true);
    renderInspector();
  }

  function selectProgram(progName) {
    const m = state.manifest;
    if (!m || !m.programs?.[progName]) return;
    state.programName = progName;
    state.analysis = analyzeProgram(m, progName);
    state.expandedStageIdxs.clear();
    state.unfoldedFamilyKeys.clear();
    state.selected = null;
    state.selectedNodeId = null;

    // Compact-first default scope:
    // - Tiny programs (<= 16 calls total, e.g. showcase / minimal): expand stages so the full graph shows on one screen.
    // - Large programs (hundreds of calls): keep Macro completely folded (~5–12 blocks total, fitting 100% on one screen!),
    //   and pre-point `focusedStageIdx` at the first multi-call layer for 1-click Stage Focus!
    const an = state.analysis;
    if (an.calls.length <= 16) {
      an.stages.forEach((st) => state.expandedStageIdxs.add(st.idx));
      state.scopeMode = "macro";
      state.focusedStageIdx = 0;
    } else {
      const firstRich = an.stages.find((st) => st.callIndices.length >= 4 && st.callIndices.length <= 30);
      state.scopeMode = "macro";
      state.focusedStageIdx = firstRich ? firstRich.idx : 0;
    }

    document.querySelectorAll(".scope-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.scope === state.scopeMode);
    });
    document.getElementById("slice-controls").classList.add("hidden");
    document.getElementById("slice-end").value = String(Math.min(an.calls.length - 1, 28));

    renderSidebarPrograms();
    renderSidebarStages();
    renderSidebarBuffers();
    if (state.activeView === "graph") {
      renderGraph(true);
      renderInspector();
    } else if (state.activeView === "memory") {
      renderMemoryDashboard();
    } else {
      renderProtocolDashboard();
    }
  }

  function loadManifestData(bundle) {
    state.meta = bundle.meta || {};
    state.manifest = bundle.manifest;

    // Detect `carry` buffers initialized once in `once: true` programs (`derive`) and never mutated in step programs
    const onceWritten = new Set();
    const stepWritten = new Set();
    const ops = state.manifest.ops || {};
    for (const prog of Object.values(state.manifest.programs || {})) {
      const targetSet = prog.once ? onceWritten : stepWritten;
      for (const call of prog.calls || []) {
        const pSpecs = (ops[call.op]?.params || []).map(parseParamSpec);
        (call.args || []).forEach((arg, ai) => {
          const dir = pSpecs[ai]?.dir;
          if (arg?.buf != null && (dir === "out" || dir === "inout")) {
            targetSet.add(arg.buf);
          }
        });
      }
    }
    state.derivedConstSet = new Set();
    for (const bname of onceWritten) {
      if (!stepWritten.has(bname)) state.derivedConstSet.add(bname);
    }

    state.vars = {};
    for (const [vn, vd] of Object.entries(state.manifest.vars || {})) {
      const mx = Number(vd.max) || 1024;
      state.vars[vn] = vn === "tokens" ? Math.min(mx, 128) : Math.min(mx, 4);
    }

    initMemoryDefaults(state.manifest);
    renderTopbarVars();

    // Choose the most interesting default serving program (round -> decode -> prefill -> first)
    const pNames = Object.keys(state.manifest.programs || {});
    const preferred = ["decode", "round", "prefill", "prefill_stage0", "step"].find((p) => pNames.includes(p)) || pNames[0] || "";
    selectProgram(preferred);
  }

  // ==================== EVENT BINDINGS ====================
  function bindEvents() {
    // View switcher tabs
    document.querySelectorAll(".view-tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.activeView = btn.dataset.view;
        document.querySelectorAll(".view-tab").forEach((b) => b.classList.toggle("active", b === btn));
        document.getElementById("view-graph").classList.toggle("active", state.activeView === "graph");
        document.getElementById("view-memory").classList.toggle("active", state.activeView === "memory");
        document.getElementById("view-protocol").classList.toggle("active", state.activeView === "protocol");
        if (state.activeView === "graph") renderGraph(false);
        else if (state.activeView === "memory") renderMemoryDashboard();
        else renderProtocolDashboard();
      });
    });

    // Left sidebar subtabs
    document.querySelectorAll(".subtab").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.activeSubtab = btn.dataset.subtab;
        document.querySelectorAll(".subtab").forEach((b) => b.classList.toggle("active", b === btn));
        document.getElementById("subtab-stages").classList.toggle("active", state.activeSubtab === "stages");
        document.getElementById("subtab-buffers").classList.toggle("active", state.activeSubtab === "buffers");
      });
    });

    // Scope buttons
    document.querySelectorAll(".scope-btn").forEach((btn) => {
      btn.addEventListener("click", () => setScopeMode(btn.dataset.scope));
    });

    document.getElementById("slice-apply").addEventListener("click", () => {
      state.sliceStart = Number(document.getElementById("slice-start").value) || 0;
      state.sliceEnd = Number(document.getElementById("slice-end").value) || 28;
      renderGraph(true);
    });

    document.getElementById("stage-filter").addEventListener("input", renderSidebarStages);
    document.getElementById("buffer-filter").addEventListener("input", renderSidebarBuffers);

    // HUD toggles
    const bindToggle = (id, key) => {
      const el = document.getElementById(id);
      el.addEventListener("click", () => {
        state[key] = !state[key];
        el.classList.toggle("active", state[key]);
        renderGraph(false);
      });
    };
    bindToggle("tog-buffers", "showBuffers");
    bindToggle("tog-states", "showStates");
    bindToggle("tog-inputs", "showInputs");
    bindToggle("tog-weights", "showWeights");
    bindToggle("tog-fold", "foldRepeated");

    document.getElementById("orient-btn").addEventListener("click", () => {
      state.orient = state.orient === "TB" ? "LR" : "TB";
      document.getElementById("orient-label").textContent = state.orient === "TB" ? "↓ TB" : "→ LR";
      renderGraph(true);
    });

    document.getElementById("trace-mode").addEventListener("change", (e) => {
      state.traceMode = e.target.value;
      if (state.selectedNodeId) applyHighlightCone(state.selectedNodeId);
    });

    document.getElementById("canvas-search").addEventListener("input", (e) => {
      state.searchQuery = e.target.value;
      state.selectedNodeId = null;
      applySearchHighlight(state.searchQuery);
    });

    // Breadcrumb clicks
    document.getElementById("stage-breadcrumb").addEventListener("click", (e) => {
      const b = e.target.closest("[data-bc-action]");
      if (!b || b.disabled) return;
      const act = b.dataset.bcAction;
      if (act === "macro") setScopeMode("macro");
      else if (act === "prev") setScopeMode("stage", Math.max(0, state.focusedStageIdx - 1));
      else if (act === "next") setScopeMode("stage", Math.min(state.analysis.stages.length - 1, state.focusedStageIdx + 1));
    });

    // Theme toggle
    document.getElementById("theme-btn").addEventListener("click", toggleTheme);

    // Canvas Pan & Wheel Zoom
    let isDragging = false;
    let dragStart = { x: 0, y: 0 };
    let moved = false;

    canvasWrap.addEventListener("mousedown", (e) => {
      if (e.button !== 0) return;
      isDragging = true;
      moved = false;
      dragStart = { x: e.clientX - state.zoom.x, y: e.clientY - state.zoom.y };
      canvasWrap.classList.add("panning");
    });

    window.addEventListener("mousemove", (e) => {
      if (!isDragging) return;
      const nx = e.clientX - dragStart.x;
      const ny = e.clientY - dragStart.y;
      if (Math.abs(nx - state.zoom.x) > 3 || Math.abs(ny - state.zoom.y) > 3) moved = true;
      state.zoom.x = nx;
      state.zoom.y = ny;
      applyZoomTransform();
    });

    window.addEventListener("mouseup", () => {
      isDragging = false;
      canvasWrap.classList.remove("panning");
    });

    canvasWrap.addEventListener("wheel", (e) => {
      e.preventDefault();
      const rect = canvasWrap.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const factor = Math.exp(-e.deltaY * 0.0014);
      const nextK = Math.max(0.08, Math.min(4.0, state.zoom.k * factor));
      state.zoom.x = mx - ((mx - state.zoom.x) / state.zoom.k) * nextK;
      state.zoom.y = my - ((my - state.zoom.y) / state.zoom.k) * nextK;
      state.zoom.k = nextK;
      applyZoomTransform();
    }, { passive: false });

    // Canvas Click & Double-Click Delegation
    svgEl.addEventListener("click", (e) => {
      if (moved) return;

      const collapsePill = e.target.closest("[data-collapse-stage]");
      if (collapsePill) {
        const idx = Number(collapsePill.dataset.collapseStage);
        state.expandedStageIdxs.delete(idx);
        renderSidebarStages();
        renderGraph(false);
        return;
      }

      const focusBtn = e.target.closest("[data-focus-stage]");
      if (focusBtn) {
        e.stopPropagation();
        const idx = Number(focusBtn.dataset.focusStage);
        const st = state.analysis?.stages[idx];
        if (st && st.callIndices.length > 1) {
          setScopeMode("stage", idx);
        }
        return;
      }

      const unfoldBtn = e.target.closest("[data-unfold-fam]");
      if (unfoldBtn) {
        e.stopPropagation();
        state.unfoldedFamilyKeys.add(unfoldBtn.dataset.unfoldFam);
        renderGraph(true);
        return;
      }

      const expandBtn = e.target.closest("[data-expand-stage]");
      if (expandBtn) {
        e.stopPropagation();
        const idx = Number(expandBtn.dataset.expandStage);
        setScopeMode("stage", idx);
        return;
      }

      const nodeEl = e.target.closest(".g-node");
      if (nodeEl) {
        const nid = nodeEl.dataset.nodeId;
        state.selectedNodeId = nid;
        applyHighlightCone(nid);
        if (nodeEl.dataset.callIdx != null) {
          state.selected = { type: "call", callIdx: Number(nodeEl.dataset.callIdx) };
        } else if (nodeEl.dataset.rk != null) {
          state.selected = { type: "resource", rk: nodeEl.dataset.rk };
        } else if (nodeEl.dataset.foldNode != null) {
          state.selected = { type: "fold", nodeId: nodeEl.dataset.foldNode };
        } else if (nodeEl.dataset.groupNode != null) {
          state.selected = { type: "group_pill", nodeId: nodeEl.dataset.groupNode };
        } else if (nodeEl.dataset.stageIdx != null) {
          state.selected = { type: "stage", stageIdx: Number(nodeEl.dataset.stageIdx) };
        }
        renderInspector();
        return;
      }

      const edgeEl = e.target.closest(".g-edge");
      if (edgeEl) {
        const ei = Number(edgeEl.dataset.edgeIdx);
        const edge = state.layoutResult?.edges[ei];
        if (edge && (edge.rk.startsWith("buf:") || edge.rk.startsWith("state:"))) {
          state.selected = { type: "resource", rk: edge.rk };
          renderInspector();
        }
        return;
      }

      // Clicked empty canvas -> clear selection
      state.selectedNodeId = null;
      state.selected = null;
      clearHighlights();
      renderInspector();
    });

    svgEl.addEventListener("dblclick", (e) => {
      const stageEl = e.target.closest(".node-stage, .node-fold");
      if (stageEl && stageEl.dataset.stageIdx != null) {
        setScopeMode("stage", Number(stageEl.dataset.stageIdx));
      }
    });

    // Zoom bar & Minimap click navigation
    document.getElementById("zoom-in").addEventListener("click", () => zoomBy(1.25));
    document.getElementById("zoom-out").addEventListener("click", () => zoomBy(0.8));
    document.getElementById("zoom-fit").addEventListener("click", fitGraphToScreen);

    document.getElementById("minimap-wrap").addEventListener("mousedown", (e) => {
      const lay = state.layoutResult;
      if (!lay) return;
      const r = minimapSvg.getBoundingClientRect();
      const rx = (e.clientX - r.left) / r.width;
      const ry = (e.clientY - r.top) / r.height;
      const gx = rx * lay.width;
      const gy = ry * lay.height;
      const cv = canvasWrap.getBoundingClientRect();
      state.zoom.x = cv.width / 2 - gx * state.zoom.k;
      state.zoom.y = cv.height / 2 - gy * state.zoom.k;
      applyZoomTransform();
    });

    // Example reset & Local JSON upload / drag-and-drop
    document.getElementById("reset-example-btn").addEventListener("click", () => {
      loadExampleManifest();
    });
    document.getElementById("upload-btn").addEventListener("click", () => {
      document.getElementById("file-input").click();
    });
    document.getElementById("file-input").addEventListener("change", (e) => {
      const f = e.target.files?.[0];
      if (f) readUploadedManifest(f);
      e.target.value = "";
    });

    const dropOverlay = document.getElementById("drop-overlay");
    window.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropOverlay.classList.remove("hidden");
    });
    dropOverlay.addEventListener("dragleave", () => dropOverlay.classList.add("hidden"));
    window.addEventListener("drop", (e) => {
      e.preventDefault();
      dropOverlay.classList.add("hidden");
      const f = e.dataTransfer?.files?.[0];
      if (f) readUploadedManifest(f);
    });

    // Keyboard shortcuts
    window.addEventListener("keydown", (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") {
        if (e.key === "Escape") e.target.blur();
        return;
      }
      if (e.key === "/") {
        e.preventDefault();
        document.getElementById("canvas-search").focus();
      } else if (e.key === "f" || e.key === "F") {
        fitGraphToScreen();
      } else if (e.key === "t" || e.key === "T") {
        toggleTheme();
      } else if (e.key === "Escape") {
        state.selectedNodeId = null;
        state.selected = null;
        clearHighlights();
        renderInspector();
      }
    });
  }

  function zoomBy(factor) {
    const rect = canvasWrap.getBoundingClientRect();
    const cx = rect.width / 2;
    const cy = rect.height / 2;
    const nextK = Math.max(0.08, Math.min(4.0, state.zoom.k * factor));
    state.zoom.x = cx - ((cx - state.zoom.x) / state.zoom.k) * nextK;
    state.zoom.y = cy - ((cy - state.zoom.y) / state.zoom.k) * nextK;
    state.zoom.k = nextK;
    applyZoomTransform();
  }

  function toggleTheme() {
    const html = document.documentElement;
    const order = ["light", "dusk", "ink"];
    const names = { light: "Paper", dusk: "Dusk", ink: "Ink" };
    const cur = html.dataset.theme || "light";
    const next = order[(order.indexOf(cur) + 1) % order.length];
    html.dataset.theme = next;
    const lbl = document.getElementById("theme-label");
    if (lbl) lbl.textContent = names[next] || "Paper";
  }

  function resolveClientConstants(obj, constants) {
    if (!constants) return obj;
    if (typeof obj === "string") return constants[obj] !== undefined ? constants[obj] : obj;
    if (Array.isArray(obj)) return obj.map((x) => resolveClientConstants(x, constants));
    if (obj && typeof obj === "object") {
      const out = {};
      for (const [k, v] of Object.entries(obj)) out[k] = resolveClientConstants(v, constants);
      return out;
    }
    return obj;
  }

  const EXAMPLE_NAME = "qwen3.8-27b-dflash2.json";
  const EXAMPLE_URL = "/studio/qwen3.8-27b-dflash2.json";

  function parseAndLoadManifestText(rawText, name, isCustom) {
    const data = JSON.parse(rawText);
    const constants = data.constants || null;
    delete data.constants;
    if (constants) {
      for (const sec of ["vars", "states", "buffers", "ops", "programs", "topology"]) {
        if (data[sec]) data[sec] = resolveClientConstants(data[sec], constants);
      }
    }
    const nameEl = document.getElementById("artifact-name");
    if (nameEl) nameEl.textContent = name;
    const resetBtn = document.getElementById("reset-example-btn");
    if (resetBtn) resetBtn.classList.toggle("hidden", !isCustom);
    state.currentPresetId = name;
    loadManifestData({ meta: { name, size_bytes: rawText.length }, manifest: data });
  }

  function readUploadedManifest(file) {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        parseAndLoadManifestText(String(reader.result), file.name, true);
      } catch (err) {
        alert("Invalid manifest JSON: " + err.message);
      }
    };
    reader.readAsText(file);
  }

  async function loadExampleManifest(customUrl) {
    const url = customUrl || EXAMPLE_URL;
    const label = customUrl ? (customUrl.split("/").pop() || "manifest.json") : EXAMPLE_NAME;
    try {
      document.getElementById("graph-telemetry").textContent = "Loading manifest…";
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const text = await res.text();
      parseAndLoadManifestText(text, label, Boolean(customUrl));
    } catch (err) {
      document.getElementById("graph-telemetry").textContent = `Load error: ${err.message}`;
    }
  }

  async function init() {
    bindEvents();
    const params = new URLSearchParams(window.location.search);
    const externalUrl = params.get("url");
    await loadExampleManifest(externalUrl || null);
  }

  init();
})();
