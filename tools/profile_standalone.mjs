#!/usr/bin/env node
// Package the same explorer and evidence as one offline, portable HTML file.
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { gzipSync, gunzipSync } from "node:zlib";
import { build } from "../website/node_modules/esbuild/lib/main.js";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const { values } = parseArgs({ options: {
  data: { type: "string", default: resolve(root, "website/public/perf/data") },
  out: { type: "string" },
} });
if (!values.out) throw Error("Usage: node tools/profile_standalone.mjs --out demo.html [--data exported-data-directory]");
const catalog = JSON.parse(await readFile(resolve(values.data, "index.json"), "utf8"));
const evidence = { "index.json": catalog };
for (const item of catalog) {
  for (const name of [item.file, item.quick]) {
    if (name.includes("/") || name.includes("\\") || name === "..") throw Error("Unsafe catalog filename");
    const bytes = await readFile(resolve(values.data, name));
    evidence[name] = JSON.parse((name.endsWith(".gz") ? gunzipSync(bytes) : bytes).toString());
  }
}
const result = await build({
  entryPoints: [resolve(root, "website/src/perf/main.tsx")],
  bundle: true, write: false, minify: true, format: "iife", jsx: "automatic",
  target: "es2020", outfile: "atlas.js", define: { "process.env.NODE_ENV": '"production"' },
});
const js = result.outputFiles.find(f => f.path.endsWith(".js")).text;
// Remote web fonts are optional in the hosted site; offline has no network dependencies.
const css = result.outputFiles.find(f => f.path.endsWith(".css")).text.replace(/@import\s*(?:url\([^)]*\)|"[^"]*"|'[^']*')\s*;/g, "");
if (css.includes("@import")) throw Error("Standalone CSS must not import network resources");
const packed = gzipSync(JSON.stringify(evidence), { level: 9 }).toString("base64");
const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>kern · Single-GPU Performance Atlas</title><style>${css}</style></head><body><div id="root"></div><script id="perf-evidence-gzip" type="application/octet-stream">${packed}</script><script>${js.replace(/<\/script/gi, "<\\/script")}</script></body></html>`;
await mkdir(dirname(resolve(values.out)), { recursive: true });
await writeFile(values.out, html);
console.log(JSON.stringify({ models: catalog.length, bytes: Buffer.byteLength(html), offline: true }));
