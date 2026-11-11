import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

const here = dirname(fileURLToPath(import.meta.url));
const schemaFile = resolve(here, "../schema/manifest-v7.schema.json");
const studioExampleFile = resolve(here, "../examples/qwen3.8-27b-dflash2.json");

// Serve/emit the golden wire-format schema (the file CI golden-checks) at
// /schema/manifest-v7.schema.json — the URL its $id declares — and the single
// Qwen3.8-27B example manifest for /studio/.
function emitManifestAssets(): Plugin {
  return {
    name: "emit-manifest-assets",
    buildStart() {
      this.addWatchFile(schemaFile);
      this.addWatchFile(studioExampleFile);
    },
    generateBundle() {
      this.emitFile({
        type: "asset",
        fileName: "schema/manifest-v7.schema.json",
        source: readFileSync(schemaFile, "utf8"),
      });
      this.emitFile({
        type: "asset",
        fileName: "studio/qwen3.8-27b-dflash2.json",
        source: readFileSync(studioExampleFile, "utf8"),
      });
    },
    configureServer(server) {
      server.middlewares.use("/schema/manifest-v7.schema.json", (_req, res) => {
        res.setHeader("Content-Type", "application/schema+json");
        res.end(readFileSync(schemaFile, "utf8"));
      });
      server.middlewares.use("/studio/qwen3.8-27b-dflash2.json", (_req, res) => {
        res.setHeader("Content-Type", "application/json");
        res.end(readFileSync(studioExampleFile, "utf8"));
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), emitManifestAssets()],
  server: {
    fs: { allow: [".."] },
  },
  build: {
    rolldownOptions: {
      input: {
        main: resolve(here, "index.html"),
        schema: resolve(here, "schema/index.html"),
        studio: resolve(here, "studio/index.html"),
        qwen38: resolve(here, "qwen38/index.html"),
        rsi: resolve(here, "rsi/index.html"),
      },
    },
  },
});
