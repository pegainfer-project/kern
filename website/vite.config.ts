import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

const here = dirname(fileURLToPath(import.meta.url));
const schemaFile = resolve(here, "../schema/manifest-v4.schema.json");

// Serve/emit the golden wire-format schema (the file CI golden-checks) at
// /schema/manifest-v4.schema.json — the URL its $id declares.
function emitManifestSchema(): Plugin {
  return {
    name: "emit-manifest-schema",
    buildStart() {
      this.addWatchFile(schemaFile);
    },
    generateBundle() {
      this.emitFile({
        type: "asset",
        fileName: "schema/manifest-v4.schema.json",
        source: readFileSync(schemaFile, "utf8"),
      });
    },
    configureServer(server) {
      server.middlewares.use("/schema/manifest-v4.schema.json", (_req, res) => {
        res.setHeader("Content-Type", "application/schema+json");
        res.end(readFileSync(schemaFile, "utf8"));
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), emitManifestSchema()],
  server: {
    fs: { allow: [".."] },
  },
  build: {
    rollupOptions: {
      input: {
        main: resolve(here, "index.html"),
        schema: resolve(here, "schema/index.html"),
        qwen38: resolve(here, "qwen38/index.html"),
      },
    },
  },
});
