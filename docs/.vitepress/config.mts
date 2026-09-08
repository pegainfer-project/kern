import { defineConfig } from "vitepress";

export default defineConfig({
  srcDir: "site",
  outDir: "../website/dist/docs",
  base: "/docs/",
  title: "kern",
  description: "Ship models as verified GPU programs.",
  cleanUrls: true,
  lastUpdated: true,
  metaChunk: true,
  sitemap: {
    hostname: "https://kern-baa.pages.dev/docs/",
  },
  head: [
    ["meta", { name: "theme-color", content: "#0647ff" }],
    ["link", { rel: "icon", href: "/docs/favicon.svg", type: "image/svg+xml" }],
  ],
  markdown: {
    lineNumbers: true,
  },
  themeConfig: {
    siteTitle: "KERN■",
    nav: [
      { text: "Guide", link: "/getting-started/" },
      { text: "Concepts", link: "/concepts/artifact" },
      { text: "Reference", link: "/reference/cli" },
      { text: "Performance", link: "https://kern-baa.pages.dev/perf/" },
    ],
    sidebar: [
      {
        text: "Getting started",
        items: [
          { text: "Quick start", link: "/getting-started/" },
          { text: "Install", link: "/getting-started/install" },
        ],
      },
      {
        text: "Concepts",
        items: [
          { text: "The model artifact", link: "/concepts/artifact" },
          { text: "Runtime and verification", link: "/concepts/runtime" },
        ],
      },
      {
        text: "Guides",
        items: [
          { text: "Run a model", link: "/guides/run-a-model" },
          { text: "Test a kernel change", link: "/guides/test-a-kernel-change" },
        ],
      },
      {
        text: "Reference",
        items: [
          { text: "Command line", link: "/reference/cli" },
          { text: "kern.toml", link: "/reference/config" },
          { text: "Manifest schema", link: "/reference/manifest" },
        ],
      },
    ],
    search: {
      provider: "local",
    },
    outline: {
      level: [2, 3],
      label: "On this page",
    },
    editLink: {
      pattern: "https://github.com/pegainfer-project/kern/edit/master/docs/site/:path",
      text: "Edit this page on GitHub",
    },
    socialLinks: [
      { icon: "github", link: "https://github.com/pegainfer-project/kern" },
    ],
    footer: {
      message: "Models ship as verified GPU programs.",
      copyright: "Released under the Apache-2.0 License.",
    },
  },
});
