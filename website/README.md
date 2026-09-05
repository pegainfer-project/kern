# Kern website

Static product page for Kern: models ship as verified GPU programs.

Production: <https://kern-baa.pages.dev>

## Local development

Requires Node.js 18 or newer.

```sh
cd website
npm ci
npm run dev
```

Create and inspect the production bundle:

```sh
npm run build
npm run preview
```

The production output is written to `dist/`.

## Performance explorer

The measured single-GPU explorer lives at `/perf/`. It loads portable evidence
from `public/perf/data/`; there are no invented benchmark values. See
[measurement and reproduction notes](../docs/performance-atlas.md) for the
manifest-driven runner, cache protocol, uncertainty limits, AI quick view
and standalone offline HTML export.

The page distinguishes measured program/trace data from calibrated op-cost
estimates and hypothetical savings. Report collection currently loads and
executes the model; viewing an exported report needs neither the model nor a
GPU. `npm run build` also produces `dist/perf/offline.html`, containing the
same explorer and evidence without network dependencies.

## Cloudflare Pages

The existing Cloudflare Pages project is named `kern`. Publish the current
production bundle with:

```sh
npm run build
wrangler pages deploy dist --project-name kern --branch master
```

Pushes to `master` that change `website/**` deploy through
`.github/workflows/deploy-website.yml`. The workflow accepts either the
`CLOUDFLARE_API_TOKEN` or `CF_API_TOKEN` Actions secret and reads the Cloudflare
account ID from the `CLOUDFLARE_ACCOUNT_ID` Actions variable.

The site can also deploy directly from this repository without a Worker or a
local Wrangler installation. Connect the repository in the Cloudflare
dashboard with these settings:

| Setting | Value |
| --- | --- |
| Repository | `pegainfer-project/kern` |
| Production branch | `master` |
| Root directory | `website` |
| Build command | `npm run build` |
| Build output directory | `dist` |

In the Cloudflare dashboard, create a Pages application, import the GitHub
repository, and enter the settings above. Pages will build from `website/` and
publish new commits to `master` automatically. Pull requests receive preview
deployments.

## Content boundaries

- Product structure and counts come from the checked-in Qwen3-4B manifests.
- Performance figures reproduce repository measurements and keep their test
  conditions visible.
- The TileFoundry section describes conceptual alignment. There is no direct
  TileFoundry-to-Kern exporter today.
- The `<3K` source metric counts production Rust files in `kern-manifest` and
  `kern-runtime`, including `kern run`; it excludes tests and tools.
