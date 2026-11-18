---
layout: false
title: kern
---

<script setup>
import { onMounted } from "vue";
import { withBase } from "vitepress";

// `/docs/` is the quick start. The site's `_redirects` answers a request for
// it; this covers a host without that file, such as the dev server.
onMounted(() => window.location.replace(withBase("/getting-started/")));
</script>
