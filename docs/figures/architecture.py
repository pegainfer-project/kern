"""The README architecture figure: python3 docs/figures/architecture.py writes architecture.svg beside it."""
from pathlib import Path
W, H = 960, 464
INK, BODY, MUTED, LINE, SOFT = "#0f172a", "#475569", "#64748b", "#94a3b8", "#e2e8f0"
ACC, ACC_INK, ACC_BG = "#ea580c", "#9a3412", "#fff7ed"
out = []
def el(s): out.append(s)
def text(x, y, s, size=13, weight=400, fill=BODY, anchor="start", spacing=None):
    ls = f' letter-spacing="{spacing}"' if spacing else ""
    el(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{ls}>{s}</text>')
def rect(x, y, w, h, r=10, fill="#ffffff", stroke=SOFT, width=1.5, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    el(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"{d}/>')
def card(x, y, w, h, title, lines, tsize=15):
    rect(x, y, w, h)
    cx = x + w / 2
    text(cx, y + h / 2 - 8 * len(lines) + 4, title, tsize, 600, INK, "middle")
    for i, l in enumerate(lines):
        text(cx, y + h / 2 - 8 * len(lines) + 24 + 17 * i, l, 12.5, 400, BODY, "middle")
def arrow(d, end):
    ex, ey = end
    el(f'<path d="{d}" fill="none" stroke="{MUTED}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>')
    el(f'<path d="M{ex-8},{ey-5} L{ex},{ey} L{ex-8},{ey+5}" fill="none" stroke="{MUTED}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>')

el(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Inter, -apple-system, \'Segoe UI\', Helvetica, Arial, sans-serif">')
el(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
el('<g transform="translate(0,-24)">')

# the package
rect(32, 56, 232, 400, 14, ACC_BG, ACC, 1.5, "6 5")
text(52, 86, "YOUR PACKAGE", 11.5, 700, ACC_INK, spacing="1.2")
for i, (t, ls) in enumerate([("manifest.json", ["typed program: buffers,", "states, ops, programs"]),
                             ("kernels/", ["any compiled code,", "pinned by SHA-256"]),
                             ("checkpoint", ["Hugging Face weights,", "bound tensor by tensor"])]):
    card(52, 104 + 116 * i, 192, 96, t, ls)

# verify
arrow("M264,256 H310", (310, 256))
rect(316, 204, 164, 104, 12, "#1e293b", "#1e293b")
text(398, 244, "verify", 16, 600, "#ffffff", "middle")
text(398, 266, "kern-manifest", 12.5, 400, "#cbd5e1", "middle")
text(398, 284, "every diagnostic, in ms", 12.5, 400, "#cbd5e1", "middle")

# fan out
arrow("M480,256 H506 V172 H526", (526, 172))
arrow("M480,256 H506 V390 H526", (526, 390))

# run
rect(532, 56, 396, 248, 14, "#f8fafc", SOFT)
text(552, 86, "RUN", 11.5, 700, MUTED, spacing="1.2")
card(552, 100, 176, 96, "kern-runtime", ["allocate · lower", "replay as CUDA graphs"])
card(744, 100, 164, 96, "kern-pool", ["pages · slots · leases", "prefix · host tier"])
el(f'<path d="M728,148 H744" stroke="{LINE}" stroke-width="2"/>')
text(552, 224, "HOSTED IN", 10.5, 700, MUTED, spacing="1.2")
for i, h in enumerate(["kern-serve", "vLLM", "SGLang"]):
    x = 552 + 122 * i
    rect(x, 236, 112, 48, 8, "#f1f5f9", SOFT)
    text(x + 56, 265, h, 13.5, 600, "#1e293b", "middle")

# harness
rect(532, 324, 396, 132, 14, "#f8fafc", SOFT)
text(552, 354, "HARNESS", 11.5, 700, MUTED, spacing="1.2")
for i, (t, l) in enumerate([("kern test", "A vs B, verdict"), ("kern bench", "time per call"), ("kern cut", "pipeline stages")]):
    card(552 + 122 * i, 366, 112, 72, t, [l], 14)

el('</g>')
el('</svg>')
(Path(__file__).parent / "architecture.svg").write_text("\n".join(out) + "\n")
