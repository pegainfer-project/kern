"""The README pool figure: python3 docs/figures/pool.py writes pool.svg beside it."""
from pathlib import Path
W, H = 960, 476
INK, BODY, MUTED, LINE, SOFT = "#0f172a", "#475569", "#64748b", "#94a3b8", "#e2e8f0"
ACC, ACC_INK, ACC_FILL = "#ea580c", "#9a3412", "#fed7aa"
DARK, BAND = "#1e293b", "#f1f5f9"
out = []
def el(s): out.append(s)
def text(x, y, s, size=13, weight=400, fill=BODY, anchor="start", spacing=None, mono=False):
    ls = f' letter-spacing="{spacing}"' if spacing else ""
    ff = ' font-family="ui-monospace, SFMono-Regular, Menlo, monospace"' if mono else ""
    el(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{ls}{ff}>{s}</text>')
def rect(x, y, w, h, r=6, fill="#ffffff", stroke=SOFT, width=1.5, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    el(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"{d}/>')
def line(d, stroke=MUTED, width=2, dash=None):
    s = f' stroke-dasharray="{dash}"' if dash else ""
    el(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"{s}/>')
def down(x, y0, y1, stroke=ACC):
    line(f"M{x},{y0} V{y1}", stroke, 2.5)
    line(f"M{x-6},{y1-9} L{x},{y1} L{x+6},{y1-9}", stroke, 2.5)
def pill(cx, cy, s, fill="#ffffff", ink=MUTED):
    w = 7 * len(s) + 16
    rect(cx - w / 2, cy - 10, w, 20, 10, fill, fill, 0)
    text(cx, cy + 4, s, 11.5, 600, ink, "middle")

BW, BH, GAP, X0 = 60, 40, 12, 444
SX, SW = 820, 108
def px(i): return X0 + (BW + GAP) * i
def mid(i): return px(i) + BW / 2
SMID = SX + SW / 2

STYLE = {"lease": (ACC_FILL, ACC, ACC_INK), "frozen": (DARK, DARK, "#ffffff")}
def obj(x, y, w, label, style, part=1.0):
    if style in ("empty", "freed"):
        stroke, ink, weight = (ACC, ACC_INK, 600) if style == "empty" else (LINE, MUTED, 400)
        rect(x, y, w, BH, fill="#ffffff", stroke=stroke, dash="4 4")
        text(x + w / 2, y + 25, label, 12.5, weight, ink, "middle")
        return
    fill, stroke, ink = STYLE[style]
    if part >= 1:
        rect(x, y, w, BH, fill=fill, stroke=stroke)
        text(x + w / 2, y + 25, label, 12.5, 600, ink, "middle")
        return
    rect(x, y, w, BH, fill="#ffffff", stroke=stroke)
    el(f'<path d="M{x+6},{y} H{x+w*part} V{y+BH} H{x+6} A6,6 0 0 1 {x},{y+BH-6} V{y+6} A6,6 0 0 1 {x+6},{y} Z" fill="{fill}"/>')
    rect(x, y, w, BH, fill="none", stroke=stroke)
    text(x + w * part / 2, y + 25, label, 12.5, 600, ink, "middle")
def swatch(x, y, style):
    if style == "empty":
        rect(x, y, 16, 16, 4, "#ffffff", ACC, 1.5, "3 3")
    else:
        rect(x, y, 16, 16, 4, STYLE[style][0], STYLE[style][1])

el(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Inter, -apple-system, \'Segoe UI\', Helvetica, Arial, sans-serif">')
el(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

text(32, 42, "A and B share a prompt prefix", 19, 700, INK)
text(32, 64, "What kern-pool does with their KV memory, in three steps.", 12.5, 400, BODY)
x = 32
for style, label in [("lease", "written by its lease"), ("frozen", "frozen: read-only, shareable"), ("empty", "leased, still empty")]:
    swatch(x, 84, style)
    text(x + 22, 97, label, 11.5, 400, BODY)
    x += 22 + 7 * len(label) + 20

Y0, RH = 148, 108
def ry(r): return Y0 + RH * r
ROWS = [
    ("lease", "LEASE A", "A takes pages for its worst case", "plus one slot, all or nothing."),
    ("retire", "CHECKPOINT", "A ends inside p3. Nothing moves:", "its pages freeze where they are."),
    ("restore", "LEASE B", "B starts at the end of A's prefix", "and keeps writing after it."),
]
for r, (op, who, c1, c2) in enumerate(ROWS):
    y = ry(r)
    text(32, y + 36, str(r + 1), 38, 700, LINE)
    text(76, y + 13, op, 15, 700, INK, mono=True)
    text(76, y + 33, c1, 12.5)
    text(76, y + 50, c2, 12.5)
    text(X0 - 16, y + 25, who, 11.5, 700, MUTED, "end", "1.2")

y = ry(0) - 12
line(f"M{px(0)},{y + 4} V{y} H{px(0) + BW} V{y + 4}", LINE, 1.5)
text(px(0), y - 8, "a page = 784 tokens of KV", 11.5, 400, BODY)
line(f"M{SX},{y + 4} V{y} H{SX + SW} V{y + 4}", LINE, 1.5)
text(SX + SW, y - 8, "a slot = one sequence's state", 11.5, 400, BODY, "end")

g1, g2 = ry(0) + BH, ry(1) + BH
band = (px(0) - 8, ry(1) - 8, px(2) + BW - px(0) + 16, RH + BH + 16)
rect(*band, 12, BAND, "#cbd5e1", 1.5)

for i in range(4): line(f"M{mid(i)},{g1 + 4} V{ry(1) - 4}", LINE, 1.5, "2 4")
line(f"M{SMID},{g1 + 4} V{ry(1) - 4}", LINE, 1.5, "2 4")
pill((mid(0) + mid(3)) / 2, (g1 + ry(1)) / 2, "same pages, frozen in place")
pill(SMID, (g1 + ry(1)) / 2, "handed over")
pill(mid(4), (g1 + ry(1)) / 2, "to the pool")

for i in range(3): line(f"M{mid(i)},{g2 + 4} V{ry(2) - 4}", LINE, 1.5, "2 4")
pill((mid(0) + mid(2)) / 2, (g2 + ry(2)) / 2, "shared, zero copy", BAND, INK)
for cx in (mid(3), SMID):
    down(cx, g2 + 5, ry(2) - 5)
    text(cx + 10, (g2 + ry(2)) / 2 + 4, "copy", 12.5, 700, ACC_INK)

y = ry(0)
for i in range(3): obj(px(i), y, BW, f"p{i}", "lease")
obj(px(3), y, BW, "p3", "lease", 0.5)
obj(px(4), y, BW, "p4", "empty")
obj(SX, y, SW, "slot s1", "lease")

y = ry(1)
for i in range(3): obj(px(i), y, BW, f"p{i}", "frozen")
obj(px(3), y, BW, "p3", "frozen", 0.5)
obj(px(4), y, BW, "freed", "freed")
obj(SX, y, SW, "slot s1", "frozen")

y = ry(2)
for i in range(3): obj(px(i), y, BW, f"p{i}", "frozen")
obj(px(3), y, BW, "p5", "lease", 0.5)
obj(px(4), y, BW, "p6", "empty")
obj(SX, y, SW, "slot s2", "lease")

y = ry(2) + BH + 32
line(f"M32,{y} H928", SOFT, 1)
text(32, y + 30, "THE RULE", 11.5, 700, ACC_INK, spacing="1.2")
text(116, y + 30, "Whole pages are shared, the partial page is copied, and only a live lease writes.", 14.5, 600, INK)

el('</svg>')
(Path(__file__).parent / "pool.svg").write_text("\n".join(out) + "\n")
