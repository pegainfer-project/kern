"""The README pool budget figure: python3 docs/figures/pool_budget.py writes pool_budget.svg beside it."""
from pathlib import Path
W, H = 960, 466
INK, BODY, MUTED, LINE, SOFT = "#0f172a", "#475569", "#64748b", "#94a3b8", "#e2e8f0"
ACC, ACC_INK, ACC_FILL, ACC_BG = "#ea580c", "#9a3412", "#fed7aa", "#fff7ed"
DARK, FAINT = "#1e293b", "#cbd5e1"
out = []
def el(s): out.append(s)
def text(x, y, s, size=13, weight=400, fill=BODY, anchor="start", spacing=None):
    ls = f' letter-spacing="{spacing}"' if spacing else ""
    el(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{ls}>{s}</text>')
def rect(x, y, w, h, r=5, fill="#ffffff", stroke=SOFT, width=1.5, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    el(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"{d}/>')
def line(d, stroke=MUTED, width=2, dash=None):
    s = f' stroke-dasharray="{dash}"' if dash else ""
    el(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{width}" stroke-linecap="round"{s}/>')

P, BW, BH = 44, 38, 34
def cx(x0, i): return x0 + P * i + P / 2

def slot(x, y, label, kind):
    fill, stroke, ink, dash = {"held": (DARK, DARK, "#ffffff", None), "new": (ACC, ACC, "#ffffff", None),
                               "void": ("none", FAINT, MUTED, "4 3")}[kind]
    rect(x - BW / 2, y, BW, BH, 5, fill, stroke, 1.5, dash)
    if label: text(x, y + BH / 2 + 5, label, 13, 700, ink, "middle")

def pages(x, y, kind):
    if kind in ("void", "gone"):
        rect(x - BW / 2, y, BW, BH, 5, "none", ACC if kind == "gone" else FAINT, 1.5, "4 3")
        return
    rect(x - BW / 2, y, BW, BH, 5, "#ffffff", LINE, 1.5)
    for k in range(4):
        rect(x - BW / 2 + 3 + 8 * k, y + 4, 6, BH - 8, 1.5, LINE if kind == "held" else "#ffffff", LINE, 1)

def chunk(x, y, moved=False):
    rect(x - BW / 2, y, BW, 26, 4, ACC_BG if moved else "#f1f5f9", ACC if moved else LINE, 2 if moved else 1.5)

el(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Inter, -apple-system, \'Segoe UI\', Helvetica, Arial, sans-serif">')
el(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

# a static split: same twelve chunks of memory, the line drawn once
X0, Y0 = 40, 50
text(X0, 34, "A STATIC SPLIT  ·  FIXED AT STARTUP", 11.5, 700, MUTED, spacing="1.2")
for i in range(5): slot(cx(X0, i), Y0, str(i + 1), "held")
for i, k in enumerate(["free"] * 4 + ["held"] * 3): pages(cx(X0, 5 + i), Y0, k)
fx = X0 + 5 * P
line(f"M{fx},{Y0 - 8} V{Y0 + BH + 8}", INK, 3)
text(cx(X0, 2), Y0 + BH + 20, "5 state slots", 12, 600, BODY, "middle")
text(cx(X0, 8), Y0 + BH + 20, "paged KV region, 4 of 7 chunks free", 12, 600, BODY, "middle")
rx = 612
rect(rx, Y0, BW, BH, 5, "#ffffff", ACC_INK, 2, "5 4")
text(rx + BW / 2, Y0 + BH / 2 + 5, "#6", 13, 700, ACC_INK, "middle")
text(rx + BW + 14, Y0 + 15, "waits: every slot is taken", 13, 700, ACC_INK)
text(rx + BW + 14, Y0 + 33, "while the KV region sits half empty", 12.5, 400, BODY)

el(f'<path d="M40,{Y0 + 80} H920" stroke="{SOFT}" stroke-width="1.5"/>')

# kern: virtual ranges reserved once, one pool of physical chunks under them
SX, PX, VY, CY = 40, 428, 220, 324
text(SX, 160, "KERN  ·  VIRTUAL RANGES RESERVED ONCE, ONE POOL OF CHUNKS UNDER THEM", 11.5, 700, MUTED, spacing="1.2")
text(SX, VY - 34, "slots: per-sequence state", 12.5, 600, INK)
text(cx(SX, 5) - BW / 2, VY - 12, "#6 and #7 run", 12.5, 700, ACC_INK)
text(PX, VY - 34, "pages: paged KV", 12.5, 600, INK)
for i in range(8): slot(cx(SX, i), VY, str(i + 1) if i < 7 else "", "held" if i < 5 else "new" if i < 7 else "void")
for i, k in enumerate(["gone"] * 2 + ["free"] * 2 + ["held"] * 3 + ["void"] * 4): pages(cx(PX, i), VY, k)
text(cx(PX, 9), VY + BH + 18, "unmapped address space", 12, 400, MUTED, "middle")
text(cx(PX, 0) - BW / 2, VY - 12, "was 2 free pages", 12.5, 700, ACC_INK)

tops, bots = VY + BH + 2, CY - 2
for i in range(5): line(f"M{cx(SX, i)},{bots} V{tops}", LINE, 1.5)
for i in range(5): line(f"M{cx(SX, 7 + i)},{bots} L{cx(PX, 2 + i)},{tops}", LINE, 1.5)
for i in range(2):
    line(f"M{cx(SX, 5 + i)},{bots} L{cx(PX, i)},{tops}", ACC_FILL, 2, "5 4")
    line(f"M{cx(SX, 5 + i)},{bots} V{tops}", ACC, 2.5)
for i in range(12): chunk(cx(SX, i), CY, i in (5, 6))
lx = cx(SX, 11) + BW / 2 + 18
text(lx, CY + 11, "physical chunks", 12.5, 600, INK)
text(lx, CY + 28, "one pool, up to 64 MiB each", 12, 400, BODY)
bx0, bx1 = cx(SX, 5) - BW / 2, cx(SX, 6) + BW / 2
line(f"M{bx0},{CY + 34} V{CY + 40} H{bx1} V{CY + 34}", ACC, 2)
text(cx(SX, 5) + P / 2, CY + 60, "remap: no bytes copied, off the serving thread", 13, 700, ACC_INK, "middle")

# legend and the takeaway
LY = 410
el(f'<path d="M40,{LY - 22} H920" stroke="{SOFT}" stroke-width="1.5"/>')
def key(x, draw, label):
    draw(x)
    text(x + 16, LY + 5, label, 12, 400, BODY)
key(40, lambda x: rect(x, LY - 7, 12, 12, 2, DARK, DARK), "held slot")
key(128, lambda x: rect(x, LY - 7, 12, 12, 2, ACC, ACC), "slot made by a remap")
key(285, lambda x: rect(x, LY - 7, 12, 12, 2, LINE, LINE, 1), "held page")
key(375, lambda x: rect(x, LY - 7, 12, 12, 2, "#ffffff", LINE, 1), "free page")
key(462, lambda x: rect(x, LY - 7, 12, 12, 2, "none", FAINT, 1.5, "3 2"), "unmapped")
key(554, lambda x: rect(x, LY - 7, 12, 12, 2, "#f1f5f9", LINE, 1.5), "chunk")
key(624, lambda x: line(f"M{x},{LY - 1} H{x + 12}", LINE, 1.5), "backs")
key(692, lambda x: line(f"M{x},{LY - 1} H{x + 12}", ACC_FILL, 2, "4 3"), "backed before the remap")
text(40, LY + 40, "One budget, no fixed line: pages and slots trade memory on demand.", 14, 700, INK)

el('</svg>')
(Path(__file__).parent / "pool_budget.svg").write_text("\n".join(out) + "\n")
