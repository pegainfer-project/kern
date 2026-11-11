"""ICML / modern research-post figure for 'Build a benchmark workload from live traffic'.

Core narrative: a 6-minute offline replay accurately predicts 7+ hours of online GPU execution.
Run `python3 docs/figures/trace-workload.py` to write `docs/figures/trace-workload.svg`.
"""

from pathlib import Path
import math
import random

W, H = 1220, 468

# Muted ICML / research-post palette
INK = "#0F172A"
SUBINK = "#1E293B"
BODY = "#334155"
MUTED = "#64748B"
RULE = "#CBD5E1"
HAIR = "#E2E8F0"
TINT_BG = "#F8FAFC"

TERRA = "#C2410C"
TERRA_MID = "#EA580C"
NAVY_BAR = "#2B5C8F"
SKY_BAR = "#6495ED"
SAGE = "#0F766E"
SAGE_BG = "#F0FDFA"
SAGE_BORDER = "#99F6E4"

SERIF = "'Nimbus Roman', 'Times New Roman', 'Liberation Serif', Georgia, serif"
SANS = "Inter, 'Liberation Sans', 'Nimbus Sans', -apple-system, 'Segoe UI', Arial, sans-serif"
MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, 'DejaVu Sans Mono', Menlo, monospace"

out: list[str] = []


def el(s: str) -> None:
    out.append(s)


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def rect(
    x: float,
    y: float,
    w: float,
    h: float,
    r: float = 8,
    fill: str = "#FFFFFF",
    stroke: str = RULE,
    sw: float = 1.0,
    dash: str | None = None,
) -> None:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    el(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r:.1f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'
    )


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    stroke: str = RULE,
    sw: float = 1.0,
    dash: str | None = None,
) -> None:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    el(
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{sw}"{d}/>'
    )


def arrow_h(x1: float, x2: float, y: float, color: str = INK, sw: float = 1.8) -> None:
    el(
        f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2 - 6:.1f}" y2="{y:.1f}" '
        f'stroke="{color}" stroke-width="{sw}" stroke-linecap="round"/>'
    )
    el(
        f'<polygon points="{x2:.1f},{y:.1f} {x2 - 8:.1f},{y - 4.3:.1f} {x2 - 8:.1f},{y + 4.3:.1f}" '
        f'fill="{color}"/>'
    )


def txt(
    x: float,
    y: float,
    s: str,
    size: float = 11.5,
    weight: int | str = 400,
    fill: str = BODY,
    anchor: str = "start",
    family: str = SANS,
    italic: bool = False,
    spacing: float | None = None,
    raw: bool = False,
) -> None:
    st = ' font-style="italic"' if italic else ""
    ls = f' letter-spacing="{spacing}"' if spacing is not None else ""
    body = s if raw else esc(s)
    el(
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{st}{ls}>{body}</text>'
    )


def fmt_hm_cell(sec: int) -> str:
    if sec <= 0:
        return "·"
    mins = round(sec / 60.0)
    if mins < 1:
        return "<1m"
    return f"{mins}m"


# ---------------------------------------------------------------------------
# Canvas & Minimal Header
# ---------------------------------------------------------------------------
el(
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
    f'width="{W}" height="{H}">'
)
rect(0, 0, W, H, r=0, fill="#FFFFFF", stroke="none", sw=0)

txt(
    28,
    38,
    "Build a Benchmark Workload from Live Traffic",
    size=20.0,
    weight=700,
    fill=INK,
    family=SERIF,
)
txt(
    W - 28,
    37,
    'Qwen3.8-27B FP8  ·  <tspan fill="#0F172A" font-weight="700">7h 21m online → 6m 20s offline</tspan>',
    size=11.2,
    weight=500,
    fill=MUTED,
    anchor="end",
    family=MONO,
    raw=True,
)
line(28, 52, W - 28, 52, stroke=INK, sw=1.1)

# ---------------------------------------------------------------------------
# Three Airy Stage Cards
# ---------------------------------------------------------------------------
PY, PH = 68, 376
PW = 348
P1X = 28
P2X = 436
P3X = 844

for px, tag, title, cmd in [
    (P1X, "(a)", "Online Trace", "--trace"),
    (P2X, "(b)", "Shape × KV Buckets", "kern workload"),
    (P3X, "(c)", "Offline Replay", "kern bench"),
]:
    rect(px, PY, PW, PH, r=8, fill="#FFFFFF", stroke=RULE, sw=1.0)
    txt(px + 18, PY + 29, f"{tag} {title}", size=14.5, weight=700, fill=INK, family=SERIF)
    bw_p = len(cmd) * 6.2 + 16
    rect(px + PW - 16 - bw_p, PY + 12, bw_p, 22, r=5, fill=TINT_BG, stroke=HAIR, sw=1.0)
    txt(px + PW - 16 - bw_p / 2, PY + 26.5, cmd, size=9.5, weight=600, fill=SUBINK, anchor="middle", family=MONO)
    line(px + 16, PY + 44, px + PW - 16, PY + 44, stroke=HAIR, sw=1.0)

# Clean arrows between cards
MID_Y = PY + PH / 2
arrow_h(P1X + PW + 10, P2X - 10, MID_Y, color=INK, sw=1.8)
txt((P1X + PW + P2X) / 2, MID_Y - 10, "1,667×", size=10.0, weight=700, fill=INK, anchor="middle", family=MONO)
txt((P1X + PW + P2X) / 2, MID_Y + 16, "fold", size=9.5, fill=MUTED, anchor="middle", italic=True, family=SERIF)

arrow_h(P2X + PW + 10, P3X - 10, MID_Y, color=INK, sw=1.8)
txt((P2X + PW + P3X) / 2, MID_Y - 10, "70×", size=10.0, weight=700, fill=INK, anchor="middle", family=MONO)
txt((P2X + PW + P3X) / 2, MID_Y + 16, "faster", size=9.5, fill=MUTED, anchor="middle", italic=True, family=SERIF)

# ===========================================================================
# PANEL (a): Online Trace (Scatter + Minimal 2-Row Parquet Peek)
# ===========================================================================
AX_L, AX_R = P1X + 56, P1X + PW - 20
AX_T, AX_B = PY + 72, PY + 206

el(f'<circle cx="{AX_L + 4}" cy="{PY + 59}" r="3.6" fill="{TERRA_MID}"/>')
txt(AX_L + 13, PY + 63, "prefill", size=9.8, weight=500, fill=BODY)
el(f'<circle cx="{AX_L + 68}" cy="{PY + 59}" r="3.6" fill="{NAVY_BAR}"/>')
txt(AX_L + 77, PY + 63, "round", size=9.8, weight=500, fill=BODY)
txt(AX_R, PY + 63, "CUPTI HES · 0 sync", size=9.2, fill=MUTED, anchor="end", family=MONO)

rect(AX_L, AX_T, AX_R - AX_L, AX_B - AX_T, r=3, fill="#FCFDFE", stroke=HAIR, sw=0.9)

for frac, lbl in [(0.14, "1M"), (0.48, "128k"), (0.84, "16k")]:
    yy = AX_T + frac * (AX_B - AX_T)
    line(AX_L, yy, AX_R, yy, stroke=HAIR, sw=0.8, dash="3 3")
    txt(AX_L - 6, yy + 3.5, lbl, size=8.8, fill=MUTED, anchor="end", family=MONO)

el(
    f'<text x="{P1X + 18}" y="{(AX_T + AX_B) / 2:.1f}" font-family="{SANS}" font-size="9.2" '
    f'fill="{MUTED}" text-anchor="middle" transform="rotate(-90 {P1X + 18} {(AX_T + AX_B) / 2:.1f})">'
    f'Summed KV</text>'
)

rng = random.Random(42)
for _ in range(110):
    burst = rng.choice([0.08, 0.22, 0.38, 0.54, 0.70, 0.86, 0.94])
    tx = min(0.96, max(0.04, burst + rng.gauss(0, 0.055)))
    is_pf = rng.random() < 0.34
    if is_pf:
        ky = min(0.92, max(0.28, rng.gauss(0.58, 0.16)))
        col, rad, op = TERRA_MID, 2.6, "0.80"
    else:
        ky = min(0.94, max(0.09, rng.gauss(0.43, 0.22)))
        col, rad, op = NAVY_BAR, 2.3, "0.70"
    el(
        f'<circle cx="{AX_L + tx * (AX_R - AX_L):.1f}" '
        f'cy="{AX_T + ky * (AX_B - AX_T):.1f}" r="{rad}" fill="{col}" fill-opacity="{op}"/>'
    )

line(AX_L, AX_T, AX_L, AX_B, stroke=SUBINK, sw=1.0)
line(AX_L, AX_B, AX_R, AX_B, stroke=SUBINK, sw=1.0)
for frac, lbl in [(0.0, "0h"), (0.5, "3h 40m"), (1.0, "7h 21m")]:
    xx = AX_L + frac * (AX_R - AX_L)
    line(xx, AX_B, xx, AX_B + 3.5, stroke=SUBINK, sw=0.9)
    anc = "end" if frac == 1.0 else ("start" if frac == 0.0 else "middle")
    txt(xx, AX_B + 14, lbl, size=8.6, fill=MUTED, anchor=anc, family=MONO)

# Clean 2-row steps.parquet card
TB_X, TB_Y, TB_W = P1X + 18, PY + 238, PW - 36
txt(TB_X, TB_Y, "steps.parquet", size=9.8, weight=700, fill=INK, family=MONO)
txt(TB_X + TB_W, TB_Y, "no prompts / tokens", size=9.0, fill=MUTED, anchor="end")

line(TB_X, TB_Y + 7, TB_X + TB_W, TB_Y + 7, stroke=INK, sw=1.2)
line(TB_X, TB_Y + 25, TB_X + TB_W, TB_Y + 25, stroke=HAIR, sw=0.9)

for cx, cname, anc in [
    (TB_X + 4, "program", "start"),
    (TB_X + 68, "shape", "start"),
    (TB_X + 236, "kv", "end"),
    (TB_X + TB_W - 4, "dur", "end"),
]:
    txt(cx, TB_Y + 19.5, cname, size=8.8, weight=700, fill=MUTED, anchor=anc, family=MONO)

for idx, (prog, shp, kv, dur, pcol) in enumerate(
    [
        ("prefill", "seqs<=1,tok<=1024", "47k", "387ms", TERRA),
        ("round", "seqs=16,tok=256", "433k", "204ms", NAVY_BAR),
    ]
):
    ry = TB_Y + 41 + idx * 20
    txt(TB_X + 4, ry, prog, size=9.0, weight=600, fill=pcol, family=MONO)
    txt(TB_X + 68, ry, shp, size=8.8, fill=BODY, family=MONO)
    txt(TB_X + 236, ry, kv, size=9.0, fill=BODY, anchor="end", family=MONO)
    txt(TB_X + TB_W - 4, ry, dur, size=9.0, fill=INK, anchor="end", family=MONO)

line(TB_X, TB_Y + 70, TB_X + TB_W, TB_Y + 70, stroke=INK, sw=1.2)

# Bottom stat pill in Panel (a)
rect(P1X + 18, PY + PH - 48, PW - 36, 32, r=6, fill=TINT_BG, stroke=HAIR, sw=1.0)
txt(
    P1X + PW / 2,
    PY + PH - 28,
    "261,672 steps  ·  7h 21m online GPU time",
    size=10.0,
    weight=700,
    fill=INK,
    anchor="middle",
    family=MONO,
)

# ===========================================================================
# PANEL (b): Shape x KV Buckets (Heatmap in Minutes + Compact TOML)
# ===========================================================================
HM_X, HM_Y = P2X + 94, PY + 72
CW, CH = 39, 22
KV_COLS = ["≤32k", "≤64k", "≤128k", "≤256k", "≤512k", "≤1M"]
HM_ROWS = [
    ("prefill", [1874, 2948, 2176, 962, 0, 0]),
    ("round g=1", [560, 1069, 894, 837, 0, 0]),
    ("round g=2", [121, 255, 593, 1429, 378, 0]),
    ("round g=4", [35, 72, 278, 1790, 619, 0]),
    ("round g=8", [2, 68, 199, 585, 1581, 159]),
    ("round g=16", [4, 8, 68, 419, 2594, 649]),
]


def heat_color(val: float) -> tuple[str, str]:
    if val <= 0:
        return ("#F8FAFC", "#CBD5E1")
    t = min(1.0, math.sqrt(val / 3000.0))
    r = int(242 - t * (242 - 28))
    g = int(246 - t * (246 - 69))
    b = int(252 - t * (252 - 135))
    ink = "#FFFFFF" if t > 0.54 else INK
    return (f"#{r:02x}{g:02x}{b:02x}", ink)


for j, kl in enumerate(KV_COLS):
    txt(HM_X + j * CW + CW / 2, HM_Y - 6, kl, size=8.5, weight=600, fill=MUTED, anchor="middle", family=MONO)

for i, (rlbl, vals) in enumerate(HM_ROWS):
    ry = HM_Y + i * CH
    is_pf = rlbl == "prefill"
    txt(HM_X - 8, ry + 14.5, rlbl, size=8.8, weight=600 if is_pf else 500, fill=TERRA if is_pf else BODY, anchor="end", family=MONO)
    for j, v in enumerate(vals):
        cx = HM_X + j * CW
        fill_c, ink_c = heat_color(v)
        rect(cx, ry, CW, CH, r=0, fill=fill_c, stroke="#FFFFFF", sw=1.0)
        lbl = fmt_hm_cell(v)
        txt(cx + CW / 2, ry + 14.5, lbl, size=8.2, weight=700 if v >= 1400 else 400, fill=ink_c, anchor="middle", family=MONO)

rect(HM_X, HM_Y, CW * len(KV_COLS), CH * len(HM_ROWS), r=0, fill="none", stroke=SUBINK, sw=0.9)

# Highlight bottom-row hotspot cell (5, 4): round g=16, <=512k = 43m
HX, HY = HM_X + 4 * CW, HM_Y + 5 * CH
rect(HX + 0.5, HY + 0.5, CW - 1, CH - 1, r=0, fill="none", stroke=TERRA_MID, sw=2.3)

CARD_Y = PY + 224
el(
    f'<line x1="{HX + CW/2:.1f}" y1="{HY + CH:.1f}" x2="{HX + CW/2:.1f}" y2="{CARD_Y - 4:.1f}" '
    f'stroke="{TERRA_MID}" stroke-width="1.6" stroke-dasharray="3 2"/>'
)
el(
    f'<polygon points="{HX + CW/2:.1f},{CARD_Y:.1f} '
    f'{HX + CW/2 - 3.6:.1f},{CARD_Y - 6:.1f} {HX + CW/2 + 3.6:.1f},{CARD_Y - 6:.1f}" fill="{TERRA_MID}"/>'
)

# Single clean horizontal TOML + bucket rule card
rect(P2X + 18, CARD_Y, PW - 36, 86, r=6, fill="#FFFBF7", stroke="#FDBA74", sw=1.1)
txt(P2X + 30, CARD_Y + 21, "served.toml", size=9.8, weight=700, fill=TERRA, family=MONO)
txt(P2X + 30, CARD_Y + 43, "groups = [16]   rows = [16]", size=9.0, fill=BODY, family=MONO)
txt(P2X + 30, CARD_Y + 62, "context = [27057]", size=9.0, fill=BODY, family=MONO)
txt(P2X + 30, CARD_Y + 77, "median shape · mean context", size=8.5, fill=MUTED, italic=True, family=SERIF)

line(P2X + 212, CARD_Y + 12, P2X + 212, CARD_Y + 74, stroke="#FED7AA", sw=1.0)
txt(HX + CW / 2, CARD_Y + 31, "weight = 12994", size=9.2, weight=700, fill=TERRA, anchor="middle", family=MONO)
txt(HX + CW / 2, CARD_Y + 50, "43m online", size=8.8, weight=600, fill=INK, anchor="middle")
txt(HX + CW / 2, CARD_Y + 67, "→ 1 sweep", size=8.6, fill=MUTED, anchor="middle", italic=True, family=SERIF)

# Bottom stat pill in Panel (b)
rect(P2X + 18, PY + PH - 48, PW - 36, 32, r=6, fill=SAGE_BG, stroke=SAGE_BORDER, sw=1.1)
txt(
    P2X + PW / 2,
    PY + PH - 28,
    "157 sweeps  ·  +0.29% CPU fidelity",
    size=10.2,
    weight=700,
    fill=SAGE,
    anchor="middle",
    family=MONO,
)

# ===========================================================================
# PANEL (c): Offline Replay (6m Offline Predicts 7h+ Online)
# ===========================================================================
rect(P3X + 18, PY + 54, PW - 36, 52, r=6, fill=SAGE_BG, stroke=SAGE, sw=1.3)
txt(
    P3X + PW / 2,
    PY + 75,
    '<tspan fill="#0F766E">6m 20s offline</tspan> run predicts <tspan fill="#0F172A">7h 21m online</tspan>',
    size=11.8,
    weight=700,
    fill=INK,
    anchor="middle",
    family=SANS,
    raw=True,
)
txt(
    P3X + PW / 2,
    PY + 96,
    'Cost(<tspan font-style="italic">M</tspan>) = '
    '<tspan font-size="11.5">∑</tspan><tspan baseline-shift="sub" font-size="7">b</tspan> '
    '<tspan font-style="italic">w</tspan><tspan baseline-shift="sub" font-size="7">b</tspan> · '
    '<tspan font-style="italic">t</tspan><tspan baseline-shift="sub" font-size="7">50</tspan>'
    '(<tspan font-style="italic">M</tspan>, <tspan font-style="italic">s</tspan><tspan baseline-shift="sub" font-size="7">b</tspan>, '
    '<tspan font-style="italic">μ</tspan><tspan baseline-shift="sub" font-size="7">b</tspan>)',
    size=10.8,
    weight=500,
    fill=SUBINK,
    anchor="middle",
    family=SERIF,
    raw=True,
)

lx = P3X + 18
for lcol, llbl, adv in [
    (TERRA_MID, "Verify MLP", 74),
    (NAVY_BAR, "Prefill Attn", 78),
    (SKY_BAR, "Verify Attn", 76),
    ("#CBD5E1", "Other", 52),
]:
    rect(lx, PY + 120, 8, 8, r=1.5, fill=lcol, stroke="none", sw=0)
    txt(lx + 11, PY + 127.5, llbl, size=8.6, fill=BODY)
    lx += adv

# Horizontal stacked bar chart on a clean 0h..8h axis
CH_L, CH_R = P3X + 106, P3X + PW - 66
CH_T, CH_B = PY + 140, PY + 286
MAX_S = 8.0 * 3600.0


def sx(sec: float) -> float:
    return CH_L + (sec / MAX_S) * (CH_R - CH_L)


for tick_h in [0, 2, 4, 6]:
    tx = sx(tick_h * 3600.0)
    line(tx, CH_T, tx, CH_B, stroke=HAIR, sw=0.8)
    txt(tx, CH_B + 14, f"{tick_h}h", size=8.6, fill=MUTED, anchor="middle", family=MONO)

BH_BAR = 26

# Bar 1: Online (26,436.7 s = 7h 21m)
B1_Y = CH_T + 8
txt(CH_L - 8, B1_Y + 12, "Online", size=9.6, weight=700, fill=INK, anchor="end")
txt(CH_L - 8, B1_Y + 23, "live trace", size=7.8, fill=MUTED, anchor="end")
rect(CH_L, B1_Y, sx(26436.7) - CH_L, BH_BAR, r=2.5, fill="#64748B", stroke="#475569", sw=0.8)
txt(sx(26436.7) + 6, B1_Y + 17, "7h 21m", size=9.0, weight=600, fill=SUBINK, family=MONO)

# Bar 2: Offline M_0 (same kernels -> 7h 25m, +0.9% match)
B2_Y = CH_T + 55
txt(
    CH_L - 8,
    B2_Y + 12,
    'Offline <tspan font-family="serif" font-style="italic">M</tspan><tspan baseline-shift="sub" font-size="7">0</tspan>',
    size=9.6,
    weight=600,
    fill=INK,
    anchor="end",
    raw=True,
)
txt(CH_L - 8, B2_Y + 23, "same kernels", size=7.8, fill=MUTED, anchor="end")
cur = 0.0
for seg_s, scol in [
    (8207.5, TERRA_MID),
    (4211.8, NAVY_BAR),
    (3182.2, SKY_BAR),
    (11072.1, "#CBD5E1"),
]:
    x0, x1 = sx(cur), sx(cur + seg_s)
    rect(x0, B2_Y, x1 - x0, BH_BAR, r=0, fill=scol, stroke="#FFFFFF", sw=0.7)
    cur += seg_s
rect(CH_L, B2_Y, sx(26673.6) - CH_L, BH_BAR, r=2.5, fill="none", stroke=SUBINK, sw=0.9)
txt(sx(26673.6) + 6, B2_Y + 12, "7h 25m", size=9.0, weight=600, fill=INK, family=MONO)
txt(sx(26673.6) + 6, B2_Y + 23, "+0.9% err", size=8.0, weight=500, fill=MUTED, family=MONO)

# Bar 3: Offline M_1 (tuned MLP kernel -> 6h 59m, -5.7% gain)
B3_Y = CH_T + 102
txt(
    CH_L - 8,
    B3_Y + 12,
    'Offline <tspan font-family="serif" font-style="italic">M</tspan><tspan baseline-shift="sub" font-size="7">1</tspan>',
    size=9.6,
    weight=700,
    fill=SAGE,
    anchor="end",
    raw=True,
)
txt(CH_L - 8, B3_Y + 23, "tuned MLP kernel", size=7.8, weight=600, fill=TERRA, anchor="end")
cur = 0.0
for seg_s, scol in [
    (8207.5 - 1520.3, TERRA_MID),
    (4211.8, NAVY_BAR),
    (3182.2, SKY_BAR),
    (11072.1, "#CBD5E1"),
]:
    x0, x1 = sx(cur), sx(cur + seg_s)
    rect(x0, B3_Y, x1 - x0, BH_BAR, r=0, fill=scol, stroke="#FFFFFF", sw=0.7)
    cur += seg_s
rect(CH_L, B3_Y, sx(25153.3) - CH_L, BH_BAR, r=2.5, fill="none", stroke=SAGE, sw=1.2)
rect(
    sx(25153.3),
    B3_Y,
    sx(26673.6) - sx(25153.3),
    BH_BAR,
    r=0,
    fill=SAGE_BG,
    stroke=SAGE,
    sw=0.9,
    dash="2 2",
)
txt(sx(26673.6) + 6, B3_Y + 12, "6h 59m", size=9.0, weight=700, fill=SAGE, family=MONO)
txt(sx(26673.6) + 6, B3_Y + 23, "−5.7% gain", size=8.0, weight=700, fill=SAGE, family=MONO)

# Reference line at online trace = 7h 21m
ref_x = sx(26436.7)
line(ref_x, CH_T + 2, ref_x, CH_B + 4, stroke=INK, sw=1.1, dash="3 2")

line(CH_L, CH_T, CH_L, CH_B, stroke=SUBINK, sw=1.0)
line(CH_L, CH_B, CH_R, CH_B, stroke=SUBINK, sw=1.0)

# Bottom stat pill in Panel (c)
rect(P3X + 18, PY + PH - 48, PW - 36, 32, r=6, fill=SAGE_BG, stroke=SAGE_BORDER, sw=1.1)
txt(
    P3X + PW / 2,
    PY + PH - 28,
    "Online match: +0.90%  ·  Kernel swap: −25m",
    size=9.8,
    weight=700,
    fill=SAGE,
    anchor="middle",
    family=MONO,
)

el("</svg>")

svg_path = Path(__file__).with_name("trace-workload.svg")
svg_path.write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"Wrote {svg_path} ({svg_path.stat().st_size} bytes)")
