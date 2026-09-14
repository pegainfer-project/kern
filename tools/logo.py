"""The kern logo: a stem and a detached chevron that read as the letter k, the pipe
`|>` and a play button, followed by e r n in geometric monoline. Run from the repo
root to regenerate docs/assets/{logo,mark}.svg and the site favicon."""
import math
import sys

W = 26            # stroke width
TOP, XH, BASE = 40, 100, 260   # ascender, x-height, baseline (outer edges)
MID = (XH + BASE) / 2


def stroke(d, color):
    return (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{W}" '
            'stroke-linecap="butt" stroke-linejoin="miter" stroke-miterlimit="4"/>')


def k_mark(x, color):
    """A stem and a detached chevron: the letter k, the pipe |>, a play button."""
    stem = f"M{x},{TOP} L{x},{BASE}"
    cap = W / 2 * math.sin(math.pi / 4)          # a 45° butt cap pokes out this far
    x0 = x + W / 2 + 0.75 * W + cap               # three quarters of a stroke of clear gap
    h = (BASE - XH) / 2 - cap
    chevron = f"M{x0},{MID - h} L{x0 + h},{MID} L{x0},{MID + h}"
    tip = x0 + h + W / 2 * math.sqrt(2)           # miter of a 90° join
    return [stroke(stem, color), stroke(chevron, color)], tip


def e(x, color):
    r = (BASE - XH) / 2 - W / 2 + 3               # rounds overshoot by 3
    cx = x + W / 2 + r
    a = math.pi / 4
    ex, ey = cx + r * math.cos(a), MID + r * math.sin(a)
    d = (f"M{cx - r},{MID} L{cx + r + W / 2},{MID} "
         f"M{cx + r},{MID} A{r},{r} 0 1 0 {ex:.1f},{ey:.1f}")
    return [stroke(d, color)], cx + r + W / 2


def r_(x, color):
    sx = x + W / 2
    rad = 50
    top = XH + W / 2 - 3
    d = (f"M{sx},{BASE} L{sx},{XH} "
         f"M{sx},{top + rad} A{rad},{rad} 0 0 1 {sx + rad},{top} L{sx + rad + 18},{top}")
    return [stroke(d, color)], sx + rad + 18


def n(x, color):
    sx = x + W / 2
    rad = 62
    top = XH + W / 2 - 3
    d = (f"M{sx},{BASE} L{sx},{top + rad} A{rad},{rad} 0 0 1 {sx + 2 * rad},{top + rad} "
         f"L{sx + 2 * rad},{BASE}")
    return [stroke(d, color)], sx + 2 * rad + W / 2


def wordmark(color, margin=40):
    parts, x = [], margin + W / 2
    for glyph, gap in ((k_mark, 30), (e, 30), (r_, 24), (n, 30)):
        ps, x = glyph(x, color)
        parts += ps
        x += gap
    width = x - gap + margin
    body = "\n  ".join(parts)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} 300" '
            f'width="{width:.0f}" height="300" role="img" aria-label="kern">\n  {body}\n</svg>\n')


def mark(color, size=300, plate=None):
    parts, tip = k_mark(0, color)
    ox, oy = (size - tip) / 2, (size - (BASE - TOP)) / 2 - TOP
    body = "\n    ".join(parts)
    bg = f'  <rect width="{size}" height="{size}" rx="{size / 8:.0f}" fill="{plate}"/>\n' if plate else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" '
            f'width="{size}" height="{size}" role="img" aria-label="kern">\n{bg}'
            f'  <g transform="translate({ox:.1f},{oy:.1f})">\n    {body}\n  </g>\n</svg>\n')


if __name__ == "__main__":
    color = "#0647ff"
    open("docs/assets/logo.svg", "w").write(wordmark(color))
    open("docs/assets/mark.svg", "w").write(mark(color))
    open("docs/site/public/favicon.svg", "w").write(mark("#ffffff", plate=color))
