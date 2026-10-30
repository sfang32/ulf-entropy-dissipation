#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""drawio2svg.py -- minimal, dependency-free renderer for mxGraphModel .drawio files.

Why: exporting through the draw.io desktop CLI needs the app installed, which is not
available here.  This renders the subset of draw.io features our paper figures use:

  * vertices        rounded / plain rectangles, containers (children in relative coords)
  * labels          multiline text (&#10; or <br>), bold, colour, centred
  * edges           orthogonal routing, dashed, coloured, arrowheads, waypoints
  * swimlanes       (rendered as a titled container)

Not supported: curves, custom shapes, HTML label markup beyond <br>.

Usage:
  python drawio2svg.py fig1-workflow.drawio [out.svg]
"""
from __future__ import annotations

import html
import re
import sys
import xml.etree.ElementTree as ET

FONT = "Helvetica, Arial, sans-serif"


def _style(s):
    out = {}
    for kv in (s or "").split(";"):
        if not kv:
            continue
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k.strip()] = v.strip()
        else:
            out[kv.strip()] = True
    return out


def _esc(t):
    return html.escape(t, quote=False)


def _lines(value):
    """Split a draw.io label into display lines."""
    v = value or ""
    v = re.sub(r"<br\s*/?>", "\n", v, flags=re.I)
    v = v.replace("&#10;", "\n").replace("\\n", "\n")
    v = re.sub(r"<[^>]+>", "", v)                  # strip any other markup
    return [l for l in html.unescape(v).split("\n")]


def _wrap(lines, width, fs):
    """Soft-wrap long labels so they stay inside their box."""
    import textwrap
    maxc = max(8, int(width / (fs * 0.56)))
    out = []
    for ln in lines:
        if len(ln) <= maxc:
            out.append(ln)
        else:
            out.extend(textwrap.wrap(ln, maxc) or [ln])
    return out


class Cell:
    __slots__ = ("id", "value", "style", "is_v", "is_e", "parent", "x", "y",
                 "w", "h", "points", "source", "target")

    def __init__(self, el):
        self.id = el.get("id")
        self.value = el.get("value") or ""
        self.style = _style(el.get("style"))
        self.is_v = el.get("vertex") == "1"
        self.is_e = el.get("edge") == "1"
        self.parent = el.get("parent")
        self.source = el.get("source")
        self.target = el.get("target")
        self.x = self.y = self.w = self.h = 0.0
        self.points = []
        g = el.find("mxGeometry")
        if g is not None:
            for k in ("x", "y"):
                if g.get(k) is not None:
                    setattr(self, k, float(g.get(k)))
            if g.get("width") is not None:
                self.w = float(g.get("width"))
            if g.get("height") is not None:
                self.h = float(g.get("height"))
            arr = g.find("Array")
            if arr is not None:
                for p in arr.findall("mxPoint"):
                    self.points.append((float(p.get("x")), float(p.get("y"))))


def load(path):
    root = ET.parse(path).getroot()
    cells = {c.get("id"): Cell(c) for c in root.iter("mxCell") if c.get("id")}
    return cells


def abs_pos(cells, c):
    """Absolute top-left, accumulating container offsets (relative coords)."""
    x, y, p = c.x, c.y, c.parent
    guard = 0
    while p and p in cells and guard < 20:
        pc = cells[p]
        if pc.is_v:
            x += pc.x
            y += pc.y
        p = pc.parent
        guard += 1
    return x, y


def _port(cells, c, kind):
    """Anchor point on a shape boundary; honours exitX/exitY and entryX/entryY."""
    x, y = abs_pos(cells, c)
    if kind == "exit":
        fx = float(c.style.get("exitX", 0.5))
        fy = float(c.style.get("exitY", 1.0))
    else:
        fx = float(c.style.get("entryX", 0.5))
        fy = float(c.style.get("entryY", 0.0))
    return (x + c.w * fx, y + c.h * fy), fx, fy


def _route(p1, s_fx, s_fy, p2, t_fx, t_fy):
    """Orthogonal route between two boundary anchors."""
    (x1, y1), (x2, y2) = p1, p2
    if abs(x1 - x2) < 1.5 or abs(y1 - y2) < 1.5:
        return [p1, p2]
    if s_fy >= 0.99:                      # leaving from the bottom
        mid = (y1 + y2) / 2 if y2 > y1 else y1 + 26
        return [p1, (x1, mid), (x2, mid), p2]
    if s_fx >= 0.99:                      # leaving to the right
        mid = (x1 + x2) / 2 if x2 > x1 else x1 + 26
        return [p1, (mid, y1), (mid, y2), p2]
    if s_fx <= 0.01:                      # leaving to the left
        mid = (x1 + x2) / 2 if x2 < x1 else x1 - 26
        return [p1, (mid, y1), (mid, y2), p2]
    mid = (y1 + y2) / 2
    return [p1, (x1, mid), (x2, mid), p2]


def render(path, out):
    cells = load(path)
    vmax_x = vmax_y = 0
    for c in cells.values():
        if c.is_v:
            x, y = abs_pos(cells, c)
            vmax_x = max(vmax_x, x + c.w)
            vmax_y = max(vmax_y, y + c.h)
    W, H = vmax_x + 40, vmax_y + 40

    parts = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
             'viewBox="0 0 %d %d">' % (W, H, W, H),
             '<rect width="100%%" height="100%%" fill="#ffffff"/>',
             '<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" '
             'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
             '<path d="M 0 0 L 10 5 L 0 10 z" fill="#666666"/></marker></defs>']

    # ---------- vertices ----------
    for c in cells.values():
        if not c.is_v:
            continue
        x, y = abs_pos(cells, c)
        fill = c.style.get("fillColor", "#ffffff")
        stroke = c.style.get("strokeColor", "#333333")
        dashed = c.style.get("dashed") == "1"
        rounded = c.style.get("rounded") == "1" or "swimlane" in c.style
        rx = min(12, c.h / 6) if rounded else 0
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        parts.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="%.1f" '
                     'fill="%s" stroke="%s" stroke-width="1.4"%s/>'
                     % (x, y, c.w, c.h, rx, fill, stroke, dash))

        sw = c.style.get("swimlane")
        if sw:                                     # title bar
            ts = float(c.style.get("startSize", 30))
            parts.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" '
                         'stroke="%s" stroke-width="1.2"/>'
                         % (x, y + ts, x + c.w, y + ts, stroke))

        lines = _wrap(_lines(c.value), c.w, float(c.style.get("fontSize", 12)))
        if not lines:
            continue
        fs = float(c.style.get("fontSize", 12))
        fcol = c.style.get("fontColor", "#1a1a1a")
        bold = c.style.get("fontStyle", "").startswith("1")
        valign = c.style.get("verticalAlign", "middle")
        if "swimlane" in c.style:
            cy = y + float(c.style.get("startSize", 30)) / 2.0
        elif valign == "top":
            cy = y + float(c.style.get("startSize", 12)) + 8 + fs / 2
        else:
            cy = y + c.h / 2 - (len(lines) - 1) * fs * 0.62
        anchor = c.style.get("align", "center")
        tx = x + c.w / 2
        if anchor == "left":
            tx, t_anchor = x + 10, "start"
        else:
            t_anchor = "middle"
        for i, ln in enumerate(lines):
            parts.append('<text x="%.1f" y="%.1f" font-family="%s" font-size="%.1f" '
                         'fill="%s" text-anchor="%s" font-weight="%s" '
                         'dominant-baseline="middle">%s</text>'
                         % (tx, cy + i * fs * 1.24, FONT, fs, fcol, t_anchor,
                            "bold" if bold else "normal", _esc(ln)))

    # ---------- edges ----------
    for c in cells.values():
        if not c.is_e:
            continue
        s, t = cells.get(c.source), cells.get(c.target)
        if s is None or t is None:
            continue
        a, sfx, sfy = _port(cells, s, "exit")
        b, tfx, tfy = _port(cells, t, "entry")
        pts = c.points[:]
        route = ([a] + pts + [b]) if pts else _route(a, sfx, sfy, b, tfx, tfy)
        col = c.style.get("strokeColor", "#666666")
        dash = ' stroke-dasharray="7 5"' if c.style.get("dashed") == "1" else ""
        d = "M %.1f %.1f " % route[0] + " ".join("L %.1f %.1f" % p for p in route[1:])
        parts.append('<path d="%s" fill="none" stroke="%s" stroke-width="1.5"%s '
                     'marker-end="url(#ar)"/>' % (d, col, dash))

    parts.append("</svg>")
    svg = "\n".join(parts)
    with open(out, "w", encoding="utf-8") as f:
        f.write(svg)
    print("[svg] %s  (%dx%d, %d bytes)" % (out, W, H, len(svg)))


def _font(size, bold=False):
    import os
    from PIL import ImageFont
    cands = ("C:/Windows/Fonts/arialbd.ttf" if bold
             else "C:/Windows/Fonts/arial.ttf",
             "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
             "C:/Windows/Fonts/msyh.ttc")
    for f in cands:
        if os.path.exists(f):
            try:
                return ImageFont.truetype(f, max(8, int(round(size))))
            except Exception:
                pass
    return ImageFont.load_default()


def render_png(path, out, scale=1.6):
    """Rasterise with Pillow (no cairo / draw.io desktop needed)."""
    from PIL import Image, ImageDraw
    cells = load(path)
    vmax_x = vmax_y = 0
    for c in cells.values():
        if c.is_v:
            x, y = abs_pos(cells, c)
            vmax_x = max(vmax_x, x + c.w)
            vmax_y = max(vmax_y, y + c.h)
    W, H = int((vmax_x + 40) * scale), int((vmax_y + 40) * scale)
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    def S(v):
        return v * scale

    for c in cells.values():
        if c.is_v:
            x, y = abs_pos(cells, c)
            fill = c.style.get("fillColor", "#ffffff")
            stroke = c.style.get("strokeColor", "#333333")
            rounded = c.style.get("rounded") == "1" or "swimlane" in c.style
            rx = min(12, c.h / 6) if rounded else 0
            box = [S(x), S(y), S(x + c.w), S(y + c.h)]
            if rx:
                d.rounded_rectangle(box, radius=S(rx), fill=fill,
                                    outline=stroke, width=max(1, int(scale)))
            else:
                d.rectangle(box, fill=fill, outline=stroke,
                            width=max(1, int(scale)))
            if c.style.get("swimlane"):
                ts = float(c.style.get("startSize", 30))
                d.line([S(x), S(y + ts), S(x + c.w), S(y + ts)], fill=stroke,
                       width=max(1, int(scale)))

    for c in cells.values():                       # text on top of all boxes
        if not c.is_v:
            continue
        fs = float(c.style.get("fontSize", 12))
        lines = _wrap(_lines(c.value), c.w, fs)
        if not lines:
            continue
        x, y = abs_pos(cells, c)
        fcol = c.style.get("fontColor", "#1a1a1a")
        bold = c.style.get("fontStyle", "").startswith("1")
        font = _font(fs * scale, bold)
        if "swimlane" in c.style:
            cy = y + float(c.style.get("startSize", 30)) / 2.0
        elif c.style.get("verticalAlign") == "top":
            cy = y + float(c.style.get("startSize", 12)) + 10 + fs / 2
        else:
            cy = y + c.h / 2 - (len(lines) - 1) * fs * 0.62
        inner = c.parent in cells and cells[c.parent].is_v
        if c.style.get("align") == "left":
            for i, ln in enumerate(lines):
                d.text((S(x + 10), S(cy + i * fs * 1.24) - (0 if not inner else 0)),
                       ln, font=font, fill=fcol, anchor="lm")
        else:
            for i, ln in enumerate(lines):
                d.text((S(x + c.w / 2), S(cy + i * fs * 1.24)), ln,
                       font=font, fill=fcol, anchor="mm")

    def arrow(p, q):
        import math
        dx, dy = q[0] - p[0], q[1] - p[1]
        n = math.hypot(dx, dy) or 1
        ux, uy = dx / n, dy / n
        L = 11 * scale
        Wd = 5.0 * scale
        bx, by = q[0] - ux * L, q[1] - uy * L
        d.polygon([q, (bx - uy * Wd, by + ux * Wd), (bx + uy * Wd, by - ux * Wd)],
                  fill="#666666")
        return (bx, by)

    for c in cells.values():
        if not c.is_e:
            continue
        s, t = cells.get(c.source), cells.get(c.target)
        if s is None or t is None:
            continue
        a, sfx, sfy = _port(cells, s, "exit")
        b, tfx, tfy = _port(cells, t, "entry")
        pts = c.points[:]
        route = ([a] + pts + [b]) if pts else _route(a, sfx, sfy, b, tfx, tfy)
        col = c.style.get("strokeColor", "#666666")
        pts_s = [(S(u), S(v)) for u, v in route]
        end = arrow(pts_s[-2], pts_s[-1])
        pts_s[-1] = end
        w = max(1, int(1.5 * scale))
        if c.style.get("dashed") == "1":
            for i in range(len(pts_s) - 1):
                (x1, y1), (x2, y2) = pts_s[i], pts_s[i + 1]
                n = 14
                for k in range(0, n, 2):
                    d.line([x1 + (x2 - x1) * k / n, y1 + (y2 - y1) * k / n,
                            x1 + (x2 - x1) * (k + 1) / n,
                            y1 + (y2 - y1) * (k + 1) / n], fill=col, width=w)
        else:
            d.line(pts_s, fill=col, width=w, joint="curve")

    img.save(out)
    print("[png] %s  (%dx%d)" % (out, W, H))


if __name__ == "__main__":
    src = sys.argv[1]
    base = src.rsplit(".", 1)[0]
    render(src, base + ".svg")
    try:
        render_png(src, base + ".png")
    except Exception as e:
        print("[png] skipped (%s: %s)" % (type(e).__name__, e))
