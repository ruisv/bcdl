"""The bird's-eye check figures, drawn as SVG and rasterised with librsvg.

Why not cv2, which draws everything else in this file's neighbourhood: cv2 has
no type system. It can put a string somewhere at a scale, and that is the end of
it — no weights, no tabular figures, no leading, no rounded corners that are
actually round, no opacity that is not a manual blend, no shadows. A check
figure that is read every day is worth a real renderer.

Why not a headless browser, which is where "HTML and CSS" usually leads: it is
a browser. librsvg gives the same cairo/pango/harfbuzz stack a browser uses for
text and vectors, takes CSS in a `<style>` block, and rasterises this figure in
about 130 ms with nothing to install beyond a conda package.

The split that makes it work: **rasters stay rasters.** A 50k-point sweep and a
semantic map are pixels, so they are built with numpy, PNG-encoded with alpha,
and embedded as data URIs; everything with an edge or a glyph — grid, boxes,
labels, cards, legends — is vector. Both halves take metres and go through the
one `BevView`, so no drawing code has to remember which way the axes were
flipped.

Requires `rsvg-convert` (conda-forge `librsvg`, in this repo's environment).
"""
import base64
import os
import shutil
import subprocess

import numpy as np

# --- palette ---------------------------------------------------------------
# Hex, because this is a stylesheet. `bgr()` converts for the numpy layers, so
# the raster and the vector halves cannot disagree about a colour.
BG = "#12151a"
PANEL = "#171b21"
CARD = "#1c212a"
LINE = "#2b323d"
GRID = "#252b34"
GRID2 = "#39414e"
TEXT = "#e9edf3"
DIM = "#8a94a3"
FAINT = "#5b6472"

ACCENT = "#5fd97a"      # our answer: the plan, and class 0
AMBER = "#f2b23c"       # ground truth / realized
CYAN = "#46c7e8"        # ego
PINK = "#e072c8"
NEUTRAL = "#dfe5ee"     # detections and sensor returns: deliberately colourless

CLASS_COLORS = [ACCENT, AMBER, CYAN, PINK]

# A stack, not a font: Open Sans if the host has it, else Ubuntu and Source Code
# Pro, which conda-forge's `openfonts` guarantees, else DejaVu. Naming one font
# that may be absent is how a figure silently re-flows on another machine.
FONT_UI = "'Open Sans','Ubuntu','DejaVu Sans',sans-serif"
FONT_NUM = "'Source Code Pro','DejaVu Sans Mono',monospace"

# Shared stylesheet. Everything visual that is not per-element lives here.
STYLE = """
  text { font-family: %s; fill: %s; }
  .h1  { font-size: 15px; font-weight: 600; letter-spacing: .2px; }
  .h2  { font-size: 12.5px; font-weight: 600; letter-spacing: .6px;
         text-transform: uppercase; fill: %s; }
  .lb  { font-size: 12.5px; fill: %s; }
  .lbw { font-size: 12.5px; }
  .sm  { font-size: 11px; fill: %s; }
  .num { font-family: %s; font-size: 12.5px; }
  .big { font-family: %s; font-size: 21px; font-weight: 500; }
  .card  { fill: %s; }
  .rule  { stroke: %s; stroke-width: 1; }
  .grid  { stroke: %s; stroke-width: 1; fill: none; }
  .grid2 { stroke: %s; stroke-width: 1; fill: none; }
  .chip  { fill: %s; fill-opacity: .92; }
""" % (FONT_UI, TEXT, DIM, DIM, FAINT, FONT_NUM, FONT_NUM, CARD, LINE, GRID,
       GRID2, "#0f1216")

DEFS = """
  <filter id="lift" x="-30%%" y="-30%%" width="160%%" height="160%%">
    <feDropShadow dx="0" dy="1.5" stdDeviation="3" flood-color="#000" flood-opacity=".5"/>
  </filter>
  <linearGradient id="panelbg" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#1a1f26"/><stop offset="1" stop-color="%s"/>
  </linearGradient>
  <radialGradient id="egoglow">
    <stop offset="0" stop-color="%s" stop-opacity=".30"/>
    <stop offset="1" stop-color="%s" stop-opacity="0"/>
  </radialGradient>
""" % (PANEL, CYAN, CYAN)


def bgr(hex_color):
    """'#rrggbb' -> the (b, g, r) tuple the numpy layers want."""
    h = hex_color.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# --- the view: metres <-> pixels --------------------------------------------

class BevView:
    """A metric window rendered north-up: +x forward is UP, +y left is LEFT."""

    def __init__(self, fwd_m, back_m, lat_m, ppm):
        self.fwd_m, self.back_m, self.lat_m, self.ppm = fwd_m, back_m, lat_m, ppm
        self.w = int(round(2 * lat_m * ppm))
        self.h = int(round((fwd_m + back_m) * ppm))

    def px(self, x, y):
        """Metres -> panel pixels, as floats: this is vector output, so there is
        no reason to round anything to a pixel grid."""
        return (self.w / 2.0 - y * self.ppm,
                self.h - 1 - (x + self.back_m) * self.ppm)


# --- raster layers (pixels stay pixels) -------------------------------------

def points_rgba(view, xy, z=None, glow=True):
    """The sweep as an RGBA raster, height-coloured, for embedding under the
    vector overlay. Returns (rgba, lo, hi) with the height bounds it used.

    Height colour is stretched over the 2..98 PERCENTILE of the actual returns,
    not over the configured z-range. A pillar config spans -3..1 m while a road
    sweep puts nearly every return in a band a few tens of centimetres thick, so
    a map over the full range collapses to one hue and the colour says nothing —
    and telling a box on a car from a box on a wall is the whole reason the
    points are coloured at all.
    """
    import cv2

    rgba = np.zeros((view.h, view.w, 4), np.uint8)
    if len(xy) == 0:
        return rgba, 0.0, 0.0
    col = np.clip((view.w / 2.0 - xy[:, 1] * view.ppm).astype(np.int32), 0, view.w - 1)
    row = np.clip((view.h - 1 - (xy[:, 0] + view.back_m) * view.ppm).astype(np.int32),
                  0, view.h - 1)
    if z is None:
        colors = np.full((len(xy), 3), 210, np.uint8)
        lo = hi = 0.0
    else:
        lo, hi = float(np.percentile(z, 2.0)), float(np.percentile(z, 98.0))
        t = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
        # Reshape to a COLUMN first: applyColorMap on a 1-D array of N values
        # returns (1, N, 3), so the obvious [:, 0] silently yields one colour and
        # paints the entire cloud with it.
        colors = cv2.applyColorMap((t * 255).astype(np.uint8).reshape(-1, 1),
                                   cv2.COLORMAP_TURBO)[:, 0]
    core = np.zeros((view.h, view.w, 3), np.uint8)
    core[row, col] = colors
    alpha = np.zeros((view.h, view.w), np.uint8)
    alpha[row, col] = 255
    if glow:
        # A wide dim splat so sparse far returns register at all, under the crisp
        # cores so near structure keeps its shape.
        k = np.ones((3, 3), np.uint8)
        soft = cv2.dilate(core, k)
        soft_a = cv2.dilate(alpha, k)
        rgba[..., :3] = np.maximum(core, (soft * 0.55).astype(np.uint8))
        rgba[..., 3] = np.maximum(alpha, (soft_a * 0.40).astype(np.uint8))
    else:
        rgba[..., :3] = core
        rgba[..., 3] = alpha
    rgba[..., :3] = rgba[..., :3][..., ::-1]           # BGR -> RGB for the PNG
    return rgba, lo, hi


def tint_rgba(mask01, color, gamma=1.0):
    """A scalar field as a single-hue RGBA raster (lidar occupancy)."""
    a = np.clip(mask01, 0, 1) ** gamma
    r, g, b = bgr(color)[::-1]
    out = np.zeros((*a.shape, 4), np.uint8)
    out[..., 0], out[..., 1], out[..., 2] = r, g, b
    out[..., 3] = (a * 255).astype(np.uint8)
    return out


def labels_rgba(labels, colors_hex, alpha=255):
    """A class-index map as an RGBA raster."""
    out = np.zeros((*labels.shape, 4), np.uint8)
    for i, hx in enumerate(colors_hex):
        m = labels == i
        if not m.any():
            continue
        r, g, b = bgr(hx)[::-1]
        out[m] = (r, g, b, 0 if i == 0 else alpha)
    return out


# --- the document -----------------------------------------------------------

class SVG:
    """An SVG document that rasterises itself."""

    def __init__(self, w, h, bg=BG):
        self.w, self.h, self.bg = int(w), int(h), bg
        self.body = []

    def add(self, s):
        self.body.append(s)
        return self

    # -- primitives ---------------------------------------------------------
    def rect(self, x, y, w, h, fill=None, cls=None, rx=0, extra=""):
        f = 'fill="%s" ' % fill if fill else ""
        c = 'class="%s" ' % cls if cls else ""
        return self.add('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="%g" %s%s%s/>'
                        % (x, y, w, h, rx, f, c, extra))

    def line(self, x0, y0, x1, y1, cls="rule", extra=""):
        return self.add('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" class="%s" %s/>'
                        % (x0, y0, x1, y1, cls, extra))

    def text(self, x, y, s, cls="lb", extra="", anchor=None):
        a = ' text-anchor="%s"' % anchor if anchor else ""
        return self.add('<text x="%.1f" y="%.1f" class="%s"%s %s>%s</text>'
                        % (x, y, cls, a, extra, esc(s)))

    def group(self, s, extra=""):
        return self.add('<g %s>%s</g>' % (extra, s))

    def raster(self, rgba, x, y, w, h, pixelated=False, opacity=1.0):
        """Embed an RGBA numpy array as a data URI."""
        import cv2

        arr = rgba[..., [2, 1, 0, 3]] if rgba.shape[2] == 4 else rgba
        ok, buf = cv2.imencode(".png", arr)
        if not ok:
            raise RuntimeError("bev_svg: PNG encode failed")
        uri = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()
        return self.add('<image x="%.1f" y="%.1f" width="%.1f" height="%.1f" '
                        'preserveAspectRatio="none" opacity="%g" %s xlink:href="%s"/>'
                        % (x, y, w, h, opacity,
                           'image-rendering="pixelated"' if pixelated else "", uri))

    # -- rasterise ----------------------------------------------------------
    def to_svg(self):
        return ('<svg xmlns="http://www.w3.org/2000/svg" '
                'xmlns:xlink="http://www.w3.org/1999/xlink" width="%d" height="%d">'
                '<style>%s</style><defs>%s</defs>'
                '<rect width="%d" height="%d" fill="%s"/>%s</svg>'
                % (self.w, self.h, STYLE, DEFS, self.w, self.h, self.bg,
                   "".join(self.body)))

    def render(self):
        """Rasterise to a BGR ndarray, the shape the rest of the harness wants."""
        import cv2

        exe = shutil.which("rsvg-convert")
        if exe is None:
            raise RuntimeError(
                "bev_svg needs rsvg-convert (conda-forge 'librsvg'); it is in this "
                "repo's environment.yml — recreate the env with scripts/bootstrap_board.sh")
        out = subprocess.run([exe, "-w", str(self.w), "-h", str(self.h), "-f", "png"],
                             input=self.to_svg().encode(), capture_output=True)
        if out.returncode != 0:
            raise RuntimeError("rsvg-convert failed: " + out.stderr.decode()[:400])
        img = cv2.imdecode(np.frombuffer(out.stdout, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("bev_svg: rsvg-convert produced no image")
        return img


# --- composed pieces --------------------------------------------------------

def panel_bg(doc, x, y, w, h, rx=8):
    doc.rect(x, y, w, h, fill="url(#panelbg)", rx=rx)
    doc.rect(x + .5, y + .5, w - 1, h - 1, fill="none", rx=rx,
             extra='stroke="%s" stroke-width="1"' % LINE)


def panel_title(doc, x, y, w, title, right=None):
    # A scrim under the band: the panel's content runs to its edges, and a title
    # sitting straight on a point cloud is unreadable exactly where the cloud is.
    doc.add('<rect x="%.1f" y="%.1f" width="%.1f" height="37" rx="8" fill="%s" '
            'fill-opacity=".82"/>' % (x, y, w, PANEL))
    doc.text(x + 14, y + 24, title, "h1")
    if right:
        doc.text(x + w - 14, y + 23, right, "sm", anchor="end")
    doc.line(x + 12, y + 36, x + w - 12, y + 36)


def polar_grid(doc, view, ox, oy, rings, spokes=True, labels=True):
    """Range rings and radial spokes on the ego.

    Polar because a lidar's returns lie on rings and its density falls off with
    range; horizontal rules invite reading the picture as a scatter plot.
    """
    cx, cy = view.px(0.0, 0.0)
    cx, cy = cx + ox, cy + oy
    if spokes:
        r = max(view.w, view.h)
        for deg in range(0, 360, 30):
            a = np.deg2rad(deg)
            doc.line(cx, cy, cx + r * np.sin(a), cy - r * np.cos(a), "grid")
    for r_m in rings:
        doc.add('<circle cx="%.1f" cy="%.1f" r="%.1f" class="grid"/>'
                % (cx, cy, r_m * view.ppm))
    if labels:
        a = np.deg2rad(35)
        for r_m in rings:
            x, y = cx + r_m * view.ppm * np.sin(a), cy - r_m * view.ppm * np.cos(a)
            if ox + 8 < x < ox + view.w - 24 and oy + 40 < y < oy + view.h:
                doc.text(x + 4, y - 4, "%d" % r_m, "sm")


def box(doc, view, ox, oy, x, y, heading, length, width, color, wedge=True, sw=1.6):
    """An oriented footprint with a heading wedge.

    The wedge is not decoration: the direction head is a two-way classifier, so a
    footprint without one is only right modulo pi and a flipped box looks perfect.
    """
    c, s = np.cos(heading), np.sin(heading)

    def corner(dx, dy):
        px, py = view.px(x + dx * c - dy * s, y + dx * s + dy * c)
        return px + ox, py + oy

    pts = [corner(length / 2, width / 2), corner(length / 2, -width / 2),
           corner(-length / 2, -width / 2), corner(-length / 2, width / 2)]
    d = " ".join("%.1f,%.1f" % p for p in pts)
    doc.add('<polygon points="%s" fill="%s" fill-opacity=".16" stroke="%s" '
            'stroke-width="%g" stroke-linejoin="round"/>' % (d, color, color, sw))
    if wedge:
        nose = [corner(length / 2 + min(0.9, length * 0.3), 0),
                corner(length / 2, width * 0.28), corner(length / 2, -width * 0.28)]
        doc.add('<polygon points="%s" fill="%s"/>'
                % (" ".join("%.1f,%.1f" % p for p in nose), color))


def cuboid(doc, uv, color, dashed_edges=False, sw=1.6, front=True):
    """A projected 3-D box: 12 edges, with the front face called out.

    This is the view that settles whether a detection is right. A footprint can
    only show where a box is; a cuboid over the photograph shows whether it
    wraps the object, at the right height, facing the right way — the three
    things the head predicts and the three a top-down figure hides.
    """
    dash = ' stroke-dasharray="5 4"' if dashed_edges else ""
    d = []
    for a, b in EDGES_3D:
        d.append("M%.1f,%.1f L%.1f,%.1f" % (uv[a][0], uv[a][1], uv[b][0], uv[b][1]))
    doc.add('<path d="%s" fill="none" stroke="%s" stroke-width="%g" '
            'stroke-linecap="round" opacity=".95"%s/>'
            % (" ".join(d), color, sw, dash))
    if front:
        pts = " ".join("%.1f,%.1f" % (uv[i][0], uv[i][1]) for i in FRONT_3D)
        doc.add('<polygon points="%s" fill="%s" fill-opacity=".18" stroke="%s" '
                'stroke-width="%g"%s/>' % (pts, color, color, sw, dash))


EDGES_3D = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)]
FRONT_3D = [0, 1, 5, 4]


def ego(doc, view, ox, oy, length=4.6, width=1.9):
    cx, cy = view.px(0.0, 0.0)
    doc.add('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="url(#egoglow)"/>'
            % (cx + ox, cy + oy, 2.2 * view.ppm))
    box(doc, view, ox, oy, 0.0, 0.0, 0.0, length, width, CYAN, sw=1.8)


def ribbon(doc, view, ox, oy, xy, color, w0=1.0, w1=0.3):
    """A trajectory as a tapering band: the taper encodes time, and a band
    survives being drawn over a busy semantic map where a thin line does not."""
    p = np.asarray(xy, np.float64)
    if len(p) < 2:
        return
    d = np.diff(p, axis=0)
    ang = np.concatenate([np.arctan2(d[:, 1], d[:, 0])[:1],
                          np.arctan2(d[:, 1], d[:, 0])])
    half = np.linspace(w0, w1, len(p)) / 2.0
    left, right = [], []
    for (x, y), a, hw in zip(p, ang, half):
        nx, ny = -np.sin(a) * hw, np.cos(a) * hw
        lx, ly = view.px(x + nx, y + ny)
        rx, ry = view.px(x - nx, y - ny)
        left.append((lx + ox, ly + oy))
        right.append((rx + ox, ry + oy))
    poly = " ".join("%.1f,%.1f" % q for q in left + right[::-1])
    doc.add('<polygon points="%s" fill="%s" fill-opacity=".38"/>' % (poly, color))
    spine = " ".join("%.1f,%.1f" % (view.px(*q)[0] + ox, view.px(*q)[1] + oy) for q in p)
    doc.add('<polyline points="%s" fill="none" stroke="%s" stroke-width="2" '
            'stroke-linecap="round" stroke-linejoin="round"/>' % (spine, color))
    for q in p[1:]:
        px, py = view.px(*q)
        doc.add('<circle cx="%.1f" cy="%.1f" r="2.6" fill="%s" stroke="%s" '
                'stroke-width="1"/>' % (px + ox, py + oy, color, PANEL))


def dashed(doc, view, ox, oy, xy, color, sw=2.0):
    pts = " ".join("%.1f,%.1f" % (view.px(*q)[0] + ox, view.px(*q)[1] + oy) for q in xy)
    doc.add('<polyline points="%s" fill="none" stroke="%s" stroke-width="%g" '
            'stroke-dasharray="7 5" stroke-linecap="round"/>' % (pts, color, sw))


class ChipPlacer:
    """Places label chips so they do not cover each other.

    Twenty boxes in a dense scene put their scores in the same few hundred
    pixels; chips at a fixed offset stack into a pile, which loses exactly the
    numbers you wanted. Candidates are tried around the anchor, a leader line is
    drawn when the chip ends up far from its box, and anything that cannot be
    placed is dropped — an unreadable label is worse than none.
    """

    CHAR_W = 6.55       # Open Sans at 12 px, close enough to reserve space
    H = 19.0

    def __init__(self, bounds):
        self.taken = []
        self.bounds = bounds        # (x0, y0, x1, y1)

    @staticmethod
    def _hits(a, b, pad=3):
        return not (a[0] + a[2] + pad < b[0] or b[0] + b[2] + pad < a[0] or
                    a[1] + a[3] + pad < b[1] or b[1] + b[3] + pad < a[1])

    def add(self, doc, s, ax, ay, color):
        w = len(s) * self.CHAR_W + 16
        h = self.H
        x0, y0, x1, y1 = self.bounds
        for dx, dy in ((9, -h / 2), (-w - 9, -h / 2), (9, -h - 9), (-w - 9, -h - 9),
                       (9, 9), (-w - 9, 9), (-w / 2, -h - 13), (-w / 2, 13),
                       (9, -h - 26), (-w - 9, -h - 26), (-w / 2, -h - 30)):
            x, y = ax + dx, ay + dy
            if not (x0 <= x and x + w <= x1 and y0 <= y and y + h <= y1):
                continue
            cand = (x, y, w, h)
            if any(self._hits(cand, t) for t in self.taken):
                continue
            if abs(dx) > 24 or abs(dy) > 24:
                doc.line(ax, ay, x + w / 2, y + h / 2, "",
                         'stroke="%s" stroke-width="1" stroke-opacity=".55"' % color)
            doc.rect(x, y, w, h, cls="chip", rx=4, extra='filter="url(#lift)"')
            doc.rect(x + 1.5, y + 3, 2.5, h - 6, fill=color, rx=1.2)
            # style=, not fill=: a class rule beats a presentation attribute, so
            # `fill="..."` on an element that carries a class silently does
            # nothing — which is how these labels came out uniformly grey.
            doc.text(x + 9, y + h - 5.5, s, "sm", extra='style="fill:%s"' % color)
            self.taken.append(cand)
            return True
        return False


def card(doc, x, y, w, h, title=None):
    doc.rect(x, y, w, h, cls="card", rx=8, extra='filter="url(#lift)"')
    if title:
        doc.text(x + 14, y + 24, title, "h2")
        doc.line(x + 14, y + 33, x + w - 14, y + 33)
        return y + 33 + 20
    return y + 20


def rows(doc, x, y, w, entries, line_h=21):
    """`(label, colour_or_None, value_or_None)` rows; values right-aligned in
    tabular figures so a column of numbers can be scanned."""
    for row in entries:
        label, color, value = (tuple(row) + (None,) * 3)[:3]
        if label is None:
            y += 9
            continue
        tx = x + 14
        if color is not None:
            doc.rect(tx, y - 8.5, 9, 9, fill=color, rx=2)
            tx += 15
        doc.text(tx, y, label, "lb" if color is not None else "lbw")
        if value is not None:
            doc.text(x + w - 14, y, value, "num", anchor="end")
        y += line_h
    return y


def height_ramp(doc, x, y, w, h, lo, hi, unit="m"):
    """The height scale, labelled with the percentile bounds the picture used —
    a legend describing a mapping the figure does not use is worse than none."""
    import cv2

    grad = np.linspace(255, 0, 256).astype(np.uint8).reshape(-1, 1)
    bar = cv2.applyColorMap(grad, cv2.COLORMAP_TURBO)
    rgba = np.dstack([bar[..., ::-1], np.full(bar.shape[:2], 255, np.uint8)])
    doc.raster(rgba, x, y, w, h)
    doc.rect(x + .5, y + .5, w - 1, h - 1, fill="none", rx=2,
             extra='stroke="%s"' % LINE)
    doc.text(x + w + 9, y + 9, "%+.1f %s" % (hi, unit), "sm")
    doc.text(x + w + 9, y + h, "%+.1f %s" % (lo, unit), "sm")


def scale_bar(doc, view, ox, oy, metres=10):
    """A bar of known length, so the figure survives being cropped into a
    document without its axis labels."""
    n = metres * view.ppm
    x1, y1 = ox + view.w - 16, oy + view.h - 16
    x0 = x1 - n
    doc.line(x0, y1, x1, y1, "", 'stroke="%s" stroke-width="2"' % DIM)
    for x in (x0, x1):
        doc.line(x, y1 - 4, x, y1 + 1, "", 'stroke="%s" stroke-width="2"' % DIM)
    doc.text((x0 + x1) / 2, y1 - 8, "%d m" % metres, "sm", anchor="middle")


def available():
    return shutil.which("rsvg-convert") is not None
