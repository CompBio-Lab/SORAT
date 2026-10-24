"""Figure 1: SORAT workflow schematic (180 mm wide).

Usage: python make_figure1.py <outdir>
Writes figure1_workflow.pdf, .svg and .png (300 dpi). Diagrammatic only; no anatomy is drawn.
A layout check fails the build if text overlaps other text or crosses a frame edge.
"""
import sys, itertools
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
W, H0, H1 = 180.0, 6.0, 58.0                      # canvas: width, y-range (mm)
# Carlito (SIL Open Font License) ships in fonts/ so the layout is identical on any machine.
from pathlib import Path as _Path
from matplotlib import font_manager as _fm
for _f in sorted((_Path(__file__).resolve().parents[1] / "fonts").glob("Carlito-*.ttf")):
    _fm.fontManager.addfont(str(_f))
plt.rcParams.update({"font.family": ["Carlito", "Liberation Sans"], "pdf.fonttype": 42, "svg.fonttype": "path",
                     "hatch.linewidth": 0.5})
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, SLATE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6b7685"
INK, MUTED, FRAME, RULE = "#161c28", "#5d6574", "#8c94a1", "#d3d7de"
# lane accents (pale fills) and label ramps (light / dark / mid, so they separate in greyscale)
CINE_T, LGE_T, EVAL_T = "#dfd5f2", "#f9f3e6", "#e6eff2"
VENT = {"RV 1": "#c9b8ec", "MYO 2": "#43286f", "LV 3": "#8a6dc4"}
ATR = {"Wall 1": "#5a4128", "Right atrium 2": "#e7d3a8", "Left atrium 3": "#a9824a"}
B, TT = 7.5, 8.5
fig = plt.figure(figsize=(W / 25.4, (H1 - H0) / 25.4))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H0, H1); ax.axis("off")
texts = []
def T(x, y, s, size=B, weight="normal", color=INK, ha="left", va="center"):
    t = ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, linespacing=1.15, zorder=8)
    texts.append(t); return t
def line(xs, ys, color=INK, lw=0.7, ls="-"): ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=5, solid_capstyle="butt")
def arrow(p, q, color=INK, lw=0.8, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=5, lw=lw, color=color, ls=ls, shrinkA=0, shrinkB=0, zorder=6))
def rect(x, y, w, h, z=4, **kw): ax.add_patch(Rectangle((x, y), w, h, zorder=z, **kw))
def frame(x0, x1, y0=9.0, y1=54.0, ls="-", fc="white", ec=FRAME, lw=0.7):
    rect(x0, y0, x1 - x0, y1 - y0, fc=fc, ec=ec, lw=lw, ls=ls, z=2)
def swatch(x, y, name, col, hatch=False):
    rect(x, y - 2.0, 4.0, 4.0, fc=col, ec=INK, lw=0.7, hatch="////" if hatch else None)
    T(x + 5.3, y, name, B)

Y_CINE, Y_LGE, TY = 42.2, 13.0, 50.4
IN, CT, MD, EV, OU = (1.0, 22.0), (27.0, 88.0), (93.0, 121.0), (126.0, 149.5), (154.5, 179.0)

# Inputs: two lanes
T(IN[0], TY, "Samplesheet", B, color=MUTED)
for y, lab, tint in ((Y_CINE, "Cine CMR", CINE_T), (Y_LGE, "Atrial LGE", LGE_T)):
    rect(IN[0], y - 2.6, IN[1] - IN[0] - 1.0, 5.2, fc=tint, ec="none", z=2)
    T(IN[0] + 1.5, y, lab, B)
    arrow((IN[1], y), (CT[0], y))

# Data contract (focal element)
frame(*CT, ec=INK, lw=1.1)
rect(CT[0] + 0.55, 20.6, CT[1] - CT[0] - 1.1, 54.0 - 20.6 - 0.55, fc=CINE_T, ec="none", z=2.5)
rect(CT[0] + 0.55, 9.55, CT[1] - CT[0] - 1.1, 20.6 - 9.55, fc=LGE_T, ec="none", z=2.5)
x = CT[0] + 3.0
T(x, TY, "Data contract", TT, "bold")
for i in range(8):
    on = i in (1, 5)
    rect(x + i * 7.1, 37.5, 6.0, 9.4, fc="white", ec=INK if on else FRAME, lw=1.5 if on else 0.6)
T(x + 1 * 7.1 + 3.0, Y_CINE, "ED", B, "bold", ha="center"); T(x + 5 * 7.1 + 3.0, Y_CINE, "ES", B, "bold", ha="center")
T(x, 33.2, "ED/ES checked on phase images", B)
T(x, 28.4, "Labels", B, color=MUTED)
for nm, col, dx in zip(VENT, VENT.values(), (0, 17.5, 37.0)): swatch(x + dx, 24.0, nm, col, hatch=nm.startswith("RV"))
for nm, col, dx in zip(ATR, ATR.values(), (0, 13.0, 35.2)): swatch(x + dx, Y_LGE, nm, col, hatch=nm.startswith("Right"))

# Models
frame(*MD); x = MD[0] + 2.5
rect(MD[0] + 0.55, 19.0, MD[1] - MD[0] - 1.1, 47.9 - 19.0, fc=CINE_T, ec="none", z=2.5)
rect(MD[0] + 0.55, 9.55, MD[1] - MD[0] - 1.1, 19.0 - 9.55, fc=LGE_T, ec="none", z=2.5)
T(x, TY, "Models", TT, "bold")
T(x + 4.8, 45.4, "CineMA", B)
for j, (c, lb) in enumerate(zip((BLUE, ORANGE, AQUA), ("ACDC", "M&Ms-1", "M&Ms-2"))):
    yc = 42.0 - j * 2.8
    rect(x, yc - 1.25, 2.6, 2.5, fc=c, ec="none"); T(x + 4.8, yc, lb, 6.5)
rows = [("nnFormer", YELLOW, 30.6), ("VSA-3L", MAGENTA, 23.0), ("Atrial nnU-Net", SLATE, Y_LGE)]
for nm, c, ym in rows:
    rect(x, ym - 3.0, 2.6, 6.0, fc=c, ec="none"); T(x + 4.8, ym, nm, B)
bus = (CT[1] + MD[0]) / 2
line([CT[1], bus], [Y_CINE, Y_CINE]); line([bus, bus], [23.0, Y_CINE])
for ym in (Y_CINE, 30.6, 23.0): arrow((bus, ym), (MD[0], ym))
arrow((CT[1], Y_LGE), (MD[0], Y_LGE))

# Evaluate and post-processing
frame(*EV, y0=31.0, fc=EVAL_T); x = EV[0] + 2.5
T(x, TY, "Evaluate", TT, "bold"); T(x, 41.0, "Scored in\noriginal space", B)
bus2 = (MD[1] + EV[0]) / 2
for ym in (30.6, 23.0, Y_LGE): line([MD[1], bus2], [ym, ym], MUTED, 0.7)
line([bus2, bus2], [Y_LGE, Y_CINE], MUTED, 0.7); arrow((MD[1], Y_CINE), (EV[0], Y_CINE), MUTED)
frame(*EV, y0=9.0, y1=17.5, ls=(0, (2.4, 1.8)))
T((EV[0] + EV[1]) / 2, 13.2, "Post-processing\n(optional)", B, color=MUTED, ha="center")
arrow(((EV[0] + EV[1]) / 2, 31.0), ((EV[0] + EV[1]) / 2, 17.9), MUTED, 0.7, (0, (2.2, 1.6)))

# Outputs
frame(*OU, fc=EVAL_T); x = OU[0] + 2.2
T(x, TY, "Outputs", TT, "bold")
T(x, 41.0, "Clinical\nmeasures", B)
line([OU[0] + 2.2, OU[1] - 2.2], [33.4, 33.4], FRAME, 0.5)
T(x, 28.6, "Per-subject\nreports", B)
arrow((EV[1], Y_CINE), (OU[0], Y_CINE))

fig.canvas.draw(); rend = fig.canvas.get_renderer(); inv = ax.transData.inverted(); boxes, bad = [], []
frames = {"ct": CT, "md": MD, "ev": EV, "ou": OU}
for t in texts:
    bb = t.get_window_extent(rend); (a, b_), (c, d) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    boxes.append((t.get_text().replace("\n", " | "), (a, b_, c, d)))
    if a < 0.5 or c > W - 0.5 or b_ < H0 + 0.5 or d > H1 - 0.5: bad.append(f"off canvas: {t.get_text()!r}")
    if a < IN[1]: continue
    for nm, (x0, x1) in frames.items():
        if a < x1 and c > x0 and not (a >= x0 + 1.0 and c <= x1 - 1.0): bad.append(f"crosses frame {nm}: {t.get_text()!r} {a:.1f}-{c:.1f}")
for (n1, p), (n2, q) in itertools.combinations(boxes, 2):
    if p[0] < q[2] and q[0] < p[2] and p[1] < q[3] and q[1] < p[3]: bad.append(f"overlap: {n1!r} / {n2!r}")
if bad: raise AssertionError("\n".join(bad))
print("layout ok; min font", min(t.get_fontsize() for t in texts))
for ext, kw in [("pdf", {}), ("svg", {}), ("png", {"dpi": 300})]: fig.savefig(f"{OUT}/figure1_workflow.{ext}", facecolor="white", **kw)
