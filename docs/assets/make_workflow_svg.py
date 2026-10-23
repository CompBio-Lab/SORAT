"""Generate docs/assets/sorat-workflow-{light,dark}.svg for the README.

Usage: python docs/assets/make_workflow_svg.py docs/assets
"""
from pathlib import Path
import sys

OUT = Path(sys.argv[1])
W, H = 960, 380
THEMES = {
    "light": dict(fg="#1f2328", muted="#59636e", fill="#f6f8fa", card="#ffffff", stroke="#d1d9e0", line="#8c959f", accent="#0969da"),
    "dark": dict(fg="#e6edf3", muted="#9198a1", fill="#151b23", card="#0d1117", stroke="#3d444d", line="#656c76", accent="#4493f8"),
}
MODELS = [("CineMA", "9 checkpoints + ensembles", "#2a78d6"), ("nnFormer", "3-D transformer", "#eda100"),
          ("VSA-3L", "MONAI 2-D U-Net", "#e87ba4"), ("Atrial nnU-Net", "LGE, biatrial", "#1baf7a")]
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"


def text(x, y, s, size=13, weight=400, fill="fg", anchor="start"):
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="var(--{fill})" text-anchor="{anchor}">{s}</text>'


def card(x, y, w, h, title, lines, accent=None):
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="var(--card)" stroke="var(--stroke)"/>']
    if accent:
        out.append(f'<rect x="{x}" y="{y + 10}" width="4" height="{h - 20}" rx="2" fill="{accent}"/>')
    out.append(text(x + 16, y + 26, title, 14, 600))
    for i, ln in enumerate(lines):
        out.append(text(x + 16, y + 48 + i * 19, ln, 12, 400, "muted"))
    return "\n".join(out)


def flow(x1, y1, x2, y2):
    mx = (x1 + x2) / 2
    d = f"M{x1},{y1} C{mx},{y1} {mx},{y2} {x2},{y2}"
    return (f'<path d="{d}" fill="none" stroke="var(--line)" stroke-width="1.5"/>'
            f'<path class="flow" d="{d}" fill="none" stroke="var(--accent)" stroke-width="1.5" stroke-dasharray="4 10"/>'
            f'<circle cx="{x2}" cy="{y2}" r="3" fill="var(--line)"/>')


def svg(theme):
    t = THEMES[theme]
    css = (".flow{animation:dash 1.6s linear infinite} @keyframes dash{to{stroke-dashoffset:-28}}"
            " @media (prefers-reduced-motion: reduce){.flow{animation:none}}")
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{FONT}">',
         f"<title>SORAT workflow</title><style>{css}</style>"]
    cols = [(24, "INPUT"), (254, "DATA CONTRACT"), (494, "MODEL CONTAINERS"), (734, "EVALUATION")]
    for x, label in cols:
        p.append(text(x, 30, label, 11, 600, "muted"))
    mid = 197
    p.append(card(24, mid - 55, 196, 110, "Samplesheet", ["Cine SAX or atrial LGE", "NIfTI · NRRD · DICOM", "Optional reference masks"]))
    p.append(card(254, mid - 55, 206, 110, "One convention", ["ED/ES frames verified", "Label maps declared", "Original geometry kept"]))
    p.append(flow(220, mid, 254, mid))
    for i, (name, sub, col) in enumerate(MODELS):
        y = 52 + i * 76
        p.append(card(494, y, 206, 62, name, [sub], accent=col))
        p.append(flow(460, mid, 494, y + 31))
        p.append(flow(700, y + 31, 734, mid))
    p.append(card(734, mid - 75, 202, 150, "In original space",
                  ["DSC · HD95 per structure", "Failure and off-heart checks", "Volumes · EF · LV mass", "Bland–Altman · ICC",
                   "Reports with provenance"]))
    p.append(text(W / 2, H - 12, "Every model runs in its own Apptainer container; Nextflow schedules the work locally or on SLURM.",
                  12, 400, "muted", "middle"))
    p.append("</svg>")
    out = "\n".join(p)
    for k, v in t.items():
        out = out.replace(f"var(--{k})", v)
    return out


OUT.mkdir(parents=True, exist_ok=True)
for theme in THEMES:
    (OUT / f"sorat-workflow-{theme}.svg").write_text(svg(theme))
print("written")
