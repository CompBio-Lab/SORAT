"""Draw docs/assets/sorat-workflow-{light,dark}.svg for the README.

The figure follows one real case through SORAT: rows of a samplesheet, the
short-axis model containers, a mask in the input geometry, and the resulting
rows of comparison/aggregated_metrics.csv (ACDC patient101, ED, RV).

Usage: python docs/assets/make_workflow_svg.py docs/assets
"""
from pathlib import Path
import sys

OUT = Path(sys.argv[1])
W, H = 960, 222
THEMES = {
    "light": dict(fg="#1f2328", muted="#59636e", card="#ffffff", head="#f6f8fa", stroke="#d1d9e0",
                  line="#8c959f", accent="#0969da", rv="#cf222e", myo="#1a7f37", lv="#0969da"),
    "dark": dict(fg="#e6edf3", muted="#9198a1", card="#0d1117", head="#151b23", stroke="#3d444d",
                 line="#656c76", accent="#4493f8", rv="#f85149", myo="#3fb950", lv="#4493f8"),
}
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
CONTAINERS = [("sorat-cinema.sif", "#2a78d6"), ("sorat-nnformer.sif", "#eda100"), ("sorat-vsa3l.sif", "#e87ba4")]


def t(x, y, s, size=12, fill="fg", mono=False, anchor="start", weight=400):
    fam = MONO if mono else SANS
    return (f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" font-weight="{weight}" '
            f'fill="var(--{fill})" text-anchor="{anchor}" xml:space="preserve">{s}</text>')


def file_card(x, y, w, name, rows):
    h = 34 + 18 * len(rows) + 8
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="var(--card)" stroke="var(--stroke)"/>',
           f'<path d="M{x},{y + 8} a8,8 0 0 1 8,-8 h{w - 16} a8,8 0 0 1 8,8 v18 h-{w} z" fill="var(--head)"/>',
           f'<line x1="{x}" y1="{y + 26}" x2="{x + w}" y2="{y + 26}" stroke="var(--stroke)"/>',
           t(x + 12, y + 18, name, 11.5, "muted", mono=True)]
    for i, r in enumerate(rows):
        out.append(t(x + 12, y + 46 + i * 18, r, 11.5, "fg", mono=True))
    return "\n".join(out), h


def flow(x1, y1, x2, y2):
    mx = (x1 + x2) / 2
    d = f"M{x1},{y1} C{mx},{y1} {mx},{y2} {x2},{y2}"
    return (f'<path d="{d}" fill="none" stroke="var(--line)" stroke-width="1.4"/>'
            f'<path class="flow" d="{d}" fill="none" stroke="var(--accent)" stroke-width="1.4" stroke-dasharray="3 11"/>')


def mask(cx, cy):
    """Schematic short-axis slice: RV crescent, myocardial ring, LV cavity."""
    return "\n".join([
        f'<path d="M{cx - 20},{cy - 30} C{cx - 62},{cy - 26} {cx - 66},{cy + 24} {cx - 22},{cy + 32} '
        f'C{cx - 40},{cy + 14} {cx - 42},{cy - 12} {cx - 20},{cy - 30} Z" fill="var(--rv)" fill-opacity="0.85"/>',
        f'<circle cx="{cx}" cy="{cy}" r="30" fill="var(--myo)" fill-opacity="0.85"/>',
        f'<circle cx="{cx}" cy="{cy}" r="19" fill="var(--lv)" fill-opacity="0.9"/>',
    ])


def svg(theme):
    v = THEMES[theme]
    css = (".flow{animation:dash 1.8s linear infinite} @keyframes dash{to{stroke-dashoffset:-28}}"
           " @media (prefers-reduced-motion: reduce){.flow{animation:none}}")
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
         f"<title>SORAT workflow</title><style>{css}</style>"]
    sheet, sh = file_card(16, 52, 248, "samplesheet.csv",
                          ["patient_id,image,ground_truth,…", "patient101,…_4d.nii.gz,…", "patient102,…_4d.nii.gz,…",
                           "site001,cine_sax/,,…"])
    p.append(sheet)
    mid = 52 + sh / 2
    p.append(t(140, 52 + sh + 24, "$ nextflow run main.nf", 11.5, "muted", mono=True, anchor="middle"))
    xs, ws = 390, 196
    for i, (name, col) in enumerate(CONTAINERS):
        y = mid - 19 + (i - 1) * 58
        p.append(f'<rect x="{xs}" y="{y}" width="{ws}" height="38" rx="8" fill="var(--card)" stroke="var(--stroke)"/>')
        p.append(f'<rect x="{xs}" y="{y + 9}" width="3" height="20" rx="1.5" fill="{col}"/>')
        p.append(t(xs + 14, y + 23, name, 11.5, "fg", mono=True))
        p.append(flow(264, mid, xs, y + 19))
        p.append(flow(xs + ws, y + 19, 636, mid))
    p.append(mask(676, mid))
    p.append(t(660, mid + 52, "in the input geometry", 11, "muted", anchor="middle"))
    metrics, mh = file_card(728, mid - 54, 216, "aggregated_metrics.csv",
                            ["model     dice_rv  hd95_rv", "cinema    0.963    1.6", "nnformer  0.965    1.6",
                             "vsa3l     0.912   10.0"])
    p.append(metrics)
    p.append(flow(706, mid, 728, mid))
    p.append(t(W / 2, H - 8, "ACDC patient101, end-diastole, right ventricle", 11, "muted", anchor="middle"))
    p.append("</svg>")
    out = "\n".join(p)
    for k, val in v.items():
        out = out.replace(f"var(--{k})", val)
    return out


OUT.mkdir(parents=True, exist_ok=True)
for theme in THEMES:
    (OUT / f"sorat-workflow-{theme}.svg").write_text(svg(theme))
