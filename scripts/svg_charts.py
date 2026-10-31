"""Grouped bar charts as static light/dark SVGs for the README.

Same palette and anatomy as the existing charts in docs/: series 1 blue,
series 2 orange (validated categorical slots 1-2), recessive grid, values in
text ink above each bar, a legend for the two series and a <title> on every
bar so hovering shows the exact value.
"""
from __future__ import annotations

from pathlib import Path

THEMES = {
    "light": {"surface": "#fcfcfb", "border": "#e1e0d9", "title": "#0b0b0b", "text": "#52514e",
              "muted": "#7d7b75", "grid": "#e1e0d9", "axis": "#c3c2b7", "series": ["#2a78d6", "#eb6834"]},
    "dark": {"surface": "#1a1a19", "border": "#2c2c2a", "title": "#ffffff", "text": "#c3c2b7",
             "muted": "#94928b", "grid": "#2c2c2a", "axis": "#45453f", "series": ["#3987e5", "#d95926"]},
}
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"


def grouped_bars(path_stem: Path, title: str, subtitle: str, categories: list[str],
                 series: dict[str, list[float | None]], maximum: float = 100.0, unit: str = "%") -> None:
    """Write <path_stem>-light.svg and <path_stem>-dark.svg."""
    width, height = 860, 360
    left, right, top, bottom = 56, 20, 100, 300
    group = (width - left - right) / len(categories)
    bar = min(34.0, (group - 24) / len(series))
    for theme_name, theme in THEMES.items():
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
               f'height="{height}" font-family="{FONT}">',
               f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="6" '
               f'fill="{theme["surface"]}" stroke="{theme["border"]}"/>']
        x = 20
        for index, name in enumerate(series):
            out.append(f'<rect x="{x}" y="15" width="12" height="12" rx="3" fill="{theme["series"][index]}"/>'
                       f'<text x="{x + 18}" y="25" font-size="12" fill="{theme["text"]}">{name}</text>')
            x += 30 + 7 * len(name)
        out.append(f'<text x="20" y="60" font-size="14" font-weight="600" fill="{theme["title"]}">{title}</text>'
                   f'<text x="20" y="77" font-size="11.5" fill="{theme["muted"]}">{subtitle}</text>')
        for tick in range(0, 101, 25):
            y = bottom - (bottom - top) * tick / 100
            out.append(f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" '
                       f'stroke="{theme["grid" if tick else "axis"]}"/>'
                       f'<text x="{left - 8}" y="{y + 4:.1f}" font-size="11" text-anchor="end" '
                       f'fill="{theme["muted"]}">{maximum * tick / 100:g}</text>')
        for c, category in enumerate(categories):
            start = left + c * group + (group - bar * len(series) - 2 * (len(series) - 1)) / 2
            for s, (name, values) in enumerate(series.items()):
                value = values[c]
                bx = start + s * (bar + 2)
                if value is None:
                    out.append(f'<text x="{bx + bar / 2:.1f}" y="{bottom - 6}" font-size="10" text-anchor="middle" '
                               f'fill="{theme["muted"]}">n/a</text>')
                    continue
                h = max((bottom - top) * min(value, maximum) / maximum, 1.0)
                y = bottom - h
                r = min(4.0, bar / 2, h)
                out.append(f'<path d="M{bx:.1f},{bottom} V{y + r:.1f} Q{bx:.1f},{y:.1f} {bx + r:.1f},{y:.1f} '
                           f'H{bx + bar - r:.1f} Q{bx + bar:.1f},{y:.1f} {bx + bar:.1f},{y + r:.1f} V{bottom} Z" '
                           f'fill="{theme["series"][s]}"><title>{category} · {name}: {value:.1f}{unit}</title></path>'
                           f'<text x="{bx + bar / 2:.1f}" y="{y - 5:.1f}" font-size="10.5" text-anchor="middle" '
                           f'fill="{theme["text"]}">{value:.1f}</text>')
            out.append(f'<text x="{left + c * group + group / 2:.1f}" y="{bottom + 18}" font-size="11.5" '
                       f'text-anchor="middle" fill="{theme["text"]}">{category}</text>')
        out.append("</svg>")
        Path(f"{path_stem}-{theme_name}.svg").write_text("\n".join(out), encoding="utf-8")
