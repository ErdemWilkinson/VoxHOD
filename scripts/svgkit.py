"""Static light/dark SVG charts for the READMEs (GitHub and Hugging Face render them as images).

Every chart is written twice, <stem>-light.svg and <stem>-dark.svg, so a README can
switch with <picture>. Categorical colours are the validated slots 1-4 (blue, orange,
aqua, yellow); values are printed in text ink next to each mark because aqua and
yellow are below 3:1 contrast on the light surface. Each mark has a <title>, so
hovering the image in a browser shows the exact value.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

THEMES = {
    "light": {"surface": "#fcfcfb", "border": "#e1e0d9", "title": "#0b0b0b", "text": "#52514e",
              "muted": "#7d7b75", "grid": "#e1e0d9", "axis": "#c3c2b7", "zone": "#f1f0ec",
              "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], "on_series": "#ffffff"},
    "dark": {"surface": "#1a1a19", "border": "#2c2c2a", "title": "#ffffff", "text": "#c3c2b7",
             "muted": "#94928b", "grid": "#2c2c2a", "axis": "#45453f", "zone": "#232321",
             "series": ["#3987e5", "#d95926", "#199e70", "#c98500"], "on_series": "#ffffff"},
}
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"


def esc(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def write(stem: Path, width: int, height: int, body: Callable[[dict], str]) -> None:
    """Write <stem>-light.svg and <stem>-dark.svg; body(theme) returns the inner SVG."""
    for name, t in THEMES.items():
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
               f'height="{height}" font-family="{FONT}">'
               f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="6" '
               f'fill="{t["surface"]}" stroke="{t["border"]}"/>{body(t)}</svg>')
        Path(f"{stem}-{name}.svg").write_text(svg, encoding="utf-8")


def header(t: dict, title: str, subtitle: str, legend: list[tuple[str, int | str]] | None = None) -> str:
    """Title, subtitle and an optional legend row. legend items: (label, series index or 'ring')."""
    out = [f'<text x="20" y="30" font-size="15" font-weight="600" fill="{t["title"]}">{esc(title)}</text>',
           f'<text x="20" y="49" font-size="12" fill="{t["muted"]}">{esc(subtitle)}</text>']
    x = 20
    for label, kind in legend or []:
        if kind == "ring":
            out.append(f'<circle cx="{x + 6}" cy="68" r="5" fill="{t["surface"]}" stroke="{t["muted"]}" stroke-width="2"/>')
        elif kind == "hollow":
            out.append(f'<circle cx="{x + 6}" cy="68" r="5" fill="{t["surface"]}" stroke="{t["muted"]}" stroke-width="1.5"/>')
        else:
            out.append(f'<rect x="{x}" y="62" width="12" height="12" rx="3" fill="{t["series"][kind]}"/>')
        out.append(f'<text x="{x + 18}" y="72" font-size="12" fill="{t["text"]}">{esc(label)}</text>')
        x += 34 + 6.6 * len(label)
    return "".join(out)


def bar_up(x: float, y: float, w: float, h: float) -> str:
    r = min(4.0, w / 2, h)
    return (f"M{x:.1f},{y + h:.1f} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
            f"H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} V{y + h:.1f} Z")


def bar_right(x: float, y: float, w: float, h: float) -> str:
    r = min(4.0, h / 2, w)
    return (f"M{x:.1f},{y:.1f} H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} "
            f"V{y + h - r:.1f} Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} H{x:.1f} Z")


def grouped_bars(stem: Path, title: str, subtitle: str, categories: list[str],
                 series: list[tuple[str, int, list[float | None]]], maximum: float = 100.0,
                 unit: str = "%", ticks: tuple = (0, 25, 50, 75, 100)) -> None:
    """series: (name, colour slot, values per category)."""
    width, height = 860, 380
    left, right, top, bottom = 56, 20, 104, 330
    group = (width - left - right) / len(categories)
    bar = min(46.0, (group - 40) / len(series))

    def body(t: dict) -> str:
        out = [header(t, title, subtitle, [(name, slot) for name, slot, _ in series])]
        for tick in ticks:
            y = bottom - (bottom - top) * tick / maximum
            out.append(f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="{t["grid" if tick else "axis"]}"/>'
                       f'<text x="{left - 8}" y="{y + 4:.1f}" font-size="11" text-anchor="end" fill="{t["muted"]}">{tick:g}{unit}</text>')
        for c, category in enumerate(categories):
            start = left + c * group + (group - bar * len(series) - 2 * (len(series) - 1)) / 2
            for s, (name, slot, values) in enumerate(series):
                value = values[c]
                bx = start + s * (bar + 2)
                if value is None:
                    out.append(f'<text x="{bx + bar / 2:.1f}" y="{bottom - 6}" font-size="10" text-anchor="middle" fill="{t["muted"]}">n/a</text>')
                    continue
                h = max((bottom - top) * min(value, maximum) / maximum, 1.0)
                out.append(f'<path d="{bar_up(bx, bottom - h, bar, h)}" fill="{t["series"][slot]}">'
                           f'<title>{esc(category)} · {esc(name)}: {fmt(value)}{unit}</title></path>'
                           f'<text x="{bx + bar / 2:.1f}" y="{bottom - h - 6:.1f}" font-size="11" text-anchor="middle" '
                           f'fill="{t["text"]}">{fmt(value)}</text>')
            out.append(f'<text x="{left + c * group + group / 2:.1f}" y="{bottom + 22}" font-size="12.5" '
                       f'text-anchor="middle" fill="{t["text"]}">{esc(category)}</text>')
        return "".join(out)

    write(stem, width, height, body)


def dumbbell(stem: Path, title: str, subtitle: str, rows: list[dict], divider_before: int | None = None,
             divider_label: str = "") -> None:
    """rows: {label, a (ring value), b (filled value), highlight (bool)}. Shows the a - b gap."""
    width = 860
    left, right, top, row_h = 150, 90, 96, 30
    height = top + row_h * len(rows) + 40
    plot_w = width - left - right

    def sx(v: float) -> float:
        return left + plot_w * v / 100

    def body(t: dict) -> str:
        out = [header(t, title, subtitle, [("float (computer)", "ring"), ("int8 (device file)", 0)])]
        bottom = top + row_h * len(rows)
        for tick in (0, 25, 50, 75, 100):
            out.append(f'<line x1="{sx(tick):.1f}" x2="{sx(tick):.1f}" y1="{top - 6}" y2="{bottom}" stroke="{t["grid"]}"/>'
                       f'<text x="{sx(tick):.1f}" y="{bottom + 18}" font-size="11" text-anchor="middle" fill="{t["muted"]}">{tick}%</text>')
        for i, r in enumerate(rows):
            y = top + i * row_h + row_h / 2
            if divider_before is not None and i == divider_before:
                yd = y - row_h / 2
                out.append(f'<line x1="20" x2="{width - 20}" y1="{yd:.1f}" y2="{yd:.1f}" stroke="{t["muted"]}" stroke-dasharray="3 3"/>'
                           f'<text x="{width - 20}" y="{yd - 5:.1f}" font-size="11" text-anchor="end" fill="{t["muted"]}">{esc(divider_label)}</text>')
            gap = r["a"] - r["b"]
            line = t["series"][1] if r.get("highlight") else t["muted"]
            out.append(f'<text x="{left - 12}" y="{y + 4:.1f}" font-size="12" text-anchor="end" fill="{t["title"]}">{esc(r["label"])}</text>'
                       f'<line x1="{sx(r["b"]):.1f}" x2="{sx(r["a"]):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{line}" stroke-width="{3 if r.get("highlight") else 2}"/>'
                       f'<circle cx="{sx(r["a"]):.1f}" cy="{y:.1f}" r="5" fill="{t["surface"]}" stroke="{t["muted"]}" stroke-width="2">'
                       f'<title>{esc(r["label"])} float: {fmt(r["a"])}%</title></circle>'
                       f'<circle cx="{sx(r["b"]):.1f}" cy="{y:.1f}" r="5.5" fill="{t["series"][0]}" stroke="{t["surface"]}" stroke-width="2">'
                       f'<title>{esc(r["label"])} int8: {fmt(r["b"])}%</title></circle>'
                       f'<text x="{sx(max(r["a"], r["b"])) + 12:.1f}" y="{y + 4:.1f}" font-size="11.5" '
                       f'font-weight="{600 if gap > 5 else 400}" fill="{t["title"] if gap > 5 else t["text"]}">−{fmt(gap)}</text>')
        return "".join(out)

    write(stem, width, height, body)


def hbar_panels(stem: Path, title: str, subtitle: str, panels: list[dict]) -> None:
    """Side-by-side horizontal bar panels. panel: {title, max, rows: [(label, value, text)], slot}."""
    width = 860
    top, row_h = 96, 34
    n_rows = max(len(p["rows"]) for p in panels)
    height = top + 26 + n_rows * row_h + 20
    pw = (width - 40) / len(panels)

    def body(t: dict) -> str:
        out = [header(t, title, subtitle)]
        for i, p in enumerate(panels):
            x0 = 20 + i * pw
            label_w, value_w = 118, 54
            bar_w = pw - label_w - value_w - 14
            out.append(f'<text x="{x0:.1f}" y="{top + 4}" font-size="12.5" font-weight="600" fill="{t["title"]}">{esc(p["title"])}</text>')
            for j, (label, value, text) in enumerate(p["rows"]):
                y = top + 18 + j * row_h
                h = row_h - 12
                w = max(2.0, bar_w * value / p["max"])
                bx = x0 + label_w
                out.append(f'<text x="{bx - 8:.1f}" y="{y + h / 2 + 4:.1f}" font-size="12" text-anchor="end" fill="{t["text"]}">{esc(label)}</text>'
                           f'<rect x="{bx:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h}" rx="4" fill="{t["zone"]}"/>'
                           f'<path d="{bar_right(bx, y, w, h)}" fill="{t["series"][p.get("slot", 0)]}"><title>{esc(p["title"])} · {esc(label)}: {esc(text)}</title></path>'
                           f'<text x="{bx + w + 6:.1f}" y="{y + h / 2 + 4:.1f}" font-size="12" font-weight="600" fill="{t["title"]}">{esc(text)}</text>')
        return "".join(out)

    write(stem, width, height, body)
